#!/usr/bin/env python3
"""Make the kpool indexer's top-k the same on every run.

The prefill and decode top-k kernels return the right pools, but in a
run-dependent order, and when pools tie at the k-th score they keep a
run-dependent subset of the tied ones. The pools expand to token ids in list
order and the sparse MLA kernel sums keys in that order, so either difference
changes the rounding. The next layer's indexer then picks different keys, and
identical requests past 2048 tokens drifted apart by up to 9.8 nats.

The kernels are asked for TIE_MARGIN extra candidates. A Triton pass reads the
candidates' scores, keeps the top k with ties going to the lowest pool id, and
writes the ids in ascending order. A row whose tie runs past the extra
candidates is rescanned in full.

The persistent and cooperative decode kernels only take k in (512, 1024, 2048),
so with those backends decode rows are only sorted.

Same contract as the other patchers: each anchor exactly once, or the build
fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sparse_indexer.py")

HELPER_ANCHOR = "@eager_break_during_capture\ndef sparse_attn_indexer_kpool(\n"
HELPER = '''from vllm.triton_utils import tl, triton

# Extra candidates asked of the top-k kernel. Pools tied at the k-th score are
# almost always among them.
_TIE_MARGIN = 32


@triton.jit
def _order_key(x):
    # float32 -> int64 in [0, 2^32) with the same order as the floats.
    b = x.to(tl.int32, bitcast=True)
    return (b ^ ((b >> 31) & 0x7FFFFFFF)).to(tl.int64) + 2147483648


@triton.jit
def _select_topk_rows_kernel(
    cand_ptr, cand_stride, logits_ptr, logits_stride, ks_ptr, ke_ptr,
    out_ptr, out_stride, K, KM,
    HAS_KS: tl.constexpr, BLOCK: tl.constexpr, SCAN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_KS:
        ks = tl.load(ks_ptr + row).to(tl.int64)
    else:
        ks = tl.zeros((), dtype=tl.int64)
    ke = tl.load(ke_ptr + row).to(tl.int64)
    lrow = logits_ptr + row * logits_stride + ks
    orow = out_ptr + row * out_stride
    offs = tl.arange(0, BLOCK)
    pad = 2147483647
    ids = tl.load(cand_ptr + row * cand_stride + offs, mask=offs < KM, other=-1)
    ids = tl.sort(tl.where(ids < 0, pad, ids))
    valid = ids != pad
    n_valid = tl.sum(valid.to(tl.int32))
    u = tl.where(valid, _order_key(tl.load(lrow + ids, mask=valid, other=0.0)), -1)
    # The k-th largest key: the largest t with at least K keys >= t.
    thr = tl.zeros((), dtype=tl.int64)
    for bit in tl.static_range(31, -1, -1):
        t = thr | (1 << bit)
        thr = tl.where(tl.sum((u >= t).to(tl.int32)) >= K, t, thr)
    tied = u == thr
    need = K - tl.sum((u > thr).to(tl.int32))
    if n_valid <= K:
        keep = valid
    else:
        keep = (u > thr) | (tied & (tl.cumsum(tied.to(tl.int32), axis=0) <= need))
    lowest = tl.min(tl.where(valid, u, 1 << 40))
    if (n_valid == KM) & (lowest == thr):
        # The tie may continue past the candidates. Every score above it is
        # a candidate, so walk the row in id order and take everything above
        # it plus the lowest-id tied pools.
        base = 0
        for start in range(0, ke - ks, SCAN):
            c = start + tl.arange(0, SCAN)
            m = c < ke - ks
            uc = _order_key(tl.load(lrow + c, mask=m, other=0.0))
            eq = m & (uc == thr)
            take = m & ((uc > thr) | (eq & (tl.cumsum(eq.to(tl.int32), axis=0) <= need)))
            tl.store(orow + base + tl.cumsum(take.to(tl.int32), axis=0) - 1, c.to(tl.int32), mask=take)
            base += tl.sum(take.to(tl.int32))
            need -= tl.sum(eq.to(tl.int32))
    else:
        n = tl.sum(keep.to(tl.int32))
        tl.store(orow + tl.cumsum(keep.to(tl.int32), axis=0) - 1, ids, mask=keep)
        tl.store(orow + offs, tl.full((BLOCK,), -1, tl.int32), mask=(offs >= n) & (offs < K))


def _select_topk_rows(cand, logits, ks, ke, out):
    """Write each row's top out.shape[1] ids to out in ascending order, lowest id first among equal scores.

    cand holds an exact top-(k + _TIE_MARGIN) per row as ids relative to ks,
    -1 padded. logits[r, ks[r]:ke[r]] are row r's scores; ks=None means 0.
    """
    if cand.shape[0] == 0:
        return
    _select_topk_rows_kernel[(cand.shape[0],)](
        cand, cand.stride(0), logits, logits.stride(0), ke if ks is None else ks, ke,
        out, out.stride(0), out.shape[1], cand.shape[1],
        HAS_KS=ks is not None, BLOCK=triton.next_power_of_2(cand.shape[1]), SCAN=1024,
    )


@triton.jit
def _sort_topk_rows_kernel(ptr, stride, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    pad = 2147483647
    v = tl.load(ptr + row * stride + offs, mask=offs < n, other=pad)
    v = tl.sort(tl.where(v < 0, pad, v))
    tl.store(ptr + row * stride + offs, tl.where(v == pad, -1, v), mask=offs < n)


def _sort_topk_rows(t: torch.Tensor) -> None:
    """Order each row's ids ascending in place, keeping the -1 padding last."""
    if t.shape[0] == 0:
        return
    _sort_topk_rows_kernel[(t.shape[0],)](
        t, t.stride(0), t.shape[1], BLOCK=triton.next_power_of_2(t.shape[1])
    )


'''
PREFILL = """            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )
"""
PREFILL_NEW = """            cand = torch.full(
                (num_rows, select_k + _TIE_MARGIN), -1, dtype=torch.int32, device=logits.device
            )
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                cand,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k + _TIE_MARGIN,
            )
            _select_topk_rows(cand, logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, topk_dst)
"""
DECODE = """        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_dst,
            select_k,
            attn_metadata_narrowed.max_seq_len,
        )
"""
DECODE_NEW = """        if topk_backend == "per_row":
            # 2-D seq_lens hold each row's own length, as the kernel reads them.
            cand = torch.full(
                (num_rows, select_k + _TIE_MARGIN), -1, dtype=torch.int32, device=logits.device
            )
            get_indexer_topk(topk_backend)(
                logits,
                seq_lens,
                next_n,
                cand,
                select_k + _TIE_MARGIN,
                attn_metadata_narrowed.max_seq_len,
            )
            _select_topk_rows(cand, logits, None, seq_lens.reshape(-1)[:num_rows], topk_dst)
        else:
""" + "".join("    " + line + "\n" for line in DECODE.splitlines()) + """            _sort_topk_rows(topk_dst)
"""

text = PATH.read_text()
if "def _select_topk_rows(" in text:
    print("[deterministic-topk] already applied")
else:
    for name, anchor in (("helper", HELPER_ANCHOR), ("prefill", PREFILL), ("decode", DECODE)):
        n = text.count(anchor)
        assert n == 1, f"[deterministic-topk] {name} anchor matched {n} times; stock tree changed"
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    text = text.replace(PREFILL, PREFILL_NEW)
    text = text.replace(DECODE, DECODE_NEW)
    PATH.write_text(text)
    print("[deterministic-topk] kpool top-k: lowest pool id wins ties, rows sorted (prefill and decode)")
