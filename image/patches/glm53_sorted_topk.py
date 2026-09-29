#!/usr/bin/env python3
"""Sort each row of the kpool indexer's top-k before it is used.

The prefill and decode top-k kernels return the right set of pools in an order
that changes from run to run. The pools expand to token ids in that order, and
the sparse MLA kernel sums keys in list order, so two identical requests past
2048 tokens round differently at the first sparse layer. The next layer's
indexer then picks a different set, and prompt logprobs past 2048 tokens moved
by up to 9.8 nats between identical requests. Sorting fixes the order without
changing the set.

Same contract as the other patchers: each anchor exactly once, or the build
fails.
"""
from pathlib import Path

PATH = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sparse_indexer.py")

HELPER_ANCHOR = "@eager_break_during_capture\ndef sparse_attn_indexer_kpool(\n"
HELPER = '''from vllm.triton_utils import tl, triton


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
DECODE = """        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_dst,
            select_k,
            attn_metadata_narrowed.max_seq_len,
        )
"""

text = PATH.read_text()
if "def _sort_topk_rows(" in text:
    print("[sorted-topk] already applied")
else:
    for name, anchor in (("helper", HELPER_ANCHOR), ("prefill", PREFILL), ("decode", DECODE)):
        n = text.count(anchor)
        assert n == 1, f"[sorted-topk] {name} anchor matched {n} times; stock tree changed"
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    text = text.replace(PREFILL, PREFILL + "            _sort_topk_rows(topk_dst)\n")
    text = text.replace(DECODE, DECODE + "        _sort_topk_rows(topk_dst)\n")
    PATH.write_text(text)
    print("[sorted-topk] kpool top-k rows sorted in prefill and decode (Triton, one launch)")
