# Experimental speedups

Compose overrides that stack on `compose/glm53.yaml`. Each one mounts files
over the image this repo builds (vLLM nightly ddd6fbca); several replace whole vLLM or
FlashInfer files, so they only match that image.

## How to use it

Requirements: the image built from this tree (`image/build.sh`) and the normal four-node setup from the main README (the same `compose/glm53.yaml` and a per-node `compose/.env` on every box). The overrides replace files inside that image, so they do not apply to other images.

1. Put both ConnectX PCIe roots on the fabric: tag both interfaces `rdma` in mentatd's `MENTAT_ANNOUNCE_IFACES` (main README, step 4), or set `FABRIC_SUBNETS` to both subnets in each node's `compose/.env`. The entrypoint derives `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX` from it, and arx and arxbig use the same two devices. Each rank's log shows them (`arx all-reduce: rank r/4 on [dev0, dev1]`). With only one device they log a warning and fall back to NCCL.
2. Start the stack with every override, on every node. Order matters: later files win.

   ```
   docker compose -f compose/glm53.yaml \
     -f experimental/compose/arx.yaml -f experimental/compose/snapshot.yaml \
     -f experimental/compose/adaptive-k.yaml -f experimental/compose/fp8.yaml \
     -f experimental/compose/megamoe.yaml -f experimental/compose/fixes.yaml \
     -f experimental/compose/sp.yaml -f experimental/compose/recoverssm.yaml up -d
   ```

3. The first boot loads the checkpoint normally (~8 min), compiles the CUDA extensions, and writes a processed-weight snapshot per rank under `weight-snapshots` in the cache mount (`CACHE_HOME`, or the `glm53_cache` volume), ~48 GB per node. Later boots restore it (~3.5 min).
4. Check: `bash smoketest/run.sh http://<head>:8002` should pass 8/8, and each rank's log should show `arx all-reduce: rank r/4` and `arxbig all-gather: rank r/4`.

### Turning pieces off

Leave a compose file out to drop that piece, or set its switch in `.env` (every node must use the same values):

| Switch | Default | What it does |
|---|---|---|
| `VLLM_ARXBIG` | 1 | arxbig RDMA collectives for prefill (arx.yaml) |
| `VLLM_ARXBIG_AG` | 0 | route prefill all-gathers through arxbig; off because its output sits in pinned memory, where GEMMs reading it run 3.7x slower |
| `VLLM_ARXBIG_RS` | 1 | RDMA reduce-scatter buffers (~0.5 GB pinned per rank); needed by `VLLM_GLM_SP_MOE_FUSED` |
| `VLLM_ARX_RING` | 0 (1 with `TP=RING4`) | for boxes cabled in a ring with no switch: arx and arxbig open QPs only to rank r-1 and r+1, and the rank between two others relays their data, with the same results as over the switch |
| `ARX_RING_PREV_HCAS` | from mentat with `TP=RING4` | with `VLLM_ARX_RING=1`: the RDMA devices, root 0 then root 1, of the port cabled to rank r-1. Each device uses the GID of its own IPv4 address |
| `ARX_RING_NEXT_HCAS` | unset | the same for the port cabled to rank r+1 |
| `VLLM_GLM_SP_TP` | 1 | sequence parallelism for forwards of `VLLM_GLM_SP_MIN_TOKENS` (1024) or more (sp.yaml) |
| `VLLM_GLM_SP_FP8_GATHER` | 1 | gather KDA attention inputs as FP8 |
| `VLLM_GLM_SP_MOE_FUSED` | 1 | MoE combine feeding the RDMA reduce-scatter; needs arx.yaml with `VLLM_ARXBIG_RS=1` and megamoe.yaml with `VLLM_MOE_PREFILL=1` |
| `VLLM_GLM_SP_MOE_QUANT_GATHER` | 1 | with `VLLM_GLM_SP_MOE_FUSED`: route each rank's own rows and gather the MoE input as NVFP4 + FP8 instead of bf16 (same results) |
| `VLLM_GLM_ARX_PREFETCH` | 1 | L2 prefetch during decode all-reduces; needs arx.yaml |
| `VLLM_MEGAMOE` | 1 | decode MoE kernel for batches of up to `VLLM_MEGAMOE_MAX_TOKENS` (8) |
| `VLLM_MOE_PREFILL` | 1 | prefill MoE kernel for batches of `VLLM_MOE_PREFILL_MIN_TOKENS` (1024) or more |
| `VLLM_MOE_PREFILL_Y8` | 1 | FP8 per-expert rows in the fused prefill MoE |
| `VLLM_TRITON_SPARSE_MLA` | 1 | Triton sparse MLA instead of FlashInfer's (fixes.yaml) |
| `MAX_NUM_SEQS` | 50 | requests decoding at once (fixes.yaml); about 50 KDA states fit in the pinned KV pool, against the stock 32 |
| `VLLM_GLM5NEXT_RECOVERSSM` | 1 | with recoverssm.yaml: KDA drafts verified from one saved state per request, bit-exact; raises `MAX_NUM_SEQS` to 64 at TP=4 and 16 at TP=2 |
| `VLLM_GLM5NEXT_DRAFT_POOL` | 1 | the drafter's KV in its own small pool instead of a page in every KV block: ~40% more KV tokens (fixes.yaml) |
| `VLLM_DENSE_W4` | in_proj, o_proj, shared experts, drafter | regex of dense layers stored as NVFP4 (the rest are FP8); add `\|lm_head$` for the opt-in NVFP4 lm_head |
| `VLLM_DENSE_FP8_LM_HEAD` | 1 | FP8 lm_head |
| `VLLM_ADAPTIVE_K_ONLINE` | 1 | adaptive-k learns the step cost while serving; 0 uses the fixed table below |
| `VLLM_ADAPTIVE_K_MODEL` | fitted on this stack | starting step cost: ms, ms per expert touched, ms per verified token |
| `VLLM_ADAPTIVE_K_COST_MS` | measured on this stack | with `VLLM_ADAPTIVE_K_ONLINE=0`: step cost per verify length |

Debug switches: `VLLM_MOE_PREFILL_CHECK=N` and `VLLM_GLM_SP_MOE_CHECK=N` also run the stock path on the first N prefill batches and log the difference.

### Snapshot version and tag

A snapshot holds processed weights. Its key covers the checkpoint, the TP layout, the MoE backend and `max_num_batched_tokens`. Two kinds of change need more:

- A code change to how weights are processed (`weight_snapshot.py`, `dense_fp8.py`, the loader patches): bump `SNAPSHOT_VERSION` in `weight_snapshot.py`. Snapshot names start with `v<version>-`, and each boot deletes the snapshots of every other version from its snapshot directory, for all TP sizes.
- A knob that changes processing (`VLLM_DENSE_W4`, `VLLM_DENSE_FP8*`): a new `VLLM_WEIGHT_SNAPSHOT_TAG` (set in fp8.yaml). Snapshots under the old tag stay on disk until the next version bump.

A shape change fails the restore with an error. A same-shape change (such as which layers are NVFP4) would restore stale weights without one.

### Quality check

`experimental/quality/quality.py` (GSM8K and HumanEval against a running server; see its docstring for the data files and the no-network HumanEval run).
`experimental/quality/prefill_block.py` measures what a long prefill does to other requests: gaps in a stream that is already generating, and time to first token for short requests arriving meanwhile.
`experimental/quality/agent_tools.py` runs five small agent tasks (list, read, search and write files in an in-memory tree) streamed and not streamed, and checks the result; `python3 agent_tools.py http://<head>:8002`.

## Results

Single stream, thinking off, 512 tokens (`dev/repro/decode.py` prompts):

| | structured | code | prose |
|---|---|---|---|
| stock, DFlash2 k=7 | 121.9 | 91.3 | 38.7 |
| all overrides | 170.3 | 120.8 | 65.8 |

Concurrent streams, aggregate tok/s at 1/2/4/8 streams: stock 85.5/65.4/99.5/147.8,
all overrides 126/102/146/198 (16 streams: 271). With a different prompt per
stream, code runs 317 tok/s at 16 streams and 451 at 32, and 50 mixed streams
run 404. RigMark (reasoning=low) code / prose / structured 107.9 / 61.7 / 157.1
tok/s. Needle retrieval 12/12 up to 507k tokens. KV cache 3.63M tokens per
rank at the 26 GiB pin and 50 sequences (2.63M before the drafter got its own
pool).

Cold prefill, tok/s (random-word prompts, nothing cached):

| | 32k | 128k |
|---|---|---|
| stock | 2,730 | 2,679 |
| all overrides | 4,946 | 4,750 |

Boot goes from about 8 minutes to about 3.5 once snapshots exist.

## The pieces

- **arx** (`arx/`): all-reduce over RDMA for tensors up to 256 KB, without
  NCCL. It takes 13-27 us where NCCL takes 80-90. The GPU publishes a
  sequence number, and a CPU thread posts the RDMA writes over both ConnectX
  roots.
- **snapshot** (`snapshot/weight_snapshot.py`, `base_loader.py`): the first
  boot writes each rank's processed weights to disk. Later boots build the
  model on dummy weights through the normal path, then overwrite every tensor
  in place (about 15 s for 46 GiB). `modelopt.py` and
  `flashinfer_cutlass_moe.py` are left from an earlier design and may no
  longer be needed.
- **fp8** (`snapshot/dense_fp8.py`): the NVFP4 checkpoint leaves about 4 GB
  per rank of dense linears in bf16, and decode reads them every step. This
  converts them to FP8 after load (per-channel weight scales, per-token
  activation scales, CUTLASS scaled_mm). Prefill-sized batches go through in
  2048-row pieces when the weight is over 16 MiB: CUTLASS rereads the whole
  weight for every row of tiles, and past L2 that comes from DRAM, so one call
  on a 16k x 6416 x 4096 GEMM runs at 68 TFLOPS and the pieces at 169. +14-20% decode, and about +0.5% NLL
  on prose. Layers matching `VLLM_DENSE_W4` (by default the KDA in_proj and
  every drafter layer, attention o_proj and the shared experts) go to NVFP4 instead, through `megamoe/megadense4.cu`
  (W4A16 at the 4-bit roofline, about twice as fast as the FP8 GEMM). They
  keep an FP8 copy for batches of more than 32 tokens. That is +6-10% decode.
  The in_proj costs about as much NLL again as FP8 did. The drafter layers only
  change acceptance. NVFP4 on every dense layer cost 2-3x more NLL, so it is
  not the default.
- **megamoe** (`megamoe/`): an NVFP4 MoE kernel for batches of up to 8 tokens.
  It reads the CUTLASS backend's own tensors, so CUTLASS still handles
  prefill. Activations stay 16-bit, which is exact in the weights, where
  CUTLASS rounds them to FP4. With `VLLM_MOE_PREFILL=1` (the default here),
  batches of 1024+ tokens take `megamoe/moe_prefill.cu`: the same W4A4 math as
  CUTLASS on the same tensors, with the token gather, SwiGLU and FP4
  requantization folded into the fc1 GEMM (hand-written block-scaled
  `mma.sync` NVFP4). About 18 ms per layer at 16k tokens against CUTLASS's
  22-26, equally close to an fp32 reference. +7-8% prefill.
- **adaptive-k** (`adaptive-k/adaptive_k.py`): a scheduler that picks how many
  of the 7 DFlash2 drafts each step verifies (2, 3, 4, 5 or 7). It uses recent
  per-position acceptance, per request and averaged over all requests, and a
  step cost it learns from measured step times: a fixed part, the MoE experts
  the step's tokens touch, and a per-token part. Weight reads follow the
  experts touched, so a 128-token step costs far less than a straight line
  through small ones, and at 16+ streams the scheduler verifies more drafts
  than a fixed table would allow. The drafter uses Triton attention, which
  avoids a mid-step host sync.
  The scheduler picks the step's draft count before the drafts exist. Once
  they do, `draft-trunc/` cuts each request's drafts where their predicted
  survival falls below `VLLM_DRAFT_TRUNC_TAU` (0.3). The prediction is vLLM's
  acceptance estimator, which scores each draft from the logit of its most
  likely token (AUC 0.91-0.93 per position on logged steps). Dead drafts keep
  the verify shape, route to expert -1 (the MoE kernels skip them) and are
  rejected, so greedy output is unchanged, token for token and logprob for
  logprob. A cut step grades only the drafts the estimator expected to
  survive, and fitting on those steps drifted it low over a long mixed
  workload. So every 4th step (`VLLM_DRAFT_TRUNC_EXPLORE`) skips the cut, and
  only those steps train the estimator. On TP=4, measured on one boot after
  the full gate: structured / code / prose decode +2 / +1-2 / +5-7%.
  `VLLM_DRAFT_TRUNC_TAU=0` turns it off.
- **recoverssm** (`fixes/recoverssm.py`, `compose/recoverssm.yaml`;
  `VLLM_GLM5NEXT_RECOVERSSM=0` turns it off): the KDA layers keep one recurrent state
  per request instead of one per draft position, and after sampling replay
  the accepted tokens through the stock kernel, so outputs are bit-identical.
  Ported from vLLM's Kimi-K3 RecoverSSM. At TP=2, 16 requests run at once
  instead of 3, and 8 streams decode 44% faster. At TP=4, 61 run at 64
  streams instead of 49 (+28% aggregate).
- **fixes** (`fixes/`): things found by profiling.
  - The fused mHC post + pre-norm decode kernel read its projection weights
    in fp32, 1.5 MB a call, though the checkpoint stores them in bf16. It
    now reads a cached bf16 copy (mhc_tilelang.py, tilelang_kernels.py):
    8.2 us a call instead of 11.9, ~1% faster decode at one stream, and
    greedy logprobs identical to the fp32 path.
  - FlashInfer's MLA planner cloned a 136 MB metadata buffer on every step,
    only to roll it back on error.
  - The GLM indexer's head gate ran a fp32 GEMM that took 69 us per layer.
  - FlashInfer's sparse MLA runs at about 10 TFLOPS on GB10 whatever the
    indices. `gb10_sparse_mla.py` is a Triton kernel for GLM's NoPE MLA (one
    program per query, all 16 heads, 32 gathered keys per block) that runs
    3.3x faster. `VLLM_TRITON_SPARSE_MLA=0` goes back to FlashInfer.
    At decode sizes (a request's 1 + 7 draft tokens) one program per query
    left most of the 48 SMs idle, and FlashInfer decoded 3-6% faster; below
    96 query tokens the kernel now splits each token's rows across programs
    and combines them (4.4x faster at 8 tokens), which recovers most of that
    without FlashInfer's per-step host sync and keeps Triton's prefill.
  - FlashKDA needs dense q, k and v, and the KDA short conv wrote them as one
    [tokens, 3 x 2048] tensor, so each KDA layer copied three 56 MB tensors
    per 16k-token step. `causal_conv1d.py` gains an `out_group` argument that
    writes each group of features as its own dense tensor, and `kda.py` uses
    it. The outputs are bit-identical; prefill +2-3%.
  - The DFlash2 drafter's KV rode the target's pool: every 2304-token block
    carried the drafter's pages for its five layers, 30% of the KV cache,
    though a sliding-window layer only ever holds its 2048-token window.
    `kv_cache_utils.py`, `kv_cache_coordinator.py`, `kv_cache_interface.py`,
    `worker_utils.py` and `warmup.py` give the drafter's group its own pool
    (sized for `max_num_seqs` windows plus one prefill chunk, 105 blocks,
    0.58 GiB) and keep its block ids out of the target's zeroing and warmup.
    At the same 26 GiB pin: 1,416 -> 1,976 blocks, 2.63M -> 3.67M KV tokens
    at 32 sequences (3.63M at the 50 fixes.yaml sets).
    Draft acceptance and speed are unchanged. `VLLM_GLM5NEXT_DRAFT_POOL=0`
    goes back; `VLLM_GLM5NEXT_DRAFT_POOL_BLOCKS` sets the pool's size.
- **sp** (`fixes/model.py`): sequence parallelism for prefill. A forward of
  at least 1024 tokens keeps the residual stream split across the TP ranks, so
  mHC and the norms run on a quarter of the tokens, with an all-gather before
  and a reduce-scatter after attention and the MLP. +17% prefill. Decode stays
  plain TP: applied to every batch, SP cost decode 10-15%. The KDA layers'
  attention inputs are gathered as per-token FP8, half the bytes, since
  in_proj's FP8 GEMM would quantize the same rows the same way (+2%).
- **arxbig** (`arx/arxbig.cu`, `VLLM_ARXBIG`, `VLLM_ARXBIG_RS`): all-gather and
  reduce-scatter over RoCE for prefill-sized SP collectives, from pinned
  buffers the ConnectX writes directly. The all-gather is ~10% faster than
  NCCL's (187 vs 165 Gb/s), but its result lands in pinned memory, and on
  GB10 a GEMM that reads its operand from pinned memory runs 3.7x slower, so
  prefill gathers stay on NCCL (`VLLM_ARXBIG_AG=0`). The point is the
  reduce-scatter. With `VLLM_GLM_SP_MOE_FUSED`, MoE layers under SP run the
  router, the shared expert and moe_prefill's fc1/fc2 in model.py, and one
  kernel writes shared + scaled routed sum (fp32, rounded once) straight into
  the send buffer, publishing rows as it goes, so the network runs under the
  finalize. It replaces the runner's scale and add passes and NCCL's
  reduce-scatter: ~12.5 -> ~8.4 ms per MoE layer at 16k tokens, prefill +2.5-3%.
  With `VLLM_MOE_PREFILL_Y8` (default) fc2 writes its per-expert rows as e4m3
  with a scale per 128 columns, halving the 1.07 GB each layer writes and reads
  back: prefill +5%, GSM8K/HumanEval/NLL/tool-call results unchanged.
  With `VLLM_GLM_SP_MOE_QUANT_GATHER` (default) each rank routes its own rows
  and quantizes them the way the experts would (NVFP4 for the routed experts,
  per-token FP8 for the shared expert's GEMM), and the gather moves those
  instead of bf16: 0.56 + 1 bytes per value instead of 2, and a quarter of the
  fp32 router GEMM per rank. The layer output matches the bf16 gather to
  within 1e-4 relative; prefill +3-4%.
  All ranks write to all peers at once and this fabric has no PFC, so
  incast drops packets and go-back-N retransmits make individual calls vary
  (4-9 ms); per-destination serialization was worse (two QPs cannot fill a
  link).
- **ring** (`TP=RING4`; `image/fabric_ring.py`, ring mode in `arx/`): TP=4 on
  four boxes cabled in a loop. A ConnectX-7 can't forward RoCE for another
  box, because RoCE addressed to its MAC goes to its own RDMA engine, so every
  collective uses only neighbour links. mentat 0.17's ring claim places rank i
  on member i in cable order and gives each rank its interface toward each
  neighbour. `fabric_ring.py` runs at each rank's Python start and turns that
  into NCCL's ring algorithm with a channel graph that receives on the port
  toward the previous rank and sends on the port toward the next, and into
  arx's devices. The graph carries the speeds NCCL computes for its own ring:
  with others it picked other protocols, the reductions summed in another
  order, and a greedy text diverged. arx and arxbig relay data for the
  opposite box through a neighbour, raw, so every rank sums the same values
  in rank order and results stay bit-identical. Each cable needs its own
  subnet for each PCIe root. On a switch, with both neighbours on the one
  port, `arx/test_ring.py` passed bit for bit, decode was ~1% slower from the
  relay and prefill 3-4% slower (2026-09-28). Not yet measured on cables.
- **arx prefetch** (`VLLM_GLM_ARX_PREFETCH`): while a decode all-reduce waits
  for its peers, its threads ask L2 for the weights the next kernels read (the
  router gate and shared expert after attention, the next in_proj after the
  MoE). About -0.6 ms per 32 ms step.

`dense_fp8.simulate_nvfp4` is a diagnostic. It rounds the dense weights
through NVFP4 to measure the quality cost: about +1% NLL on prose.

## Sources

Most of `fixes/`, and a few files elsewhere, are vLLM or FlashInfer files with
our changes, bind-mounted over the originals in the image. They keep their
Apache-2.0 headers. The rest is ours.

| Origin | Files |
|---|---|
| vLLM, changed | `arx/cuda_communicator.py`; `fixes/attention.py`, `causal_conv1d.py`, `flashinfer_mla_sparse_sm90.py`, `gdn_attn.py`, `kda.py`, `kv_cache_coordinator.py`, `kv_cache_interface.py`, `kv_cache_utils.py`, `model.py`, `tilelang_kernels.py`, `warmup.py`, `worker_utils.py`; `snapshot/base_loader.py`, `flashinfer_cutlass_moe.py`, `modelopt.py` |
| FlashInfer, changed | `fixes/_fa_common.py` |
| ours | `arx/arx.py`, `arx_vllm.cu`, `arxbig.cu`; `adaptive-k/adaptive_k.py`; `fixes/gb10_sparse_mla.py`, `recoverssm.py` (after vLLM's Kimi-K3 RecoverSSM); `megamoe/`; `snapshot/weight_snapshot.py`, `dense_fp8.py`; `quality/` |

Other people's work in here:

- arx and arxbig take the RDMA path MTU from the ports rather than assuming
  4096, from [Chuck](https://github.com/chuck-ads)'s
  [#6](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/pull/6).
- `tilelang_kernels.py` carries vLLM's copy of SGLang's mHC kernel, and
  `causal_conv1d.py` vLLM's adaptation of Tri Dao's causal-conv1d, both
  credited in the file.
- The decode split in `gb10_sparse_mla.py` is the Flash-Decoding scheme (Dao
  et al., 2023): split each query's keys across programs, then merge the
  partial softmax results.

## Quality

`quality/quality.py` (GSM8K first 250 and HumanEval, greedy, thinking off;
HumanEval programs run by `quality/run_he.py` in a no-network container),
against stock on the same boxes:

| | stock | all overrides |
|---|---|---|
| GSM8K (250) | 97.2% | 97.2% |
| HumanEval pass@1 | 156/164 | 156/164 |
| count to 200, thinking off, 5 runs | 0 corrupt | 0 corrupt |
| tool call at 42k context, 40 greedy runs | 10 diverge | 0 diverge |
| NLL vs bf16 dense layers | within ±0.006 | prose +0.008 to +0.025, code within ±0.007 |
