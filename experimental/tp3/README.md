# TP=3 (three boxes)

GLM-5.3-Flash at tensor parallel 3. Four dimensions do not divide by 3, so
they are zero-padded; the padding is exact (zero heads and zero expert
columns add nothing) and the checkpoint on disk is not rewritten.

| | stock | padded | per rank |
|---|---|---|---|
| MLA heads | 64 | 66 | 22 |
| KDA heads | 64 | 66 | 22 |
| MoE intermediate | 2048 | 2304 | 768 |
| vocab | 154880 | 154944 | 51648 |
| DFlash2 drafter q / kv heads | 32 / 8 | 36 / 9 | 12 / 3 |

## Pieces

- `make_padded.py SRC_MODEL DST_MODEL SRC_DRAFT DST_DRAFT`: the target dir is
  hardlinks of every checkpoint file plus a `config.json` with the padded sizes
  and `tp_pad_orig`; the drafter (small) is padded offline. Run it inside the image.
- `tp3pad.py`: pads the target's tensors as `load_weights` reads them (hooked
  from `experimental/fixes/model.py`; a no-op when the config has no `tp_pad_orig`).
- `vocab_parallel_embedding.py`: pads the vocab on to a multiple of 64 x TP.
- `experimental/fixes/gb10_sparse_mla.py`: the Triton sparse-MLA kernel took
  `tl.arange(0, H)` over heads, which needs a power of two; 22 heads per rank
  now run in the next power-of-two tile with a head mask.
- `experimental/snapshot/dense_fp8.py`: the KDA in_proj at TP=3 has N=8726
  rows, not a multiple of 16, and stayed bf16; rows are now zero-padded to 16
  and the output trimmed.
- `image/entrypoint.sh`: a `TP=3` case. KV pin 12 GiB (26 GiB had the
  host OOM-killer take rank 0 during graph capture), block size 3456 (vLLM
  lifts it to 3584, 14% mamba page padding against 46% at the 4608 it picks
  from 2304).

## Running it

Everything else is the four-box setup from the main README (image, mentatd,
fabric), on three boxes.

1. **Padded model dirs, on every box.** The target dir is hardlinks, so it must
   sit on the same filesystem as the checkpoint; the drafter is copied and
   padded. Run inside the image this repo builds:

       docker run --rm -v /srv/models:/srv/models --entrypoint python3 \
         -v "$PWD/experimental/tp3":/tp3:ro spark-glm53:v9 /tp3/make_padded.py \
         /srv/models/glm-5.3-flash-nvfp4 /srv/models/glm-5.3-flash-nvfp4-tp3 \
         /srv/models/glm-5.3-flash-dflash2 /srv/models/glm-5.3-flash-dflash2-tp3

2. **compose/.env on every box** (tp3.yaml takes `TP` from the `.env`, defaulting
   to 3):

       MODEL_HOST_DIR=/srv/models/glm-5.3-flash-nvfp4-tp3
       DFLASH_HOST_DIR=/srv/models/glm-5.3-flash-dflash2-tp3

3. **Start** with every override, `recoverssm.yaml` included, and `tp3.yaml` last (it
   restates adaptive-k.yaml's `EXTRA_ARGS` and adds a data-parallel vision
   tower, whose 16 heads do not split by 3):

       ./glm53 -f experimental/compose/tp3.yaml up -d

   The first boot pads while it
   loads and writes a weight snapshot under its own tag (`tp3pad-...`, set in
   tp3.yaml; a TP=4 snapshot must never restore here); later boots restore it.

The entrypoint's `TP=3` case sets the KV pin (12 GiB), block size (3456),
`MAX_NUM_SEQS` 64 with RecoverSSM (32 without) and the adaptive-k starting cost. With RecoverSSM on, the head keeps ~6.6 GB of
host memory free while serving; a 16 GiB pin would leave ~2.6 GB, so do not raise it.

## Triangle, no switch (TP=RING3)

The same padded checkpoint on three GB10 boxes cabled as a triangle: each
box's two ConnectX-7 ports go to its two neighbours, one cable per port, and
the third box's port cables back to the first. No switch. Set `TP=RING3` in
`compose/.env`; the entrypoint maps it to its `TP=3` case (KV pin, block
size, `MAX_NUM_SEQS`, weight padding) and adds the ring wiring.

- **Addresses**: each cable carries two 100G halves, one per PCIe root, and
  root 0 faces root 0 on the neighbour, root 1 faces root 1 (check with
  `ib_write_bw -x 0` between the unaddressed devices: the link-local RoCE v1
  GID needs no address). Give each half its own `/24`, MTU 9000: six subnets
  for three cables. A device joins the fabric only when its netdev carries an
  IPv4. `fabric_ring.py` expects both roots of each port, as on RING4.

- **Compose overlays**: the TP=3 list above, with `TP=RING3` in
  `compose/.env` (tp3.yaml inherits it).

- **mentat**: 0.17.1 places ring claims of any size (a ring closes only at
  three members or more), so `MENTAT_CLAIM_LAYOUT=ring` needs no member
  count; the entrypoint sets it and the claim takes the three bundles of the
  TP=3 group. Tag the LAN interface `lan` and both roots' fabric netdevs
  `rdma` in `MENTAT_ANNOUNCE_IFACES` (step 4 of the main README), for example
  `en*f*np*=connectx+rdma,en*=lan`. Each rank
  gets its neighbours' addresses and the local interface toward each from
  mentat, and `fabric_ring.py` finds both roots' devices behind each.

- **NCCL**: subnet-aware routing (`NCCL_IB_SUBNET_AWARE_ROUTING=1`, no pinned
  GID index), not a ring graph. NCCL takes one device per channel for both
  directions of a graph, so a graph cannot receive from one neighbour and
  send to the other on different cables. The GID index differs per cable and
  moves on reboot; NCCL finds each device's address GID itself.

- **arx**: ring mode, with `ARX_RING_PREV_HCAS` / `ARX_RING_NEXT_HCAS` from
  `fabric_ring.py`. In a ring of three every peer is a neighbour, so nothing
  is relayed. arxbig's ring reduce-scatter assumes four ranks, so the
  entrypoint sets `VLLM_ARXBIG=0` and prefill-sized collectives stay on NCCL.

## Tried and not kept

At TP=3, on this stack, one change at a time against the defaults above:
sequence parallel off (-16% prefill), `VLLM_ARXBIG_AG=1` (-26 to -30%
prefill), `max_num_batched_tokens` 8192 (-4% at 22k), megamoe
`VLLM_MEGAMOE_MAX_TOKENS=16` and its tile variants (within noise), 72 MLA
heads instead of 66 (-10% prose decode, measured on the PR #4 base). A MoE intermediate of 2112 (704 per
rank) would pad less, but megamoe needs a multiple of 128 per rank.

## Tests

`dev/patch-tests/_glm53_tp3pad_test.py` (CPU) and
`dev/patch-tests/_glm53_tp3_sparse_mla_test.py` (one GB10); each file says how
to run it.

## Measured

Two sets of three boxes (one 200G switch, both ConnectX ports), each booted
clean with the defaults above (RecoverSSM on, `MAX_NUM_SEQS` 64), on this
branch over main 6db84d4 plus the capture-size fix from its own PR
(`tp-cudagraph-tiers`: an explicit capture cap dropped adaptive-k's decode
sizes 3, 5, 6, 10, 12 and 15). KV held 1.48M tokens then; with #16 the head log reports 1,905,585. On main 2288993 plus #20 and the zeroer fix, set 1 re-measured within noise: RigMark 43.9 / 76.2 / 113.7, cold prefill 8k / 32k / 64k 3312 / 3637 / 3587.

| | set 1 | set 2 |
|---|---|---|
| RigMark prose / code / structured, tok/s | 43.8 / 75.7 / 113.8 | 44.0 / 76.2 / 114.0 |
| RigMark cold prefill 8k / 32k / 64k, tok/s | 3295 / 3602 / 3565 | 3325 / 3610 / 3583 |
| `gate/prefill.py` cold 32k / 128k, tok/s (2 seeds) | 3574, 3533 / 3440, 3449 | 3561, 3554 / 3422, 3437 |
| `gate/conc_workload.py code` 1 / 2 / 4 / 8 streams, tok/s | 86.5 / 101.8 / 137.4 / 163.7 | 87.4 / 100.5 / 153.7 / 170.2 |
| `conc_workload.py mixed` 8 / 16 / 32 / 48 streams, tok/s | 134.1 / 195.5 / 262.7 / 335.9 | 141.0 / 200.3 / 260.0 / 333.9 |
| `dev/repro/needle.py` 32k to 480k | 12/12 | 12/12 |

At 48 streams, `MAX_NUM_SEQS` 64 gave +24% over 32 (269 tok/s); single-stream
decode is the same at both. A 30-minute soak of the same TP=3 changes on main
14cc721 (`MAX_NUM_SEQS` 32, 16 mixed streams back to back, 824 requests) had
no errors or restarts.

## TP=6 (six boxes)

The same padded checkpoint splits by 6: 11 MLA heads, 11 KDA heads and 384
MoE columns per rank. The pieces above cover it: the KDA in_proj has N=4491
per rank and takes the same row padding to 16, and the sparse-MLA kernel runs
11 heads in a 16-row tile. Only the drafter needs a different pad: its 8 kv
heads need 12 (48 q heads) to split by 6.

1. **Padded model dirs, on every box**, as step 1 above but with the drafter
   padded for 6:

       ... /tp3/make_padded.py SRC_MODEL DST_MODEL SRC_DRAFT DST_DRAFT \
         --draft-heads 48 --draft-kv-heads 12 --tp 6

2. **compose/.env on every box**: `MODEL_HOST_DIR` / `DFLASH_HOST_DIR` at the
   padded dirs, as for TP=3.

3. **Start** with the TP=3 command, `tp6.yaml` in place of `tp3.yaml`. It sets
   `TP=6` and its own snapshot tag (`tp6pad-...`).

The entrypoint's `TP=6` case sets the TP=4 KV pin (26 GiB; weights are ~1/6
per rank), block size 1792, `MAX_NUM_SEQS` 64 with RecoverSSM and the
adaptive-k starting cost. With 11 KDA heads per rank the mamba page is ~1573
tokens of attention page: the default 2304 pads it 46%, 1792 (the smallest
multiple of 256 above it) 14%, for a 1.7% bigger pool (4.47M tokens) and a
faster cached TTFT (0.677 against 0.903 s).

Tried and not kept at TP=6 (screened on an earlier build): sequence parallel off (-25% prefill),
`max_num_batched_tokens` 32768 (no prefill gain, and it needs a bigger arxbig
slot), `VLLM_MEGAMOE_MAX_TOKENS=16` (within noise).

`dev/patch-tests/_glm53_tp3_sparse_mla_test.py` includes H=11.

Measured on six boxes (one 200G switch, both ConnectX ports, GPU clocks locked
at 1989 MHz), booted clean with the entrypoint defaults on this branch's TP=6
changes over main 3df22b8 plus the padded-config import guard (#24):

| | |
|---|---|
| RigMark prose / code / structured, tok/s | 66.5 / 120.3 / 176.4 |
| RigMark cold prefill 8k / 32k / 64k, tok/s | 4776 / 4902 / 4903 |
| `gate/prefill.py` cold 32k / 128k, tok/s (2 seeds) | 4900, 4914 / 4737, 4724 |
| `gate/conc_workload.py code` 1 / 2 / 4 / 8 streams, tok/s | 145.0 / 153.8 / 218.6 / 281.7 |
| `conc_workload.py mixed` 8 / 16 / 32 / 48 / 64 streams, tok/s | 196.6 / 298.7 / 415.3 / 506.7 / 596.5 |
| `dev/repro/needle.py` 32k to 480k | 12/12 |

A 30-minute soak at 64 mixed streams (2700 requests) had no errors,
preemptions or restarts.

## On cables, with no switch

Three boxes with one cable per pair also form a ring. `TP=RING3` sets it up:
subnet-aware NCCL routing, arx in ring mode (`VLLM_ARX_RING=1`) with arxbig
off, since arxbig's ring does not take three ranks
([experimental/README.md](../README.md)). The section below has the start
command; the RING3 section above has the fabric layout.

Through a switch, with both neighbours on one port, ring and mesh at three
ranks gave the same bits and the same all-reduce latency: 14.5 / 21.5 / 45.7 /
72.9 us in ring mode against 13.6 / 24.2 / 44.9 / 72.6 in mesh mode, at
8 / 64 / 256 / 512 KB.

@calvarado2004 measured a cabled triangle on 45b438b, with every overlay and
RecoverSSM off, from their own launcher with the `TP=3` case's arguments and on
their own harness, so these aren't comparable with the main README's table
([#52](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/52)):

- arx took decode from 116.2 / 88.9 / 47.6 to 128.9 / 94.5 / 49.1 tok/s
  (count / code / prose) and cold prefill at 21.9k from 2,503 to 2,938 tok/s.
- With `--max-model-len 1048576` at the 12 GiB pin the KV pool held 1,849,306
  tokens, and needles at 291,517, 600,405 and 955,301 tokens were found in
  104, 236 and 422 s.
- The first boot took 753 s and later boots 184-204 s; `smoketest/run.sh`
  passed 8/8.

### RING3 start command

`compose/.env` on every box:

    TP=RING3
    HEAD_HOST=<head LAN address>

Then, on every box:

    ./glm53 -f experimental/compose/tp3.yaml -f experimental/compose/seqcap32.yaml up -d

`seqcap32.yaml` is optional. It caps requests at 32 and graphs at 256 tokens, and spends the ~5 GiB that frees on KV
(18 GiB pin), for long-context agents that never run more than 32 requests.

Synthetic agent turns at temperature 0, thinking off. The first three rows are from 2026-09-30 on main 6861dd9;
the last is from 2026-10-03 on main c748079 (with #50's draft cut and #59).

| agent turns at 131k context, whole-turn tok/s | 1 agent | 3 agents |
|---|---|---|
| 12 GiB pin, 64 requests, arx off (1,905,585 KV tokens) | 45.3 | 25.1 |
| `seqcap32.yaml`, arx off (2,960,882) | 49.7 | 23.8, 23.5 |
| `seqcap32.yaml`, arx ring (2,960,882) | 53.6 | 24.4, 25.3 |
| `seqcap32.yaml`, arx ring, c748079 (2,960,882) | 54.7 | 27.1, 26.1 |

Decode per stream at 3 agents, 2026-09-30: 34.4 / 30.3, 31.8 / 43.0, 37.0. Cold prefill with arx: 130k tokens
3,413 tok/s, 262k 3,210 tok/s (2,870 without); on c748079 3,396 and 3,201. `smoketest/run.sh` 8/8 on both.
