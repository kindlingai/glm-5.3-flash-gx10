# GLM-5.3-Flash on ASUS GX10 (GB10): TP=2, 3, 4 or 6

Need help? Join us on discord: https://discord.gg/M7XTrRJW3

nvidia/GLM-5.3-Flash-NVFP4 served by vLLM on two, three, four or six GB10
boxes (TP=2, 3, 4 or 6), over RoCE. The base is an unmodified vLLM nightly
with a few small patches, a newer FlashKDA, and
[mentat](https://github.com/mmastrac/mentat) in place of Ray. What makes it fast:

- DFlash2 speculative decoding, with a scheduler that picks the draft length
  each step from a live model of step cost and acceptance.
- RecoverSSM: drafts are verified from one saved KDA state per request, so
  more requests fit in the KV pool.
- NVFP4 MoE kernels for GB10: a decode kernel that keeps activations in 16
  bits, and a fused prefill kernel.
- A Triton sparse MLA kernel in place of FlashInfer's.
- FP8 and NVFP4 copies of the dense layers the checkpoint leaves in bf16. This
  is the only change with a measured accuracy cost: about 1% NLL on prose.
- RDMA collectives over both ConnectX-7 PCIe roots in place of NCCL's.
- Sequence-parallel prefill, gathering activations already quantized.
- The drafter's KV in its own small pool, and a pinned KV cache.
- Processed-weight snapshots, so a restart takes about 3 minutes.

Each of these lives in `experimental/`, replaces files inside the image this
repo builds, and can be turned off. [experimental/README.md](experimental/README.md)
has the details.

| | TP=2 | TP=3 | TP=4 | TP=6 |
|---|---|---|---|---|
| prefill @32k, cold | 2,931 tok/s | 3,864 tok/s | 4,934 tok/s | 4,907 tok/s |
| prefill @128k, cold | 2,871 tok/s | 3,736 tok/s | 4,776 tok/s | 4,731 tok/s |
| decode\*, code / prose / structured | 66.2 / 42.6 / 94.7 tok/s | 88.0 / 54.2 / 121.4 tok/s | 117.3 / 70.5 / 164.3 tok/s | – |
| mixed\*, 4 / 8 streams, aggregate | 83 / 110 tok/s | 107 / 138 tok/s | 144 / 194 tok/s | – |
| KV pool (fp8_e4m3) | 1.10M tokens, 8 GiB pin | 1.91M tokens, 12 GiB pin | 4.40M tokens, 26 GiB pin | 4.47M tokens, 26 GiB pin |
| longest request | 160k tokens | 524k tokens | 524k tokens | 524k tokens |
| requests decoding at once | 16 | 64 | 64 | 64 |
| boot, once snapshots exist | ~3 min | ~3 min | ~3 min | not measured |
| needle recall | 6/6 up to 128k tokens | 12/12 up to 507k tokens | 12/12 up to 507k tokens | 12/12 up to 480k tokens |

\* modified rigmark-like approach, see [gate/bench](gate/bench/).

Prefill is cold: random words, nothing cached.

We measured TP=2, 3 and 4 on boots that restored weight snapshots. TP=4 prefill
is the median of four such boots. The first boot, which loads the checkpoint
and writes the snapshots, leaves less memory free and runs slower (128k prefill
at TP=2 lost 7-15%), so restart once before measuring. TP=6 ran on other boxes
and builds, with GPU clocks locked at 1989 MHz.
[experimental/tp3/README.md](experimental/tp3/README.md) has the details.

TP=4 is the default. For two boxes, set `TP=2` in `compose/.env` on both. Each
box then holds twice the weights, so the entrypoint shrinks the KV pin, context
length, request limit and batch budget to fit.
[experimental/tp3/README.md](experimental/tp3/README.md) has the steps for TP=3
and TP=6.

`TP=RING4` runs TP=4 on four boxes cabled in a loop, with no switch: each box's
two ConnectX-7 ports go to its two neighbours. mentat places the ranks in cable
order, and every collective uses only neighbour links, relaying traffic for the
opposite box through a neighbour. It needs mentat 0.17.1 on every daemon
and a subnet per cable for each PCIe root. Through our switch, the relay hop
cost about 1% of decode. On real ring cables, another user measured RigMark
decode at 100.7 / 59.9 / 146.7 tok/s for code / prose / structured, and cold
prefill at 4,745 / 4,737 tok/s at 32k / 64k (mentat 0.17.1, every overlay,
RecoverSSM off). That is 12% below the switched TP=4 column on code, 9% on
structured, level on prose, and about 5% below our RigMark prefill.
[experimental/README.md](experimental/README.md) has the details.

[model.yaml](model.yaml) has the checkpoints and the memory footprint, and
[KNOBS.md](KNOBS.md) lists every environment variable the entrypoint reads.
[NOTES.md](NOTES.md) has the measurements and diagnosis behind the choices
here.

[kindling.json](kindling.json) has the results table above in a form a program
can read. The Kindling AI site shows this recipe's numbers from it. A PR that
changes the table changes kindling.json too.

## What you need

- **Four ASUS GX10 or other GB10 boxes** (sm_121a, 128 GB unified memory), or
  two, three or six for TP=2, 3 or 6.
  The model takes all of each box: ~91.9 GiB of GPU allocations per rank, with
  1.5-3 GiB left free (2026-09-23). Nothing else runs beside it.
- **Each box running headless** (`multi-user.target`). These boxes ship with a
  GNOME desktop enabled, and a desktop session takes memory and GPU time from
  the model: `sudo systemctl set-default multi-user.target && sudo systemctl
  isolate multi-user.target`. The preflight warns while one is running.
- **A ConnectX-7 fabric between them**, through one switch (ours is a
  MikroTik CRS812 at 200G), or cabled in a loop for `TP=RING4`, with RoCE
  working. Through a switch, each box needs a static IPv4 on its ConnectX
  interface, all in one subnet, MTU 9000. For full prefill speed also give
  the ConnectX-7's second PCIe root an address in a second subnet on every
  box (see `FABRIC_SUBNETS` in Tuning). mentatd tells the model which
  subnets these are (step 4).
- **A LAN between them** that your clients can reach. mentat identifies
  each box by its LAN address, and the API is served on it.
- **The weights on each box's local disk**, not on NFS: every rank reads the
  whole checkpoint, and an NFS mount races the network at boot.
  - [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4),
    181 GiB, at `/srv/models/glm-5.3-flash-nvfp4` (`MODEL_HOST_DIR`)
  - [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
    the drafter, 2.2 GiB, at `/srv/models/glm-5.3-flash-dflash2`
    (`DFLASH_HOST_DIR`). It is licensed CC BY-NC-ND
    4.0, non-commercial; check that before you serve it. `SPEC_METHOD=mtp`
    uses the checkpoint's own MTP head instead, slower but with no second
    download. Start that without `experimental/compose/adaptive-k.yaml`, which
    is tuned for DFlash2; the entrypoint refuses the combination.

  `hf download` is resumable:

      hf download nvidia/GLM-5.3-Flash-NVFP4 --local-dir /srv/models/glm-5.3-flash-nvfp4
      hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir /srv/models/glm-5.3-flash-dflash2
- **Docker with the NVIDIA container runtime and Compose v2** on every box,
  and `/dev/infiniband` present on the host.

## Ports

| port | what | where |
|---|---|---|
| 6379, 6380 | mentatd control and HTTP | every box |
| 6381 | mentatd-serve: the OpenAI API and merged MCP for clients | one box |
| 6382/udp | mentatd announcements | every box |
| 8002 | vLLM's OpenAI API (`/v1/chat/completions`, `/v1/models`, `/metrics`) | head only |
| 8082 | status page and MCP (`/mcp`) | every box |

Point clients at mentatd-serve on `:6381`, not at vLLM on `:8002`. It
health-gates, so a request during a boot or a reload waits instead of
failing, and it routes by model name, so another model on the same boxes
answers on the same address. mentatd-serve is a separate process from the
daemon by design: nothing that routes inference traffic runs inside the thing
that holds cluster membership.

## Layout

| path | what |
|---|---|
| `image/` | Dockerfile, entrypoint, patches, `verify-base.py`, `self-test.py`, chat template, `build.sh` |
| `compose/glm53.yaml` | the model, the same file on every box |
| `.env.example` | optional overrides for `compose/.env` |
| `kindling.json` | the results table, machine-readable, for the Kindling AI site |
| `smoketest/` | `run.sh <base> [served-name]` |
| `.submodules/spark-agent` | the status server, reached through the `vllm` symlink |
| `dev/` | not in the image: the corruption diagnosis and repros, kernel and patch tests, the step tap |

mentatd and mentatd-serve come from the [mentat](https://github.com/mmastrac/mentat)
repo, with their own compose files (step 4).

## 1. Get the repo onto every box

    git clone --recursive https://github.com/mmastrac/glm-5.3-flash-4x-gx10
    # or, in an existing clone: git submodule update --init

The status server comes from the
[spark-agent](https://github.com/mmastrac/spark-agent) submodule, and the
image build fails without it.

## 2. Build the image

    image/build.sh                       # tags the build.sh default
    TAG=spark-glm53:mine image/build.sh  # or your own tag
    BASE=... image/build.sh              # override the pinned nightly

The build context is the repo root (`docker build -f image/Dockerfile .`), so
the `vllm` symlink into the submodule stays inside it. Build on a GX10: there
is no cross-build for aarch64 here. A plain `rsync -a` copy of the repo also
builds, because it keeps the hidden `.submodules/` and the symlink; a bare
`scp -r *` misses the first. Build once and copy the image to the other three
boxes (`docker save | ssh ... docker load`, or a registry): every rank must
run the same image.

The base is `vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4`
(the tag is the full commit), which carries `glm5_next`
and DFlash2 upstream, and the patches are small anchored edits that fail the
build if the tree moves under them. The one thing the build compiles is
FlashKDA (see Patches), in a builder stage that took 98 s on a GX10.
`image/verify-base.py` then checks the finished tree. The image embeds mentat
0.17.1. Run the same version everywhere: `mentatd` and `mentatd-serve` on
every box should be 0.17.1 too.

## 3. Overrides in compose/.env (optional)

`compose/.env` is optional. Each box reads its LAN address and fabric subnets
from its own mentatd (step 4), the boxes elect a head once all of them have
registered, and the paths and the image have defaults. Put a value in
`compose/.env` only to override one. Compose reads `.env` from `compose/`,
beside the compose file, whatever directory you run it from.
`.env.example` lists the usual overrides.

| variable | unset | set it when |
|---|---|---|
| `IMAGE` | `image/build.sh`'s default tag | you built under another tag |
| `MODEL_HOST_DIR` | `/srv/models/glm-5.3-flash-nvfp4` | the checkpoint is elsewhere on this box |
| `DFLASH_HOST_DIR` | `/srv/models/glm-5.3-flash-dflash2` | the drafter is elsewhere on this box |
| `CACHE_HOME`, `LOG_DIR` | the volumes `glm53_cache` and `glm53_logs` | you want the JIT caches or the logs in a host directory |
| `HEAD_HOST` | elected: the box with the lowest LAN address | you want a fixed head: its LAN address, the same on every box. Each box then works out its own role |
| `ROLE` | from `HEAD_HOST` | never, unless you set it on every box: `head` on one, `worker` on the rest |
| `VLLM_HOST_IP` | the address mentatd tags `lan` | mentatd tags nothing `lan` on this box |
| `FABRIC_SUBNETS` | one subnet per address mentatd tags `rdma` | mentatd tags no fabric address `rdma` |
| `CLUSTER_SUBNET` | the first of `FABRIC_SUBNETS` | you set one subnet by hand, the older form of `FABRIC_SUBNETS` |

With `HEAD_HOST` unset, the boxes elect the one with the lowest LAN address
as head, so with all of them up it is always the same box. Each box logs
one `election:` line naming the candidates, the head and its own role. Set
`HEAD_HOST` on every box or on none: a box with a fixed head does not take
part in the election. The weight snapshots under `CACHE_HOME` are per TP rank,
so moving the head reshuffles the ranks, and the first boot after that loads
the checkpoint in full and writes new snapshots.

Leave every tuned knob out: each has its default in `image/entrypoint.sh`, and
a copy in `.env` silently wins over the measured value.

The volumes survive `docker compose down` and go with `down -v`. The engine
log is `/logs/vllm.log` inside the container (`vllm-head.log` or
`vllm-worker.log` when `ROLE` is set), readable through the status page's MCP
tools or `docker exec glm53 tail /logs/vllm.log`.

## 4. Start mentatd, and mentatd-serve on one box

mentat has its own repo, compose files and `.env`. On every box, in a
checkout of [mmastrac/mentat](https://github.com/mmastrac/mentat) at `v0.17.1`:

    VERSION=0.17.1 ./build.sh
    cat > .env <<'EOF'
    MENTAT_PEERS=<another box's LAN address>:6379
    MENTAT_ANNOUNCE_IFACES=en*f*np*=connectx+rdma,en*=lan
    EOF
    docker compose -f mentatd.yaml up -d

Or skip the build and add `IMAGE=mmastrac/mentatd:0.17.1` to that `.env`
(`mmastrac/mentatd-serve:0.17.1` for the router): the published images cover
arm64.

The model reads its networking from `MENTAT_ANNOUNCE_IFACES`. Tag the LAN
interface `lan` and every ConnectX interface that holds a fabric address
`rdma`. The line above does both on a GX10: `en*f*np*` matches both PCIe
roots (`enp1s0f0np0` and `enP2p1s0f0np0`, or their `f1` twins). A pattern that
misses the second root, such as `enp1s0f*np*`, leaves NCCL on one root, and
the preflight warns about it. To check a box:

    curl -s localhost:6380/status | jq .addr_tags

The daemon names the box by its default route's address, which must be the
LAN address. Set `MENTAT_NODE_IP` in mentat's `.env` when it is not. The
model container registers with the daemon on its own box
(`127.0.0.1:6379`). That daemon passes the registration on to the daemon
mentat elected as its head. With `HEAD_HOST` set, the container registers
with the daemon at `HEAD_HOST:6379` instead. Then, on one box:

    docker compose -f mentatd-serve.yaml up -d

mentat's `mentatd.yaml` explains the other settings, such as signing
announcements (`MENTAT_SECRET`, which must then be set on every box).

## 5. Start the model

On every box:

    ./glm53 up -d

`glm53` runs `docker compose` with `compose/glm53.yaml` and every overlay in
`experimental/compose`, in the order the numbers at the top were measured with
(later files win). It passes the rest of its arguments to `docker compose`, so
`./glm53 down`, `./glm53 logs -f` and `./glm53 ps` work too, and
`./glm53 -f my.yaml up -d` adds your own files after the overlays. Started with
`compose/glm53.yaml` alone (`./glm53 --stock`), the stack runs vLLM's stock
kernels at about half the speed, so the entrypoint refuses to start that way
unless `compose/.env` sets `ALLOW_STOCK=1`.

From cold, the boxes may start in any order: registration retries until the
daemon answers, and the head waits for all four GPUs before it loads. The
cold boot takes about eight minutes to reach the API. The status page on `:8082` answers from
container start, so there is something to read while it loads.

**Replacing a running stack is different.** Recreating all four ranks at once
lets a starting rank join a group that is still tearing down, and the whole
group then hangs just past NCCL setup: every rank `running`, restart count 0,
CPU ~1%, no weights loading, and nothing in any log after the
`custom_all_reduce` warning. Seen on 2026-09-10 after several rapid recreate
cycles. Take every rank down, confirm all four containers are gone, then start
them again a few seconds later:

    # on each box
    ./glm53 down --timeout 60
    # confirm on all four: docker ps -a | grep glm53  ->  nothing
    # then up again; with fixed roles, the head first

The project name is pinned to `glm53`. A stack started under another project
name (an older checkout run from a different directory, say) must be taken
down with its own compose file first, or the two collide on the container
name.

## 6. Check it

    smoketest/run.sh http://<mentatd-serve box>:6381

Eight cases, each with an answer that can be checked, because this model can
load cleanly, report healthy and serve fluent nonsense. The first fails
against any model that is not a GLM-5.3 checkpoint. Exit status is the number
of failures. See [smoketest/README.md](smoketest/README.md).

The first run after a fresh install is slow. Kernels compile the first time
each batch shape and draft length comes up, and on an empty cache that happens
inside your first requests: the first request can take minutes, and the first
pass over several concurrency levels runs well below the numbers at the top
(code at 4 streams: 127 tok/s on the first pass, 196 after a restart). The
compiled kernels are kept in `CACHE_HOME`, so restarts and later boots start
warm. Run your workload once, or restart once, before timing anything.

## Diagnostics

Each container runs spark-agent's status server on `:8082` from container
start, before the model is loadable. It does not proxy inference. The router's
`/mcp` merges its tools under the `glm53__` prefix: `node_status`,
`cluster_status`, `metrics`, `serve_args`, `throughput`,
`latency_percentiles`, `cache_sizes`, `ray_status`, `versions`, and
`find_files` / `search_files` over a fixed set of roots. `/memory` on the
status page says whether torch or something else holds a box's memory.

> **`:8082` is unauthenticated and binds `0.0.0.0`.** Anyone who can reach the
> port can read engine state and search file contents under `SEARCH_ROOTS`
> (the installed vLLM package, `/root/.cache`, `/logs` and `/cache` by
> default). Paths are resolved with `realpath` so symlinks cannot escape those
> roots, arguments are passed as argv rather than through a shell, and output
> is capped at 16 KB, but there is no authentication. Fine on an isolated
> network; put it behind something, or narrow `SEARCH_ROOTS`, on any network
> you do not control.

Two checks run at every start and print to the container log; neither stops
the boot:

- `preflight`, on every node, lists what makes the stack slow or fragile:
  a fabric port down or below 200 Gb/s, a port MTU too small for RoCE's 4096-byte path MTU (under 4200), a PCIe link
  below its maximum, one ConnectX root instead of two, link flaps, RDMA
  retransmit counters, GPU clock-limit events, other GPU processes, too little
  host memory for the TP size and KV pin, page cache the GPU cannot use yet
  (it evicts the model files' cached pages itself), swap in use, a model directory on
  NFS, no disk for the first weight snapshot, a memlock limit, and a running
  desktop session. `PREFLIGHT=0` skips it.
- `fabric check`, on every rank before vLLM starts, with the results printed
  on the head: an NCCL all-reduce over the fabric with its bus bandwidth against
  the ~95 Gb/s per ConnectX root a healthy link gives, and every version,
  override file and knob that must match across nodes, with the ones that do
  not. A fabric can link up at full rate and still move 12 Gb/s until the
  boxes' power is drained; this is where that shows. `FABRIC_CHECK=0` skips
  it; a node that cannot meet the others within `FABRIC_CHECK_TIMEOUT_S`
  (120) skips it too.

Host-level facts (GPU, PCI, RDMA counters, dmesg, systemd) come from
spark-agent's separate per-machine agent, which is not part of this recipe.
The container tools above cover the engine; diagnosing the fabric or the box
needs that agent or plain ssh.

## Tuning

The bias throughout is that **a long prefill must never block a short request**,
and that a single stream should be fast, rather than maximising aggregate
throughput at concurrency. Every value below is the entrypoint's default.
`FABRIC_SUBNETS` comes from mentatd's tags, and a patch bakes `busy_loop_s`
into the image.

| knob | value | why |
|---|---|---|
| `LONG_PREFILL_TOKEN_THRESHOLD` | 2304 | Caps one prefill's share of each scheduler step. Left at the default (budget − 256) a 120k prefill takes the whole step and a 12-token request waits 78–90 s; at 2304 it waited 4.83 s (2026-09-06). Must be a multiple of 2304, the KDA block size, because prefix caching snaps chunk ends to it: 2048 yields alternating 2048/256-token chunks. Costs nothing: the 200k prefill got *faster*. |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | Measured the same as 8192 at 200k once chunks are capped (234.1 s against 237.7 s, 2026-09-06). 8192 at TP=2, which leaves memory for its KV pin. |
| `KV_CACHE_MEMORY` | 26 GiB | 8 GiB at TP=2, 12 GiB at TP=3. Pinned, `--gpu-memory-utilization` no longer sizes the pool, and vLLM says so at startup. At 28 GiB the head sat near 1 GiB free and eight long requests had a worker OOM-killed. |
| `FABRIC_SUBNETS` | every address mentatd tags `rdma` | Each GB10's ConnectX-7 sits on two PCIe roots and one root tops out near 110 Gb/s. NCCL over both doubles all-reduce bandwidth (110 to 190 Gb/s) and took a 126k prefill from 2,412 to 2,680 tok/s (2026-09-26); decode did not move. Needs an IPv4 on the second root's interface in its own subnet, MTU 9000, the same RoCE v2 GID index on both roots, and both interfaces tagged `rdma` (step 4). To set it by hand instead, quote it and separate the subnets with spaces (`FABRIC_SUBNETS="198.18.0. 198.19.0."`). With only `CLUSTER_SUBNET` set, NCCL uses that one root. The numbers at the top use both. |
| `MAX_NUM_SEQS` | 32 | Each running request holds a KDA recurrent state for every verify position (1+k = 8 at k=7) out of the KV pool, so the pool caps this, not throughput. With RecoverSSM, which keeps one state per request, it is 64 at TP=3, 4 and 6 and 16 at TP=2. |
| DFlash2 `k=7` | | Decodes 121.9 / 91.3 / 38.7 tok/s structured / code / prose on this image without the overrides (`dev/repro/decode.py`, thinking off). On an earlier image (2026-09-23) it gave 109.8 / 88.8 / 52.6, and the checkpoint's own MTP head at k=4 gave 57.2 / 54.4 / 45.6. Costs ~41% of the KV pool: 3.44M tokens with speculation off, 2.02M with it at the same pin, on the pre-nightly image (2026-09-06). |
| `MOE_BACKEND` | `flashinfer_cutlass` | NVFP4 weights and activations, quantizing activations with the checkpoint's own input scales. The MoE kernels in `experimental/` read its processed tensors, so they need it. `marlin` keeps activations in 16 bits and ignores the input scales. It ran on earlier images and is untested on this one. |
| `SAFETENSORS_LOAD_STRATEGY` | eager | Loads in 511 s against 690 s for lazy. Unpinned, eager's buffers cost 38% of the KV cache; with the pin they cost nothing. |
| `busy_loop_s` | 0.002 | See Patches. Raises decode and drops the SoC ~20 °C. |
| `GPU_MEM_UTIL` | 0.88 | 0.90 passes every startup check and wedges the box hours later. See Troubleshooting. |

If you are serving many concurrent users instead, raise `MAX_NUM_SEQS` as far
as the KDA state allows, drop the KV pin back toward 20 GiB, and consider a
larger `LONG_PREFILL_TOKEN_THRESHOLD`: the fairness reserve costs a solo user
about 5% and buys nothing when every step has several requests in it anyway.

### The display carveout (kindling-spark-os)

The GB10 firmware reserves 2 GiB of memory for a display that the driver never
uses on this chip. [kindling-spark-os](https://github.com/kindlingai/kindling-spark-os)
runs `dispramd`, which lends that range to CUDA processes. On a host where its
socket (`/run/dispram/dispram.sock`) exists, `./glm53` adds
`experimental/compose/dispram.yaml`. That file mounts the socket and the host's
vLLM plugin, which adds the 1.99 GiB to the pinned KV pool and maps the pool's
tail onto the carveout. `DISPRAM=0` leaves it out. On DGX OS nothing changes.

| | KV without | KV with | |
|---|---|---|---|
| TP=2 (8 GiB pin) | 1.10M tokens | 1,379,896 | +25% |
| TP=4 (26 GiB pin) | 4,395,612 | 4,740,713 | +8% |
| TP=4, pin raised to 28 GiB | 4,395,612 | 5,083,602 | +16% |

The 64 KiB kernel also returns ~2 GiB of RAM, so at TP=4 `dispram.yaml` raises
the pin by 2 GiB (`KV_EXTRA_MEMORY`). With the pool full (twelve 511k-token
prompts) the head still had 9.5 GiB free and no swap in use. TP=2 has no room
for it, and TP=3 and 6 are unmeasured, so their pins stay as they are.

Every rank needs it, because vLLM sizes the pool by the smallest rank.
kindling-spark-os keeps only `/home` and the docker directories from the root
disk, so `/srv/models` does not exist there: set `MODEL_HOST_DIR` and
`DFLASH_HOST_DIR` to the checkpoints' real directories.

## Patches

Applied at build time from `image/patches/`; each asserts its anchor matches
exactly once. `image/verify-base.py` then checks the finished tree, reading
files as text (an import-based check needs a GPU driver that does not exist
during `docker build`).

| file | what it does | source |
|---|---|---|
| `glm53-flash_SM121.py` | Makes the model run on GB10 at all. On capability 12 the nightly offers only `FLASHINFER_MLA_SPARSE_SM120`, which needs the packed `fp8_ds_mla` layout with `pe_dim == 64`; this checkpoint is NoPE, so the engine dies in `concat_and_cache_mla` after a full weight load. Lists the SM90 sparse-MLA path for capability 12 and swaps FA3 for FA2. | [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark), MIT |
| `mia_retarget.py` | Two of MiaAI's edits target code that has moved since: the indexer allocation, now in `models/glm5next/nvidia/sparse_indexer.py`, and FlashInfer 0.7.0's FA2 fp8 gate. Rewrites their paths and anchors in a copy, so the vendored file stays verbatim. | ours |
| `sm90_fp8_kv_dtype.py` | The SM90 backend planned an fp8 KV cache as uint8. | ours |
| `gb10_plugin_backend.py` | Lets `VLLM_GLM53_CUDA_SPARSE_MLA` pick between the two sm_121 MLA kernels, which otherwise collide silently. | ours |
| `glm53_mtp_bf16.py` | nvidia's `config.json` says the MTP layer is NVFP4, but its weights are BF16, so MTP dies at load. Excludes it from quantisation when the checkpoint holds no scales for it. | ours |
| `glm53_eagle3_aux.py` | DFlash2 reads auxiliary hidden states from target layers 5, 14, 24, 33 and 42; upstream GLM5next does not expose them. | ours |
| `glm53_dflash2_kv_groups.py` | The GLM-5-Next KV grouper gives up on the drafter's sliding-window layers and the model dies unifying page sizes. Keeps the target's groups and adds the drafter's. | ours |
| `vllm-58720-routed-experts.patch` | Indexes the expert mapping once per load instead of scanning it for every checkpoint tensor. Merged after this nightly. | [vllm#58720](https://github.com/vllm-project/vllm/pull/58720) |
| `image/flashkda/` | Rebuilds `vllm/_flashkda_C` from FlashKDA 17a037d. The nightly's b59532f rounds the KDA recurrent state to bf16 every 16 tokens, and long prefills then corrupt tool-call output; 17a037d keeps it in fp32. | [vllm#58846](https://github.com/vllm-project/vllm/pull/58846), open |
| `glm53_reasoning_always_parsed.py` | Thinking off maps to low reasoning effort (`image/chat-template.jinja`), so the model always emits a short `<think>` block. vLLM's `glm47_moe` stops parsing `<think>` when thinking is off, and the trace would land in `content`. | ours |
| `glm47_failclosed.py` | Tool-call parser plugin, opt-in with `TOOL_PARSER=glm47_failclosed` (the default is the stock `glm47`). Checks each call against the tools the request offered; a call with a bad name or argument keys comes back as a retryable call whose sentinel argument names the mistake, instead of being dropped or leaking into history. Containment, not a cure. | ours, after [NNNtrance](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark) #7 and #11 |
| `spin_wait.py` | vLLM's shm queue spins for `busy_loop_s` (1 s) after each message; on GB10 the CPU and GPU share one power budget, so the spin costs ~20 °C and decode. 0.002 keeps the fast path; 0 (always block) measured slower. | [nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) |
| `worker_memory_cap.py` | Caps each worker's share of unified memory (`TORCH_MEM_FRACTION`, 0.92). vLLM never calls `set_per_process_memory_fraction`, so nothing else bounds a worker. | ours |
| `spark_mem_trace.py` | Names whatever crosses that bound, instead of leaving an OOM anonymous. | ours |
| `link_cuda_headers.sh` | The base ships CUDA libraries without their headers where nvcc looks, which breaks FlashInfer JIT at link time. | ours |

LibertAI's sparse-MLA kernel plugin is not mounted. To use it, build it for
sm_121a into a directory on every box and add an override file:

    # compose/ext.yaml
    services:
      glm53:
        volumes:
          - /srv/ext:/opt/ext:ro
        environment:
          - PYTHONPATH=/opt/ext
          - VLLM_GLM53_CUDA_SPARSE_MLA=1

    ./glm53 -f compose/ext.yaml up -d

`VLLM_GLM53_CUDA_SPARSE_MLA` without the plugin drops the working SM90 path
and leaves the one that fails on this checkpoint. Leave the plugin's
`VLLM_GLM53_MOE_INPUT_SCALE` unset: it applies one constant activation scale,
and this checkpoint carries real per-projection scales.

`dev/patch-tests/` holds tests for two of the patches and for the entrypoint's
mentat discovery (`_entrypoint_discovery_test.sh`); the image does not use
them. The old recipe's `gb10_topk_fallback.py` is now a flag
(`--sparse-indexer-topk-backend per_row`). `thinking_budget_guard.py` and
`glm53_kpool_tail_ring.py` (the spec-decode tail ring,
[vllm#58454](https://github.com/vllm-project/vllm/pull/58454)) are upstream.

## Troubleshooting

**Prefill at half speed, every metric healthy.** If NCCL all-reduce crawls
(~12 Gb/s) while `ib_write_bw` reads a healthy 109 Gb/s and no error counter
moves, the ConnectX-7 has latched a slow fallback state from the DAC cables
being hot-plugged. **Power off and unplug for a minute**: a reboot does not
clear it, and neither does a NIC hotplug reset. The tell is that NCCL Tree
beats Ring. Healthy on both roots is Ring 180-191 Gb/s against Tree 55-93
(NOTES.md). A single unidirectional stream
cannot see this, which is why the RDMA test passes; a ring collective, sending
and receiving at once, can.

**The box wedges hours after a clean start.** `GPU_MEM_UTIL=0.90` passes every
startup check and is worth +28% KV, then takes the box down with no ssh, no
userspace and ping only (2026-08-27). Unified memory means the CUDA allocation
*is* host memory, so none of it is reclaimable and the OOM killer cannot help.
0.88 is the default.

**Long output repeats or skips when thinking is off.** "Count from 1 to 200"
comes back as `35 36 37 37`, or jumps ahead, or starts copying the prompt.
This is the model, not the stack. GLM-5.3-Flash has no non-thinking mode: its
official template always opens `<think>` under `Reasoning Effort: Max`, and an
empty `<think></think>` is out of distribution. Every checkpoint (nvidia,
RedHatAI, the official FP8), both MoE kernels, TP=2 and TP=4, and Hugging
Face's own `glm5_next` implementation fail the same way (2026-09-23). The
image's template (`image/chat-template.jinja`) maps `thinking: false` /
`enable_thinking: false` to `Reasoning Effort: Low` instead: a few dozen tokens
of reasoning and a clean answer. `reasoning_effort` (`low`, `high`, default
`max`) also works directly, at the top level of the request. If long output
still breaks with thinking off, check that `CHAT_TEMPLATE` isn't pointing at the
checkpoint's own template.

**Intermittent corrupted tokens.** The LibertAIDAI modelopt NVFP4 build emits
them mid-word, inside rare tokens: invisible in English, reproducible with a
Korean prompt, and identical under both MoE kernels, both attention backends,
and with speculation off (2026-09-06). The nvidia build this recipe uses and
the compressed-tensors builds (RedHatAI NVFP4, INT4 AWQ) are clean on the same
stack. Independently reported by tonyd2wild.

**Workers restart forever with `group 'glm53' already has an active driver session`,
and the head waits for GPUs.** More than one node is running as the head: every
head starts a driver, mentat allows one per group, and the others exit and
restart while the real head never sees its workers. With roles set by hand,
set `ROLE=worker` in `compose/.env` on every node except the head, keep
`HEAD_HOST` the head's address everywhere, then take all four down and start
them again. An image built from this tree refuses a head whose `VLLM_HOST_IP` is not
`HEAD_HOST`, with a FATAL line naming the fix. Elected boxes agree on one
head, but a box with `ROLE=head` in its `.env` beside them is a second one:
set the roles on every box or on none. If the error remains with the roles
right, an earlier head's session is still held: with all four down, run
`mentat stop --group glm53` against the head's daemon (or restart its mentatd).

**Boot waits at `electing: waiting for 4 agents in group glm53, have 3`.** A
box has not registered. Every box starts its agent before the election, so
the missing one is not running, cannot reach its own mentatd, or runs under
another `MENTAT_GROUP`. `docker exec mentatd mentatd status --group glm53`
on any box lists the agents mentat's head holds: each box should appear once
with `alive=true`.

**Boot hangs at `waiting for 4 GPUs, have 1`, with `HEAD_HOST` set.** Every
box must set `HEAD_HOST` to the head's address. Seen 2026-09-06 on a
mentat that held each agent on the daemon it registered with: point a box at
its own daemon and the head lists the agent yet marks it `alive=false
degraded=true`, while that box's own daemon sees only its own agent. Confirm
with `docker exec mentatd mentatd status` on the head: every agent should read
`alive=true`.

**The model waits in placement with nothing in the container log.** If
`MENTAT_ANNOUNCE_IFACES` tags a link `rdma` and the daemons' probes over that
link fail, mentat will not place the four ranks, and the group waits for
`MENTAT_PG_PENDING_TIMEOUT_MS` (10 minutes). `pending_reason` in the
daemon's `/status` (HTTP, port 6380) names the constraint. Fix the link, or set `MENTAT_ISLAND_PLACEMENT=off` and restart
the daemons.

**The second boot deadlocks after NCCL setup.** A persisted FlashInfer
autotune cache keys some MoE entries per rank, so rank 0 loads a tuning the
others lack and they wait on each other forever (2026-09-23). The entrypoint
keeps the cache ephemeral at TP>1 and wipes it at every start, which costs
about two minutes of autotune a boot; `AUTOTUNE_CACHE=persist` brings the old
behaviour back.

**The first requests after a fresh install are slow.** Kernels compile the
first time each batch shape comes up, mid-serve, and the first request can
take minutes. It happens once per shape per `CACHE_HOME`: clearing that cache
or its volume brings it back. Run the workload once before timing anything.

**A value in `.env` does nothing.** `.env` only substitutes into the compose
file; a variable reaches the container only if `compose/glm53.yaml` lists it
under `environment`. And `.env` must be in `compose/`, not in the repo root.

**NCCL fails every TP init after a reboot.** Do not pin `NCCL_IB_GID_INDEX`.
The RoCE GID table is indexed by (address, RoCE version) in the order
addresses appeared, so boxes do not agree and a removed address leaves a hole;
the right index moved from 6 to 5 on one box across a reboot (2026-08-26). The
entrypoint derives it at every start. To read a table:

    for i in $(seq 0 9); do p=/sys/class/infiniband/<dev>/ports/1; \
      echo "$i $(cat $p/gid_attrs/types/$i) $(cat $p/gids/$i)"; done

The right entry is the RoCE v2 one for the box's static fabric address.

## Credits

[mmastrac/mentat](https://github.com/mmastrac/mentat) ·
[mmastrac/spark-agent](https://github.com/mmastrac/spark-agent) ·
[tonyd2wild](https://github.com/tonyd2wild) ·
[MiaAI-Lab](https://github.com/MiaAI-Lab) (sm_121 patches, see
`image/patches/LICENSE.MiaAI-Lab`) ·
[tonyliu312](https://github.com/tonyliu312) (the KV pin) ·
[nacyot](https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) (spin wait) ·
[alexellis](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless),
and [RigMark](https://github.com/alexellis/rigmark) for the gate ·
[incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (the DFlash2 drafter) ·
[NNNtrance](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark) (the fail-closed
tool parser idea) ·
[Chuck](https://github.com/chuck-ads) (the RDMA path MTU, [#6](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/pull/6), and the
streamed tool call and page cache reports, [#7](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/issues/7)) ·
[ayayalar](https://github.com/ayayalar) (the first-boot stall report behind the
fabric check, [#5](https://github.com/mmastrac/glm-5.3-flash-4x-gx10/issues/5)) ·
[stevededrick](https://github.com/stevededrick) (the arx idle backoff,
[#59](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/59)) ·
[sdougbrown](https://github.com/sdougbrown) (the EAGLE block drop,
[#54](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/54), and the stale JIT
lock, [#53](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/53)) ·
[calvarado2004](https://github.com/calvarado2004) (the arx ring at three ranks,
[#57](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/57), and the stock tool
parser default, [#58](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/58)) ·
[tfolkman](https://github.com/tfolkman) (how a killed worker leaves the JIT lock,
[#53](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/53)) ·
[strusty](https://github.com/strusty) (megamoe and rows with no routed experts,
[#63](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/63), and the DFlash
block drop at TP=3, [#66](https://github.com/kindlingai/glm-5.3-flash-gx10/issues/66))

The files in `experimental/` that replace vLLM and FlashInfer files keep their
Apache-2.0 headers, and [experimental/README.md](experimental/README.md#sources)
lists which ones they are.
