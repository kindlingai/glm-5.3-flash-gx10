# Working on this repo

This is the recipe I run GLM-5.3-Flash with on four GB10 boxes. People copy it,
so a change that makes it slower or subtly wrong costs more than the change
was worth. Most of what follows is about proving a change is neither before it
lands.

If you're an agent working here, read this whole file first. If you're a
person, the same rules apply to you.

## How changes land

Every change goes up as a pull request against `main`. One change per PR,
with the gate results in the description. I merge.

Before anything leaves your machine, scrub it. No IP addresses, hostnames, box
names, usernames or paths from your own setup. The compose files use relative
paths and `.env` values for a reason.

Don't push to a PR branch until the change passes the gate below. Pushing
something "to see what CI thinks" doesn't work here: there's no CI that can run
this model, and the PR is the record of what was measured.

A docs-only change doesn't need the gate. A setup fix that helps someone
starting fresh (an entrypoint guard, a `.env.example` comment, a
troubleshooting entry) needs its own logic tested, not the speed suite.

## The gate

A change passes when:

1. every quality check below is inside its bar, and
2. prefill and decode are at par or faster than the last full gate, on
   repeated samples, not one run.

Run it on the head node, from the repo root, with the stack up and nothing
else using it:

```
gate/run.sh full my-change
```

`gate/run.sh quick my-change` is the reduced gate. It skips HumanEval, the
count probe, RigMark and half the needle and prefill samples. It's fine for
checking your own work as you go. It's only enough for a PR when I've said
so for that change.

The full gate takes a couple of hours. GSM8K and HumanEval need their data in
`QUALITY_DIR` (see `experimental/quality/quality.py`). RigMark needs a checkout
of [RigMark](https://github.com/alexellis/rigmark) at c5a0db0 in
`RIGMARK_DIR`. The gate skips it without one, and then the PR isn't done.

### Quality

| Check | Script | Bar |
|---|---|---|
| Smoketest | `smoketest/run.sh` | 8/8 |
| NLL on three fixed texts | `gate/nll.py compare` | notes +0.008 to +0.025, the others within ±0.007 |
| GSM8K, first 250 | `experimental/quality/quality.py` | 95.5 to 98.5% |
| HumanEval | `quality.py`, then `run_he.py` with no network | 154 to 156 of 164 |
| Count to 200, 5 runs | `gate/count.py` | 0 corrupt |
| Tool call at 42k context, 40 runs | `dev/repro/toolcall_corruption.py` | 0 diverging on the last gate (stock: 10) |
| Agent tools | `experimental/quality/agent_tools.py` | 10/10, 0 bad args |
| Needle, 32k to 480k | `dev/repro/needle.py` | 12/12 |
| RigMark output gates | RigMark `bench.py` | 6/6 |

The bars are wide because greedy decoding here isn't reproducible across
boots. FlashInfer's autotuner picks different kernels on each boot, the
rounding changes, and borderline answers flip. HumanEval has about a dozen
problems that pass or fail from boot to boot, and
`experimental/quality/he_detail.py` shows which ones flipped between runs. GSM8K runs 8 requests at once, so
batching changes run to run too: the same config on the same boot has scored
96.4 and then 98.4.

So one number near an edge proves nothing. A real regression either falls well
outside a bar or moves several checks together. The last one I caught (NVFP4
for the attention projections) took GSM8K to 95.6 and HumanEval to 154 at
once. When you're unsure, compare the two paths inside one boot, where
generated text is byte for byte deterministic.

NLL isn't. `nll-ref.json` was recorded with the dense layers left in bf16
(`fp8.yaml` left out), and FP8 dense moved the notes text up by about 0.015.
Since then it has read anywhere from +0.008 to +0.025, sometimes 0.006 apart
on the same boot. Watch the other two texts, and watch for all three moving
the same way. To add a text, drop it in `gate/texts/`, boot without
`fp8.yaml`, and record a new reference with `gate/nll.py record`.

### Speed

| Measure | Script | Samples |
|---|---|---|
| Prefill, 32k and 128k, cold | `gate/prefill.py` | 4 (mean) |
| Decode, structured / code / prose | `gate/decode.py` | median of 3 |
| Decode, older prompt set | `gate/decode_stream.py` | median of 3 |
| Concurrent streams 1 to 16, mixed prompts | `gate/conc.py` | 1 |
| Preemptions | `/metrics` | must stay 0 |

Par means within the noise of the last full gate: about 0.5% for prefill and
1 to 2% for decode. Anything lower is a loss, even if it's small. I would
rather not have a change than carry a 1% regression forever.

The last full gate (2026-09-28):

| | |
|---|---|
| Prefill 32k / 128k | 4,946 / 4,750 tok/s |
| Decode structured / code / prose | 170.3 / 120.8 / 65.8 tok/s |
| Decode, older prompts | 162.2 / 127.6 / 89.2 tok/s |
| Concurrent 1 / 2 / 4 / 8 / 16 | 125 / 103 / 146 / 192 / 251 tok/s |
| RigMark code / prose / structured | 107.9 / 61.7 / 157.1 tok/s |

When your PR lands with better numbers, update this table in the same PR.

## Extra checks for some changes

The gate covers the common cases. Some changes need more.

Kernels and anything else that changes numerics get a bit-exact or
tolerance test in `dev/patch-tests/` or `dev/kernel-tests/` first, against the
path they replace. Say in the PR which one it is and why a tolerance is safe,
if you needed one.

Anything that touches scheduling or runs differently with more than one
request gets the concurrency sweep too. `gate/conc_workload.py` runs one
workload (code, prose, structured or mixed) at the concurrency levels you give
it, each stream with its own prompt:

```
for w in code prose mixed; do python3 gate/conc_workload.py $w 1 2 4 8 16 32 64; done
```

Report it against the same sweep on `main`. The targets I care about are 155
tok/s at 4 streams and 350 at 16, on the mixed set.

Memory and KV changes need the boot log's `GPU KV cache size` line before and
after, and a run at `MAX_NUM_SEQS` streams with vLLM's `Running:` count, to
show they all fit.

Anything that changes how weights are processed needs a new
`VLLM_WEIGHT_SNAPSHOT_TAG`, or a boot without restoring. A shape change fails
the restore loudly. A same-shape change restores stale weights and says
nothing.

## Things that have cost a day

Measure prefill on cold prompts. A repeated prompt measures the prefix cache:
a 200k probe that took 198 s cold came back in 6 s warm. `gate/prefill.py`
draws fresh text for each run.

Count tokens, not stream events. With speculative decoding one event carries
several tokens, and events per second undercount decode badly.

Keep the FlashInfer autotune cache ephemeral. If the four ranks read caches
that differ, TP=4 deadlocks.

Don't benchmark on a box with a desktop session running. It shares the
unified memory and the power budget with the model, and the entrypoint warns
about it.

A worker at `ROLE=head` crash-loops on mentat's "group already has an active
driver session" while the real head waits for GPUs. The entrypoint refuses to
start that way now, but it's the first thing to check when a boot hangs.
