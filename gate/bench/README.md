# bench.py: decode speed with error bars

A decode benchmark built for speculative decoding, where speed depends on how easy each
answer is to draft. A single run per prompt can't separate two setups closer than about
10%. This one repeats each workload with different prompts and reports how sure it is.

The approach follows [RigMark](https://github.com/alexellis/rigmark): fixed prompts
behind a per-run tag, warmups, and checks on the output. The code is our own.

## Workloads

| workload | what runs |
|---|---|
| code | one stream, 512 tokens, a different coding prompt each run |
| prose | one stream, 512 tokens, a different explanation prompt each run |
| structured | one stream, 512 tokens, a different number list each run |
| counting | one stream, count to 200; any wrong line fails the run |
| mixed4, mixed8 | 4 or 8 concurrent streams of code, prose and structured prompts |

Every workload gets 2 warmup runs that aren't recorded, then 16 recorded runs (8 batches
for mixed). Requests use temperature 0 with thinking off. Each carries a run tag fixed by
the series name, the workload and the run number. So run 5 of `code` sends the same
prompt and tag on every setup, and a server can't answer it from its prefix cache. Every
run is also checked for Hangul and replacement characters, which is how corruption on
this model has shown up before.

## What it reports

For each run: output tokens, time to first token, decode time (first token to last), and
the speculative-decoding counters from `/metrics`. Those give tokens accepted per verify
step, which depends on how easy the output was to draft, and time per step, which depends
on the setup and on how many drafts adaptive-k chose to verify.

For each workload:

- tok/s: total output tokens / total decode time, with a bootstrap 95% interval
- spread: the run-to-run standard deviation, as a share of the mean
- detects: the smallest difference an unpaired comparison of this many runs would find
  (95% confidence, 80% power). Pairing does much better, because the spread is mostly
  prompt-to-prompt difference.
- outliers: runs with a modified Z-score above 3.5

`compare` pairs two result files run by run, and reports B / A as a geometric-mean ratio
with a bootstrap 95% interval and a two-sided exact Wilcoxon signed-rank p-value,
Holm-corrected across workloads. It splits the ratio into tokens per step and time per
step, so you can tell a faster setup from easier answers.

## Usage

```
python3 gate/bench/bench.py run --label mysetup --out mysetup.json
python3 gate/bench/bench.py compare base.json mysetup.json
```

`GATE_URL` (default `http://127.0.0.1:8002`) and `GATE_MODEL` (default `glm53`) pick the
server. Runs pair only within one `--series`. A full run takes about 15 minutes at TP=4.

## Check: two boots of the same build

Two separate TP=4 boots of one build, compared with `compare`:

| workload | B / A | 95% interval |
|---|---|---|
| code | 0.997 | 0.989 to 1.003 |
| prose | 0.996 | 0.992 to 1.000 |
| structured | 1.004 | 1.000 to 1.010 |
| counting | 1.000 | 0.998 to 1.002 |
| mixed4 | 1.015 | 0.996 to 1.037 |
| mixed8 | 1.003 | 0.978 to 1.032 |

None differ (Holm-corrected p ≥ 0.95). Single-stream code varies 15% from run to run, and
pairing still puts two boots within about 1% of each other.

## Results

RESULTS
