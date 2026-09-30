#!/usr/bin/env python3
"""Decode benchmark with repeated runs, warmups, run tags and error bars.

    bench.py run [--label L] [--series NAME] [--runs 16] [--batches 8] [--warmup 2] [--out FILE]
    bench.py compare A.json B.json

run: every workload gets --warmup unrecorded runs, then its recorded runs. Recorded run i
of a workload always sends the same prompt with the same run tag, both fixed by the series
name, so two setups measured in one series can be paired run by run. The tag keeps a
server from answering a repeat from its prefix cache.

  code, prose, structured   one stream, 512 tokens, a different prompt each run (prompts.py)
  counting                  one stream, count to 200; any wrong line fails the run
  mixed4, mixed8            4 or 8 concurrent streams of code, prose and structured, one
                            batch per run (--batches)

Each run records its output tokens, time to first token, decode time (first token to last)
and the speculative-decoding counters from /metrics. Those split decode speed into tokens
accepted per verify step, which depends on how easy this output was to draft, and time per
step. Time per step also grows with the drafts verified per step, which adaptive-k chooses,
so both are reported.

For each workload, run prints output tokens / decode time over all runs with a bootstrap
95% interval, the run-to-run spread, and the smallest difference that an unpaired
comparison of this many runs would detect (95% confidence, 80% power). The spread includes
the difference between prompts, which pairing removes, so a paired compare detects smaller
differences than this. It flags outlier runs by modified Z-score (above 3.5).

compare pairs the two files' runs by workload and index, and reports B / A as a geometric
mean ratio with a bootstrap 95% interval and a two-sided Wilcoxon signed-rank p-value,
Holm-corrected across workloads.
"""
import argparse, hashlib, json, math, os, random, re, statistics, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gate/prompts.py
from prompts import prompt_for

BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
SYSTEM = "Answer the request. Ignore the run tag."
COUNT = "Count from 1 to 200, one number per line, digits only."
SINGLE = {"code": 512, "prose": 512, "structured": 512, "counting": 768}
MIXED = {"mixed4": 4, "mixed8": 8}
BOOT = 2000
Z95, Z80 = 1.96, 0.8416
# Hangul and the replacement character: GLM-5.3 corruption has shown up as both.
SUSPECT = re.compile("[가-힣�]")


def run_tag(series, work, i):
    return hashlib.blake2b(f"{series}/{work}/{i}".encode(), digest_size=6).hexdigest()


def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}\S*\s+([0-9.e+]+)$", raw, re.M))
    return (get("spec_decode_num_accepted_tokens_total"), get("spec_decode_num_drafts_total"),
            get("spec_decode_num_draft_tokens_total"))


def stream(prompt, tag, max_tokens):
    """One streamed request: output tokens, send and first-token and last times, text."""
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"Run tag {tag}.\n\n{prompt}"}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    sent = time.monotonic()
    first, tokens, text = None, 0, []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("choices"):
                delta = d["choices"][0].get("delta", {})
                piece = delta.get("content") or delta.get("reasoning") or ""
                if piece and first is None:
                    first = time.monotonic()
                text.append(delta.get("content") or "")
            if d.get("usage"):
                tokens = d["usage"]["completion_tokens"]
    return tokens, sent, first, time.monotonic(), "".join(text)


def check(work, text):
    if SUSPECT.search(text):
        return "suspect characters"
    if work == "counting" and [l.strip() for l in text.strip().split("\n")] != [str(k) for k in range(1, 201)]:
        return "wrong count"
    return "" if text.strip() else "empty"


def run_single(work, series, i, warm=False):
    tag = run_tag(series, work, f"warm{i}" if warm else i)
    prompt = COUNT if work == "counting" else prompt_for(work, i)
    a0, d0, k0 = metrics()
    tokens, sent, first, end, text = stream(prompt, tag, SINGLE[work])
    a1, d1, k1 = metrics()
    decode = end - first if first else float("nan")
    steps = d1 - d0 or max(tokens - 1, 1)  # no speculative decoding: one token per step
    return {"work": work, "i": i, "tag": tag, "tokens": tokens, "ttft": first - sent if first else None,
            "decode_s": decode, "tps": tokens / decode if decode > 0 else 0.0,
            "tok_per_step": 1 + (a1 - a0) / steps if d1 > d0 else 1.0, "ms_per_step": 1000 * decode / steps,
            "drafts_per_step": (k1 - k0) / steps if d1 > d0 else 0.0, "problem": check(work, text)}


def run_mixed(work, series, b, warm=False):
    n = MIXED[work]
    jobs = [(prompt_for("mixed", b * n + s), run_tag(series, work, f"warm{b}.{s}" if warm else f"{b}.{s}")) for s in range(n)]
    out = [None] * n

    def go(s):
        out[s] = stream(jobs[s][0], jobs[s][1], 512)

    a0, d0, k0 = metrics()
    ts = [threading.Thread(target=go, args=(s,)) for s in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    a1, d1, k1 = metrics()
    t0 = min(o[2] for o in out if o[2]); t1 = max(o[3] for o in out)
    tokens = sum(o[0] for o in out)
    problems = [p for p in (check("mixed", o[4]) for o in out) if p]
    return {"work": work, "i": b, "tag": ",".join(j[1] for j in jobs), "tokens": tokens,
            "ttft": statistics.median(o[2] - o[1] for o in out if o[2]), "decode_s": t1 - t0, "tps": tokens / (t1 - t0),
            "per_stream": statistics.median(o[0] / (o[3] - o[2]) for o in out if o[2]),
            "tok_per_step": 1 + (a1 - a0) / (d1 - d0) if d1 > d0 else 1.0,
            "drafts_per_step": (k1 - k0) / (d1 - d0) if d1 > d0 else 0.0, "problem": "; ".join(problems)}


def bootstrap(xs, stat, rng):
    vals = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def modified_z(xs):
    med = statistics.median(xs)
    mad = statistics.median(abs(x - med) for x in xs)
    return [0.0 if mad == 0 else 0.6745 * (x - med) / mad for x in xs]


def summarize(runs):
    rng = random.Random(0)
    rate = lambda rs: sum(r["tokens"] for r in rs) / sum(r["decode_s"] for r in rs)
    rates = [r["tps"] for r in runs]
    cv = statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else float("nan")
    lo, hi = bootstrap(runs, rate, rng)
    return {"runs": len(runs), "tps": rate(runs), "ci": [lo, hi], "cv": cv,
            "detects": (Z95 + Z80) * math.sqrt(2) * cv / math.sqrt(len(runs)),
            "tok_per_step": statistics.median(r["tok_per_step"] for r in runs),
            "drafts_per_step": statistics.median(r["drafts_per_step"] for r in runs),
            "ms_per_step": statistics.median(r["ms_per_step"] for r in runs) if "ms_per_step" in runs[0] else None,
            "ttft_ms": 1000 * statistics.median(r["ttft"] for r in runs if r["ttft"] is not None),
            "outliers": [r["i"] for r, z in zip(runs, modified_z(rates)) if abs(z) > 3.5],
            "problems": sum(1 for r in runs if r["problem"])}


def cmd_run(a):
    works = [w for w in a.workloads.split(",")]
    stream("hi", "warm", 4)
    res = {"label": a.label, "series": a.series, "base": BASE, "model": MODEL,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": [], "summary": {}}
    for w in works:
        run, count = (run_single, a.runs) if w in SINGLE else (run_mixed, a.batches)
        for j in range(a.warmup):
            run(w, a.series, j, warm=True)
        rs = []
        for i in range(count):
            rs.append(run(w, a.series, i))
            if rs[-1]["problem"]:
                print(f"  {w} run {i}: {rs[-1]['problem']}", flush=True)
        res["runs"] += rs
        s = res["summary"][w] = summarize(rs)
        step = f"{s['ms_per_step']:6.1f}" if s["ms_per_step"] else "     -"
        print(f"{w:<11}{s['runs']:>5}  {s['tps']:7.1f}  [{s['ci'][0]:6.1f}, {s['ci'][1]:6.1f}]  {100 * s['cv']:5.1f}%"
              f"  {100 * s['detects']:5.1f}%  {s['tok_per_step']:5.2f}  {s['drafts_per_step']:5.2f}  {step}  {s['ttft_ms']:6.0f}"
              f"  {s['problems']} failed" + (f", outliers {s['outliers']}" if s["outliers"] else ""), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def wilcoxon(ds):
    """Two-sided exact Wilcoxon signed-rank p-value; zero differences are dropped."""
    ds = [d for d in ds if d != 0]
    n = len(ds)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda k: abs(ds[k]))
    ranks = [0] * n
    k = 0
    while k < n:  # ties share the average rank; doubled ranks stay integers
        j = k
        while j + 1 < n and abs(ds[order[j + 1]]) == abs(ds[order[k]]):
            j += 1
        for m in range(k, j + 1):
            ranks[order[m]] = k + j + 2
        k = j + 1
    w = sum(r for r, d in zip(ranks, ds) if d > 0)
    counts = {0: 1}  # number of sign patterns giving each positive-rank sum
    for r in ranks:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = 2 ** n
    below = sum(c for s, c in counts.items() if s <= w) / total
    above = sum(c for s, c in counts.items() if s >= w) / total
    return min(1.0, 2 * min(below, above))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, worst = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        worst = max(worst, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = worst
    return adj


def cmd_compare(a):
    A, B = (json.load(open(p)) for p in (a.a, a.b))
    if A["series"] != B["series"]:
        sys.exit("the files belong to different series, so their runs cannot be paired")
    rng = random.Random(0)
    rows = []
    for w in A["summary"]:
        if w not in B["summary"]:
            continue
        ra = {r["i"]: r for r in A["runs"] if r["work"] == w and not r["problem"]}
        rb = {r["i"]: r for r in B["runs"] if r["work"] == w and not r["problem"]}
        idx = sorted(set(ra) & set(rb))
        ratio = lambda key: [math.log(rb[i][key] / ra[i][key]) for i in idx if rb[i].get(key) and ra[i].get(key)]
        ds = ratio("tps")
        if len(ds) < 2:
            print(f"{w:<11}{len(ds):>5}  too few paired runs")
            continue
        lo, hi = bootstrap(ds, statistics.mean, rng)
        rows.append([w, len(idx), math.exp(statistics.mean(ds)), math.exp(lo), math.exp(hi), wilcoxon(ds),
                     math.exp(statistics.mean(ratio("tok_per_step"))),
                     math.exp(statistics.mean(ratio("ms_per_step"))) if ra[idx[0]].get("ms_per_step") else None])
    for row, p in zip(rows, holm([r[5] for r in rows])):
        w, n, g, lo, hi, _, step_ratio, ms = row
        verdict = "differs" if p < 0.05 else "no detectable difference"
        step = f"{ms:6.3f}" if ms else "     -"
        print(f"{w:<11}{n:>5}  {g:6.3f}  [{lo:6.3f}, {hi:6.3f}]  p {p:6.4f}  {step_ratio:6.3f}  {step}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", default="")
    r.add_argument("--series", default="glm53-bench-1")
    r.add_argument("--runs", type=int, default=16)
    r.add_argument("--batches", type=int, default=8)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--workloads", default="code,prose,structured,counting,mixed4,mixed8")
    r.add_argument("--out", default="bench.json")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        print(f"{'workload':<11} runs    tok/s   95% interval      spread  detects  tok/step  drafts/step  ms/step  TTFT ms")
        cmd_run(a)
    else:
        print(f"{'workload':<11} pairs   B / A   95% interval       p (Holm)  tok/step  ms/step")
        cmd_compare(a)


if __name__ == "__main__":
    main()
