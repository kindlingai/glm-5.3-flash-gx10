"""Aggregate decode tok/s with N concurrent streams of one workload, each stream a
different prompt (identical prompts would route to the same experts).

    python3 conc_workload.py WORKLOAD N [N ...]      WORKLOAD: code | prose | structured | mixed
"""
import json, os, re, statistics, sys, threading, time, urllib.request

from prompts import prompt_for

BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")


def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}\S*\s+([0-9.e+]+)$", raw, re.M))
    return get("spec_decode_num_accepted_tokens_total"), get("spec_decode_num_drafts_total")


def one(prompt, out, i):
    body = json.dumps({"model": MODEL, "max_tokens": 512, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=900)
    first = n = None
    for line in r:
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if first is None and d.get("choices"):
            first = time.monotonic()
        if d.get("usage"):
            n = d["usage"]["completion_tokens"]
    out[i] = (n, first, time.monotonic())


work = sys.argv[1]
for N in [int(a) for a in sys.argv[2:]]:
    prompts = [prompt_for(work, i) for i in range(N)]
    a0, d0 = metrics()
    out = [None] * N
    ts = [threading.Thread(target=one, args=(prompts[i], out, i)) for i in range(N)]
    for t in ts: t.start()
    for t in ts: t.join()
    a1, d1 = metrics()
    t0 = min(o[1] for o in out); t1 = max(o[2] for o in out)
    per = [o[0] / (o[2] - o[1]) for o in out]
    al = 1 + (a1 - a0) / max(d1 - d0, 1)  # tokens per verify step per request
    print(f"{work:<10} streams {N:2d}: aggregate {sum(o[0] for o in out) / (t1 - t0):6.1f} tok/s,"
          f" per stream median {statistics.median(per):5.1f}, accepted per step {al:.2f}", flush=True)
