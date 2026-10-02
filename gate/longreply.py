"""Long replies at temperature 1.0: does decode drift away from prefill as a reply grows?

The NLL gate scores fixed texts, so it only sees the prefill side. This generates
REPLIES long replies (16,000 tokens, thinking high, temperature 1.0, unique
cache_salt each) with their decode-time logprobs, rescores the same token ids as
a prompt (prompt_logprobs), and compares the two per 2,000-token window.

On this stack decode and prefill differ by design (batch-size invariance traded
for speed), so the absolute gap is not the signal: a flat 0.11-0.24 nats mean
|diff| from the first window to the last is normal here. The signal is growth:
a reply that degrades as it gets longer (a broken drafter, a cache or state bug
that compounds) shows windows late in the reply well above its first ones.

  FAIL  late windows (second half) average > 1.8x the first two windows, or any
        window > 0.40, or a 16-token run repeated > 8 times in one window
  PASS  otherwise (a reply that stops before 4,000 tokens is skipped; at least
        one must reach it, else INCONCLUSIVE)

Calibration, on per-window results saved from an equivalent checker (same
windows, decode vs prompt_logprobs): 22 16K replies from unaffected builds all
PASS (10 on this stack at 1173cc1..78b9540: late/first 0.92-1.37, max window
0.244; 12 on a vLLM-main build). On an 8-bit drafter that degraded long replies,
7 of 9 replies FAIL (late/first 2.31-5.59, or a window at 0.995); the 2 that
PASS show no growth within that reply.

usage: longreply.py [REPLIES]   (GATE_URL, GATE_MODEL as for the other gate tools)
Each 16K reply took ~36 min here with ~30 other streams running; quiet is faster
(not measured on this stack).
"""
import concurrent.futures as cf, json, os, sys, time, urllib.request
from collections import Counter

BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
REPLIES = int(sys.argv[1]) if sys.argv[1:2] and sys.argv[1].isdigit() else 2
TOKENS, WIN, MIN_LEN = 16000, 2000, 4000
MESSAGES = [{"role": "user", "content":
    "Write a complete, production-quality Python implementation of a persistent key-value store with a write-ahead "
    "log, crash recovery, compaction, range scans and a small CLI. Explain each design decision as you go, then "
    "write thorough tests for every component, including crash-recovery tests that kill the process mid-write."}]


def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def generate(i):
    t = time.time()
    d = post("/v1/chat/completions", {
        "model": MODEL, "messages": MESSAGES, "temperature": 1.0, "max_tokens": TOKENS, "reasoning_effort": "high",
        "cache_salt": f"longreply-{time.time()}-{i}", "logprobs": True, "top_logprobs": 1,
        "return_tokens_as_token_ids": True})
    c = d["choices"][0]
    toks = [(int(x["token"].split(":", 1)[1]), x["logprob"]) for x in (c.get("logprobs") or {}).get("content") or []]
    return {"i": i, "secs": round(time.time() - t), "finish": c["finish_reason"], "tokens": toks}


def rescore(toks):
    pids = post("/tokenize", {"model": MODEL, "messages": MESSAGES, "add_generation_prompt": True})["tokens"]
    lp = post("/v1/completions", {"model": MODEL, "prompt": pids + [t for t, _ in toks], "max_tokens": 1,
                                  "temperature": 0, "prompt_logprobs": 0})["choices"][0]["prompt_logprobs"][len(pids):]
    return [next(iter(x.values()))["logprob"] if x else None for x in lp]


def windows(toks, res):
    """Per window of >= 500 tokens: (start, mean |diff|, per 1k > 1 nat, top 16-token repeat)."""
    out = []
    for k in range(0, len(toks), WIN):
        pairs = [(a[1], b) for a, b in zip(toks[k:k + WIN], res[k:k + WIN]) if b is not None]
        if len(pairs) < 500:
            continue
        diffs = [abs(a - b) for a, b in pairs]
        ids = [t for t, _ in toks[k:k + WIN]]
        runs = Counter(tuple(ids[j:j + 16]) for j in range(len(ids) - 16))
        out.append((k, sum(diffs) / len(diffs), 1000 * sum(x > 1 for x in diffs) / len(diffs), max(runs.values())))
    return out


def verdict(per_reply):
    """per_reply: list of window lists. Returns (verdict, reasons)."""
    scored = [w for w in per_reply if w and (w[-1][0] + WIN) >= MIN_LEN]
    if not scored:
        return "INCONCLUSIVE", [f"no reply reached {MIN_LEN} tokens"]
    bad = []
    for n, w in enumerate(scored):
        first = sum(x[1] for x in w[:2]) / len(w[:2])
        late = [x[1] for x in w[len(w) // 2:]]
        ratio = (sum(late) / len(late)) / first if first else 0
        if ratio > 1.8:
            bad.append(f"reply {n}: late windows {ratio:.2f}x the first two")
        if max(x[1] for x in w) > 0.40:
            bad.append(f"reply {n}: a window at {max(x[1] for x in w):.3f} mean |diff|")
        if max(x[3] for x in w) > 8:
            bad.append(f"reply {n}: a 16-token run repeated {max(x[3] for x in w)} times")
    return ("FAIL", bad) if bad else ("PASS", [])


def selftest():
    w = lambda *d: [(k * WIN, v, 0, 1) for k, v in enumerate(d)]
    clean = w(0.165, 0.150, 0.180, 0.190, 0.200, 0.190, 0.210, 0.180)        # this stack, flat
    grows = w(0.090, 0.085, 0.120, 0.180, 0.300, 0.350, 0.380, 0.390)        # collapses with length
    assert verdict([clean, clean]) == ("PASS", [])
    assert verdict([clean, grows])[0] == "FAIL"
    assert verdict([w(0.2, 0.5, 0.2, 0.2)])[0] == "FAIL"                     # one window past 0.40
    assert verdict([[(0, 0.1, 0, 30), (2000, 0.1, 0, 1)]])[0] == "FAIL"      # a loop
    assert verdict([w(0.1)])[0] == "INCONCLUSIVE"                            # 2,000 tokens only
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        selftest(); sys.exit(0)
    with cf.ThreadPoolExecutor(REPLIES) as ex:
        outs = list(ex.map(generate, range(REPLIES)))
    per_reply = []
    for o in outs:
        res = rescore(o["tokens"])
        w = windows(o["tokens"], res)
        per_reply.append(w)
        print(f"reply {o['i']}: {len(o['tokens'])} tokens in {o['secs']} s, finish {o['finish']}")
        for k, d, n, rep in w:
            print(f"  {k:5d}+  mean |diff| {d:.3f}   > 1 nat {n:3.0f}/1k   top 16-token repeat x{rep}")
    v, why = verdict(per_reply)
    print(v + ("" if not why else ": " + "; ".join(why)))
