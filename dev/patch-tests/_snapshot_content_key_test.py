#!/usr/bin/env python3
"""experimental/snapshot/weight_snapshot.py: the snapshot key sees tensor bytes (#62).

Writes small safetensors shards and checks _content_fingerprint:
  - two checkpoints with byte-identical headers but different tensor data
    (the DFlash2 7d74cdd -> bf582e4 case) get different fingerprints
  - the same bytes in another directory get the same fingerprint (a moved or
    re-mounted checkpoint still restores)
  - a change at the start, middle or end of the data is seen
  - a repo id that is not a directory, or a directory without shards, is None
CPU only, no vLLM needed (vllm.logger is stubbed when absent):

    python3 dev/patch-tests/_snapshot_content_key_test.py
"""
import importlib.util
import json
import logging
import os
import struct
import sys
import tempfile
import types

try:
    import vllm.logger  # noqa: F401
except Exception:
    stub = types.ModuleType("vllm.logger")
    stub.init_logger = logging.getLogger
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = stub

here = os.path.dirname(os.path.abspath(__file__))
path = next(p for p in (os.path.join(here, "weight_snapshot.py"),
                        os.path.join(here, "../../experimental/snapshot/weight_snapshot.py")) if os.path.exists(p))
spec = importlib.util.spec_from_file_location("weight_snapshot", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

N = 3 * m._SAMPLE + 12345  # data longer than the three samples


def shard(d, name, data):
    header = json.dumps({"w": {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}}).encode()
    with open(os.path.join(d, name), "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + data)


def ckpt(root, sub, data):
    d = os.path.join(root, sub)
    os.makedirs(d)
    shard(d, "model-00001-of-00002.safetensors", data)
    shard(d, "model-00002-of-00002.safetensors", bytes(reversed(data)))
    return d


failures = 0


def check(name, ok):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    failures += not ok


with tempfile.TemporaryDirectory() as root:
    base = bytes(i % 251 for i in range(N))
    old = m._content_fingerprint(ckpt(root, "old", base))
    new_data = bytearray(base)
    new_data[0] ^= 1
    new_data[N // 2] ^= 1
    new = m._content_fingerprint(ckpt(root, "new", bytes(new_data)))
    check("same headers, different tensor data -> different key", old != new)
    check("same bytes elsewhere -> same key", m._content_fingerprint(ckpt(root, "moved", base)) == old)
    start = 8 + len(json.dumps({"w": {"dtype": "U8", "shape": [N], "data_offsets": [0, N]}}).encode())
    for label, off in (("start", 0), ("middle", (N - m._SAMPLE) // 2), ("end", N - 1)):
        b = bytearray(base)
        b[off] ^= 1
        check(f"one byte changed at the {label} -> different key",
              m._content_fingerprint(ckpt(root, label, bytes(b))) != old)
    check("repo id -> None", m._content_fingerprint("zai-org/GLM-5.3-Flash") is None)
    os.makedirs(os.path.join(root, "empty"))
    check("no shards -> None", m._content_fingerprint(os.path.join(root, "empty")) is None)

print(f"{failures} failure(s)")
sys.exit(1 if failures else 0)
