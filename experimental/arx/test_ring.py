#!/usr/bin/env python3
"""Bit-exact and speed test for arx and arxbig, in mesh or ring mode, without vLLM.

Every rank builds every rank's inputs from fixed seeds and computes the
result the way the kernels do (bf16 values summed in fp32 in rank order,
rounded once). Each collective's output must match that reference bit for
bit, and each rank's output must match what the other ranks expect it to
hold. Covered: the arx all-reduce, arxbig's all_gather and reduce_scatter,
and the fused MoE finalize + rs_finish path (with routing weights that are
powers of two, so the reference can round the same way the kernel's FMAs
do). Calls also go out in back-to-back bursts of mixed collectives, checked
only afterwards, so buffer reuse and relays run at full speed. It then
prints all-reduce latency and all-gather / reduce-scatter bandwidth.

Mesh mode needs every box on the switch; ring mode needs the ring cabling,
or the switch with both ARX_RING_*_HCAS set to the NCCL_IB_HCA devices (that
checks the relay logic, not the speed). The two modes can't share one
cabling, so each is its own run. Since both must match the same reference,
passing both means they agree.

Run one container per box from the repo root, rank 0 first, all within a
minute. The serving stack must be down. Rank r must be ring member r in
cable order (rank r's next port cabled to rank r+1's prev port).

    sudo docker run --rm --gpus all --network host --ipc host --ulimit memlock=-1 \\
      --device /dev/infiniband:/dev/infiniband -v "$PWD/experimental/arx:/arx:ro" \\
      -v /tmp/arx-ext:/root/.cache/torch_extensions \\
      -e RANK=<0..3> -e WORLD_SIZE=4 -e MASTER_ADDR=<rank 0's LAN address> -e MASTER_PORT=29610 \\
      -e VLLM_HOST_IP=<this box's LAN address> $FABRIC \\
      --entrypoint python3 <serving image> /arx/test_ring.py --mode <mesh|ring>

with FABRIC, for mesh, the entrypoint's values (its log prints them):
    -e NCCL_IB_HCA=<root 0 device>,<root 1 device> -e NCCL_IB_GID_INDEX=<index>
and for ring, the two functions of the port facing each neighbour, e.g.
    -e ARX_RING_PREV_HCAS=rocep1s0f0,roceP2p1s0f0 -e ARX_RING_NEXT_HCAS=rocep1s0f1,roceP2p1s0f1

The /tmp/arx-ext mount keeps the compiled extensions between runs.
WORLD_SIZE=2 works the same way. Every rank prints its failures and rank 0
ends with "RESULT <mode>: PASS" or "FAIL"; the exit code is nonzero on a
failure. A rank that makes no progress for 120 s says where it was and exits.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys
import threading
import time
import zlib

import torch
import torch.distributed as dist

BF16 = torch.bfloat16


def lan_iface(ip: str) -> str:
    out = subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) > 3 and f[3].split("/")[0] == ip:
            return f[1]
    sys.exit(f"no interface holds VLLM_HOST_IP={ip}")


def gen(shape, dtype, *key) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(zlib.crc32(repr(key).encode()))
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)
    return (torch.randn(shape, generator=g, device="cuda") * 4).to(dtype)


def rank_order_sum(xs) -> torch.Tensor:
    acc = torch.zeros(xs[0].shape, dtype=torch.float32, device="cuda")
    for x in xs:
        acc += x.float()
    return acc.to(BF16)


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(raw(t).cpu().numpy().tobytes()).hexdigest()[:16]


class Watchdog:
    def __init__(self, rank: int, secs: float):
        self.rank, self.secs, self.what, self.t = rank, secs, "setup", time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def at(self, what: str) -> None:
        self.what, self.t = what, time.time()

    def _run(self) -> None:
        while True:
            time.sleep(5)
            if time.time() - self.t > self.secs:
                print(f"rank {self.rank}: no progress for {self.secs:.0f} s in {self.what}", flush=True)
                os._exit(3)


class Test:
    def __init__(self, mode: str):
        self.mode = mode
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.checks = self.failures = 0
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import arx

        dev = torch.device("cuda", 0)
        self.ar = arx.ArxCommunicator(dist.group.WORLD, dev)
        self.big = arx.ArxBig(dist.group.WORLD, dev)
        if self.ar.disabled or self.big.disabled or not self.big.rs:
            sys.exit("arx or arxbig did not come up; see the warnings above")
        self.dog = Watchdog(self.rank, 120)  # after the extensions compile

    # Each op issues one collective and returns (output, reference function,
    # kind): "same" outputs are the whole reference on every rank, "chunk"
    # outputs are this rank's chunk of it.

    def allreduce(self, n: int, key):
        ins = [gen((n,), BF16, "ar", key, s) for s in range(self.world)]
        out = torch.empty_like(ins[self.rank])
        self.ar.ext.allreduce(ins[self.rank], out, [], [])
        return out, lambda: rank_order_sum(ins), "same"

    def all_gather(self, nbytes: int, dtype, key):
        n = nbytes // torch.empty((), dtype=dtype).element_size()
        ins = [gen((n,), dtype, "ag", key, s) for s in range(self.world)]
        out = self.big.all_gather(ins[self.rank]).clone()  # later gathers overwrite the returned slot
        return out, lambda: torch.cat(ins), "same"

    def reduce_scatter(self, chunk_bytes: int, key):
        n = chunk_bytes // 2 * self.world
        ins = [gen((n,), BF16, "rs", key, s) for s in range(self.world)]
        out = self.big.ext.reduce_scatter(ins[self.rank])
        return out, lambda: rank_order_sum(ins), "chunk"

    def finalize_rs(self, T: int, Tpad: int, H: int, key, topk: int = 2):
        def inputs(s):
            R = T * topk
            y = gen((R, H), BF16, "fy", key, s)
            shared = gen((T, H), BF16, "fs", key, s)
            g = torch.Generator(device="cuda")
            g.manual_seed(zlib.crc32(repr(("fp", key, s)).encode()))
            pos = torch.randint(0, R, (T * topk,), generator=g, device="cuda", dtype=torch.int32)
            w = 2.0 ** torch.randint(-2, 2, (T, topk), generator=g, device="cuda").float()
            return y, pos, w, shared

        mine = inputs(self.rank)
        seq = self.big.ext.moe_finalize_rs(*mine, T, Tpad, None)
        out = self.big.ext.rs_finish(seq, Tpad // self.world, H)

        def ref():
            rows = []
            for s in range(self.world):
                y, pos, w, shared = inputs(s)
                acc = shared.float()
                p = pos.view(T, topk).long()
                for q in range(topk):
                    acc = acc + w[:, q:q + 1] * y[p[:, q]].float()
                r = torch.zeros((Tpad, H), dtype=BF16, device="cuda")
                r[:T] = acc.to(BF16)
                rows.append(r)
            return rank_order_sum(rows)

        return out, ref, "chunk"

    def check(self, name: str, out, ref_fn, kind: str) -> None:
        self.dog.at(f"check {name}")
        torch.cuda.synchronize()
        ref = ref_fn()
        if kind == "chunk":
            chunks = ref.chunk(self.world)
            want = [digest(c) for c in chunks]
            ok = torch.equal(raw(out), raw(chunks[self.rank]))
        else:
            want = [digest(ref)] * self.world
            ok = torch.equal(raw(out), raw(ref))
        got: list = [None] * self.world
        dist.all_gather_object(got, digest(out))
        self.checks += 1
        bad = [j for j in range(self.world) if got[j] != want[j]]
        if not ok or bad:
            self.failures += 1
            first = ""
            if not ok:
                exp = chunks[self.rank] if kind == "chunk" else ref
                diff = (raw(out) != raw(exp)).nonzero()
                first = f", first differing byte {diff[0].item()} of {raw(exp).numel()}"
            print(f"[{self.mode}] rank {self.rank} FAIL {name}: matches reference {ok}{first}; "
                  f"ranks whose output differs from what this rank expects: {bad}", flush=True)

    def issue(self, op: str, size: int, key):
        if op == "ar":
            return self.allreduce(size // 2, key)
        if op == "ag":
            return self.all_gather(size, BF16, key)
        if op == "ag8":
            return self.all_gather(size, torch.uint8, key)
        if op == "rs":
            return self.reduce_scatter(size, key)
        Tpad = max(size // (2 * 1024), 8 * self.world)
        Tpad -= Tpad % (8 * self.world)
        return self.finalize_rs(Tpad - Tpad // 7, Tpad, 1024, key)

    def burst(self, name: str, ops) -> None:
        """Issues every op back to back, then checks them all."""
        self.dog.at(f"burst {name}")
        issued = [(f"{name}[{i}] {op} {size}", *self.issue(op, size, (name, i))) for i, (op, size) in enumerate(ops)]
        for label, out, ref_fn, kind in issued:
            self.check(label, out, ref_fn, kind)

    def bench(self, label: str, fn, iters: int) -> float:
        """Seconds per call, the slowest rank's."""
        self.dog.at(f"bench {label}")
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        t: list = [None] * self.world
        dist.all_gather_object(t, a.elapsed_time(b) / 1e3 / iters)
        return max(t)

    def say(self, msg: str) -> None:
        if self.rank == 0:
            print(f"[{self.mode}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mesh", "ring"], required=True)
    p.add_argument("--repeat", type=int, default=3, help="checks per size")
    p.add_argument("--bursts", type=int, default=20, help="bursts of 16 mixed collectives")
    args = p.parse_args()

    os.environ["VLLM_ARX_RING"] = "1" if args.mode == "ring" else "0"
    os.environ["VLLM_ARXBIG_RS"] = "1"
    if "GLOO_SOCKET_IFNAME" not in os.environ and "VLLM_HOST_IP" in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = lan_iface(os.environ["VLLM_HOST_IP"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                            rank=rank, world_size=world)
    for k in sorted(os.environ):
        if k.startswith(("NCCL_IB", "ARX_", "VLLM_ARX", "GLOO_")):
            print(f"[{args.mode}] rank {rank} {k}={os.environ[k]}", flush=True)
    T = Test(args.mode)
    KB, MB = 1 << 10, 1 << 20

    T.say("checking sizes")
    for n in (8, 64, 1024, 4096, 32768, 131072, 262144):
        for i in range(args.repeat):
            T.check(f"allreduce {n * 2} B", *T.allreduce(n, ("size", i)))
    for nbytes in (16, 48, 4 * KB, 1 * MB + 16, 8 * MB, 32 * MB):
        for i in range(args.repeat):
            T.check(f"all_gather {nbytes} B", *T.all_gather(nbytes, BF16, ("size", i)))
            T.check(f"all_gather u8 {nbytes} B", *T.all_gather(nbytes, torch.uint8, ("size", i)))
    for chunk in (16, 48, 4 * KB, 1 * MB + 16, 8 * MB, 32 * MB):
        for i in range(args.repeat):
            T.check(f"reduce_scatter {chunk} B/chunk", *T.reduce_scatter(chunk, ("size", i)))
    for Tn, Tpad, H in ((100, 64 * world, 1024), (1000, 1024, 4096), (16000, 16384, 4096)):
        for i in range(args.repeat):
            T.check(f"finalize_rs T={Tn} Tpad={Tpad} H={H}", *T.finalize_rs(Tn, Tpad, H, ("size", i)))

    T.say("checking bursts of mixed collectives")
    rng = random.Random(1234)  # the same sequence on every rank
    sizes = {"ar": [16, 8 * KB, 64 * KB, 512 * KB], "ag": [16, 64 * KB, 1 * MB, 4 * MB],
             "ag8": [48, 1 * MB], "rs": [16, 64 * KB, 1 * MB, 4 * MB], "fin": [1 * MB, 8 * MB]}
    for b in range(args.bursts):
        ops = []
        for _ in range(16):
            op = rng.choice(["ar", "ar", "ar", "ag", "ag8", "rs", "rs", "fin"])
            ops.append((op, rng.choice(sizes[op])))
        T.burst(f"burst{b}", ops)
    T.burst("allreduce x64", [("ar", 8 * KB)] * 64)
    T.burst("reduce_scatter x8", [("rs", 8 * MB)] * 8)
    T.burst("finalize_rs x4", [("fin", 8 * MB)] * 4)

    T.say("all-reduce latency (arx)")
    for n in (4096, 32768, 131072, 262144):
        x = gen((n,), BF16, "bench")
        out = torch.empty_like(x)
        t = T.bench(f"allreduce {n}", lambda: T.ar.ext.allreduce(x, out, [], []), 2000)
        T.say(f"  allreduce {n * 2 // KB:4d} KB  {t * 1e6:7.1f} us")
    T.say("all-gather / reduce-scatter bandwidth (arxbig), data received per rank")
    for nbytes in (1 * MB, 8 * MB, 32 * MB):
        x = gen((nbytes // 2,), BF16, "bench")
        t = T.bench(f"all_gather {nbytes}", lambda: T.big.all_gather(x), 50)
        T.say(f"  all_gather     {nbytes // MB:3d} MB/rank  {t * 1e3:7.3f} ms  "
              f"{(world - 1) * nbytes * 8 / t / 1e9:6.1f} Gb/s")
    for chunk in (1 * MB, 8 * MB, 32 * MB):
        x = gen((chunk // 2 * world,), BF16, "bench")
        t = T.bench(f"reduce_scatter {chunk}", lambda: T.big.ext.reduce_scatter(x), 50)
        T.say(f"  reduce_scatter {chunk // MB:3d} MB/chunk {t * 1e3:7.3f} ms  "
              f"{(world - 1) * chunk * 8 / t / 1e9:6.1f} Gb/s")
    y, pos, w, shared = (gen((32768, 4096), BF16, "bench"), torch.randint(0, 32768, (32768,), device="cuda",
                         dtype=torch.int32), torch.ones((16384, 2), device="cuda"), gen((16384, 4096), BF16, "bench"))

    def fin():
        T.big.ext.rs_finish(T.big.ext.moe_finalize_rs(y, pos, w, shared, 16384, 16384, None), 16384 // world, 4096)

    t = T.bench("finalize_rs", fin, 20)
    T.say(f"  finalize_rs + rs_finish, 16k x 4096 bf16: {t * 1e3:.3f} ms")

    T.dog.at("result")
    fails: list = [None] * world
    dist.all_gather_object(fails, T.failures)
    T.say(f"RESULT {args.mode}: {'PASS' if sum(fails) == 0 else 'FAIL'}: {T.checks} checks per rank, "
          f"failures per rank {fails}")
    dist.barrier()
    os._exit(0 if sum(fails) == 0 else 1)  # the proxy threads never exit


if __name__ == "__main__":
    main()
