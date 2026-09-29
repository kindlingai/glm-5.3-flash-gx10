#!/usr/bin/env python3
"""Before vLLM starts, every rank meets on the head, compares what must be the
same on every node, and times an NCCL all-reduce over the fabric.

A fabric can come up linked at full rate with healthy PCIe and still move
12 Gb/s (2026-09-26: fixed only by draining the boxes' power), and a node can
run a stale copy of an override file. Neither fails anything; both show up
later as a slow or subtly different model. The head prints what it found.

Warnings only: it always exits 0, and a rank that finds no peers within
FABRIC_CHECK_TIMEOUT_S skips the check.

On a ring (FABRIC_LAYOUT=ring) the ranks compare facts through the TCP store
and skip the all-reduce. This runs before mentat places the ranks in cable
order, so an NCCL collective here would route to diagonal boxes that share
no cable, and hang.
"""
import datetime
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import time

PORT = int(os.environ.get("FABRIC_CHECK_PORT", "29511"))
TIMEOUT_S = int(os.environ.get("FABRIC_CHECK_TIMEOUT_S", "120"))
PROBE_MIB = 256
# Bus bandwidth an all-reduce reaches on a healthy fabric, per fabric device
# (one per ConnectX PCIe root); below 80% of it the check warns.
GBPS_PER_DEVICE = float(os.environ.get("FABRIC_CHECK_GBPS_PER_DEVICE", "95"))

# Knobs every rank must agree on. Per-node values (addresses, device names,
# the GID index, which the kernel assigns per boot) are listed, not compared.
SAME_ENV = ("TP", "MAX_NUM_BATCHED_TOKENS", "MAX_MODEL_LEN", "BLOCK_SIZE", "KV_CACHE_DTYPE", "KV_CACHE_MEMORY",
            "GPU_MEM_UTIL", "SPEC_METHOD", "SPEC_TOKENS", "MOE_BACKEND", "EXTRA_ARGS", "MTP", "HEAD_HOST")


def sha(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        return "missing"


def mounted_files() -> dict:
    """Files bind-mounted over the image (the compose overrides), by mount point."""
    out = {}
    with open("/proc/self/mountinfo") as f:
        for line in f:
            target = line.split()[4]
            if target.startswith(("/usr/local/lib/", "/opt/")) and os.path.isfile(target):
                out["mount " + target.rsplit("/dist-packages/", 1)[-1]] = sha(target)
    return out


def facts() -> tuple[dict, dict]:
    """(values that must match on every rank, per-node values to list)."""
    import torch

    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    model = os.environ.get("MODEL_DIR", "/models/glm-5.3-flash-nvfp4")
    same = {
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "driver": driver,
        "entrypoint": sha("/entrypoint.sh"),
        "discover": sha("/discover.sh"),
        "model config": sha(f"{model}/config.json"),
        "chat template": sha(os.environ.get("CHAT_TEMPLATE") or "/usr/local/share/glm53-chat-template.jinja"),
        "fabric devices": str(len([d for d in os.environ.get("NCCL_IB_HCA", "").split(",") if d])),
    }
    same.update({k: os.environ.get(k, "") for k in SAME_ENV})
    same.update({k: v for k, v in os.environ.items() if k.startswith("VLLM_") and k != "VLLM_HOST_IP"})
    same.update(mounted_files())
    node = {"host": socket.gethostname(), "gid": os.environ.get("NCCL_IB_GID_INDEX", "?"),
            "devices": os.environ.get("NCCL_IB_HCA", "?"), "ip": os.environ.get("VLLM_HOST_IP", "?")}
    return same, node


def main() -> None:
    world = int(os.environ.get("TP", "4"))
    head = os.environ.get("ROLE") == "head"
    if world < 2:
        return
    import torch
    import torch.distributed as dist

    ring = os.environ.get("FABRIC_LAYOUT") == "ring"
    timeout = datetime.timedelta(seconds=TIMEOUT_S)
    try:
        store = dist.TCPStore(os.environ["HEAD_HOST"], PORT, world_size=world, is_master=head, timeout=timeout,
                              wait_for_workers=False)
        rank = 0 if head else int(store.add("rank", 1))
        if rank >= world:
            print(f"fabric check: more than {world} ranks joined; skipped")
            return
        if ring:
            store.set(f"facts{rank}", json.dumps(facts()))
            everything = [json.loads(store.get(f"facts{r}")) for r in range(world)] if rank == 0 else None
        else:
            torch.cuda.set_device(0)
            dist.init_process_group("nccl", store=store, rank=rank, world_size=world, timeout=timeout,
                                    device_id=torch.device("cuda", 0))
    except Exception as e:  # noqa: BLE001 -- a missing peer must not stop the boot
        print(f"fabric check: skipped, no rendezvous with all {world} ranks within {TIMEOUT_S}s ({e})")
        return
    if ring:
        if rank == 0:
            report(world, everything, None)
        else:
            print(f"fabric check: rank {rank} took part; the head prints the results")
        return

    same, node = facts()
    everything = [None] * world
    dist.all_gather_object(everything, (same, node))

    x = torch.ones(PROBE_MIB << 19, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 5 * 1e3
    bus = (PROBE_MIB << 20) * 8 / (ms / 1e3) / 1e9 * 2 * (world - 1) / world
    dist.destroy_process_group()
    if rank != 0:
        print(f"fabric check: rank {rank} took part; the head prints the results")
        return
    report(world, everything, (ms, bus))


def report(world: int, everything: list, probe: tuple[float, float] | None) -> None:
    """The head's table: each rank, the all-reduce timing (None on a ring), and any value that differs."""
    print(f"fabric check ({world} ranks):")
    print("  rank  host          ip               gid  devices")
    for r, (_, n) in enumerate(everything):
        print(f"  {r:<5} {n['host']:<13} {n['ip']:<16} {n['gid']:<4} {n['devices']}")
    if probe is None:
        print("  ok   all-reduce skipped on a ring: this runs before mentat places the ranks in cable order")
    else:
        ms, bus = probe
        devices = min(int(s["fabric devices"]) for s, _ in everything)
        expect = GBPS_PER_DEVICE * max(devices, 1)
        status = "ok  " if bus >= 0.8 * expect else "WARN"
        print(f"  {status} all-reduce {PROBE_MIB} MiB: {ms:.1f} ms, bus bandwidth {bus:.0f} Gb/s "
              f"(expected ~{expect:.0f} with {devices} device{'s' if devices != 1 else ''} per node)")
    keys = sorted(set().union(*(s.keys() for s, _ in everything)))
    bad = 0
    for k in keys:
        vals = [s.get(k, "(absent)") for s, _ in everything]
        if len(set(vals)) > 1:
            bad += 1
            per = ", ".join(f"{n['host']}={v}" for v, (_, n) in zip(vals, everything))
            print(f"  WARN differs: {k}: {per}")
    if not bad:
        print(f"  ok   {len(keys)} values match on every rank")


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"  # the entrypoint sets INFO for vLLM; it would bury the table
    main()
