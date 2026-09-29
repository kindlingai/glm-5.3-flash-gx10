"""All-reduce over RoCE for GB10 tensor-parallel groups, without NCCL.

NCCL costs about 80-90 us per decode-sized all-reduce on these boxes; this
costs 13 us at 8 KB and 27 us at 64 KB. Each rank writes its partial to its
peers with RDMA over both ConnectX roots (see arx_vllm.cu). Larger all-reduces,
which are prefill-sized, stay on NCCL, whose bandwidth wins there.

Enabled with VLLM_ARX_ALLREDUCE=1. The RDMA devices and GID index come from
NCCL_IB_HCA (exactly two, one per root, in the same subnet order on every
rank) and NCCL_IB_GID_INDEX.

Ring mode (VLLM_ARX_RING=1) is for groups of 4 or 2 cabled in a ring with no
switch, with rank r's "next" port cabled to rank r+1. arx and arxbig open QPs
only to r-1 and r+1, and the rank in between relays raw data for the rank
beyond, so results match mesh mode bit for bit (see the .cu files).
ARX_RING_PREV_HCAS and ARX_RING_NEXT_HCAS name the two RDMA devices (root 0,
root 1) of the port facing each neighbour. Each port has its own subnet, so
each device uses the RoCE v2 GID of the IPv4 address on its own netdev.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_ext = None
_taken = False  # the extension holds one connection per process


def _hcas(var: str) -> list[str]:
    return [h.strip("=^").split(":")[0] for h in os.environ.get(var, "").split(",") if h]


def roce_v2_gid(dev: str) -> int:
    """GID index of dev's RoCE v2 GID for an IPv4 address on dev's own netdev.

    The kernel fills the GID table from the netdev's addresses, and ndevs
    names the netdev an entry came from, so a VLAN or other upper device's
    entries are skipped.
    """
    base = f"/sys/class/infiniband/{dev}"
    netdevs = set(os.listdir(f"{base}/device/net"))
    port = f"{base}/ports/1"
    found = []
    for name in sorted(os.listdir(f"{port}/gids"), key=int):
        try:
            with open(f"{port}/gids/{name}") as f:
                gid = f.read().strip()
            with open(f"{port}/gid_attrs/types/{name}") as f:
                kind = f.read().strip()
            with open(f"{port}/gid_attrs/ndevs/{name}") as f:
                ndev = f.read().strip()
        except OSError:  # empty entries refuse to report their attributes
            continue
        if kind == "RoCE v2" and ndev in netdevs and gid.startswith("0000:0000:0000:0000:0000:ffff:"):
            found.append(int(name))
    if not found:
        raise RuntimeError(f"arx: {dev} has no RoCE v2 GID for an IPv4 address on {sorted(netdevs)}")
    if len(found) > 1:
        logger.warning("arx: %s has IPv4 GIDs at %s; using %d", dev, found, found[0])
    return found[0]


def fabric(world: int, who: str):
    """(devices, GID index per device, ring) for this rank, or None to use NCCL."""
    if os.environ.get("VLLM_ARX_RING") == "1":
        prev, nxt = _hcas("ARX_RING_PREV_HCAS"), _hcas("ARX_RING_NEXT_HCAS")
        if len(prev) != 2 or len(nxt) != 2:
            logger.warning("%s ring mode needs two devices each in ARX_RING_PREV_HCAS and ARX_RING_NEXT_HCAS, "
                           "got %r and %r; using NCCL", who, prev, nxt)
            return None
        if world not in (2, 4):
            raise ValueError(f"{who} ring mode supports groups of 2 or 4 ranks, not {world}")
        devs = prev + nxt
        return devs, [roce_v2_gid(d) for d in devs], True
    hcas = _hcas("NCCL_IB_HCA")
    if len(hcas) != 2:
        logger.warning("%s needs two RDMA devices in NCCL_IB_HCA, got %r; using NCCL", who, hcas)
        return None
    gid = int(os.environ.get("NCCL_IB_GID_INDEX", "5"))
    return hcas, [gid, gid], False


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            "arx_vllm",
            [os.path.join(os.path.dirname(__file__), "arx_vllm.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a"],
            extra_ldflags=["-libverbs"],
        )
    return _ext


# (address, bytes) ranges for the next all-reduce to prefetch into L2 while it waits.
_prefetch: tuple = ()


def set_prefetch(tensors) -> None:
    """The next arx all-reduce asks L2 for these tensors' bytes while it waits for peers."""
    global _prefetch
    _prefetch = tuple((t.data_ptr(), t.numel() * t.element_size()) for t in tensors if t is not None)


class ArxCommunicator:
    def __init__(self, group: ProcessGroup, device: torch.device):
        global _taken
        self.disabled = True
        if _taken:
            logger.warning("arx already serves another group in this process; using NCCL here")
            return
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        fab = fabric(world, "arx")
        if fab is None:
            return
        devs, gids, ring = fab
        with torch.cuda.device(device):
            ext = _load()
            info = ext.prepare(rank, world, devs, gids, ring)
            infos: list[bytes] = [b""] * world
            dist.all_gather_object(infos, info, group=group)
            ext.connect(infos)
        dist.barrier(group=group)
        _taken = True
        self.ext = ext
        self.max_bytes = min(int(os.environ.get("VLLM_ARX_MAX_BYTES", 256 << 10)), ext.max_bytes())
        self.disabled = False
        logger.info("arx all-reduce: rank %d/%d%s on %s, gids %s, up to %d bytes", rank, world,
                    " ring (prev, next)" if ring else "", devs, gids, self.max_bytes)

    def should_use(self, t: torch.Tensor) -> bool:
        return (
            t.dtype == torch.bfloat16
            and t.is_cuda
            and t.is_contiguous()
            and t.numel() % 8 == 0
            and t.numel() * 2 <= self.max_bytes
        )

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        global _prefetch
        out = torch.empty_like(t)
        pf, _prefetch = _prefetch, ()
        self.ext.allreduce(t, out, [p for p, _ in pf], [b for _, b in pf])
        return out


_big_ext = None


class ArxBig:
    """Prefill-sized all-gathers over RoCE (arxbig.cu): ~10% faster than NCCL's.

    Enabled with VLLM_ARXBIG=1 next to arx. all_gather returns a view of a
    pinned buffer that the fourth all-gather after it overwrites, so a caller
    that keeps the result longer must copy it.
    """

    def __init__(self, group: ProcessGroup, device: torch.device):
        global _big_ext
        self.disabled = True
        self.rs = False
        self.gather = False
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        fab = fabric(world, "arxbig")
        if fab is None:
            return
        devs, gids, ring = fab
        slot = int(os.environ.get("VLLM_ARXBIG_SLOT_MB", "132")) << 20
        slot -= slot % (world * 16)
        with torch.cuda.device(device):
            if _big_ext is None:
                from torch.utils.cpp_extension import load

                _big_ext = load("arxbig", [os.path.join(os.path.dirname(__file__), "arxbig.cu")],
                                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a"],
                                extra_ldflags=["-libverbs"])
            self.rs = os.environ.get("VLLM_ARXBIG_RS") == "1"
            info = _big_ext.prepare(rank, world, devs, gids, ring, slot, slot if self.rs else 0)
            infos: list[bytes] = [b""] * world
            dist.all_gather_object(infos, info, group=group)
            _big_ext.connect(infos)
            dist.barrier(group=group)
        self.ext, self.world, self.slot = _big_ext, world, slot
        self.min_bytes = int(os.environ.get("VLLM_ARXBIG_MIN_KB", "1024")) << 10
        self.gather = os.environ.get("VLLM_ARXBIG_AG") == "1"
        self.disabled = False
        logger.info("arxbig all-gather: rank %d/%d%s, %d MiB slots", rank, world, " ring" if ring else "", slot >> 20)

    def should_gather(self, t: torch.Tensor) -> bool:
        total = t.numel() * t.element_size() * self.world
        return (t.is_cuda and t.dim() >= 1 and (t.numel() * t.element_size()) % 16 == 0
                and self.min_bytes <= total <= self.slot)

    def all_gather(self, t: torch.Tensor) -> torch.Tensor:
        return self.ext.all_gather(t.contiguous())
