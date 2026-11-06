"""NCCL and arx settings for a rank that mentat placed on a ring of boxes.

Runs at interpreter start (fabric_ring.pth) and does nothing unless mentat
gave this process MENTAT_FABRIC_LAYOUT=ring, which it does for each rank of a
TP=RING4 or TP=RING3 claim. On a ring each box cables one port to the
previous rank and the other to the next, so a rank sends to next and receives
from prev on different ports, and the diagonal ranks share no cable. This
sets:

- NCCL_IB_HCA to both PCIe roots' functions of both ports, NCCL_ALGO=Ring so
  NCCL only talks to neighbours, and NCCL_GRAPH_FILE to a ring whose channels
  receive on the ports toward prev and send on the ports toward next. NCCL
  cannot infer that wiring, since it assumes every NIC reaches every peer.
  With FABRIC_RING_GRAPH=0 (TP=RING3) it sets NCCL_IB_SUBNET_AWARE_ROUTING
  instead of the graph and NCCL_ALGO.
- the GID from each port's own address, since each cable has its own subnet
  and no one GID index fits every device.

Each port must show two RDMA devices, one per PCIe root, each on a netdev with
an IPv4 address (a RoCE v2 GID). Otherwise this stops the rank and names the
port and netdev to fix, instead of letting NCCL fail later or a rank run on
half the fabric.
- VLLM_ARX_RING, ARX_RING_PREV_HCAS and ARX_RING_NEXT_HCAS for arx.

NCCL reads its environment on first use, so this must run before vLLM
imports it.
"""
import os
import socket
import struct
import sys
from typing import Optional

# Where /sys is mounted; the tests point this at a fake tree.
SYS_ROOT = "/sys"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _gid_ipv4(gid: str) -> Optional[str]:
    """The IPv4 behind a v4-mapped GID such as
    0000:0000:0000:0000:0000:ffff:0a63:0301, or None for any other GID."""
    groups = gid.split(":")
    if len(groups) != 8 or groups[5] != "ffff" or any(g != "0000" for g in groups[:5]):
        return None
    try:
        return socket.inet_ntoa(struct.pack(">HH", int(groups[6], 16), int(groups[7], 16)))
    except (ValueError, OSError):
        return None


def _netdevs(dev: str) -> list[str]:
    try:
        return sorted(os.listdir(f"{SYS_ROOT}/class/infiniband/{dev}/device/net"))
    except OSError:
        return []


def _has_ipv4_gid(dev: str) -> bool:
    """Whether dev has a RoCE v2 GID for an IPv4 address on one of its own
    netdevs. The driver fills the GID table from the netdevs' addresses, so an
    unaddressed netdev leaves its device with link-local GIDs only."""
    port = f"{SYS_ROOT}/class/infiniband/{dev}/ports/1"
    netdevs = set(_netdevs(dev))
    try:
        names = os.listdir(f"{port}/gids")
    except OSError:
        return False
    return any(
        _read(f"{port}/gid_attrs/types/{n}") == "RoCE v2"
        and _read(f"{port}/gid_attrs/ndevs/{n}") in netdevs
        and _gid_ipv4(_read(f"{port}/gids/{n}")) is not None
        for n in names
    )


def _port_devices(iface: str) -> list[str]:
    """RDMA devices behind the port that carries iface, one per PCIe root.

    Each ConnectX-7 port is two PCI functions on two roots with the same
    bus:device.function (0000:01:00.1 and 0002:01:00.1), and each function
    has its own RDMA device. Sorted by PCI address, which is also the order
    libibverbs and NCCL list them in.
    """
    pci = os.path.basename(os.path.realpath(f"{SYS_ROOT}/class/net/{iface}/device"))
    bdf = pci.split(":", 1)[1]
    devs = []
    for fn in sorted(os.listdir(f"{SYS_ROOT}/bus/pci/devices")):
        if fn.split(":", 1)[1] != bdf:
            continue
        ib = f"{SYS_ROOT}/bus/pci/devices/{fn}/infiniband"
        if os.path.isdir(ib):
            devs += [(fn, d) for d in os.listdir(ib)]
    return [d for _, d in sorted(devs)]


def _check_port(side: str, iface: str, devs: list[str]) -> None:
    """Stop the rank unless the port has two RDMA devices, each addressed."""
    where = f"the port toward {side} ({iface})"
    if len(devs) != 2:
        raise RuntimeError(
            f"fabric_ring: {where} has {len(devs)} RDMA device(s) {devs}; a ring needs "
            "two, one per PCIe root. Check that both roots' functions of this port "
            "are up (rdma link show).")
    bare = [d for d in devs if not _has_ipv4_gid(d)]
    if bare:
        names = ", ".join(f"{d} (netdev {'/'.join(_netdevs(d)) or 'none'})" for d in bare)
        raise RuntimeError(
            f"fabric_ring: on {where}, {names} has no IPv4 RoCE v2 GID. Give that "
            "netdev an address in its own subnet, matching the neighbour's same "
            "root, or NCCL and arx cannot open queue pairs on it.")


def _graph_xml(prev_net: list[int], next_net: list[int], nchannels: int) -> str:
    """A ring graph for one GPU per node: channel c receives on NCCL net device
    prev_net[c % 2] and sends on next_net[c % 2], alternating PCIe roots."""
    chans = "".join(
        f'<channel><net dev="{prev_net[c % 2]}"/><gpu dev="0"/>'
        f'<net dev="{next_net[c % 2]}"/></channel>'
        for c in range(nchannels)
    )
    # The speeds and path types are what NCCL computes for its own ring on
    # these boxes. It picks protocols and chunk sizes from them, which set
    # the order its reductions sum in.
    return (
        '<graphs version="1">'
        f'<graph id="0" pattern="4" crossnic="1" nchannels="{nchannels}" speedintra="0.24" '
        'speedinter="0.24" latencyinter="0" typeintra="LOC" typeinter="P2C" samechannels="1">'
        f"{chans}</graph></graphs>\n"
    )


def _nets(name: str, computed: list[int], ndevs: int) -> list[int]:
    """NCCL net device numbers for one side of the ring: `computed`, unless the
    environment variable `name` lists one number per PCI root (for example
    FABRIC_RING_NEXT_NET=1,3). The override lets a box whose cables do not
    match the computed numbering set them by hand."""
    v = os.environ.get(name, "").strip()
    if not v:
        return computed
    try:
        nets = [int(x) for x in v.split(",")]
    except ValueError:
        raise RuntimeError(f"{name}={v!r}: want comma-separated NCCL net device numbers, one per PCI root")
    if len(nets) != len(computed) or not all(0 <= n < ndevs for n in nets):
        raise RuntimeError(f"{name}={v!r}: want {len(computed)} numbers, each from 0 to {ndevs - 1}")
    return nets


def _setup() -> None:
    if os.environ.get("MENTAT_FABRIC_LAYOUT") != "ring":
        return
    prev_iface = os.environ["MENTAT_FABRIC_PREV_IFACE"]
    nxt_iface = os.environ["MENTAT_FABRIC_NEXT_IFACE"]
    prev = _port_devices(prev_iface)
    nxt = _port_devices(nxt_iface)
    _check_port("prev", prev_iface, prev)
    _check_port("next", nxt_iface, nxt)
    # On a switch both "ports" can be the same one; list each device once.
    devs = sorted(set(prev + nxt), key=lambda d: os.path.realpath(f"{SYS_ROOT}/class/infiniband/{d}/device"))
    index = {d: i for i, d in enumerate(devs)}
    nchannels = int(os.environ.get("NCCL_MAX_NCHANNELS") or 8)
    path = f"/tmp/nccl-ring-graph.{os.getpid()}.xml"
    prev_net = _nets("FABRIC_RING_PREV_NET", [index[d] for d in prev], len(devs))
    next_net = _nets("FABRIC_RING_NEXT_NET", [index[d] for d in nxt], len(devs))
    if os.environ.get("FABRIC_RING_DEBUG") == "1":
        sys.stderr.write(
            f"fabric_ring: rank {os.environ.get('MENTAT_FABRIC_RANK', '?')} of {os.environ.get('MENTAT_FABRIC_SIZE', '?')}, "
            f"prev {prev_iface} {prev} -> nets {prev_net}, next {nxt_iface} {nxt} -> nets {next_net}, "
            f"NCCL_IB_HCA order {devs}\n")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,NET")
    with open(path, "w") as f:
        f.write(_graph_xml(prev_net, next_net, nchannels))
    if os.environ.get("FABRIC_RING_GRAPH", "1") != "0":
        os.environ.update({"NCCL_ALGO": "Ring", "NCCL_GRAPH_FILE": path})
    else:
        # NCCL takes one device per channel for both directions of a ring
        # graph, so in a triangle a graph sends to next over the cable to
        # prev. Subnet-aware routing opens each peer's queue pairs on the
        # device in that peer's subnet.
        os.environ["NCCL_IB_SUBNET_AWARE_ROUTING"] = "1"
    os.environ.update({
        "NCCL_IB_HCA": "=" + ",".join(devs),
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_CROSS_NIC": "1",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "VLLM_ARX_RING": "1",
        "ARX_RING_PREV_HCAS": ",".join(prev),
        "ARX_RING_NEXT_HCAS": ",".join(nxt),
    })
    os.environ.pop("NCCL_IB_GID_INDEX", None)


def _main() -> None:
    # Python prints an exception raised in a .pth import and carries on, which
    # would start the rank without its ring env. Stop the process instead.
    try:
        _setup()
    except Exception as e:  # noqa: BLE001 - any failure here leaves the ring unset
        sys.stderr.write(f"FATAL: {e}\n" if isinstance(e, RuntimeError) else f"FATAL: fabric_ring: {e!r}\n")
        sys.stderr.flush()
        os._exit(1)


_main()
