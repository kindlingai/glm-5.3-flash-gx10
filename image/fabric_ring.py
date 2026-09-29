"""NCCL and arx settings for a rank that mentat placed on a ring of boxes.

Runs at interpreter start (fabric_ring.pth) and does nothing unless mentat
gave this process MENTAT_FABRIC_LAYOUT=ring, which it does for each rank of a
TP=RING4 claim. On a ring each box cables one port to the previous rank and
the other to the next, so a rank sends to next and receives from prev on
different ports, and the diagonal ranks share no cable. This sets:

- NCCL_IB_HCA to both PCIe roots' functions of both ports, NCCL_ALGO=Ring so
  NCCL only talks to neighbours, and NCCL_GRAPH_FILE to a ring whose channels
  receive on the ports toward prev and send on the ports toward next. NCCL
  cannot infer that wiring, since it assumes every NIC reaches every peer.
- the GID from each port's own address, since each cable has its own subnet
  and no one GID index fits every device.
- VLLM_ARX_RING, ARX_RING_PREV_HCAS and ARX_RING_NEXT_HCAS for arx.

NCCL reads its environment on first use, so this must run before vLLM
imports it.
"""
import os


def _port_devices(iface: str) -> list[str]:
    """RDMA devices behind the port that carries iface, one per PCIe root.

    Each ConnectX-7 port is two PCI functions on two roots with the same
    bus:device.function (0000:01:00.1 and 0002:01:00.1), and each function
    has its own RDMA device. Sorted by PCI address, which is also the order
    libibverbs and NCCL list them in.
    """
    pci = os.path.basename(os.path.realpath(f"/sys/class/net/{iface}/device"))
    bdf = pci.split(":", 1)[1]
    devs = []
    for fn in sorted(os.listdir("/sys/bus/pci/devices")):
        if fn.split(":", 1)[1] != bdf:
            continue
        ib = f"/sys/bus/pci/devices/{fn}/infiniband"
        if os.path.isdir(ib):
            devs += [(fn, d) for d in os.listdir(ib)]
    return [d for _, d in sorted(devs)]


def _graph_xml(nccl_index: dict[str, int], prev: list[str], nxt: list[str], nchannels: int) -> str:
    """A ring graph for one GPU per node: channel c receives on prev[c % 2]
    and sends on next[c % 2], alternating PCIe roots across channels."""
    chans = "".join(
        f'<channel><net dev="{nccl_index[prev[c % 2]]}"/><gpu dev="0"/>'
        f'<net dev="{nccl_index[nxt[c % 2]]}"/></channel>'
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


def _setup() -> None:
    if os.environ.get("MENTAT_FABRIC_LAYOUT") != "ring":
        return
    prev = _port_devices(os.environ["MENTAT_FABRIC_PREV_IFACE"])
    nxt = _port_devices(os.environ["MENTAT_FABRIC_NEXT_IFACE"])
    if len(prev) != 2 or len(nxt) != 2:
        raise RuntimeError(f"fabric_ring: expected two RDMA devices per port, got prev={prev} next={nxt}")
    # On a switch both "ports" can be the same one; list each device once.
    devs = sorted(set(prev + nxt), key=lambda d: os.path.realpath(f"/sys/class/infiniband/{d}/device"))
    index = {d: i for i, d in enumerate(devs)}
    nchannels = int(os.environ.get("NCCL_MAX_NCHANNELS") or 8)
    path = f"/tmp/nccl-ring-graph.{os.getpid()}.xml"
    with open(path, "w") as f:
        f.write(_graph_xml(index, prev, nxt, nchannels))
    os.environ.update({
        "NCCL_IB_HCA": "=" + ",".join(devs),
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_ALGO": "Ring",
        "NCCL_CROSS_NIC": "1",
        "NCCL_GRAPH_FILE": path,
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_ROCE_VERSION_NUM": "2",
        "VLLM_ARX_RING": "1",
        "ARX_RING_PREV_HCAS": ",".join(prev),
        "ARX_RING_NEXT_HCAS": ",".join(nxt),
    })
    os.environ.pop("NCCL_IB_GID_INDEX", None)


_setup()
