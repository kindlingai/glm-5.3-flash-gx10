#!/usr/bin/env python3
"""image/fabric_ring.py: the NCCL and arx env a ring of boxes boots with.

Fakes /sys/class/infiniband, /sys/class/net and /sys/bus/pci under a temp tree
(the module's SYS_ROOT) and checks:
  - with both PCIe roots of each port addressed, NCCL_IB_HCA, the graph
    channels and the ARX_RING_* env are what they were before the port check
  - NCCL_IB_GID_INDEX is dropped: ring mode picks each device's GID by its
    own address, and the cables' indexes differ
  - with the second root's netdevs unaddressed (no IPv4 RoCE v2 GID), setup
    stops and names the device, its netdev and the fix
  - a port with no RDMA devices stops setup and names the interface
  - a failure at interpreter start exits the process with status 1 instead of
    letting Python print the error and carry on
CPU only, reads nothing from the real host:

    python3 dev/patch-tests/_fabric_ring_test.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fabric_ring_under_test",
    os.path.join(HERE, "..", "..", "image", "fabric_ring.py"),
)
fabric_ring = importlib.util.module_from_spec(_spec)
os.environ.pop("MENTAT_FABRIC_LAYOUT", None)  # keep the import a no-op
_spec.loader.exec_module(fabric_ring)

# Device names follow the real boxes: first root roce<lowercase>*, second
# root roceP2*, both roots' functions sharing one BDF tail.
A_IFACE = "enp1s0f0np0"  # netdev of the port toward prev, first root
B_IFACE = "enp1s0f1np1"  # netdev of the port toward next, first root
A_DEVS = ("rocep1s0f0", "roceP2p1s0f0")  # pci 0000:01:00.1, 0002:01:00.1
B_DEVS = ("rocep1s0f1", "roceP2p1s0f1")  # pci 0000:01:00.2, 0002:01:00.2
LINK_LOCAL = "fe80:0000:0000:0000:0000:0000:0000:0002"


def v4_gid(a: int, b: int, c: int, d: int) -> str:
    return "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" % (a, b, c, d)


# bdf per device, first root then second
# (pci bus:device.function, RDMA device, the netdev of the device itself).
# The second root is domain 0002, as on the real boxes.
PORTS = [
    ("0000:01:00.1", A_DEVS[0], "enp1s0f0np0"),
    ("0000:01:00.2", B_DEVS[0], "enp1s0f1np1"),
    ("0002:01:00.1", A_DEVS[1], "enP2p1s0f0np0"),
    ("0002:01:00.2", B_DEVS[1], "enP2p1s0f1np1"),
]
# Sorting key the module uses for the union: the real path of each device's
# PCI function, so first-root devices sort before second-root ones here.
UNION_ORDER = (
    A_DEVS[0], B_DEVS[0], A_DEVS[1], B_DEVS[1]
)


class FakeSys:
    """A /sys tree with two ConnectX ports (prev, next) and one device per
    root per port. A device that is not in `addressed` has only link-local
    GIDs, like the real second root before an address is put on its netdev."""

    def __init__(self, addressed: dict[str, tuple[int, int, int, int]]):
        self.root = tempfile.mkdtemp()
        self.sys = os.path.join(self.root, "sys")
        for bdf, dev, netdev in PORTS:
            base = os.path.join(self.sys, "class", "infiniband", dev)
            self._gid(dev, 0, LINK_LOCAL, "RoCE v1", netdev)
            self._gid(dev, 1, LINK_LOCAL, "RoCE v2", netdev)
            if dev in addressed:
                self._gid(dev, 2, v4_gid(*addressed[dev]), "RoCE v2", netdev)
            pci = os.path.join(self.sys, "bus", "pci", "devices", bdf)
            # The PCI function owns its netdevs; both are reached through
            # /sys/class/infiniband/<dev>/device.
            os.makedirs(os.path.join(pci, "net"), exist_ok=True)
            open(os.path.join(pci, "net", netdev), "w").close()
            pci_ib = os.path.join(pci, "infiniband")
            os.makedirs(pci_ib)
            os.symlink(os.path.relpath(base, pci_ib), os.path.join(pci_ib, dev))
            # /sys/class/infiniband/<dev>/device -> its PCI function
            os.symlink(os.path.relpath(pci, base), os.path.join(base, "device"))
            # The addressed (first-root) netdevs are the ones mentat names;
            # each points at its own PCI function.
            if bdf.startswith("0000:"):
                net = os.path.join(self.sys, "class", "net", netdev)
                os.makedirs(net, exist_ok=True)
                os.symlink(os.path.relpath(pci, net), os.path.join(net, "device"))

    def _gid(self, dev: str, index: int, gid: str, kind: str, ndev: str) -> None:
        port = os.path.join(self.sys, "class", "infiniband", dev, "ports", "1")
        for sub, value in (("gids", gid), (os.path.join("gid_attrs", "types"), kind),
                           (os.path.join("gid_attrs", "ndevs"), ndev)):
            path = os.path.join(port, sub)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, str(index)), "w") as f:
                f.write(value + "\n")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.root)


ENV_KEYS = ("MENTAT_FABRIC_LAYOUT", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
            "NCCL_ALGO", "NCCL_GRAPH_FILE", "NCCL_MAX_NCHANNELS",
            "VLLM_ARX_RING", "ARX_RING_PREV_HCAS", "ARX_RING_NEXT_HCAS")


class RingTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        self.fake = FakeSys({})
        os.environ["MENTAT_FABRIC_LAYOUT"] = "ring"
        os.environ["MENTAT_FABRIC_PREV_IFACE"] = A_IFACE
        os.environ["MENTAT_FABRIC_NEXT_IFACE"] = B_IFACE
        os.environ.pop("NCCL_MAX_NCHANNELS", None)
        os.environ.pop("NCCL_IB_GID_INDEX", None)
        for k in ("VLLM_ARX_RING", "ARX_RING_PREV_HCAS", "ARX_RING_NEXT_HCAS"):
            os.environ.pop(k, None)
        fabric_ring.SYS_ROOT = self.fake.sys

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.fake.cleanup()

    # -- helpers ----------------------------------------------------------

    def addr(self, second_root: bool = True):
        """Addressed dict: first-root devices 10.99.x.1, second 10.99.x.2."""
        out = {A_DEVS[0]: (10, 99, 3, 1), B_DEVS[0]: (10, 99, 2, 2)}
        if second_root:
            out[A_DEVS[1]] = (10, 99, 13, 1)
            out[B_DEVS[1]] = (10, 99, 22, 1)
        return out

    # -- two devices per port: the old RING4 behaviour, unchanged ---------

    def test_two_devices_per_port_unchanged(self):
        fake = FakeSys(self.addr())
        try:
            fabric_ring.SYS_ROOT = fake.sys
            os.environ["NCCL_IB_GID_INDEX"] = "5"
            fabric_ring._setup()
            self.assertEqual(
                os.environ["NCCL_IB_HCA"],
                "=" + ",".join(UNION_ORDER),
                "first-root devices then second-root, sorted by PCI address",
            )
            self.assertNotIn("NCCL_IB_GID_INDEX", os.environ)
            self.assertEqual(os.environ["NCCL_ALGO"], "Ring")
            graph = ET.parse(os.environ["NCCL_GRAPH_FILE"]).getroot()
            self.assertEqual(graph.tag, "graphs")
            g = graph.find("graph")
            self.assertEqual(g.get("nchannels"), "8")
            chans = g.findall("channel")
            self.assertEqual(len(chans), 8)
            prev = [UNION_ORDER.index(d) for d in A_DEVS]
            nxt = [UNION_ORDER.index(d) for d in B_DEVS]
            for c, ch in enumerate(chans):
                net = [int(d.get("dev")) for d in ch.findall("net")]
                self.assertEqual(net, [prev[c % 2], nxt[c % 2]])
                self.assertEqual([int(x.get("dev")) for x in ch.findall("gpu")], [0])
            # arx ring env with both devices per side, roots in order
            self.assertEqual(os.environ["VLLM_ARX_RING"], "1")
            self.assertEqual(os.environ["ARX_RING_PREV_HCAS"], ",".join(A_DEVS))
            self.assertEqual(os.environ["ARX_RING_NEXT_HCAS"], ",".join(B_DEVS))
        finally:
            fake.cleanup()

    def test_unaddressed_second_root_names_netdev(self):
        fake = FakeSys(self.addr(second_root=False))
        try:
            fabric_ring.SYS_ROOT = fake.sys
            with self.assertRaises(RuntimeError) as cm:
                fabric_ring._setup()
            msg = str(cm.exception)
            self.assertIn(A_IFACE, msg)
            self.assertIn(A_DEVS[1], msg)
            self.assertIn("enP2p1s0f0np0", msg)
            self.assertIn("address", msg)
            self.assertNotIn("NCCL_GRAPH_FILE", os.environ)
        finally:
            fake.cleanup()

    def test_port_without_devices_names_iface(self):
        fake = FakeSys(self.addr())
        try:
            fabric_ring.SYS_ROOT = fake.sys
            os.environ["MENTAT_FABRIC_NEXT_IFACE"] = "enp9s0f0np0"
            os.makedirs(os.path.join(fake.sys, "bus", "pci", "devices", "0000:09:00.0"))
            net = os.path.join(fake.sys, "class", "net", "enp9s0f0np0")
            os.makedirs(net)
            os.symlink(os.path.join(fake.sys, "bus", "pci", "devices", "0000:09:00.0"),
                       os.path.join(net, "device"))
            with self.assertRaisesRegex(RuntimeError, r"toward next \(enp9s0f0np0\) has 0 RDMA"):
                fabric_ring._setup()
        finally:
            fake.cleanup()


class StartupTest(unittest.TestCase):
    def test_failure_exits_the_process(self):
        env = dict(os.environ, MENTAT_FABRIC_LAYOUT="ring",
                   MENTAT_FABRIC_PREV_IFACE="no-such-iface0", MENTAT_FABRIC_NEXT_IFACE="no-such-iface1")
        path = os.path.join(HERE, "..", "..", "image", "fabric_ring.py")
        r = subprocess.run([sys.executable, "-c", f"import runpy; runpy.run_path({path!r}); print('continued')"],
                           env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("continued", r.stdout)
        self.assertIn("FATAL", r.stderr)


if __name__ == "__main__":
    unittest.main()