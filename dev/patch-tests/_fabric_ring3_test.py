#!/usr/bin/env python3
"""image/fabric_ring.py with FABRIC_RING_GRAPH, the switch TP=RING3 turns off.

Stubs the port lookup (no /sys needed) and checks that:
  - by default (TP=RING4) the rank gets NCCL_ALGO=Ring and the ring graph file
  - with FABRIC_RING_GRAPH=0 it gets NCCL_IB_SUBNET_AWARE_ROUTING=1 instead,
    and the same NCCL_IB_HCA and ARX_RING_* devices
CPU only:

    python3 dev/patch-tests/_fabric_ring3_test.py
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fabric_ring_under_test", os.path.join(HERE, "..", "..", "image", "fabric_ring.py"))
fabric_ring = importlib.util.module_from_spec(_spec)
os.environ.pop("MENTAT_FABRIC_LAYOUT", None)  # keep the import a no-op
_spec.loader.exec_module(fabric_ring)

PORTS = {"enp1s0f1np1": ["rocep1s0f1", "roceP2p1s0f1"],  # toward prev
         "enp1s0f0np0": ["rocep1s0f0", "roceP2p1s0f0"]}  # toward next
KEYS = ("MENTAT_FABRIC_LAYOUT", "MENTAT_FABRIC_PREV_IFACE", "MENTAT_FABRIC_NEXT_IFACE",
        "FABRIC_RING_GRAPH", "NCCL_ALGO", "NCCL_GRAPH_FILE", "NCCL_IB_SUBNET_AWARE_ROUTING",
        "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "VLLM_ARX_RING", "ARX_RING_PREV_HCAS",
        "ARX_RING_NEXT_HCAS", "NCCL_MAX_NCHANNELS")


class GraphSwitchTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in KEYS}
        for k in KEYS:
            os.environ.pop(k, None)
        os.environ.update(MENTAT_FABRIC_LAYOUT="ring", MENTAT_FABRIC_PREV_IFACE="enp1s0f1np1",
                          MENTAT_FABRIC_NEXT_IFACE="enp1s0f0np0", NCCL_IB_GID_INDEX="3")
        self.real = fabric_ring._port_devices
        fabric_ring._port_devices = lambda iface: list(PORTS[iface])

    def tearDown(self):
        fabric_ring._port_devices = self.real
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def run_setup(self):
        fabric_ring._setup()
        return {k: os.environ.get(k) for k in KEYS}

    def test_ring_graph_by_default(self):
        env = self.run_setup()
        self.assertEqual(env["NCCL_ALGO"], "Ring")
        self.assertTrue(os.path.isfile(env["NCCL_GRAPH_FILE"]))
        self.assertIsNone(env["NCCL_IB_SUBNET_AWARE_ROUTING"])

    def test_subnet_routing_without_graph(self):
        default = self.run_setup()
        for k in ("NCCL_ALGO", "NCCL_GRAPH_FILE"):
            os.environ.pop(k, None)
        os.environ["FABRIC_RING_GRAPH"] = "0"
        env = self.run_setup()
        self.assertIsNone(env["NCCL_ALGO"])
        self.assertIsNone(env["NCCL_GRAPH_FILE"])
        self.assertEqual(env["NCCL_IB_SUBNET_AWARE_ROUTING"], "1")
        self.assertIsNone(env["NCCL_IB_GID_INDEX"])
        for k in ("NCCL_IB_HCA", "VLLM_ARX_RING", "ARX_RING_PREV_HCAS", "ARX_RING_NEXT_HCAS"):
            self.assertEqual(env[k], default[k], k)
        self.assertEqual(env["ARX_RING_PREV_HCAS"], "rocep1s0f1,roceP2p1s0f1")


if __name__ == "__main__":
    unittest.main()
