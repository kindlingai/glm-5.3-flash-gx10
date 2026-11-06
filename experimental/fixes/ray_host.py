"""Actor host: the process the mentat agent spawns for each actor.

DIAGNOSTIC COPY (hermes, 2026-10-02): adds SIGTERM interception with a
stack dump of all threads and faulthandler.enable(), plus traceback.print_exc()
in both except BaseException handlers per NOTES.md, to find who kills the
head worker with signal 15 during loading.
"""

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

import faulthandler
import signal
import traceback

waiting_for_dump = {"flag": False}


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _on_sigterm(signum, frame):
    _log("=== HERMES DIAG: caught SIGTERM!! ===")
    _log("signal=%s frame=%r" % (signum, frame))
    _log("--- threading.enumerate() ---")
    for th in threading.enumerate():
        _log("  thread %s daemon=%s alive=%s" % (th, th.daemon, th.is_alive()))
    _log("--- stacks of all threads ---")
    for th_id, stack in sys._current_frames().items():
        _log("--- thread 0x%x ---" % th_id)
        traceback.print_stack(stack, file=sys.stderr)
    faulthandler.dump_traceback_later(1, repeat=False)
    waiting_for_dump["flag"] = True
    # Keep the process alive so agents/daemon see it and we can probe it.
    # Do NOT exit here: we want to see whether the killer retries (SIGKILL).


def _on_sigkill_fallback():
    return


PROTO = "0.99"


def major_matches(peer):
    def major(v):
        if not isinstance(v, str):
            return None
        head, dot, tail = v.partition(".")
        if not dot or not head.isdigit() or not tail.isdigit():
            return None
        if not (head.isascii() and tail.isascii()):
            return None
        return head.lstrip("0") or "0"

    cutover = {"0", "1"}
    a, b = major(PROTO), major(peer)
    if a is None or b is None:
        return False
    return a == b or {a, b} <= cutover


def _send(sock, header, payload=b""):
    hb = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("<II", len(hb), len(payload)) + hb + payload)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("agent closed the actor socket")
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock):
    hlen, plen = struct.unpack("<II", _recv_exact(sock, 8))
    header = json.loads(_recv_exact(sock, hlen))
    payload = _recv_exact(sock, plen)
    return header, payload


def _dumps(obj):
    from ray import cloudpickle

    try:
        return cloudpickle.dumps(obj)
    except Exception as e:
        import pickle

        return pickle.dumps(RuntimeError(f"unpicklable object {type(obj).__name__}: {e!r}"))


def _watch_agent(agent_pid):
    while True:
        time.sleep(5)
        try:
            os.kill(agent_pid, 0)
        except OSError:
            _log(
                f"mentat host: agent pid {agent_pid} is gone. Exiting so this "
                "actor is not orphaned",
                flush=True,
            )
            os._exit(1)
        if os.getppid() != agent_pid:
            _log(
                "mentat host: reparented (agent died); exiting",
                flush=True,
            )
            os._exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    _send(sock, {"t": "host_hello", "req": 0, "proto": PROTO})

    agent_pid = int(os.environ.get("MENTAT_AGENT_PID", "0") or "0")
    if agent_pid:
        threading.Thread(target=_watch_agent, args=(agent_pid,), daemon=True).start()

    header, payload = _recv(sock)
    if header.get("t") != "ctor":
        _log(f"mentat host: expected ctor, got {header}")
        return 1
    offered = header.get("proto", "")
    if not major_matches(offered):
        _log(
            f"mentat host: agent proto {offered!r}, this host {PROTO}",
            flush=True,
        )
        return 1

    import pickle

    try:
        cls, ctor_args, ctor_kwargs = pickle.loads(payload)
        instance = cls(*ctor_args, **ctor_kwargs)
    except BaseException as e:  # noqa: BLE001 -- must report, then die
        _send(sock, {"t": "ctor_err", "req": 0, "error": repr(e)}, _dumps(e))
        _log("=== HERMES DIAG: exception during actor ctor ===")
        traceback.print_exc()
        raise
    _send(sock, {"t": "ctor_ok", "req": 0})
    _log("=== HERMES DIAG: ctor_ok sent, python pid=%d ===" % os.getpid())

    while True:
        header, payload = _recv(sock)
        if header.get("t") != "host_call":
            _log(f"mentat host: unexpected frame {header}")
            continue
        ref_id = header["ref_id"]
        method = header["method"]
        try:
            call_args, call_kwargs = pickle.loads(payload)
            result = getattr(instance, method)(*call_args, **call_kwargs)
            _send(sock, {"t": "host_result", "req": 0, "ref_id": ref_id, "ok": True}, _dumps(result))
        except BaseException as e:  # noqa: BLE001 -- errors belong to the caller
            _send(
                sock,
                {"t": "host_result", "req": 0, "ref_id": ref_id, "ok": False},
                _dumps(e),
            )
            _log(f"=== HERMES DIAG: exception in method {method!r} ===")
            traceback.print_exc()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    faulthandler.enable()
    faulthandler.register(signal.SIGTERM, all_threads=True)
    sys.exit(main())
