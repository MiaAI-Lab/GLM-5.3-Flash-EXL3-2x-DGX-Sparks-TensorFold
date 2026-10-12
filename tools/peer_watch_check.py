#!/usr/bin/env python3
"""Patch 0115's checks (issue #129: a worker restarted alone could never rejoin, and rank 0 served a dead pair).

CPU only, no GPU and no model: the ranks are processes on one machine meeting at TensorFold's own rendezvous,
`tensorfold.cuda.comm.NCCL` with the real TCPStore, with only the NCCL library calls replaced by a stand-in (the
check needs no CUDA; the stand-in records the NCCL id each rank starts with). A clean stop goes through TensorFold's
own `cuda.http.serve` (a SIGTERM to rank 0, whose shutdown hook stands in for the spill tier's flush). Run it in the
image:

    docker run --rm --network none --entrypoint python \\
      -v "$PWD/tools/peer_watch_check.py:/x.py:ro" <image> /x.py

or with any python that has torch and TensorFold's package on its path. Each case prints PASS or FAIL; exit 1 on
any FAIL. TENSORFOLD_PEER_TIMEOUT_S is 2 in the cases unless they say otherwise. The cases:
  gone      rank 1 killed and not back: rank 0 ends (code 1) within the timeout plus a few beats
  back      rank 1 killed and started again at once: rank 0 ends ("started again"), the new rank 1 never starts
            with the old rank 0's NCCL id, and a restart of both ranks forms a new pair with a new id
  idle      a healthy pair, rank 1 waiting on the doorbell (the engine's store connection busy for an hour), for 4x
            the timeout: both ranks keep running
  headloss  rank 0 killed while rank 1 waits in a collective that never returns: rank 1 ends (code 1), not before
            the timeout and within it plus a few beats
  orphan    three ranks, rank 1 in a collective: rank 2 killed, rank 0 ends, then rank 1 ends too (the rank that
            did not restart; without it three ranks never form again under a restart policy)
  stop      a clean stop of rank 0 (SIGTERM, its shutdown taking 1.5 s) while rank 1 is still writing: rank 1 still
            writes 5 s after rank 0's exit (more than the timeout), then exits 0; no rank ended by the watch
  cadence   at TENSORFOLD_PEER_TIMEOUT_S=40 a rank beats every 2 s (not an eighth of 40)
  stopkill  rank 1 killed during rank 0's clean stop (4 s): rank 0 finishes the stop and exits 0
  setting   TENSORFOLD_PEER_TIMEOUT_S: unset 120, 0 ends no rank, a bad value fails at start
On TensorFold without the patch, `gone`, `back`, `headloss`, `orphan`, `cadence` and `setting` fail.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

FAILS: list[str] = []
ENDED = "ends, since a rank that left"          # the watch's own line when it ends a rank


def result(ok: bool, case: str, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'} {case}: {what}", flush=True)
    if not ok:
        FAILS.append(case)


# ------------------------------------------------------------------------------------------------- one rank (a child)
class _Fn:
    """A stand-in for one ctypes function: argtypes / restype assignable, as comm.NCCL sets them."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class _FakeNccl:
    """ncclCommInitRank as NCCL's: collective, it returns once every rank called it with the same id (a marker file
    a rank in PEER_CHECK_DIR); the id each rank starts with is printed (UID ...)."""

    def __init__(self) -> None:
        def unique_id(ref):
            ref._obj.internal[:] = [b - 128 for b in os.urandom(128)]
            return 0

        def init_rank(comm_ref, world, uid, rank):
            key = bytes(b & 0xFF for b in uid.internal).hex()[:32]
            print(f"UID {key}", flush=True)
            meet = Path(os.environ["PEER_CHECK_DIR"])
            (meet / f"{key}.{rank}").touch()
            while not all((meet / f"{key}.{r}").exists() for r in range(world)):
                time.sleep(0.02)
            return 0

        self.ncclGetErrorString = _Fn(lambda code: b"stand-in")
        self.ncclGetUniqueId = _Fn(unique_id)
        self.ncclCommInitRank = _Fn(init_rank)
        for name in ("ncclAllGather", "ncclSend", "ncclRecv", "ncclGroupStart", "ncclGroupEnd"):
            setattr(self, name, _Fn(lambda *a: 0))


class _App:
    """What ``cuda.http.serve`` needs of an app at a stop: ``on_exit`` (the spill tier's flush hook) takes ``hold`` s."""

    def __init__(self, hold: float) -> None:
        self.hold = hold

    def on_exit(self) -> None:
        time.sleep(self.hold)
        print("HOOK DONE", flush=True)


def child(role: str, port: int, hold: float, world: int, rank: int) -> int:
    import torch

    from tensorfold.cuda import comm

    comm._library = _FakeNccl                         # the NCCL library calls only; the rendezvous is TensorFold's
    torch.cuda.current_device = lambda: 0
    nccl = comm.NCCL(rank, world, "127.0.0.1", port)
    print("READY", flush=True)
    if role == "rank0":                               # serving: the HTTP thread and the engine wait on other things
        while True:
            time.sleep(3600)
    if role == "serve":                               # rank 0 serving through TensorFold's server loop until SIGTERM
        from tensorfold.cuda.http import serve

        serve(_App(hold), "127.0.0.1", int(os.environ["PEER_CHECK_HTTP"]))    # SIGTERM's handler set as it listens
        print("STOPPED", flush=True)
        return 0
    if role == "busy":                                # writing (a clean stop's flush) until released, at most ``hold`` s
        end = time.monotonic() + hold
        while not (Path(os.environ["PEER_CHECK_DIR"]) / "release").exists() and time.monotonic() < end:
            time.sleep(0.05)
        print("FLUSHED", flush=True)
        return 0
    if role == "stuck":                               # in a collective with a rank that is gone: never returns
        while True:
            time.sleep(3600)
    nccl.store.wait(["tf_bell/never"], timedelta(hours=1))   # an idle rank past 0: the doorbell, as the engine waits
    return 0


# ------------------------------------------------------------------------------------------------------- the cases
class Rank:
    def __init__(self, role: str, port: int, timeout: str | None, tmp: Path, name: str, hold: float = 0.0,
                 world: int = 2, rank: int | None = None) -> None:
        env = dict(os.environ)
        env.pop("TENSORFOLD_PEER_TIMEOUT_S", None)
        if timeout is not None:
            env["TENSORFOLD_PEER_TIMEOUT_S"] = timeout
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PEER_CHECK_DIR"] = str(tmp)
        self.http = free_port()                       # rank 0's API port in the "serve" role
        env["PEER_CHECK_HTTP"] = str(self.http)
        if rank is None:
            rank = 0 if role in ("rank0", "serve") else 1
        self.log = tmp / f"{name}.log"
        self.f = open(self.log, "w")
        self.p = subprocess.Popen([sys.executable, __file__, "--role", role, str(port), str(hold), str(world),
                                   str(rank)], env=env, stdout=self.f, stderr=subprocess.STDOUT)
        self.ended_at: float | None = None

    def text(self) -> str:
        self.f.flush()
        return self.log.read_text(errors="replace")

    def uid(self) -> str | None:
        lines = [ln for ln in self.text().splitlines() if ln.startswith("UID ")]
        return lines[0][4:] if lines else None

    def wait_for(self, word: str, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if word in self.text():
                return True
            if self.p.poll() is not None:
                return word in self.text()
            time.sleep(0.05)
        return False

    def exited_within(self, seconds: float) -> int | None:
        try:
            code = self.p.wait(seconds)
        except subprocess.TimeoutExpired:
            return None
        self.ended_at = time.monotonic()
        return code

    def stop(self) -> None:
        """A clean stop, as docker stop sends it: SIGTERM once the server listens (its handler is set by then)."""

        end = time.monotonic() + 60
        while time.monotonic() < end and self.p.poll() is None:
            try:
                socket.create_connection(("127.0.0.1", self.http), timeout=1).close()
                break
            except OSError:
                time.sleep(0.05)
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGKILL)
            self.p.wait()
            self.ended_at = time.monotonic()

    def tail(self) -> str:
        return " | ".join(self.text().strip().splitlines()[-3:])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def group(port: int, timeout: str | None, tmp: Path, tag: str, roles: list[str], holds: list[float] | None = None):
    """The ranks, highest first (as start.sh launches them); True when they all formed one set with one id."""

    world = len(roles)
    ranks: list[Rank] = [None] * world   # type: ignore[list-item]
    for r in range(world - 1, -1, -1):
        ranks[r] = Rank(roles[r], port, timeout, tmp, f"{tag}-r{r}", (holds or [0.0] * world)[r], world, r)
    ok = all(x.wait_for("READY", 60) for x in ranks) and len({x.uid() for x in ranks}) == 1
    return ranks, ok


def pair(port: int, timeout: str | None, tmp: Path, tag: str, r1_role: str = "rank1", r0_role: str = "rank0",
         holds: list[float] | None = None):
    (r0, r1), ok = group(port, timeout, tmp, tag, [r0_role, r1_role], holds)
    return r0, r1, ok


def case_gone(tmp: Path) -> None:
    r0, r1, ok = pair(free_port(), "2", tmp, "gone")
    try:
        result(ok, "gone", "the pair forms")
        if not ok:
            return
        r1.kill()
        code = r0.exited_within(2 + 4)
        result(code == 1, "gone", f"rank 0 ends with code 1 within 6 s of rank 1's death (got {code}; {r0.tail()})")
        result("rank 1 showed no sign of life" in r0.text(), "gone", "rank 0 says why it ended")
    finally:
        r0.kill(); r1.kill()


def case_back(tmp: Path) -> None:
    port = free_port()
    r0, r1, ok = pair(port, "30", tmp, "back")
    rb = rr0 = rr1 = None
    try:
        result(ok, "back", "the pair forms")
        if not ok:
            return
        old = r0.uid()
        r1.kill()
        rb = Rank("rank1", port, "30", tmp, "back-r1b")             # the worker's container, back at once
        from torch.distributed import TCPStore

        store = TCPStore("127.0.0.1", port, 2, False, timeout=timedelta(seconds=10), wait_for_workers=False)
        end = time.monotonic() + 60                                 # until it reaches the rendezvous (imports first)
        while time.monotonic() < end and rb.uid() is None and rb.p.poll() is None:
            try:
                if store.check(["tf_join/1"]) and store.add("tf_join/1", 0) > 1:
                    break
            except Exception:                    # noqa: BLE001  (rank 0, and its store, already gone)
                break
            time.sleep(0.05)
        code = r0.exited_within(6)
        result(code == 1, "back", f"rank 0 ends with code 1 within 6 s of a new rank 1 reaching the rendezvous (got "
                                  f"{code}; {r0.tail()})")
        result("rank 1 started again" in r0.text(), "back", "rank 0 says why it ended")
        rb_code = rb.exited_within(15)
        result(rb.uid() != old, "back", f"the new rank 1 never starts with the old rank 0's NCCL id ({rb.uid()})")
        result(rb_code not in (None, 0), "back", f"the new rank 1 ends once that rank 0 is gone (got {rb_code})")
        r0.kill()
        rr1 = Rank("rank1", port, "30", tmp, "back-r1c")            # the supervisor restarts both ranks
        rr0 = Rank("rank0", port, "30", tmp, "back-r0b")
        ok2 = rr0.wait_for("READY", 60) and rr1.wait_for("READY", 60)
        result(ok2 and rr0.uid() == rr1.uid() and rr0.uid() != old, "back",
               f"the restarted ranks form a new pair with a new id ({rr0.uid()} / {rr1.uid()})")
    finally:
        for r in (r0, r1, rb, rr0, rr1):
            if r is not None:
                r.kill()


def case_idle(tmp: Path) -> None:
    r0, r1, ok = pair(free_port(), "2", tmp, "idle")
    try:
        result(ok, "idle", "the pair forms")
        if not ok:
            return
        code = r0.exited_within(8)
        result(code is None, "idle", f"rank 0 keeps running for 4x the timeout beside an idle rank 1 (got {code}; "
                                     f"{r0.tail()})")
        result(r1.p.poll() is None, "idle", f"rank 1 keeps running ({r1.tail()})")
    finally:
        r0.kill(); r1.kill()


def follower_ends(case: str, follower: Rank, gone_at: float, timeout: float, label: str) -> None:
    """``follower`` (in a collective that never returns) must end with code 1 no sooner than the timeout less two
    beats after rank 0 went at ``gone_at``, and no later than the timeout plus 4 s, saying why."""

    code = follower.exited_within(timeout + 4 + 2)
    after = (follower.ended_at - gone_at) if follower.ended_at is not None else None
    result(code == 1 and after is not None and timeout - 0.5 <= after <= timeout + 4, case,
           f"{label} ends with code 1 between {timeout - 0.5:g} and {timeout + 4:g} s after rank 0 went (got {code}"
           f"{'' if after is None else f' after {after:.1f} s'}; {follower.tail()})")
    result("rank 0 showed no sign of life" in follower.text(), case, f"{label} says why it ended")


def case_headloss(tmp: Path) -> None:
    r0, r1, ok = pair(free_port(), "2", tmp, "headloss", r1_role="stuck")
    try:
        result(ok, "headloss", "the pair forms")
        if not ok:
            return
        time.sleep(1)
        r0.kill()
        follower_ends("headloss", r1, r0.ended_at, 2.0, "rank 1")
    finally:
        r0.kill(); r1.kill()


def case_orphan(tmp: Path) -> None:
    (r0, r1, r2), ok = group(free_port(), "2", tmp, "orphan", ["rank0", "stuck", "rank1"])
    try:
        result(ok, "orphan", "three ranks form")
        if not ok:
            return
        r2.kill()
        code = r0.exited_within(2 + 4)
        result(code == 1 and "rank 2 showed no sign of life" in r0.text(), "orphan",
               f"rank 0 ends with code 1 on rank 2's death (got {code}; {r0.tail()})")
        if code is not None:
            follower_ends("orphan", r1, r0.ended_at, 2.0, "rank 1 (the rank that did not restart)")
    finally:
        r0.kill(); r1.kill(); r2.kill()


def case_stop(tmp: Path) -> None:
    r0, r1, ok = pair(free_port(), "2", tmp, "stop", r1_role="busy", r0_role="serve", holds=[1.5, 120.0])
    try:
        result(ok, "stop", "the pair forms")
        if not ok:
            return
        r0.stop()
        code0 = r0.exited_within(30)
        result(code0 == 0 and "HOOK DONE" in r0.text() and "STOPPED" in r0.text(), "stop",
               f"rank 0 runs its shutdown hook and exits 0 (got {code0}; {r0.tail()})")
        time.sleep(2 + 3)                                # the timeout and more: a watch that ends rank 1 has by now
        alive = r1.p.poll() is None
        (tmp / "release").touch()                        # rank 1's writing is done
        code1 = r1.exited_within(10)
        result(alive and code1 == 0 and "FLUSHED" in r1.text(), "stop",
               f"rank 1 is still writing 5 s after rank 0's exit, then finishes and exits 0 (got {code1}; "
               f"{r1.tail()})")
        result(ENDED not in r0.text() + r1.text(), "stop", "the watch ended no rank")
    finally:
        r0.kill(); r1.kill()


def case_cadence(tmp: Path) -> None:
    port = free_port()
    r0, r1, ok = pair(port, "40", tmp, "cadence")
    try:
        result(ok, "cadence", "the pair forms")
        if not ok:
            return
        from torch.distributed import TCPStore

        store = TCPStore("127.0.0.1", port, 2, False, timeout=timedelta(seconds=10), wait_for_workers=False)
        first = store.add("tf_alive/1", 0)
        time.sleep(10.5)
        beats = store.add("tf_alive/1", 0) - first
        result(beats >= 4, "cadence", f"at TENSORFOLD_PEER_TIMEOUT_S=40 rank 1 beats every 2 s, not every 5 "
                                      f"({beats} beats in 10.5 s), so a clean stop's mark is seen within 2 s")
    finally:
        r0.kill(); r1.kill()


def case_stopkill(tmp: Path) -> None:
    r0, r1, ok = pair(free_port(), "2", tmp, "stopkill", r0_role="serve", holds=[4.0, 0.0])
    try:
        result(ok, "stopkill", "the pair forms")
        if not ok:
            return
        r0.stop()
        time.sleep(0.3)
        r1.kill()
        code0 = r0.exited_within(15)
        result(code0 == 0 and "STOPPED" in r0.text() and ENDED not in r0.text(), "stopkill",
               f"rank 0 finishes its clean stop and exits 0 though rank 1 died during it (got {code0}; {r0.tail()})")
    finally:
        r0.kill(); r1.kill()


def case_setting(tmp: Path) -> None:
    from tensorfold.cuda import comm

    fn = getattr(comm, "peer_timeout", None)
    result(fn is not None, "setting", "comm.peer_timeout exists")
    if fn is not None:
        result(fn("") == 120.0 and fn("0") == 0.0 and fn("2.5") == 2.5, "setting", "unset 120, 0 and 2.5 read as such")
        for bad in ("abc", "-1", "nan"):
            try:
                fn(bad)
                result(False, "setting", f"{bad!r} is refused")
            except ValueError:
                result(True, "setting", f"{bad!r} is refused")
    r0, r1, ok = pair(free_port(), "0", tmp, "off")
    try:
        result(ok, "setting", "0: the pair forms")
        if ok:
            r1.kill()
            result(r0.exited_within(4) is None, "setting", "0: rank 0 is never ended by the watch")
    finally:
        r0.kill(); r1.kill()
    r0, r1, ok = pair(free_port(), "0", tmp, "off1", r1_role="stuck")
    try:
        if ok:
            r0.kill()
            result(r1.exited_within(4) is None, "setting", "0: a rank past 0 is never ended by the watch")
    finally:
        r0.kill(); r1.kill()
    port = free_port()
    bad = Rank("rank0", port, "soon", tmp, "bad-r0")
    try:
        code = bad.exited_within(30)
        result(code not in (None, 0) and "TENSORFOLD_PEER_TIMEOUT_S" in bad.text(), "setting",
               f"a bad value fails rank 0 at start, naming the setting (got {code}; {bad.tail()})")
    finally:
        bad.kill()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--role":
        return child(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    import torch

    from tensorfold.cuda import comm

    print(f"torch {torch.__version__}, {comm.__file__}", flush=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for case in (case_gone, case_back, case_idle, case_headloss, case_orphan, case_stop, case_cadence,
                     case_stopkill, case_setting):
            case(tmp)
    print(f"{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failed check(s){': ' + ', '.join(sorted(set(FAILS))) if FAILS else ''}",
          flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
