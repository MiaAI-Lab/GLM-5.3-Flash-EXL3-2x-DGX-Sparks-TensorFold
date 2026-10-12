"""CPU-only GLM queue lifecycle regression using the real scheduler classes.

Run after applying the recipe patches to TensorFold v0.6.0:
  python3 -B tools/test_queued_cancellation.py --source-root /path/to/src
Use --expect-stock before patch 0073 to reproduce the original full-lanes
gap. No Torch/CUDA, sockets, package installation or source writes are needed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import queue
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def classes(source, names, namespace):
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == len(names) + 1, "missing or duplicate source class"
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<scheduler>", "exec"), namespace)


class Decoder:
    watchdog = 0

    def __init__(self):
        self.streams, self.requeue, self.admitted, self.finished, self.fits_calls = {}, [], [], [], []
        self.room = True

    def live(self):
        return len(self.streams)

    def fits(self, stream):
        self.fits_calls.append(stream)
        return self.room

    def admit(self, stream):
        self.admitted.append(stream)
        self.streams[id(stream)] = stream

    def round(self):
        return [s for s in self.streams.values() if s.done]

    def finish(self, done):
        self.finished.extend(done)
        for stream in done:
            self.streams.pop(id(stream), None)

    def stats(self, stream):
        return stream.stats()


def wait(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "caller/scheduler did not make bounded progress"
        time.sleep(0.002)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-root", type=Path, required=True, help="directory containing tensorfold/")
    cli.add_argument("--expect-stock", action="store_true")
    args = cli.parse_args()
    root = args.source_root / "tensorfold"
    Stream = module("queue_streams", root / "cuda/streams.py").Stream
    Cancelled = module("tensorfold.server.cancellation", root / "server/cancellation.py").RequestCancelled
    ns = dict(queue=queue, threading=threading, itertools=itertools, Any=Any, Callable=Callable,
              Stream=Stream, time=time, _unwatch=lambda _: None)
    source = (root / "families/glm5_next/cuda/multi.py").read_text(encoding="utf-8")
    classes(source, {"NoRoom"}, ns)
    classes((root / "cuda/scheduler.py").read_text(encoding="utf-8"), {"Waiting", "Scheduler"}, ns)
    classes(source, {"GlmScheduler"}, ns)

    def scheduler(holders=0, capacity=4):
        obj = object.__new__(ns["GlmScheduler"])
        obj.decoder, obj.max_streams = Decoder(), capacity
        obj.waiting, obj.held, obj.boxes, obj.yields, obj.gather_s = ns["Waiting"](), None, {}, 0, 0
        # patch 0088's spill tier (off): its scheduler state
        obj.flushing, obj.flush_lock, obj.loading, obj.ready, obj.staged = None, threading.Lock(), [], [], set()
        obj.max_in_system = None              # patch 0082's admission cap: off
        for _ in range(holders):
            stream = Stream([1], 100)
            obj.decoder.streams[id(stream)] = stream
        return obj

    def ask(obj, poll, background=False, emit_hook=None):
        reply = {}
        def emit(tokens):
            return False if emit_hook is None else emit_hook(tokens)
        emit.cancelled = poll
        def run():
            try:
                reply["stats"] = obj.submit([1], 100, None, False, emit, background=background)
            except Exception as exc:
                reply["error"] = exc
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        wait(lambda: obj.waiting.qsize() == 1)
        return thread, reply

    def cancelled_reply(thread, reply, error=Cancelled):
        thread.join(2)
        assert not thread.is_alive() and isinstance(reply.get("error"), error)

    checks = []
    obj, event = scheduler(4), threading.Event()
    thread, reply = ask(obj, event.is_set)
    holders = tuple(obj.decoder.streams)
    event.set()
    if args.expect_stock:
        time.sleep(0.12)
        obj._iteration()
        assert thread.is_alive() and obj.waiting.qsize() == 1 and not reply
        _, box = obj.waiting.get_nowait()  # test cleanup, not stock cancellation behavior
        box.put(("error", Cancelled("test cleanup")))
        cancelled_reply(thread, reply)
        print(json.dumps({"stock_gap_reproduced": True, "device_calls": 0}))
        return
    wait(lambda: obj.waiting.queue[0][2][0].emit.cancel_requested())
    obj._iteration()
    cancelled_reply(thread, reply)
    assert obj.waiting.empty() and tuple(obj.decoder.streams) == holders
    assert not obj.decoder.admitted and not obj.decoder.fits_calls
    checks.append("full lanes: caller acknowledged without admission or touching holders")

    obj, event = scheduler(8, 8), threading.Event()
    thread, reply = ask(obj, event.is_set)
    holders = tuple(obj.decoder.streams)
    event.set()
    wait(lambda: obj.waiting.queue[0][2][0].emit.cancel_requested())
    obj._iteration()
    cancelled_reply(thread, reply)
    assert obj.waiting.empty() and tuple(obj.decoder.streams) == holders and not obj.decoder.admitted
    checks.append("eight full lanes: same pre-admission cancellation")

    obj, event = scheduler(3), threading.Event()
    thread, reply = ask(obj, event.is_set)
    obj.decoder.room = False
    obj._admit()
    assert obj.held is not None
    event.set()
    obj._iteration()
    cancelled_reply(thread, reply)
    assert obj.held is None and not obj.decoder.admitted
    checks.append("memory-held cancellation")

    obj = scheduler()
    thread, reply = ask(obj, lambda: True)
    obj._admit(obj.waiting.get_nowait())
    cancelled_reply(thread, reply)
    assert not obj.decoder.fits_calls
    checks.append("pre-cancelled idle admission")

    obj, entries = scheduler(4), []
    for n, (background, cancel) in enumerate([(False, False), (False, True), (True, False), (False, False)]):
        stream, box = Stream([n], 100, background=background), queue.Queue()
        stream.emit = lambda _: False
        stream.emit.cancelled = lambda cancel=cancel: cancel
        stream.emit.cancel_requested = stream.emit.cancelled
        obj.waiting.put((stream, box))
        entries.append((stream, box))
    obj._cancel_waiting()
    assert isinstance(entries[1][1].get_nowait()[1], Cancelled)
    assert [obj.waiting.get_nowait()[0].prompt[0] for _ in range(3)] == [0, 3, 2]
    obj._cancel_waiting()
    assert entries[1][1].empty()
    checks.append("foreground FIFO, background priority and exactly one acknowledgement")

    obj, event = scheduler(), threading.Event()
    thread, reply = ask(obj, event.is_set)
    obj._admit()
    stream = next(iter(obj.decoder.streams.values()))
    event.set()
    wait(stream.emit.cancel_requested)
    obj._cancel_waiting()
    assert thread.is_alive() and obj.decoder.live() == 1 and not reply
    stream.take([2])
    obj._iteration()
    cancelled_reply(thread, reply)
    assert obj.decoder.live() == 0 and obj.decoder.finished == [stream]
    checks.append("active disconnect raises only after normal decoder finish")

    for method in ("_yield", "_requeue"):
        obj, event = scheduler(), threading.Event()
        thread, reply = ask(obj, event.is_set, background=True)
        obj._admit()
        stream = next(iter(obj.decoder.streams.values()))
        if method == "_yield":
            obj.max_streams = 1
            obj.waiting.put((Stream([99], 100), queue.Queue()))
        else:
            obj.decoder.streams.clear()  # decoder has already freed the lane before reporting requeue
            obj.decoder.requeue = [stream]
        getattr(obj, method)()
        again = next(e[2][0] for e in obj.waiting.queue if e[2][0].background)
        assert again.emit is stream.emit
        event.set()
        wait(again.emit.cancel_requested)
        obj._cancel_waiting()
        cancelled_reply(thread, reply)
        assert id(stream) not in obj.boxes
        checks.append(method + " preserves cancellation and reply ownership")

    def broken_poll():
        raise ValueError("callback failed")
    obj = scheduler(4)
    thread, reply = ask(obj, broken_poll)
    wait(lambda: obj.waiting.queue[0][2][0].emit.cancel_requested())
    obj._iteration()
    cancelled_reply(thread, reply, ValueError)
    assert obj.decoder.live() == 4
    checks.append("poll failure is isolated to its caller")

    class ObservedBox(queue.Queue):
        def __init__(self):
            super().__init__()
            self.gets, self.first_get, self.next_get = 0, threading.Event(), threading.Event()
        def get(self, *args, **kwargs):
            self.gets += 1
            (self.first_get if self.gets == 1 else self.next_get).set()
            return super().get(*args, **kwargs)

    # A caller's older False poll returns only after another poll has latched True.
    entered, release = threading.Event(), threading.Event()
    calls, calls_lock, box = [0], threading.Lock(), ObservedBox()
    def overlap_poll():
        with calls_lock:
            calls[0] += 1
            n = calls[0]
        if n == 1:
            entered.set()
            assert release.wait(2)
            return False
        return True
    obj = scheduler(4)
    ns["queue"] = SimpleNamespace(Queue=lambda: box, Empty=queue.Empty)
    thread, reply = ask(obj, overlap_poll)
    try:
        assert entered.wait(2)
        stream = obj.waiting.queue[0][2][0]
        assert stream.emit.cancelled()
        release.set()
        assert box.first_get.wait(2)  # the caller has consumed its older False result
        assert stream.emit.cancel_requested(), "an older False poll unlatched disconnect"
        obj._iteration()
        cancelled_reply(thread, reply)
    finally:
        release.set()
        box.put(("error", Cancelled("test cleanup")))
        thread.join(2)
        ns["queue"] = queue
    checks.append("overlapping polls cannot undo a latched disconnect")

    # Normal emit=True stop/gate cuts are not socket disconnects, including replay.
    for method in ("_yield", "_requeue"):
        permit, emitted, box = threading.Event(), threading.Event(), ObservedBox()
        def normal_stop(tokens):
            emitted.set()
            assert permit.wait(2)
            return True
        obj = scheduler()
        ns["queue"] = SimpleNamespace(Queue=lambda: box, Empty=queue.Empty)
        thread, reply = ask(obj, lambda: False, background=True, emit_hook=normal_stop)
        try:
            obj._admit()
            stream = next(iter(obj.decoder.streams.values()))
            stream.take([2])
            assert emitted.wait(2)
            if method == "_yield":
                obj.max_streams = 1
                obj.waiting.put((Stream([99], 100), queue.Queue()))
            else:
                obj.decoder.streams.clear()
                obj.decoder.requeue = [stream]
            getattr(obj, method)()
            if method == "_yield":
                assert obj.waiting.get_nowait()[0].prompt == [99]  # dummy foreground test owner
            permit.set()
            assert box.next_get.wait(2)  # emit=True has been consumed by the caller
            obj._cancel_waiting()
            assert obj.waiting.qsize() == 1 and thread.is_alive() and not reply
            again = obj.waiting.queue[0][2][0]
            assert not again.emit.cancel_requested()
            obj._admit()
            again.take([2, 3])
            obj._iteration()
            thread.join(2)
            assert not thread.is_alive() and "stats" in reply and "error" not in reply
            assert obj.decoder.finished[-1] is again
        finally:
            permit.set()
            box.put(("error", Cancelled("test cleanup")))
            thread.join(2)
            ns["queue"] = queue
        checks.append(method + " retains normal stop/gate completion, not RequestCancelled")

    obj, fail = scheduler(), threading.Event()
    def active_poll():
        if fail.is_set():
            raise ValueError("active poll failed")
        return False
    thread, reply = ask(obj, active_poll)
    obj._admit()
    stream = next(iter(obj.decoder.streams.values()))
    fail.set()
    wait(stream.emit.cancel_requested)
    obj._cancel_waiting()
    assert thread.is_alive() and obj.decoder.live() == 1
    stream.take([2])
    obj._iteration()
    cancelled_reply(thread, reply, ValueError)
    assert obj.decoder.live() == 0 and obj.decoder.finished == [stream]
    checks.append("active poll failure propagates only after decoder finish")

    # Delivery failures latch an error, not a normal stop, and never let the
    # caller return before active decoder cleanup. Preserve the original error.
    failure = ValueError("delivery failed")
    deliveries = []
    def broken_emit(tokens):
        deliveries.append(tokens)
        raise failure
    obj = scheduler()
    thread, reply = ask(obj, lambda: False, emit_hook=broken_emit)
    box = obj.waiting.queue[0][2][1]
    try:
        obj._admit()
        stream = next(iter(obj.decoder.streams.values()))
        stream.take([2])
        wait(stream.emit.cancel_requested)
        obj._cancel_waiting()
        assert thread.is_alive() and not reply and obj.decoder.live() == 1
        assert stream.emit.cancel_error() is failure and not stream.done
        stream.take([2, 3])
        obj._iteration()
        cancelled_reply(thread, reply, ValueError)
        assert reply["error"] is failure and len(deliveries) == 1
        assert obj.decoder.finished == [stream] and obj.decoder.live() == 0 and not obj.boxes
    finally:
        box.put(("error", failure))
        thread.join(2)
    checks.append("delivery failure stops once and propagates after active decoder finish")

    # A delivery callback can be in flight while the scheduler yields/requeues
    # its stream. Test failure both before and after replay, including a
    # memory-held replay; queue removal remains owned by the scheduler.
    for method in ("_yield", "_requeue"):
        for timing in ("before", "after", "held"):
            permit, emitted, box = threading.Event(), threading.Event(), ObservedBox()
            failure = ValueError(method + " delivery " + timing)
            deliveries = []
            def delayed_failure(tokens):
                deliveries.append(tokens)
                emitted.set()
                assert permit.wait(2)
                raise failure
            obj = scheduler()
            ns["queue"] = SimpleNamespace(Queue=lambda: box, Empty=queue.Empty)
            thread, reply = ask(obj, lambda: False, background=True, emit_hook=delayed_failure)
            try:
                obj._admit()
                stream = next(iter(obj.decoder.streams.values()))
                stream.take([2])
                assert emitted.wait(2)
                if timing == "before":
                    permit.set()
                    assert box.next_get.wait(2)
                    assert stream.emit.cancel_requested() and thread.is_alive() and not reply
                if method == "_yield":
                    obj.max_streams = 1
                    obj.waiting.put((Stream([99], 100), queue.Queue()))
                else:
                    obj.decoder.streams.clear()  # decoder freed the lane before reporting requeue
                    obj.decoder.requeue = [stream]
                getattr(obj, method)()
                if method == "_yield":
                    assert obj.waiting.get_nowait()[0].prompt == [99]
                again = obj.waiting.queue[0][2][0]
                assert again.emit is stream.emit and id(stream) not in obj.boxes
                if timing == "held":
                    obj.decoder.room = False
                    obj._admit()
                    assert obj.held is not None and obj.held[0] is again
                if timing != "before":
                    assert not again.emit.cancel_requested() and thread.is_alive() and not reply
                    permit.set()
                    assert box.next_get.wait(2)
                assert again.emit.cancel_requested() and again.emit.cancel_error() is failure
                assert thread.is_alive() and not reply
                obj._cancel_waiting()
                cancelled_reply(thread, reply, ValueError)
                assert reply["error"] is failure and len(deliveries) == 1
                assert obj.waiting.empty() and obj.held is None and not obj.boxes
                assert obj.decoder.admitted == [stream] and obj.decoder.live() == 0
                obj._cancel_waiting()
                assert box.empty()  # one terminal reply only
            finally:
                permit.set()
                box.put(("error", failure))
                thread.join(2)
                ns["queue"] = queue
            checks.append(method + " delivery failure " + timing + " replay keeps cleanup/error ownership")

    # The decoder may already have finished while the caller is still inside
    # delivery. Its queued terminal reply must not turn that failure into stats.
    permit, emitted, box = threading.Event(), threading.Event(), ObservedBox()
    failure = ValueError("delivery failed after finish")
    def late_failure(tokens):
        emitted.set()
        assert permit.wait(2)
        raise failure
    obj = scheduler()
    ns["queue"] = SimpleNamespace(Queue=lambda: box, Empty=queue.Empty)
    thread, reply = ask(obj, lambda: False, emit_hook=late_failure)
    try:
        obj._admit()
        stream = next(iter(obj.decoder.streams.values()))
        stream.take([2])
        assert emitted.wait(2)
        stream.done = True
        obj._iteration()
        assert obj.decoder.finished == [stream] and obj.decoder.live() == 0 and not obj.boxes
        assert thread.is_alive() and not reply
        permit.set()
        cancelled_reply(thread, reply, ValueError)
        assert reply["error"] is failure and "stats" not in reply
    finally:
        permit.set()
        box.put(("error", failure))
        thread.join(2)
        ns["queue"] = queue
    checks.append("delivery failure after decoder finish is not swallowed by terminal stats")
    print(json.dumps({"status": "PASS", "tests": len(checks), "checks": checks, "device_calls": 0,
                      "source_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
