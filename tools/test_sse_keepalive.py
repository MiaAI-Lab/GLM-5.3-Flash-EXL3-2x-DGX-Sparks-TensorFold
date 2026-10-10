"""CPU-only: a streamed reply that is silent (waiting for a lane, filling a long prompt) writes SSE keep-alive comments,
on every streaming route, and the events a client parses stay the same bytes (patch 0112, issue #106).

  python3 -B tools/test_sse_keepalive.py --source-root /path/to/patched/src [--stock-root /path/to/unpatched/src]
Use --expect-stock on a source without patch 0112 to see every stream stay silent while its request waits.
Runs the real CUDA-server handler (and through it the Messages and Responses routes) on a loopback port with a stub app
whose run() is silent for a few seconds, once with TENSORFOLD_SSE_KEEPALIVE_S=1 and once with 0, each in its own process,
with ids and clocks fixed so the runs can be compared byte for byte. Checks, with patch 0112:
- every streamed route (chat, completions, Messages, Responses) gets ': keepalive' about once a second while silent
  and none after its last event, error events included; a reply that keeps sending tokens gets none;
- each stream with its comment lines taken out is byte-identical to the stream with 0 (headers included) and parses to
  the same events by the SSE format's rules; with --stock-root, the replies with 0 are byte-identical to the unpatched
  source's;
- non-streamed replies are byte-identical either way; with 0 no keep-alive thread ever starts;
- no keep-alive thread outlives its request: a client that leaves ends it with its cancelled request, or by the failed
  write when the request goes on; a bad TENSORFOLD_SSE_KEEPALIVE_S stops the server at import.
No torch or GPU.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time

SILENCE = 3.2               # a streamed request's stub run: silent this long, then one token
SHORT = 1.5                 # a non-streamed request's: longer than a keep-alive interval
DEAF = 6.0                  # a request that goes on after its client left (no cancellation check)
COMMENT = b": keepalive\n\n"
ROUTES = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")
LAST = {"/v1/chat/completions": b"data: [DONE]\n\n", "/v1/completions": b"data: [DONE]\n\n",
        "/v1/messages": b"event: message_stop\n", "/v1/responses": b"event: response.completed\n"}
BAD = ("abc", "-1", "nan", "inf", "")


def body_for(route, stream, text):
    if route == "/v1/completions":
        return {"model": "stub", "stream": stream, "max_tokens": 4, "prompt": text}
    if route == "/v1/responses":
        return {"model": "stub", "stream": stream, "max_output_tokens": 16, "input": text, "store": False}
    return {"model": "stub", "stream": stream, "max_tokens": 4, "messages": [{"role": "user", "content": text}]}


def exchange(port, route, body, leave_on=None):
    """POST ``body``; every received piece with its arrival time (s), until EOF (or until ``leave_on`` arrives: then
    the client closes the connection, as one that gives up does)."""

    data = json.dumps(body).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    t0 = time.monotonic()
    sock.sendall(f"POST {route} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    pieces, raw = [], b""
    while True:
        piece = sock.recv(65536)
        if not piece:
            break
        pieces.append((time.monotonic() - t0, piece))
        raw += piece
        if leave_on is not None and leave_on in raw:
            break
    sock.close()
    return pieces


# -- the child: one setting, every request, raw bytes out -----------------------------------------------------------


def child(root, seconds):
    os.environ["TENSORFOLD_SSE_KEEPALIVE_S"] = seconds
    import uuid
    counter = iter(range(1, 1 << 30))
    uuid.uuid4 = lambda: uuid.UUID(int=next(counter))           # the same ids in every run
    time.time = lambda: 1767225600.0                               # the same created / Date in every run
    sys.path.insert(0, root)
    try:
        from tensorfold.cuda.http import make_handler
    except ValueError as exc:                                      # a bad setting: reported to the parent
        print(json.dumps({"import_error": str(exc)}))
        return
    from tensorfold.server.cancellation import RequestCancelled
    from tensorfold.server.errors import RequestError
    from tensorfold.server.http import Server

    class Stub:
        model_ids, served_name = ["stub"], "stub"

        def prepare(self, body, chat):
            return None

        def admit(self, body, chat, prepared):
            return None

        def reply_model(self, body):
            return "stub"

        def run(self, body, chat, emit, prepared=None, cancelled=None):
            text = json.dumps(body)
            deaf = "deaf" in text
            end = time.monotonic() + (DEAF if deaf else SILENCE if body.get("stream") else SHORT)
            sent = 0
            while time.monotonic() < end:                  # waiting for a lane / filling a prompt
                if not deaf and cancelled is not None and cancelled():
                    raise RequestCancelled()
                threading.Event().wait(0.05)
                if "busy" in text and time.monotonic() > end - SILENCE + 0.4 * (sent + 1):
                    sent += 1
                    emit({"content": f"t{sent} "})         # a reply that keeps sending: a token every 0.4 s
            if "please fail" in text:
                raise RequestError("the stub refused this request")
            emit({"content": "OK"})
            return {"final": None, "call_deltas": [], "calls": [], "finish": "stop", "stats": {}, "content": "OK",
                    "reasoning": "", "prompt_tokens": 5, "completion_tokens": 1}

    server = Server(("127.0.0.1", 0), make_handler(Stub()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def beats():
        return sum(t.name == "sse-keepalive" for t in threading.enumerate())

    out = {"cases": {}, "threads_after": {}, "threads_most": 0, "leave": {}}
    sampling = threading.Event()

    def sample():                                     # the most keep-alive threads alive at once, over every request
        while not sampling.wait(0.02):
            out["threads_most"] = max(out["threads_most"], beats())

    threading.Thread(target=sample, daemon=True).start()
    cases = [(route, stream, "hi") for route in ROUTES for stream in (True, False)]
    cases += [("/v1/chat/completions", True, "please fail"), ("/v1/messages", True, "please fail"),
              ("/v1/chat/completions", True, "busy"), ("/v1/messages", True, "busy")]
    for route, stream, text in cases:
        key = f"{route} stream={stream} {text}"
        pieces = exchange(port, route, body_for(route, stream, text))
        out["cases"][key] = [(t, base64.b64encode(p).decode()) for t, p in pieces]
        out["threads_after"][key] = beats()
    # a client that leaves at its first keep-alive: its request is cancelled ("hi"), or goes on ("deaf") and the
    # comments end at the first write that fails
    for text in ("hi", "deaf"):
        exchange(port, "/v1/chat/completions", body_for("/v1/chat/completions", True, text), leave_on=COMMENT)
        left = time.monotonic()
        while time.monotonic() - left < DEAF + 1 and beats():
            time.sleep(0.05)
        out["leave"][text] = {"threads": beats(), "seconds": time.monotonic() - left}
        time.sleep(DEAF if text == "deaf" else 0)       # the deaf request's run ends before the server does
    sampling.set()
    server.shutdown()
    print(json.dumps(out))


# -- the parent: the settings compared ------------------------------------------------------------------------------


def sse_events(body):
    """The events an SSE client dispatches (the format's parsing rules: comment lines and unknown fields skipped)."""

    events, kind, data = [], "message", []
    for line in body.replace(b"\r\n", b"\n").split(b"\n"):
        if not line:
            if data:
                events.append((kind, b"\n".join(data)))
            kind, data = "message", []
            continue
        if line.startswith(b":"):
            continue
        name, _, value = line.partition(b":")
        value = value[1:] if value.startswith(b" ") else value
        if name == b"event":
            kind = value.decode()
        elif name == b"data":
            data.append(value)
    return events


def start_child(root, seconds):
    return subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "--source-root", root, "--child", seconds],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish_child(proc, what):
    out, err = proc.communicate(timeout=300)
    if proc.returncode:
        sys.exit(f"the child for {what} failed:\n{err}")
    return json.loads(out.strip().splitlines()[-1])


def split(pieces):
    """(head, body, [(arrival time, the body's length so far)])"""

    raw, marks = b"", []
    for t, p in pieces:
        raw += base64.b64decode(p)
        marks.append((t, len(raw)))
    cut = raw.index(b"\r\n\r\n") + 4
    return raw[:cut], raw[cut:], [(t, end - cut) for t, end in marks if end > cut]


def silence(pieces):
    """The longest the client heard nothing, from its request to the end of the reply (s)."""

    times = [0.0] + [t for t, _ in pieces]
    return max(b - a for a, b in zip(times, times[1:]))


def comment_times(body, marks):
    times, at = [], 0
    while (at := body.find(COMMENT, at)) >= 0:
        times.append(next(t for t, end in marks if end >= at + len(COMMENT)))
        at += len(COMMENT)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--stock-root", help="an unpatched source: the replies with 0 must equal its replies byte for byte")
    ap.add_argument("--expect-stock", action="store_true")
    ap.add_argument("--child")
    args = ap.parse_args()
    if args.child is not None:
        return child(args.source_root, args.child)

    procs = {"on": start_child(args.source_root, "1"), "off": start_child(args.source_root, "0")}
    if args.stock_root:
        procs["stock"] = start_child(args.stock_root, "1")
    runs = {name: finish_child(proc, name) for name, proc in procs.items()}
    on, off = runs["on"], runs["off"]
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)

    if args.expect_stock:
        for key, pieces in on["cases"].items():
            head, body, marks = split(pieces)
            check(b"\n:" not in b"\n" + body, f"{key}: a comment line in the stock stream")
            if "stream=True" in key and "busy" not in key:
                quiet = silence(pieces)
                check(quiet >= SILENCE - 0.5, f"{key}: no {SILENCE} s silence ({quiet:.1f} s)")
                print(f"stock {key}: nothing received for {quiet:.1f} s while its request waits")
        if failures:
            sys.exit("FAILED (expected the stock behaviour):\n  " + "\n  ".join(failures))
        print("stock: every streamed route is silent while its request waits, as expected")
        return

    for key, pieces in on["cases"].items():
        head, body, marks = split(pieces)
        head0, body0, _ = split(off["cases"][key])
        check(head == head0, f"{key}: the headers differ between keep-alives on and off")
        check(body.replace(COMMENT, b"") == body0, f"{key}: the stream without its comments differs from the one with 0")
        check(COMMENT not in body0, f"{key}: a comment with TENSORFOLD_SSE_KEEPALIVE_S=0")
        if args.stock_root:
            check(runs["stock"]["cases"][key] and split(runs["stock"]["cases"][key])[:2] == (head0, body0),
                  f"{key}: the reply with 0 differs from the unpatched source's")
        check(on["threads_after"][key] == 0 and off["threads_after"][key] == 0,
              f"{key}: a keep-alive thread outlived its request ({on['threads_after'][key]})")
        times = comment_times(body, marks)
        if "stream=False" in key:
            check(body == body0 and not times, f"{key}: the non-streamed reply changed")
            print(f"{key}: byte-identical, no comment")
            continue
        check(sse_events(body) == sse_events(body0) and sse_events(body0), f"{key}: the parsed events differ")
        if "busy" in key:
            check(not times, f"{key}: keep-alives while tokens were flowing: {times}")
            print(f"{key}: tokens every 0.4 s, no keep-alive; the same {len(sse_events(body0))} events")
            continue
        check(len(times) >= int(SILENCE) - 1, f"{key}: {len(times)} keep-alives in {SILENCE} s of silence: {times}")
        check(all(0.9 <= b - a <= 1.6 for a, b in zip(times, times[1:])), f"{key}: keep-alives not ~1 s apart: {times}")
        quiet = silence(pieces)
        check(quiet < 1.6, f"{key}: still a {quiet:.1f} s silence")
        last = LAST[key.split(" ")[0]] if "please fail" not in key else b"data: "
        check(body.rfind(COMMENT) < body.rfind(last), f"{key}: a keep-alive after the stream's last event")
        print(f"{key}: {len(times)} keep-alives at {[round(t, 1) for t in times]} s, the longest silence "
              f"{quiet:.1f} s; without them the same {len(body0)} bytes and {len(sse_events(body0))} events as with 0")
    if args.stock_root:
        print("with 0: every reply byte-identical to the unpatched source's")
    check(off["threads_most"] == 0, f"TENSORFOLD_SSE_KEEPALIVE_S=0 started {off['threads_most']} keep-alive threads")
    check(on["threads_most"] == 1, f"{on['threads_most']} keep-alive threads at once for one request at a time")
    print(f"keep-alive threads alive at once: {on['threads_most']} with 1, {off['threads_most']} with 0")
    for text, limit in (("hi", 1.0), ("deaf", 3.5)):
        leave = on["leave"][text]
        check(leave["threads"] == 0 and leave["seconds"] < limit,
              f"client left ({text}): {leave['threads']} keep-alive threads after {leave['seconds']:.1f} s")
        print(f"client left ({'its request cancelled' if text == 'hi' else 'its request going on'}): "
              f"the keep-alive thread ended {leave['seconds']:.2f} s later")

    refused = {}
    for bad in BAD:
        refused[bad] = "TENSORFOLD_SSE_KEEPALIVE_S" in finish_child(start_child(args.source_root, bad), bad).get(
            "import_error", "")
        check(refused[bad], f"TENSORFOLD_SSE_KEEPALIVE_S={bad!r} was accepted")
    print(f"bad settings refused at import: {refused}")
    if failures:
        sys.exit("FAILED:\n  " + "\n  ".join(failures))
    print("OK")


if __name__ == "__main__":
    main()
