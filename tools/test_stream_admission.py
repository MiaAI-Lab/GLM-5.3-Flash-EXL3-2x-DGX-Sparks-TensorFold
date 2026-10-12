"""CPU-only: a full server answers a STREAMED request 429 + Retry-After, not 200 and an error event (issue #50).

  python3 -B tools/test_stream_admission.py --source-root /path/to/patched/src
Use --expect-stock on a source without patch 0095 (patches 0081-0083 only) to see the 200 with the error inside the stream.
Runs the real CUDA-server handler on a loopback port with a stub app whose scheduler refuses like ``Scheduler._check_admission``
when its cap is reached; ``App.admit`` itself is taken from the source file and run on a fake engine. No torch or GPU.
"""
from __future__ import annotations

import argparse
import ast
import json
import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace


def admit_method(root):
    source = (Path(root) / "tensorfold/cuda/server.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "App":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "admit":
                    ns = {"Any": object, "PreparedRequest": object, "is_title_request": lambda messages, tools: False}
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(
                        module="__future__", names=[ast.alias(name="annotations")], level=0), item], type_ignores=[])),
                        "<admit>", "exec"), ns)
                    return ns["admit"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from tensorfold.cuda.http import make_handler
    from tensorfold.server.errors import CapacityError, RequestError

    full = [True]

    class Scheduler:
        def _check_admission(self, background):
            if full[0] and not background:
                raise CapacityError("all serving lanes are busy; retry shortly")

    admit = admit_method(args.source_root)

    class App:
        model_ids, served_name = ["m"], "m"
        engine = SimpleNamespace(scheduler=Scheduler())

        def reply_model(self, body):
            return "m"

        def prepare(self, body, chat):
            return SimpleNamespace(tools=None)

        def run(self, body, chat, emit, prepared=None, cancelled=None):
            Scheduler()._check_admission(body.get("priority") == "background")       # Scheduler.submit's check
            emit({"content": "hi"})
            return {"final": {}, "calls": None, "call_deltas": [], "finish": "stop", "content": "hi", "reasoning": None,
                    "stats": {}, "prompt_tokens": 1, "completion_tokens": 1}

    if admit is not None:
        App.admit = admit
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App()))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(**extra):
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}], **extra}).encode()
        s = socket.create_connection(server.server_address, timeout=5)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        data = b""
        while chunk := s.recv(65536):
            data += chunk
            if b"[DONE]" in data or b"\r\n\r\n" in data and b"Content-Length" in data and data.endswith(b"}"):
                break
        s.close()
        return data

    for stream in (False, True):
        reply = post(stream=stream)
        status = int(reply.split()[1])
        if args.expect_stock and stream:
            assert status == 200 and b"lanes are busy" in reply, reply[:200]
            print("stock: a streamed request to a full server is a 200 with the error inside the stream")
            return
        assert status == 429 and b"Retry-After: 5" in reply, (stream, reply[:300])
    assert int(post(stream=True, priority="background").split()[1]) == 200, "background requests queue by design"
    full[0] = False
    assert int(post(stream=True).split()[1]) == 200
    print("streamed and plain requests to a full server: 429 + Retry-After: 5; with room: 200")


if __name__ == "__main__":
    main()
