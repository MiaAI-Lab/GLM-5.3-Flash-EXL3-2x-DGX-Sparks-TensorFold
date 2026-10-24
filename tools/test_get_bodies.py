"""CPU-only in-process HTTP test of GET and DELETE requests that carry a body (patch 0113).

  python3 -B tools/test_get_bodies.py --source-root /path/to/patched/src
Use --expect-stock on the unpatched source to reproduce the failure: do_GET and do_DELETE never read a body, so a
GET with a body (scanners send them) leaves its bytes on the keep-alive connection, and the server parses them as the
start of the next request: a stdlib "Bad request syntax" 400 for whatever request comes next on that connection. Behind
a pooling proxy (nginx upstream keepalive) that next request can be another user's.
Runs the real CUDA-server handler (tensorfold.cuda.http) on a loopback port with a stub app; no torch, GPU or models.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from http.server import ThreadingHTTPServer


def start(root):
    sys.path.insert(0, root)
    from tensorfold.cuda.http import make_handler

    class App:
        model_ids = ["m"]
        served_name = "m"

        def reply_model(self, body):
            return "m"

        def prepare(self, body, chat):
            return body

        def run(self, body, chat, emit, prepared=None, cancelled=None):
            return {"content": "ok", "reasoning": None, "calls": None, "finish": "stop",
                    "stats": {}, "prompt_tokens": 3, "completion_tokens": 1}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def read_response(sock_file):
    """One response read by its Content-Length, as a strict keep-alive client would; (status, headers, body)."""

    line = sock_file.readline()
    if not line.startswith(b"HTTP/1."):
        raise AssertionError(f"not a status line (framing lost): {line!r}")
    status = int(line.split()[1])
    headers = {}
    while (h := sock_file.readline()) not in (b"\r\n", b""):
        k, _, v = h.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, sock_file.read(int(headers.get("content-length", 0)))


def with_body(method: str, path: str, body: bytes) -> bytes:
    return f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


def chunked(method: str, path: str, body: bytes) -> bytes:
    return (f"{method} {path} HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n".encode()
            + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")


MODELS = b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n"
PROBE = b"<?php system($_GET['c']); ?>"          # what the scanners put in a GET body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    server = start(args.source_root)
    port = server.server_address[1]

    def conn():
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        return s, s.makefile("rb")

    if args.expect_stock:
        s, f = conn()
        s.sendall(with_body("GET", "/v1/models", PROBE))
        assert read_response(f)[0] == 200
        s.sendall(MODELS)                                   # the next request on the same connection
        status, _, body = read_response(f)
        assert status == 400 and b"Bad request" in body, (status, body)
        print(f"stock: the GET's body was parsed as the next request: {status} {body[:80]!r}")
        return

    msgs = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
    # 1. a GET with a body (Content-Length), then more requests on the same connection: framing intact
    s, f = conn()
    s.sendall(with_body("GET", "/v1/models", PROBE))
    status, _, body = read_response(f)
    assert status == 200 and json.loads(body)["data"][0]["id"] == "m", (status, body)
    s.sendall(MODELS)
    assert read_response(f)[0] == 200, "a GET's body leaked into the next request"
    s.sendall(with_body("POST", "/v1/chat/completions", msgs))
    status, _, body = read_response(f)
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "ok", (status, body)
    # 2. a chunked body on a GET (an unknown route: the 404 path), then the next request
    s.sendall(chunked("GET", "/nope", PROBE))
    assert read_response(f)[0] == 404
    s.sendall(MODELS)
    assert read_response(f)[0] == 200, "a chunked GET body leaked into the next request"
    # 3. a DELETE with a body, then the next request
    s.sendall(with_body("DELETE", "/v1/responses/resp_none", PROBE))
    status, _, body = read_response(f)
    assert status in (200, 404), (status, body)
    s.sendall(MODELS)
    assert read_response(f)[0] == 200, "a DELETE's body leaked into the next request"
    # 4. GETs without a body still answer at once (nothing to read, nothing waited for)
    for _ in range(3):
        s.sendall(MODELS)
        assert read_response(f)[0] == 200
    s.close()
    # 4b. a body of 100 KiB, over the 64 KiB a GET may carry: answered, connection closed, nothing buffered
    s, f = conn()
    s.sendall(f"GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: {100 * 1024}\r\n\r\n".encode())
    status, headers, _ = read_response(f)
    assert status == 200 and headers.get("connection") == "close", (status, headers)
    s.close()
    # 5. a GET body past the limit: refused framing closes the connection instead of reading on
    s, f = conn()
    s.sendall(f"GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: {97 * 1024 ** 2}\r\n\r\n".encode())
    s.settimeout(5)
    status, headers, _ = read_response(f)
    assert status == 200 and headers.get("connection") == "close", (status, headers)
    s.close()
    print("GET and DELETE bodies: read before the response, framing intact on keep-alive connections")


if __name__ == "__main__":
    main()
