"""CPU-only in-process HTTP test of chunked request bodies (issues #67 and #74).

  python3 -B tools/test_chunked_bodies.py --source-root /path/to/patched/src
Use --expect-stock on the unpatched source to reproduce the reports: a Transfer-Encoding: chunked body is read as empty
("messages must be a list", 400), and its unread chunks, taken for the next request, draw the stdlib's HTML
"Error response" after the JSON, with no status line (HTTP/0.9 mode), past the declared Content-Length.
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
    from tensorfold.server.errors import RequestError

    class App:
        model_ids = ["m"]
        served_name = "m"

        def reply_model(self, body):
            return "m"

        def prepare(self, body, chat):
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            return body

        def run(self, body, chat, emit, prepared=None, cancelled=None):
            text = body["messages"][-1]["content"]
            return {"content": "echo:" + str(text), "reasoning": None, "calls": None, "finish": "stop",
                    "stats": {}, "prompt_tokens": 3, "completion_tokens": 2}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def chunked(payload: bytes, size=7, ext="", trailer=b"") -> bytes:
    out = b""
    for i in range(0, len(payload), size):
        part = payload[i:i + size]
        out += f"{len(part):x}{ext}\r\n".encode() + part + b"\r\n"
    return out + b"0\r\n" + trailer + b"\r\n"


def request(path, body: bytes, *, chunks=None, extra=b"") -> bytes:
    head = f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n".encode() + extra
    if chunks is not None:
        return head + b"Transfer-Encoding: chunked\r\n\r\n" + chunks
    return head + f"Content-Length: {len(body)}\r\n\r\n".encode() + body


def read_response(sock_file):
    """One response read by its Content-Length, as a strict keep-alive client would; (status, headers, body)."""

    line = sock_file.readline()
    if not line.startswith(b"HTTP/1.") :
        raise AssertionError(f"not a status line (framing lost): {line!r}")
    status = int(line.split()[1])
    headers = {}
    while (h := sock_file.readline()) not in (b"\r\n", b""):
        k, _, v = h.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers, sock_file.read(int(headers.get("content-length", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    server = start(args.source_root)
    port = server.server_address[1]
    msgs = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()

    def conn():
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        return s, s.makefile("rb")

    if args.expect_stock:
        s, f = conn()
        s.sendall(request("/v1/chat/completions", msgs, chunks=chunked(msgs)))
        status, _, body = read_response(f)
        extra = f.read1(65536) if hasattr(f, "read1") else b""
        s.settimeout(1)
        try:
            extra = extra or s.recv(65536)
        except OSError:
            pass
        assert status == 400 and b"messages must be a list" in body, (status, body)
        print(f"stock: 400 {body.decode()}; bytes after the declared Content-Length: {extra[:60]!r}")
        return

    # 1. a chunked chat completion, then more requests on the same connection: exact framing throughout
    s, f = conn()
    s.sendall(request("/v1/chat/completions", msgs, chunks=chunked(msgs)))
    status, _, body = read_response(f)
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "echo:hi", (status, body)
    s.sendall(request("/v1/chat/completions", msgs))                       # Content-Length, same socket
    assert read_response(f)[0] == 200
    s.sendall(request("/v1/chat/completions", b"{}", chunks=chunked(b"{}", ext=";name=v", trailer=b"X-T: 1\r\n")))
    status, _, body = read_response(f)                                     # no messages: a clean 400 and a usable socket
    assert status == 400 and b"messages must be a list" in body, (status, body)
    s.sendall(request("/v1/chat/completions", msgs, chunks=chunked(msgs, size=1)))      # one byte a chunk
    assert read_response(f)[0] == 200
    s.sendall(b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n")
    assert read_response(f)[0] == 200, "framing lost after the chunked requests"
    # 2. the chunks arriving in separate TCP segments
    raw = request("/v1/chat/completions", msgs, chunks=chunked(msgs, size=5))
    for i in range(0, len(raw), 9):
        s.sendall(raw[i:i + 9])
    assert read_response(f)[0] == 200
    # 3. a refused path with a chunked body: 404, the body consumed, the socket still framed
    s.sendall(request("/nope", msgs, chunks=chunked(msgs)))
    assert read_response(f)[0] == 404
    s.sendall(request("/v1/chat/completions", msgs))
    assert read_response(f)[0] == 200, "a refused chunked body leaked into the next request"
    s.close()
    # 3b. /v1/responses, chunked
    s, f = conn()
    rbody = json.dumps({"model": "m", "input": "hi", "store": False}).encode()
    s.sendall(request("/v1/responses", rbody, chunks=chunked(rbody, size=6)))
    status, _, body = read_response(f)
    assert status == 200 and json.loads(body)["status"] == "completed", (status, body)
    s.sendall(request("/v1/responses", rbody))
    assert read_response(f)[0] == 200, "framing lost after a chunked /v1/responses"
    s.close()
    # 4. malformed chunking: a 400 and the connection closed
    for bad in (b"zz\r\nabc\r\n0\r\n\r\n", b"3\r\nabcde\r\n0\r\n\r\n", b"3\r\nabc"):
        s, f = conn()
        s.sendall(request("/v1/chat/completions", b"", chunks=bad))
        s.shutdown(socket.SHUT_WR) if bad.endswith(b"abc") else None
        status, headers, body = read_response(f)
        assert status == 400 and any(w in body.decode() for w in ("chunk", "incomplete")), (bad, status, body)
        assert headers.get("connection") == "close", headers
        s.close()
    # 5. a body past the limit is refused as soon as its chunks say so, and the connection closed
    s, f = conn()
    s.sendall(request("/v1/chat/completions", b"", chunks=f"{97 * 1024 ** 2:x}\r\n".encode()))
    status, headers, body = read_response(f)
    assert status == 400 and "96 MiB" in body.decode() and headers.get("connection") == "close", (status, body, headers)
    s.close()
    # 6. Transfer-Encoding with another coding is refused
    s, f = conn()
    s.sendall(request("/v1/chat/completions", b"", chunks=b"", extra=b"X: 1\r\n").replace(b"chunked", b"gzip, chunked"))
    status, headers, body = read_response(f)
    assert status == 400 and "Transfer-Encoding" in body.decode(), (status, body)
    s.close()
    print("chunked request bodies: framing intact on keep-alive connections, malformed and oversize bodies refused and closed")


if __name__ == "__main__":
    main()
