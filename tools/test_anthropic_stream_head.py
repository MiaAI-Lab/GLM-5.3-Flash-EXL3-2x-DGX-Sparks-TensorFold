"""CPU-only in-process HTTP test of the Anthropic stream head's usage (patch 0112).

  python3 -B tools/test_anthropic_stream_head.py --source-root /path/to/patched/src [--expect-stock]
Use --expect-stock on a tree without the patch to reproduce the report: a streamed /v1/messages reply opens with
message_start.usage all zeros (the prompt's token count only arrives in the final message_delta), so a client that
reads prompt counts from message_start (Claude Code's context-window indicator) shows ~0% used. With the patch the
head names the prepared prompt's whole size and the final delta keeps the engine's input/cache_read split, the two
totals equal.

Runs the real CUDA-server handler (tensorfold.cuda.http) on a loopback port with a stub app: prepare returns a
24-token prompt, the streamed reply's usage reports 24 prompt tokens of which 3 were cached, so the head must say
24 and the final delta 21 + 3. No torch, GPU or models.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

PROMPT_TOKENS = 24
CACHED_TOKENS = 3
COMPLETION_TOKENS = 2


def start(root: str):
    sys.path.insert(0, root)
    from tensorfold.cuda.http import make_handler

    class App:
        model_ids = ["m"]
        served_name = "m"

        def reply_model(self, body):
            return "m"

        def prepare(self, body, chat):
            # the shape the CUDA app's prepare returns: prompt, the tokens the reply's usage counts
            return SimpleNamespace(prompt=list(range(PROMPT_TOKENS)), max_tokens=8, tools=[],
                                   thinking=False, sampling=None, ignore_eos=False, stop=(),
                                   vision=None, grammar=None, think_budget=0)

        def admit(self, body, chat, prepared):
            pass

        def run(self, body, chat, emit, prepared=None, cancelled=None):
            emit({"content": "hi"})
            return {"final": {}, "calls": [], "call_deltas": [], "calls_streamed": 0, "finish": "stop",
                    "content": "hi", "reasoning": None, "prompt_tokens": len(prepared.prompt),
                    "completion_tokens": COMPLETION_TOKENS, "cached_tokens": CACHED_TOKENS, "stats": {}}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def request(path: str, body: bytes) -> bytes:
    return (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode() + body


def exchange(port: int, path: str, body: bytes) -> tuple[int, bytes]:
    """One request on its own connection: the status and the bytes after the headers."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    f = s.makefile("rb")
    s.sendall(request(path, body))
    status = int(f.readline().split()[1])
    headers = {}
    while (line := f.readline()) not in (b"\r\n", b""):
        name, _, value = line.decode().partition(":")
        headers[name.strip().lower()] = value.strip()
    rest = f.read(int(headers["content-length"])) if "content-length" in headers else f.read()
    s.close()
    return status, rest


def events(body: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: ")
            and line[6:] != b"[DONE]"]


def head_usage(body: bytes) -> dict:
    starts = [e for e in events(body) if e.get("type") == "message_start"]
    assert len(starts) == 1, f"expected one message_start, got {len(starts)}"
    return starts[0]["message"]["usage"]


def delta_usage(body: bytes) -> dict:
    deltas = [e for e in events(body) if e.get("type") == "message_delta"]
    assert len(deltas) == 1, f"expected one message_delta, got {len(deltas)}"
    return deltas[0]["usage"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    server = start(args.source_root)
    port = server.server_address[1]
    stream = json.dumps({"model": "m", "max_tokens": 8, "stream": True,
                         "messages": [{"role": "user", "content": "hi"}]}).encode()
    _, body = exchange(port, "/v1/messages", stream)
    start_usage, delta = head_usage(body), delta_usage(body)

    if args.expect_stock:
        assert start_usage["input_tokens"] == 0 and start_usage["cache_read_input_tokens"] == 0, start_usage
        assert delta["input_tokens"] == PROMPT_TOKENS - CACHED_TOKENS, delta
        assert delta["cache_read_input_tokens"] == CACHED_TOKENS, delta
        print(f"stock: message_start.usage {start_usage}; the count only arrives in the final "
              f"message_delta {delta} ({delta['input_tokens']}+{delta['cache_read_input_tokens']}={PROMPT_TOKENS})")
        return

    assert start_usage == {"input_tokens": PROMPT_TOKENS, "output_tokens": 0,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, start_usage
    assert delta == {"input_tokens": PROMPT_TOKENS - CACHED_TOKENS, "output_tokens": COMPLETION_TOKENS,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": CACHED_TOKENS,
                     "output_tokens_details": {"thinking_tokens": 0}}, delta
    assert start_usage["input_tokens"] == delta["input_tokens"] + delta["cache_read_input_tokens"], \
        f"head {start_usage} disagrees with the delta's total {delta}"
    # a whole (non-streamed) reply keeps its usage, and a plain chat stream is untouched by the wire's new attribute
    _, body = exchange(port, "/v1/messages", json.dumps({"model": "m", "max_tokens": 8,
                                                         "messages": [{"role": "user", "content": "hi"}]}).encode())
    whole = json.loads(body)
    assert whole["usage"]["input_tokens"] == PROMPT_TOKENS - CACHED_TOKENS, whole["usage"]
    _, body = exchange(port, "/v1/chat/completions", stream)
    first = [e for e in events(body) if e.get("choices")][0]
    assert first["choices"][0]["delta"] == {"role": "assistant"}, first
    last = [e for e in events(body) if e.get("choices")][-1]
    assert last.get("usage", {}).get("prompt_tokens") == PROMPT_TOKENS, last
    print(f"patched: message_start.usage {start_usage}; final message_delta.usage {delta} "
          f"({delta['input_tokens']}+{delta['cache_read_input_tokens']}={PROMPT_TOKENS})")


if __name__ == "__main__":
    main()
