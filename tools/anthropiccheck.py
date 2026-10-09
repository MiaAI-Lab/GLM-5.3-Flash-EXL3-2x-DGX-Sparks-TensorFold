#!/usr/bin/env python3
"""The Anthropic Messages surface (patch 0084) against the live server: message shape, count_tokens, streamed
tool_use, a tool_result turn, stop_sequences, the 400 error shape, and a chunked body.

Usage: tools/anthropiccheck.py      (API_URL / PORT as in client.py). Exit code 1 on failure.
"""
import http.client
import json
import os
import sys
import urllib.error
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import API_URL, open_url  # noqa: E402

MESSAGES = API_URL + "/v1/messages"
COUNT = MESSAGES + "/count_tokens"
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
HEADERS = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
TOOLS = [{"name": "add_tags", "description": "Attach tags to a document.",
          "input_schema": {"type": "object", "required": ["doc_id", "tags"],
                           "properties": {"doc_id": {"type": "integer"},
                                          "tags": {"type": "array", "items": {"type": "string"}}}}}]

fails = []


def post(url: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(url, json.dumps(body).encode(), HEADERS)
    return json.load(open_url(req, timeout))


def blocks(message: dict, kind: str) -> list:
    return [b for b in message.get("content", []) if b.get("type") == kind]


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{name}: {'ok' if ok else 'FAIL'} ({detail})", flush=True)
    if not ok:
        fails.append(name)


def main() -> None:
    text_body = {"model": MODEL, "max_tokens": 64, "temperature": 0, "system": "You are terse.",
                 "messages": [{"role": "user", "content": "Say exactly: hello from tensorfold"}]}
    msg = post(MESSAGES, text_body)
    texts = " ".join(b.get("text", "") for b in blocks(msg, "text"))
    check("message", msg.get("type") == "message" and "hello from tensorfold" in texts
          and msg.get("usage", {}).get("input_tokens", 0) > 0 and msg.get("usage", {}).get("output_tokens", 0) > 0,
          f"stop_reason {msg.get('stop_reason')}, usage {msg.get('usage')}")

    count = post(COUNT, text_body)
    check("count_tokens", abs(count.get("input_tokens", -1) - msg["usage"]["input_tokens"]) <= 8,
          f"count {count.get('input_tokens')} vs usage {msg['usage']['input_tokens']}")

    stream_body = {"model": MODEL, "max_tokens": 512, "temperature": 0, "tools": TOOLS, "stream": True,
                   "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' "
                                                              "using the tool."}]}
    req = urllib.request.Request(MESSAGES, json.dumps(stream_body).encode(), HEADERS)
    conn = open_url(req, 600)
    events, call_args, tool_name, stop_reason = [], {}, None, None
    for raw in conn:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:])
        events.append(event["type"])
        if event["type"] == "content_block_start" and event["content_block"].get("type") == "tool_use":
            tool_name = event["content_block"]["name"]
            call_args[event["index"]] = ""
        if event["type"] == "content_block_delta" and event["delta"].get("type") == "input_json_delta":
            call_args[event["index"]] += event["delta"]["partial_json"]
        if event["type"] == "message_delta":
            stop_reason = event["delta"].get("stop_reason")
    call = json.loads(next(iter(call_args.values()), "{}"))
    check("streaming tool_use", "message_start" in events and tool_name == "add_tags"
          and isinstance(call.get("tags"), list) and len(call.get("tags", [])) >= 3
          and stop_reason == "tool_use" and "message_stop" in events,
          f"{len(events)} events, {tool_name}({json.dumps(call)}), stop {stop_reason}")

    loop_body = {"model": MODEL, "max_tokens": 128, "temperature": 0,
                 "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' "
                                                           "using the tool."},
                              {"role": "assistant", "content": [
                                  {"type": "tool_use", "id": "call_1", "name": "add_tags",
                                   "input": {"doc_id": 42, "tags": ["urgent", "finance", "q3"]}}]},
                              {"role": "user", "content": [
                                  {"type": "tool_result", "tool_use_id": "call_1",
                                   "content": "saved tags: urgent, finance, q3"}]}]}
    msg2 = post(MESSAGES, loop_body)
    texts2 = " ".join(b.get("text", "") for b in blocks(msg2, "text"))
    check("tool_result turn", msg2.get("type") == "message" and bool(texts2.strip()),
          f"stop {msg2.get('stop_reason')}, reply {texts2.strip()[:60]!r}")

    stop_body = {"model": MODEL, "max_tokens": 64, "temperature": 0, "stop_sequences": ["END"],
                 "messages": [{"role": "user",
                               "content": "Reply with exactly this and nothing else: alpha beta END gamma"}]}
    msg3 = post(MESSAGES, stop_body)
    texts3 = " ".join(b.get("text", "") for b in blocks(msg3, "text"))
    check("stop_sequences", msg3.get("stop_reason") == "stop_sequence" and msg3.get("stop_sequence") == "END"
          and "gamma" not in texts3, f"stop {msg3.get('stop_reason')} {msg3.get('stop_sequence')!r}, "
                                     f"text {texts3.strip()[:40]!r}")

    bad = {"model": MODEL, "messages": [{"role": "user", "content": "no max_tokens here"}]}
    req = urllib.request.Request(MESSAGES, json.dumps(bad).encode(), HEADERS)
    try:
        urllib.request.urlopen(req, timeout=60)
        check("400 error shape", False, "the request was accepted")
    except urllib.error.HTTPError as exc:
        err = json.loads(exc.read() or b"{}")
        check("400 error shape", exc.code == 400 and err.get("type") == "error"
              and err.get("error", {}).get("type") == "invalid_request_error",
              f"{exc.code} {err.get('error', {}).get('type')}")

    raw = json.dumps({"model": MODEL, "max_tokens": 32, "temperature": 0,
                      "messages": [{"role": "user", "content": "Say exactly: chunked ok"}]}).encode()
    host, _, port = API_URL.split("://")[1].partition(":")
    conn = http.client.HTTPConnection(host, port or 8888, timeout=600)
    conn.putrequest("POST", "/v1/messages")
    for k, v in HEADERS.items():
        conn.putheader(k, v)
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()
    conn.send(f"{len(raw):x}\r\n".encode() + raw + b"\r\n0\r\n\r\n")
    resp = conn.getresponse()
    msg4 = json.loads(resp.read() or b"{}")
    texts4 = " ".join(b.get("text", "") for b in blocks(msg4, "text"))
    check("chunked body", resp.status == 200 and "chunked ok" in texts4,
          f"{resp.status} {texts4.strip()[:40]!r}")

    print("anthropiccheck: " + ("all ok" if not fails else f"FAILED {', '.join(fails)}"), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
