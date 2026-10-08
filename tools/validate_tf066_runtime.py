#!/usr/bin/env python3
"""Bounded live checks for the two-Spark TensorFold v0.6.6 deployment.

Prints scalar JSON records only. Prompts and replies stay in process memory.
The explicit --long-context mode tests 32k and 128k; it never requests 1M.
"""

import argparse
import base64
import binascii
import concurrent.futures
import json
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib


MODEL = "GLM-5.3-Flash-EXL3"


class CheckError(Exception):
    def __init__(self, kind, status=None):
        super().__init__(kind)
        self.kind = kind
        self.status = status


def report(name, started, ok, **values):
    row = {"test": name, "pass": bool(ok), "elapsed_s": round(time.monotonic() - started, 3)}
    row.update(values)
    print(json.dumps(row, sort_keys=True), flush=True)
    return bool(ok)


def post(base, route, payload, timeout):
    request = urllib.request.Request(
        base + route, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise CheckError("http", response.status)
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise CheckError("http", exc.code) from None


def get(base, route, timeout=5):
    try:
        with urllib.request.urlopen(base + route, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as exc:
        raise CheckError("http", exc.code) from None


def chat(model, prompt, *, max_tokens=128, thinking=False, **extra):
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    body.update(extra)
    return body


def answer(reply):
    return reply["choices"][0]["message"].get("content") or ""


def tokens(reply):
    return int((reply.get("usage") or {}).get("completion_tokens") or 0)


def metrics(base):
    values = {}
    for line in get(base, "/metrics").splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition(" ")
            if key in ("tensorfold:requests_running", "tensorfold:requests_waiting",
                       "tensorfold_health:cached_tokens_total"):
                values[key] = float(value)
    return values


def metric(values, name):
    if name not in values:
        raise CheckError("schema")
    return values[name]


def png_halves(width=128, height=64):
    left = b"\xff\x00\x00" * (width // 2)
    right = b"\x00\x00\xff" * (width // 2)
    raw = (b"\x00" + left + right) * height

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def run_check(name, function):
    started = time.monotonic()
    try:
        values = function()
        ok = values.pop("pass")
        if not ok:
            values.setdefault("failure_kind", "semantic")
        return report(name, started, ok, **values)
    except CheckError as exc:
        return report(name, started, False, error_class=exc.kind, http_status=exc.status)
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
        return report(name, started, False, error_class="schema", exception_type=type(exc).__name__)
    except Exception as exc:
        return report(name, started, False, error_class=type(exc).__name__)


def functional(base, model, timeout):
    results = []

    def preflight():
        health = json.loads(get(base, "/health"))
        models = json.loads(get(base, "/v1/models"))
        counters = metrics(base)
        idle = (health.get("ok") is True and not health.get("busy")
                and metric(counters, "tensorfold:requests_running") == 0
                and metric(counters, "tensorfold:requests_waiting") == 0)
        names = [item.get("id") for item in models.get("data", [])]
        return {"pass": idle and model in names, "idle": idle, "model_found": model in names,
                "failure_kind": "state" if not idle or model not in names else None}

    results.append(run_check("preflight", preflight))
    if not results[-1]:
        return False

    def normal():
        reply = post(base, "/v1/chat/completions", chat(model, "Reply with exactly BLUE."), timeout)
        return {"pass": answer(reply).strip().upper() == "BLUE", "completion_tokens": tokens(reply)}

    results.append(run_check("chat_nonstream", normal))

    def stream():
        payload = chat(model, "Reply with exactly GREEN.", stream=True)
        request = urllib.request.Request(
            base + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                frames = []
                for raw in response:
                    if raw.startswith(b"data: "):
                        frames.append(raw[6:].strip())
                    if len(frames) > 1000:
                        raise CheckError("too_many_frames")
        except urllib.error.HTTPError as exc:
            raise CheckError("http", exc.code) from None
        chunks = [json.loads(frame) for frame in frames if frame != b"[DONE]"]
        text = "".join((choice.get("delta") or {}).get("content") or ""
                       for item in chunks for choice in item.get("choices", []))
        return {"pass": bool(chunks) and frames[-1] == b"[DONE]" and text.strip().upper() == "GREEN",
                "frames": len(chunks), "done": bool(frames) and frames[-1] == b"[DONE]"}

    results.append(run_check("chat_sse", stream))

    def reasoning():
        prompt = ("A library has 5 shelves with 8 books on each shelf. It lends 13 books, "
                  "then receives 7 books. How many books does it have now? Reply with the number only.")
        off = post(base, "/v1/chat/completions", chat(model, prompt), timeout)
        on = post(base, "/v1/chat/completions",
                  chat(model, prompt, max_tokens=512, thinking=True,
                       reasoning_effort="low", thinking_budget=96), timeout)
        off_msg = off["choices"][0]["message"]
        on_msg = on["choices"][0]["message"]
        off_reason = off_msg.get("reasoning_content") or ""
        on_reason = on_msg.get("reasoning_content") or ""
        off_numbers = re.findall(r"\b\d+\b", off_msg.get("content") or "")
        on_numbers = re.findall(r"\b\d+\b", on_msg.get("content") or "")
        return {"pass": off_numbers == ["34"] and not off_reason
                and on_numbers == ["34"],
                "off_reasoning_tokens_present": bool(off_reason),
                "on_reasoning_tokens_present": bool(on_reason),
                "on_reasoning_observation": "present" if on_reason else "absent",
                "off_completion_tokens": tokens(off), "on_completion_tokens": tokens(on)}

    results.append(run_check("reasoning_on_off", reasoning))

    def tools():
        definition = {"type": "function", "function": {
            "name": "multiply", "description": "Multiply two integers.",
            "parameters": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"]}}}
        first = chat(model, "Use the multiply tool for 7 times 8.", max_tokens=256,
                     tools=[definition],
                     tool_choice={"type": "function", "function": {"name": "multiply"}})
        first_reply = post(base, "/v1/chat/completions", first, timeout)
        msg = first_reply["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if len(calls) != 1:
            return {"pass": False, "call_count": len(calls)}
        call = calls[0]
        args = json.loads(call["function"]["arguments"])
        typed = call["function"]["name"] == "multiply" and args == {"a": 7, "b": 8}
        first["messages"].extend([msg, {"role": "tool", "tool_call_id": call["id"], "content": "56"}])
        first["tool_choice"] = "none"
        second = post(base, "/v1/chat/completions", first, timeout)
        second_msg = second["choices"][0]["message"]
        roundtrip = "56" in (second_msg.get("content") or "") and not second_msg.get("tool_calls")
        return {"pass": typed and roundtrip, "typed_args": typed, "roundtrip": roundtrip,
                "completion_tokens": tokens(second)}

    results.append(run_check("tool_roundtrip", tools))

    def responses():
        reply = post(base, "/v1/responses", {
            "model": model, "input": "Reply with exactly ORANGE.",
            "temperature": 0, "max_output_tokens": 128,
            "reasoning": {"effort": "none"}, "store": False,
        }, timeout)
        parts = []
        for item in reply.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text") or "")
        text = reply.get("output_text") or "".join(parts)
        return {"pass": text.strip().upper() == "ORANGE", "output_items": len(reply.get("output", []))}

    results.append(run_check("responses_route", responses))

    def anthropic():
        request = urllib.request.Request(
            base + "/v1/messages",
            data=json.dumps({"model": model, "messages": [{"role": "user", "content": "Reply with exactly PURPLE."}],
                             "max_tokens": 128, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise CheckError("http", exc.code) from None
        text = "".join(part.get("text") or "" for part in reply.get("content", []) if part.get("type") == "text")
        return {"pass": text.strip().upper() == "PURPLE", "content_parts": len(reply.get("content", []))}

    results.append(run_check("anthropic_route", anthropic))

    def anthropic_count():
        reply = post(base, "/v1/messages/count_tokens", {
            "model": model,
            "messages": [{"role": "user", "content": "Count the tokens in this harmless sentence."}],
        }, timeout)
        count = reply["input_tokens"]
        return {"pass": isinstance(count, int) and count > 0, "input_tokens": count}

    results.append(run_check("anthropic_count_tokens", anthropic_count))

    def vision():
        image = "data:image/png;base64," + base64.b64encode(png_halves()).decode()
        payload = chat(model, "")
        payload["messages"] = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": "What are the colors of the left and right halves? Reply with two words: left then right."},
        ]}]
        reply = post(base, "/v1/chat/completions", payload, timeout)
        words = re.findall(r"[a-z]+", answer(reply).lower())
        colors = [word for word in words if word in {
            "red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"}]
        return {"pass": colors == ["red", "blue"], "completion_tokens": tokens(reply),
                "color_count": len(colors)}

    results.append(run_check("vision_png", vision))

    def prefix():
        nonce = uuid.uuid4().hex
        system = "Shared synthetic system block " + nonce + "\n" + (
            "Maple amber lantern orbit violet quartz river silver meadow copper.\n" * 130)
        def one(word):
            body = chat(model, "")
            body["messages"] = [{"role": "system", "content": system},
                                {"role": "user", "content": "Reply with exactly " + word + "."}]
            return post(base, "/v1/chat/completions", body, timeout)
        before = metrics(base)
        first = one("ALPHA")
        middle = metrics(base)
        second = one("BETA")
        after = metrics(base)
        cached_first = metric(middle, "tensorfold_health:cached_tokens_total") - metric(
            before, "tensorfold_health:cached_tokens_total")
        cached_second = metric(after, "tensorfold_health:cached_tokens_total") - metric(
            middle, "tensorfold_health:cached_tokens_total")
        return {"pass": answer(first).strip().upper() == "ALPHA"
                and answer(second).strip().upper() == "BETA"
                and cached_second > cached_first,
                "first_cached_delta": int(cached_first), "second_cached_delta": int(cached_second)}

    results.append(run_check("prefix_reuse", prefix))

    def queue_cancel():
        running_key = "tensorfold:requests_running"
        waiting_key = "tensorfold:requests_waiting"
        def long_reply(i):
            body = chat(model, "Count upward from " + str(i * 100 + 1)
                        + ", one integer per line. Continue for at least 1000 lines.",
                        max_tokens=768, ignore_eos=True)
            reply = post(base, "/v1/chat/completions", body, timeout)
            return bool(answer(reply).strip()) and tokens(reply) == 768
        peak = 0
        waited = False
        drained = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(long_reply, i) for i in range(4)]
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not all(f.done() for f in futures):
                peak = max(peak, int(metric(metrics(base), running_key)))
                if peak == 4:
                    break
                time.sleep(.05)
            if peak != 4:
                return {"pass": False, "peak_running": peak, "queue_observed": False}
            host = urllib.parse.urlsplit(base)
            body = json.dumps(chat(model, "Count from 1 to 1000.", max_tokens=768,
                                   ignore_eos=True, stream=True)).encode()
            request = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host.hostname}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                       "Connection: close\r\n\r\n").encode() + body
            sock = socket.create_connection((host.hostname, host.port or 80), timeout=5)
            try:
                sock.sendall(request)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    waited = metric(metrics(base), waiting_key) >= 1
                    if waited:
                        break
                    time.sleep(.05)
            finally:
                sock.close()
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                drained = metric(metrics(base), waiting_key) == 0
                if drained:
                    break
                time.sleep(.05)
            incumbent_ok = all(f.result(timeout=timeout) for f in futures)
        recovery = post(base, "/v1/chat/completions", chat(model, "Reply with exactly READY."), timeout)
        idle = metrics(base)
        recovered = answer(recovery).strip().upper() == "READY"
        return {"pass": waited and drained and incumbent_ok and recovered
                and metric(idle, running_key) == 0 and metric(idle, waiting_key) == 0,
                "peak_running": peak, "queue_observed": waited, "cancel_drained": drained,
                "incumbents_ok": incumbent_ok, "lane_recovered": recovered}

    results.append(run_check("four_lanes_queue_cancel", queue_cancel))
    return all(results)


def long_context(base, model, tiers, timeout):
    results = []
    for target in tiers:
        def needle():
            marker = "violet-" + uuid.uuid4().hex[:12]
            nonce = uuid.uuid4().hex
            line = "Maple amber lantern orbit violet quartz river silver meadow copper."
            def prompt(rows):
                lines = [f"{i:05d}: {line}" for i in range(rows)]
                lines.insert(int(rows * .6), "Remember this passphrase: " + marker)
                return "Synthetic retrieval trial " + nonce + ".\n" + "\n".join(lines) + (
                    "\nWhat is the passphrase? Reply with the passphrase only.")
            def count(rows):
                payload = {"messages": [{"role": "user", "content": prompt(rows)}],
                           "chat_template_kwargs": {"enable_thinking": False}}
                return int(post(base, "/tokenize", payload, min(timeout, 60))["count"])
            sample = count(200)
            rows = max(200, round(200 * target / sample))
            actual = count(rows)
            if not .95 * target <= actual <= 1.05 * target:
                rows = max(200, round(rows * target / actual))
                actual = count(rows)
            body = chat(model, prompt(rows), max_tokens=128)
            reply = post(base, "/v1/chat/completions", body, timeout)
            return {"pass": .95 * target <= actual <= 1.05 * target and marker in answer(reply),
                    "prompt_tokens": actual, "completion_tokens": tokens(reply),
                    "needle_found": marker in answer(reply)}
        results.append(run_check("needle_" + str(target), needle))
    return all(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--functional", action="store_true", help="API, vision, cache, and lane checks (default)")
    mode.add_argument("--long-context", action="store_true", help="explicit 32k and 128k needle checks")
    mode.add_argument("--self-check", action="store_true", help="CPU-only PNG and request-shape check")
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--long-timeout", type=int, default=360)
    parser.add_argument("--tiers", type=int, nargs="+", default=[32000, 128000])
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urllib.parse.urlsplit(base).scheme != "http" or min(args.timeout, args.long_timeout) < 10:
        parser.error("base-url must use http and timeouts must be at least 10 seconds")
    if args.self_check:
        started = time.monotonic()
        data = png_halves()
        length = struct.unpack(">I", data[33:37])[0]
        raw = zlib.decompress(data[41:41 + length])
        valid = (data.startswith(b"\x89PNG\r\n\x1a\n")
                 and struct.unpack(">II", data[16:24]) == (128, 64)
                 and data[37:41] == b"IDAT"
                 and len(raw) == 64 * (1 + 128 * 3)
                 and raw[1:4] == b"\xff\x00\x00"
                 and raw[1 + 64 * 3:1 + 65 * 3] == b"\x00\x00\xff")
        return 0 if report("self_check", started, valid, png_bytes=len(data)) else 1
    if args.long_context:
        if any(tier not in (32000, 128000) for tier in args.tiers):
            parser.error("long-context tiers are limited to 32000 and 128000")
        return 0 if long_context(base, args.model, args.tiers, args.long_timeout) else 1
    return 0 if functional(base, args.model, args.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
