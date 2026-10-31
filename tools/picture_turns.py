#!/usr/bin/env python3
"""A chat that gets one new picture a turn (an agent reading screenshots) must only read the new picture each time: its
earlier pictures keep their size, so the prompt still resumes from its kept state.

Usage: tools/picture_turns.py [pictures] [size]      (API_URL / PORT as in client.py). ``pictures`` 1920x1080 PNGs
(default 12: past the 8 that TensorFold's own 16,384-token picture budget holds at full size), one a turn, each
followed by a text turn, after ~0.82 x ``size`` tokens of text (default 40000). Exit code 1 when a turn after the
first (the text, read cold) reads more than what it added, or when an earlier picture changes size.

With a picture cap that depends on the number of pictures (TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS=16384) the 9th
picture's turn reads the whole chat again and every later one reads it from the first picture on.
"""
import base64
import json
import os
import random
import struct
import sys
import time
import urllib.request
import zlib

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import URL, open_url, prose  # noqa: E402

MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
WIDTH, HEIGHT, BLOCK = 1920, 1080, 40
SLACK = 512                              # a turn's own text and reply, and the 64-token grid kept states sit on


def picture(rng: random.Random) -> str:
    """A 1920x1080 PNG of random colour blocks (2,040 tokens at the default cap), as a data URL."""

    rows = []
    for _ in range(HEIGHT // BLOCK):
        line = b"".join(bytes(rng.randrange(256) for _ in range(3)) * BLOCK for _ in range(WIDTH // BLOCK))
        rows.append((b"\0" + line) * BLOCK)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def chat(messages: list) -> tuple[int, int, float]:
    body = {"model": MODEL, "messages": messages, "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    reply = json.load(open_url(req, 1800))
    usage = reply["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    messages.append({"role": "assistant", "content": reply["choices"][0]["message"].get("content") or "ok"})
    return usage["prompt_tokens"], cached, time.time() - t0


def main() -> None:
    pictures = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    rng = random.Random(time.time_ns())  # new pictures and text every run: nothing resumes from an earlier run
    messages = [{"role": "system", "content": "You are terse. " + prose(64, rng.randrange(1 << 30))},
                {"role": "user", "content": "Notes for later, reply ok.\n" + prose(size, rng.randrange(1 << 30))}]
    bad, first = [], 0
    print("turn         prompt  cached     new  seconds")
    before, cached, secs = chat(messages)
    print(f"   text     {before:7d} {cached:7d} {before - cached:7d}  {secs:7.1f}  (cold)", flush=True)
    for n in range(1, pictures + 1):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"Screenshot {n}. One word for it."},
            {"type": "image_url", "image_url": {"url": picture(rng)}}]})
        prompt, cached, secs = chat(messages)
        grew = prompt - before
        first = first or grew
        note = ""
        if prompt - cached > max(grew, 0) + SLACK:
            note = "  <- re-read its history"
            bad.append(f"picture {n}'s turn read {prompt - cached} tokens for a picture of {grew}")
        if grew < 0.75 * first:              # the earlier pictures shrank: the prompt grew by less than one picture
            note = note or "  <- earlier pictures resized"
            bad.append(f"picture {n} added {grew} tokens to the prompt, the first one {first}")
        print(f"picture {n:2d}  {prompt:7d} {cached:7d} {prompt - cached:7d}  {secs:7.1f}{note}", flush=True)
        messages.append({"role": "user", "content": "One more word."})
        prompt, cached, secs = chat(messages)
        if prompt - cached > SLACK:
            bad.append(f"the text turn after picture {n} read {prompt - cached} tokens")
        print(f"   text {n:2d}  {prompt:7d} {cached:7d} {prompt - cached:7d}  {secs:7.1f}", flush=True)
        before = prompt
    if bad:
        sys.exit("FAIL: " + "; ".join(bad))
    print(f"OK: {pictures} pictures, every turn read only what it added")


if __name__ == "__main__":
    main()
