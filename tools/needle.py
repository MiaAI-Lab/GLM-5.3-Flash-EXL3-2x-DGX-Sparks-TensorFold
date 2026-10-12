#!/usr/bin/env python3
"""Needle in a haystack at ~195k tokens: a passphrase hidden at ~60% depth of random prose, asked for greedily.

Usage: tools/needle.py [label] [size]   (size: client.prose's length parameter, the prompt comes out at ~0.8 x size
tokens; default 248000, ~195k tokens. API_URL / PORT as in client.py). Exit code 1 if the answer is wrong.
"""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import URL, open_url, prose  # noqa: E402

SECRET = "violet-harbor-7291"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "needle"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 248_000
    hay = prose(size, 777).split(". ")
    at = int(len(hay) * 0.6)
    hay.insert(at, f"Remember this: the secret passphrase is {SECRET}")
    prompt = ". ".join(hay) + "\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only."
    body = {"model": os.environ.get("MODEL", "GLM-5.3-Flash-EXL3"), "max_tokens": 2048, "temperature": 0, "seed": 1234,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(open_url(req, 3600))
    content = reply["choices"][0]["message"].get("content") or ""
    ok = SECRET in content
    print(f"{label}: needle {reply['usage']['prompt_tokens']} tok, prefill {reply['tensorfold'].get('prefill_s')} s, "
          f"total {time.time() - t0:.1f} s, answer {content.strip()[-80:]!r}: {'CORRECT' if ok else 'WRONG'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
