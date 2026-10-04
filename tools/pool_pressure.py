#!/usr/bin/env python3
"""Two conversations taking turns keep each other's prompt state when the shared pool is nearly full (two agents at
once, each with a long history): one's next turn must not evict the other's kept state while the pool's free rows
would hold it after a compaction.

Usage: tools/pool_pressure.py [size]     (API_URL / PORT as in client.py). Run it on a freshly started server with
`PARALLEL` above 1 and nothing else sending requests: it places prompts by filling the pool. A smaller pool runs it
faster (e.g. CONTEXT=131072 KV_POOL_GIB=0.5). Each conversation's history is ~size tokens (default 30000, at least
24000). Exit code 1 when a turn resumes less than 90% of the conversation's previous prompt; 2 when the pool could not
be filled as the check needs (the kept-prompt cap, TF_GLM_CACHE_ENTRIES, reached first).

Conversations A and B each take a turn (B's state lands right after A's in the pool), other conversations fill the
pool until fewer free rows are left than A's next turn takes, then A takes a turn that adds ~4000 tokens (its rows
must grow into B's or move), then B takes its next turn.
"""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import API_URL, URL, open_url, prose  # noqa: E402

MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")


def health() -> dict:
    return json.load(open_url(urllib.request.Request(API_URL + "/health"), 30))


def chat(messages: list) -> tuple[dict, int, int, float]:
    body = {"model": MODEL, "messages": messages, "max_tokens": 32, "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    reply = json.load(open_url(req, 1800))
    usage = reply["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return reply["choices"][0]["message"], usage["prompt_tokens"], cached, time.time() - t0


def turn(name: str, convo: list, before: int = 0) -> tuple[int, float]:
    """The turn's prompt tokens, and the share of ``before`` (the conversation's previous prompt) it resumed."""
    msg, prompt, cached, secs = chat(convo)
    convo.append({"role": "assistant", "content": msg.get("content") or ""})
    share = min(1.0, cached / before) if before else 1.0
    print(f"{name}: {prompt} prompt tokens, {cached} cached" + (f" ({share:.0%} of its previous prompt)" if before
                                                               else "") + f", {secs:.1f} s")
    return prompt, share


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    if size < 24000:
        sys.exit("size: at least 24000 (the free rows left must hold A's next turn's ~4000 more)")
    h = health()
    if "pool_tokens" not in h:
        sys.exit("run it on a server started with PARALLEL above 1 (no shared pool otherwise)")
    if h.get("kept_prompts") or h.get("requests_running"):
        sys.exit(f"the server holds {h.get('kept_prompts')} kept prompts and runs {h.get('requests_running')} "
                 "requests: start it again, and send nothing else while this runs")
    words = int(size / 0.82)
    a = [{"role": "user", "content": "Conversation A. Summarize this log in one line.\n" + prose(words, 1)}]
    b = [{"role": "user", "content": "Conversation B. Summarize this log in one line.\n" + prose(words, 2)}]
    a1, _ = turn("A, turn 1 (cold)", a)
    b1, _ = turn("B, turn 1 (cold)", b)
    pool = h["pool_tokens"]
    longest = min(h["context_length"] - 4096, 262144)      # long cold prompts take minutes each
    now = health()
    left, kept, n = now["pool_free_tokens"], now["kept_prompts"], 0
    while left > size // 2:              # fewer free rows than A's next turn takes (and size // 4 or more: it adds ~4000)
        n += 1
        fill = min(left - size // 4 - 2048, longest)
        chat([{"role": "user", "content": f"Filler {n}. Name the most common word.\n" + prose(int(fill / 0.82), 100 + n)}])
        now = health()
        if now["kept_prompts"] <= kept:
            print(f"the kept-prompt cap ({kept}) was reached before the pool filled: inconclusive (raise "
                  "TF_GLM_CACHE_ENTRIES or pass a larger size)")
            sys.exit(2)
        left, kept = now["pool_free_tokens"], now["kept_prompts"]
    print(f"{n} other conversations: {left} of the pool's {pool} rows free")
    a.append({"role": "user", "content": "A tool result:\n" + prose(int(4000 / 0.82), 3)})
    _, worst = turn("A, turn 2 (adds ~4000 tokens)", a, a1)
    b.append({"role": "user", "content": "And in one word?"})
    worst = min(worst, turn("B, turn 2", b, b1)[1])
    ok = worst >= 0.9
    print("pool_pressure:", "OK" if ok else f"FAILED: a turn resumed only {worst:.0%} of its previous prompt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
