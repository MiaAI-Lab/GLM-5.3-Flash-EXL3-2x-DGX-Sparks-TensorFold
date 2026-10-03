#!/usr/bin/env python3
"""An agent's conversation keeps its prompt state while other conversations with the same system prompt run (a coding
agent and its sub-agents, or several chats on one system prompt): each of its next turns must resume from its kept
state, not re-read its history.

Usage: tools/prompt_reuse.py [size]      (API_URL / PORT as in client.py). The conversation's history is ~0.82 x size
tokens (default 40000). Exit code 1 when a turn resumes less than 90% of its prompt. Needs a server started with
`PARALLEL` above 1.

Each round: one request of another conversation with the same system prompt, then the agent's next turn.
"""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import URL, open_url, prose  # noqa: E402

MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
SYSTEM = {"role": "system", "content": "You are a careful coding agent. Read the material, then answer briefly. "
          + prose(2000, 1)}


def chat(messages: list) -> tuple[dict, int, int, float]:
    body = {"model": MODEL, "messages": messages, "max_tokens": 64, "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    reply = json.load(open_url(req, 1800))
    usage = reply["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return reply["choices"][0]["message"], usage["prompt_tokens"], cached, time.time() - t0


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    agent = [SYSTEM, {"role": "user", "content": "Summarize this log in one line.\n" + prose(size, 2)}]
    msg, prompt, cached, secs = chat(agent)
    print(f"agent turn 0: {prompt} prompt tokens, {cached} cached, {secs:.1f} s (cold)")
    worst = 1.0
    for turn in (1, 2, 3):
        chat([SYSTEM, {"role": "user", "content": f"Sub-task {turn}: name three uses of a hash map."}])
        agent += [{"role": "assistant", "content": msg.get("content") or ""},
                  {"role": "user", "content": f"Note {turn}: " + prose(200, 10 + turn)}]
        msg, prompt, cached, secs = chat(agent)
        worst = min(worst, cached / prompt)
        print(f"agent turn {turn} (after another conversation): {prompt} prompt tokens, {cached} cached "
              f"({cached / prompt:.0%}), {secs:.1f} s")
    ok = worst >= 0.9
    print("prompt_reuse:", "OK" if ok else f"FAILED: a turn resumed only {worst:.0%} of its prompt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
