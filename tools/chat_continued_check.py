#!/usr/bin/env python3
"""Checks for patch 0100 (a conversation keeps its id when a stream continues it without its own turn state), run
inside the image that scripts/prepare.sh built:

    docker run --rm --entrypoint python -v "$PWD/tools/chat_continued_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

continued_chat on stubbed kept states (no GPU or model): issue #104's in-turn rewind before the kept turns continues
its conversation; another conversation behind the same system prompt and opening message, an edit of the first message,
shared and picture states start or skip as before; the most recently used match wins; the counters and the log line of
a large prefill (rank 0 only). Exit code 1 when a check fails.
"""
import contextlib
import io
from types import SimpleNamespace as NS

import tensorfold.families.glm5_next.cuda.multi as M

SYSTEM = list(range(100000, 100900))                 # a 900-token system prompt every conversation shares
OPEN = list(range(200000, 200100))                   # and the same 100-token opening message
fails: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def state(ids, chat, own=True, images=False):
    return NS(ids=list(ids), chat=chat, own=own, images=images)


def convo(seed, turns, first=SYSTEM + OPEN):
    """A conversation's prompts: the first one, then each turn appending a reply and a tool result of its own."""
    prompts, ids = [list(first)], list(first)
    for t in range(turns):
        ids = ids + [seed * 1000 + t * 50 + k for k in range(50)]
        prompts.append(list(ids))
    return prompts


big = list(range(500000))
check("common_prefix", M.common_prefix([1, 2, 3, 4], [1, 2, 9]) == 2 and M.common_prefix([1, 2, 3], [1, 2, 3, 4]) == 3
      and M.common_prefix([], [1]) == 0 and M.common_prefix(big, big[:123457] + [-1] + big[123458:]) == 123457)

a, b = convo(7, 6), convo(8, 6)
roots = {1: len(a[0])}
kept = [state(SYSTEM, 0, own=False), state(a[5], 1), state(a[6], 1)]
check("a rewind before the kept turns continues its conversation (#104)",
      M.continued_chat(kept, a[3] + [9, 9, 9], roots) is kept[-1])
check("another conversation behind the same system prompt and opening starts its own",
      M.continued_chat(kept, b[2], roots) is None)
check("a prompt that has not repeated the first reply yet starts its own",
      M.continued_chat(kept, a[0] + [5] * 10, roots) is None)
edited = SYSTEM + OPEN[:50] + [0] * 50 + a[2][len(a[0]):]
check("an edit of the first message starts a new conversation", M.continued_chat(kept, edited, roots) is None)
check("shared, picture and forgotten conversations' states are skipped",
      M.continued_chat([state(a[4], 1, own=False)], a[2] + [1], roots) is None
      and M.continued_chat([state(a[4], 1, images=True)], a[2] + [1], roots) is None
      and M.continued_chat([state(a[4], 2)], a[2] + [1], roots) is None)
two = M.continued_chat([state(a[3], 1), state(a[4], 2)], a[2] + [1], {1: len(a[0]), 2: len(a[0])})
check("the most recently used match wins", two is not None and two.chat == 2)

d = object.__new__(M.MultiDecoder)
d.rank, d.chats_continued, d.chats_continued_prefill = 0, 0, 0
long = convo(7, 200)
prior, prompt = state(long[200], 1), long[150] + [9]
out = io.StringIO()
with contextlib.redirect_stdout(out):
    d._continued(1, prior, NS(shared=True, ids=SYSTEM), len(SYSTEM), prompt)
text = out.getvalue()
check("counted and logged: where it resumed, from what, where it went back",
      (d.chats_continued, d.chats_continued_prefill) == (1, len(prompt) - len(SYSTEM))
      and "conversation 1: resumed at 900 of" in text and "from a shared prefix" in text
      and f"went back to token {len(long[150])}" in text)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    d._continued(1, prior, None, len(prompt) - 10, prompt)                  # a small prefill: counted, not logged
    d.rank = 1
    d._continued(1, prior, None, 0, prompt)                                 # rank 1 counts, never logs
check("a small prefill and rank 1: counted, not logged", d.chats_continued == 3 and out.getvalue() == "")

print("all passed" if not fails else f"{len(fails)} failed")
raise SystemExit(1 if fails else 0)
