#!/usr/bin/env python3
"""Checks for patch 0119 (a conversation keeps its id when a stream continues it without its own turn state), run
inside the image that scripts/prepare.sh built:

    docker run --rm --entrypoint python -v "$PWD/tools/chat_continued_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

continued_chat on stubbed kept states (no GPU or model): issue #104's in-turn rewind before the kept turns continues
its conversation; another conversation behind the same system prompt and opening message, an edit of the first message,
shared and picture states start or skip as before; the most recently used match wins; the counters and the log line of
a large prefill (rank 0 only). Then MultiDecoder._admitted itself on a stubbed slot state (forward.State) and a real
Pool: the conversation an admitted stream joins (a rewind resumed from nothing joins its conversation and is counted,
another conversation behind the same opening, draft:false and a picture prompt get new ids, a session whose opening and
first reply are the same as another's joins it), and chat_roots pruned past 4 x max(16, TF_GLM_CACHE_ENTRIES)
conversations to those a kept state is in. Exit code 1 when a check fails.
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


# -- MultiDecoder._admitted itself: which conversation an admitted stream joins, and chat_roots pruned ----------------
from tensorfold.families.glm5_next.cuda import forward as F      # noqa: E402
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Pool   # noqa: E402


class FakeState:                                     # forward.State: only what _admitted calls with no kept prompt
    def __init__(self, *a, **kw):
        pass

    def reset(self):
        pass


def admitting(entries=8):
    m = object.__new__(M.MultiDecoder)
    m.rank, m.w = 0, NS(world=1, device="cpu")
    m.g, m.e = NS(costs=None, cache_entries=entries), NS(rows=0, caches=None, slots=8)
    m.pool, m.grid, m.drafts, m.kept, m.lanes = Pool(64 * ALIGN), 0, None, [], {}
    m.next_sid = m.next_kid = m.next_order = m.next_chat = 0
    m.chat_roots, m.chats_continued, m.chats_continued_prefill = {}, 0, 0
    m._kept_by_id = lambda kid: None
    m.boot = "check"                                 # (the spill tier's blocks name a stream's lineage after it)
    return m


def admit(m, prompt, draft=True, images=False):
    """An ADMIT with no kept prompt to resume from (a rewind past every kept state, a state read back from disk or
    dropped): the stream's conversation, and the stream."""

    sid = m.next_sid
    a = dict(kid=-1, prompt=list(prompt), copy=True, base=(sid % 32) * ALIGN, size=ALIGN, sid=sid, slot=sid % 8,
             code=[0], images=images, draft=draft, keyed=None, shared=[])
    s = NS()
    m._admitted(a, s, None, None)
    lane = m.lanes[sid]
    m.pool.remove(lane.extent)                       # (its extent free for the next one; a test pool of 64 blocks)
    return lane.chat


State = F.State
F.State = FakeState
try:
    m = admitting()
    one = convo(7, 3)
    c1 = admit(m, one[0])
    check("_admitted: a first prompt starts conversation 1 and records its first prompt's length",
          c1 == 1 and m.chat_roots == {1: len(one[0])})
    m.kept = [state(SYSTEM, 0, own=False), state(one[3], c1)]
    c = admit(m, one[1] + [9, 9])
    check("_admitted: a rewind before the kept turns, resumed from nothing, joins its conversation (#104)",
          c == c1 and m.chats_continued == 1 and m.chats_continued_prefill == len(one[1]) + 2 and m.next_chat == 1)
    other = convo(8, 3)
    c2 = admit(m, other[1])
    check("_admitted: another conversation behind the same system prompt and opening gets its own id",
          c2 == 2 and m.chats_continued == 1 and m.chat_roots[2] == len(other[1]))
    c = admit(m, one[1] + [9], draft=False)
    check("_admitted: draft:false is never counted as a continuation (a new id)", c == 3 and m.chats_continued == 1)
    c = admit(m, one[1] + [9], images=True)
    check("_admitted: a prompt with pictures is not matched (a new id)", c == 4 and m.chats_continued == 1)
    twin = SYSTEM + OPEN + one[1][len(one[0]):len(one[0]) + M.CHAT_MATCH] + [77] * 40
    c = admit(m, twin)
    check("_admitted: a second session whose opening and first reply (temperature 0) are the same joins the first: "
          "the match cannot tell them apart (README)", c == c1 and m.chats_continued == 2)

    m = admitting(entries=8)                         # chat_roots holds at most 4 x max(16, entries) = 64 conversations
    limit = 4 * max(16, m.g.cache_entries)
    firsts = [convo(100 + i, 0)[0][:-1] + [300000 + i] for i in range(limit)]
    for f in firsts:
        admit(m, f)
    check("_admitted: chat_roots grows to its limit without pruning", len(m.chat_roots) == limit)
    m.kept = [state(firsts[0] + [1] * 40, 1), state(firsts[5] + [2] * 40, 6), state(SYSTEM, 0, own=False)]
    newest = admit(m, convo(999, 0)[0][:-1] + [399999])
    check("_admitted: past it, only the conversations a kept state is in stay, and the new one",
          set(m.chat_roots) == {1, 6, newest} and newest == limit + 1)
    c = admit(m, firsts[0] + [1] * 40 + [5])
    check("_admitted: a pruned-to conversation is still continued", c == 1)
    c = admit(m, firsts[2] + [5] * 40)
    check("_admitted: a pruned conversation's prompt starts a new one (its root is gone)", c == newest + 1)
finally:
    F.State = State

print("all passed" if not fails else f"{len(fails)} failed")
raise SystemExit(1 if fails else 0)
