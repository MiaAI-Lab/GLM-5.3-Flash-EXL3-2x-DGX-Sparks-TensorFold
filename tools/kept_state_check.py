#!/usr/bin/env python3
"""Checks for patch 0089 (the kept-state byte budget and per-conversation quota), run inside the image that
scripts/prepare.sh built:

    docker run --rm --entrypoint python -v "$PWD/tools/kept_state_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

MultiDecoder's _keep, _drop and _evict on the real Pool with a stubbed snapshot (no GPU or model): both limits off by
default (nothing dropped, nothing spilled), the quota keeping a conversation's newest states and not counting shared
ones, the byte budget evicting to its size with one EVICT op a victim, neither limit writing to the spill tier while the
entry cap's drops do. Exit code 1 when a check fails.
"""
from types import SimpleNamespace as NS

import torch

import tensorfold.families.glm5_next.cuda.multi as M
import tensorfold.families.glm5_next.cuda.decode as D
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Pool, align_up

MB = 1 << 20
fails: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def fake_take(e, ids, pending, mtp, drafter):
    return NS(ids=list(ids), rec=torch.zeros(e.size * MB // 4, dtype=torch.float32), conv=torch.zeros(1), pending=None,
              drafter_rows=None, head=None, tail=None, rows=None, nbytes=0, mtp_len=-1)


D.take_snapshot = fake_take
nx = [0]


def dec(entries=32, size=100, cap=0):
    m = object.__new__(M.MultiDecoder)
    m.pool = Pool(1000 * ALIGN)
    m.kept, m.lanes, m.partial, m.ops, m.spilled = [], {}, None, [], []
    m.e = NS(size=size, use=lambda st: None)
    m.g = NS(cache_entries=entries)
    m.rank, m.next_kid, m.kept_bytes_cap = 0, 0, cap
    m.disk = object()                                    # the spill tier "on": _spill records what it is asked
    m._spill = lambda c, leaving=False, keep=frozenset(): m.spilled.append(c.kid) or True
    m._emit = lambda op, p: m.ops.append((op, list(p)))
    m._ctx = lambda lane: None
    return m


def lane(m, chat, toks):
    x = m.pool.add(nx[0] * ALIGN * 2, align_up(toks))
    nx[0] += 1
    x.kept, x.owner = [], None
    return NS(st=None, s=NS(prompt=list(range(chat * 1000, chat * 1000 + toks + 10))), dflash=False, key=None,
              images=False, extent=x, shared=[], chat=chat)


check("defaults: no quota, no byte cap", M.KEEP_PER_CHAT == 0 and M.KEPT_BYTES_CAP == 0)

m = dec()
l = lane(m, 1, 64)
for i in range(6):
    m._keep(l, 64 + i, own=True)
check("off: all six of one chat kept, nothing evicted or spilled", len(m.kept) == 6 and not m.ops and not m.spilled)

M.KEEP_PER_CHAT = 2
m = dec()
l = lane(m, 1, 64)
for i in range(5):
    m._keep(l, 64 + i, own=True)
check("quota keeps 2 own states of one chat", sum(1 for c in m.kept if c.own) == 2)
check("quota keeps the newest", sorted(len(c.ids) for c in m.kept) == [67, 68])
check("quota drops are not spilled", not m.spilled)
sh = lane(m, 2, 64)
m._keep(sh, 64)
m._keep(sh, 66)
check("shared states are not counted", sum(1 for c in m.kept if not c.own) == 2)
a, b = lane(m, 3, 64), lane(m, 4, 64)
for i in range(3):
    m._keep(a, 64 + i, own=True)
    m._keep(b, 64 + i, own=True)
check("two chats each hold their own quota", sum(1 for c in m.kept if c.own and c.chat == 3) == 2
      and sum(1 for c in m.kept if c.own and c.chat == 4) == 2)

M.KEEP_PER_CHAT = 0
m = dec(cap=250 * MB)
ls = [lane(m, i, 64) for i in range(6)]
for i, l in enumerate(ls):
    m._keep(l, 64 + i)
check("byte cap: evicts down to 250 MB (2 states of 100 MB)", m._held_bytes() <= 250 * MB and len(m.kept) == 2)
check("byte cap: one EVICT op a victim, marked no-spill",
      [p[1] for op, p in m.ops if op == M.EVICT] == [0, 0, 0, 0])
check("byte cap: nothing spilled", not m.spilled)

m = dec(entries=2)
ls = [lane(m, i, 64) for i in range(4)]
for i, l in enumerate(ls):
    m._keep(l, 64 + i)
check("entry cap: still spills what it drops", len(m.kept) == 2 and len(m.spilled) == 2)

# rank 1 applies the EVICT: the same drop, no spill
r = dec(cap=250 * MB)
r.rank = 1
x = lane(r, 9, 64)
r._keep(x, 64)
c = r.kept[0]
r._drop(c, spill=False)
check("rank 1: EVICT without spill drops and does not write", not r.kept and not r.spilled)

print("all passed" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
