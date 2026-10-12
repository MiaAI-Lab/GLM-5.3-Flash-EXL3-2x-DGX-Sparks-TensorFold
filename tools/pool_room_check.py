#!/usr/bin/env python3
"""Checks for patch 0074 (the shared pool compacts before it evicts), run inside the image that scripts/prepare.sh
built:

    docker run --rm --entrypoint python -v "$PWD/tools/pool_room_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

MultiDecoder's room-making (_grow, _room) on the real Pool and a CPU arena, no GPU or model: a conversation whose
extent cannot grow in place and a new request, each while the pool's free rows cover them in several ranges (nothing
evicted, every kept prompt's rows intact, each extent moved once at most, rank 1 replaying the ops to the same
extents and rows), the same for a decoding stream's extent, too few free rows (only the least recently used state it
needs evicted), and protected extents. Exit code 1 when a check fails.
"""
from types import SimpleNamespace as NS

import torch

from tensorfold.families.glm5_next.cuda.multi import MOVE, MultiDecoder
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up

B = ALIGN                                   # one block of rows
failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f" ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def decoder(blocks):
    rows = blocks * B
    m = object.__new__(MultiDecoder)
    m.pool, m.kept, m.lanes, m.partial = Pool(rows), [], {}, None
    m.arena = Arena(rows, [Plane(torch.zeros(rows, dtype=torch.int32)), Plane(torch.zeros(rows // 4, dtype=torch.int32), 4)])
    m.ops = []
    m._emit = lambda op, p: m.ops.append((op, list(p)))
    m.disk = None                        # patch 0088's spill tier: off
    return m


def moves(m, x):
    """How many times rank 0 moved extent x."""
    return sum(1 for op, p in m.ops if op == MOVE and p[0] == x.eid)


def replays(m, r):
    """Rank 1 (``r``, built like ``m`` before the call) applies rank 0's ops and ends with the same extents and rows."""
    msg = [v for op, p in m.ops for v in (op, len(p), *p)]
    r.apply(msg)
    same = [(x.eid, x.base, x.size) for x in r.pool.extents] == [(x.eid, x.base, x.size) for x in m.pool.extents]
    return same and all(bool((a.tensor == b.tensor).all()) for a, b in zip(m.arena.planes, r.arena.planes))


def keep(m, at_block, tokens, name):
    """A kept prompt of ``tokens`` at block ``at_block``, its rows tagged with its own number; appended as the most
    recently used."""
    x = m.pool.add(at_block * B, align_up(tokens))
    tag = len(m.kept) + 1
    for p in m.arena.planes:
        p.view(x.base, x.size)[:-(-tokens // p.div)] = tag
    c = NS(ids=[0] * tokens, key=[0] * tokens, extent=x, kid=tag, shared=False, name=name, tag=tag,
           rec=0, conv=0, tail=0, head=0, drafter_rows=0)
    x.kept = [c]
    m.kept.append(c)
    return x


def alive(m):
    return [c.name for c in m.kept]


def rows_intact(m):
    """Every kept prompt's rows still hold its tag at its extent's current base."""
    for c in m.kept:
        for p in m.arena.planes:
            got = p.view(c.extent.base, c.extent.size)[:-(-len(c.ids) // p.div)]
            if not bool((got == c.tag).all()):
                return False
    return True


# A conversation's next turn needs 2 more blocks; the rows right after it are another kept prompt and no free gap
# holds the grown extent, but the pool's free rows (5 blocks in 4 holes) cover it. Least recently used: "other" (the
# second agent's only state).
def fragmented():
    m = decoder(24)
    keep(m, 0, 6 * B, "other")            # blocks 0-5
    keep(m, 7, 2 * B, "old1")             # hole at 6
    keep(m, 10, 2 * B, "old2")            # hole at 9
    a = keep(m, 13, 6 * B, "agent")       # hole at 12
    keep(m, 19, 3 * B, "neighbour")       # right after the agent; hole at 22-23
    m.kept = [c for c in m.kept if c.name != "agent"] + [c for c in m.kept if c.name == "agent"]
    return m, a


m, a = fragmented()
r, _ = fragmented()
check("fragmented: the free rows cover the growth, no gap does",
      m.pool.free_rows() >= 2 * B and m.pool.room_after(a) == 0 and m.pool.place(8 * B, ignore=[a]) is None)
check("grow: succeeds", m._grow(a, 8 * B, protect=[a]))
check("grow: compacts instead of evicting (every kept prompt stays)",
      alive(m) == ["other", "old1", "old2", "neighbour", "agent"], alive(m))
check("grow: the agent's extent has its rows", a.size >= 8 * B)
check("grow: every kept prompt's rows moved intact", rows_intact(m))
check("grow: the neighbour moves once (up, out of the agent's way), not down and back up",
      moves(m, m.pool.extents[-1]) == 1, m.ops)
check("grow: rank 1 replaying the ops ends with the same extents and rows", replays(m, r))

# The same while the agent's extent is a decoding stream's (its rows up to the stream's position, rebound when moved).
m, a = fragmented()
bound = []
m.kept = [c for c in m.kept if c.name != "agent"]
a.kept, a.owner = [], 7
for p in m.arena.planes:
    p.view(a.base, a.size)[:] = 99
m.lanes = {7: NS(st=NS(pos=6 * B - 5, bind=lambda base, size: bound.append((base, size))))}
check("decoding grow: succeeds", m._grow(a, 8 * B, protect=[a]))
check("decoding grow: compacts instead of evicting", alive(m) == ["other", "old1", "old2", "neighbour"], alive(m))
check("decoding grow: the stream's rows moved intact",
      all(bool((p.view(a.base, a.size)[:-(-(6 * B - 5) // p.div)] == 99).all()) for p in m.arena.planes))
check("decoding grow: the stream is bound to where its extent is", not bound or bound[-1] == (a.base, a.size), bound)
check("decoding grow: the other kept prompts' rows moved intact", rows_intact(m))

# A new request needs 4 blocks: the free rows (4 holes) cover it, no gap does.
def holes():
    m = decoder(16)
    keep(m, 0, 3 * B, "other")
    keep(m, 4, 3 * B, "old1")
    keep(m, 8, 3 * B, "old2")
    keep(m, 12, 3 * B, "old3")
    return m


m, r = holes(), holes()
base = m._room(4 * B)
check("room: a base for the new extent", base is not None and m.pool.place(4 * B) == base)
check("room: compacts instead of evicting", alive(m) == ["other", "old1", "old2", "old3"], alive(m))
check("room: every kept prompt's rows moved intact", rows_intact(m))
check("room: rank 1 replaying the ops ends with the same extents and rows", replays(m, r))

# Too little free rows: evict least recently used, only until the free rows cover the request, then compact. Here the
# oldest ("other", 3 blocks, not next to a hole big enough) is enough; evicting until a gap opens would also drop more.
m = decoder(16)
keep(m, 0, 3 * B, "other")            # hole at 3
keep(m, 4, 5 * B, "old1")             # hole at 9
keep(m, 10, 5 * B, "old2")            # hole at 15
base = m._room(5 * B)
check("room short: a base", base is not None)
check("room short: evicts only the least recently used state it needs", alive(m) == ["old1", "old2"], alive(m))
check("room short: the survivors' rows moved intact", rows_intact(m))

# Protected extents are never evicted, compaction or not.
m = decoder(10)
p = keep(m, 0, 4 * B, "protected")
keep(m, 5, 4 * B, "old")
check("protect: room for 4 blocks evicts the unprotected one", m._room(4 * B, [p]) is not None and alive(m) == ["protected"],
      alive(m))

print(f"{'FAILED ' + str(len(failures)) if failures else 'all passed'}")
raise SystemExit(1 if failures else 0)
