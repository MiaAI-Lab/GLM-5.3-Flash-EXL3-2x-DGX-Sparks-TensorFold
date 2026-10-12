#!/usr/bin/env python3
"""Checks for patch 0097 (the kept-state count follows a share of the pool's memory), run inside the image that
scripts/prepare.sh built:

    docker run --rm --entrypoint python -v "$PWD/tools/kept_cap_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

engine.kept_entries' arithmetic (off by default: the configured count, no note; a share of the budget; the three quarter
ceiling that keeps the reservation inside the start-up estimate), and MultiDecoder._keep on the real Pool with a stubbed
snapshot (no GPU or model): the derived count keeps more than 32 states while the pool has room, the default evicts at
32, and the cached key arrays equal the ones built each time. Exit code 1 when a check fails.
"""
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

import tensorfold.families.glm5_next.cuda.decode as D
import tensorfold.families.glm5_next.cuda.multi as M
from tensorfold.families.glm5_next.cuda.engine import kept_entries
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Pool, align_up

MB, GIB = 1 << 20, 1 << 30
fails: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# -- the count --------------------------------------------------------------------------------------------------------
STATE, BUDGET = 45 * MB, int(12.5 * GIB)               # one state's fixed cost; KV_POOL_GIB's budget
check("off: the configured count, no note", kept_entries(32, 0, BUDGET, STATE) == (32, ""))
n, note = kept_entries(32, 25, BUDGET, STATE)
check("25%: a quarter of the budget in states (71 of 45 MiB)", n == 71 and note and n * STATE <= BUDGET * 0.25)
check("never below the configured count", kept_entries(128, 5, BUDGET, STATE)[0] == 128)
check("a share past 75% is read as 75%", kept_entries(32, 100, BUDGET, STATE)[0] == int(BUDGET * 0.75 // STATE))
n, note = kept_entries(400, 0, BUDGET, STATE)
check("an explicit count past 3/4 of the budget is cut (the estimate would not cover it)",
      n == int(BUDGET * 0.75 // STATE) and "three quarters" in note and n * STATE <= BUDGET * 0.75)
check("a small budget still keeps one", kept_entries(32, 0, 10 * MB, STATE)[0] == 1)
check("the same on every rank (pure)", kept_entries(32, 25, BUDGET, STATE) == kept_entries(32, 25, BUDGET, STATE))

# -- _keep with the count ---------------------------------------------------------------------------------------------
def fake_take(e, ids, pending, mtp, drafter):
    return NS(ids=list(ids), rec=torch.zeros(1), conv=torch.zeros(1), pending=None, drafter_rows=None, head=None,
              tail=None, rows=None, nbytes=0, mtp_len=-1)


D.take_snapshot = fake_take


def dec(entries):
    m = object.__new__(M.MultiDecoder)
    m.pool = Pool(4000 * ALIGN)
    m.kept, m.lanes, m.partial, m.ops, m.spilled = [], {}, None, [], []
    m.e = NS(size=1, use=lambda st: None)
    m.g = NS(cache_entries=entries)
    m.rank, m.next_kid, m.kept_bytes_cap, m.disk = 0, 0, 0, None
    m._emit = lambda op, p: m.ops.append((op, list(p)))
    m._ctx = lambda lane: None
    return m


def one_shots(m, count, first=0):
    for i in range(first, first + count):
        x = m.pool.add(i * ALIGN * 2, align_up(64))
        x.kept, x.owner = [], None
        lane = NS(st=None, s=NS(prompt=list(range(i * 1000, i * 1000 + 80))), dflash=False, key=None, images=False,
                  extent=x, shared=[], chat=i)
        m._keep(lane, 64, own=True)


m = dec(32)
one_shots(m, 60)
check("default 32: 60 finished conversations leave 32 states", len(m.kept) == 32)
m = dec(kept_entries(32, 25, BUDGET, STATE)[0])
one_shots(m, 60)
check("25% share: all 60 kept while the pool has room", len(m.kept) == 60 and m.pool.free_rows() > 0)
one_shots(m, 40, first=60)
check("25% share: past 71 the count still bounds the states", len(m.kept) == 71)

# -- the cached key array ---------------------------------------------------------------------------------------------
keys = [list(range(i, i + 200_000)) for i in range(32)]
cs = [NS(key=k) for k in keys]
t0 = time.perf_counter()
fresh = [np.asarray(c.key, dtype=np.int64) for c in cs]
t_fresh = time.perf_counter() - t0
first = [M.known_key(c) for c in cs]
t0 = time.perf_counter()
again = [M.known_key(c) for c in cs]
t_again = time.perf_counter() - t0
check("known_key equals np.asarray(key, int64)", all(np.array_equal(a, b) for a, b in zip(fresh, first)))
check("known_key is built once", all(a is b for a, b in zip(first, again)) and t_again < t_fresh / 50)
print(f"     32 keys of 200k tokens: {t_fresh * 1000:.0f} ms each admission before, {t_again * 1000:.2f} ms now")

print("all passed" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
