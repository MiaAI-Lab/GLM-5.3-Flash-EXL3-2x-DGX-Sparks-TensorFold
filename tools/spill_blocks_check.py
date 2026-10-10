#!/usr/bin/env python3
"""Checks for patch 0110 (the spill tier's block store, tensorfold/cuda/spill_blocks.py, and its hooks in multi.py),
CPU only, run inside the image that scripts/prepare.sh built (no GPU, no model, no server); its files go to a scratch
folder under $TMPDIR:

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/spill_blocks_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

A real BlockStore over a real Arena on the host (a latents plane, div 1, and a pooled-key plane, div 4 with 2 pad
rows), with a stand-in small state, and MultiDecoder's own load and move paths on a real Pool:
- the block a kept state's end cuts holds the rows of the keep, not what the stream writes after it (its pooled-key
  row still filling at the keep);
- compaction moving an extent a stored prompt is being read into goes on without waiting for the disk, the read's next
  blocks land at the new base and none at the old one, and a move that starts while a block's copy is being issued
  waits for that copy and moves it too;
- a restored state keeps its point's shared-prefix flag, rank 0's on every rank (LOADED carries it);
- the same ids kept again as a shared prefix make the stored point shared, its json rewritten through a temporary
  name and the folder fsynced (a restart reads it so), also when they were still waiting or being written; a point
  dropped during the rewrite does not get its json back.
Exit code 1 when a check fails; tools/spill_blocks_mutants.sh runs it on faithful mutants of each fix.
"""
import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

from tensorfold.cuda import spill as SP
from tensorfold.cuda import spill_blocks as S
from tensorfold.families.glm5_next.cuda import multi as M
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up

ROWS = 8 * ALIGN
TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "60"))
fails: list[str] = []
ROOT = tempfile.mkdtemp(prefix="spill-blocks-check-")


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


def _hung() -> None:
    print(f"FAIL timed out after {TIMEOUT:g} s", flush=True)
    os._exit(1)


threading.Timer(TIMEOUT, _hung).start()


class Snap:
    """A stand-in for decode.Snapshot: the fields ``_finish_load`` moves to the device, and the multi-stream ones
    ``_SPILL_SKIP`` leaves out."""


CLASSES = [f"{Snap.__module__}:{Snap.__qualname__}"]


def snap(ids, shared=False) -> Snap:
    s = Snap()
    s.ids, s.shared, s.lineage = list(ids), shared, ""
    s.rec, s.conv = torch.arange(64, dtype=torch.float32) + len(ids), torch.ones(8)
    s.tail, s.head, s.drafter_rows = None, None, None
    return s


def arena(seed: int) -> Arena:
    g = torch.Generator().manual_seed(seed)
    lat = torch.randint(-2**31, 2**31 - 1, (ROWS, 4), generator=g, dtype=torch.int32)
    pk = torch.randint(-2**15, 2**15 - 1, (ROWS // 4 + 2, 8), generator=g, dtype=torch.int16)
    return Arena(ROWS, [Plane(lat, 1), Plane(pk, 4, 2)])


def store(ar: Arena, name: str, rank: int = 0) -> S.BlockStore:
    cfg = SP.SpillConfig(root=os.path.join(ROOT, name), gib=1.0, min_tokens=1, min_free_gib=0.0)
    os.makedirs(cfg.root, exist_ok=True)
    return S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=rank, device="cpu", classes=CLASSES, quiet=True)


def rows(ar: Arena, base: int, n: int) -> list[torch.Tensor]:
    """The rows tokens [0, n) of an extent at ``base`` occupy, each plane (a pooled-key row still filling included)."""

    return [p.tensor[base // p.div:base // p.div + -(-n // p.div)].clone() for p in ar.planes]


def span(ar: Arena, a: int, b: int) -> list[torch.Tensor]:
    """Every plane's rows of tokens [a, b) (both multiples of 4)."""

    return [p.tensor[a // p.div:b // p.div].clone() for p in ar.planes]


def same(a: list, b: list) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def ids_of(n: int, seed: int = 0) -> list[int]:
    return [int(v) for v in np.random.default_rng(seed).integers(1, 150000, n)]


def until(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.005)
    return True


def load_rows(st: S.BlockStore, ar: Arena, key: str, x, n: int):
    """A read of point ``key`` into extent ``x`` (the store's own reader), then its rows; Nones when the read fails."""

    try:
        small, pt = st.finish_load(st.load_async(key, x))
    except Exception as exc:                        # noqa: BLE001  (a failed read: the checks below say so)
        print(f"     (read of {key} failed: {exc!r})")
        return None, None, None
    return small, pt, rows(ar, x.base, n)


def stored(st: S.BlockStore, ar: Arena, n: int, seed: int, base: int = 0, shared: bool = False):
    """A state of ``n`` tokens kept at ``base`` and written: (its ids, its rows as kept)."""

    ids = ids_of(n, seed)
    x = Pool(ROWS).add(base, align_up(n))
    st.persist(ids, snap(ids), extent=x, lineage=f"s{seed}", shared=shared, skip=M.MultiDecoder._SPILL_SKIP)
    if not st.drain(TIMEOUT / 2):
        print("     (a write did not end)")
    return ids, rows(ar, base, n)


def decoder(ar: Arena, st: S.BlockStore, rank: int = 0) -> M.MultiDecoder:
    """A MultiDecoder with only what its spill-tier paths use: a real Pool over ``ar`` guarded by ``_guard_pool``,
    ``st`` as its disk, one rank's view (no collectives), its ops left in ``outbox``."""

    m = object.__new__(M.MultiDecoder)
    m.pool, m.arena, m.disk, m.rank = Pool(ROWS), ar, st, rank
    m.w = NS(world=1, device="cpu")
    m.g = NS(cache_entries=8)
    m.kept, m.lanes, m.partial, m.loads, m.outbox = [], {}, None, {}, []
    m.next_lid = m.next_kid = 0
    m.broken, m.idle = None, False
    m._flush = lambda: None
    m._guard_pool()
    return m


def gate_reads(st: S.BlockStore, hold) -> threading.Event:
    """Reads of the blocks ``hold(name)`` names wait for the returned event (the disk slow)."""

    gate, orig = threading.Event(), st._read_block

    def gated(h, crc, buf, nb):
        if hold(h):
            gate.wait(TIMEOUT)
        return orig(h, crc, buf, nb)

    st._read_block = gated
    return gate


# -- review 1: the block a state's end cuts is frozen at the keep ---------------------------------------------------
ar = arena(1)
pool = Pool(ROWS)
st = store(ar, "tail")
n = 2 * ALIGN + 6                                   # a tail of 6 tokens: its last pooled-key row holds 2 of 4
x = pool.add(0, align_up(n))
ids = ids_of(n)
kept = rows(ar, x.base, n)
check("tail: a state with a tail is queued", st.persist(ids, snap(ids), extent=x, lineage="a",
                                                        skip=M.MultiDecoder._SPILL_SKIP) == "queued")
g = torch.Generator().manual_seed(9)
for p in ar.planes:                                 # the stream goes on decoding in its extent past the keep
    a = n // p.div
    p.tensor[a:a + 8] = torch.randint(-100, 100, tuple(p.tensor[a:a + 8].shape), generator=g, dtype=p.tensor.dtype)
check("tail: the stream's next pooled-key row is the one the keep ended in", not same(rows(ar, x.base, n), kept))
check("tail: drained", st.drain(TIMEOUT / 2))
y = pool.add(4 * ALIGN, align_up(n))
_, pt, got = load_rows(st, ar, S.point_key(ids), y, n)
check("tail: the stored rows are those of the keep (the filling row as it was)", got is not None and same(got, kept))
check("tail: the tail block is not staged from the pool later", st.stats["staged_blocks"] == 2)

# -- review 2: compaction moves an extent being read without waiting for the read ----------------------------------
n = 3 * ALIGN
ar = arena(2)
st = store(ar, "move")
m = decoder(ar, st)
ids, kept = stored(st, ar, n, 21)
pt = st.index[S.point_key(ids)]
gate = gate_reads(st, lambda h: h != pt.blocks[0])  # the first block comes, the disk holds the rest
below = m.pool.add(0, 2 * ALIGN)                    # an extent below the read's, gone before the compaction
hi, lo, _ = S.key_ints(pt.key)
h = m._start_load(hi, lo, n, 2 * ALIGN, align_up(n), 0)
x = h.x
check("move: the read's first block is in the pool", until(lambda: st.stats["loaded_bytes"] > 0))
m.pool.remove(below)
t = threading.Thread(target=m._compact, daemon=True)
t.start()
t.join(3.0)
check("move: compaction moves the extent being read without waiting for the disk", not t.is_alive() and x.base == 0)
left = (align_up(n), 2 * ALIGN + align_up(n))       # the rows it left (past its new end)
for p in ar.planes:
    p.tensor[left[0] // p.div:left[1] // p.div] = 7
before = span(ar, *left)
gate.set()
t.join(TIMEOUT / 2)
check("move: the read ends", until(lambda: h.lj.done.is_set()))
check("move: the stored prompt is kept after its read", m.complete_load(h) and len(x.kept) == 1)
check("move: its rows are at the extent's new base, as they were kept", same(rows(ar, x.base, n), kept))
check("move: no block of the read landed at the old base", same(span(ar, *left), before))


class Holding:
    """Extent ``x``'s base as the reader reads it to issue a block's copies; the first read hands out the base, then
    waits for ``go`` (a move starting now finds that block's copies being issued)."""

    def __init__(self, x) -> None:
        self.x, self.entered, self.go = x, threading.Event(), threading.Event()

    @property
    def base(self) -> int:
        b = self.x.base
        if not self.entered.is_set():
            self.entered.set()
            self.go.wait(TIMEOUT / 4)
        return b


ar = arena(3)
st = store(ar, "lock")
m = decoder(ar, st)
ids, kept = stored(st, ar, n, 31)
pt = st.index[S.point_key(ids)]
gate = gate_reads(st, lambda h: True)
below = m.pool.add(0, 2 * ALIGN)
hi, lo, _ = S.key_ints(pt.key)
h = m._start_load(hi, lo, n, 2 * ALIGN, align_up(n), 0)
x = h.x
hold = Holding(x)
h.lj.where = hold                                   # (the reader reads the extent's base through it)
gate.set()
check("lock: a block's copies are being issued", until(hold.entered.is_set))
m.pool.remove(below)
t = threading.Thread(target=m._compact, daemon=True)
t.start()
t.join(0.5)
check("lock: the move waits for the copies being issued", t.is_alive())
hold.go.set()
t.join(TIMEOUT / 2)
check("lock: the read ends", until(lambda: h.lj.done.is_set()))
check("lock: the stored prompt is kept after its read", m.complete_load(h) and len(x.kept) == 1)
check("lock: its rows are at the extent's new base, as they were kept", x.base == 0 and same(rows(ar, x.base, n), kept))


# -- review 3, shared-flag 1 and 2: a restore keeps the shared-prefix flag, rank 0's on every rank -----------------
def two_ranks(flags: tuple[bool, bool], name: str):
    """Rank 0 and rank 1 (each its arena, disk and pool), rank 0's ops applied by rank 1's own ``apply``; the same
    state stored on each with its own shared flag (``flags``), then read back through LOAD and LOADED."""

    ms = []
    for rank, flag in enumerate(flags):
        a = arena(40 + rank)
        s = store(a, f"{name}{rank}", rank)
        ids, _ = stored(s, a, n, 41, shared=flag)
        ms.append(decoder(a, s, rank))
    m0, m1 = ms
    sent = []

    def flush():
        msg, m0.outbox = m0.outbox, []
        sent.extend(m1.parse(msg))
        m1.apply(msg)

    m0._flush = flush
    hi, lo, _ = S.key_ints(S.point_key(ids))
    m0._emit(M.LOAD, [hi, lo, n, 0, align_up(n), 0])         # as ``prestage`` sends it
    m0._flush()
    h = m0._start_load(hi, lo, n, 0, align_up(n), 0)
    until(lambda: h.lj.done.is_set() and m1.loads[0].lj.done.is_set())
    ok = m0.complete_load(h)
    return ok, m0, m1, [p for op, p in sent if op == M.LOADED]


ok, m0, m1, loaded = two_ranks((True, False), "shared")
check("shared: a point rank 0 holds as shared is restored on both ranks",
      ok and len(m0.kept) == 1 and len(m1.kept) == 1)
check("shared: LOADED carries rank 0's flag", loaded == [[0, 1]])
check("shared: rank 0 restores it as a shared-prefix state", ok and m0.kept[0].shared is True)
check("shared: rank 1 too, though its own point says otherwise", ok and m1.kept[0].shared is True)
ok, m0, m1, loaded = two_ranks((False, True), "own")
check("shared: a point rank 0 holds as not shared is restored unshared on both ranks",
      ok and not m0.kept[0].shared and not m1.kept[0].shared and loaded == [[0, 0]])


# -- shared-flag 3: the same ids kept again as a shared prefix make the stored point shared, on disk too ------------
def json_of(st: S.BlockStore, key: str) -> dict:
    try:
        with open(st._point_path(key, ".json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def keep(st: S.BlockStore, ids: list, shared: bool, lineage: str) -> str:
    x = Pool(ROWS).add(0, align_up(len(ids)))
    return st.persist(ids, snap(ids, shared), extent=x, lineage=lineage, shared=shared,
                      skip=M.MultiDecoder._SPILL_SKIP)


log: list = []                                     # (what, path) of the writer's puts and folder fsyncs
_put, _fsync_dir, hooks = S._put, S._fsync_dir, {}


def put_logged(path, data, owner, direct=False):
    log.append(("put", os.path.basename(path)))
    for suffix, fn in list(hooks.items()):          # (before the file is written)
        if path.endswith(suffix):
            hooks.pop(suffix)
            fn()
    return _put(path, data, owner, direct)


def fsync_dir_logged(path):
    log.append(("fsync_dir", os.path.basename(path)))
    return _fsync_dir(path)


S._put, S._fsync_dir = put_logged, fsync_dir_logged
n2 = ALIGN + 100
ar = arena(5)
st = store(ar, "upgrade")
ids, _ = stored(st, ar, n2, 51)                     # written as a conversation's own turn
key = S.point_key(ids)
log.clear()
check("upgrade: the same ids kept again as shared are not written again", keep(st, ids, True, "s51") == "stored")
check("upgrade: the point is shared now", st.index[key].shared is True)
check("upgrade: its json says so", until(lambda: json_of(st, key).get("shared") is True, 5))
check("upgrade: the json goes through a temporary name, fsynced, renamed, then the folder is fsynced",
      until(lambda: log[-2:] == [("put", f"{key}.json"), ("fsync_dir", "points")], 5))
check("upgrade: a restart reads it shared", store(ar, "upgrade").index.get(key, NS(shared=None)).shared is True)

ids = ids_of(n2, 52)
key = S.point_key(ids)
check("upgrade: a state waiting to be written", keep(st, ids, False, "s52") == "queued")
check("upgrade: kept again as shared meanwhile", keep(st, ids, True, "s52") == "stored")
st.drain(TIMEOUT / 4)
check("upgrade: it is written as shared", st.index[key].shared is True
      and until(lambda: json_of(st, key).get("shared") is True, 5))

ids = ids_of(n2, 53)
key = S.point_key(ids)
hooks[f"{key}.ids"] = lambda: keep(st, ids, True, "s53")   # kept again as shared while its files are written
check("upgrade: a state written as its own", keep(st, ids, False, "s53") == "queued")
st.drain(TIMEOUT / 4)
check("upgrade: kept again as shared during its write, its json is rewritten shared",
      st.index[key].shared is True and until(lambda: json_of(st, key).get("shared") is True, 5))

ids, _ = stored(st, ar, n2, 54)
key = S.point_key(ids)


def drop():
    with st.lock:
        st._remove(key, "dropped")


hooks[f"{key}.json"] = drop                         # dropped while its json is rewritten
keep(st, ids, True, "s54")
check("upgrade: a point dropped during the rewrite", until(lambda: not hooks, 5) and key not in st.index)
time.sleep(0.2)
check("upgrade: ... loses the json the rewrite put back", not os.path.exists(st._point_path(key, ".json")))
S._put, S._fsync_dir = _put, _fsync_dir

print("all passed" if not fails else f"FAILED: {fails}", flush=True)
os._exit(1 if fails else 0)
