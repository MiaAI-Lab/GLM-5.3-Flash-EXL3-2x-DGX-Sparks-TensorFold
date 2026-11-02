#!/usr/bin/env python3
"""Checks for patch 0128 (the spill tier's block store, tensorfold/cuda/spill_blocks.py, and its hooks in multi.py),
CPU only, run inside the image that scripts/prepare.sh built (no GPU, no model, no server); its files go to a scratch
folder under $TMPDIR:

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/spill_blocks_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

A real BlockStore over a real Arena on the host (a latents plane, div 1, and a pooled-key plane, div 4 with 2 pad
rows), with a stand-in small state, and MultiDecoder's own load and move paths on a real Pool:
- the block a kept state's end cuts holds the rows of the keep, not what the stream writes after it (its pooled-key
  row still filling at the keep);
- a read only reads: its reader copies no row (no CUDA work off the engine's thread); the engine copies the blocks
  read so far between rounds (rank 0's step, LOADPART to rank 1 copying as far, the rest at LOADED), on its own thread;
- compaction moving an extent a stored prompt is being read into goes on without waiting for the disk, and the read's
  later blocks land at the new base and none at the old one;
- a restored state keeps its point's shared-prefix flag, rank 0's on every rank (LOADED carries it);
- the same ids kept again as a shared prefix make the stored point shared, its json rewritten through a temporary
  name and the folder fsynced (a restart reads it so), also when they were still waiting or being written; a point
  dropped during the rewrite does not get its json back, and a drain (a clean stop) waits for the rewrite;
- a read that fails on a missing block file drops that name and the points naming it, not the points whose blocks
  are links to the same bytes (they still read back); one that fails on damaged bytes drops every name of them;
- every op in multi's table has a sender, and /metrics carries the prefix wait's counters;
- each of a point's files (.ids, .st, .json) has its pages dropped from the page cache after its fsync;
- a decode round settles the reads once its verify pass is queued (every slot the last step copied out back to the
  reader, after the newest copy only), so beside a decoding stream a read slot turns every round;
- a staged block records its write slot's own event (made at start), and the writer waits on that one;
- LOADED on a rank with more blocks left than read slots returns: the wait for a block frees the slots the step's
  own copies hold (a rank 1 whose rank 0 failed early hung there for ever);
- the read-ahead (16 blocks unless TF_GLM_SPILL_READ_BLOCKS says otherwise; on two Sparks the preallocated read
  buffers hold the first ten and 9 fits them, so the pool pays nothing for 9) keeps spill.preallocate's read buffers
  and pins only the rest before the pool is sized, in power-of-two pieces of at most spill.CHUNK (torch's pinned
  allocator rounds an allocation up to one), the bytes it returns (the pool's cost) those pieces only; ring carves
  at least that many stored blocks out of them and pins none later;
- the room, files first: each eviction step deletes the files it should (an old turn its point files only, a fork
  and a whole conversation their own blocks too, never a block another point names) and goes in the order old turns,
  abandoned forks, whole conversations least recently used, shared prefixes last; a point being read is never taken,
  nor a block the state being written names;
- old turns trimmed as newer ones are written (SPILL_TURNS: the newest on the path stay, forks off it and shared
  prefixes untouched, a turn being read kept; 0 keeps every turn), the quota (a conversation over its share trims its
  own old turns and forks, no other's), the free-disk floor (a disk full of other things refuses without evicting;
  short of the floor, room made in the eviction order);
- a restart: temporary names, points without their json, with damaged ids or a missing block, blocks no point names
  and foreign files go, the complete point reads back; older builds lose this rank's folder (the newest other stays);
  a failed write leaves no temporary file;
- temporary files are opened O_EXCL | O_NOFOLLOW (a symlink planted at the name is not written through, a leftover
  file replaced); one engine a folder and rank (``spill.hold``: a second engine, of any build, the one-stream store
  too, gets ``Busy`` before it prunes anything, the engine then runs with the tier off; a kill -9 lets it go);
- release has its own cap (TF_GLM_SPILL_RELEASE_MIB, by default an eighth of the memory reserve): past it a state's
  rows are not copied and it is not stored; the host copies (and the block a keep cuts) are let go once written;
- the prefix wait is off unless TF_GLM_PREFIX_WAIT=1; what it waits for and what not, when it lets go, and the
  scheduler (the spill tier off) holding the request apart while the queue goes on.
Exit code 1 when a check fails; tools/spill_blocks_mutants.sh runs it on faithful mutants of each fix.
"""
import inspect
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

from tensorfold.cuda import health as H
from tensorfold.cuda import spill as SP
from tensorfold.cuda import spill_blocks as S
from tensorfold.cuda.scheduler import Waiting
from tensorfold.families.glm5_next.cuda import multi as M
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up
from tensorfold.server import metrics as MT

ROWS = 8 * ALIGN
# TEST_TIMEOUT_S: the whole run's watchdog (a hang prints a FAIL and exits), and the longest single wait is half of it.
# 900 s by default: one run takes ~15-60 s, and a slower CPU went past the old 60 s (tools/spill_blocks_mutants.sh
# sets 120 for its runs).
TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "900"))
fails: list[str] = []
ROOT = tempfile.mkdtemp(prefix="spill-blocks-check-")


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


def _hung() -> None:
    print(f"FAIL timed out after {TIMEOUT:g} s", flush=True)
    os._exit(1)


_watchdog = threading.Timer(TIMEOUT, _hung)
_watchdog.daemon = True                               # (a crash ends the run at once)
_watchdog.start()


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


OPEN: dict = {}                                      # (folder, rank) -> the store holding it (``spill.hold``)


def store(ar: Arena, name: str, rank: int = 0, gib: float = 1.0, floor: float = 0.0, **kw) -> S.BlockStore:
    """A store on ``ROOT/name``; one already open there for ``rank`` is closed first (the engine before it ended: a
    restart)."""

    cfg = SP.SpillConfig(root=os.path.join(ROOT, name), gib=gib, min_tokens=1, min_free_gib=floor)
    os.makedirs(cfg.root, exist_ok=True)
    old = OPEN.pop((cfg.root, rank), None)
    if old is not None:
        old.close()
    st = S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=rank, device="cpu", classes=CLASSES, quiet=True, **kw)
    OPEN[(cfg.root, rank)] = st
    return st


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
    m.load_turn = 0
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
check("move: the read's first block is in the pool", until(lambda: (m.step_loads(), st.stats["loaded_bytes"] > 0)[1]))
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


# -- the reader reads, the engine copies (between rounds, on its thread; LOADPART takes rank 1 as far) ------------------
ar = arena(3)
st = store(ar, "engine")
m = decoder(ar, st)
ids, kept = stored(st, ar, n, 31)
pt = st.index[S.point_key(ids)]
hi, lo, _ = S.key_ints(pt.key)
h = m._start_load(hi, lo, n, 2 * ALIGN, align_up(n), 0)
x = h.x
for p in ar.planes:
    p.tensor[x.base // p.div:(x.base + align_up(n)) // p.div] = 5
before = rows(ar, x.base, n)
check("engine: the reader reads every block and copies none", until(lambda: h.lj.done.is_set())
      and len(h.lj.queued) == h.lj.total and st.stats["loaded_bytes"] == 0 and same(rows(ar, x.base, n), before))
threads, copy_in = set(), st._copy_in
st._copy_in = lambda *a: (threads.add(threading.get_ident()), copy_in(*a))[1]
m.step_loads()
check("engine: a step copies the blocks read, on the engine's own thread",
      h.lj.next == h.lj.total and threads == {threading.get_ident()})
check("engine: LOADPART tells rank 1 how far", [p for op, p in m.parse(m.outbox) if op == M.LOADPART] == [[0, h.lj.total]])
check("engine: the stored prompt is kept after its read", m.complete_load(h) and len(x.kept) == 1)
check("engine: its rows as they were kept", same(rows(ar, x.base, n), kept))
ar = arena(4)
st = store(ar, "engine1", 1)
m1 = decoder(ar, st, 1)
ids, kept = stored(st, ar, n, 31)
hi, lo, _ = S.key_ints(S.point_key(ids))
m1.apply([M.LOAD, 6, hi, lo, n, 2 * ALIGN, align_up(n), 0])
h = m1.loads[0]
m1.apply([M.LOADPART, 2, 0, 2])
check("engine: rank 1 copies as far as LOADPART says (waiting for its reads), no further", h.lj.next == 2)
m1.apply([M.LOADED, 2, 0, 0])
check("engine: rank 1 copies the rest at LOADED, its rows as kept", len(m1.kept) == 1 and same(rows(ar, h.x.base, n), kept))


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

ids, _ = stored(st, ar, n2, 55)
key = S.point_key(ids)
hooks[f"{key}.json"] = lambda: time.sleep(0.3)     # a slow rewrite
keep(st, ids, True, "s55")
check("upgrade: a drain (a clean stop) waits for the rewrite", st.drain(TIMEOUT / 4)
      and json_of(st, key).get("shared") is True)
S._put, S._fsync_dir = _put, _fsync_dir


# -- review 4: a missing block file drops that name, a damaged one every name linked to its bytes -----------------
def linked(name: str):
    """A (2 blocks), B (other ids, the same rows: its blocks linked to A's) and C (A's ids and one block more, A's
    lineage: it names A's blocks), written; a decoder over the store."""

    a = arena(60)
    s = store(a, name)
    ia, ib = ids_of(2 * ALIGN, 61), ids_of(2 * ALIGN, 62)
    ic = ia + ids_of(ALIGN, 63)
    kept = rows(a, 0, 2 * ALIGN)
    for ids, lineage in ((ia, "a"), (ib, "b"), (ic, "a")):
        keep(s, ids, False, lineage)
        s.drain(TIMEOUT / 4)
    return a, s, decoder(a, s), [S.point_key(i) for i in (ia, ib, ic)], kept


def read_fails(m: M.MultiDecoder, key: str, n: int) -> bool:
    """A read of ``key`` through LOAD / LOADED (``_finish_load`` forgets it): True when it failed."""

    hi, lo, _ = S.key_ints(key)
    h = m._start_load(hi, lo, n, 4 * ALIGN, align_up(n), m.next_lid)
    until(lambda: m.load_ready(h))
    return not m.complete_load(h)


ar, st, m, (ka, kb, kc), kept = linked("lost")
pa, pb = st.index[ka], st.index[kb]
check("forget: B's blocks are links to A's (the same bytes)", st.stats["dedup_blocks"] == 2
      and [st.have[b][1] for b in pa.blocks] == [st.have[b][1] for b in pb.blocks])
os.remove(st._block_path(pa.blocks[0]))              # one name of the bytes gone
check("forget: the read of A fails", read_fails(m, ka, 2 * ALIGN))
check("forget: A goes, and C, which names the missing block", ka not in st.index and kc not in st.index)
check("forget: B, linked to the missing name's bytes, stays with its files",
      kb in st.index and all(os.path.exists(st._block_path(b)) for b in pb.blocks))
y = Pool(ROWS).add(4 * ALIGN, 2 * ALIGN)
_, _, got = load_rows(st, ar, kb, y, 2 * ALIGN)
check("forget: B still reads back whole", got is not None and same(got, kept))

ar, st, m, (ka, kb, kc), kept = linked("damaged")
pa, pb = st.index[ka], st.index[kb]
with open(st._block_path(pa.blocks[0]), "r+b") as fh:   # the bytes of every name linked to them
    fh.write(b"\xff" * 64)
check("forget: the read of a damaged A fails", read_fails(m, ka, 2 * ALIGN))
check("forget: damaged bytes drop every point naming any of their names (A, B, C)",
      ka not in st.index and kb not in st.index and kc not in st.index)
check("forget: ... and those names' files", not any(os.path.exists(st._block_path(b)) for b in pb.blocks[:1]))


# -- review 5: every op has a sender; the prefix wait's counters reach /metrics ------------------------------------
src = inspect.getsource(M)
unsent = [name for name in M.OPS.values() if f"self._emit({name}," not in src]
check("ops: every op in the table has a sender", not unsent)
of = H.of
H.of = lambda app: NS(snapshot=lambda app: {"prefix_waits": 6, "prefix_wait_tokens": 98688})
lines: list = []
MT._health(lines, object())
H.of = of
check("metrics: the prefix wait's counters", f"{MT.HEALTH}prefix_waits_total 6" in lines
      and f"{MT.HEALTH}prefix_wait_tokens_total 98688" in lines
      and f"# TYPE {MT.HEALTH}prefix_waits_total counter" in lines)


# -- a point's small files leave no pages behind: each one's pages are dropped after its fsync ---------------------
events: list = []                                   # (what, file name) in order, on the writer thread


def fname(fd: int) -> str:
    try:
        return os.path.basename(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        return "?"


fsync, drop_cache = os.fsync, S._drop_cache
os.fsync = lambda fd: (events.append(("fsync", fname(fd))), fsync(fd))[1]
S._drop_cache = lambda fd: (events.append(("drop", fname(fd))), drop_cache(fd))[1]
ar = arena(7)
st = store(ar, "pages")
ids, _ = stored(st, ar, ALIGN + 100, 71)
os.fsync, S._drop_cache = fsync, drop_cache
key = S.point_key(ids)
for suffix in (".ids", ".st", ".json"):
    mine = [what for what, f in events if f == f".{key}{suffix}"]
    check(f"pages: {suffix} dropped from the page cache after its fsync", mine == ["fsync", "drop"])

# -- a read slot turns every round: beside a decoding stream the step between rounds frees only the slots the step
# before copied out, so a slot turned every second round (profiled on GPU: the reader waited for slots 80% of a 226k
# read, 14-18 blocks copied a round through 32 slots).
class Stop(Exception):
    pass


def round_order(loading: bool) -> list:
    """What a decode round does, in order, up to its sampling: its verify pass queued, the reads settled, sampled."""

    a = arena(100)
    s = store(a, f"round{int(loading)}")
    m = decoder(a, s)
    order: list = []
    lane = NS(copies=None, dflash=False, depth=0, s=NS(out=[7], sampling=None, stop_eos=False), constraint=None,
              st=NS(pos=0), window=None)
    m.profile, m.tune, m.drafts, m.window, m.verify = None, NS(depth="", sampler=""), None, 64, NS()

    def forward(verify, segments):
        order.append("forward")
        return NS(logits=[torch.zeros(1, 4)])

    def sample(w, parts):
        order.append("sample")
        raise Stop

    fwd, smp = M.forward_streams, M.sample_streams
    M.forward_streams, M.sample_streams = forward, sample
    s.settle = lambda: order.append("settle")
    if loading:
        m.loads[0] = NS()
    try:
        m._round([lane])
    except Stop:
        pass
    finally:
        M.forward_streams, M.sample_streams = fwd, smp
    return order


check("round: a decode round settles the reads once its verify pass is queued, before its readback",
      round_order(True) == ["forward", "settle", "sample"])
check("round: with no read under way a round settles nothing", round_order(False) == ["forward", "sample"])


class StreamEv:
    """A copy's event on one stream: waiting for it means every event recorded before it is done too."""

    def __init__(self, line: list) -> None:
        self.line, self.done, self.synced = line, False, 0
        line.append(self)

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synced += 1
        for e in self.line[:self.line.index(self) + 1]:
            e.done = True


def drained(q) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return sorted(out)


st = store(arena(104), "settle")
try:
    st.settle()
    early = True
except Exception as exc:                            # noqa: BLE001
    print(f"     (settle before the slots: {exc!r})")
    early = False
check("settle: before the read slots exist it does nothing", early)
st.ring()
drained(st.free_reads)
line: list = []
evs = [StreamEv(line) for _ in range(3)]
st.read_events = list(zip((4, 5, 6), evs))
st._recycle()
check("settle: a step's recycling between rounds waits for no copy", not drained(st.free_reads)
      and all(e.synced == 0 for e in evs) and len(st.read_events) == 3)
st.settle()
check("settle: it waits for the newest copy only, and every slot copied out comes back",
      drained(st.free_reads) == [4, 5, 6] and not st.read_events and [e.synced for e in evs] == [0, 0, 1])


class Rec:
    """A slot's event: counts its records and the writer's waits on it."""

    def __init__(self) -> None:
        self.records = self.syncs = 0

    def record(self, stream=None) -> None:
        self.records += 1

    def synchronize(self) -> None:
        self.syncs += 1


ar = arena(102)
st = store(ar, "wevents")
st.ring()
st.slot_events = [Rec() for _ in st.slots]          # (made at start on a GPU; none on the host)
ids, kept = stored(st, ar, 3 * ALIGN - 6, 103)      # two blocks staged by the pumps, the tail copied at keep
staged = st.stats["staged_blocks"]
check("write: a staged block records its slot's own event, made at start, and the writer waits on that one (no event "
      "made or dropped a block)", staged == 2 and sum(r.records for r in st.slot_events) == staged
      and sum(r.syncs for r in st.slot_events) == staged)
_, _, back = load_rows(st, ar, S.point_key(ids), Pool(ROWS).add(0, align_up(3 * ALIGN - 6)), 3 * ALIGN - 6)
check("write: its rows read back as kept", back is not None and same(back, kept))

# -- review (wojo#2): the reads in flight share one TF_GLM_SPILL_LOAD_MS pause between rounds, each first in turn ----
class SlowDisk:
    """A disk whose every step copies for as long as it may, 30 ms at most."""

    def __init__(self) -> None:
        self.calls: list = []

    def step(self, lj, upto=None, budget=None):
        self.calls.append(lj.name)
        time.sleep(min(budget, 0.03))
        return 1


ar = arena(9)
m = decoder(ar, store(ar, "budget"))
m.disk, m.load_budget = SlowDisk(), 0.04
m.loads = {i: NS(done=False, err=None, lid=i, lj=NS(error=None, next=0, name=i)) for i in range(3)}
t = time.perf_counter()
m.step_loads()
took = time.perf_counter() - t
check(f"budget: three reads in flight share one 40 ms pause (took {took * 1000:.0f} ms)", took < 0.06)
first = []
for _ in range(3):
    m.disk.calls.clear()
    m.step_loads()
    first.append(m.disk.calls[0])
check(f"budget: each pause starts with the next read ({first})", sorted(first) == [0, 1, 2])

# -- a restore beside decoding streams ends only once its budgeted copies are done (LOADED copies nothing then) ---------
ar = arena(10)
m = decoder(ar, store(ar, "ready"))
done = threading.Event()
done.set()
h = NS(done=False, err=None, lj=NS(done=done, error=None, next=3, total=10))
m.lanes = {1: NS(decoding=True, paused=False, s=NS(done=False))}
check("ready: a read done with blocks left to copy waits while a stream decodes", not m.load_ready(h))
h.lj.next = 10
check("ready: every block copied, the restore ends", m.load_ready(h))
h.lj.next = 3
m.lanes = {}
check("ready: nothing decoding, the rest is copied at once", m.load_ready(h))

# -- LOADED on a rank that copied little: the wait for a block frees the read slots the step's own copies hold ---------
# Rank 0's read failed early (LOADED came with no LOADPART before it) and rank 1 has more blocks left than read slots:
# its step copies a slot's worth, and the reader needs those slots back to read the next block. They went back only at
# the top of the next step, so the wait for that block starved the reader for ever (the rank-1 watchdog then exits).
saved, ROWS = ROWS, 22 * ALIGN
os.environ.update(TF_GLM_SPILL_READ_BLOCKS="8", TF_GLM_SPILL_STAGE_MIB="1")
try:
    ar = arena(500)
    st = store(ar, "hang", 1)
    nt = 20 * ALIGN
    ids, kept = stored(st, ar, nt, 501)
    m1 = decoder(ar, st, 1)
finally:
    ROWS = saved
    for k in ("TF_GLM_SPILL_READ_BLOCKS", "TF_GLM_SPILL_STAGE_MIB"):
        os.environ.pop(k, None)
hi, lo, _ = S.key_ints(S.point_key(ids))
m1.apply([M.LOAD, 6, hi, lo, nt, 0, align_up(nt), 0])
h = m1.loads[0]
check("hang: 20 blocks through 8 read slots", len(st.read_slots) == 8 and h.lj.total == 20)
until(lambda: len(h.lj.queued) >= 8)
t = threading.Thread(target=lambda: m1.apply([M.LOADED, 2, 0, 0]), daemon=True)
t.start()
t.join(8.0)
check("hang: LOADED on a rank that copied nothing before returns (the wait frees the slots its copies hold)",
      not t.is_alive())
check("hang: the state is kept, its rows as stored", not t.is_alive() and len(m1.kept) == 1
      and same(rows(ar, h.x.base, nt), kept))

# The same with the read slots carved as pin_read_ahead leaves them: a preallocated buffer that holds two and a half
# blocks, the rest in one power-of-two piece, so the ring holds more slots than the 7 asked and some sit in each buffer.
pinned, per = dict(SP._PINNED), S._up(st.block_bytes)
saved, ROWS = ROWS, 22 * ALIGN
os.environ.update(TF_GLM_SPILL_READ_BLOCKS="7", TF_GLM_SPILL_STAGE_MIB="1")
try:
    SP._PINNED.clear()
    SP._PINNED["read"] = [torch.zeros(2 * per + per // 2, dtype=torch.uint8)]
    took = S.pin_read_ahead(per)
    ar = arena(510)
    st = store(ar, "hangC", 1)
    ids, kept = stored(st, ar, nt, 511)
    m1 = decoder(ar, st, 1)
finally:
    ROWS = saved
    for k in ("TF_GLM_SPILL_READ_BLOCKS", "TF_GLM_SPILL_STAGE_MIB"):
        os.environ.pop(k, None)
hi, lo, _ = S.key_ints(S.point_key(ids))
m1.apply([M.LOAD, 6, hi, lo, nt, 0, align_up(nt), 0])
h = m1.loads[0]
check("hang, carved slots: 20 blocks through more than the 7 slots asked, carved from the preallocated buffer and one "
      "piece", took > 0 and 7 < len(st.read_slots) < 20 and h.lj.total == 20)
until(lambda: len(h.lj.queued) >= len(st.read_slots))
t = threading.Thread(target=lambda: m1.apply([M.LOADED, 2, 0, 0]), daemon=True)
t.start()
t.join(8.0)
check("hang, carved slots: LOADED on a rank that copied nothing before returns", not t.is_alive())
check("hang, carved slots: the state is kept, its rows as stored", not t.is_alive() and len(m1.kept) == 1
      and same(rows(ar, h.x.base, nt), kept))
SP._PINNED.clear()
SP._PINNED.update(pinned)

saved, chunk, env = dict(SP._PINNED), SP.CHUNK, os.environ.pop("TF_GLM_SPILL_READ_BLOCKS", None)
check("read-ahead: 16 blocks by default", S.read_blocks() == 16)
SP._PINNED.clear()
SP._PINNED["read"] = [torch.empty(SP.CHUNK, dtype=torch.uint8) for _ in range(2)]   # (preallocate's, 2 x 64 MiB)
check("read-ahead: the default at two Sparks' block size keeps preallocate's read buffers and pins 64 + 32 MiB more, "
      "what the pool pays (the engine's bytes a pool block there: 13.6-14.0 MB, a stored block 12.7 MB)",
      S.pin_read_ahead(14_029_224) == 96 << 20 and len(SP._PINNED["read"]) == 4)
os.environ["TF_GLM_SPILL_READ_BLOCKS"] = "9"
SP._PINNED["read"] = [torch.empty(SP.CHUNK, dtype=torch.uint8) for _ in range(2)]
check("read-ahead: 9 blocks fit preallocate's read buffers on two Sparks, so the pool pays nothing for them",
      S.pin_read_ahead(14_029_224) == 0 and len(SP._PINNED["read"]) == 2)
os.environ["TF_GLM_SPILL_READ_BLOCKS"] = "21"
st = store(arena(105), "readahead")
pe = 3 * S._up(st.block_bytes)       # (the engine's bytes a pool block, more than a stored block's, as on the Sparks)
SP.CHUNK = 1 << (4 * pe - 1).bit_length()                  # (small pieces, a power of two as 64 MiB is)
pre = [torch.zeros(SP.CHUNK, dtype=torch.uint8) for _ in range(2)]   # (preallocate's two read buffers)
SP._PINNED.clear()
SP._PINNED["read"] = list(pre)
took = S.pin_read_ahead(pe)
new = SP._PINNED["read"][2:]
check("read-ahead: preallocate's read buffers kept, only the rest pinned before the pool, and only that charged",
      len(SP._PINNED["read"]) > 2 and all(a is b for a, b in zip(SP._PINNED["read"], pre))
      and took == sum(b.numel() for b in new)
      and took + 2 * SP.CHUNK >= 21 * pe > took + 2 * SP.CHUNK - max(b.numel() for b in new))
check("read-ahead: the rest in power-of-two pieces of at most spill.CHUNK",
      bool(new) and all(b.numel() & (b.numel() - 1) == 0 and b.numel() <= SP.CHUNK for b in new))
st.ring()
check("read-ahead: ring carves at least the blocks asked from them and pins none later",
      len(st.read_slots) >= 21 and {x.untyped_storage().data_ptr() for x in st.read_slots}
      == {b.untyped_storage().data_ptr() for b in SP._PINNED["read"]})
SP._PINNED.clear()
SP._PINNED.update(saved)
SP.CHUNK = chunk
os.environ.pop("TF_GLM_SPILL_READ_BLOCKS", None)
if env is not None:
    os.environ["TF_GLM_SPILL_READ_BLOCKS"] = env


# == the store's room, files and settings (review 2026-10-24): files deleted first, then the order, quota, turns, floor
os.environ["TF_GLM_SPILL_STAGE_MIB"] = "2"            # (a small write ring a store: these make many stores)
def fill(ar: Arena, base: int, n: int, seed: int) -> None:
    """Rows of their own for tokens [base, base + n) (no two states' blocks the same bytes: no dedup links)."""

    g = torch.Generator().manual_seed(1000 + seed)
    for p in ar.planes:
        a, b = base // p.div, (base + align_up(n)) // p.div
        p.tensor[a:b] = torch.randint(-2**15, 2**15 - 1, tuple(p.tensor[a:b].shape), generator=g,
                                      dtype=torch.int32).to(p.tensor.dtype)


SEEDS = iter(range(10_000))
BANK: dict = {}                                        # a block's name -> its rows (one computation: the same rows)


def put(st: S.BlockStore, ar: Arena, ids: list, lineage: str, shared: bool = False, used: float | None = None):
    """A state of ``ids`` kept at base 0 and written: rows of their own for blocks no state named before, the same
    rows for a block named before (what a later turn or a fork computes there); its key, its rows as kept, what
    persist said."""

    fill(ar, 0, len(ids), next(SEEDS))
    for s0, e0, h in S.block_hashes(np.asarray(ids, dtype=np.int64), B, lineage):
        for p, t in zip(ar.planes, BANK.setdefault(h, span(ar, s0, e0))):
            p.tensor[s0 // p.div:e0 // p.div] = t
    x = Pool(ROWS).add(0, align_up(len(ids)))
    kept = rows(ar, 0, len(ids))
    r = st.persist(ids, snap(ids, shared), extent=x, lineage=lineage, shared=shared, skip=M.MultiDecoder._SPILL_SKIP)
    st.drain(TIMEOUT / 4)
    key = S.point_key(ids)
    if used is not None and key in st.index:
        st.index[key].used = used
    return key, kept, r


def files_of(st: S.BlockStore, key: str) -> list[str]:
    return [sfx for sfx in (".ids", ".st", ".json") if os.path.exists(st._point_path(key, sfx))]


def block_files(st: S.BlockStore) -> set[str]:
    return {f[:-4] for f in os.listdir(st.blocks_dir) if f.endswith(".bin")}


def reads_back(st: S.BlockStore, ar: Arena, key: str, ids: list, kept) -> bool:
    st.ring()                                           # (the engine makes it at start)
    fill(ar, 0, len(ids), 9999)                         # (other rows there first)
    y = Pool(ROWS).add(0, align_up(len(ids)))
    _, _, got = load_rows(st, ar, key, y, len(ids))
    return got is not None and same(got, kept)


B = ALIGN
SYS = ids_of(B, 500)                                   # a shared system prompt, one block
LA, LB = S.fork("sys", "a", B), S.fork("sys", "b", B)  # two conversations that resumed from it
TA = [SYS + ids_of(k * B, 501) for k in (1, 2, 3)]     # conversation A: three turns, one block more each
FA = TA[1] + ids_of(B, 502)                            # a fork of A's second turn (an edited last message)
TB = [SYS + ids_of(k * B, 503) for k in (1, 2)]        # conversation B: two turns


def tree(name: str, **kw):
    """S (shared), A1-A3, A2f (a fork off A2), B1, B2 written, each its own rows; used: S, B1, B2, A1, A2, A2f, A3."""

    ar = arena(200)
    st = store(ar, name, **kw)
    k = {}
    k["S"], _, _ = put(st, ar, SYS, "sys", shared=True, used=1.0)
    k["B1"], _, _ = put(st, ar, TB[0], LB, used=2.0)
    k["B2"], _, _ = put(st, ar, TB[1], LB, used=3.0)
    k["A1"], _, _ = put(st, ar, TA[0], LA, used=4.0)
    k["A2"], _, _ = put(st, ar, TA[1], LA, used=5.0)
    k["A2f"], _, _ = put(st, ar, FA, S.fork(LA, "f", 3 * B), used=6.0)
    k["A3"], _, _ = put(st, ar, TA[2], LA, used=7.0)
    return ar, st, k


# -- eviction: one victim at a time, each one's files gone, the blocks another point names kept ---------------------
ar, st, k = tree("order", turns=0)
name = {v: n for n, v in k.items()}
check("room: seven points written, nothing trimmed (turns 0)", len(st.index) == 7 and st.stats["trimmed_turns"] == 0)
check("room: the victims in order: old turns (least recently used first), the abandoned fork, whole conversations "
      "(least recently used first), the shared prefix last",
      [(name[key], why) for key, why in st._victims()] ==
      [("B1", "pruned_turns"), ("A1", "pruned_turns"), ("A2", "pruned_turns"), ("A2f", "pruned_forks"),
       ("B2", "pruned_conversations"), ("A3", "pruned_conversations"), ("S", "pruned_shared")])
blk = {n: set(st.index[key].blocks) for n, key in k.items()}
gone: list = []
for step in range(7):
    before = set(st.index)
    st._make_room(1)                                   # (any shortfall: the next victim alone goes)
    gone += [name[key] for key in before - set(st.index)]
check("room: _make_room takes one at a time; once A1 and A2 are gone the fork is a conversation of its own (B2, "
      "least recently used, goes before it)", gone == ["B1", "A1", "A2", "B2", "A2f", "A3", "S"])
ar, st, k = tree("files", turns=0)
st._make_room(1)
check("files: an old turn's files go (.ids, .st, .json)", not files_of(st, k["B1"]))
check("files: ... its blocks stay (the newer turn names them all)", blk["B1"] <= block_files(st))
st._make_room(1)
st._make_room(1)
check("files: two more old turns' files go, no block", not files_of(st, k["A1"]) and not files_of(st, k["A2"])
      and blk["A2"] <= block_files(st))
st._make_room(1)
check("files: a whole conversation's files and own blocks go", not files_of(st, k["B2"])
      and not (blk["B2"] - blk["S"]) & block_files(st))
check("files: ... the shared prefix's block stays", blk["S"] <= block_files(st))
st._make_room(1)
check("files: the fork goes with its own block", not (blk["A2f"] - blk["A2"]) & block_files(st)
      and not files_of(st, k["A2f"]))
check("files: ... the blocks it shared with the newest turn stay", blk["A2"] <= block_files(st)
      and len(files_of(st, k["A3"])) == 3)
check("files: the counters say why", (st.stats["pruned_turns"], st.stats["pruned_conversations"]) == (3, 2))
check("files: stored bytes match the blocks left on disk",
      st.block_bytes_on_disk == sum(os.stat(st._block_path(b)).st_size for b in block_files(st)))

# -- a point being read is never evicted; making room never takes a block a state being written names ------------
ar, st, k = tree("reading", turns=0)
st.reading.add(k["B1"])
st._make_room(st.stored())                             # (more than everything)
check("reading: a point being read stays while everything else goes", set(st.index) == {k["B1"]}
      and len(files_of(st, k["B1"])) == 3 and blk["B1"] <= block_files(st))
st.reading.clear()
ar = arena(201)
st = store(ar, "pinned", gib=1.0, turns=0)
ka, kept_a, _ = put(st, ar, TA[2], LA, used=1.0)        # conversation A's state (older)
kb, _, _ = put(st, ar, TB[1], LB, used=2.0)             # conversation B's (newer)
st.cap = st.stored() + st.block_bytes                   # room for one block more, not for C's block and small part
ids_c = TA[2] + ids_of(B, 504)                          # C extends A's state (its lineage: names A's four blocks)
wrote = st.stats["written_blocks"]
kc, kept_c, r = put(st, ar, ids_c, LA)
check("pinned: C is written once room is made", r == "queued" and kc in st.index)
check("pinned: A's point went first (least recently used conversation)", ka not in st.index)
check("pinned: ... but not the blocks C names", all(b in block_files(st) for b in st.index[kc].blocks))
check("pinned: C wrote only its new block", st.stats["written_blocks"] == wrote + 1)
check("pinned: C reads back whole", reads_back(st, ar, kc, ids_c, kept_c))

# -- old turns trimmed as newer ones are written (SPILL_TURNS) -------------------------------------------------------
ar = arena(202)
st = store(ar, "turns", turns=2)
ks, _, _ = put(st, ar, SYS, "sys", shared=True)
k1, _, _ = put(st, ar, TA[0], LA)
k2, kept2, _ = put(st, ar, TA[1], LA)
kf, keptf, _ = put(st, ar, FA, S.fork(LA, "f", 3 * B))
check("turns: a fork's own path counts too (A2f, A2 kept: A1 trimmed)", k1 not in st.index and k2 in st.index
      and st.stats["trimmed_turns"] == 1)
k3, _, _ = put(st, ar, TA[2], LA)
check("turns: A3 written, A2 still among the newest two of its path", k2 in st.index)
before_blocks = block_files(st)
ids4 = TA[2] + ids_of(B, 505)
k4, kept4, _ = put(st, ar, ids4, LA)
check("turns: A4 written, A2 trimmed (the newest 2 on the path stay: A4, A3)", k2 not in st.index
      and k3 in st.index and k4 in st.index and st.stats["trimmed_turns"] == 2)
check("turns: a trimmed turn's files go", not files_of(st, k1) and not files_of(st, k2))
check("turns: ... and no block (the newest turn names them)", before_blocks <= block_files(st))
check("turns: the fork off the path is untouched, and still reads back", kf in st.index
      and reads_back(st, ar, kf, FA, keptf))
check("turns: the shared prefix is never trimmed", ks in st.index)
check("turns: the newest turn reads back whole", reads_back(st, ar, k4, ids4, kept4))
check("turns: a restart finds the same points", set(store(ar, "turns", turns=2).index) == {ks, kf, k3, k4})
ar = arena(203)
st = store(ar, "turns0", turns=0)
for ids in [SYS] + TA:
    put(st, ar, ids, "sys" if ids is SYS else LA, shared=ids is SYS)
check("turns: SPILL_TURNS 0 keeps every turn until room is needed", len(st.index) == 4
      and st.stats["trimmed_turns"] == 0)
ar = arena(204)
st = store(ar, "turnsread", turns=1)
put(st, ar, SYS, "sys", shared=True)
k1, _, _ = put(st, ar, TA[0], LA)
st.reading.add(k1)
put(st, ar, TA[1], LA)
check("turns: a turn being read is not trimmed", k1 in st.index)
st.reading.clear()
env = os.environ.pop("TF_GLM_SPILL_TURNS", None)
for raw, want in (("0", 0), ("2", 2), (None, 4)):
    if raw is None:
        os.environ.pop("TF_GLM_SPILL_TURNS", None)
    else:
        os.environ["TF_GLM_SPILL_TURNS"] = raw
    got = int(os.environ.get("TF_GLM_SPILL_TURNS") or 4)    # (multi._spill_store's reading of it)
    check(f"turns: TF_GLM_SPILL_TURNS={raw} reads as {want}", got == want)
os.environ.pop("TF_GLM_SPILL_TURNS", None)
if env is not None:
    os.environ["TF_GLM_SPILL_TURNS"] = env
check("turns: multi reads TF_GLM_SPILL_TURNS that way", 'int(os.environ.get("TF_GLM_SPILL_TURNS") or 4)'
      in inspect.getsource(M.MultiDecoder._spill_store))

# -- the quota: a conversation over its share trims its own old turns and forks first -------------------------------
ar, st, k = tree("quota", turns=0, quota=0.0)
own_a = st._conv_bytes([k["A1"], k["A2"], k["A2f"], k["A3"]])
st.quota = (own_a - 1) / st.cap                        # A holds just over its share, B well under
ids4 = TA[2] + ids_of(B, 506)
k4, _, r = put(st, ar, ids4, LA, used=8.0)
check("quota: A's next turn is written", r == "queued" and k4 in st.index)
check("quota: A's own old turns and its fork went first", st.stats["quota_trims"] == 1
      and not {k["A1"], k["A2"], k["A2f"]} & set(st.index))
check("quota: A's newest state before the write stays", k["A3"] in st.index)
check("quota: the other conversation and the shared prefix are untouched",
      {k["B1"], k["B2"], k["S"]} <= set(st.index))
ar, st, k = tree("quota-under", turns=0)
st.quota = 0.9
put(st, ar, TA[2] + ids_of(B, 507), LA)
check("quota: under its share nothing is trimmed", st.stats["quota_trims"] == 0 and len(st.index) == 8)

# -- the free-disk floor ----------------------------------------------------------------------------------------------
ar = arena(205)
st = store(ar, "floor", floor=1.0 / 1024)               # 1 MiB floor
k1, _, _ = put(st, ar, TA[0], LA)
files0 = set(os.listdir(st.points_dir)), block_files(st)
st.disk_free = lambda: 0                               # the disk full of other things
k2, _, r = put(st, ar, TB[0], LB)
check("floor: a disk full of other things refuses the write", r == "no_space" and st.stats["no_space"] == 1)
check("floor: ... without evicting anything", set(st.index) == {k1})
check("floor: ... and leaves no file behind", (set(os.listdir(st.points_dir)), block_files(st)) == files0)
ar, st, k = tree("floor-room", turns=0, floor=1.0 / 1024)
start = st.stored()
need = 2 * st.block_bytes                              # what a write of one new block needs, about
st.disk_free = lambda: st.floor + need - 4096 + (start - st.stored())   # (what goes comes back as free space)
k4, _, r = put(st, ar, TA[2] + ids_of(B, 508), LA)
check("floor: short of the floor, room is made in the eviction order and the write goes on", r == "queued"
      and k4 in st.index and st.stats["no_space"] == 0)
check("floor: ... the first victim went, the rest stayed", k["B1"] not in st.index
      and {k["A3"], k["B2"], k["S"]} <= set(st.index))

# -- a restart with torn, orphaned and foreign files: only complete points stay ------------------------------------------
ar = arena(206)
st = store(ar, "torn", turns=0)
ks, kept_s, _ = put(st, ar, SYS, "sys", shared=True)
k1, _, _ = put(st, ar, TA[0], LA)
k2, _, _ = put(st, ar, TB[0], LB)
k3, _, _ = put(st, ar, TB[0] + ids_of(B, 509), LB)
ids_p = ids_of(B, 530)                                  # a picture prompt: its ids are keyed, its tokens after them
fill(ar, 0, B, 531)
st.persist(ids_p, snap(ids_p), extent=Pool(ROWS).add(0, B), lineage="pic", tokens=ids_of(B, 532),
           skip=M.MultiDecoder._SPILL_SKIP)
st.drain(TIMEOUT / 4)
kp = S.point_key(ids_p)
own1 = set(st.index[k1].blocks) - set(st.index[ks].blocks)
own3 = set(st.index[k3].blocks) - set(st.index[k2].blocks)
os.remove(st._point_path(k1, ".json"))                 # a write cut before its json: incomplete
with open(st._point_path(k2, ".ids"), "r+b") as fh:    # its ids damaged (their CRC-32 in its json)
    fh.write(b"\x00" * 8)
with open(st._point_path(kp, ".ids"), "r+b") as fh:    # a picture point's tokens damaged (its keyed ids intact):
    fh.seek(8 * B)                                      # only the CRC-32 tells
    fh.write(b"\x01" * 8)
os.remove(st._block_path(sorted(own3)[0]))             # a block of k3 gone
for d in (st.points_dir, st.blocks_dir):
    open(os.path.join(d, ".left-over.tmp"), "wb").write(b"x")      # temporary names a crash left
orphan = st._block_path("0" * 32)
open(orphan, "wb").write(b"y" * 100)                   # a block no point names
foreign = st._point_path("feedfacefeedface-7", ".json")
open(foreign, "w").write(json.dumps({"format": "another", "key": "feedfacefeedface-7"}))
st = store(ar, "torn", turns=0)
check("torn: only the complete point stays", set(st.index) == {ks})
check("torn: a point without its json loses its other files", not files_of(st, k1))
check("torn: ... and the blocks only it named", not own1 & block_files(st))
check("torn: a point whose ids fail their CRC goes with its files", not files_of(st, k2))
check("torn: ... also when only a picture point's tokens are damaged (its key still matches)", kp not in st.index
      and not files_of(st, kp))
check("torn: a point missing a block goes with its files and its other blocks", not files_of(st, k3)
      and not own3 & block_files(st))
check("torn: temporary names left by a crash go", not [f for d in (st.points_dir, st.blocks_dir)
                                                     for f in os.listdir(d) if f.startswith(".")])
check("torn: a block no point names goes", not os.path.exists(orphan))
check("torn: a json of another format goes", not os.path.exists(foreign))
check("torn: the counts match what is left", len(st.have) == len(st.index[ks].blocks) == len(block_files(st)))
check("torn: the complete point reads back", reads_back(st, ar, ks, SYS, kept_s))

# -- older builds' folders pruned (this rank's only), the newest other kept -------------------------------------------
ar = arena(207)
root = os.path.join(ROOT, "builds")
for i, (age, ranks) in enumerate(((300, (0, 1)), (200, (0,)), (100, (0,)))):
    for r in ranks:
        os.makedirs(os.path.join(root, f"spill-{i:016x}", f"rank{r}"), exist_ok=True)
        open(os.path.join(root, f"spill-{i:016x}", f"rank{r}", "f"), "wb").write(b"z")
    t = time.time() - age
    os.utime(os.path.join(root, f"spill-{i:016x}"), (t, t))
st = store(ar, "builds")
check("builds: the newest other build stays", os.path.isdir(os.path.join(root, f"spill-{2:016x}", "rank0")))
check("builds: older builds lose this rank's folder", not os.path.exists(os.path.join(root, f"spill-{1:016x}"))
      and not os.path.exists(os.path.join(root, f"spill-{0:016x}", "rank0")))
check("builds: ... not another rank's", os.path.exists(os.path.join(root, f"spill-{0:016x}", "rank1", "f")))

# -- a failed write leaves no temporary file ------------------------------------------------------------------------------
d = os.path.join(ROOT, "failed")
os.makedirs(d, exist_ok=True)
fsync = S.os.fsync


def full(fd):
    raise OSError(28, "No space left on device")


S.os.fsync = full
try:
    S._put(os.path.join(d, "x.json"), b"{}" * 100, None)
    raised = False
except OSError:
    raised = True
finally:
    S.os.fsync = fsync
check("failed write: raises, and leaves neither the file nor its temporary name", raised and not os.listdir(d))

# -- temporary files: made new (O_EXCL), never through a symlink (O_NOFOLLOW) ----------------------------------------------
check("tmp: temporary files are opened O_CREAT | O_EXCL | O_NOFOLLOW",
      bool(S._NEW & os.O_EXCL and S._NEW & os.O_CREAT and S._NEW & os.O_NOFOLLOW))
d = os.path.join(ROOT, "symlink")
os.makedirs(d, exist_ok=True)
victim = os.path.join(ROOT, "victim.txt")
open(victim, "w").write("not yours")
for direct in (False, True):
    path = os.path.join(d, f"p{int(direct)}.json")
    os.symlink(victim, os.path.join(d, f".p{int(direct)}.json"))           # planted at the temporary name
    S._put(path, b"mine", None, direct=direct)
    check(f"tmp: a symlink planted at the temporary name is not written through (direct={direct})",
          open(victim).read() == "not yours" and open(path, "rb").read() == b"mine"
          and not os.path.islink(path) and not os.path.lexists(os.path.join(d, f".p{int(direct)}.json")))
open(os.path.join(d, ".left.json"), "wb").write(b"old bytes, longer than the new ones")
S._put(os.path.join(d, "left.json"), b"new", None)
check("tmp: a temporary file left at the name is replaced, not appended to or reused",
      open(os.path.join(d, "left.json"), "rb").read() == b"new")
ar = arena(208)
st = store(ar, "symlink-block")
ids = ids_of(B, 510)
h0 = S.block_hashes(np.asarray(ids), B, "sl")[0][2]
os.symlink(victim, os.path.join(st.blocks_dir, f".{h0}.bin"))
fill(ar, 0, B, 511)
kept = rows(ar, 0, B)
st.persist(ids, snap(ids), extent=Pool(ROWS).add(0, B), lineage="sl", skip=M.MultiDecoder._SPILL_SKIP)
st.drain(TIMEOUT / 4)
check("tmp: a block written over a symlink planted at its temporary name leaves the target alone",
      open(victim).read() == "not yours" and not os.path.islink(st._block_path(h0)))
check("tmp: ... and the block reads back", reads_back(st, ar, S.point_key(ids), ids, kept))

# -- one engine a folder (and rank): flock on .lock-rank<r> --------------------------------------------------------------
ar = arena(209)
st = store(ar, "lock")
ids_l = ids_of(B, 512)
kl, _, _ = put(st, ar, ids_l, "l")
cfg = SP.SpillConfig(root=os.path.join(ROOT, "lock"), gib=1.0, min_tokens=1, min_free_gib=0.0)
try:
    S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=0, device="cpu", classes=CLASSES, quiet=True)
    msg = ""
except SP.Busy as exc:
    msg = str(exc)
check("lock: a second engine on the folder, same rank, is refused (Busy)", bool(msg))
check("lock: ... saying which folder and which process holds it", cfg.root in msg and f"pid {os.getpid()}" in msg)
old_build = os.path.join(cfg.root, "spill-00000000000000ab", "rank0")   # an older build's folder the other
os.makedirs(old_build, exist_ok=True)                                   # engine would prune (it keeps the newest:
open(os.path.join(old_build, "f"), "wb").write(b"z")                    # this engine's)
t = time.time() - 1000
os.utime(os.path.dirname(old_build), (t, t))
try:
    S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=0, device="cpu", classes=CLASSES, quiet=True,
                 signature="another build")
    other = False
except SP.Busy:
    other = True
check("lock: an engine of another build is refused too, before it prunes anything", other
      and st.has(kl) and os.path.isdir(st.dir) and len(files_of(st, kl)) == 3
      and os.path.exists(os.path.join(old_build, "f")))
r1 = S.BlockStore(cfg, planes=ar.planes, block=ALIGN, rank=1, device="cpu", classes=CLASSES, quiet=True)
check("lock: another rank on the same folder is not", r1 is not None)
r1.close()
try:
    SP.SpillStore(SP.SpillConfig(root=cfg.root, gib=1.0, min_tokens=1, min_free_gib=0.0), device="cpu", quiet=True)
    one = False
except SP.Busy:
    one = True
check("lock: the one-stream store (PARALLEL=1) is refused on a folder in use", one)
st = store(ar, "lock")                                  # (the first closed: a restart)
check("lock: once the first engine ended, the folder opens again with its points", kl in st.index)
code = ("import sys, time; from tensorfold.cuda import spill as SP; SP.hold(sys.argv[1], 0); "
        "print('held', flush=True); time.sleep(60)")
st.close()
OPEN.pop((cfg.root, 0), None)
proc = subprocess.Popen([sys.executable, "-c", code, cfg.root], stdout=subprocess.PIPE, text=True)
line = proc.stdout.readline().strip()
try:
    store(ar, "lock")
    held = False
except SP.Busy as exc:
    held = f"pid {proc.pid}" in str(exc)
proc.kill()
proc.wait()
check("lock: held by another process (its pid named)", line == "held" and held)
check("lock: a kill -9 of that process lets it go", kl in store(ar, "lock").index)


class _BusyStore:
    def __init__(self, *a, **kw):
        raise SP.Busy("/spill is in use by another engine as rank 0 (pid 1)")


dec = object.__new__(M.MultiDecoder)
dec.g = NS(spill_cfg=cfg, model_dir=ROOT, drafter_dir=None, drafter=None, agreed=(), _gather_ints=None)
dec.arena, dec.w, dec.grid, dec.drafts, dec.rank, dec.pool = ar, NS(world=1, device="cpu"), 0, None, 0, Pool(ROWS)
saved = S.BlockStore, SP.build_id, SP.weights_id
S.BlockStore, SP.build_id, SP.weights_id = _BusyStore, (lambda *a: "b"), (lambda *a: "w")
try:
    off = dec._spill_store()
except SP.Busy:
    off = "raised"
finally:
    S.BlockStore, SP.build_id, SP.weights_id = saved
check("lock: the engine runs with the tier off (Busy on every rank alike)", off is None)

# -- release: rows copied to host memory have their own cap (TF_GLM_SPILL_RELEASE_MIB) ----------------------------------
env = {k: os.environ.pop(k, None) for k in ("TF_GLM_SPILL_RELEASE_MIB", "TENSORFOLD_MEMORY_RESERVE_GIB")}
os.environ["TENSORFOLD_MEMORY_RESERVE_GIB"] = "4"     # (two Sparks: 14.5, so 1.8 GiB; 4 fits any test machine)
check("release cap: by default an eighth of the memory reserve (4 GiB: 512 MiB)", S.release_cap() == 512 << 20)
os.environ.pop("TENSORFOLD_MEMORY_RESERVE_GIB")
check("release cap: without a reserve set, an eighth of TensorFold's (at least 4 GiB: 512 MiB or more)",
      S.release_cap() >= 512 << 20 and S.release_cap() < S.BACKLOG)
os.environ["TENSORFOLD_MEMORY_RESERVE_GIB"] = "100000"
check("release cap: a reserve TensorFold would refuse falls back to its default, not an error at start",
      S.release_cap() >= 512 << 20)
os.environ.pop("TENSORFOLD_MEMORY_RESERVE_GIB")
os.environ["TF_GLM_SPILL_RELEASE_MIB"] = "64"
check("release cap: TF_GLM_SPILL_RELEASE_MIB=64 is 64 MiB", S.release_cap() == 64 << 20)
os.environ["TF_GLM_SPILL_RELEASE_MIB"] = "0"
check("release cap: 0 is allowed (release copies nothing)", S.release_cap() == 0)
bad = []
for raw in ("-1", "lots"):
    os.environ["TF_GLM_SPILL_RELEASE_MIB"] = raw
    try:
        S.release_cap()
    except ValueError:
        bad.append(raw)
check("release cap: a negative or non-number value is refused", bad == ["-1", "lots"])
for kk, v in env.items():
    os.environ.pop(kk, None)
    if v is not None:
        os.environ[kk] = v


def stalled(st: S.BlockStore):
    """The writer held at its next block until the returned event (a slow disk)."""

    gate, orig = threading.Event(), st._write_block

    def slow(*a):
        gate.wait(TIMEOUT)
        return orig(*a)

    st._write_block = slow
    return gate


ar = arena(210)
st = store(ar, "release")
st.release_cap = 4 * st.block_bytes                    # room for four blocks
gate = stalled(st)
m = decoder(ar, st)
ids1, ids2 = ids_of(3 * B, 513), ids_of(2 * B, 514)
x1 = m.pool.add(0, 3 * B)
fill(ar, 0, 3 * B, 515)
kept1 = rows(ar, 0, 3 * B)
st.persist(ids1, snap(ids1), extent=x1, lineage="r1", skip=M.MultiDecoder._SPILL_SKIP)
x2 = m.pool.add(3 * B, 2 * B)
fill(ar, 3 * B, 2 * B, 516)
st.persist(ids2, snap(ids2), extent=x2, lineage="r2", skip=M.MultiDecoder._SPILL_SKIP)
m.pool.remove(x1)                                       # (its rows go before they are written: the pool's guard)
check("release: rows within the cap are copied to the host", st.stats["released_blocks"] == 3
      and st.host_bytes == 3 * st.block_bytes)
m.pool.remove(x2)
check("release: past the cap a state's rows are not copied, the state is dropped", st.stats["released_blocks"] == 3
      and st.stats["release_dropped"] == 1 and st.host_bytes == 3 * st.block_bytes)
fill(ar, 0, 5 * B, 517)                                 # (other streams write those rows now)
gate.set()
st.drain(TIMEOUT / 4)
check("release: the copied state is stored as it was kept", reads_back(st, ar, S.point_key(ids1), ids1, kept1))
check("release: the dropped one is not stored, and wrote nothing", S.point_key(ids2) not in st.index
      and st.stats["write_failed"] == 1 and len(block_files(st)) == 3)
check("release: the host copies are let go once written", st.host_bytes == 0 and st.backlog == 0)
check("release: /health shows the cap and what it holds", st.info()["release_cap_mb"] == (4 * st.block_bytes) >> 20
      and "host_mb" in st.info() and "release_dropped" in st.info())
ar = arena(211)
st = store(ar, "release-tail")
gate = stalled(st)
n = B + 6
fill(ar, 0, n, 518)
st.persist(ids_of(n, 519), snap(ids_of(n, 519)), extent=Pool(ROWS).add(0, align_up(n)), lineage="t",
           skip=M.MultiDecoder._SPILL_SKIP)
tail = S._span_bytes(ar.planes, B, n)
check("release: the block a kept state's end cuts counts in the same host bytes", st.host_bytes == tail)
gate.set()
st.drain(TIMEOUT / 4)
check("release: ... until it is written", st.host_bytes == 0)

# -- the prefix wait: off by default (it changes admission with the spill tier off too) ---------------------------------
probe = "from tensorfold.families.glm5_next.cuda import multi as M; print(M.PREFIX_WAIT)"
seen = {}
for raw in (None, "0", "1"):
    e = {kk: v for kk, v in os.environ.items() if kk != "TF_GLM_PREFIX_WAIT"}
    if raw is not None:
        e["TF_GLM_PREFIX_WAIT"] = raw
    seen[raw] = subprocess.run([sys.executable, "-c", probe], env=e, capture_output=True, text=True).stdout.strip()
check("prefix wait: off unless TF_GLM_PREFIX_WAIT=1", seen == {None: "False", "0": "False", "1": "True"})


def waiter(lanes, hit=0):
    d = object.__new__(M.MultiDecoder)
    d.g = NS(shared=B, serial_only=False)
    d.drafts, d.prefix_waits, d.prefix_wait_tokens = None, 0, 0
    d._resume = lambda prompt: NS(ids=prompt[:hit]) if hit else None
    d.lanes = {ln.sid: ln for ln in lanes}
    return d


def lane(sid, key, shared, pos, **kw):
    return NS(sid=sid, decoding=False, s=NS(done=False), key=list(key), images=None, dflash=True, shared=set(shared),
              st=NS(pos=pos), **kw)


sysp = ids_of(3 * B, 520)                              # a 6k-token system prompt another request is prefilling
prompt = sysp + ids_of(B, 521)
req = NS(prompt=prompt, draft=True, vision=None)
filling = lane(1, sysp + ids_of(B, 522), {3 * B}, B)  # at token 2048 of 6144, its shared point ahead
saved_pw = M.PREFIX_WAIT
M.PREFIX_WAIT = False
check("prefix wait: off, a request never waits", waiter([filling]).prefix_wait(req) is None)
M.PREFIX_WAIT = True
d = waiter([filling])
check("prefix wait: on, it waits for the shared point another request is still prefilling",
      d.prefix_wait(req) == (1, 3 * B) and d.prefix_waits == 1 and d.prefix_wait_tokens == 3 * B)
check("prefix wait: not for a point already kept (what it adds is under the shared-prefix minimum)",
      waiter([filling], hit=3 * B - 100).prefix_wait(req) is None)
check("prefix wait: not when its prompt leaves the other's before the point",
      waiter([filling]).prefix_wait(NS(prompt=sysp[:2 * B] + ids_of(2 * B, 523), draft=True, vision=None)) is None)
check("prefix wait: not when its prompt ends at the point (nothing left to prefill)",
      waiter([filling]).prefix_wait(NS(prompt=sysp, draft=True, vision=None)) is None)
past, pics = lane(1, filling.key, {3 * B}, B), lane(1, filling.key, {3 * B}, B)
past.decoding, pics.images = True, [1]
check("prefix wait: not for a stream past its prompt (decoding) or with pictures",
      waiter([past]).prefix_wait(req) is None and waiter([pics]).prefix_wait(req) is None)
done = lane(1, filling.key, {3 * B}, B)
done.s.done = True
check("prefix wait: not for a stream that ended", waiter([done]).prefix_wait(req) is None)
check("prefix wait: not for a point it has passed", waiter([lane(1, filling.key, {3 * B}, 3 * B)]).prefix_wait(req)
      is None)
check("prefix wait: not for a request without drafts or with pictures",
      waiter([filling]).prefix_wait(NS(prompt=prompt, draft=False, vision=None)) is None
      and waiter([filling]).prefix_wait(NS(prompt=prompt, draft=True, vision=object())) is None)
nodf = waiter([lane(1, filling.key, {3 * B}, B)])
nodf.drafts = object()
nodf.lanes[1].dflash = False
check("prefix wait: not for a stream whose states keep no DFlash2 window", nodf.prefix_wait(req) is None)
d = waiter([filling])
w = (1, 3 * B)
check("prefix wait: not released while the point is still ahead", not d.prefix_ready(w))
filling.st.pos = 3 * B
check("prefix wait: released once the point is kept", d.prefix_ready(w))
filling.st.pos = B
filling.decoding = True
check("prefix wait: released when the stream left its prompt", d.prefix_ready(w))
filling.decoding = False
check("prefix wait: released when the stream went", waiter([]).prefix_ready(w))

# the scheduler with the spill tier off (no prestage): a waiting request is held apart while the queue goes on

class SchedDecoder:
    def __init__(self, w):
        self.w, self.admitted, self.streams = w, [], {}

    def live(self):
        return len(self.streams)

    def fits(self, s):
        return True

    def admit(self, s):
        self.admitted.append(s)
        self.streams[id(s)] = s

    def prefix_wait(self, s):
        return self.w.prefix_wait(s)

    def prefix_ready(self, p):
        return self.w.prefix_ready(p)


def sched(w):
    o = object.__new__(M.GlmScheduler)
    o.decoder, o.max_streams = SchedDecoder(w), 4
    o.waiting, o.held, o.boxes, o.yields, o.gather_s = Waiting(), None, {}, 0, 0
    o.flushing, o.flush_lock, o.loading, o.ready, o.staged, o.prefix = None, threading.Lock(), [], [], set(), []
    o.max_in_system = None
    return o


def item(p):
    return (NS(prompt=p, draft=True, vision=None, background=False, done=False, emit=NS()), queue.Queue())


filling = lane(1, sysp + ids_of(B, 522), {3 * B}, B)
for on in (False, True):
    M.PREFIX_WAIT = on
    o = sched(waiter([filling]))
    a, b = item(prompt), item(ids_of(2 * B, 524))
    o.waiting.put(a)
    o.waiting.put(b)
    o._admit()
    if not on:
        check("prefix wait off: both requests admitted at once (admission as without the patch)",
              o.decoder.admitted == [a[0], b[0]] and not o.prefix)
        continue
    check("prefix wait on: the request sharing the prefix is held apart, the next one admitted",
          o.decoder.admitted == [b[0]] and [it for _, it in o.prefix] == [a])
    o._admit()
    check("prefix wait on: still held while the point is ahead", o.decoder.admitted == [b[0]])
    filling.st.pos = 3 * B
    o._admit()
    check("prefix wait on: admitted once the point is kept, before the queue", o.decoder.admitted == [b[0], a[0]]
          and not o.prefix and getattr(a[0], "prefix_waited", False))
M.PREFIX_WAIT = saved_pw
conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "config.sh")
if os.path.exists(conf):                                # (the recipe's tree; not when the check is run alone)
    check("prefix wait: config.sh defaults PREFIX_WAIT to 0", 'PREFIX_WAIT="${PREFIX_WAIT:-0}"' in open(conf).read())

print("all passed" if not fails else f"FAILED: {fails}", flush=True)
os._exit(1 if fails else 0)
