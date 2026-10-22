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
  at least that many stored blocks out of them and pins none later.
Exit code 1 when a check fails; tools/spill_blocks_mutants.sh runs it on faithful mutants of each fix.
"""
import inspect
import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

from tensorfold.cuda import health as H
from tensorfold.cuda import spill as SP
from tensorfold.cuda import spill_blocks as S
from tensorfold.families.glm5_next.cuda import multi as M
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up
from tensorfold.server import metrics as MT

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
print("all passed" if not fails else f"FAILED: {fails}", flush=True)
os._exit(1 if fails else 0)
