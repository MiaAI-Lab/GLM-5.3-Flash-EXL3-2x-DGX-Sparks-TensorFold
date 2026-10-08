#!/usr/bin/env python3
"""Checks for patch 0109 (the NVMe KV tier, TF_GLM_KV_TIER), CPU only, run inside the image that scripts/prepare.sh
built (no GPU, no model, no server); its files go to a scratch folder under $TIER_TEST_DIR (default: /tmp):

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/kv_tier_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

Prints PASS / FAIL lines and ALL PASS; exit 1 when a check fails (a hang prints a FAIL line after $TEST_TIMEOUT_S,
default 300).

Part 0, the tier (a real Tier over a fake arena with the real planes' geometry: latents div 1, pooled keys div 4 with
2 pad rows): a kept state waits with its rows in the pool, a pump stages at most PUMP_BLOCKS blocks, a drain writes the
rest; a later turn writes only its new blocks, another lineage its own; a load at another base gives the rows, ids and
small state back bit for bit and writes no row outside the state's (the next extent's first pooled row included);
prefix matching (longest, longer_than, grid, the whole prompt only with its head row); rows released before they are
overwritten are written as they were; the ring's size; the write queue's cap and the free-disk floor skip before any
host copy; the cap on disk drops least recently used points and unnamed blocks; one disk tier at a time.

Part 1, the files: fsync order seen through a fake ``os.fsync`` / ``os.replace`` (a new tier's folder and root, each
block fsynced before its rename, the blocks' directory before the point, the ids and small state before their renames,
the points' directory before and after the json, every fsync and CRC off the engine loop); the json's CRC-32s equal the
files'; files and folders take the tier directory's owner; a load checks every file on the reader threads and fails -
None, no exception, nothing copied from the bad block on - on a corrupted, truncated or missing block, a block of the
wrong size with a matching CRC, a damaged small state (which torch.load alone accepts), a point whose lineage was
edited onto another computation's blocks; ``forget`` drops the point, the bad block (even one a waiting state names)
and every point naming it, keeps blocks other points or a waiting state name; a point whose reused block vanished
before its write is not written; a restart after a crash between the block writes and the point (or between the ids
and the json) removes what no complete point names and keeps the rest loadable; a restart skips points with damaged ids
or a short CRC list; other builds' folders are pruned (newest TF_GLM_KV_TIER_KEEP_BUILDS kept, default 1, never this
rank's own, never another rank's); ``make`` keys the folder on the build, the weights, the agreed settings (prefill
rows among them) and the format, and the real build key is stable.

Part 2, the protocol: two real ``MultiDecoder``s (rank 0 deciding, a follower applying its messages in-process) over
real Pools and Arenas, each rank its own Tier. A load with nothing else decoding, and one in slices beside a decoding
stream (each KV_PART reaches the follower before rank 0 copies), become the same kept prompt on both ranks; a point the
follower lacks is not loaded (KV_QUERY). A block damaged on the follower, the follower's small state damaged, a block
damaged on rank 0 during a sliced load: every rank drops the point (KV_CHECK, KV_END), frees the load's extent, keeps
nothing, the request gets the pool's hit (prefill); a later request does not load the same point again."""
import sys as _sys
import traceback as _tb

_sys.excepthook = lambda t, v, tb: (_tb.print_exception(t, v, tb), print(f"FAIL raised {t.__name__}: {v}", flush=True))
import errno
import faulthandler
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import zlib

import numpy as np
import torch

from tensorfold.families.glm5_next.cuda import kvtier
from tensorfold.families.glm5_next.cuda import multi as M
from tensorfold.families.glm5_next.cuda import pool as P

TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "300"))
faulthandler.dump_traceback_later(int(TIMEOUT) + 30, exit=True)


def _hung():
    print(f"FAIL timed out after {TIMEOUT:g} s: a write or a load never finished", flush=True)
    faulthandler.dump_traceback()
    os._exit(1)


_watch = threading.Timer(TIMEOUT, _hung)
_watch.daemon = True
_watch.start()
torch.manual_seed(0)
B = kvtier.BLOCK
SENT = 0xA5
fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    fails += not ok


def arena(tokens):
    planes = [P.Plane(torch.randint(0, 256, (tokens, 528), dtype=torch.uint8)),
              P.Plane(torch.randn(tokens, 64).to(torch.bfloat16)),
              P.Plane(torch.randint(0, 256, (tokens // 4 + 2, 144), dtype=torch.uint8), 4, 2)]
    return P.Arena(tokens, planes)


class Snap:
    def __init__(self, ids, lineage):
        self.ids = [int(v) for v in ids]
        self.lineage = lineage
        self.rec = torch.randn(3, 2, 8, 8)
        self.conv = torch.randn(3, 3, 16).to(torch.bfloat16)
        self.tail = [(torch.tensor([4, 5]), torch.randn(2, 2, 16).to(torch.bfloat16))]
        self.head = None
        self.drafter_end = len(self.ids)
        self.drafter_rows = [torch.randn(2, 5, 8)]


def key_of(snap):
    return kvtier.point_key(np.asarray(snap.ids, dtype=np.int32))


def settle(t, secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        if t.jobs.empty() and not t.inflight and not t.pending:
            return True
        time.sleep(0.01)
    return False


def write(t, snap, src, base=0):
    t.persist(snap, src, base)
    t.drain(src)
    return settle(t) and t.has(key_of(snap))


def rows_equal(src, dst, base_s, base_d, s, e):
    for p, q in zip(src.planes, dst.planes):
        a, b = kvtier._plane_rows(p, s, e)
        x = p.tensor[base_s // p.div + a:base_s // p.div + b]
        y = q.tensor[base_d // q.div + a:base_d // q.div + b]
        if not torch.equal(x.contiguous().view(-1).view(torch.uint8), y.contiguous().view(-1).view(torch.uint8)):
            return False
    return True


def sentinel(a):
    for p in a.planes:
        p.tensor.view(-1).view(torch.uint8).fill_(SENT)


def untouched(a, base, s, e):
    for p in a.planes:
        lo, hi = kvtier._plane_rows(p, s, e)
        if not bool((p.tensor[base // p.div + lo:base // p.div + hi].contiguous().view(-1).view(torch.uint8)
                     == SENT).all()):
            return False
    return True


def flip(path, at):
    with open(path, "r+b") as f:
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0xFF]))


def load_quiet(t, key, dst, base):
    try:
        return t.load(key, dst, base, "cpu"), None
    except Exception as exc:                          # a failed check must be a None, never an exception
        return None, exc


def ids(seed, n):
    """Token ids of a conversation ``seed``: the same seed, the same prefix (point keys are the ids alone)."""
    return np.random.default_rng(seed).integers(0, 150000, size=8 * B).astype(np.int32)[:n]


IDS = ids(99, 8 * B)
root = tempfile.mkdtemp(prefix="kvtier-check-", dir=os.environ.get("TIER_TEST_DIR"))
try:
    # -- part 0: the tier (writes at keep through the ring, blocks shared by later turns, loads, limits) ------------
    tg = kvtier.Tier(root + "/g", 1.0, 16, 0, {"test": "g"}, gain=1)
    srcg = arena(8 * B)
    n1, n2 = 3 * B + 100, 4 * B + 37
    g1 = Snap(ids(40, n1), "A")
    tg.persist(g1, srcg, 2 * B)
    check("a kept state waits with its rows in the pool: nothing written before a pump",
          len(tg.pending) == 1 and not tg.has(key_of(g1)) and tg.stats["staged_blocks"] == 0)
    tg.pump(srcg)
    check(f"one pump stages at most {kvtier.PUMP_BLOCKS} blocks", 0 < tg.stats["staged_blocks"] <= kvtier.PUMP_BLOCKS)
    tpb = kvtier.Tier(root + "/pb", 1.0, 16, 0, {"test": "pb"}, gain=1)
    tpb.persist(Snap(ids(44, 7 * B), "PB"), srcg, 0)          # 7 blocks waiting: more than one pump's worth
    tpb.pump(srcg)
    check(f"a pump between rounds stages exactly {kvtier.PUMP_BLOCKS} of 7 waiting blocks",
          tpb.stats["staged_blocks"] == kvtier.PUMP_BLOCKS)
    tpb.drain(srcg)
    settle(tpb)
    want1 = [P.Plane(p.tensor.clone(), p.div, p.pad) for p in srcg.planes]
    tg.drain(srcg)
    ok = settle(tg) and tg.has(key_of(g1))
    nblk = len(list(tg.blocks.glob("*.bin")))
    check(f"the state is written: 3 whole blocks and a tail ({nblk})", ok and nblk == 4)
    write(tg, Snap(ids(40, n2), "A"), srcg, 2 * B)
    nblk2 = len(list(tg.blocks.glob("*.bin")))
    check(f"a later turn of the same conversation writes only its new blocks ({nblk} -> {nblk2})", nblk2 == nblk + 2)
    gb = Snap(ids(40, 2 * B), "B")
    write(tg, gb, srcg, 0)
    check("the same ids from another computation (lineage) write blocks of their own",
          len(list(tg.blocks.glob("*.bin"))) == nblk2 + 2)
    dstg = arena(8 * B)
    sentinel(dstg)
    got, exc = load_quiet(tg, key_of(g1), dstg, 3 * B)
    check("a load puts the rows back at another base, bit for bit, ids and small state too",
          got is not None and got[0] == g1.ids and torch.equal(got[1]["rec"], g1.rec)
          and all(torch.equal(p.tensor[2 * B // p.div:-(-(2 * B + n1) // p.div)],
                              q.tensor[3 * B // q.div:-(-(3 * B + n1) // q.div)]) for p, q in zip(want1, dstg.planes)))
    check("a load writes no row outside the state's own (the rest of its extent, the next extent's first pooled row, "
          "the row before)", untouched(dstg, 3 * B, n1, P.align_up(n1)) and P.align_up(n1) == 4 * B
          and bool((dstg.planes[2].tensor[7 * B // 4] == SENT).all()) and all(
              bool((p.tensor[3 * B // p.div - 1].contiguous().view(-1).view(torch.uint8) == SENT).all())
              for p in dstg.planes))
    pr = [int(v) for v in ids(40, 5 * B)]
    k2g = key_of(Snap(ids(40, n2), "A"))
    check("the longest kept prefix of a prompt is found", tg.best(pr, 0, 0, lambda k: False) == k2g)
    check("nothing longer than longer_than", tg.best(pr, n2, 0, lambda k: False) is None)
    check("on a prompt grid only states on it", tg.best(pr, 0, 64, lambda k: False) == key_of(gb))
    check("the whole prompt only with its head row kept",
          tg.best(pr[:n1], 0, 0, lambda k: False) == key_of(gb) and tg.best(pr[:n1], 0, 0, lambda k: True) == key_of(g1))
    # rows about to be overwritten while their state waits are copied first (release)
    x = types.SimpleNamespace(base=0)
    g3 = Snap(ids(41, 2 * B + 5), "C")
    tg.persist(g3, srcg, x)
    want3 = [P.Plane(p.tensor.clone(), p.div, p.pad) for p in srcg.planes]
    tg.release(x, 0, srcg)
    for p in srcg.planes:
        p.tensor[:-(-(2 * B + 5) // p.div)].view(-1).view(torch.uint8).fill_(0x3C)
    tg.drain(srcg)
    settle(tg)
    d3 = arena(8 * B)
    got3, _ = load_quiet(tg, key_of(g3), d3, 0)
    check("a waiting state's rows released before they are overwritten are written as they were",
          got3 is not None and tg.stats["released_blocks"] >= 2 and all(
              torch.equal(p.tensor[:-(-(2 * B + 5) // p.div)], q.tensor[:-(-(2 * B + 5) // q.div)])
              for p, q in zip(want3, d3.planes)))
    check("the write ring holds TF_GLM_KV_TIER_STAGE_MIB (256 MiB here: at least two blocks)",
          len(tg.slots) == max(2, (256 << 20) // tg.slots[0].numel()) and tg.free_slots.qsize() == len(tg.slots))
    # limits: the write queue's cap, the free-disk floor (both before any host copy), the cap on disk
    tq = kvtier.Tier(root + "/q", 1.0, 16, 0, {"test": "q"}, gain=1)
    tq.backlog_max = 1024
    real_cat, copies = torch.cat, []
    torch.cat = lambda *a, **k: copies.append(1) or real_cat(*a, **k)
    try:
        tq.persist(Snap(ids(42, 2 * B), "Q"), srcg, 0)
        tq.backlog_max, tq.min_free = 1 << 40, shutil.disk_usage(tq.dir).free + 1
        tq.persist(Snap(ids(43, 2 * B), "Q"), srcg, 0)
    finally:
        torch.cat = real_cat
    check("over the write queue's cap and under the free-disk floor: skipped, counted apart, no host copy",
          tq.stats["skipped_backlog"] == 1 and tq.stats["skipped_disk"] == 1 and not copies and not tq.pending
          and tq.backlog == 0 and not tq.inflight)
    check("the free-disk floor is off unless TF_GLM_KV_TIER_MIN_FREE_GIB is set",
          kvtier.Tier(root + "/q2", 1.0, 16, 0, {"test": "q2"}, gain=1).min_free == 0)
    tc_ = kvtier.Tier(root + "/cap", 0.000001, 16, 0, {"test": "cap"}, gain=1)
    for k in range(3):
        write(tc_, Snap(ids(50 + k, B + 5), "P"), srcg, 0)
        time.sleep(0.02)
    check("under the cap least recently used points go first, then blocks no point names",
          len(tc_.index) <= 1 and {p.stem for p in tc_.blocks.glob("*.bin")} <= {b for m in tc_.index.values()
                                                                                 for b in m["blocks"]})
    os.environ["TF_GLM_KV_TIER"] = root + "/one"
    try:
        kvtier.make(types.SimpleNamespace(g=types.SimpleNamespace(spill_cfg=object())))
        refused = ""
    except ValueError as e:
        refused = str(e)
    finally:
        del os.environ["TF_GLM_KV_TIER"]
    check("one disk tier at a time: TF_GLM_KV_TIER beside --spill-gib stops the start",
          "TF_GLM_KV_TIER" in refused and "--spill-gib" in refused)

    # -- part 1a: the write order -------------------------------------------------------------------------------
    real_fsync, real_replace, real_crc = os.fsync, os.replace, zlib.crc32
    events, crc_threads = [], []

    def rec_fsync(fd):
        events.append(("fsync", os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")),
                       threading.current_thread().name))
        return real_fsync(fd)

    def rec_replace(a, b):
        events.append(("replace", os.path.realpath(str(a)), os.path.realpath(str(b)), threading.current_thread().name))
        return real_replace(a, b)

    class Z:
        def __getattr__(self, k):
            return getattr(zlib, k)

        def crc32(self, *a):
            crc_threads.append(threading.current_thread().name)
            return real_crc(*a)

    os.fsync = rec_fsync
    try:
        t = kvtier.Tier(root + "/a", 1.0, 16, 0, {"test": "a"}, gain=1)
    finally:
        os.fsync = real_fsync
    check("a new tier fsyncs its folder and the root",
          [e[1] for e in events] == [os.path.realpath(t.dir), os.path.realpath(root + "/a")])
    events.clear()
    src = arena(8 * B)
    s1 = Snap(ids(1, 3 * B + 100), "W")                     # 3 whole blocks + a tail
    os.fsync, os.replace, kvtier.zlib = rec_fsync, rec_replace, Z()
    try:
        ok_write = write(t, s1, src, 0)
    finally:
        os.fsync, os.replace, kvtier.zlib = real_fsync, real_replace, zlib
    k1 = key_of(s1)
    check("a state is written and indexed", ok_write)
    main = threading.current_thread().name
    check("every fsync of a write runs on the writer thread, none on the engine loop",
          bool(events) and all(e[-1] == "kvtier-writer" for e in events))
    check("every CRC-32 of a write runs off the engine loop", bool(crc_threads) and main not in crc_threads)
    pts, blk = os.path.realpath(t.points), os.path.realpath(t.blocks)

    def idx(kind, *want):
        for i, e in enumerate(events):
            if e[0] == kind and all(w is None or e[1 + j] == w for j, w in enumerate(want)):
                return i
        return -1

    def last_idx(kind, *want):
        found = -1
        for i, e in enumerate(events):
            if e[0] == kind and all(w is None or e[1 + j] == w for j, w in enumerate(want)):
                found = i
        return found

    names = t.index[k1]["blocks"] if k1 in t.index else []
    order = len(names) == 4
    brep = []
    for h in names:
        fin, tmp = os.path.join(blk, f"{h}.bin"), os.path.join(blk, f".{h}.bin")
        r = idx("replace", tmp, fin)
        f = idx("fsync", tmp)
        order &= 0 <= f < r
        brep.append(r)
    check("each block is fsynced under its temporary name before its rename", order)
    rj = idx("replace", os.path.join(pts, f".{k1}.json"), os.path.join(pts, f"{k1}.json"))
    fdir_b = last_idx("fsync", blk)
    check("the blocks' directory is fsynced after every block's rename and before the point's json",
          rj >= 0 and bool(brep) and max(brep) < fdir_b < rj)
    seq = True
    for suffix in (".ids.npy", ".pt"):
        f = idx("fsync", os.path.join(pts, f".{k1}{suffix}"))
        r = idx("replace", os.path.join(pts, f".{k1}{suffix}"), os.path.join(pts, f"{k1}{suffix}"))
        seq &= 0 <= f < r
        dirs = [i for i, e in enumerate(events) if e[0] == "fsync" and e[1] == pts and r < i < rj]
        seq &= bool(dirs)
    check("the ids and the small state: fsynced, renamed, then the points' directory fsynced, before the json", seq)
    fj = idx("fsync", os.path.join(pts, f".{k1}.json"))
    check("the json is fsynced before its rename and the points' directory after it",
          0 <= fj < rj < last_idx("fsync", pts))
    meta = json.loads((t.points / f"{k1}.json").read_text())
    check("the json carries each block's CRC-32, the ids' and the small state's",
          meta.get("crcs") == [real_crc((t.blocks / f"{h}.bin").read_bytes()) for h in names]
          and meta.get("ids_crc") == real_crc((t.points / f"{k1}.ids.npy").read_bytes())
          and meta.get("small_crc") == real_crc((t.points / f"{k1}.pt").read_bytes()))
    check("no temporary file is left", not list(t.points.glob(".*")) and not list(t.blocks.glob(".*")))

    # files private (they hold prompts' tokens) and owned by the tier directory's owner (a server run as root in a
    # container)
    modes = {(p.stat().st_mode & 0o777, p.is_dir()) for p in [t.dir, *t.dir.rglob("*")]}
    check(f"every file 0600 and every folder 0700 ({sorted(modes)})", modes == {(0o600, False), (0o700, True)})
    if os.geteuid() == 0:
        ro = root + "/own"
        os.makedirs(ro)
        os.chown(ro, 4321, 4322)
        to = kvtier.Tier(ro, 1.0, 16, 0, {"test": "own"}, gain=1)
        write(to, Snap(ids(2, 2 * B + 1), "O"), src, 0)
        owners = {(p.stat().st_uid, p.stat().st_gid) for p in [to.dir, *to.dir.rglob("*")]}
        check(f"every file and folder takes the tier directory's owner ({owners})", owners == {(4321, 4322)})
    else:
        print("info: not root: ownership not checked", flush=True)

    # -- part 1a, a failing disk: a write that fails (ENOSPC) at a block's or a point file's fsync or rename leaves
    # no temporary file behind and no point
    for at, what in (("rename", ".bin"), ("rename", ".json"), ("fsync", ".bin"), ("fsync", ".ids.npy")):
        tf = kvtier.Tier(root + f"/full-{at}{what}", 1.0, 16, 0, {"test": f"full-{at}{what}"}, gain=1)
        sf = Snap(ids(30, 2 * B + 3), "E")

        def full_replace(a, b, what=what):
            if str(b).endswith(what):
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_replace(a, b)

        def full_fsync(fd, what=what):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(what):
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_fsync(fd)

        if at == "rename":
            os.replace = full_replace
        else:
            os.fsync = full_fsync
        try:
            ok_full = write(tf, sf, src, 0)
        finally:
            os.fsync, os.replace = real_fsync, real_replace
        check(f"a disk full at a {what} file's {at}: the state is not kept, no temporary file is left",
              not ok_full and key_of(sf) not in tf.index
              and not list(tf.points.glob(".*")) and not list(tf.blocks.glob(".*")))

    # -- part 1b: loads -----------------------------------------------------------------------------------------
    dst = arena(8 * B)
    crc_threads.clear()
    kvtier.zlib = Z()
    try:
        got, exc = load_quiet(t, k1, dst, 2 * B)
    finally:
        kvtier.zlib = zlib
    check("an undamaged load comes back (rows at another base, ids, small state)",
          got is not None and got[0] == s1.ids and rows_equal(src, dst, 0, 2 * B, 0, len(s1.ids))
          and torch.equal(got[1]["rec"], s1.rec))
    check("a load's CRC-32s run on the reader threads, not the engine loop", bool(crc_threads) and main not in crc_threads)

    HOWS = ("corrupt", "truncated", "missing", "short-with-crc")

    def damaged(how, h_index):
        """A fresh state of its own lineage, damaged by ``how``; its load into a sentinel arena."""
        s = Snap(ids(20 + HOWS.index(how), 3 * B + 100), f"D-{how}")
        if not write(t, s, src, 0):
            return None
        k = key_of(s)
        h = t.index[k]["blocks"][h_index]
        f = t.blocks / f"{h}.bin"
        if how == "corrupt":
            flip(f, f.stat().st_size // 2)
        elif how == "truncated":
            os.truncate(f, f.stat().st_size // 2)
        elif how == "missing":
            f.unlink()
        elif how == "short-with-crc":                  # a writer bug: a short block whose CRC-32 matches it
            data = f.read_bytes()[:f.stat().st_size - 4096]
            f.write_bytes(data)
            t.have_blocks[h] = real_crc(data)
        d = arena(8 * B)
        sentinel(d)
        crc0, lf0 = t.stats["crc_failed"], t.stats["load_failed"]
        ld = kvtier.Loading(t, k, d, kvtier._At(2 * B), "cpu")
        try:
            ld.step(ld.total)
            res, exc = ld.finish(), None
        except Exception as e:
            res, exc = None, e
        s0, e0, _ = ld.hashes[h_index]
        return dict(res=res, exc=exc, ld=ld, h=h, k=k, crc=t.stats["crc_failed"] - crc0,
                    lf=t.stats["load_failed"] - lf0, rest_untouched=untouched(d, 2 * B, s0, len(s.ids)),
                    before=rows_equal(src, d, 0, 2 * B, 0, s0))

    for how, crc_expected in zip(HOWS, (1, 1, 0, 0)):
        r = damaged(how, 1)
        check(f"{how} block: the load fails with None (no exception), the block named bad, counted",
              r is not None and r["res"] is None and r["exc"] is None and r["ld"].bad == {r["h"]}
              and r["lf"] == 1 and r["crc"] == crc_expected)
        check(f"{how} block: nothing from the bad block on was written; the blocks before it were",
              r is not None and r["rest_untouched"] and r["before"])
        if how == "short-with-crc":
            check("short-with-crc block: failed on its size", r is not None and "the planes take" in str(r["ld"].failed))

    # a damaged small state that torch.load alone accepts: only its CRC-32 stops it
    s = Snap(ids(3, 2 * B + 50), "S")
    write(t, s, src, 0)
    ks = key_of(s)
    pt = t.points / f"{ks}.pt"
    raw = pt.read_bytes()
    at = raw.find(s.rec.numpy().tobytes())
    flip(pt, at + 17)
    try:
        alone = torch.load(pt, map_location="cpu", weights_only=False)
        accepted = not torch.equal(alone["rec"], s.rec)
    except Exception:
        accepted = False
    print(f"info: torch.load alone {'accepts' if accepted else 'rejects'} the damaged small state", flush=True)
    r, exc = load_quiet(t, ks, arena(8 * B), 0)
    check("damaged small state: the load fails with None, no block named bad",
          at >= 0 and r is None and exc is None)

    # a point whose lineage was edited onto another computation's blocks of the same ids
    sa, sb = Snap(ids(4, 2 * B), "LA"), Snap(ids(4, 3 * B), "LB")
    write(t, sa, src, 0)
    src_b = arena(8 * B)
    write(t, sb, src_b, 0)
    ka = key_of(sa)
    meta = json.loads((t.points / f"{ka}.json").read_text())
    meta["lineage"] = "LB"
    (t.points / f"{ka}.json").write_text(json.dumps(meta))
    t2 = kvtier.Tier(root + "/a", 1.0, 16, 0, {"test": "a"}, gain=1)
    r, exc = load_quiet(t2, ka, arena(8 * B), 0)
    check("a point whose lineage names another computation's blocks fails its load (never their rows)",
          ka in t2.index and r is None and exc is None)

    # -- part 1c: forget ------------------------------------------------------------------------------------------
    p1, p2, p3 = Snap(ids(5, 2 * B), "F"), Snap(ids(5, 3 * B), "F"), Snap(ids(6, 2 * B + 9), "G")
    for sn in (p1, p2, p3):
        write(t, sn, src, 0)
    k1_, k2_, k3_ = key_of(p1), key_of(p2), key_of(p3)
    b2 = t.index[k2_]["blocks"]
    t.forget(k2_, ())                                     # another rank failed: no bad block here
    r1, _ = load_quiet(t, k1_, arena(8 * B), 0)
    check("forget(no bad block): the point goes, its own block too, the blocks a shorter point shares stay",
          k2_ not in t.index and not (t.blocks / f"{b2[2]}.bin").exists() and all(
              (t.blocks / f"{h}.bin").exists() for h in b2[:2]) and r1 is not None)
    write(t, p2, src, 0)
    bad = t.index[k1_]["blocks"][0]
    t.forget(k1_, {bad})
    check("forget(a bad block): the point, the bad block and every point naming it go; other points stay",
          k1_ not in t.index and k2_ not in t.index and not (t.blocks / f"{bad}.bin").exists()
          and not any((t.points / f"{k}{x}").exists() for k in (k1_, k2_) for x in (".json", ".pt", ".ids.npy"))
          and t.has(k3_))
    # a bad block goes even when a waiting state names it: that state is then not written (never a point over it)
    w1, w2 = Snap(ids(16, 2 * B), "WB"), Snap(ids(16, 3 * B), "WB")
    write(t, w1, src, 0)
    t.persist(w2, src, 0)                                 # waiting: its first two blocks are w1's
    badw = t.index[key_of(w1)]["blocks"][1]
    t.forget(key_of(w1), {badw})
    t.drain(src)
    check("forget(a bad block a waiting state names): the block goes, the waiting state is not written over it",
          settle(t) and not (t.blocks / f"{badw}.bin").exists() and key_of(w2) not in t.index)
    # a waiting state's reused blocks survive a forget of the point they came from
    q1, q2 = Snap(ids(7, 2 * B), "Q"), Snap(ids(7, 3 * B), "Q")
    write(t, q1, src, 0)
    t.persist(q2, src, 0)                                 # waiting: its first two blocks are q1's
    t.forget(key_of(q1), ())
    t.drain(src)
    check("forget keeps the blocks a waiting state names: that state is still written", settle(t) and t.has(key_of(q2)))
    # a point whose reused block vanished before its write is not written
    v1, v2 = Snap(ids(8, 2 * B), "V"), Snap(ids(8, 3 * B), "V")
    write(t, v1, src, 0)
    t.persist(v2, src, 0)
    t.have_blocks.pop(t.index[key_of(v1)]["blocks"][0])
    t.drain(src)
    check("a point whose reused block is gone before its write is not written", settle(t) and key_of(v2) not in t.index)
    check("a point missing a block is not complete (what KV_QUERY answers)", key_of(v1) in t.index and not t.has(key_of(v1)))

    # -- part 1d: restart -------------------------------------------------------------------------------------------
    root_c = root + "/c"
    tc = kvtier.Tier(root_c, 1.0, 16, 0, {"test": "c"}, gain=1)
    keep = Snap(ids(9, 2 * B + 5), "K")
    write(tc, keep, src, 0)
    real_put = kvtier._put
    crashed = {}
    for at_name, lin in ((".json", "X1"), (".ids.npy", "X2")):
        s = Snap(ids(10 + len(at_name), 3 * B + 7), lin)

        def put(path, w, at_name=at_name):
            if path.name.endswith(at_name):
                raise RuntimeError("crash")
            return real_put(path, w)

        kvtier._put = put
        try:
            tc.persist(s, src, 0)
            tc.drain(src)
            settle(tc)
        finally:
            kvtier._put = real_put
        crashed[at_name] = kvtier.block_hashes(np.asarray(s.ids, np.int32), lin)
    (tc.points / ".leftover.json").write_text("{")     # a temporary file a crash cut off
    on_disk = {p.stem for p in tc.blocks.glob("*.bin")}
    staged = all(h in on_disk for v in crashed.values() for _, _, h in v)
    tr = kvtier.Tier(root_c, 1.0, 16, 0, {"test": "c"}, gain=1)
    left = {p.stem for p in tr.blocks.glob("*.bin")}
    check("restart after a crash between the blocks and the point: its blocks and files are removed",
          staged and not any(h in left for v in crashed.values() for _, _, h in v)
          and sorted(p.name for p in tr.points.iterdir()) == sorted(f"{key_of(keep)}{x}" for x in (".json", ".pt",
                                                                                                  ".ids.npy")))
    r, _ = load_quiet(tr, key_of(keep), arena(8 * B), 0)
    check("restart after a crash: the complete point still loads", r is not None and r[0] == keep.ids)
    # damaged ids, a short CRC list: skipped and removed at restart
    d1, d2 = Snap(ids(13, 2 * B + 11), "I1"), Snap(ids(14, 2 * B + 13), "I2")
    write(tr, d1, src, 0)
    write(tr, d2, src, 0)
    flip(tr.points / f"{key_of(d1)}.ids.npy", 200)
    m2 = json.loads((tr.points / f"{key_of(d2)}.json").read_text())
    m2["crcs"] = m2["crcs"][:-1]
    (tr.points / f"{key_of(d2)}.json").write_text(json.dumps(m2))
    tr2 = kvtier.Tier(root_c, 1.0, 16, 0, {"test": "c"}, gain=1)
    check("restart: a point with damaged ids is skipped and removed",
          key_of(d1) not in tr2.index and not (tr2.points / f"{key_of(d1)}.json").exists())
    check("restart: a point whose CRC list is short is skipped", key_of(d2) not in tr2.index)
    check("restart: the undamaged point stays", tr2.has(key_of(keep)))

    # -- part 1e: other builds' folders ----------------------------------------------------------------------------
    rb = root + "/builds"
    for name, age in (("glm-aaaa-r0", 300), ("glm-bbbb-r0", 100), ("glm-cccc-r1", 900)):
        os.makedirs(f"{rb}/{name}/points")
        fp = f"{rb}/{name}/fingerprint.json"
        with open(fp, "w") as f:
            f.write("{}")
        os.utime(fp, (time.time() - age, time.time() - age))
    os.environ.pop("TF_GLM_KV_TIER_KEEP_BUILDS", None)
    tb = kvtier.Tier(rb, 1.0, 16, 0, {"test": "b1"}, gain=1)
    check("other builds: the most recently started one is kept by default, older ones removed, another rank's kept",
          os.path.isdir(f"{rb}/glm-bbbb-r0") and not os.path.exists(f"{rb}/glm-aaaa-r0")
          and os.path.isdir(f"{rb}/glm-cccc-r1") and tb.pruned_builds == 1)
    kb = Snap(ids(15, 2 * B + 3), "B")
    write(tb, kb, src, 0)
    os.environ["TF_GLM_KV_TIER_KEEP_BUILDS"] = "0"
    tb2 = kvtier.Tier(rb, 1.0, 16, 0, {"test": "b1"}, gain=1)
    check("KEEP_BUILDS 0: a restart of the same build keeps its own folder and states", tb2.has(key_of(kb))
          and not os.path.exists(f"{rb}/glm-bbbb-r0") and os.path.isdir(f"{rb}/glm-cccc-r1"))
    os.environ.pop("TF_GLM_KV_TIER_KEEP_BUILDS")

    # -- part 1f: make's folder key ---------------------------------------------------------------------------------
    import tensorfold.cuda.spill as spill

    real_b, real_w = spill.build_id, spill.weights_id
    stamp = {"build": "b1", "weights": "w1"}
    spill.build_id = lambda m, d=None: stamp["build"]
    spill.weights_id = lambda *d: stamp["weights"]
    os.environ.update({"TF_GLM_KV_TIER": root + "/mk", "TF_GLM_KV_TIER_STAGE_MIB": "8",
                       "TF_GLM_KV_TIER_KEEP_BUILDS": "9"})
    small = arena(2 * B)

    def made(agreed=(1, 2, 3)):
        g = types.SimpleNamespace(spill_cfg=None, model_dir=root, drafter=None, drafter_dir=None, agreed=list(agreed))
        dec = types.SimpleNamespace(g=g, w=types.SimpleNamespace(world=1, rank=0), arena=small, grid=64,
                                    e=types.SimpleNamespace(caches=types.SimpleNamespace(kv="fp8"),
                                                            slots=types.SimpleNamespace(rec=torch.zeros(1, 1, 2, 3),
                                                                                        conv=torch.zeros(1, 4, 5))))
        return kvtier.make(dec).dir

    try:
        base_dir = made()
        same = made() == base_dir
        stamp["build"] = "b2"
        other_build = made() != base_dir
        stamp["build"] = "b1"
        stamp["weights"] = "w2"
        other_weights = made() != base_dir
        stamp["weights"] = "w1"
        other_agreed = made((1, 2, 4)) != base_dir     # e.g. another TF_GLM_PREFILL_ROWS
        real_format = kvtier.FORMAT
        kvtier.FORMAT = real_format + 1
        other_format = made() != base_dir
        kvtier.FORMAT = real_format
    finally:
        spill.build_id, spill.weights_id = real_b, real_w
        for k in ("TF_GLM_KV_TIER", "TF_GLM_KV_TIER_STAGE_MIB", "TF_GLM_KV_TIER_KEEP_BUILDS"):
            os.environ.pop(k, None)
    check("make: the same build and settings find the same folder", same)
    check("make: another build is another folder", other_build)
    check("make: other weight files are another folder", other_weights)
    check("make: other agreed settings (prefill rows among them) are another folder", other_agreed)
    check("make: another file format is another folder", other_format)
    # the real build key (spill.build_id: the family's and tensorfold.cuda's sources, the runtime) over a model folder
    with open(os.path.join(root, "config.json"), "w") as f:
        f.write('{"model_type": "glm5_next"}')
    os.environ.update({"TF_GLM_KV_TIER": root + "/mk2", "TF_GLM_KV_TIER_STAGE_MIB": "8"})
    try:
        real_dir = made()
        fp_real = json.loads((real_dir / "fingerprint.json").read_text())
        check("make: the real build key finds the same folder again and names the sources and the runtime",
              made() == real_dir and "sources=" in fp_real["build"] and "torch=" in fp_real["build"])
    finally:
        for k in ("TF_GLM_KV_TIER", "TF_GLM_KV_TIER_STAGE_MIB"):
            os.environ.pop(k, None)

    # -- part 2: the protocol -------------------------------------------------------------------------------------------
    class Net:
        def __init__(self):
            self.follower, self.box = None, None

    class G:
        cache_entries = 32

        def __init__(self, rank, net):
            self.rank, self.net = rank, net

        def _ring(self):
            pass

        def _share(self, msg):
            f = self.net.follower
            f.apply(M.unseal(msg, f.received, 1))
            f.received = (f.received + 1) % M.SEAL_MOD

        def _gather_ints(self, vals):
            if self.rank:
                self.net.box = list(vals)
                return None
            return [list(vals), self.net.box]

    def rank_decoder(rank, net, tier_, pool_rows, arena_):
        d = M.MultiDecoder.__new__(M.MultiDecoder)
        d.g, d.rank, d.w = G(rank, net), rank, types.SimpleNamespace(device="cpu")
        d.pool, d.arena, d.grid, d.rows_max = P.Pool(pool_rows), arena_, 64, 16
        d.tune = types.SimpleNamespace(async_msg=False, lone=False)
        d.fill_budget, d.partial, d.short_round_due, d.decode_due = 0.0, None, False, False
        d.lanes, d.kept, d.outbox, d.requeue = {}, [], [], []
        d.next_sid = d.next_kid = d.next_order = 0
        d.idle, d.broken, d.sent, d.received, d.watchdog, d.iteration_since = False, None, 0, 0, 0, None
        d.drafts, d.tier = None, tier_
        d.loading, d.load_ms, d.load_block_ms = None, kvtier.load_ms(), 2.0
        d.disk, d.kept_bytes_cap, d.next_chat = None, 0, 0
        return d

    def dummy_kept(d, kid, base, ids):
        x = d.pool.add(base, P.align_up(len(ids)))
        c = types.SimpleNamespace(ids=list(ids), kid=kid, extent=x, key=None, images=False, drafter_end=-1,
                                  drafter_rows=None, head=None, rec=None, conv=None, tail=None, shared=False)
        x.kept.append(c)
        d.kept.append(c)
        d.next_kid = max(d.next_kid, kid + 1)
        return c

    busy_lane = types.SimpleNamespace(decoding=True, paused=False, s=types.SimpleNamespace(done=False), extent=None)
    N2 = 6 * B + 640                                       # on the prompt grid (64): 7 blocks
    POOL = 12 * B
    real_step, step_log, tier_rank = kvtier.Loading.step, [], {}

    def logged_step(self, upto):
        step_log.append((tier_rank.get(id(self.tier)), self.next, int(upto)))
        return real_step(self, upto)

    kvtier.Loading.step = logged_step
    for case in ("good", "good, sliced", "follower lacks it", "follower block", "follower small",
                 "rank 0 block, sliced"):
        step_log.clear()
        net = Net()
        ranks, srcs, snaps = [], [], []
        for r in (0, 1):
            tt = kvtier.Tier(root + f"/proto-{case}-r{r}".replace(" ", "_"), 1.0, 1, r, {"test": "p", "rank": r},
                             gain=1)
            s_src = arena(POOL)
            sn = Snap(IDS[:N2], f"P-{case}")              # the same ids and lineage on both ranks, each rank its rows
            write(tt, sn, s_src, 0)
            ranks.append(rank_decoder(r, net, tt, POOL, arena(POOL)))
            tier_rank[id(tt)] = r
            srcs.append(s_src)
            snaps.append(sn)
        r0, r1 = ranks
        net.follower = r1
        key = key_of(snaps[0])
        for d in ranks:
            dummy_kept(d, 0, 0, IDS[:B])                   # the pool's own shorter hit
        hit = r0.kept[0]
        if case == "follower block":
            flip(r1.tier.blocks / f"{r1.tier.index[key]['blocks'][3]}.bin", 1000)
        elif case == "follower small":
            flip(r1.tier.points / f"{key}.pt", (r1.tier.points / f"{key}.pt").stat().st_size - 100)
        elif case == "rank 0 block, sliced":
            flip(r0.tier.blocks / f"{r0.tier.index[key]['blocks'][2]}.bin", 1000)
            r0.lanes[99] = busy_lane
            r0.load_ms = 0.01                             # one block a slice: blocks are left after the bad one
        elif case == "good, sliced":
            r0.lanes[99] = busy_lane
            r0.load_ms = 0.01                             # a slice of one block each (its copy takes longer)
        elif case == "follower lacks it":
            r1.tier.forget(key, ())
        before = [[(x.base, x.size) for x in d.pool.extents] for d in ranks]
        prompt = [int(v) for v in IDS[:N2]] + [11, 12, 13]
        try:
            got = r0._from_disk(prompt, hit)
            held = False
        except M.NoRoom:
            got, held = None, True
        if held:
            steps = 0
            while r0.loading is not None and steps < 400:
                time.sleep(0.01)
                r0._load_step()
                if r0.outbox:
                    r0._flush()
                steps += 1
            check(f"{case}: the sliced load ended (loading cleared on both ranks)",
                  r0.loading is None and r1.loading is None)
        if case.startswith("good"):
            ok = got is not None and got is not hit if case == "good" else held
            ok = ok and r0.loading is None and r1.loading is None
            for d, s_src, sn in zip(ranks, srcs, snaps):
                c = d.kept[-1]
                ok = ok and c.ids == sn.ids and rows_equal(s_src, d.arena, 0, c.extent.base, 0, N2)
                ok = ok and torch.equal(c.rec, sn.rec)
            check(f"{case}: the load becomes the same kept prompt on both ranks, each with its own rows", ok)
            if case == "good, sliced":
                pairs = [(step_log[i], step_log[i + 1]) for i in range(0, len(step_log) - 1, 2)]
                check(f"good, sliced: {len(pairs)} slices, each sent before rank 0 copied (the follower's first)",
                      len(step_log) % 2 == 0 and len(pairs) >= 2 and
                      all(a[0] == 1 and b[0] == 0 and a[1:] == b[1:] for a, b in pairs))
            continue
        if case == "follower lacks it":
            check("follower lacks it: KV_QUERY answers no, nothing loads, rank 0 keeps its point",
                  got is hit and not held and r0.loading is None and r1.loading is None and not step_log
                  and key in r0.tier.index and len(r0.kept) == 1
                  and [[(x.base, x.size) for x in d.pool.extents] for d in ranks] == before)
            continue
        if not held:
            check(f"{case}: _from_disk returns the pool's hit (the request prefills past it)", got is hit)
        check(f"{case}: no rank keeps the loaded state", all(len(d.kept) == 1 for d in ranks))
        check(f"{case}: the load's extent is freed on both ranks",
              [[(x.base, x.size) for x in d.pool.extents] for d in ranks] == before)
        check(f"{case}: every rank dropped the point", all(key not in d.tier.index and
                                                          not (d.tier.points / f"{key}.json").exists() for d in ranks))
        n_out = len(r0.outbox)
        again = r0._from_disk(prompt, hit)
        check(f"{case}: a later request does not load the same point again",
              again is hit and r0.loading is None and len(r0.outbox) == n_out)
        for d in ranks:
            d.tier._readers.shutdown(wait=True)
    kvtier.Loading.step = real_step
finally:
    shutil.rmtree(root, ignore_errors=True)

print("ALL PASS" if not fails else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
