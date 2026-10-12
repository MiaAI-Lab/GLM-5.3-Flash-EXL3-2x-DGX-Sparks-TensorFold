#!/usr/bin/env python3
"""Checks for patch 0085 (the split selection's last step over the chunks before its own only), run on a GPU inside
the image that scripts/prepare.sh built:

    docker run --rm --gpus all --entrypoint python -v "$PWD/tools/select_split_check.py:/c.py" tensorfold-glm53:v0.6.0 \
      /c.py

The patched sparse.select_split against the kernel before the patch (copied below; the whole token and count
buffers) and against an independent torch oracle (each sparse row's 512 best complete pools, ties to the lower pool,
ascending, then its incomplete pool's tokens):
  - segmented windows through seg_select_tokens (the indexer's own scores) at the engine's scratch width (a pool of
    5,791,744 rows: 512 chunk programs a row), a stream-sized one (1,048,592: 128) and a small one (65,536: 8 programs
    of 2,048 scores); 1 to 4 streams, dense and sparse rows, rows across a chunk boundary, 2,060 to 1.04M tokens
  - tie-heavy (8 values), all-equal and signed-zero / -inf / huge scores straight into select_split
  - SPLIT_BLK 1, 4 and 32 give the bits of the default 16
Needs ~1 GiB of GPU memory (a free Spark, or the server stopped). Exit code 1 when a check fails.
"""
import sys
from functools import partial

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import kv8, segments, sparse
from tensorfold.families.glm5_next.cuda.sparse import _order_key, _split_digits, _visible_bound

H, D, KEY = 32, 128, 128
WIDTHS = {"cp512": 5_791_744, "cp128": 1_048_592, "cp8": 65_536}
K, PL, W = sparse.TOPK_POOLS, sparse.POOL, sparse.TOKENS
dev = torch.device("cuda")
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"), flush=True)
    fails += 0 if ok else 1


@triton.jit
def _select_split_before(S, s_stride, NPC, POS, SPR, TOT, HIST, GT, OUT, TOK, CNT, W: tl.constexpr,
                           K: tl.constexpr, PL: tl.constexpr, CHS: tl.constexpr, CP: tl.constexpr,
                           STEP: tl.constexpr, SEG: tl.constexpr, VIS: tl.constexpr, TOKS: tl.constexpr,
                           BLK: tl.constexpr):
    """_select_split before patch 0085 (step 4: one CP x 256 tile masked to the earlier chunks); BLK unused."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    if SEG:
        if tl.load(SPR + r) == 0:
            if STEP == 4:
                if c == 0:
                    tl.store(CNT + r, 0)
            return
        qpos = tl.load(POS + r).to(tl.int64)
        NP = (qpos + 1) // PL
        npool = NP
    else:
        if VIS:
            qpos = tl.load(POS).to(tl.int64) + r
            NP = _visible_bound(NPC, tl.load(POS), r, K)
        else:
            qpos = r * 0
            NP = NPC
        npool = (qpos + 1) // PL
    nch = (NP + CHS - 1) // CHS
    if c >= nch:
        return
    bins = tl.arange(0, 256)
    i = c * CHS + tl.arange(0, CHS)
    ok = i < NP
    u = _order_key(tl.load(S + r * s_stride + i, mask=ok, other=0.0))
    prefix, fixed, need, d = _split_digits(TOT, r, STEP, K)
    own = HIST + (r * CP + c) * 256
    if STEP > 0 and STEP < 4:
        above = tl.sum(tl.where(bins > d, tl.load(own + bins), 0), 0)
        if STEP > 1:
            above += tl.load(GT + r * CP + c)
        tl.store(GT + r * CP + c, above)
    if STEP < 4:
        match = ok & ((u & fixed) == prefix)
        hist = tl.histogram(((u >> (24 - 8 * STEP)) & 0xFF).to(tl.int32), 256, mask=match)
        tl.store(own + bins, hist)
        tl.atomic_add(TOT + (r * 4 + STEP) * 256 + bins, hist)
    else:
        ci = tl.arange(0, CP)
        before = ci < c
        last = tl.load(HIST + (r * CP + ci[:, None]) * 256 + bins[None, :], mask=before[:, None], other=0)
        gt = tl.load(GT + r * CP + ci, mask=before, other=0) + tl.sum(tl.where(bins[None, :] > d, last, 0), 1)
        equal_seen = tl.sum(tl.sum(tl.where(bins[None, :] == d, last, 0), 1), 0)
        written = tl.sum(gt, 0) + tl.minimum(equal_seen, need)
        e = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((e == 1) & (tl.cumsum(e, 0) - e + equal_seen < need))
        t = take.to(tl.int32)
        slot = (written + tl.cumsum(t, 0) - t).to(tl.int64)
        if SEG or TOKS:
            j = tl.arange(0, PL)
            tl.store(TOK + r * W + slot[:, None] * PL + j[None, :],
                     (i.to(tl.int64)[:, None] * PL + j[None, :]).to(tl.int32), mask=take[:, None])
            if c == 0:
                jt = tl.arange(0, 4)
                jok = jt < PL - 1
                tail = npool * PL + jt
                tok = jok & (tail <= qpos)
                tl.store(TOK + r * W + K * PL + jt, tl.where(tok, tail, -1).to(tl.int32), mask=jok)
                n = K * PL + tl.sum(tok.to(tl.int64), 0)
                tl.store(CNT + r, tl.where(npool > K, n, 0).to(tl.int32))
        else:
            tl.store(OUT + r * K + slot, i.to(tl.int64), mask=take)


PATCHED = sparse._select_split


def order_key(s):
    """_order_key in int64: the float32 scores' order as unsigned 32-bit keys (-0 counted as +0)."""
    b = (s.float() + 0.0).view(torch.int32).to(torch.int64)
    return (b & 0xFFFFFFFF) ^ torch.where(b < 0, torch.full_like(b, 0xFFFFFFFF), torch.full_like(b, 0x80000000))


def oracle(scores, pos, spr):
    """Each sparse row's K best complete pools (ties to the lower pool) ascending as tokens, its incomplete pool's
    tokens after them (-1 past its position), and its count (0: dense rows, and rows of K pools or fewer)."""
    R = scores.shape[0]
    toks = torch.full((R, W), -1, dtype=torch.int32)
    cnt = torch.zeros(R, dtype=torch.int32)
    for r in range(R):
        q = int(pos[r])
        npool = (q + 1) // PL
        if not int(spr[r]) or npool <= K:
            continue
        best = torch.sort(-order_key(scores[r, :npool]), stable=True).indices[:K]
        sel = torch.sort(best).values
        toks[r, :K * PL] = (sel[:, None] * PL + torch.arange(PL)).reshape(-1).to(torch.int32)
        n = K * PL
        for j in range(PL - 1):
            t = npool * PL + j
            toks[r, K * PL + j] = t if t <= q else -1
            n += int(t <= q)
        cnt[r] = n
    return toks, cnt


def make(contexts, rows_each):
    R = sum(rows_each)
    rows = segments.SegRows(R, dev, max_segs=max(segments.SEGS_DEFAULT, len(contexts)))
    segs, base = [], 0
    for ctx, n in zip(contexts, rows_each):
        segs.append((base, ctx - n, n))
        base += -(-ctx // segments.EXTENT) * segments.EXTENT
    pk = kv8.quantize_rows(torch.randn(base // PL + 2, KEY, device=dev))
    rows.set(segs)
    torch.cuda.synchronize()
    qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
    ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
    return qi, ikr[:, KEY:], pk, rows, R


def run(fn, scratch, R, kernel=None):
    """fn's tokens and counts (buffers reset first, so every unwritten slot compares too)."""
    scratch.tokens.fill_(-1)
    scratch.counts.fill_(-7)
    if kernel is not None:
        sparse._select_split = kernel
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        sparse._select_split = PATCHED
    return scratch.tokens[:R].cpu(), scratch.counts[:R].cpu()


def compare(name, got, ref, o, pos, spr):
    same = torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    bad = [r for r in range(got[0].shape[0]) if int(got[1][r]) != int(o[1][r])
           or (int(o[1][r]) > 0 and not torch.equal(got[0][r], o[0][r]))]
    check(f"{name}: the bits before the patch", same,
          f"{int((got[0] != ref[0]).sum())} tokens, {int((got[1] != ref[1]).sum())} counts differ")
    check(f"{name}: oracle", not bad, f"rows {bad[:6]} (pos {[int(pos[r]) for r in bad[:6]]}, "
          f"count {[int(got[1][r]) for r in bad[:6]]} vs {[int(o[1][r]) for r in bad[:6]]})")


def special(shape, kind, npool_max):
    g = torch.Generator(device=dev).manual_seed(5)
    s = torch.zeros(shape, device=dev)
    if kind == "ties":
        s[:, :npool_max] = torch.randint(0, 8, (shape[0], npool_max), device=dev, generator=g).float()
    elif kind == "special":
        v = torch.randn(shape[0], npool_max, device=dev, generator=g)
        pick = torch.randint(0, 8, v.shape, device=dev, generator=g)
        v = torch.where(pick == 0, torch.full_like(v, -0.0), v)
        v = torch.where(pick == 1, torch.full_like(v, float("-inf")), v)
        v = torch.where(pick == 2, torch.full_like(v, 3.0e38), v)
        v = torch.where(pick == 3, torch.full_like(v, -3.0e38), v)
        v = torch.where(pick == 4, torch.zeros_like(v), v)
        s[:, :npool_max] = v
    return s                                      # "equal": all zeros


def seg_case(name, contexts, rows_each, widths, blk_sweep=False):
    qi, wts, pk, rows, R = make(contexts, rows_each)
    pos, spr = rows.pos[:R].cpu(), rows.sparse[:R].cpu()
    npool_max = (int(pos.max()) + 1) // PL
    for wname in widths:
        scratch = segments.SelectScratch(R, WIDTHS[wname], dev)
        label = f"{name} {wname}"
        call = partial(sparse.seg_select_tokens, qi, wts, pk, rows, scratch)
        got = run(call, scratch, R)
        scores = scratch.scores[:R].cpu()
        ref = run(call, scratch, R, _select_split_before)
        compare(label, got, ref, oracle(scores, pos, spr), pos, spr)
        for kind in ("ties", "equal", "special"):
            s = special(scratch.scores[:R].shape, kind, npool_max)
            direct = partial(sparse.select_split, s, K, scratch=scratch.split, rows=rows, tokens=scratch.tokens[:R],
                             counts=scratch.counts[:R])
            compare(f"{label} {kind} scores", run(direct, scratch, R), run(direct, scratch, R, _select_split_before),
                    oracle(s.cpu(), pos, spr), pos, spr)
            del s, direct
        if blk_sweep:
            for blk in (1, 4, 32):
                sparse.SPLIT_BLK = blk
                try:
                    alt = run(call, scratch, R)
                finally:
                    sparse.SPLIT_BLK = 16
                check(f"{label}: SPLIT_BLK {blk} = 16's bits", torch.equal(alt[0], got[0]) and torch.equal(alt[1], got[1]))
        del scratch, call
        torch.cuda.empty_cache()


check("patch 0085 applied: SPLIT_BLK 16", getattr(sparse, "SPLIT_BLK", None) == 16, f"{getattr(sparse, 'SPLIT_BLK', None)}")
torch.manual_seed(0)
seg_case("1 x 16 @ 2,060 (dense/sparse edge)", [2_060], [16], ("cp512", "cp128", "cp8"))
seg_case("1 x 16 @ 49,160 (chunk edge)", [49_160], [16], ("cp512", "cp128", "cp8"))
seg_case("1 x 16 @ 35k", [35_000], [16], ("cp512", "cp128", "cp8"))
seg_case("1 x 16 @ 226k", [226_000], [16], ("cp512", "cp128"))
seg_case("1 x 16 @ 590k", [590_000], [16], ("cp512", "cp128"), blk_sweep=True)
seg_case("1 x 3 @ 1.04M", [1_040_000], [3], ("cp512", "cp128"))
seg_case("4 streams mixed", [1_000, 50_000, 300_000, 700_000], [3, 8, 5, 16], ("cp512", "cp128"), blk_sweep=True)
print(f"{'ALL PASS' if fails == 0 else f'{fails} FAIL'}", flush=True)
sys.exit(1 if fails else 0)
