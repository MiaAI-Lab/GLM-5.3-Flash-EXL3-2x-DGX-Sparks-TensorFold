#!/usr/bin/env python3
"""Checks for patch 0086 (the indexer's prompt scoring, one row and SCORE_LOOP pool blocks a program), run on a GPU
inside the image that scripts/prepare.sh built:

    docker run --rm --gpus all --entrypoint python -v "$PWD/tools/scores_loop_check.py:/c.py" tensorfold-glm53:v0.6.0 \
      /c.py

The patched sparse._select_prompt against a copy of it with the kernel before the patch (copied below, launched as
before: 4 rows and one 64-pool block a program, 4 warps): every score block each hands to top_pools is captured and
compared BIT FOR BIT (the whole [rows, NP] block, -inf columns included; buffers pre-filled so an unwritten score shows
up), then the selection's counts and counted rows' tokens. Cases, for FP8 and bf16 pooled keys: 512-row chunks at row
positions 0 (dense), 1,800 (across SPARSE_FROM), 35k, 113k, 300k and the cache's end; 1, 300 and 1,024 rows (two
blocks: the second block's offset). Needs ~3 GiB of GPU memory. Exit code 1 when a check fails.
"""
import sys

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import kv8, sparse
from tensorfold.families.glm5_next.cuda.kv8 import row_scales

H, D, KEY = 32, 128, 128
CAP = 1_048_576
dev = torch.device("cuda")
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"), flush=True)
    fails += 0 if ok else 1


@triton.jit
def _scores_before(QI, W, w_stride, PK, PKS, OUT, POS, R, NP, scale, wscale, H: tl.constexpr, HP: tl.constexpr,
                   D: tl.constexpr, BP: tl.constexpr, RB: tl.constexpr, RS: tl.constexpr, FP8: tl.constexpr):
    """sparse._scores before patch 0086, verbatim."""

    rb = tl.program_id(0)
    pb = tl.program_id(1)
    P = tl.load(POS)
    p = pb * BP + tl.arange(0, BP)
    visible = (P + tl.minimum((rb + 1) * RB, R)) // 4
    if pb * BP >= visible:
        for i in tl.static_range(RB):
            r = rb * RB + i
            if r < R:
                tl.store(OUT + r * NP + p, float("-inf"), mask=p < NP)
        return
    d = tl.arange(0, D)
    hh = tl.arange(0, HP)
    hok = hh < H
    kok = p < (P + rb * RB + RB) // 4
    k = tl.load(PK + p[:, None].to(tl.int64) * RS + d[None, :], mask=kok[:, None], other=0.0).to(tl.bfloat16)
    if FP8:
        ks = row_scales(PKS, p.to(tl.int64), kok, D, RS)
    for i in tl.static_range(RB):
        r = rb * RB + i
        if r < R:
            npool = (P + r + 1) // 4
            q = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.bfloat16)
            kr = tl.where((p < npool)[:, None], k, 0.0)
            dots = tl.dot(q, tl.trans(kr))
            if FP8:
                dots = dots * ks[None, :]
            w = tl.load(W + r * w_stride + hh, mask=hok, other=0.0).to(tl.float32) * wscale
            sc = tl.sum(w[:, None] * tl.maximum(dots * scale, 0.0), axis=0)
            sc = tl.where(p < npool, sc, float("-inf"))
            tl.store(OUT + r * NP + p, sc, mask=p < NP)


def old_select_prompt(qi, wts, pk, pos, R, np_max, pos_dev):
    """sparse._select_prompt before patch 0086 (the old kernel at its default 4 rows a program)."""
    NP = sparse.pool_count(pos, R, np_max)
    B = min(R, sparse.SELECT_ROWS)
    buf = torch.full((B * sparse.pool_bucket(pos, R, np_max),), 7.0, dtype=torch.float32, device=dev)
    pkv, pks, rs, fp8 = kv8.parts(pk)
    width = sparse.TOPK_POOLS * sparse.POOL + sparse.POOL - 1
    tokens = torch.empty((R, width), dtype=torch.int32, device=dev)
    counts = torch.empty((R,), dtype=torch.int32, device=dev)
    for a in range(0, R, B):
        n = min(B, R - a)
        at = pos_dev if a == 0 else pos_dev + a
        scores = buf[:n * NP].view(n, NP)
        _scores_before[(triton.cdiv(n, 4), triton.cdiv(NP, 64))](qi[a:a + n], wts[a:a + n], wts.stride(0), pkv, pks,
                                                                 scores, at, n, NP, D ** -0.5,
                                                                 1.0 / 5.656854249492381, H=H, HP=32, D=D, BP=64,
                                                                 RB=4, RS=rs, FP8=fp8, num_warps=4)
        pools = sparse.top_pools(scores, sparse.TOPK_POOLS)
        sparse._tokens[(n,)](pools, at, tokens[a:a + n], counts[a:a + n], W=width, K=sparse.TOPK_POOLS,
                             PL=sparse.POOL, BLOCK=1024, num_warps=4)
    return tokens, counts


captured = []
_top_pools = sparse.top_pools


def capturing_top_pools(scores, k, pos=None):
    captured.append(scores.clone())
    return _top_pools(scores, k, pos)


def run(fn, *args):
    """fn's tokens and counts, and the score blocks it handed to top_pools (the empty-allocated buffers are pre-filled
    with 7.0 so an unwritten score shows up)."""
    captured.clear()
    empty = torch.empty

    def filled(*a, **kw):
        t = empty(*a, **kw)
        if t.dtype == torch.float32:
            t.fill_(7.0)
        return t
    sparse.top_pools, torch.empty = capturing_top_pools, filled
    try:
        tokens, counts = fn(*args)
        torch.cuda.synchronize()
    finally:
        sparse.top_pools, torch.empty = _top_pools, empty
    return tokens, counts, list(captured)


check("patch 0086 applied: SCORE_LOOP 32", getattr(sparse, "SCORE_LOOP", None) == 32, f"{getattr(sparse, 'SCORE_LOOP', None)}")
torch.manual_seed(0)
raw = torch.randn(CAP // sparse.POOL + 2, KEY, device=dev)
caches = {"fp8": kv8.quantize_rows(raw), "bf16": raw.to(torch.bfloat16)}
np_max = raw.shape[0] - 2
cases = [(P, 512) for P in (0, 1_800, 35_000, 113_000, 300_000, CAP - 512)] + [(40_000, 1), (40_000, 300), (113_000, 1_024)]
for kind, pk in caches.items():
    for P, R in cases:
        qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
        ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
        wts = ikr[:, KEY:]
        pos_dev = torch.tensor([P], device=dev, dtype=torch.int64)
        t0, c0, s0 = run(old_select_prompt, qi, wts, pk, P, R, np_max, pos_dev)
        t1, c1, s1 = run(sparse._select_prompt, qi, wts, pk, P, R, np_max, pos_dev)
        bits = len(s0) == len(s1) and all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(s0, s1))
        diff = sum(int((a.view(torch.int32) != b.view(torch.int32)).sum()) for a, b in zip(s0, s1)) if len(s0) == len(s1) else -1
        check(f"{kind} {R} rows at {P}: every score bit ({len(s1)} block(s))", bits, f"{diff} differ, blocks {len(s0)}/{len(s1)}")
        counted = c0 > 0
        same = torch.equal(c0, c1) and torch.equal(t0[counted], t1[counted])
        check(f"{kind} {R} rows at {P}: counts and counted rows' tokens", same,
              f"counts differ {int((c0 != c1).sum())}, tokens differ {int((t0[counted] != t1[counted]).sum())}")
        del qi, ikr, s0, s1
    torch.cuda.empty_cache()
print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
