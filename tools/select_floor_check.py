#!/usr/bin/env python3
"""tools/select_floor_check.py [--time]: patch 0101 - the prompt rows' selection (sparse.prompt_pools / _select_floor:
one pass that compacts the pools at or above a sampled floor, then _select_rows' radix select over them in registers,
or _select_rows itself when the candidates miss [K, CAP]) - checked on the GPU against the reference (_top_pools: a
stable sort's k best, ties to the lower pool, ascending) and against the old path's _select_rows (BLOCK 1,024, 4 warps,
as top_pools launched it for prompt rows) for EVERY row. Run in the image built with patch 0101, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/select_floor_check.py:/c.py" <image> /c.py
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

Kernel cases, at rows of 1,000 .. 148,480 scores (every prompt_tile; 64 rows = BLOCK 1,024 and 8 rows = BLOCK 4,096):
random, bf16-rounded (ties, keys with zero low bits), a few distinct values, all equal, +0 / -0 ties, -inf past each
row's own visible pools (some rows with fewer than 512 visible: count 0), and rows built for each path (the floor
recomputed here from the sample's formula, so each crafted row is checked to take the path it is built for): ties at
the K-th best inside the candidates, 511 and 512 candidates (around K), CAP + 1 and CAP + 2,000 candidates with the
best pools last (dropped past CAP), exactly CAP, the K-th best below a masked lane's 0.0 (the last BLOCK 4,096 step
past the row's end); a row of NP candidates stores nothing past its 2 * CAP scratch, and a short scratch is refused.
End to end: sparse._select_prompt (patched) against a copy of the one before patch 0101 (top_pools), every token and
count of every row, and each score block's pools against _top_pools; FP8 and bf16 pooled keys; 512-row chunks at row
positions 0, 1,800, 35k, 113k, 300k, 450k, 590k and the cache's end; 1, 32, 64, 300, 577 (512 + 65), 520 (512 + 8) and
1,024 rows.

--time: the selection of one 512-row block, top_pools (before) against prompt_pools, CUDA events, on the scores
_scores gives random FP8 keys at 35k, 113k, 300k and 590k (the indexer is replicated: the same shapes at TP=2 and
TP=3)."""
import statistics
import sys

import torch
import triton

from tensorfold.families.glm5_next.cuda import kv8, sparse

H, D, KEY = 32, 128, 128
CAP = 1_048_576
K = sparse.TOPK_POOLS
dev = torch.device("cuda")
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"), flush=True)
    fails += 0 if ok else 1


def keys(x):
    """_order_key in torch (int64): unsigned order = the scores' order, -0 as +0."""
    bits = (x + 0.0).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 2 ** 31, bits ^ 0xFFFFFFFF, bits | 2 ** 31)


def sample_cols(np_):
    """_sample_floor's sample columns for rows of np_ scores (those at or past np_ load as 0.0)."""
    s = torch.arange(sparse.PROMPT_SECTORS * 8, device=dev)
    return ((s // 8) * np_ // sparse.PROMPT_SECTORS) // 8 * 8 + s % 8


def candidates(x):
    """Each row's candidate count by _select_floor's floor, recomputed (-1: the row is one tile)."""
    R, NP = x.shape
    cap, target = sparse.prompt_tile(NP)
    if NP <= cap:
        return torch.full((R,), -1, dtype=torch.int64, device=dev)
    k = keys(x)
    j = sample_cols(NP)
    su = torch.where(j < NP, k[:, j.clamp(max=NP - 1)], 2 ** 31)
    jj = max(target * sparse.PROMPT_SECTORS * 8 // NP, 1)
    floor = (torch.topk(su, jj, dim=1).values[:, -1:] >> 16) << 16
    return (k >= floor).sum(1)


def prod_rows(x):
    out = torch.empty((x.shape[0], K), dtype=torch.int64, device=dev)
    sparse._select_rows[(x.shape[0],)](x, out, x.shape[1], x, K=K, BLOCK=1024, VIS=False, num_warps=4)
    return out


paths = {"tile": 0, "candidates": 0, "fallback": 0}


def kernel_case(name, x, want=None):
    """prompt_pools(x) against _top_pools and the old path's _select_rows, every row; want: the path every row must
    take ("tile", "candidates", "fallback")."""
    x = x.contiguous()
    cap = sparse.prompt_tile(x.shape[1])[0]
    n = candidates(x)
    kind = torch.where(n < 0, 0, torch.where((n >= K) & (n <= cap), 1, 2))
    for i, p in enumerate(paths):
        paths[p] += int((kind == i).sum())
    if want is not None:
        w = list(paths).index(want)
        check(f"{name}: every row takes the {want} path", bool((kind == w).all()), f"candidates {n.tolist()[:8]}")
    ref = sparse._top_pools(x, K)
    got = sparse.prompt_pools(x, K)
    old = prod_rows(x)
    torch.cuda.synchronize()
    bad = (got != ref).any(1)
    check(f"{name}: pools = _top_pools and _select_rows, every row", torch.equal(got, ref) and torch.equal(old, ref),
          f"{int(bad.sum())} rows differ (first {bad.nonzero()[:4].flatten().tolist()}), "
          f"_select_rows equal {torch.equal(old, ref)}")


check("patch 0101 applied: prompt_pools", hasattr(sparse, "prompt_pools") and hasattr(sparse, "_select_floor"))
if fails:
    print("1 FAIL", flush=True)
    sys.exit(1)
torch.manual_seed(0)
for NP in (1_000, 1_024, 4_096, 5_120, 9_216, 12_288, 13_312, 28_672, 32_768, 33_792, 75_776, 148_480):
    for R in (64, 8):
        g = torch.Generator(device=dev).manual_seed(NP + R)
        x = torch.randn(R, NP, device=dev, generator=g)
        kernel_case(f"{NP} x {R} random", x)
        kernel_case(f"{NP} x {R} bf16-rounded", x.to(torch.bfloat16).float())
        kernel_case(f"{NP} x {R} 8 values", torch.randint(0, 8, (R, NP), device=dev, generator=g).float())
        kernel_case(f"{NP} x {R} all equal", torch.full((R, NP), 0.25, device=dev))
        z = torch.where(torch.rand(R, NP, device=dev, generator=g) < 0.5, -0.0, 0.0)
        z[:, :400] = 1.0
        z[:, 700:] = -1.0 - torch.rand(R, NP - 700, device=dev, generator=g)
        kernel_case(f"{NP} x {R} +0 / -0 ties at the K-th", z[:, torch.randperm(NP, device=dev, generator=g)])
        vis = torch.randint(0, NP + 1, (R,), device=dev, generator=g)
        vis[: R // 4] = torch.randint(0, K + 1, (R // 4,), device=dev, generator=g)       # rows with count 0
        y = torch.where(torch.arange(NP, device=dev) < vis[:, None], x, float("-inf"))
        kernel_case(f"{NP} x {R} -inf past each row's pools", y)

# crafted rows for the candidates' paths (rows of 148,480 scores: CAP 4,096)
NP = 148_480
cap = sparse.prompt_tile(NP)[0]
cols = sample_cols(NP)
rest = torch.ones(NP, dtype=torch.bool, device=dev)
rest[cols] = False
others = rest.nonzero().flatten()
g = torch.Generator(device=dev).manual_seed(78)
x = torch.randn(8, NP, device=dev, generator=g) * 0.5                                   # below 4.0
spots = others[torch.randperm(others.numel(), device=dev, generator=g)[:700]]
x[:, spots[:400]] = 5.0
x[:, spots[400:]] = 4.0                                                                  # the K-th best: a 4.0 tie
kernel_case("crafted ties at the K-th best among the candidates", x, "candidates")
x = torch.rand(8, NP, device=dev, generator=g)
x[:, cols[:511]] = 9.0
kernel_case("crafted 511 candidates (one short of K)", x, "fallback")
x = torch.rand(8, NP, device=dev, generator=g)
x[:, cols[:512]] = 9.0
kernel_case("crafted 512 candidates (K)", x, "candidates")
x = torch.full((8, NP), 0.5, device=dev)
x[:, cols] = 1.0
x[:, others[-2000:]] = 2.0                                                               # the best pools come last
kernel_case("crafted CAP + 2,000 candidates, the best last", x, "fallback")
x = torch.full((8, NP), 0.5, device=dev)
x[:, cols] = 1.0
x[:, others[-(cap - cols.numel() + 1):]] = 2.0
kernel_case("crafted CAP + 1 candidates, the best last", x, "fallback")
x = torch.full((8, NP), 0.5, device=dev)
x[:, cols[:cap // 2]] = 1.0
x[:, others[-(cap - cap // 2):]] = 2.0
kernel_case("crafted exactly CAP candidates, the best last", x, "candidates")
NP = 16_000                                         # 8 rows: BLOCK 4,096, the last step's 384 lanes past NP
cols = sample_cols(NP)
rest = torch.ones(NP, dtype=torch.bool, device=dev)
rest[cols] = False
x = -1.0 - torch.rand(8, NP, device=dev, generator=g)                                  # below a masked lane's 0.0
x[:, rest.nonzero().flatten()[:400]] = 1.0                                              # unsampled: the floor is < 0
kernel_case("crafted K-th best below 0.0 with lanes past the row", x, "candidates")
NP = 148_480
cap = sparse.prompt_tile(NP)[0]
scratch = torch.full((2 * cap + NP,), 0x5A5A5A5A, dtype=torch.int32, device=dev)
sparse.prompt_pools(torch.full((1, NP), 0.25, device=dev), K, scratch)
torch.cuda.synchronize()
check("a row of NP candidates stores nothing past its 2 * CAP scratch", bool((scratch[2 * cap:] == 0x5A5A5A5A).all()),
      f"{int((scratch[2 * cap:] != 0x5A5A5A5A).sum())} written")
try:
    sparse.prompt_pools(torch.zeros((2, NP), device=dev), K, scratch[:4 * cap - 1])
    refused = False
except ValueError:
    refused = True
check("prompt_pools refuses a scratch under R * 2 * CAP int32", refused)
check(f"every path taken (rows: {paths})", all(v > 0 for v in paths.values()))


def old_select_prompt(qi, wts, pk, pos, R, np_max, pos_dev):
    """sparse._select_prompt before patch 0101 (top_pools), verbatim otherwise."""
    dev = qi.device
    NP = sparse.pool_count(pos, R, np_max)
    B = min(R, sparse.SELECT_ROWS)
    buf = torch.empty((B * sparse.pool_bucket(pos, R, np_max),), dtype=torch.float32, device=dev)
    H = wts.shape[1]
    D = qi.shape[1] // H
    wscale = 1.0 / 5.656854249492381 if H == 32 else H ** -0.5
    width = sparse.TOPK_POOLS * sparse.POOL + sparse.POOL - 1
    tokens = torch.empty((R, width), dtype=torch.int32, device=dev)
    counts = torch.empty((R,), dtype=torch.int32, device=dev)
    pkv, pks, rs, fp8 = kv8.parts(pk)
    for a in range(0, R, B):
        n = min(B, R - a)
        at = pos_dev if a == 0 else pos_dev + a
        scores = buf[:n * NP].view(n, NP)
        sparse._scores[(n, triton.cdiv(triton.cdiv(NP, 64), sparse.SCORE_LOOP))](
            qi[a:a + n], wts[a:a + n], wts.stride(0), pkv, pks, scores, at, n, NP, D ** -0.5, wscale, H=H,
            HP=max(16, triton.next_power_of_2(H)), D=D, BP=64, L=sparse.SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)
        pools = sparse.top_pools(scores, sparse.TOPK_POOLS)
        sparse._tokens[(n,)](pools, at, tokens[a:a + n], counts[a:a + n], W=width, K=sparse.TOPK_POOLS, PL=sparse.POOL,
                             BLOCK=1024, num_warps=4)
    return tokens, counts


blocks = []
_prompt_pools = sparse.prompt_pools


def checking_prompt_pools(scores, k, scratch=None):
    out = _prompt_pools(scores, k, scratch)
    blocks.append(bool(torch.equal(out, sparse._top_pools(scores, k))))
    return out


def run(fn, *args):
    """fn's tokens and counts, every empty-allocated buffer pre-filled (an unwritten or stale value shows up)."""
    empty = torch.empty

    def filled(*a, **kw):
        t = empty(*a, **kw)
        t.fill_(7.0 if t.dtype == torch.float32 else 0x5A5A5A5A if t.dtype == torch.int32 else -5)
        return t
    sparse.prompt_pools, torch.empty = checking_prompt_pools, filled
    try:
        tokens, counts = fn(*args)
        torch.cuda.synchronize()
    finally:
        sparse.prompt_pools, torch.empty = _prompt_pools, empty
    return tokens, counts


raw = torch.randn(CAP // sparse.POOL + 2, KEY, device=dev)
caches = {"fp8": kv8.quantize_rows(raw), "bf16": raw.to(torch.bfloat16)}
np_max = raw.shape[0] - 2
cases = ([(P, 512) for P in (0, 1_800, 35_000, 113_000, 300_000, 450_000, 590_000, CAP - 512)]
         + [(40_000, 1), (590_000, 32), (40_000, 64), (300_000, 300), (300_000, 577), (590_000, 520), (113_000, 1_024)])
for kind, pk in caches.items():
    for P, R in cases:
        qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
        ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
        wts = ikr[:, KEY:]
        pos_dev = torch.tensor([P], device=dev, dtype=torch.int64)
        t0, c0 = run(old_select_prompt, qi, wts, pk, P, R, np_max, pos_dev)
        blocks.clear()
        t1, c1 = run(sparse._select_prompt, qi, wts, pk, P, R, np_max, pos_dev)
        check(f"{kind} {R} rows at {P}: each block's pools = _top_pools ({len(blocks)} block(s))",
              len(blocks) == triton.cdiv(R, sparse.SELECT_ROWS) and all(blocks), f"{blocks}")
        check(f"{kind} {R} rows at {P}: every row's tokens and count", torch.equal(c0, c1) and torch.equal(t0, t1),
              f"counts differ {int((c0 != c1).sum())}, tokens differ {int((t0 != t1).sum())}")
        del qi, ikr, t0, t1
    torch.cuda.empty_cache()


def bench(fn, reps=20):
    """Median of reps timed calls (CUDA events) after 3 warm-up calls, in ms."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


if "--time" in sys.argv and fails == 0:
    pk = caches["fp8"]
    pkv, pks, rs, fp8 = kv8.parts(pk)
    R = 512
    for P in (35_000, 113_000, 300_000, 590_000):
        NP = sparse.pool_count(P, R, np_max)
        qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
        wts = torch.randn(R, H, device=dev, dtype=torch.bfloat16)
        scores = torch.empty((R, NP), device=dev)
        sparse._scores[(R, triton.cdiv(triton.cdiv(NP, 64), sparse.SCORE_LOOP))](
            qi, wts, wts.stride(0), pkv, pks, scores, torch.tensor([P], device=dev), R, NP, D ** -0.5, 1 / 32 ** 0.5,
            H=H, HP=H, D=D, BP=64, L=sparse.SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)
        cand = torch.empty((R * 2 * sparse.prompt_tile(NP)[0],), dtype=torch.int32, device=dev)
        t0 = bench(lambda: sparse.top_pools(scores, K))
        t1 = bench(lambda: sparse.prompt_pools(scores, K, cand))
        print(f"TIME {R} rows at {P}: {NP} pools, top_pools {t0:.3f} ms -> prompt_pools {t1:.3f} ms "
              f"({100 * (t1 / t0 - 1):+.1f}%)", flush=True)
        del scores, cand, qi, wts
print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
