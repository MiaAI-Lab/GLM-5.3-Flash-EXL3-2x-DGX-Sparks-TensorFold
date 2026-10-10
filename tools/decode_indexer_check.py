#!/usr/bin/env python3
"""tools/decode_indexer_check.py [--time]: patch 0107 - the DSA indexer of a decode window (sparse.seg_select_tokens):
  - the Triton scoring's grid, SEG_SCORE_GRID, 512 programs a segment (was 256): speed only;
  - the scoring of segments of SEG_SCORE_CUDA_ROWS (4) or more rows from seg_scores.cu, which stores _seg_scores'
    scores with the prompt scoring's instruction sequence (prompt_scores.cu, patch 0105), the segments of fewer rows
    from _seg_scores as before (TF_GLM_SEG_SCORES_CUDA=0: _seg_scores for all);
  - the selection of windows of SEG_FLOOR_ROWS (8) or more rows by _seg_select_floor, one program a sparse row and one
    pass over its scores (patch 0101's method), instead of select_split's five passes (TF_GLM_SEG_SELECT_FLOOR=0:
    select_split).
Run in the image built with patch 0107, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/decode_indexer_check.py:/c.py" <image> /c.py

Scoring, through sparse.seg_select_tokens with TF_GLM_SEG_SCORES_CUDA=1 against =0: the whole scores scratch
(pre-filled with a NaN sentinel, so every store and every non-store counts), tokens and counts bit-identical:
windows of 1-6 streams and 1-16 rows a stream at 180k-600k (the scratch as wide as the engine's: the whole pool), a
stream straddling the dense limit beside a dense segment, unaligned positions, uneven rows; every non-NaN e4m3 code,
arbitrary key scales, large / tiny / zero queries, negative head weights; other CTA counts and stages (speed only);
one CUDA graph replayed over new positions; the routing (the extension and _seg_scores each run once, over the pool
table the extension writes; bf16 caches stay on Triton); the extension alone stores its own segments only; the
extension's pool table sized for any window (MAX_SEGS) at its first use and never replaced, so a window of more
segments after a graph was captured does not move it.
Selection, _seg_select_floor against select_split, the whole tokens [R, 2051] and counts [R]: the entry point at 8-32
rows; crafted scores (tie-heavy: more than CAP candidates, bf16-rounded, -inf / -0.0 / +0.0, a sample that misleads
the floor: fewer than K candidates, plain random), rows sparse below the dense limit (count 0), an overflowing row
that must store nothing past its own 2 * CAP scratch words; other speed constants; one CUDA graph over new positions;
the routing (8+ rows on, fewer rows / =0 / a scratch too small: select_split).
Grid: the Triton scoring launched with min(512, pool blocks) programs a segment, the same scores as with 256.
Start-up: decode.Engine's constructor loads (on a fresh image, builds) the scoring extension before any request for
FP8 caches with DSA layers and the switch on; not with TF_GLM_SEG_SCORES_CUDA=0, bf16 caches or no DSA layer.
Not used - a failed build or load, or another Triton release than sparse.TRITON_PTX: nothing raised, logged once
and not retried, the switch reads off, and the window gets the Triton scoring's scores, tokens and counts.
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

--time: the whole indexer call (scores + selection; cold: 11 calls a CUDA graph, each its own layer's keys, L2 flushed
between replays), v1.10 (grid 256, both switches 0) -> grid 512 -> + CUDA scoring -> + one-pass selection. The
indexer is replicated: its shapes are the same at TP=2 and TP=3."""
import os
import statistics
import sys
from types import SimpleNamespace

import torch

from tensorfold.families.glm5_next.cuda import decode, forward, hcsplit, kv8, sparse
from tensorfold.families.glm5_next.cuda.segments import EXTENT, SegRows, SelectScratch

dev = torch.device("cuda")
IH, ID = 32, 128
POOL_ROWS = 5_791_744
SENT = 0x7FC0BEEF
fails = 0
g = torch.Generator(device=dev).manual_seed(95)


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


check(all(hasattr(sparse, a) for a in ("seg_scores_cuda", "_seg_scores_ext", "seg_select_floor", "_seg_select_floor")),
      "patch 0107 applied: sparse.seg_scores_cuda, sparse.seg_select_floor")
if fails:
    print("1 FAIL", flush=True)
    sys.exit(1)
SHIPPED = (sparse.SEG_SCORE_CTAS, sparse.SEG_SCORE_STAGES)
FLOOR = (sparse.SEG_FLOOR_ROWS, sparse.SEG_FLOOR)
flush = torch.empty((256 << 20,), dtype=torch.uint8, device=dev)


def layout(streams, room=0):
    """[(ctx, rows)] -> segments [(base, pos, rows)] and the arena's token rows (``room``: tokens past each stream's
    window inside its extent, for positions moved later)."""
    segs, base = [], 0
    for ctx, n in streams:
        segs.append((base, ctx, n))
        base += -(-(ctx + n + 64 + room) // EXTENT) * EXTENT
    return segs, base


def keys(rows_, adversarial=False):
    if not adversarial:
        return kv8.quantize_rows(torch.randn(rows_, ID, device=dev, generator=g) *
                                 torch.rand(rows_, 1, device=dev, generator=g) * 4)
    pk = kv8.quantize_rows(torch.zeros(rows_, ID, device=dev))
    codes = torch.randint(0, 255, (rows_, ID), device=dev, generator=g, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes)            # no NaN codes (0x7F / 0xFF)
    pk[:, :ID] = codes.to(torch.uint8)
    sc = torch.rand(rows_, device=dev, generator=g) * 3 + 1e-3                  # arbitrary scales
    pk[:, ID:ID + 4] = sc.view(torch.uint8).view(rows_, 4)
    return pk


def queries(R, adversarial=False):
    qi = torch.randn(R, IH * ID, device=dev, generator=g)
    wts = torch.randn(R, IH, device=dev, generator=g)
    if adversarial:
        mag = torch.tensor([1e-30, 1e-3, 1.0, 1e3, 3e4], device=dev)
        pick = mag[torch.randint(0, 5, (R, IH, 1), device=dev, generator=g)]
        qi = qi * pick.reshape(R, IH, 1).expand(R, IH, ID).reshape(R, -1)
        qi[:, ::7] = 0.0
        qi[:, 3::11] = -0.0
        wts = wts * 100
    return qi.to(torch.bfloat16), wts.to(torch.bfloat16)


def window(streams, adversarial=False, max_segs=None, room=0):
    segs, base = layout(streams, room)
    R = sum(n for _, n in streams)
    rows = SegRows(R, dev, max_segs=max_segs or max(4, len(streams)))
    rows.set(segs)
    pk = keys(base // sparse.POOL + 2, adversarial)
    qi, wts = queries(R, adversarial)
    return qi, wts, pk, rows, R, segs


def run(qi, wts, pk, rows, sel, cuda):
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "1" if cuda else "0"
    sel.scores.view(torch.int32).fill_(SENT)
    sel.tokens.fill_(-7)
    sel.counts.fill_(-7)
    t, c = sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    torch.cuda.synchronize()
    return sel.scores.clone(), t.clone(), c.clone()


def same3(a, b):
    return (torch.equal(a[0].view(torch.int32), b[0].view(torch.int32)) and torch.equal(a[1], b[1])
            and torch.equal(a[2], b[2]))


def bits(name, streams, adversarial=False, arms=(None,)):
    qi, wts, pk, rows, R, _ = window(streams, adversarial)
    sel = SelectScratch(R, POOL_ROWS, dev)
    ref = run(qi, wts, pk, rows, sel, False)
    stored = int((ref[0].view(torch.int32)[:R] != SENT).sum())
    for arm in arms:
        sparse.SEG_SCORE_CTAS, sparse.SEG_SCORE_STAGES = SHIPPED if arm is None else arm
        got = run(qi, wts, pk, rows, sel, True)
        diff = int((got[0].view(torch.int32) != ref[0].view(torch.int32)).sum())
        check(same3(got, ref), f"scores {name}, CTAs {sparse.SEG_SCORE_CTAS} stages {sparse.SEG_SCORE_STAGES}: "
                               f"scores ({stored} stored), tokens, counts bit-identical ({diff} score words differ)")
    sparse.SEG_SCORE_CTAS, sparse.SEG_SCORE_STAGES = SHIPPED
    del sel
    torch.cuda.empty_cache()


def entry(qi, wts, pk, rows, sel, floor):
    os.environ["TF_GLM_SEG_SELECT_FLOOR"] = "1" if floor else "0"
    sel.tokens.fill_(-7)
    sel.counts.fill_(-7)
    t, c = sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    torch.cuda.synchronize()
    return t.clone(), c.clone()


def direct(scores, rows, sel, R, floor, sync=True):
    """One selection over given scores [R, NP]: _seg_select_floor as seg_select_tokens launches it, or select_split."""
    if sync:
        sel.tokens.fill_(-7)
        sel.counts.fill_(-7)
    tok, cnt = sel.tokens[:R], sel.counts[:R]
    if floor:
        cap, target, block, warps = sparse.SEG_FLOOR
        sparse._seg_select_floor[(R,)](scores, scores.stride(0), rows.pos, rows.sparse, tok, cnt, sel.split,
                                       W=sparse.TOKENS, K=sparse.TOPK_POOLS, PL=sparse.POOL, CAP=cap,
                                       NS=sparse.PROMPT_SECTORS, TARGET=target, BLOCK=block, num_warps=warps)
    else:
        sparse.select_split(scores, sparse.TOPK_POOLS, scratch=sel.split, rows=rows, tokens=tok, counts=cnt)
    if sync:
        torch.cuda.synchronize()
        return tok.clone(), cnt.clone()


def same2(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def differ(a, b):
    return int((a[0] != b[0]).sum()) + int((a[1] != b[1]).sum())


def sampled(NP):
    """The columns _sample_floor reads in a row of NP scores."""
    s = torch.arange(sparse.PROMPT_SECTORS * 8, device=dev)
    return torch.unique(((s // 8) * NP // sparse.PROMPT_SECTORS) // 8 * 8 + s % 8)


def construct(loader, kv, kinds=("dsa",)):
    """decode.Engine's constructor with stand-ins for the buffers, caches, slots and state it builds; ``loader``
    (``module.name``) counted, nothing built. Returns its calls."""
    mod, name = loader
    calls = []
    real = getattr(mod, name)
    saved = [(decode, "Buffers"), (decode, "State"), (forward, "Caches"), (forward, "Slots"), (forward, "index_ring"),
             (hcsplit, "buffer_rows")]
    saved = [(m, a, getattr(m, a)) for m, a in saved]
    stub = SimpleNamespace
    try:
        setattr(mod, name, lambda: calls.append(name))
        decode.Buffers = decode.State = forward.Caches = forward.Slots = lambda *a, **k: stub(rows=1)
        forward.index_ring = lambda *a, **k: 0
        hcsplit.buffer_rows = lambda rows, *a, **k: rows
        os.environ["TF_GLM_L2PF"] = "0"                  # no L2 prefetch (it reads real weights)
        w = stub(meta={}, layers=[stub(kind=k) for k in kinds], world=3, rank=0, mtp=None, head=stub(n=1),
                 cfg=stub(hidden=4096), device=torch.device("cuda"))
        decode.Engine(w, kv=kv)
    finally:
        setattr(mod, name, real)
        for m, a, v in saved:
            setattr(m, a, v)
    return len(calls)


def cold_us(fn, layers=11, iters=15):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    times = []
    for _ in range(iters):
        flush.fill_(1)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / layers)
    return statistics.median(times)


def scoring():
    C = sparse.SPARSE_FROM
    for name, streams in (("4 x 8 @ 596k", [(596_000, 8)] * 4), ("4 x 4 @ 596k", [(596_001, 4)] * 4),
                          ("4 x 1 @ 596k", [(596_002, 1)] * 4), ("1 x 16 @ 596k", [(596_003, 16)]),
                          ("1 x 1 @ 596k", [(596_000, 1)]), ("1 x 5 @ 180k", [(180_001, 5)]),
                          ("1 x 16 @ 180k", [(180_002, 16)]), ("2 x 8 @ 300k", [(300_000, 8), (300_003, 8)]),
                          ("2 mixed 180k x 3 + 600k x 5", [(180_003, 3), (600_000, 5)]),
                          ("4 mixed 180k/300k/450k/600k x 8", [(180_000, 8), (300_001, 8), (450_002, 8), (600_003, 8)]),
                          ("4 uneven rows 1/16/3/12", [(250_000, 1), (400_001, 16), (590_002, 3), (200_003, 12)]),
                          ("dense limit straddled + dense segment",
                           [(C - 3, 8), (1_000, 4), (C + 1, 16), (300_000, 2)]),
                          ("6 streams x 5 @ 200k-450k", [(200_000 + 50_000 * i + i, 5) for i in range(6)]),
                          ("9 / 10 / 11 rows (a CTA's second group of 1 / 2 / 3 rows)",
                           [(450_001, 9), (300_002, 10), (596_003, 11)])):
        bits(name, streams)
    bits("adversarial 4 x 8 @ 300k", [(300_001, 8), (300_002, 8), (299_999, 8), (300_000, 8)], adversarial=True)
    bits("adversarial 1 x 16 @ 180k", [(180_003, 16)], adversarial=True)
    bits("4 x 8 @ 596k", [(596_000, 8)] * 4, arms=((1, 2), (7, 3), (48, 4), (512, 2), (96, 4)))
    bits("dense limit straddled", [(C - 3, 8), (C + 70, 3)], arms=((1, 4), (7, 2), (512, 3)))
    bits("1 x 16 @ 180k", [(180_002, 16)], arms=((3, 2), (200, 4)))

    streams = [(400_000, 8), (590_000, 8), (200_000, 8), (500_000, 8)]
    qi, wts, pk, rows, R, segs = window(streams, room=8_192)
    sel = SelectScratch(R, POOL_ROWS, dev)
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "1"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    for k, shift in enumerate((0, 37, -150_001, 4_097)):
        moved = [(b, max(1_000, p + shift * (i + 1) // 2), n) for i, (b, p, n) in enumerate(segs)]
        rows.set(moved)
        ref = run(qi, wts, pk, rows, sel, False)
        sel.scores.view(torch.int32).fill_(SENT)
        sel.tokens.fill_(-7)
        sel.counts.fill_(-7)
        gr.replay()
        torch.cuda.synchronize()
        got = (sel.scores.clone(), sel.tokens[:R].clone(), sel.counts[:R].clone())
        check(same3(got, ref), f"scores: CUDA graph replay {k} (positions {[p for _, p, _ in moved]}) bit-identical")
    gr = sel = None
    torch.cuda.empty_cache()

    calls, launched = [], []
    real, real_tri = sparse._seg_scores_ext, sparse._seg_scores

    class Tri:                                       # _seg_scores' launches recorded: grid and the pool table it reads
        def __getitem__(self, grid):
            def go(*a, **kw):
                launched.append((tuple(grid), a[12]))
                return real_tri[grid](*a, **kw)
            return go

    sparse._seg_scores_ext = lambda: calls.append(1) or real()
    sparse._seg_scores = Tri()
    try:
        qi, wts, pk, rows, R, _ = window([(300_000, 4)])
        sel = SelectScratch(R, 400_000, dev)
        run(qi, wts, pk, rows, sel, True)
        check(len(calls) == 1 and len(launched) == 1 and launched[0][1] is getattr(sel, "seg_scores_pools", None),
              f"TF_GLM_SEG_SCORES_CUDA=1: the extension called {len(calls)} time(s), _seg_scores launched "
              f"{len(launched)} time(s) over the table the extension writes (want 1, 1)")
        calls.clear()
        launched.clear()
        ref = run(qi, wts, pk, rows, sel, False)
        blocks = -(-sel.scores.shape[1] // 64)
        check(not calls and len(launched) == 1 and launched[0][1] is rows.seg_pools
              and launched[0][0] == (rows.max_segs, min(512, blocks)),
              f"TF_GLM_SEG_SCORES_CUDA=0: the extension called {len(calls)} time(s), _seg_scores launched "
              f"{len(launched)} time(s) over the window's own pool counts, grid {launched[0][0] if launched else None} "
              f"(want 0, 1, ({rows.max_segs}, {min(512, blocks)}))")
        check(sparse.SEG_SCORE_GRID == 512, f"SEG_SCORE_GRID {sparse.SEG_SCORE_GRID} (want 512)")
        sparse.SEG_SCORE_GRID = 256
        got = run(qi, wts, pk, rows, sel, False)
        sparse.SEG_SCORE_GRID = 512
        check(same3(got, ref), "the Triton scoring at grid 256 (before the patch) stores the same bits as at 512")
        os.environ.pop("TF_GLM_SEG_SCORES_CUDA", None)
        check(sparse.seg_scores_cuda(), "TF_GLM_SEG_SCORES_CUDA unset: on")
        calls.clear()
        pkb = torch.zeros(pk.shape[0], ID, dtype=torch.bfloat16, device=dev)            # a bf16 cache
        sparse.seg_select_tokens(qi, wts, pkb, rows, sel)
        torch.cuda.synchronize()
        check(not calls, f"a bf16 cache: the extension called {len(calls)} time(s) (want 0: FP8 only)")
    finally:
        sparse._seg_scores_ext, sparse._seg_scores = real, real_tri
    # the extension's pool table: sized for any window at its first use, never replaced (a captured graph keeps it)
    from tensorfold.families.glm5_next.cuda.segments import MAX_SEGS

    qa_, wa_, pa_, ra_, ra_n, _ = window([(300_000, 4)])
    qb_, wb_, pb_, rb_, rb_n, _ = window([(200_000 + 50_000 * i + i, 2) for i in range(6)], max_segs=MAX_SEGS)
    sel = SelectScratch(max(ra_n, rb_n), 600_000, dev)
    first, err = None, None
    try:
        run(qa_, wa_, pa_, ra_, sel, True)
        first = getattr(sel, "seg_scores_pools", None)
        run(qb_, wb_, pb_, rb_, sel, True)
    except RuntimeError as e:                    # a table too small for the second window: the extension refuses it
        err = f"{type(e).__name__}: {str(e)[:120]}"
    check(err is None and first is not None and sel.seg_scores_pools is first and first.numel() == MAX_SEGS,
          f"the extension's pool table: {MAX_SEGS} entries at its first use ({None if first is None else first.numel()}),"
          f" the same tensor for a {rb_.max_segs}-segment window after a {ra_.max_segs}-segment one ({err or 'no error'})")
    sel = None
    torch.cuda.empty_cache()

    qi, wts, pk, rows, R, _ = window([(300_001, 3), (400_002, 4), (250_003, 1), (500_000, 9), (C - 3, 6)], max_segs=6)
    sel = SelectScratch(R, 600_000, dev)
    tp = torch.full((rows.max_segs,), -5, dtype=torch.int32, device=dev)
    sel.scores.view(torch.int32).fill_(SENT)
    rs = kv8.parts(pk)[2]
    sparse._seg_scores_ext().seg_scores(qi, wts, wts.stride(0), pk, rs, sel.scores, sel.scores.stride(0), rows.pos,
                                        rows.sparse, rows.seg_start, rows.seg_rows, rows.seg_pbase, rows.seg_pools, tp,
                                        rows.max_segs, ID ** -0.5, 1.0 / 5.656854249492381, sparse.SEG_SCORE_CTAS,
                                        sparse.SEG_SCORE_STAGES, sparse.SEG_SCORE_CUDA_ROWS)
    torch.cuda.synchronize()
    small = rows.seg_rows < sparse.SEG_SCORE_CUDA_ROWS
    want_tp = torch.where(small, rows.seg_pools, torch.zeros_like(rows.seg_pools))
    check(torch.equal(tp, want_tp), f"the extension's pool table for _seg_scores {tp.tolist()} == the counts of the "
                                    f"segments under {sparse.SEG_SCORE_CUDA_ROWS} rows, 0 else {want_tp.tolist()}")
    written = (sel.scores.view(torch.int32)[:R] != SENT).any(1)
    seg_of = rows.seg[:R].long()
    want_w = (~small[seg_of]) & (rows.sparse[:R] != 0) & (rows.seg_pools[seg_of] > 0)
    check(torch.equal(written, want_w), f"rows the extension stored {written.int().tolist()} == the sparse rows of "
                                        f"segments of {sparse.SEG_SCORE_CUDA_ROWS}+ rows {want_w.int().tolist()}")
    sel = None
    torch.cuda.empty_cache()
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "1"


def selection():
    C = sparse.SPARSE_FROM
    for name, streams in (("4 x 8 @ 596k", [(596_000, 8)] * 4), ("1 x 16 @ 596k", [(596_003, 16)]),
                          ("1 x 8 @ 596k", [(596_001, 8)]), ("4 x 3 @ 596k", [(596_002, 3)] * 4),
                          ("2 x 8 @ 300k", [(300_000, 8), (300_003, 8)]), ("1 x 16 @ 180k", [(180_002, 16)]),
                          ("4 mixed 180k/300k/450k/600k x 8", [(180_000, 8), (300_001, 8), (450_002, 8), (600_003, 8)]),
                          ("dense limit straddled + dense segment",
                           [(C - 3, 8), (1_000, 4), (C + 1, 16), (300_000, 2)]),
                          ("6 streams x 3 @ 200k-450k", [(200_000 + 50_000 * i + i, 3) for i in range(6)])):
        qi, wts, pk, rows, R, _ = window(streams)
        sel = SelectScratch(R, POOL_ROWS, dev)
        ref = entry(qi, wts, pk, rows, sel, False)
        got = entry(qi, wts, pk, rows, sel, True)
        check(same2(got, ref), f"selection {name}: tokens and counts identical to select_split's "
                               f"({int((ref[1] > 0).sum())} sparse rows, {differ(got, ref)} words differ)")
        sel = None
        torch.cuda.empty_cache()

    streams = [(596_000, 4), (300_001, 4), (C + 5, 4), (C - 2, 4), (180_002, 4), (14_000, 4)]
    _, _, _, rows, R, _ = window(streams, max_segs=6)
    sel = SelectScratch(R, 600_000, dev)
    NP = sel.scores.shape[1]
    cases = {}
    cases["tie-heavy integers (8 values)"] = torch.randint(0, 8, (R, NP), device=dev, generator=g).float()
    cases["bf16-rounded scores"] = torch.randn(R, NP, device=dev, generator=g).to(torch.bfloat16).float()
    mixed = torch.randn(R, NP, device=dev, generator=g)
    pick = torch.rand(R, NP, device=dev, generator=g)
    mixed[pick < 0.2] = float("-inf")
    mixed[(pick >= 0.2) & (pick < 0.3)] = -0.0
    mixed[(pick >= 0.3) & (pick < 0.4)] = 0.0
    cases["-inf / -0.0 / +0.0 mixed"] = mixed
    few = torch.zeros(R, NP, device=dev)                  # the sample's top values that the rest of the row lacks
    for r in range(R):
        npool = (int(rows.pos[r]) + 1) // 4
        cols = sampled(npool)
        few[r, cols[torch.randperm(cols.numel(), device=dev, generator=g)[:60]]] = 10.0
    cases["few high scores, all in the sample (under K candidates)"] = few
    cases["plain random"] = torch.randn(R, NP, device=dev, generator=g)
    for name, sc in cases.items():
        ref = direct(sc, rows, sel, R, False)
        got = direct(sc, rows, sel, R, True)
        check(same2(got, ref), f"selection, {name}: identical ({differ(got, ref)} words differ)")
    segs_lo = [(0, 1_500, 4), (EXTENT * 2, 2_045, 4), (EXTENT * 4, 2_300, 4), (EXTENT * 6, 1_100, 4)]
    rows_lo = SegRows(16, dev, max_segs=4)
    rows_lo.set(segs_lo, sparse_from=1_000)
    sel_lo = SelectScratch(16, 600_000, dev)
    sc = torch.randn(16, sel_lo.scores.shape[1], device=dev, generator=g)
    ref = direct(sc, rows_lo, sel_lo, 16, False)
    got = direct(sc, rows_lo, sel_lo, 16, True)
    check(same2(got, ref) and int((ref[1] == 0).sum()) > 0,
          f"selection, rows sparse from position 1,000 (up to K visible pools: count 0): identical "
          f"({differ(got, ref)} words differ; {int((ref[1] == 0).sum())} rows with count 0)")
    sel_lo = None
    # a row with more than CAP candidates writes only its own 2 * CAP candidate slots: one overflowing sparse row
    # (tie-heavy scores) followed by 7 dense rows, which return before touching their regions
    _, _, _, rows_ov, R_ov, _ = window([(596_000, 1), (1_000, 7)])
    sel_ov = SelectScratch(R_ov, 600_000, dev)
    sc = torch.randint(0, 8, (R_ov, sel_ov.scores.shape[1]), device=dev, generator=g).float()
    ref = direct(sc, rows_ov, sel_ov, R_ov, False)
    sel_ov.split.fill_(0x7F7F7F7F)
    got = direct(sc, rows_ov, sel_ov, R_ov, True)
    cap = sparse.SEG_FLOOR[0]
    keys_full = bool((sel_ov.split[:cap] != 0x7F7F7F7F).all())
    outside = int((sel_ov.split[2 * cap:] != 0x7F7F7F7F).sum())
    check(same2(got, ref) and keys_full and outside == 0,
          f"selection, an overflowing row (more than CAP {cap} candidates: key slots full {keys_full}) beside dense "
          f"rows: identical ({differ(got, ref)} words differ), {outside} scratch words past its 2 * CAP changed")
    sel_ov = None
    sc = cases["bf16-rounded scores"]
    ref = direct(sc, rows, sel, R, False)
    for arm in ((2048, 1024, 4096, 8), (8192, 2048, 2048, 4), (4096, 512, 1024, 8), (4096, 3000, 8192, 8)):
        sparse.SEG_FLOOR = arm
        got = direct(sc, rows, sel, R, True)
        check(same2(got, ref), f"selection, SEG_FLOOR {arm}: identical ({differ(got, ref)} words differ)")
    sparse.SEG_FLOOR_ROWS, sparse.SEG_FLOOR = FLOOR
    sel = None
    torch.cuda.empty_cache()

    streams = [(400_000, 8), (590_000, 8), (200_000, 8), (500_000, 8)]
    qi, wts, pk, rows, R, segs = window(streams, room=8_192)
    sel = SelectScratch(R, POOL_ROWS, dev)
    os.environ["TF_GLM_SEG_SELECT_FLOOR"] = "1"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        sparse.seg_select_tokens(qi, wts, pk, rows, sel)
    for k, shift in enumerate((0, 37, -150_001, 4_097)):
        rows.set([(b, max(1_000, p + shift * (i + 1) // 2), n) for i, (b, p, n) in enumerate(segs)])
        ref = entry(qi, wts, pk, rows, sel, False)
        sel.tokens.fill_(-7)
        sel.counts.fill_(-7)
        gr.replay()
        torch.cuda.synchronize()
        got = (sel.tokens[:R].clone(), sel.counts[:R].clone())
        check(same2(got, ref), f"selection: CUDA graph replay {k} identical ({differ(got, ref)} words differ)")
    gr = sel = None
    torch.cuda.empty_cache()

    launched = []
    real = sparse._seg_select_floor

    class Rec:
        def __getitem__(self, grid):
            launched.append(grid)
            return real[grid]

    sparse._seg_select_floor = Rec()
    try:
        for name, streams, room, env, want in (
                ("8 rows, on", [(300_000, 8)], POOL_ROWS, "1", 1),
                (f"{sparse.SEG_FLOOR_ROWS - 1} rows, on", [(300_000, sparse.SEG_FLOOR_ROWS - 1)], POOL_ROWS, "1", 0),
                ("8 rows, TF_GLM_SEG_SELECT_FLOOR=0", [(300_000, 8)], POOL_ROWS, "0", 0),
                ("8 rows, unset", [(300_000, 8)], POOL_ROWS, None, 1),
                ("8 rows, a scratch too small for the candidates", [(5_000, 8)], 6_000, "1", 0)):
            qi, wts, pk, rows, R, _ = window(streams)
            sel = SelectScratch(R, room, dev)
            launched.clear()
            if env is None:
                os.environ.pop("TF_GLM_SEG_SELECT_FLOOR", None)
            else:
                os.environ["TF_GLM_SEG_SELECT_FLOOR"] = env
            sparse.seg_select_tokens(qi, wts, pk, rows, sel)
            torch.cuda.synchronize()
            check(len(launched) == want, f"selection routing, {name}: _seg_select_floor launched {len(launched)} "
                                         f"time(s) (want {want})")
            sel = None
    finally:
        sparse._seg_select_floor = real
    os.environ["TF_GLM_SEG_SELECT_FLOOR"] = "1"


def timing():
    arms = (("v1.10", 256, "0", "0"), ("grid 512", 512, "0", "0"), ("+ CUDA scoring", 512, "1", "0"),
            ("+ one-pass selection", 512, "1", "1"))
    for name, streams in (("4 x 8 @ 596k", [(596_000, 8)] * 4), ("4 x 4 @ 596k", [(596_000, 4)] * 4),
                          ("4 x 2 @ 596k", [(596_000, 2)] * 4), ("4 x 1 @ 596k", [(596_000, 1)] * 4),
                          ("1 x 16 @ 596k", [(596_000, 16)]), ("1 x 8 @ 596k", [(596_000, 8)]),
                          ("1 x 3 @ 596k", [(596_000, 3)]), ("1 x 16 @ 180k", [(180_000, 16)]),
                          ("2 x 8 @ 300k", [(300_000, 8)] * 2), ("4 x 8 @ 180k", [(180_000, 8)] * 4)):
        qi, wts, pk, rows, R, _ = window(streams)
        layers = [pk] + [keys(pk.shape[0]) for _ in range(10)]
        sel = SelectScratch(R, POOL_ROWS, dev)
        res = []
        for _, grid, sc, fl in arms:
            sparse.SEG_SCORE_GRID = grid
            os.environ["TF_GLM_SEG_SCORES_CUDA"], os.environ["TF_GLM_SEG_SELECT_FLOOR"] = sc, fl
            res.append(cold_us(lambda: [sparse.seg_select_tokens(qi, wts, lk, rows, sel) for lk in layers]))
        sparse.SEG_SCORE_GRID = 512
        print(f"TIME the indexer call, {name}, us a call: " + ", ".join(f"{a[0]} {t:.1f}" for a, t in zip(arms, res))
              + f" ({100 * (res[-1] / res[0] - 1):+.1f}%; x11 layers a round {11e-3 * (res[0] - res[-1]):.2f} ms)",
              flush=True)
        sel = layers = None
        torch.cuda.empty_cache()


def main():
    scoring()
    selection()
    loader = (sparse, "_seg_scores_ext")
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "1"
    check(construct(loader, "fp8") == 1, "Engine(FP8 caches, DSA layers): the scoring extension loaded at start")
    check(construct(loader, "bf16") == 0, "Engine(bf16 caches): not loaded (the kernel is FP8 only)")
    check(construct(loader, "fp8", ("mla",)) == 0, "Engine(no DSA layer): not loaded")
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "0"
    check(construct(loader, "fp8") == 0, "TF_GLM_SEG_SCORES_CUDA=0: not loaded")
    # the extension not used (a failed build or load; another Triton release): logged once, Triton serves
    import contextlib
    import io

    from tensorfold.cuda import build

    qi, wts, pk, rows, R, _ = window([(300_000, 8), (450_001, 4)])
    sel = SelectScratch(R, 500_000, dev)
    ref = run(qi, wts, pk, rows, sel, False)
    real_load, real_ptx, real_ext = build.load, sparse.TRITON_PTX, sparse._SEG_SCORES_EXT
    for case, ptx, want_tries in (("a failed build", real_ptx, 1), ("another Triton release", "0.0", 0)):
        tries, buf = [], io.StringIO()

        def broken(*a, **k):
            tries.append(1)
            raise RuntimeError("Error building extension 'tensorfold_glm_segscores_v1': nvcc not found")

        build.load, sparse.TRITON_PTX, sparse._SEG_SCORES_EXT, sparse._SEG_SCORES_FAILED = broken, ptx, None, None
        raised = first = second = got = None
        on = True
        try:
            with contextlib.redirect_stdout(buf):
                first, second = sparse._seg_scores_ext(), sparse._seg_scores_ext()
                on = sparse.seg_scores_cuda()
                got = run(qi, wts, pk, rows, sel, True)
        except Exception as exc:
            raised = exc
        finally:
            build.load, sparse.TRITON_PTX, sparse._SEG_SCORES_EXT, sparse._SEG_SCORES_FAILED = (real_load, real_ptx,
                                                                                                real_ext, None)
        logged = buf.getvalue().count("seg_scores.cu is not used")
        check(raised is None and first is None and second is None and len(tries) == want_tries and not on
              and logged == 1,
              f"{case}: nothing raised ({raised!r}), logged once ({logged}), loader called {len(tries)} "
              f"(want {want_tries}), the switch reads off ({not on})")
        check(got is not None and same3(got, ref), f"{case}: the window gets the Triton scoring's bits")
    os.environ["TF_GLM_SEG_SCORES_CUDA"] = "1"
    if "--time" in sys.argv and fails == 0:
        timing()
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    with torch.no_grad():
        main()
