#!/usr/bin/env python3
"""tools/seg_chunks_check.py [--time]: patch 0108 - the decode windows' latent attention chunk pass (latent._seg_chunks,
Triton) as a CUDA kernel (seg_chunks.cu) for FP8 arenas and 17-24 heads a rank (TP=3's 22 / 21) that computes every
chunk partial with the Triton kernel's own instruction sequence; Triton's _merge reads them as before. Run in the image
built with patch 0108, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/seg_chunks_check.py:/c.py" <image> /c.py

latent.seg_attention with the CUDA chunk pass (TF_GLM_SEG_CHUNKS_CUDA=1, SEG_CHUNKS_CTAS as shipped) against the
Triton one (=0, the head tile seg_head_block(R, H), TF_GLM_LATENT_STAGES 3): every output bit (out pre-filled with
NaN) AND every chunk partial (the scratch's PO / PM / PL slices pre-filled with NaN, so every store and every
non-store counts) identical, over:
  - windows at 22 and 21 heads: 4 x 8 / 4 x 4 / 4 x 2 / 4 x 1 @ 596k, 1 x 16 / 1 x 8 / 1 x 1 @ 596k, 2 x 8 @ 300k,
    4 x 8 @ 180k, 3 x 5 @ 420k, uneven streams (rows 1 / 7 / 3 / 5) - each stream's rows sharing its selected pools
    plus their own incomplete pool, as seg_select gives them;
  - dense rows (keys 0 .. pos below the dense limit: pos 0, 30, 31, 32, 510, 511, 512, 1023, 2047, 2049 ..), mixed
    with sparse ones in one window, and sparse rows of 0, 1, 31-33, 511-513, 1024, 2050 and 2051 tokens (chunks
    straddled, empty chunks, partial key tiles), their token lists padded past the count with -1 and with ids far
    past the arena (never read);
  - adversarial data: every e4m3 code including NaN codes (0x7F / 0xFF) in rows that are read, arbitrary
    (non-power-of-two) row scales and a 192^-0.5 softmax scale, queries from 1e-30 to 3e4 with +-0, a row whose
    keys are all -0;
  - both head tiles (16 and 32: the same partials) and other SEG_CHUNKS_CTAS (1, 7, 48, 0);
  - one CUDA graph of the CUDA path captured, then replayed over new positions, tokens and queries;
  - routing: the extension is called once a window on an FP8 arena with 22 / 21 heads, never with the switch at 0, on
    a bf16 arena, or with 16 / 32 heads (32: TP=2);
  - start-up: decode.Engine's constructor loads (on a fresh image, builds) the extension before any request for FP8
    caches with DSA layers and the switch on; not with TF_GLM_SEG_CHUNKS_CUDA=0, bf16 caches or no DSA layer;
  - not used - a failed build or load, or another Triton release than sparse.TRITON_PTX: nothing raised, logged once
    and not retried, the switch reads off, and seg_attention gives the Triton kernel's output and partials.
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

--time: the whole seg_attention call (chunk pass and merge), Triton -> CUDA, cold (11 calls a CUDA graph, each its
own selected tokens, L2 flushed between replays), 22 heads (TP=3 rank 0) and 21; at 32 heads (TP=2) the kernel does
not run."""
import os
import statistics
import sys
from types import SimpleNamespace

import torch

from tensorfold.families.glm5_next.cuda import decode, forward, hcsplit, kv8, latent, sparse
from tensorfold.families.glm5_next.cuda.segments import SegRows

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(97)
LW = 512
TOK = sparse.TOKENS
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


class Count:
    """latent's CUDA extension, its ``seg_chunks`` calls counted."""

    def __init__(self, ext):
        self.ext, self.n = ext, 0

    def seg_chunks(self, *a):
        self.n += 1
        return self.ext.seg_chunks(*a)


def cache_for(rows, codes="normal", scales="pow2"):
    c = kv8.zeros(rows, LW, "fp8", dev)
    if codes == "all":                     # every e4m3 code, NaN included
        c[:, :LW] = torch.randint(0, 256, (rows, LW), dtype=torch.uint8, device=dev, generator=g)
    else:
        c[:, :LW] = torch.randint(0, 0x7E, (rows, LW), dtype=torch.uint8, device=dev, generator=g)
        c[:, :LW] |= torch.randint(0, 2, (rows, LW), dtype=torch.uint8, device=dev, generator=g) * 0x80
    if scales == "pow2":
        s = 2.0 ** torch.randint(-12, -4, (rows,), device=dev, generator=g).float()
    else:                                  # arbitrary: products with the scores round
        s = torch.rand((rows,), device=dev, generator=g) * 0.01 + 1e-4
    c[:, LW:LW + 4] = s.to(torch.float32).view(torch.uint8).reshape(rows, 4)
    return c


def window(spec, ctx_of, H, cache_rows=None, q="normal", codes="normal", scales="pow2"):
    """spec: per stream (rows, ctx, sparse?) -> (qa, cache, rows, tokens, counts). Sparse streams' rows share the
    stream's 2,048 selected pool tokens plus their own incomplete pool (seg_select's shape). Each stream's extent of
    the arena holds its rows' keys (ctx + rows + 64 rows), one after another."""
    R = sum(r for r, _, _ in spec)
    ext = [0]
    for nr, ctx, _ in spec:
        ext.append(ext[-1] + ctx + nr + 64)
    cache = cache_for(cache_rows or ext[-1], codes, scales)
    rows = SegRows(R, dev, max_segs=max(4, min(len(spec), 8)))
    pos, base, spr = [], [], []
    toks = torch.full((R, TOK + 1), -1, dtype=torch.int32, device=dev)
    counts = torch.zeros((R,), dtype=torch.int32, device=dev)
    r0 = 0
    for si, (nr, ctx, sp) in enumerate(spec):
        if sp:
            pools = torch.randperm(ctx // 4 - 1, device=dev, generator=g)[:sparse.TOPK_POOLS].sort().values
            t = (pools[:, None] * 4 + torch.arange(4, device=dev)[None, :]).reshape(-1)
        for j in range(nr):
            p = ctx + j
            pos.append(p)
            base.append(ext[si])
            spr.append(1 if sp else 0)
            if sp:
                sel = torch.cat([t, torch.arange(p // 4 * 4, p + 1, device=dev)])[:TOK]
                toks[r0 + j, :sel.numel()] = sel.to(torch.int32)
                counts[r0 + j] = sel.numel()
        r0 += nr
    rows.pos.copy_(torch.tensor(pos, dtype=torch.int32))
    rows.base.copy_(torch.tensor(base, dtype=torch.int32))
    rows.sparse.copy_(torch.tensor(spr, dtype=torch.int32))
    if q == "normal":
        qa = (torch.randn((R, H, LW), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    else:                                  # 1e-30 .. 3e4 and +-0
        mag = 10.0 ** (torch.rand((R, H, LW), device=dev, generator=g) * 34.5 - 30)
        sgn = torch.randint(0, 2, (R, H, LW), device=dev, generator=g).float() * 2 - 1
        qa = (mag * sgn).to(torch.bfloat16)
        qa[torch.rand((R, H, LW), device=dev, generator=g) < 0.05] = 0.0
        qa[torch.rand((R, H, LW), device=dev, generator=g) < 0.05] = -0.0
    return qa, cache, rows, toks, counts


def run(qa, cache, rows, toks, counts, cuda, hb=None, ctas=0, scale=0.0625):
    R, H, _ = qa.shape
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1" if cuda else "0"
    latent.SEG_CHUNKS_CTAS = ctas
    s = latent.LatentScratch(max(R, 32), H, NCH, dev, lw=LW)
    n = NCH * R * H
    for t in (s.po, s.pm, s.pl):
        t.fill_(float("nan"))
    out = torch.full((R, H, LW), float("nan"), dtype=torch.bfloat16, device=dev)
    latent.seg_attention(qa, cache, rows, toks, counts, s, scale=scale, out=out, hb=hb)
    torch.cuda.synchronize()
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
    latent.SEG_CHUNKS_CTAS = SHIPPED_CTAS
    return out, s.po[:n * LW].clone(), s.pm[:n].clone(), s.pl[:n].clone()


def same(a, b):
    a, b = a.contiguous(), b.contiguous()
    v = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.view(v), b.view(v)), int((a.view(v) != b.view(v)).sum())


def compare(name, w, hb=None, ctas=None, scale=0.0625):
    qa, cache, rows, toks, counts = w
    ref = run(qa, cache, rows, toks, counts, False, scale=scale)
    got = run(qa, cache, rows, toks, counts, True, hb=hb, ctas=SHIPPED_CTAS if ctas is None else ctas, scale=scale)
    res = [same(a, b) for a, b in zip(got, ref)]
    ok = all(r[0] for r in res)
    check(ok, f"{name}: out / PO / PM / PL identical to Triton's ({', '.join(str(r[1]) for r in res)} differ)")
    return ref


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


def timing():
    flush = torch.empty((256 << 20,), dtype=torch.uint8, device=dev)

    def graph_us(call):
        call()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            call()
        times = []
        for _ in range(20):
            flush.fill_(1)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gr.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) * 1e3 / 11)
        return statistics.median(times)

    for H in (22, 21):
        for name, spec in (("4 x 8 @ 596k", [(8, 596000, 1)] * 4), ("4 x 4 @ 596k", [(4, 596000, 1)] * 4),
                           ("4 x 2 @ 596k", [(2, 596000, 1)] * 4), ("1 x 16 @ 596k", [(16, 596000, 1)]),
                           ("1 x 8 @ 596k", [(8, 596000, 1)]), ("1 x 1 @ 596k", [(1, 596000, 1)]),
                           ("2 x 8 @ 300k", [(8, 300000, 1)] * 2), ("4 x 8 @ 180k", [(8, 180000, 1)] * 4)):
            qa, cache, rows, toks, counts = window(spec, None, H)
            tl = [toks] + [window(spec, None, H, cache_rows=16)[3] for _ in range(10)]   # each layer its own tokens
            R = qa.shape[0]
            s = latent.LatentScratch(max(R, 32), H, NCH, dev, lw=LW)
            out = torch.empty((R, H, LW), dtype=torch.bfloat16, device=dev)
            res = []
            for env in ("0", "1"):
                os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = env
                res.append(graph_us(lambda: [latent.seg_attention(qa, cache, rows, t, counts, s, scale=0.0625, out=out)
                                             for t in tl]))
            os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
            print(f"TIME {H} heads, {name}: Triton {res[0]:.1f} -> CUDA {res[1]:.1f} us a call "
                  f"({100 * (res[1] / res[0] - 1):+.1f}%; x11 layers a round {11e-3 * (res[0] - res[1]):.2f} ms)",
                  flush=True)
            torch.cuda.empty_cache()


check(hasattr(latent, "seg_chunks_cuda") and hasattr(latent, "_seg_chunks_ext"),
      "patch 0108 applied: latent.seg_chunks_cuda")
if fails:
    print("1 FAIL", flush=True)
    sys.exit(1)
NCH = latent.seg_chunks()
SHIPPED_CTAS = latent.SEG_CHUNKS_CTAS


def main():
    # ---- decode windows
    for H in (22, 21):
        for name, spec in (("4 x 8 @ 596k", [(8, 596000, 1)] * 4), ("4 x 4 @ 596k", [(4, 596000, 1)] * 4),
                           ("4 x 2 @ 596k", [(2, 596000, 1)] * 4), ("4 x 1 @ 596k", [(1, 596000, 1)] * 4),
                           ("1 x 16 @ 596k", [(16, 596000, 1)]), ("1 x 8 @ 596k", [(8, 596000, 1)]),
                           ("1 x 1 @ 596k", [(1, 596000, 1)]), ("2 x 8 @ 300k", [(8, 300000, 1)] * 2),
                           ("4 x 8 @ 180k", [(8, 180000, 1)] * 4), ("3 x 5 @ 420k", [(5, 420000, 1)] * 3),
                           ("uneven 1 / 7 / 3 / 5 @ 180k-600k",
                            [(1, 600000, 1), (7, 180000, 1), (3, 333333, 1), (5, 512345, 1)])):
            ref = compare(f"H {H} {name}", window(spec, None, H))
            if H == 22 and name == "4 x 8 @ 596k":
                check(not torch.isnan(ref[0].float()).any(),
                      "H 22 4 x 8 @ 596k: Triton's out has no NaN (a real case)")
    # ---- dense rows, mixed windows, ragged counts
    for H in (22, 21):
        spec = [(1, p, 0) for p in (0, 30, 31, 32, 510, 511, 512, 1023, 2047)] + [(3, 2049, 0), (4, 596000, 1)]
        compare(f"H {H} dense rows (pos 0 .. 2051) + a sparse stream", window(spec, None, H))
        qa, cache, rows, _, _ = window([(4, 2000, 0), (3, 700, 0), (2, 100, 0)], None, H)
        compare(f"H {H} a window without sparse rows (no token lists)", (qa, cache, rows, None, None))
        qa, cache, rows, toks, counts = window([(12, 300000, 1)], None, H)
        cnts = [0, 1, 31, 32, 33, 511, 512, 513, 1024, 2050, 2051, 1537]
        counts.copy_(torch.tensor(cnts, dtype=torch.int32))
        for r, c in enumerate(cnts):
            toks[r, c:] = -1 if r % 2 else 2 ** 30        # padding never read: -1, or far past the arena
        compare(f"H {H} sparse counts {cnts} (padded with -1 / 2^30)", (qa, cache, rows, toks, counts))
    # ---- adversarial data
    for H in (22, 21):
        compare(f"H {H} every e4m3 code (NaN codes read), arbitrary scales, softmax 192^-0.5",
                window([(8, 200000, 1)] * 2, None, H, codes="all", scales="any"), scale=192 ** -0.5)
        compare(f"H {H} queries 1e-30 .. 3e4 and +-0, arbitrary scales",
                window([(6, 250000, 1)] * 2, None, H, q="wide", scales="any"))
    qa, cache, rows, toks, counts = window([(4, 100000, 1), (3, 1500, 0)], None, 22)
    for r in range(7):                    # the rows' keys all -0 (code 0x80) in row 0's / row 4's chunk 0
        ids = (toks[r, :512].long() if r < 4 else torch.arange(512, device=dev)) + rows.base[r].long()
        cache[ids, :LW] = 0x80
    compare("H 22 keys all -0 in the first chunk", (qa, cache, rows, toks, counts))
    # ---- head tiles and CTA counts (speed constants: the same bits)
    w = window([(8, 596000, 1)] * 2 + [(2, 1800, 0)], None, 22, codes="all", scales="any")
    for hb in (16, 32):
        compare(f"H 22 head tile {hb} (CUDA) vs the default tile (Triton)", w, hb=hb)
    for ctas in (1, 7, 48, 0):
        compare(f"H 22 SEG_CHUNKS_CTAS {ctas}", w, ctas=ctas)
    # ---- one CUDA graph replayed over new positions, tokens and queries
    H = 22
    qa, cache, rows, toks, counts = window([(8, 400000, 1)] * 3, None, H)
    s = latent.LatentScratch(32, H, NCH, dev, lw=LW)
    out = torch.full((24, H, LW), float("nan"), dtype=torch.bfloat16, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        latent.seg_attention(qa, cache, rows, toks, counts, s, scale=0.0625, out=out)
    torch.cuda.current_stream().wait_stream(side)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        latent.seg_attention(qa, cache, rows, toks, counts, s, scale=0.0625, out=out)
    for k in range(3):
        q2, _, r2, t2, c2 = window([(8, 399000 - 1000 * k, 1), (8, 380000 + 777 * k, 1), (8, 399000 - 555 * k, 1)],
                                   None, H, cache_rows=16)
        qa.copy_(q2)
        toks.copy_(t2)
        counts.copy_(c2)
        rows.pos.copy_(r2.pos)
        out.fill_(float("nan"))
        gr.replay()
        torch.cuda.synchronize()
        ref = run(qa, cache, rows, toks, counts, False)[0]
        check(same(out, ref)[0], f"CUDA graph replay {k} over new positions / tokens / queries: out identical "
                                 f"({same(out, ref)[1]} differ)")
    del gr
    # ---- routing
    ext = latent._seg_chunks_ext()
    cnt = Count(ext)
    latent._SEG_CHUNKS_EXT = cnt
    try:
        for name, H, kind, env, want in (("fp8, 22 heads", 22, "fp8", "1", 1), ("fp8, 21 heads", 21, "fp8", "1", 1),
                                         ("fp8, 22 heads, TF_GLM_SEG_CHUNKS_CUDA=0", 22, "fp8", "0", 0),
                                         ("bf16 arena, 22 heads", 22, "bf16", "1", 0),
                                         ("fp8, 16 heads", 16, "fp8", "1", 0), ("fp8, 32 heads", 32, "fp8", "1", 0)):
            qa, cache, rows, toks, counts = window([(4, 300000, 1)] * 2, None, H)
            if kind == "bf16":
                cache = kv8.dequantize(cache).to(torch.bfloat16).contiguous()
            s = latent.LatentScratch(32, H, NCH, dev, lw=LW)
            out = torch.empty((8, H, LW), dtype=torch.bfloat16, device=dev)
            os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = env
            cnt.n = 0
            try:
                latent.seg_attention(qa, cache, rows, toks, counts, s, scale=0.0625, out=out)
                torch.cuda.synchronize()
                check(cnt.n == want, f"routing {name}: the extension ran {cnt.n} time(s) (want {want})")
            except Exception as ex:          # e.g. the extension refusing a shape routed to it
                check(False, f"routing {name}: {type(ex).__name__}: {str(ex)[:120]}")
    finally:
        os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
        latent._SEG_CHUNKS_EXT = ext
    os.environ.pop("TF_GLM_SEG_CHUNKS_CUDA", None)
    check(latent.seg_chunks_cuda(), "TF_GLM_SEG_CHUNKS_CUDA unset: on")
    loader = (latent, "_seg_chunks_ext")
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
    check(construct(loader, "fp8") == 1, "Engine(FP8 caches, DSA layers): the extension loaded at start")
    check(construct(loader, "bf16") == 0, "Engine(bf16 caches): not loaded (the kernel is FP8 only)")
    check(construct(loader, "fp8", ("mla",)) == 0, "Engine(no DSA layer): not loaded")
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "0"
    check(construct(loader, "fp8") == 0, "TF_GLM_SEG_CHUNKS_CUDA=0: not loaded")
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
    # ---- the extension not used (a failed build or load; another Triton release): logged once, Triton serves
    import contextlib
    import io

    from tensorfold.cuda import build
    from tensorfold.families.glm5_next.cuda import sparse

    qa, cache, rows, toks, counts = window([(8, 300000, 1), (5, 450000, 1)], None, 22)
    ref = run(qa, cache, rows, toks, counts, False)
    real_load, real_ptx = build.load, sparse.TRITON_PTX
    for case, ptx, want_tries in (("a failed build", real_ptx, 1), ("another Triton release", "0.0", 0)):
        tries, buf = [], io.StringIO()

        def broken(*a, **k):
            tries.append(1)
            raise RuntimeError("Error building extension 'tensorfold_glm_segchunks_v1': nvcc not found")

        build.load, sparse.TRITON_PTX, latent._SEG_CHUNKS_EXT, latent._SEG_CHUNKS_FAILED = broken, ptx, None, None
        raised = first = second = got = None
        on = True
        try:
            with contextlib.redirect_stdout(buf):
                first, second = latent._seg_chunks_ext(), latent._seg_chunks_ext()
                on = latent.seg_chunks_cuda()
                got = run(qa, cache, rows, toks, counts, True)
        except Exception as exc:
            raised = exc
        finally:
            build.load, sparse.TRITON_PTX, latent._SEG_CHUNKS_EXT, latent._SEG_CHUNKS_FAILED = (real_load, real_ptx,
                                                                                                ext, None)
        logged = buf.getvalue().count("seg_chunks.cu is not used")
        check(raised is None and first is None and second is None and len(tries) == want_tries and not on
              and logged == 1,
              f"{case}: nothing raised ({raised!r}), logged once ({logged}), loader called {len(tries)} "
              f"(want {want_tries}), the switch reads off ({not on})")
        check(got is not None and all(same(a, b)[0] for a, b in zip(got, ref)),
              f"{case}: seg_attention gives the Triton kernel's output and partials")
    os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "1"
    if "--time" in sys.argv and fails == 0:
        timing()
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
