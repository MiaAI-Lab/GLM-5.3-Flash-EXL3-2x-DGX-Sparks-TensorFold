#!/usr/bin/env python3
"""tools/prompt_scores_check.py [--time]: patch 0105 - a prompt chunk's DSA indexer scoring (sparse._scores, Triton)
as a CUDA kernel (prompt_scores.cu) that computes every score with the Triton kernel's own instruction sequence. Run in
the image built with patch 0105, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/prompt_scores_check.py:/c.py" <image> /c.py

1. Kernel: prompt_scores.cu's whole [R, NP] output (pre-filled with NaN) equals the Triton kernel's (sparse._scores,
   unchanged by the patch) bit for bit: 512-row blocks and a 333-row tail at row positions 2,050 / 35k / 113k / 300k /
   590k, int32 and int64 positions, FP8 pooled keys from kv8.quantize_rows, head weights a strided view of the
   index-key rows (as forward lays them out); also arbitrary (not power-of-two) pool scales, which pin the epilogue's
   product order.
2. Oracle: against float64 scores from the dequantized keys (sum_h w_h relu(scale ks_p q_h . k_p)), finite entries
   within fp32 tolerance, -inf exactly where the pool is past the row's position.
3. The entry point: sparse._select_prompt over a 2,048-row chunk at 113k (four 512-row blocks) gives the same tokens
   and counts with TF_GLM_SCORES_CUDA=1 (the CUDA kernel, counted: 4 launches) and =0 (Triton, none).
4. Start-up: decode.Engine's constructor loads (and on a fresh image, builds) the extension, before any request, for
   FP8 caches with DSA layers and the switch on; not with TF_GLM_SCORES_CUDA=0, a bf16 cache or no DSA layer (the
   constructor run with stand-ins for the buffers, caches, slots and state it builds; the loader counted).
5. The extension not used - a failed build or load (the loader raising), or another Triton release than
   sparse.TRITON_PTX (the loader then never called): nothing raised, logged once and not retried, the switch reads off,
   and sparse._select_prompt gives the Triton kernel's tokens and counts.
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

--time: a 512-row block's scores, Triton -> CUDA, at 35k, 113k, 300k and 590k (CUDA events; the indexer is
replicated, so its shapes are the same at TP=2 and TP=3)."""
import os
import sys
from types import SimpleNamespace

import torch
import triton

from tensorfold.families.glm5_next.cuda import decode, forward, hcsplit, kv8, sparse

H, D, KEY = 32, 128, 128
CAP = 1_048_576
WSCALE = 1.0 / 5.656854249492381
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def ms(fn, n=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


class Count:
    """sparse's CUDA extension, its ``scores`` calls counted."""

    def __init__(self, ext):
        self.ext, self.calls = ext, 0

    def scores(self, *a):
        self.calls += 1
        return self.ext.scores(*a)


def oracle(qi, wts, pk, P, R, NP):
    codes = pk[:NP, :KEY].contiguous().view(torch.float8_e4m3fn).double()                 # [NP, 128] key values
    ks = pk[:NP, KEY:KEY + 4].contiguous().view(torch.float32).double()[:, 0]               # [NP] scales
    q = qi.view(R, H, D).double()
    w = wts.float().double() * float(torch.tensor(WSCALE, dtype=torch.float32))
    dots = torch.einsum("rhd,pd->rhp", q, codes) * ks[None, None, :] * float(torch.tensor(D ** -0.5))
    s = (w[:, :, None] * dots.clamp(min=0)).sum(1)                                          # [R, NP]
    vis = torch.arange(NP, device=qi.device)[None, :] < ((P + torch.arange(R, device=qi.device) + 1) // 4)[:, None]
    return s, vis


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


def main():
    check(hasattr(sparse, "prompt_scores_cuda") and hasattr(sparse, "_scores_ext"),
          "patch 0105 applied: sparse.prompt_scores_cuda")
    if fails:
        print("1 FAIL", flush=True)
        sys.exit(1)
    dev = torch.device("cuda")
    torch.manual_seed(84)
    pk = kv8.quantize_rows(torch.randn(CAP // sparse.POOL + 2, KEY, device=dev))
    pkv, pks, rs, fp8 = kv8.parts(pk)
    ext = sparse._scores_ext()
    timing = "--time" in sys.argv
    for P in (2_050, 35_000, 113_000, 300_000, 590_000):
        for R in (512, 333):
            NP = sparse.pool_count(P, R, pk.shape[0] - 2)
            qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
            ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
            wts = ikr[:, KEY:]
            pos = torch.tensor([P], device=dev, dtype=torch.int32 if R == 512 else torch.int64)
            ref = torch.full((R, NP), float("nan"), device=dev)
            out = torch.full((R, NP), float("nan"), device=dev)

            def tri():
                sparse._scores[(R, triton.cdiv(triton.cdiv(NP, 64), sparse.SCORE_LOOP))](
                    qi, wts, wts.stride(0), pkv, pks, ref, pos, R, NP, D ** -0.5, WSCALE, H=H, HP=32, D=D, BP=64,
                    L=sparse.SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)

            def cud():
                ext.scores(qi, wts, wts.stride(0), pk, rs, out, pos, R, NP, D ** -0.5, WSCALE)

            tri()
            cud()
            torch.cuda.synchronize()
            diff = int((out.view(torch.int32) != ref.view(torch.int32)).sum())
            check(diff == 0, f"P {P} rows {R} NP {NP} ({pos.dtype}): CUDA scores == Triton's bit for bit ({diff} of "
                             f"{R * NP} differ)")
            if P <= 35_000 and R == 333:
                s, vis = oracle(qi, wts, pk, P, R, NP)
                fin = torch.isfinite(out)
                err = torch.where(vis, (out.double() - s).abs(), torch.zeros_like(s)).max() / s.abs().max()
                check(bool(torch.equal(fin, vis)) and err.item() < 1e-5,
                      f"P {P} rows {R}: -inf exactly past each row's pools, max error {err.item():.2e} of max")
            if timing and R == 512 and P >= 35_000:
                t0, t1 = ms(tri), ms(cud)
                print(f"TIME a 512-row block's scores at {P} ({NP} pools): Triton {t0:.3f} -> CUDA {t1:.3f} ms "
                      f"({100 * (t1 / t0 - 1):+.1f}%)", flush=True)
    # arbitrary (not power-of-two) pool scales: kv8's are powers of two, which makes the scale's place in the product
    # chain exact either way; arbitrary ones pin Triton's order, (dot * ks) * scale
    pk2 = pk.clone()
    pk2[:, KEY:KEY + 4].view(torch.float32)[:, 0] = torch.rand(pk2.shape[0], device=dev) * 3 + 0.1
    p2v, p2s, _, _ = kv8.parts(pk2)
    P, R = 35_000, 512
    NP = sparse.pool_count(P, R, pk2.shape[0] - 2)
    qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
    ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
    wts = ikr[:, KEY:]
    pos = torch.tensor([P], device=dev, dtype=torch.int32)
    ref = torch.full((R, NP), float("nan"), device=dev)
    out = torch.full((R, NP), float("nan"), device=dev)
    sparse._scores[(R, triton.cdiv(triton.cdiv(NP, 64), sparse.SCORE_LOOP))](
        qi, wts, wts.stride(0), p2v, p2s, ref, pos, R, NP, D ** -0.5, WSCALE, H=H, HP=32, D=D, BP=64,
        L=sparse.SCORE_LOOP, RS=rs, FP8=fp8, num_warps=4)
    ext.scores(qi, wts, wts.stride(0), pk2, rs, out, pos, R, NP, D ** -0.5, WSCALE)
    torch.cuda.synchronize()
    diff = int((out.view(torch.int32) != ref.view(torch.int32)).sum())
    check(diff == 0, f"arbitrary pool scales, P {P} rows {R}: CUDA == Triton bit for bit ({diff} of {R * NP} differ)")
    # the entry point, both ways
    P, R = 113_000, 2048
    qi = torch.randn(R, H * D, device=dev, dtype=torch.bfloat16)
    ikr = torch.randn(R, KEY + H, device=dev, dtype=torch.bfloat16)
    pos_dev = torch.tensor([P], device=dev, dtype=torch.int32)
    got = {}
    for flag in ("1", "0"):
        os.environ["TF_GLM_SCORES_CUDA"] = flag
        counter = Count(ext)
        sparse._SCORES_EXT = counter
        try:
            tok, cnt = sparse._select_prompt(qi, ikr[:, KEY:], pk, P, R, pk.shape[0] - 2, pos_dev)
            torch.cuda.synchronize()
        finally:
            sparse._SCORES_EXT = ext
        got[flag] = (tok.clone(), cnt.clone(), counter.calls)
    same = torch.equal(got["1"][0], got["0"][0]) and torch.equal(got["1"][1], got["0"][1])
    check(same and got["1"][2] == 4 and got["0"][2] == 0,
          f"_select_prompt 2,048 rows at 113k: tokens and counts equal CUDA vs Triton ({same}), CUDA launches "
          f"{got['1'][2]} (want 4) / {got['0'][2]} (want 0)")
    os.environ.pop("TF_GLM_SCORES_CUDA", None)
    check(sparse.prompt_scores_cuda(), "TF_GLM_SCORES_CUDA unset: on")
    # start-up
    loader = (sparse, "_scores_ext")
    os.environ["TF_GLM_SCORES_CUDA"] = "1"
    check(construct(loader, "fp8") == 1, "Engine(FP8 caches, DSA layers): the scoring extension loaded at start")
    check(construct(loader, "bf16") == 0, "Engine(bf16 caches): not loaded (the kernel is FP8 only)")
    check(construct(loader, "fp8", ("mla",)) == 0, "Engine(no DSA layer): not loaded")
    os.environ["TF_GLM_SCORES_CUDA"] = "0"
    check(construct(loader, "fp8") == 0, "TF_GLM_SCORES_CUDA=0: not loaded")
    # the extension not used (a failed build or load; another Triton release): the start goes on, logged once, and
    # the Triton kernel serves with the same tokens
    import contextlib
    import io

    from tensorfold.cuda import build

    os.environ["TF_GLM_SCORES_CUDA"] = "1"
    real_load, real_ptx = build.load, sparse.TRITON_PTX
    check(sparse.triton_matches_ptx(), f"this image's Triton {triton.__version__} is TRITON_PTX {sparse.TRITON_PTX}")
    sparse.TRITON_PTX = triton.__version__.split(".")[0] + ".99"
    check(not sparse.triton_matches_ptx(), f"another minor release ({sparse.TRITON_PTX}) is not this Triton's")
    sparse.TRITON_PTX = real_ptx
    for case, ptx, want_tries in (("a failed build", real_ptx, 1), ("another Triton release", "0.0", 0)):
        tries, out = [], io.StringIO()

        def broken(*a, **k):
            tries.append(1)
            raise RuntimeError("Error building extension 'tensorfold_glm_scores_v1': nvcc not found")

        build.load, sparse.TRITON_PTX, sparse._SCORES_EXT, sparse._SCORES_FAILED = broken, ptx, None, None
        raised = first = second = tok = cnt = None
        on = True
        try:
            with contextlib.redirect_stdout(out):
                first, second = sparse._scores_ext(), sparse._scores_ext()
                on = sparse.prompt_scores_cuda()
                tok, cnt = sparse._select_prompt(qi, ikr[:, KEY:], pk, P, R, pk.shape[0] - 2, pos_dev)
                torch.cuda.synchronize()
        except Exception as exc:
            raised = exc
        finally:
            build.load, sparse.TRITON_PTX, sparse._SCORES_EXT, sparse._SCORES_FAILED = real_load, real_ptx, ext, None
        logged = out.getvalue().count("prompt_scores.cu is not used")
        check(raised is None and first is None and second is None and len(tries) == want_tries and not on
              and logged == 1,
              f"{case}: nothing raised ({raised!r}), logged once ({logged}), loader called {len(tries)} "
              f"(want {want_tries}), the switch reads off ({not on})")
        check(tok is not None and torch.equal(tok, got["0"][0]) and torch.equal(cnt, got["0"][1]),
              f"{case}: _select_prompt gives the Triton kernel's tokens and counts")
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
