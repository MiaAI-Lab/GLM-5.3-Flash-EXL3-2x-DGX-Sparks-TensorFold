#!/usr/bin/env python3
"""tools/sparse_onepass_check.py [--time]: patch 0106 - a prompt chunk's sparse latent attention
(latent._sparse_onepass, Triton) as a CUDA kernel (sparse_onepass.cu) for FP8 caches and 17-24 heads a rank (TP=3's
22 / 21) that computes every output with the Triton kernel's own instruction sequence. Run in the image built with
patch 0106, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/sparse_onepass_check.py:/c.py" <image> /c.py

1. Kernel: the whole [R, H, 512] output (pre-filled with NaN, so rows it must leave alone and slots it never writes
   show) equals the Triton kernel's (latent._sparse_onepass, launched by sparse_onepass with TF_GLM_ONEPASS_CUDA=0)
   bit for bit: 2,048-row prompt chunks, 22 heads (and 21), FP8 latent rows from kv8.quantize_rows, softmax scale
   1/16, int32 token lists 2,051 wide at positions 35k / 113k / 300k / 590k, two selections ("recent": each row's
   last 2,051 tokens; "spread": 512 random pools below the row), counts 2,048-2,051 with a few short and empty rows;
   ragged counts 0-2,051, 1,024- and 333-row chunks, padding past the counts that must never be read (-1, a row of
   NaN codes, out of range), NaN codes in rows that are read (Triton's NaN bits), and arbitrary (not power-of-two)
   row scales with a non-power-of-two softmax scale, which pin the order of every product in the chain.
2. Oracle: against float64 attention from the dequantized rows (softmax(scale q . k) . k), within bf16 tolerance.
3. The entry point: latent.sparse_onepass launches the CUDA kernel (counted) with TF_GLM_ONEPASS_CUDA=1 and gives the
   bits it gives with =0 (Triton, not launched); 32 heads a rank (TP=2) and a bf16 cache stay on Triton.
4. Start-up: decode.Engine's constructor loads (on a fresh image, builds) the extension before any request for FP8
   caches with DSA layers and the switch on; not with TF_GLM_ONEPASS_CUDA=0, a bf16 cache or no DSA layer (the
   constructor run with stand-ins for what it builds; the loader counted).
5. The extension not used - a failed build or load (the loader raising), or another Triton release than
   sparse.TRITON_PTX (the loader then never called): nothing raised, logged once and not retried, the switch reads off,
   and latent.sparse_onepass gives the Triton kernel's output.
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

--time: a 2,048-row chunk at 113k and 590k, "recent" and "spread" selections, Triton -> CUDA (CUDA events), 22 and
21 heads (TP=3's ranks); at 32 heads (TP=2) the kernel does not run."""
import os
import statistics
import sys
from types import SimpleNamespace

import torch

from tensorfold.families.glm5_next.cuda import decode, forward, hcsplit, kv8, latent

R, LW, NSEL = 2048, 512, 2051
SCALE = 256 ** -0.5
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def ms(fn, n=5):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / n)
    return statistics.median(out)


class Count:
    """latent's CUDA extension, its ``onepass`` calls counted."""

    def __init__(self, ext):
        self.ext, self.calls = ext, 0

    def onepass(self, *a):
        self.calls += 1
        return self.ext.onepass(*a)


def selection(kind, P, rows, dev):
    """[rows, 2,051] int32 token lists of rows at positions P .. P + rows - 1 (ascending)."""
    r = torch.arange(rows, device=dev)
    if kind == "recent":
        return (P + r[:, None] - (NSEL - 1) + torch.arange(NSEL, device=dev)[None, :]).to(torch.int32).contiguous()
    out = torch.empty((rows, NSEL), dtype=torch.int32, device=dev)
    for a in range(0, rows, 256):
        b = min(rows, a + 256)
        pools = torch.sort(torch.rand(b - a, P // 4, device=dev).topk(512, dim=1).indices, dim=1).values
        out[a:b, :2048] = (pools[:, :, None] * 4 + torch.arange(4, device=dev)).reshape(b - a, 2048).to(torch.int32)
    out[:, 2048:] = (((P + r) // 4 * 4)[:, None] + torch.arange(3, device=dev)[None, :]).to(torch.int32)
    return out


def run(qa, cache, tok, counts, scale, cuda):
    os.environ["TF_GLM_ONEPASS_CUDA"] = "1" if cuda else "0"
    out = torch.full_like(qa, float("nan"))
    latent.sparse_onepass(qa, cache, tok, counts, out, scale)
    torch.cuda.synchronize()
    return out


def compare(name, qa, cache, tok, counts, scale=SCALE):
    ref = run(qa, cache, tok, counts, scale, False)
    out = run(qa, cache, tok, counts, scale, True)
    a, b = ref.view(torch.int16), out.view(torch.int16)
    diff = int((a != b).sum())
    left = int(out[counts == 0].isnan().all()) if bool((counts == 0).any()) else 1
    check(diff == 0 and left == 1, f"{name}: CUDA == Triton bit for bit ({diff} of {out.numel()} differ; rows with "
                                   f"count 0 left alone: {bool(left)})")
    return ref, out


def oracle(qa, cache, tok, counts, out, rows, scale):
    """Max |out - float64 attention| / max |attention| over the given rows."""
    codes = cache[:, :LW].contiguous().view(torch.float8_e4m3fn)
    ks = cache[:, LW:LW + 4].contiguous().view(torch.float32)[:, 0]
    worst = 0.0
    for r in rows:
        n = int(counts[r])
        ids = tok[r, :n].long()
        kv = codes[ids].double() * ks[ids].double()[:, None]                       # [n, 512] dequantized rows
        s = (qa[r].double() @ kv.T) * float(torch.tensor(scale, dtype=torch.float32))
        o = torch.softmax(s, dim=1) @ kv
        worst = max(worst, ((out[r].double() - o).abs().max() / o.abs().max()).item())
    return worst


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
    check(hasattr(latent, "onepass_cuda") and hasattr(latent, "_onepass_ext"),
          "patch 0106 applied: latent.onepass_cuda")
    if fails:
        print("1 FAIL", flush=True)
        sys.exit(1)
    dev = torch.device("cuda")
    torch.manual_seed(85)
    ext = latent._onepass_ext()
    timing = "--time" in sys.argv
    cap = 600_000 + R + 8
    cache = kv8.quantize_rows(torch.randn(cap, LW, device=dev))
    for P in (35_000, 113_000, 300_000, 590_000):
        for kind in ("recent", "spread"):
            qa = (torch.randn(R, 22, LW, device=dev) * 2).to(torch.bfloat16)
            tok = selection(kind, P, R, dev)
            counts = (NSEL - torch.randint(0, 4, (R,), device=dev)).to(torch.int32)
            counts[:8] = torch.tensor([0, 1, 5, 31, 32, 33, 100, 0], dtype=torch.int32)
            ref, out = compare(f"P {P} {kind} 22 heads", qa, cache, tok, counts)
            if P == 35_000 and kind == "spread":
                err = oracle(qa, cache, tok, counts, out, [1, 2, 6, 100, 777, 2047], SCALE)
                check(err < 2e-2, f"P {P} {kind}: against float64 attention, max error {err:.2e} of max (bf16 out)")
    P = 113_000
    qa = (torch.randn(R, 21, LW, device=dev) * 2).to(torch.bfloat16)
    tok = selection("spread", P, R, dev)
    counts = torch.full((R,), NSEL, dtype=torch.int32, device=dev)
    compare(f"P {P} spread 21 heads", qa, cache, tok, counts)
    qa = (torch.randn(R, 22, LW, device=dev) * 2).to(torch.bfloat16)
    ragged = torch.randint(0, NSEL + 1, (R,), device=dev).to(torch.int32)
    compare(f"P {P} spread ragged counts 0-2,051", qa, cache, tok, ragged)
    for rows in (1024, 333):
        compare(f"P {P} spread {rows}-row chunk", qa[:rows].contiguous(), cache, tok[:rows].contiguous(),
                ragged[:rows].contiguous())
    # rows past a count are never read: the cache's first and last rows NaN codes, the lists' padding -1 / the last
    # row / out of range (no listed token is row 0)
    poison = cache.clone()
    poison[0, :LW] = 0x7F
    poison[-1, :LW] = 0x7F
    tok2 = torch.where(tok == 0, torch.full_like(tok, 4), tok)
    cut = torch.randint(1, NSEL, (R,), device=dev).to(torch.int32)
    past = torch.arange(NSEL, device=dev)[None, :] >= cut[:, None]
    tok2[past] = torch.tensor([-1, cap - 1, 2 ** 31 - 1], dtype=torch.int32, device=dev)[
        torch.randint(0, 3, (int(past.sum()),), device=dev)]
    ref, out = compare(f"P {P} padding past the counts (-1 / a NaN row / huge)", qa, poison, tok2, cut)
    check(not bool(out.isnan().any()), "padding: no NaN reached an output row")
    # NaN codes in rows that are read (only a NaN input writes one): the outputs that see them carry Triton's NaN bits
    nanc = cache.clone()
    nanc[tok[::7, 100].long(), 5] = 0x7F
    full = torch.full((R,), NSEL, dtype=torch.int32, device=dev)
    ref, out = compare(f"P {P} NaN codes in read rows (Triton's NaN bits)", qa, nanc, tok, full)
    check(bool(ref.isnan().any()), f"NaN codes: the reference has NaN outputs ({int(ref.isnan().sum())})")
    # arbitrary row scales and softmax scale: every product's order pinned
    arb = cache.clone()
    arb[:, LW:LW + 4].view(torch.float32)[:, 0] = torch.rand(arb.shape[0], device=dev) * 3 + 0.1
    for kind in ("recent", "spread"):
        tok = selection(kind, 35_000, R, dev)
        counts = (NSEL - torch.randint(0, 4, (R,), device=dev)).to(torch.int32)
        compare(f"arbitrary row scales, softmax scale 192^-0.5, P 35000 {kind}", qa, arb, tok, counts, 192 ** -0.5)
    # the entry point: launches counted; TP=2's 32 heads and a bf16 cache stay on Triton
    tok = selection("spread", 113_000, R, dev)
    counts = torch.full((R,), NSEL, dtype=torch.int32, device=dev)
    got = {}
    for flag in (True, False):
        counter = Count(ext)
        latent._ONEPASS_EXT = counter
        try:
            got[flag] = (run(qa, cache, tok, counts, SCALE, flag), counter.calls)
        finally:
            latent._ONEPASS_EXT = ext
    same = torch.equal(got[True][0].view(torch.int16), got[False][0].view(torch.int16))
    check(same and got[True][1] == 1 and got[False][1] == 0,
          f"sparse_onepass entry: CUDA {got[True][1]} launch (want 1) / Triton {got[False][1]} (want 0), outputs "
          f"equal {same}")
    for name, q, c in (("32 heads a rank (TP=2)", (torch.randn(R, 32, LW, device=dev)).to(torch.bfloat16), cache),
                       ("a bf16 cache", qa, torch.randn(cache.shape[0], LW, device=dev).to(torch.bfloat16))):
        counter = Count(ext)
        latent._ONEPASS_EXT = counter
        try:
            out = run(q, c, tok, counts, SCALE, True)
            ok = not bool(out[counts > 0].isnan().any())
        except Exception as e:      # a launch the kernel refuses is a failure of the routing, reported as one
            print(f"  {name}: {type(e).__name__}: {str(e)[:200]}")
            ok = False
        finally:
            latent._ONEPASS_EXT = ext
        check(counter.calls == 0 and ok, f"{name}: stays on Triton (CUDA launches {counter.calls}, want 0)")
    os.environ.pop("TF_GLM_ONEPASS_CUDA", None)
    check(latent.onepass_cuda(), "TF_GLM_ONEPASS_CUDA unset: on")
    # start-up
    loader = (latent, "_onepass_ext")
    os.environ["TF_GLM_ONEPASS_CUDA"] = "1"
    check(construct(loader, "fp8") == 1, "Engine(FP8 caches, DSA layers): the extension loaded at start")
    check(construct(loader, "bf16") == 0, "Engine(bf16 caches): not loaded (the kernel is FP8 only)")
    check(construct(loader, "fp8", ("mla",)) == 0, "Engine(no DSA layer): not loaded")
    os.environ["TF_GLM_ONEPASS_CUDA"] = "0"
    check(construct(loader, "fp8") == 0, "TF_GLM_ONEPASS_CUDA=0: not loaded")
    # the extension not used (a failed build or load; another Triton release): logged once, Triton serves
    import contextlib
    import io

    from tensorfold.cuda import build
    from tensorfold.families.glm5_next.cuda import sparse

    tok = selection("spread", 113_000, R, dev)
    counts = torch.full((R,), NSEL, dtype=torch.int32, device=dev)
    ref = run(qa, cache, tok, counts, SCALE, False)                # Triton on these inputs
    real_load, real_ptx = build.load, sparse.TRITON_PTX
    for case, ptx, want_tries in (("a failed build", real_ptx, 1), ("another Triton release", "0.0", 0)):
        tries, buf = [], io.StringIO()

        def broken(*a, **k):
            tries.append(1)
            raise RuntimeError("Error building extension 'tensorfold_glm_onepass_v1': nvcc not found")

        build.load, sparse.TRITON_PTX, latent._ONEPASS_EXT, latent._ONEPASS_FAILED = broken, ptx, None, None
        raised = first = second = res = None
        on = True
        try:
            with contextlib.redirect_stdout(buf):
                first, second = latent._onepass_ext(), latent._onepass_ext()
                on = latent.onepass_cuda()
                res = run(qa, cache, tok, counts, SCALE, True)
        except Exception as exc:
            raised = exc
        finally:
            build.load, sparse.TRITON_PTX, latent._ONEPASS_EXT, latent._ONEPASS_FAILED = real_load, real_ptx, ext, None
        logged = buf.getvalue().count("sparse_onepass.cu is not used")
        check(raised is None and first is None and second is None and len(tries) == want_tries and not on
              and logged == 1,
              f"{case}: nothing raised ({raised!r}), logged once ({logged}), loader called {len(tries)} "
              f"(want {want_tries}), the switch reads off ({not on})")
        check(res is not None and torch.equal(res.view(torch.int16), ref.view(torch.int16)),
              f"{case}: sparse_onepass gives the Triton kernel's output")
    if timing and fails == 0:
        for heads in (22, 21):
            qa = (torch.randn(R, heads, LW, device=dev) * 2).to(torch.bfloat16)
            out = torch.empty_like(qa)
            for P in (113_000, 590_000):
                for kind in ("recent", "spread"):
                    tok = selection(kind, P, R, dev)
                    counts = torch.full((R,), NSEL, dtype=torch.int32, device=dev)
                    os.environ["TF_GLM_ONEPASS_CUDA"] = "0"
                    t0 = ms(lambda: latent.sparse_onepass(qa, cache, tok, counts, out, SCALE))
                    os.environ["TF_GLM_ONEPASS_CUDA"] = "1"
                    t1 = ms(lambda: latent.sparse_onepass(qa, cache, tok, counts, out, SCALE))
                    print(f"TIME {heads} heads, a 2,048-row chunk at {P} ({kind}): Triton {t0:.3f} -> CUDA {t1:.3f} ms "
                          f"({100 * (t1 / t0 - 1):+.1f}%)", flush=True)
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
