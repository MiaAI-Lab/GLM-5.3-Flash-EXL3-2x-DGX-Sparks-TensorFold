#!/usr/bin/env python3
"""tools/prompt_tile_check.py [--time]: patch 0102 - GLM's dense 4-bit projections of a prompt chunk (forward.mm ->
the shared prefill matmul) launch forward.prompt_tile's tile (9 where K >= 1024 and the grid has 96 or more 128 x 128
blocks, else 3) instead of tile 0. Run in the image built with patch 0102, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/prompt_tile_check.py:/c.py" <image> /c.py

At every projection shape a 2,048-row prompt chunk runs on rank 0, at TP=3 (22 of 64 heads) and TP=2 (32), from the
model's geometry (tp.split_sizes; the row counts are hcsplit's pieces of the chunk):
1. forward.prompt_tile gives the measured table at TP=3, and tile 0 with TF_GLM_PROMPT_TILE=0;
2. forward.mm on a prompt chunk's buffers launches the shared prefill matmul with that tile (recorded) and its output
   equals tile 0's bit for bit (bf16 and fp32 outputs as the call sites use them, pre-filled with NaN so an unwritten
   slot shows): a tile only picks which warp computes an output, each output is the same fp32 mma chain over K;
3. the output against the dequantized weights times x in float64 (the kernel computes the projection).
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check).

--time: each shape's time a call on tile 0 and on the rule (CUDA-graph replays), and a chunk's total weighted by the
calls a chunk makes."""
import sys
from types import SimpleNamespace

import torch

from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.glm5_next.cuda import forward, qmm, tp

HEADS, LIN, DENSE, SHARED, HID = 64, 64, 12288, 2048, 4096       # GLM-5.3-Flash's split dimensions and hidden size
KDA, DSA, MOE, DEN = 34, 11, 42, 3                                # layers of each kind (calls a chunk per row piece)


def shapes(world):
    """(label, M, N, K, fp32 out, calls a chunk) for rank 0 of ``world``."""
    h, lh = tp.split_sizes(HEADS, world, 1)[0], tp.split_sizes(LIN, world, 1)[0]
    dn, sh = tp.split_sizes(DENSE, world, 128)[0], tp.split_sizes(SHARED, world, 128)[0]
    if world == 3:   # hcsplit's pieces of a 2,048-row chunk at three ranks: 683 own rows (384 + 299), 1,365 others
        front, post, moe = ((384, 1), (299, 1), (1365, 1)), ((384, 2), (299, 1), (298, 1), (683, 1)), \
            ((384, 3), (299, 2), (298, 1))
    else:            # two ranks: 1,024 own rows in two 512-row pieces, 1,024 others
        front, post, moe = ((512, 2), (1024, 1)), ((512, 4),), ((512, 4),)
    out = []
    for m, c in front:
        out += [(f"kda proj {m}", m, 3 * lh * 128 + 256 + lh, HID, False, KDA * c),
                (f"kda fb/gb {m}", m, lh * 128, 128, False, 2 * KDA * c),
                (f"dsa proj {m}", m, 2048, HID, False, DSA * c), (f"dsa q_b {m}", m, h * 256, 1536, False, DSA * c),
                (f"dense gu {m}", m, 2 * dn, HID, False, DEN * c)]
    for m, c in post:
        out += [(f"kda o {m}", m, HID, lh * 128, True, KDA * c), (f"dsa o {m}", m, HID, h * 256, True, DSA * c),
                (f"dense down {m}", m, HID, dn, True, DEN * c)]
    for m, c in moe:
        out += [(f"shared gu {m}", m, 2 * sh, HID, False, MOE * c), (f"shared down {m}", m, HID, sh, True, MOE * c)]
    return out + [("idx kw 2048", 2048, 256, HID, False, DSA), ("idx qb 2048", 2048, 4096, 1536, False, DSA)]


EXPECTED = {   # TP=3 (m, n, k) -> the tile measured fastest on a GB10 (every tile of the extension tried)
    (384, 8726, 4096): 9, (299, 8726, 4096): 9, (1365, 8726, 4096): 9,
    (384, 2816, 128): 3, (299, 2816, 128): 3, (1365, 2816, 128): 3,
    (384, 2048, 4096): 3, (299, 2048, 4096): 3, (1365, 2048, 4096): 9,
    (384, 5632, 1536): 9, (299, 5632, 1536): 9, (1365, 5632, 1536): 9,
    (384, 8192, 4096): 9, (299, 8192, 4096): 9, (1365, 8192, 4096): 9,
    (384, 4096, 2816): 9, (299, 4096, 2816): 9, (298, 4096, 2816): 9, (683, 4096, 2816): 9,
    (384, 4096, 5632): 9, (299, 4096, 5632): 9, (298, 4096, 5632): 9, (683, 4096, 5632): 9,
    (384, 4096, 4096): 9, (299, 4096, 4096): 9, (298, 4096, 4096): 9, (683, 4096, 4096): 9,
    (384, 1536, 4096): 3, (299, 1536, 4096): 3, (298, 1536, 4096): 3,
    (384, 4096, 768): 3, (299, 4096, 768): 3, (298, 4096, 768): 3,
    (2048, 256, 4096): 3, (2048, 4096, 1536): 9,
}
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def make(n, k, dev, g):
    npad = (n + 127) // 128 * 128
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (npad // 64, k // 64, 8, 32, 2), dtype=torch.int32,
                      device=dev, generator=g)
    s = (torch.rand((k // 64, npad), device=dev, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    b = (torch.rand((k // 64, npad), device=dev, generator=g) * 0.02 - 0.01).to(torch.bfloat16)
    return qmm.Q4(w, s, b, n, k)


def oracle(x, q):
    words, s, b = shared.unpack(shared.Q4(q.weight, q.scales, q.biases, q.n, q.k, q.gs))
    w = qmm.dequantize(words, s, b).to(torch.bfloat16).double()       # [n, k], bf16(q * s + b) as the kernel rounds
    return x.double() @ w.t()


def graph_time(fn, reps=50):
    """Best of 5 timings of reps calls replayed from one CUDA graph, in us a call."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        a.record()
        for _ in range(reps // 10):
            gr.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / reps * 1e3)
    return best


def main():
    check(hasattr(forward, "prompt_tile"), "patch 0102 applied: forward.prompt_tile")
    if fails:
        print("1 FAIL", flush=True)
        sys.exit(1)
    timing = "--time" in sys.argv
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(83)
    table = {(m, n, k): forward.prompt_tile(m, n, k) for _, m, n, k, _, _ in shapes(3)}
    bad = {key: (t, EXPECTED.get(key)) for key, t in table.items() if EXPECTED.get(key) != t}
    check(not bad and len(table) == len(EXPECTED), f"prompt_tile at TP=3 = the measured table over {len(table)} shapes "
                                                   f"({bad or 'all match'})")
    was = forward.PROMPT_TILES
    forward.PROMPT_TILES = False                       # what TF_GLM_PROMPT_TILE=0 sets at import
    check(all(forward.prompt_tile(m, n, k) == 0 for m, n, k in table), "TF_GLM_PROMPT_TILE=0: tile 0 everywhere")
    forward.PROMPT_TILES = was
    seen = []
    orig = shared.prefill_matmul

    def rec(x, q, *, f32=False, tile=0, out=None):
        seen.append(tile)
        return orig(x, q, f32=f32, tile=tile, out=out)

    b = SimpleNamespace(prefill=True, sk=None)
    for world in (3, 2):
        old_t = new_t = 0.0
        for label, m, n, k, f32, calls in shapes(world):
            label = f"TP={world} {label}"
            q = make(n, k, dev, g)
            x = torch.randn((m, k), device=dev, generator=g).to(torch.bfloat16)
            dt, iv = (torch.float32, torch.int32) if f32 else (torch.bfloat16, torch.int16)
            ref = torch.full((m, n), float("nan"), device=dev, dtype=dt)
            orig(x, q, f32=f32, tile=0, out=ref)                          # the kernel as forward.mm launched it before
            out = torch.full((m, n), float("nan"), device=dev, dtype=dt)
            seen.clear()
            forward.shared.prefill_matmul = rec
            try:
                forward.mm(b, x, q, None, out, f32=f32)
            finally:
                forward.shared.prefill_matmul = orig
            want = forward.prompt_tile(m, n, k)
            same = torch.equal(out.view(iv), ref.view(iv)) and not torch.isnan(out).any()
            check(seen == [want] and same, f"{label}: mm launched tile {seen} (want [{want}]), output == tile 0's: "
                                           f"{same}")
            if label.endswith(("1365", "683", "1024")) or "shared down" in label or "idx" in label:
                r = oracle(x, q)
                err = ((out.double() - r).abs().max() / r.abs().max()).item()
                check(err < (2e-4 if f32 else 1e-2), f"{label}: against the float64 oracle, max error {err:.2e} of max")
            if timing:
                t0 = graph_time(lambda: orig(x, q, f32=f32, tile=0, out=ref))
                t1 = graph_time(lambda: forward.mm(b, x, q, None, out, f32=f32))
                old_t, new_t = old_t + calls * t0, new_t + calls * t1
                print(f"TIME {label} ({m} x {n} x {k}, {calls} calls): tile 0 {t0:.1f} us, tile {want} {t1:.1f} us "
                      f"({100 * (t1 / t0 - 1):+.1f}%)", flush=True)
        if timing:
            print(f"TIME TP={world}: a 2,048-row chunk's dense 4-bit projections {old_t / 1e3:.1f} -> "
                  f"{new_t / 1e3:.1f} ms "
                  f"({100 * (new_t / old_t - 1):+.1f}%)", flush=True)
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
