#!/usr/bin/env python3
"""tools/seg_head_block_check.py [--time]: patch 0104 - latent.seg_head_block(R, H): a rank of 17-24 heads (TP=3:
22 / 21) takes its heads in one 32-head tile a row from 5 rows, not 8; 16 (TP=4) and 32 (TP=2) heads keep the 8-row rule.
Run in the image built with patch 0104, on one GPU:
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/seg_head_block_check.py:/c.py" <image> /c.py

1. bits: latent.seg_attention under the new rule returns, for every call, the whole output (pre-filled with NaN) of the
   same call with the old rule's tiles (32 heads from 8 rows, else 16) forced through ``hb``: windows of 1-16 rows in
   1, 2 and 4 streams at ~180k / ~300k / ~596k, 22 and 21 heads, sparse rows over 2,051 selected tokens, and a dense
   window (rows over their stream's first keys, ``tokens`` None); heads a tile never change a row's bits;
2. mechanism: the launch grid of the Triton chunk pass (_seg_chunks, recorded) has ceil(H / 32) head blocks exactly
   where the rule says: from 5 rows at 22 and 21 heads, from 8 rows at 32 heads (TP=2), else ceil(H / 16).
The CUDA chunk pass of patch 0108 (when the image has it) is switched off here (TF_GLM_SEG_CHUNKS_CUDA=0): it gives
either tile the same bits and does not read the head tile. One PASS / FAIL line a check; exits 1 on any FAIL (on an
image without the patch: at the first check).

--time: cold (L2 flushed), 11 calls a CUDA graph (each its own keys), the new rule against the old one at 4-8 rows,
22 heads (TP=3 rank 0); at 32 heads (TP=2) the rule does not change."""
import inspect
import os
import statistics
import sys

os.environ["TF_GLM_SEG_CHUNKS_CUDA"] = "0"
os.environ["TF_GLM_LATENT_STAGES"] = "3"

import torch  # noqa: E402
import triton  # noqa: E402

from tensorfold.families.glm5_next.cuda import kv8, latent, sparse  # noqa: E402
from tensorfold.families.glm5_next.cuda.segments import SegRows  # noqa: E402

fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def old_rule(R):
    """seg_head_block before the patch (SEG_WIDE_ROWS = 8 at any head count)."""
    return latent.HB_WIDE if R >= 8 else latent.HB


check("H" in inspect.signature(latent.seg_head_block).parameters and hasattr(latent, "SEG_FEW_HEADS"),
      "patch 0104 applied: seg_head_block(R, H)")
if fails:
    print("1 FAIL", flush=True)
    sys.exit(1)
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(88)
LW = 512
TOK = sparse.TOKENS
grids = []
real = latent._seg_chunks


class Recorder:
    def __getitem__(self, grid):
        grids.append(tuple(grid))
        return real[grid]


latent._seg_chunks = Recorder()


def cache_for(rows):
    c = kv8.zeros(rows, LW, "fp8", dev)
    codes = torch.randint(0, 0x7E, (rows, LW), dtype=torch.int32, device=dev, generator=g)
    sign = torch.randint(0, 2, (rows, LW), dtype=torch.int32, device=dev, generator=g) * 0x80
    c[:, :LW] = (codes | sign).to(torch.uint8)
    c[:, LW:LW + 4] = torch.tensor([2.0 ** -8], dtype=torch.float32, device=dev).view(torch.uint8).repeat(rows, 1)
    return c


def selections(streams, rows_per, ctx):
    toks = torch.zeros((streams * rows_per, TOK + 1), dtype=torch.int32, device=dev)
    for st in range(streams):
        pools = torch.randperm(ctx // 4 - 1, device=dev, generator=g)[:sparse.TOPK_POOLS].sort().values
        t = (pools[:, None] * 4 + torch.arange(4, device=dev)[None, :]).reshape(-1)
        for j in range(rows_per):
            p = ctx + j
            sel = torch.cat([t, torch.arange(p // 4 * 4, p + 1, device=dev)])[:TOK]
            toks[st * rows_per + j, :sel.numel()] = sel.to(torch.int32)
    return toks


def window(streams, rows_per, ctx, H, dense=False):
    R = streams * rows_per
    stride = ctx + 4096
    cache = cache_for(streams * stride)
    rows = SegRows(R, dev, max_segs=max(4, streams))
    rows.pos.copy_(torch.tensor([ctx + j for _ in range(streams) for j in range(rows_per)], dtype=torch.int32))
    rows.base.copy_(torch.tensor([s * stride for s in range(streams) for _ in range(rows_per)], dtype=torch.int32))
    qa = (torch.randn((R, H, LW), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    s = latent.LatentScratch(max(R, 32), H, latent.seg_chunks(), dev, lw=LW)
    if dense:
        rows.sparse.fill_(0)
        return R, cache, rows, None, None, qa, s
    rows.sparse.fill_(1)
    counts = torch.full((R,), TOK, dtype=torch.int32, device=dev)
    return R, cache, rows, selections(streams, rows_per, ctx), counts, qa, s


def run(R, cache, rows, toks, counts, qa, s, hb):
    out = torch.full(qa.shape, float("nan"), dtype=torch.bfloat16, device=dev)
    grids.clear()
    latent.seg_attention(qa, cache, rows, toks, counts, s, scale=0.0625, out=out, hb=hb)
    torch.cuda.synchronize()
    return out, list(grids)


for H in (22, 21):
    for streams, ctx, rlist, dense in ((1, 180000, range(1, 17), False), (2, 300000, (1, 2, 3, 4, 6), False),
                                       (4, 596000, (1, 2, 3, 6), False), (1, 1500, (1, 3, 5, 7, 9), True)):
        for rp in rlist:
            R, cache, rows, toks, counts, qa, s = window(streams, rp, ctx, H, dense)
            new, g_new = run(R, cache, rows, toks, counts, qa, s, None)
            old, _ = run(R, cache, rows, toks, counts, qa, s, old_rule(R))
            want = latent.HB_WIDE if R >= 5 else latent.HB
            check(not torch.isnan(new.float()).any() and torch.equal(new.view(torch.int16), old.view(torch.int16))
                  and len(g_new) == 1 and g_new[0][1] == triton.cdiv(H, want),
                  f"{H} heads, {streams} x {rp} rows ({R}) at {ctx}{' dense' if dense else ''}: the old rule's bits, "
                  f"grid {g_new} ({triton.cdiv(H, want)} head blocks wanted)")
            del cache, s
            torch.cuda.empty_cache()
check(all(latent.seg_head_block(R, 32) == old_rule(R) for R in range(1, 25)),
      "32 heads a rank (TP=2): the old rule at 1-24 rows")
check(all(latent.seg_head_block(R, H) == (latent.HB_WIDE if R >= 5 else latent.HB)
          for H in (17, 21, 22, 24) for R in range(1, 25)), "17-24 heads a rank (TP=3): 32-head tiles from 5 rows")
check(all(latent.seg_head_block(R, H) == old_rule(R) for H in (16, 25) for R in range(1, 25)),
      "16 heads (TP=4) and 25 heads a rank: the old rule")

if "--time" in sys.argv and fails == 0:
    latent._seg_chunks = real
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

    for streams, rp, ctx in ((1, 4, 180000), (1, 5, 180000), (1, 6, 180000), (1, 7, 180000), (1, 8, 180000),
                             (1, 5, 596000), (1, 6, 596000), (2, 3, 300000), (4, 2, 596000)):
        R, cache, rows, _, counts, qa, s = window(streams, rp, ctx, 22)
        tl = [selections(streams, rp, ctx) for _ in range(11)]           # each call (a layer) its own keys
        out = torch.empty(qa.shape, dtype=torch.bfloat16, device=dev)
        res = [graph_us(lambda hb=hb: [latent.seg_attention(qa, cache, rows, t, counts, s, scale=0.0625, out=out,
                                                            hb=hb) for t in tl]) for hb in (old_rule(R), None)]
        print(f"TIME 22 heads, {streams} x {rp} rows at {ctx}: old rule {res[0]:.1f} -> {res[1]:.1f} us a call "
              f"({100 * (res[1] / res[0] - 1):+.1f}%)", flush=True)
        del cache, s, tl
        torch.cuda.empty_cache()
print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
sys.exit(1 if fails else 0)
