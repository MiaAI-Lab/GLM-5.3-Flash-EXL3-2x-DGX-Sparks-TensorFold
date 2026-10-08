"""0102 and 0103 on the GPU at the indexer's real shapes (32 heads of 128, pooled keys FP8 or bf16):

- 0103: _scores_pair gives _scores' bits for every row, and the prompt selection with TF_GLM_PROMPT_SELECT (stacked
  scoring, compacted top-k) the same tokens and counts as without, from a few thousand to 128k tokens, for one row to a
  full block, and for rows of many equal scores (the top-k's fallback);
- 0102: each rank's own rows of a 2,048-row chunk (TP=2, 3, 4) selected alone (select_pools) give the whole chunk's
  tokens and counts (pool_tokens)."""

import pytest
import torch
import triton

from tensorfold.families.glm5_next.cuda import kv8, sparse

H, D = 32, 128


def case(ctx: int, R: int, kind: str, seed: int, equal: bool = False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    np_max = ctx // 4 + 2
    k = torch.randn(np_max, D, device="cuda", generator=g)
    if equal:
        k[: np_max // 2] = k[0]
    pk = kv8.quantize_rows(k.to(torch.bfloat16)) if kind == "fp8" else k.to(torch.bfloat16)
    qi = (torch.randn(R, H * D, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    wts = torch.randn(R, H, device="cuda", generator=g).to(torch.bfloat16)
    pos = ctx - R
    return qi, wts, pk, pos, np_max, torch.tensor([pos], dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("ctx,R,kind", [(32768, 512, "fp8"), (131072, 300, "bf16"), (4096, 7, "fp8")])
def test_scores_pair_bits(ctx, R, kind):
    qi, wts, pk, pos, np_max, pos_dev = case(ctx, R, kind, ctx + R)
    NP = sparse.pool_count(pos, R, np_max)
    pkv, pks, rs, fp8 = kv8.parts(pk)
    a = torch.empty((R, NP), dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    args = (qi, wts, wts.stride(0), pkv, pks)
    sparse._scores[(triton.cdiv(R, 4), triton.cdiv(NP, 64))](*args, a, pos_dev, R, NP, D ** -0.5, 1 / 5.656854249492381,
                                                             H=H, HP=32, D=D, BP=64, RB=4, RS=rs, FP8=fp8, num_warps=4)
    sparse._scores_pair[(triton.cdiv(R, 2), triton.cdiv(NP, 128))](*args, b, pos_dev, R, NP, D ** -0.5,
                                                                   1 / 5.656854249492381, H=H, HP=32, D=D, BP=128,
                                                                   RB=2, RS=rs, FP8=fp8, num_warps=4)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("ctx,R,kind,equal", [(4096, 512, "fp8", False), (8192, 171, "fp8", False),
                                              (32768, 1, "bf16", False), (131072, 512, "fp8", False),
                                              (2200, 64, "fp8", False), (65536, 97, "fp8", True)])
def test_prompt_select_same_tokens(monkeypatch, ctx, R, kind, equal):
    qi, wts, pk, pos, np_max, pos_dev = case(ctx, R, kind, ctx * 7 + R, equal)
    got = {}
    for on in (False, True):
        monkeypatch.setattr(sparse, "PROMPT_SELECT", on)
        got[on] = sparse._select_prompt(qi, wts, pk, pos, R, np_max, pos_dev)
    assert torch.equal(got[False][1], got[True][1]) and torch.equal(got[False][0], got[True][0])


@pytest.mark.parametrize("ctx,kind", [(2600, "fp8"), (32768, "fp8"), (32768, "bf16"), (131072, "fp8")])
def test_own_rows_give_the_chunks_tokens(ctx, kind):
    R = 2048
    qi, wts, pk, pos, np_max, pos_dev = case(ctx, R, kind, ctx)
    tok, cnt = sparse._select_prompt(qi, wts, pk, pos, R, np_max, pos_dev)
    counted = cnt > 0
    for world in (2, 3, 4):
        share = -(-R // world)
        pools = torch.empty((R, sparse.TOPK_POOLS), dtype=torch.int32, device="cuda")
        for r in range(world):
            lo, hi = r * share, min(R, r * share + share)
            sparse.select_pools(qi[lo:hi], wts[lo:hi], pk, pos + lo, hi - lo, np_max, pos_dev + lo, pools[lo:hi])
        t2, c2 = torch.empty_like(tok), torch.empty_like(cnt)
        sparse.pool_tokens(pools, pos_dev, t2, c2)
        assert torch.equal(cnt, c2) and torch.equal(tok[counted], t2[counted]), world
