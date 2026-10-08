"""0101 on the GPU: msa.cu (TF_GLM_MSA=1) at a rank's heads (32: TP=2, 22 / 21: TP=3, 16: TP=4) on an FP8 latent
cache: within bf16 rounding of exact attention (no further than sparse_onepass), the same bits on a repeat and for any
block of rows, rows without a sparse count left alone; and sparse_onepass's 4-warp launch at up to 16 heads keeps
the 8 warps' bits (FP8 and bf16)."""

import pytest
import torch

from tensorfold.families.glm5_next.cuda import kv8, latent, msa

LW, W, CTX = 512, 2051, 8192


def lists(R: int, g: torch.Generator) -> torch.Tensor:
    tok = torch.full((R, W), -1, dtype=torch.int32)
    for r in range(R):
        pools = torch.randperm(CTX // 4 - 1, generator=g)[:512].sort().values
        tok[r, :2048] = (pools[:, None] * 4 + torch.arange(4)).reshape(-1).to(torch.int32)
        tok[r, 2048:2050] = torch.tensor([CTX - 3, CTX - 2], dtype=torch.int32)
    return tok.cuda()


@pytest.fixture(scope="module")
def case():
    g = torch.Generator().manual_seed(5)
    lat = (torch.randn(CTX, LW, generator=g) * 2).to(torch.bfloat16).cuda()
    R = 96
    tok = lists(R, g)
    cnt = torch.randint(1, 2051, (R,), generator=g, dtype=torch.int32)
    cnt[:5] = torch.tensor([1, 15, 16, 17, 2050], dtype=torch.int32)
    cnt[5:9] = 0
    return lat, tok, cnt.cuda()


@pytest.mark.parametrize("H", [32, 22, 21, 16, 9, 1])
def test_msa_against_exact_and_its_own_bits(case, H):
    lat, tok, cnt = case
    cache = kv8.quantize_rows(lat)
    deq = kv8.dequantize(cache)
    R = tok.shape[0]
    g = torch.Generator().manual_seed(H)
    qa = (torch.randn(R, H, LW, generator=g) * 0.05).to(torch.bfloat16).cuda()
    out = torch.full_like(qa, 7.0)
    msa.prompt(qa, cache, tok, cnt, out, 0.0625)
    one = torch.full_like(qa, 7.0)
    latent.sparse_onepass(qa, cache, tok, cnt, one, 0.0625)
    worst = worst_one = 0.0
    for r in range(R):
        n = int(cnt[r])
        if n == 0:
            assert torch.equal(out[r], torch.full_like(out[r], 7.0))
            continue
        k = deq[tok[r, :n].long()]
        exact = torch.softmax((qa[r].float() @ k.T) * 0.0625, -1) @ k
        worst = max(worst, (out[r].float() - exact).abs().max().item())
        worst_one = max(worst_one, (one[r].float() - exact).abs().max().item())
    assert worst <= max(worst_one, 1e-3) * 1.25
    again = torch.full_like(qa, 7.0)
    msa.prompt(qa, cache, tok, cnt, again, 0.0625)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
    blocks = torch.full_like(qa, 7.0)
    for a, z in ((0, 1), (1, 40), (40, R)):
        msa.prompt(qa[a:z].contiguous(), cache, tok[a:z].contiguous(), cnt[a:z].contiguous(), blocks[a:z], 0.0625)
    assert torch.equal(out.view(torch.int16), blocks.view(torch.int16))


@pytest.mark.parametrize("kind", ["fp8", "bf16"])
@pytest.mark.parametrize("H", [16, 12, 1])
def test_onepass_four_warps_same_bits(case, kind, H):
    lat, tok, cnt = case
    cache = kv8.quantize_rows(lat) if kind == "fp8" else lat
    g = torch.Generator().manual_seed(100 + H)
    qa = (torch.randn(tok.shape[0], H, LW, generator=g) * 0.05).to(torch.bfloat16).cuda()
    new, old = torch.zeros_like(qa), torch.zeros_like(qa)
    latent.sparse_onepass(qa, cache, tok, cnt, new, 0.0625)
    latent.sparse_onepass(qa, cache, tok, cnt, old, 0.0625, launch=latent.ONEPASS_LAUNCH)
    assert torch.equal(new.view(torch.int16), old.view(torch.int16))
