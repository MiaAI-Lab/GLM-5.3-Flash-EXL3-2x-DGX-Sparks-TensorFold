"""0103's compacted top-k (sparse._select_compact) in Triton's interpreter: from the same scores, the same pools as
_select_rows (top_pools), whether a row's candidates fit the compacted buffer or not (its fallback over every score),
with ties, -inf columns and rows of fewer visible pools than k; the switch's default."""

import importlib

import pytest
import torch

from tensorfold.families.glm5_next.cuda import sparse


def scores_case(R: int, NP: int, seed: int, kind: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(R, NP, generator=g)
    if kind == "ties":                  # many equal values around the threshold
        s = torch.round(s * 4) / 4
    elif kind == "inf":                 # rows with few visible pools: -inf past them
        for r in range(R):
            s[r, int(torch.randint(1, NP, (1,), generator=g)):] = float("-inf")
    elif kind == "zeros":
        s[:, ::3] = 0.0
        s[:, 1::3] = -0.0
    return s.contiguous()


@pytest.mark.parametrize("kind", ["plain", "ties", "inf", "zeros"])
@pytest.mark.parametrize("cap", [8, 2048])
def test_compact_same_pools(monkeypatch, kind, cap):
    monkeypatch.setattr(sparse, "COMPACT_CAP", cap)
    R, NP, K = 3, 1536, 512
    s = scores_case(R, NP, 3 + cap, kind)
    want = sparse.top_pools(s, K)
    got = sparse.select_compact(s, K)
    assert torch.equal(want, got)


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv("TF_GLM_PROMPT_SELECT", raising=False)
    assert importlib.reload(sparse).PROMPT_SELECT is True
    monkeypatch.setenv("TF_GLM_PROMPT_SELECT", "0")
    assert importlib.reload(sparse).PROMPT_SELECT is False
    monkeypatch.delenv("TF_GLM_PROMPT_SELECT")
    importlib.reload(sparse)
