"""0101: a prompt chunk's sparse attention. On CPU: sparse_onepass's launch rule (4 warps for one 16-row tile, the
same bits as 8 on the GPU: tests/gpu/test_prompt_msa_gpu.py), TF_GLM_MSA's default and what msa.cu takes, and the
switch joining the ranks' startup comparison. The kernels themselves: tests/gpu."""

import importlib
import inspect

import torch

from tensorfold.families.glm5_next.cuda import kv8, latent, msa


def test_launch_rule(monkeypatch):
    got = []

    class Spy:
        def __getitem__(self, grid):
            def run(*a, **kw):
                got.append((grid, kw["HBT"], kw["num_warps"]))
            return run

    monkeypatch.setattr(latent, "_sparse_onepass", Spy())
    cache = kv8.zeros(64, 512, "fp8", "cpu")
    tok = torch.zeros((3, 2051), dtype=torch.int32)
    cnt = torch.zeros((3,), dtype=torch.int32)
    for H in (16, 12, 22, 32):
        qa = torch.zeros((3, H, 512), dtype=torch.bfloat16)
        latent.sparse_onepass(qa, cache, tok, cnt, torch.empty_like(qa), 0.0625)
    assert got == [((3, 1, 1), 16, 4), ((3, 1, 1), 16, 4), ((3, 1, 1), 32, 8), ((3, 1, 1), 32, 8)]
    # an explicit tile or launch (benchmarks) is taken as given
    got.clear()
    qa = torch.zeros((3, 16, 512), dtype=torch.bfloat16)
    latent.sparse_onepass(qa, cache, tok, cnt, torch.empty_like(qa), 0.0625, launch=(8, 1, True))
    assert got == [((3, 1, 1), 16, 8)]


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv("TF_GLM_MSA", raising=False)
    assert importlib.reload(msa).ENABLED is False
    monkeypatch.setenv("TF_GLM_MSA", "1")
    assert importlib.reload(msa).ENABLED is True
    monkeypatch.delenv("TF_GLM_MSA")
    importlib.reload(msa)


def test_supported_fp8_only():
    qa = torch.zeros((2, 32, 512), dtype=torch.bfloat16)
    assert not msa.supported(qa, kv8.zeros(8, 512, "fp8", "cpu"))          # CPU tensors: never
    meta = torch.zeros((2, 32, 512), dtype=torch.bfloat16, device="meta")

    class Cuda:                                                           # what supported() reads of a tensor
        def __init__(self, t):
            self.is_cuda, self.dtype, self.shape, self.dim = True, t.dtype, t.shape, t.dim

    fp8 = Cuda(torch.zeros((8, 528), dtype=torch.uint8, device="meta"))
    bf16 = Cuda(torch.zeros((8, 512), dtype=torch.bfloat16, device="meta"))
    assert msa.supported(Cuda(meta), fp8)
    assert not msa.supported(Cuda(meta), bf16)                             # bf16 caches keep sparse_onepass
    assert not msa.supported(Cuda(torch.zeros((2, 33, 512), dtype=torch.bfloat16, device="meta")), fp8)


def test_startup_comparison_names_it():
    from tensorfold.families.glm5_next.cuda import engine

    src = inspect.getsource(engine)
    assert "int(MSA)]" in src and "TF_GLM_MSA)" in src
