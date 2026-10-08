"""0105 (TF_GLM_SIDE=1) on the GPU: a decode window's MoE shared expert on a second stream beside the router and the
routed EXL3 experts. On the tiny checkpoint with a MoE layer (flash_fakes, moe=True; one rank, 4-bit dense weights):
decode windows of 1, 4 and 16 rows, eager and replayed from a CUDA graph, give the one-stream path's logits bit for
bit, window after window; prompt buffers get no side stream; TF_GLM_SIDE takes 0 or 1."""

import pytest
import torch

import flash_fakes


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return flash_fakes.write_checkpoint(tmp_path_factory.mktemp("flashside"), moe=True)


def run(folder, side: bool, graph: bool, monkeypatch):
    monkeypatch.setenv("TF_GLM_SIDE", "1" if side else "0")
    for k, v in dict(TF_GLM_DENSE="q4", TF_GLM_KVB="bf16", TF_GLM_KV="fp8", TF_GLM_LATENT="1",
                     TF_GLM_L2PF="0").items():
        monkeypatch.setenv(k, v)
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=0, world=1, device="cuda", mtp=False)
    b = F.Buffers(w, 16, capacity=512)
    assert (b.side is not None) == side
    on_side = []
    real = F.shared_front

    def shared_front(*a, **k):                 # where the shared expert ran: the side stream, or the main one
        on_side.append(b.side is not None and torch.cuda.current_stream() == b.side)
        return real(*a, **k)

    monkeypatch.setattr(F, "shared_front", shared_front)
    assert F.Buffers(w, 64, capacity=512, prefill=True).side is None
    st = F.State(w, 512, 16, kv="fp8")
    g = torch.Generator().manual_seed(3)
    out = []
    for R in (16, 1, 4, 16, 1):
        toks = torch.randint(2, flash_fakes.V, (R,), generator=g).tolist()
        if graph:
            F.stage(w, st, b, toks)
            nch = F.chunks_for(st, R)
            cg = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                warm = st.clone()
                F.compute(w, warm, b, R, nch=nch)
                torch.cuda.synchronize()
                with torch.cuda.graph(cg, stream=s):
                    F.compute(w, st, b, R, nch=nch)
            torch.cuda.synchronize()
            cg.replay()
            logits = b.logits[:R]
        else:
            logits = F.forward(w, st, b, toks)
        torch.cuda.synchronize()
        out.append(logits.clone())
        F.commit(w, st, b, R, R)
    assert on_side and all(x == side for x in on_side)
    return out


@pytest.mark.parametrize("graph", [False, True])
def test_side_stream_same_bits(folder, monkeypatch, graph):
    one = run(folder, False, False, monkeypatch)
    two = run(folder, True, graph, monkeypatch)
    for a, b in zip(one, two):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_switch(monkeypatch):
    from tensorfold.families.glm5_next.cuda import forward as F

    monkeypatch.delenv("TF_GLM_SIDE", raising=False)
    assert F.side_stream() is False
    monkeypatch.setenv("TF_GLM_SIDE", "1")
    assert F.side_stream() is True
    monkeypatch.setenv("TF_GLM_SIDE", "2")
    with pytest.raises(ValueError):
        F.side_stream()
