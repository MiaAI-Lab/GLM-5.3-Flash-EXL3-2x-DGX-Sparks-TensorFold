"""TP4 CUDA chunk pass: exact outputs AND partials, boundaries, sparse rows, graph replay."""
import pytest
import torch

from tensorfold.families.glm5_next.cuda import latent, kv8, sparse
from tensorfold.families.glm5_next.cuda.segments import SegRows, EXTENT


@pytest.mark.parametrize("rows,position", [(1, 0), (4, 31), (4, 32), (4, 511), (4, 512),
                                          (7, 2047), (8, 2048), (16, 2051), (32, 8192), (64, 8192)])
def test_exact_partials_and_graph(monkeypatch, rows, position):
    if torch.cuda.get_device_capability() != (12, 1) or latent.triton.__version__ != "3.7.1":
        pytest.skip("CUDA replacement is qualified only on GB10 / Triton 3.7.1")
    monkeypatch.setenv("TF_GLM_SEG_CHUNKS_CUDA", "1")
    assert latent._seg_chunks_ext() is not None
    torch.manual_seed(rows + position)
    capacity = ((position + rows + EXTENT - 1) // EXTENT + 1) * EXTENT
    cache = kv8.quantize_rows(torch.randn(capacity, 512, device="cuda", dtype=torch.bfloat16))
    qa = torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
    segments = SegRows(rows, "cuda", max_segs=8)
    segments.set([(EXTENT, position, rows)])
    tokens = torch.full((rows, sparse.TOKENS), -1, device="cuda", dtype=torch.int32)
    counts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    for i in range(rows):
        if position + i >= sparse.SPARSE_FROM:
            n = min(sparse.TOKENS, position + i + 1)
            tokens[i, :n] = torch.randperm(position + i + 1, device="cuda")[:n].sort().values.int()
            counts[i] = n
    scratch = latent.LatentScratch(rows, 16, latent.seg_chunks(), "cuda")
    out = torch.empty_like(qa)
    def run(hb=None, tokenless=False):
        latent.seg_attention(qa, cache, segments, None if tokenless else tokens,
                             None if tokenless else counts, scratch, scale=256**-.5, out=out, hb=hb)
    def tensors():
        return out, scratch.po, scratch.pm, scratch.pl
    def exact(reference):
        for actual, expected in zip(tensors(), reference):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    run(hb=16)                           # explicit tile bypasses the opt-in CUDA path
    reference = [t.clone() for t in tensors()]
    run(); torch.cuda.synchronize(); exact(reference)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): run()
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): run()
    for _ in range(2):
        qa.normal_(); run(hb=16); reference = [t.clone() for t in tensors()]
        out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize(); exact(reference)
    if position + rows < sparse.SPARSE_FROM:
        run(hb=16, tokenless=True); reference = [t.clone() for t in tensors()]
        run(tokenless=True); torch.cuda.synchronize(); exact(reference)
