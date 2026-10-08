"""A tiny GLM-5.3-Flash checkpoint for the CPU and GPU tests, in the layout of
Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (BF16 dense weights, the same tensor names), with DSA layers and dense
MLPs: no KDA layer and, unless asked (``moe``, GPU tests: a last layer of EXL3 routed experts and a BF16 shared
expert), no routed expert, whose kernels are CUDA-only, so that a prompt's DSA path (projections, the indexer, its
pooled keys and top-k, latent attention, the row split's glue) runs in Triton's interpreter. Several ranks load it as the engine does (``load_rank``)
and talk over gloo (``GlooComm``: the NCCL calls the row split makes)."""

from __future__ import annotations

import json
import os
import struct
import types
from pathlib import Path

import torch

D, V, S = 256, 256, 4                 # hidden, vocabulary, hyper-connection streams
HEADS, QL, KL = 4, 128, 128           # attention heads (one to four a rank), q_a and kv_a ranks
IH, ID = 4, 128                       # indexer heads and width (the kernels' 128)
DENSE = 512                           # the dense MLP: 128-wide units, so it splits at four ranks too
MOE, EXPERTS = 512, 8                 # an expert's width (128-wide Hadamard blocks a rank) and count
LAYERS = 2


def text_config(layers: int = LAYERS, moe: bool = False) -> dict:
    return {
        "model_type": "glm5_next_text", "hidden_size": D, "num_hidden_layers": layers, "vocab_size": V,
        "rms_norm_eps": 1e-5, "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "q_lora_rank": QL,
        "kv_lora_rank": KL, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0, "v_head_dim": 256,
        "n_routed_experts": EXPERTS, "num_experts_per_tok": 2, "moe_intermediate_size": MOE, "n_shared_experts": 1,
        "intermediate_size": DENSE, "routed_scaling_factor": 2.5, "norm_topk_prob": True, "hc_mult": S,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": IH, "index_head_dim": ID, "index_topk": 2048,
        "index_kpool": 4, "swiglu_limit": 10.0, "layer_types": ["deepseek_sparse_attention"] * layers,
        "mlp_layer_types": ["dense"] * (layers - 1) + ["sparse" if moe else "dense"],
        "first_k_dense_replace": layers - 1 if moe else layers, "eos_token_id": [1],
        "num_nextn_predict_layers": 0, "linear_attn_config": {"num_heads": 4, "head_dim": 128},
    }


def write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, at = {}, [], 0
    names = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16", torch.int16: "I16",
             torch.int32: "I32"}
    for name, t in tensors.items():
        t = t.contiguous()
        raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for blob in blobs:
            f.write(blob)


def write_checkpoint(folder: Path, seed: int = 0, layers: int = LAYERS, moe: bool = False) -> Path:
    """The tiny checkpoint in ``folder`` (config.json, model.safetensors); returns the folder. ``moe``: the last
    layer's MLP is a MoE of EXL3 routed experts (random trellis tiles, as the real checkpoint stores them) and a BF16
    shared expert."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    P = "model.language_model."

    def bf16(*shape, scale=0.05, offset=0.0):
        return (offset + scale * torch.randn(*shape, generator=g)).to(torch.bfloat16)

    def f32(*shape, scale=0.1, offset=0.0):
        return (offset + scale * torch.randn(*shape, generator=g)).float()

    t = {P + "embed_tokens.weight": bf16(V, D, scale=0.5), P + "norm.weight": bf16(D, scale=0.02, offset=1.0),
         "lm_head.weight": bf16(V, D)}
    for i in range(layers):
        p = P + f"layers.{i}."
        t[p + "input_layernorm.weight"] = bf16(D, scale=0.02, offset=1.0)
        t[p + "post_attention_layernorm.weight"] = bf16(D, scale=0.02, offset=1.0)
        for site in ("attn", "ffn"):
            t[p + f"hc_{site}_fn"] = bf16(S * S + 2 * S, S * D, scale=0.02)
            t[p + f"hc_{site}_base"] = f32(S * S + 2 * S)
            t[p + f"hc_{site}_scale"] = f32(3, scale=0.05, offset=0.5)
        a = p + "self_attn."
        t[a + "q_a_proj.weight"] = bf16(QL, D)
        t[a + "kv_a_proj_with_mqa.weight"] = bf16(KL, D)
        t[a + "q_a_layernorm.weight"] = bf16(QL, scale=0.02, offset=1.0)
        t[a + "kv_a_layernorm.weight"] = bf16(KL, scale=0.02, offset=1.0)
        t[a + "q_b_proj.weight"] = bf16(HEADS * 256, QL, scale=0.08)
        t[a + "kv_b_proj.weight"] = bf16(HEADS * 512, KL, scale=0.08)
        t[a + "o_proj.weight"] = bf16(D, HEADS * 256, scale=0.03)
        t[a + "indexer.wk.weight"] = bf16(ID, D, scale=0.08)
        t[a + "indexer.weights_proj.weight"] = bf16(IH, D, scale=0.08)
        t[a + "indexer.wq_b.weight"] = bf16(IH * ID, QL, scale=0.08)
        t[a + "indexer.k_norm.weight"] = bf16(ID, scale=0.02, offset=1.0)
        t[a + "indexer.k_norm.bias"] = bf16(ID, scale=0.02)
        t[a + "indexer.index_kpool_compress_gate"] = bf16(ID, D, scale=0.08)
        t[a + "indexer.index_kpool_compress_ape"] = bf16(4, ID, scale=0.3)
        m = p + "mlp."
        if moe and i == layers - 1:
            t[m + "gate.weight"] = bf16(EXPERTS, D)
            t[m + "gate.e_score_correction_bias"] = f32(EXPERTS, scale=0.01)
            for e in range(EXPERTS):
                for proj, n, k in (("gate_proj", MOE, D), ("up_proj", MOE, D), ("down_proj", D, MOE)):
                    x = m + f"experts.{e}.{proj}."
                    t[x + "trellis"] = torch.randint(-2**15, 2**15, (k // 16, n // 16, 64), generator=g,
                                                     dtype=torch.int32).to(torch.int16)
                    t[x + "suh"] = (0.03 * torch.randn(k, generator=g)).to(torch.float16)
                    t[x + "svh"] = (0.03 * torch.randn(n, generator=g)).to(torch.float16)
                    t[x + "mcg"] = torch.tensor([0xCBAC1FED - 2**32], dtype=torch.int32)
            sh = m + "shared_experts."
            t[sh + "gate_proj.weight"] = bf16(MOE, D)
            t[sh + "up_proj.weight"] = bf16(MOE, D)
            t[sh + "down_proj.weight"] = bf16(D, MOE, scale=0.03)
            continue
        t[m + "gate_proj.weight"] = bf16(DENSE, D)
        t[m + "up_proj.weight"] = bf16(DENSE, D)
        t[m + "down_proj.weight"] = bf16(D, DENSE, scale=0.03)
    write_safetensors(folder / "model.safetensors", t)
    config = {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
              "text_config": text_config(layers, moe),
              "quantization_config": {"quant_method": "exl3", "bits": 4.0, "scope": "routed_experts_only"}}
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def cpu_cuda_stubs() -> None:
    """What the loader and buffers ask of torch.cuda on a machine without one: a current stream whose events are
    already done, an empty cache to free."""

    done = types.SimpleNamespace(synchronize=lambda: None, query=lambda: True)
    torch.cuda.current_stream = lambda *a, **k: types.SimpleNamespace(record_event=lambda *a, **k: done,
                                                                       wait_stream=lambda *a, **k: None,
                                                                       wait_event=lambda *a, **k: None)
    torch.cuda.empty_cache = lambda: None


def env(kv: str = "fp8") -> None:
    """The engine settings the tests run with: BF16 dense weights and kv_b (the Q4 prompt matmul is CUDA-only), the
    latent cache in ``kv``, the interpreter."""

    os.environ.update(TRITON_INTERPRET="1", TF_GLM_DENSE="bf16", TF_GLM_KVB="bf16", TF_GLM_KV=kv, TF_GLM_LATENT="1",
                      TF_GLM_L2PF="0")


def load_rank(folder: Path, rank: int, world: int):
    from tensorfold.families.glm5_next.cuda.weights import load

    w = load(folder, rank=rank, world=world, device="cpu", mtp=False)
    w.meta["long_context"] = True
    return w


class GlooComm:
    """The communicator calls the row split and the gathers make (``tensorfold.cuda.comm.NCCL``'s), over gloo; it
    counts the bytes each rank sends. CUDA tensors go through host copies on the current stream (each waits for that
    stream's work before it, as NCCL's would; the host blocks meanwhile), so a GPU test's side stream really runs its
    exchanges beside the main stream's queued kernels."""

    def __init__(self, world: int) -> None:
        import torch.distributed as dist

        self.dist, self.world, self.world_size = dist, world, world
        self.sent = 0

    def barrier(self) -> None:
        self.dist.barrier()

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.sent += send.numel() * send.element_size() * (self.world - 1)
        host = send.contiguous().reshape(-1).cpu()
        parts = [torch.empty_like(host) for _ in range(self.world)]
        self.dist.all_gather(parts, host)
        recv.view(self.world, -1).copy_(torch.stack(parts))

    def exchange(self, sends, recvs, peer: int) -> None:
        self.exchange_all({peer: sends}, {peer: recvs})

    def exchange_all(self, sends: dict, recvs: dict) -> None:
        work, back = [], []
        for p in sorted(sends):
            for t in sends[p]:
                self.sent += t.numel() * t.element_size()
                work.append(self.dist.isend(t.contiguous().cpu(), p))
            for t in recvs[p]:
                tmp = torch.empty(t.shape, dtype=t.dtype)
                work.append(self.dist.irecv(tmp, p))
                back.append((t, tmp))
        for item in work:
            item.wait()
        for t, tmp in back:
            t.copy_(tmp)
