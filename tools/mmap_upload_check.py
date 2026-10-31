#!/usr/bin/env python3
"""Check for patch 0098 (glm-mmap-uploads): the DFlash2 drafter's and the GLM vision tower's loaders copy each tensor
out of the safetensors mmap before it goes to the GPU. Run inside the image scripts/prepare.sh built (no GPU needed):

    docker run --rm --entrypoint python -v "$PWD/tools/mmap_upload_check.py:/c.py" tensorfold-glm53:v0.6.0 /c.py

Each loader runs here on the CPU over a small checkpoint this script writes, with ``Tensor.to`` watched: no tensor a
loader hands to a device copy may lie in the safetensors file's mapping (its address inside a mapping of that file in
/proc/self/maps, read at the moment of the copy). Why it matters: on kindling spark-os's 64 KiB-page kernel a pageable
copy to the GPU straight from a file mmap hangs in the driver (cuMemcpyHtoDAsync) once the process holds GPU memory,
and both loaders run after the main weights; the same copy from an ordinary host buffer is instant. The loaders as
TensorFold ships them fail this check. Exit code 1 when a check fails.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

failures = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def file_ranges(path):
    """[start, end) of every mapping of ``path`` in this process right now."""

    real = os.path.realpath(path)
    out = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].strip() == real:
                lo, hi = parts[0].split("-")
                out.append((int(lo, 16), int(hi, 16)))
    return out


class Watch:
    """While active, every ``Tensor.to`` call records whether its source tensor's bytes lie in a mapping of ``path``."""

    def __init__(self, path):
        self.path = path
        self.calls = 0
        self.from_mmap = 0

    def __enter__(self):
        self.orig = torch.Tensor.to
        watch = self

        def to(t, *args, **kwargs):
            lo = t.data_ptr()
            hi = lo + t.numel() * t.element_size()
            watch.calls += 1
            if any(a < hi and lo < b for a, b in file_ranges(watch.path)):
                watch.from_mmap += 1
            return watch.orig(t, *args, **kwargs)

        torch.Tensor.to = to
        return self

    def __exit__(self, *exc):
        torch.Tensor.to = self.orig
        return False


def bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def vision_tower(root):
    from tensorfold.vision.glm import GlmVisionTower

    d = Path(root) / "vision"
    d.mkdir()
    vision = {"hidden_size": 64, "num_heads": 4, "depth": 1, "rms_norm_eps": 1e-6, "patch_size": 14,
              "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 64}
    (d / "config.json").write_text(json.dumps({"vision_config": vision}))
    names = {"model.visual.a.weight": bf16(256, 64), "model.visual.b.weight": bf16(64),
             "model.visual.c.weight": torch.randn(64, 64)}           # float32: the load converts it to bf16
    save_file(names, str(d / "model-00001-of-00001.safetensors"))
    index = {"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}}
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    with Watch(d / "model-00001-of-00001.safetensors") as w:
        tower = GlmVisionTower(d, "cpu")
    check("vision tower: the loader copies its tensors to the device", w.calls >= len(names), f"{w.calls} copies")
    check("vision tower: no device copy reads the safetensors mmap", w.from_mmap == 0,
          f"{w.from_mmap} of {w.calls} copies from the mapped file")
    check("vision tower: the weights arrive as written",
          all(torch.equal(tower.t[n[len("model.visual."):]], t.to(torch.bfloat16)) for n, t in names.items()))


class Stop(Exception):
    pass


def drafter(root):
    from tensorfold.families.glm5_next.cuda import dflash2

    d = Path(root) / "dflash2"
    d.mkdir()
    cfg = {"hidden_size": 64, "head_dim": 16, "num_attention_heads": 4, "num_key_value_heads": 2,
           "intermediate_size": 128, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000.0},
           "num_hidden_layers": 1, "sliding_window": 17, "is_causal": True,
           "dflash_config": {"mask_token_id": 0, "conv_group_size": 1, "conv_kernel_size": 2, "block_size": 8,
                             "selector_top_k": 4, "target_layer_ids": [1]}}
    (d / "config.json").write_text(json.dumps(cfg))
    fc = bf16(64, 128)
    save_file({"fc.weight": fc, "hidden_norm.weight": bf16(64)}, str(d / "model.safetensors"))
    seen = []

    def quantize4(t):                     # the first upload's result; nothing past it is needed here
        seen.append(t)
        raise Stop

    orig = dflash2._quantize4
    dflash2._quantize4 = quantize4
    try:
        with Watch(d / "model.safetensors") as w:
            try:
                dflash2.Drafter(d, SimpleNamespace(device="cpu", rank=0, world=1))
            except Stop:
                pass
    finally:
        dflash2._quantize4 = orig
    check("drafter: the loader reached its first upload", len(seen) == 1 and w.calls >= 1, f"{w.calls} copies")
    check("drafter: no device copy reads the safetensors mmap", w.from_mmap == 0,
          f"{w.from_mmap} of {w.calls} copies from the mapped file")
    check("drafter: the weight arrives as written", len(seen) == 1 and torch.equal(seen[0], fc))


def main():
    with tempfile.TemporaryDirectory() as root:
        vision_tower(root)
        drafter(root)
    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
