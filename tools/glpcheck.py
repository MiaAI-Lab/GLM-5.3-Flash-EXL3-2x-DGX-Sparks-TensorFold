#!/usr/bin/env python3
"""Validate a GLP (GGUF Layer Projection) control vector offline: the same fail-closed gates the engine applies at
boot (patch 0054, tensorfold/families/glm5_next/cuda/glp.py -- keep the two in sync), no GPU and no server needed.

Usage: tools/glpcheck.py <control.gguf> [--layers 45] [--width 16384]
Exit code 0 and a summary when the file would be served; 1 with the refusal reason otherwise.

The format: spec/GLP.md in https://github.com/msuiche/weightless (https://weightless.msuiche.com).
"""
import argparse
import hashlib
import math
import re
import struct
import sys

import numpy as np

HOOK = "residual_stream_post_layer"

_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

_DIRECTION_RE = re.compile(r"direction\.(\d+)(?:\.(\d+))?")


def _int_meta(meta: dict, key: str, default: int) -> int:
    v = meta.get(key)
    if v is None or v == "":
        return default
    return int(float(str(v)))


def read_gguf(path: str) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: bad magic, not a GGUF file")
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
    if ver != 3:
        raise ValueError(f"{path}: GGUF version {ver}, expected 3")
    off = 24

    def rd_string(o):
        (n,) = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8"), o + n

    meta = {}
    for _ in range(n_kv):
        key, off = rd_string(off)
        (vtype,) = struct.unpack_from("<I", data, off)
        off += 4
        if vtype == 8:
            val, off = rd_string(off)
        elif vtype == 9:
            etype, cnt = struct.unpack_from("<IQ", data, off)
            off += 12
            if etype == 8:
                val = []
                for _ in range(cnt):
                    s, off = rd_string(off)
                    val.append(s)
            else:
                val = struct.unpack_from(f"<{cnt}{_FM[etype][1]}", data, off)
                off += _SZ[etype] * cnt
        else:
            (val,) = struct.unpack_from(_FM[vtype], data, off)
            off += _SZ[vtype]
        meta[key] = val

    infos = []
    for _ in range(n_tensors):
        name, off = rd_string(off)
        (nd,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", data, off)
        off += 8 * nd
        (tt,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", data, off)
        off += 8
        infos.append((name, dims, tt, toff))

    align = meta.get("general.alignment", 32)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, tt, toff in infos:
        if tt != 0:
            raise ValueError(f"{path}: {name} is not F32 (ggml type {tt})")
        n = 1
        for d in dims:
            n *= d
        tensors[name] = np.frombuffer(data, dtype="<f4", count=n, offset=base + toff).copy()
    return meta, tensors


def check(path: str, layers: int, width: int) -> dict:
    meta, tensors = read_gguf(path)

    mode = meta.get("glp.mode")
    if mode is None:
        raise ValueError(f"{path}: no glp.mode. Refusing to guess: an additive control vector and a projective one "
                         "are different operations.")
    if mode != "project":
        raise ValueError(f"{path}: glp.mode={mode!r}, but this engine only implements projective ablation "
                         "(x -= alpha*(x.d)d). Refusing to apply.")
    file_hook = meta.get("glp.hook_point")
    if file_hook != HOOK:
        raise ValueError(f"{path}: glp.hook_point={file_hook!r} does not match this engine's hook ({HOOK}). "
                         "Refusing to apply at the wrong site.")
    version = _int_meta(meta, "glp.spec_version", 1)
    if version not in (1, 2):
        raise ValueError(f"{path}: glp.spec_version={meta.get('glp.spec_version')!r} is not implemented by this "
                         "reader (1 and 2 are). Refusing.")
    rank_declared = _int_meta(meta, "glp.rank", 1)
    if version == 1 and (rank_declared != 1 or "glp.dir_scales" in meta or "glp.layer_scales" in meta):
        raise ValueError(f"{path}: glp.rank={rank_declared} / alpha-scaling keys require glp.spec_version 2 "
                         "(this file declares 1). Refusing.")
    scales = sorted(set(meta) & {"glp.dir_scales", "glp.layer_scales"})
    if scales:
        raise ValueError(f"{path}: carries {', '.join(scales)}, which this engine does not implement. Refusing to "
                         "ignore alpha multipliers.")
    if rank_declared != 1:
        raise ValueError(f"{path}: glp.rank={rank_declared} (subspace) -- this engine implements rank 1 only. "
                         "Refusing to serve a partially applied subspace.")
    alpha = float(meta.get("glp.alpha_default", 1.0))
    if not math.isfinite(alpha):
        raise ValueError(f"{path}: glp.alpha_default must be finite, not {meta.get('glp.alpha_default')!r}")

    grouped: dict[int, dict[int, np.ndarray]] = {}
    for name, arr in tensors.items():
        m = _DIRECTION_RE.fullmatch(name)
        if m is None:
            if name.startswith("direction."):
                raise ValueError(f"{path}: malformed tensor name {name!r} -- 'direction.' must be followed by an "
                                 "integer layer id")
            continue
        idx, j = int(m.group(1)), int(m.group(2) or 0)
        if idx < 1:
            raise ValueError(f"{path}: {name} is invalid; layer 0 cannot be expressed in this container")
        if j != 0:
            raise ValueError(f"{path}: {name} names a rank-{j + 1} direction; this engine implements rank 1 only. "
                             "Refusing.")
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all() or not np.any(arr):
            raise ValueError(f"{path}: {name} must be finite and nonzero")
        grouped.setdefault(idx, {})[j] = arr
    if not grouped:
        raise ValueError(f"{path}: no direction.<N> tensors found")

    declared = meta.get("glp.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            raise ValueError(f"{path}: glp.layer_ids_zero_based is not a comma list of integers: {declared!r}") \
                from None
        if want and want != sorted(grouped):
            raise ValueError(f"{path}: glp.layer_ids_zero_based declares layers {want[0]}..{want[-1]} "
                             f"({len(want)} entries) but the direction tensors resolve to {sorted(grouped)[0]}.."
                             f"{sorted(grouped)[-1]} ({len(grouped)} entries). The tensor names are what get "
                             "applied, so this file would steer the wrong layers. Re-export it.")

    for idx, js in grouped.items():
        if not 0 <= idx < layers:
            raise ValueError(f"{path}: direction layer {idx} out of range for this model ({layers} layers)")
        vec = js[0]
        if vec.size != width:
            raise ValueError(f"{path}: direction layer {idx} is {vec.size} wide; this model's widened stream is "
                             f"{width} (streams * hidden). Refusing to steer a different geometry.")
        norm = float(np.linalg.norm(vec.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"{path}: direction layer {idx} must be finite and nonzero")

    return {"meta": meta, "layers": sorted(grouped), "alpha": alpha, "width": width}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--layers", type=int, default=45, help="decoder layers of the model (GLM-5.3-Flash: 45)")
    ap.add_argument("--width", type=int, default=16384, help="the widened stream width (streams * hidden: 4*4096)")
    args = ap.parse_args()
    try:
        got = check(args.file, args.layers, args.width)
    except (ValueError, OSError, struct.error) as exc:
        print(f"glpcheck: {exc}", file=sys.stderr)
        sys.exit(1)
    meta = got["meta"]
    with open(args.file, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"{args.file}: servable")
    print(f"  mode=project hook={HOOK} spec_version={meta.get('glp.spec_version', 1)} rank={meta.get('glp.rank', 1)}")
    print(f"  layers {got['layers'][0]}..{got['layers'][-1]} ({len(got['layers'])} steered), width {got['width']}, "
          f"alpha_default {got['alpha']}")
    base = meta.get("general.base_model.0.name") or meta.get("glp.base_model") or "?"
    rev = str(meta.get("general.base_model.0.version") or meta.get("glp.base_revision") or "?")[:12]
    print(f"  base model: {base} @ {rev}; content sha256 {sha[:16]}...")


if __name__ == "__main__":
    main()
