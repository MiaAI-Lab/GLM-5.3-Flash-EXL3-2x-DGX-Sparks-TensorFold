"""CPU tests of the GLM-5.3-Flash patches: Triton kernels run in Triton's interpreter (TRITON_INTERPRET=1) on CPU
tensors, so they need no GPU; several ranks run as processes over gloo. scripts/test-cpu.sh runs them in a
memory-capped container of the recipe's image; TF_SRC points at a patched TensorFold source tree to test instead of
the image's installed one."""

import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
if os.environ.get("TF_SRC"):
    sys.path.insert(0, os.environ["TF_SRC"])
sys.path.insert(0, os.path.dirname(__file__))


def _patch_interpreter_dot() -> None:
    """Triton's interpreter multiplies bf16 operands of tl.dot as their raw uint16 bits (numpy has no bf16), which
    gives garbage; widen them to fp32 first (exact), as the tensor cores multiply bf16 values exactly into fp32."""
    try:
        import numpy as np
        import triton.language as tl
        from triton.runtime import interpreter as itp
    except ImportError:
        return
    builder = getattr(itp, "InterpreterBuilder", None)
    if builder is None or getattr(builder, "_glm53_dot", False):
        return
    original = builder.create_dot
    import torch

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        def wide(x):
            sc = getattr(x.dtype, "scalar", x.dtype)
            if sc == tl.bfloat16:
                bits = x.data.astype(np.uint32) << 16
                return itp.TensorHandle(bits.view(np.float32), tl.float32)
            if sc == tl.float8e4nv:
                f = torch.from_numpy(np.ascontiguousarray(x.data).view(np.uint8).copy()).view(torch.float8_e4m3fn)
                return itp.TensorHandle(f.float().numpy(), tl.float32)
            return x
        return original(self, wide(a), wide(b), d, input_precision, max_num_imprecise_acc)

    builder.create_dot = create_dot

    cast = builder.cast_impl

    def cast_impl(self, src, dst_type):
        """fp32 -> bf16 rounds to nearest even, as the GPU's conversion does (the interpreter truncates)."""
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            bits = np.ascontiguousarray(src.data, dtype=np.float32).view(np.uint32).astype(np.uint64)
            nan = np.isnan(np.ascontiguousarray(src.data, dtype=np.float32))
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            rounded = np.where(nan, np.uint16(0x7FC0), rounded)
            return itp.TensorHandle(rounded.reshape(np.shape(src.data)), tl.bfloat16)
        return cast(self, src, dst_type)

    builder.cast_impl = cast_impl
    builder.create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)

    # fp8 e4m3 (the FP8 latent cache): the interpreter emulates conversions bit by bit, not as the GPU rounds; torch's
    # float8_e4m3fn conversions round like the hardware (to nearest even, saturating at 448 as the kernels clamp)
    import torch

    def to_f32(handle):
        sc = getattr(handle.dtype, "scalar", handle.dtype)
        data = np.ascontiguousarray(handle.data)
        if sc == tl.bfloat16:
            return (data.astype(np.uint32) << 16).view(np.float32)
        if sc == tl.float8e4nv:
            return torch.from_numpy(data.view(np.uint8).copy()).view(torch.float8_e4m3fn).float().numpy()
        return data.astype(np.float32)

    fp_to_fp = builder.create_fp_to_fp

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        s8, d8 = src.dtype.scalar == tl.float8e4nv, dst_type.scalar == tl.float8e4nv
        if d8:
            x = torch.from_numpy(to_f32(src).copy()).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            return itp.TensorHandle(x.reshape(np.shape(src.data)), tl.float8e4nv)
        if s8:
            f = to_f32(src).reshape(np.shape(src.data))
            return self.cast_impl(itp.TensorHandle(f, tl.float32), dst_type) \
                if dst_type.scalar != tl.float32 else itp.TensorHandle(f, tl.float32)
        return fp_to_fp(self, src, dst_type, rounding_mode)

    builder.create_fp_to_fp = create_fp_to_fp
    builder.create_fp_ext = lambda self, src, dst_type: (
        create_fp_to_fp(self, src, dst_type, None) if src.dtype.scalar == tl.float8e4nv else cast(self, src, dst_type))
    builder._glm53_dot = True


if os.environ.get("TRITON_INTERPRET") == "1":
    _patch_interpreter_dot()
