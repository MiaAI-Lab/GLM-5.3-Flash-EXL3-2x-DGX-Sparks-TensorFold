"""PRMT operand identity and Q4 reduction identity; synthetic inputs, no model weights."""
import os
from pathlib import Path
import subprocess
import sys

import torch


def test_prmt_operands():
    from tensorfold.families.glm5_next.cuda import exl3_mm
    from torch.utils.cpp_extension import load_inline
    source = Path(exl3_mm.__file__).with_name("exl3.cu").read_text()
    # Compile the actual installed helpers, not a second implementation.
    helpers = source.split("__device__ __forceinline__ void mma16816(")[0]
    kernel = r'''
__global__ void compare_unpack(const uint32_t* words, uint32_t* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t a[2], b[2], c[2], d[2];
    decode_tile(words[i], threadIdx.x & 31, a, b);
    decode_tile_p(words[i], threadIdx.x & 31, c, d);
    result[i] = (a[0] == c[0] && a[1] == c[1] && b[0] == d[0] && b[1] == d[1]);
}
} // namespace
at::Tensor compare_unpack_cuda(at::Tensor words) {
    auto result = at::empty_like(words);
    compare_unpack<<<words.numel()/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        (const uint32_t*)words.data_ptr(), (uint32_t*)result.data_ptr(), words.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
'''
    ext = load_inline(name="tf_tp4_prmt_operand_check_v1",
        cpp_sources="at::Tensor compare_unpack_cuda(at::Tensor);", cuda_sources=helpers+kernel,
        functions=["compare_unpack_cuda"], extra_cuda_cflags=["-O3"], verbose=False)
    torch.manual_seed(73)
    words = torch.randint(-2**31, 2**31-1, (65536,), device="cuda", dtype=torch.int32)
    for pattern in (words, torch.zeros_like(words), torch.full_like(words, -1),
                    torch.arange(65536, device="cuda", dtype=torch.int32)):
        assert bool(ext.compare_unpack_cuda(pattern).all())


QMM_CHECK = r'''
import sys, torch
from tensorfold.cuda.kernels import qmm
torch.manual_seed(91)
result = {}
for n,k in ((1024,4096),(7168,4096),(8192,4096)):
    words=torch.randint(-2**31,2**31-1,(n,k//8),device='cuda',dtype=torch.int32)
    scales=(torch.rand(n,k//64,device='cuda')*.02+.001).bfloat16()
    biases=(torch.randn(n,k//64,device='cuda')*.05).bfloat16()
    q=qmm.pack(words,scales,biases,64)
    for rows in (1,16,32,64,65):
        x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)
        for f32 in (False,True):
            y=qmm.matmul(x,q,sk=4,f32=f32)
            result[n,rows,f32,'eager']=y.cpu()
            sums=qmm.group_sums(x)
            part=torch.empty(4,rows,n,device='cuda',dtype=torch.float32)
            out=torch.empty_like(y)
            def invoke():qmm.matmul(x,q,xs=sums,sk=4,f32=f32,out=out,part=part)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):invoke()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):invoke()
            x.normal_();sums.copy_(qmm.group_sums(x))
            graph.replay();torch.cuda.synchronize()
            result[n,rows,f32,'graph']=out.cpu()
torch.save(result,sys.argv[1])
'''


def test_qmm_reductions_and_graphs(tmp_path):
    # The C++ choice is process-lifetime state; use separate processes so the
    # first path cannot silently remain active in the second half of the test.
    outputs = []
    for clusters in ("1", "0"):
        file = tmp_path / (clusters + ".pt")
        env = dict(os.environ, TF_GLM_QMM_CLUSTERS=clusters)
        subprocess.run([sys.executable, "-c", QMM_CHECK, str(file)], env=env, check=True, timeout=600)
        outputs.append(torch.load(file, weights_only=True))
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        assert torch.equal(outputs[0][key].view(torch.uint8), outputs[1][key].view(torch.uint8)), key
