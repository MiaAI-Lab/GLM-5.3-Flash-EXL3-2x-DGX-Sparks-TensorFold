#!/usr/bin/env python3
"""tools/qmm_noclusters_check.py [--time]: patch 0103 - decode windows' 4-bit matmuls (tensorfold/cuda/kernels/qmm)
reduce their K slices through the partials buffer and reduce_kernel instead of a thread-block cluster when the window
has at most 64 rows and the weight holds fewer than 28 Mi values. Run in the image built with patch 0103, on one GPU
(sm_90 or newer, where the clusters exist):
docker run --rm --gpus all --entrypoint python -v "$PWD/tools/qmm_noclusters_check.py:/c.py" <image> /c.py

NEW = the installed extension (qmm._ext()); OLD = the same sources with qmm_clusters' rule put back to the one before
the patch (clusters for every 2-8 reduced K slices on sm_90+), built here. At GLM-5.3-Flash's rank-0 decode shapes at
TP=4, TP=3 and TP=2 (every dense projection, the shared expert, DFlash2's matrices) and 1-128 rows:
1. bits: NEW's output equals OLD's bit for bit through both entry points (``qmm`` with the row bucket, ``qmm_cfg`` with
   every decode tile config 0-15 for rows <= 16), outputs pre-filled with NaN;
2. mechanism: NEW launches reduce_kernel (the buffer path) exactly when sk > 1, rows <= 64 and n * k < 28 Mi
   (torch.profiler's kernel names), OLD never does; with TF_GLM_QMM_CLUSTERS=1 (read once a process: a child
   process) NEW runs the cluster where it would take the buffer;
3. graphs: matmuls on three side streams captured in one CUDA graph and replayed with new inputs equal OLD's eager
   outputs (the buffer path's partials are allocated per call, stream-ordered);
4. the extension loads under a new name (tensorfold_qmm_v5), so a kernel folder holding v4's build does not serve it.
One PASS / FAIL line a check; exits 1 on any FAIL (on an image without the patch: at the first check); an error that
stops the check (a CUDA fault in the extension) is a FAIL too.

--time: CUDA-graph us a call, OLD -> NEW, cold L2 (enough weight copies to pass the L2), rows 1, 4, 8, 32, 64, at
TP=4, 3 and 2."""
import hashlib
import os
import subprocess
import re
import shutil
import sys
import tempfile

import torch

from tensorfold.cuda.build import load
from tensorfold.cuda.kernels import qmm as installed
from tensorfold.families.glm5_next.cuda import tp

SRC = ("qmm.cpp", "qmm.cu", "qmm_prefill.cu", "qmm_prefill8.cu")
OLD_RULE = ("bool qmm_clusters(int SK, bool reduce, long long, long long, long long) {\n"
            "    return SK > 1 && SK <= 8 && reduce && at::cuda::getCurrentDeviceProperties()->major >= 9;\n}\n")
ROWS = (1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128)
fails = 0


def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    fails += not ok


def shapes(world):
    """(label, n, k, fp32 out) of rank 0's decode matmuls at ``world`` ranks (DFlash2 by tp.draft_parts); "edge": two
    shapes either side of the 28 Mi rule (no GLM-5.3-Flash matrix sits there)."""
    if world == "edge":
        return [("28 Mi values exactly", 7168, 4096, False), ("just under 28 Mi", 7040, 4096, False)]
    h, lh = tp.split_sizes(64, world, 1)[0], tp.split_sizes(64, world, 1)[0]
    dn, sh = tp.split_sizes(12288, world, 128)[0], tp.split_sizes(2048, world, 128)[0]
    vocab = tp.split_sizes(154880, world, 64)[0]
    dq, dkv, dmlp = (x[0] for x in tp.draft_parts(32, 8, 12288, world))
    return [("kda proj", 3 * lh * 128 + 256 + lh, 4096, False), ("kda o", 4096, lh * 128, True),
            ("dsa o", 4096, h * 256, True), ("shared gate/up", 2 * sh, 4096, False), ("shared down", 4096, sh, True),
            ("kda f_b/g_b", lh * 128, 128, False), ("dsa q_a|kv_a", 2048, 4096, False),
            ("dsa q_b", h * 256, 1536, False), ("idx wq_b", 4096, 1536, False), ("idx wk|w", 160, 4096, False),
            ("dense gate/up", 2 * dn, 4096, False), ("dense down", 4096, dn, True),
            ("draft taps fc", 4096, 20480, False), ("draft ctx kv", 2 * dkv * 128, 4096, False),
            ("draft qkv", (dq + 2 * dkv) * 128, 4096, False), ("draft o_proj", 4096, dq * 128, True),
            ("draft gate/up", 2 * dmlp, 4096, False), ("draft down", 4096, dmlp, True),
            ("draft head", vocab, 4096, False)]


def old_build(kdir):
    """The installed sources with qmm_clusters' body put back to the rule before the patch, built under its own name."""
    work = tempfile.mkdtemp(prefix="qmm-old-")
    for f in os.listdir(kdir):
        if f.endswith((".cu", ".cpp", ".cuh", ".h")):
            shutil.copy(os.path.join(kdir, f), work)
    cu = os.path.join(work, "qmm.cu")
    src = open(cu).read()
    new, n = re.subn(r"bool qmm_clusters\(int SK, bool reduce, long long M, long long N, long long K\) \{\n.*?\n\}\n",
                     lambda _: OLD_RULE, src, flags=re.S)
    if n != 1:
        return None, work
    open(cu, "w").write(new)
    h = hashlib.sha256(b"".join(open(os.path.join(work, f), "rb").read() for f in sorted(os.listdir(work))))
    return load(name=f"qmm_check_old_{h.hexdigest()[:12]}", sources=[os.path.join(work, f) for f in SRC],
                extra_cuda_cflags=["-O3"], verbose=False), work


def main():
    kdir = os.path.dirname(installed.__file__)
    src = open(os.path.join(kdir, "qmm.cu")).read()
    check("long long M, long long N, long long K)" in src and "CLUSTER_VALUES" in src,
          "patch 0103 applied: qmm_clusters takes the window's shape")
    if fails:
        print("1 FAIL", flush=True)
        sys.exit(1)
    py = open(os.path.join(kdir, "qmm.py")).read()
    check('name="tensorfold_qmm_v5"' in py and 'name="tensorfold_qmm_v4"' not in py,
          "the extension loads as tensorfold_qmm_v5 (not v4's build from a shared kernel folder)")
    timing = "--time" in sys.argv
    dev = torch.device("cuda")
    check(torch.cuda.get_device_capability()[0] >= 9, "the GPU has thread-block clusters (sm_90 or newer)")
    eo, work = old_build(kdir)
    check(eo is not None, "OLD: qmm_clusters' body found and put back to the rule before the patch")
    if eo is None:
        print(f"{fails} FAIL", flush=True)
        sys.exit(1)
    en = installed._ext()
    g = torch.Generator(device=dev).manual_seed(86)

    def weights(n, k):
        words = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, k // 8), dtype=torch.int32, device=dev, generator=g)
        sc = (torch.rand((n, k // 64), device=dev, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
        bi = ((torch.rand((n, k // 64), device=dev, generator=g) - 0.5) * 0.02).to(torch.bfloat16)
        return installed.pack(words, sc, bi, 64)

    def run(ext, x, xs, q, sk, f32, cfg=None):
        out = torch.full((x.shape[0], q.n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        if cfg is None:
            ext.qmm(x, xs, q.weight, q.scales, q.biases, out, None, q.n, sk, 64, installed.bucket(x.shape[0]), f32,
                    True)
        else:
            ext.qmm_cfg(x, xs, q.weight, q.scales, q.biases, out, None, q.n, sk, f32, cfg)
        return out

    def bits(t):
        return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)

    def reduces(ext, x, xs, q, sk, f32, cfg=None):
        """Whether the call launched reduce_kernel (the buffer path), from the profiler's kernel names."""
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            run(ext, x, xs, q, sk, f32, cfg)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        return any("qmm_kernel" in nm for nm in names), any("reduce_kernel" in nm for nm in names)

    if "--clusters-child" in sys.argv:              # the opt-out check's child: TF_GLM_QMM_CLUSTERS=1 in its env
        n, k = 2 * tp.split_sizes(2048, 3, 128)[0], 4096
        x = (torch.randn((1, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        _, red = reduces(en, x, installed.group_sums(x, 64), weights(n, k), installed.split_k(n, k, 64), False)
        print(f"CHILD {'buffer' if red else 'cluster'}", flush=True)
        sys.exit(0)

    seen = set()
    for world in (4, 3, 2, "edge"):
        for label, n, k, f32 in shapes(world):
            if (n, k, f32) in seen:
                continue
            seen.add((n, k, f32))
            label = (f"TP={world} " if world != "edge" else "") + f"{label} ({n} x {k})"
            q = weights(n, k)
            sk = installed.split_k(n, k, 64)
            differ, wrong, calls = [], [], 0
            for m in ROWS:
                x = (torch.randn((m, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                xs = installed.group_sums(x, 64)
                ref = run(eo, x, xs, q, sk, f32)
                if torch.isnan(ref.float()).any():
                    differ.append(f"m={m}: OLD left a slot unwritten")
                for cfg in [None] + (list(range(16)) if m <= 16 else []):
                    calls += 1
                    try:
                        if not torch.equal(bits(run(en, x, xs, q, sk, f32, cfg)), bits(ref)):
                            differ.append(f"m={m} cfg {cfg}")
                    except RuntimeError as exc:          # the extension refusing a call is a failure, reported
                        differ.append(f"m={m} cfg {cfg}: NEW raised {str(exc).splitlines()[0][:100]}")
                want = sk > 1 and m <= 64 and n * k < (28 << 20)
                for cfg in (None, 0) if m <= 16 else (None,):
                    try:
                        ran_n, red_n = reduces(en, x, xs, q, sk, f32, cfg)
                        ran_o, red_o = reduces(eo, x, xs, q, sk, f32, cfg)
                    except RuntimeError as exc:
                        wrong.append(f"m={m} cfg {cfg}: raised {str(exc).splitlines()[0][:100]}")
                        continue
                    if not ran_n or not ran_o or red_n != want or (sk > 1 and red_o):
                        wrong.append(f"m={m} cfg {cfg}: NEW {'buffer' if red_n else 'cluster'}, OLD "
                                     f"{'buffer' if red_o else 'cluster'}, want {'buffer' if want else 'cluster'}")
            check(not differ, f"{label}: {calls} NEW calls == OLD bit for bit at rows {ROWS[0]}-{ROWS[-1]} "
                              f"({differ[:3] or 'all equal'})")
            check(not wrong, f"{label}: sk {sk}, the buffer path exactly at rows <= 64 below 28 Mi values "
                             f"({wrong[:3] or 'as wanted'})")
            del q
    child = subprocess.run([sys.executable, os.path.abspath(__file__), "--clusters-child"], capture_output=True,
                           text=True, env={**os.environ, "TF_GLM_QMM_CLUSTERS": "1"})
    check("CHILD cluster" in child.stdout,
          "TF_GLM_QMM_CLUSTERS=1: a 1-row TP=3 shared gate/up runs its cluster, the rule before the patch "
          f"({(child.stdout.strip().splitlines() or [child.stderr.strip()[-200:]])[-1]})")

    picks = [s for s in shapes(3) if s[0] in ("kda o", "shared gate/up", "dsa q_a|kv_a", "draft qkv")]
    qs = [weights(n, k) for _, n, k, _ in picks]
    m = 2
    xin = [torch.zeros((m, k), dtype=torch.bfloat16, device=dev) for _, _, k, _ in picks]
    xsin = [torch.zeros((m, k // 64), dtype=torch.float32, device=dev) for _, _, k, _ in picks]
    outs = [torch.zeros((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev) for _, n, _, f32 in picks]
    streams = [torch.cuda.Stream() for _ in range(3)]

    def body():
        main_st = torch.cuda.current_stream()
        for i, (_, n, k, f32) in enumerate(picks):
            st = streams[i % 3]
            st.wait_stream(main_st)
            with torch.cuda.stream(st):
                for _ in range(3):          # repeated calls: partials buffers recycled within the stream
                    en.qmm(xin[i], xsin[i], qs[i].weight, qs[i].scales, qs[i].biases, outs[i], None, n,
                           installed.split_k(n, k, 64), 64, installed.bucket(m), f32, True)
            main_st.wait_stream(st)

    bad = []
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        for rep in range(2):
            for i, (_, n, k, f32) in enumerate(picks):
                xin[i].copy_((torch.randn((m, k), device=dev, generator=g) * (0.5 + rep)).to(torch.bfloat16))
                xsin[i].copy_(installed.group_sums(xin[i], 64))
                outs[i].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for i, (label, n, k, f32) in enumerate(picks):
                ref = run(eo, xin[i], xsin[i], qs[i], installed.split_k(n, k, 64), f32)
                if not torch.equal(bits(outs[i]), bits(ref)):
                    bad.append(f"replay {rep}: {label}")
    except RuntimeError as exc:
        bad.append(f"raised {str(exc).splitlines()[0][:100]}")
    check(not bad, f"a CUDA graph of matmuls on three side streams, replayed twice with new inputs == OLD eager "
                   f"({bad or 'all equal'})")

    if timing:
        def timed(ext, x, xs, qcopies, sk, f32, reps=20):
            out = run(ext, x, xs, qcopies[0], sk, f32)

            def body2():
                for q in qcopies:
                    ext.qmm(x, xs, q.weight, q.scales, q.biases, out, None, q.n, sk, 64, installed.bucket(x.shape[0]),
                            f32, True)
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                body2()
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body2()
            gr.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                gr.replay()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1e3 / reps / len(qcopies)

        for world in (4, 3, 2):
            for label, n, k, f32 in shapes(world):
                q0 = weights(n, k)
                copies = max(2, min(64, -(-(256 << 20) // q0.nbytes())))
                qc = [q0] + [installed.Q4(q0.weight.clone(), q0.scales.clone(), q0.biases.clone(), n, k, 64)
                             for _ in range(copies - 1)]
                sk = installed.split_k(n, k, 64)
                line = []
                for m in (1, 4, 8, 32, 64):
                    x = (torch.randn((m, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                    xs = installed.group_sums(x, 64)
                    to, tn = timed(eo, x, xs, qc, sk, f32), timed(en, x, xs, qc, sk, f32)
                    line.append(f"{m}: {to:.1f} -> {tn:.1f}")
                print(f"TIME TP={world} {label} ({n} x {k}, sk {sk}) us a call, rows " + ", ".join(line), flush=True)
                del qc, q0
                torch.cuda.empty_cache()
    shutil.rmtree(work, ignore_errors=True)
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:   # noqa: BLE001 - an error in the extension under test (e.g. a CUDA fault) is a failure
        print(f"FAIL the check stopped on {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}", flush=True)
        sys.exit(1)
