"""0102 and 0104 with real CUDA streams: two, three and four ranks as processes sharing this GPU (gloo between them, its
exchanges through host copies on the issuing stream), the tiny all-DSA checkpoint (flash_fakes) with the recipe's
4-bit dense projections, FP8 caches, and the row split's overlap on (its side stream, the glue's fronts). From the
same caches, a prompt tail prefilled with the own-row front, in two halves, and both, also through the layer-sliced
forward (compute_steps: the split joined at each layer's yield, main-stream work queued meanwhile, then resumed),
leaves the unsplit path's head rows, final rows and every cache row bit for bit: the events order the side stream's
shares, exchanges and glue against the main stream's blocks (a missing wait shows up as other bits)."""

import hashlib
import multiprocessing as mp
import os
import socket

import pytest
import torch

import flash_fakes

P0 = 2000
CHUNKINGS = ([160], [96, 64], [70, 90], [158, 1, 1])
CAP = 2304


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return flash_fakes.write_checkpoint(tmp_path_factory.mktemp("flashstreams"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank(folder, rank, world, port, exchange, out_q):
    try:
        out_q.put((rank, _body(folder, rank, world, port, exchange)))
    except BaseException:
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))


def _body(folder, rank, world, port, exchange):
    import torch.distributed as dist

    os.environ.update(TF_GLM_DENSE="q4", TF_GLM_KVB="bf16", TF_GLM_KV="fp8", TF_GLM_LATENT="1", TF_GLM_L2PF="0")
    torch.empty(1, device="cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 0.75 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import kv8, microbatch
    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings, pad_rows
    from tensorfold.families.glm5_next.cuda.weights import load

    microbatch.GRID = 2              # no KDA layer here: halves need not sit on its 64-row grid
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    w = load(folder, rank=rank, world=world, device="cuda", mtp=False)
    w.meta["long_context"] = True
    w.comm = flash_fakes.GlooComm(world)
    rows = 160
    b = F.Buffers(w, rows, capacity=CAP, prefill=True, pad=pad_rows(rows, world))
    common = dict(split=True, overlap=True, fronts=True, min_rows=2, micro_min=2, exchange=exchange)
    modes = {"split": SplitSettings(**common), "own": SplitSettings(**common, own_front=True),
             "micro": SplitSettings(**common, microbatch=True),
             "micro_own": SplitSettings(**common, microbatch=True, own_front=True),
             # the layer-sliced forward (compute_steps): the split joined at each layer's yield and resumed
             "micro_steps": SplitSettings(**common, microbatch=True, own_front=True),
             "own_steps": SplitSettings(**common, own_front=True)}
    splits = {m: HcSplit(w, b, s) for m, s in modes.items()}
    tokens = torch.randint(2, flash_fakes.V, (P0 + 160,), generator=torch.Generator().manual_seed(7)).tolist()
    base = F.State(w, CAP, 64, kv="fp8")
    g = torch.Generator().manual_seed(11)
    for kc in base.kc:
        kc[:P0].copy_(kv8.quantize_rows((2 * torch.randn(P0, kv8.width(kc), generator=g)).to(torch.bfloat16)))
    for ik, ig, pk in base.index:
        ik[:P0].copy_(torch.randn(P0, ik.shape[1], generator=g).to(torch.bfloat16))
        ig[:P0].copy_(torch.randn(P0, ig.shape[1], generator=g).to(torch.bfloat16))
        n = P0 // 4
        pk[:n].copy_(kv8.quantize_rows(torch.randn(n, kv8.width(pk), generator=g).to(torch.bfloat16)))
    base.set_pos(P0)
    out = {}
    for ci, sizes in enumerate(CHUNKINGS):
        for mode in ("unsplit", *modes):
            st = base.clone()
            b.split = None if mode == "unsplit" else splits[mode]
            digest = hashlib.sha256()
            at = P0
            for R in sizes:
                b.ids[:R].copy_(torch.tensor(tokens[at:at + R], dtype=torch.int32))
                if mode.endswith("_steps"):
                    gen = F.compute_steps(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                    while True:
                        try:
                            next(gen)
                            torch.cuda._sleep(2_000_000)       # a decode round's main-stream work meanwhile
                        except StopIteration as stop:
                            logits = stop.value
                            break
                else:
                    logits = F.compute(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                torch.cuda.synchronize()
                for t in (logits, b.hidden[:R]):
                    digest.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                F.commit(w, st, b, R, R)
                at += R
            for t in [kc[:at] for kc in st.kc] + [x[:at] for trio in st.index for x in trio[:2]] + \
                    [trio[2][:at // 4] for trio in st.index]:
                digest.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            out[(ci, mode)] = digest.hexdigest()
    dist.barrier()
    dist.destroy_process_group()
    return out


@pytest.mark.parametrize("world,exchange", [(2, "p2p"), (3, "p2p"), (4, "p2p"), (3, "gather"), (4, "gather")])
def test_streams_keep_the_bits(folder, world, exchange):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_rank, args=(str(folder), r, world, port, exchange, q)) for r in range(world)]
    for p in procs:
        p.start()
    got = dict(q.get(timeout=1800) for _ in procs)
    for p in procs:
        p.join(60)
    for r, res in got.items():
        assert "error" not in res, f"rank {r}:\n{res['error']}"
    for r, res in got.items():
        for ci, sizes in enumerate(CHUNKINGS):
            ref = res[(ci, "unsplit")]
            for mode in ("split", "own", "micro", "micro_own", "micro_steps", "own_steps"):
                assert res[(ci, mode)] == ref, f"rank {r} chunks {sizes}: {mode} != unsplit"
