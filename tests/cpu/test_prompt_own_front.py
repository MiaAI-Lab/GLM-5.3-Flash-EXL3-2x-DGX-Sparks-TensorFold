"""0102 (TF_GLM_PROMPT_OWN_FRONT=1, ``own_front``) and 0104 (TF_GLM_PROMPT_MICROBATCH=1, ``microbatch``) on CPU: Triton's interpreter, the tiny all-DSA checkpoint
(flash_fakes), two, three and four ranks over gloo. With the row split on, each rank runs the DSA blocks' replicated
front (q_a / kv_a and their norms, the indexer's keys, gates, queries and top-k pools) on its own rows and the ranks
share its outputs instead of the normed rows.

From the same caches (a context of 2,000 tokens filled with random rows), the tail of a prompt is prefilled in several
chunkings that cross the sparse limit (rows past position 2,050 attend their top-2,048 tokens): with the own-row
front, the head rows, final hidden rows and every cache row written (latents, index keys, gates, pooled keys) equal
the unchanged split path's and the unsplit path's bit for bit on every rank, and the caches are identical on every
rank; the own-row front ran on every DSA layer of every split chunk; chunks of two halves (with and without the
own-row front: the halves' chains in place, no side stream on CPU), also through the layer-sliced forward
(compute_steps, which joins the split at every layer and resumes it), leave the same bits; the bytes on the wire change
by exactly the
shares added less the normed rows no longer swapped (``wire_delta``; at the real widths that is fewer bytes: 6.8 KiB a
row instead of 8 KiB, and 4.8 KiB before the rows go sparse)."""

import hashlib
import multiprocessing as mp
import socket

import pytest
import torch

import flash_fakes

P0 = 2000                                      # tokens already in the caches
CHUNKINGS = ([70, 3, 87], [150, 9, 1])          # tests/gpu runs more chunkings, fast
CAP = 2304


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return flash_fakes.write_checkpoint(tmp_path_factory.mktemp("flashfront"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wire_delta(world: int, sizes: list[int]) -> int:
    """Bytes a rank sends with the own-row front less without, over a prompt tail in chunks ``sizes`` (those of 2+ rows
    split): a DSA layer shares its index keys and weights (bf16), gates (fp32), q_a / kv_a norms' rows (bf16) and, when
    the chunk has sparse rows, its pools (int32), a whole share of H rows to every peer; the glue before every DSA
    layer but the first (whose input rows the unsplit hc_pre made on every rank) no longer swaps its normed rows."""

    from tensorfold.families.glm5_next.cuda import sparse

    F = flash_fakes
    keys = (F.ID + F.IH) * 2 + F.ID * 4
    rows = (F.QL + F.KL) * 2
    pools = sparse.TOPK_POOLS * 4
    normed = F.D * 2
    total, at = 0, P0
    for R in sizes:
        if R < 2:                              # below the split's min_rows: unsplit, nothing shared or swapped
            at += R
            continue
        H = -(-R // world)
        sparse_rows = at + R - 1 >= sparse.SPARSE_FROM
        per_layer = keys + rows + (pools if sparse_rows else 0)
        total += (world - 1) * H * (F.LAYERS * per_layer - (F.LAYERS - 1) * normed)
        at += R
    return total


def halves_cut(R: int, world: int) -> int:
    """microbatch.cut with the tests' grid of 2."""
    import math

    unit = 2 * world // math.gcd(2, world)
    return max(unit, (R // 2) // unit * unit)


def test_settings():
    """TF_GLM_PROMPT_OWN_FRONT: 0 (default) or 1, only with the row split; it joins the ranks' startup comparison."""
    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings

    off = SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1"})
    on = SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PROMPT_OWN_FRONT": "1"})
    assert not off.own_front and on.own_front
    assert len(on.code()) == len(off.code()) == len(SplitSettings().code()) and on.code() != off.code()
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_PROMPT_OWN_FRONT": "1"})
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PROMPT_OWN_FRONT": "2"})
    assert "own rows" in on.describe() and "own rows" not in off.describe()


# -- ranks ----------------------------------------------------------------------------------------------------------
def _rank(folder, rank, world, port, exchange, out_q):
    try:
        out_q.put((rank, _body(folder, rank, world, port, exchange)))
    except BaseException:
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))


def _fill(w, st, g):
    """Random rows of every cache below P0: latents, index keys and gates, pooled keys (FP8 rows as kv8 writes)."""
    from tensorfold.families.glm5_next.cuda import kv8

    for kc in st.kc:
        kc[:P0].copy_(kv8.quantize_rows((2 * torch.randn(P0, kv8.width(kc), generator=g)).to(torch.bfloat16)))
    for ik, ig, pk in st.index:
        ik[:P0].copy_(torch.randn(P0, ik.shape[1], generator=g).to(torch.bfloat16))
        ig[:P0].copy_(torch.randn(P0, ig.shape[1], generator=g).to(torch.bfloat16))
        n = P0 // 4
        pk[:n].copy_(kv8.quantize_rows(torch.randn(n, kv8.width(pk), generator=g).to(torch.bfloat16)))
    st.set_pos(P0)


def digest(t) -> str:
    """A tensor's dtype, shape and bytes, as one string."""
    t = t.contiguous()
    return f"{t.dtype} {tuple(t.shape)} " + hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def log(rank, what) -> None:
    import sys
    import time

    print(f"[{time.strftime('%H:%M:%S')}] rank {rank}: {what}", file=sys.stderr, flush=True)


def _body(folder, rank, world, port, exchange):
    import torch.distributed as dist

    flash_fakes.env("fp8")
    torch.set_num_threads(1)
    import conftest  # noqa: F401  (the interpreter patches)

    flash_fakes.cpu_cuda_stubs()
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import microbatch, own_front
    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings, pad_rows

    microbatch.GRID = 2              # no KDA layer here: halves need not sit on its 64-row grid

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    w = flash_fakes.load_rank(folder, rank, world)
    log(rank, "loaded")
    comm = flash_fakes.GlooComm(world)
    w.comm = comm
    rows = 160
    b = F.Buffers(w, rows, capacity=CAP, prefill=True, pad=pad_rows(rows, world))
    # micro_steps: the halves with the own-row front through the layer-sliced forward (compute_steps), which joins
    # the split at every layer's yield and resumes it, as the engine's concurrent fill does
    modes = ("unsplit", "split", "own", "micro", "micro_own", "micro_steps")
    splits = {mode: HcSplit(w, b, SplitSettings(split=True, min_rows=2, exchange=exchange,
                                                own_front=mode in ("own", "micro_own", "micro_steps"),
                                                microbatch=mode.startswith("micro"), micro_min=2))
              for mode in modes[1:]}
    calls = {"front": 0, "halved": 0}
    real = own_front.front
    real_steps = microbatch.layer_steps

    def counted(*a, **k):
        calls["front"] += 1
        return real(*a, **k)

    def halved(*a, **k):
        calls["halved"] += 1
        return real_steps(*a, **k)

    own_front.front = counted
    microbatch.layer_steps = halved
    tokens = torch.randint(2, flash_fakes.V, (P0 + 160,), generator=torch.Generator().manual_seed(7)).tolist()
    base = F.State(w, CAP, 64, kv="fp8")
    _fill(w, base, torch.Generator().manual_seed(11))
    out = {}
    for ci, sizes in enumerate(CHUNKINGS):
        for mode in modes:
            st = base.clone()
            b.split = None if mode == "unsplit" else splits[mode]
            heads, hidden, before, sent, halves = [], [], calls["front"], comm.sent, calls["halved"]
            at = P0
            for R in sizes:
                b.ids[:R].copy_(torch.tensor(tokens[at:at + R], dtype=torch.int32))
                if mode.endswith("_steps"):
                    gen = F.compute_steps(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                    while True:
                        try:
                            next(gen)
                        except StopIteration as stop:
                            logits = stop.value
                            break
                else:
                    logits = F.compute(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                heads.append(logits.clone())
                hidden.append(b.hidden[:R].clone())
                F.commit(w, st, b, R, R)
                at += R
            caches = [kc[:at].clone() for kc in st.kc] + [x[:at].clone() for trio in st.index for x in trio[:2]] + \
                [trio[2][:at // 4].clone() for trio in st.index]
            # digests, not tensors: the parent compares bytes, and a rank may exit before it reads shared storage
            out[(ci, mode)] = {"heads": [digest(t) for t in heads], "hidden": [digest(t) for t in hidden],
                               "caches": [digest(t) for t in caches], "fronts": calls["front"] - before,
                               "sent": comm.sent - sent, "halved": calls["halved"] - halves}
            log(rank, f"chunks {sizes} {mode} done")
    # the caches every rank wrote, for the cross-rank comparison in the parent
    out["digest"] = {k: v["caches"] for k, v in out.items()}
    dist.barrier()
    dist.destroy_process_group()
    return out


def run_ranks(folder, world, exchange):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_rank, args=(str(folder), r, world, port, exchange, q)) for r in range(world)]
    for p in procs:
        p.start()
    import queue
    import time

    got, start = {}, time.monotonic()
    while len(got) < world:                    # a rank that dies (or fails) is reported at once, with its exit code
        try:
            r, res = q.get(timeout=20)
            got[r] = res
        except queue.Empty:
            dead = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode not in (None, 0) and r not in got]
            assert not dead, f"rank(s) exited without a result: {dead}"
            assert time.monotonic() - start < 5400, "ranks still running after 90 minutes"
    for p in procs:
        p.join(60)
    for r, res in got.items():
        assert "error" not in res, f"rank {r}:\n{res['error']}"
    return got


@pytest.mark.parametrize("world,exchange", [(2, "p2p"), (3, "gather"), (4, "p2p")])
def test_own_front_same_bits(folder, world, exchange):
    got = run_ranks(folder, world, exchange)
    dsa_layers = flash_fakes.LAYERS
    for r, res in got.items():
        for ci, sizes in enumerate(CHUNKINGS):
            ref, split, own = res[(ci, "unsplit")], res[(ci, "split")], res[(ci, "own")]
            for mode in ("split", "own", "micro", "micro_own", "micro_steps"):
                for name in ("heads", "hidden", "caches"):
                    assert ref[name] == res[(ci, mode)][name], f"rank {r} chunks {sizes}: {name} {mode} != unsplit"
            # every chunk whose halves are 2+ rows each ran halved
            want = sum(1 for R in sizes if R - halves_cut(R, world) >= 2 and halves_cut(R, world) >= 2)
            assert all(res[(ci, m)]["halved"] == want for m in ("micro", "micro_own", "micro_steps")), (r, sizes)
            # the own-row front ran on every DSA layer of every split chunk (min_rows 2: every chunk of 2+ rows)
            assert own["fronts"] == dsa_layers * sum(1 for R in sizes if R >= 2), (r, sizes, own["fronts"])
            assert split["fronts"] == 0
            assert own["sent"] - split["sent"] == wire_delta(world, sizes), (r, sizes, own["sent"], split["sent"])
    # every rank wrote the same caches
    digests = [got[r]["digest"] for r in range(world)]
    assert all(d == digests[0] for d in digests[1:])
