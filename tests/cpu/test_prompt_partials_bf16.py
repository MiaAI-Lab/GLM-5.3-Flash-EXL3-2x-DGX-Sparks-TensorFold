"""0106: TF_GLM_PROMPT_PARTIALS=bf16 (``hcsplit``, ``glue.hc_post``) on CPU: Triton's interpreter, the tiny all-DSA
checkpoint (flash_fakes), two, three and four ranks over gloo.

A row-split prompt chunk sends each rank's partial of the peers' rows as bf16 instead of fp32 (p2p: half the bytes),
and every rank's partial of a row is rounded to bf16 before the rank-ordered fp32 sum, so that a row's bits depend on
neither the chunking nor the exchange. Checked:

- the kernel: ``hc_post`` / ``hc_post_pair`` / ``hc_post_ranks`` against a torch reference of the arithmetic: fp32
  partials as before (bit for bit the old kernel's sum and rounding), ``round_parts`` rounding each partial first, bf16
  partials the same as the rounded fp32 ones;
- the settings: fp32 by default, bf16 only with the row split, in the ranks' startup comparison (``code``);
- end to end, from the same caches (a context of 2,040 tokens, rows past 2,050 attend their top-2,048 tokens), a
  prompt tail in two chunkings: with bf16 partials the unsplit chunk (every chunk shorter than the split's minimum rows:
  ``hc_post`` rounds), the split chunk, the own-row front, two micro-batches a chunk (also with the own-row front,
  through the layer-sliced forward) leave the same final hidden rows and every cache row bit for bit, on every
  rank, p2p and all-gather, and the same hidden rows and caches in either chunking; the caches are identical on
  every rank;
- fp32 (the default) is untouched: split == unsplit as before, and it differs from bf16 (the prompt's bits change once
  by design: not quality-checked on Flash, so opt-in);
- the bytes on the wire: with p2p exactly half the partials' bytes less (``partial_bytes``); the all-gather exchange
  sends the same."""

import hashlib
import os
import multiprocessing as mp
import socket

import pytest
import torch

import flash_fakes

P0 = 2040                                      # tokens already in the caches
CHUNKINGS = ([40, 3, 21], [55, 8, 1])      # 64 rows past 2,040: the rows past 2,050 are sparse
CAP = 2304


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return flash_fakes.write_checkpoint(tmp_path_factory.mktemp("flashbf16"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- settings -------------------------------------------------------------------------------------------------------
def test_settings_from_env():
    """TF_GLM_PROMPT_PARTIALS: fp32 (default) or bf16, only with the row split; it joins the startup comparison."""
    from tensorfold.families.glm5_next.cuda.hcsplit import PARTIALS, SplitSettings

    assert PARTIALS == ("fp32", "bf16")
    off = SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1"})
    on = SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PROMPT_PARTIALS": "bf16"})
    assert off.partials == "fp32" and on.partials == "bf16"
    assert on.code() != off.code() and len(on.code()) == len(off.code()) == len(SplitSettings().code())
    assert SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PROMPT_PARTIALS": "fp32"}).code() == off.code()
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_PROMPT_PARTIALS": "bf16"})
    with pytest.raises(ValueError):
        SplitSettings.from_env({"TF_GLM_HC_SPLIT": "1", "TF_GLM_PROMPT_PARTIALS": "fp16"})
    assert "bf16" in on.describe() and "bf16" not in off.describe()


# -- the kernel -----------------------------------------------------------------------------------------------------
def reference(x, parts, post, comb, rounded: bool):
    """hc_post's arithmetic in torch: the partials (rounded to bf16 each if ``rounded``) summed in fp32 rank 0 first,
    the sum rounded to bf16, then post * branch + the streams mixed by comb in fixed order, rounded to bf16."""
    world, rows, d = parts.shape
    p = parts.to(torch.bfloat16).float() if rounded else parts.float()
    acc = p[0].clone()
    for k in range(1, world):
        acc = acc + p[k]
    branch = acc.to(torch.bfloat16).float()
    xs = [x[:, s * d:(s + 1) * d].float() for s in range(4)]
    out = torch.empty_like(x)
    for s in range(4):
        c = [comb[:, k * 4 + s:k * 4 + s + 1] for k in range(4)]
        mixed = ((c[0] * xs[0] + c[1] * xs[1]) + c[2] * xs[2]) + c[3] * xs[3]
        out[:, s * d:(s + 1) * d] = (post[:, s:s + 1] * branch + mixed).to(torch.bfloat16)
    return out


def test_hc_post_rounding():
    import conftest  # noqa: F401  (the interpreter patches)

    from tensorfold.families.glm5_next.cuda import glue

    g = torch.Generator().manual_seed(7)
    for world in (2, 3, 4):
        rows, d = 5, 256
        parts = (torch.randn((world, rows, d), generator=g) * torch.logspace(-3, 2, d)).contiguous()
        x = (torch.randn((rows, 4 * d), generator=g) * 4).to(torch.bfloat16)
        post = torch.rand((rows, 4), generator=g) * 2
        comb = torch.rand((rows, 16), generator=g)
        rounded = parts.to(torch.bfloat16).contiguous()

        def same(a, b):
            return torch.equal(a.view(torch.int16), b.view(torch.int16))

        out = torch.empty_like(x)
        glue.hc_post(x, out, parts, post, comb)                         # fp32, the default: the old arithmetic
        assert same(out, reference(x, parts, post, comb, False))
        glue.hc_post(x, out, parts, post, comb, round_parts=True)       # fp32 partials rounded first
        assert same(out, reference(x, parts, post, comb, True))
        assert not same(out, reference(x, parts, post, comb, False))     # the rounding changes bits
        want = out.clone()
        glue.hc_post(x, out, rounded, post, comb)                       # bf16 partials: rounded already
        assert same(out, want)
        glue.hc_post(x, out, rounded, post, comb, round_parts=True)     # nothing more to round
        assert same(out, want)
        if world == 2:                                                  # two apart (the split's own and received rows)
            for g0, g1, ref_round in ((parts[0], parts[1], True), (rounded[0], rounded[1], False)):
                glue.hc_post_pair(x, out, g0.contiguous(), g1.contiguous(), post, comb, round_parts=ref_round)
                assert same(out, want)
            with pytest.raises(ValueError):
                glue.hc_post_pair(x, out, parts[0].contiguous(), rounded[1].contiguous(), post, comb)
        if world == 3:                                                  # rows of one gathered buffer
            glue.hc_post_ranks(x, out, parts[0], rows * d, world, post, comb, round_parts=True)
            assert same(out, want)


# -- ranks ----------------------------------------------------------------------------------------------------------
def digest(t) -> str:
    t = t.contiguous()
    return f"{t.dtype} {tuple(t.shape)} " + hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


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


FP32 = ("f_unsplit", "f_split")
BF16 = ("b_unsplit", "b_split", "b_micro", "b_steps")      # b_steps: halves with the own-row front, layer-sliced


def _rank(folder, rank, world, port, exchange, chunkings, out_q):
    try:
        out_q.put((rank, _body(folder, rank, world, port, exchange, chunkings)))
    except BaseException:
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))


def _body(folder, rank, world, port, exchange, chunkings):
    import torch.distributed as dist

    flash_fakes.env("fp8")
    torch.set_num_threads(1)
    import conftest  # noqa: F401

    flash_fakes.cpu_cuda_stubs()
    from tensorfold.families.glm5_next.cuda import forward as F
    from tensorfold.families.glm5_next.cuda import microbatch
    from tensorfold.families.glm5_next.cuda.hcsplit import HcSplit, SplitSettings, pad_rows

    microbatch.GRID = 2              # no KDA layer here: halves need not sit on its 64-row grid
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    w = flash_fakes.load_rank(folder, rank, world)
    comm = flash_fakes.GlooComm(world)
    w.comm = comm
    rows = 64
    b = F.Buffers(w, rows, capacity=CAP, prefill=True, pad=pad_rows(rows, world))

    def settings(mode):
        bf = mode.startswith("b_")
        return SplitSettings(split=True, min_rows=2 if mode not in ("f_unsplit", "b_unsplit") else 1000,
                             exchange=exchange, partials="bf16" if bf else "fp32",
                             own_front=mode == "b_steps",
                             microbatch=mode in ("b_micro", "b_steps"), micro_min=2)

    # the bf16 unsplit run has the split attached, its minimum rows past every chunk: layer_forward rounds
    splits = {mode: HcSplit(w, b, settings(mode)) for mode in FP32[1:] + BF16}
    splits["f_unsplit"] = None
    halved = {"n": 0}
    real_steps = microbatch.layer_steps

    def counted(*a, **k):
        halved["n"] += 1
        return real_steps(*a, **k)

    microbatch.layer_steps = counted
    tokens = torch.randint(2, flash_fakes.V, (P0 + 64,), generator=torch.Generator().manual_seed(7)).tolist()
    base = F.State(w, CAP, 64, kv="fp8")
    _fill(w, base, torch.Generator().manual_seed(11))
    out = {}
    for ci, sizes in enumerate(CHUNKINGS[:chunkings]):
        for mode in FP32 + BF16:
            st = base.clone()
            b.split = splits[mode]
            hidden, sent, n_halved = [], comm.sent, halved["n"]
            at = P0
            for R in sizes:
                b.ids[:R].copy_(torch.tensor(tokens[at:at + R], dtype=torch.int32))
                if mode == "b_steps":
                    gen = F.compute_steps(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                    while True:
                        try:
                            next(gen)
                        except StopIteration:
                            break
                else:
                    F.compute(w, st, b, R, nch=F.chunks_for(st, R), host_pos=st.pos)
                hidden.append(b.hidden[:R].clone())
                F.commit(w, st, b, R, R)
                at += R
            caches = [kc[:at].clone() for kc in st.kc] + [x[:at].clone() for trio in st.index for x in trio[:2]] + \
                [trio[2][:at // 4].clone() for trio in st.index]
            out[(ci, mode)] = {"hidden": digest(torch.cat(hidden)), "caches": [digest(t) for t in caches],
                               "sent": comm.sent - sent, "halved": halved["n"] - n_halved}
    dist.barrier()
    dist.destroy_process_group()
    return out


def run_ranks(folder, world, exchange, chunkings):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_rank, args=(str(folder), r, world, port, exchange, chunkings, q)) for r in range(world)]
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


def partial_bytes(world: int, sizes: list[int]) -> int:
    """The fp32 partials' bytes a rank sends with p2p over a prompt tail in chunks ``sizes`` (those of 2+ rows split):
    two sites a layer, its partial of each peer's share of ceil(R / world) rows, hidden values a row, 4 bytes each."""
    F = flash_fakes
    return sum(F.LAYERS * 2 * (world - 1) * -(-R // world) * F.D * 4 for R in sizes if R >= 2)


def halves_cut(R: int, world: int) -> int:
    """microbatch.cut with the tests' grid of 2."""
    import math

    unit = 2 * world // math.gcd(2, world)
    return max(unit, (R // 2) // unit * unit)


slow = pytest.mark.skipif(not os.environ.get("TF_SLOW_TESTS"), reason="the full matrix (hours in the interpreter): "
                          "TF_SLOW_TESTS=1")
# one chunking by default: 2 ranks p2p, 3 ranks all-gather, 4 ranks p2p (own-row front and micro-batches in b_steps);
# the rest of the matrix, and both chunkings (hidden rows and caches equal across them), with TF_SLOW_TESTS=1
CASES = [pytest.param(2, "p2p", 1), pytest.param(3, "gather", 1), pytest.param(4, "p2p", 1)] + \
    [pytest.param(w, x, 2, marks=[pytest.mark.slow, slow]) for w, x in
     ((2, "p2p"), (2, "gather"), (3, "p2p"), (3, "gather"), (4, "p2p"), (4, "gather"))]


@pytest.mark.parametrize("world,exchange,chunkings", CASES)
def test_bf16_partials_same_bits(folder, world, exchange, chunkings):
    got = run_ranks(folder, world, exchange, chunkings)
    for r, res in got.items():
        # fp32 (the default) is untouched: split == unsplit; bf16 partials are other bits than fp32's
        for ci, sizes in enumerate(CHUNKINGS[:chunkings]):
            f_ref = res[(ci, "f_unsplit")]
            assert res[(ci, "f_split")]["hidden"] == f_ref["hidden"] and res[(ci, "f_split")]["caches"] == f_ref["caches"]
            ref = res[(ci, "b_unsplit")]
            assert ref["hidden"] != f_ref["hidden"], "bf16 partials left fp32's bits: the rounding is not on"
            for mode in BF16:
                assert res[(ci, mode)]["hidden"] == ref["hidden"], f"rank {r} chunks {sizes}: hidden {mode}"
                assert res[(ci, mode)]["caches"] == ref["caches"], f"rank {r} chunks {sizes}: caches {mode}"
            want = sum(1 for R in sizes if R - halves_cut(R, world) >= 2 and halves_cut(R, world) >= 2)
            assert all(res[(ci, m)]["halved"] == want for m in ("b_micro", "b_steps")), (r, sizes)
            assert want > 0
            # the wire: p2p sends half the partials' bytes less; the all-gather exchange sends fp32 either way
            sent_f, sent_b = res[(ci, "f_split")]["sent"], res[(ci, "b_split")]["sent"]
            if exchange == "p2p":
                assert sent_f - sent_b == partial_bytes(world, sizes) // 2, (r, sizes, sent_f, sent_b)
                assert sent_f > sent_b
            else:
                assert sent_f == sent_b, (r, sizes, sent_f, sent_b)
        # either chunking: the same hidden rows and cache rows, bit for bit
        for mode in BF16 if chunkings == 2 else ():
            assert res[(0, mode)]["hidden"] == res[(1, mode)]["hidden"], f"rank {r}: hidden {mode} across chunkings"
            assert res[(0, mode)]["caches"] == res[(1, mode)]["caches"], f"rank {r}: caches {mode} across chunkings"
    # every rank wrote the same caches
    for key in got[0]:
        assert all(got[r][key]["caches"] == got[0][key]["caches"] for r in range(1, world)), key
