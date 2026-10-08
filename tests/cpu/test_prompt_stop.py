"""0109: a single-stream prompt votes after each chunk whether rank 0's client left (TF_GLM_PROMPT_STOP, default on).

The ranks run as threads over a rendezvous communicator (``Ranks``: the one all-gather the vote makes); the engine's real
``decode.prefill`` loop and ``GlmEngine._run_once`` / ``generate`` run over a shell whose chunk forward is a stand-in
that commits one deterministic state row a prompt row (each row a function of every row before it, so a row is equal
only when the whole prefix was). Tests: the setting; the vote (any rank's wish stops every rank; a broken poll never
wishes; a stop is sticky); all ranks stop after the same chunk; the committed rows equal those of a run of that prefix
alone and the stopped state resumes a longer prompt exactly; nothing stops without a wish or with TF_GLM_PROMPT_STOP=0
or on the last chunk; ``_run_once`` / ``generate`` report the stop (and ``RequestCancelled``) and leave the engine
ready; and the ranks' startup comparison covers the setting. The last test runs a real tiny model on two gloo ranks."""

import hashlib
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

import conftest  # noqa: F401
from tensorfold.families.glm5_next.cuda import decode
from tensorfold.families.glm5_next.cuda.decode import PromptStop, PromptStopped, prompt_stop_on

CHUNK = 64


def test_setting(monkeypatch):
    assert prompt_stop_on({}) and prompt_stop_on({"TF_GLM_PROMPT_STOP": ""}) and prompt_stop_on({"TF_GLM_PROMPT_STOP": "1"})
    assert not prompt_stop_on({"TF_GLM_PROMPT_STOP": "0"}) and not prompt_stop_on({"TF_GLM_PROMPT_STOP": " 0 "})
    for bad in ("2", "on", "-1"):
        with pytest.raises(ValueError):
            prompt_stop_on({"TF_GLM_PROMPT_STOP": bad})
    monkeypatch.delenv("TF_GLM_PROMPT_STOP", raising=False)
    assert prompt_stop_on()
    monkeypatch.setenv("TF_GLM_PROMPT_STOP", "0")
    assert not prompt_stop_on()


class Ranks:
    """``world`` ranks as threads; ``comm(rank)`` is that rank's all-gather over the shared rendezvous."""

    def __init__(self, world):
        self.world, self.barrier, self.slots, self.gathers = world, threading.Barrier(world, timeout=30), [None] * world, 0

    def comm(self, rank):
        owner = self

        class Comm:
            def all_gather(self, send, recv):
                owner.slots[rank] = send.clone()
                owner.barrier.wait()
                recv.copy_(torch.cat([owner.slots[r].reshape(-1) for r in range(owner.world)]))
                owner.barrier.wait()
                if rank == 0:
                    owner.gathers += 1

        return Comm()

    def run(self, fn):
        out, errors = [None] * self.world, []

        def go(r):
            try:
                out[r] = fn(r)
            except BaseException as exc:                    # noqa: BLE001
                errors.append((r, exc))
                self.barrier.abort()

        threads = [threading.Thread(target=go, args=(r,)) for r in range(self.world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors, errors
        return out


def weights(ranks, rank):
    return SimpleNamespace(comm=ranks.comm(rank), world=ranks.world, device=torch.device("cpu"), mtp=None, rank=rank)


class FakeEngine:
    """decode.Engine's surface ``prefill`` uses, its forward a stand-in: a row's state hashes every token before it."""

    def __init__(self, w, rows=CHUNK):
        self.w, self.prefill_rows, self.grid = w, rows, 0
        self.st = SimpleNamespace(pos=0)
        self.pbuf = None
        self.constraint = self.window = self.head = self.prompt_stop = None
        self.state, self.chunks = [], []
        self.last_hidden = None

    def reset(self):
        self.st.pos = 0
        self.state, self.chunks = [], []

    def sample(self, last, positions, sampling):
        return [int(last.sum()) % 251]

    def follow(self, tokens):
        pass


def fake_chunk(e, prompt, start, R, **kw):
    assert e.st.pos == start and len(e.state) == start
    h = hashlib.sha256(repr(prompt[:start]).encode()).hexdigest() if not e.state else e.state[-1]
    for t in prompt[start:start + R]:
        h = hashlib.sha256((h + str(t)).encode()).hexdigest()
        e.state.append(h)
    e.st.pos += R
    e.chunks.append((start, R))
    return torch.tensor([[float(start + R)]])


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    monkeypatch.setattr(decode, "prefill_chunk", fake_chunk)


def tokens(n):
    return [(j * 37 + 5) % 250 + 2 for j in range(n)]


def stop_run(world, n, polls_true_at, *, rank_with_poll=0, sample=False):
    """Every rank prefills ``n`` rows with a vote; ``rank_with_poll``'s poll turns true after chunk ``polls_true_at``
    (1-based; None never). Returns per rank: (stopped_at or None, state rows, chunks, votes taken)."""
    ranks = Ranks(world)
    prompt = tokens(n)

    def go(rank):
        e = FakeEngine(weights(ranks, rank))
        seen = [0]

        def poll():
            return polls_true_at is not None and len(e.chunks) >= polls_true_at

        e.prompt_stop = PromptStop(e.w, poll if rank == rank_with_poll else None)
        try:
            first = decode.prefill(e, prompt, None, drafter=None, sample=sample, mtp=False)
            return None, list(e.state), list(e.chunks), first
        except PromptStopped as stopped:
            return stopped.at, list(e.state), list(e.chunks), None

    return ranks.run(go), ranks


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("after", [1, 2, 3])
def test_all_ranks_stop_after_the_same_chunk(world, after):
    res, ranks = stop_run(world, 5 * CHUNK + 9, after)
    assert [r[0] for r in res] == [after * CHUNK] * world
    assert all(r[2] == [(k * CHUNK, CHUNK) for k in range(after)] for r in res)
    assert ranks.gathers == after                        # one tiny gather a chunk, none past the stop


@pytest.mark.parametrize("who", [1, 2])
def test_any_rank_may_wish(who):
    res, _ = stop_run(3, 4 * CHUNK + 1, 2, rank_with_poll=who)
    assert [r[0] for r in res] == [2 * CHUNK] * 3


@pytest.mark.parametrize("after", [1, 2, 4])
def test_committed_rows_equal_a_prefix_run(after):
    """The stopped state is the state of a run of that prefix alone, row for row, on every rank."""
    n = 5 * CHUNK + 9
    res, _ = stop_run(2, n, after)
    at = after * CHUNK
    ref, _ = stop_run(2, at, None)                     # the prefix as a prompt of its own: ends in the same chunks
    for got, want in zip(res, ref):
        assert got[0] == at and want[0] is None
        assert got[1] == want[1] and len(got[1]) == at and got[2] == want[2]
    assert res[0][1] == res[1][1]


def test_stopped_state_resumes_a_longer_prompt_exactly():
    """Rows committed before a stop are a prefix state: the rest of the prompt filled on top gives the full run's rows."""
    n = 4 * CHUNK + 20
    full, _ = stop_run(2, n, None)
    ranks = Ranks(2)
    prompt = tokens(n)

    def go(rank):
        e = FakeEngine(weights(ranks, rank))
        e.prompt_stop = PromptStop(e.w, (lambda: len(e.chunks) >= 2) if rank == 0 else None)
        with pytest.raises(PromptStopped):
            decode.prefill(e, prompt, None, sample=False, mtp=False)
        e.prompt_stop = None
        for start in range(e.st.pos, n, CHUNK):
            fake_chunk(e, prompt, start, min(CHUNK, n - start))
        return list(e.state)

    resumed = ranks.run(go)
    assert resumed[0] == resumed[1] == full[0][1]


def test_no_wish_no_stop_and_the_last_chunk_is_not_voted():
    n = 3 * CHUNK + 5
    res, ranks = stop_run(2, n, None, sample=True)
    assert all(r[0] is None and len(r[1]) == n and r[3] is not None for r in res)
    assert ranks.gathers == 3                            # after chunks 1-3 of 4: the last chunk's end is the prompt's
    res, ranks = stop_run(2, 2 * CHUNK, 99)              # a wish that comes only after the last chunk: no vote then
    assert all(r[0] is None for r in res) and ranks.gathers == 1
    res, _ = stop_run(2, CHUNK, 1)                       # a one-chunk prompt: nothing to stop
    assert all(r[0] is None and len(r[1]) == CHUNK for r in res)


def test_prefill_without_a_vote_is_unchanged():
    ranks = Ranks(2)
    prompt = tokens(3 * CHUNK + 5)

    def go(rank):
        e = FakeEngine(weights(ranks, rank))
        assert e.prompt_stop is None
        first = decode.prefill(e, prompt, None, sample=True, mtp=False)
        return first, len(e.state), e.chunks

    a, b = ranks.run(go)
    assert a == b and a[1] == len(prompt) and ranks.gathers == 0


def test_promptstop_vote_rules():
    ranks = Ranks(2)
    state = {}

    def go(rank):
        def broken():
            raise RuntimeError("poll failed")

        flip = [False]
        vote = PromptStop(weights(ranks, rank), broken if rank == 1 else (lambda: flip[0]))
        a = vote()                                       # nobody wishes (a broken poll never does)
        flip[0] = True
        b = vote()                                       # rank 0 wishes: both stop
        flip[0] = False
        c = vote()                                       # sticky
        return a, b, c, vote.stop

    assert ranks.run(go) == [(False, True, True, True)] * 2


def test_single_rank_needs_no_gather():
    vote = PromptStop(SimpleNamespace(comm=None, world=1, device=torch.device("cpu")), lambda: True)
    assert vote() is True
    vote = PromptStop(SimpleNamespace(), None)
    assert vote() is False


# -- the engine's wiring -----------------------------------------------------------------------------------------------

def engine_shell(ranks, rank, poll=None):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    g = GlmEngine.__new__(GlmEngine)
    g.rank, g.w, g.e = rank, weights(ranks, rank), FakeEngine(weights(ranks, rank))
    g.e.w = g.w
    g.live, g.cache, g.shared, g.grid, g.drafter, g.eos = [], [], 0, 0, None, ()
    g._drafters = lambda code: (False, False, False)
    g._take_over = lambda keep: None
    g._remember = lambda snap: g.cache.append(snap)
    g.taken = []

    def on_tokens(new):
        g.taken.extend(new)
        return False

    if poll is not None:
        on_tokens.cancelled = poll
    return g, on_tokens


def run_once(world, n, after, env=None, monkeypatch=None):
    ranks = Ranks(world)
    prompt = tokens(n)

    def go(rank):
        g = None
        g, on = engine_shell(ranks, rank, (lambda: len(g.e.chunks) >= after) if rank == 0 and after else None)
        stats = g._run_once(prompt, 1, None, True, on, [0, 0, 0, 0], None, False)
        return stats, g

    return ranks.run(go)


def test_run_once_stops_every_rank_alike(monkeypatch):
    monkeypatch.delenv("TF_GLM_PROMPT_STOP", raising=False)
    n = 4 * CHUNK + 3
    out = run_once(2, n, 2)
    for stats, g in out:
        assert stats["prompt_stopped"] == 2 * CHUNK and "decode_s" not in stats
        assert g.live == tokens(n)[:2 * CHUNK] and g.e.st.pos == 2 * CHUNK and g.e.prompt_stop is None
        assert g.taken == []                                     # no first token was sampled or emitted
    assert out[0][1].e.state == out[1][1].e.state


def test_run_once_without_a_wish_or_with_the_setting_off(monkeypatch):
    n = 4 * CHUNK + 3
    monkeypatch.delenv("TF_GLM_PROMPT_STOP", raising=False)
    for stats, g in run_once(2, n, None):
        assert "prompt_stopped" not in stats and g.e.st.pos == n and len(g.taken) == 1 and g.e.prompt_stop is None
    monkeypatch.setenv("TF_GLM_PROMPT_STOP", "0")                # a poll that is true from the start changes nothing
    out = run_once(2, n, 1)
    for stats, g in out:
        assert "prompt_stopped" not in stats and g.e.st.pos == n and len(g.taken) == 1 and g.e.prompt_stop is None
    full = run_once(2, n, None)
    assert out[0][1].e.state == full[0][1].e.state


def test_generate_raises_cancelled_after_a_stopped_prompt():
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.server.cancellation import RequestCancelled

    g = GlmEngine.__new__(GlmEngine)
    g.limit, g.serial_only, g.scheduler, g.vision, g.spill = 10_000, False, None, None, None
    g.request, g.policy, g.comm, g.cache = None, "auto", SimpleNamespace(), []
    g._effective = lambda code: code
    g._resume = lambda prompt, code: None
    g._shared = lambda prompt, hit: []
    g._ring = g._spill_wait = lambda *a: None
    g._share = lambda payload: None
    g.dump = None
    calls = []
    g._run = lambda *a, **k: calls.append(a) or {"prompt_stopped": 192, "prefill_s": 0.1}
    with pytest.raises(RequestCancelled, match="192 rows"):
        g.generate(tokens(300), 16, None, lambda new: False)
    g._run = lambda *a, **k: {"prefill_s": 0.1}
    assert g.generate(tokens(300), 16, None, lambda new: False)["drafts"] is True


def test_startup_comparison_covers_the_setting():
    """The ranks must agree on TF_GLM_PROMPT_STOP (the vote is a collective): it is in the settings they compare."""
    import inspect

    from tensorfold.families.glm5_next.cuda import engine

    source = inspect.getsource(engine.GlmEngine.__init__)
    assert "mine.append(int(prompt_stop_on()))" in source and "TF_GLM_PROMPT_STOP" in source


# -- a real model: two gloo ranks, the tiny checkpoint, the interpreter ---------------------------------------------

def _model_rank(folder, rank, world, port, out_q):
    try:
        out_q.put((rank, _model_body(folder, rank, world, port)))
    except BaseException:
        import traceback

        out_q.put((rank, {"error": traceback.format_exc()}))


def _model_body(folder, rank, world, port):
    import flash_fakes
    import torch.distributed as dist

    flash_fakes.env("fp8")
    torch.set_num_threads(1)
    flash_fakes.cpu_cuda_stubs()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    w = flash_fakes.load_rank(folder, rank, world)
    w.comm = flash_fakes.GlooComm(world)
    e = decode.Engine(w, capacity=512, max_rows=4, prefill_rows=CHUNK, kv="fp8", long_context=True)
    prompt = [(j * 37 + 5) % 250 + 2 for j in range(3 * CHUNK + 20)]

    def caches(n):
        st = e.st
        out = [kc[:n].clone() for kc in st.kc] + [x[:n].clone() for trio in st.index for x in trio[:2]]
        h = hashlib.sha256()
        for t in out:
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    seen = [0]
    real = decode.prefill_chunk

    def counting(*a, **k):
        seen[0] += 1
        return real(*a, **k)

    decode.prefill_chunk = counting

    def cpu_stage(w_, st, b, tokens):                      # forward.stage without the pinned buffer and its event
        b.ids[:len(tokens)].copy_(torch.tensor(list(tokens), dtype=torch.int32))
        return len(tokens)

    decode.stage = cpu_stage
    e.prompt_stop = PromptStop(w, (lambda: seen[0] >= 2) if rank == 0 else None)
    try:
        decode.prefill(e, prompt, None, sample=False, mtp=False)
        stopped = None
    except PromptStopped as s:
        stopped = s.at
    pos, stopped_caches = e.st.pos, caches(2 * CHUNK)
    e.prompt_stop = None
    decode.prefill(e, prompt[:2 * CHUNK], None, sample=False, mtp=False)         # the prefix alone
    prefix_caches = caches(2 * CHUNK)
    decode.prefill(e, prompt, None, sample=False, mtp=False)                     # no vote: the whole prompt
    dist.barrier()
    dist.destroy_process_group()
    return {"stopped": stopped, "pos": pos, "stopped_caches": stopped_caches, "prefix_caches": prefix_caches,
            "full_pos": e.st.pos}


def test_real_model_two_ranks_stop_at_the_same_chunk(tmp_path):
    import multiprocessing as mp
    import queue
    import socket
    import time

    import flash_fakes

    folder = flash_fakes.write_checkpoint(tmp_path / "ckpt")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_model_rank, args=(str(folder), r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    got, start = {}, time.monotonic()
    while len(got) < 2:
        try:
            r, res = q.get(timeout=20)
            got[r] = res
        except queue.Empty:
            dead = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode not in (None, 0) and r not in got]
            assert not dead, f"rank(s) exited without a result: {dead}"
            assert time.monotonic() - start < 1800, "ranks still running"
    for p in procs:
        p.join(60)
    for r, res in got.items():
        assert "error" not in res, f"rank {r}:\n{res['error']}"
        assert res["stopped"] == 2 * CHUNK and res["pos"] == 2 * CHUNK       # both ranks, the same chunk
        assert res["stopped_caches"] == res["prefix_caches"]                  # the rows of a prefix run, bit for bit
        assert res["full_pos"] == 3 * CHUNK + 20
