"""TF_NCCL_RING_INPLACE (0106) on CPU: the ring exchange (``tensorfold.cuda.comm.NCCL._exchange_ring``, TF_NCCL_RING=1,
0098) with each link's all-gather in place (``_ring_links``, the default) against 0098's pack / all-gather / unpack
steps (``_ring_steps``, TF_NCCL_RING_INPLACE=0).

N ranks as threads of one process, each link's two-rank NCCL all-gather replaced by a blocking fake that copies raw
bytes between the two ranks' buffers as NCCL does (in place when the input is the rank's own slot of the output).
Checked, for rings of three to six ranks (four Sparks are the real ring), several tensors a peer of mixed dtypes and odd
sizes (the message to the rank across the ring, N even, split in two halves):

- every message arrives bit for bit, in place and with the steps;
- the two modes leave the same bytes in every receive buffer (the in-place chains are a rewrite of the steps, so the
  exchange's result, and with it the prompt's bits, is the same);
- in place, no link call copies a rank's own input; with the steps there are copies (what in place saves);
- the two ends of a link name the same count in both modes (a rank sends what the neighbour expects);
- the default is in place; TF_NCCL_RING_INPLACE=0 keeps the steps.

The streams order nothing here (the fake links run at the call, in one thread each): the ordering of the in-place
chains (which link waits for which, the caller's stream joined at the end) is for a GPU to check."""

import contextlib
import ctypes
import inspect
import threading

import pytest
import torch

from tensorfold.cuda import comm as C


class Stream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass


class Event:
    def record(self, stream=None):
        pass


class Link:
    """One two-rank link: both ends post (send, recv, count), then each copies both slots into its own output."""

    def __init__(self) -> None:
        self.posted = {}
        self.barrier = threading.Barrier(2, timeout=30)
        self.local_copies = 0
        self.calls = 0
        self.lock = threading.Lock()


class Lib:
    def __init__(self, links: dict) -> None:
        self.links = links

    def ncclAllGather(self, send, recv, count, dtype, handle, stream):
        a, slot = handle
        link = self.links[a]
        link.posted[slot] = (send, recv, count)
        link.barrier.wait()
        for s in (0, 1):
            src, _, n = link.posted[s]
            assert n == count, "the two ends of a link name the same count"
            dst = recv + s * count
            if s == slot and src == dst:
                continue                      # in place: NCCL leaves the rank's own slot alone
            if s == slot:
                with link.lock:
                    link.local_copies += 1
            ctypes.memmove(dst, src, count)
        with link.lock:
            link.calls += 1
        link.barrier.wait()                   # neither end reuses its buffers before the other has read them
        return 0

    def ncclGetErrorString(self, code):
        return b"fake"


def ranks(N: int, inplace: bool):
    links = {a: Link() for a in range(N)}
    lib = Lib(links)
    out = []
    for me in range(N):
        n = object.__new__(C.NCCL)
        n.rank, n.world, n.ring, n.inplace, n.lib = me, N, True, inplace, lib
        n.links = {1: (me, 0), -1: ((me - 1) % N, 1)}
        n.link_streams = {1: Stream(), -1: Stream()}
        out.append(n)
    return out, links


@pytest.fixture
def cpu_cuda(monkeypatch):
    """The ring exchange's CUDA calls on CPU: streams that order nothing (the fake links run at the call), buffers
    allocated on the CPU."""
    real_empty, real_zeros = torch.empty, torch.zeros

    def drop(fn):
        def call(*a, **k):
            k.pop("device", None)
            return fn(*a, **k)
        return call

    monkeypatch.setattr(torch, "empty", drop(real_empty))
    monkeypatch.setattr(torch, "zeros", drop(real_zeros))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)


# what a row-split layer sends (bf16 partials of a share of rows, bf16 normed rows), and odd sizes
SPECS = [[(37, torch.float32)], [(64, torch.bfloat16), (5, torch.int32), (129, torch.float32)],
         [(1, torch.bfloat16)], [(2048, torch.bfloat16)], [(4096, torch.bfloat16), (4096, torch.bfloat16)]]


def payload(src: int, dst: int, i: int, n: int, dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * src + 100 * dst + i)
    size = torch.empty((), dtype=dtype).element_size()
    bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (n * size // 4 + 1,), generator=g, dtype=torch.int64).to(torch.int32)
    return bits.view(torch.int8)[:n * size].view(dtype).clone()


def run(N: int, spec, inplace: bool):
    """Every rank's receive buffers after one exchange of ``spec``'s tensors between every pair, the sends, the links."""
    comms, links = ranks(N, inplace)
    sends = [{t: [payload(me, t, i, n, dt) for i, (n, dt) in enumerate(spec)] for t in range(N) if t != me}
             for me in range(N)]
    recvs = [{t: [torch.full((n,), 7, dtype=dt) for n, dt in spec] for t in range(N) if t != me} for me in range(N)]
    errors = []

    def go(me):
        try:
            comms[me]._exchange_ring(sends[me], recvs[me])
        except BaseException as exc:          # noqa: BLE001
            errors.append((me, exc))
            for link in links.values():
                link.barrier.abort()

    threads = [threading.Thread(target=go, args=(me,)) for me in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return sends, recvs, links


def test_default_is_in_place():
    """TF_NCCL_RING_INPLACE defaults to 1 (the chains in place); only "0" keeps 0098's steps."""
    src = inspect.getsource(C.NCCL.__init__)
    assert 'os.environ.get("TF_NCCL_RING_INPLACE", "1").strip() != "0"' in src
    assert hasattr(C.NCCL, "_ring_links") and hasattr(C.NCCL, "_ring_steps")
    body = inspect.getsource(C.NCCL._exchange_ring)
    assert "self._ring_links(" in body and "self._ring_steps(" in body


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_in_place_equals_the_steps(cpu_cuda, N):
    for spec in SPECS:
        sends, fast, fast_links = run(N, spec, inplace=True)
        _, slow, slow_links = run(N, spec, inplace=False)
        for me in range(N):
            for src in range(N):
                if src == me:
                    continue
                for i, (n, dt) in enumerate(spec):
                    want = sends[src][me][i]
                    assert torch.equal(fast[me][src][i].view(torch.int8), want.view(torch.int8)), \
                        ("in place", N, spec, me, src, i)
                    assert torch.equal(slow[me][src][i].view(torch.int8), want.view(torch.int8)), \
                        ("steps", N, spec, me, src, i)
                    # the two modes: the same bytes in the receive buffer
                    assert torch.equal(fast[me][src][i].view(torch.int8), slow[me][src][i].view(torch.int8))
        # in place: no link call copies a rank's own input; the steps pack into a separate input each call
        assert sum(link.local_copies for link in fast_links.values()) == 0
        assert sum(link.local_copies for link in slow_links.values()) > 0
        # the same links carry the same number of all-gathers (both ends of each count once)
        assert sum(link.calls for link in fast_links.values()) == sum(link.calls for link in slow_links.values())


def test_four_rank_ring_row_split_shapes(cpu_cuda):
    """Four Sparks on a ring, the row-split exchange of a layer: every rank sends each peer a [n, 4096] bf16 block
    (the partials, TF_GLM_PROMPT_PARTIALS=bf16) and an fp32 one: in place and with the steps the receive buffers hold
    the same bytes, and each is what the sender sent."""
    spec = [(8 * 4096, torch.bfloat16), (8 * 4096, torch.float32)]
    sends, fast, _ = run(4, spec, inplace=True)
    _, slow, _ = run(4, spec, inplace=False)
    for me in range(4):
        for src in range(4):
            if src == me:
                continue
            for i in range(2):
                assert torch.equal(fast[me][src][i].view(torch.int8), slow[me][src][i].view(torch.int8))
                assert torch.equal(fast[me][src][i].view(torch.int8), sends[src][me][i].view(torch.int8))
