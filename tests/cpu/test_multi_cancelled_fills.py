"""0109's scheduler half: a filling stream whose client left ends before its next chunk, under --parallel.

The scheduler (``MultiDecoder._fill_turn`` / ``_iterate``, the real code) runs on CPU over a shell whose forward work is
replaced by deterministic stand-ins that depend on one stream alone (a chunk advances its prompt position; a stream's
token k is a function of its prompt and k). Tests: a filling stream whose client left is returned, done, before any
further chunk, with and without sliced fills (TF_GLM_FILL_BUDGET_MS), a decoding stream is not touched by its poll,
and a stream whose client is still there fills on."""

from types import SimpleNamespace

import pytest

import conftest  # noqa: F401
from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda.multi import Lane, MultiDecoder

FILL = 64                                   # TF_GLM_FILL_ROWS of the shell: prompt chunks while others decode
ENGINE_CHUNK = 256                          # the engine's own chunk (an idle server's)


def token(prompt, k):
    """Token k of a stream's reply: a function of its prompt and k alone, as a stream served alone computes it."""
    h = 17
    for t in prompt:
        h = (h * 31 + t) % 1000003
    return (h * 131 + k * 7919) % 251 + 2


def shell(*, share=0.5, budget=0, group=False):
    """A MultiDecoder with the scheduler's real methods and stand-in forwards; ``log`` records the work in order."""
    dec = object.__new__(MultiDecoder)
    dec.lanes, dec.partial, dec.group = {}, None, group
    dec.fill_budget, dec.profiles, dec.profile = budget, None, None
    dec.fill_rows, dec.e = FILL, SimpleNamespace(prefill_rows=ENGINE_CHUNK)
    dec.share, dec.credit = share, 0.0
    dec.decode_due = dec.short_round_due = False
    dec.eos, dec.rows_max, dec.lone_rows = (), 4, 0
    dec.tune = SimpleNamespace(lone=False)
    dec.outbox, dec.log = [], []
    dec._emit = lambda *a, **k: None
    dec._flush = lambda: None
    dec._grow = lambda *a, **k: True
    dec._ends = lambda lane: ()

    def fill(lane, stop, layers=None):
        dec.log.append(("fill", lane.sid, lane.st.pos, stop))
        lane.st.pos = stop
        n = len(lane.s.prompt)
        if stop < n:
            return None
        lane.decoding = True
        return token(lane.s.prompt, 0)

    def rnd(lanes, depths=None, lone=False, short=False):
        dec.log.append(("round", tuple(l.sid for l in lanes)))
        for l in lanes:
            l.s.take([token(l.s.prompt, len(l.s.out))], ())

    dec._fill, dec._round = fill, rnd
    return dec


def admit(dec, prompt, count, *, background=False, emit=None):
    s = Stream(list(prompt), count, None, background=background)
    s.emit = emit if emit is not None else (lambda new: None)
    s.sid = dec.next_sid = getattr(dec, "next_sid", -1) + 1
    lane = Lane(s=s, sid=s.sid, slot=s.sid, extent=SimpleNamespace(), st=SimpleNamespace(pos=0), order=s.sid, code=[0])
    dec.lanes[s.sid] = lane
    return lane


def run(dec, arrivals):
    """Iterate to the end; ``arrivals``: {iteration: [(prompt, count, kwargs)]}. Returns the streams."""
    streams, i = [], 0
    while True:
        for prompt, count, kw in arrivals.get(i, ()):
            streams.append(admit(dec, prompt, count, **kw).s)
        if not dec.lanes:
            if i >= max(arrivals):
                return streams
            i += 1
            continue
        done = dec._iterate()
        for s in done:
            dec.lanes.pop(s.sid, None)
        i += 1
        assert i < 500, dec.log


def prompt(n, seed):
    return [(seed * 7 + j * 13) % 250 + 2 for j in range(n)]


@pytest.mark.parametrize("budget", [0, 5])
def test_cancelled_filling_stream_ends_before_its_next_chunk(budget):
    """A filling stream whose client left is returned, done, before any further chunk, with and without sliced fills;
    a decoding stream is not touched by its poll."""
    gone = [False]
    dec = shell(budget=budget)
    lead = admit(dec, prompt(12, 1), 8)
    lead.s.emit.cancelled = lambda: True                       # a decoding stream's poll is the round's business
    left = admit(dec, prompt(200, 2), 3)
    left.s.emit.cancelled = lambda: gone[0]
    dec.lanes[lead.sid].decoding = True
    dec.log.clear()
    gone[0] = True
    done = dec._iterate()
    assert done == [left.s] and left.s.done and not lead.s.done
    assert dec.log == [] and left.st.pos == 0                  # no chunk, no round
    assert left.s.finished > 0
    assert dec.lanes[left.sid] is left                         # finish() retires it (and tells rank 1)


def test_not_cancelled_filling_stream_fills_on():
    dec = shell()
    admit(dec, prompt(100, 1), 3).s.emit.cancelled = lambda: False
    assert dec._iterate() == [] and dec.log == [("fill", 0, 0, 100)]
