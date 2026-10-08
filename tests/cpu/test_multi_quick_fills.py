"""0108: quick fills under --parallel (TF_GLM_QUICK_ROWS), and a cancelled filling stream ending before its next chunk.

The scheduler (``MultiDecoder._fill_turn`` / ``_iterate``, the real code) runs on CPU over a shell whose forward work is
replaced by deterministic stand-ins that depend on one stream alone (a chunk advances its prompt position; a stream's
token k is a function of its prompt and k), which is exactly the property the exactness argument rests on: the order of
chunks and rounds between streams moves no stream's rows or tokens. Tests: the setting, the order of chunks and
rounds with and without quick fills (a long prompt still alternates until its rest is quick; a background prompt is
never quick; a decoding-free server fills as before), every reply and every stream's chunk boundaries equal with
quick fills on and off, and a filling stream whose client left ending before its next chunk (with and without
TF_GLM_FILL_BUDGET_MS)."""

from types import SimpleNamespace

import pytest

import conftest  # noqa: F401
from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda.multi import Lane, MultiDecoder, quick_rows

FILL = 64                                   # TF_GLM_FILL_ROWS of the shell: prompt chunks while others decode
ENGINE_CHUNK = 256                          # the engine's own chunk (an idle server's)


def test_quick_rows_setting(monkeypatch):
    assert quick_rows(1024, "") == 1024 and quick_rows(512, "") == 512 and quick_rows(4096, "") == 1024
    assert quick_rows(1024, "0") == 0 and quick_rows(1024, "200") == 200 and quick_rows(256, "2000") == 256
    assert quick_rows(1024, " 7 ") == 7
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(ValueError):
            quick_rows(1024, bad)
    monkeypatch.delenv("TF_GLM_QUICK_ROWS", raising=False)
    assert quick_rows(2048) == 1024
    monkeypatch.setenv("TF_GLM_QUICK_ROWS", "0")
    assert quick_rows(2048) == 0
    monkeypatch.setenv("TF_GLM_QUICK_ROWS", "-3")
    with pytest.raises(ValueError):
        quick_rows(2048)


def token(prompt, k):
    """Token k of a stream's reply: a function of its prompt and k alone, as a stream served alone computes it."""
    h = 17
    for t in prompt:
        h = (h * 31 + t) % 1000003
    return (h * 131 + k * 7919) % 251 + 2


def shell(quick, *, share=0.5, budget=0, group=False):
    """A MultiDecoder with the scheduler's real methods and stand-in forwards; ``log`` records the work in order."""
    dec = object.__new__(MultiDecoder)
    dec.lanes, dec.partial, dec.group = {}, None, group
    dec.fill_budget, dec.profiles, dec.profile = budget, None, None
    dec.fill_rows, dec.e = FILL, SimpleNamespace(prefill_rows=ENGINE_CHUNK)
    dec.share, dec.credit, dec.quick = share, 0.0, quick_rows(FILL, str(quick))
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


def burst():
    """A stream decoding, then two short prompts and one longer than a chunk arriving together."""
    return {0: [(prompt(20, 1), 12, {})], 2: [(prompt(9, 2), 6, {}), (prompt(30, 3), 5, {}), (prompt(90, 4), 5, {})]}


@pytest.mark.parametrize("quick", ["", "0"])
def test_burst_fills_first(quick):
    """Default (quick = the 64-row chunk): the two short prompts fill in consecutive iterations; the long one (90 rows,
    64-row chunks) waits for the share's turn behind a round, then its quick 26-row rest fills next. TF_GLM_QUICK_ROWS=0:
    every chunk alternates with a round."""
    dec = shell(quick or 1024)
    assert dec.quick == (0 if quick else FILL)
    streams = run(dec, burst())
    fills = [i for i, e in enumerate(dec.log) if e[0] == "fill"]
    assert [dec.log[i][1] for i in fills] == [0, 1, 2, 3, 3], dec.log
    if not quick:
        at = fills[1]
        assert [e[0] for e in dec.log[at:at + 6]] == ["fill", "fill", "round", "fill", "fill", "round"], dec.log
        assert dec.log[at + 3][1:] == (3, 0, 64) and dec.log[at + 4][1:] == (3, 64, 90)
    else:
        assert all(dec.log[i + 1][0] == "round" for i in fills[1:]), dec.log
    for s in streams:
        assert s.out == [token(s.prompt, k) for k in range(s.count)]


def test_long_prompt_alternates_until_its_rest_is_quick():
    dec = shell(FILL)
    streams = run(dec, {0: [(prompt(10, 1), 40, {})], 1: [(prompt(300, 2), 3, {})]})
    long_fills = [e for e in dec.log if e[0] == "fill" and e[1] == 1]
    assert [(e[2], e[3]) for e in long_fills] == [(0, 64), (64, 128), (128, 192), (192, 256), (256, 300)]
    at = [i for i, e in enumerate(dec.log) if e[0] == "fill" and e[1] == 1]
    gaps = [at[k + 1] - at[k] for k in range(len(at) - 1)]
    assert gaps[:3] == [2, 2, 2] and gaps[3] == 1, (gaps, dec.log)    # 236 rows left are not quick; the last 44 are
    assert streams[1].out == [token(streams[1].prompt, k) for k in range(3)]


def test_background_prompt_is_never_quick():
    dec = shell(FILL)
    lead = admit(dec, prompt(12, 1), 8)
    assert dec._iterate() == [] and lead.decoding
    bg = admit(dec, prompt(10, 2), 4, background=True)
    assert not dec._quick(bg)
    assert dec._fill_turn() is None and dec.credit == 0.5       # the share's turn is a round's
    fg = admit(dec, prompt(10, 3), 4)
    assert dec._quick(fg) and dec._fill_turn() is fg            # quick: before the background prompt, no turn taken
    assert dec.credit == 0.5


def test_quick_picks_the_oldest_quick_foreground_prompt():
    dec = shell(FILL)
    lead = admit(dec, prompt(12, 1), 30)
    dec._iterate()
    long = admit(dec, prompt(200, 2), 3)                        # older, not quick
    a, b = admit(dec, prompt(40, 3), 3), admit(dec, prompt(8, 4), 3)
    assert not dec._quick(long) and dec._quick(a) and dec._quick(b) and lead.decoding
    assert dec._fill_turn() is a                                # next_fill: the oldest foreground quick one


def test_idle_server_fills_as_before():
    """No stream decoding: the quick rule is not consulted and the oldest foreground prompt fills in the engine's chunk."""
    dec = shell(FILL)
    admit(dec, prompt(500, 1), 3)
    admit(dec, prompt(8, 2), 3)
    assert dec._fill_turn() is dec.lanes[0]
    assert dec._stop(dec.lanes[0]) == ENGINE_CHUNK


@pytest.mark.parametrize("share", [0.5, 1.0, 0.25])
def test_replies_and_chunks_equal_without(share):
    """Quick fills move no stream's rows or tokens: replies, and every stream's own chunk boundaries, equal off."""
    arrivals = {0: [(prompt(20, 1), 12, {})], 1: [(prompt(150, 5), 6, {})],
                2: [(prompt(9, 2), 6, {}), (prompt(30, 3), 5, {}), (prompt(90, 4), 5, {}),
                    (prompt(33, 6), 4, {"background": True})], 6: [(prompt(60, 7), 5, {})]}
    got = {}
    for quick in (1024, 0):
        dec = shell(quick, share=share)
        streams = run(dec, arrivals)
        got[quick] = ([s.out for s in streams],
                      {sid: [(e[2], e[3]) for e in dec.log if e[0] == "fill" and e[1] == sid] for sid in range(len(streams))})
        for s in streams:
            assert s.out == [token(s.prompt, k) for k in range(s.count)]
    assert got[1024] == got[0]


def test_first_tokens_come_sooner_with_quick_fills():
    """The point: in a burst behind a decoding stream, the short prompts' first tokens (iteration index of their last
    chunk) come no later, and the second one strictly sooner."""
    when = {}
    for quick in (1024, 0):
        dec = shell(quick)
        run(dec, burst())
        last = {}
        for i, e in enumerate(dec.log):
            if e[0] == "fill":
                last[e[1]] = i
        when[quick] = last
    assert when[1024][1] <= when[0][1] and when[1024][2] < when[0][2]


@pytest.mark.parametrize("budget", [0, 5])
def test_cancelled_filling_stream_ends_before_its_next_chunk(budget):
    """A filling stream whose client left is returned, done, before any further chunk, with and without sliced fills;
    a decoding stream is not touched by its poll."""
    gone = [False]
    dec = shell(FILL, budget=budget)
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
    dec = shell(FILL)
    admit(dec, prompt(100, 1), 3).s.emit.cancelled = lambda: False
    assert dec._iterate() == [] and dec.log == [("fill", 0, 0, 100)]
