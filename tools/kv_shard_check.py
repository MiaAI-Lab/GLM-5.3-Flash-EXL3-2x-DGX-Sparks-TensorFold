#!/usr/bin/env python3
"""Checks for patch 0129 (KV sharding, TF_GLM_KV_SHARD: tensorfold/families/glm5_next/cuda/shard.py and its pool and
placement hooks in pool.py / multi.py), CPU only, run inside the image that scripts/prepare.sh built (no GPU, no model,
no server):

    docker run --rm --entrypoint python -e CUDA_VISIBLE_DEVICES= -v "$PWD/tools/kv_shard_check.py:/c.py" \
      tensorfold-glm53:v0.6.0 /c.py

A real Pool split in parts (one a rank; two halves at two ranks) and MultiDecoder's own placement and eviction paths:
- when no part has room, evictions happen in one part only, the one whose newest needed eviction is the oldest (the
  same states one pool would lose, as near as parts allow), never in a part the new extent does not go to;
- an explicit part (a resume into its owner's part) and a grown extent evict only in their own part;
- a new stream goes to the part with fewer live streams (each row's owner computes its decode attention), then the
  one with more free rows; free rows are cut at every part's end and no extent crosses one;
- with one pool (no shard) placement and eviction are as before (least recently used first);
- the sharded pool's size (each token's latent once over N ranks, its pooled keys on every rank) and its parts at
  two, three and four ranks;
- (9-) the v5 hardening: a filling prompt's twin is pinned (compaction goes around it, never moves it, never counts
  its rows as room), a placement no part can make evicts nothing, ``fits`` looks at parts, the TP3 decode exchange
  (uneven heads) has equal sends and receives, the ranks agree on one pool size, bf16 rows travel as bytes, the sharded
  pool never exceeds its memory budget, and the shard-specific refusals, signature and asserts.
- (10) the cross-part carry (TF_GLM_KV_SHARD_CARRY): ``_place_hit``'s rules (a full part no longer drops a hit: another part's
  room keeps it; free rows elsewhere spare other conversations' states; a gap of 2 live streams carries to the lighter
  part, never toward a busier one, never under 2,048 tokens, never by evicting) against the old placement code as the
  reference (carry off, or one pool: identical decisions, evictions, moves and counters), rows that are pinned (a twin,
  a slice in flight, a read) never carried, ADMIT copy 2 and XMOVE flushed before the collective and applied alike on
  rank 1, ``Arena.carry`` on simulated ranks (two and three: FP8 and bf16 bytes, the display span by kernel, a partial
  pooled-key row, pad rows, who sends and who receives), live streams relocated when they cannot grow, and the spill
  tier (block names carry the part, unsharded names unchanged, a carried state and a state moved with writes waiting
  restore byte for byte). tools/kv_shard_mutants.sh runs it on faithful mutants of each.
The decode and prompt kernels' equality (64 heads at once == 32 + 32; a compact copy of the selected rows == the
arena; the marks-and-prefix-sums history == a sort and a search) needs a GPU and is checked on two Sparks."""
import os, random, sys
from types import SimpleNamespace
from tensorfold.families.glm5_next.cuda import multi
from tensorfold.families.glm5_next.cuda.pool import Pool, align_up

A = 4096
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); (fails.append(name) if not ok else None)

def section(name, fn):
    """Run one group of checks; an exception (a missing function, a refusal that did not come) is a FAIL, not a crash."""
    try:
        fn()
    except Exception as exc:                       # noqa: BLE001
        check(f"{name}: raised {type(exc).__name__}: {str(exc)[:90]}", False)
        if os.environ.get("KV_CHECK_TRACE"):          # (a failing section's traceback, for whoever is debugging it)
            import traceback; traceback.print_exc()

class Stub:
    _room = multi.MultiDecoder._room; _victims = multi.MultiDecoder._victims
    _gathered = multi.MultiDecoder._gathered; _compact = multi.MultiDecoder._compact
    _settle = multi.MultiDecoder._settle; _grow = multi.MultiDecoder._grow
    _eviction_cost = getattr(multi.MultiDecoder, "_eviction_cost", None)
    _cheaper_half = getattr(multi.MultiDecoder, "_cheaper_half", None)
    _revoke_twins = getattr(multi.MultiDecoder, "_revoke_twins", lambda self, half: False)
    def __init__(self, rows, half):
        self.pool = Pool(rows, half=half); self.kept = []; self.evicted = []; self.moves = 0
        self.lanes, self.partial, self.twins = {}, None, {"made": 0, "revoked": 0}
        self.no_part = 0
    def _emit(self, op, payload): return []
    def _evict(self, c, spill=True):
        self.evicted.append(c.name); self.kept = [k for k in self.kept if k is not c]
        c.extent.kept = [k for k in c.extent.kept if k is not c]; self._settle(c.extent)
    def _move(self, x, base, size): self.moves += 1; self.pool.move(x, base, size)
    def _resize(self, x, size): self.pool.resize(x, size)
    def keep(self, name, base, n):
        x = self.pool.add(base, n); c = SimpleNamespace(name=name, ids=[0] * n, extent=x); x.kept.append(c)
        self.kept.append(c); return c

H = 64 * A
# 1. halves: LRU interleaved; the half whose needed victims are oldest is chosen, evictions only there
s = Stub(2 * H, H)
s.keep("a0", 0, 20 * A); s.keep("b0", H, 20 * A); s.keep("a1", 20 * A, 20 * A); s.keep("b1", H + 20 * A, 30 * A)
s.keep("a2", 40 * A, 24 * A); s.keep("b2", H + 50 * A, 14 * A)
# free: half0 0, half1 0. need 30A: half0 needs a0+a1 (newest index 2), half1 needs b0+b1? b0 frees 20 -> 20<30, b1 -> 50: newest idx 3
base = s._room(30 * A)
check("1 room in the half losing older states", base is not None and s.pool.half_of(base) == 0)
check("1 evictions only there (a0, a1)", s.evicted == ["a0", "a1"])
# 2. explicit half: evicts only in it, even when the globally oldest is in the other
s = Stub(2 * H, H)
s.keep("a0", 0, 64 * A); s.keep("b0", H, 32 * A); s.keep("b1", H + 32 * A, 32 * A)
base = s._room(16 * A, half=1)
check("2 explicit half evicts only its own", s.evicted == ["b0"] and s.pool.half_of(base) == 1)
# 3. unsharded unchanged: global LRU
s = Stub(2 * H, 0)
s.keep("a0", 0, 64 * A); s.keep("b0", H, 32 * A); s.keep("b1", H + 32 * A, 32 * A)
base = s._room(16 * A)
check("3 one pool: least recently used first", s.evicted == ["a0"])
# 4. placement: fewer live streams first when both fit
s = Stub(2 * H, H)
x = s.pool.add(0, 8 * A); x.owner = 1; y = s.pool.add(8 * A, 8 * A); y.owner = 2
z = s.pool.add(H, 40 * A)        # kept, no owner
check("4 fewer live streams first", s.pool.half_of(s.pool.place(4 * A)) == 1)
y.owner = None; x.owner = None
check("4 tie: more free rows", s.pool.half_of(s.pool.place(4 * A)) == 0)
# 5. grow evicts only in its half
s = Stub(2 * H, H)
c0 = s.keep("a0", 0, 32 * A); s.keep("b0", H, 32 * A); s.keep("a1", 32 * A, 32 * A)
g = s.pool.add(H + 32 * A, 16 * A); g.owner = 7
ok = s._grow(g, 40 * A)
check("5 grow evicts in its own half only", ok and s.evicted == ["b0"])
# 6. random: with halves every eviction is in the half the placement lands in; never more than needed in it
rnd = random.Random(5)
bad = 0
for trial in range(300):
    s = Stub(2 * H, H)
    for h in (0, 1):
        at = h * H
        while True:
            n = rnd.randrange(1, 20) * A
            if at + n > (h + 1) * H: break
            if rnd.random() < 0.8: s.keep(f"{h}-{at}", at, n)
            at += n
    rnd.shuffle(s.kept)
    need = rnd.randrange(1, 64) * A
    before = list(s.kept)
    base = s._room(need)
    if base is None: continue
    hb = s.pool.half_of(base)
    ev = [c for c in before if c.name in s.evicted]
    if any(s.pool.half_of(c.extent) != hb for c in ev if c.extent in s.pool.extents) or \
       any(int(c.name.split("-")[0]) != hb for c in ev):
        bad += 1; continue
    # minimality: undoing the newest eviction leaves too little room in that half
    if ev:
        freed_last = len(ev[-1].ids)
        if s.pool.free_rows(hb) - align_up(freed_last) >= need and len(ev) > 0:
            bad += 1
check("6 random: evictions in one half, none beyond the need", bad == 0)
# 7. N parts (TP3/TP4): placement, cuts at every part's end, eviction in the part losing the oldest states
for N in (3, 4):
    P = 32 * A
    s = Stub(N * P, P)
    check(f"7 N={N} parts", s.pool.parts == N)
    gaps = s.pool.gaps()
    check(f"7 N={N} free rows cut at every part's end", gaps == [(i * P, P) for i in range(N)])
    x = s.pool.add(0, 8 * A); x.owner = 1
    y = s.pool.add(P, 8 * A); y.owner = 2
    check(f"7 N={N} new stream to a part with no live stream", s.pool.half_of(s.pool.place(4 * A)) == 2)
    try:
        s.pool.add(P - A, 2 * A); crossed = True
    except ValueError:
        crossed = False
    check(f"7 N={N} no extent across a part's end", not crossed)
    s = Stub(N * P, P)
    names = []
    for h in range(N):                      # each part full: two kept prompts of 16 A; LRU interleaved by part
        for j in range(2):
            names.append(s.keep(f"{h}-{j}", h * P + j * 16 * A, 16 * A))
    order = [c for j in range(2) for c in names if c.name.endswith(f"-{j}")]
    s.kept = order                           # oldest first: every part's first, then every part's second
    base = s._room(24 * A)                   # needs both of one part's prompts: the part whose second is oldest
    check(f"7 N={N} eviction in one part, the one losing the oldest", s.evicted == ["0-0", "0-1"] and s.pool.half_of(base) == 0)
# 8. sharded sizes and setup at N ranks
from tensorfold.families.glm5_next.cuda import shard
for N, want in ((2, 1.88), (3, 2.66), (4, 3.36)):
    rows = shard.sizes(2_000_000, 11, "fp8", 1_048_576, N)
    check(f"8 N={N} sizes: whole parts of extents ({rows}, x{rows / 2_000_000:.2f})", rows % (N * 2048) == 0 and abs(rows / 2_000_000 - want) < 0.06)
    shard.setup(1, N, rows, heads=[22, 21, 21] if N == 3 else None)
    check(f"8 N={N} setup: part {rows // N}", shard.HALF == rows // N and shard.WORLD == N and shard.owner(shard.HALF * (N - 1)) == N - 1)
shard.setup(0, 2, 4096 * 512)

# ---- 9- : the v5 review fixes ------------------------------------------------------------------------------------------
import torch

H = 32 * A                      # (a part of 32 A rows from here on)

class Real(Stub):
    """Stub with the real _move / _used / _untwin / fits (a twin lane has a State stub that records its binds)."""
    _move = multi.MultiDecoder._move; _used = multi.MultiDecoder._used; _untwin = multi.MultiDecoder._untwin
    fits = multi.MultiDecoder.fits; _need = multi.MultiDecoder._need
    _shard_health = getattr(multi.MultiDecoder, "_shard_health", None)
    _pinned = getattr(multi.MultiDecoder, "_pinned", None)
    _revoke = multi.MultiDecoder._revoke_twins
    def __init__(self, rows, half):
        super().__init__(rows, half); self.rank = 0; self.disk = None; self.rows_max = 2048; self.ops = []
        self.carry = getattr(multi, "carry_stats", dict)()
        self.arena = SimpleNamespace(copy=lambda a, b, n: None)
        self.lanes = Lanes()
    def _emit(self, op, payload): self.ops.append((op, list(payload))); return []
    def _move(self, x, base, size):
        return multi.MultiDecoder._move(self, x, base, size)

class Lanes(dict):
    """The lanes by sid; a live stream the test did not make a lane for reads as one at position 0."""
    def __missing__(self, sid): return SimpleNamespace(sid=sid, twin=None, st=St(), extent=None, decoding=False)

class St:
    def __init__(self, pos=0): self.twin_base = None; self.twin_size = None; self.pos = pos; self.binds = []
    def bind(self, b, n): self.binds.append((b, n))

def twin(s, base, n, lane_x, pos=A, sid=1):
    """A twin of the lane filling in ``lane_x``: an extent (flagged like Pool.add(twin=True), so the same on a tree whose
    add does not take it) that the lane's state points at."""
    t = s.pool.add(base, n); t.twin = True
    st = St(pos); st.twin_base = base; st.twin_size = n
    lane = SimpleNamespace(sid=sid, twin=[t], st=st, extent=lane_x, decoding=False)
    s.lanes[sid] = lane
    return t, lane

def twin_pinned():
    # the reviewer's scenario: part 1 = free 8A | twin of a sliced fill 8A | free 8A | kept 8A; a placement of 10A
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])
    k = s.keep("k", H + 24 * A, 8 * A)
    base = s._room(10 * A, half=1)
    check("9 twin: compaction leaves a twin where it is", t.base == H + 8 * A and lane.st.twin_base == t.base)
    check("9 twin: no extent overlaps a live twin",
          all(y is t or y.end <= t.base or y.base >= t.end for y in s.pool.extents))
    check("9 twin: the room came from evicting the kept prompt (gathered rows are around the twin)",
          base is not None and base == H + 16 * A and s.evicted == ["k"])
    check("9 twin: the twin was not revoked mid-fill", lane.twin is not None and s.twins["revoked"] == 0)
section("9 twin", twin_pinned)

def twin_not_room():
    # free 8A | twin 8A | free 8A: 16A free but never together: not gatherable, no victims, nothing moves
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])
    s.pool.add(H + 24 * A, 8 * A).owner = 9          # part 1: free 8A | twin 8A | free 8A | a live stream 8A
    check("9 twin: its rows are not gatherable room (16A free, 8A together at most)", s._gathered(12 * A, 1) is False)
    check("9 twin: no eviction can make room around it", s._eviction_cost(12 * A, (), 1) is None)
    check("9 twin: _room finds none and moves nothing", s._room(12 * A, half=1) is None and t.base == H + 8 * A
          and not [op for op, _ in s.ops if op == multi.MOVE])
section("9 twin room", twin_not_room)

def twin_move_refused():
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 8 * A, 8 * A, x)
    for what, fn in (("pool.move", lambda: s.pool.move(t, H, 8 * A)), ("_move", lambda: multi.MultiDecoder._move(s, t, H, 8 * A))):
        try:
            fn(); ok = False
        except (ValueError, RuntimeError):
            ok = True
        check(f"9 twin: {what} of a twin is refused", ok and t.base == H + 8 * A)
section("9 twin move", twin_move_refused)

def twin_random():
    rnd = random.Random(11); bad = []
    for trial in range(400):
        s = Real(2 * H, H)
        keep_n = 0
        # part 0: a filling stream; part 1: its twin among kept prompts and a live stream
        x = s.pool.add(0, 8 * A); x.owner = 1
        at = H; items = []
        while True:
            n = rnd.randrange(1, 7) * A
            if at + n > 2 * H: break
            kind = rnd.choice(["free", "kept", "kept", "live", "twin"])
            if kind == "twin" and not items.count("twin"):
                t, lane = twin(s, at, n, x); items.append("twin"); s.twin_obj = t
            elif kind == "kept": s.keep(f"k{keep_n}", at, n); keep_n += 1; items.append("kept")
            elif kind == "live": s.pool.add(at, n).owner = 100 + len(items); items.append("live")
            at += n + rnd.choice([0, 0, A])
        if "twin" not in items: continue
        t = s.twin_obj
        lane = s.lanes[1]
        # the lane's fill is pending (twin kept) in half the trials: the other half it may be revoked, then rows vanish
        pend = rnd.random() < 0.5
        s.partial = SimpleNamespace(contains=lambda sid: pend and sid == 1, group=[(lane, 8 * A)]) if pend else None
        base0 = t.base
        rnd.shuffle(s.kept)
        need = rnd.randrange(1, 20) * A
        ev0 = len(s.evicted)
        r = s._room(need, half=1) if rnd.random() < 0.5 else s._room(need)
        if lane.twin is not None and t.base != base0: bad.append((trial, "twin moved")); continue
        if lane.twin is not None and any(y is not t and y.base < t.end and t.base < y.end for y in s.pool.extents):
            bad.append((trial, "overlap")); continue
        if r is None and len(s.evicted) != ev0: bad.append((trial, "evicted for nothing")); continue
        if r is not None:
            try: s.pool.add(r, need)
            except ValueError: bad.append((trial, "placed on taken rows"))
    check(f"9 twin: random layouts: twin never moves or overlaps, no eviction for nothing ({bad[:2]})", not bad)
section("9 twin random", twin_random)

def no_part():
    # two parts of 32A, each a 24A live stream and an 8A kept prompt; 16A wanted (the reviewer's repro 2)
    for half in (None, 1):
        s = Real(2 * H, H)
        s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2
        s.keep("k0", 24 * A, 8 * A); s.keep("k1", H + 24 * A, 8 * A)
        r = s._room(16 * A, half=half)
        check(f"9 room (half={half}): none when no part can make it, and no kept prompt is evicted",
              r is None and s.evicted == [] and len(s.kept) == 2)
        check(f"9 room (half={half}): the refusal is counted", s.no_part == 1)
    s = Real(2 * H, H)
    s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2
    s.keep("k0", 24 * A, 8 * A); s.keep("k1", H + 24 * A, 8 * A)
    r = s._room(8 * A)
    check("9 room: still evicts the least needed when a part can (8A: one prompt)", r is not None and len(s.evicted) == 1)
section("9 room", no_part)

def fits_parts():
    def lanes(s):
        for sid, x in enumerate([e for e in s.pool.extents if e.owner is not None], 1):
            s.lanes[sid] = SimpleNamespace(sid=sid, twin=None, st=St(), extent=x, decoding=False)
    need_prompt = 16 * A - 2048                  # _need = 16A
    # (a) 12A free in each part (24A together): fits() used to say yes
    s = Real(2 * H, H)
    s.pool.add(0, 20 * A).owner = 1; s.pool.add(H, 20 * A).owner = 2; lanes(s)
    check("9 fits: free rows spread over parts are not room for one prompt", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is False)
    # (b) a kept prompt makes the room in one part
    s = Real(2 * H, H)
    s.pool.add(0, 16 * A).owner = 1; s.pool.add(H, 20 * A).owner = 2; lanes(s)
    s.keep("k", 16 * A, 8 * A)
    check("9 fits: free + kept rows of ONE part cover it", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is True)
    # (c) twin rows are not room
    s = Real(2 * H, H)
    x = s.pool.add(0, 20 * A); x.owner = 1
    t, lane = twin(s, H + 12 * A, 20 * A, x)
    s.pool.add(H, 12 * A).owner = 2; lanes(s); s.lanes[1] = lane
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 20 * A)])      # (its fill is in flight)
    check("9 fits: a pinned twin's rows are not evictable room", s.fits(SimpleNamespace(prompt=[0] * need_prompt)) is False)
    # (d) decoding streams in the part need headroom
    s = Real(2 * H, H)
    s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 24 * A).owner = 2; lanes(s)
    s.keep("k", H + 24 * A, 8 * A)
    for l in s.lanes.values(): l.decoding = True
    check("9 fits: headroom for the part's decoding stream counts (8A of room: a prompt needing 8A - 2048 fits, 8A does not)",
          s.fits(SimpleNamespace(prompt=[0] * (8 * A - 4096))) is True and s.fits(SimpleNamespace(prompt=[0] * (8 * A - 2048))) is False)
    # (e) unsharded unchanged
    s = Real(2 * H, 0)
    s.pool.add(0, 40 * A).owner = 1; lanes(s); s.keep("k", 40 * A, 8 * A)
    check("9 fits: one pool as before", s.fits(SimpleNamespace(prompt=[0] * (20 * A))) is True
          and s.fits(SimpleNamespace(prompt=[0] * (30 * A))) is False)
section("9 fits", fits_parts)

def fits_revocable_twin():
    # the re-review's case: a 30A prompt between chunks of its fill (no slice in flight) in part 0, its 30A twin in
    # part 1. The twin can be given back, so a 10A request fits (the scheduler holds a request fits() refuses and
    # nothing calls _room, so a twin counted as pinned would hold it for the other prompt's whole fill)
    def build(pending):
        s = Real(2 * H, H)
        x = s.pool.add(0, 30 * A); x.owner = 1
        t, lane = twin(s, H, 30 * A, x)
        if pending:
            s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 30 * A)])
        return s, x, t, lane
    prompt = SimpleNamespace(prompt=[0] * (8 * A))
    s, x, t, lane = build(False)
    check("9 fits: a twin between chunks is room (it is given back)", s.fits(prompt) is True)
    if s._shard_health is not None:
        check("9 health: its rows count as gatherable room", s._shard_health()["parts"][1]["gatherable"] == H)
    base = s._room(s._need(8 * A))
    check("9 fits: and _room does place it, after one UNTWIN", base is not None and lane.twin is None
          and [op for op, _ in s.ops].count(multi.UNTWIN) == 1)
    s, x, t, lane = build(True)
    check("9 fits: a twin of the fill part way through its layers is not room", s.fits(prompt) is False)
    check("9 fits: and _room refuses it too, keeping the twin", s._room(s._need(8 * A)) is None and lane.twin is not None)
section("9 fits revocable twin", fits_revocable_twin)

def grow_twin():
    s = Real(2 * H, H)
    x = s.pool.add(0, 8 * A); x.owner = 1
    t, lane = twin(s, H + 16 * A, 8 * A, x)
    g = s.pool.add(H + 4 * A, 4 * A); g.owner = 7          # free 4A | g 4A | free 8A | twin | free 8A
    s.pool.add(H + 12 * A, 2 * A).owner = 8
    s.partial = SimpleNamespace(contains=lambda sid: sid == 1, group=[(lane, 8 * A)])      # (the twin stays)
    ok = s._grow(g, 14 * A)
    check("9 grow: around a pinned twin the segment's free rows gather after the extent",
          ok and g.size == 14 * A and t.base == H + 16 * A and
          all(y is t or y.end <= t.base or y.base >= t.end for y in s.pool.extents))
    check("9 grow: beyond the segment's rows it is refused, the twin still where it was",
          s._grow(g, 20 * A) is False and t.base == H + 16 * A)
section("9 grow", grow_twin)

def pool_segments():
    from tensorfold.families.glm5_next.cuda.pool import Pool as P
    p = P(2 * H, half=H)
    t = p.add(8 * A, 8 * A); t.twin = True
    p.add(H, 16 * A)
    check("9 pool: segments cut at a twin and at the part", p.segments(0) == [(0, 8 * A), (16 * A, H)])
    check("9 pool: compactable is the largest segment, not the sum", p.compactable(0) == H - 16 * A and p.compactable(1) == H - 16 * A)
section("9 pool", pool_segments)

def tp3_exchange():
    heads = [22, 21, 21]; R = 5; LW = 8; N = 3
    torch.manual_seed(0)
    truth = torch.randn(R, sum(heads), LW).bfloat16()
    owners = torch.tensor([0, 1, 2, 1, 0])
    bufs, oalls, sends, recvs = [], [], [], []
    for r in range(N):
        shard.RANK = r
        hp = shard.head_plan(heads, r)
        send, recv = shard.exchange_buffers(hp, R, LW, "cpu")
        b = SimpleNamespace(heads=heads, offs=hp.offs, peers=hp.peers, send=send, recv=recv)
        oall = torch.full_like(truth, float("nan"))                      # only the rows it owns are computed
        for i in range(R):
            if int(owners[i]) == r: oall[i] = truth[i]
        bufs.append(b); oalls.append(oall); sends.append(shard.pack_heads(b, oall, R))
    # the exchange_all contract (comm.py): each pair's send and receive alike in numel and dtype, contiguous
    for r in range(N):
        for p in bufs[r].peers:
            s_, r_ = sends[r][0][p][0], sends[p][1][r][0]
            if s_.numel() != r_.numel() or s_.dtype != r_.dtype or not (s_.is_contiguous() and r_.is_contiguous()):
                raise ValueError(f"exchange_all: rank {r} -> {p} sends {tuple(s_.shape)}, rank {p} receives {tuple(r_.shape)}")
            r_.copy_(s_)
    good = True
    for r in range(N):
        shard.RANK = r
        out = torch.empty(R, heads[r], LW, dtype=torch.bfloat16)
        shard.unpack_heads(bufs[r], oalls[r], owners, out, R)
        want = truth[:, bufs[r].offs[r]:bufs[r].offs[r] + heads[r]]
        good &= bool(torch.equal(out, want))
    check("9 tp3: the decode exchange of uneven heads (22/21/21): pairs alike, every rank gets its own heads", good)
    shard.RANK = 0
section("9 tp3", tp3_exchange)

def tp3_scratch():
    # the buffers the decode window really allocates (_scratch_n): comm.exchange_all refuses, on each rank, a peer whose
    # send and receive differ in size, and at 22/21/21 heads they did (rank 0 sent 21 heads' worth, expected 22)
    from tensorfold.families.glm5_next.cuda import latent
    saved = (torch.cuda.is_current_stream_capturing, latent.LatentScratch, shard.WORLD, shard.RANK, shard.HEADS)
    torch.cuda.is_current_stream_capturing = lambda: False
    latent.LatentScratch = lambda *a, **k: None
    try:
        shard.WORLD, shard.HEADS = 3, [22, 21, 21]
        bad = []
        for r in range(3):
            shard.RANK = r; shard._buf.clear()
            b = shard._scratch_n(8, [22, 21, 21][r], 16, "cpu", 4)
            for p in b.peers:
                if b.send[p].numel() != b.recv[p].numel():
                    bad.append((r, p, b.send[p].numel(), b.recv[p].numel()))
        check(f"9 tp3: every rank's send and receive alike per peer, as exchange_all demands ({bad[:2]})", not bad)
    finally:
        shard._buf.clear()
        torch.cuda.is_current_stream_capturing, latent.LatentScratch, shard.WORLD, shard.RANK, shard.HEADS = saved
section("9 tp3 scratch", tp3_scratch)

def pool_agree():
    # two ranks' unsharded pools differ by a few extents (their KDA state sizes): the ranks take the smallest
    a, b = 2_000_000 // 2048 * 2048, 2_000_000 // 2048 * 2048 - 3 * 2048
    halves = []
    for rows in (a, b):
        shard.setup(0, 3, shard.budget(rows, 11, "fp8", 3), heads=[22, 21, 21]); halves.append(shard.HALF)
    check("9 agree: ranks sizing alone can disagree on a part's rows (the failure)", halves[0] != halves[1])
    p, d = shard.agree([[a, 1000], [b, 500], [a, 700]])
    check("9 agree: the smallest pool and display rows on every rank", (p, d) == (b, 500))
    halves = []
    for r in range(3):
        shard.setup(r, 3, shard.budget(p, 11, "fp8", 3), heads=[22, 21, 21]); halves.append(shard.HALF)
    check("9 agree: then every rank's part is the same", len(set(halves)) == 1)
    shard.setup(0, 2, 4096 * 512)
section("9 agree", pool_agree)

def bf16_bytes():
    lc = torch.randn(64, 512).bfloat16()
    lcb = shard.as_bytes(lc)
    check("9 bf16: a bf16 plane's rows are 1,024 bytes, a uint8 (FP8) plane's unchanged",
          lcb.shape == (64, 1024) and lcb.dtype == torch.uint8 and shard.as_bytes(lcb) is lcb and shard.as_bytes(None) is None)
    hist = torch.tensor([3, 5, 9, 40])
    buf = torch.empty((8, lcb.shape[1]), dtype=torch.uint8)
    torch.index_select(lcb, 0, hist, out=buf[:4])
    old = shard.DTYPE
    shard.DTYPE = torch.bfloat16
    got = shard.as_cache(buf[:6])
    check("9 bf16: the received bytes viewed back as the plane's rows", got.dtype == torch.bfloat16 and torch.equal(got[:4], lc[hist]))
    shard.DTYPE = torch.uint8
    check("9 fp8: received rows stay bytes", shard.as_cache(buf).dtype == torch.uint8)
    shard.DTYPE = old
section("9 bf16", bf16_bytes)

def floor_budget():
    # the window a request may use is what a part holds (limit = part - max_rows, the engine's own cap), noted not refused
    # unless a part would be absurdly small
    cap = lim = 1_048_576
    for N, unsh in ((2, 1_118_208), (2, 1_060_864), (3, 1_060_864), (4, 1_060_864)):
        have = shard.budget(unsh, 11, "fp8", N)
        part = have // N
        rows, note = shard.plan(unsh, 11, "fp8", lim, N, 16)
        want = f"up to {min(lim, part - 16)} tokens"
        check(f"9 floor N={N} pool {unsh}: a part of {part} short of the {lim} window: shrink, one note with the numbers, "
              f"never past the budget", rows == have and rows <= shard.budget(unsh, 11, "fp8", N) and part - 16 < lim
              and want in note and str(part) in note and str(lim) in note and "\n" not in note)
    rows, note = shard.plan(4_000_000, 11, "fp8", lim, 2, 16)
    check("9 floor: a budget that holds the windows is untouched", rows == shard.budget(4_000_000, 11, "fp8", 2) and not note)
    for N, unsh, what in ((2, 100_000, "a part of ~94K"), (2, 250_000, "a part below a quarter of the window"),
                          (4, 150_000, "a part of ~140K at four ranks")):
        have = shard.budget(unsh, 11, "fp8", N)
        try:
            shard.plan(unsh, 11, "fp8", lim, N, 16); refused = False
        except ValueError as exc:
            refused = str(have) in str(exc) and str(have // N) in str(exc) and str(lim) in str(exc)
        check(f"9 floor N={N}: {what} ({have // N} tokens) is refused with the numbers", refused)
    rows, note = shard.plan(1_060_864, 11, "fp8", 262_144, 2, 16)
    check("9 floor: a smaller window that a part holds is untouched", not note)
section("9 floor", floor_budget)

def refusals():
    check("9 refuse: serial verify with the shard says why", "serial" in (shard.unsupported("serial") or ""))
    check("9 refuse: batched verify is fine", shard.unsupported("batched") is None)
section("9 refuse", refusals)

def hardening():
    from tensorfold.families.glm5_next.cuda import forward, segments
    shard.ON = True
    try:
        shard.setup(0, 2, 4096 * 512)
        # Caches.bind: a twin the extent's size
        ca = SimpleNamespace(arena=SimpleNamespace(view=lambda i, b, n: (i, b, n)), kc=[0], vc=[None], mtp=None, index=None)
        st = St(); st.twin_base = shard.HALF + 2048; st.twin_size = 4096
        try:
            forward.Caches.bind(ca, st, 0, 8192); ok = False
        except ValueError:
            ok = True
        check("9 bind: a twin of another size than its extent is refused", ok)
        forward.Caches.bind(ca, st, 0, 4096)
        check("9 bind: the same size binds", True)
        # SegRows.fill: a window past the trash rows
        sr = object.__new__(segments.SegRows); sr.rows, sr.max_segs, sr.ring = shard.TRASH + 64, 4, None
        try:
            sr.fill([(0, 0, shard.TRASH + 1)], sparse_from=None); ok = False
        except ValueError:
            ok = True
        check("9 window: more rows than the trash rows hold is refused", ok)
        try:
            shard._broadcast(None, None, None, 0, 8, "cpu", 0, -1, 1, 0); ok = False
        except ValueError:
            ok = True
        check("9 broadcast: no owner (root -1) is refused", ok)
        s = Real(2 * H, H)
        try:
            s._untwin(SimpleNamespace(sid=4, twin=None, st=St(), extent=None)); ok = False
        except RuntimeError:
            ok = True
        check("9 untwin: a lane with no twin is refused", ok)
        h = Real(2 * H, H)
        x = h.pool.add(0, 8 * A); x.owner = 1; t, lane = twin(h, H, 8 * A, x); h.pool.add(H + 8 * A, 4 * A); h.keep("q", H + 16 * A, 4 * A)
        hh = h._shard_health()
        check("9 health: per-part rows (free, twin, kept, live) and the fetch counters",
              hh["parts"][1]["twin_rows"] == 8 * A and hh["parts"][1]["kept_rows"] == 4 * A + 4 * A and "fetches" in hh["fetch"])
    finally:
        shard.ON = False
        shard.setup(0, 2, 4096 * 512)
section("9 hardening", hardening)

def signature():
    import types, tensorfold.cuda.spill as spill, tensorfold.cuda.spill_blocks as sb
    got = {}
    class Store:
        def __init__(self, cfg, **kw): got["sig"] = kw["signature"]
        def ring(self): pass
    saved = (spill.build_id, spill.weights_id, sb.BlockStore)
    spill.build_id = lambda *a: "b"; spill.weights_id = lambda *a: "w"; sb.BlockStore = Store
    try:
        def make(sharded):
            shard.ON = sharded
            d = SimpleNamespace(
                g=SimpleNamespace(spill_cfg=object(), drafter=None, model_dir="m", drafter_dir=None, agreed=[1, 2],
                                  _gather_ints=None),
                arena=SimpleNamespace(planes=[SimpleNamespace(tensor=torch.zeros(4, 512, dtype=torch.bfloat16), div=1, pad=0)]),
                grid=0, drafts=None, w=SimpleNamespace(world=2, device="cpu"), rank=0, pool=Pool(2 * H, half=H if sharded else 0),
                _guard_pool=lambda: None)
            multi.MultiDecoder._spill_store(d)
            return got["sig"]
        a, b = make(False), make(True)
    finally:
        shard.ON = False
        spill.build_id, spill.weights_id, sb.BlockStore = saved
    check("9 spill: sharded and unsharded runs have different stores", a != b)
    check("9 spill: an unsharded run's signature is unchanged (no kvshard marker)", "kvshard" not in a and "kvshard" in b)
section("9 spill", signature)

# ---- 10 : the cross-part carry (TF_GLM_KV_SHARD_CARRY): decisions, the ops and their order ------------------------------
# A hit used to be forced into its state's part: a full part dropped it (the prompt prefilled again) or evicted other
# conversations' states, while the other part sat empty. Now rank 0 may carry the rows into another part
# (``Arena.carry``): ADMIT copy=2, XMOVE for a live stream. Only to avoid a real cost: never to even the parts out, never
# when the hit's part fits the request without evicting anything. These are the decisions (``_place_hit``, ``_relocate``),
# the ops and their order around the collective, the follower's side, and what the old code decided.
import collections
import contextlib
from tensorfold.families.glm5_next.cuda import pool as poolmod
from tensorfold.families.glm5_next.cuda.pool import Arena, Plane
from tensorfold.engine import grammar as _grammar  # noqa: F401  (imported by admit)


@contextlib.contextmanager
def carry_on(on=True):
    saved = multi.KV_CARRY
    multi.KV_CARRY = on
    try:
        yield
    finally:
        multi.KV_CARRY = saved


def _mm(name):
    return getattr(multi.MultiDecoder, name, None)


class FakeArena:
    """What the decoder asks of its arena, recorded in the decoder's event list (its order against the ops matters)."""
    def __init__(self, ev): self.ev = ev; self.planes = [SimpleNamespace(div=4)]
    def copy(self, a, b, n): self.ev.append(("copy", a, b, n))
    def carry(self, a, b, n, comm): self.ev.append(("carry", a, b, n))
    def sharded_bytes(self, n): return n * 5808


class Car(Real):
    """Real + the carry's decisions and ops, with the events (ops emitted, syncs, evictions, copies, carries) in order."""
    _place_hit = _mm("_place_hit"); _place = _mm("_place"); _pins = _mm("_pins"); _pressed = _mm("_pressed")
    _fits_unevicted = _mm("_fits_unevicted"); _placeable = _mm("_placeable")
    _refused = _mm("_refused"); _room_free = _mm("_room_free"); _free_elsewhere = _mm("_free_elsewhere")
    _grow_fits = _mm("_grow_fits"); _rescue = _mm("_rescue"); _relocate = _mm("_relocate"); _xmove = _mm("_xmove")
    _legal = _mm("_legal"); _move_to = _mm("_move_to"); _move_lane = _mm("_move_lane")
    _carry = _mm("_carry"); _recent = _mm("_recent"); _touch = _mm("_touch")
    _guard_pool = _mm("_guard_pool"); _release_rows = _mm("_release_rows")

    def __init__(self, rows, half):
        super().__init__(rows, half)
        self.carry = multi.carry_stats(); self.tune = SimpleNamespace(lone=False); self.events = []
        self.arena = FakeArena(self.events); self.g = SimpleNamespace(comm=None); self.partial = None
        self.w = SimpleNamespace(world=2); self.xticks = 0; self.entered = {}; self.carry_kind = None
    def _emit(self, op, payload): self.events.append(("emit", op)); return super()._emit(op, payload)
    def _sync(self): self.events.append(("sync",))
    def _evict(self, c, spill=True): self.events.append(("evict", c.name)); return super()._evict(c, spill)


def place(s, hit, need, key):
    """_place_hit, and the count ``admit`` makes once the ADMIT is queued."""
    base, copy, h = s._place_hit(hit, need, key)
    if copy == 2 and s.carry_kind:
        s.carry[s.carry_kind] += 1
    return base, copy, h


def put_state(s, x, cid, n, own=False, name=None):
    """A kept state of extent x: n tokens of conversation ``cid`` (its ids and key are [cid] * n, so a prompt of the same
    conversation extends it and another's does not)."""
    c = SimpleNamespace(name=name or f"s{len(s.kept)}", ids=[cid] * n, key=[cid] * n, extent=x, own=own, shared=False,
                        kid=len(s.kept), chat=cid, images=False, head=None, drafter_end=-1, drafter_rows=None)
    x.kept.append(c); s.kept.append(c)
    return c


def lane_of(s, x, sid, pos=None, **kw):
    """A decoding stream in extent x (``pos`` tokens in; default: the extent's size)."""
    lane = SimpleNamespace(sid=sid, twin=None, st=St(x.size if pos is None else pos), extent=x, decoding=True, xwhy=None,
                           paused=False, s=SimpleNamespace(done=False, count=10 ** 9, out=[], emit=None), depth=0, head=None,
                           moved_at=-(1 << 30), came_from=-1, **kw)
    s.lanes[sid] = lane
    x.owner = sid
    return lane


def rescue_layout():
    """Part 0: a live stream (24A) and the kept hit (8A): full. Part 1 empty."""
    s = Car(2 * H, H)
    s.pool.add(0, 24 * A).owner = 1
    x = s.pool.add(24 * A, 8 * A)
    return s, x, put_state(s, x, 7, 8 * A, own=True)


def pressure_layout(live1=0, hit_n=8 * A):
    """Part 0: a live stream (12A), the hit's extent (8A), another conversation's kept prompt y (8A) and 4A free: a request
    of 14A fits only by evicting y. Part 1: ``live1`` rows of a live stream."""
    s = Car(2 * H, H)
    s.pool.add(0, 12 * A).owner = 1
    x = s.pool.add(12 * A, 8 * A)
    hit = put_state(s, x, 7, hit_n, own=True, name="hit")
    y = put_state(s, s.pool.add(20 * A, 8 * A), 3, 8 * A, name="y")
    if live1:
        s.pool.add(H, live1 * A).owner = 2
    return s, x, hit, y


def rescue_cases():
    with carry_on():
        s, x, hit = rescue_layout()
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 rescue: a full part no longer drops the hit: it is carried into the part with room",
              copy == 2 and h is hit and s.pool.half_of(base) == 1 and s.evicted == [] and hit in s.kept)
        check("10 rescue: the hit's extent and kept state stay (a carry copies)", x in s.pool.extents and x.kept == [hit])
        check("10 rescue: counted as rescued, and it is not a refusal (no_part back where it was)",
              s.carry["rescued"] == 1 and s.carry["pressure"] == s.carry["others"] == 0 and s.no_part == 0)
        check("10 rescue: nothing moved or evicted to make the room", not [e for e in s.events if e[0] in ("evict", "emit")])
        # the old code's refusal count survives a rescue that is refused: a live stream's extent holding a small hit, its part
        # full (the copy beside it is refused: counted), the hit under the minimum (no rescue), the request placed elsewhere
        for on in (True, False):
            with carry_on(on):
                s = Car(2 * H, H)
                s.pool.add(0, 24 * A).owner = 1
                x = s.pool.add(24 * A, 8 * A); hit = put_state(s, x, 7, 1024); x.owner = 3
                yy = put_state(s, s.pool.add(H, 32 * A), 3, 32 * A, name="yy")
                base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
                check(f"10 rescue (carry {int(on)}): a hit under the minimum in a live stream's extent whose part is full is dropped, "
                      f"the other part's kept prompt evicted, and the refusal of the part's own try counted once, as before",
                      h is None and s.evicted == ["yy"] and s.no_part == 1)
        # the other part has the room, but not a block for each stream it holds and the arriving one: free, not roomy
        s, x, hit = rescue_layout()
        s.pool.add(H, 16 * A).owner = 2                                 # (the 16A would fit, its headroom would not)
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 rescue: no headroom in the free rows of the other part: the pressure rule leaves it alone, the rescue "
              "(evicting there as the drop would have) still keeps the hit", copy == 2 and s.pool.half_of(base) == 1 and s.carry["rescued"] == 1)
        # the other part is full of kept prompts: the oldest go (as dropping the hit would have evicted them), the hit stays
        s, x, hit = rescue_layout()
        s.kept = []
        q0 = put_state(s, s.pool.add(H, 16 * A), 3, 16 * A, name="q0"); q1 = put_state(s, s.pool.add(H + 16 * A, 16 * A), 4, 16 * A, name="q1")
        s.kept.append(hit)
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 rescue: a part full of kept prompts gives up its oldest, the hit stays and its rows travel",
              copy == 2 and h is hit and base == H and s.evicted == ["q0"] and q1 in s.kept and s.carry["rescued"] == 1)
        # a grow that cannot work evicts nothing: part 0 holds the hit's extent (4A), another kept prompt (4A), a live
        # stream (20A) and 4A free; 16A wanted. Growing the extent in place would evict the other prompt and still fall
        # short (12A), so the hit is carried instead and the other conversation keeps its state
        s = Car(2 * H, H)
        s.pool.add(0, 20 * A).owner = 1
        x = s.pool.add(20 * A, 4 * A); hit = put_state(s, x, 7, 4 * A, own=True, name="hit")
        y = put_state(s, s.pool.add(24 * A, 4 * A), 3, 4 * A, name="y")
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 rescue: a grow that cannot work evicts no other conversation's state before the hit is carried",
              copy == 2 and h is hit and s.evicted == [] and y in s.kept and s.pool.half_of(base) == 1)
        # a live stream's extent holding the hit (a shared prefix its stream wrote): the copy needs room beside it
        s = Car(2 * H, H)
        s.pool.add(0, 24 * A).owner = 1
        x = s.pool.add(24 * A, 8 * A); hit = put_state(s, x, 7, 8 * A); x.owner = 3
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 rescue: also when the hit lies in a live stream's extent (the copy has no room beside it)",
              copy == 2 and h is hit and s.pool.half_of(base) == 1 and s.carry["rescued"] == 1 and s.evicted == [])
        # nothing anywhere: NoRoom, and the counters are what the old code left
        for on in (True, False):
            with carry_on(on):
                s = Car(2 * H, H)
                s.pool.add(0, 24 * A).owner = 1; s.pool.add(H, 32 * A).owner = 2
                x = s.pool.add(24 * A, 8 * A); hit = put_state(s, x, 7, 8 * A)
                try:
                    place(s, hit, 16 * A, [7] * (20 * A)); raised = False
                except multi.NoRoom:
                    raised = True
                check(f"10 rescue (carry {int(on)}): no room in any part is NoRoom, the refusal counted once, nothing evicted",
                      raised and s.no_part == 1 and s.evicted == [])
    # carry off: the old drop, exactly
    with carry_on(False):
        s, x, hit = rescue_layout()
        base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
        check("10 carry off: the hit is dropped as before (a fresh prompt in the other part)",
              h is None and copy == 0 and s.pool.half_of(base) == 1 and s.no_part == 0 and s.carry["hits"] == 0
              and not s.carry["rescued"])
section("10 rescue", rescue_cases)


def lane_twin(s, x, sid=5):
    """x's stream (a lane filling in it) holds a twin in part 1."""
    t, lane = twin(s, H + 24 * A, 8 * A, x, sid=sid)
    x.owner = sid
    return lane


def pins_cases():
    pins = (("loading", lambda s, x: setattr(x, "loading", object())),
            ("twin", lambda s, x: setattr(x, "twin", True)),
            ("twin", lambda s, x: lane_twin(s, x)),
            ("partial", lambda s, x: (lane_of(s, x, 5), setattr(s, "partial", SimpleNamespace(contains=lambda sid: sid == 5, group=[])))))
    for why, setup in pins:
        with carry_on():
            s, x, hit = rescue_layout()
            setup(s, x)
            base, copy, h = place(s, hit, 16 * A, [7] * (20 * A))
            check(f"10 pins: rows pinned ({why}) are not carried (the rescue): the hit is dropped as before, refusal counted once",
                  h is None and copy == 0 and s.carry["refused"][why] == 1 and s.carry["hits"] == s.carry["rescued"] == 0
                  and not [e for e in s.events if e[0] == "carry"])
        with carry_on():
            s, x, hit, y = pressure_layout()
            setup(s, x)
            base, copy, h = place(s, hit, 14 * A, [7] * (20 * A))
            check(f"10 pins: rows pinned ({why}) are not carried (the pressure rule): the request is placed in its part as before",
                  copy != 2 and s.carry["refused"][why] == 1 and s.carry["hits"] == s.carry["pressure"] == 0)
    with carry_on():                       # a pinned twin: the fill part way through its layers
        s, x, hit = rescue_layout()
        x.twin = True
        lane = SimpleNamespace(sid=8, twin=[x], st=St(), extent=None, decoding=False)
        s.lanes[8] = lane
        s.partial = SimpleNamespace(contains=lambda sid: sid == 8, group=[(lane, A)])
        check("10 pins: a pinned twin extent says pinned (before twin)", s._pins(x) == "pinned")
        x.twin = False; s.partial = None
        check("10 pins: nothing pins an ordinary kept extent", s._pins(x) is None)
section("10 pins", pins_cases)


def pressure_cases():
    key = [7] * (20 * A)
    need = 14 * A
    with carry_on():
        s, x, hit, y = pressure_layout()
        check("10 pressure: the setup: its part would evict y to place the request, and could (so it is no rescue)",
              not s._fits_unevicted(hit, need, []) and s._placeable(hit, need))
        base, copy, h = place(s, hit, need, key)
        check("10 pressure: fitting the request would evict a kept prompt and the other part has the room free: the hit is "
              "carried there, y keeps its state", copy == 2 and h is hit and s.pool.half_of(base) == 1 and s.evicted == [] and y in s.kept)
        check("10 pressure: counted as pressure only; nothing moved in the other part", s.carry["pressure"] == 1
              and s.carry["rescued"] == s.carry["others"] == 0 and not [e for e in s.events if e[0] == "emit"])
        # the same pool, the other part without headroom for its streams (free exactly the request): evict as before
        s, x, hit, y = pressure_layout(live1=18)
        base, copy, h = place(s, hit, need, key)
        check("10 pressure: free rows without the headroom (a block for each stream there and the one arriving) are no room: "
              "no carry, y is evicted and the extent grows in place as before",
              copy == 0 and base == x.base and s.evicted == ["y"] and s.carry["hits"] == 0 and s.carry["refused"]["no_target"] == 1)
        s, x, hit, y = pressure_layout(live1=17)                        # (1A more free than the request: its headroom, a block each
        base, copy, h = place(s, hit, need, key)                        # for the stream there and the arriving one, is 2 blocks = A)
        check("10 pressure: ... and with the headroom free it carries", copy == 2 and s.pool.half_of(base) == 1 and s.evicted == [])
        # P fits without evicting anything: never, however much emptier the other part is
        s = Car(2 * H, H)
        s.pool.add(0, 12 * A).owner = 1
        x = s.pool.add(12 * A, 8 * A); hit = put_state(s, x, 7, 8 * A, own=True)
        base, copy, h = place(s, hit, need, key)
        check("10 no pressure: its part fits the request without evicting (12A of live streams, 20A of room): no carry, the "
              "extent grows in place, though the other part is empty",
              copy == 0 and base == x.base and s.carry["hits"] == 0 and not any(s.carry["refused"].values()))
        s = Car(2 * H, H)                                                 # ...and with live streams all on one side
        for i in range(5):
            s.pool.add(i * 4 * A, 4 * A).owner = 10 + i
        x = s.pool.add(20 * A, 8 * A); hit = put_state(s, x, 7, 8 * A, own=True)
        base, copy, h = place(s, hit, 10 * A, key)
        check("10 no pressure: 5 live streams against none is no reason (decode spread is not worth a move)",
              copy == 0 and s.carry["hits"] == 0 and not any(s.carry["refused"].values()))
        # free rows that are not together: compacting gathers them, no eviction, no carry
        s = Car(2 * H, H)
        s.pool.add(0, 4 * A).owner = 1; s.pool.add(8 * A, 4 * A).owner = 2
        x = s.pool.add(16 * A, 4 * A); hit = put_state(s, x, 7, 4 * A, own=True); x.owner = 3
        base, copy, h = place(s, hit, 10 * A, key)
        check("10 no pressure: scattered free rows of its own part (gathered by compacting) fit it: no carry",
              copy == 1 and s.pool.half_of(base) == 0 and s.carry["hits"] == 0 and s.evicted == [])
        # a hit under the minimum is never carried: placed as before
        s, x, hit, y = pressure_layout(hit_n=1024)
        base, copy, h = place(s, hit, need, key)
        check("10 pressure: a hit under 2,048 tokens is never carried: refused as small, y evicted as before",
              copy == 0 and s.carry["refused"]["small"] == 1 and s.carry["hits"] == 0 and s.evicted == ["y"])
        s, x, hit, y = pressure_layout(hit_n=2048)
        check("10 pressure: a hit of exactly 2,048 tokens carries", place(s, hit, need, key)[1] == 2)
        # the rescue and the others branch have the minimum too
        s, x, hit = rescue_layout()
        s.kept[0].ids = s.kept[0].key = [7] * 1024
        base, copy, h = place(s, hit, 16 * A, key)
        check("10 rescue: a hit under 2,048 tokens is dropped as before (cheaper to recompute than to carry)",
              h is None and s.carry["refused"]["small"] == 1)
        # no ping-pong: the conversation was carried to part 1 because part 0 lacked room; its next turn, kept there, grows
        # there (room), and is not carried back while part 0 still lacks room
        s, x, hit = rescue_layout()
        base, copy, h = place(s, hit, 16 * A, key)
        nx = s.pool.add(base, 16 * A)                                     # the stream's extent, then its kept state (turn 2)
        turn2 = put_state(s, nx, 7, 12 * A, own=True, name="turn2")
        b2, c2, _ = place(s, turn2, 20 * A, [7] * (24 * A))
        check("10 no ping-pong: the next turn grows in the part it was carried to (room there): not carried back",
              c2 == 0 and b2 == nx.base)
        s.pool.add(nx.end, 2 * H - nx.end).owner = 9                      # part 1 fills up; part 0 still has no room
        try:
            b3, c3, _ = place(s, turn2, 28 * A, [7] * (30 * A))
        except multi.NoRoom:
            c3 = None
        check("10 no ping-pong: a turn that part 1 cannot fit is not carried back into part 0 while part 0 lacks room",
              c3 != 2)
section("10 pressure", pressure_cases)


def others_cases():
    def lay(live0=16, live1=0):
        s = Car(2 * H, H)
        s.pool.add(0, live0 * A).owner = 1
        if live1:
            s.pool.add(H, live1 * A).owner = 2
        x = s.pool.add(live0 * A, 16 * A)
        hit = put_state(s, x, 7, 8 * A, name="hit"); other = put_state(s, x, 9, 16 * A, name="other")
        return s, x, hit, other
    with carry_on():
        s, x, hit, other = lay()
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others: the part has no free rows for the copy: another part's free rows carry it, the other conversation keeps its state",
              copy == 2 and h is hit and s.pool.half_of(base) == 1 and s.evicted == [] and other in x.kept and s.carry["others"] == 1)
        check("10 others: counted as others only", s.carry["rescued"] == s.carry["pressure"] == 0)
        s, x, hit, other = lay(live0=8)
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others: free rows in its own part still get the copy (copy 1), nothing carried",
              copy == 1 and s.pool.half_of(base) == 0 and s.carry["others"] == 0 and s.carry["hits"] == 0)
        s, x, hit, other = lay(live1=32)
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others: no free rows anywhere: the others are evicted and the extent grows in place, as before",
              copy == 0 and h is hit and base == x.base and s.evicted == ["other"] and s.carry["refused"]["no_target"] == 1)
        s, x, hit, other = lay()
        hit.ids = hit.key = [7] * 1024
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others: a hit under 2,048 tokens is not carried (small), the others are evicted as before",
              copy == 0 and s.carry["refused"]["small"] == 1 and s.evicted == ["other"])
        s, x, hit, other = lay()
        s.tune.lone = True
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others (a lone setting that would overwrite them): the others kept by carrying the hit, nothing evicted",
              copy == 2 and s.evicted == [] and s.carry["pressure"] == 1)
    with carry_on(False):
        s, x, hit, other = lay()
        base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 others (carry off): the others are evicted and the extent grows in place, as before",
              copy == 0 and base == x.base and s.evicted == ["other"] and s.carry["refused"]["no_target"] == 0)
section("10 others", others_cases)


# -- the old code as the reference: the same decisions with the carry off or without parts --------------------------------
def old_place(s, hit, need, key):
    """MultiDecoder.admit's placement block as of kvshard-v7, verbatim."""
    copy = 0
    base = None
    if hit is not None and hit.extent.owner is None:
        x = hit.extent
        others = [c for c in x.kept if len(c.ids) > len(hit.ids) and key[:len(c.ids)] != c.key]
        if others and not s.tune.lone:
            base = s.pool.place(need, half=s.pool.half_of(x) if s.pool.half else None)
            copy = int(base is not None)
        if base is None:
            for c in others:
                s._evict(c)
            if s._grow(x, need, protect=[x]):
                base = x.base
            else:
                hit = None
    if base is None:
        base = s._room(need, [hit.extent] if hit is not None else [],
                       half=s.pool.half_of(hit.extent) if hit is not None and s.pool.half else None)
        if base is None and hit is not None:
            hit = None
            base = s._room(need)
        if base is None:
            raise multi.NoRoom("no room")
        copy = int(hit is not None)
    return base, copy, hit


def rand_car(seed, parts=2, sharded=True, heavy=False):
    """A random pool of live (some with a stream), kept, loading and free ranges, twins of filling prompts and a slice in
    flight, in ``parts`` parts: (decoder, hit, need, key); the same seed builds the same pool, so one decoder can run
    the old code and another the new."""
    rnd = random.Random(seed)
    s = Car(parts * H, H if sharded else 0)
    states, nxt, lives, twins = [], 0, [], []
    for h in range(parts):
        at = h * H
        while True:
            n = rnd.randrange(1, 9) * A
            if at + n > (h + 1) * H:
                break
            kind = rnd.choice(["live", "live", "live", "kept", "free", "load", "twin"] if heavy else
                              ["live", "kept", "kept", "kept", "free", "load", "twin"])
            if kind == "live":
                x = s.pool.add(at, n)
                x.owner = 100 + at // A
                if rnd.random() < 0.5:
                    lane_of(s, x, x.owner, pos=rnd.randrange(1, n // A + 1) * A - 512)
                lives.append(x)
            elif kind == "kept":
                x = s.pool.add(at, n)
                for L in sorted({rnd.randrange(1, n // A + 1) * A for _ in range(rnd.choice([1, 1, 2]))}):
                    states.append(put_state(s, x, rnd.choice([3, 5, 7]), L, own=rnd.random() < 0.5, name=f"k{nxt}"))
                    nxt += 1
            elif kind == "load" and sharded:
                x = s.pool.add(at, n)
                x.loading = SimpleNamespace(n=n // 2, lj=None)
            elif kind == "twin" and sharded and lives:
                t = s.pool.add(at, n, twin=True)
                twins.append(t)
            at += n + rnd.choice([0, 0, A])
    for t in twins:                          # each twin belongs to a filling prompt living in another part
        homes = [x for x in lives if s.pool.half_of(x) != s.pool.half_of(t)]
        if not homes:
            s.pool.remove(t)
            continue
        x = rnd.choice(homes)
        lane = s.lanes.get(x.owner) or lane_of(s, x, x.owner, pos=A)
        lane.decoding = False
        lane.twin = (lane.twin or []) + [t]
        lane.st.twin_base, lane.st.twin_size = t.base, t.size
    filling = [l for l in s.lanes.values() if not l.decoding]
    if filling and rnd.random() < 0.5:
        chosen = {rnd.choice(filling).sid}
        s.partial = SimpleNamespace(contains=lambda sid: sid in chosen, group=[(l, A) for l in filling if l.sid in chosen])
    rnd.shuffle(s.kept)
    hit = rnd.choice(states) if states and rnd.random() < 0.9 else None
    need = rnd.randrange(2, 20) * A
    key = ([hit.ids[0]] if hit is not None else [0]) * (need + A)
    s.tune.lone = rnd.random() < 0.15
    return s, hit, need, key


def run_new(s, hit, need, key):
    try:
        base, copy, h = s._place_hit(hit, need, key)
        return ("ok", base, copy, h.name if h is not None else None)
    except multi.NoRoom:
        return ("noroom",)


def run_old(s, hit, need, key):
    try:
        base, copy, h = old_place(s, hit, need, key)
        return ("ok", base, copy, h.name if h is not None else None)
    except multi.NoRoom:
        return ("noroom",)


def snapshot_of(s):
    return (sorted((x.base, x.size, x.owner is not None, tuple(c.name for c in x.kept), x.twin) for x in s.pool.extents),
            list(s.evicted), s.no_part, [e for e in s.events if e[0] in ("emit", "evict")])


def old_equivalence():
    bad = []
    for sharded, on in ((True, False), (False, True), (False, False)):
        n = 0
        for seed in range(500):
            a = rand_car(seed, sharded=sharded)
            b = rand_car(seed, sharded=sharded)
            with carry_on(on):
                got = run_new(*a)
            want = run_old(*b)
            n += 1
            if got != want or snapshot_of(a[0]) != snapshot_of(b[0]):
                bad.append((sharded, on, seed, got, want))
        check(f"10 old code ({'parts, carry off' if sharded else 'one pool, carry ' + ('on' if on else 'off')}): {n} random pools (with "
              f"streams, twins, stored prompts being read and a slice in flight) place, evict, move and count exactly as "
              f"kvshard-v7 did ({bad[:1]})", not [b for b in bad if b[0] == sharded and b[1] == on])


def new_invariants():
    bad, carried, saved_evictions, dropped_kept, same_no_part, no_part_checked = [], 0, 0, 0, 0, 0
    kinds = collections.Counter()
    for seed in range(1200):
        for heavy in (False, True):
            a = rand_car(seed, heavy=heavy)
            b = rand_car(seed, heavy=heavy)
            pins = a[0]._pins(a[1].extent) if a[1] is not None else None
            with carry_on(False):
                old = run_new(*b)
            with carry_on():
                new = run_new(*a)
            s, hit = a[0], a[1]
            tag = (seed, heavy)
            if (old[0] == "noroom") != (new[0] == "noroom"):
                bad.append((tag, "room", old, new)); continue
            if new[0] == "noroom":
                continue
            _, base, copy, kept = new
            old_evicted = b[0].evicted
            if copy == 2:
                carried += 1
                kinds[s.carry_kind] += 1
                if hit is None or kept != hit.name or s.pool.half_of(base) == s.pool.half_of(hit.extent) or pins is not None \
                        or len(hit.ids) < multi.CARRY_MIN:
                    bad.append((tag, "copy 2 not across parts, or pinned, or small", new, pins)); continue
                # a carry avoids a real cost: the old code evicted something, or dropped the hit
                if not (old_evicted or old[3] is None):
                    bad.append((tag, "carried where the old code evicted nothing and kept the hit", old, new)); continue
                if old_evicted or old[3] is None:
                    saved_evictions += 1
                if s.carry_kind != "rescued" and [e for e in s.events if e[0] == "evict"]:
                    bad.append((tag, "a pressure/others carry evicted something", s.events)); continue
            else:
                # the part fits the request without evicting: the old code's decision, exactly, hit kept
                if old[3] is not None and not old_evicted and old != new:
                    bad.append((tag, "no eviction in the old code and the new one decided otherwise", old, new)); continue
                if copy == 1 and hit is not None and s.pool.half_of(base) != s.pool.half_of(hit.extent):
                    bad.append((tag, "copy 1 across parts", new)); continue
                if old[3] is not None and new[3] is None:
                    bad.append((tag, "dropped a hit the old code kept", old, new)); continue
                no_part_checked += 1
                if s.no_part == b[0].no_part:
                    same_no_part += 1
            try:
                s.pool.add(base, a[2])
            except ValueError:
                if copy != 0:
                    bad.append((tag, "placed on taken rows", new))
    check(f"10 random pools: a carry only where the old code evicted or dropped, never where its part fits the request without "
          f"evicting (decided as before, hit kept), pressure and others carries evict nothing, nothing pinned or under 2,048 "
          f"tokens, NoRoom the same ({carried} carried {dict(kinds)}, {saved_evictions} evictions or drops avoided; {bad[:1]})", not bad)
    check("10 random pools: the carry actually happens in every kind (the sweep reaches them)",
          carried > 40 and kinds["pressure"] >= 3 and kinds["rescued"] >= 3)
    check(f"10 random pools: where nothing is carried the refusal count (/health refused_no_part) is the old code's "
          f"({same_no_part} of {no_part_checked})", same_no_part == no_part_checked > 100)


def grow_fits_sound():
    bad = checked = 0
    for seed in range(2500):
        s, hit, need, key = rand_car(seed, heavy=True)
        if hit is None or hit.extent.owner is not None:
            continue
        x = hit.extent
        no = not s._grow_fits(x, need)
        if no:
            checked += 1
            if s._grow(x, need, protect=[x]):         # (on a pool this check does not reuse)
                bad += 1
    check(f"10 grow fits: never says no to a grow that works, with streams, twins and stored prompts being read "
          f"({checked} refusals checked)", bad == 0 and checked > 20)


def unevicted_sound():
    """``_fits_unevicted`` against what the old code does: it says yes exactly when the old code places the request
    without evicting anything."""
    bad = yes = no = 0
    for seed in range(2500):
        a = rand_car(seed, heavy=bool(seed % 2))
        s, hit, need, key = a
        if hit is None:
            continue
        others = [c for c in hit.extent.kept if len(c.ids) > len(hit.ids) and key[:len(c.ids)] != c.key] \
            if hit.extent.owner is None else []
        fits = s._fits_unevicted(hit, need, others)
        b = rand_car(seed, heavy=bool(seed % 2))
        with carry_on(False):
            old = run_new(*b)
        silent = old[0] == "ok" and old[3] is not None and not b[0].evicted
        if fits and not silent:
            bad += 1
        if (not fits) and silent:
            bad += 1
        yes += fits; no += not fits
    check(f"10 fits unevicted: yes exactly where the old code placed the request without evicting anything and kept the hit "
          f"({yes} yes, {no} no; {bad} differ)", bad == 0 and yes > 50 and no > 50)
section("10 old equivalence", old_equivalence)
section("10 invariants", new_invariants)
section("10 grow fits", grow_fits_sound)
section("10 unevicted", unevicted_sound)


# -- the ops, in order: ADMIT copy=2 flushes before the collective; both ranks end with the same pool ---------------------
from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda import forward as _forward


class FakeState:
    """forward.State without tensors of the pool: what _admitted touches."""
    def __init__(self, w, size, rows, caches=None, base=0, slots=None, slot=0, **kw):
        self.size, self.base, self.slot, self.pos = size, base, slot, 0
        self.rec = [torch.zeros(2)]; self.cur = [0]; self.conv = torch.zeros(2); self.index = None; self.kc = []
        self.twin_base = self.twin_size = None; self.binds = []
    def reset(self): pass
    def set_pos(self, p): self.pos = p
    def bind(self, b, n): self.base, self.size = b, n; self.binds.append((b, n))


class Adm(Car):
    """Car with admit and _admitted themselves (rank 0 decides and applies, rank 1 applies what it is sent)."""
    admit = _mm("admit"); _admitted = _mm("_admitted"); _parse_admit = staticmethod(multi.MultiDecoder._parse_admit)
    _resume = _mm("_resume"); _kept_by_id = _mm("_kept_by_id"); _rewrites = _mm("_rewrites"); _need = _mm("_need")
    apply = _mm("apply"); parse = staticmethod(multi.MultiDecoder.parse); _continued = _mm("_continued")

    def __init__(self, rows, half, rank=0):
        super().__init__(rows, half)
        self.rank = rank; self.outbox = []
        self.g = SimpleNamespace(limit=10 ** 9, serial_only=False, vision=None, shared=0, opener=None, comm=None,
                                 cache_entries=8, costs=None)
        self.e = SimpleNamespace(rows=2048, caches=None, slots=None); self.w = SimpleNamespace(world=2)
        self.count = 4; self.drafts = None; self.dflash_code = [0, 0, 0, 0]; self.grid = 0
        self.next_sid = self.next_kid = self.next_order = self.next_chat = 0
        self.broken = None; self.boot = "boot"; self.chat_roots = {}; self.chats_continued = self.chats_continued_prefill = 0
        self.arena.planes = [SimpleNamespace(div=4)]
    def _check(self): pass
    def _emit(self, op, payload):
        self.events.append(("emit", op)); self.ops.append((op, [int(v) for v in payload]))
        return multi.MultiDecoder._emit(self, op, payload)        # (outbox: what rank 1 is sent)
    def _sync(self): self.events.append(("sync",))


@contextlib.contextmanager
def fake_state():
    saved = _forward.State
    _forward.State = FakeState
    try:
        yield
    finally:
        _forward.State = saved


def hit_state(s, x, cid, n, own=False, name="hit"):
    """A kept state a request can resume from (the fields _admitted and _resume read)."""
    c = put_state(s, x, cid, n, own=own, name=name)
    c.rec, c.conv, c.tail, c.mtp_len, c.lineage, c.kid = torch.zeros(2), torch.zeros(2), None, -1, "", s.next_kid
    s.next_kid += 1
    return c


def adm_rescue(rank=0):
    """The rescue pool as an Adm: part 0 full (a live stream and the kept hit), part 1 empty."""
    s = Adm(2 * H, H, rank)
    s.pool.add(0, 24 * A).owner = 1
    x = s.pool.add(24 * A, 8 * A)
    return s, x, hit_state(s, x, 7, 8 * A, own=True)


def admit_order():
    with carry_on(), fake_state():
        s, x, hit = adm_rescue()
        s.admit(Stream([7] * (14 * A), 4, None, True))
        ev = [e[0] if e[0] != "emit" else e for e in s.events]
        i, j, k = ev.index(("emit", multi.ADMIT)), ev.index("sync"), ev.index("carry")
        check("10 admit: copy 2 is flushed (_sync) after the ADMIT is queued and BEFORE the carry runs", i < j < k)
        check("10 admit: the ADMIT says copy 2 and names the hit", s.ops[-1][0] == multi.ADMIT and s.ops[-1][1][5] == 2
              and s.ops[-1][1][4] == hit.kid)
        check("10 admit: exactly one sync, and the carry is the hit's rows to the new extent",
              ev.count("sync") == 1 and s.events[k] == ("carry", hit.extent.base, s.pool.get(max(e.eid for e in s.pool.extents)).base, 8 * A)
              and not [e for e in s.events if e[0] == "copy"])
        check("10 admit: carry stats on this rank (hits, tokens, bytes)",
              s.carry["hits"] == 1 and s.carry["tokens"] == 8 * A and s.carry["bytes"] == 8 * A * 5808 and s.carry["live"] == 0)
        # copy 1 and copy 0 flush nothing of their own
        s = Adm(2 * H, H)
        s.pool.add(H, 4 * A).owner = 2                            # (one live stream a part: no balance carry)
        x = s.pool.add(0, 8 * A); hit = hit_state(s, x, 7, 8 * A); x.owner = 3     # the hit's extent is a live stream's
        s.admit(Stream([7] * (9 * A), 4, None, True))
        check("10 admit: a copy within the part (copy 1) has no sync of its own",
              "sync" not in [e[0] for e in s.events] and s.ops[-1][1][5] == 1
              and [e for e in s.events if e[0] == "copy"] == [("copy", 0, s.pool.get(max(e.eid for e in s.pool.extents)).base, 8 * A)])
        s = Adm(2 * H, H)
        s.pool.add(0, 24 * A).owner = 1
        x = s.pool.add(24 * A, 4 * A); hit = hit_state(s, x, 7, 4 * A)
        s.pool.add(H, 24 * A).owner = 2
        s.admit(Stream([7] * (7 * A), 4, None, True))
        check("10 admit: taking the extent over in place (copy 0) has no sync either",
              "sync" not in [e[0] for e in s.events] and s.ops[-1][1][5] == 0 and [e for e in s.events if e[0] in ("carry", "copy")] == [])
section("10 admit order", admit_order)


def two_ranks():
    # rank 0 admits, rank 1 applies the message: the same pool, the same kept order, the same carry
    with carry_on(), fake_state():
        r0, x0, h0 = adm_rescue(0)
        r1, x1, h1 = adm_rescue(1)
        r0.admit(Stream([7] * (14 * A), 4, None, True))
        msg = list(r0.outbox)
        r1.apply(msg)
        shape = lambda s: sorted((x.base, x.size, x.owner is not None, [c.name for c in x.kept]) for x in s.pool.extents)
        check("10 two ranks: rank 1 applying the ADMIT ends with rank 0's pool", shape(r0) == shape(r1))
        check("10 two ranks: rank 1 carries the same rows to the same place",
              [e for e in r0.events if e[0] == "carry"] == [e for e in r1.events if e[0] == "carry"] != [])
        check("10 two ranks: neither copies instead", not [e for e in r0.events + r1.events if e[0] == "copy"])
        check("10 two ranks: the same kept order (the carried own state is not touched)",
              [c.name for c in r0.kept] == [c.name for c in r1.kept] and r0.kept[-1] is h0)
        check("10 two ranks: rank 1 counts the carry too", r1.carry["hits"] == 1 and r1.carry["tokens"] == 8 * A)
        # a shared prefix (not the conversation's own turn state) stays for the others: touched as ever
        for own in (True, False):
            r = []
            for rank in (0, 1):
                s, x, hit = adm_rescue(rank)
                hit.own = own
                other = put_state(s, s.pool.add(H, 4 * A), 5, 4 * A, name="old")
                s.kept = [hit, other]                              # (the hit is the older)
                r.append(s)
            r[0].admit(Stream([7] * (14 * A), 4, None, True))
            r[1].apply(list(r[0].outbox))
            last = [[c.name for c in s.kept][-1] for s in r]
            check(f"10 admit: after a carry the resumed state {'keeps its place (its own turn state: the LRU reclaims it first)' if own else 'is touched (a shared prefix stays for the others)'}, on both ranks",
                  last == (["old", "old"] if own else ["hit", "hit"]))
    # a malformed copy 2 fails closed
    with carry_on(), fake_state():
        s = Adm(2 * H, H, 1)
        x = s.pool.add(0, 8 * A); hit = hit_state(s, x, 7, 8 * A)
        a = dict(sid=0, slot=0, base=16 * A, size=16 * A, kid=hit.kid, copy=2, count=4, stop_eos=True, draft=True, sampling=None,
                 shared=[], code=[0, 0, 0, 0], prompt=[7] * (9 * A), packed=[], positions=[], images=False, keyed=None)
        try:
            s._admitted(a, Stream([7] * (9 * A), 4, None, True), None, None); ok = False
        except RuntimeError:
            ok = True
        check("10 follower: a copy 2 inside one part fails closed", ok and not [e for e in s.events if e[0] == "carry"])
        s = Adm(2 * H, H, 1)
        s.pool.add(0, 8 * A)
        try:
            s._admitted(dict(a, kid=-1, base=H), Stream([7] * (9 * A), 4, None, True), None, None); ok = False
        except RuntimeError:
            ok = True
        check("10 follower: a copy 2 of no kept prompt fails closed", ok)
section("10 two ranks", two_ranks)


def recent_cases():
    s = Car(2 * H, H)
    a = put_state(s, s.pool.add(0, 4 * A), 1, 4 * A, own=True, name="a")
    b = put_state(s, s.pool.add(H, 4 * A), 2, 4 * A, own=True, name="b")
    c = put_state(s, s.pool.add(8 * A, 4 * A), 3, 4 * A, own=False, name="c")
    s._recent(a, 2)
    check("10 recent: a carried own turn state keeps its place in the LRU order", [k.name for k in s.kept] == ["a", "b", "c"])
    s._recent(c, 2)
    check("10 recent: a carried shared prefix is touched (other conversations resume from it)", [k.name for k in s.kept] == ["a", "b", "c"][:2] + ["c"])
    s._recent(a, 1)
    check("10 recent: a copy within the part touches an own state as ever", [k.name for k in s.kept][-1] == "a")
    s._recent(b, 0)
    check("10 recent: so does taking the extent over", [k.name for k in s.kept][-1] == "b")
section("10 recent", recent_cases)


# -- a live stream moves whole: XMOVE ---------------------------------------------------------------------------------------
class FakeDisk:
    """The spill tier as far as a move touches it: what it is asked to release, and the extent's base then; ``waiting``:
    the rows of states this rank has still to write (what ``_used`` counts), per extent."""
    def __init__(self, waiting=None): self.releases = []; self.pending = []; self.waiting = []; self.want = waiting
    def release(self, x, keep): self.releases.append((x.eid, keep, x.base))
    def stop_load(self, lj, why): pass
    def put(self, x, n):
        self.pending.append(SimpleNamespace(point=SimpleNamespace(n=n), where=x, next=0, new=[1]))


def xlay(pos=8 * A - 1024, kept=0, lane2=True):
    """Part 0: stream 1 (extent 8A, ``pos`` tokens in) and stream 2 (24A): full; part 1 empty. ``lane2`` False: stream 2
    is a live extent without a lane to move."""
    s = Car(2 * H, H)
    x = s.pool.add(0, 8 * A)
    lane = lane_of(s, x, 1, pos=pos)
    y = s.pool.add(8 * A, 24 * A)
    if lane2:
        lane_of(s, y, 2, pos=20 * A)
    else:
        y.owner = 2
    if kept:
        put_state(s, x, 7, kept, own=True, name="k")
    return s, x, lane


def xmove_cases():
    with carry_on():
        s, x, lane = xlay(lane2=False)
        eid, old = x.eid, x.base
        check("10 xmove: the lane cannot grow in its part (the setup)", s._grow(x, lane.st.pos + 2048, protect=[x]) is False)
        ok = s._relocate(lane)
        size = align_up(lane.st.pos + s.rows_max)
        check("10 xmove: relocated: the same extent, now in the other part, big enough for the round",
              ok and x.eid == eid and s.pool.half_of(x) == 1 and x.size == size and x.owner == 1 and x in s.pool.extents)
        ev = [e for e in s.events if e[0] in ("emit", "sync", "carry")]
        check("10 xmove: XMOVE is queued, flushed (_sync), and only then does the carry run, of the used rows",
              ev == [("emit", multi.XMOVE), ("sync",), ("carry", old, x.base, lane.st.pos)])
        check("10 xmove: the op names the extent, its new base, size and the rows used (rank 0's count rides along)",
              s.ops[-1] == (multi.XMOVE, [eid, x.base, size, lane.st.pos]))
        check("10 xmove: the stream is rebound to its new rows", lane.st.binds[-1] == (x.base, x.size))
        check("10 xmove: no overlap in the pool, the old rows are free",
              all(a.end <= b.base for a, b in zip(s.pool.extents, s.pool.extents[1:])) and old == 0
              and s.pool.place(8 * A, half=0) == 0)
        check("10 xmove: headroom: a block for the arriving stream is free after it in the part it went to",
              s.pool.room_after(x) >= 2048)
        check("10 xmove: counted (live, tokens, bytes), the stream remembers where it was and when", s.carry["live"] == 1
              and s.carry["tokens"] == lane.st.pos and s.carry["bytes"] == lane.st.pos * 5808 and s.carry["hits"] == 0
              and lane.xwhy is None and lane.came_from == 0 and lane.moved_at == s.xticks and s.entered == {1: s.xticks})
        # kept states travel with the extent
        s, x, lane = xlay(kept=8 * A, lane2=False)
        k = x.kept[0]
        s._relocate(lane)
        check("10 xmove: the extent's kept prompts move with it (still in its list and the kept order)",
              s.pool.half_of(x) == 1 and x.kept == [k] and k in s.kept and k.extent is x)
        check("10 xmove: the rows carried are the used ones (kept states and the stream's position, whichever is more)",
              [e for e in s.events if e[0] == "carry"][0][3] == max(8 * A, lane.st.pos))
        # the free rows are not together: compacting the target part first (MOVEs before the XMOVE)
        s, x, lane = xlay(lane2=False)
        for i in range(4):
            put_state(s, s.pool.add(H + i * 8 * A, 4 * A), 20 + i, 4 * A)
        ok = s._relocate(lane)
        ops = [e[1] for e in s.events if e[0] == "emit"]
        check("10 xmove: free rows gathered by compacting the other part first (MOVEs, then the XMOVE), nothing evicted",
              ok and ops[-1] == multi.XMOVE and multi.MOVE in ops and set(ops[:-1]) == {multi.MOVE} and s.evicted == [])
        # headroom in the target: the rows without a block for each stream there and the arriving one are no room
        s, x, lane = xlay(lane2=False)
        size = align_up(lane.st.pos + s.rows_max)
        s.pool.add(H, 32 * A - size).owner = 7                          # (exactly size free, no headroom)
        r = [s._relocate(lane), s.carry["refused"]["no_target"]]
        s2, x2, lane2_ = xlay(lane2=False)
        s2.pool.add(H, 32 * A - size - 2 * 2048).owner = 7              # (size and a block each for the stream there and the arriving)
        check("10 xmove: free rows without headroom are no room (pause as before); with a block for each stream there and the "
              "arriving one it moves", r == [False, 1] and s2._relocate(lane2_) is True)
        # the cheapest stream moves: a small one whose leaving lets the stream that cannot grow grow in place
        s = Car(2 * H, H)
        a = s.pool.add(0, 16 * A); la = lane_of(s, a, 1, pos=16 * A - 1024)           # needs 2048 more rows
        b = s.pool.add(16 * A, 4 * A); lb = lane_of(s, b, 2, pos=3 * A)
        s.pool.add(20 * A, 12 * A).owner = 3
        ok = s._relocate(la)
        ev = [e for e in s.events if e[0] == "emit"]
        check("10 xmove: the smallest stream that frees the rows moves, not the stream that cannot grow (3A carried, not 16A): "
              "the stream then grows in place", ok and s.pool.half_of(b) == 1 and s.pool.half_of(a) == 0 and a.base == 0
              and ev.count(("emit", multi.XMOVE)) == 1 and [e for e in s.events if e[0] == "carry"][0][3] == 3 * A
              and a.size >= align_up(la.st.pos + s.rows_max) and lb.moved_at == s.xticks and la.moved_at < 0)
        # one carry a relocation: the bystander moved but the stream still cannot grow -> it pauses, no second carry
        s = Car(2 * H, H)
        a = s.pool.add(0, 16 * A); la = lane_of(s, a, 1, pos=16 * A - 1024)
        b = s.pool.add(16 * A, 4 * A); lb = lane_of(s, b, 2, pos=3 * A)
        s.pool.add(20 * A, 12 * A).owner = 3
        s._grow = lambda *a_, **k_: False
        ok = s._relocate(la)
        check("10 xmove: after a bystander's move the stream grows or pauses: one XMOVE, never a second carry",
              ok is False and [e for e in s.events if e[0] == "emit"].count(("emit", multi.XMOVE)) == 1
              and s.pool.half_of(b) == 1 and s.pool.half_of(a) == 0)
        # ...but not one that something pins; then the stream itself
        s = Car(2 * H, H)
        a = s.pool.add(0, 16 * A); la = lane_of(s, a, 1, pos=16 * A - 1024)
        b = s.pool.add(16 * A, 4 * A); lb = lane_of(s, b, 2, pos=3 * A); lb.twin = [s.pool.add(H + 24 * A, 4 * A, twin=True)]
        s.pool.add(20 * A, 12 * A).owner = 3
        ok = s._relocate(la)
        check("10 xmove: the smaller stream is pinned (a twin): the stream that cannot grow moves itself",
              ok and s.pool.half_of(a) == 1 and s.pool.half_of(b) == 0)
        # ...nor one that is not small: the stream itself is the cheaper carry
        s = Car(2 * H, H)
        a = s.pool.add(0, 4 * A); la = lane_of(s, a, 1, pos=4 * A - 1024)
        b = s.pool.add(4 * A, 20 * A); lb = lane_of(s, b, 2, pos=19 * A)
        s.pool.add(24 * A, 8 * A).owner = 3
        ok = s._relocate(la)
        check("10 xmove: when the stream that cannot grow is the smaller, it moves itself", ok and s.pool.half_of(a) == 1
              and s.pool.half_of(b) == 0)
        # cooldown: a stream that moved is not moved again within the window, nor back; no stream leaves a part another entered
        K = multi.RELOCATE_HOLD
        s, x, lane = xlay(lane2=False)
        s.xticks = 1000
        lane.moved_at, lane.came_from = 1000 - K + 1, 1
        check("10 xmove: a stream that moved less than RELOCATE_HOLD rounds ago is not moved again",
              s._relocate(lane) is False and s.carry["refused"]["no_target"] == 1 and not s.events)
        s.xticks = 1000 - K + 1 + K
        check("10 xmove: ... and may after the window", s._relocate(lane) is True)
        # three parts, so the part it came from is not the only one with room: only the cooldown stops it
        def three():
            t = Car(3 * H, H)
            tx = t.pool.add(0, 8 * A); tl = lane_of(t, tx, 1, pos=8 * A - 1024)
            t.pool.add(8 * A, 24 * A).owner = 2
            t.xticks = 1000
            tl.moved_at, tl.came_from = 1000 - K + 1, 1
            return t, tx, tl
        t, tx, tl = three()
        check("10 xmove: inside the window it stays put though a third part has the room (the cooldown, not the part it left)",
              t._relocate(tl) is False and not t.events and t._move_to(tl, 8 * A) == 2)
        t, tx, tl = three()
        tl.moved_at = 1000 - K
        check("10 xmove: ... and the window passed it may move again",
              t._relocate(tl) is True and t.pool.half_of(tx) in (1, 2))
        s, x, lane = xlay(lane2=False)
        s.xticks = 1000
        s.entered[0] = 1000 - K + 1
        check("10 xmove: no stream leaves a part another entered within the window (moves do not answer one another)",
              s._relocate(lane) is False and not s.events)
        s.xticks = 1000 + 1
        check("10 xmove: ... and may after it", s._relocate(lane) is True)
        s, x, lane = xlay(lane2=False)
        s.xticks = 1000
        lane.moved_at, lane.came_from = 1000 - K + 1, 1
        check("10 xmove: never back into the part a stream just left (_move_to), though it has the room",
              s._move_to(lane, 8 * A) is None)
        lane.moved_at = 1000 - K
        check("10 xmove: ... once the window has passed it may", s._move_to(lane, 8 * A) == 1)
        # refusals: counted once a reason, nothing queued, nothing moved
        def refuse(why, setup):
            s, x, lane = xlay(lane2=False)
            setup(s, x, lane)
            r = [s._relocate(lane), s._relocate(lane)]
            check(f"10 xmove: a stream with {why} is not relocated, counted once; nothing queued or moved",
                  r == [False, False] and s.carry["refused"][why] == 1 and not s.events and x.base == 0)
        refuse("loading", lambda s, x, l: setattr(x, "loading", object()))
        refuse("partial", lambda s, x, l: setattr(s, "partial", SimpleNamespace(contains=lambda sid: sid == 1, group=[(l, A)])))

        def tw(s, x, l):
            t = s.pool.add(H + 8 * A, 8 * A); t.twin = True
            l.twin = [t]
        refuse("twin", tw)

        def pn(s, x, l):                                 # (the extent itself a pinned twin of a fill in flight)
            x.twin = True
            ex = s.pool.add(H, 4 * A); ex.owner = 9
            f = SimpleNamespace(sid=9, twin=[x], st=St(), extent=ex, decoding=False)
            s.lanes[9] = f
            s.partial = SimpleNamespace(contains=lambda sid: sid == 9, group=[(f, A)])
        refuse("pinned", pn)
        s, x, lane = xlay(lane2=False)
        s.pool.add(H, 32 * A).owner = 3
        r = [s._relocate(lane), s._relocate(lane)]
        check("10 xmove: no part with the room is no_target (counted once a stream), nothing evicted or moved",
              r == [False, False] and s.carry["refused"]["no_target"] == 1 and not s.events)
        lane.xwhy = None
        s._relocate(lane)
        check("10 xmove: and counted again when the reason is new again", s.carry["refused"]["no_target"] == 2)
    with carry_on(False):
        s, x, lane = xlay()
        check("10 xmove: carry off: never", s._relocate(lane) is False and not s.events and not any(s.carry["refused"].values()))
section("10 xmove", xmove_cases)


class Fol(Car):
    """Car that also applies a message (rank 1), with what XMOVE reads."""
    apply = _mm("apply"); parse = staticmethod(multi.MultiDecoder.parse)
    def __init__(self, rows, half):
        super().__init__(rows, half); self.rank = 1; self.outbox = []
    def _emit(self, op, payload):
        self.events.append(("emit", op)); self.ops.append((op, [int(v) for v in payload]))
        return multi.MultiDecoder._emit(self, op, payload)


def xa_of(r):
    return next(x for x in r.pool.extents if x.owner == 1)


def xmove_two_ranks():
    with carry_on():
        s0, x0, l0 = xlay(kept=8 * A, lane2=False)
        s1, x1, l1 = xlay(kept=8 * A, lane2=False)
        r0, r1 = Fol(2 * H, H), Fol(2 * H, H)
        for r, s in ((r0, s0), (r1, s1)):
            r.pool, r.lanes, r.kept = s.pool, s.lanes, s.kept
        r0.rank = 0
        ok = r0._relocate(l0)
        r1.apply(list(r0.outbox))
        shape = lambda s: sorted((x.eid, x.base, x.size, x.owner, len(x.kept)) for x in s.pool.extents)
        check("10 two ranks: rank 1 applying the XMOVE ends with rank 0's pool, its stream rebound the same",
              ok and shape(r0) == shape(r1) and l1.st.binds == l0.st.binds and l1.st.binds[-1] == (x1.base, x1.size)
              and s1.pool.half_of(x1) == 1)
        check("10 two ranks: rank 1 carries the same rows between the same rows",
              [e for e in r1.events if e[0] == "carry"] == [e for e in r0.events if e[0] == "carry"] != []
              and r1.carry["live"] == 1 and r1.carry["tokens"] == r0.carry["tokens"])
        # the rows to send are rank 0's count, whatever each rank's spill tier still has waiting (it advances them alone)
        def pair(w0, w1):
            a, xa, la = xlay(lane2=False); b, xb, lb = xlay(lane2=False)
            ra, rb = Fol(2 * H, H), Fol(2 * H, H)
            for r, s in ((ra, a), (rb, b)):
                r.pool, r.lanes, r.kept = s.pool, s.lanes, s.kept
            ra.rank = 0
            ra.disk, rb.disk = FakeDisk(), FakeDisk()
            if w0: ra.disk.put(xa, w0)
            if w1: rb.disk.put(xb, w1)
            ra._relocate(la)
            return ra, rb, la
        ra, rb, la = pair(8 * A, 7 * A)
        rb.apply(list(ra.outbox))
        na, nb = [e[3] for e in ra.events if e[0] == "carry"], [e[3] for e in rb.events if e[0] == "carry"]
        want = la.st.pos                     # kept prompts and the stream's position: what every rank agrees on
        check("10 xmove: ranks whose spill tiers have different writes waiting carry the same number of rows: rank 0's "
              "count without the waiting writes (released to the host before the move), in the op",
              na == nb == [want] and ra.ops[-1][1][3] == want and ra._used(xa_of(ra), pending=False) == want)
        ra, rb, la = pair(7 * A, 8 * A)
        try:
            rb.apply(list(ra.outbox)); raised = False
        except RuntimeError:
            raised = True
        nb = [e[3] for e in rb.events if e[0] == "carry"]
        check("10 xmove: a rank with more writes waiting than rank 0 does not refuse (rank 0 would be left in its send): "
              "it carries rank 0's count", not raised and nb == [e[3] for e in ra.events if e[0] == "carry"] == [want])
section("10 xmove two ranks", xmove_two_ranks)


class Loop(Car):
    """Car with the real _iterate and what its decode branch reads."""
    _iterate = _mm("_iterate"); _fill_turn = _mm("_fill_turn")
    def __init__(self, rows, half):
        super().__init__(rows, half)
        self.fill_budget = 0; self.group = False; self.profiles = None; self.lone_rows = 0; self.short_round_due = False
        self.decode_due = False; self.credit = 0.0; self.share = 0.5; self.profile = None; self.outbox = []
    def _flush(self): self.events.append(("flush",))
    def _round(self, lanes, **kw): self.events.append(("round", [l.sid for l in lanes]))
    def _give_back(self): self.events.append(("giveback",)); return []


def decode_loop():
    for on in (True, False):
        s, x, lane = xlay(lane2=False)
        l = Loop(2 * H, H)
        l.pool, l.lanes = s.pool, s.lanes
        with carry_on(on):
            l._iterate()
        evs = [e[0] if e[0] != "emit" else e[1] for e in l.events]
        if on:
            check("10 decode loop: a stream that cannot grow is relocated, then decodes in the round (XMOVE, sync, carry, then ROUND)",
                  evs[:3] == [multi.XMOVE, "sync", "carry"] and ("round", [1]) in l.events)
            check("10 decode loop: it is not paused, and the round counted", not lane.paused and l.pool.half_of(x) == 1
                  and l.xticks == 1)
        else:
            check("10 decode loop (carry off): it pauses, as before",
                  lane.paused and x.base == 0 and not any(e == ("emit", multi.XMOVE) for e in l.events))
section("10 decode loop", decode_loop)


class Grow(Loop):
    """The loop with a round that moves every stream on by STEP tokens."""
    STEP = 512
    def _round(self, lanes, **kw):
        self.events.append(("round", [l.sid for l in lanes]))
        for l in lanes:
            l.st.pos += self.STEP


def thrash():
    """Many rounds over a pool that cannot hold what its streams grow into: moves do not answer one another. (The review's
    probe: stream 1 moved 0 -> 1, stream 3 back 1 -> 0 eight rounds later; two whole-extent transfers for one lack of room.)"""
    K = multi.RELOCATE_HOLD
    results = []
    for variant in range(5):
        s = Grow(2 * H, H)
        layout = [[(1, 0, 16 * A, 16 * A - 1500), (2, 16 * A, 16 * A, 16 * A - 9000), (3, H, 12 * A, 12 * A - 6000), (4, H + 12 * A, 3 * A, 3 * A - 7000)],
                  [(1, 0, 8 * A, 8 * A - 1500), (2, 8 * A, 8 * A, 8 * A - 900), (3, 16 * A, 8 * A, 8 * A - 600), (4, 24 * A, 8 * A, 8 * A - 2000),
                   (5, H, 16 * A, 16 * A - 3000)],
                  [(1, 0, 12 * A, 12 * A - 800), (2, 12 * A, 12 * A, 12 * A - 400), (3, H, 20 * A, 20 * A - 2500), (4, H + 20 * A, 4 * A, 4 * A - 2000)],
                  [(1, 0, 8 * A, 8 * A - 1000), (2, 8 * A, 8 * A, 8 * A - 1000), (3, 16 * A, 8 * A, 8 * A - 1000), (4, 24 * A, 8 * A, 8 * A - 1000),
                   (5, H, 4 * A, 3 * A)],
                  [(1, 0, 10 * A, 10 * A - 1000), (2, 10 * A, 10 * A, 10 * A - 1000), (3, 20 * A, 12 * A, 12 * A - 1000), (4, H, 6 * A, 5 * A),
                   (5, H + 6 * A, 6 * A, 5 * A)]][variant]
        for sid, base, size, pos in layout:
            x = s.pool.add(base, size)
            lane_of(s, x, sid, pos=pos)
        s.outbox = []
        moves = []                                   # (round, sid, from part, to part)
        by_eid = {l.extent.eid: l.sid for l in s.lanes.values()}
        where = {l.sid: s.pool.half_of(l.extent) for l in s.lanes.values()}
        with carry_on():
            for r in range(3000):
                n0 = len(s.ops)
                s._iterate()
                for op, p in s.ops[n0:]:
                    if op == multi.XMOVE:
                        sid = by_eid[p[0]]
                        to = p[1] // H
                        moves.append((r, sid, where[sid], to))
                        where[sid] = to
        gaps_ok = all(b[0] - a[0] >= K for sid in {m[1] for m in moves} for a, b in
                      zip([m for m in moves if m[1] == sid], [m for m in moves if m[1] == sid][1:]))
        back = [(a, b) for a in moves for b in moves if b[1] == a[1] and 0 < b[0] - a[0] < K and b[3] == a[2]]
        counter = [(a, b) for a in moves for b in moves if 0 < b[0] - a[0] < K and b[2] == a[3] and b[1] != a[1]]
        results.append((len(moves), gaps_ok, back, counter, len({m[1] for m in moves})))
        check(f"10 thrash [{variant}]: over 3000 rounds of growing streams, a stream moves at most once per {K} rounds, never "
              f"back into the part it left within them, and no stream leaves a part another entered within them "
              f"({len(moves)} moves, {len(counter)} counter-moves)", gaps_ok and not back and not counter)
        check(f"10 thrash [{variant}]: the moves are bounded (at most one per stream per window)",
              len(moves) <= len(s.lanes) * (3000 // K + 1))
    check("10 thrash: the sweep moves streams (it reaches the rule)", sum(r[0] for r in results) >= 3)
section("10 thrash", thrash)


def move_guard():
    s = Car(2 * H, H); s.disk = FakeDisk(); s._guard_pool()
    x = s.pool.add(0, 8 * A); x.owner = 1
    s.lanes[1] = SimpleNamespace(sid=1, twin=None, st=St(4 * A), extent=x, decoding=True)
    s._move(x, 4 * A, 8 * A)
    check("10 spill: a move inside the part releases nothing (as before)", s.disk.releases == [])
    old = x.base
    s._xmove(x, H, 8 * A, 4 * A)
    check("10 spill: a move across parts first has the tier copy the waiting writes' rows to the host (release from 0, "
          "while the extent is still at its old base)", s.disk.releases == [(x.eid, 0, old)])
section("10 move guard", move_guard)


def pool_cross():
    p = Pool(2 * H, half=H)
    x = p.add(0, 8 * A)
    try:
        p.move(x, H, 8 * A); refused = False
    except ValueError:
        refused = True
    check("10 pool: a move to another part is still refused without cross", refused and x.base == 0)
    check("10 pool: ...and is made with cross=True (returns the old base)", p.move(x, H, 8 * A, cross=True) == 0 and x.base == H)
    try:
        p.move(x, H - A, 8 * A, cross=True); crossed = True
    except ValueError:
        crossed = False
    check("10 pool: even with cross an extent never straddles a part's end", not crossed and x.base == H)
    t = p.add(0, 4 * A, twin=True)
    try:
        p.move(t, H + 16 * A, 4 * A, cross=True); ok = False
    except ValueError:
        ok = True
    check("10 pool: a twin never moves, across parts or not", ok)
    y = p.add(H + 8 * A, 4 * A)
    try:
        p.move(x, H + 4 * A, 8 * A, cross=True); ok = False
    except ValueError:
        ok = True
    check("10 pool: a cross move onto taken rows is refused", ok)
section("10 pool cross", pool_cross)


def ops_and_settings():
    check("10 ops: XMOVE is in the table, after the older ops", multi.OPS[multi.XMOVE] == "XMOVE" and multi.XMOVE == max(multi.OPS))
    ok = multi.MultiDecoder.parse([multi.XMOVE, 4, 4, 5, 6, 7]) == [(multi.XMOVE, [4, 5, 6, 7])]
    try:
        multi.MultiDecoder.parse([multi.XMOVE, 3, 4, 5, 6]); bad = False
    except RuntimeError:
        bad = True
    check("10 ops: XMOVE carries four values (the extent, base, size, rows used); other lengths are refused", ok and bad)
    check("10 settings: TF_GLM_KV_SHARD_CARRY is 1 unless 0; anything else is refused",
          multi.kv_carry("") is True and multi.kv_carry("1") is True and multi.kv_carry("0") is False)
    try:
        multi.kv_carry("yes"); bad = False
    except ValueError:
        bad = True
    check("10 settings: a typo in it is refused, not read as on", bad)
    check("10 settings: the minimum hit and the move cooldown are constants, and there is no count-based balance",
          multi.CARRY_MIN == 2048 and multi.RELOCATE_HOLD == 256 and not hasattr(multi, "CARRY_GAP")
          and not hasattr(multi, "kv_carry_min") and not hasattr(multi.MultiDecoder, "_carry_target"))


def health_carry():
    s = Car(2 * H, H)
    h = s._shard_health()
    want = {"hits", "rescued", "pressure", "others", "live", "tokens", "bytes", "ms_total", "ms_max", "refused"}
    check("10 health: kv_shard.carry has hits, rescued, pressure, others, live, tokens, bytes, ms_total, ms_max, refused",
          set(h["carry"]) == want)
    check("10 health: refused has pinned, twin, loading, partial, no_target, small, all 0 at start",
          h["carry"]["refused"] == dict(pinned=0, twin=0, loading=0, partial=0, no_target=0, small=0)
          and all(h["carry"][k] == 0 for k in want - {"refused"}))
    with carry_on():
        s, x, hit = rescue_layout()
        place(s, hit, 16 * A, [7] * (20 * A))
        s2, x2, lane = xlay(lane2=False)
        s2.carry = s.carry
        s2._relocate(lane)
        c = s._shard_health()["carry"]
        check("10 health: counters reported (rescued 1, live 1, tokens/bytes/ms of the carries)",
              c["rescued"] == 1 and c["live"] == 1 and c["tokens"] == lane.st.pos and c["bytes"] == lane.st.pos * 5808
              and c["ms_total"] >= 0 and c["ms_max"] >= 0 and c["hits"] == 0)
        s.carry["hits"] += 1
        check("10 health: the reported dict is a copy", s._shard_health()["carry"] is not s.carry
              and s._shard_health()["carry"]["refused"] is not s.carry["refused"])
section("10 ops and settings", ops_and_settings)
section("10 health", health_carry)


# -- Arena.carry on simulated ranks: byte equality, who sends, who receives ---------------------------------------------------
import collections
import ctypes
import math
import os
import tempfile

SH = 4 * 2048                    # a part: 4 pool blocks (rows)


class SimNccl:
    """ncclSend / ncclRecv between the simulated ranks of one process: bytes through FIFO mailboxes a (sender, receiver)
    pair, as NCCL matches a pair's calls in order; ``shard.RANK`` says which rank is calling. A receive with no send
    before it raises: the driver runs the sender first (the real ranks run side by side)."""
    comm = None

    def __init__(self):
        self.box = collections.defaultdict(collections.deque); self.log = []; self.spans = []; self.touched = []
    lib = property(lambda self: self)
    def _check(self, code): assert code == 0
    def _span(self, ptr, nbytes):
        if any(ptr < hi and lo < ptr + nbytes for lo, hi in self.spans):
            self.touched.append(("nccl", ptr))
    def ncclSend(self, ptr, count, dtype, peer, comm, stream):
        self._span(ptr, count * 4)
        self.box[(shard.RANK, peer)].append(ctypes.string_at(ptr, count * 4)); self.log.append(("send", shard.RANK, peer, count))
        return 0
    def ncclRecv(self, ptr, count, dtype, peer, comm, stream):
        self._span(ptr, count * 4)
        data = self.box[(peer, shard.RANK)].popleft()
        assert len(data) == count * 4, f"a receive of {count * 4} bytes meets a send of {len(data)}"
        ctypes.memmove(ptr, data, len(data)); self.log.append(("recv", shard.RANK, peer, count))
        return 0


def sim_arenas(world, seed=0):
    """One arena a rank over ``world`` parts of SH rows: a sharded FP8-like plane (uint8, 528 B a row), a sharded bf16
    plane, a sharded plane in the display span, the pooled keys (div 4, pad 2, 36 B) and a per-token plane every rank
    holds whole; replicated planes equal on every rank, sharded ones each rank's own."""
    rows = world * SH
    shard.setup(0, world, rows)
    out = []
    g = torch.Generator().manual_seed(seed)
    rep_pk = torch.randint(0, 256, (rows // 4 + 2, 36), generator=g, dtype=torch.uint8)
    rep_tok = torch.randn(rows, 8, generator=g).bfloat16()
    for r in range(world):
        planes = [Plane(torch.randint(0, 256, (SH + shard.TRASH, 528), generator=g, dtype=torch.uint8), 1, 0, shard=True),
                  Plane(torch.randn(SH + shard.TRASH, 64, generator=g).bfloat16(), 1, 0, shard=True),
                  Plane(torch.randint(0, 256, (SH + shard.TRASH, 528), generator=g, dtype=torch.uint8), 1, 0, spanned=True, shard=True),
                  Plane(rep_pk.clone(), 4, 2), Plane(rep_tok.clone(), 1, 0)]
        out.append(Arena(rows, planes))
    return out


def bytes_eq(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def carry_sim(world, src, dst, n, piece=None, order=None):
    """Carry n tokens from pool row src to dst on ``world`` simulated ranks: the arenas before and after, the transport."""
    ars = sim_arenas(world)
    before = [[p.tensor.clone() for p in ar.planes] for ar in ars]
    nc = SimNccl()
    for ar in ars:
        for p in ar.planes:
            if p.spanned:
                nc.spans.append((p.tensor.data_ptr(), p.tensor.data_ptr() + p.tensor.numel() * p.tensor.element_size()))
    saved = (shard._stream, shard._piece, torch.Tensor.copy_)
    shard._stream = lambda: 0
    if piece is not None:
        shard._piece = lambda rb: piece
    orig_copy = torch.Tensor.copy_

    def guarded(self, other, *a, **k):          # a memcpy on the display span would be what the span forbids
        for t in (self, other):
            if isinstance(t, torch.Tensor) and t.numel() and any(
                    t.data_ptr() < hi and lo < t.data_ptr() + t.numel() * t.element_size() for lo, hi in nc.spans):
                nc.touched.append(("copy_", t.data_ptr()))
        return orig_copy(self, other, *a, **k)
    torch.Tensor.copy_ = guarded
    so = shard.owner(src)
    try:
        for r in (order if order is not None else [so] + [r for r in range(world) if r != so]):
            shard.RANK = r
            ars[r].carry(src, dst, n, nc)
    finally:
        shard._stream, shard._piece = saved[0], saved[1]
        torch.Tensor.copy_ = orig_copy
        shard.RANK = 0
    return ars, before, nc


def expected(ars, before, world, src, dst, n):
    """What every rank's planes hold after: only the destination (its owner's sharded rows; every rank's replicated rows)."""
    so, do = shard.owner(src), shard.owner(dst)
    exp = [[t.clone() for t in b] for b in before]
    for r in range(world):
        for k, p in enumerate(ars[r].planes):
            if p.shard:
                if r == do:
                    exp[r][k][dst - do * SH:dst - do * SH + n] = before[so][k][src - so * SH:src - so * SH + n]
            else:
                m = -(-n // p.div)
                exp[r][k][dst // p.div:dst // p.div + m] = before[r][k][src // p.div:src // p.div + m]
    return exp


def carry_cases():
    shard.ON = False
    try:
        for world, src, dst, n, piece, label in (
                (2, 2048, SH + 4096, 3001, None, "two ranks, a cut not a multiple of 4 (the last pooled-key row partly filled)"),
                (2, 0, SH, 2048, 7, "two ranks, sent 7 rows at a time"),
                (2, SH + 2048, 4096, 5000, 1000, "two ranks, the other way (part 1 to part 0), 1000-row pieces"),
                (3, 4096, 2 * SH + 2048, 4097, 513, "TP3: part 0 to part 2, rank 1 a bystander"),
                (3, 2 * SH, 2048, 3000, None, "TP3: part 2 to part 0"),
                (3, SH + 2048, 2 * SH + 4096, 2049, 100, "TP3: part 1 to part 2, rank 0 a bystander")):
            ars, before, nc = carry_sim(world, src, dst, n, piece)
            exp = expected(ars, before, world, src, dst, n)
            so, do = shard.owner(src), shard.owner(dst)
            ok = all(bytes_eq(p.tensor, exp[r][k]) for r, ar in enumerate(ars) for k, p in enumerate(ar.planes))
            check(f"10 carry [{label}]: every plane of every rank is the source's rows at the destination, nothing else touched "
                  f"(FP8 bytes, bf16, the display span's plane, the pooled keys with a partial last row, the pad rows)", ok)
            sends = [e for e in nc.log if e[0] == "send"]; recvs = [e for e in nc.log if e[0] == "recv"]
            paired = (bool(sends) and all(e[1] == so and e[2] == do for e in sends) and all(e[1] == do and e[2] == so for e in recvs)
                      and [e[3] for e in sends] == [e[3] for e in recvs] and not any(nc.box.values()))
            check(f"10 carry [{label}]: only the source's owner sends (to the destination's), only the destination's receives "
                  f"(from it); every send meets its receive in order; a bystander does nothing", paired)
            planes = sum(1 for p in ars[0].planes if p.shard)
            if piece is None:
                check(f"10 carry [{label}]: one piece a sharded plane when the rows fit one send ({planes} sends)", len(sends) == planes)
            else:
                want = planes * -(-n // piece)
                check(f"10 carry [{label}]: pieces of {piece} rows ({want} sends)", len(sends) == want)
            check(f"10 carry [{label}]: the display span is reached only by kernel (no copy_, no transport on it)", not nc.touched)
            check(f"10 carry [{label}]: the source's rows stay", all(bytes_eq(ars[so].planes[k].tensor[src - so * SH:src - so * SH + n],
                                                                            before[so][k][src - so * SH:src - so * SH + n])
                                                                    for k, p in enumerate(ars[so].planes) if p.shard))
        # one part: a plain local copy, no transport
        ars, before, nc = carry_sim(2, 0, 4096, 3000)
        check("10 carry: within one part it is the local copy (nothing sent)", not nc.log and bytes_eq(
            ars[0].planes[0].tensor[4096:7096], before[0][0][0:3000]))
        ars, before, nc = carry_sim(2, 0, SH, 0)
        check("10 carry: zero tokens is a no-op", not nc.log and all(bytes_eq(p.tensor, before[r][k]) for r, ar in enumerate(ars) for k, p in enumerate(ar.planes)))
        try:
            carry_sim(2, 0, 2 * SH - 1000, 3000); bad = False
        except ValueError:
            bad = True
        check("10 carry: a destination past the arena is refused", bad)
        check("10 carry: sharded_bytes is the sharded planes' row bytes a token (528 + 128 + 528)",
              sim_arenas(2)[0].sharded_bytes(10) == 10 * (528 + 128 + 528))
    finally:
        shard.setup(0, 2, 4096 * 512)
section("10 carry sim", carry_cases)


# -- the spill tier: block names carry the part; a carried state restores byte for byte -----------------------------------
import numpy as np
import tensorfold.cuda.spill as SP
import tensorfold.cuda.spill_blocks as S

SB = 2048                                  # the pool's block
SPH = 12 * SB                              # a part of twelve blocks
SPR = 2 * SPH
SROOT = tempfile.mkdtemp(prefix="kv-shard-carry-")
_n = [0]


class SnapS:
    """A stand-in for decode.Snapshot (the fields the tier stores and the ones _SPILL_SKIP leaves out)."""


SCLASSES = [f"{SnapS.__module__}:{SnapS.__qualname__}"]


def snap_s(ids, lineage, extent):
    s = SnapS()
    s.ids, s.shared, s.lineage = list(ids), False, lineage
    s.rec, s.conv = torch.arange(64, dtype=torch.float32) + len(ids), torch.ones(8)
    s.tail = s.head = s.drafter_rows = None
    s.extent, s.key, s.images, s.own, s.chat, s.kid = extent, list(ids), False, True, 1, 0
    return s


def sp_arena(sharded):
    if sharded:
        shard.setup(0, 2, SPR)
        lat = torch.zeros((shard.HALF + shard.TRASH, 4), dtype=torch.int32)
        planes = [Plane(lat, 1, 0, shard=True)]
    else:
        planes = [Plane(torch.zeros((SPR, 4), dtype=torch.int32), 1, 0)]
    return Arena(SPR, planes + [Plane(torch.zeros((SPR // 4 + 2, 8), dtype=torch.int16), 4, 2)])


def sp_dec(ar, rank, sharded, nc=None):
    """A MultiDecoder with what the tier's paths use, one rank's view: a real Pool guarded by _guard_pool, a real
    BlockStore, the simulated transport as its comm."""
    _n[0] += 1
    cfg = SP.SpillConfig(root=os.path.join(SROOT, f"s{_n[0]}"), gib=1.0, min_tokens=1, min_free_gib=0.0)
    os.makedirs(cfg.root, exist_ok=True)
    shard.RANK = rank
    st = S.BlockStore(cfg, planes=ar.planes, block=SB, rank=rank, device="cpu", classes=SCLASSES, quiet=True)
    m = object.__new__(multi.MultiDecoder)
    m.pool = Pool(SPR, half=SPH if sharded else 0)
    m.arena, m.disk, m.rank = ar, st, rank
    m.w = SimpleNamespace(world=2, device="cpu"); m.g = SimpleNamespace(cache_entries=8, comm=nc)
    m.kept, m.lanes, m.partial, m.loads, m.outbox = [], {}, None, {}, []
    m.next_lid = m.next_kid = 0; m.load_turn = 0; m.broken = None; m.idle = False
    m.drafts = None; m.grid = 0; m.carry = multi.carry_stats(); m._flush = lambda: None
    m._guard_pool()
    return m


def sp_fill(ars, base, a, b, seed):
    """Tokens [a, b) (a a multiple of 4) of the extent at ``base``: the owner's latent rows, every rank's pooled keys; the
    rows written, to compare against."""
    g = torch.Generator().manual_seed(seed)
    own = shard.owner(base) if shard.ON or ars[0].planes[0].shard else 0
    lat = torch.randint(-2 ** 31, 2 ** 31 - 1, (b - a, 4), generator=g, dtype=torch.int32)
    pk = torch.randint(-2 ** 15, 2 ** 15 - 1, (-(-b // 4) - a // 4, 8), generator=g, dtype=torch.int16)
    for r, ar in enumerate(ars):
        if ar.planes[0].shard:
            if r == own:
                lo = base - own * SPH
                ar.planes[0].tensor[lo + a:lo + b] = lat
        else:
            ar.planes[0].tensor[base + a:base + b] = lat
        ar.planes[1].tensor[base // 4 + a // 4:base // 4 + a // 4 + pk.shape[0]] = pk
    return lat, pk


def sp_rows(ar, rank, base, n):
    """What rank ``rank`` holds of tokens [0, n) of the extent at ``base``: (latent rows or None, pooled-key rows)."""
    if ar.planes[0].shard:
        lat = ar.planes[0].tensor[base - rank * SPH:base - rank * SPH + n].clone() if shard.owner(base) == rank else None
    else:
        lat = ar.planes[0].tensor[base:base + n].clone()
    return lat, ar.planes[1].tensor[base // 4:base // 4 + -(-n // 4)].clone()


def sp_keep(decs, base, ids, lineage, size=None):
    """The state kept in a new extent of every rank's pool and written (queued) by every rank; the kept snapshots."""
    out = []
    for r, d in enumerate(decs):
        shard.RANK = r
        x = d.pool.add(base, size or align_up(len(ids)))
        c = snap_s(ids, lineage, x)
        x.kept.append(c); d.kept.append(c)
        d._persist(c)
        out.append(c)
    return out


def sp_drain(decs):
    for r, d in enumerate(decs):
        shard.RANK = r
        if not d.disk.drain(30):
            print("     (a write did not end)")


def sp_restore(decs, ars, ids, base, n):
    """Read the state of ``ids`` back into a new extent at ``base`` on every rank: (rows per rank, the points)."""
    out, pts = [], []
    key = S.point_key(ids)
    for r, (d, ar) in enumerate(zip(decs, ars)):
        shard.RANK = r
        x = d.pool.add(base, align_up(n))
        try:
            small, pt = d.disk.finish_load(d.disk.load_async(key, x))
        except Exception as exc:                    # noqa: BLE001
            print(f"     (rank {r}: the read failed: {exc!r})")
            out.append(None); pts.append(None)
            continue
        out.append(sp_rows(ar, r, base, n)); pts.append(pt)
    return out, pts


def same_rows(got, lat, pk, rank, owner):
    if got is None:
        return False
    glat, gpk = got
    return torch.equal(gpk, pk) and (glat is None if rank != owner else torch.equal(glat, lat))


def spill_names():
    L = "boot-3|boot-9@2048"
    ids = np.arange(5 * SB + 77, dtype=np.int64)
    n0 = [h for _, _, h in S.block_hashes(ids, SB, S.in_part(L, 0))]
    n1 = [h for _, _, h in S.block_hashes(ids, SB, S.in_part(L, 1))]
    check("10 spill: the part salts every block name (no block of the same ids in two parts shares a name)",
          len(n0) == len(n1) == 6 and not set(n0) & set(n1))
    check("10 spill: idempotent: salting a salted lineage replaces the part, never stacks",
          S.in_part(S.in_part(L, 0), 0) == S.in_part(L, 0) and S.in_part(S.in_part(L, 0), 1) == S.in_part(L, 1)
          and S.in_part(S.in_part(L, 1), 1).count("#") == 1)
    check("10 spill: the forks after the first computation are untouched, and forking a salted lineage keeps the salt at the head",
          S.in_part(L, 1) == "boot-3#1|boot-9@2048" and S.fork(S.in_part(L, 1), "x", 5000) == "boot-3#1|boot-9@2048|x@5000"
          and S.in_part(S.fork(S.in_part(L, 1), "x", 5000), 0) == "boot-3#0|boot-9@2048|x@5000" and S.in_part("", 1) == "#1")
    same = [h for _, _, h in S.block_hashes(ids, SB, S.in_part(L, 1))]
    check("10 spill: the same part gives the same names (a conversation's turns in one part still share their blocks)", same == n1)
section("10 spill names", spill_names)


def spill_unsharded():
    ar = sp_arena(False)
    d = sp_dec(ar, 0, False)
    ids = list(np.random.default_rng(3).integers(1, 150000, 3 * SB))
    sp_fill([ar], 0, 0, len(ids), 1)
    (c,) = sp_keep([d], 0, ids, "u-5|u-7@1024")
    sp_drain([d])
    pt = d.disk.index[S.point_key(ids)]
    check("10 spill: an unsharded store's lineage and block names are unchanged (no part in them, no half field)",
          d._lineage(c) == "u-5|u-7@1024" and pt.lineage == "u-5|u-7@1024" and "#" not in pt.lineage and not pt.fields
          and pt.blocks == [h for _, _, h in S.block_hashes(np.asarray(ids, dtype=np.int64), SB, "u-5|u-7@1024")])
section("10 spill unsharded", spill_unsharded)


def spill_carried_restore():
    n1, n2 = 2 * SB + 100, 3 * SB + 152
    ids2 = [int(v) for v in np.random.default_rng(5).integers(1, 150000, n2)]
    ids1 = ids2[:n1]
    for carried in (True, False):
        nc = SimNccl()
        saved = shard._stream
        shard._stream = lambda: 0
        try:
            ars = [sp_arena(True), sp_arena(True)]
            decs = [sp_dec(ars[r], r, True, nc) for r in range(2)]
            lat1, pk1 = sp_fill(ars, 0, 0, n1, 11)
            sp_keep(decs, 0, ids1, "boot-1")                           # turn 1, kept in part 0 and written
            sp_drain(decs)
            base2 = SPH if carried else 3 * SB                          # turn 2: carried into part 1, or beside it in part 0
            xs = []
            for r in (0, 1):                                            # (rank 0 first: it sends)
                shard.RANK = r
                xs.append(decs[r].pool.add(base2, align_up(n2)))
                if carried:
                    decs[r]._carry(0, base2, n1, "admit")
                else:
                    decs[r].arena.copy(0, base2, n1)                    # copy 1: within the part
            sp_fill(ars, base2, n1 - n1 % 4, n2, 12)                    # the stream's own rows from the resume point on
            lin2 = S.fork("boot-1", "boot-2", n1)
            for r in (0, 1):
                shard.RANK = r
                c = snap_s(ids2, lin2, xs[r])
                xs[r].kept.append(c); decs[r].kept.append(c); decs[r]._persist(c)
            sp_drain(decs)
            part = 1 if carried else 0
            tag = "carried into part 1" if carried else "in place in part 0, control"
            p1 = [d.disk.index[S.point_key(ids1)] for d in decs]
            p2 = [d.disk.index[S.point_key(ids2)] for d in decs]
            check(f"10 spill [{tag}]: every rank wrote the point under the part's lineage and recorded its half",
                  all(p.lineage == S.in_part(lin2, part) and p.fields.get("half") == part for p in p2)
                  and all(p.lineage == S.in_part("boot-1", 0) and p.fields.get("half") == 0 for p in p1))
            if carried:
                check(f"10 spill [{tag}]: its blocks are named for the new part: none shared with the turn it grew from",
                      not set(p2[0].blocks) & set(p1[0].blocks) and not set(p2[1].blocks) & set(p1[1].blocks))
            else:
                full = n1 // SB
                check(f"10 spill [{tag}]: a turn in the same part still names the blocks of the turn before it ({full} whole blocks)",
                      p2[0].blocks[:full] == p1[0].blocks[:full] and p2[1].blocks[:full] == p1[1].blocks[:full])
            rbase = SPH + 6 * SB if carried else 8 * SB
            got, pts = sp_restore(decs, ars, ids2, rbase, n2)
            want = [sp_rows(ars[r], r, base2, n2) for r in (0, 1)]
            ok = all(got[r] is not None and torch.equal(got[r][1], want[r][1]) and (got[r][0] is None) == (want[r][0] is None)
                     and (want[r][0] is None or torch.equal(got[r][0], want[r][0])) for r in (0, 1))
            check(f"10 spill [{tag}]: the state restores byte for byte on both ranks (the owner's latent rows, every rank's pooled keys)",
                  ok and all(p is not None and p.n == n2 for p in pts))
            if carried:
                check(f"10 spill [{tag}]: and its first {n1} tokens are what turn 1 wrote in part 0",
                      got[1] is not None and got[1][0] is not None and torch.equal(got[1][0][:n1], lat1)
                      and torch.equal(got[1][1][:n1 // 4], pk1[:n1 // 4]) and got[0][0] is None)
            lins = {pt.lineage for pt in pts if pt is not None}
            check(f"10 spill [{tag}]: the restored point keeps the salted lineage, and salting it for its part again changes nothing "
                  f"(a restored state persisted again names the same blocks)", lins == {S.in_part(lin2, part)}
                  and all(S.in_part(l, part) == l for l in lins))
        finally:
            shard._stream = saved
            shard.RANK = 0
            shard.setup(0, 2, 4096 * 512)
section("10 spill carried", spill_carried_restore)


def spill_pending_move():
    # a state waiting to be written when its extent moves to the other part: the tier copies its rows to the host first
    # (the old owner's), so what is written is what part 0 held, under part 0's name
    n = 3 * SB
    ids = [int(v) for v in np.random.default_rng(8).integers(1, 150000, n)]
    nc = SimNccl()
    saved = shard._stream
    shard._stream = lambda: 0
    try:
        ars = [sp_arena(True), sp_arena(True)]
        decs = [sp_dec(ars[r], r, True, nc) for r in range(2)]
        lat, pk = sp_fill(ars, 0, 0, n, 21)
        cs = sp_keep(decs, 0, ids, "boot-4")
        waiting = [len(d.disk.pending) for d in decs]
        for r in (0, 1):
            shard.RANK = r
            decs[r]._xmove(decs[r].pool.get(cs[r].extent.eid), SPH + 2 * SB, 4 * SB, n)
        sp_drain(decs)
        check("10 spill xmove: the writes still waiting when the extent changed part were released to the host first (both ranks)",
              all(w > 0 for w in waiting) and all(d.disk.stats["released_blocks"] > 0 for d in decs))
        check("10 spill xmove: the extent is in part 1 now and the point records the part its rows were written from (0)",
              all(d.pool.half_of(cs[r].extent) == 1 for r, d in enumerate(decs))
              and all(d.disk.index[S.point_key(ids)].fields.get("half") == 0 for d in decs)
              and all(d.disk.index[S.point_key(ids)].lineage == S.in_part("boot-4", 0) for d in decs))
        got, pts = sp_restore(decs, ars, ids, 7 * SB, n)
        check("10 spill xmove: restored into part 0 it is byte for byte what part 0's owner held before the move",
              got[0] is not None and got[1] is not None and torch.equal(got[0][0], lat) and torch.equal(got[0][1], pk)
              and got[1][0] is None and torch.equal(got[1][1], pk))
        # and the moved extent has the rows (the carry): its owner now holds them
        moved = [sp_rows(ars[r], r, SPH + 2 * SB, n) for r in (0, 1)]
        check("10 spill xmove: the moved extent holds the rows in part 1 (rank 1) and the pooled keys on both",
              moved[0][0] is None and torch.equal(moved[1][0], lat) and torch.equal(moved[0][1], pk) and torch.equal(moved[1][1], pk))
    finally:
        shard._stream = saved
        shard.RANK = 0
        shard.setup(0, 2, 4096 * 512)
section("10 spill xmove", spill_pending_move)


# -- review fixes: the time of a carry, TP3 warm-up, the lone setting, counting at the ADMIT -----------------------------------
import inspect
from tensorfold.families.glm5_next.cuda import engine as _engine
from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings


def carry_ms():
    # the time /health reports is the transfer's: the ranks that send or receive measure it, with three ranks or more the
    # longest of them is shared in one all-gather every rank joins; two ranks need none
    gathered = []
    def mk(world, rank, rows):
        s = Car(world * H, H)
        s.w = SimpleNamespace(world=world); s.rank = rank
        def gi(values):
            gathered.append((rank, list(values)))
            if rows is None:
                raise AssertionError("no all-gather at two ranks")
            return rows
        s.g = SimpleNamespace(comm=None, _gather_ints=gi)
        return s
    s = mk(3, 1, [[0], [0], [7000]])                                       # rank 1 owns neither part 0 nor part 2
    s._carry(0, 2 * H, 4096, "admit")
    check("10 carry time: a rank that owns neither part sends nothing to the gather but 0, and reports the longest transfer "
          "(7.0 ms) of the ranks that took part", gathered == [(1, [0])] and abs(s.carry["ms_total"] - 7.0) < 1e-9
          and abs(s.carry["ms_max"] - 7.0) < 1e-9)
    gathered.clear()
    s = mk(3, 0, [[4000], [0], [7000]])
    s._carry(0, 2 * H, 4096, "live")
    check("10 carry time: an involved rank shares what it measured, and reports the longest too",
          len(gathered) == 1 and gathered[0][0] == 0 and gathered[0][1][0] >= 0 and abs(s.carry["ms_total"] - 7.0) < 1e-9)
    gathered.clear()
    s = mk(2, 0, None)
    s._carry(0, H, 4096, "admit")
    check("10 carry time: with two ranks (both take part) it is the rank's own, with no all-gather", not gathered
          and s.carry["ms_total"] >= 0 and s.carry["hits"] == 1)
section("10 carry time", carry_ms)


def warm_up():
    from tensorfold.cuda.comm import _DTYPES

    class Rec:
        comm = None
        def __init__(self): self.log = []; self.lib = self
        def _check(self, code): assert code == 0
        def ncclSend(self, ptr, count, dtype, peer, comm, stream): self.log.append(("send", peer, count)); return 0
        def ncclRecv(self, ptr, count, dtype, peer, comm, stream): self.log.append(("recv", peer, count)); return 0
    saved = shard._stream
    shard._stream = lambda: 0
    try:
        for world in (2, 3, 4):
            shard.setup(0, world, world * 2048 * 4)
            logs = []
            for r in range(world):
                shard.RANK = r
                rec = Rec()
                shard.warm(rec, "cpu")
                logs.append(rec.log)
            pairs = [(a, b) for a in range(world) for b in range(a + 1, world)]
            if world == 2:
                check("10 warm-up: with two ranks there is nothing to warm (the fetch's sends are the first, as ever)", not any(logs))
                continue
            ok = True
            for a, b in pairs:
                ok &= [e[:2] for e in logs[a] if e[1] == b] == [("send", b), ("recv", b)]
                ok &= [e[:2] for e in logs[b] if e[1] == a] == [("recv", a), ("send", a)]
            check(f"10 warm-up: {world} ranks: every pair sends and receives once each way, the lower rank sending first, the "
                  f"higher receiving first (each call meets its partner's)", ok and all(e[2] == 4 for l in logs for e in l)
                  and all(len(logs[r]) == 2 * (world - 1) for r in range(world)))
            order = []
            for r in range(world):
                seen = []
                for e in logs[r]:
                    pr = tuple(sorted((r, e[1])))
                    if not seen or seen[-1] != pr:
                        seen.append(pr)
                order.append(seen)
            check(f"10 warm-up: {world} ranks: the pairs come in the same order on every rank (a, then b), none skipped",
                  all(o == [p for p in pairs if r in p] for r, o in enumerate(order)))
    finally:
        shard._stream = saved
        shard.RANK = 0
        shard.setup(0, 2, 4096 * 512)
    check("10 warm-up: the engine calls it once the shard is set up, on every rank",
          "shard.warm(self.comm" in inspect.getsource(_engine))
section("10 warm-up", warm_up)


def lone_setting():
    t = MultiSettings.from_env({})
    check("10 lone: unset, TF_GLM_MULTI_LONE is on in the settings (the review's finding)", t.lone is True)
    saved = shard.ON
    try:
        shard.ON = True
        u = multi.shard_tune(t)
        check("10 lone: with the shard on it is off, whatever the default (the others branch and its carry are not skipped)",
              u.lone is False and t.lone is True and MultiSettings.from_env({"TF_GLM_MULTI_LONE": "0"}).lone is False
              and multi.shard_tune(MultiSettings.from_env({"TF_GLM_MULTI_LONE": "0"})).lone is False)
        shard.ON = False
        check("10 lone: without the shard it is as set", multi.shard_tune(t) is t and multi.shard_tune(t).lone is True)
        shard.ON = True
        with carry_on(False):
            check("10 lone: TF_GLM_KV_SHARD_CARRY=0 leaves it as set (the decoder as before the carry)",
                  multi.shard_tune(t) is t and multi.shard_tune(t).lone is True)
        shard.ON = False
        shard.ON = True
        s = Car(2 * H, H); s.tune = multi.shard_tune(t)
        s.pool.add(0, 16 * A).owner = 1
        x = s.pool.add(16 * A, 16 * A)
        hit = put_state(s, x, 7, 8 * A, name="hit"); other = put_state(s, x, 9, 16 * A, name="other")
        with carry_on():
            base, copy, h = place(s, hit, 6 * A, [7] * (20 * A))
        check("10 lone: with LONE unset and the shard on the others branch runs: the shared rows are carried, the other "
              "conversation keeps its state", copy == 2 and s.carry["others"] == 1 and s.evicted == [] and other in x.kept)
    finally:
        shard.ON = saved
    check("10 lone: the decoder takes its settings through it", "shard_tune(" in inspect.getsource(multi.MultiDecoder.__init__))
    check("10 lone: and an explicit LONE=1 is still refused at start", "TF_GLM_MULTI_LONE" in inspect.getsource(_engine))
section("10 lone", lone_setting)


def counted_at_admit():
    class Boom(Adm):
        def _emit(self, op, payload):
            if op == multi.ADMIT:
                raise RuntimeError("the op could not be queued")
            return super()._emit(op, payload)
    def lay(cls, q_live):
        s = cls(2 * H, H)
        s.pool.add(0, 24 * A).owner = 1
        x = s.pool.add(24 * A, 8 * A); hit_state(s, x, 7, 8 * A, own=True)
        if q_live:
            s.pool.add(H, q_live * A).owner = 2
        return s
    with carry_on(), fake_state():
        for label, q_live, kind in (("free rows elsewhere", 0, "rescued"), ("rescue, evicting there", 17, "rescued")):
            s = lay(Boom, q_live)
            try:
                s.admit(Stream([7] * (14 * A), 4, None, True)); raised = False
            except RuntimeError:
                raised = True
            check(f"10 counters [{label}]: a placement whose ADMIT never got queued is not counted (decision made, "
                  f"carry_kind={s.carry_kind}, hits/rescued/pressure/others all 0)",
                  raised and s.carry_kind == kind and s.carry["rescued"] == s.carry["pressure"] == s.carry["others"] == 0)
            s = lay(Adm, q_live)
            s.admit(Stream([7] * (14 * A), 4, None, True))
            check(f"10 counters [{label}]: and counted once when it was", s.carry["rescued"] == 1 and s.carry["hits"] == 1
                  and s.carry["pressure"] == s.carry["others"] == 0)
section("10 counters at the ADMIT", counted_at_admit)


def no_import_per_piece():
    check("10 shard: the send and the receive of a piece import nothing (the dtype table is read once, at import)",
          all("import" not in inspect.getsource(f) for f in (shard._send, shard._recv)))
section("10 no import per piece", no_import_per_piece)


def tp4_guard():
    """(11, kv-shard-tp4-exp) Four ranks, the tp4 prompt settings and the carry on a ring all run with the shard; the
    warm-up opens no point-to-point connection on a ring (the neighbour links are the communicator's own) nor without
    the carry; the own-row front and microbatching reach the owner's rows through ``forward._sharded_attend``."""

    keys = shard.PROMPT_SETTINGS + ("TF_NCCL_RING", "TF_GLM_KV_SHARD_CARRY")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        check("11 guard: two, three and four ranks run", all(shard.unsupported("batched", n) is None for n in (2, 3, 4)))
        check("11 guard: serial verify still refused", shard.unsupported("serial", 4) is not None)
        for k in shard.PROMPT_SETTINGS:
            os.environ[k] = "1"
            check(f"11 guard: {k}=1 runs with the shard at four ranks", shard.unsupported("batched", 4) is None)
            os.environ.pop(k)
        os.environ["TF_NCCL_RING"] = "1"
        check("11 guard: the carry on a four-rank ring runs (relayed)", shard.unsupported("batched", 4) is None)
        os.environ.pop("TF_NCCL_RING")

        class NoSend:
            nccl = None

            def __getattr__(self, name):
                raise AssertionError(f"warm-up touched the communicator ({name})")

        saved_world = shard.WORLD

        def warm_touches():
            try:
                shard.warm(NoSend(), "cpu")
                return False
            except Exception:
                return True
        try:
            shard.WORLD = 4
            os.environ["TF_GLM_KV_SHARD_CARRY"] = "0"
            check("11 warm-up: carry off, four ranks: no point-to-point connection", not warm_touches())
            os.environ["TF_GLM_KV_SHARD_CARRY"] = "1"
            os.environ["TF_NCCL_RING"] = "1"
            check("11 warm-up: carry on, on a ring: none either (the relay uses the neighbour links)", not warm_touches())
            os.environ.pop("TF_NCCL_RING")
            check("11 warm-up: carry on, on a switch: the pairs' connections", warm_touches())
        finally:
            shard.WORLD = saved_world
        check("11 guard: the engine passes its rank count",
              "shard.unsupported(verify_kind(), world)" in inspect.getsource(_engine))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
section("11 tp4 guard", tp4_guard)


def exp_paths():
    """(13) The own-row front and microbatching with the shard, checked by structure (their kernels need a GPU)."""

    from tensorfold.families.glm5_next.cuda import forward as fw
    try:                                     # (the tp4 branch's own-row front; main has none)
        from tensorfold.families.glm5_next.cuda import own_front as of
    except ImportError:
        of = None
    rest = inspect.getsource(of.rest) if of is not None else None
    if rest is not None:
        check("13 own-front: the owner only writes the chunk's latents into the cache (sharded, not a twin)",
              "sharded = shard.ON and w.comm is not None and getattr(st, \"twin_base\", None) is None" in rest
              and "if not sharded or lc is not None:" in rest)
        check("13 own-front: sharded rows take their tokens from the shared pools, then the owner's rows through "
              "_sharded_attend with the stream's owner as root",
              "sparse.pool_tokens(pools[:R], pos_dev, tokens[:R], counts[:R])" in rest and "_sharded_attend(" in rest
              and "shard.owner(st.base)" in rest)
        check("13 own-front: the unsharded path is the tp4 one (dense and sparse attention on lc)",
              "dense_attention(qa[:dense], lc, pos_dev" in rest and "sparse_prompt(qa, lc, tokens[:R], counts[:R], ol, scale)" in rest)
    att = inspect.getsource(fw._sharded_attend)
    i, j, k2 = att.find("wait_stream(side)"), att.find("shard.fetch("), att.find("side.wait_stream(")
    check("13 fetch: ordered with the split's side stream (it waits for the side stream, then the side stream for it)",
          0 <= i < j < k2)
    check("13 fetch: the dsa_rows path and the own-row front share it",
          "_sharded_attend(" in inspect.getsource(fw._sharded_rows))
    sp = inspect.getsource(fw._shard_sparse)
    check("13 kernel: sharded rows use the unsharded path's sparse kernel choice (sparse_prompt where the base has it)",
          "_shard_sparse(qa, att, tok, counts, ol, scale, b.prefill)" in att and 'globals().get("sparse_prompt")' in sp
          and "latent.SPARSE_ONEPASS" in sp)

    class B:
        split = SimpleNamespace(stream="side")
    check("13 side stream: found on a row-split chunk, none without a split",
          fw._side_stream(B()) == "side" and fw._side_stream(SimpleNamespace()) is None
          and fw._side_stream(SimpleNamespace(split=None)) is None)
section("13 exp paths", exp_paths)


class SimRing:
    """A switchless ring's links for ``relay_rows``: ``links`` as the calling rank sees them (to the next rank: link r,
    this rank its rank 0; to the previous: link r - 1, its rank 1), and ``lib.ncclAllGather`` as a two-rank mailbox:
    a call posts its half and takes the partner's if already posted (the sender never needs the receiver's half; the
    ranks run in path order, so each receive finds its sender's piece). No ncclSend / ncclRecv exist here."""

    def __init__(self, world):
        import collections
        self.world, self.box, self.log, self.touched, self.spans = world, collections.defaultdict(collections.deque), [], [], []
        ring = self

        class AllGather:
            argtypes = None

            def __call__(self, send, recv, count, dtype, link, stream):
                import ctypes
                me = (send - recv) // (count * 4)
                assert me in (0, 1) and dtype == 2, (me, dtype)
                for ptr in (send, recv):
                    if any(lo <= ptr < hi for lo, hi in ring.spans):
                        ring.touched.append(("allgather", ptr))
                ring.box[(link, me)].append(ctypes.string_at(send, count * 4))
                got = bool(ring.box[(link, 1 - me)])
                if got:
                    data = ring.box[(link, 1 - me)].popleft()
                    ctypes.memmove(recv + (1 - me) * count * 4, data, len(data))
                ring.log.append(("ag", shard.RANK, link, me, count, got))
                return 0

        self.lib = type("Lib", (), {})()
        self.lib.ncclAllGather = AllGather()

        def no(*a, **k):
            raise AssertionError("a point-to-point send or receive on the ring")
        self.lib.ncclSend = self.lib.ncclRecv = self.lib.ncclBroadcast = no

    @property
    def links(self):
        r = shard.RANK
        return {1: ("L", r), -1: ("L", (r - 1) % self.world)}

    def _check(self, code):
        assert code == 0


def ring_carry_sim(world, src, dst, n, piece=None):
    ars = sim_arenas(world)
    before = [[p.tensor.clone() for p in ar.planes] for ar in ars]
    nc = SimRing(world)
    for ar in ars:
        for p in ar.planes:
            if p.spanned:
                nc.spans.append((p.tensor.data_ptr(), p.tensor.data_ptr() + p.tensor.numel() * p.tensor.element_size()))
    saved = (shard._stream, shard._piece, os.environ.get("TF_NCCL_RING"))
    shard._stream = lambda: 0
    if piece is not None:
        shard._piece = lambda rb: piece
    os.environ["TF_NCCL_RING"] = "1"
    so, do = shard.owner(src), shard.owner(dst)
    path = shard.ring_path(so, do, world)
    try:
        for r in path + [r for r in range(world) if r not in path]:
            shard.RANK = r
            ars[r].carry(src, dst, n, nc)
    finally:
        shard._stream, shard._piece = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("TF_NCCL_RING", None)
        else:
            os.environ["TF_NCCL_RING"] = saved[2]
        shard.RANK = 0
    return ars, before, nc, path


def ring_carry_cases():
    shard.ON = False
    try:
        check("12 ring path: neighbours one hop, the shorter way round, the opposite rank the way up",
              shard.ring_path(0, 1, 4) == [0, 1] and shard.ring_path(0, 3, 4) == [0, 3] and shard.ring_path(3, 0, 4) == [3, 0]
              and shard.ring_path(0, 2, 4) == [0, 1, 2] and shard.ring_path(2, 0, 4) == [2, 3, 0]
              and shard.ring_path(1, 3, 4) == [1, 2, 3] and shard.ring_path(0, 2, 3) == [0, 2] and shard.ring_path(1, 0, 3) == [1, 0])
        for world, src, dst, n, piece, label in (
                (4, 2048, SH + 4096, 3001, None, "part 0 to its neighbour 1"),
                (4, 4096, 2 * SH + 2048, 4097, 513, "part 0 to the opposite part 2 (relayed by rank 1), 513-row pieces"),
                (4, 2 * SH + 1024, 4096, 2500, 700, "part 2 to the opposite part 0 (the way up: relayed by rank 3)"),
                (4, 3 * SH, 2048, 2049, None, "part 3 to part 0 across the ring's seam"),
                (4, SH + 2048, 3 * SH + 4096, 3000, 1000, "part 1 to the opposite part 3 (relayed by rank 2)"),
                (4, 4096, 3 * SH + 2048, 2000, 7, "part 0 down to part 3, 7 rows a piece"),
                (3, 4096, 2 * SH + 2048, 3000, 999, "TP3 ring: part 0 to 2, one hop down")):
            ars, before, nc, path = ring_carry_sim(world, src, dst, n, piece)
            exp = expected(ars, before, world, src, dst, n)
            ok = all(bytes_eq(p.tensor, exp[r][k]) for r, ar in enumerate(ars) for k, p in enumerate(ar.planes))
            check(f"12 ring carry [{label}]: every plane of every rank is the source's rows at the destination, nothing "
                  f"else touched (relays included)", ok)
            planes = sum(1 for p in ars[0].planes if p.shard)
            pieces = -(-n // (piece or n))
            calls = {r: [e for e in nc.log if e[1] == r] for r in range(world)}
            want = {r: (0 if r not in path else (1 if r in (path[0], path[-1]) else 2) * planes * pieces) for r in range(world)}
            check(f"12 ring carry [{label}]: only the path's ranks call ({path}), each hop once a piece and plane "
                  f"(ends one hop, relays two); no point-to-point", all(len(calls[r]) == want[r] for r in range(world)))
            recv = [e for e in nc.log if e[5]]
            check(f"12 ring carry [{label}]: every hop's receiver found its sender's piece ({len(recv)} = hops x pieces x planes)",
                  len(recv) == (len(path) - 1) * pieces * planes and not any(e[1] == path[0] for e in recv))
            check(f"12 ring carry [{label}]: the display span is reached only by kernel", not nc.touched)
        os.environ["TF_NCCL_RING"] = "1"
        shard.setup(0, 4, 4 * SH)
        check("12 relayed: a four-rank ring relays, two ranks never do", shard.relayed())
        check("12 relayed: only over a communicator with the ring links (main's has none: direct sends, as before)",
              shard.relayed(SimpleNamespace(links={1: "next", -1: "prev"})) and not shard.relayed(SimpleNamespace())
              and not shard.relayed(SimpleNamespace(links={})))
        shard.setup(0, 2, 2 * SH)
        check("12 relayed: (two ranks: no)", not shard.relayed())
        os.environ.pop("TF_NCCL_RING")
        shard.setup(0, 4, 4 * SH)
        check("12 relayed: four ranks on a switch send straight", not shard.relayed())
    finally:
        os.environ.pop("TF_NCCL_RING", None)
        shard.setup(0, 2, 4096 * 512)
section("12 ring carry", ring_carry_cases)
print("FAILED" if fails else "ALL PASS", len(fails))
sys.stdout.flush()
os._exit(1 if fails else 0)
