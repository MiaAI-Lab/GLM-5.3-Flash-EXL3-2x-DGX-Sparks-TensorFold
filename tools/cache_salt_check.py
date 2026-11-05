#!/usr/bin/env python3
"""Patch 0130's checks (cache_salt: vLLM's per-request prefix-cache isolation, which RigMark's cold
prefill relies on), CPU only, self-contained (the fork-point check's harness inlined: its upstream file is not merged).

The parallel engine through two real ``MultiDecoder``s (rank 1 applying rank 0's messages in-process, the model
stubbed), the serial engine through two real ``GlmEngine.generate`` / ``follow_one`` with the model stubbed the same
way, and the HTTP side's ``GlmApp`` field check and request hand-off:

  1. app: ``cache_salt`` absent / null passes, a non-empty string of at most 256 characters passes, "" / 5 / [] / {}
     and a string longer than 256 are refused by ``_check_fields`` (before a stream opens), and ``run`` hands the
     salt (or None) to ``engine.request``.
  2. parallel: a prompt kept unsalted resumes unsalted (the control), not under a salt; kept under salt A it resumes
     under A only (not B, not unsalted); the identical prompt under a fresh salt prefills from 0 (RigMark's cold).
     Both ranks keep the same states, rank 1's salted ones marked "" (salted, salt unknown there).
  3. parallel + NVMe tier (a recording stand-in on both ranks; only in a tree with PR #118's kvtier.py, required when
     CACHE_SALT_TIER=1): unsalted kept states are written, salted ones never (on either rank); a salted request never
     looks a state up on the tier, an unsalted one does.
  4. parallel + --spill-gib: a salted state is unspillable ("salted"); a salted request is never prestaged from the
     store, an unsalted one is looked up.
  5. parallel: shared-prefix points come only from kept prompts of the request's own salt.
  6. serial: the same as 2 through ``generate`` (rank 0) and ``follow_one`` (rank 1, reading the header's salted flag),
     a salted state is ``_unwritable`` ("salted") for the spill tier, ``_shared`` reads only its own salt's prompts.
  7. both engines: a prompt kept salted and the same prompt kept unsalted both stay (neither resumes the other, so
     replacing one with the other only lost a state): an unsalted replay after a salted request of the same ids
     resumes as before, and the other way round; two salts of the same ids still replace each other (rank 1 knows
     only that a state is salted); both ranks keep the same states in the same order, rank 1's serial resume taking
     the state as salted as rank 0's.
  8. routes: a /v1/responses request's ``cache_salt`` reaches the chat completion that runs it (and a malformed one
     is still refused by the app's check), and /v1/messages - whose protocol has no salt - refuses it instead of
     running the request silently unsalted.

Prints ok / FAIL lines and "all passed"; exit 1 when a check fails."""
import sys as _sys  # an uncaught raise prints a FAIL line (a kill needs one)
import traceback as _tb

_sys.excepthook = lambda t, v, tb: (_tb.print_exception(t, v, tb), print(f"FAIL raised {t.__name__}: {v}", flush=True))
import collections  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

os.environ.setdefault("TF_GLM_MULTI_WATCHDOG_S", "0")
import torch  # noqa: E402

import tensorfold.families.glm5_next.cuda.decode as D  # noqa: E402
import tensorfold.families.glm5_next.cuda.engine as ENG  # noqa: E402
import tensorfold.families.glm5_next.cuda.forward as F  # noqa: E402
from tensorfold.server.errors import RequestError  # noqa: E402
from tensorfold.server import anthropic_translate, responses_translate  # noqa: E402
import tensorfold.families.glm5_next.cuda.multi as M  # noqa: E402
from tensorfold.cuda.server import App  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.glm5_next.cuda.app import GlmApp, cache_salt  # noqa: E402
from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings  # noqa: E402
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane  # noqa: E402

TIMEOUT = float(os.environ.get("TEST_TIMEOUT_S", "300"))
_watch = threading.Timer(TIMEOUT, lambda: (print(f"FAIL timed out after {TIMEOUT:g} s", flush=True), os._exit(1)))
_watch.daemon = True
_watch.start()

# -- the two ranks, the model stubbed (the harness of the fork-point check, tools/fork_point_check.py upstream) ------
GRID, LEAST, ENTRIES = 64, 256, 32     # the recipe's grid, shared-prefix minimum and kept-state cap
GMASK, SOP, SYSTEM, USER, ASSISTANT, THINK, EOS = 151331, 151333, 151335, 151336, 151337, 151350, 151329
REPLY = 7                              # the stubbed sampler's token
POOL = 512 * ALIGN
fails: list[str] = []


def check(name: str, ok) -> None:
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


class SlotState:
    """``forward.State``: only the slot's position matters to the bookkeeping."""

    def __init__(self, *a, **k):
        self.pos, self.cur, self.rec, self.conv = 0, [0], [torch.zeros(1)], torch.zeros(1)

    def reset(self):
        self.pos = 0

    def set_pos(self, n):
        self.pos = n

    def mark(self, point):           # ``State.mark`` (where a chunk keeps a state inside it), in trees that have it
        pass


def prefill_chunk(e, prompt, start, rows, **k):
    e.st.pos = start + rows
    return torch.zeros(1, 4)


def take_snapshot(e, ids, pending, *, mtp, drafter=None, inner=False):
    return D.Snapshot(list(ids), torch.zeros(1), torch.zeros(1), None, -1, -1, tail=[])


F.State, D.prefill_chunk, D.take_snapshot = SlotState, prefill_chunk, take_snapshot
D.put_ring_tail = lambda st, hit: None
M.sample_streams = lambda w, parts: [[REPLY] for _ in parts]


class Engine:
    def __init__(self, rows=POOL):
        self.w = NS(device="cpu", cfg=NS(eos=[EOS], hidden=8), vocab_offset=0, comm=None, layers=[None], world=2)
        self.slots, self.rows, self.prefill_rows, self.st = NS(count=4), 16, 2048, None
        self.caches = NS(rows=rows, arena=Arena(rows, [Plane(torch.zeros(rows, 1, dtype=torch.uint8))]))

    def use(self, st):
        prev, self.st = self.st, st
        return prev


class Net:
    follower = None


class Glm:
    """The ``GlmEngine`` the decoder reads (its settings) and talks through (rank 0 to rank 1, in-process)."""

    def __init__(self, rank, net):
        self.rank, self.net, self.e = rank, net, Engine()
        self.w = self.e.w
        self.grid, self.drafter, self.limit, self.serial_only = GRID, None, 1 << 20, False
        self.shared, self.opener, self.cache_entries, self.copy, self.spill_cfg = LEAST, USER, ENTRIES, None, None
        self.multi_window, self.costs, self.model_dir = 64, None, "."
        self.vision = None

    def _ring(self):
        pass

    def _share(self, msg):
        f = self.net.follower
        f.apply(M.unseal(msg, f.received, 1))
        f.received = (f.received + 1) % M.SEAL_MOD


def ranks():
    net = Net()                                  # the recipe's TF_GLM_MULTI_LONE=0 (scripts/config.sh)
    tune = MultiSettings.from_env({"TF_GLM_MULTI_LONE": "0"})
    r0, r1 = (M.MultiDecoder(Glm(r, net), 4, verify=object(), tune=tune) for r in (0, 1))
    net.follower = r1
    return r0, r1


rng = random.Random(120)


def text(n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def system_block(n):
    return [GMASK, SOP, SYSTEM] + text(n - 3)


def turn(msg):
    return [USER] + msg + [ASSISTANT, THINK]


def prompt_of(n_system: int, k: int) -> list[int]:
    return system_block(n_system) + turn(text(k))


# -- 1: the HTTP side ---------------------------------------------------------------------------------------------
def app_side():
    eng = NS(generate=lambda prompt, max_tokens, sampling, on_tokens, draft=True: None, request=NS())
    app = object.__new__(GlmApp)
    app.engine = eng
    for body, ok in (({}, True), ({"cache_salt": None}, True), ({"cache_salt": "s" * 43}, True),
                     ({"cache_salt": "s" * 256}, True), ({"cache_salt": "s" * 257}, False),
                     ({"cache_salt": ""}, False), ({"cache_salt": 5}, False), ({"cache_salt": []}, False),
                     ({"cache_salt": {}}, False)):
        problem = app._check_fields(dict(body, prompt=[1, 2, 3]))
        check(f"app: cache_salt {body.get('cache_salt', '(absent)')!r} {'passes' if ok else 'is refused'} "
              f"({problem!r})", (problem is None) == ok and (ok or "cache_salt" in str(problem)))
    check("app: cache_salt() returns the salt", cache_salt({"cache_salt": "abc"}) == "abc" and cache_salt({}) is None)
    saved = App.run
    try:
        App.run = lambda self, body, chat, emit, prepared=None, cancelled=None: {}
        app.template = NS()
        app.run({"prompt": [1], "cache_salt": "xyz"}, False, lambda d: True)
        first = eng.request.cache_salt
        app.run({"prompt": [1]}, False, lambda d: True)
        check(f"app: run hands the salt to engine.request, and None after an unsalted request ({first!r}, "
              f"{eng.request.cache_salt!r})", first == "xyz" and eng.request.cache_salt is None)
    finally:
        App.run = saved


# -- 8: the translated routes --------------------------------------------------------------------------------------
class _Store:                                   # the Responses translator's store, as test_responses_include's
    def conversation(self, _id):
        return []


def routes():
    body = {"model": "x", "store": False,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Say OK."}]}]}
    chat = responses_translate.translate({**body, "cache_salt": "A"}, _Store()).chat
    check(f"routes: a Responses request's salt reaches its chat completion ({chat.get('cache_salt')!r})",
          chat.get("cache_salt") == "A")
    chat = responses_translate.translate({**body, "cache_salt": 5}, _Store()).chat     # malformed: refused by the app
    app = object.__new__(GlmApp)
    app.engine = NS(request=NS())
    check(f"routes: a Responses request's malformed salt is refused by the app, not the translator "
          f"({app._check_fields(dict(chat, prompt=[1, 2, 3]))!r})",
          "cache_salt" in str(app._check_fields(dict(chat, prompt=[1, 2, 3]))))
    check("routes: a Responses request without a salt carries none",
          "cache_salt" not in responses_translate.translate(body, _Store()).chat)
    messages = {"model": "x", "max_tokens": 8, "messages": [{"role": "user", "content": "Say OK."}]}
    try:
        anthropic_translate.translate({**messages, "cache_salt": "A"})
        refused = None
    except RequestError as exc:
        refused = str(exc)
    check(f"routes: /v1/messages refuses a salt instead of running it unsalted ({refused!r})",
          refused is not None and "cache_salt" in refused)
    check("routes: /v1/messages without a salt translates as before",
          "cache_salt" not in anthropic_translate.translate(messages))


# -- the parallel engine --------------------------------------------------------------------------------------------
def run_salted(r0, prompt, salt):
    """Admit one prompt under ``salt`` (None: unsalted), run it to its first token; what it resumed."""
    s = Stream(list(prompt), 1, None, draft=True, stop_eos=True, vision=None)
    s.glm = {"salt": salt}
    r0.admit(s)
    while not s.done:
        r0.finish(r0.round())
    return s.cached


def salts_on_ranks(r0, r1):
    return ([(len(c.ids), getattr(c, "salt", None) is not None) for c in r0.kept]
            == [(len(c.ids), getattr(c, "salt", None) is not None) for c in r1.kept])


def parallel_isolation():
    r0, r1 = ranks()
    p = prompt_of(700, 3)                        # 705 ids: its own state kept at the grid point 704
    point = len(p) // GRID * GRID
    longer = p + [REPLY] + text(40)          # goes on past that point: resumes 704 when allowed
    base = run_salted(r0, p, None)
    control = run_salted(r0, longer, None)
    check(f"parallel: unsalted, a longer prompt resumes the unsalted state ({base}, {control}; point {point})",
          base == 0 and control == point)
    a = run_salted(r0, longer, "A")
    check(f"parallel: under salt A the unsalted state is not resumed ({a})", a == 0)
    a2 = run_salted(r0, longer + text(70), "A")
    check(f"parallel: under salt A again, A's own state resumes ({a2}, its state {len(longer) // GRID * GRID})",
          a2 == len(longer) // GRID * GRID)
    b = run_salted(r0, longer + text(70), "B")
    check(f"parallel: under salt B, A's state is not resumed ({b})", b == 0)
    q = prompt_of(900, 5)                        # RigMark's pair: the same prompt cold then replayed, one salt
    cold = run_salted(r0, q, "pair-1")
    warm = run_salted(r0, q, "pair-1")           # an immediate replay: resumes its kept grid point
    cold2 = run_salted(r0, q, "pair-2")
    check(f"parallel: one salt per pair: cold {cold}, replay {warm}, the same prompt under a fresh salt {cold2}",
          cold == 0 and warm == len(q) // GRID * GRID and cold2 == 0)
    r = prompt_of(1016, 5)                       # 1,024 ids, on the grid: a replay resumes the whole prompt
    plain = [run_salted(r0, r, None), run_salted(r0, r, None)]
    r2 = prompt_of(1016, 5)
    salted = [run_salted(r0, r2, "w"), run_salted(r0, r2, "w"), run_salted(r0, r2, "v")]
    check(f"parallel: an on-grid prompt replays as the unsalted control does ({plain}), under one salt alike and "
          f"cold under another ({salted})", plain[0] == 0 and plain[1] > 0 and salted == [0, plain[1], 0])
    un = run_salted(r0, q + text(64), None)
    check(f"parallel: an unsalted request resumes no salted state ({un})", un == 0)
    check("parallel: both ranks keep the same states, salted alike", salts_on_ranks(r0, r1))
    check("parallel: rank 1 marks a salted state '' (salted, the salt unknown there), rank 0 keeps the salt",
          {getattr(c, "salt", None) for c in r1.kept} <= {None, ""} and "" in {getattr(c, "salt", None)
                                                                                for c in r1.kept}
          and {"A", "B", "pair-1", "pair-2"} & {getattr(c, "salt", None) for c in r0.kept})


def parallel_same_ids():
    r0, r1 = ranks()
    r = prompt_of(1016, 5)                       # 1,024 ids, on the grid: a replay resumes the whole prompt
    got = [run_salted(r0, r, None), run_salted(r0, r, "s"), run_salted(r0, r, None), run_salted(r0, r, "s")]
    check(f"parallel: kept salted and unsalted, the same ids: each replay resumes its own ({got})",
          got == [0, 0, len(r), len(r)])
    q = prompt_of(1016, 5)
    two = [run_salted(r0, q, "a"), run_salted(r0, q, "b"), run_salted(r0, q, "a")]
    check(f"parallel: two salts of the same ids replace each other, as on rank 1 ({two})", two == [0, 0, 0])
    check("parallel: both ranks keep the same states, salted alike (the same ids kept both ways)",
          salts_on_ranks(r0, r1) and [c.kid for c in r0.kept] == [c.kid for c in r1.kept])


class TierRec:
    """The NVMe tier, recording: every state queued for disk, every lookup; it holds nothing (``best`` -> None)."""

    pending = False
    gain, least, head_saved = 1, 1, True

    def __init__(self):
        self.persisted, self.looked = [], []

    def wants(self, n):
        return True

    def persist(self, c, arena, extent):
        self.persisted.append((len(c.ids), getattr(c, "salt", None)))

    def best(self, prompt, least, grid, head_saved):
        self.looked.append(len(prompt))
        return None

    def release(self, *a):
        pass

    def pump(self, *a):
        pass

    def drain(self, *a):
        pass

    def health(self):
        return {}


def parallel_tier():
    if importlib.util.find_spec("tensorfold.families.glm5_next.cuda.kvtier") is None:
        # main has no NVMe tier (kvtier.py, PR #118's patch 0109); CACHE_SALT_TIER=1 makes this a failure
        check("tier: no NVMe tier in this tree (main; CACHE_SALT_TIER makes its absence a failure)",
              os.environ.get("CACHE_SALT_TIER") != "1")
        print("skip tier: no NVMe tier in this tree")
        return
    r0, r1 = ranks()
    r0.tier, r1.tier = TierRec(), TierRec()
    p = prompt_of(700, 3)
    run_salted(r0, p, None)
    unsalted_written = [t.persisted[:] for t in (r0.tier, r1.tier)]
    looked = len(r0.tier.looked)
    run_salted(r0, prompt_of(800, 3), "A")
    check(f"tier: an unsalted state is written on both ranks ({unsalted_written})",
          all(w and all(s is None for _, s in w) for w in unsalted_written))
    check(f"tier: a salted state is written on neither rank ({r0.tier.persisted}, {r1.tier.persisted})",
          r0.tier.persisted == unsalted_written[0] and r1.tier.persisted == unsalted_written[1])
    check(f"tier: the unsalted request looked the tier up, the salted one did not ({r0.tier.looked})",
          looked == 1 and len(r0.tier.looked) == 1)


class DiskRec:
    """--spill-gib's store, recording lookups; it holds nothing. ``find`` is the store's lookup on main (``0088``);
    ``wants`` / ``best`` / ``stats`` are the block store's (PR #124), so the check runs on either tree."""

    cfg = NS(min_tokens=1)

    def __init__(self):
        self.found = 0
        self.stats = collections.Counter()

    def wants(self, n):
        return n >= self.cfg.min_tokens

    def find(self, prompt, grid):
        self.found += 1
        return None

    def best(self, prompt, have, grid):
        self.found += 1
        return None


def parallel_spill():
    r0, r1 = ranks()
    p = prompt_of(700, 3)
    run_salted(r0, p, "A")
    run_salted(r0, prompt_of(700, 3), None)
    disk = DiskRec()
    r0.disk, r0.loads = disk, {}                 # (``_spill_store`` makes ``loads`` with a store)
    salted = [c for c in r0.kept if getattr(c, "salt", None) is not None]
    plain = [c for c in r0.kept if getattr(c, "salt", None) is None]
    check(f"spill: a salted state is unspillable as 'salted' ({[r0._unspillable(c) for c in salted]})",
          salted and all(r0._unspillable(c) == "salted" for c in salted))
    check(f"spill: an unsalted one is not refused for a salt ({[r0._unspillable(c) for c in plain]})",
          plain and all(r0._unspillable(c) != "salted" for c in plain))
    s = Stream(list(p), 1, None, draft=True, stop_eos=True, vision=None)
    s.glm = {"salt": "A"}
    got_salted = r0.prestage(s)
    salted_finds = disk.found
    s2 = Stream(list(p), 1, None, draft=True, stop_eos=True, vision=None)
    s2.glm = {}
    r0.prestage(s2)
    check(f"spill: a salted request is not prestaged from the store, an unsalted one is looked up "
          f"({salted_finds}, {disk.found})", got_salted is None and salted_finds == 0 and disk.found == 1)


def parallel_shared():
    r0, r1 = ranks()
    head = system_block(1200)
    run_salted(r0, head + turn(text(500)), "A")     # kept: its 1,792-id state; shares 1,200+ ids with what follows
    known = []
    real = ENG.shared_points

    def spy(prompt, cut, point, grid, least, k, opener=None):
        known.append(len(k))
        return real(prompt, cut, point, grid, least, k, opener)

    ENG.shared_points = spy
    try:
        run_salted(r0, head + turn(text(600)), "B")
        run_salted(r0, head + turn(text(650)), "A")
    finally:
        ENG.shared_points = real
    check(f"shared: salt B's request sees no kept prompt of salt A, A's sees its own ({known})",
          known[0] == 0 and known[1] >= 1)


# -- the serial engine ----------------------------------------------------------------------------------------------
def serial_prefill(e, prompt, sampling, *, mtp=True, drafter=None, resume=None, feed=None, sample=True,
                   keep_head=False):
    if keep_head:
        e.head = torch.zeros(1)
    return REPLY


def serial_snapshot(e, ids, pending, *, mtp, drafter=None, inner=False):
    return D.Snapshot(list(ids), torch.zeros(1), torch.zeros(1), None, 0, -1, tail=[])


def serial_engine(rank, wire):
    g = object.__new__(ENG.GlmEngine)
    g.rank, g.limit, g.serial_only, g.policy, g.drafter, g.vision = rank, 1 << 20, False, "0", None, None
    g.w = NS(mtp=None, cfg=NS(quant="exl3"))
    g.request = NS(policy=None, stop_eos=True, cache_salt=None)
    g.scheduler, g.spill, g.grid, g.shared, g.opener, g.cache = None, None, GRID, 0, None, []
    g.cache_entries, g.cache_bytes, g.eos, g.costs, g.copy, g.dump = 64, 1 << 40, [EOS], None, None, None
    g.e = NS(constraint=None, window=None, vote=None, head=None, last_hidden=None)
    g.comm, g.live = object(), []
    g._ring = lambda: None
    g._take_over = lambda keep: None
    g._await_bell = lambda: None
    if rank == 0:
        g._share = lambda msg: wire.append(list(msg))
    else:
        g._share = lambda msg: wire.pop(0)
    return g


def serial():
    saved = (D.prefill, D.take_snapshot)
    D.prefill, D.take_snapshot = serial_prefill, serial_snapshot
    try:
        wire: list = []
        g0, g1 = serial_engine(0, wire), serial_engine(1, wire)

        def gen(prompt, salt):
            g0.request.cache_salt = salt
            stats = g0.generate(list(prompt), 1, None, lambda t: None)
            g1.follow_one()
            return stats["cached"]

        p = prompt_of(700, 3)
        point = len(p) // GRID * GRID
        longer = p + [REPLY] + text(40)
        got = [gen(p, None), gen(longer, None), gen(longer, "A"), gen(longer + text(70), "A"),
               gen(longer + text(70), "B")]
        check(f"serial: unsalted resumes unsalted; A not unsalted; A its own; B not A's ({got})",
              got == [0, point, 0, len(longer) // GRID * GRID, 0])
        q = prompt_of(900, 5)
        pair = [gen(q, "pair-1"), gen(q, "pair-1"), gen(q, "pair-2"), gen(q, None)]
        check(f"serial: one salt per pair: cold, the replay resumes its grid point, a fresh salt cold, unsalted cold "
              f"({pair})", pair == [0, len(q) // GRID * GRID, 0, 0])
        r = prompt_of(1016, 5)                   # 1,024 ids, on the grid: a whole replay (its head's row kept)
        whole = [gen(r, "w"), gen(r, "w"), gen(r, "v"), gen(r, None), gen(r, None)]
        check(f"serial: an on-grid prompt replays whole under its salt only, as unsalted does ({whole})",
              whole == [0, len(r), 0, 0, len(r)])
        check("serial: rank 1 keeps the same states, salted ones marked ''",
              [(len(c.ids), c.salt is not None) for c in g0.cache] == [(len(c.ids), c.salt is not None)
                                                                      for c in g1.cache]
              and {c.salt for c in g1.cache} == {None, ""})
        check(f"serial: the wire is drained ({len(wire)} left)", not wire)
        w2: list = []
        h0, h1 = serial_engine(0, w2), serial_engine(1, w2)

        def gen2(prompt, salt):
            h0.request.cache_salt = salt
            stats = h0.generate(list(prompt), 1, None, lambda t: None)
            h1.follow_one()
            return stats["cached"]

        r2 = prompt_of(1016, 5)
        both = [gen2(r2, None), gen2(r2, "s"), gen2(r2, "s"), gen2(r2, None), gen2(r2, "s")]
        check(f"serial: kept salted and unsalted, the same ids: each replay resumes its own ({both})",
              both == [0, 0, len(r2), len(r2), len(r2)])
        check("serial: both ranks keep the same states in the same order (rank 1 resumes the state as salted as "
              "rank 0's)", [(len(c.ids), c.salt is not None) for c in h0.cache]
              == [(len(c.ids), c.salt is not None) for c in h1.cache] and len(h0.cache) == 2)
        g0.live = []
        salted = [c for c in g0.cache if c.salt is not None]
        plain = [c for c in g0.cache if c.salt is None]
        check(f"serial: a salted state is unwritable as 'salted' ({[g0._unwritable(c) for c in salted]})",
              salted and all(g0._unwritable(c) == "salted" for c in salted))
        check(f"serial: an unsalted one is not refused for a salt ({[g0._unwritable(c) for c in plain]})",
              plain and all(g0._unwritable(c) != "salted" for c in plain))
        g0.shared = 256
        head = system_block(1200)
        g0.cache = [serial_snapshot(None, head + text(600), None, mtp=False)]
        g0.cache[0].salt = "A"
        probe = head + turn(text(300))
        check(f"serial: shared points from the own salt's prompts only ({g0._shared(probe, None, 'A')} under A, "
              f"{g0._shared(probe, None, 'B')} under B)",
              g0._shared(probe, None, "A") and g0._shared(probe, None, "B") == [])
        g0.shared, g0.cache, g1.cache = 0, [], []
        finds: list = []
        g0.spill = NS(cfg=NS(min_tokens=1 << 30), has=lambda ids: False,
                      find=lambda prompt, grid, usable: finds.append(len(prompt)))
        g0._spill_wait = lambda: None
        gen(prompt_of(500, 3), "S")
        gen(prompt_of(510, 3), None)
        check(f"serial: a salted request looks nothing up in --spill-gib's store, an unsalted one does ({finds})",
              finds == [len(prompt_of(510, 3))])
        g0.spill = None
        sent: list = []
        g0.scheduler = NS(submit=lambda *a, **k: (sent.append(k.get("glm")), {})[1])
        g0.request.cache_salt = "Z"
        g0.generate([1, 2, 3], 1, None, lambda t: None)
        g0.request.cache_salt = None
        g0.generate([1, 2, 3], 1, None, lambda t: None)
        check(f"--parallel: generate hands the salt to the scheduler's stream ({sent})",
              [g.get("salt") for g in sent] == ["Z", None])
    finally:
        D.prefill, D.take_snapshot = saved


def main() -> int:
    for part in (app_side, routes, parallel_isolation, parallel_same_ids, parallel_tier, parallel_spill,
                 parallel_shared, serial):
        try:
            part()
        except Exception as exc:                 # a part that raises fails the check; the rest still run
            _tb.print_exc()
            check(f"{part.__name__} ran ({type(exc).__name__}: {exc})", False)
    print("all passed" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
