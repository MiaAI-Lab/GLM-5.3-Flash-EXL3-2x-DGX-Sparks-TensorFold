#!/usr/bin/env python3
"""Checks for the fork-point patch (a kept state that a second conversation forks from is a shared prefix), run inside
the image that scripts/prepare.sh built, no GPU:

    docker run --rm --network none --entrypoint python -v "$PWD/tools/fork_point_check.py:/c.py" <image> /c.py

Two real MultiDecoders, rank 0 deciding and rank 1 applying its messages in-process (``admit``, ``round``, ``finish``,
``apply``: the scheduler's calls), with only the model stubbed: the prompt chunk, the snapshot of the slot's state, the
vision tower and the sampler. Prompts are token ids shaped as the chat template lays them out (a system block, then
``<|user|>``, the message and the generation prompt); the grid (64), the shared-prefix minimum (256), the kept-state
cap (32) and TF_GLM_MULTI_LONE=0 are the recipe's defaults; prompts are filled one at a time (MULTI_PREFILL=0,
FILL_BUDGET_MS=0: the grouped and sliced fills keep states through the same ``_keep``). Exit code 1 when a check fails.

  1. Issue #121's script: 40 short chats fill the cap, then three conversations on one new system prompt: "Hey", a
     long opener, "Hi". The third must resume the system prompt (before: the first conversation's end state, which is
     the system block's state, was dropped as superseded once the second kept a longer state in its extent).
  2. An agent's sessions on one system prompt: three openers, one of them runs 7 turns, two new sessions: all resume.
  3. TF_GLM_KEEP_PER_CHAT=1, the cap not full: the third conversation still resumes (the second one forked, so it is
     not the first one's next turn and does not take the system block's state as its own earlier turn).
  4. TF_GLM_KEEP_PER_CHAT=2: a conversation that forks from another's first state starts a chain of its own, so its
     turns do not push the other conversation's latest turn out.
  5. Two conversations on a new system prompt admitted together (both prefill it): the second state with the same ids
     replaces the first and is a shared prefix, so the next conversation still resumes after one of them goes on; and
     a shared-prefix state replaced by one with the same ids stays one.
  6. Unchanged: a conversation going on as its writer did is not a fork (chat W waits while chat L runs 40 turns, then
     W's turn 2 resumes W's own state, and L's earlier turns are the ones dropped).
  7. A turn whose prompt ended on the grid: the first reply that goes on from its state is its line, a second (a
     retry) forks, and the state stays past the cap for a third.
  8. Pictures: the same ids with another picture's content (keyed ids) are a fork.
  9. The spill tier (--spill-gib) stores the line past a state and restores it.
 10. The rule (``multi.forks``) on its own.
Every scenario also checks that both ranks keep the same states with the same flags.
"""
import os

os.environ.setdefault("TF_GLM_MULTI_WATCHDOG_S", "0")

import random  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

import torch  # noqa: E402

import tensorfold.families.glm5_next.cuda.decode as D  # noqa: E402
import tensorfold.families.glm5_next.cuda.forward as F  # noqa: E402
import tensorfold.families.glm5_next.cuda.multi as M  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings  # noqa: E402
from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane  # noqa: E402

GRID, LEAST, ENTRIES = 64, 256, 32
GMASK, SOP, SYSTEM, USER, ASSISTANT, THINK, EOS = 151331, 151333, 151335, 151336, 151337, 151350, 151329
REPLY = 7                              # the stubbed sampler's token
POOL = 512 * ALIGN
fails: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


# -- the model, stubbed: positions move, states are placeholders ----------------------------------------------------
class SlotState:
    """``forward.State``: only the slot's position matters to the bookkeeping."""

    def __init__(self, *a, **k):
        self.pos, self.cur, self.rec, self.conv = 0, [0], [torch.zeros(1)], torch.zeros(1)

    def reset(self):
        self.pos = 0

    def set_pos(self, n):
        self.pos = n


def prefill_chunk(e, prompt, start, rows, **k):
    e.st.pos = start + rows
    return torch.zeros(1, 4)


def take_snapshot(e, ids, pending, *, mtp, drafter=None):
    return D.Snapshot(list(ids), torch.zeros(1), torch.zeros(1), None, -1, -1, tail=[])


F.State, D.prefill_chunk, D.take_snapshot = SlotState, prefill_chunk, take_snapshot
D.put_ring_tail = lambda st, hit: None
M.sample_streams = lambda w, parts: [[REPLY] for _ in parts]


class Engine:
    def __init__(self, planes=None, rows=POOL):
        self.w = NS(device="cpu", cfg=NS(eos=[EOS], hidden=8), vocab_offset=0, comm=None, layers=[None], world=2)
        self.slots, self.rows, self.prefill_rows, self.st = NS(count=4), 16, 2048, None
        planes = planes(rows) if planes is not None else [Plane(torch.zeros(rows, 1, dtype=torch.uint8))]
        self.caches = NS(rows=rows, arena=Arena(rows, planes))

    def use(self, st):
        prev, self.st = self.st, st
        return prev


class Net:
    follower = None


class Glm:
    """The ``GlmEngine`` the decoder reads (its settings) and talks through (rank 0 to rank 1, in-process)."""

    def __init__(self, rank, net, planes=None, rows=POOL):
        self.rank, self.net, self.e = rank, net, Engine(planes, rows)
        self.w = self.e.w
        self.grid, self.drafter, self.limit, self.serial_only = GRID, None, 1 << 20, False
        self.shared, self.opener, self.cache_entries, self.copy, self.spill_cfg = LEAST, USER, ENTRIES, None, None
        self.multi_window, self.costs, self.model_dir = 64, None, "."
        self.vision = NS(features=lambda prepared: torch.zeros(len(prepared.positions), 8))   # the tower, stubbed

    def _ring(self):
        pass

    def _share(self, msg):
        f = self.net.follower
        f.apply(M.unseal(msg, f.received, 1))
        f.received = (f.received + 1) % M.SEAL_MOD


def ranks(glm=Glm, planes=None, rows=POOL):
    net = Net()                                  # the recipe's TF_GLM_MULTI_LONE=0 (scripts/config.sh)
    tune = MultiSettings.from_env({"TF_GLM_MULTI_LONE": "0"})
    r0, r1 = (M.MultiDecoder(glm(r, net, planes, rows), 4, verify=object(), tune=tune) for r in (0, 1))
    net.follower = r1
    return r0, r1


# -- prompts, as the chat template lays them out -------------------------------------------------------------------
rng = random.Random(121)


def text(n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def system_block(n):
    return [GMASK, SOP, SYSTEM] + text(n - 3)


def turn(msg):
    return [USER] + msg + [ASSISTANT, THINK]


class Picture:
    """An image prompt's prepared pictures (``s.vision``): their rows' positions and content keys."""

    def __init__(self, positions, key):
        self.positions, self.item_rows, self.item_keys = list(positions), [len(positions)], [key]

    def from_item(self, first):
        return self


def run(r0, *prompts):
    """Admit the prompts together (a prompt, or (prompt, Picture)), run until each has its first token (a reply of one
    token), finish them; what each resumed (``tensorfold.cached``)."""

    streams = []
    for p in prompts:
        p, vision = p if isinstance(p, tuple) else (p, None)
        streams.append(Stream(list(p), 1, None, draft=True, stop_eos=True, vision=vision))
    for s in streams:
        r0.admit(s)
    while not all(s.done for s in streams):
        r0.finish(r0.round())
    return [s.cached for s in streams]


def same_on_ranks(r0, r1):
    return [(c.kid, len(c.ids), bool(c.shared)) for c in r0.kept] == [(c.kid, len(c.ids), bool(c.shared))
                                                                       for c in r1.kept]


def fill_cap(r0, count=40):
    for i in range(count):                       # "Filler conversation i: lorem ipsum ..." + "Say OK."
        run(r0, system_block(310) + turn(text(4)))


# -- 1: the issue's script ----------------------------------------------------------------------------------------
def issue_script():
    r0, r1 = ranks()
    fill_cap(r0)
    check(f"the 40 short chats fill the cap ({len(r0.kept)} kept states)", len(r0.kept) == ENTRIES)
    sys_ = system_block(14831)                   # its <|user|> at 14,831: in the grid step that starts at 14,784
    hey, opener, hi = sys_ + turn(text(1)), sys_ + turn(text(48)), sys_ + turn(text(1))
    got = [run(r0, p)[0] for p in (hey, opener, hi)]
    print(f"     cached: 'Hey' {got[0]}, long opener {got[1]}, 'Hi' {got[2]} (prompts {len(hey)}, {len(opener)}, "
          f"{len(hi)})")
    check("conversation 1 ('Hey') prefills the new system prompt", got[0] == 0)
    check("conversation 2 (a long opener) resumes the system block (14,784)", got[1] == 14784)
    check("conversation 3 ('Hi') resumes the system block too, past the cap (#121)", got[2] == 14784)
    state = next((c for c in r0.kept if len(c.ids) == 14784), None)
    check("the system block's state is a shared prefix, not a turn state of the first conversation",
          state is not None and state.shared and not state.own)
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 2: an agent's sessions ---------------------------------------------------------------------------------------
def agent_sessions():
    r0, r1 = ranks()
    fill_cap(r0)
    sys_ = system_block(25540)                  # its <|user|> at 25,540: in the grid step that starts at 25,536
    point = 25536
    hello, howdy, buenos = (sys_ + turn(text(k)) for k in (2, 60, 6))  # the second ends past 25,600: a longer state
    got = [run(r0, p)[0] for p in (hello, howdy, buenos)]
    check(f"three sessions on one system prompt: the 2nd and 3rd resume it ({got})", got == [0, point, point])
    conv, seen = howdy, []                                              # "Howdy" goes on, 7 turns
    for _ in range(7):
        conv = conv + [REPLY] + text(30) + turn(text(rng.randrange(5, 200)))
        seen.append(run(r0, conv)[0])
    check(f"the session that goes on resumes its own last state every turn ({seen})",
          seen[0] == 25600 and all(b > a for a, b in zip(seen, seen[1:])))
    got = [run(r0, sys_ + turn(text(k)))[0] for k in (4, 9)]
    check(f"two new sessions after it resume the system prompt ({got})", got == [point, point])
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 3 and 4: TF_GLM_KEEP_PER_CHAT ----------------------------------------------------------------------------------
def keep_per_chat():
    saved = M.KEEP_PER_CHAT
    try:
        M.KEEP_PER_CHAT = 1
        r0, r1 = ranks()
        sys_ = system_block(14831)
        got = [run(r0, sys_ + turn(text(k)))[0] for k in (1, 48, 1)]
        check(f"TF_GLM_KEEP_PER_CHAT=1, room left: the third conversation resumes the system block ({got})",
              got == [0, 14784, 14784])
        M.KEEP_PER_CHAT = 2
        r0, r1 = ranks()
        sys_ = system_block(14831)
        a1 = sys_ + turn(text(1))                                       # A: turn 1 ends at the system block's point
        run(r0, a1)
        a2 = a1 + [REPLY] + text(40) + turn(text(300))
        run(r0, a2)                                                     # A: turn 2
        b = sys_ + turn(text(48))                                       # B forks at the system block, 2 turns
        run(r0, b)
        run(r0, b + [REPLY] + text(40) + turn(text(300)))
        a2_point = len(a2) // GRID * GRID
        got = run(r0, a2 + [REPLY] + text(40) + turn(text(20)))[0]      # A: turn 3
        check(f"TF_GLM_KEEP_PER_CHAT=2: a conversation forked from another's first state starts a chain of its own, "
              f"so the other's latest turn stays (A's turn 3 resumed {got}, its turn 2 state is {a2_point})",
              got == a2_point)
        check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))
    finally:
        M.KEEP_PER_CHAT = saved


# -- 5: the same ids written by two conversations -------------------------------------------------------------------
def admitted_together():
    r0, r1 = ranks()
    fill_cap(r0)
    sys_ = system_block(14831)
    a, b = sys_ + turn(text(1)), sys_ + turn(text(2))
    got = run(r0, a, b)
    check(f"two conversations admitted together both prefill the new system prompt ({got})", got == [0, 0])
    state = next((c for c in r0.kept if len(c.ids) == 14784), None)
    check("the second state with the system block's ids replaces the first and is a shared prefix",
          state is not None and state.shared and not state.own
          and sum(1 for c in r0.kept if len(c.ids) == 14784) == 1)
    got = run(r0, b + [REPLY] + text(40) + turn(text(300)))[0]         # B goes on (as B's own prompt did)
    check(f"B's turn 2 resumes the system block ({got})", got == 14784)
    got = run(r0, sys_ + turn(text(5)))[0]
    check(f"a new conversation still resumes the system block after B went on ({got})", got == 14784)
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


def shared_replaced():
    r0, r1 = ranks()
    fill_cap(r0)
    sys_ = system_block(14831)
    r = sys_ + turn(text(1))
    q = r + [REPLY] + text(40) + turn(text(300))     # R's next turn, sent while R's first still prefills
    got = run(r0, q, r)                              # Q (cold) keeps the system block as a shared point, R then
    check(f"a conversation and its next turn admitted together both prefill ({got})", got == [0, 0])
    state = next((c for c in r0.kept if len(c.ids) == 14784), None)   # R's own state, the same ids as Q's point
    check("a shared-prefix state replaced by a state with the same ids stays a shared prefix",
          state is not None and state.shared and not state.own)
    run(r0, r + [REPLY] + text(40) + turn(text(30)))                   # R goes on (as its prompt did) from it
    got = run(r0, sys_ + turn(text(5)))[0]
    check(f"a new conversation still resumes the system block ({got})", got == 14784)
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 6: what must not change ----------------------------------------------------------------------------------------
def own_turns():
    r0, r1 = ranks()
    fill_cap(r0)
    w = system_block(900) + turn(text(20))                              # W's <|user|> in its end's grid step
    w_point = len(w) // GRID * GRID
    run(r0, w)                                                          # W: turn 1, then waits
    conv = system_block(705) + turn(text(30))                           # L: 40 turns (<|user|> in its end's step)
    flagged = 0
    for _ in range(40):
        run(r0, conv)
        flagged += sum(1 for c in r0.kept if c.shared)
        conv = conv + [REPLY] + text(30) + turn(text(rng.randrange(5, 90)))
    check(f"a conversation going on as its writer did never makes its own turn states shared ({flagged} flagged)",
          flagged == 0)
    got = run(r0, w + [REPLY] + text(30) + turn(text(10)))[0]
    check(f"W's turn 2 after L's 40 turns resumes W's own state ({got} of {w_point}): L dropped its own earlier turns",
          got == w_point)
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 7: a state whose writer's prompt ended there --------------------------------------------------------------------
def retries_on_grid():
    r0, r1 = ranks()
    fill_cap(r0, ENTRIES - 4)                    # room for the 4 states below, then the cap is reached
    x1 = system_block(14844) + turn(text(1))     # ends on the grid (14,848): its own state ends its prompt
    run(r0, x1)
    run(r0, x1 + [REPLY] + text(40) + turn(text(30)))                  # its turn 2
    run(r0, x1 + [REPLY] + text(40) + turn(text(30)))                  # turn 2 again, another reply (a retry)
    fill_cap(r0, 2)
    got = run(r0, x1 + [REPLY] + text(40) + turn(text(10)))[0]        # a third reply
    check(f"a turn's state that two replies went on from (its prompt ended on the grid) is kept past the cap ({got})",
          got == len(x1))
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 8: pictures ----------------------------------------------------------------------------------------------------
IMAGE = 151363


def pictures():
    r0, r1 = ranks()
    fill_cap(r0)
    sys_ = system_block(14784)                   # its <|user|> at the grid point 14,784
    rows = list(range(14785, 14805))             # 20 picture rows after <|user|>
    one = sys_ + [USER] + [IMAGE] * 20 + text(3) + [ASSISTANT, THINK]
    run(r0, (one, Picture(rows, 5)))
    two = one + [REPLY] + text(40) + turn(text(30))                     # the same ids, another picture
    got = run(r0, (two, Picture(rows, 6)))[0]
    check(f"the same question about another picture resumes the system block ({got})", got == 14784)
    got = run(r0, sys_ + turn(text(1)))[0]
    check(f"another picture is a fork too: a text conversation after it still resumes the system block ({got})",
          got == 14784)
    check("both ranks keep the same states with the same flags", same_on_ranks(r0, r1))


# -- 9: the spill tier (--spill-gib) stores and restores the line past a state ----------------------------------------
def spill_line():
    from tensorfold.cuda import spill

    r0, r1 = ranks()
    sys_ = system_block(14831)
    hey = sys_ + turn(text(1))
    run(r0, hey)
    snap = next(c for c in r0.kept if len(c.ids) == 14784)
    items, layer = spill.encode(snap, skip=M.MultiDecoder._SPILL_SKIP)
    back = spill.decode(layer, dict(items), [f"{D.Snapshot.__module__}:{D.Snapshot.__qualname__}"])
    line = getattr(back, "after", None)
    check("a spilled state's line past it (its writer's prompt) comes back from the spill tier",
          line is not None and tuple(line) == tuple(hey[14784:]))
    check("and still tells another conversation from the writer's next turn",
          M.forks(line, (sys_ + turn(text(48)))[14784:]) and not M.forks(line, (hey + [REPLY] + text(9))[14784:]))


# -- 10: the rule itself ----------------------------------------------------------------------------------------------
def fork_rule():
    check("the rule: the writer's next turn (its line, then more) is no fork", not M.forks((1, 2, 3), [1, 2, 3, 4]))
    check("the rule: a prompt ending inside the writer's line, agreeing, is no fork", not M.forks((1, 2, 3), [1, 2]))
    check("the rule: another id where both have one is a fork", M.forks((1, 2, 3), [1, 9]) and M.forks([5], (6, 7)))
    check("the rule: nothing known on either side is no fork", not M.forks((), [1]) and not M.forks((1,), []))


def run_parts(*parts) -> None:
    for part in parts or (issue_script, agent_sessions, keep_per_chat, admitted_together, shared_replaced, own_turns,
                          retries_on_grid, pictures, spill_line, fork_rule):
        try:
            part()
        except Exception as exc:                 # a part that raises fails the check; the rest still run
            check(f"{part.__name__} ran ({type(exc).__name__}: {exc})", False)


def main() -> int:
    run_parts()
    print("all passed" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
