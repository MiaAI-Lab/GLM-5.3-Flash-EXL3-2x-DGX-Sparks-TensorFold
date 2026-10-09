"""CPU-only test of patch 0091's loop guard with the real call_gate.generate_gated.

  python3 -B tools/test_loop_guard.py --source-root /path/to/patched/src
No Torch/CUDA, sockets or source writes. Replays token streams: a clean block, an exact cycle, a mostly-"the" block,
a loop after </think> (not watched), and a loop that starts late, and checks the close lands and the reply goes on."""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

THINK_END, CLOSE = 7, [10, 7, 11]


def load(root, name, rel):
    spec = importlib.util.spec_from_file_location(name, Path(root) / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(gates, stream, rounds=4, cap=3000):
    """Feed ``stream`` (the model's tokens) in rounds of ``rounds`` through generate_gated; after a cut the model
    goes on with the 'answer' [50, 51, 52]. Returns the reply."""
    gated = sys.modules["tf_call_gate"].generate_gated
    reply_out = []

    def generate(ids, count, take):
        tail = list(stream)[len(reply_out):]       # the model goes on where the reply stopped
        if CLOSE[-1] in ids:                       # after the close it answers
            tail = [50, 51, 52, 2]
        while tail and count > 0:
            chunk, tail = tail[:rounds], tail[rounds:]
            chunk = chunk[:count]
            if take(chunk):
                return {}
            count -= len(chunk)
        return {}

    out = []

    def on_tokens(new):
        out.extend(new)
        reply_out[:] = out
        return False

    gated(generate, [0], cap, gates, on_tokens)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    root = ap.parse_args().source_root
    load(root, "tf_call_gate", "tensorfold/engine/call_gate.py")
    guard_mod = load(root, "tf_loop_guard", "tensorfold/engine/loop_guard.py")
    rnd = random.Random(1)
    prose = [rnd.randrange(100, 5000) for _ in range(900)]
    def fresh():
        return guard_mod.LoopGuard(CLOSE, THINK_END)

    # 1. clean thinking, then </think> and an answer: untouched
    clean = prose[:600] + [THINK_END, 50, 51, 2]
    g = fresh(); out = run([g], clean)
    assert out == clean and g.fires == 0, "a clean block must not fire"
    # 2. exact cycle of period 4 after 300 tokens of prose: closed within a window and a round or two
    cyc = prose[:300] + [9, 8, 9, 6] * 400
    g = fresh(); out = run([g], cyc)
    assert g.fires == 1 and CLOSE == out[-len(CLOSE) - 4:-4], (g.fires, out[-12:])
    assert out[-4:] == [50, 51, 52, 2] and len(out) < 300 + 256 + 4 + 8 + len(CLOSE) + 8, len(out)
    # 3. a block that is half 'the' (id 5) between other words
    mixed = []
    for i in range(1200):
        mixed += [5, prose[i % len(prose)]] if i % 3 else [5, 5, prose[i % len(prose)]]
    g = fresh(); out = run([g], mixed)
    assert g.fires == 1 and len(out) < 700, (g.fires, len(out))
    # 4. a loop after </think> is the answer's business, not watched
    after = prose[:50] + [THINK_END] + [3] * 800
    g = fresh(); out = run([g], after)
    assert g.fires == 0 and out == after
    # 5. near cycle (one token in every 12 differs) and common-word prose never fire
    near = []
    for i in range(3000):
        near.append(prose[i % 11] if i % 12 else rnd.randrange(6000, 9000))
    g = fresh(); run([g], near, cap=3000); assert g.fires == 0
    zipf = rnd.choices(range(100, 5100), weights=[1 / r for r in range(1, 5001)], k=5000)    # the top token ~11%
    g = fresh(); run([g], zipf, cap=5000); assert g.fires == 0, "natural token frequencies must not fire"
    # 6. two guards are independent (one lane collapses, the other does not)
    a, b = fresh(), fresh()
    run([a], cyc); run([b], clean)
    assert a.fires == 1 and b.fires == 0
    # 7. the budget and the guard together: the earlier cut wins, and the reply still ends once
    tb = sys.modules["tf_call_gate"].ThinkBudget(350, CLOSE, THINK_END)
    g = fresh(); out = run([tb, g], cyc)
    assert out.count(THINK_END) == 1, out.count(THINK_END)
    print("OK: loop guard")


main()
