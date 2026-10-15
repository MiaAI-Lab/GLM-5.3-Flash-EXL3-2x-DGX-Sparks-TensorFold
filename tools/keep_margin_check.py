#!/usr/bin/env python3
"""CPU checks for patch 0109 (TF_GLM_KEEP_MARGIN: kept prompt states short of the prompt's end).

  python3 -B tools/keep_margin_check.py --source-root /path/to/patched/src

Loads the patched engine.py on its own (no torch, GPU or model) and checks
grid_point and keep_margin: off (0) is the old rule for every length and resume point; on, a kept state is on the grid,
at least the margin short of the end and past the resume point. Then the reported miss, replayed on token positions: a
conversation is warmed with one last user message and asked again with another (the repro in the patch's issue), so the
two prompts share all but their last ``tail`` tokens; the second resumes at the longest kept state that lies inside the
shared part. Off, it resumes at 0 whenever the warmed prompt's length mod 64 is below ``tail`` (tail / 64 of all
lengths); with a margin of 64 never, for any tail up to 64. Exit code 1 when a check fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

fails: list[str] = []


def check(name: str, ok: bool) -> None:
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def load(root: str):
    sys.path.insert(0, root)                       # engine.py's module level reads tensorfold.cuda.geometry
    path = os.path.join(root, "tensorfold", "families", "glm5_next", "cuda", "engine.py")
    spec = importlib.util.spec_from_file_location("glm_engine_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_point(n: int, cut: int, grid: int):
    """grid_point before patch 0109."""
    if not grid or n % grid == 0:
        return n
    point = n // grid * grid
    return point if point > cut else None


def resumes(e, warm: int, tail: int, margin: int, grid: int = 64) -> int:
    """Where the second prompt resumes: the warmed prompt (``warm`` tokens, resumed at 0) kept its state at
    grid_point; the second shares its first ``warm - tail`` tokens. A kept state is usable only inside that."""
    point = e.grid_point(warm, 0, grid, margin)
    return point if point is not None and point <= warm - tail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, help="the patched TensorFold src directory")
    e = load(ap.parse_args().source_root)

    # -- keep_margin ----------------------------------------------------------------------------------------------------
    check("unset and 0: off", e.keep_margin("") == 0 and e.keep_margin("0") == 0 and e.keep_margin(" ") == 0)
    check("a number of tokens", e.keep_margin("64") == 64 and e.keep_margin(" 200 ") == 200)
    for bad in ("-1", "abc", "4097", "1.5"):
        try:
            e.keep_margin(bad)
            check(f"{bad!r} refused", False)
        except ValueError:
            check(f"{bad!r} refused", True)

    # -- grid_point -----------------------------------------------------------------------------------------------------
    same = all(e.grid_point(n, cut, g) == old_point(n, cut, g) == e.grid_point(n, cut, g, 0)
               for g in (0, 32, 64, 128) for n in range(1, 700) for cut in range(0, n, 7))
    check("margin 0: the old rule for every length, resume point and grid", same)
    ok = True
    for g in (0, 32, 64, 128):
        for m in (1, 16, 64, 100):
            for n in range(1, 900):
                for cut in range(0, n, 5):
                    p = e.grid_point(n, cut, g, m)
                    if p is None:
                        ok &= (n - m) // (g or 1) * (g or 1) <= cut
                    else:
                        ok &= p > cut and n - p >= m and (not g or p % g == 0) and (not g or n - p < m + g)
    check("margin on: on the grid, past the resume point, margin to margin + grid - 1 short of the end", ok)

    # -- the reported miss ----------------------------------------------------------------------------------------------
    warms = range(17_000, 17_000 + 64 * 4)                      # every length mod 64, four times
    for tail in (1, 8, 20, 64):
        off = sum(resumes(e, w, tail, 0) == 0 for w in warms) / len(warms)
        on = sum(resumes(e, w, tail, 64) == 0 for w in warms)
        check(f"a {tail}-token branching tail: off resumes at 0 for {off:.0%} of lengths, margin 64 for none",
              abs(off - tail / 64) < 1e-9 and on == 0)
    w = 200_000 + 5                                              # a 200k prompt whose last 8 tokens change
    check("200k, an 8-token tail: off from 0, margin 64 from within 127 tokens of the end",
          resumes(e, w, 8, 0) == 0 and w - resumes(e, w, 8, 64) <= 127)

    print("all checks pass" if not fails else f"{len(fails)} check(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
