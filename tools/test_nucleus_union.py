"""CPU-only check that TENSORFOLD_NUCLEUS_UNION=1 draws exactly what the default path draws (issue #91).

  python3 -B tools/test_nucleus_union.py --source-root /path/to/patched/src
Needs numpy only (torch is stubbed: the draw itself is numpy). Emulates ``nucleus_rows`` over R ranks' vocabulary
shards the way ``_shares`` builds them, runs the real ``_draw`` with the flag off and on, and compares the token and
its share of the mass over many random rows, ranks 2 to 4, top_p, min_p and tie-heavy distributions.
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np


def load(root):
    sys.path.insert(0, root)
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    from tensorfold.cuda import sampling
    from tensorfold.engine.exact_sampling import Sampling
    return sampling, Sampling


def shares(sampling, scaled, mass, count, ranks):
    """``_shares`` on numpy: per rank the top ``count`` (value, id, mass) per row, padded, plus shard sums and widths."""

    rows, vocab = scaled.shape
    cuts = np.linspace(0, vocab, ranks + 1).astype(int)
    vs, ids, ms, sums, wid = [], [], [], [], []
    for k in range(ranks):
        lo, hi = cuts[k], cuts[k + 1]
        v, i, m = np.full((rows, count), -np.inf), np.full((rows, count), -1, np.int64), np.zeros((rows, count), np.int64)
        for r in range(rows):
            order = np.lexsort((np.arange(lo, hi), -scaled[r, lo:hi]))[:count]   # topk (ties: lowest id first)
            n = len(order)
            v[r, :n], i[r, :n], m[r, :n] = scaled[r, lo:hi][order], order + lo, mass[r, lo:hi][order]
        vs.append(v), ids.append(i), ms.append(m)
        sums.append(mass[:, lo:hi].sum(axis=1)), wid.append(np.full(rows, hi - lo))
    return np.stack(vs), np.stack(ids), np.stack(ms), np.stack(sums), np.stack(wid)


def nucleus(sampling, scaled, positions, smp, ranks, union):
    os.environ["TENSORFOLD_NUCLEUS_UNION"] = "1" if union else "0"
    top = scaled.max(axis=1)
    mass = np.floor(np.exp(scaled - top[:, None]) * sampling.MASS).astype(np.int64)
    got = shares(sampling, scaled, mass, sampling.NUCLEUS, ranks)
    drawn = sampling._draw(got, positions, smp)
    trips = 1
    if drawn is None and sampling.union_cover() and int(got[4].max()) > sampling.WIDER:
        drawn, trips = sampling._draw(shares(sampling, scaled, mass, sampling.WIDER, ranks), positions, smp), 2
    if drawn is None:
        drawn, trips = sampling._draw(shares(sampling, scaled, mass, int(got[4].max()), ranks), positions, smp), trips + 1
    return drawn, trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--vocab", type=int, default=20000)
    ap.add_argument("--cases", type=int, default=60)
    args = ap.parse_args()
    sampling, Sampling = load(args.source_root)
    rng = np.random.default_rng(7)
    trips_off = trips_on = n = 0
    for case in range(args.cases):
        ranks = int(rng.integers(2, 5))
        kind = case % 5
        logits = rng.normal(size=(3, args.vocab)) * (1.0, 4.0, 0.3, 0.0, 1.0)[kind]
        if kind == 3:
            logits = np.round(logits + rng.normal(size=logits.shape) * 0.0, 0)
            logits[:, rng.integers(0, args.vocab, 8)] = 2.0              # a few tied peaks
        if kind == 0:
            logits[:, :40] += 12.0                                       # concentrated
        if kind == 4:                                                    # ~4000 broad candidates over the ranks
            for r in range(3):
                logits[r, rng.choice(args.vocab, 4000, replace=False)] += 7.0
        temp = float((1.0, 0.7, 1.3)[case % 3])
        top_p = float((0.95, 0.9, 0.6, 0.99)[case % 4])
        min_p = float((0.0, 0.0, 0.05)[case % 3])
        smp = Sampling(int(rng.integers(1, 10**6)), temp, 0, top_p, min_p)
        scaled = logits / temp
        positions = [int(x) for x in rng.integers(0, 10**5, 3)]
        a, ta = nucleus(sampling, scaled, positions, smp, ranks, False)
        b, tb = nucleus(sampling, scaled, positions, smp, ranks, True)
        assert a == b, f"case {case} differs: {a} vs {b}"
        n += len(a)
        trips_off += ta
        trips_on += tb
    print(f"{n} draws identical over {args.cases} cases; gathers off {trips_off}, on {trips_on}")


if __name__ == "__main__":
    main()
