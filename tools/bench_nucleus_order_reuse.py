"""Compare host-side _draw latency to a saved pre-patch sampling.py (numpy only).

Apply the recipe patches excluding 0131, save tensorfold/cuda/sampling.py, then apply
0131 and run:
  python tools/bench_nucleus_order_reuse.py --source-root /path/to/patched/src \
      --baseline-sampler /path/to/pre-0131-sampling.py
This times CPU sampling only, excluding candidate construction, GPU work and communication.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np

from test_nucleus_union import load, shares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--baseline-sampler", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1:
        parser.error("repeats and iterations must be positive")
    current, Sampling = load(args.source_root)
    spec = importlib.util.spec_from_file_location("baseline_sampling", args.baseline_sampler)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rng = np.random.default_rng(7)
    broad = rng.normal(size=(1, 131072))
    peaked = broad.copy()
    peaked[:, :40] += 12.0
    results = []
    for label, values, count, union in (
        ("covered_1024", peaked, 1024, "1"),
        ("covered_16384", peaked, 16384, "1"),
        ("uncovered_1024", broad, 1024, "1"),
        ("whole_shards", peaked, 65536, "1"),
        ("union_disabled", peaked, 1024, "0"),
    ):
        mass = np.floor(np.exp(values - values.max(axis=1)[:, None]) * current.MASS).astype(np.int64)
        got = shares(current, values, mass, count, 2)
        smp = Sampling(7, 0.7, 0, 0.95, 0.0)
        durations = [[], []]
        functions = [baseline._draw, current._draw]
        with patch.dict(os.environ, {"TENSORFOLD_NUCLEUS_UNION": union}):
            expected = functions[0](got, [17], smp)
            assert functions[1](got, [17], smp) == expected, label
            for fn in functions:
                for _ in range(20):
                    fn(got, [17], smp)
            for repeat in range(args.repeats):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    start = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        functions[index](got, [17], smp)
                    durations[index].append((time.perf_counter_ns() - start) / args.iterations / 1000)
        before, after = map(statistics.median, durations)
        results.append(dict(case=label, fallback=expected is None, before_us=before,
                            after_us=after, reduction_percent=100 * (1 - after / before),
                            samples_us=durations))
    print(json.dumps(dict(platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, rows=1, ranks=2, vocab=131072,
                         repeats=args.repeats, iterations=args.iterations, results=results), indent=2))


if __name__ == "__main__":
    main()
