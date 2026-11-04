"""CPU regression checks for shared union-coverage/draw ordering (numpy only).

python tools/test_nucleus_order_reuse.py --source-root /path/to/patched/src
"""
from __future__ import annotations

import argparse
import os
import unittest
from unittest.mock import patch

import numpy as np

from test_nucleus_union import load, shares


class OrderReuseTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TENSORFOLD_NUCLEUS_UNION": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def candidates(self, values, masses, count=2):
        return shares(sampling, np.array(values, dtype=float, ndmin=2),
                      np.array(masses, dtype=np.int64, ndmin=2), count, 2)

    def test_partial_coverage_sorts_and_accumulates_once(self):
        got = self.candidates([5, 0, -10, 4, -1, -10], [100, 1, 0, 80, 1, 0])
        with patch.object(np, "lexsort", wraps=np.lexsort) as sort, \
                patch.object(np, "cumsum", wraps=np.cumsum) as cumulative:
            result = sampling._draw(got, [17], Sampling(7, 1.0, 0, 0.5, 0.0))
        self.assertEqual(result, [(0, 100 / 182)])
        self.assertEqual(sort.call_count, 1, "coverage and draw must share their ordering")
        self.assertEqual(cumulative.call_count, 1, "coverage and draw must share cumulative mass")

    def test_cutoff_tie_requires_more_candidates(self):
        got = self.candidates([5, 4, 3, 4, 4, 3], [100] * 6)
        self.assertIsNone(sampling._draw(got, [17], Sampling(7, 1.0, 0, 0.5, 0.0)))

    def test_insufficient_candidate_mass_requires_more_candidates(self):
        got = self.candidates([5, 4, 3, 5, 4, 3], [100] * 6, count=1)
        self.assertIsNone(sampling._draw(got, [17], Sampling(7, 1.0, 0, 0.9, 0.0)))

    def test_exact_mass_boundary_excludes_next_token(self):
        got = self.candidates([5, 0, -10, 4, -1, -10], [100, 0, 0, 100, 0, 0])
        for seed in range(30):
            self.assertEqual(sampling._draw(got, [seed], Sampling(seed, 1.0, 0, 0.5, 0.0)),
                             [(0, 0.5)])

    def test_padding_ties_min_p_and_disabled_union_match_whole_shards(self):
        values = np.array([[5, 4, 4, 0, -10], [2, 2, 2, 2, 2]], dtype=float)
        masses = np.floor(np.exp(values - values.max(axis=1)[:, None]) * sampling.MASS).astype(np.int64)
        got = shares(sampling, values, masses, 4, 2)
        for top_p in (0.5, 0.95, 1.0):
            for min_p in (0.0, 0.2):
                for seed in range(10):
                    smp = Sampling(seed, 1.0, 0, top_p, min_p)
                    with patch.dict(os.environ, {"TENSORFOLD_NUCLEUS_UNION": "0"}):
                        expected = sampling._draw(got, [13, 29], smp)
                    self.assertEqual(sampling._draw(got, [13, 29], smp), expected)

    def test_later_uncovered_row_requests_fallback(self):
        got = self.candidates([[5, 0, -10, 4, -1, -10], [5, 4, 3, 4, 4, 3]],
                              [[100, 1, 0, 80, 1, 0], [100] * 6])
        self.assertIsNone(sampling._draw(got, [17, 18], Sampling(7, 1.0, 0, 0.5, 0.0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args, rest = parser.parse_known_args()
    sampling, Sampling = load(args.source_root)
    unittest.main(argv=[__file__, *rest])
