"""Mission 15 item 6: no checkable edge = NEUTRAL, not zero and not one.

`_relational_agreement` returns 0.0 when a side has no relations, and the shipped
quality is `s*(1-beta) + beta*rel` -- so on graphs where most nodes are degree-0 a
pair is scored at HALF its appearance value purely for having nothing to check.  Under
`MATCH_DEGREE0_PAIR_USES_APPEARANCE` such a pair takes its own appearance.

Requirements pinned here:
  * flag off  -> the degree-0 pair keeps the shipped `s*(1-beta)` (defaults identical),
  * flag on   -> its quality IS its appearance `s` (neither 0 nor 1),
  * a perfectly agreeing CONNECTED pair scores >= the appearance-only value.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P
from matching.alignment import global_consistency_alignment

BETA = 0.5


def _run(nodes, edges_a, edges_b, sim=1.0, only=None):
    """`only=(u, v)` restricts the similarity to that one pair (all others 0)."""
    def sim_fn(u, v):
        return sim if (only is None or (u, v) == only) else 0.0
    return global_consistency_alignment(
        nodes, nodes, sim_fn, edges_a=edges_a, edges_b=edges_b,
        systematicity_weight=BETA)


#: a graph that HAS edges, on nodes 1-2; node 0 is isolated -- the real case: the
#: defect is invisible when the whole graph has no edges (`rel` is then 1.0).
CONNECTED = ((1, 2, "CLS"),)


class Degree0Tests(unittest.TestCase):
    def setUp(self):
        self._old = P.MATCH_DEGREE0_PAIR_USES_APPEARANCE

    def tearDown(self):
        P.MATCH_DEGREE0_PAIR_USES_APPEARANCE = self._old

    def test_shipped_quality_halves_a_degree0_pair(self):
        P.MATCH_DEGREE0_PAIR_USES_APPEARANCE = False
        a = _run([0, 1, 2], CONNECTED, CONNECTED, only=(0, 0))
        self.assertEqual(a.mapping, {0: 0})
        self.assertAlmostEqual(a.quality[0], 0.5, places=6)

    def test_degree0_pair_is_its_appearance_when_on(self):
        P.MATCH_DEGREE0_PAIR_USES_APPEARANCE = True
        a = _run([0, 1, 2], CONNECTED, CONNECTED, only=(0, 0))
        self.assertAlmostEqual(a.quality[0], 1.0, places=6)
        b = _run([0, 1, 2], CONNECTED, CONNECTED, sim=0.3, only=(0, 0))
        self.assertAlmostEqual(b.quality[0], 0.3, places=6)

    def test_connected_agreeing_pair_never_scores_below_appearance(self):
        P.MATCH_DEGREE0_PAIR_USES_APPEARANCE = True
        a = _run([1, 2], CONNECTED, CONNECTED, sim=0.4)
        self.assertEqual(sorted(a.mapping), [1, 2])
        self.assertGreaterEqual(min(a.quality.values()), 0.4 - 1e-9)


if __name__ == "__main__":
    unittest.main()
