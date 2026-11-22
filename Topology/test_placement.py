"""Task 4 PLACEMENT — invariants that must hold whatever the arms measured.

These are the claims the mission makes checkable without truth, plus the
determinism and darkness guarantees.  Run:  .venv/bin/python Topology/test_placement.py
"""
import math
import os
import sys
import unittest

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import parameters as P
from matching.placement import (PlacementVector, PAYLOAD_CLASSES, wrap_pi,
                                debin_theta)
from matching.placement_scorer import (support_from_residual, payload_residual,
                                       separation_auc, d_prime)


class PayloadTests(unittest.TestCase):
    def test_reversal_raises(self):
        """Reversal is NOT -direction: the far end has its own frame and scale."""
        v = PlacementVector(direction=0.4, distance=2.0, log_scale=0.0,
                            source_theta=0.1, source_scale=10.0, tolerance=1.0)
        with self.assertRaises(Exception):
            v.reversed()

    def test_direction_is_signed_over_full_turn(self):
        for d in (-math.pi + 1e-6, -1.0, 0.0, 1.0, math.pi - 1e-6):
            v = PlacementVector(direction=d, distance=1.0, log_scale=0.0,
                                source_theta=0.0, source_scale=1.0, tolerance=1.0)
            self.assertAlmostEqual(v.direction, d, places=12)

    def test_wrap_pi_range(self):
        for x in (-10.0, -math.pi, 0.0, math.pi, 10.0):
            self.assertLessEqual(abs(wrap_pi(x)), math.pi + 1e-12)

    def test_payload_classes_exclude_proximity(self):
        """CONTACT / CO_LOCATED / PROXIMITY / SURFACE_GROUPING may not carry."""
        banned = {"CONTACT", "CO_LOCATED", "PROXIMITY", "SURFACE_GROUPING"}
        self.assertFalse(banned & set(PAYLOAD_CLASSES))


class ScorerTests(unittest.TestCase):
    def test_identity_residual_is_zero(self):
        v = PlacementVector(direction=0.7, distance=3.0, log_scale=0.0,
                            source_theta=0.2, source_scale=5.0, tolerance=1.0)
        self.assertEqual(payload_residual(v, v, 1.0), 0.0)

    def test_support_is_bounded_and_truncated(self):
        self.assertAlmostEqual(support_from_residual(0.0), 1.0, places=12)
        for r in (0.5, 1.0, 2.0):
            s = support_from_residual(r)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        # the cutoff is STRICTLY beyond 3 sigma: at exactly 3 sigma the
        # Gaussian still returns exp(-4.5) ~ 0.0111, which is bounded support.
        self.assertAlmostEqual(support_from_residual(3.0), math.exp(-4.5),
                               places=12)
        self.assertEqual(support_from_residual(3.001), 0.0)
        self.assertEqual(support_from_residual(1e9), 0.0)

    def test_support_is_neutral_at_half_only_by_construction(self):
        """The modifier must be neutral at support 0.5 and never unbounded."""
        w = P.PLACEMENT_WEIGHT
        for s in (0.0, 0.5, 1.0):
            f = 1.0 + w * (s - 0.5)
            self.assertGreaterEqual(f, 1.0 - w / 2 - 1e-12)
            self.assertLessEqual(f, 1.0 + w / 2 + 1e-12)
        self.assertAlmostEqual(1.0 + w * (0.5 - 0.5), 1.0, places=12)

    def test_uncheckable_is_not_rewarded(self):
        """No checkable edge => None => the caller must leave similarity alone."""
        from matching.placement_scorer import node_support
        self.assertIsNone(node_support(0, {}, {}, {}, 1.0))

    def test_separation_metrics(self):
        # perfect separation
        auc = separation_auc([0.0, 0.1, 0.2], [1.0, 1.1, 1.2])
        self.assertAlmostEqual(auc, 1.0, places=12)
        # d' is undefined for a degenerate (zero-variance) population and the
        # scorer returns None rather than dividing by zero; use spread.
        self.assertGreater(d_prime([0.0, 0.1, 0.2, 0.15],
                                   [1.0, 1.1, 1.2, 1.15]), 1.0)
        self.assertIsNone(d_prime([0.0] * 5, [1.0] * 5))


class DarknessTests(unittest.TestCase):
    def test_every_placement_flag_is_dark(self):
        """Part 6 measured a failure, so the flags stay dark."""
        self.assertFalse(P.PLACEMENT_PAYLOAD_ENABLED)
        self.assertFalse(P.PLACEMENT_SCORER_ENABLED)
        self.assertFalse(P.PLACEMENT_GATE_ENABLED)
        self.assertIsNone(P.PLACEMENT_TOLERANCE)

    def test_one_pre_stated_weight(self):
        self.assertEqual(P.PLACEMENT_WEIGHT, 0.25)

    def test_spine_widening_is_dark(self):
        """The reach fix was rejected on frame validity, so the knob is empty."""
        self.assertEqual(tuple(P.ORIENTATION_SPINE_EXTRA), ())

    def test_alignment_is_unchanged_when_dark(self):
        from matching.alignment import global_consistency_alignment
        nodes = list(range(5))
        sim = [[1.0 if i == j else 0.1 for j in range(5)] for i in range(5)]
        a = global_consistency_alignment(nodes, nodes, lambda u, v: sim[u][v],
                                         sim=sim)
        fake = {(0, 1, "X"): 1}
        b = global_consistency_alignment(
            nodes, nodes, lambda u, v: sim[u][v], sim=sim,
            placement_payloads_a=fake, placement_payloads_b=fake,
            placement_tolerance=1.0)
        self.assertEqual(a.mapping, b.mapping)


if __name__ == "__main__":
    unittest.main(verbosity=2)
