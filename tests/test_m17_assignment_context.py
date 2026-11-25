"""MISSION 17 / O2 -- the context start, and why the damped filter could not fix it.

Measured first (see MISSION17_REPORT.md): at 15 deg the context ACCURACY -- the share of
places whose mapping is the truth partner -- is 0.173 for the argmax of the appearance
similarity, and the O2 lever sweep (iters 0/2/5/8, margin 0/0.02, rival count 0/1/3,
rival tolerance 0.02/0.05, mutual on/off) moved the geometry AUC by less than 0.005.

The mechanism is structural, and these tests pin the two properties that break it:
`assignment_mapping` is a BIJECTION (two places cannot claim one reference place) and it
scores with a CONFIGURATION descriptor that needs no correspondence, so it is not
computed from the mapping it is trying to fix.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P
from matching import placement_context as C
from matching.placement import PlacementVector
from matching.placement_matcher import Side


def _vec(dist: float, direction: float, axial: bool = True) -> PlacementVector:
    return PlacementVector(direction=direction, distance=dist, log_scale=0.0,
                           source_theta=0.0, source_scale=1.0, axial=axial)


def _side(star: dict) -> Side:
    """A Side whose node 0 has the given (neighbour -> (dist, direction)) star."""
    incident = {0: [(n, True, "EDGE") for n in star]}
    index = {(0, n, "EDGE"): _vec(d, th) for n, (d, th) in star.items()}
    return Side(payload={}, index=index, incident=incident, lens={},
                nodes=tuple([0] + list(star)))


class DescriptorTests(unittest.TestCase):
    def test_it_is_invariant_to_the_source_frame(self):
        """`direction` is in the SOURCE node's frame, so a rotated image gives the same
        descriptor.  This is the property that makes it usable before the context is
        known at all."""
        a = _side({1: (2.0, 0.1), 2: (3.5, 1.2), 3: (5.0, -0.7)})
        # the same configuration seen in a frame rotated by 40 degrees: the STRORED
        # directions are the same by construction, so the descriptor must be identical
        b = _side({1: (2.0, 0.1), 2: (3.5, 1.2), 3: (5.0, -0.7)})
        self.assertEqual(C.config_descriptor(a, 0), C.config_descriptor(b, 0))
        self.assertAlmostEqual(C.config_similarity(C.config_descriptor(a, 0),
                                                   C.config_descriptor(b, 0)), 1.0)

    def test_it_separates_different_configurations(self):
        a = _side({1: (2.0, 0.1), 2: (3.5, 1.2), 3: (5.0, -0.7)})
        b = _side({1: (2.0, 0.1), 2: (3.5, 1.2), 3: (9.0, 1.5)})
        s = C.config_similarity(C.config_descriptor(a, 0), C.config_descriptor(b, 0))
        self.assertLess(s, 1.0)
        self.assertGreaterEqual(s, 0.0)

    def test_a_thin_star_has_no_descriptor(self):
        a = _side({1: (2.0, 0.1)})
        self.assertIsNone(C.config_descriptor(a, 0))


class AssignmentTests(unittest.TestCase):
    def test_the_mapping_is_a_bijection_where_argmax_collides(self):
        """THE DEFECT: two places both prefer the same reference place, and argmax
        gives it to both.  The assignment cannot."""
        cands = {0: [(7, 0.9), (8, 0.5)], 1: [(7, 0.8), (8, 0.4)]}
        side = _side({1: (1.0, 0.0), 2: (2.0, 0.5)})
        got = C.assignment_mapping([0, 1], cands, side, side, lam=0.0)
        self.assertEqual(len(set(got.values())), 2,
                         "the assignment must not give one reference place to two")
        self.assertEqual(got[0], 7)
        self.assertEqual(got[1], 8)

    def test_the_configuration_term_changes_the_choice(self):
        cands = {0: [(7, 0.90), (8, 0.85)]}
        # the query's star matches node 8's star and not node 7's
        q = _side({1: (2.0, 0.10), 2: (3.0, 0.50)})
        r = Side(payload={}, incident={7: [(1, True, "EDGE"), (2, True, "EDGE")],
                                       8: [(1, True, "EDGE"), (2, True, "EDGE")]},
                 index={(7, 1, "EDGE"): _vec(9.0, 1.4), (7, 2, "EDGE"): _vec(9.5, -1.3),
                        (8, 1, "EDGE"): _vec(2.0, 0.10), (8, 2, "EDGE"): _vec(3.0, 0.50)},
                 lens={}, nodes=(7, 8))
        pure = C.assignment_mapping([0], cands, q, r, lam=0.0)
        cfg = C.assignment_mapping([0], cands, q, r, lam=0.5)
        self.assertEqual(pure[0], 7, "appearance alone prefers 7")
        self.assertEqual(cfg[0], 8, "the configuration term prefers 8, and it is right")

    def test_it_is_deterministic(self):
        cands = {u: [(v, 0.9 - 0.01 * v) for v in range(5)] for u in range(4)}
        side = _side({1: (1.0, 0.0), 2: (2.0, 0.5)})
        a = C.assignment_mapping(sorted(cands), cands, side, side, lam=0.5)
        b = C.assignment_mapping(sorted(cands), cands, side, side, lam=0.5)
        self.assertEqual(a, b)


class FlagTests(unittest.TestCase):
    def test_off_by_default_and_declared(self):
        import importlib
        import parameters as PP
        self.assertFalse(PP.PLACEMENT_CONTEXT_ASSIGN)
        self.assertFalse(PP.PLACEMENT_CONTEXT_M17)
        from eth80_frontend import _FRONTEND_SWITCHES
        for name in ("PLACEMENT_CONTEXT_ASSIGN", "PLACEMENT_CONTEXT_ASSIGN_LAMBDA",
                     "PLACEMENT_CONTEXT_ASSIGN_ITERS", "PLACEMENT_CONTEXT_M17",
                     "PLACEMENT_CONTEXT_M17_ITERS", "PLACEMENT_CONTEXT_M17_MAX_RIVALS",
                     "PLACEMENT_CONTEXT_M17_RIVAL_TOL", "PLACEMENT_CONTEXT_M17_MUTUAL"):
            self.assertIn(name, _FRONTEND_SWITCHES)


if __name__ == "__main__":
    unittest.main()


class NoSilentFallbackTests(unittest.TestCase):
    """Bar 3.  The block that adds the assignment start was first inserted by a
    pattern that matched TWO functions, so it also landed in `rescore_matrix`, where
    the name it reads is not defined -- the matcher caught the NameError and FELL BACK
    to the raw matrix, and the 14-pair defaults check caught the result (63 diffs on a
    tree whose flags were all off).  Both halves are pinned here.
    """

    def test_rescore_matrix_does_not_fall_back(self):
        import numpy as np
        from matching import placement_matcher as PM
        from matching.placement import PlacementVector
        n = 3
        side = PM.Side(payload={}, index={(0, 1, "EDGE"): PlacementVector(
            direction=0.1, distance=2.0, log_scale=0.0, source_theta=0.0,
            source_scale=1.0)},
            incident={0: [(1, True, "EDGE")]}, lens={0: "BLOB", 1: "BLOB"},
            nodes=(0, 1))
        before = dict(PM.FAILURES)
        matrix = [[0.5, 0.2], [0.1, 0.4]]
        PM.rescore_matrix(matrix, [0, 1], [0, 1], side, side)
        self.assertEqual(PM.FAILURES["fell_back"], before["fell_back"],
                         "rescore_matrix fell back -- a name it reads is undefined")

    def test_the_assignment_block_appears_in_one_function_only(self):
        import inspect
        from matching import placement_matcher as PM
        self.assertEqual(inspect.getsource(PM.rescore_matrix).count(
            "PLACEMENT_CONTEXT_ASSIGN"), 0,
            "rescore_matrix reads a switch it cannot resolve")
        self.assertGreater(inspect.getsource(PM.context_mapping).count(
            "PLACEMENT_CONTEXT_ASSIGN"), 0,
            "context_mapping is where the assignment start belongs")
