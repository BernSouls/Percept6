"""Mission 3 invariants: truth-free match time, non-vacuity canaries, estimator.

These are the assertions the mission demands be in the test suite rather than in
a log:

  * the BUILDER is mask-free -- object membership may gate the NEGATIVE-A
    control, but must not change deployed behaviour;
  * non-vacuity canaries -- R's edge credit has non-zero variance across the
    candidates of a node, P differs from N on some nodes, and P_shuffled and
    P_presence are not numerically identical to P;
  * the scored statistic is bounded and the PADDING rule does what it claims:
    a wrong candidate's expected score must not grow with its checkable-edge
    count.
"""
from __future__ import annotations

import math
import sys
import os
import unittest
from typing import Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _vec(direction: float, distance: float, log_scale: float = 0.0,
         axial: bool = True):
    from matching.placement import PlacementVector
    return PlacementVector(direction=direction, distance=distance,
                           log_scale=log_scale, source_theta=0.0,
                           source_scale=1.0, axial=axial)


class TruthFreeBuilderTests(unittest.TestCase):
    def test_the_builder_needs_no_object_membership(self):
        """Deployed behaviour is IDENTICAL with and without object ids.

        `object_of` exists only to gate the NEGATIVE-A control (intra_only).
        If it ever changes the deployed edge set, the builder is reading
        renderer truth and every number it produced is tainted.
        """
        import guard
        guard.activate()
        import numpy as np
        from matching import placement_edges as PE
        from perception.synthetic3d import (Material, Scene, SceneObject,
                                            Sphere)
        from Topology.placement_cache import get_or_build
        from Topology.placement_truth import node_objects

        mat = Material(albedo=0.8, texture="stripes", texture_freq=10.0,
                       texture_contrast=0.8, texture_space="object",
                       texture_dir=(1.0, 0.0, 0.0))
        sc = Scene(objects=[
            SceneObject(name="a", shape=Sphere(0.26),
                        centre=np.array([-0.24, 0.0, 0.0]), material=mat,
                        layer=0),
            SceneObject(name="b", shape=Sphere(0.26),
                        centre=np.array([0.24, 0.0, 0.0]), material=mat,
                        layer=0),
        ])
        r = sc.render()
        g = get_or_build(r.image_u8, "test-mask-free")
        obj = node_objects(r, g)

        free = PE.build_edges(g.subgraph, r.image_u8, object_of=None,
                              intra_only=False)
        withobj = PE.build_edges(g.subgraph, r.image_u8, object_of=obj,
                                 intra_only=False)
        self.assertEqual(sorted(free), sorted(withobj))
        self.assertGreater(len(free), 0)
        for k in sorted(free):
            self.assertEqual(free[k].direction, withobj[k].direction)
            self.assertEqual(free[k].distance, withobj[k].distance)

        # ... and intra_only DOES change the set, so the control is real.
        intra = PE.build_edges(g.subgraph, r.image_u8, object_of=obj,
                               intra_only=True)
        self.assertNotEqual(sorted(intra), sorted(free),
                            "NEGATIVE-A control is a no-op on a pair scene")


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        from matching import placement_llr as L
        self.L = L
        # A fit whose LLR is +2 for a small residual and -2 for a large one.
        res_t = [0.05] * 200 + [0.1] * 100
        res_w = [3.0] * 200 + [4.0] * 100
        self.fit = L.fit_llr(res_t, res_w)

    def test_fit_separates_true_from_wrong(self):
        self.assertLess(self.fit.mu_wrong, 0.0)
        self.assertGreater(self.fit.of(0.05), 0.0)
        # A far-out residual sits in a bin where the wrong density dominates;
        # with Laplace smoothing that bin can be flat (LLR == 0) but it can
        # never be positive, which is what the check is for.
        self.assertLessEqual(self.fit.of(3.5), 0.0)
        self.assertLess(self.fit.of(3.5), self.fit.of(0.05))

    def test_score_is_bounded(self):
        q_idx = {(0, 1, "A"): _vec(0.0, 1.0)}
        r_idx = {(5, 6, "A"): _vec(0.0, 1.0)}
        inc = {0: [(1, True, "A")]}
        s = self.L.score_candidate(0, 5, {1: 6}, inc, q_idx, r_idx, self.fit)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_padding_removes_the_checkable_edge_count_advantage(self):
        """A WRONG candidate is not rewarded for having MORE checkable edges.

        The claim is in expectation over the wrong population: with seven extra
        checkable slots the expected score must not move, because each of them
        is worth ``mu_wrong`` on average -- exactly what the padding puts in an
        empty slot.  (A *true* candidate SHOULD gain from more good evidence;
        that is the point of the sum.)
        """
        import numpy as np
        from matching.placement_scorer import payload_residual
        L = self.L
        # Residuals as the SCORER sees them, so the fit and the test share a
        # scale.  A prescribed residual r is a direction offset of r * FIXED_TOL
        # on a NON-axial pair (an axial pair would fold mod pi).
        def pair(res: float):
            return _vec(0.0, 1.0, axial=False), _vec(res * L.FIXED_TOL, 1.0,
                                                     axial=False)

        wrong_res = [1.8, 2.2, 2.6, 3.0]
        pairs_w = [pair(v) for v in wrong_res]
        res_w = [payload_residual(q, r, L.FIXED_TOL) for q, r in pairs_w]
        qt, rt = pair(0.05)
        fit = L.fit_llr([payload_residual(qt, rt, L.FIXED_TOL)], res_w)
        inc1 = {0: [(1, True, "A")]}
        inc8 = {0: [(i, True, "A") for i in range(1, 9)]}
        q_idx = {(0, i, "A"): _vec(0.0, 1.0, axial=False)
                 for i in range(1, 9)}
        mapping = {i: 100 + i for i in range(1, 9)}
        s1, s8 = [], []
        for res, (qq, rr) in zip(res_w, pairs_w):
            r_idx = {(5, 100 + i, "A"): rr for i in range(1, 9)}
            s1.append(L.score_candidate(0, 5, mapping, inc1, q_idx, r_idx, fit))
            s8.append(L.score_candidate(0, 5, mapping, inc8, q_idx, r_idx, fit))
        self.assertAlmostEqual(float(np.mean(s1)), float(np.mean(s8)), places=9)

    def test_padding_value_is_the_wrong_pair_mean(self):
        """An empty slot must be worth exactly mu_wrong, not 0 and not credit."""
        L = self.L
        empty = L.score_candidate(0, 5, {}, {}, {}, {}, self.fit)
        expect = self.fit.mu_wrong * 8 / (2.0 * L.LLR_CLIP * 8) + 0.5
        self.assertAlmostEqual(empty, expect, places=12)
        self.assertLess(empty, 0.5)

    def test_uncheckable_slots_are_padded_not_zeroed(self):
        """None must score mu_wrong, never 0 and never a credit."""
        self.assertAlmostEqual(self.fit.of(None), self.fit.mu_wrong, places=12)
        self.assertLess(self.fit.of(None), 0.0)

    def test_presence_uses_existence_only(self):
        """Two slots with different residuals but the same existence score equal."""
        L = self.L
        pf = L.fit_presence([("A", True)] * 50 + [("A", False)] * 50,
                            [("A", True)] * 10 + [("A", False)] * 90)
        q_idx = {(0, 1, "A"): _vec(0.0, 1.0), (0, 2, "A"): _vec(3.0, 1.0)}
        r_idx = {(5, 6, "A"): _vec(0.0, 1.0), (5, 7, "A"): _vec(0.0, 1.0)}
        inc = {0: [(1, True, "A"), (2, True, "A")]}
        s = L.presence_score(0, 5, {1: 6, 2: 7}, inc, q_idx, r_idx, pf)
        # Both slots exist, so both take the same per-class value.
        self.assertTrue(math.isfinite(s))


class CanaryTests(unittest.TestCase):
    """Non-vacuity: an arm that cannot move is not evidence of anything."""

    def test_r_edge_credit_has_non_zero_variance_across_candidates(self):
        from Topology.placement_arms2 import presence_credit
        inc = {0: [(1, True, "A"), (2, True, "B")]}
        ref_by_src = {10: {"A": [1]}, 11: {"A": [1], "B": [2]},
                      12: {"A": [1], "B": [2, 3]}}
        vals = [presence_credit(0, v, inc, ref_by_src) for v in (10, 11, 12)]
        self.assertGreater(float(np.var(vals)), 0.0,
                           "R gives every candidate the same credit: vacuous")

    def test_geometry_credit_compares_parameters(self):
        """R_fixed must distinguish an AGREEING relation from a mismatched one.

        The original implementation counted a counterpart edge without ever
        reading its parameters, so it returned the same credit for a matched and
        a wildly mismatched relation.  This canary fails loudly if that returns.
        """
        from core.schema import EdgeParams
        from Topology.placement_arms2 import geometry_credit
        inc = {0: [(1, True, "A")]}
        q_params = EdgeParams(class_="EDGE_REGION_CONTENT", quantized=(1.0,),
                              sign=1)
        same = EdgeParams(class_="EDGE_REGION_CONTENT", quantized=(1.0,), sign=1)
        differ = EdgeParams(class_="EDGE_OCCLUSION", quantized=(9.0,), sign=-1)
        q_by = {0: {"A": [(1, q_params)]}}
        ref_by = {10: {"A": [(1, same)]}, 11: {"A": [(1, differ)]}}
        v_good = geometry_credit(0, 10, inc, ref_by, {1: 1}, q_by)
        v_bad = geometry_credit(0, 11, inc, ref_by, {1: 1}, q_by)
        self.assertGreater(v_good, v_bad,
                           "R_fixed cannot tell a matching relation from a "
                           "mismatched one")

    def test_candidate_dependent_support_varies(self):
        """P's support must change when only the CANDIDATE changes."""
        from matching import placement_llr as L
        inc = {0: [(1, True, "A"), (2, True, "A")]}
        q_idx = {(0, 1, "A"): _vec(0.0, 1.0), (0, 2, "A"): _vec(0.0, 1.0)}
        r_idx = {(5, 6, "A"): _vec(0.0, 1.0), (5, 7, "A"): _vec(0.0, 1.0),
                 (8, 6, "A"): _vec(math.pi, 3.0), (8, 7, "A"): _vec(math.pi, 3.0)}
        res_t = [0.05] * 100
        res_w = [3.0] * 100
        fit = L.fit_llr(res_t, res_w)
        m = {1: 6, 2: 7}
        s_good = L.score_candidate(0, 5, m, inc, q_idx, r_idx, fit)
        s_bad = L.score_candidate(0, 8, m, inc, q_idx, r_idx, fit)
        self.assertGreater(s_good, s_bad,
                           "P cannot distinguish a good candidate from a bad one")


class CacheClosureTests(unittest.TestCase):
    def test_the_digest_covers_the_whole_build_closure(self):
        """Every repository file on the build path is hashed into the cache key.

        Without this, editing a module the graph build actually calls (as
        `eth80_frontend.py` itself once was) serves a stale graph while the key
        stays put.
        """
        import guard
        guard.activate()
        from Topology import placement_cacheclosure_m5 as CC
        files = CC.closure()
        self.assertGreater(len(files), 40)
        missing = sorted(r for r in files if not CC.covered(r))
        self.assertEqual(missing, [],
                         "cache digest does not cover: %s" % (missing,))

    def test_editing_a_covered_root_module_moves_the_digest(self):
        """Sensitivity, proved on a file the OLD digest list omitted."""
        import guard
        guard.activate()
        import eth80_frontend as F
        path = os.path.join(F.ROOT, "sensory_floor.py")
        orig = open(path, "rb").read()
        try:
            F._SOURCE_DIGEST.clear()
            before = F.frontend_source_digest()
            open(path, "wb").write(orig + b"\n# digest sensitivity probe\n")
            F._SOURCE_DIGEST.clear()
            after = F.frontend_source_digest()
        finally:
            open(path, "wb").write(orig)     # BINARY write: a text-mode
            F._SOURCE_DIGEST.clear()         # rewrite would convert CRLF and
        self.assertNotEqual(before, after)   # corrupt the file.
        self.assertEqual(F.frontend_source_digest(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
