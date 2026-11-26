"""Mission 23 bundle tests: the T2 guard, the two final orders, the bootstrap and the read.

No graphs, no images: these pin the SHAPE of what Mission 23 added -- that T2 cannot be scored on a
gallery that still contains the query's own object, that a fusion of two orders over the same
candidates can neither add nor drop one, that the paired bootstrap is deterministic and centred,
and that the read's registered view order and chance columns are what the report says.
"""
from __future__ import annotations

import inspect
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]


class TestT2Gallery(unittest.TestCase):
    """Item 1's whole point: the query's own object must not be in the gallery."""

    def test_it_removes_the_query_own_object(self):
        from matching import placement_classscore as CS
        keys = ["apple/1", "apple/2", "car/1", "car/2"]
        out = CS.t2_gallery(keys, "apple/1")
        self.assertEqual(out, ["apple/2", "car/1", "car/2"])
        self.assertEqual(len(out), len(keys) - 1)
        self.assertNotIn("apple/1", out)

    def test_it_refuses_a_gallery_without_the_query_own_object(self):
        from matching import placement_classscore as CS
        with self.assertRaises(ValueError):
            CS.t2_gallery(["apple/2", "car/1"], "apple/1")

    def test_scoring_refuses_a_gallery_that_still_contains_it(self):
        """THE TEST THE MISSION ASKED FOR: it fails if the query's own object is present."""
        from matching import placement_classscore as CS
        inst = {0: 0.9, 1: 0.8, 2: 0.1}
        class_of = {0: "apple", 1: "apple", 2: "car"}
        # the reduced gallery scores normally ...
        cs = CS.class_scores_t2(inst, class_of, "apple/1", ["apple/2", "car/1"], agg="top2")
        self.assertAlmostEqual(cs["apple"], 1.7)
        # ... and the UNREDUCED one raises instead of quietly returning an inflated number
        with self.assertRaises(ValueError):
            CS.class_scores_t2(inst, class_of, "apple/1", ["apple/1", "apple/2", "car/1"])

    def test_reduced_scoring_equals_plain_scoring(self):
        from matching import placement_classscore as CS
        inst = {0: 0.9, 1: 0.8}
        class_of = {0: "apple", 1: "apple"}
        a = CS.class_scores_t2(inst, class_of, "apple/9", ["apple/2", "car/1"], agg="max")
        b = CS.class_scores(inst, class_of, agg="max")
        self.assertEqual(a, b)


class TestFinalOrders(unittest.TestCase):
    def test_plain_reorder_drops_nothing_and_is_stable(self):
        from Topology import placement_m23_eth80 as H
        cands = [3, 7, 1]
        out = H.order_plain(cands, {3: 0.2, 7: 0.9, 1: 0.5})
        self.assertEqual(out, [7, 1, 3])
        self.assertEqual(sorted(out), sorted(cands))
        # a candidate with no score keeps a place, at the end, by index
        self.assertEqual(H.order_plain(cands, {7: 0.9}), [7, 1, 3])

    def test_fusion_keeps_the_candidate_set(self):
        from Topology import placement_m23_eth80 as H
        cands = [0, 4, 9, 2]
        out = H.order_fused(cands, {0: 0.1, 4: 0.2, 9: 0.3, 2: 0.4})
        self.assertEqual(sorted(out), sorted(cands))
        self.assertEqual(len(out), len(cands))

    def test_fusion_agrees_with_a_confident_matcher(self):
        """When the matcher's order IS the door's order the fusion cannot disagree with it."""
        from Topology import placement_m23_eth80 as H
        cands = [5, 6, 7]
        sc = {5: 0.9, 6: 0.5, 7: 0.1}            # same order as `cands`
        self.assertEqual(H.order_fused(cands, sc), H.order_plain(cands, sc))

    def test_fusion_is_deterministic_and_k_is_fixed(self):
        from Topology import placement_m23_eth80 as H
        cands = [2, 0, 1]
        sc = {2: 0.4, 0: 0.4, 1: 0.4}
        a = H.order_fused(cands, sc)
        b = H.order_fused(cands, sc)
        self.assertEqual(a, b)
        src = inspect.getsource(H.order_fused)
        self.assertIn("k: int = 60", src)        # no tuned weight, pinned


class TestBootstrap(unittest.TestCase):
    def test_it_is_deterministic_at_the_registered_seed(self):
        from Topology import placement_m23_eth80 as H
        rows = {("o%d" % i): (float(i % 2), float((i + 1) % 2)) for i in range(40)}
        a = H.bootstrap_paired(rows)
        b = H.bootstrap_paired(rows)
        self.assertEqual(a, b)
        self.assertEqual(a["n_objects"], 40)

    def test_identical_arms_differ_by_nothing(self):
        from Topology import placement_m23_eth80 as H
        rows = {("o%d" % i): (float(i % 2), float(i % 2)) for i in range(30)}
        r = H.bootstrap_paired(rows)
        self.assertEqual(r["diff"], 0.0)
        self.assertGreaterEqual(r["p_bootstrap"], 0.5)

    def test_ci_is_ordered_and_brackets_the_estimate(self):
        from Topology import placement_m23_eth80 as H
        rows = {("o%d" % i): (1.0 if i < 30 else 0.0, 0.0) for i in range(40)}
        r = H.bootstrap_paired(rows)
        lo, hi = r["ci95"]
        self.assertLessEqual(lo, r["diff"] + 1e-12)
        self.assertGreaterEqual(hi, r["diff"] - 1e-12)


class TestReadArithmetic(unittest.TestCase):
    def test_chance_columns_and_the_zero_pairs_rule(self):
        from Topology.placement_m23_stages import summarise_read
        rows = [{"object": "o1", "cls": "a", "view": "-035-045", "assembled": 1, "deployed": 0,
                 "appearance": 0, "door_in_s": 1, "deployed_in_s": 1, "needed_pairs": 1},
                {"object": "o2", "cls": "b", "view": "-035-045", "assembled": 0, "deployed": 0,
                 "appearance": 0, "door_in_s": 0, "deployed_in_s": 0, "needed_pairs": 0}]
        r = summarise_read(rows, ("-035-045",), 8, "fused", 80)
        self.assertAlmostEqual(r["chance_top1"], 1.0 / 80)
        self.assertAlmostEqual(r["chance_recall_s"], 8.0 / 80)
        self.assertAlmostEqual(r["rates"]["assembled"], 0.5)
        self.assertAlmostEqual(r["rates"]["deployed"], 0.0)
        self.assertEqual(r["queries_without_pairs"], 1)

    def test_the_registered_view_order_keeps_the_never_read_view(self):
        """The amendment in the registration: a partial read is whole, unseen views."""
        from Topology import placement_m23_eth80 as H
        doc = open(os.path.join(ROOT, "Topology", "results", "placement",
                                "MISSION23_PREREGISTRATION.md")).read()
        self.assertIn("`-035-045`, `-045-180`,\n   `-022-000`", doc.replace("\r", ""))
        self.assertEqual(H.VIEWS[0], "-022-000")     # the module default, unchanged for M22 compat


class TestPower(unittest.TestCase):
    def test_the_registered_mde_is_below_the_planning_effect(self):
        import json
        path = os.path.join(ROOT, "Topology", "results", "placement", "m23_power.json")
        d = json.load(open(path))
        self.assertLess(d["cells"]["w=1.0"]["mde"], 0.375)
        self.assertLess(d["cells"]["w=0.0"]["mde"], 0.375)


if __name__ == "__main__":
    unittest.main()
