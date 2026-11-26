"""Mission 25 -- more stored views per object, and the guarantees that come with it.

    the rule            k = 1 is `000-000`; k = 4 is the coverage-optimal set; queries are never
                        stored; the covering radii are the declared ones; the rule takes no score
    k = 1 is M24        a max over one view is that view, so the one-view path cannot drift
    the best view       every shortlisted object reports the stored view the door chose for it
    the ceiling         the shortlist is S OBJECTS, so chance stays S/N_objects
    the pair cache      keyed on the matcher's own digest, so a re-read is free and an edit to the
                        matcher is not
    the equivalence     the recorded k = 1 check against the Mission-23 rows (80/80 and 44/44)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from matching import placement_views as V                                      # noqa: E402

CORPUS = os.path.join(ROOT, "data", "ETH-80-master")
QUERIES = ("035-045", "045-180")


def _p(i: int, view: str) -> str:
    return os.path.join(CORPUS, "1", str(i), "apple%d-%s.png" % (i, view))


def _have_images() -> bool:
    return all(os.path.exists(_p(i, "000-000")) for i in (1, 2, 3))


class TestTheRule(unittest.TestCase):
    """Item 1: the rule, the distances, the reserved views."""

    def test_grid_is_41_views_and_spacing_is_22(self) -> None:
        self.assertEqual(len(V.GRID), 41)
        self.assertAlmostEqual(V.min_spacing(), 22.0, places=6)

    def test_k1_is_the_base_view(self) -> None:
        self.assertEqual(V.reference_views(1, QUERIES), ["000-000"])
        self.assertAlmostEqual(V.covering_radius(["000-000"]), 90.0, places=6)

    def test_k4_is_the_declared_optimum(self) -> None:
        refs = V.reference_views(4, QUERIES)
        self.assertEqual(refs, ["000-000", "045-000", "066-117", "066-243"])
        self.assertAlmostEqual(V.covering_radius(refs), 65.5, places=2)
        self.assertIn("000-000", refs)                     # pinned: k = 1 is a subset of k = 4

    def test_queries_are_never_stored(self) -> None:
        for k in (1, 2, 3, 4):
            refs = V.reference_views(k, QUERIES)
            self.assertEqual(set(refs) & set(QUERIES), set(), "k=%d stored a query view" % k)
            self.assertEqual(len(refs), k)

    def test_rule_is_deterministic_and_takes_no_score(self) -> None:
        a = V.reference_views(4, QUERIES)
        b = V.reference_views(4, QUERIES)
        self.assertEqual(a, b)
        import inspect
        params = list(inspect.signature(V.reference_views).parameters)
        self.assertEqual(params, ["k", "queries", "base", "pin_base"])   # nothing score-shaped

    def test_declared_query_distances(self) -> None:
        """The two numbers the report leans on: 35.0 -> 30.0 and 45.0 -> 45.0 degrees."""
        k1 = V.distances_to_nearest(V.reference_views(1, QUERIES), QUERIES)
        k4 = V.distances_to_nearest(V.reference_views(4, QUERIES), QUERIES)
        self.assertAlmostEqual(k1["035-045"]["deg"], 35.0, places=6)
        self.assertAlmostEqual(k4["035-045"]["deg"], 30.0, places=6)
        self.assertAlmostEqual(k1["045-180"]["deg"], 45.0, places=6)
        self.assertAlmostEqual(k4["045-180"]["deg"], 45.0, places=6)

    def test_label_round_trip(self) -> None:
        self.assertEqual(V.label_of("/x/apple1-035-045.png"), "035-045")
        self.assertEqual(V.view_suffix("035-045"), "-035-045")


class TestTheBundleNamesTheViews(unittest.TestCase):
    def test_bundle_carries_references_and_reserved(self) -> None:
        from matching import placement_bundle as PB
        b = PB.PlacementBundle.load()
        self.assertTrue(b.reference_views, "the bundle does not name its stored views")
        self.assertEqual(set(b.reserved), set(QUERIES))
        self.assertEqual(set(b.reference_views) & set(b.reserved), set())
        for v in b.reference_views:
            self.assertIn(v, V.GRID)
        self.assertIn("rule", b.references)
        self.assertGreaterEqual(int(b.references.get("k", 0)), 1)

    def test_bundle_refuses_a_stored_query(self) -> None:
        from matching import placement_bundle as PB
        from matching import placement_recognise as RC
        b = PB.PlacementBundle.load()
        bad = PB.PlacementBundle(
            name=b.name, mission=b.mission, date=b.date, shortlist=b.shortlist, rrf_k=b.rrf_k,
            rerank=b.rerank, door_n_hits=b.door_n_hits, appearance=b.appearance,
            matcher_weight=b.matcher_weight, recipe=dict(b.recipe),
            recipe_extra=dict(b.recipe_extra), keypoint_class=b.keypoint_class,
            reference_views=["000-000", "035-045"], reserved=list(QUERIES),
            references=dict(b.references), provenance=dict(b.provenance),
            code_digest=b.code_digest)
        with self.assertRaises(PB.BundleRefusal):
            RC._validate(bad)


class TestMaxOverViews(unittest.TestCase):
    """The arithmetic that makes k = 1 equal Mission 24 and k = 4 a max, not a sum."""

    def test_one_view_is_the_identity(self) -> None:
        from matching import placement_recognise as RC
        x = np.asarray([[0.1, 0.9, 0.3]])
        got = RC._max_over_views(x, 3, 1)
        np.testing.assert_array_equal(got, x)

    def test_max_over_views_is_a_max_not_a_mean_or_sum(self) -> None:
        from matching import placement_recognise as RC
        x = np.asarray([[0.1, 0.9, 0.3, 0.2, 0.5, 0.4]])       # 3 objects x 2 views
        got = RC._max_over_views(x, 3, 2)
        np.testing.assert_allclose(got, [[0.9, 0.3, 0.5]])
        self.assertNotAlmostEqual(float(got[0, 0]), float(x[0, :2].mean()))

    def test_best_view_ties_go_to_the_lowest_index(self) -> None:
        from matching import placement_recognise as RC
        x = np.asarray([[0.4, 0.4, 0.1, 0.7]])
        self.assertEqual(RC._argmax_over_views(x, 2, 2), [0, 1])

    def test_chance_is_objects_not_views(self) -> None:
        """S = 8 objects on an 80-object gallery: chance 0.10 whatever k is."""
        from Topology import placement_m25_read as R
        rows = []
        for arm in ("k1", "k4"):
            for view in R.VIEWS:
                for i in range(80):
                    rows.append({"arm": arm, "view": view, "object": "o%02d" % i, "cls": "c",
                                 "in_shortlist": 0, "assembled": 0, "best_view_truth": "000-000",
                                 "sec": {"door": 1.0, "matcher": 1.0, "total": 2.0}})
        out = R.summarise(rows)
        self.assertAlmostEqual(out["chance"]["door_recall_s"], 0.1, places=9)
        self.assertAlmostEqual(out["chance"]["top1"], 0.0125, places=9)
        for arm, d in out["arms"].items():
            self.assertEqual(d["n"], 160)


@unittest.skipUnless(_have_images(), "ETH-80 corpus not present")
class TestEntryPointWithSeveralViews(unittest.TestCase):
    """The entry point stores k views per object and reports which one won."""

    def test_best_view_is_reported_for_every_shortlisted_object(self) -> None:
        from matching import placement_recognise as RC
        vs = ["000-000", "090-000"]
        items = [("apple/%d" % i, {v: _p(i, v) for v in vs}) for i in (1, 2, 3)]
        g = RC.build_gallery(items, views=vs, workers=1)
        self.assertEqual(g.views, vs)
        self.assertEqual(g.n, 3)
        self.assertEqual(g.n_flat, 6)
        r = RC.recognise(_p(4, "035-045"), g, workers=0, matcher=False)
        self.assertEqual(sorted(r.best_view), sorted(r.shortlist))
        for j, v in r.best_view.items():
            self.assertIn(v, vs)
            self.assertIn(j, range(3))
        self.assertEqual(len(r.shortlist), min(8, 3))

    def test_gallery_refuses_a_missing_view(self) -> None:
        from matching import placement_recognise as RC
        items = [("apple/1", {"000-000": _p(1, "000-000")})]
        with self.assertRaises(ValueError):
            RC.build_gallery(items, views=["000-000", "090-000"], workers=1)


class TestPairCache(unittest.TestCase):
    """Item 5: a pair score is cached against the matcher, not against the call."""

    def test_key_depends_on_the_weight_and_the_matcher(self) -> None:
        from matching import placement_recognise as RC
        a = RC._pair_key("/q/x.png", "/r/y.png", 0.375)
        b = RC._pair_key("/q/x.png", "/r/y.png", 0.5)
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith(RC.matcher_digest()))

    def test_round_trip_and_foreign_digest_is_ignored(self) -> None:
        from matching import placement_recognise as RC
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "pairs.jsonl")
            RC.append_pair_cache([{"q": "/q/x.png", "r": "/r/y.png", "w": 0.375, "B": 0.5}], p)
            got = RC.load_pair_cache(p)
            self.assertEqual(len(got), 1)
            with open(p, "a") as fh:                        # a line from another matcher
                fh.write(json.dumps({"d": "0000000000000000", "q": "z.png", "r": "w.png",
                                     "w": 0.375, "B": 0.9}) + "\n")
            self.assertEqual(len(RC.load_pair_cache(p)), 1)


class TestK1EquivalenceArtifact(unittest.TestCase):
    """Item 3: the recorded check that k = 1 is Mission 24, on the record's own data."""

    def test_recorded_k1_equivalence(self) -> None:
        p = os.path.join(ROOT, "Topology", "results", "placement", "m25_k1_equivalence.json")
        if not os.path.exists(p):
            self.skipTest("m25_k1_equivalence.json not produced yet")
        d = json.load(open(p))
        self.assertEqual(d["door"]["mismatched"], 0, "the k = 1 door is not Mission 24's door")
        self.assertEqual(d["matcher"]["mismatched"], 0, "the k = 1 order is not Mission 24's")
        self.assertEqual(d["door"]["equal"], d["door"]["checked"])
        self.assertGreaterEqual(d["matcher"]["queries_with_full_pair_cache"], 40)


if __name__ == "__main__":
    unittest.main()
