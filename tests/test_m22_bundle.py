"""Mission 22 bundle tests: the front doors, the assembler, the class score and the power code.

Everything here is arithmetic and structure -- no graphs, no images -- so the suite stays fast
and these tests stay honest about what they pin: the SHAPE of the mechanism (what it may read,
what it may not drop, how it ties, how it normalises) rather than a number that a rebuild could
move.
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


def _entry(rows, spacing=None):
    return {"path": "x", "places": rows, "dropped": {"no_sig": 0, "no_config": 0,
                                                     "no_position": 0, "no_scale": 0},
            "spacing_median": 0.0 if spacing is None else spacing,
            "n_payload_places": len(rows), "sec": 0.0}


def _row(n, x, y, s=1.0, theta=None, dz=None, key=None):
    dz = [0.0] * 4 if dz is None else dz
    return {"n": int(n), "lens": "L", "key": ("L", (1, 2, 3, 4)) if key is None else key,
            "Dz": list(dz), "Mz": [1.0] * len(dz), "x": float(x), "y": float(y),
            "s": float(s), "theta": theta}


class TestLengthUnit(unittest.TestCase):
    """The defect that cost the door its recall: a zero length unit."""

    def test_colocated_places_are_not_neighbours(self):
        from matching import placement_frontdoor as FD
        e = _entry([_row(1, 10.0, 10.0), _row(2, 10.0, 10.0), _row(3, 14.0, 10.0)])
        sc, unit = FD.local_scales(e)
        self.assertGreater(unit, 0.0)
        self.assertAlmostEqual(unit, 4.0, places=6)      # median of {4, 4, 4}
        self.assertEqual(len(sc), 3)

    def test_unit_is_positive_even_when_every_place_coincides(self):
        from matching import placement_frontdoor as FD
        e = _entry([_row(i, 5.0, 5.0) for i in range(6)])
        sc, unit = FD.local_scales(e)
        self.assertTrue(np.isfinite(unit))
        self.assertGreater(unit, 0.0)

    def test_empty_entry_does_not_divide_by_zero(self):
        from matching import placement_frontdoor as FD
        sc, unit = FD.local_scales(_entry([]))
        self.assertEqual(sc, [])
        self.assertGreater(unit, 0.0)

    def test_side_spacing_is_not_the_unit(self):
        # the shipped Side.spacing_median is 0 on most of these images; the door must not use it
        from matching import placement_frontdoor as FD
        e = _entry([_row(1, 0.0, 0.0), _row(2, 3.0, 0.0)], spacing=0.0)
        sc, unit = FD.local_scales(e)
        self.assertGreater(unit, 1.0)


class TestVote(unittest.TestCase):
    def test_identical_entries_vote_one_transform(self):
        from matching import placement_frontdoor as FD
        # every place carries a DISTINCT configuration row, so its only true partner is itself
        rows = [_row(i, 3.0 * i, 2.0 * i, dz=[float(i)] * 4) for i in range(8)]
        idx = FD.PlaceIndex([_entry(rows)], ["k"])
        out = idx.vote(_entry(rows), None, n_hits=1)
        self.assertEqual(out["votes"], 8)
        # every vote is the identity transform: the histogram holds one cell
        self.assertEqual(out["bins"], 1)
        self.assertGreater(out["scores"][0], 0.0)

    def test_the_measured_spacing_defect_stays_fixed(self):
        # with the shipped Side.spacing_median (0.0 on these images) every vote got its own
        # bin; the door must now produce a real histogram from the leaf geometry alone
        from matching import placement_frontdoor as FD
        rows = [_row(i, 3.0 * i, 2.0 * i, dz=[float(i)] * 4) for i in range(8)]
        idx = FD.PlaceIndex([_entry(rows)], ["k"])
        out = idx.vote(_entry(rows), 0.0, n_hits=1)
        self.assertEqual(out["bins"], 1)

    def test_rotation_bin_wraps_at_pi(self):
        from matching import placement_frontdoor as FD
        nb = int(round(math.pi / FD.BINS["theta"]))
        # dtheta just below pi and just above 0 must land in the same axial bin
        a = (math.pi - 1e-3) / FD.BINS["theta"]
        b = (1e-3) / FD.BINS["theta"]
        self.assertEqual(int(round(a)) % nb, int(round(b)) % nb)

    def test_cfg_distance_is_symmetric_and_needs_two_slots(self):
        from matching import placement_frontdoor as FD
        a = _row(1, 0.0, 0.0, dz=[0.0, 0.0, 0.0])
        b = _row(2, 0.0, 0.0, dz=[1.0, 1.0, 1.0])
        self.assertAlmostEqual(FD.cfg_distance(a, b), FD.cfg_distance(b, a), places=12)
        thin = {"Dz": [1.0, 0.0, 0.0], "Mz": [1.0, 0.0, 0.0]}
        self.assertEqual(FD.cfg_distance(a, thin), float("inf"))


class TestCountDoor(unittest.TestCase):
    def test_normalisation_is_the_stated_one(self):
        from matching import placement_frontdoor as FD
        gal = [_entry([_row(i, 2.0 * i, 0.0) for i in range(4)])]
        q = _entry([_row(0, 0.0, 0.0), _row(1, 2.0, 0.0)])
        idx = FD.PlaceIndex(gal, ["a"])
        out, diag = FD.count_scores(idx, [q], n_hits=1)
        used = diag[0]["places_used"]
        self.assertEqual(used, 2)
        # each of the two query places counts exactly one partner, in the one gallery object
        self.assertAlmostEqual(out[0, 0], 2.0 / math.sqrt(used * len(gal[0]["places"])), places=12)

    def test_a_place_with_nothing_in_the_pool_scores_nothing(self):
        from matching import placement_frontdoor as FD
        gal = [_entry([_row(i, 2.0 * i, 0.0) for i in range(4)])]
        q = _entry([_row(0, 0.0, 0.0, key=("Z", (9, 9)))])
        idx = FD.PlaceIndex(gal, ["a"])
        out, diag = FD.count_scores(idx, [q], n_hits=1)
        self.assertEqual(diag[0]["places_used"], 0)
        self.assertEqual(float(out[0, 0]), 0.0)

    def test_scores_are_independent_of_the_gallery_order(self):
        # distinct configuration rows, so the ranking is by distance and not by tie-break
        from matching import placement_frontdoor as FD
        a = _entry([_row(i, 2.0 * i, 0.0, dz=[0.1 * i] * 4) for i in range(4)])
        b = _entry([_row(i, 2.0 * i, 1.0, dz=[0.1 * i + 0.05] * 4) for i in range(4)])
        q = _entry([_row(0, 0.0, 0.0, dz=[0.0] * 4)])
        s1, _ = FD.count_scores(FD.PlaceIndex([a, b], ["a", "b"]), [q], n_hits=2)
        s2, _ = FD.count_scores(FD.PlaceIndex([b, a], ["b", "a"]), [q], n_hits=2)
        self.assertAlmostEqual(float(s1[0, 0]), float(s2[0, 1]), places=12)
        self.assertAlmostEqual(float(s1[0, 1]), float(s2[0, 0]), places=12)


class TestAppearance(unittest.TestCase):
    def test_feature_is_unit_length_and_deterministic(self):
        import tempfile
        from PIL import Image
        from matching import placement_frontdoor as FD
        rng = np.random.default_rng(3)
        img = (rng.random((32, 40, 3)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.png")
            Image.fromarray(img).save(p)
            f1 = FD.appearance_feature(p)
            f2 = FD.appearance_feature(p)
        self.assertAlmostEqual(float(np.linalg.norm(f1)), 1.0, places=9)
        self.assertTrue(np.array_equal(f1, f2))
        self.assertEqual(f1.shape, (76,))                       # 4x4x4 colour + 12 gradient


class TestFusionAndShortlist(unittest.TestCase):
    def test_rrf_has_no_weight_and_prefers_agreement(self):
        # A ranks 0 first and B ranks it last; 1 is second in A and first in B, and RRF must
        # put 1 on top.  A weighted sum could be tuned to disagree; RRF has nothing to tune.
        from matching import placement_frontdoor as FD
        a = np.array([[9.0, 8.0, 7.0, 1.0]])
        b = np.array([[1.0, 9.0, 8.0, 7.0]])
        f = FD.rank_fuse(a, b)
        self.assertEqual(int(np.argmax(f[0])), 1)

    def test_rrf_is_scale_free(self):
        from matching import placement_frontdoor as FD
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[0.0, 5.0, 1.0]])
        f1 = FD.rank_fuse(a, b)
        f2 = FD.rank_fuse(a * 1000.0, b + 7.0)
        self.assertTrue(np.allclose(f1, f2))

    def test_shortlist_keeps_size_and_ties_by_index(self):
        from matching import placement_frontdoor as FD
        m = np.array([[0.5, 0.5, 0.5, 0.9]])
        self.assertEqual(FD.shortlist(m, 0, 2), [3, 0])   # best first, then the lowest index
        self.assertEqual(len(FD.shortlist(m, 0, 4)), 4)
        self.assertEqual(FD.shortlist(m, 0, 2), FD.shortlist(m, 0, 2))


class TestAssemble(unittest.TestCase):
    def test_it_reorders_and_never_drops(self):
        from matching import placement_assemble as AS
        short = [4, 7, 9, 1]
        scores = {4: 0.2, 7: 0.9, 9: 0.5, 1: 0.1}
        ordered, tab = AS.assemble(short, lambda j: scores[j])
        self.assertEqual(ordered, [7, 9, 4, 1])
        self.assertEqual(sorted(ordered), sorted(short))

    def test_an_unscoreable_candidate_is_kept_and_not_promoted(self):
        from matching import placement_assemble as AS
        def score_of(j):
            if j == 9:
                raise RuntimeError("no placement side")
            return {4: 0.2, 7: 0.9, 1: 0.1}[j]
        ordered, tab = AS.assemble([4, 7, 9, 1], score_of)
        self.assertEqual(ordered[-1], 9)
        self.assertEqual(len(ordered), 4)

    def test_top1_counts_absence_as_a_miss(self):
        from matching import placement_assemble as AS
        self.assertEqual(AS.top1([], 3), 0)
        self.assertEqual(AS.top1(None, 3), 0)
        self.assertEqual(AS.top1([3], 3), 1)
        self.assertEqual(AS.top1([1, 3], 3), 0)


class TestClassScore(unittest.TestCase):
    def setUp(self):
        from matching import placement_classscore as CS
        self.CS = CS
        self.class_of = {0: "a", 1: "a", 2: "b", 3: "c", 4: "c"}

    def test_top2_max_mean_are_what_they_say(self):
        s = {0: 0.9, 1: 0.4, 2: 0.5, 3: 0.2, 4: 0.1}
        self.assertAlmostEqual(self.CS.class_scores(s, self.class_of, "top2")["a"], 1.3)
        self.assertAlmostEqual(self.CS.class_scores(s, self.class_of, "max")["a"], 0.9)
        self.assertAlmostEqual(self.CS.class_scores(s, self.class_of, "mean")["a"], 0.65)
        self.assertAlmostEqual(self.CS.class_scores(s, self.class_of, "top2")["c"], 0.3)

    def test_a_class_with_one_shortlisted_instance_scores_that_instance(self):
        s = {2: 0.7}
        for agg in ("top2", "max", "mean"):
            self.assertAlmostEqual(self.CS.class_scores(s, self.class_of, agg)["b"], 0.7)

    def test_unknown_aggregate_is_an_error_not_a_default(self):
        with self.assertRaises(KeyError):
            self.CS.class_scores({0: 1.0}, self.class_of, "median")

    def test_it_cannot_read_the_query_label(self):
        # structural: the only arguments are evidence and the GALLERY's class table
        sig = inspect.signature(self.CS.class_scores)
        self.assertEqual(list(sig.parameters), ["instance_scores", "class_of", "agg"])
        # behavioural: deleting the query's own class from the table changes nothing about the
        # other classes, and the query's class score is a function of its instances only
        s = {0: 0.9, 1: 0.4, 2: 0.5, 3: 0.2}
        full = self.CS.class_scores(s, self.class_of, "top2")
        subset = self.CS.class_scores({k: v for k, v in s.items() if self.class_of[k] != "a"},
                                      self.class_of, "top2")
        self.assertEqual(set(full) - set(subset), {"a"})
        self.assertEqual(subset, {k: v for k, v in full.items() if k != "a"})

    def test_class_top1_reads_the_label_once_and_breaks_ties_by_name(self):
        self.assertEqual(self.CS.class_top1({"a": 1.0, "b": 2.0}, "b"), 1)
        self.assertEqual(self.CS.class_top1({"a": 1.0, "b": 2.0}, "a"), 0)
        # an exact tie is broken on the class NAME, deterministically and label-blind
        self.assertEqual(self.CS.class_top1({"a": 1.0, "b": 1.0}, "b"), 1)
        self.assertEqual(self.CS.class_top1({"a": 1.0, "b": 1.0}, "a"), 0)
        self.assertEqual(self.CS.class_top1({}, "a"), 0)

    def test_chance(self):
        # 8 classes, 2 instances each, gallery 16: a random top-1 is the right class 2/16,
        # and a random top-8 misses both same-class instances with probability C(14,8)/C(16,8)
        self.assertAlmostEqual(self.CS.chance_top1(2, 16), 0.125)
        self.assertAlmostEqual(self.CS.chance_at_s(2, 16, 8),
                               1.0 - 3003.0 / 12870.0, places=9)
        self.assertAlmostEqual(self.CS.chance_at_s(0, 16, 8), 0.0)
        self.assertAlmostEqual(self.CS.chance_at_s(15, 16, 8), 1.0)


class TestPower(unittest.TestCase):
    def test_the_registration_numbers(self):
        from Topology import placement_m22_power as PW
        self.assertAlmostEqual(PW.mde(16), 0.706, places=3)
        self.assertAlmostEqual(PW.power(16, 0.5), 0.405, places=3)
        self.assertLess(PW.mde(40), 0.5)          # FIT can resolve the planning effect
        self.assertGreater(PW.mde(16), 0.5)       # FRESH cannot

    def test_power_grows_with_n_and_with_effect(self):
        from Topology import placement_m22_power as PW
        self.assertLess(PW.power(16, 0.6), PW.power(40, 0.6))
        self.assertLess(PW.power(40, 0.3), PW.power(40, 0.6))

    def test_mde_is_undefined_when_even_the_largest_effect_is_unreachable(self):
        from Topology import placement_m22_power as PW
        self.assertTrue(math.isnan(PW.mde(1)))


class TestRanksMetrics(unittest.TestCase):
    """The chance column.  Mission 21 printed the top-1 chance beside recall@S; it is S/N."""

    def _objs(self, n, per_class=5):
        return [{"key": "k%d" % i, "cls": "c%d" % (i // per_class)} for i in range(n)]

    def test_a_flat_score_matrix_reproduces_the_chance_exactly(self):
        from Topology import placement_m22_eth80 as H
        n, s = 20, 8
        objs = self._objs(n)
        keys = ["k%d" % i for i in range(n)]
        m = H.ranks_metrics(np.zeros((n, n)), objs, keys, s)
        # every score ties, so the truth's rank is its index + 1: exactly the first s indices hit
        self.assertAlmostEqual(m["recall_s"], s / n, places=12)
        self.assertAlmostEqual(m["chance_recall_s"], s / n, places=12)
        self.assertAlmostEqual(m["top1"], 1.0 / n, places=12)
        self.assertAlmostEqual(m["chance_top1"], 1.0 / n, places=12)
        self.assertEqual(m["n"], n)

    def test_a_perfect_ranking_and_a_missing_truth(self):
        from Topology import placement_m22_eth80 as H
        n, s = 4, 2
        objs = self._objs(n, per_class=2)
        keys = ["k%d" % i for i in range(n)]
        eye = np.eye(n)
        m = H.ranks_metrics(eye, objs, keys, s)
        self.assertEqual((m["recall_s"], m["top1"]), (1.0, 1.0))
        # a truth that is not in the gallery at all is a miss, and it does not inflate n
        m2 = H.ranks_metrics(eye, objs + [{"key": "nope", "cls": "c0"}], keys, s)
        self.assertEqual(m2["n"], n)
        self.assertEqual(m2["top1"], 1.0)


if __name__ == "__main__":
    unittest.main()
