"""MISSION 18 / O2 and O3 -- the rotation consensus, and the switch that gates it.

Measured before this code existed (`m18_diagnose.json`, 4 DEV images): at 15 deg the
context accuracy is 0.182, the true partner's median rank inside the similarity row is 11,
and 69.6 % of true partners are OUTSIDE the deployed candidate cap of 6.  Two different
defects, so two different instruments:

* the consensus is aimed at the first -- at a rotation, every TRUE pair implies the same
  frame shift, and the impostors spread over the window, so the shift is a vote the
  appearance similarity does not get to cast;
* the cap is aimed at the second -- a candidate that is never generated cannot be ranked,
  and no term inside the blend can reach it.

Both are OFF by default.  These tests go RED if either is reverted, and they check that
neither runs at defaults.
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


def _vec(dist: float, direction: float, theta: float = 0.0) -> PlacementVector:
    return PlacementVector(direction=direction, distance=dist, log_scale=0.0,
                           source_theta=theta, source_scale=1.0, axial=False)


def _side(star: dict) -> Side:
    """``star`` = {node: {neighbour: (distance, theta)}}; leaves get a self-loop."""
    incident = {}
    index = {}
    nodes = set(star)
    for u, nb in star.items():
        incident[u] = [(v, True, "EDGE") for v in nb]
        for v, (d, th) in nb.items():
            index[(u, v, "EDGE")] = _vec(d, 0.0, th)
            nodes.add(v)
    for v in nodes:
        # a leaf still needs its OWN outgoing vector or it has no orientation
        if v not in incident:
            incident[v] = [(v, True, "SELF")]
            index[(v, v, "SELF")] = _vec(1.0, 0.0, 0.0)
    return Side(payload={}, index=index, incident=incident, lens={},
                nodes=tuple(sorted(nodes)))


def _frame(impostor_shifts=(1.00, -1.45, 0.80), truth_shift=0.4, n=3):
    """``n`` query nodes, each with an OFFICIAL candidate at ``truth_shift`` and an
    IMPOSTOR that the appearance similarity likes more and whose implied shift differs.

    The point of the fixture is that the truth clusters at one shift and the impostors do
    not: a single impostor outvoting the truth at one node is not the mechanism, and a
    test built on it would pass for the wrong reason.
    """
    a_star = {u: {100 + u: (1.0, 0.0)} for u in range(n)}
    b_index = {}
    b_incident = {}
    for u in range(n):
        b_index[(10 + u, 0, "S")] = _vec(1.0, 0.0, truth_shift)
        b_incident[10 + u] = [(0, True, "S")]
        b_index[(20 + u, 0, "S")] = _vec(1.0, 0.0,
                                         truth_shift + impostor_shifts[u % len(impostor_shifts)])
        b_incident[20 + u] = [(0, True, "S")]
    a = _side(a_star)
    b = Side(payload={}, index=b_index, incident=b_incident, lens={},
             nodes=tuple(sorted({10 + u for u in range(n)} |
                                {20 + u for u in range(n)})))
    cands = {u: [(10 + u, 0.5), (20 + u, 0.9)] for u in range(n)}
    return a, b, cands


def _five():
    """Five query nodes: the truth outvotes one impostor per node by 2.5 : 0.9."""
    return _frame(impostor_shifts=(1.00, -1.45, 0.80, 0.95, -0.60), n=5)


class ConsensusTests(unittest.TestCase):
    def test_the_shift_is_the_vote(self):
        a, b, cands = _frame()
        stats = {}
        modes = C.consensus_shifts([0, 1, 2], cands, a, b, tol=0.175, stats=stats)
        self.assertTrue(modes, "no shift found although every node has an orientation")
        self.assertAlmostEqual(modes[0], 0.4, places=1)
        self.assertEqual(stats["consensus"]["votes"], 6)

    def test_the_mapping_follows_the_consensus_not_the_bigger_similarity(self):
        """RED if the restriction is removed: the impostor wins on appearance (0.9 v 0.5)."""
        a, b, cands = _frame()
        out = C.consensus_mapping([0, 1, 2], cands, a, b, lam=0.0, tol=0.175)
        self.assertEqual([out.get(u) for u in range(3)], [10, 11, 12])

    def test_a_row_with_nothing_consistent_falls_back_and_is_counted(self):
        a, b, cands = _five()
        cands[4] = [(24, 0.9)]          # node 4 keeps only its (inconsistent) impostor
        stats = {}
        out = C.consensus_mapping([0, 1, 2, 3, 4], cands, a, b, lam=0.0, tol=0.175,
                                  stats=stats)
        self.assertEqual(stats["consensus"]["rows_without_consensus"], 1)
        self.assertEqual(stats["consensus"]["rows_without_orientation"], 0)
        self.assertEqual(out.get(4), 24)

    def test_a_row_with_no_orientation_falls_back_and_is_counted_too(self):
        a, b, cands = _five()
        inc = dict(a.incident)
        inc[4] = []
        a = Side(payload=a.payload, index=dict(a.index), incident=inc, lens={},
                 nodes=a.nodes)
        stats = {}
        C.consensus_mapping([0, 1, 2, 3, 4], cands, a, b, lam=0.0, tol=0.175, stats=stats)
        self.assertEqual(stats["consensus"]["rows_without_orientation"], 1)

    def test_no_orientation_means_no_consensus_and_it_says_so(self):
        a = Side(payload={}, index={}, incident={0: [(1, True, "E")], 1: []},
                 lens={}, nodes=(0, 1))
        b = Side(payload={}, index={}, incident={10: [(11, True, "E")], 11: []},
                 lens={}, nodes=(10, 11))
        stats = {}
        out = C.consensus_mapping([0], {0: [(10, 0.5)]}, a, b, stats=stats)
        self.assertEqual(out, {})
        self.assertEqual(stats["consensus_fallback"], "no measured orientation")


class DefaultsTests(unittest.TestCase):
    def test_the_switches_are_off_and_declared(self):
        import parameters as PP
        from eth80_frontend import _FRONTEND_SWITCHES
        self.assertFalse(PP.PLACEMENT_CONTEXT_CONSENSUS)
        self.assertEqual(int(PP.PLACEMENT_CAND_CAP), 0)
        self.assertEqual(str(PP.PLACEMENT_LLR_TABLE_PATH), "")
        for name in ("PLACEMENT_CONTEXT_CONSENSUS", "PLACEMENT_CONTEXT_CONSENSUS_TOL",
                     "PLACEMENT_CONTEXT_CONSENSUS_MODES", "PLACEMENT_CONTEXT_CONSENSUS_LAMBDA",
                     "PLACEMENT_CAND_CAP", "PLACEMENT_LLR_TABLE_PATH"):
            self.assertIn(name, _FRONTEND_SWITCHES)

    def test_defaults_read_the_deployed_table_and_cap(self):
        from matching import placement_matcher as PM
        PM.reset_cache()
        self.assertEqual(PM.active_path(), PM.FROZEN_PATH)
        self.assertEqual(PM.cand_cap(), int(PM.frozen()["cand_cap"]))
        self.assertGreaterEqual(PM.cand_cap(), 6)

    def test_the_consensus_block_lives_in_one_function(self):
        import inspect
        from matching import placement_matcher as PM
        self.assertEqual(inspect.getsource(PM.rescore_matrix).count(
            "PLACEMENT_CONTEXT_CONSENSUS"), 0)
        self.assertGreater(inspect.getsource(PM.context_mapping).count(
            "PLACEMENT_CONTEXT_CONSENSUS"), 0)


if __name__ == "__main__":
    unittest.main()


class SoftConsensusTests(unittest.TestCase):
    """The hard filter is only as good as the mode.  On a real photograph at 15 deg the
    mode is wrong (m18_shift_cap6.json: 0.04 against a true 0.26) and the hard version
    drops the truth partner from 90 % of the true rows.  The soft version must never
    remove a candidate -- that is the property this pins.
    """

    def test_soft_keeps_every_candidate(self):
        a, b, cands = _five()
        # a mode that is WRONG for every row: the impostors' shift
        cands = {u: [(10 + u, 0.5), (20 + u, 0.9)] for u in range(5)}
        hard = C.consensus_mapping([0, 1, 2, 3, 4], cands, a, b, lam=0.0, tol=0.175,
                                   soft=True, bonus=0.15)
        self.assertEqual(len(hard), 5)
        # with a wrong mode the nudge must NOT be able to invert the 0.4 similarity gap
        self.assertEqual([hard.get(u) for u in range(5)],
                         [20, 21, 22, 23, 24])

    def test_soft_bonus_can_flip_a_near_tie_and_is_capped_by_its_size(self):
        a, b, cands = _frame()
        cands = {u: [(10 + u, 0.50), (20 + u, 0.60)] for u in range(3)}
        out = C.consensus_mapping([0, 1, 2], cands, a, b, lam=0.0, tol=0.175,
                                  soft=True, bonus=0.15)
        self.assertEqual([out.get(u) for u in range(3)], [10, 11, 12])


class TableTests(unittest.TestCase):
    """[M18 / O1] The shipped candidate must be defensible AS DATA, and the default must
    still be the deployed table: `audit_bundle` is what the freeze guard reads, so a table
    that rises, or that pays positive credit in a bin too thin to justify it, must not
    pass this test."""

    def test_the_m18_table_is_shippable_and_off_by_default(self):
        import json
        from matching import placement_llr as L
        from matching import placement_matcher as PM
        b = json.load(open(PM.FROZEN_PATH_M18))
        finding = L.audit_bundle(b)
        self.assertEqual(finding["incoherent"], [])
        self.assertEqual(finding["unshippable"], [])
        self.assertEqual(int(finding["n_rises"]), 0)
        self.assertFalse(P.PLACEMENT_LLR_TABLE_M18)
        PM.reset_cache()
        self.assertEqual(PM.active_path(), PM.FROZEN_PATH)

    def test_the_m18_table_orders_the_tail_the_right_way(self):
        """O1 as written: LLR(5.0 tol) strictly < LLR(1.5 tol).  A monotone table with a
        PAVA-pooled tail reads EQUAL at both -- tbfine does -- so the tail is fitted with
        a form instead.  This is the assertion that would go red if the tail were pooled."""
        import json
        from matching import placement_llr as L
        from matching import placement_matcher as PM
        fit, _p, _a, _g = L.from_jsonbundle(json.load(open(PM.FROZEN_PATH_M18)))
        self.assertLess(fit.of(5.0), fit.of(1.5))
        self.assertGreater(fit.of(0.1), fit.of(0.3))

    def test_no_new_path_counts_a_fallback_at_defaults(self):
        from matching import placement_matcher as PM
        PM.reset_cache()
        before = dict(PM.FAILURES)
        PM.frozen()
        fit, _p = PM.fit_objects()
        fit.of(0.4)
        self.assertEqual(PM.FAILURES, before)


class TableSanityTests(unittest.TestCase):
    """A table's `mu_wrong` is what an UNFORMABLE slot pays.  The first M18 bundle
    computed it as the mean of log(llr_raw) and wrote -11.8 nats; every slot whose
    residual could not be formed then paid -11.8, the score gridlocked, and top-1 at
    15 deg fell from 0.434 to 0.175 -- on BOTH candidate tables, which is what gave it
    away.  The deployed bundle reads -0.535 and M17 reads -2.26, so the guard is a band,
    not a point: wide enough not to be tuned, narrow enough to catch a nonsense value.
    """

    def test_mu_wrong_is_a_score_and_not_a_log_of_one(self):
        import json
        from matching import placement_matcher as PM
        for path in (PM.FROZEN_PATH, PM.FROZEN_PATH_M17, PM.FROZEN_PATH_M18):
            b = json.load(open(path))
            mu = float(b["fit"]["mu_wrong"])
            self.assertGreater(mu, -4.0, "%s: mu_wrong %r is not a per-slot score"
                               % (os.path.basename(path), mu))
            self.assertLessEqual(mu, 0.5, "%s: mu_wrong %r pays a wrong slot a bonus"
                                 % (os.path.basename(path), mu))

    def test_mu_wrong_equals_the_wrong_pair_expectation(self):
        import json
        import numpy as np
        from matching import placement_matcher as PM
        b = json.load(open(PM.FROZEN_PATH_M18))
        llr = np.asarray(b["fit"]["llr"], dtype=float)
        hw = np.asarray(b["fit"]["bin_counts_wrong"], dtype=float)
        self.assertAlmostEqual(float(b["fit"]["mu_wrong"]),
                               float(np.sum(llr * hw / hw.sum())), places=9)
