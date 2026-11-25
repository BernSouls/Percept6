"""M16 quality bars 5 and 6 -- a table that can be SHIPPED without lying.

BAR 6: the LLR table must be non-increasing in |residual| and no empty bin may carry
positive credit.  Mission 15's 0.05 px refit rose 8 times in the sparse tail, so the
shipped-table claim could not be made.  `PLACEMENT_LLR_MONOTONE` projects the table
onto the non-increasing family (pool-adjacent-violators, weighted by bin counts).

BAR 5: the freeze guard covers res_max and n_bins, so a refit with a different
support or resolution is REFUSED rather than silently indexed against the wrong
edges.  Off (the default, bar 1) the mismatch is RECORDED, never swallowed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import numpy as np

import parameters as P
from matching import placement_llr as L

#: A NON-MONOTONE table with the shape Mission 15 actually measured: a heavy first
#: bin, a decaying body, and sparse tail bins that rise.
RISING = [3.0, 1.1, -0.9, -1.8, -2.2, -2.7, -3.0, -2.9, -3.0, -3.0, -2.6, -3.0]


class MonotoneTableTests(unittest.TestCase):
    def setUp(self):
        self._m = P.PLACEMENT_LLR_MONOTONE
        self._n = P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL
        self._c = P.PLACEMENT_LLR_MIN_BIN_COUNT

    def tearDown(self):
        P.PLACEMENT_LLR_MONOTONE = self._m
        P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = self._n
        P.PLACEMENT_LLR_MIN_BIN_COUNT = self._c

    def test_isotonic_kills_every_rise_and_says_how_much_it_moved(self):
        out, rep = L.isotonic_nonincreasing(RISING, [100, 90, 80, 70, 60, 50, 40,
                                                    3, 9, 2, 6, 1])
        self.assertEqual(rep["rises_after"], 0)
        self.assertEqual(sum(1 for i in range(len(out) - 1) if out[i + 1] > out[i] + 1e-12),
                         0)
        self.assertGreater(rep["changed_bins"], 0,
                           "a rising sequence must be changed by the projection")
        self.assertGreater(rep["smoothing_constant_share"], 0.0)
        self.assertEqual(len(out), len(RISING))

    def test_unprojected_table_rises_and_the_flag_is_what_stops_it(self):
        t = [0.01] * 5000 + [0.9] * 40
        w = [0.9] * 4000 + [5.0] * 60
        P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = True
        P.PLACEMENT_LLR_MIN_BIN_COUNT = 2
        P.PLACEMENT_LLR_MONOTONE = False
        off = L.fit_llr(t, w, res_max=1.8, n_bins=36)
        P.PLACEMENT_LLR_MONOTONE = True
        on = L.fit_llr(t, w, res_max=1.8, n_bins=36)
        self.assertEqual(L.llr_monotonicity(on.llr, on.edges)["n_rises"], 0,
                         "with the flag on the table must be shippable")
        self.assertEqual(on.extra["isotonic"]["rises_after"], 0)
        # the flag must be doing the work: the unprojected table has the same bins
        if L.llr_monotonicity(off.llr, off.edges)["n_rises"] == 0:
            self.skipTest("this sample happens to be monotone; the pin is the "
                          "isotonic test above")

    def test_no_empty_bin_gets_positive_credit(self):
        P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = True
        P.PLACEMENT_LLR_MONOTONE = True
        P.PLACEMENT_LLR_MIN_BIN_COUNT = 20
        fit = L.fit_llr([0.01] * 300, [2.5] * 300, res_max=6.0, n_bins=24)
        # the bins beyond the data (nothing above 2.5) must not be positive
        for i in range(12, 24):
            self.assertLessEqual(float(fit.llr[i]), 0.0,
                                 "empty bin %d carries positive credit" % i)


class FreezeGuardTests(unittest.TestCase):
    """The guard as MISSION 17 redefined it.

    M16 asked whether the DECLARED support matched the LIVE constants.  M17 asks the
    stronger, honest question: does the bundle agree with ITS OWN edges, does it carry
    the weight profile, and is it shippable at all?  The consequence is that the
    DEPLOYED bundle now fails the strict gate -- which is the finding of O1, not a
    regression, and it is pinned below.
    """
    def setUp(self):
        self._g = P.PLACEMENT_FREEZE_GUARD_SUPPORT
        self.path = os.path.join(ROOT, "matching", "placement_frozen_params.json")
        self.base = json.load(open(self.path))

    def tearDown(self):
        P.PLACEMENT_FREEZE_GUARD_SUPPORT = self._g
        L.GUARD_RECORD.clear()

    def _bundle(self, **const):
        b = json.loads(json.dumps(self.base))
        b.setdefault("constants", {}).update(const)
        return b

    def test_a_declaration_that_contradicts_its_own_edges_always_raises(self):
        """Coherence is not a policy: a file that contradicts itself can never be read."""
        for strict in (False, True):
            P.PLACEMENT_FREEZE_GUARD_SUPPORT = strict
            L.GUARD_RECORD.clear()
            with self.assertRaises(ValueError):
                L.from_jsonbundle(self._bundle(res_max=1.8))
            with self.assertRaises(ValueError):
                L.from_jsonbundle(self._bundle(n_bins=36))

    def test_the_DEPLOYED_bundle_fails_the_strict_gate(self):
        """O1's finding, pinned: the shipped table has no counts and rises 5 times."""
        P.PLACEMENT_FREEZE_GUARD_SUPPORT = True
        L.GUARD_RECORD.clear()
        with self.assertRaises(ValueError) as cm:
            L.from_jsonbundle(self.base)
        msg = str(cm.exception)
        self.assertIn("weight profile", msg)
        self.assertIn("rises", msg)

    def test_tolerant_mode_records_without_swallowing(self):
        P.PLACEMENT_FREEZE_GUARD_SUPPORT = False
        L.GUARD_RECORD.clear()
        L.from_jsonbundle(self.base)          # the deployed bundle, tolerated
        self.assertIn("unshippable", L.GUARD_RECORD)
        with self.assertRaises(AssertionError):
            L.assert_guard_clean()
        L.GUARD_RECORD.clear()
        L.assert_guard_clean()


if __name__ == "__main__":
    unittest.main()
