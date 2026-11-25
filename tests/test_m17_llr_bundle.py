"""MISSION 17 / O1 -- a table that is defensible, as data, without moving a default.

Two bundles exist.  OFF (the default) the deployed one is read, byte-identically to
Mission 16.  ON (`PLACEMENT_LLR_TABLE_M17`) the refitted one is read: pooled places-mode
residuals, empty-bin rule and monotone projection on, 36 bins of 0.05 tolunits over
[0, 1.8), WITH its per-bin counts.

The defect this pins: the deployed table pays **+0.597 nats at 5.0 tolunits and -2.237
at 1.5** -- a candidate five tolerances away is scored 2.83 nats ABOVE one that is one
and a half away.  It is the Laplace prior of an empty bin read as support, it has 5
rises and 16 of 24 bins positive, and it stores no counts, so it cannot even be
projected honestly.
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
from matching import placement_matcher as PM

OLD = os.path.join(ROOT, "matching", "placement_frozen_params.json")
NEW = os.path.join(ROOT, "matching", "placement_frozen_params_m17.json")


class DeployedTableIsTheDefectTests(unittest.TestCase):
    def setUp(self):
        self.b = json.load(open(OLD))
        self.fit = L._fit_from_dict(self.b["fit"])

    def test_it_rewards_distance(self):
        self.assertGreater(self.fit.of(5.0), 0.0,
                           "the deployed table pays positive credit in the tail")
        self.assertLess(self.fit.of(1.5), 0.0)
        self.assertGreater(self.fit.of(5.0), self.fit.of(1.5),
                           "pinned defect: a farther placement outscores a nearer one")

    def test_it_has_no_weight_profile(self):
        f = L.audit_bundle(self.b)
        self.assertTrue(any("weight profile" in x for x in f["unshippable"]))
        self.assertGreaterEqual(f["n_rises"], 5)
        self.assertGreaterEqual(f["positive_bins"], 16)


class RefittedTableTests(unittest.TestCase):
    def setUp(self):
        self.b = json.load(open(NEW))
        self.fit = L._fit_from_dict(self.b["fit"])
        self.audit = L.audit_bundle(self.b)

    def test_it_is_clean(self):
        self.assertEqual(self.audit["incoherent"], [])
        self.assertEqual(self.audit["unshippable"], [])
        self.assertEqual(self.audit["n_rises"], 0)
        self.assertEqual(self.audit["positive_bins"], 2)

    def test_it_carries_its_weight_profile(self):
        ct = self.b["fit"]["bin_counts_true"]
        cw = self.b["fit"]["bin_counts_wrong"]
        self.assertEqual(len(ct), 36)
        self.assertEqual(len(cw), 36)
        self.assertEqual(sum(ct), self.b["fit"]["n_true"])
        self.assertEqual(sum(cw), self.b["fit"]["n_wrong"])

    def test_no_thin_bin_carries_positive_credit(self):
        """A bin below MIN_BIN_COUNT is "never measured" for this purpose.

        The refitted table has NO fully empty bin -- the finer support means every bin
        has data, which is one of the reasons it is defensible -- so the check is the
        stronger one: no bin thinner than the fitted minimum may carry positive credit.
        """
        ct = np.array(self.b["fit"]["bin_counts_true"])
        cw = np.array(self.b["fit"]["bin_counts_wrong"])
        llr = np.array(self.b["fit"]["llr"])
        thin = (ct + cw) < int(self.b["constants"]["min_bin_count"])
        self.assertTrue(thin.any(), "the tail must actually be thin for this to bite")
        self.assertTrue((llr[thin] <= 0.0).all())
        empty = (ct + cw) == 0
        if empty.any():
            self.assertTrue((llr[empty] <= 0.0).all())

    def test_it_does_not_reward_distance(self):
        self.assertLessEqual(self.fit.of(5.0), self.fit.of(1.5))
        self.assertLessEqual(self.fit.of(1.0), self.fit.of(0.05))

    def test_strict_guard_accepts_it_and_refuses_the_deployed_one(self):
        old = P.PLACEMENT_FREEZE_GUARD_SUPPORT
        try:
            P.PLACEMENT_FREEZE_GUARD_SUPPORT = True
            L.GUARD_RECORD.clear()
            L.from_jsonbundle(self.b)
            self.assertEqual(L.GUARD_RECORD, {})
            with self.assertRaises(ValueError):
                L.from_jsonbundle(json.load(open(OLD)))
        finally:
            P.PLACEMENT_FREEZE_GUARD_SUPPORT = old
            L.GUARD_RECORD.clear()

    def test_a_dropped_weight_profile_is_caught(self):
        b = json.loads(json.dumps(self.b))
        b["fit"].pop("bin_counts_true")
        b["fit"].pop("bin_counts_wrong")
        f = L.audit_bundle(b)
        self.assertTrue(any("weight profile" in x for x in f["unshippable"]))


class SelectorTests(unittest.TestCase):
    def tearDown(self):
        P.PLACEMENT_LLR_TABLE_M17 = False
        PM.reset_cache()
        L.GUARD_RECORD.clear()

    def test_off_reads_the_deployed_table(self):
        P.PLACEMENT_LLR_TABLE_M17 = False
        PM.reset_cache()
        self.assertTrue(PM.active_path().endswith("placement_frozen_params.json"))
        fit, _p = PM.fit_objects()
        self.assertEqual(len(fit.llr), 24)
        self.assertEqual(float(fit.edges[-1]), 6.0)

    def test_on_reads_the_refitted_table(self):
        P.PLACEMENT_LLR_TABLE_M17 = True
        PM.reset_cache()
        self.assertTrue(PM.active_path().endswith("placement_frozen_params_m17.json"))
        fit, _p = PM.fit_objects()
        self.assertEqual(len(fit.llr), 36)
        self.assertAlmostEqual(float(fit.edges[-1]), 1.8)

    def test_the_two_tables_disagree_where_it_matters(self):
        P.PLACEMENT_LLR_TABLE_M17 = False
        PM.reset_cache()
        old, _p = PM.fit_objects()
        P.PLACEMENT_LLR_TABLE_M17 = True
        PM.reset_cache()
        new, _p2 = PM.fit_objects()
        # a far candidate: the deployed table REWARDS it, the refit punishes it
        self.assertGreater(old.of(5.0), new.of(5.0))
        self.assertGreater(old.of(5.0), 0.0)
        self.assertLess(new.of(5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
