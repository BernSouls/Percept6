"""M16 quality bars 2, 3 and 5, pinned on the keypoint class constitution.

BAR 2 -- A TEST THAT FAILS IF THE FIX IS REVERTED.  Every test below is written
against the DEFECT, not against the fix: with `KP_GRADED_SIMILARITY` off the
assertions are the shipped (broken) behaviour, and with it on they are the
requirement.  Reverting the fix turns the second group red; `test_the_defect_...`
turns red if someone "fixes" it silently while the flag is off.

BAR 3 -- NO SILENT FALLBACK.  An unknown variant raises; the refusals are counted.

BAR 5 -- THE FREEZE GUARD covers res_max and n_bins.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import numpy as np

import parameters as P
from core import keypoint_classes as KPC
from core.registries import CONSTITUTIONAL_RANGES, CIRCULAR_CYCLE_LENGTHS
from core.types import TypedSignature
from lenses.keypoints import Keypoint, THETA_BINS, signature_of


def _kp(theta_deg: float, *, kind: str = "blob", scale: float = 1.6,
        coherence: float = 0.8, dim: int = 2) -> Keypoint:
    return Keypoint(kind=kind, x=10.0, y=10.0, scale=scale,
                    theta=math.radians(theta_deg), coherence=coherence,
                    resp=1.0, var_x=1.0, var_y=1.0, dim=dim)


def _sig(theta_deg: float, **kw):
    kpc_was = dict(CONSTITUTIONAL_RANGES)
    return signature_of(_kp(theta_deg, **kw))


class GradedKeypointTests(unittest.TestCase):
    def setUp(self):
        self._flag = P.KP_GRADED_SIMILARITY
        self._variant = P.KP_GRADED_VARIANT
        self._saved = {c: CONSTITUTIONAL_RANGES.get(c) for c in KPC.KEYPOINT_CLASSES}
        self._saved_cycle = {c: CIRCULAR_CYCLE_LENGTHS.get(c)
                             for c in KPC.KEYPOINT_CLASSES}
        # [M32] isolate: another test module (test_m15_manifest) imports the shipped path, which
        # registers the keypoint classes; without this the "off" arm reads a graded kernel and the
        # defect test measures the fix.  Pre-existing at M31's tip; the leak is the test's, not
        # the code's -- the suite must be order-independent.
        KPC.REGISTRATIONS.clear()
        for _c in KPC.KEYPOINT_CLASSES:
            CONSTITUTIONAL_RANGES.pop(_c, None)
            CIRCULAR_CYCLE_LENGTHS.pop(_c, None)

    def tearDown(self):
        P.KP_GRADED_SIMILARITY = self._flag
        P.KP_GRADED_VARIANT = self._variant
        for c, v in self._saved.items():
            if v is None:
                CONSTITUTIONAL_RANGES.pop(c, None)
            else:
                CONSTITUTIONAL_RANGES[c] = v
            cy = self._saved_cycle.get(c)
            if cy is None:
                CIRCULAR_CYCLE_LENGTHS.pop(c, None)
            else:
                CIRCULAR_CYCLE_LENGTHS[c] = cy

    # ------------------------------------------------------------------ bar 2
    #: A theta that sits 4 deg inside a bin edge: 5 deg of rotation then CROSSES a
    #: bin boundary.  The defect is a bin-edge defect, so the offset matters and is
    #: stated -- from 0 deg a 5 deg rotation stays inside bin 0 and nothing breaks
    #: (measured, and pinned below so the claim is not overstated).
    EDGE = 11.0

    def test_the_defect_is_that_one_bin_costs_everything(self):
        """Off: equality is 1.0 and ONE bin is 0.0 -- the shipped defect."""
        P.KP_GRADED_SIMILARITY = False
        KPC.REGISTRATIONS.clear()
        same = _sig(0.0)
        self.assertAlmostEqual(same.similarity(_sig(0.0)), 1.0, places=9)
        one_bin = _sig(360.0 / THETA_BINS)          # exactly one theta bin away
        self.assertEqual(same.similarity(one_bin), 0.0,
                         "the defect: a 15 deg rotation zeroes the pair")
        self.assertEqual(_sig(self.EDGE).similarity(_sig(self.EDGE + 5.0)), 0.0,
                         "5 deg across a bin edge zeroes the pair too")
        # HONESTY: inside one bin nothing breaks, which is why the defect looks
        # intermittent -- it is an EDGE effect, not a rotation effect.
        self.assertAlmostEqual(same.similarity(_sig(5.0)), 1.0, places=9)

    def test_fix_graded_similarity_survives_a_rotation(self):
        P.KP_GRADED_SIMILARITY = True
        P.KP_GRADED_VARIANT = "narrow"
        KPC.REGISTRATIONS.clear()
        same = _sig(self.EDGE)
        for deg, floor in ((5.0, 0.5), (15.0, 0.15)):
            s = same.similarity(_sig(self.EDGE + deg))
            self.assertGreater(s, floor,
                               "a %g deg rotation must stay matchable, got %r"
                               % (deg, s))
            self.assertLess(s, 1.0, "a rotated pair must not score as identical")
        # and it must DEGRADE, not saturate.  The kernel sees BIN distance, so the
        # comparison steps a whole bin: 5 deg (1 bin from the edge) vs 25 deg
        # (2 bins).  Sub-bin resolution is the POPULATION arm's job, measured in
        # Topology/placement_m16_similarity.py, not claimed here.
        self.assertLess(same.similarity(_sig(self.EDGE + 25.0)),
                        same.similarity(_sig(self.EDGE + 5.0)))

    def test_fix_separates_an_impostor(self):
        P.KP_GRADED_SIMILARITY = True
        P.KP_GRADED_VARIANT = "narrow"
        KPC.REGISTRATIONS.clear()
        same = _sig(0.0)
        near = same.similarity(_sig(15.0))
        far = same.similarity(_sig(90.0))
        kind = same.similarity(_sig(0.0, kind="corner"))
        self.assertGreater(near, far, "15 deg must beat 90 deg")
        self.assertLess(kind, 0.01, "a different KIND is not the same place")

    def test_wide_variant_is_wider_than_narrow(self):
        P.KP_GRADED_SIMILARITY = True
        KPC.REGISTRATIONS.clear()
        P.KP_GRADED_VARIANT = "narrow"
        a = _sig(0.0)
        narrow = a.similarity(_sig(15.0))
        KPC.REGISTRATIONS.clear()
        P.KP_GRADED_VARIANT = "wide"
        b = _sig(0.0)
        wide = b.similarity(_sig(15.0))
        self.assertGreater(wide, narrow)

    # ------------------------------------------------------------------ bar 3
    def test_an_unknown_variant_raises_and_is_counted(self):
        KPC.REFUSALS.clear()
        with self.assertRaises(ValueError):
            KPC.register("no-such-variant")
        self.assertIn("register:no-such-variant", KPC.REFUSALS)
        self.assertEqual(KPC.register("no-such-variant", strict=False), {})

    def test_registration_is_declared_before_use(self):
        """Bar 9: the switches are in the front-end digest, not invented later."""
        from eth80_frontend import _FRONTEND_SWITCHES
        self.assertIn("KP_GRADED_SIMILARITY", _FRONTEND_SWITCHES)
        self.assertIn("KP_GRADED_VARIANT", _FRONTEND_SWITCHES)

    def test_the_SHIPPED_DEFAULTS_are_the_arm_the_report_measured(self):
        """The report calls one arm "shipped"; this pins which one."""
        import parameters as P
        self.assertFalse(P.KP_GRADED_SIMILARITY, "the mission turns nothing on")
        self.assertFalse(P.KP_GRADED_POPULATION,
                         "population is measured and NOT shipped: it wins the "
                         "detection proxy and loses geometry-AUC and top-1 at 15 deg")
        self.assertEqual(P.KP_GRADED_VARIANT, "wide")
        self.assertEqual(P.KP_GRADED_POP_VARIANT, "theta2-balanced")
        self.assertFalse(P.PLACEMENT_LLR_MONOTONE)
        self.assertFalse(P.PLACEMENT_FREEZE_GUARD_SUPPORT)
        for name in ("KP_GRADED_POPULATION", "KP_GRADED_POP_VARIANT",
                     "PLACEMENT_LLR_MONOTONE", "PLACEMENT_FREEZE_GUARD_SUPPORT"):
            from eth80_frontend import _FRONTEND_SWITCHES
            self.assertIn(name, _FRONTEND_SWITCHES)

    def test_registration_reports_what_it_installed(self):
        KPC.REGISTRATIONS.clear()
        got = KPC.register("narrow")
        self.assertEqual(sorted(got["classes"]), sorted(KPC.KEYPOINT_CLASSES))
        self.assertEqual(CONSTITUTIONAL_RANGES["Blob"], tuple(got["ranges"]))
        self.assertEqual(CIRCULAR_CYCLE_LENGTHS["Blob"], {0: 12})


if __name__ == "__main__":
    unittest.main()
