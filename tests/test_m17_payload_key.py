"""MISSION 17 / O5 -- a graph that carries no payload must not serve a payload arm.

The population vector rides in the signature payload and NOT in the graph cache key's
node content.  Mission 16's equivalence build measured the consequence by accident:
identical node keys and digest, different AUC, because the graph built with
`KP_GRADED_POPULATION` off has no payload to read.

Both directions are pinned here:
  * payload PRESENT  -> the population branch runs and the reading is the graded one;
  * payload ABSENT   -> the read is COUNTED (`POPULATION_PAYLOAD_MISSES`), and with
    `KP_POPULATION_PAYLOAD_STRICT` it RAISES instead of quietly scoring on the kernel.
That is the "refuses to serve" half of the acceptance; the other half is the digest,
which already lists `KP_GRADED_POPULATION` (`test_m14_switch_digest`).
"""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P
from core import keypoint_classes as KPC
from core import types as T
from lenses.keypoints import Keypoint, signature_of


def _kp(deg: float) -> Keypoint:
    return Keypoint(kind="blob", x=1.0, y=1.0, scale=1.6, theta=math.radians(deg),
                    coherence=0.8, resp=1.0, var_x=1.0, var_y=1.0, dim=2)


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(P, k) for k in
                       ("KP_GRADED_SIMILARITY", "KP_GRADED_POPULATION",
                        "KP_POPULATION_PAYLOAD_STRICT", "POPULATION_CODING_ENABLED")}
        P.KP_GRADED_SIMILARITY = True
        P.KP_GRADED_POPULATION = True
        P.KP_POPULATION_PAYLOAD_STRICT = False
        P.POPULATION_CODING_ENABLED = True
        KPC.unregister()
        KPC.register("wide")
        KPC.register_population()
        T.POPULATION_PAYLOAD_MISSES.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(P, k, v)
        KPC.unregister()
        T.POPULATION_PAYLOAD_MISSES.clear()

    def test_present_payload_serves_the_arm(self):
        a = signature_of(_kp(20.0))
        b = signature_of(_kp(25.0))          # same 15-degree bin, sub-bin apart
        self.assertTrue(a.population and b.population)
        s = a.similarity(b)
        self.assertEqual(T.POPULATION_PAYLOAD_MISSES, {})
        self.assertLess(s, 0.9995,
                        "the population branch must resolve below the bin, not return "
                        "the exact-equality 1.0")

    def test_absent_payload_is_counted(self):
        P.KP_GRADED_POPULATION = False        # what a graph built without the payload has
        a = signature_of(_kp(20.0))
        b = signature_of(_kp(25.0))
        P.KP_GRADED_POPULATION = True         # the reading side asks for the arm
        self.assertFalse(a.population)
        s = a.similarity(b)
        self.assertEqual(T.POPULATION_PAYLOAD_MISSES.get("Blob"), 1)
        self.assertEqual(s, 1.0,
                         "pinned defect: without the payload the pair reads as the "
                         "binary exact-equality 1.0 and calls itself the population arm")

    def test_strict_raises_instead_of_falling_back(self):
        P.KP_GRADED_POPULATION = False
        a = signature_of(_kp(20.0))
        b = signature_of(_kp(25.0))
        P.KP_GRADED_POPULATION = True
        P.KP_POPULATION_PAYLOAD_STRICT = True
        with self.assertRaises(T.PopulationPayloadMissing):
            a.similarity(b)

    def test_the_switch_is_declared_before_use(self):
        from eth80_frontend import _FRONTEND_SWITCHES
        for name in ("KP_POPULATION_PAYLOAD_STRICT", "KP_GRADED_POPULATION",
                     "PLACEMENT_LLR_TABLE_M17"):
            self.assertIn(name, _FRONTEND_SWITCHES)


if __name__ == "__main__":
    unittest.main()
