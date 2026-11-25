"""MISSION 20 / item 1 -- the M19 design as one bundle, and a manifest that cannot lie.

M19 shipped the descriptor harness-side only, and the result manifests recorded PROCESS
DEFAULTS: a row could carry `desc: (24, 0.25)` while the manifest said "descriptor off".
Two things are pinned here:

* the bundle is OFF by default and, when on, derives every switch it claims (so there is
  one source of truth, not a second one written by hand);
* `m15_manifest.effective_flags()` reports what the matcher will actually use -- checked in
  a SUBPROCESS, because the flags are derived at import and a test that pokes the module
  afterwards would be testing its own poking.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P


class BundleTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(P.PLACEMENT_M19)
        from eth80_frontend import _FRONTEND_SWITCHES
        for name in ("PLACEMENT_M19", "PLACEMENT_M19_W", "PLACEMENT_M19_K"):
            self.assertIn(name, _FRONTEND_SWITCHES)

    def test_the_bundle_derives_and_the_manifest_agrees(self):
        code = """
import sys
sys.path[:0] = [%r]
import parameters as P
from matching import placement_matcher as PM
from matching.m15_manifest import effective_flags
f = effective_flags()
# what the DEPLOYED call site records, on a two-cell matrix
PM.reset_cache()
stats = {}
PM.rescore_matrix([[0.5, 0.0], [0.0, 0.0]], [0, 1], [0, 1],
                  type("S", (), {"incident": {}, "index": {}})(),
                  type("S", (), {"incident": {}, "index": {}})(), stats=stats)
e = stats["effective"]
print(int(P.PLACEMENT_M19), int(f["PLACEMENT_NODE_DESC"]),
      f["PLACEMENT_NODE_DESC_W"], f["PLACEMENT_NODE_DESC_K"],
      int(f["PLACEMENT_LLR_TABLE_M18"]), int(f["PLACEMENT_CONTEXT_ASSIGN"]),
      e["llr_table"], e["cand_cap_resolved"], e["PLACEMENT_NODE_DESC_W"],
      e["PLACEMENT_M19"])
""" % ROOT
        env = dict(os.environ, PLACEMENT_M19="1")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, timeout=180)
        self.assertEqual(out.returncode, 0, out.stderr[-1200:])
        got = out.stdout.split()
        self.assertEqual(got[0], "1", "the child did not see the bundle")
        self.assertEqual(got[1], "1",
                         "the manifest says descriptor OFF while the bundle is ON")
        self.assertEqual(float(got[2]), 0.25, "the bundle must default w to 0.25")
        self.assertEqual(int(got[3]), 24)
        self.assertEqual(got[4], "1", "the bundle must select the M18 table")
        self.assertEqual(got[5], "1", "the bundle must select the assignment start")
        # the resolver outputs, recorded at the call site
        self.assertEqual(got[6], "placement_frozen_params_m18.json")
        self.assertEqual(int(got[7]), 24, "the bundle must set the candidate cap to 24")
        # and the call site's own view of the switches agrees with the manifest
        self.assertEqual(float(got[8]), 0.25)
        self.assertEqual(got[9].lower(), "true",
                         "the call site's own view must show the bundle on")

    def test_defaults_report_the_deployed_table_and_no_descriptor(self):
        from matching.m15_manifest import effective_flags
        f = effective_flags()
        self.assertFalse(f["PLACEMENT_NODE_DESC"])
        self.assertFalse(f["PLACEMENT_LLR_TABLE_M18"])
        from matching import placement_matcher as PM
        PM.reset_cache()
        self.assertEqual(PM.active_path(), PM.FROZEN_PATH)


if __name__ == "__main__":
    unittest.main()
