"""Mission 15 item 9: a number may not be written without the vector that produced it."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from matching import m15_manifest as M


class ManifestTests(unittest.TestCase):
    def test_manifest_has_every_required_field(self):
        m = M.manifest()
        for key in ("flags", "frame", "weight", "endpoints", "code"):
            self.assertIn(key, m)
        self.assertTrue(m["flags"], "the effective flag vector must not be empty")
        self.assertIn("PLACEMENT_FRAME_KEYPOINT", m["flags"])
        self.assertIn("FE_KEYPOINTS", m["flags"])
        self.assertTrue(m["frame"]["fallback"])
        self.assertIn("value", m["weight"])
        self.assertTrue(m["endpoints"]["lenses"])
        self.assertRegex(m["code"], r"^[0-9a-f]{12}$")

    def test_check_refuses_a_bare_number(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.json")
            with self.assertRaises(ValueError):
                M.check({"auc": 0.5}, p)
            self.assertFalse(os.path.exists(p))
            with self.assertRaises(ValueError):
                M.check({"manifest": {"flags": {}}}, p)

    def test_write_result_attaches_the_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.json")
            M.write_result(p, {"auc": 0.5})
            d = json.load(open(p))
            self.assertEqual(d["auc"], 0.5)
            self.assertIn("flags", d["manifest"])

    def test_digest_is_stable_within_a_run(self):
        self.assertEqual(M.code_digest(), M.code_digest())

    def test_every_m15_result_file_on_disk_carries_a_manifest(self):
        """Item 9 on the files themselves, not only on the API.

        A result file that reaches `Topology/results/placement/` without a manifest is
        a number nobody can attribute; this fails if one does.
        """
        import glob
        # [MISSION 24] the M15 result files moved to archive/ with their scripts; the pin follows
        # them, because "a number nobody can attribute" is a property of the file, not of the
        # directory it sits in.  Both places are scanned, so a new one cannot slip in either.
        paths = sorted(glob.glob(os.path.join(
            ROOT, "Topology", "results", "placement", "m15_*.json")) +
            glob.glob(os.path.join(
                ROOT, "archive", "Topology", "results", "placement", "m15_*.json")))
        self.assertTrue(paths, "no M15 result file found to check")
        for path in paths:
            payload = json.load(open(path))
            M.check(payload, path)


if __name__ == "__main__":
    unittest.main()
