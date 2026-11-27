"""M27: the pair cache must survive a different checkout prefix, and the record export must load.

Item 1a's requirement is a test that FAILS if a path prefix change makes a cached pair a miss --
that is exactly the defect that made M26 buy 209 pairs it already owned.  Item 1b's requirement is
that the committed npz export reproduces the door, which is checked by `m27_export_records --check`
(80/80 shortlists) and pinned here in its cheap form: the exported cell and the live cell carry the
same places, keys and configuration rows.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC          # noqa: E402
from matching import placement_recognise as RC   # noqa: E402


class TestPairCachePrefixInvariance(unittest.TestCase):
    def test_committed_pairs_are_found_by_basename(self) -> None:
        cache = RC.load_pair_cache()
        self.assertGreater(len(cache), 500, "the committed pair files did not load")
        self.assertIn("m23_read_pairs.jsonl", RC.ADOPTED)

    def test_a_different_prefix_is_still_a_hit(self) -> None:
        q = os.path.join(ROOT, "data", "ETH-80-master", "1", "1", "apple1-035-045.png")
        r = os.path.join(ROOT, "data", "ETH-80-master", "1", "1", "apple1-000-000.png")
        w = 0.375
        cache = RC.load_pair_cache()
        key_here = RC._pair_key(q, r, w)
        self.assertIn(key_here, cache)
        other = "/somewhere/else/entirely/data/ETH-80-master/1/1/apple1-035-045.png"
        other_ref = "/somewhere/else/entirely/data/ETH-80-master/1/1/apple1-000-000.png"
        self.assertEqual(RC._pair_key(other, other_ref, w), key_here)

    def test_a_temp_legacy_file_with_a_foreign_prefix_is_adopted(self) -> None:
        """The failure this pins: a committed file whose paths point at another checkout."""
        import matching.placement_recognise as M
        tmpdir = tempfile.mkdtemp()
        name = "m22_assemble_dev_pairs.jsonl"          # one of the adopted names
        # names no real cache row can collide with: the pair itself must be the adopted one
        rows = [{"q": "X:/old/place/ETH-80-master/9/9/zzznine9-035-045.png",
                 "r": "X:/old/place/ETH-80-master/9/9/zzznine9-000-000.png", "B": 0.123}]
        path = os.path.join(tmpdir, name)
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        old_dir = M.LEGACY_PAIR_DIR
        try:
            M.LEGACY_PAIR_DIR = tmpdir
            cache = RC.load_pair_cache(legacy=True)
        finally:
            M.LEGACY_PAIR_DIR = old_dir
        key = RC._pair_key("/home/anywhere/data/ETH-80-master/9/9/zzznine9-035-045.png",
                           "elsewhere/zzznine9-000-000.png", 0.375)
        self.assertIn(key, cache, "a foreign path prefix turned a cached pair into a miss")
        self.assertAlmostEqual(cache[key], 0.123)

    def test_the_digest_guard_still_holds_for_the_own_cache(self) -> None:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "pairs.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps({"d": "stale", "q": "a.png", "r": "b.png", "w": 0.375,
                                 "B": 0.5}) + "\n")
            fh.write(json.dumps({"d": RC.matcher_digest(), "q": "a.png", "r": "b.png",
                                 "w": 0.375, "B": 0.7}) + "\n")
        cache = RC.load_pair_cache(path, legacy=False)
        self.assertEqual(len(cache), 1)
        self.assertAlmostEqual(list(cache.values())[0], 0.7)

    def test_written_rows_are_basename_keyed(self) -> None:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "pairs.jsonl")
        RC.append_pair_cache([{"q": "/a/b/c.png", "r": "/d/e/f.png", "w": 0.375, "B": 0.25}],
                             path=path)
        row = json.loads(open(path).read().strip())
        self.assertEqual(row["q"], "c.png")
        self.assertEqual(row["r"], "f.png")


class TestRecordExport(unittest.TestCase):
    def test_export_round_trip_carries_every_field_the_door_reads(self) -> None:
        from matching import placement_records as PR
        if not REC.npz_available():
            self.skipTest("data/placement_records_m27.npz not built")
        keys = [o["key"] for o in REC.objects()]
        view = "000-000"
        with np.load(REC.NPZ, allow_pickle=True) as z:
            cell = PR._cell_from(z, keys[0], view)
        live_p = REC.cache_path(keys[0], view)
        if not os.path.exists(live_p):
            self.assertGreater(len(cell["places"]), 100)
            return
        live = REC.load(keys[0], view)
        self.assertEqual(len(cell["places"]), len(live["places"]))
        a, b = cell["places"][0], live["places"][0]
        self.assertEqual(a["lens"], b["lens"])
        self.assertEqual(a["n"], b["n"])
        np.testing.assert_allclose(a["key"][1], b["key"][1], atol=1e-6)
        np.testing.assert_allclose(a["Dz"], b["Dz"], atol=1e-5)
        np.testing.assert_allclose(a["Mz"], b["Mz"], atol=1e-6)
        self.assertEqual(cell["dropped"], live["dropped"])

    def test_export_size_is_committable(self) -> None:
        if not REC.npz_available():
            self.skipTest("no export")
        mb = os.path.getsize(REC.NPZ) / 1e6
        self.assertLess(mb, 40.0, "the export is %.1f MB; the commit budget is 40 MB" % mb)


if __name__ == "__main__":
    unittest.main()
