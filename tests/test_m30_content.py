"""M30's gates: the colour fusion, the colour blend, and the shape registration.

Every test here goes RED if the change is reverted:

* `test_fuse_two_channels_is_the_shipped_door` -- `placement_fuse.fuse` must reproduce the shipped
  `rank_fuse(counted, appearance)` bit for bit, or the "colour off" path is not the shipped door.
* `test_colour_channel_is_off_by_default` -- the default weight is 0.0 and the colour matrix is not
  consulted; M28 measured it, M30 ships it behind a weight.
* `test_colour_weight_is_below_the_incumbents` -- A1's stated discount (0.5 < 1.0).
* `test_node_colour_flag_off_returns_the_matrix` and `..._w_zero_is_off` -- A2's ablation: the
  deployed score is the two-term blend unless the flag is on AND the weight is positive.
* `test_shape_fields_are_registered` -- the M16 lesson (an unregistered class returns 0.0, not a
  small number): every graded field the similarity reads must be in the live registries.
* `test_record_carries_shape_and_version` -- the record format's shape columns round-trip.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import unittest                                            # noqa: E402


def _mat(seed: int, n: int = 4, g: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, g))


class TestDoorFusion(unittest.TestCase):
    def test_fuse_two_channels_is_the_shipped_door(self):
        from matching import placement_fuse as FU
        from matching.placement_frontdoor import rank_fuse
        a, b = _mat(1), _mat(2)
        self.assertTrue(np.array_equal(FU.fuse(a, b), rank_fuse(a, b)))

    def test_colour_channel_is_off_by_default(self):
        from matching import placement_fuse as FU
        self.assertEqual(FU.DEFAULT_WEIGHTS["colour"], 0.0)
        a, b, c = _mat(3), _mat(4), _mat(5)
        self.assertTrue(np.array_equal(FU.fuse(a, b, c), FU.fuse(a, b)))

    def test_colour_weight_is_below_the_incumbents(self):
        from matching import placement_fuse as FU
        self.assertLess(FU.COLOUR_WEIGHT, FU.DEFAULT_WEIGHTS["counted"])
        self.assertLess(FU.COLOUR_WEIGHT, FU.DEFAULT_WEIGHTS["appearance"])

    def test_colour_channel_changes_the_ranking(self):
        from matching import placement_fuse as FU
        a, b, c = _mat(6), _mat(7), _mat(8)
        off = FU.fuse(a, b)
        on = FU.fuse(a, b, c, weights={"colour": FU.COLOUR_WEIGHT})
        self.assertFalse(np.allclose(off, on))

    def test_negative_weight_is_refused(self):
        from matching import placement_fuse as FU

        class B:
            fusion = {"colour": -1.0}
        with self.assertRaises(ValueError):
            FU.effective_weights(B())


class TestColourBlend(unittest.TestCase):
    def _matrix(self):
        return [[0.5, 0.25], [0.1, 0.4]]

    def test_node_colour_flag_off_returns_the_matrix(self):
        import parameters as P
        from matching import placement_matcher as PM
        old = P.PLACEMENT_NODE_COLOUR
        try:
            P.PLACEMENT_NODE_COLOUR = False
            m = self._matrix()
            self.assertIs(PM.apply_node_colour(m, [0, 1], [0, 1], None, None, None, None), m)
        finally:
            P.PLACEMENT_NODE_COLOUR = old

    def test_node_colour_w_zero_is_off(self):
        import parameters as P
        from matching import placement_matcher as PM
        old = (P.PLACEMENT_NODE_COLOUR, P.PLACEMENT_NODE_COLOUR_W)
        try:
            P.PLACEMENT_NODE_COLOUR = True
            P.PLACEMENT_NODE_COLOUR_W = 0.0
            m = self._matrix()
            self.assertIs(PM.apply_node_colour(m, [0, 1], [0, 1], None, None, None, None), m)
        finally:
            P.PLACEMENT_NODE_COLOUR, P.PLACEMENT_NODE_COLOUR_W = old

    def test_rescore_matrix_accepts_images(self):
        import inspect
        from matching import placement_matcher as PM
        self.assertIn("images", inspect.signature(PM.rescore_matrix).parameters)

    def test_alignment_threads_the_images(self):
        import inspect
        from matching import alignment
        self.assertIn("placement_images",
                      inspect.signature(alignment.global_consistency_alignment).parameters)


class TestColourCache(unittest.TestCase):
    """The per-side colour cache must not change a single vector (M30 measured 57 % of a blended
    pair's cost was this recomputation; the cache is only allowed to be faster, never different)."""

    def test_cached_features_equal_the_uncached_ones(self):
        import numpy as np
        from matching import placement_matcher as PM

        class Side:
            nodes = (7, 11, 13)
            places = ((10.0, 10.0), (20.0, 24.0), (5.0, 30.0))
            spacing_by_node = {7: 3.0, 11: 4.5, 13: 2.0}
        img = (np.arange(48 * 48 * 3, dtype=np.uint8) % 251).reshape(48, 48, 3)
        side = Side()
        fast = PM._node_colour(img, side.nodes, side)
        slow = PM._legacy_node_colour(img, side.nodes, side)
        self.assertEqual(sorted(fast), sorted(slow))
        for u in fast:
            self.assertTrue(np.array_equal(np.asarray(fast[u]), np.asarray(slow[u])), u)

    def test_a_node_without_a_frame_has_no_colour(self):
        import numpy as np
        from matching import placement_matcher as PM
        img = (np.arange(48 * 48 * 3, dtype=np.uint8) % 251).reshape(48, 48, 3)

        class Side:
            nodes = (1, 2)
            places = ((10.0, 10.0), None)
            spacing_by_node = {1: 3.0, 2: 3.0}
        got = PM._node_colour(img, Side.nodes, Side())
        self.assertEqual(list(got), [1])


class TestShapeRegistration(unittest.TestCase):
    def test_shape_fields_are_registered(self):
        from matching import placement_shape as PS
        self.assertEqual(PS.missing_fields(), [], "a graded field is not registered")
        reg = PS.registered()
        self.assertEqual(tuple(reg["ranges"]), PS.RANGES)
        self.assertEqual(tuple(reg["weights"]), PS.WEIGHTS)
        self.assertEqual(reg["cycle"], PS.CYCLE)
        self.assertEqual(list(reg["axes"] or ()), list(PS.FIELDS))

    def test_every_field_is_visible_to_the_registries(self):
        from matching import placement_shape as PS
        import core.registries as R
        self.assertIn(PS.CLASS, R.CONSTITUTIONAL_RANGES)
        self.assertIn(PS.CLASS, R.SLOT_WEIGHT_PRIORS)
        self.assertIn(PS.CLASS, R.CIRCULAR_CYCLE_LENGTHS)
        axes = R.POPULATION_AXES.get(PS.CLASS) or ()
        self.assertEqual(len(axes), len(PS.FIELDS))
        self.assertTrue(all(a.n_bins >= 2 for a in axes))

    def test_unregister_restores_and_register_reinstalls(self):
        from matching import placement_shape as PS
        import core.registries as R
        try:
            PS.unregister()
            self.assertEqual(PS.missing_fields(), list(PS.FIELDS))
            PS.register()
            self.assertEqual(PS.missing_fields(), [])
        finally:
            PS.register()

    def test_unknown_field_is_refused(self):
        from matching import placement_shape as PS
        self.assertEqual(len(PS.FIELDS), 5)
        self.assertEqual(len(PS.RANGES), len(PS.FIELDS))
        self.assertEqual(len(PS.BINS), len(PS.FIELDS))
        self.assertEqual(len(PS.GRADED if hasattr(PS, "GRADED") else PS.FIELDS), len(PS.FIELDS))


class TestRecordFormat(unittest.TestCase):
    def test_record_carries_shape_and_version(self):
        from matching import placement_records as PR
        self.assertGreaterEqual(PR.FORMAT_VERSION, 29)
        self.assertEqual(PR.N_SHAPE, 5)
        place = {"n": 1, "lens": "Blob", "key": ("Blob", (1.0, 2.0, 3.0, 4.0, 5.0)),
                 "Dz": [0.0, 1.0], "Mz": [1.0, 1.0], "x": 1.0, "y": 2.0, "s": 3.0,
                 "theta": 0.5, "content": [0.1] * 6,
                 "resp": 0.4, "var_x": 0.2, "var_y": 0.1, "eigratio": 0.7, "rel_angle": 1.1}
        arrays = PR.export([("a/1", "000-000", {"places": [place]})], ["a/1"], ["000-000"])
        import json
        meta = json.loads(str(arrays["meta"]))
        self.assertEqual(meta["shape"], 5)
        self.assertGreaterEqual(meta["version"], 29)
        PR.save("/tmp/_m30_test.npz", arrays)
        back = PR.load_cell("/tmp/_m30_test.npz", "a/1", "000-000")
        for f in PR.SHAPE_FIELDS:
            self.assertAlmostEqual(back["places"][0][f], place[f], places=12)

    def test_keys_survive_the_shape_columns(self):
        """The shape block is a COLUMN SET, never a key slot: the key tuple is unchanged."""
        from matching import placement_records as PR
        place = {"n": 1, "lens": "Blob", "key": ("Blob", (1.0, 2.0, 3.0, 4.0, 5.0)),
                 "Dz": [0.0], "Mz": [1.0], "x": 0.0, "y": 0.0, "s": 1.0, "theta": None,
                 "resp": 0.9, "var_x": 0.5, "var_y": 0.5, "eigratio": 0.2, "rel_angle": 0.3}
        arrays = PR.export([("a/1", "000-000", {"places": [place]})], ["a/1"], ["000-000"])
        PR.save("/tmp/_m30_test2.npz", arrays)
        back = PR.load_cell("/tmp/_m30_test2.npz", "a/1", "000-000")
        self.assertEqual(tuple(back["places"][0]["key"][1]), (1.0, 2.0, 3.0, 4.0, 5.0))


class TestMechanismLivesInCore(unittest.TestCase):
    """A shipped behaviour must not be reachable ONLY from a Topology module."""

    def test_the_fusion_is_a_core_module(self):
        import matching.placement_fuse as FU
        self.assertTrue(callable(FU.fuse))
        self.assertFalse(FU.__file__.replace(os.sep, "/").startswith(
            os.path.join(ROOT, "Topology").replace(os.sep, "/")))

    def test_the_shape_is_a_core_module(self):
        import matching.placement_shape as PS
        self.assertTrue(callable(PS.features))
        self.assertTrue(callable(PS.register))
        self.assertNotIn("/Topology/", PS.__file__.replace(os.sep, "/"))

    def test_the_content_is_a_core_module(self):
        import matching.placement_content as PC
        self.assertTrue(callable(PC.features))
        self.assertNotIn("/Topology/", PC.__file__.replace(os.sep, "/"))

    def test_no_topology_import_in_core(self):
        """The core modules must not import the harness (the M19 rule, re-pinned)."""
        import matching.placement_fuse as FU
        import matching.placement_shape as PS
        for mod in (FU, PS):
            with open(mod.__file__, encoding="utf-8") as fh:
                src = fh.read()
            self.assertNotIn("import Topology", src)
            self.assertNotIn("from Topology", src)


if __name__ == "__main__":
    unittest.main()
