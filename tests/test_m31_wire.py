"""M31's gates: the product ships the fused door, and only the core can reach it.

Every test here goes RED if the change is reverted:

* `test_bundle_ships_the_fused_door` -- door.kind "fused3", colour weight 0.5, rrf_k 60,
  shortlist 8, content "colour_rings", matcher blend {w_cfg 0.25, w_col 0.10}.
* `test_fused3_requires_colour_content` / `test_gallery_refuses_missing_content` /
  `test_recognise_refuses_a_keypoint_gallery` -- the three refusals (no silent 0.0 anywhere).
* `test_recognise_reproduces_the_m30_door` -- the product's shortlist equals the M30 fused door's,
  and with the colour weight at 0 equals the two-channel shipped door's (item 1d's two equalities,
  on a records-only subset; `Topology/m31_wire.py --equalities` runs all 80 x 2).
* `test_composition_is_a_supplement` -- M31's composition fix: with the configuration term off,
  `apply_node_colour` returns `(1 - w_col) * matrix + w_col * colour`, i.e. the colour enters as a
  supplement and never rescales the incumbent (M30 read `PLACEMENT_NODE_DESC_W` = 0.5 as `w_cfg`
  and so computed 0.4 * sim + 0.1 * colour).
* `test_the_door_is_core` -- `recognise`, the fused door and the colour content are importable and
  usable without any Topology module.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import unittest                                            # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
_IMG = os.path.join(ROOT, "data", "ETH-80-master", "1", "1", "apple1-000-000.png")
REF = "000-000"
VIEWS = ("035-045", "045-180")


def _have_records() -> bool:
    return os.path.exists(RECORDS)


class TestBundle(unittest.TestCase):
    def test_bundle_ships_the_fused_door(self):
        # [M35] the shipped door is the object-level bag door; `fused3` stays implemented as the
        # legacy path (the fusion weights below are still the ones that path reads).
        from matching import placement_bundle as PB
        b = PB.PlacementBundle.load()
        self.assertEqual(b.door_kind, "bag")
        self.assertAlmostEqual(b.fusion_weight, 0.5, places=12)
        self.assertEqual(int(b.rrf_k), 60)
        self.assertEqual(int(b.shortlist), 8)
        self.assertEqual(b.content_kind, "colour_rings")
        self.assertAlmostEqual(float(b.matcher_blend["w_cfg"]), 0.25, places=12)
        self.assertAlmostEqual(float(b.matcher_blend["w_col"]), 0.10, places=12)

    def test_fused3_requires_colour_content(self):
        from matching import placement_bundle as PB
        from matching import placement_recognise as PR
        b = PB.PlacementBundle.load()
        bad = dataclasses.replace(b, content={"kind": "keypoint"})
        with self.assertRaises(PB.BundleRefusal):
            PR._validate(bad)

    def test_blend_must_be_convex(self):
        from matching import placement_bundle as PB
        from matching import placement_recognise as PR
        b = PB.PlacementBundle.load()
        bad = dataclasses.replace(b, matcher_blend={"w_cfg": 0.95, "w_col": 0.10})
        with self.assertRaises(PB.BundleRefusal):
            PR._validate(bad)

    def test_colour_weight_zero_is_the_two_channel_door(self):
        from matching import placement_bundle as PB
        from matching import placement_recognise as PR
        # [M35] same as above: the fused door is exercised explicitly, it is no longer what ships.
        b = dataclasses.replace(PB.PlacementBundle.load(), door_kind="fused3", door_params={})
        zero = dataclasses.replace(b, door_params={**dict(b.door_params), "colour_weight": 0.0})
        PR._validate(zero)                       # legal: 0 is the shipped two-channel door
        self.assertEqual(zero.fusion_weight, 0.0)


def _stripped_records(n: int = 2):
    """Real records with their colour content removed -- the mismatch the door must refuse."""
    from Topology import m26_records as REC
    from matching import placement_records as PR
    objs = REC.objects()[:n]
    cells = PR.load_cells(RECORDS, [REF])
    recs = []
    for o in objs:
        rec = cells[(o["key"], REF)]
        rec = {"places": [dict(p, content=None) for p in rec["places"]]}
        recs.append([rec])
    return objs, recs


@unittest.skipUnless(_have_records(), "the M30 records are not in this checkout")
class TestRefusals(unittest.TestCase):
    def test_gallery_refuses_missing_content(self):
        from matching import placement_bundle as PB
        from matching import placement_gallery as GL
        objs, recs = _stripped_records(2)
        items = [(o["key"], [(REF, GL._path_of(o, REF))]) for o in objs]
        with self.assertRaises(PB.BundleRefusal):
            GL._assemble(items, recs, "d", {"build": 0.0}, content_kind="colour_rings")
        g = GL._assemble(items, recs, "d", {"build": 0.0}, content_kind="keypoint")
        self.assertEqual(g.content_kind, "keypoint")

    def test_recognise_refuses_a_keypoint_gallery(self):
        from matching import placement_bundle as PB
        from matching import placement_gallery as GL
        from matching import placement_recognise as PR
        objs, recs = _stripped_records(2)
        items = [(o["key"], [(REF, GL._path_of(o, REF))]) for o in objs]
        g = GL._assemble(items, recs, "d", {"build": 0.0}, content_kind="keypoint")
        b = PB.PlacementBundle.load()
        with self.assertRaises(PB.BundleRefusal):
            PR.recognise(GL._path_of(objs[0], REF), g, bundle=b, matcher=False)

    def test_from_records_refuses_an_uncovered_cell(self):
        from matching import placement_bundle as PB
        from matching import placement_gallery as GL
        with self.assertRaises(PB.BundleRefusal):
            GL.from_records(RECORDS, [{"key": "not/an-object", "ref": _IMG}], REF)


class TestComposition(unittest.TestCase):
    def test_composition_is_a_supplement(self):
        import parameters as P
        from matching import placement_matcher as PM
        old = (P.PLACEMENT_NODE_COLOUR, P.PLACEMENT_NODE_COLOUR_W, P.PLACEMENT_NODE_DESC)
        try:
            P.PLACEMENT_NODE_COLOUR = True
            P.PLACEMENT_NODE_COLOUR_W = 0.10
            P.PLACEMENT_NODE_DESC = False        # no configuration term in the matrix
            m = [[0.5, 0.25], [0.1, 0.4]]
            cos = np.asarray([1.0, 0.0, 1.0, 0.0])

            class Side:
                nodes = (0, 1)
                places = ((10.0, 10.0), (20.0, 20.0))
                spacing_by_node = {0: 3.0, 1: 3.0}
            from matching import placement_matcher as PM2
            PM2._COLOUR_CACHE.clear()
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            orig = PM2._node_colour
            PM2._node_colour = lambda _img, nodes, side: {int(u): np.asarray([cos[u], 0.0, 0.0,
                                                                             0.0, 0.0, 0.0])
                                                         for u in nodes}
            try:
                out = PM.apply_node_colour(m, [0, 1], [0, 1], Side(), Side(), img, img)
            finally:
                PM2._node_colour = orig
            used = [(i, j) for i in range(2) for j in range(2) if m[i][j] > 0.0]
            for i, j in used:
                expect = (1.0 - 0.10) * m[i][j] + 0.10 * float(cos[i] * cos[j])
                self.assertAlmostEqual(float(out[i][j]), expect, places=12)
        finally:
            P.PLACEMENT_NODE_COLOUR, P.PLACEMENT_NODE_COLOUR_W, P.PLACEMENT_NODE_DESC = old


@unittest.skipUnless(_have_records(), "the M30 records are not in this checkout")
class TestProductDoor(unittest.TestCase):
    """Item 1d on a records-only subset (the full 80 x 2 is `m31_wire.py --equalities`)."""

    n_obj = 3

    @classmethod
    def setUpClass(cls):
        from Topology import m26_records as REC
        from matching import placement_gallery as GL
        from matching import placement_records as PR
        cls.objs = REC.objects()[:cls.n_obj]
        cls.gal = GL.from_records(RECORDS, REC.objects(), REF, content="colour_rings")
        cls.cells = PR.load_cells(RECORDS, [REF] + list(VIEWS))
        cls.path = {o["key"]: GL._path_of(o, "000-000") for o in REC.objects()}

    def _query(self, obj, view):
        from matching import placement_gallery as GL
        rec = self.cells[(obj["key"], view)]
        return rec, GL._path_of(obj, view)

    def test_recognise_reproduces_the_m30_door(self):
        from matching import placement_bundle as PB
        from matching import placement_frontdoor as FD
        from matching import placement_fuse as FU
        from matching import placement_recognise as PR2
        # [M35] the shipped bundle now names the OBJECT-LEVEL bag door.  This test's claim is about
        # the FUSED door's arithmetic, which is still implemented (legacy), so it runs that kind
        # explicitly: reverting `fuse` (or the colour channel it reads) still goes red here.
        b = dataclasses.replace(PB.PlacementBundle.load(), door_kind="fused3", door_params={})
        cells = 0
        for view in VIEWS:
            for o in self.objs:
                qrec, qpath = self._query(o, view)
                app = FD.appearance_feature(qpath)
                r = PR2.recognise(qpath, self.gal, bundle=b, matcher=False, record=qrec,
                                  appearance=app, s=8, use_pair_cache=False)
                cnt, _d = FD.count_scores(self.gal.index, [qrec], n_hits=int(b.door_n_hits))
                ref_app = np.asarray([app], dtype=np.float64) @ self.gal.appearance.T
                colour = np.asarray(FU.colour_channel(self.gal.flat_records(), [qrec]),
                                    dtype=np.float64)
                # [M34] the shipped door fuses a FIFTH channel; the hand-built arithmetic here has
                # to name it or this test would be checking a door the bundle no longer describes.
                frag = None
                if float(b.fragment_weight) != 0.0:
                    from matching import placement_anchors as PA
                    from matching import placement_fragment as PFM
                    frag = PFM.channel(
                        [{"places": r["places"]} for r in self.gal.flat_records()],
                        qrec["places"],
                        [PA.for_path(p) for p in self.gal.flat_paths()],
                        PA.for_path(qpath))
                fused = FU.fuse(cnt, ref_app, colour, fragment=frag,
                                weights={"colour": float(b.fusion_weight),
                                         "fragment": float(b.fragment_weight)}, k=int(b.rrf_k))
                order = sorted(range(self.gal.n), key=lambda j: (-fused[0, j], j))[:8]
                self.assertEqual(list(r.shortlist), order,
                                 "fused door mismatch on %s at %s" % (o["key"], view))
                cells += 1
        self.assertEqual(cells, 2 * self.n_obj)

    def test_colour_weight_zero_reproduces_the_shipped_door(self):
        from matching import placement_bundle as PB
        from matching import placement_frontdoor as FD
        from matching import placement_recognise as PR2
        # [M35] same as above: the fused door is exercised explicitly, it is no longer what ships.
        b = dataclasses.replace(PB.PlacementBundle.load(), door_kind="fused3", door_params={})
        zero = dataclasses.replace(b, door_params={**dict(b.door_params), "colour_weight": 0.0})
        for view in VIEWS:
            for o in self.objs:
                qrec, qpath = self._query(o, view)
                app = FD.appearance_feature(qpath)
                r = PR2.recognise(qpath, self.gal, bundle=zero, matcher=False, record=qrec,
                                  appearance=app, s=8, use_pair_cache=False)
                cnt, _d = FD.count_scores(self.gal.index, [qrec], n_hits=int(b.door_n_hits))
                ref_app = np.asarray([app], dtype=np.float64) @ self.gal.appearance.T
                # [M34] colour at zero removes COLOUR and nothing else: the fragment channel the
                # bundle also fuses stays, so the comparison is the four-term door, not rank_fuse.
                from matching import placement_fuse as FU2
                from matching import placement_anchors as PA2
                from matching import placement_fragment as PFM2
                frag2 = None
                if float(b.fragment_weight) != 0.0:
                    frag2 = PFM2.channel(
                        [{"places": r["places"]} for r in self.gal.flat_records()],
                        qrec["places"],
                        [PA2.for_path(p) for p in self.gal.flat_paths()],
                        PA2.for_path(qpath))
                shipped = FU2.fuse(cnt, ref_app, None, fragment=frag2,
                                   weights={"colour": 0.0,
                                            "fragment": float(b.fragment_weight)},
                                   k=int(b.rrf_k))
                order = sorted(range(self.gal.n), key=lambda j: (-shipped[0, j], j))[:8]
                self.assertEqual(list(r.shortlist), order,
                                 "shipped door mismatch on %s at %s" % (o["key"], view))


class TestTheDoorIsCore(unittest.TestCase):
    def test_the_door_is_core(self):
        import matching.placement_recognise as PR
        import matching.placement_gallery as GL
        import matching.placement_fuse as FU
        for mod in (PR, GL, FU):
            self.assertNotIn("/Topology/", mod.__file__.replace(os.sep, "/"))
            with open(mod.__file__, encoding="utf-8") as fh:
                src = fh.read()
            self.assertNotIn("import Topology", src)
            self.assertNotIn("from Topology", src)

    def test_the_readme_names_the_product(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("placement_recognise", text)
        self.assertIn("fused3", text)


if __name__ == "__main__":
    unittest.main()
