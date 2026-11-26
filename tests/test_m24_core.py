"""Mission 24 -- the core entry point's own guarantees, each a test that fails on revert.

    no Topology import      a core package must not import from `Topology/` or edit sys.path
    refusal                 a missing bundle, or a bundle frozen for other code, is a REFUSAL
    no env needed           recognise builds and ranks with no PLACEMENT_* variable set
    no leak                 every switch the entry point applies is restored afterwards
    parallel == serial      the 8 pair scores may run in a pool; the ORDER of the result may not
                            depend on it
    recipe drift            the recipe in core is the recipe the Mission-20 record used
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from matching import placement_bundle as PB                                    # noqa: E402

CORE_PACKAGES = ("matching", "core", "parameters", "perception", "lenses", "dynamics", "memory",
                 "learning", "action")
ROOT_MODULES = ("eth80_frontend.py", "concept_graph.py", "consolidator.py",
                "derived_lenses.py", "sensory_floor.py")

CORPUS = os.path.join(ROOT, "data", "ETH-80-master")


def _core_files():
    for pkg in CORE_PACKAGES:
        d = os.path.join(ROOT, pkg)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                yield os.path.join(pkg, f), os.path.join(d, f)
    for f in ROOT_MODULES:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            yield f, p


class TestNoTopologyImport(unittest.TestCase):
    """A core package imports nothing from Topology/ and never edits sys.path at import."""

    def test_no_topology_imports(self) -> None:
        bad = []
        for rel, path in _core_files():
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name == "Topology" or a.name.startswith("Topology."):
                            bad.append((rel, node.lineno, a.name))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "Topology" or node.module.startswith("Topology."):
                        bad.append((rel, node.lineno, node.module))
        self.assertEqual(bad, [], "core imports from Topology/: %r" % (bad[:5],))

    def test_no_sys_path_editing(self) -> None:
        """No module-level `sys.path` mutation in a core package (the CLI may, in __main__)."""
        bad = []
        for rel, path in _core_files():
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            guarded = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    src = ast.dump(node.test)
                    if "__name__" in src:
                        for sub in ast.walk(node):
                            guarded.add(id(sub))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AugAssign, ast.Call)):
                    continue
                target = "sys.path"
                hit = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                            and sub.value.id == "sys" and sub.attr == "path":
                        hit = True
                if hit and id(node) not in guarded:
                    bad.append((rel, getattr(node, "lineno", 0)))
        self.assertEqual(bad, [], "core edits sys.path at import: %r" % (bad[:5],))


class TestBundleRefusal(unittest.TestCase):
    """A missing or stale bundle is a refusal, never a default."""

    def test_missing_bundle_refuses(self) -> None:
        with self.assertRaises(PB.BundleRefusal) as cm:
            PB.PlacementBundle.load(os.path.join(tempfile.mkdtemp(), "nope.json"))
        self.assertIn("refuses", str(cm.exception))

    def test_stale_bundle_refuses(self) -> None:
        d = json.load(open(PB.BUNDLE_PATH))
        d["code_digest"] = "0" * 64
        p = os.path.join(tempfile.mkdtemp(), "stale.json")
        with open(p, "w") as fh:
            json.dump(d, fh)
        with self.assertRaises(PB.BundleRefusal) as cm:
            PB.PlacementBundle.load(p)
        self.assertIn("re-freeze", str(cm.exception))

    def test_bundle_is_frozen_for_this_code(self) -> None:
        b = PB.PlacementBundle.load()
        self.assertEqual(b.code_digest, PB.compute_code_digest())
        self.assertEqual(int(b.shortlist), 8)
        self.assertEqual(int(b.rrf_k), 60)
        self.assertEqual(b.rerank, "plain")
        self.assertAlmostEqual(float(b.matcher_weight), PB.shipped_weight(), places=12)

    def test_recipe_mismatch_refuses(self) -> None:
        from matching import placement_recognise as RC
        b = PB.PlacementBundle.load()
        bad = PB.PlacementBundle(
            name=b.name, mission=b.mission, date=b.date, shortlist=b.shortlist, rrf_k=b.rrf_k,
            rerank=b.rerank, door_n_hits=b.door_n_hits, appearance=b.appearance,
            matcher_weight=b.matcher_weight,
            recipe=dict(b.recipe, FE_KEYPOINTS=not b.recipe["FE_KEYPOINTS"]),
            recipe_extra=dict(b.recipe_extra), keypoint_class=b.keypoint_class,
            provenance=dict(b.provenance), code_digest=b.code_digest)
        with self.assertRaises(PB.BundleRefusal):
            RC._validate(bad)

    def test_unknown_rerank_refuses(self) -> None:
        from matching import placement_recognise as RC
        b = PB.PlacementBundle.load()
        bad = PB.PlacementBundle(
            name=b.name, mission=b.mission, date=b.date, shortlist=b.shortlist, rrf_k=b.rrf_k,
            rerank="fused", door_n_hits=b.door_n_hits, appearance=b.appearance,
            matcher_weight=b.matcher_weight, recipe=dict(b.recipe),
            recipe_extra=dict(b.recipe_extra), keypoint_class=b.keypoint_class,
            provenance=dict(b.provenance), code_digest=b.code_digest)
        with self.assertRaises(PB.BundleRefusal):
            RC._validate(bad)


def _gallery_items():
    base = os.path.join(CORPUS, "1")
    out = []
    for inst in ("1", "2", "3"):
        p = os.path.join(base, inst, "apple%s-000-000.png" % inst)
        if os.path.exists(p):
            out.append(("apple/%s" % inst, p))
    return out


@unittest.skipUnless(os.path.isdir(CORPUS) and len(_gallery_items()) == 3,
                     "ETH-80 corpus not present (fetch it with Topology/fetch_corpus.py --fetch)")
class TestEntryPointNeedsNoEnv(unittest.TestCase):
    """The entry point works with no PLACEMENT_* variable set, and leaks nothing when it does."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.items = _gallery_items()
        cls.query = os.path.join(CORPUS, "1", "4", "apple4-035-045.png")

    def _snapshot(self):
        import parameters as P
        return {k: getattr(P, k) for k in dir(P) if k.startswith("PLACEMENT_") and
                not callable(getattr(P, k))}

    def test_no_env_and_no_leak(self) -> None:
        from matching import placement_recognise as RC
        saved = {k: v for k, v in os.environ.items() if k.startswith("PLACEMENT_")}
        for k in saved:
            del os.environ[k]
        before = self._snapshot()
        try:
            g = RC.build_gallery(self.items, workers=1)
            r = RC.recognise(self.query, g, workers=0, matcher=False)
        finally:
            os.environ.update(saved)
        after = self._snapshot()
        changed = {k: (before[k], after[k]) for k in before if before[k] != after.get(k)}
        self.assertEqual(changed, {}, "the entry point leaked switches: %r" % (changed,))
        self.assertEqual(len(r.shortlist), min(8, len(self.items)))
        self.assertEqual(sorted(r.shortlist), list(range(len(self.items))))
        self.assertEqual(r.unscored, r.shortlist)          # nothing scored, nothing claimed
        self.assertEqual(len(r.ranking), len(self.items))
        for k in ("appearance", "door", "matcher", "total"):
            self.assertIn(k, r.seconds)


class TestParallelEqualsSerial(unittest.TestCase):
    """The pool may not change the answer: same ranking, same scores, both orders of work."""

    @unittest.skipUnless(os.path.isdir(CORPUS) and len(_gallery_items()) == 3,
                         "ETH-80 corpus not present")
    def test_parallel_order_equals_serial(self) -> None:
        from matching import placement_frontend as FE
        from matching import placement_recognise as RC
        g = RC.build_gallery(_gallery_items(), workers=1)
        q = os.path.join(CORPUS, "1", "4", "apple4-035-045.png")
        real = FE.pair_score

        def fake(qpath, rpath, weight=None):
            # deterministic, distinct per pair, and cheap: the ORDER is what is being tested
            return float(sum(bytearray(os.path.basename(rpath).encode())) % 997) / 997.0
        FE.pair_score = fake                                       # type: ignore[assignment]
        try:
            ser = RC.recognise(q, g, workers=0)
            par = RC.recognise(q, g, workers=2)
        finally:
            FE.pair_score = real                                   # type: ignore[assignment]
        self.assertEqual(ser.shortlist, par.shortlist)
        self.assertEqual(ser.ranking, par.ranking)
        self.assertEqual(ser.scores, par.scores)
        self.assertEqual(len(ser.scores), len(ser.shortlist))
        # and the order is the shipped one, not the door's
        self.assertEqual(ser.ranking, RC.order_plain(ser.shortlist, ser.scores)
                         + [j for j in ser.door_order if j not in set(ser.shortlist)])


class TestRecipeIsTheRecordedRecipe(unittest.TestCase):
    """The recipe that moved into core is the recipe the Mission-20 record was built with."""

    def test_core_recipe_equals_harness_recipe(self) -> None:
        from matching import placement_frontend as FE
        sys.path.insert(0, os.path.join(ROOT, "Topology"))
        from Topology import placement_m20_eth80 as M20
        self.assertEqual(dict(FE.RECIPE), dict(M20.RECIPE))
        self.assertEqual(FE.RECIPE_EXTRA["KP_GRADED_POPULATION"], True)
        self.assertEqual(FE.RECIPE_KEYPOINT_CLASS, "wide")


if __name__ == "__main__":
    unittest.main()
