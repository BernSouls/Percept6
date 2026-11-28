"""M26 item 2 + item 8: the vectorised registry kernel, and the properties the door needs.

The parity test is the one that must FAIL ON REVERT: if ``matching/placement_kernel.py`` ever stops
implementing the registry's kernel (the defect this mission exists to fix is exactly an equality
function masquerading as a kernel), the vectorised block stops agreeing with
``core.types.TypedSignature.similarity`` on the quantised branch and this file goes red.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from matching import placement_frontend as FE          # noqa: E402
from matching import placement_kernel as K             # noqa: E402
from matching import placement_frontdoor as FD         # noqa: E402
from core.types import ABSENT, TypedSignature          # noqa: E402

CLASSES = ("CornerPoint", "Blob", "BoundaryArc", "Keypoint")
THETA = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)
COHERENCE = (0.0, 1.0, 2.0, 3.0)
SCALE = (1.0, 1.6, 2.5, 4.0)
KIND = (0.0, 1.0, 2.0)
DIM = (1.0, 2.0)


def sig(cls: str, q) -> TypedSignature:
    return TypedSignature(kind="CANONICAL", class_=cls,
                          quantized=tuple(float(x) for x in q), population=())


def random_tuples(rng, n: int) -> np.ndarray:
    return np.column_stack([rng.choice(THETA, n), rng.choice(COHERENCE, n),
                            rng.choice(SCALE, n), rng.choice(KIND, n), rng.choice(DIM, n)])


def real_tuples(cls: str, limit: int = 1200):
    """Quantised tuples of one class from the cached ETH-80 records, or None if none are cached."""
    try:
        from Topology import m26_records as REC
    except Exception:
        return None
    if not os.path.isdir(REC.CACHE):
        return None
    out = []
    for name in sorted(os.listdir(REC.CACHE)):
        if len(out) >= limit:
            break
        if not name.endswith(".pkl"):
            continue
        try:
            rec = REC.load(*name[:-4].split("__"))
        except Exception:
            continue
        out.extend(tuple(float(x) for x in r["key"][1]) for r in rec["places"] if r["lens"] == cls)
    return np.asarray(out[:limit], dtype=np.float64) if out else None


class TestKernelParity(unittest.TestCase):
    """Item 2: the block equals the reference kernel to 1e-9, per class, random and real."""

    @classmethod
    def setUpClass(cls) -> None:
        FE.apply_recipe()

    def _check(self, cls: str, A: np.ndarray, B: np.ndarray) -> None:
        blk = K.similarity_block(cls, A, B)
        self.assertEqual(blk.shape, (A.shape[0], B.shape[0]))
        worst = 0.0
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                ref = sig(cls, A[i]).similarity(sig(cls, B[j]))
                worst = max(worst, abs(ref - blk[i, j]))
        self.assertLessEqual(worst, 1e-9, "class %s: block %g away from the reference" % (cls, worst))

    def test_parity_random_pairs_per_class(self) -> None:
        rng = np.random.default_rng(20261126)
        for cls in CLASSES:
            with self.subTest(cls=cls):
                A = random_tuples(rng, 50)
                B = random_tuples(rng, 50)
                self._check(cls, A, B)          # 2,500 pairs per class

    def test_parity_real_pairs_per_class(self) -> None:
        rng = np.random.default_rng(7)
        for cls in ("CornerPoint", "Blob", "BoundaryArc"):
            R = real_tuples(cls, 60)
            if R is None or len(R) < 4:
                self.skipTest("no cached ETH-80 records for real pairs")
            A = R[rng.integers(0, len(R), 40)]
            B = R[rng.integers(0, len(R), 40)]
            with self.subTest(cls=cls):
                self._check(cls, A, B)
                # the SAME real pairs, shuffled onto each other: exact duplicates must score 1.0
                same = K.similarity_pairs(cls, R[:50], R[:50])
                np.testing.assert_allclose(same, 1.0, atol=1e-12)

    def test_not_an_equality_function(self) -> None:
        """The revert this mission exists to prevent: an equality kernel would pass nothing here."""
        FE.apply_recipe()
        A = np.asarray([[0.0, 0.0, 1.0, 0.0, 1.0]])
        B = np.asarray([[1.0, 0.0, 1.0, 0.0, 1.0]])       # one theta bin apart
        v = float(K.similarity_block("Blob", A, B)[0, 0])
        self.assertGreater(v, 0.5, "one theta bin must not annihilate the similarity")
        self.assertLess(v, 1.0)
        far = np.asarray([[6.0, 0.0, 1.0, 0.0, 1.0]])
        self.assertLess(float(K.similarity_block("Blob", A, far)[0, 0]), v)

    def test_cyclic_theta_and_categorical_slots(self) -> None:
        FE.apply_recipe()
        a = np.asarray([[0.0, 0.0, 1.0, 0.0, 1.0]])
        b_near = np.asarray([[11.0, 0.0, 1.0, 0.0, 1.0]])   # wraps: 12 - 11 = 1 bin
        b_direct = np.asarray([[1.0, 0.0, 1.0, 0.0, 1.0]])  # one bin, no wrap
        b_two = np.asarray([[2.0, 0.0, 1.0, 0.0, 1.0]])     # two bins
        wrap = float(K.similarity_block("Blob", a, b_near)[0, 0])
        direct = float(K.similarity_block("Blob", a, b_direct)[0, 0])
        two = float(K.similarity_block("Blob", a, b_two)[0, 0])
        self.assertAlmostEqual(wrap, direct, places=12)   # 11 -> 0 is ONE bin, like 0 -> 1
        self.assertAlmostEqual(wrap, np.exp(-(4.0 / 9.0) * (1.0 / 4.0) ** 2), places=12)
        self.assertGreater(wrap, two)
        cat = np.asarray([[0.0, 0.0, 1.0, 1.0, 1.0]])
        base = np.asarray([[0.0, 0.0, 1.0, 0.0, 1.0]])
        self.assertAlmostEqual(float(K.similarity_block("Blob", base, cat)[0, 0]),
                               np.exp(-1.0), places=12)   # one categorical slot, w/W = 1/9, sigma 1/3

    def test_absent_slots_renormalise(self) -> None:
        FE.apply_recipe()
        q = np.asarray([[0.0, ABSENT, 1.0, 0.0, 1.0]])
        p = np.asarray([[1.0, 2.0, 1.0, 0.0, 1.0]])
        v = float(K.similarity_block("Blob", q, p)[0, 0])
        self.assertGreater(v, 0.0)
        all_absent = np.asarray([[ABSENT] * 5])
        self.assertEqual(float(K.similarity_block("Blob", all_absent, p)[0, 0]), 0.0)

    def test_unregistered_class_is_equality_and_counted(self) -> None:
        K.FALLBACKS.clear()
        a = np.asarray([[0.0, 0.0]])
        b = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        blk = K.similarity_block("NOT_A_REGISTERED_CLASS", a, b)
        self.assertEqual(float(blk[0, 0]), 1.0)
        self.assertEqual(float(blk[0, 1]), 0.0)
        self.assertEqual(K.FALLBACKS.get("NOT_A_REGISTERED_CLASS"), 1)   # counted, never silent


class TestGradedDoorProperties(unittest.TestCase):
    """Item 8: determinism, order independence, and a label-free idf."""

    @classmethod
    def setUpClass(cls) -> None:
        FE.apply_recipe()
        from Topology import m26_records as REC
        cls.REC = REC
        cls.keys = []
        for o in REC.objects():
            if all(os.path.exists(REC.cache_path(o["key"], v)) for v in ("000-000", "035-045")):
                cls.keys.append(o["key"])
            if len(cls.keys) >= 6:
                break
        if len(cls.keys) < 4:
            raise unittest.SkipTest("not enough cached records for the door properties")

    def _index(self):
        gal = [self.REC.load(k, "000-000") for k in self.keys]
        return FD.PlaceIndex(gal, self.keys)

    def _queries(self):
        return [self.REC.load(k, "035-045") for k in self.keys]

    def test_per_object_max_is_order_independent(self) -> None:
        idx = self._index()
        qs = self._queries()
        base, _ = FD.graded_scores(idx, qs)
        shuffled = []
        rng = np.random.default_rng(3)
        for e in idx.entries:
            rows = list(e["places"])
            rng.shuffle(rows)
            shuffled.append(dict(e) | {"places": rows})
        idx2 = FD.PlaceIndex(shuffled, self.keys)
        other, _ = FD.graded_scores(idx2, qs)
        np.testing.assert_allclose(base, other, atol=1e-12)

    def test_idf_is_label_free_and_permutation_equivariant(self) -> None:
        idx = self._index()
        qs = self._queries()
        base, _ = FD.graded_scores(idx, qs)
        order = [2, 0, 3, 1] + list(range(4, len(self.keys)))
        idx2 = FD.PlaceIndex([idx.entries[j] for j in order], [self.keys[j] for j in order])
        other, _ = FD.graded_scores(idx2, qs)
        np.testing.assert_allclose(base[:, order], other, atol=1e-12)

    def test_graded_door_is_deterministic_across_hash_seeds(self) -> None:
        code = (
            "import sys,numpy as np;sys.path.insert(0,%r)\n"
            "import matching.placement_frontend as FE;FE.apply_recipe()\n"
            "from Topology import m26_records as REC\n"
            "from matching import placement_frontdoor as FD\n"
            "keys=%r\n"
            "idx=FD.PlaceIndex([REC.load(k,'000-000') for k in keys],keys)\n"
            "M,_=FD.graded_scores(idx,[REC.load(k,'035-045') for k in keys])\n"
            "print(np.round(M,12).tolist())\n" % (ROOT, list(self.keys)))
        outs = []
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env=env, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            outs.append(r.stdout.strip())
        self.assertEqual(len(set(outs)), 1, "the graded door changed with the hash seed")


class TestCountedDoorUnchanged(unittest.TestCase):
    """Item 8: the counted door's arithmetic is the M22 arithmetic, still."""

    @classmethod
    def setUpClass(cls) -> None:
        FE.apply_recipe()
        from Topology import m26_records as REC
        cls.REC = REC
        cls.keys = []
        for o in REC.objects():
            if all(os.path.exists(REC.cache_path(o["key"], v)) for v in ("000-000", "035-045")):
                cls.keys.append(o["key"])
            if len(cls.keys) >= 3:
                break
        if len(cls.keys) < 3:
            raise unittest.SkipTest("no cached records")

    def test_count_scores_equals_the_reference_formula(self) -> None:
        import math
        gal = [self.REC.load(k, "000-000") for k in self.keys]
        idx = FD.PlaceIndex(gal, self.keys)
        q = self.REC.load(self.keys[0], "035-045")
        got, diag = FD.count_scores(idx, [q], n_hits=8)
        # the M22 formula, written out again: each live query place counts the objects of its 8
        # nearest pool places by the frozen ruler, score = count / sqrt(used * object places)
        used, cnt = 0, {}
        for row in q["places"]:
            if int(np.count_nonzero(row["Mz"])) < 2:
                continue
            pool = idx.by_key.get(idx._keyfn(row))
            if not pool:
                continue
            used += 1
            ranked = sorted(pool, key=lambda t: (FD.cfg_distance(row, t[1]), t[0], t[1]["n"]))
            for (obj, ref) in ranked[:8]:
                if int(np.count_nonzero(ref["Mz"])) >= 2:
                    cnt[obj] = cnt.get(obj, 0) + 1
        for obj, c in cnt.items():
            want = c / math.sqrt(max(1, used) * max(1, idx.n_places[obj]))
            self.assertAlmostEqual(float(got[0, obj]), want, places=12)
        self.assertEqual(diag[0]["places_used"], used)

    def test_shortlist_rule_fixed_is_the_old_behaviour(self) -> None:
        from matching import placement_bundle as PB
        b = PB.PlacementBundle.load()
        scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2999, 0.2])
        order = list(range(len(scores)))
        from matching import placement_recognise as RC
        fixed = RC._shortlist_size(scores, order, b, 8) if b.shortlist_rule == "fixed" else None
        if fixed is not None:
            self.assertEqual(fixed, 8)
        # the adaptive rule only ever WIDENS, never re-orders or shortens
        class _B:
            shortlist_rule = "adaptive"
            shortlist_wide = 16
            margin_tau = 0.5
        self.assertEqual(RC._shortlist_size(scores, order, _B(), 8), 16)
        class _Clear:
            shortlist_rule = "adaptive"
            shortlist_wide = 16
            margin_tau = 0.0
        self.assertEqual(RC._shortlist_size(scores, order, _Clear(), 8), 8)


class TestDoorKindAndCore(unittest.TestCase):
    def test_bundle_kind_is_one_the_tree_implements(self) -> None:
        from matching import placement_bundle as PB
        b = PB.PlacementBundle.load()
        # [M35] M31 shipped the fused three-channel door, M35 ships the object-level bag door; every
        # kind the loader accepts is one the tree implements, which is the claim this test makes.
        self.assertIn(b.door_kind, ("counted", "graded", "fused3", "bag"))
        if b.door_kind == "counted":
            self.assertEqual(b.door_params, {})

    def test_unknown_door_kind_is_refused(self) -> None:
        import json
        import tempfile
        from matching import placement_bundle as PB
        b = PB.PlacementBundle.load()
        with open(PB.BUNDLE_PATH) as fh:
            d = json.load(fh)
        d["door"]["kind"] = "magic"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(d, fh)
            p = fh.name
        try:
            with self.assertRaises(PB.BundleRefusal):
                PB.PlacementBundle.load(p)
        finally:
            os.unlink(p)

    def test_core_does_not_import_topology(self) -> None:
        for name in ("placement_kernel.py", "placement_frontdoor.py", "placement_gallery.py",
                     "placement_recognise.py", "placement_bundle.py"):
            p = os.path.join(ROOT, "matching", name)
            tree = ast.parse(open(p, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertFalse(a.name.startswith("Topology"),
                                         "%s imports %s" % (name, a.name))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith("Topology"),
                                     "%s imports %s" % (name, node.module))


if __name__ == "__main__":
    unittest.main()
