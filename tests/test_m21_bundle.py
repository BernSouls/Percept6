"""MISSION 21 -- the tests that pin each change, and go RED if it is reverted.

Each test names the property the mission asked for and would fail if the code went back to
the Mission-20 convention.  Nothing here builds a graph: the stage-1 ruler, the descriptor
variants, the chance arithmetic, the attribution tagger and the coverage control are pure
functions, so the whole file runs in seconds.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]


def _params(k: int = 4, sigma_d: float = 0.35, sigma_a: float = 0.5) -> dict:
    return {"K": k, "sigma_d": sigma_d, "sigma_a": sigma_a,
            "mu_D": [0.0] * k, "sd_D": [1.0] * k, "mu_cos": [0.0] * k,
            "sd_cos": [1.0] * k, "mu_sin": [0.0] * k, "sd_sin": [1.0] * k,
            "keep_D": [1] * k}


def _raw(rows, k: int = 4) -> dict:
    D = np.zeros((len(rows), k), dtype=float)
    A = np.zeros((len(rows), k), dtype=float)
    M = np.zeros((len(rows), k), dtype=float)
    for r, (ds, angs) in enumerate(rows):
        n = min(len(ds), len(angs), k)
        D[r, :n] = ds[:n]
        A[r, :n] = angs[:n]
        M[r, :n] = 1.0
    return {"D": D, "A": A, "M": M, "ids": list(range(len(rows)))}


class FrozenRulerTests(unittest.TestCase):
    def test_params_file_exists_with_provenance(self):
        from matching import placement_prefilter as PF
        p = PF.load_params()
        self.assertIsNotNone(p, "matching/placement_prefilter_params.json must exist")
        for key in ("K", "sigma_d", "sigma_a", "mu_D", "sd_D", "keep_D", "mu_cos", "sd_cos",
                    "mu_sin", "sd_sin", "fitted_on", "chosen_by", "code_digest"):
            self.assertIn(key, p)
        self.assertEqual(len(p["mu_D"]), p["K"])
        self.assertEqual(len(p["keep_D"]), p["K"])
        self.assertGreater(p["fitted_on"]["n_images"], 0)
        self.assertEqual(p["fitted_on"]["split"], "FIT")

    def test_scoring_refuses_without_params(self):
        """No silent fallback: a missing ruler is a refusal, not a quietly different number."""
        from matching import placement_prefilter as PF
        a = PF.standardize(_raw([([1.0, 2.0], [0.0, 1.0])]), _params())
        old_cache, old_path = PF._PARAMS_CACHE, PF.PARAMS_PATH
        try:
            PF._PARAMS_CACHE = None
            PF.PARAMS_PATH = "/nonexistent/params.json"
            with self.assertRaises(RuntimeError):
                PF.score_bags_v2(a, a, None)
        finally:
            PF._PARAMS_CACHE, PF.PARAMS_PATH = old_cache, old_path

    def test_standardization_is_a_function_of_the_image_alone(self):
        from matching import placement_prefilter as PF
        p = _params()
        bag = _raw([([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]), ([4.0, 5.0], [1.0, 2.0])])
        a1 = PF.standardize(bag, p)
        _ = PF.standardize(_raw([([9.0, 9.0, 9.0], [3.0, 3.0, 3.0])]), p)
        a2 = PF.standardize(bag, p)
        self.assertTrue(np.array_equal(a1["D"], a2["D"]))
        self.assertTrue(np.array_equal(a1["M"], a2["M"]))

    def test_score_matches_an_independent_reimplementation(self):
        """RED if the score goes back to per-reference std/keep.

        The reference implementation uses ONLY the frozen parameters, so a return of the
        Mission-20 convention (any statistic of the reference bag) makes the numbers differ.
        """
        from matching import placement_prefilter as PF
        p = _params()
        A = PF.standardize(_raw([([1.0, 2.0], [0.0, 1.0]), ([2.0, 3.0], [0.5, 2.0])]), p)
        B = PF.standardize(_raw([([1.5, 2.5], [0.1, 1.1]), ([3.0, 4.0], [1.0, 2.5]),
                                 ([0.5, 1.0], [2.0, 3.0])]), p)
        got = PF.score_bags_v2(A, B, p)["score"]
        sims = []
        for i in range(A["D"].shape[0]):
            best = 0.0
            for j in range(B["D"].shape[0]):
                n, d2, a2 = 0, 0.0, 0.0
                for t in range(p["K"]):
                    if A["M"][i, t] * B["M"][j, t] <= 0:
                        continue
                    n += 1
                    d2 += (A["D"][i, t] - B["D"][j, t]) ** 2
                    a2 += ((A["C"][i, t] - B["C"][j, t]) ** 2
                           + (A["S"][i, t] - B["S"][j, t]) ** 2)
                if n < 2:
                    continue
                best = max(best, math.exp(-0.5 * (d2 / (n * p["sigma_d"] ** 2)
                                                  + a2 / (n * p["sigma_a"] ** 2))))
            sims.append(best)
        ref = float(np.mean([s for s in sims if s > 0]))
        self.assertAlmostEqual(got, ref, places=12)

    def test_no_per_reference_statistic_in_the_scoring_path(self):
        import inspect
        from matching import placement_prefilter as PF
        src = (inspect.getsource(PF.score_bags_v2) + inspect.getsource(PF.pair_terms)
               + inspect.getsource(PF.score_terms) + inspect.getsource(PF.standardize))
        self.assertNotIn(".std(", src)
        self.assertNotIn("keep = sd", src)
        self.assertNotIn("B[:, keep]", src)


class PaddingAndKTests(unittest.TestCase):
    def test_padding_is_a_mask_not_a_repeated_last_entry(self):
        from matching import placement_prefilter as PF
        bag = _raw([([1.0, 2.0], [0.0, 1.0]), ([], [])], k=5)
        self.assertEqual(bag["M"][0].tolist(), [1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(bag["D"][0].tolist(), [1.0, 2.0, 0.0, 0.0, 0.0])
        self.assertEqual(float(bag["M"][1].sum()), 0.0)

    def test_descriptor_variant_default_and_shapes(self):
        import parameters as P
        from matching import placement_context as Ctx
        self.assertEqual(P.PLACEMENT_NODE_DESC_VARIANT, "sorted")
        self.assertEqual(Ctx.descriptor_variant(), "sorted")
        ds = [1.0, 2.0, 3.0, 5.0]
        self.assertEqual(Ctx._shape(ds, "sorted"), ds)
        self.assertEqual(Ctx._shape(ds, "rank"), [0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(Ctx._shape(ds, "angles"), [0.0, 0.0, 0.0, 0.0])
        mu = sum(ds) / 4
        for a, b in zip(Ctx._shape(ds, "ratio"), [x - mu for x in ds]):
            self.assertAlmostEqual(a, b, places=12)


class ScaleFindingTests(unittest.TestCase):
    def test_config_similarity_is_not_invariant_to_a_uniform_scale(self):
        """The measured mechanism, pinned as a number.

        A uniform scale s multiplies image distances by s, i.e. ADDS log(s) to every log
        distance; the estimator differences the vectors, so the sum of squares grows by
        n log(s)^2 and the similarity is exp(-0.5 log(s)^2 / sigma_d^2) -- not 1.  Centring
        each place ("ratio") removes the term, which is why that variant exists.
        """
        from matching import placement_context as Ctx
        d = ([1.2, 1.9, 2.4, 3.0], [0.1, 0.9, 2.0, -1.4])
        c = math.log(1.15)
        shifted = ([x + c for x in d[0]], list(d[1]))
        got = Ctx.config_similarity(d, shifted)
        want = math.exp(-0.5 * c * c / (Ctx.SIGMA_D ** 2))
        self.assertAlmostEqual(got, want, places=12)
        self.assertLess(got, 0.95)
        centred_a = (Ctx._shape(d[0], "ratio"), d[1])
        centred_b = (Ctx._shape(shifted[0], "ratio"), d[1])
        self.assertAlmostEqual(Ctx.config_similarity(centred_a, centred_b), 1.0, places=12)


class ChanceTests(unittest.TestCase):
    def test_chance_is_per_task_and_not_one_over_gallery(self):
        from Topology.placement_m21_shortlist import _hyper_ge
        self.assertAlmostEqual(_hyper_ge(4, 39, 8), 0.61745, places=4)
        self.assertAlmostEqual(_hyper_ge(1, 39, 8), 8 / 39, places=3)
        self.assertEqual(_hyper_ge(0, 39, 8), 0.0)


class AttributionTests(unittest.TestCase):
    def test_tagger_ranks_by_the_composite_it_tests(self):
        """RED if the tagger ranks by the shipped score: AGGREGATION becomes unreachable."""
        from Topology import placement_m21_attrib as A
        st = A.selftest()
        self.assertTrue(st["ok"], st)
        self.assertEqual(st["tag_S0"], "")
        self.assertEqual(st["tag_S1"], "AGGREGATION")

    def test_tagger_on_real_runs_is_not_degenerate(self):
        path = os.path.join(ROOT, "Topology", "results", "placement", "m21_attrib.json")
        if not os.path.exists(path):
            self.skipTest("run Topology/placement_m21_attrib.py first")
        d = json.load(open(path))
        runs = [r for r in d["runs"] if "arm" in r]
        self.assertTrue(runs)
        for r in runs:
            self.assertEqual(sum(r["tags"].values()) + r["top1"], r["queries"])
        self.assertTrue([r for r in runs if r["arm"] in ("S1", "S2")
                         and sum(r["tags"].values()) > 0])


class CoverageTests(unittest.TestCase):
    def test_coverage_subsamples_and_recomputes(self):
        from Topology import placement_m21_coverage as C
        row = {"q": "q1", "r": "r1", "B": 0.5, "E": 0.9, "S3": 0.9, "den": 100,
               "places": [[1, 1, 0.9, 1], [2, 2, 0.9, 1], [3, 3, 0.9, 1], [4, 4, 0.5, 0]]}
        rec = C.recompute(row, 2, 7, 0, 8.0)
        self.assertEqual(rec["mapped"], 2)
        self.assertLessEqual(rec["judged"], 2)
        self.assertAlmostEqual(rec["E"], sum(0.9 - 0.5 for _ in range(rec["judged"]))
                               / (rec["judged"] + 8.0), places=12)
        self.assertAlmostEqual(rec["S3"], sum(0.9 - 0.5 for _ in range(rec["judged"]))
                               / (100 + 8.0), places=12)
        self.assertEqual(C.recompute(row, 2, 7, 0, 8.0), rec)

    def test_coverage_is_deterministic_across_processes(self):
        code = (
            "import sys; sys.path[:0]=[%r,%r]\n"
            "from Topology import placement_m21_coverage as C\n"
            "row={'q':'a','r':'b','B':0.1,'E':0.0,'S3':0.0,'den':50,"
            "'places':[[i,i,0.7,1] for i in range(20)]}\n"
            "print(C.recompute(row,5,11,3,8.0))\n" % (ROOT, os.path.join(ROOT, "Topology")))
        outs = []
        for seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PLACEMENT_M19="1")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env, cwd=ROOT)
            outs.append(r.stdout.strip())
        self.assertTrue(outs[0], outs)
        self.assertEqual(outs[0], outs[1])


class PairTableTests(unittest.TestCase):
    def test_pair_terms_are_symmetric_before_aggregation(self):
        from matching import placement_prefilter as PF
        p = _params()
        A = PF.standardize(_raw([([1.0, 2.0], [0.0, 1.0]), ([3.0, 1.0], [2.0, 0.5])]), p)
        B = PF.standardize(_raw([([1.2, 2.2], [0.1, 1.1]), ([0.0, 1.0], [0.3, 2.0]),
                                 ([2.0, 2.0], [1.0, 1.0])]), p)
        t1, n1 = PF.pair_terms(A, B)
        t2, n2 = PF.pair_terms(B, A)
        self.assertTrue(np.allclose(t1["d2"], t2["d2"].T))
        self.assertTrue(np.allclose(t1["a2"], t2["a2"].T))
        self.assertTrue(np.array_equal(t1["live"], t2["live"].T))
        self.assertTrue(np.allclose(n1, n2.T))
        self.assertIsNotNone(PF.score_terms(t1, 0.35, 0.5))


if __name__ == "__main__":
    unittest.main()
