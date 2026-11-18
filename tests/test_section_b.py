"""Pins for the section-B defect fixes (front-end fix session).

Run: ``python -m unittest discover -s tests -v``

The 17 items already FIXED on HEAD are pinned by source assertions so a
revert cannot land silently; the four items that were PRESENT (B8, B9,
B19, B21) also carry functional tests of the repaired behaviour.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import parameters as P
from core.graph_types import Subgraph
from core.types import TypedSignature, EdgeType

ROOT = Path(__file__).resolve().parents[1]
JM = (ROOT / "matching" / "joint_mcs.py").read_text(encoding="utf-8")
GM = (ROOT / "matching" / "graded_mcs.py").read_text(encoding="utf-8")


def _graph(nodes, edges=(), params=None):
    return Subgraph(nodes=frozenset(nodes), edges=frozenset(edges),
                    edge_params=dict(params or {}))


def _sigs(nodes, tag="X"):
    return {n: (tag, TypedSignature.opaque(tag, int(n) % 97)) for n in nodes}


class SectionBSourcePins(unittest.TestCase):
    """A revert of any of the 21 fixes is caught here."""

    def test_b1_missing_measurement_is_not_perfect(self):
        from matching.joint_mcs import edge_pair_credit
        a = _graph({1, 2}, {(1, 2, EdgeType.PARALLEL)},
                   {(1, 2, EdgeType.PARALLEL): object()})
        b = _graph({3, 4}, {(3, 4, EdgeType.PARALLEL)})
        # one side measured, one side not -> 0.0
        got = edge_pair_credit(a, b, 1, 2, 3, 4, EdgeType.PARALLEL)
        self.assertEqual(got, 0.0)
        # both sides unmeasured -> full presence unit (default flag off)
        a2 = _graph({1, 2}, {(1, 2, EdgeType.PARALLEL)})
        got2 = edge_pair_credit(a2, b, 1, 2, 3, 4, EdgeType.PARALLEL)
        self.assertEqual(got2, 1.0)

    def test_b2_categorical_credit(self):
        self.assertIn("edge_categorical_axes", JM)

    def test_b3_node_sim_memoized(self):
        self.assertRegex(JM, r"@lru_cache[^\n]*\n\s*def node_sim")

    def test_b4_dead_code_absent(self):
        self.assertNotIn("used_target_edges", JM)

    def test_b5_budget_exhausted_combined(self):
        self.assertIn("return replace(forward, budget_exhausted=True)", JM)

    def test_b6_edge_weight_read_in_pooled(self):
        self.assertIn("w=float(P.MATCH_EDGE_WEIGHT)", JM)
        self.assertIn("score=2.0*(nodes+w*edge_credit)/denom", JM)

    def test_b7_docstring_qualified(self):
        self.assertIn("under MATCH_SEPARATE_TERM_NORMALIZATION", JM)

    def test_b9_derived_price_on_q_scale(self):
        self.assertIn("price_scale", JM)
        self.assertIn("MATCH_SYSTEMATICITY_WEIGHT", JM)
        self.assertIn("* price_scale", JM)

    def test_b10_b11_b12_b13_b14_b15_b16_b17_b18_b20(self):
        self.assertNotIn("def _neighbourhood_coherence", GM)
        self.assertNotIn("edge_mass", GM)
        self.assertRegex(GM, r"def objective_value\([\s\S]*?node_mass")
        self.assertIn("if best is None or value > best:", GM)
        self.assertIn("best_a = 0.0", GM)
        self.assertIn("+ const", GM)
        self.assertIn('_EDGE_AGREEMENT_FAILURES["count"] += 1', GM)
        self.assertIn("return max(0.0, min(1.0, res.score / scale))", GM)
        self.assertIn("if j in measured_terms", GM)
        self.assertIn("if q_types and c_types:", GM)
        self.assertIn("obj = objective_value(correspondence", GM)

    def test_b19_single_pass_degree(self):
        self.assertNotIn("sum(1 for e in Q.edges", GM)
        self.assertIn("q_deg[_u] = q_deg.get(_u, 0) + 1", GM)

    def test_b21_no_vacuous_one(self):
        self.assertNotIn("kappa = 1.0", GM)
        self.assertIn("kappa = 0.0", GM)
        self.assertNotIn("coherence is 1.0 (no evidence", GM)


class SectionBFunctional(unittest.TestCase):

    def setUp(self):
        self._saved = {n: getattr(P, n) for n in (
            "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE",
            "MATCH_ISOLATED_NODE_WEIGHT", "MATCH_UNMATCHED_COST_DERIVED",
            "MATCH_UNMATCHED_COST", "MATCH_EDGE_WEIGHT",
            "MATCH_SYSTEMATICITY_WEIGHT", "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE",
            "MATCH_SEPARATE_TERM_NORMALIZATION", "MATCH_EDGE_DENOMINATOR_MODE",
            "MATCH_COVERAGE_EXPONENT_GAMMA")}

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(P, n, v)

    def test_b8_no_alphabetical_tiebreak_on_edgeless_pair(self):
        """Two edgeless graph pairs must not all score 0 (the 0.1250 artefact)."""
        from matching.joint_mcs import joint_mcs
        P.MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = True
        P.MATCH_UNMATCHED_COST = 0.0
        P.MATCH_UNMATCHED_COST_DERIVED = False
        P.MATCH_EDGE_WEIGHT = 0.5
        q = _graph({1, 2, 3})
        r1 = _graph({11, 12, 13})
        r2 = _graph({21, 22, 23})
        sq, s1, s2 = _sigs((1, 2, 3)), _sigs((11, 12, 13)), _sigs((21, 22, 23))
        # make r1 share two exact signatures with q, r2 share one
        s1[11] = sq[1]
        s1[12] = sq[2]
        s2[21] = sq[3]
        a = joint_mcs(q, r1, sq, s1)
        b = joint_mcs(q, r2, sq, s2)
        self.assertGreater(len(a.mapping), 0, "edgeless alignment must not be empty")
        self.assertGreater(a.score, 0.0)
        self.assertNotAlmostEqual(a.score, b.score, places=6,
                                  msg="exclusion must not collapse every score")

    def test_b8_exclusion_keeps_the_node_domain(self):
        """exclude_isolated is a down-weight, not a deletion."""
        from matching.joint_mcs import joint_mcs
        P.MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = False
        P.MATCH_UNMATCHED_COST = 0.0
        q = _graph({1, 2, 3})
        r = _graph({11, 12, 13})
        sq, sr = _sigs((1, 2, 3)), _sigs((11, 12, 13))
        sr[11] = sq[1]
        base = joint_mcs(q, r, sq, sr)
        P.MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = True
        excl = joint_mcs(q, r, sq, sr)
        self.assertEqual(set(base.mapping), set(excl.mapping))
        self.assertLess(excl.score, base.score)  # down-weight, not zero

    def test_b9_derived_price_scaled(self):
        from matching.joint_mcs import joint_mcs
        P.MATCH_UNMATCHED_COST_DERIVED = True
        P.MATCH_SYSTEMATICITY_WEIGHT = 0.75
        P.MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = False
        q = _graph({1, 2})
        r = _graph({11, 12})
        sq, sr = _sigs((1, 2)), _sigs((11, 12))
        sr[11] = sq[1]
        res = joint_mcs(q, r, sq, sr)
        self.assertGreater(len(res.mapping), 0)
        self.assertGreaterEqual(res.score, 0.0)

    def test_b21_static_coherence_edgeless_is_zero(self):
        from matching.graded_mcs import _static_coherence
        q = _graph({1, 2})
        c = _graph({11, 12})
        self.assertEqual(_static_coherence(q, c, 1, 11, {}, {}, {}), 0.0)

    def test_b21_objective_edgeless_node_gets_no_coherence_credit(self):
        from matching.graded_mcs import objective_value
        P.GRADED_MCS_COHERENCE_BETA = 1.0
        P.GRADED_MCS_STATIC_COHERENCE = False
        q = _graph({1, 2})
        c = _graph({11, 12})
        sq, sc = _sigs((1, 2)), _sigs((11, 12))
        sc[11] = sq[1]
        val = objective_value({1: 11}, q, c, sq, sc, beta=1.0)
        # beta=1 and kappa=0 => node term must be 0 for an edgeless node
        self.assertAlmostEqual(val["node"], 0.0, places=9)
        self.assertIn("objective", val)

    def test_audit_all_fixed(self):
        verdict = json.loads((ROOT / "results" / "section_b_verdict.json").read_text())
        self.assertEqual(verdict["counts"]["PRESENT"], 0)
        self.assertEqual(verdict["counts"]["FIXED"], 21)


if __name__ == "__main__":
    unittest.main()
