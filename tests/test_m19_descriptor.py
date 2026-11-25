"""MISSION 19 / O1 and O2 -- the node descriptor, the band it re-ranks, and the adaptive cap.

The signature similarity is gated by five quantised slots, so places that share a key are
ordered only by the graded population term: the measured rank-1 share at 15 deg is 0.059
with 43-51 % of places sharing their full quantised key.  The configuration descriptor is a
per-node geometry needing no mapping; blending it into the head of each row is what these
tests pin.

The band (`(1-w) * s_k` and below keeps its raw value) is the part that is easy to get
wrong, so it is pinned against a brute-force full-row blend: the two must produce the same
ordering on random data.  If the band is narrowed by mistake, that equivalence breaks
before any acceptance number moves.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P
from matching import placement_context as C
from matching import placement_matcher as PM
from matching.placement import PlacementVector


def _vec(dist: float, direction: float, theta: float = 0.0) -> PlacementVector:
    return PlacementVector(direction=direction, distance=dist, log_scale=0.0,
                           source_theta=theta, source_scale=1.0, axial=False)


def _side(star: dict) -> PM.Side:
    incident, index, nodes = {}, {}, set(star)
    for u, nb in star.items():
        incident[u] = []
        for v, dist in nb.items():
            incident[u].append((v, True, "E"))
            index[(u, v, "E")] = _vec(dist, 0.35 * v)
            nodes.add(v)
    for v in nodes:
        if v not in incident:
            incident[v] = []
    return PM.Side(payload={}, index=index, incident=incident, lens={},
                   nodes=tuple(sorted(nodes)))


class NodeDescriptorTests(unittest.TestCase):
    def setUp(self):
        self._saved = (P.PLACEMENT_NODE_DESC, P.PLACEMENT_NODE_DESC_K,
                       P.PLACEMENT_NODE_DESC_W, P.PLACEMENT_NODE_DESC_MAXCELLS)
        P.PLACEMENT_NODE_DESC = False
        P.PLACEMENT_NODE_DESC_MAXCELLS = 0

    def tearDown(self):
        (P.PLACEMENT_NODE_DESC, P.PLACEMENT_NODE_DESC_K,
         P.PLACEMENT_NODE_DESC_W, P.PLACEMENT_NODE_DESC_MAXCELLS) = self._saved

    def test_off_by_default_and_the_matrix_is_the_same_object(self):
        m = [[0.5, 0.1], [0.2, 0.4]]
        out = PM.apply_node_descriptor(m, [0, 1], [7, 8], _side({0: {1: 1.0}}),
                                       _side({7: {8: 1.0}}))
        self.assertIs(out, m, "the descriptor must not touch the matrix when it is off")

    def test_it_prefers_the_matching_configuration(self):
        """Two candidates with the SAME similarity: only the configuration can separate
        them, and the one whose star matches must win.  RED if the blend is removed."""
        P.PLACEMENT_NODE_DESC = True
        P.PLACEMENT_NODE_DESC_K = 2
        P.PLACEMENT_NODE_DESC_W = 0.5
        a = _side({0: {1: 2.0, 2: 4.0}, 1: {}, 2: {}})
        b = _side({10: {11: 2.0, 12: 4.0}, 20: {21: 8.0, 22: 1.0}})
        m = [[0.6, 0.6]]
        out = PM.apply_node_descriptor(m, [0], [10, 20], a, b)
        self.assertGreater(out[0][0], out[0][1])

    def test_the_whole_row_blend_matches_a_brute_force_blend(self):
        """Default (`MAXCELLS = 0`) blends EVERY positive cell, so the row keeps one score
        scale and the ordering is exactly the brute-force blend.  RED if the blend is
        restricted to the head: a head cell with a poor configuration is attenuated to
        `(1-w)*s`, and an unblended tail cell would then outscore it on a different scale."""
        import random
        P.PLACEMENT_NODE_DESC = True
        P.PLACEMENT_NODE_DESC_K = 3
        P.PLACEMENT_NODE_DESC_W = 0.5
        P.PLACEMENT_NODE_DESC_MAXCELLS = 0
        rng = random.Random(7)
        n_b = 12
        star_b = {100 + j: {1000 + j: 1.0 + j, 1001 + j: 3.0 - 0.1 * j}
                  for j in range(n_b)}
        b = _side(star_b)
        a = _side({0: {1: 2.0, 2: 5.0}, 1: {}, 2: {}})
        nodes_b = [100 + j for j in range(n_b)]
        for _trial in range(20):
            row = [0.0 if rng.random() < 0.25 else round(rng.random(), 3)
                   for _ in range(n_b)]
            out = PM.apply_node_descriptor([row], [0], nodes_b, a, b)[0]
            dq = C.config_descriptor(a, 0)
            full = list(row)
            for j in range(n_b):
                if row[j] > 0.0:
                    c = C.config_similarity(dq, C.config_descriptor(b, nodes_b[j]))
                    full[j] = 0.5 * row[j] + 0.5 * c
            self.assertEqual(
                sorted(range(n_b), key=lambda j: (-out[j], nodes_b[j])),
                sorted(range(n_b), key=lambda j: (-full[j], nodes_b[j])),
                "the row blend disagrees with the brute-force blend")

    def test_the_head_restriction_is_the_thing_that_breaks_the_scale(self):
        """A one-cell cap must visibly change a row whose tail beats an attenuated head,
        and must be COUNTED while doing it."""
        P.PLACEMENT_NODE_DESC = True
        P.PLACEMENT_NODE_DESC_K = 1
        P.PLACEMENT_NODE_DESC_W = 0.5
        P.PLACEMENT_NODE_DESC_MAXCELLS = 0
        a = _side({0: {1: 2.0, 2: 5.0}, 1: {}, 2: {}})
        b = _side({10: {11: 2.0, 12: 4.0}, 20: {21: 8.0, 22: 1.0}})
        row = [0.9, 0.55]
        whole = PM.apply_node_descriptor([row], [0], [10, 20], a, b)[0]
        P.PLACEMENT_NODE_DESC_MAXCELLS = 1
        head = PM.apply_node_descriptor([row], [0], [10, 20], a, b)[0]
        self.assertEqual(head[1], row[1], "a truncated row's tail keeps its raw value")
        self.assertNotEqual(whole, head, "the two policies must differ on this row")

    def test_a_truncated_band_is_counted(self):
        P.PLACEMENT_NODE_DESC = True
        P.PLACEMENT_NODE_DESC_K = 2
        P.PLACEMENT_NODE_DESC_MAXCELLS = 1
        a = _side({0: {1: 2.0, 2: 4.0}, 1: {}, 2: {}})
        b = _side({10: {11: 2.0, 12: 4.0}, 20: {21: 8.0, 22: 1.0}})
        before = int(PM.FAILURES.get("desc_capped", 0))
        PM.apply_node_descriptor([[0.9, 0.9]], [0], [10, 20], a, b)
        self.assertGreater(int(PM.FAILURES.get("desc_capped", 0)), before,
                           "a truncated re-rank band must be counted, not silent")

    def test_determinism_and_ties_broken_on_node_id(self):
        P.PLACEMENT_NODE_DESC = True
        P.PLACEMENT_NODE_DESC_K = 2
        P.PLACEMENT_NODE_DESC_W = 0.5
        a = _side({0: {1: 2.0, 2: 4.0}, 1: {}, 2: {}})
        b = _side({10: {11: 2.0, 12: 4.0}, 20: {21: 2.0, 22: 4.0}})
        m = [[0.7, 0.7]]
        one = PM.apply_node_descriptor(m, [0], [10, 20], a, b)
        two = PM.apply_node_descriptor(m, [0], [10, 20], a, b)
        self.assertEqual(one, two)


class AdaptiveCapTests(unittest.TestCase):
    def setUp(self):
        self._saved = (P.PLACEMENT_CAP_ADAPTIVE, P.PLACEMENT_CAP_ADAPTIVE_K,
                       P.PLACEMENT_CAP_ADAPTIVE_MAX)
        P.PLACEMENT_CAP_ADAPTIVE = False

    def tearDown(self):
        (P.PLACEMENT_CAP_ADAPTIVE, P.PLACEMENT_CAP_ADAPTIVE_K,
         P.PLACEMENT_CAP_ADAPTIVE_MAX) = self._saved

    def test_the_cap_is_clamped_and_monotone_in_the_row(self):
        P.PLACEMENT_CAP_ADAPTIVE = True
        P.PLACEMENT_CAP_ADAPTIVE_K = 0.125
        P.PLACEMENT_CAP_ADAPTIVE_MAX = 64
        got = [PM.adaptive_cap(24, n) for n in (0, 37, 114, 192, 315, 586, 5000)]
        self.assertEqual(got, sorted(got), "the cap must not fall as the row grows")
        self.assertTrue(all(24 <= g for g in got), "the base cap must never be undercut")
        self.assertTrue(all(g <= 64 for g in got), "the ceiling must hold")
        self.assertEqual(PM.adaptive_cap(24, 586), 64)
        self.assertEqual(PM.adaptive_cap(24, 114), 24)

    @staticmethod
    def _matrix():
        return [[0.9, 0.8, 0.7, 0.0], [0.1, 0.2, 0.0, 0.0]]

    def test_off_is_exactly_top_candidates(self):
        m = self._matrix()
        nb = [4, 5, 6, 7]
        P.PLACEMENT_CAP_ADAPTIVE = False
        self.assertEqual(PM.top_candidates_adaptive(m, nb, 2),
                         PM.top_candidates(m, nb, 2))

    def test_on_it_grows_only_the_dense_row(self):
        P.PLACEMENT_CAP_ADAPTIVE = True
        P.PLACEMENT_CAP_ADAPTIVE_K = 1.0
        P.PLACEMENT_CAP_ADAPTIVE_MAX = 8
        rows = PM.top_candidates_adaptive(self._matrix(), [4, 5, 6, 7], 1)
        self.assertEqual(len(rows[0]), 3, "a row of 3 should give cap 3")
        self.assertEqual(len(rows[1]), 2, "a row of 2 should give cap 2")


if __name__ == "__main__":
    unittest.main()


class DedupTests(unittest.TestCase):
    """[M19 / O2] The cap spent on distinct keys instead of copies.

    It can lose the truth -- dedup drops a candidate that duplicates a key already taken --
    so the property pinned here is the MECHANISM, not a promise of recall: the head is one
    candidate per distinct key, the row is still `cap` long, and a disallowed row is left
    exactly as `top_candidates` left it.
    """

    def setUp(self):
        self._saved = P.PLACEMENT_CAP_DEDUP
        P.PLACEMENT_CAP_DEDUP = False

    def tearDown(self):
        P.PLACEMENT_CAP_DEDUP = self._saved

    @staticmethod
    def _matrix():
        return [[0.9, 0.8, 0.7, 0.6, 0.5]]

    def test_off_is_exactly_top_candidates(self):
        m = self._matrix()
        nb = [1, 2, 3, 4, 5]
        keys = [("a",), ("a",), ("b",), ("b",), ("c",)]
        self.assertEqual(PM.top_candidates_dedup(m, nb, 3, keys),
                         PM.top_candidates(m, nb, 3))

    def test_on_it_spends_the_head_on_distinct_keys(self):
        P.PLACEMENT_CAP_DEDUP = True
        m = self._matrix()
        nb = [1, 2, 3, 4, 5]
        keys = [("a",), ("a",), ("b",), ("b",), ("c",)]
        rows = PM.top_candidates_dedup(m, nb, 3, keys)
        self.assertEqual([t[1] for t in rows[0]], [1, 3, 5],
                         "the head must be one candidate per distinct key, best first")

    def test_it_still_fills_the_cap_when_keys_run_out(self):
        P.PLACEMENT_CAP_DEDUP = True
        m = self._matrix()
        nb = [1, 2, 3, 4, 5]
        keys = [("a",), ("a",), ("a",), ("a",), ("a",)]
        rows = PM.top_candidates_dedup(m, nb, 4, keys)
        self.assertEqual(len(rows[0]), 4, "a single-key row must still fill its cap")
        self.assertEqual([t[1] for t in rows[0]], [1, 2, 3, 4])
