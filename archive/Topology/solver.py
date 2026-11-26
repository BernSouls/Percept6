"""Cached pair solver for matcher tuning.

One call to :func:`solve_direction` runs the real global-consistency alignment
(``matching.alignment.global_consistency_alignment``) with a SUPPLIED node
similarity matrix, so the expensive signature comparisons are computed once
per graph pair and reused by every alignment configuration.  The returned
:class:`Facts` object stores the correspondence and the per-edge facts
(target edge present, graded/presence credit, checkable counts), so the score
formula can be re-evaluated for every (edge weight, gamma, denominator,
isolated-handling) cell without re-matching.

Nothing here is a re-implementation of the matcher: the mapping comes from the
tree's own aligner, the node similarity from ``memory.cores._sig_similarity``
(through the same dispatch ``joint_mcs`` uses), and the edge credit from
``matching.joint_mcs.edge_pair_credit``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import guard

guard.activate()

import parameters as P                                    # noqa: E402
from core.schema import edge_categorical_axes, edge_graded_axes  # noqa: E402
from core.types import UNDIRECTED_EDGE_TYPES             # noqa: E402
from matching.alignment import global_consistency_alignment  # noqa: E402
from matching.joint_mcs import edge_pair_credit          # noqa: E402
from matching.structural_similarity import _lookup_edge_params, _pair_edge_sim_typed  # noqa: E402


def _edge_key(e):
    a, b, t = e
    return (min(a, b), max(a, b), getattr(t, "value", str(t)), a, b)


def _edge_sets(edges: Iterable):
    und, direct = set(), set()
    for s, d, et in edges:
        if et in UNDIRECTED_EDGE_TYPES:
            und.add((min(s, d), max(s, d), et))
        else:
            direct.add((s, d, et))
    return und, direct


def _target(et, u, v):
    return (min(u, v), max(u, v), et) if et in UNDIRECTED_EDGE_TYPES else (u, v, et)


def _sg(g):
    """The Subgraph behind a stored Graph wrapper, or the value itself."""
    return getattr(g, "subgraph", g)


def node_sim_matrix(a, b, sigs_a, sigs_b,
                    isolated_weight: Optional[float] = None) -> List[List[float]]:
    """[i][j] over the SORTED node lists, exactly the matcher's node_sim.

    With `isolated_weight` the matcher's own down-weight is applied HERE --
    inside the node similarity -- which is where PRODUCTION applies it
    (``matching/joint_mcs.py::_directional_joint_mcs``: a node with no
    incident relation earns ``MATCH_ISOLATED_NODE_WEIGHT`` of its signature
    similarity).  Applying it at SCORE time instead (the historical offline
    behaviour) changed the correspondence as well as the number: production
    mapped 229 nodes against the offline 230 even at w=1.0, because the
    down-weight moves WHO is paired before the alignment ever runs.
    """
    from memory.cores import _sig_similarity
    a, b = _sg(a), _sg(b)
    na, nb = sorted(a.nodes), sorted(b.nodes)
    deg_a: Dict[int, int] = {}
    for s, d, _t in a.edges:
        deg_a[s] = deg_a.get(s, 0) + 1
        deg_a[d] = deg_a.get(d, 0) + 1
    deg_b: Dict[int, int] = {}
    for s, d, _t in b.edges:
        deg_b[s] = deg_b.get(s, 0) + 1
        deg_b[d] = deg_b.get(d, 0) + 1
    cache: Dict[Tuple[int, int], float] = {}

    def sim(u, v):
        key = (u, v)
        hit = cache.get(key)
        if hit is not None:
            return hit
        x, y = sigs_a.get(u), sigs_b.get(v)
        if x is None or y is None or x[0] != y[0]:
            got = 0.0
        else:
            got = _sig_similarity(x[1], y[1])
            if (isolated_weight is not None and isolated_weight < 1.0
                    and (deg_a.get(u, 0) == 0 or deg_b.get(v, 0) == 0)):
                got *= isolated_weight
        cache[key] = got
        return got

    return [[sim(u, v) for v in nb] for u in na]


def solve_direction(a, b, sigs_a, sigs_b, sim, *, beta: float,
                    unmatched_cost: float, unmapped_credit: float,
                    rounds: int = 4):
    """One directional alignment with the supplied similarity matrix."""
    P.MATCH_ALIGN_UNMAPPED_CREDIT = float(unmapped_credit)
    a, b = _sg(a), _sg(b)
    na, nb = sorted(a.nodes), sorted(b.nodes)
    edge_sim = lambda x, y, u, v, et: edge_pair_credit(a, b, x, y, u, v, et)
    return global_consistency_alignment(
        na, nb, lambda u, v: 0.0, edges_a=a.edges, edges_b=b.edges,
        unmatched_cost=float(unmatched_cost),
        systematicity_weight=float(beta),
        # beta == 0 makes the refinement weight identical to the seed matrix;
        # the fixed point cannot move, so the rounds are skipped (the final
        # quality also uses beta == 0).
        rounds=int(rounds) if beta > 0.0 else 1,
        max_pair_evals=P.MATCH_ALIGN_MAX_PAIR_EVALS,
        sim=sim, budget_refinement_only=True, edge_similarity=edge_sim,
        # [NEEDS #4 / transfer equivalence] The SAME occlusion-credit geometry
        # production passes (matching/joint_mcs.py::_directional_joint_mcs),
        # so the offline alignment sees the same candidate values, not a
        # geometry-free subworld -- otherwise the two arms diverge in the
        # mapping itself (229 vs 228 nodes), which no score-formula fix can
        # repair.
        anchors_a=getattr(a, "anchor_of_node", None),
        anchors_b=getattr(b, "anchor_of_node", None),
        junctions_a=getattr(a, "t_junctions", None) or (),
        junctions_b=getattr(b, "t_junctions", None) or (),
    )


def derived_unmatched_cost(sim: Sequence[Sequence[float]]) -> float:
    positive = [s for row in sim for s in row if s > 0.0]
    return (math.fsum(positive) / len(positive)) if positive else 0.0


@dataclass
class Facts:
    na: int
    nb: int
    ea: int
    eb: int
    node_rows: List[Tuple[int, int, float, bool]]     # u, v, sim, isolated
    edge_rows_a: List[Tuple[bool, bool, float, bool, bool]]
    # both_mapped, present, credit, graded, presence_only
    edge_rows_b: List[Tuple[bool, bool, float, bool, bool]]
    mapping: Dict[int, int]

    def checkable(self):
        ca = sum(1 for both, *_ in self.edge_rows_a if both)
        cb = sum(1 for both, *_ in self.edge_rows_b if both)
        return ca, cb

    def presence_checkable(self):
        # [NEEDS #4] the presence-only relations (no measured axes) among the
        # checkable ones: production's denominator subtracts these in the
        # checkable / per_side arms (joint_mcs.py: `min(checkable_a -
        # presence_checkable_a, ...)`), so the advantage belongs to the
        # numerator-crediting side.  They leave the mass on both paths.
        pa = sum(1 for r in self.edge_rows_a if r[0] and r[1] and not r[3])
        pb = sum(1 for r in self.edge_rows_b if r[0] and r[1] and not r[3])
        return pa, pb

    def credits(self, presence_counts: bool = True):
        ca = math.fsum(r[2] for r in self.edge_rows_a
                       if r[0] and r[1] and (presence_counts or not r[4]))
        cb = math.fsum(r[2] for r in self.edge_rows_b
                       if r[0] and r[1] and (presence_counts or not r[4]))
        pa = sum(1 for r in self.edge_rows_a if r[4])
        pb = sum(1 for r in self.edge_rows_b if r[4])
        return ca, cb, pa, pb


def facts(a, b, sigs_a, sigs_b, sim, alignment) -> Facts:
    """Extract everything the score formula needs from one directional solve."""
    a, b = _sg(a), _sg(b)
    na, nb = sorted(a.nodes), sorted(b.nodes)
    mapping = dict(alignment.mapping)
    rev = {v: u for u, v in mapping.items()}
    a_und, a_dir = _edge_sets(a.edges)
    b_und, b_dir = _edge_sets(b.edges)
    deg_a, deg_b = {}, {}
    for s, d, _t in a.edges:
        deg_a[s] = deg_a.get(s, 0) + 1
        deg_a[d] = deg_a.get(d, 0) + 1
    for s, d, _t in b.edges:
        deg_b[s] = deg_b.get(s, 0) + 1
        deg_b[d] = deg_b.get(d, 0) + 1
    idx_a = {n: i for i, n in enumerate(na)}
    idx_b = {n: i for i, n in enumerate(nb)}
    node_rows = [(u, v, float(sim[idx_a[u]][idx_b[v]]),
                  deg_a.get(u, 0) == 0 or deg_b.get(v, 0) == 0)
                 for u, v in sorted(mapping.items())]

    def row(x, y, et, u, v, target_set, p_side, q_side, x_first):
        both = u is not None and v is not None
        if not both:
            return (False, False, 0.0, False, False)
        present = _target(et, u, v) in target_set
        if not present:
            return (True, False, 0.0, False, False)
        p = _lookup_edge_params(p_side.edge_params, x, y, et)
        q = _lookup_edge_params(q_side.edge_params, u, v, et)
        graded = edge_graded_axes(p, q) > 0
        if graded:
            credit = edge_pair_credit(p_side, q_side, x, y, u, v, et)
            return (True, True, float(credit), True, False)
        raw = (1.0 if p is None or q is None
               else _pair_edge_sim_typed(x, y, et, p, u, v, et, q, False))
        return (True, True, float(raw), False, True)

    rows_a = []
    for x, y, et in sorted(a.edges, key=_edge_key):
        u, v = mapping.get(x), mapping.get(y)
        rows_a.append(row(x, y, et, u, v, b_und if et in UNDIRECTED_EDGE_TYPES else b_dir,
                          a, b, True))
    rows_b = []
    for x, y, et in sorted(b.edges, key=_edge_key):
        u, v = rev.get(x), rev.get(y)
        rows_b.append(row(x, y, et, u, v, a_und if et in UNDIRECTED_EDGE_TYPES else a_dir,
                          b, a, False))
    return Facts(len(na), len(nb), len(a.edges), len(b.edges),
                 node_rows, rows_a, rows_b, mapping)


def score(f: Facts, *, w: float, gamma: float, denom_mode: str,
          isolated: str, presence_counts: bool = True,
          formula: str = "pooled", delta: float = 0.0):
    """One score cell, from cached facts only.

    ``formula="pooled"`` is the historical additive form
    ``2*(node_credit + w*edge_credit) / (na+nb+edge_mass)``.  In that form
    node credit runs several times the edge credit (median ratio about 5 in
    the 96-pair dev sample), so w cannot make the edges decisive; it is kept as a labelled arm.

    ``formula="normalized"`` puts the two terms on their own scales:
    ``node_term = node_credit/(na+nb)``, ``edge_term = edge_credit/edge_mass``
    and combines ``2*((1-w)*node_term*node_cov**delta +
    w*edge_term*edge_cov**gamma)`` with ``node_cov = 2*matched/(na+nb)``
    and ``edge_cov = preserved/checkable``.  There w is a real blend weight.
    """
    iso_w = {"keep": 1.0, "downweight": 0.5, "exclude": 0.0}[isolated]
    node_credit = math.fsum(s * (iso_w if iso else 1.0)
                            for _u, _v, s, iso in f.node_rows)
    ca, cb, pa, pb = f.credits(presence_counts)
    edge_credit = ca + cb
    checkable_a, checkable_b = f.checkable()
    checkable = checkable_a + checkable_b
    preserved = (sum(1 for r in f.edge_rows_a if r[0] and r[1])
                 + sum(1 for r in f.edge_rows_b if r[0] and r[1]))
    if formula == "normalized":
        node_mass = float(f.na + f.nb)
        node_term = (node_credit / node_mass) if node_mass else 0.0
        edge_mass = float(checkable)
        edge_term = (edge_credit / edge_mass) if edge_mass else 0.0
        node_cov = (2.0 * len(f.node_rows) / node_mass) if node_mass else 0.0
        edge_cov = (preserved / float(checkable)) if checkable else 0.0
        node_component = node_term * (node_cov ** delta)
        edge_component = edge_term * (edge_cov ** gamma)
        total = 2.0 * ((1.0 - w) * node_component + w * edge_component)
        return total, {"node_credit": node_credit, "edge_credit": edge_credit,
                       "edge_mass": edge_mass, "denom": node_mass + edge_mass,
                       "node_term": node_term, "edge_term": edge_term,
                       "node_cov": node_cov, "edge_cov": edge_cov,
                       "node_component": node_component,
                       "edge_component": edge_component}
    if denom_mode == "all":
        edge_mass = float(max(0, f.ea + f.eb - (2 * max(pa, pb) if not presence_counts else 0)))
    elif denom_mode == "checkable":
        pac, pbc = f.presence_checkable()
        edge_mass = float(2 * min(checkable_a - pac, checkable_b - pbc))
    elif denom_mode == "per_side":
        pac, pbc = f.presence_checkable()
        edge_mass = float((checkable_a - pac) + (checkable_b - pbc))
    else:
        raise ValueError(denom_mode)
    edge_mass = float(max(0.0, edge_mass))
    if gamma != 1.0:
        edge_mass = float(edge_mass ** gamma) if edge_mass > 0.0 else 0.0
    denom = float(f.na + f.nb) + edge_mass
    total = 2.0 * (node_credit + w * edge_credit) / denom if denom else 0.0
    return total, {"node_credit": node_credit, "edge_credit": edge_credit,
                   "edge_mass": edge_mass, "denom": denom,
                   "node_term": (node_credit / float(f.na + f.nb))
                   if f.na + f.nb else 0.0,
                   "edge_term": (edge_credit / edge_mass) if edge_mass else 0.0,
                   "node_cov": (2.0 * len(f.node_rows) / float(f.na + f.nb))
                   if f.na + f.nb else 0.0,
                   "edge_cov": (preserved / float(checkable)) if checkable else 0.0}
