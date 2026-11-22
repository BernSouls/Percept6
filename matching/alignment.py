"""matching/alignment.py -- global-consistency graph alignment.

The correspondence layer of the fuzzy graded matcher (Phase 5).

Given two node sets, a node-similarity function (the *population-overlap*
similarity produced by ``TypedSignature.similarity``) and the typed edge sets,
this module produces a bijective correspondence that maximises not just local
node similarity but GLOBAL CONSISTENCY: neighbouring nodes that are connected in
one graph are preferred to map onto neighbouring connected nodes in the other
(systematicity). It is seeded from the existing approximate Hungarian
(``_max_weight_bipartite_matching``) and refined by bounded fixed-point
re-assignment plus a coupled swap search.

Decisions (mine, stated so they can be audited):

* **THE JOINT OBJECTIVE.** A pair (u, v) is worth

      q(u, v) = sim(u, v) · (1 − β) + β · rel(u, v)

  -- the APPEARANCE term and the STRUCTURE term are ADDITIVE, so a relation
  that agrees can outbid a better-looking but unconnected candidate, and both
  terms stay inside [0, 1].  The old matrix was ``sim · (1 − β + β · rel)``,
  which MULTIPLIED the structure into the appearance: a pair whose relations
  all contradicted could never beat an unrelated pair, so the relations could
  not change WHO PAIRS -- only how much a pair already chosen was worth.

* **NO EVIDENCE IS NOT PERFECT AGREEMENT.** A node with no relations has NO
  relational support: ``rel`` is 0.0, not 1.0.  Under the old reading the
  ABSENCE of structure was worth 1.82x the PRESENCE of a tenth of it, and
  58 % of the frame nodes are degree-0, so this was the common case.

* **THE CANDIDATE IS EVALUATED IN THE WORLD IT INDUCES.** A pair is scored
  against the mapping it would produce -- with the partner it displaces
  released -- not against the previous round's frozen map.  The frozen map
  made a wrong pair look *better* than the right one (the right pair's
  partner still pointed at the right pair's own image, so the relation it
  should have preserved was looked for as a self-loop).

* **COUPLED MOVES.** The assignment is one-to-one on NODES; the relations
  are decided with it.  After every sweep the mapping is offered the coupled
  swap of every relation's endpoints, accepted only on a strict increase of
  the joint objective.  A swap is the only move that can change WHO PAIRS
  for a pair of coupled nodes.

* **MONOTONIC, AND THE BEST IS RETURNED.** A round that does not improve the
  joint objective is not accepted, a repeated mapping ends the search, and
  the best mapping seen is the one returned.

* **EXPLICIT UNMATCHED COST.** A pair is only ever matched when its NET
  SURPLUS over the price of leaving both nodes unmatched is positive
  (``max(0, q − unmatched_cost)``).  The assignment therefore maximises what
  is actually gained, not the gross quality: two mediocre pairs can no
  longer outbid one good one.

* **SYMMETRIC COVERAGE.** Coverage is the Dice overlap ``2|M| / (|A|+|B|)``,
  so a mapping that explains both sides is rewarded and a mapping that covers
  one side by leaving the other entirely unmatched is not. Forward/backward
  edge coverage are reported separately and CONJUNCTED (min), never averaged.

* **SYSTEMATICITY.** The conjunctive (min) fraction of typed edges preserved in
  each direction. Min, not mean: a clean A side must not compensate a
  contradicted B side.

* **DETERMINISM.** Every iteration sorts its inputs; ties break on
  ``(weight desc, row asc, col asc)``; the returned mapping is a sorted dict.

Pure, stateless, budgeted, no randomness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from core.types import UNDIRECTED_EDGE_TYPES
from matching.structural_similarity import (
    _max_weight_bipartite_matching,
    _t_junction_near_nodes,
    _tj_spatial_index,
)
import parameters as P

#: (u, v) -> similarity in [0, 1]. 0.0 means "not comparable".
SimFn = Callable[[int, int], float]
#: node -> list of (neighbour, edge_type, outward). `outward` True when the
#: node is the edge's source.
Incidence = Dict[int, List[Tuple[int, object, bool]]]
Edge = Tuple[int, int, object]
#: the tolerance of an objective comparison.  Below it a move is not a move.
_EPS = 1e-12
#: how many times the final pruning may re-read the relations it pruned by.
_PRUNE_PASSES = 8


@dataclass(frozen=True)
class Alignment:
    """Result of a global-consistency alignment.

    mapping      : A-node -> B-node, sorted by A-node for determinism.
    quality      : A-node -> final pair quality in [0, 1]
                   (= sim·(1−β) + β·rel).
    systematicity: conjunctive typed-edge preservation, in [0, 1].
    fwd_edge_cov : fraction of A edges preserved in B, in [0, 1].
    bwd_edge_cov : fraction of B edges preserved in A, in [0, 1].
    node_coverage: symmetric Dice 2|M| / (|A|+|B|), in [0, 1].
    unmatched_a  : sorted A nodes left unmatched.
    unmatched_b  : sorted B nodes left unmatched.
    rounds_used  : refinement sweeps actually performed.
    budget_exhausted: True when the pair-evaluation budget stopped refinement.
    """
    mapping: Dict[int, int]
    quality: Dict[int, float]
    systematicity: float
    fwd_edge_cov: float
    bwd_edge_cov: float
    node_coverage: float
    unmatched_a: Tuple[int, ...] = ()
    unmatched_b: Tuple[int, ...] = ()
    rounds_used: int = 0
    budget_exhausted: bool = False

    def edge_coverage(self) -> float:
        """The CONJUNCTIVE edge coverage: the MIN of the two directions.

        A mean let a clean A side compensate a contradicted B side, and the
        MEAN of two 1.0s is 1.0 -- so an alignment that preserved NOTHING
        read as FULL coverage.  The codebase's own convention (the
        systematicity term, and the docstrings here and in
        ``matching/fuzzy.py``) is the conjunction: min, not mean.
        """
        return min(self.fwd_edge_cov, self.bwd_edge_cov)


class _Incidence(dict):
    """The incidence map, plus the geometry the occlusion credit needs.

    ``matching/structural_similarity.py`` reads an unmapped partner as an
    OCCLUSION (a T-junction near either end) and pays the partial credit.
    The aligner had no access to the geometry, so it paid nothing and an
    occluded relation read as a contradiction.  The incidence therefore
    carries the anchors and the junctions of its side.
    """

    __slots__ = ("anchors", "junctions", "tj_index", "classes")

    def __init__(self, *args, anchors=None, junctions=None, tj_index=None,
                 classes=None):
        super().__init__(*args)
        self.anchors = anchors or {}
        self.junctions = junctions or ()
        self.tj_index = tj_index
        #: endpoint pair -> the schema classes of the relations that span it
        #: ([A9] the compatibility layer)
        self.classes = classes or {}


def _edge_sort_key(t: Edge) -> Tuple:
    """The canonical order of a typed edge: endpoints, then the edge type.

    The edge type is ordered by ``(is_enum, value, name)`` so that an
    ``EdgeType`` enum member and a U30 CATEGORY STRING have a TOTAL order.
    The old key was ``getattr(t[2], "value", 0)``, which collapsed every
    category string to 0 and so left the order of two same-endpoint edges
    of different categories to the sort's own stability.
    """
    et = t[2]
    value = getattr(et, "value", None)
    if value is not None:
        return (min(t[0], t[1]), max(t[0], t[1]), 0, float(value),
                str(getattr(et, "name", "")), t[0], t[1])
    return (min(t[0], t[1]), max(t[0], t[1]), 1, 0.0, str(et), t[0], t[1])


def build_incidence(
    edges: Iterable[Edge],
    anchors: Optional[Dict[int, Sequence[float]]] = None,
    junctions: Sequence[Any] = (),
) -> Incidence:
    """node -> [(neighbour, et, outward)] from a typed edge iterable.

    [A10] The incidence is CANONICAL: an undirected relation appears ONCE
    per endpoint pair -- its two directions are one relation -- and a
    self-loop appears ONCE at its node.  The old code appended a self-loop
    twice and appended BOTH orientations of an undirected relation, so a
    node was charged twice for one relation while ``_edge_indexes``
    (which canonicalises into a set) could match it only once: the
    denominator of the agreement was inflated relative to the number of
    relations that could be matched at all.

    An undirected relation is oriented canonically (low id -> high id) on
    BOTH sides, so the two graphs are read the same way and the edge
    similarity of an undirected pair never sees a reversed endpoint order.
    """
    radius = float(getattr(P, "T_JUNCTION_RADIUS", 0.0) or 0.0)
    inc = _Incidence(
        anchors=anchors,
        junctions=tuple(junctions or ()),
        tj_index=_tj_spatial_index(tuple(junctions or ()), radius),
        classes=_pair_class_index(edges),
    )
    seen_undirected: Set[Tuple[int, int, object]] = set()
    for e in sorted(edges, key=_edge_sort_key):
        s, d, et = e
        if s == d:
            # a self-loop is ONE relation
            inc.setdefault(s, []).append((d, et, True))
            continue
        if et in UNDIRECTED_EDGE_TYPES:
            lo, hi = (s, d) if s < d else (d, s)
            if (lo, hi, et) in seen_undirected:
                continue
            seen_undirected.add((lo, hi, et))
            inc.setdefault(lo, []).append((hi, et, True))
            inc.setdefault(hi, []).append((lo, et, False))
            continue
        inc.setdefault(s, []).append((d, et, True))
        inc.setdefault(d, []).append((s, et, False))
    return inc


def _edge_indexes(edges: Iterable[Edge]) -> Tuple[Set[Edge], Set[Edge]]:
    directed: Set[Edge] = set()
    undirected: Set[Edge] = set()
    for s, d, et in edges:
        if et in UNDIRECTED_EDGE_TYPES:
            undirected.add((min(s, d), max(s, d), et))
        else:
            directed.add((s, d, et))
    return directed, undirected


#: The schema classes each edge type may be matched by.  The lookup is a
#: pure function of the type, so the cache cannot change an answer.
_COMPAT_CLASSES: Dict[str, frozenset] = {}


def _edge_class_name(et: object) -> str:
    """The schema class of an edge type: ``EDGE_`` + the type's name."""
    name = getattr(et, "name", None) or str(et)
    return name if name.startswith("EDGE_") else "EDGE_" + name


def _compatible_classes(et: object) -> frozenset:
    """The schema classes a relation of this type may be matched by.

    [A9] THE COMPATIBILITY LAYER WAS UNREACHABLE FROM THE LIVE ALIGNER.
    The aligner compared the edge types for EQUALITY, so an
    ORIENTATION_CONTINUITY relation could never be preserved by a
    CONTOUR_CONTINUITY one -- the very relations the polymorphic machinery
    exists for, and the reason ``core.schema.get_compatible_types`` exists
    at all.  An unknown class is compatible only with itself, so the
    behaviour is unchanged for every type outside a declared family.
    """
    cls = _edge_class_name(et)
    got = _COMPAT_CLASSES.get(cls)
    if got is None:
        from core.schema import get_compatible_types
        got = frozenset(get_compatible_types(cls))
        _COMPAT_CLASSES[cls] = got
    return got


def _pair_class_index(edges: Iterable[Edge]) -> Dict[Tuple[int, int], Set[str]]:
    """Endpoint pair -> the schema classes of the relations that span it.

    The key is the canonical pair for an undirected relation and the
    directed pair otherwise, so a DIRECTED relation is only ever matched by
    a relation pointing the same way.
    """
    out: Dict[Tuple[int, int], Set[str]] = {}
    for (s, d, et) in edges:
        key = (min(s, d), max(s, d)) if et in UNDIRECTED_EDGE_TYPES else (s, d)
        out.setdefault(key, set()).add(_edge_class_name(et))
    return out


def _preserved_edge(
    a_src: int, a_dst: int, et: object,
    mapping: Dict[int, int],
    b_dir: Set[Edge], b_und: Set[Edge],
    b_classes: Optional[Dict[Tuple[int, int], Set[str]]] = None,
) -> bool:
    b_src = mapping.get(a_src)
    b_dst = mapping.get(a_dst)
    if b_src is None or b_dst is None:
        return False
    if et in UNDIRECTED_EDGE_TYPES:
        lo, hi = (b_src, b_dst) if b_src < b_dst else (b_dst, b_src)
        if (lo, hi, et) in b_und:
            return True
        # [A9] a relation of a COMPATIBLE class preserves it too
        if b_classes:
            return bool(_compatible_classes(et) & b_classes.get((lo, hi), frozenset()))
        return False
    if (b_src, b_dst, et) in b_dir:
        return True
    if b_classes:
        return bool(_compatible_classes(et) & b_classes.get((b_src, b_dst),
                                                            frozenset()))
    return False


def systematicity(
    mapping: Dict[int, int],
    edges_a: Sequence[Edge],
    edges_b: Sequence[Edge],
) -> Tuple[float, float, float]:
    """Return (conjunctive, forward, backward) typed-edge preservation.

    conjunctive = min(forward, backward), with an edgeless side read as 1.0
    (vacuous), matching the codebase's conjunction convention.

    [A10] The denominators are the CANONICAL relation counts (the same
    sets the edge index matches against), not the raw edge-list lengths: an
    undirected relation stored under both orientations is ONE relation, and
    a raw length counted it twice while only one match was possible.
    """
    a_dir, a_und = _edge_indexes(edges_a)
    b_dir, b_und = _edge_indexes(edges_b)
    # [A9] the schema classes each side carries at every endpoint pair, so a
    # relation of a COMPATIBLE class can preserve a relation.
    a_classes = _pair_class_index(edges_a)
    b_classes = _pair_class_index(edges_b)
    rev = {v: u for u, v in mapping.items()}

    n_a = len(a_dir) + len(a_und)
    n_b = len(b_dir) + len(b_und)
    both_empty = (n_a == 0 and n_b == 0)
    if not both_empty and (n_a == 0 or n_b == 0):
        return 0.0, 0.0, 0.0
    if both_empty:
        return 1.0, 1.0, 1.0

    fwd = 0
    for (s, d, et) in a_dir:
        if _preserved_edge(s, d, et, mapping, b_dir, b_und, b_classes):
            fwd += 1
    for (lo, hi, et) in a_und:
        if _preserved_edge(lo, hi, et, mapping, b_dir, b_und, b_classes):
            fwd += 1
    fwd = fwd / float(n_a)

    bwd = 0
    for (s, d, et) in b_dir:
        if _preserved_edge(s, d, et, rev, a_dir, a_und, a_classes):
            bwd += 1
    for (lo, hi, et) in b_und:
        if _preserved_edge(lo, hi, et, rev, a_dir, a_und, a_classes):
            bwd += 1
    bwd = bwd / float(n_b)
    return min(fwd, bwd), fwd, bwd


def _relational_agreement(
    u: int,
    v: int,
    mapping: Dict[int, int],
    inc_a: Incidence,
    inc_b: Incidence,
    b_dir: Set[Edge],
    b_und: Set[Edge],
    a_dir: Set[Edge],
    a_und: Set[Edge],
    rev: Dict[int, int],
    edge_similarity=None,
    *,
    checkable: bool = False,
    candidate: Optional[Tuple[int, int]] = None,
    edge_weight_fn=None,
) -> float:
    """Conjunctive mapped-context preservation for pair (u, v).

    Fraction of u's typed relations whose images under `mapping` exist at v,
    min the same from v's side.

    [A1] A node with NO relations has NO relational support: that side reads
    0.0, not 1.0.  The old reading made the ABSENCE of structure worth more
    than the PRESENCE of it: with ``w = sim·(1−β+β·rel)`` at β=0.5 an
    isolated pair scored ``sim`` while a pair preserving a tenth of its
    relations scored ``0.55·sim`` -- a 1.82x reward for having no structure
    at all, on a corpus where 58 % of the frame nodes are degree-0.

    [A4] `checkable` selects the DENOMINATOR.  With it OFF (the default) the
    denominator is EVERY relation of the side, so an unmapped neighbour
    costs the side it is on -- the symmetry the pins fix.  With it ON the
    denominator is only the CHECKABLE mass: the relations whose partner IS
    mapped (plus the relations whose partner is unmapped but OCCLUDED, which
    are explained rather than unknown).  A pair is then scored on the
    evidence it can actually be checked against instead of being diluted by
    relations that are not mapped yet -- which is what left the early
    rounds with no gradient to climb.

    [A5] An unmapped partner is not automatically a contradiction: when a
    T-junction lies near either end of the relation, the relation is
    OCCLUDED and earns the partial occlusion credit, exactly as
    ``matching/structural_similarity.py`` reads it.

    [A2] `candidate` is the pair being EVALUATED.  The relation evidence is
    read in the world that pair induces: the candidate's own image is the
    candidate's partner, and the node the candidate DISPLACES is released
    (its own image is then unknown, so it earns nothing).  Reading the
    evidence against the previous round's frozen map made a wrong pair look
    better than the right one: the right pair's partner still pointed at the
    right pair's own image, so the relation it should have preserved was
    looked for as a self-loop and came back absent.
    """
    unmapped_credit = float(getattr(P, "MATCH_ALIGN_UNMAPPED_CREDIT", 0.0))
    occlusion_credit = float(getattr(P, "PARTIAL_OCCLUSION_CREDIT", 0.0))
    # [PART 4] weight-aware counting: when enabled, each matched edge's
    # contribution is scaled by its graded weight (edge_weight_fn), and the
    # denominator accumulates the weight instead of a unit count.  Old graphs
    # without weights default to 1.0, so this reduces to the old behaviour.
    use_weight = bool(getattr(P, "EDGE_MATCH_USE_CONFIDENCE", False)) and edge_weight_fn is not None

    def _ew(a1, a2, et):
        return float(edge_weight_fn(a1, a2, et)) if use_weight else 1.0

    anchors_a = getattr(inc_a, "anchors", None) or {}
    anchors_b = getattr(inc_b, "anchors", None) or {}
    junc_a = getattr(inc_a, "junctions", None) or ()
    junc_b = getattr(inc_b, "junctions", None) or ()
    tj_a = getattr(inc_a, "tj_index", None)
    tj_b = getattr(inc_b, "tj_index", None)
    # [A9] the schema classes each side carries at every endpoint pair
    a_classes = getattr(inc_a, "classes", None) or {}
    b_classes = getattr(inc_b, "classes", None) or {}

    cand_u = cand_v = None
    if candidate is not None:
        cand_u, cand_v = candidate

    edges_u = inc_a.get(u)
    if edges_u:
        terms = []
        denominator = 0
        for w, et, outward in edges_u:
            mw = mapping.get(w)
            if cand_u is not None:
                if w == cand_u:
                    # the candidate's own relation: its partner IS the
                    # candidate's image
                    mw = cand_v
                elif mw == cand_v:
                    # w is the node the candidate DISPLACES: under the
                    # candidate its image is unknown
                    mw = None
            if mw is None:
                # [A5] an unmapped partner is not a contradiction when the
                # relation is occluded by a T-junction.
                if _t_junction_near_nodes(anchors_a.get(u), anchors_a.get(w),
                                          junc_a, index=tj_a):
                    terms.append(occlusion_credit)
                    denominator += 1
                else:
                    terms.append(unmapped_credit)
                    if not checkable:
                        denominator += _ew(u, w, et)
                continue
            denominator += _ew(u, w, et)
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (v, mw) if v < mw else (mw, v)
                ok = (lo, hi, et) in b_und
                if not ok:
                    # [A9] a relation of a COMPATIBLE class preserves it
                    ok = bool(_compatible_classes(et)
                              & b_classes.get((lo, hi), frozenset()))
            else:
                want = (v, mw, et) if outward else (mw, v, et)
                ok = want in b_dir
                if not ok:
                    key = (v, mw) if outward else (mw, v)
                    ok = bool(_compatible_classes(et) & b_classes.get(key, frozenset()))
            if ok:
                if et in UNDIRECTED_EDGE_TYPES:
                    # the endpoints are canonical on BOTH sides, so the edge
                    # similarity never sees a reversed order
                    a1, a2 = (u, w) if u < w else (w, u)
                    b1, b2 = (v, mw) if v < mw else (mw, v)
                else:
                    a1, a2 = (u, w) if outward else (w, u)
                    b1, b2 = (v, mw) if outward else (mw, v)
                if edge_similarity is None:
                    terms.append(_ew(a1, a2, et))
                else:
                    got = edge_similarity(a1, a2, b1, b2, et)
                    if got is None:
                        # [B7] INCOMPARABLE: the two relations have no common
                        # ontology, so they are NOT COMPARED -- no credit and
                        # no charge (the "score it 0" reading made an absent
                        # counterpart look like evidence AGAINST the pair).
                        denominator -= _ew(a1, a2, et)
                        continue
                    terms.append(_ew(a1, a2, et) * got)
        agree_u = (math.fsum(terms) / denominator) if denominator else 0.0
    else:
        # [A1] no relations is NO relational support, not perfect agreement.
        agree_u = 0.0

    edges_v = inc_b.get(v)
    if edges_v:
        terms = []
        denominator = 0
        for x, et, outward in edges_v:
            ux = rev.get(x)
            if cand_u is not None:
                if x == cand_v:
                    ux = cand_u
                elif ux == cand_u:
                    ux = None
            if ux is None:
                if _t_junction_near_nodes(anchors_b.get(v), anchors_b.get(x),
                                          junc_b, index=tj_b):
                    terms.append(occlusion_credit)
                    denominator += _ew(v, x, et)
                else:
                    terms.append(unmapped_credit)
                    if not checkable:
                        denominator += _ew(v, x, et)
                continue
            denominator += _ew(v, x, et)
            if et in UNDIRECTED_EDGE_TYPES:
                lo, hi = (u, ux) if u < ux else (ux, u)
                ok = (lo, hi, et) in a_und
                if not ok:
                    # [A9] a relation of a COMPATIBLE class preserves it
                    ok = bool(_compatible_classes(et)
                              & a_classes.get((lo, hi), frozenset()))
            else:
                # the image of the v-side edge v->x (outward=True means v is
                # the source) is u->ux.
                ok = ((u, ux, et) if outward else (ux, u, et)) in a_dir
                if not ok:
                    key = (u, ux) if outward else (ux, u)
                    ok = bool(_compatible_classes(et) & a_classes.get(key, frozenset()))
            if ok:
                if et in UNDIRECTED_EDGE_TYPES:
                    a1, a2 = (u, ux) if u < ux else (ux, u)
                    b1, b2 = (v, x) if v < x else (x, v)
                else:
                    a1, a2 = (u, ux) if outward else (ux, u)
                    b1, b2 = (v, x) if outward else (x, v)
                if edge_similarity is None:
                    terms.append(_ew(a1, a2, et))
                else:
                    got = edge_similarity(a1, a2, b1, b2, et)
                    if got is None:
                        # [B7] INCOMPARABLE -- not compared at all (no credit,
                        # no charge); see the u side.
                        denominator -= _ew(a1, a2, et)
                        continue
                    terms.append(_ew(a1, a2, et) * got)
        agree_v = (math.fsum(terms) / denominator) if denominator else 0.0
    else:
        agree_v = 0.0
    return min(agree_u, agree_v)


def _assign(sim: List[List[float]],
            nodes_a: List[int],
            nodes_b: List[int],
            rel_weight: Optional[Callable[[int, int], float]] = None,
            beta: float = 0.0,
            skip_zero: bool = False,
            unmatched_cost: float = 0.0) -> Dict[int, int]:
    """One bipartite assignment.

    When `rel_weight` is given the matrix is

        q(i, j) = sim(i, j)·(1 − β) + β·rel(i, j)

    -- the APPEARANCE and the STRUCTURE term are ADDITIVE, so a relation
    that agrees can outbid a better-looking candidate whose context
    contradicts, and both terms stay inside [0, 1].  The old matrix was
    ``sim·(1 − β + β·rel)``, which multiplied the structure into the
    appearance: the relations could only discount a pair, never outbid
    one, and an isolated pair was worth MORE than a connected one.

    The weights the assignment maximises are the NET SURPLUS
    ``max(0, q − unmatched_cost)``: leaving a node unmatched is a decision
    with a PRICE, so a pair is matched only when it beats that price and
    the assignment maximises what is actually gained.  (The old code
    maximised the gross quality and filtered afterwards, so two mediocre
    pairs could outbid one good one.)

    A candidate of similarity 0 is never comparable, so its weight is 0.0
    whatever its relational agreement is.

    `skip_zero`: a candidate of similarity 0 keeps the weight 0.0 without
    its relational agreement being evaluated -- the matrix is the same and
    the evaluation is saved."""
    n_b = len(nodes_b)
    if rel_weight is None:
        q = sim
    elif skip_zero:
        q = [
            [(sim[i][j] * (1.0 - beta) + beta * rel_weight(i, j))
             if sim[i][j] > 0.0 else 0.0
             for j in range(n_b)]
            for i in range(len(nodes_a))
        ]
    else:
        q = [
            [(sim[i][j] * (1.0 - beta) + beta * rel_weight(i, j))
             if sim[i][j] > 0.0 else 0.0
             for j in range(n_b)]
            for i in range(len(nodes_a))
        ]
    if unmatched_cost > 0.0:
        w = [[max(0.0, q[i][j] - unmatched_cost) for j in range(n_b)]
             for i in range(len(nodes_a))]
    else:
        w = q
    pairs = _max_weight_bipartite_matching(w)
    m: Dict[int, int] = {}
    for i, j in pairs:
        if w[i][j] > 0.0:
            m[nodes_a[i]] = nodes_b[j]
    return m


def _incident_edge_types(edges: Sequence[Edge]) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for x, y, et in edges:
        out.setdefault(x, set()).add(et)
        out.setdefault(y, set()).add(et)
    return out


def _relational_seed_matrix(matrix: List[List[float]], nodes_a: List[int],
                            nodes_b: List[int], edges_a: Sequence[Edge],
                            edges_b: Sequence[Edge]) -> List[List[float]]:
    """Seed weights biased by incident relation-type overlap.

    The old seed is a pure node-similarity Hungarian.  This keeps the node
    term but multiplies it by (1 + 2 * overlap) so a pair that already
    agrees on WHAT relations leave each node is the first one assigned.  It
    is a seed only; the fixed point still uses the full typed relational
    agreement.

    [LEAD] The overlap is a MULTISET overlap over ``(type, direction)``:
    the old one compared the SETS of incident types, so ten relations of
    one type read the same as one, and a directed relation read the same
    whichever way it pointed.  The Dice coefficient over the multisets
    counts the multiplicity and the direction.
    """
    ta = _incident_relation_signatures(edges_a)
    tb = _incident_relation_signatures(edges_b)
    out: List[List[float]] = []
    for i, u in enumerate(nodes_a):
        row = []
        sa = ta.get(u, {})
        for j, v in enumerate(nodes_b):
            sb = tb.get(v, {})
            overlap = _dice_multiset(sa, sb)
            row.append(float(matrix[i][j]) * (1.0 + 2.0 * overlap))
        out.append(row)
    return out


def _incident_relation_signatures(edges: Sequence[Edge]) -> Dict[int, Dict[Any, int]]:
    """node -> {(relation type, direction): how many}."""
    out: Dict[int, Dict[Any, int]] = {}
    for x, y, et in edges:
        if et in UNDIRECTED_EDGE_TYPES:
            key = (et, "")
            out.setdefault(x, {})
            out.setdefault(y, {})
            out[x][key] = out[x].get(key, 0) + 1
            out[y][key] = out[y].get(key, 0) + 1
        else:
            out.setdefault(x, {})
            out.setdefault(y, {})
            ko = (et, "out")
            ki = (et, "in")
            out[x][ko] = out[x].get(ko, 0) + 1
            out[y][ki] = out[y].get(ki, 0) + 1
    return out


def _dice_multiset(a: Dict[Any, int], b: Dict[Any, int]) -> float:
    """The Dice coefficient of two count multisets, in [0, 1]."""
    if not a or not b:
        return 0.0
    total_a = sum(a.values())
    total_b = sum(b.values())
    shared = 0
    for key, n in a.items():
        m = b.get(key)
        if m:
            shared += min(n, m)
    return (2.0 * shared) / float(total_a + total_b)


def _clamped_cosine(a, b) -> float:
    """Cosine of two vectors clamped to [0, 1].  1.0 when either is absent
    (the seed is left unchanged for a node with no address code)."""
    import numpy as np
    if a is None or b is None:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (na * nb))))


def _address_seed_factor(
    nodes_a: Sequence[int],
    nodes_b: Sequence[int],
    addrs_a: Optional[Dict[int, Any]],
    addrs_b: Optional[Dict[int, Any]],
    k: Optional[int],
) -> Optional[List[List[float]]]:
    """[B2] Element-wise clamped-cosine factor over node address vectors.

    Returns a matrix [i][j] of cosines (clamped 0..1) between the level-``k``
    address vectors of node i (A) and node j (B), or None when no addresses
    are supplied.  A node with no address vector contributes a factor of 1.0
    (its seed is unchanged).  This is a SEED adjustment only: it multiplies
    the seed matrix and introduces NO pruning of candidate pairs.
    """
    if not addrs_a or not addrs_b:
        return None
    if k is None:
        keys = [kk for kk in addrs_a if isinstance(kk, int)]
        k = max(keys) if keys else None
        if k is None or k not in addrs_b:
            return None
    la = addrs_a.get(k, {})
    lb = addrs_b.get(k, {})
    return [[_clamped_cosine(la.get(u), lb.get(v)) for v in nodes_b]
            for u in nodes_a]


def global_consistency_alignment(
    nodes_a: Sequence[int],
    nodes_b: Sequence[int],
    sim_fn: SimFn,
    edges_a: Sequence[Edge] = (),
    edges_b: Sequence[Edge] = (),
    *,
    unmatched_cost: float = 0.0,
    systematicity_weight: float = 0.5,
    rounds: int = 6,
    max_pair_evals: Optional[int] = None,
    sim: Optional[Sequence[Sequence[float]]] = None,
    budget_refinement_only: bool = True,
    seed_evals_charged: int = 0,
    edge_similarity=None,
    anchors_a: Optional[Dict[int, Sequence[float]]] = None,
    anchors_b: Optional[Dict[int, Sequence[float]]] = None,
    junctions_a: Sequence[Any] = (),
    junctions_b: Sequence[Any] = (),
    addrs_a: Optional[Dict[int, Any]] = None,
    addrs_b: Optional[Dict[int, Any]] = None,
    address_k: Optional[int] = None,
    edge_weight_fn=None,
    placement_payloads_a: Optional[Dict[Any, Any]] = None,
    placement_payloads_b: Optional[Dict[Any, Any]] = None,
    placement_tolerance: Optional[float] = None,
) -> Alignment:
    """Global-consistency alignment seeded from the approximate Hungarian.

    [D2 PREREQUISITES] `max_pair_evals` is the budget of the REFINEMENT
    (P.MATCH_ALIGN_MAX_PAIR_EVALS: "pair-evaluation budget for the
    refinement"). Historically the seed's similarity matrix was charged to
    it when this function built the matrix (`sim_fn`) and not when the
    caller supplied one (`sim`), so the same inputs gave two answers by
    entry point, and on any pair with more candidates than the budget the
    `sim_fn` route returned the unrefined seed.

    [A11] `budget_refinement_only` now DEFAULTS to True, which is the
    accounting the tree's own parameter declares
    (P.MATCH_ALIGN_BUDGET_REFINEMENT_ONLY = True) and the accounting every
    live caller already used: the seed is never charged, and a candidate
    of similarity 0 is neither evaluated nor charged (its weight is 0.0
    whatever its agreement).  False is the historical accounting -- the
    `sim_fn` route charges the whole n x m matrix -- and `seed_evals_charged`
    lets a caller that supplies `sim` reproduce that charge exactly.  The
    two routes agree under the default.

    Steps:
      1. Build the node-similarity matrix (or use a supplied `sim`, indexed
         ``[i][j]`` over the SORTED node lists); prune pairs below
         `unmatched_cost`.
      2. Seed the correspondence with the approximate Hungarian on it.
      3. Refine: re-assign on ``sim·(1−β) + β·rel`` -- the relations are
         read against the world each candidate induces -- then offer the
         coupled swap of every relation's endpoints.  A round that does not
         improve the joint objective is not accepted, a repeated mapping
         ends the search, and the BEST mapping seen is returned.
      4. Keep only pairs whose final quality beats `unmatched_cost`, as a
         FIXED POINT: the relations are re-read against the mapping that
         survived the pruning, until it stops changing.

    `unmatched_cost == 0.0` recovers the classic "any positive similarity may
    match" behaviour; raising it makes unmatched an explicit, priced decision.
    """
    nodes_a = sorted(set(nodes_a))
    nodes_b = sorted(set(nodes_b))
    # [A8] the empty result's coverage slots: a graph pair that HAS edges
    # and matched NOTHING has preserved nothing, so its edge coverage is
    # 0.0, not the 1.0 the old empty result carried (which made a
    # zero-match alignment read as FULLY covered).
    both_edgeless = not edges_a and not edges_b
    empty_cov = 1.0 if both_edgeless else 0.0
    empty = Alignment({}, {}, empty_cov, empty_cov, empty_cov, 0.0,
                      tuple(nodes_a), tuple(nodes_b))
    if not nodes_a or not nodes_b:
        return empty

    # 1. similarity matrix (supplied or built), pruned by the unmatched cost.
    pair_evals = 0
    if sim is None:
        sim = [[0.0] * len(nodes_b) for _ in nodes_a]
        for i, u in enumerate(nodes_a):
            for j, v in enumerate(nodes_b):
                sim[i][j] = float(sim_fn(u, v))
                pair_evals += 1
    else:
        pair_evals = int(seed_evals_charged)
    if budget_refinement_only:
        pair_evals = 0
    matrix: List[List[float]] = [
        [(0.0 if float(sim[i][j]) < unmatched_cost else float(sim[i][j]))
         for j in range(len(nodes_b))]
        for i in range(len(nodes_a))
    ]

    # [B2] Address seed: multiply the SEED matrix element-wise by the clamped
    # cosine of the per-graph address vectors.  Nothing else -- no pruning of
    # candidate pairs (the unmatched_cost prune above is unchanged).
    addr_factor = _address_seed_factor(nodes_a, nodes_b, addrs_a, addrs_b,
                                       address_k)
    if addr_factor is not None:
        for i in range(len(nodes_a)):
            for j in range(len(nodes_b)):
                matrix[i][j] *= addr_factor[i][j]

    # [PLACEMENT / dark] Part 3's scorer as a MODIFIER on node similarity.
    #
    # sim'(u, v) = sim(u, v) * (1 + w * (support(u) - 0.5))
    #
    # with ONE pre-stated w (P.PLACEMENT_WEIGHT) -- no sweep, no fitted beta.
    # The modifier is NEUTRAL at support 0.5, bounded to [1 - w/2, 1 + w/2],
    # and a node with NO CHECKABLE EDGE keeps its similarity exactly (support
    # None => no credit and no penalty; the mission forbids rewarding mere
    # presence).  It is applied to the SEED matrix using the seed's own
    # provisional mapping, so the support is read against a correspondence
    # the aligner produced, never against truth.
    #
    # Dark by default: PLACEMENT_ENABLED is False and every argument is
    # optional, so the shipped path is untouched (a caller that passes
    # nothing gets exactly the old matrix).
    if (bool(getattr(P, "PLACEMENT_SCORER_ENABLED", False))
            and placement_payloads_a is not None
            and placement_payloads_b is not None
            and placement_tolerance):
        try:
            from matching.placement_scorer import node_support
            _prov: Dict[int, int] = {}
            for _i, _u in enumerate(nodes_a):
                _bj, _bv = -1, 0.0
                for _j, _v in enumerate(nodes_b):
                    if matrix[_i][_j] > _bv:
                        _bv, _bj = matrix[_i][_j], _j
                if _bj >= 0 and _bv > 0.0:
                    _prov[int(_u)] = int(nodes_b[_bj])
            _w = float(getattr(P, "PLACEMENT_WEIGHT", 0.25))
            _mc = float(getattr(P, "PLACEMENT_MAX_NEIGHBOUR_CREDIT", 1.0))
            for _i, _u in enumerate(nodes_a):
                _rep = node_support(int(_u), _prov, placement_payloads_a,
                                    placement_payloads_b,
                                    float(placement_tolerance),
                                    max_credit=_mc)
                if _rep is None or _rep["support"] is None:
                    continue          # uncheckable: no credit, no penalty
                _f = 1.0 + _w * (float(_rep["support"]) - 0.5)
                for _j in range(len(nodes_b)):
                    matrix[_i][_j] *= _f
        except Exception:
            # A dark feature may never break the aligner.
            pass

    if not any(any(row) for row in matrix):
        return empty

    inc_a = build_incidence(edges_a, anchors=anchors_a, junctions=junctions_a)
    inc_b = build_incidence(edges_b, anchors=anchors_b, junctions=junctions_b)
    a_dir, a_und = _edge_indexes(edges_a)
    b_dir, b_und = _edge_indexes(edges_b)
    has_edges = bool(edges_a) or bool(edges_b)
    idx_a = {n: i for i, n in enumerate(nodes_a)}
    idx_b = {n: j for j, n in enumerate(nodes_b)}

    conjunction = bool(getattr(P, "MATCH_RELATIONAL_CONJUNCTION", False))
    anneal = bool(getattr(P, "MATCH_ALIGN_BETA_ANNEAL", False))
    final_beta = 1.0 if conjunction else float(systematicity_weight)

    def _rel(u: int, v: int, mp: Dict[int, int], rv: Dict[int, int],
             candidate=None) -> float:
        """One charged relational-agreement evaluation."""
        nonlocal pair_evals
        pair_evals += 1
        return _relational_agreement(
            u, v, mp, inc_a, inc_b, b_dir, b_und, a_dir, a_und, rv,
            edge_similarity, checkable=True, candidate=candidate,
            edge_weight_fn=edge_weight_fn,
        )

    def _quality(u: int, v: int, mp: Dict[int, int], rv: Dict[int, int],
                 beta: float) -> float:
        """The pair quality: the appearance term plus the structure term."""
        s = matrix[idx_a[u]][idx_b[v]]
        if s <= 0.0:
            return 0.0
        rel = _rel(u, v, mp, rv)
        return s * (1.0 - beta) + beta * rel

    def _objective(mp: Dict[int, int], beta: float) -> float:
        """THE JOINT OBJECTIVE of a mapping: the sum of its pair qualities.

        This is the same quantity the assignment maximises, so the
        refinement's moves and the assignment are measured on ONE scale.
        """
        rv = {v: u for u, v in mp.items()}
        return math.fsum(
            _quality(u, v, mp, rv, beta) for u, v in sorted(mp.items())
        )

    def _swap_candidates(mp: Dict[int, int]) -> List[Tuple[int, int]]:
        """The coupled moves: the node pairs whose images may be swapped.

        A swap is the only move that can change WHO PAIRS for two coupled
        nodes: the assignment moves one node at a time, so it can never
        exchange two partners that only make sense together.
        """
        rv = {v: u for u, v in mp.items()}
        out: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()

        def _add(a: int, b: int) -> None:
            if a == b:
                return
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append(key)

        for (s, d, _et) in edges_a:
            if s in mp and d in mp:
                _add(s, d)
        for (s, d, _et) in edges_b:
            us = rv.get(s)
            ud = rv.get(d)
            if us is not None and ud is not None:
                _add(us, ud)
        return sorted(out)

    def _swap_delta(mp: Dict[int, int], u1: int, u2: int, v1: int, v2: int,
                    beta: float) -> float:
        """The change of the joint objective if u1 and u2 exchange images.

        Only the pairs that can change are re-read: the pairs incident to
        u1/u2 on the A side (their neighbours' images changed) and the
        pairs incident to v1/v2 on the B side (their preimages changed).
        Every other pair reads the same relations under both mappings.

        The SWAPPED pairs are accounted separately: the pair (u1, v1)
        exists only in the old mapping and (u1, v2) only in the new one, so
        they cannot be differenced against each other -- each is compared
        with its counterpart on the OTHER side of the swap.
        """
        trial = dict(mp)
        trial[u1] = v2
        trial[u2] = v1
        rv = {v: u for u, v in mp.items()}
        trv = {v: u for u, v in trial.items()}
        affected_a = {u1, u2}
        for u in (u1, u2):
            for (w, _et, _o) in inc_a.get(u) or ():
                affected_a.add(w)
        affected_b = {v1, v2}
        for v in (v1, v2):
            for (x, _et, _o) in inc_b.get(v) or ():
                affected_b.add(x)
        # the pairs that exist in BOTH mappings
        pairs: Set[Tuple[int, int]] = set()
        for u in affected_a - {u1, u2}:
            v = mp.get(u)
            if v is not None and trial.get(u) == v:
                pairs.add((u, v))
        for v in affected_b - {v1, v2}:
            u = rv.get(v)
            if u is not None and trv.get(v) == u:
                pairs.add((u, v))
        delta = 0.0
        for (u, v) in sorted(pairs):
            delta += (_quality(u, v, trial, trv, beta)
                      - _quality(u, v, mp, rv, beta))
        delta += (_quality(u1, v2, trial, trv, beta)
                  + _quality(u2, v1, trial, trv, beta)
                  - _quality(u1, v1, mp, rv, beta)
                  - _quality(u2, v2, mp, rv, beta))
        return delta

    def _swap_pass(mp: Dict[int, int], beta: float) -> Tuple[Dict[int, int], bool]:
        """The coupled swap search: accept a swap only on a strict gain."""
        exhausted = False
        for _sweep in range(4):
            improved = False
            for (u1, u2) in _swap_candidates(mp):
                if max_pair_evals is not None and pair_evals >= max_pair_evals:
                    exhausted = True
                    break
                v1 = mp.get(u1)
                v2 = mp.get(u2)
                if v1 is None or v2 is None or v1 == v2:
                    continue
                if _swap_delta(mp, u1, u2, v1, v2, beta) > _EPS:
                    mp[u1] = v2
                    mp[u2] = v1
                    improved = True
            if exhausted or not improved:
                break
        return mp, exhausted

    seed_matrix = matrix
    if bool(getattr(P, "MATCH_ALIGN_RELATIONAL_SEED", False)) and has_edges:
        seed_matrix = _relational_seed_matrix(matrix, nodes_a, nodes_b,
                                               edges_a, edges_b)
    mapping = _assign(seed_matrix, nodes_a, nodes_b, unmatched_cost=unmatched_cost)

    rounds_used = 0
    budget_exhausted = False
    if has_edges and rounds > 1:
        n_ref = max(1, rounds - 1)
        best_mapping = dict(mapping)
        best_objective = _objective(mapping, final_beta)
        current_objective = best_objective
        seen: Set[Tuple[Tuple[int, int], ...]] = {
            tuple(sorted(mapping.items()))
        }
        for round_index in range(max(0, rounds - 1)):
            if max_pair_evals is not None and pair_evals >= max_pair_evals:
                budget_exhausted = True
                break
            rev = {v: u for u, v in mapping.items()}

            def rel_weight(i: int, j: int, _mapping=mapping, _rev=rev) -> float:
                # [A2] the candidate is evaluated in the world it induces
                u = nodes_a[i]
                v = nodes_b[j]
                return _rel(u, v, _mapping, _rev, candidate=(u, v))

            if conjunction and anneal:
                # Pre-registered schedule: beta_k = (k+1)/(rounds-1),
                # k=0..rounds-2, so the last refinement round runs at 1.0.
                beta_k = min(1.0, (round_index + 1) / n_ref)
            elif conjunction:
                beta_k = 1.0
            else:
                beta_k = float(systematicity_weight)
            new_mapping = _assign(
                matrix, nodes_a, nodes_b, rel_weight,
                beta=beta_k,
                skip_zero=bool(budget_refinement_only),
                unmatched_cost=unmatched_cost,
            )
            # the coupled moves: the relations may change WHO PAIRS
            new_mapping, exhausted = _swap_pass(new_mapping, final_beta)
            if exhausted:
                budget_exhausted = True
            rounds_used += 1
            key = tuple(sorted(new_mapping.items()))
            if key in seen:
                # [A3] a repeated mapping ends the search ONLY when the
                # schedule has run out.  Breaking earlier (as the old
                # `if new_mapping == mapping: break` did) stopped the
                # ANNEAL at a beta below the final one and then scored the
                # result at the final beta -- a mapping optimised for one
                # relational weight and judged at another.
                if not (conjunction and anneal) or beta_k >= final_beta - _EPS:
                    break
            seen.add(key)
            objective = _objective(new_mapping, final_beta)
            if objective > best_objective + _EPS:
                best_objective = objective
                best_mapping = dict(new_mapping)
            if objective > current_objective + _EPS:
                # [A3] a round that does not improve is not accepted
                current_objective = objective
                mapping = new_mapping
            elif key == tuple(sorted(mapping.items())):
                break
        mapping = best_mapping

    # 2. Final quality and the unmatched floor, as a FIXED POINT.
    #    [A7] the relations are re-read against the mapping that SURVIVED
    #    the pruning: the old code read them against the pre-pruning map, so
    #    a pair kept the credit of a partner that had just been dropped.
    filtered = dict(mapping)
    quality: Dict[int, float] = {}
    for _pass in range(_PRUNE_PASSES):
        rev_filtered = {v: u for u, v in filtered.items()}
        new_filtered: Dict[int, int] = {}
        new_quality: Dict[int, float] = {}
        for u, v in sorted(filtered.items()):
            s = matrix[idx_a[u]][idx_b[v]]
            if s <= 0.0:
                continue
            if has_edges:
                rel = _rel(u, v, filtered, rev_filtered)
            else:
                rel = 1.0
            q = s * (1.0 - final_beta) + final_beta * rel
            if q <= unmatched_cost:
                continue
            new_filtered[u] = v
            new_quality[u] = q
        settled = (new_filtered == filtered)
        filtered = new_filtered
        quality = new_quality
        if settled:
            break

    sys_conj, fwd, bwd = systematicity(filtered, edges_a, edges_b)
    n_match = len(filtered)
    node_cov = (2.0 * n_match) / float(len(nodes_a) + len(nodes_b))
    unmatched_a = tuple(u for u in nodes_a if u not in filtered)
    unmatched_b = tuple(v for v in nodes_b if v not in set(filtered.values()))

    return Alignment(
        mapping=dict(sorted(filtered.items())),
        quality=quality,
        systematicity=sys_conj,
        fwd_edge_cov=fwd,
        bwd_edge_cov=bwd,
        node_coverage=node_cov,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        rounds_used=rounds_used,
        budget_exhausted=budget_exhausted,
    )
