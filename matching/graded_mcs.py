"""[STAGE 10] GRADED GLOBAL MCS -- one global optimisation over the
correspondence, with the nodes and the relations decided TOGETHER.

    maximize   W_N · (1/|Q|) · Σ_{u,v} s(u,v)·x[u,v]
             + W_E · (1/|E_A|) · Σ_{e,e'} a(e,e')·y[(u,v),(u',v')]
             − W_U · [ (1/|Q|)·unmatched_q + (1/|C|)·unmatched_r ]

    s.t.  Σ_v x[u,v] ≤ 1,  Σ_u x[u,v] ≤ 1,  and y is the AND of its two
          endpoint node-pairs (y ≤ x, y ≤ x', y ≥ x + x' − 1, y ≥ 0).

WHY THIS IS SIMULTANEOUS.  The edge term is live ONLY when both of the
edge's endpoint node-pairs are selected.  A node pair whose neighbourhood
conflicts therefore earns less than an alternative pairing that preserves
the relations -- and the SOLVER makes that trade-off, globally, in one
decision.  There is no node-first pass, no edge audit afterwards and no
admission gate: nothing is rejected for disagreeing, it simply scores less.

GRADED, NOT EXACT.  `s` is the graded signature similarity and `a` the
graded polymorphic edge agreement; both live in [0, 1].

Pre-registered in diagnostics/phase10_preregistration.md BEFORE this was
written.  The contract is the one `_global_correspondence` returns:
`(score, correspondence, diagnostics)`, with `correspondence` the
`Dict[int, int]` from the query node to the reference node.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import parameters as P
from core.schema import edge_categorical_axes, edge_graded_axes
from matching.structural_similarity import _lookup_edge_params_oriented

__all__ = ["graded_mcs", "GradedMCSResult"]


class GradedMCSResult:
    """The result of one global solve.

    Attributes:
        score         the objective value (the same scale the matcher uses)
        correspondence {query node: reference node}
        diagnostics   the solver's statistics, the three terms of the
                      objective, the pruning and the truncation reports
    """

    __slots__ = ("score", "correspondence", "diagnostics")

    def __init__(self, score: float, correspondence: Dict[int, int],
                 diagnostics: Dict[str, Any]) -> None:
        self.score = float(score)
        self.correspondence = dict(correspondence)
        self.diagnostics = dict(diagnostics)

    def __iter__(self):
        return iter((self.score, self.correspondence, self.diagnostics))

    def __getitem__(self, i):
        return (self.score, self.correspondence, self.diagnostics)[i]


# --------------------------------------------------------------------------
# the two kernels
# --------------------------------------------------------------------------
def _sig_similarity(a: Any, b: Any) -> float:
    """The GRADED node similarity: `TypedSignature.similarity` (the graded
    feature distance, with the V4 CONTOUR turning-function alignment).

    A missing signature is 0.0 ("no evidence"), never an error.  The
    similarity is NOT restricted by lens type: two nodes of different kinds
    may still be similar, and the candidate floor (not a lens-type rule)
    decides.
    """
    if a is None or b is None:
        return 0.0
    return float(a.similarity(b))


#: [G5] The number of edge agreements that could not be computed.  A
#: malformed edge must not take the whole solve down; the edge simply
#: weighs nothing -- but the count was CLAIMED in the comment of
#: `_edge_agreement` and nothing counted it, so a solve that was quietly
#: losing relations looked like a solve that had none.
_EDGE_AGREEMENT_FAILURES: Dict[str, int] = {"count": 0}


def _edge_agreement(q_s: Any, q_d: int, q_et: Any, q_ep: Any,
                    r_s: Any, r_d: int, r_et: Any, r_ep: Any) -> float:
    """The GRADED edge agreement of two edges.

    `matching/structural_similarity.py::_pair_edge_sim_typed` does the work:
    the polymorphic projection, `get_compatible_types` for a cross-type
    pair, `reverse_edge_params` / `flipped_copy()` for the undirected
    reversal, the circular wrap on the cyclic axes, and the UNKNOWN/absent
    axes weighing nothing.

    [G5] The failure IS counted (the comment used to claim a count that
    did not exist).
    """
    from matching.structural_similarity import _pair_edge_sim_typed
    try:
        return float(_pair_edge_sim_typed(q_s, q_d, q_et, q_ep,
                                           r_s, r_d, r_et, r_ep))
    except Exception:                              # noqa: BLE001
        # a malformed edge must not take the whole solve down; the edge
        # simply weighs nothing.  The count is reported.
        _EDGE_AGREEMENT_FAILURES["count"] += 1
        return 0.0


def _effective_edge_weights(Q: Any, C: Any) -> Optional[Dict[Any, float]]:
    """The EFFECTIVE edge weights, exactly as the solver applies them.

    [G1] The solver multiplied each edge term by the effective weights of
    its two edges; `objective_value` did not, so the two counted the same
    relation differently.  They now read the SAME weights from the SAME
    place.
    """
    if not getattr(P, "GRADED_MCS_USE_EFFECTIVE_EDGE_WEIGHTS", True):
        return None
    try:
        from memory.cores import _effective_edge_weights as _w
    except Exception:                               # noqa: BLE001
        return None
    try:
        out: Dict[Any, float] = {}
        out.update(_w(Q))
        out.update(_w(C))
        return out
    except Exception:                               # noqa: BLE001
        return None


def _static_coherence(Q: Any, C: Any, u: int, v: int,
                       memo: Dict[Any, float],
                       q_inc: Dict[int, List[Any]],
                       c_inc: Dict[int, List[Any]]) -> float:
    """THE STATIC COHERENCE of a candidate pair: how well v's relations
    match u's, WITHOUT the mapping.

    This is the design's own notion of the unit ("a node-with-its-edges"):
    the node's signature says what it IS, and this says whether the
    counterpart has the relations it has.  A node pair whose neighbourhood
    has no counterpart scores 0 -- "a node pair with disagreeing edges is
    NOT a match, no matter how similar the nodes look".

    It is MAPPING-INDEPENDENT, so it is a constant of the candidate pair and
    costs no variable and no constraint.

    The agreements are memoised on the edge PAIR, so the cost is the number
    of DISTINCT edge pairs the candidates touch, not their product.
    """
    rels_u = q_inc.get(u) or []
    rels_v = c_inc.get(v) or []
    if not rels_u or not rels_v:
        # [B21] NO EVIDENCE IS NOT PERFECT AGREEMENT.  matching/alignment.py
        # reads an edgeless side as 0.0 relational support unconditionally;
        # the MILP path must agree, or the absence of structure earns the
        # full coherence credit and displaces structured pairs.
        return 0.0
    total = 0.0
    for (eu, epu) in rels_u:
        best = 0.0
        for (ev, epv) in rels_v:
            key = (eu, ev)
            got = memo.get(key)
            if got is None:
                got = _edge_agreement(eu[0], eu[1], eu[2], epu,
                                      ev[0], ev[1], ev[2], epv)
                memo[key] = got
            if got > best:
                best = got
        total += best
    return total / len(rels_u)


def _incident(sg: Any, params: Dict[Any, Any], node: int) -> List[Tuple[Any, Any]]:
    """The relations incident to a node, as (edge, edge params)."""
    out = []
    for e in sg.edges:
        if e[0] == node or e[1] == node:
            ep = params.get(e) or params.get((e[1], e[0], e[2]))
            out.append((e, ep))
    return out


# --------------------------------------------------------------------------
# the candidate set
# --------------------------------------------------------------------------
def _candidate_pairs(nodes_q: Sequence[int], nodes_c: Sequence[int],
                      sim: Any, floor: float, top_k: int,
                      max_node_vars: int,
                      lens_q: Optional[Dict[int, str]] = None,
                      lens_c: Optional[Dict[int, str]] = None,
                      lens_blocked: bool = False,
                      q_types: Optional[Dict[int, set]] = None,
                      c_types: Optional[Dict[int, set]] = None) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """The candidate node pairs: the top-k by similarity per query node,
    UNIONED with the top-k per reference node, filtered by the floor, and
    UNIONED with the STRUCTURALLY supported pairs.

    With `lens_blocked` the candidates are restricted to the SAME LENS TYPE
    on both sides -- the population the incumbent matcher scores (its
    correspondence is a per-lens-type assignment).  MEASURED (stage 10): over
    the all-pairs population the solver matches 262 nodes per pair against the
    incumbent's 159, and the extra matches are the ones that DISPLACE the
    true pairs (the attribute nodes are matched to frame nodes).

    [G8] THE STRUCTURALLY SUPPORTED PAIRS.  The pruning was by pure node
    similarity, so a pair whose node similarity is low but whose RELATIONS
    agree -- exactly the pair the edge term exists to find -- could be
    eliminated before the joint solve ever saw it.  A pair is therefore also
    a candidate when the two nodes carry a COMMON relation type and the node
    similarity is positive (a positive similarity is the "comparable"
    condition; a pair that is not comparable is never a candidate whatever
    its structure).

    The order is canonical: the query nodes in ascending id, the candidates
    of a node by DESCENDING similarity with the reference id as the
    tie-break.  The same input therefore gives the same candidate list, in
    the same order, whatever the hash seed.
    """
    def allowed(u: int, v: int) -> bool:
        if not lens_blocked:
            return True
        return lens_q.get(u) == lens_c.get(v)

    cands: Dict[int, List[int]] = {u: [] for u in nodes_q}
    # per query node: the best k by similarity
    for u in nodes_q:
        row = sorted(((sim(u, v), v) for v in nodes_c if allowed(u, v)),
                     key=lambda t: (-t[0], t[1]))
        cands[u] = [v for s, v in row[:top_k] if s > floor]
    # per reference node: the best k by similarity (the union)
    for v in nodes_c:
        col = sorted(((sim(u, v), u) for u in nodes_q if allowed(u, v)),
                     key=lambda t: (-t[0], t[1]))
        for s, u in col[:top_k]:
            if s > floor and v not in cands[u]:
                cands[u].append(v)
    # [G8] the structurally supported pairs: a common relation type, a
    # similarity above the candidate FLOOR (the floor is absolute: a pair at
    # or below it is not a candidate at all), and at most top_k of them per
    # query node so the candidate population stays bounded.  They bypass the
    # TOP-K cut, which is what could eliminate the true partner before the
    # joint solve saw it.
    if q_types and c_types:
        for u in nodes_q:
            su = q_types.get(u) or set()
            if not su:
                continue
            row = []
            for v in nodes_c:
                if not allowed(u, v):
                    continue
                if v in cands[u]:
                    continue
                sv = c_types.get(v) or set()
                if not (su & sv):
                    continue
                s = sim(u, v)
                if s <= floor:
                    # the candidate FLOOR is absolute: a pair at or below it
                    # is not a candidate at all, whatever its structure
                    continue
                row.append((s, v))
            row.sort(key=lambda t: (-t[0], t[1]))
            for _s, v in row[:top_k]:
                cands[u].append(v)
    # the canonical order: by query id, then by reference id
    pairs: List[Tuple[int, int]] = []
    for u in nodes_q:
        for v in sorted(cands[u]):
            pairs.append((u, v))
    truncated = 0
    if len(pairs) > max_node_vars:
        # the lowest similarities go first, and the truncation is REPORTED
        scored = sorted(((sim(u, v), u, v) for u, v in pairs),
                        key=lambda t: (-t[0], t[1], t[2]))
        keep = scored[:max_node_vars]
        truncated = len(pairs) - len(keep)
        pairs = sorted(((u, v) for _s, u, v in keep), key=lambda t: (t[0], t[1]))
    report = {
        "candidates": len(pairs),
        "candidate_floor": floor,
        "top_k": top_k,
        "node_var_truncation": truncated,
    }
    return pairs, report


def _incident_relation_types(sg: Any) -> Dict[int, set]:
    """node -> the set of relation types incident to it."""
    out: Dict[int, set] = {}
    for (x, y, et) in getattr(sg, "edges", ()) or ():
        out.setdefault(x, set()).add(et)
        out.setdefault(y, set()).add(et)
    return out


# --------------------------------------------------------------------------
# the objective's three terms
# --------------------------------------------------------------------------
def _edge_terms(Q: Any, C: Any, q_sigs: Dict[int, Tuple[str, Any]],
                c_sigs: Dict[int, Tuple[str, Any]],
                cand: Dict[int, List[int]],
                pair_index: Dict[Tuple[int, int], int],
                edge_weight: Optional[Dict[Any, float]]) -> Tuple[
                    List[Tuple[int, int, float]], Dict[str, Any]]:
    """The edge terms: every (query edge, reference edge) whose TWO endpoint
    node-pairs are both candidates.

    Each term is `(x_index_1, x_index_2, agreement)`; the solver's `y`
    variable says whether both node-pairs are selected.
    """
    from core.types import is_undirected_edge

    q_params = getattr(Q, "edge_params", None) or {}
    r_params = getattr(C, "edge_params", None) or {}

    def _ep(sg_params, e):
        p = sg_params.get(e)
        if p is None:
            p = sg_params.get((e[1], e[0], e[2]))
        return p

    terms: List[Tuple[int, int, float]] = []
    #: the indices of the terms that are EVIDENCE (measured on both sides)
    measured: Set[int] = set()
    n_computed = 0
    n_incompatible = 0
    failures_before = _EDGE_AGREEMENT_FAILURES["count"]
    # the reference side's edges, indexed by their endpoint pair
    r_edges: Dict[Tuple[int, int], List[Any]] = {}
    for (s, d, et) in C.edges:
        r_edges.setdefault((s, d), []).append(et)
        if is_undirected_edge(et):
            r_edges.setdefault((d, s), []).append(et)
    for (u, u2, et) in sorted(Q.edges,
                              key=lambda e: (e[0], e[1],
                                             getattr(e[2], "name", str(e[2])))):
        # [LEAD] The ORIENTATION each side's parameters are stored in is
        # resolved from the store, and the endpoints are handed over in it:
        # `_pair_edge_sim_typed` decides whether to reverse the second side
        # from the ORDER OF THE NODE IDS it is given, which is independent
        # of the order the parameters are stored in, so the same relation
        # was read forwards or backwards by accident.
        ep_a, flip_a = _lookup_edge_params_oriented(q_params, u, u2, et)
        q_s, q_d = (u2, u) if flip_a else (u, u2)
        # [G1] ONE TERM PER (QUERY EDGE, CANDIDATE PAIR).  The terms used to
        # be ONE per query edge -- the BEST candidate pair -- so the solver
        # could not express the credit of any mapping whose relation did not
        # use that best pair, and it maximised a different function from the
        # one `objective_value` scores truth on: the whole truth-vs-optimum
        # comparison was invalid.  There is one term per candidate PAIR now,
        # and within a pair the BEST compatible reference edge (so one
        # preserved relation is still one objective unit: the y variables
        # are linked to the two endpoint pairs, and the one-to-one
        # constraints make at most one of them selectable).
        for v in cand.get(u, ()):
            i1 = pair_index.get((u, v))
            if i1 is None:
                continue
            for v2 in cand.get(u2, ()):
                i2 = pair_index.get((u2, v2))
                if i2 is None:
                    continue
                best = None
                measured_here = False
                for et_r in r_edges.get((v, v2), ()):
                    ep_b, flip_b = _lookup_edge_params_oriented(
                        r_params, v, v2, et_r)
                    r_s, r_d = (v2, v) if flip_b else (v, v2)
                    a = _edge_agreement(q_s, q_d, et, ep_a, r_s, r_d, et_r, ep_b)
                    n_computed += 1
                    if a <= 0.0:
                        n_incompatible += 1
                        continue
                    # [G7] IS THIS TERM EVIDENCE?  A missing parameter
                    # scores 0.5 (neutral) and a Gaussian is never zero, so
                    # a relation that was NOT measured on both sides counts
                    # as support like a real one.  A term is EVIDENCE when
                    # both sides carry parameters and the pair has graded
                    # axes (or a categorical agreement) to read.
                    if (ep_a is not None and ep_b is not None
                            and (edge_graded_axes(ep_a, ep_b) > 0
                                 or edge_categorical_axes(ep_a, ep_b) > 0)):
                        measured_here = True
                    wa = wb = 1.0
                    if edge_weight:
                        wa = float(edge_weight.get((u, u2, et), 1.0))
                        wb = float(edge_weight.get((v, v2, et_r), 1.0))
                    value = a * wa * wb
                    if best is None or value > best:
                        best = value
                if best is not None:
                    terms.append((i1, i2, best))
                    if measured_here:
                        measured.add(len(terms) - 1)
    report = {"edge_terms": len(terms), "edge_evaluations": n_computed,
              "edge_zero_agreements": n_incompatible,
              "edge_agreement_failures":
                  _EDGE_AGREEMENT_FAILURES["count"] - failures_before,
              # [G7] the terms that are EVIDENCE (measured on both sides)
              "measured_edge_terms": measured}
    return terms, report


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------
def node_weight_fn(Q: Any, C: Any,
                   q_sigs: Dict[int, Tuple[str, Any]],
                   c_sigs: Dict[int, Tuple[str, Any]],
                   use_anchor: Optional[bool] = None):
    """THE NODE WEIGHT of the correspondence: the function the solver
    maximises AND the function `objective_value` evaluates with, so the two
    are always on the same scale.

    It is the graded signature similarity, TIMES the anchor-proximity factor
    when P.GRADED_MCS_USE_ANCHOR_PRIOR is on -- the tree's LIVE node weight
    (memory/cores.py::_mcs_accumulate's `_anchor_similarity`)."""
    if use_anchor is None:
        use_anchor = bool(getattr(P, "GRADED_MCS_USE_ANCHOR_PRIOR", False))
    _has, q_norm, c_norm = _anchor_factor(Q, C) if use_anchor \
        else (False, None, None)
    use_anchor = use_anchor and _has
    cache: Dict[Tuple[int, int], float] = {}

    def weight(u: int, v: int) -> float:
        k = (u, v)
        got = cache.get(k)
        if got is None:
            got = _sig_similarity(q_sigs[u][1], c_sigs[v][1])
            if use_anchor:
                a = q_norm.get(u)
                b = c_norm.get(v)
                if a is not None and b is not None:
                    got *= math.exp(-2.5 * math.hypot(a[0] - b[0],
                                                      a[1] - b[1]))
            cache[k] = got
        return got

    return weight, use_anchor


def objective_value(mapping: Dict[int, int], Q: Any, C: Any,
                    q_sigs: Dict[int, Tuple[str, Any]],
                    c_sigs: Dict[int, Tuple[str, Any]],
                    beta: Optional[float] = None,
                    node_mass: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    """THE OBJECTIVE, evaluated at one mapping.

    The same definition the solver maximises, so that a mapping can be
    compared with the solver's own optimum on the same scale:

        node term   Σ_i A_i·x_i·(1 − β + β·κ(i)),  A_i = W_N·s_i·w_i/|Q|,
                    κ(i) = (the relations of i the mapping preserves)/deg(i)
        edge term   Σ_e W_E·a(e,e')·y_e/|E_A|,  y_e = both endpoints matched
        unmatched   −W_U·[(1−|M|)/|Q| + (1−|M|)/|C|]

    [G1 / G2] The function READS THE SAME THINGS THE SOLVER READS: the
    node masses (`node_mass` -- it was read but not even a PARAMETER, so
    the call raised NameError), the effective edge weights, and the static
    coherence.  It is therefore the same function the solver maximises, and
    a mapping can be compared with the solver's optimum on one scale.
    """
    from core.types import is_undirected_edge
    if beta is None:
        beta = float(getattr(P, "GRADED_MCS_COHERENCE_BETA", 0.0))
    W_N = float(getattr(P, "GRADED_MCS_NODE_WEIGHT", 1.0))
    W_E = float(getattr(P, "GRADED_MCS_EDGE_WEIGHT", 1.0))
    W_U = float(getattr(P, "GRADED_MCS_UNMATCHED_COST", 0.10))
    static_coh = bool(getattr(P, "GRADED_MCS_STATIC_COHERENCE", False))

    frame_only = bool(getattr(P, "GRADED_MCS_FRAME_ONLY", False))
    if frame_only:
        from matching.relation_frame import is_frame
        nodes_q = [n for n in Q.nodes
                   if n in q_sigs and is_frame(q_sigs[n][0])]
        nodes_c = [n for n in C.nodes
                   if n in c_sigs and is_frame(c_sigs[n][0])]
    else:
        nodes_q = [n for n in Q.nodes if n in q_sigs]
        nodes_c = [n for n in C.nodes if n in c_sigs]
    n_q = max(1, len(nodes_q))
    n_c = max(1, len(nodes_c))
    n_e = max(1, len(Q.edges))
    q_params = getattr(Q, "edge_params", None) or {}
    r_params = getattr(C, "edge_params", None) or {}
    edge_weight = _effective_edge_weights(Q, C)
    r_edges: Dict[Tuple[int, int], List[Any]] = {}
    for (s, d, et) in C.edges:
        r_edges.setdefault((s, d), []).append(et)
        if is_undirected_edge(et):
            r_edges.setdefault((d, s), []).append(et)
    deg: Dict[int, int] = {}
    for (u, u2, _et) in Q.edges:
        deg[u] = deg.get(u, 0) + 1
        deg[u2] = deg.get(u2, 0) + 1
    # the static coherence, when it is the conditioning (the solver's own
    # choice: with it on, the node's credit is conditioned by the
    # mapping-INDEPENDENT coherence and the relation-conditioned term is
    # not added again)
    coh: Dict[Tuple[int, int], float] = {}
    if static_coh:
        q_inc = {u: _incident(Q, q_params, u) for u in nodes_q}
        c_inc = {v: _incident(C, r_params, v) for v in nodes_c}
        coh_memo: Dict[Any, float] = {}
    kept: Dict[int, int] = {}
    kept_at: Dict[int, List[Tuple[int, float]]] = {}
    edge_term = 0.0
    n_preserved = 0
    for (u, u2, et) in Q.edges:
        v = mapping.get(u)
        v2 = mapping.get(u2)
        if v is None or v2 is None:
            continue
        # [LEAD] the orientation each side's parameters are stored in (see
        # `_edge_terms`)
        ep_a, flip_a = _lookup_edge_params_oriented(q_params, u, u2, et)
        q_s, q_d = (u2, u) if flip_a else (u, u2)
        # [G1] the BEST compatible reference edge at the MAPPED pair, with
        # the SAME effective weights the solver applies.
        best_a = 0.0
        for et_r in r_edges.get((v, v2), ()):
            ep_b, flip_b = _lookup_edge_params_oriented(r_params, v, v2, et_r)
            r_s, r_d = (v2, v) if flip_b else (v, v2)
            a = _edge_agreement(q_s, q_d, et, ep_a, r_s, r_d, et_r, ep_b)
            if a <= 0.0:
                continue
            if edge_weight:
                a = a * (float(edge_weight.get((u, u2, et), 1.0))
                         * float(edge_weight.get((v, v2, et_r), 1.0)))
            if a > best_a:
                best_a = a
        if best_a > 0.0:
            edge_term += W_E * best_a / n_e
            n_preserved += 1
            kept[u] = kept.get(u, 0) + 1
            kept[u2] = kept.get(u2, 0) + 1
            kept_at.setdefault(u, []).append((v2, best_a))
            kept_at.setdefault(u2, []).append((v, best_a))
    node_term = 0.0
    weight, _ua = node_weight_fn(Q, C, q_sigs, c_sigs)
    for u, v in mapping.items():
        if u not in q_sigs or v not in c_sigs:
            continue
        w = 1.0
        if node_mass:
            w = float(node_mass.get(u, 1.0)) * float(node_mass.get(v, 1.0))
        A = W_N * weight(u, v) * w / n_q
        d = deg.get(u, 0)
        if static_coh and d:
            # the mapping-INDEPENDENT coherence is the conditioning
            kappa = _static_coherence(Q, C, u, v, coh_memo, q_inc, c_inc)
        elif d:
            # the AGREEMENT-WEIGHTED coherence: the fraction of the node's
            # relations the mapping preserves, weighted by HOW WELL each is
            # preserved
            tot = 0.0
            for (_v2, a) in kept_at.get(u, ()):
                tot += a
            kappa = tot / d
        else:
            # [B21] edgeless => kappa = 0.0 (alignment.py's convention),
            # not the vacuous 1.0 that paid full credit for no structure.
            kappa = 0.0
        node_term += A * (1.0 - beta + beta * kappa)
    unmatched = -W_U * ((len(nodes_q) - len(mapping)) / n_q
                        + (len(nodes_c) - len(mapping)) / n_c)
    return {"node": node_term, "edge": edge_term, "unmatched": unmatched,
            "objective": node_term + edge_term + unmatched,
            "mapped": len(mapping), "preserved": n_preserved}


def _anchor_factor(Q: Any, C: Any):
    """The ANCHOR-PROXIMITY FACTOR of the tree's live correspondence.

    The incumbent matcher (memory/cores.py::_mcs_accumulate) weights a node
    pair by

        signature_similarity(u, v) * exp(-2.5 * hypot(dx, dy))

    on the positions NORMALISED INTO EACH SUBGRAPH'S BOUNDING BOX (the
    `_anchor_similarity` closure, and `_anchor_weights` for the blocked
    path).  The graded global MCS must use the SAME node weight: the
    signature alone does not identify the counterpart -- measured, the
    renderer truth's pairs have a LOWER signature similarity (0.93) than the
    wrong pairs the solver prefers (0.97) -- while the position does.

    This returns `(factor, q_norm, c_norm)`, or `(None, None, None)` when
    either side has no usable box.
    """
    def _box(sg):
        anchor_map = getattr(sg, "anchor_of_node", None) or {}
        if not anchor_map:
            return None
        xs = [float(p[0]) for p in anchor_map.values() if p and len(p) >= 2]
        ys = [float(p[1]) for p in anchor_map.values() if p and len(p) >= 2]
        if len(xs) < 2 or len(ys) < 2:
            return None
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    def _norm(sg):
        box = _box(sg)
        if box is None:
            return None
        anchor_map = getattr(sg, "anchor_of_node", None) or {}
        out = {}
        for n, p in anchor_map.items():
            if not p or len(p) < 2:
                continue
            x, y = float(p[0]), float(p[1])
            out[n] = ((x - box[0]) / max(box[2], 1e-9),
                      (y - box[1]) / max(box[3], 1e-9))
        return out or None

    q_norm = _norm(Q)
    c_norm = _norm(C)
    if q_norm is None or c_norm is None:
        return None, None, None
    return True, q_norm, c_norm


def graded_mcs(Q: Any, C: Any,
               q_sigs: Dict[int, Tuple[str, Any]],
               c_sigs: Dict[int, Tuple[str, Any]],
               node_mass: Optional[Dict[int, float]] = None,
               ) -> GradedMCSResult:
    """Solve the graded global MCS for one pair of graphs.

    Returns `(score, correspondence, diagnostics)`.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import Bounds, LinearConstraint, milp

    t0 = time.time()
    W_N = float(getattr(P, "GRADED_MCS_NODE_WEIGHT", 1.0))
    W_E = float(getattr(P, "GRADED_MCS_EDGE_WEIGHT", 1.0))
    W_U = float(getattr(P, "GRADED_MCS_UNMATCHED_COST", 0.10))
    floor = float(getattr(P, "GRADED_MCS_CANDIDATE_FLOOR", 0.05))
    top_k = int(getattr(P, "GRADED_MCS_TOP_K", 10))
    max_node_vars = int(getattr(P, "GRADED_MCS_MAX_NODE_VARS", 60000))
    max_edge_vars = int(getattr(P, "GRADED_MCS_MAX_EDGE_VARS", 300000))
    max_seconds = float(getattr(P, "GRADED_MCS_MAX_SECONDS", 120.0))
    # [STAGE 10] THE DETERMINISTIC WORK LIMIT.  A wall-clock limit is NOT
    # deterministic: the same program returns a DIFFERENT mapping on a
    # different machine (or the same machine at a different load), which
    # breaks the project's determinism mandate.  The bound is therefore the
    # solver's NODE count -- the same search work on every machine -- with the
    # time limit kept only as a documented safety net (it should not bind for
    # a node-limited solve).
    node_limit = int(getattr(P, "GRADED_MCS_NODE_LIMIT", 20000))

    nodes_q = sorted(n for n in Q.nodes if n in q_sigs)
    nodes_c = sorted(n for n in C.nodes if n in c_sigs)
    diagnostics: Dict[str, Any] = {
        "solver": "scipy.optimize.milp (HiGHS)",
        "nodes_q": len(nodes_q), "nodes_c": len(nodes_c),
        "weights": {"W_N": W_N, "W_E": W_E, "W_U": W_U},
    }
    # [STAGE 10, THE FIX] THE FRAME POPULATION.  The correspondence is over
    # the FRAME (matching/relation_frame.py: the lens types that carry
    # structural relations).  The attribute nodes attach only through
    # CO_LOCATED, which is PROXIMITY and excluded from the growth, so an
    # attribute pair has no structural edge to disagree about: under the
    # vacuous-1.0 convention ("relation-free nodes are identified by
    # signature alone") it earns the FULL credit and, because the
    # one-to-one constraint is global, it DISPLACES a frame pair.
    #
    # This is the SECOND, INDEPENDENT fix: the coherence credit changes what
    # a pair is worth, this changes WHO may be matched at all.  They are
    # measured separately so the attribution is unambiguous.
    frame_only = bool(getattr(P, "GRADED_MCS_FRAME_ONLY", False))
    if frame_only:
        from matching.relation_frame import is_frame
        nodes_q = [n for n in nodes_q if is_frame(q_sigs[n][0])]
        nodes_c = [n for n in nodes_c if is_frame(c_sigs[n][0])]
    diagnostics["frame_only"] = frame_only
    diagnostics["nodes_q"] = len(nodes_q)
    diagnostics["nodes_c"] = len(nodes_c)
    if not nodes_q or not nodes_c:
        diagnostics["status"] = "empty"
        return GradedMCSResult(0.0, {}, diagnostics)

    # ---- the node similarity ----
    sim_cache: Dict[Tuple[int, int], float] = {}
    # [STAGE 10, THE FIX] the node weight: the graded signature similarity,
    # TIMES the anchor proximity when P.GRADED_MCS_USE_ANCHOR_PRIOR is on --
    # the tree's LIVE correspondence weight.  With the prior off the weight is
    # the signature alone (the stage-10 pre-registered arm).
    sim, use_anchor = node_weight_fn(Q, C, q_sigs, c_sigs)
    diagnostics["anchor_prior"] = bool(use_anchor)

    # ---- the candidates ----
    lens_q = {u: (q_sigs[u][0] if u in q_sigs else "") for u in nodes_q}
    lens_c = {v: (c_sigs[v][0] if v in c_sigs else "") for v in nodes_c}
    lens_blocked = bool(getattr(P, "GRADED_MCS_LENS_BLOCKED", False))
    pairs, cand_report = _candidate_pairs(nodes_q, nodes_c, sim, floor, top_k,
                                          max_node_vars, lens_q, lens_c,
                                          lens_blocked,
                                          _incident_relation_types(Q),
                                          _incident_relation_types(C))
    diagnostics.update(cand_report)
    diagnostics["lens_blocked"] = lens_blocked
    if not pairs:
        diagnostics["status"] = "no_candidates"
        return GradedMCSResult(0.0, {}, diagnostics)
    pair_index = {p: i for i, p in enumerate(pairs)}
    cand: Dict[int, List[int]] = {u: [] for u in nodes_q}
    for (u, v) in pairs:
        cand[u].append(v)

    # ---- the edge terms ----
    edge_weight = _effective_edge_weights(Q, C)
    edge_terms, edge_report = _edge_terms(
        Q, C, q_sigs, c_sigs, cand, pair_index, edge_weight)
    diagnostics.update(edge_report)
    measured_terms = set(edge_report.get("measured_edge_terms") or ())
    # the cap: the lowest agreements go first, and the truncation is REPORTED
    if len(edge_terms) > max_edge_vars:
        edge_terms.sort(key=lambda t: (-t[2], t[0], t[1]))
        diagnostics["edge_var_truncation"] = len(edge_terms) - max_edge_vars
        edge_terms = edge_terms[:max_edge_vars]
        edge_terms.sort(key=lambda t: (t[0], t[1]))
    else:
        diagnostics["edge_var_truncation"] = 0

    n_x = len(pairs)
    n_y = len(edge_terms)
    diagnostics["variables"] = {"x": n_x, "y": n_y,
                                "total": n_x + n_y}

    # ---- the objective ----
    n_q = max(1, len(nodes_q))
    n_c = max(1, len(nodes_c))
    n_e = max(1, len(Q.edges))
    # the unmatched term contributes +W_U·(1/|Q| + 1/|C|) per MATCH, so it
    # enters the objective as a per-x coefficient (the pre-registration
    # spells the algebra out).
    bonus = W_U * (1.0 / n_q + 1.0 / n_c)
    obj = np.zeros(n_x + n_y, dtype=np.float64)
    # [STAGE 10] THE COHERENCE CONDITIONING (beta).  The design says: "a node
    # pair with disagreeing edges is NOT a match, no matter how similar the
    # nodes look" -- the node's credit must be CONDITIONAL on the unit's
    # coherence, the agreement-weighted fraction of the node's relations the
    # mapping preserves:
    #
    #   node credit(i) = A_i · x_i · (1 − β + β·κ(i)),
    #                     A_i = W_N·s_i/|Q|,
    #                     κ(i) = (1/deg(i))·Σ_{e at i} a(e,e')·y_e
    #
    # Writing that conditioning as an objective is EXACTLY equivalent to
    # moving the node's credit ONTO ITS UNITS, because y_e IS the conjunction
    # x_{i1} ∧ x_{i2} (so x_i ∧ y_e = y_e for the units at i):
    #
    #   Σ_i A_i·x_i·(1 − β + β·κ(i))
    #     = (1 − β)·Σ_i A_i·x_i
    #       + β·Σ_e a_e·( A_{i1}/deg(u_{i1}) + A_{i2}/deg(u_{i2}) )·y_e
    #
    # NO new variable and NO new constraint: the coherence enters through the
    # node and the unit coefficients.
    #
    # With beta = 0 this is the plain additive objective (the stage-10
    # pre-registered arm); with beta = 1 it is the design's own reading: a
    # node earns its similarity ONLY through the units it preserves.
    beta = float(getattr(P, "GRADED_MCS_COHERENCE_BETA", 0.0))
    static_coh = bool(getattr(P, "GRADED_MCS_STATIC_COHERENCE", False))
    # [B19] ONE PASS over |E|, not O(n_q x |E|).
    q_deg = {}
    for (_u, _u2, _et) in Q.edges:
        q_deg[_u] = q_deg.get(_u, 0) + 1
        q_deg[_u2] = q_deg.get(_u2, 0) + 1
    if static_coh:
        q_params = getattr(Q, "edge_params", None) or {}
        c_params = getattr(C, "edge_params", None) or {}
        q_inc = {u: _incident(Q, q_params, u) for u in nodes_q}
        c_inc = {v: _incident(C, c_params, v) for v in nodes_c}
        coh_memo: Dict[Any, float] = {}
        coh = {}
        for (u, v) in pairs:
            coh[(u, v)] = _static_coherence(Q, C, u, v, coh_memo, q_inc, c_inc)
        diagnostics["static_coherence_mean"] = (
            sum(coh.values()) / max(1, len(coh)))
    # the node credits A_i (before the coherence) and the unit rewards
    A_vec = np.zeros(n_x, dtype=np.float64)
    # [G1] the node credit BEFORE the coherence conditioning -- the credit
    # the unit term carries when it conditions a node.
    A_full = np.zeros(n_x, dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        w = 1.0
        if node_mass:
            w = float(node_mass.get(u, 1.0)) * float(node_mass.get(v, 1.0))
        A = W_N * sim(u, v) * w / n_q
        # [B21] A node with NO relations has NO coherence (kappa = 0.0),
        # the same reading alignment.py uses: it earns only the appearance
        # part (1 - beta) of its credit, never the full unconditional A.
        has_edges = q_deg.get(u, 0) > 0
        if static_coh:
            # _static_coherence returns 0.0 for an edgeless side, so this
            # one expression covers both cases.
            A = A * (1.0 - beta + beta * coh[(u, v)])
            b_eff = 0.0
        elif not has_edges:
            A = A * (1.0 - beta)
            b_eff = 0.0
        else:
            b_eff = beta
        A_full[i] = A
        A_vec[i] = A * (1.0 - b_eff)
    E_vec = np.zeros(n_y, dtype=np.float64)
    for j, (i1, i2, a) in enumerate(edge_terms):
        coef = W_E * a / n_e
        if beta and not static_coh:
            u1, v1 = pairs[i1]
            u2, v2 = pairs[i2]
            d1 = max(1, q_deg.get(u1, 1))
            d2 = max(1, q_deg.get(u2, 1))
            # THE UNIT'S AGREEMENT MULTIPLIES THE NODE CREDIT IT CARRIES.
            # The node's coherence is the AGREEMENT-WEIGHTED fraction of its
            # relations the mapping preserves:
            #     κ(i) = (1/deg(i))·Σ_{e at i} a(e,e')·y_e
            # so a unit that agrees poorly carries almost no node credit --
            # "a node pair with disagreeing edges is NOT a match, no matter
            # how similar the nodes look".
            #
            # [G1] THE CREDIT IT CARRIES IS THE NODE'S OWN CREDIT (A_i, with
            # the node masses and the anchor prior), not the bare signature
            # similarity: the old code used `sim(u,v)`, so the solver and
            # `objective_value` disagreed whenever the node masses or the
            # anchor prior were on.
            coef += beta * a * (A_full[i1] / d1 + A_full[i2] / d2)
        E_vec[j] = coef
    # the constant part of the unmatched term
    const = -W_U * (1.0 + 1.0)
    diagnostics["coherence_beta"] = beta

    # ---- the constraints ----
    rows = []
    cols = []
    data = []
    ub = []
    lb = []
    # the one-to-one rows are indexed by the node's POSITION in the canonical
    # node list (the node ids are not indices)
    q_index = {u: i for i, u in enumerate(nodes_q)}
    c_index = {v: i for i, v in enumerate(nodes_c)}
    # one-to-one, query side: Σ_v x[u,v] ≤ 1
    for u in nodes_q:
        for v in cand.get(u, ()):
            i = pair_index[(u, v)]
            rows.append(q_index[u])
            cols.append(i)
            data.append(1.0)
        ub.append(1.0)
        lb.append(-np.inf)
    # one-to-one, reference side: Σ_u x[u,v] ≤ 1
    for v in nodes_c:
        users = [pair_index[(u, v)] for u in nodes_q if (u, v) in pair_index]
        for i in users:
            rows.append(len(nodes_q) + c_index[v])
            cols.append(i)
            data.append(1.0)
        ub.append(1.0)
        lb.append(-np.inf)
    # the y linearisation: y ≤ x1, y ≤ x2, y ≥ x1 + x2 − 1
    base = len(nodes_q) + len(nodes_c)
    for j, (i1, i2, _a) in enumerate(edge_terms):
        y = n_x + j
        # y - x1 ≤ 0
        rows.append(base + 3 * j + 0)
        cols.append(y)
        data.append(1.0)
        rows.append(base + 3 * j + 0)
        cols.append(i1)
        data.append(-1.0)
        ub.append(0.0)
        lb.append(-np.inf)
        # y - x2 ≤ 0
        rows.append(base + 3 * j + 1)
        cols.append(y)
        data.append(1.0)
        rows.append(base + 3 * j + 1)
        cols.append(i2)
        data.append(-1.0)
        ub.append(0.0)
        lb.append(-np.inf)
        # x1 + x2 - y ≤ 1
        rows.append(base + 3 * j + 2)
        cols.append(i1)
        data.append(1.0)
        rows.append(base + 3 * j + 2)
        cols.append(i2)
        data.append(1.0)
        rows.append(base + 3 * j + 2)
        cols.append(y)
        data.append(-1.0)
        ub.append(1.0)
        lb.append(-np.inf)

    # [FIX 3 / JOINT] A selected node pair must be supported by at least
    # one selected unit (a preserved edge): x_i <= sum_{e incident i} y_e.
    # Without this, x can be 1 with no y, and the MILP maps every node that
    # has any similarity -- the edge term is then a rounding error.
    support = {}
    for j, (i1, i2, _a) in enumerate(edge_terms):
        support.setdefault(i1, []).append(j)
        support.setdefault(i2, []).append(j)
    support_rows = 0
    if bool(getattr(P, "GRADED_MCS_NODE_REQUIRES_EDGE", False)):
        # [G7] THE SUPPORT MUST BE A MEASURED RELATION, AND ONLY A NODE
        # WITH RELATIONS NEEDS ONE.
        #
        # The old constraint applied to EVERY matched pair: a node with no
        # incident edge term could not be matched at all (x_i <= 0), so a
        # truth that mapped such a node was INFEASIBLE.  And the support it
        # did accept was any incident edge -- but a missing parameter
        # scores 0.5 (neutral) and a Gaussian is never zero, so any
        # same-type neighbour counted as support and the constraint said
        # almost nothing.
        for i in range(n_x):
            js = [j for j in support.get(i, ()) if j in measured_terms]
            if not js:
                # a node with no MEASURED relation needs no support
                continue
            row = base + 3 * n_y + support_rows
            rows.append(row); cols.append(i); data.append(1.0)
            for j in js:
                rows.append(row); cols.append(n_x + j); data.append(-1.0)
            ub.append(0.0); lb.append(-np.inf)
            support_rows += 1

    A = sp.coo_matrix((data, (rows, cols)),
                       shape=(base + 3 * n_y + support_rows, n_x + n_y)).tocsr()
    constraints = LinearConstraint(A, np.array(lb), np.array(ub))
    bounds = Bounds(np.zeros(n_x + n_y), np.ones(n_x + n_y))
    # [STAGE 10] THE VARIABLES ARE BINARY.  `Bounds(0, 1)` alone leaves them
    # CONTINUOUS, and the solver then returns the LP RELAXATION -- which it
    # reports as "optimal" (status 0) while a feasible INTEGER point beats
    # it.  That is exactly what the truth-as-witness check caught: with lambda
    # set to the truth's own ratio the solver returned -0.599 where the truth
    # (a feasible point of the same subproblem) achieves 0.
    # With GRADED_MCS_LP_RELAXATION on (a DIAGNOSTIC mode only) the
    # integrality is dropped and the continuous optimum -- the proven bound --
    # is what comes back.
    lp_relaxation = bool(getattr(P, "GRADED_MCS_LP_RELAXATION", False))
    integrality = (np.zeros(n_x + n_y, dtype=int) if lp_relaxation
                   else np.ones(n_x + n_y, dtype=int))
    diagnostics["integrality"] = "continuous (the LP relaxation)"
    if not lp_relaxation:
        diagnostics["integrality"] = "binary"

    # [STAGE 10, ADDENDUM 4] THE RATIO OBJECTIVE (Dinkelbach).
    #
    # The measured discriminator is PER MATCH, not in total: the truth
    # preserves 2.30 relations per matched node, the solver's mapping 0.81 --
    # the same TOTAL.  So the objective maximises the evidence PER MATCH:
    #
    #     max  (W_N·Σ A_i·x_i + W_E·Σ a_e·y_e) / |M|
    #
    # which is fractional, and is solved by Dinkelbach's method on the SAME
    # ILP (no new variable, no new constraint, one outer loop):
    #
    #     lambda <- 0
    #     repeat:  solve  max  N(x) - lambda·|M|;  lambda <- N/|M|
    #     stop when lambda stabilises (the tolerance and the cap are
    #     pre-registered)
    #
    # The unmatched (coverage) term does not appear: under the ratio it is a
    # per-match constant, and the coverage preference is exactly what the
    # diagnosis says must go.
    ratio_mode = str(getattr(P, "GRADED_MCS_RATIO_MODE", "none"))
    ratio_tol = float(getattr(P, "GRADED_MCS_RATIO_TOLERANCE", 1e-5))
    ratio_iters = int(getattr(P, "GRADED_MCS_RATIO_MAX_ITERS", 12))
    # a FORCED lambda (the diagnostic): the Dinkelbach loop is skipped and the
    # single subproblem is solved with this lambda, and its value at the
    # solution is reported.  It is how the TRUTH is used as a feasible
    # witness: with lambda set to the truth's own ratio, the truth is a
    # feasible point of that subproblem and achieves exactly 0 there, so a
    # solver that returns LESS than 0 has missed a feasible point it had.
    lam_fixed = getattr(P, "GRADED_MCS_RATIO_LAMBDA_FIXED", None)
    diagnostics["ratio_mode"] = ratio_mode

    def _solve(vec: np.ndarray):
        try:
            r = milp(c=vec, constraints=constraints, bounds=bounds,
                      integrality=integrality,
                      options={"node_limit": node_limit,
                               # [AUDIT 2026-11-15 / RUN BUDGET] GRADED_MCS_MAX_SECONDS
                               # is wired into HiGHS as time_limit.  Before this
                               # fix the declared safety net was silently dropped
                               # and the solve ran unbounded in wall clock
                               # (2,178 CPU-sec spent on one arms run, no result).
                               "time_limit": max_seconds,
                               "presolve": True, "mip_rel_gap": 0.0})
            return r, str(r.status)
        except Exception as exc:                   # noqa: BLE001
            diagnostics["solver_error"] = "%s: %s" % (type(exc).__name__, exc)
            return None, "error"

    t_solve = time.time()
    status = "unknown"
    result = None
    if ratio_mode == "none":
        obj[:n_x] = -(A_vec + bonus)
        obj[n_x:] = -E_vec
        result, status = _solve(obj)
    elif lam_fixed is not None:
        # the forced-lambda diagnostic: ONE subproblem, and its value at the
        # solution
        lam = float(lam_fixed)
        if ratio_mode == "relations":
            obj[:n_x] = -lam
        else:
            obj[:n_x] = -(A_vec - lam)
        obj[n_x:] = -E_vec
        result, status = _solve(obj)
        diagnostics["ratio_lambda"] = lam
        diagnostics["ratio_iterations"] = 1
        diagnostics["ratio_forced"] = True
        if result is not None and getattr(result, "x", None) is not None:
            xs = result.x
            m = sum(1 for i in range(n_x) if xs[i] > 0.5)
            if ratio_mode == "relations":
                num = sum(E_vec[j] for j in range(n_y) if xs[n_x + j] > 0.5)
            else:
                num = (sum(A_vec[i] for i in range(n_x) if xs[i] > 0.5)
                       + sum(E_vec[j] for j in range(n_y) if xs[n_x + j] > 0.5))
            diagnostics["ratio_subproblem_value"] = num - lam * m
            diagnostics["ratio_match_count"] = m
        if getattr(result, "mip_node_count", None) is not None:
            diagnostics["mip_node_count"] = int(result.mip_node_count)
        if getattr(result, "mip_gap", None) is not None:
            diagnostics["mip_gap"] = float(result.mip_gap)
        if getattr(result, "mip_dual_bound", None) is not None:
            diagnostics["mip_dual_bound"] = float(result.mip_dual_bound)
    else:
        # the Dinkelbach loop
        lam = 0.0
        history = []
        it = 0
        for it in range(1, ratio_iters + 1):
            if ratio_mode == "relations":
                # the relation evidence per match (the node term absent)
                obj[:n_x] = -lam
            else:
                # the whole objective per match
                obj[:n_x] = -(A_vec - lam)
            obj[n_x:] = -E_vec
            result, status = _solve(obj)
            if result is None or getattr(result, "x", None) is None:
                break
            xs = result.x
            m = sum(1 for i in range(n_x) if xs[i] > 0.5)
            num = 0.0
            if ratio_mode == "relations":
                num = sum(E_vec[j] for j in range(n_y) if xs[n_x + j] > 0.5)
            else:
                num = (sum(A_vec[i] for i in range(n_x) if xs[i] > 0.5)
                       + sum(E_vec[j] for j in range(n_y) if xs[n_x + j] > 0.5))
            lam_new = num / max(1, m)
            history.append(lam_new)
            if abs(lam_new - lam) <= ratio_tol:
                lam = lam_new
                break
            lam = lam_new
        diagnostics["ratio_lambda"] = lam
        diagnostics["ratio_iterations"] = it
        diagnostics["ratio_history"] = history
        if len(history) > 1:
            diagnostics["ratio_converged"] = (
                abs(history[-1] - history[-2]) <= ratio_tol)
        else:
            diagnostics["ratio_converged"] = False
    solve_time = time.time() - t_solve
    diagnostics["status"] = status
    diagnostics["solve_seconds"] = solve_time
    diagnostics["total_seconds"] = time.time() - t0
    diagnostics["constraints"] = int(A.shape[0])

    correspondence: Dict[int, int] = {}
    node_term = edge_term = 0.0
    if result is not None and getattr(result, "x", None) is not None:
        xs = result.x
        for i, (u, v) in enumerate(pairs):
            if xs[i] > 0.5:
                correspondence[u] = v
        # the preserved relations, for the trace
        preserved = []
        for j, (i1, i2, a) in enumerate(edge_terms):
            if xs[n_x + j] > 0.5:
                u, v = pairs[i1]
                u2, v2 = pairs[i2]
                preserved.append((u, u2, v, v2, a))
        diagnostics["preserved_relations"] = len(preserved)
        # the coherence of the SELECTED pairs: the fraction of each matched
        # node's relations the mapping preserves coherently.  This is the
        # design's own notion of the unit, and it is what the coherence
        # charge (beta) acts on.
        if beta:
            deg = {}
            for (u, u2, _et) in Q.edges:
                deg[u] = deg.get(u, 0) + 1
                deg[u2] = deg.get(u2, 0) + 1
            kept = {}
            for (u, u2, v, v2, _a) in preserved:
                kept.setdefault(u, 0)
                kept[u] += 1
                kept.setdefault(u2, 0)
                kept[u2] += 1
            vals = [kept.get(u, 0) / max(1, deg.get(u, 1))
                    for u in correspondence]
            diagnostics["mean_coherence"] = (float(sum(vals) / len(vals))
                                             if vals else 0.0)
        if getattr(result, "mip_gap", None) is not None:
            diagnostics["mip_gap"] = float(result.mip_gap)
        if getattr(result, "mip_dual_bound", None) is not None:
            diagnostics["mip_dual_bound"] = float(result.mip_dual_bound)
        if getattr(result, "mip_node_count", None) is not None:
            diagnostics["mip_node_count"] = int(result.mip_node_count)
        if getattr(result, "fun", None) is not None:
            # the solver's own value, and the objective it corresponds to
            # (`objective_value` == -c.x + const; the const is the constant
            # part of the unmatched term)
            diagnostics["solver_value"] = float(result.fun)
            diagnostics["solver_objective_value"] = (
                -float(result.fun) + const)
    unmatched = (len(nodes_q) + len(nodes_c)) - 2 * len(correspondence)
    # [G1 / LEAD] THE REPORTED SCORE IS THE OBJECTIVE.  It used to be
    # recomputed from the correspondence with a formula that ignored beta,
    # the static coherence and the node masses -- the very things the MILP
    # maximises -- so the reported score was NOT the objective the solver
    # maximised, and the truth could not be compared with the optimum on
    # one scale.  It is `objective_value` now: the same function.
    obj = objective_value(correspondence, Q, C, q_sigs, c_sigs, beta=beta,
                          node_mass=node_mass)
    node_term = float(obj["node"])
    edge_term = float(obj["edge"])
    unmatched_term = float(obj["unmatched"])
    diagnostics["terms"] = {"node": node_term, "edge": edge_term,
                            "unmatched": unmatched_term,
                            "unmatched_nodes": unmatched}
    score = float(obj["objective"])
    diagnostics["score"] = score
    if solve_time >= max_seconds:
        diagnostics["truncated"] = True
    return GradedMCSResult(score, correspondence, diagnostics)


def match_evidence(Q: Any, C: Any,
                   q_sigs: Dict[int, Tuple[str, Any]],
                   c_sigs: Dict[int, Tuple[str, Any]],
                   **kwargs: Any) -> Tuple[float, Dict[int, int]]:
    """The `evidence_score` contract: `(score, correspondence)`.

    The score is the objective value, put on the [0, 1] scale the old matcher
    uses: the objective's maximum is `W_N + W_E` (the node term at 1, the
    edge term at 1, the unmatched term at 0), so it is divided by that.

    [G6] THE SCORE IS CLAMPED TO [0, 1].  The objective can be NEGATIVE
    (the unmatched term is negative and can outweigh the credit), so the
    division alone produced a score below 0 while the contract claims a
    [0, 1] score.
    """
    res = graded_mcs(Q, C, q_sigs, c_sigs, **kwargs)
    scale = (float(getattr(P, "GRADED_MCS_NODE_WEIGHT", 1.0))
             + float(getattr(P, "GRADED_MCS_EDGE_WEIGHT", 1.0)))
    if scale <= 0:
        scale = 1.0
    return max(0.0, min(1.0, res.score / scale)), res.correspondence
