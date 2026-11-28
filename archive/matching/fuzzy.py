"""matching/fuzzy.py -- unified fuzzy graded structural matcher (Phase 5).

One entry point that composes the Phase 5 layers:

  * node similarity       : the population-overlap similarity already produced
                            by ``TypedSignature.similarity`` (chance-corrected
                            Pearson overlap against the uniform null);
  * global alignment      : ``matching/alignment`` global-consistency
                            alignment seeded from the approximate Hungarian,
                            with an explicit unmatched cost and symmetric
                            coverage;
  * calibration           : ``matching/calibration`` null-calibrated,
                            floor-free, background-weighted scoring;
  * consensus             : ``matching/consensus`` multi-vote consensus with
                            stored disagreement (``fuzzy_consensus``);
  * search                : ``matching/search`` best-first extension with
                            anytime budgets (available for template growth).

The pairwise result is a ``FuzzyMatch`` carrying every component separately, so
a caller (or an ablation) can read or recombine them; ``score`` is the documented
default combination.

Pure and deterministic. No corpus state, no scoring tuned against an outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import parameters as P
from core.graph_types import Subgraph
from core.types import UNDIRECTED_EDGE_TYPES

from .calibration import NullModel, mean_pairwise_similarity
from .structural_similarity import (
    _lookup_edge_params,
    _lookup_edge_params_oriented,
    _pair_edge_sim_typed,
    _sig_sim,
)

SignatureProvider = object  # (node_id) -> Optional[Tuple[lens_type, TypedSignature]]


@dataclass(frozen=True)
class FuzzyMatch:
    """Components of a fuzzy graded match between two subgraphs."""
    mapping: Dict[int, int]
    quality: Dict[int, float]
    node_coverage: float              # symmetric Dice count 2|M|/(|A|+|B|)
    node_coverage_weighted: float     # similarity-weighted Dice
    node_coverage_calibrated: float   # weighted Dice calibrated above chance
    node_quality_mean: float
    chance: float                     # null expected weighted coverage
    systematicity: float              # conjunctive typed-edge preservation
    fwd_edge_cov: float
    bwd_edge_cov: float
    edge_coverage: float              # min(fwd, bwd)
    edge_param_mean: float            # mean graded edge-param sim over kept edges
    unmatched_a: Tuple[int, ...]
    unmatched_b: Tuple[int, ...]
    score: float
    budget_exhausted: bool = False


def _edge_param_mean(
    sg_a: Subgraph,
    sg_b: Subgraph,
    mapping: Mapping[int, int],
    use_confidence: bool,
) -> float:
    if not sg_a.edges and not sg_b.edges:
        return 1.0  # both sides edgeless: genuinely vacuous
    if not sg_a.edges or not sg_b.edges:
        # One side HAS structure that the other cannot confirm.  This is
        # not "no evidence to contradict"; it is total absence of agreement.
        return 0.0
    b_und = {
        (min(s, d), max(s, d), et)
        for (s, d, et) in sg_b.edges if et in UNDIRECTED_EDGE_TYPES
    }
    b_dir = {(s, d, et) for (s, d, et) in sg_b.edges if et not in UNDIRECTED_EDGE_TYPES}
    terms = []
    for (s, d, et) in sorted(sg_a.edges, key=lambda t: (min(t[0], t[1]), max(t[0], t[1]), getattr(t[2], "value", 0), t[0], t[1])):
        ms = mapping.get(s)
        md = mapping.get(d)
        if ms is None or md is None:
            continue
        if et in UNDIRECTED_EDGE_TYPES:
            lo, hi = (ms, md) if ms < md else (md, ms)
            present = (lo, hi, et) in b_und
        else:
            present = (ms, md, et) in b_dir
        if not present:
            continue
        ep_a, flip_a = _lookup_edge_params_oriented(sg_a.edge_params, s, d, et)
        ep_b, flip_b = _lookup_edge_params_oriented(sg_b.edge_params, ms, md, et)
        # [LEAD] An edge's parameters may be DIRECTIONAL, and the two sides
        # must be read in the SAME convention: `_pair_edge_sim_typed` decides
        # whether to reverse the second side from the ORDER OF THE NODE IDS
        # it is handed, which is independent of the order the parameters are
        # stored in, so the same relation was compared forwards or backwards
        # by accident.  The orientation each side's parameters are stored in
        # is resolved once, from the store, and the endpoints are handed over
        # in it.
        if flip_a:
            s, d = d, s
        if flip_b:
            ms, md = md, ms
        terms.append(_pair_edge_sim_typed(
            s, d, et, ep_a, ms, md, et, ep_b, use_confidence,
        ))
    return math.fsum(terms) / len(terms) if terms else 0.0


def fuzzy_match(
    sg_a: Subgraph,
    sg_b: Subgraph,
    sig_provider,
    *,
    unmatched_cost: Optional[float] = None,
    systematicity_weight: Optional[float] = None,
    rounds: Optional[int] = None,
    max_pair_evals: Optional[int] = None,
    edge_weight: Optional[float] = None,
    use_confidence: bool = False,
) -> FuzzyMatch:
    """Fuzzy graded match of `sg_a` onto `sg_b`.

    Node similarity is ``_sig_sim`` (which routes through the
    population-overlap path when both signatures carry a population). The
    alignment is global-consistency refinement seeded from the approximate
    Hungarian; node coverage is calibrated against the null expected coverage
    (the mean pairwise similarity).

    [F2] THE LENS TYPE IS PART OF THE CANDIDATE.  The old code kept only
    the signature -- ``(sig_provider(u) or (None, None))[1]`` -- and dropped
    the lens type with it, so a node of one lens type could be paired with a
    node of another whenever their signatures happened to be similar, while
    every other matcher in the tree (joint_mcs, relation_walker,
    structural_similarity) pairs within a lens type.  The matrix is
    therefore built WITHIN the lens types: a cross-type cell is 0.0 (not
    comparable) and is never evaluated.

    [F3] THE RELATIONAL TERM IS A MULTIPLIER, NOT AN ADDITIVE FLOOR.  The
    score is ``node_cal · ((1 − w) + w · edge_term)``: there is no ADDITIVE
    floor (a zero node term gives a zero score), but the node term is
    paid at ``1 − w`` when the relations agree with nothing.  The old
    docstring claimed "NO 0.85 floor" and was read as "the relations can
    never discount the node term"; they can, and they do.
    """
    if unmatched_cost is None:
        unmatched_cost = float(getattr(P, "MATCH_UNMATCHED_COST", 0.0))
    if systematicity_weight is None:
        systematicity_weight = float(getattr(P, "MATCH_SYSTEMATICITY_WEIGHT", 0.5))
    if rounds is None:
        rounds = int(getattr(P, "MATCH_ALIGN_ROUNDS", 4))
    if max_pair_evals is None:
        max_pair_evals = getattr(P, "MATCH_ALIGN_MAX_PAIR_EVALS", 200000)
    if edge_weight is None:
        edge_weight = float(getattr(P, "MATCH_EDGE_WEIGHT", 0.5))

    nodes_a = sorted(sg_a.nodes)
    nodes_b = sorted(sg_b.nodes)

    # The similarity matrix is built once here so the null (mean pairwise
    # similarity) and the alignment share it. Indexed [i][j] over the sorted
    # node lists.
    #
    # [F2] the lens type is KEPT: the candidate population is the
    # same-lens-type pairs, so the matrix is blocked by lens type and a
    # cross-type cell is 0.0 (not comparable) rather than being computed
    # and possibly matched.
    sigs_a: List[Tuple[Optional[str], Any]] = []
    for u in nodes_a:
        info = sig_provider(u)
        sigs_a.append((info[0], info[1]) if info else (None, None))
    sigs_b: List[Tuple[Optional[str], Any]] = []
    for v in nodes_b:
        info = sig_provider(v)
        sigs_b.append((info[0], info[1]) if info else (None, None))
    matrix: List[List[float]] = [
        [0.0] * len(nodes_b) for _ in nodes_a
    ]
    for i, (lt_a, sa) in enumerate(sigs_a):
        if lt_a is None or sa is None:
            continue
        for j, (lt_b, sb) in enumerate(sigs_b):
            if lt_b is None or sb is None or lt_b != lt_a:
                continue
            matrix[i][j] = _sig_sim(sa, sb)

    from .alignment import global_consistency_alignment
    from core.frontend_contract import frontend_contract_active
    edge_similarity = None
    if frontend_contract_active():
        from matching.joint_mcs import edge_pair_credit
        def edge_similarity(x, y, u, v, et):
            return edge_pair_credit(sg_a, sg_b, x, y, u, v, et)
    al = global_consistency_alignment(
        nodes_a,
        nodes_b,
        None,
        edges_a=sg_a.edges,
        edges_b=sg_b.edges,
        unmatched_cost=unmatched_cost,
        systematicity_weight=systematicity_weight,
        rounds=rounds,
        max_pair_evals=max_pair_evals,
        sim=matrix,
        edge_similarity=edge_similarity,
        # [A5] the geometry the occlusion credit needs: an unmapped partner
        # whose relation is occluded by a T-junction is not a contradiction.
        anchors_a=getattr(sg_a, "anchor_of_node", None),
        anchors_b=getattr(sg_b, "anchor_of_node", None),
        junctions_a=getattr(sg_a, "t_junctions", None) or (),
        junctions_b=getattr(sg_b, "t_junctions", None) or (),
        budget_refinement_only=bool(
            getattr(P, "MATCH_ALIGN_BUDGET_REFINEMENT_ONLY", True)),
    )

    # Registered off-correspondence null: selected pairs are evidence, not
    # null alternatives. With no alternatives the null is unobserved; use
    # zero as the reference and discount evidence by one half below.
    #
    # [F2] the alternatives are the CANDIDATE pairs (the same-lens-type
    # ones).  A cross-type pair is not a candidate at all, so it is not an
    # alternative either -- counting the zero cells of the blocked matrix
    # would make the null smaller than the null of the candidate set.
    alternatives = []
    for i, (lt_a, _sa) in enumerate(sigs_a):
        for j, (lt_b, _sb) in enumerate(sigs_b):
            if lt_a is None or lt_b is None or lt_a != lt_b:
                continue
            if al.mapping.get(nodes_a[i]) != nodes_b[j]:
                alternatives.append(matrix[i][j])
    chance = sum(alternatives) / len(alternatives) if alternatives else 0.0
    ntot = len(nodes_a) + len(nodes_b)
    weighted = 2.0 * sum(al.quality.values()) / ntot if ntot else 0.0
    node_cal = NullModel(chance).calibrate(weighted)
    if not alternatives and nodes_a and nodes_b:
        node_cal *= 0.5
    quality_mean = (
        sum(al.quality.values()) / len(al.quality) if al.quality else 0.0
    )
    edge_param_mean = _edge_param_mean(sg_a, sg_b, al.mapping, use_confidence)
    # No relations on BOTH sides is a neutral relational factor, not a
    # half-credit discount. Otherwise retain the shipped weighted blend of
    # node evidence and relational coverage.
    if not sg_a.edges and not sg_b.edges:
        edge_term = 1.0
        score = node_cal
    else:
        edge_term = al.edge_coverage() * edge_param_mean
        score = node_cal * ((1.0 - edge_weight) + edge_weight * edge_term)

    return FuzzyMatch(
        mapping=dict(al.mapping),
        quality=dict(al.quality),
        node_coverage=al.node_coverage,
        node_coverage_weighted=weighted,
        node_coverage_calibrated=node_cal,
        node_quality_mean=quality_mean,
        chance=chance,
        systematicity=al.systematicity,
        fwd_edge_cov=al.fwd_edge_cov,
        bwd_edge_cov=al.bwd_edge_cov,
        edge_coverage=al.edge_coverage(),
        edge_param_mean=edge_param_mean,
        unmatched_a=al.unmatched_a,
        unmatched_b=al.unmatched_b,
        score=score,
        budget_exhausted=al.budget_exhausted,
    )


def fuzzy_consensus(
    reference: Subgraph,
    exemplars: Sequence[Subgraph],
    sig_provider,
    *,
    identity_slots: Sequence[int] = (),
    cycles: Optional[Mapping[int, float]] = None,
    voter_weights: Optional[Sequence[float]] = None,
    min_support: Optional[float] = None,
):
    """Generalise a template from exemplars aligned onto a reference.

    Each exemplar is fuzzy-matched onto the reference; for every reference node
    the matched exemplar's quantized slot values become one vote. Returns
    ``(per_node_consensus, edge_support)`` where ``per_node_consensus`` maps a
    reference node to a ``VoteConsensus`` and ``edge_support`` maps each
    reference edge key to its vote share, filtered by ``min_support`` so that no
    edge is invented.
    """
    from .consensus import (
        generalize_edges,
        support_weights,
        vote_consensus,
    )

    if min_support is None:
        min_support = float(getattr(P, "MATCH_MIN_CONSENSUS_SUPPORT", 0.5))

    ref_nodes = sorted(reference.nodes)
    slot_values: Dict[int, List[List[float]]] = {n: [] for n in ref_nodes}
    pop_values: Dict[int, List[Tuple[Tuple[float, ...], ...]]] = {n: [] for n in ref_nodes}
    edge_presence: List[Dict[object, float]] = []

    for ex in exemplars:
        fm = fuzzy_match(ex, reference, sig_provider)
        # fm.mapping: exemplar node -> reference node. Invert to reference -> ex.
        by_ref: Dict[int, int] = {rv: ev for ev, rv in fm.mapping.items()}
        for rv in ref_nodes:
            info = sig_provider(by_ref[rv]) if rv in by_ref else None
            sig = info[1] if info else None
            slot_values[rv].append(list(sig.quantized) if sig else [])
            pop_values[rv].append(getattr(sig, "population", ()) if sig else ())
        # Which reference edges did this exemplar preserve?
        # [F4] THE EDGE SETS ARE BUILT ONCE PER EXEMPLAR.  They were
        # rebuilt inside the loop over the reference's edges -- twice, once
        # as the undirected set and once as `set(ex.edges)` -- so the cost
        # was O(|E_ref| x |E_ex|) instead of O(|E_ref| + |E_ex|).
        ex_und = {
            (min(a, b), max(a, b), c) for (a, b, c) in ex.edges
            if c in UNDIRECTED_EDGE_TYPES
        }
        ex_dir = set(ex.edges)
        present: Dict[object, float] = {}
        for (s, d, et) in reference.edges:
            es, ed = by_ref.get(s), by_ref.get(d)
            if es is None or ed is None:
                continue
            if et in UNDIRECTED_EDGE_TYPES:
                want = (min(es, ed), max(es, ed), et)
                present[(s, d, et)] = 1.0 if want in ex_und else 0.0
            else:
                present[(s, d, et)] = 1.0 if (es, ed, et) in ex_dir else 0.0
        edge_presence.append(present)

    per_node = {}
    # [B2] THE VOTER WEIGHTS STAY ALIGNED WITH THE VOTERS.  The rows used
    # to be COMPACTED (a voter that did not fill a slot was dropped) while
    # the weights stayed indexed by exemplar, so the surviving values were
    # zipped with the FIRST weights -- the weights belonged to other
    # voters.  A compaction is only sound with uniform weights, so with a
    # non-uniform set the abstention mask is kept (one entry per voter),
    # which is what MATCH_CONSENSUS_COUNT_ABSTENTIONS does.
    count_abstentions = bool(getattr(
        P, "MATCH_CONSENSUS_COUNT_ABSTENTIONS", False))
    weights_uniform = (voter_weights is None
                       or len({float(w) for w in voter_weights}) <= 1)
    for rv in ref_nodes:
        cols = slot_values[rv]
        n_slots = max((len(c) for c in cols), default=0)
        # [PHASE2 D7] Keep the absence mask (behind
        # MATCH_CONSENSUS_COUNT_ABSTENTIONS, default OFF = the
        # historical behaviour): a missing exemplar is an ABSTENTION,
        # not a shorter voter list, and the weights stay aligned to the
        # exemplar index.
        if count_abstentions or not weights_uniform:
            rows = [
                [c[i] if i < len(c) else None for c in cols]
                for i in range(n_slots)
            ]
            n_voters = len(exemplars)
        else:
            rows = [
                [c[i] for c in cols if i < len(c)]
                for i in range(n_slots)
            ]
            n_voters = None
        per_node[rv] = vote_consensus(
            rows,
            identity_slots=identity_slots,
            cycles=cycles,
            weights=voter_weights,
            populations=pop_values[rv],
            n_voters=n_voters,
        )

    support = support_weights(edge_presence, voter_weights)
    kept_edges = generalize_edges(support, min_support)
    return per_node, kept_edges
