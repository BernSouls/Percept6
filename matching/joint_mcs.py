"""Unsupervised, balanced, graded one-to-one graph correspondence.

This score intentionally uses unweighted node and edge masses: unlike the
historical mcs_similarity its node credit is graded and its matched-edge mass
is not amplified by axis-count facilitation. Correspondence is a bounded
fixed-point approximation, not a certificate of a globally optimal MCS.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import parameters as P
from core.schema import (edge_categorical_axes, edge_graded_axes,
                         edge_params_incomparable)
from core.types import UNDIRECTED_EDGE_TYPES
from matching.alignment import global_consistency_alignment
from matching.structural_similarity import (_lookup_edge_params,
                                             _lookup_edge_params_oriented,
                                             _pair_edge_sim_typed)


def _edge_type_key(t):
    """Total order over enum edge types and named U30 category strings."""
    value = getattr(t, "value", None)
    return (0, value, "") if value is not None else (1, 0, str(t))


def edge_key(e):
    a,b,t=e
    return (min(a,b),max(a,b),_edge_type_key(t),a,b)


def edge_pair_credit(a, b, x, y, u, v, et, edge_weight_fn=None):
    """Typed edge credit; missing parameters yield presence, not physics.

    [LEAD / UNDIRECTED REVERSAL] An edge's parameters may be DIRECTIONAL, and
    a comparison must read both sides in the SAME convention.
    `_pair_edge_sim_typed` decides whether to reverse the second side from
    the ORDER OF THE NODE IDS it is handed -- not from the order the
    parameters are stored in.  The two are independent, so the pair was
    reversed or not by accident: the same undirected relation read two
    different ways depending on which end the caller happened to name
    first.  The ORIENTATION each side's parameters are stored in is
    therefore resolved once, from the store, and the endpoints are handed
    over in it: the reversal inside `_pair_edge_sim_typed` then fires
    exactly when the two sides' conventions differ.

    [B3] ``edge_weight_fn`` is threaded through for the weight-aware path;
    the returned credit is the typed SIMILARITY (the min-weight numerator
    credit is applied by the caller's matched-pairs loop).
    """
    p, flip_a = _lookup_edge_params_oriented(a.edge_params, x, y, et)
    q, flip_b = _lookup_edge_params_oriented(b.edge_params, u, v, et)
    if edge_params_incomparable(p, q):
        # [B7] INCOMPARABLE IS NOT CONTRADICTORY.  The typed comparison would
        # score this pair 0.0 -- "no common ontology" read as "these two
        # relations disagree".  A relation with no counterpart is not
        # evidence AGAINST a match, so it is not compared at all.
        return None
    if flip_a:
        x, y = y, x
    if flip_b:
        u, v = v, u
    # [PHASE 3 / F3] I5 must be JOINT: a presence-only relation is a count,
    # not physics, so it earns no relational agreement inside the alignment
    # either.  The flag is read here, where the aligner asks for the edge
    # weight, so gating it can change WHO PAIRS.
    if bool(getattr(P, "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE", False)):
        if p is None or q is None:
            return 0.0
        if edge_graded_axes(p, q) == 0:
            # [THREE-WAY EVIDENCE] a categorical agreement (a T junction is
            # a T junction) is physical organization, not a bare count.
            cat = edge_categorical_axes(p, q)
            if cat <= 0:
                return 0.0
            if bool(getattr(P, "MATCH_CATEGORICAL_AGREEMENT_GRADED", False)):
                # [TUNE / EDGES] GRADED, not binary: the credit is the
                # fraction of identity axes that agree, so a two-of-three
                # categorical match is not the same evidence as a complete
                # one.
                slots = getattr(p, "identity_slots", None) or ()
                total = len(slots) if slots else 1
                return min(1.0, cat / float(max(1, total)))
            return 1.0
    elif p is None and q is None:
        # [J2] BOTH sides carry no measurement: the relation exists on both
        # sides and nothing else is known about it.  That is the
        # historical full unit of evidence.
        return 1.0
    elif p is None or q is None:
        # [J2] A MISSING MEASUREMENT IS NOT A PERFECT ONE.  One side's
        # parameters are unknown, so the pair cannot be compared at all;
        # reading it as 1.0 made an unmeasured relation look like a perfect
        # match.  (F6's own contract: "a pair with params on ONE side only
        # stays 0.0 under either setting".)
        return 0.0
    return _pair_edge_sim_typed(x, y, et, p, u, v, et, q, False)


#: [MISSION 11] counters for the placement-aware pair score.  They are per process
#: (a read aggregates them) and they exist so a dark run can prove it was not
#: vacuous: a score that is "placement aware" but judged nothing must be visible.
PLACEMENT_SCORE_STATS = {"pairs_scored": 0, "pairs_no_sides": 0,
                         "mapped_nodes": 0, "judged_nodes": 0,
                         "nonzero_evidence": 0, "strict_failures": 0}


def _reset_placement_score_stats():
    for k in PLACEMENT_SCORE_STATS:
        PLACEMENT_SCORE_STATS[k] = 0


def placement_evidence(mapping, side_a, side_b, fit, pfit, a_e, a_g):
    """(E, n_judged, share_judged) -- the coverage-neutral placement evidence.

    Written down before the fit (see parameters/_text.py::PLACEMENT_SCORE_EVIDENCE):

        E = sum over JUDGED mapped pairs of (plus_score(pair) - 0.5) / (n_judged + k)

    A judged pair is one with at least one slot the frozen estimator can check.
    Each pair is centred at the estimator's own 0.5, so "no evidence" is 0 and a
    negative E is evidence against.  The denominator is the number judged plus the
    shrinkage k, so E is a mean quality and not a count of correspondences.

    Truth-free: it reads the mapping the matcher produced and the two payload
    sides; no ground truth, no absolute position.
    """
    from matching import placement_llr as L

    # [MISSION 12 / step 2] THE FIT-CHOSEN READING OF E.  With
    # PLACEMENT_E_NORMALISED the pair contributes the MEAN over the slots it
    # actually judged (`plus_score(normalise=True)`) instead of the sum over the
    # cap, and a pair that judged fewer than PLACEMENT_E_MIN_JUDGED slots is
    # REFUSED (0.5 exactly, i.e. 0 to E) rather than read on one measurement.
    # Both flags are dark, so this function is bit-identical with them off.
    _norm = bool(getattr(P, "PLACEMENT_E_NORMALISED", False))
    _min_j = int(getattr(P, "PLACEMENT_E_MIN_JUDGED", 0) or 0)
    judged = 0
    total = 0.0
    mapped = 0
    for u, v in sorted(mapping.items()):
        mapped += 1
        slots = L.slots_for_candidate(u, v, mapping, side_a.incident,
                                      side_a.index, side_b.index)
        if not slots:
            continue
        judged += 1
        term = L.plus_score(u, v, mapping, side_a.incident, side_a.index,
                            side_b.index, fit, pfit, a_e, a_g,
                            normalise=_norm, min_judged=_min_j)
        total += float(term) - 0.5
    k = float(getattr(P, "PLACEMENT_SCORE_SHRINK", 8.0))
    evidence = total / (judged + k) if judged else 0.0
    share = (judged / mapped) if mapped else 0.0
    return evidence, judged, share


@dataclass(frozen=True)
class JointMCS:
    score: float
    mapping: dict
    matched_edges: int
    graded_edges: int
    graded_edge_similarity_sum: float
    presence_only_edges: int
    budget_exhausted: bool
    #: [PHASE 3] the mass the presence-only (identity-only) edges WOULD have
    #: earned as evidence.  It is reported so the mechanism is visible: with
    #: MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE the score does not count
    #: it, but the trace says how much was withheld.
    presence_only_credit: float = 0.0
    #: [PHASE 3] the numerator's two parts and their normalizations.  The
    #: node term and the edge term are what the density arm separates, so
    #: they are reported whether the arm is on or off.
    node_credit: float = 0.0
    edge_credit: float = 0.0
    node_term: float = 0.0
    edge_term: float = 0.0
    denominator: float = 0.0
    #: [MISSION 11] the placement-aware pair score.  `score_shipped` is the number
    #: the mission's baseline arms report (and, with the flag off, equals `score`
    #: exactly); `score` is `score_shipped + w * placement_evidence` once the dark
    #: flag is on.  The three diagnostics are exposed so a read can say how much of
    #: the score was placement and how many pairs it could actually judge.
    score_shipped: float = 0.0
    placement_evidence: float = 0.0
    placement_judged: int = 0
    placement_share_judged: float = 0.0
    @property
    def mean_graded_edge_similarity(self):
        return self.graded_edge_similarity_sum / self.graded_edges if self.graded_edges else None


def _as_image_graph(subgraph: Any, sigs: Any) -> Any:
    """A minimal ImageGraph-shaped view of (subgraph, sigs).

    The placement payload builder needs `.subgraph` and `.sigs` and nothing else;
    production call sites already hold the two pieces separately, so this avoids
    forcing every caller to build a wrapper.
    """
    from types import SimpleNamespace
    return SimpleNamespace(subgraph=subgraph, sigs=dict(sigs or {}))


def _directional_joint_mcs(a, b, sigs_a, sigs_b, addrs_a=None, addrs_b=None,
                           address_k=None, placement_images=None):
    """Two graphs in; no labels, training sets, filtering or learned masses.

    Each matched node contributes its signature similarity; each preserved
    same-type edge contributes its parameter similarity, or one unit of
    topology-only presence if either side has no parameters. Identical
    predicates with no shared graded axes are never called a physical reading.
    The numerator is twice the total matched node/edge credit; denominator is
    the number of nodes plus edges on BOTH sides.

    [J4] THE EDGE WEIGHT IS READ IN THE POOLED BRANCH.  The pooled score is
    the offline pooled form, term for term:
    ``2·(node_credit + w·edge_credit) / (na + nb + edge_mass)`` with
    ``w = MATCH_EDGE_WEIGHT``.  The pooled branch used to ignore ``w``
    entirely, so the production number was the offline w=1.0 cell of the
    tuned surface while the SELECTED cell was a different number: no cell
    of the surface was production-equivalent and the selected configuration
    could not be deployed.  The separate-term arm is the OTHER offline form
    (``2·((1−w)·node_term + w·edge_term)``, the two terms on their own
    masses), and it stays behind its own flag.

    [PHASE 3] TWO ARMS, both behind a flag whose default is the historical
    behaviour:

    MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE -- a matched relation with
    no measured axes on either side is a COUNT, not physics (the tree's own
    rule, core/schema.py::edge_graded_axes).  A CATEGORICAL agreement (the
    same topology class) is physical organization and still earns credit.
    In the pooled branch the two edges it spans leave the denominator;
    under MATCH_SEPARATE_TERM_NORMALIZATION the edge term is charged against
    ALL edges on both sides, so an unpreserved relation always costs.

    MATCH_SEPARATE_TERM_NORMALIZATION -- the node term and the edge term
    are normalized by their OWN masses and combined with
    MATCH_EDGE_WEIGHT, so the held-out graph's edge density cannot
    dominate the pooled denominator (measured: the class-matrix column
    mean correlates -0.995 with the held-out edge count).
    """
    from memory.cores import _sig_similarity
    not_evidence = bool(getattr(P, "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE", False))
    separate = bool(getattr(P, "MATCH_SEPARATE_TERM_NORMALIZATION", False))
    exclude_isolated = bool(getattr(
        P, "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE", False))
    # [B3] Edge-weight credit mode: matched pairs credit min(weight on each
    # side) in the numerator; the denominators carry the raw weights.
    min_credit = bool(getattr(P, "MATCH_EDGE_MIN_CREDIT", False))

    def _edge_weight(graph, x, y, et):
        """Graded weight of one edge: EdgeParams.confidence (default 1.0)."""
        p = _lookup_edge_params(getattr(graph, "edge_params", None), x, y, et)
        return float(getattr(p, "confidence", 1.0)) if p is not None else 1.0

    # [B3] edge_weight_fn threaded through edge_pair_credit and the alignment.
    # The alignment's internal weight-aware relational path is gated by its
    # own flag (EDGE_MATCH_USE_CONFIDENCE); this wiring makes the weight
    # available without touching that pin.
    def _align_edge_weight(x, y, et):
        return _edge_weight(a, x, y, et)
    # [B8] EXCLUSION IS A DOWN-WEIGHT, NOT A DELETION.  Removing isolated
    # nodes from na/nb made an edgeless graph pair produce an EMPTY alignment:
    # every class then scored 0 and argmax fell to the alphabetical tie-break
    # (the "no-edges 0.1250" artefact).  The correspondence domain is now the
    # SAME for every arm; exclusion is the declared isolated-node weight, and
    # a floor keeps the node term measurable when neither graph has edges.
    na, nb = sorted(a.nodes), sorted(b.nodes)
    if exclude_isolated:
        isolated_floor = float(getattr(P, "MATCH_EXCLUDE_ISOLATED_FLOOR", 0.5))
        isolated_w = min(float(getattr(P, "MATCH_ISOLATED_NODE_WEIGHT", 1.0)),
                         isolated_floor)
    else:
        isolated_w = float(getattr(P, "MATCH_ISOLATED_NODE_WEIGHT", 1.0))
    deg_a = {}
    for s0, d0, _et0 in a.edges:
        deg_a[s0] = deg_a.get(s0, 0) + 1
        deg_a[d0] = deg_a.get(d0, 0) + 1
    deg_b = {}
    for s0, d0, _et0 in b.edges:
        deg_b[s0] = deg_b.get(s0, 0) + 1
        deg_b[d0] = deg_b.get(d0, 0) + 1
    # isolated_w is declared here (the [B8] down-weight rule) but NOT applied
    # inside node_sim.  [T3] The baseline instrument (solver.py / TUNING_REPORT
    # line 39) applies the isolated down-weight at SCORE time, per-node, AFTER
    # the alignment has run on the RAW similarity.  Applying it pre-alignment
    # (inside node_sim) changes the correspondence itself -- the alignment sees
    # a deflated matrix and maps fewer nodes -- so the score is not the baseline
    # cell.  The flag value is unchanged; only the placement moves to score
    # time, where the baseline instrument reads it.
    # [MISSION 12 / step 3] The CONTENT-keyed similarity cache.  `node_sim` is
    # memoised on the node ids of this one directional call; the same signature
    # CONTENT recurs across pairs and across a retrieval run, and the similarity
    # is a pure function of that content (plus the two tolerance flags, which
    # are part of the cache's namespace).  The cached value IS the shipped
    # `_sig_similarity` call, so with the flag off this branch does not exist
    # and with it on the returned floats are the same floats.
    if bool(getattr(P, "MATCH_SIG_SIM_CACHE", False)):
        from matching import sig_sim_cache as _SC

        @lru_cache(maxsize=None)
        def node_sim(u, v):
            x, y = sigs_a.get(u), sigs_b.get(v)
            if x is None or y is None or x[0] != y[0]:
                return 0.0
            return _SC.sim(x[1], y[1], x[0])
    else:
        @lru_cache(maxsize=None)
        def node_sim(u,v):
            x,y=sigs_a.get(u),sigs_b.get(v)
            if x is None or y is None or x[0]!=y[0]: return 0.0
            return _sig_similarity(x[1],y[1])

    unmatched_cost = float(P.MATCH_UNMATCHED_COST)
    if bool(getattr(P, "MATCH_UNMATCHED_COST_DERIVED", False)):
        # [F2] c = mean positive node similarity over the same-lens candidate
        # matrix.  [B9] THE PRICE MUST BE ON q's SCALE.  A relative criterion
        # (the mean of the raw similarities) compared against
        # q = sim*(1-beta) + beta*rel made every >=1-side-edgeless pair
        # unmatchable at high beta with the isolated down-weight on
        # (q = 0.125*sim < c=0.15 collapsed a mapping from 229 to 56 nodes).
        # node_sim is memoized (B3), and the price is scaled by exactly the
        # coefficients q applies to a node similarity, so the comparison is
        # a cost accounting in the objective's own units.
        positive = [s for u in na for v in nb
                    if (s := node_sim(u, v)) > 0.0]
        price_scale = ((1.0 - float(P.MATCH_SYSTEMATICITY_WEIGHT))
                       * isolated_w)
        unmatched_cost = ((math.fsum(positive) / len(positive)) * price_scale
                          if positive else 0.0)

    # [MISSION 9] Build the placement sides for the credited scorer.  The
    # builder needs the PIXELS (the source frame is read off the field), so a
    # caller that wants the scorer must hand them over; when it does not, the
    # flag is on and STRICT is on, the aligner raises instead of skipping.
    _side_a = _side_b = None
    if bool(getattr(P, "PLACEMENT_MATCHER_ENABLED", False)) and placement_images:
        from matching import placement_matcher as _PM
        _img_a, _img_b = placement_images[0], placement_images[1]
        if _img_a is not None and _img_b is not None:
            # [MISSION 12 / step 1b] ONE SIDE PER IMAGE.  The forward and the
            # reverse directional call both build both sides, so the same image's
            # payload was built up to four times per pair.  `build_side_cached` is
            # the same pure function keyed on (graph object, pixels, switches):
            # it returns the identical Side, so every value downstream is
            # unchanged, and the repeat builds disappear.
            _side_a = _PM.build_side_cached(_as_image_graph(a, sigs_a), _img_a)
            _side_b = _PM.build_side_cached(_as_image_graph(b, sigs_b), _img_b)

    @lru_cache(maxsize=None)
    def edge_sim(x,y,u,v,et):
        return edge_pair_credit(a,b,x,y,u,v,et,edge_weight_fn=_align_edge_weight)

    al=global_consistency_alignment(
        na,nb,node_sim,edges_a=a.edges,edges_b=b.edges,
        unmatched_cost=unmatched_cost,
        systematicity_weight=float(P.MATCH_SYSTEMATICITY_WEIGHT),
        rounds=int(P.MATCH_ALIGN_ROUNDS),max_pair_evals=P.MATCH_ALIGN_MAX_PAIR_EVALS,
        budget_refinement_only=bool(P.MATCH_ALIGN_BUDGET_REFINEMENT_ONLY),
        # [A5] the geometry the occlusion credit needs
        anchors_a=getattr(a,"anchor_of_node",None),
        anchors_b=getattr(b,"anchor_of_node",None),
        junctions_a=getattr(a,"t_junctions",None) or (),
        junctions_b=getattr(b,"t_junctions",None) or (),
        edge_similarity=edge_sim,
        # [B2] the vector address is a fast code of node similarity; it ONLY
        # seeds the alignment (multiplies the seed matrix).  It never enters
        # the objective.
        addrs_a=getattr(a,"addrs",None) if addrs_a is None else addrs_a,
        addrs_b=getattr(b,"addrs",None) if addrs_b is None else addrs_b,
        address_k=address_k,
        edge_weight_fn=_align_edge_weight,
        # [MISSION 9 / dark] the credited scorer.  Sides are built HERE, from
        # the graph and the pixels it came from, only when the flag is on and a
        # caller supplied the images; otherwise both stay None and the aligner's
        # new block never runs.
        placement_side_a=_side_a,
        placement_side_b=_side_b,
        placement_images=(_img_a, _img_b) if _side_a is not None else None)
    assert len(set(al.mapping.values()))==len(al.mapping)
    b_und={(min(x,y),max(x,y),et) for x,y,et in b.edges if et in UNDIRECTED_EDGE_TYPES}
    b_dir={(x,y,et) for x,y,et in b.edges if et not in UNDIRECTED_EDGE_TYPES}
    a_und={(min(x,y),max(x,y),et) for x,y,et in a.edges if et in UNDIRECTED_EDGE_TYPES}
    a_dir={(x,y,et) for x,y,et in a.edges if et not in UNDIRECTED_EDGE_TYPES}
    credits=[];graded=[];presence=0
    presence_credit=0.0;presence_pairs=0
    # [B3] raw weight of the presence-only edges (subtracted from the
    # weighted denominator, mirroring the count subtraction).
    presence_weight=0.0
    # [J8] the presence-only relations among the CHECKABLE ones (both
    # endpoints mapped): they can never earn, so they leave the edge mass.
    presence_checkable_a=0
    presence_checkable_b=0
    for x,y,et in sorted(a.edges,key=edge_key):
        u,v=al.mapping.get(x),al.mapping.get(y)
        if u is None or v is None:continue
        target=(min(u,v),max(u,v),et) if et in UNDIRECTED_EDGE_TYPES else (u,v,et)
        present=(target in b_und if et in UNDIRECTED_EDGE_TYPES else target in b_dir)
        if not present:continue
        p=_lookup_edge_params(a.edge_params,x,y,et);q=_lookup_edge_params(b.edge_params,u,v,et)
        # [B7] INCOMPARABLE IS NOT CONTRADICTORY: a relation with no common
        # ontology is neither credited NOR charged -- it leaves both the
        # numerator and the denominator.
        if edge_params_incomparable(p,q):
            continue
        if edge_graded_axes(p,q)>0:
            value=edge_sim(x,y,u,v,et)
            graded.append(value)
            if min_credit:
                # [B3] numerator: min(weight on each side) * similarity
                wa=_edge_weight(a,x,y,et); wb=_edge_weight(b,u,v,et)
                credits.append(min(wa, wb) * value)
            else:
                credits.append(value)
        else:
            # [PHASE 3 / I5] NO MEASURED AXES: this is the topological fact
            # that the relation exists on both sides -- a membership flag, a
            # parameter-less predicate -- and the tree's own rule calls that
            # "a count", not physics.  The historical behaviour credits it a
            # FULL unit of evidence; the flag withholds that credit and
            # removes the two edges it spans from the denominator.
            #
            # [J1] THE CATEGORICAL CREDIT IS READ HERE.  A CATEGORICAL
            # agreement -- the same topology class on both sides, e.g. "a T
            # junction joining these two contours" -- is physical
            # organization, and `edge_pair_credit` already reads it (the
            # identity axes that agree).  The loop used the RAW typed
            # similarity instead, which reads a categorical agreement as
            # 0.0 (no common MEASURED axis), so a T junction that was a T
            # junction on both sides earned NOTHING here while the
            # alignment itself was told it was evidence.
            raw = edge_sim(x,y,u,v,et)
            # [J2] the WITHHELD mass is what the relation would have earned
            # without the flag, read with the same (fixed) rules: a relation
            # with no parameters on BOTH sides is the historical full unit, a
            # relation measured on ONE side only is a missing measurement and
            # worth nothing, and a relation with parameters on both sides is
            # worth its typed similarity.
            withheld=(1.0 if (p is None and q is None)
                      else (0.0 if (p is None or q is None)
                            else _pair_edge_sim_typed(x,y,et,p,u,v,et,q,False)))
            presence+=1
            presence_credit+=withheld
            presence_checkable_a+=1
            if not_evidence and raw <= 0.0:
                # the presence-only edges (one on each side) can never earn,
                # so they must not dilute the mass that can
                presence_pairs+=1
                if min_credit:
                    presence_weight += (_edge_weight(a,x,y,et)
                                        + _edge_weight(b,u,v,et))
            else:
                if min_credit:
                    # [B3] the categorical credit is also weighted by the
                    # weaker side's edge weight
                    wa=_edge_weight(a,x,y,et); wb=_edge_weight(b,u,v,et)
                    credits.append(min(wa, wb) * raw)
                else:
                    credits.append(raw)
    # [T3] SCORE-TIME ISOLATED DOWN-WEIGHT.  The baseline instrument
    # (solver.py / TUNING_REPORT line 39) applies the isolated-node weight
    # AFTER the alignment, per mapped node, to the node credit only.  node_sim
    # is raw now; here each mapped pair's contribution is scaled by
    # isolated_w when either endpoint is isolated.  This is the placement the
    # baseline cell declares (isolated=downweight), not a new mechanism.
    nodes=math.fsum(node_sim(u,v) * (
        isolated_w if (deg_a.get(u, 0) == 0 or deg_b.get(v, 0) == 0) else 1.0
    ) for u,v in sorted(al.mapping.items()))
    edge_credit=math.fsum(credits)
    rev={v:k for k,v in al.mapping.items()}
    # [TUNE / DENOMINATOR] the checkable masses and the image-side credit.
    # checkable_a counts A edges with BOTH endpoints mapped; checkable_b the
    # same from B's side through the inverse mapping.  The b-side credit is
    # only needed by the non-historical denominator modes.
    mode=str(getattr(P, "MATCH_EDGE_DENOMINATOR_MODE", "all"))
    gamma=float(getattr(P, "MATCH_COVERAGE_EXPONENT_GAMMA", 1.0))
    checkable_a=sum(1 for x,y,_et in a.edges
                    if al.mapping.get(x) is not None and al.mapping.get(y) is not None)
    checkable_b=sum(1 for x,y,_et in b.edges
                    if rev.get(x) is not None and rev.get(y) is not None)
    # [B3] raw edge-weight masses for the weighted denominator
    raw_mass_a = math.fsum(_edge_weight(a, x, y, et) for x, y, et in a.edges)
    raw_mass_b = math.fsum(_edge_weight(b, x, y, et) for x, y, et in b.edges)
    # [B3] raw weight of the CHECKABLE edges (both endpoints mapped), for the
    # checkable/per_side denominator arms.
    checkable_mass_a = math.fsum(
        _edge_weight(a, x, y, et) for x, y, et in a.edges
        if al.mapping.get(x) is not None and al.mapping.get(y) is not None)
    checkable_mass_b = math.fsum(
        _edge_weight(b, x, y, et) for x, y, et in b.edges
        if rev.get(x) is not None and rev.get(y) is not None)
    b_credit=0.0
    if mode != "all":
        for x,y,et in sorted(b.edges,key=edge_key):
            u,v=rev.get(x),rev.get(y)
            if u is None or v is None:continue
            target=(min(u,v),max(u,v),et) if et in UNDIRECTED_EDGE_TYPES else (u,v,et)
            present=(target in a_und) if et in UNDIRECTED_EDGE_TYPES else (target in a_dir)
            if not present:continue
            p=_lookup_edge_params(b.edge_params,x,y,et);q=_lookup_edge_params(a.edge_params,u,v,et)
            # [B7] an incomparable relation leaves the denominator too
            if edge_params_incomparable(p,q):
                continue
            if edge_graded_axes(p,q)>0:
                b_credit+=edge_sim(u,v,x,y,et)
            else:
                # [J1] the categorical credit is read here too, for the
                # reason given in the A-side loop.
                raw=edge_sim(u,v,x,y,et)
                presence_checkable_b+=1
                if not not_evidence or raw > 0.0:
                    b_credit+=raw
    if mode == "all" and gamma == 1.0:
        if min_credit:
            # [B3] denominators carry the raw weights, not the counts
            denom=len(na)+len(nb)+raw_mass_a+raw_mass_b
            if presence_pairs:
                denom=max(0,denom-presence_weight)
        else:
            denom=len(na)+len(nb)+len(a.edges)+len(b.edges)
            if presence_pairs:
                # the presence-only edges (one on each side) can never earn, so they
                # must not dilute the mass that can
                denom=max(0,denom-2*presence_pairs)
        node_denom=len(na)+len(nb)
        if separate:
            # [FIX 2] The edge term charges against ALL edges on both sides.
            # Preserving 10% of the edges must score at most 0.10 on this term.
            edge_denom=(raw_mass_a+raw_mass_b) if min_credit else (len(a.edges)+len(b.edges))
        else:
            edge_denom=(max(0, raw_mass_a+raw_mass_b-presence_weight)
                        if min_credit else
                        max(0,len(a.edges)+len(b.edges)-2*presence_pairs))
        node_term=(nodes/node_denom) if node_denom else 0.0
        edge_term=(edge_credit/edge_denom) if edge_denom else 0.0
        # [J4] MATCH_EDGE_WEIGHT IS READ IN THE POOLED BRANCH.  It was not:
        # the pooled branch always gave the edge term its FULL weight, so
        # the parameter had no effect on the production score at all.  The
        # production number was the offline w=1.0 cell of the tuned surface
        # exactly, while the SELECTED cell was a different number (0.669147
        # against 0.867295), so NO cell of the surface was
        # production-equivalent and the selected configuration could not be
        # deployed.  The form is the offline pooled form, term for term:
        #     2·(node_credit + w·edge_credit) / (na + nb + edge_mass)
        # so the production score now EQUALS the tuned cell it names.
        w=float(P.MATCH_EDGE_WEIGHT)
        if separate:
            # [FIX 2] the separate-term arm is the OTHER offline form: the
            # two terms on their OWN masses, blended with the declared
            # weight.
            score=2.0*((1.0-w)*node_term+w*edge_term)
        else:
            score=2.0*(nodes+w*edge_credit)/denom if denom else 0.0
        edge_credit_total=edge_credit
        edge_mass=float(edge_denom)
        denom=node_denom+edge_mass
    else:
        # [TUNE / DENOMINATOR] the arms the grid measures.  The node mass is
        # always na+nb; only the edge mass changes.  gamma < 1 makes the
        # charge for an unmatched edge sublinear instead of a full unit.
        #
        # [J8] THE PRESENCE-ONLY EDGES LEAVE THE EDGE MASS IN EVERY ARM.
        # They can never earn (a relation with no measured axes has no
        # physics to read), so they must not dilute the mass that can.  The
        # pooled branch already did this; the checkable and per_side arms
        # counted them, so the same graph scored differently depending on
        # which denominator was declared -- for a reason that has nothing to
        # do with the physics.
        if mode == "checkable":
            if min_credit:
                edge_mass=float(2*min(checkable_mass_a, checkable_mass_b))
            else:
                edge_mass=float(2*min(checkable_a-presence_checkable_a,
                                      checkable_b-presence_checkable_b))
        elif mode == "per_side":
            if min_credit:
                edge_mass=float(checkable_mass_a+checkable_mass_b)
            else:
                edge_mass=float((checkable_a-presence_checkable_a)
                                +(checkable_b-presence_checkable_b))
        else:
            if min_credit:
                edge_mass=float(max(0,raw_mass_a+raw_mass_b-presence_weight))
            else:
                edge_mass=float(max(0,len(a.edges)+len(b.edges)-2*presence_pairs))
        edge_mass=max(0.0,edge_mass)
        if gamma != 1.0:
            edge_mass=float(edge_mass**gamma) if edge_mass>0.0 else 0.0
        edge_credit_total=edge_credit+b_credit
        node_denom=len(na)+len(nb)
        denom=node_denom+edge_mass
        node_term=(nodes/node_denom) if node_denom else 0.0
        edge_term=(edge_credit_total/edge_mass) if edge_mass else 0.0
        # [J4] the same declared weight here: the arms of the grid are
        # measured with the weight they declare.
        w=float(P.MATCH_EDGE_WEIGHT)
        score=2.0*(nodes+w*edge_credit_total)/denom if denom else 0.0
    # [MISSION 11 / step 1] THE PAIR SCORE ITSELF.  Placement has so far only
    # moved the MAPPING; the number a retrieval arm ranks by is this one, and it
    # knows nothing about the evidence that produced the correspondence.  With
    # the dark flag on, one term is ADDED to the shipped score:
    #     score' = score + w * E
    # with E the coverage-neutral, shrinkage-damped mean centred placement credit
    # (see placement_evidence above and parameters/_text.py for the definition
    # written down before the fit).  Flag off: `score` is byte-for-byte the
    # shipped number and nothing below runs.
    score_shipped = float(score)
    ev = 0.0
    judged = 0
    share = 0.0
    if bool(getattr(P, "PLACEMENT_SCORE_EVIDENCE", False)):
        if _side_a is not None and _side_b is not None:
            from matching import placement_matcher as _PM
            _fit, _pfit = _PM.fit_objects()
            _b = _PM.frozen()
            ev, judged, share = placement_evidence(
                dict(al.mapping), _side_a, _side_b, _fit, _pfit,
                float(_b["_a_exist"]), float(_b["_a_geom"]))
            w_score = float(getattr(P, "PLACEMENT_SCORE_WEIGHT", 0.0))
            score = score_shipped + w_score * ev
            PLACEMENT_SCORE_STATS["pairs_scored"] += 1
            PLACEMENT_SCORE_STATS["mapped_nodes"] += len(al.mapping)
            PLACEMENT_SCORE_STATS["judged_nodes"] += judged
            if abs(ev) > 0.0:
                PLACEMENT_SCORE_STATS["nonzero_evidence"] += 1
        else:
            # Flag on with no sides: in TEST MODE that is a wiring error and it
            # raises; in production it is counted and the shipped score stands.
            PLACEMENT_SCORE_STATS["pairs_no_sides"] += 1
            if bool(getattr(P, "PLACEMENT_MATCHER_STRICT", False)):
                PLACEMENT_SCORE_STATS["strict_failures"] += 1
                raise RuntimeError(
                    "PLACEMENT_SCORE_EVIDENCE needs placement_images for both "
                    "sides; got none (strict mode)")
    return JointMCS(float(score),dict(al.mapping),len(credits),len(graded),math.fsum(graded),presence,al.budget_exhausted,presence_credit,nodes,edge_credit_total,node_term,edge_term,float(denom),score_shipped,float(ev),int(judged),float(share))


def joint_mcs(a, b, sigs_a, sigs_b, addrs_a=None, addrs_b=None, address_k=None, placement_images=None):
    """Symmetric score: retain the better of both directional heuristic solves.

    This does not certify optimality. Reversing graph arguments considers the
    same two candidates, so the scalar score is symmetric even if the bounded
    heuristic's seed or refinement behaves differently in the two directions.

    [B2] ``addrs_a``/``addrs_b`` are the per-graph vector addresses (a side
    structure beside the graph).  They seed the alignment only.
    """
    forward = _directional_joint_mcs(a, b, sigs_a, sigs_b, addrs_a, addrs_b,
                                     address_k, placement_images)
    reverse = _directional_joint_mcs(
        b, a, sigs_b, sigs_a, addrs_b, addrs_a, address_k,
        (placement_images[1], placement_images[0]) if placement_images else None)
    if forward.score >= reverse.score:
        # symmetric budget reporting: the other direction's exhaustion is
        # still information about this pair.
        if reverse.budget_exhausted and not forward.budget_exhausted:
            return replace(forward, budget_exhausted=True)
        return forward
    return JointMCS(reverse.score, {v:k for k,v in reverse.mapping.items()},
                    reverse.matched_edges, reverse.graded_edges,
                    reverse.graded_edge_similarity_sum, reverse.presence_only_edges,
                    forward.budget_exhausted or reverse.budget_exhausted,
                    reverse.presence_only_credit, reverse.node_credit,
                    reverse.edge_credit, reverse.node_term, reverse.edge_term,
                    reverse.denominator, reverse.score_shipped,
                    reverse.placement_evidence, reverse.placement_judged,
                    reverse.placement_share_judged)
