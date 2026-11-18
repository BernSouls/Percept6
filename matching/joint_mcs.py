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


def edge_pair_credit(a, b, x, y, u, v, et):
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
    @property
    def mean_graded_edge_similarity(self):
        return self.graded_edge_similarity_sum / self.graded_edges if self.graded_edges else None


def _directional_joint_mcs(a, b, sigs_a, sigs_b):
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
    # isolated_w is bound above with the [B8] down-weight rule.
    @lru_cache(maxsize=None)
    def node_sim(u,v):
        x,y=sigs_a.get(u),sigs_b.get(v)
        if x is None or y is None or x[0]!=y[0]: return 0.0
        got = _sig_similarity(x[1],y[1])
        if isolated_w < 1.0 and (deg_a.get(u, 0) == 0 or deg_b.get(v, 0) == 0):
            got *= isolated_w
        return got

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

    @lru_cache(maxsize=None)
    def edge_sim(x,y,u,v,et):
        return edge_pair_credit(a,b,x,y,u,v,et)

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
        edge_similarity=edge_sim)
    assert len(set(al.mapping.values()))==len(al.mapping)
    b_und={(min(x,y),max(x,y),et) for x,y,et in b.edges if et in UNDIRECTED_EDGE_TYPES}
    b_dir={(x,y,et) for x,y,et in b.edges if et not in UNDIRECTED_EDGE_TYPES}
    a_und={(min(x,y),max(x,y),et) for x,y,et in a.edges if et in UNDIRECTED_EDGE_TYPES}
    a_dir={(x,y,et) for x,y,et in a.edges if et not in UNDIRECTED_EDGE_TYPES}
    credits=[];graded=[];presence=0
    presence_credit=0.0;presence_pairs=0
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
            else:
                credits.append(raw)
    nodes=math.fsum(node_sim(u,v) for u,v in sorted(al.mapping.items()))
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
        denom=len(na)+len(nb)+len(a.edges)+len(b.edges)
        if presence_pairs:
            # the presence-only edges (one on each side) can never earn, so they
            # must not dilute the mass that can
            denom=max(0,denom-2*presence_pairs)
        node_denom=len(na)+len(nb)
        if separate:
            # [FIX 2] The edge term charges against ALL edges on both sides.
            # Preserving 10% of the edges must score at most 0.10 on this term.
            edge_denom=len(a.edges)+len(b.edges)
        else:
            edge_denom=max(0,len(a.edges)+len(b.edges)-2*presence_pairs)
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
            edge_mass=float(2*min(checkable_a-presence_checkable_a,
                                  checkable_b-presence_checkable_b))
        elif mode == "per_side":
            edge_mass=float((checkable_a-presence_checkable_a)
                            +(checkable_b-presence_checkable_b))
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
    return JointMCS(score,dict(al.mapping),len(credits),len(graded),math.fsum(graded),presence,al.budget_exhausted,presence_credit,nodes,edge_credit_total,node_term,edge_term,float(denom))


def joint_mcs(a, b, sigs_a, sigs_b):
    """Symmetric score: retain the better of both directional heuristic solves.

    This does not certify optimality. Reversing graph arguments considers the
    same two candidates, so the scalar score is symmetric even if the bounded
    heuristic's seed or refinement behaves differently in the two directions.
    """
    forward = _directional_joint_mcs(a, b, sigs_a, sigs_b)
    reverse = _directional_joint_mcs(b, a, sigs_b, sigs_a)
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
                    reverse.denominator)
