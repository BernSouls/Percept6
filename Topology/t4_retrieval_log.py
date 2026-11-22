"""Task 4 RETRIEVAL — Part 1 component logging + W-P builder.

Re-run R, N, W on the 8x128 retrieval set, logging per-(query,ref,arm) ALL
score components for BOTH directions (forward and reverse), plus which
direction the shipped score selected.  Also provides:

  * _rewired_graph_preserve_params  — W-P: degree-preserving rewiring that
    KEEPS each edge's original parameters (geometrically inconsistent with the
    new endpoints; that inconsistency is the control, not corrected).
  * edge_term_under_mapping          — standalone edge_credit / edge_term
    computed under a FIXED mapping (for 1b: N's mapping held fixed).

2-worker multiprocessing.  Writes JSONL, one record per (query, ref, arm).
"""
from __future__ import annotations
import json, os, sys, time, math, random
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR  # noqa: E402
from t4_retrieval_measure import (_arm_graph, WINNER, BETA, QUERY_VIEWS,
                                  CLASSES, REF_VIEW, ARM_B1)
from tune import _rewire_edges  # noqa: E402
import corpus
from frontend import build_cached
from frontend import load as fload

# ---- re-export the directional matcher so we can log BOTH directions ----
from matching.joint_mcs import _directional_joint_mcs  # noqa: E402
from matching.structural_similarity import (  # noqa: E402
    _lookup_edge_params, _lookup_edge_params_oriented, _pair_edge_sim_typed)
from core.schema import edge_params_incomparable, edge_graded_axes  # noqa: E402
from matching.joint_mcs import edge_pair_credit, edge_key  # noqa: E402
from core.types import UNDIRECTED_EDGE_TYPES  # noqa: E402
import parameters as P  # noqa: E402


# --------------------------------------------------------------------------
# W-P builder
# --------------------------------------------------------------------------
def _rewired_graph_preserve_params(g, seed):
    """Degree-preserving rewiring that KEEPS each edge's original parameters.

    Same endpoint-swap as _rewire_edges, but each rewired edge carries the
    parameter values of the original edge it replaced.  The parameters are now
    geometrically inconsistent with the new endpoints; that inconsistency is
    the control and is NOT corrected.
    """
    sg = g.subgraph
    # [DETERMINISM] canonical order: sg.edges is a frozenset and its
    # iteration order is salted per process, which made the rewiring (and the
    # resulting edge count) differ between runs.
    edges = sorted(sg.edges, key=lambda e: (e[0], e[1], str(e[2])))
    # [BUGFIX] the parameters live on the SUBGRAPH (core.types.Graph has no
    # top-level edge_params); reading g.edge_params silently found none, so
    # this control came out byte-identical to W (parameter-stripped) and the
    # 1b control never existed.  Read the subgraph's, then fall back.
    ep = (getattr(sg, "edge_params", None)
          or getattr(g, "edge_params", None) or {})

    def param_of(s, d, et):
        p = ep.get((s, d, et))
        if p is None and et in UNDIRECTED_EDGE_TYPES:
            p = ep.get((min(s, d), max(s, d), et))
            if p is None:
                p = ep.get((max(s, d), min(s, d), et))
        return p

    rng = random.Random(seed)
    und = [(min(a, b), max(a, b), t) for a, b, t in edges
           if t in UNDIRECTED_EDGE_TYPES]
    und_p = [param_of(a, b, t) for a, b, t in und]
    direct = [(a, b, t) for a, b, t in edges if t not in UNDIRECTED_EDGE_TYPES]
    direct_p = [param_of(a, b, t) for a, b, t in direct]

    out = set(und)
    for _ in range(20 * max(1, len(und))):
        if len(und) < 2:
            break
        i, j = rng.sample(range(len(und)), 2)
        a, b, t1 = und[i]
        c, d, t2 = und[j]
        if t1 != t2 or a == b or c == d:
            continue
        cand = [(a, d, t1), (c, b, t2)]
        if any(x == y for x, y, _ in cand):
            continue
        if any((min(x, y), max(x, y), t) in out and
               (min(x, y), max(x, y), t) not in ((min(a, b), max(a, b), t1),
                                                 (min(c, d), max(c, d), t2))
               for x, y, t in cand):
            continue
        before = set(out)
        out.discard((a, b, t1))
        out.discard((c, d, t2))
        out.update(cand)
        if len(out) != len(before):
            out = before
            continue
        und[i] = cand[0]
        und[j] = cand[1]
        # und_p[i], und_p[j] unchanged: each edge keeps its ORIGINAL params.

    by_type = {}
    for idx, (a, b, t) in enumerate(direct):
        by_type.setdefault(t, []).append(idx)
    re_direct = set()
    re_direct_p = {}
    for t, idxs in by_type.items():
        dests = [direct[i][1] for i in idxs]
        for _ in range(100):
            rng.shuffle(dests)
            if all(direct[i][0] != d for i, d in zip(idxs, dests)):
                break
        for i, d in zip(idxs, dests):
            a, b, _t = direct[i]
            nd = d if a != d else b
            re_direct.add((a, nd, t))
            re_direct_p[(a, nd, t)] = direct_p[i]

    new_ep = {}
    for (s, d, t), p in zip(und, und_p):
        if p is not None:
            new_ep[(s, d, t)] = p
            if t in UNDIRECTED_EDGE_TYPES:
                new_ep[(min(s, d), max(s, d), t)] = p
    for (s, d, t), p in re_direct_p.items():
        if p is not None:
            new_ep[(s, d, t)] = p
    return SimpleNamespace(subgraph=replace(sg, edges=frozenset(out | re_direct),
                                             edge_params=new_ep),
                           sigs=g.sigs, addrs=getattr(g, "addrs", None))


def _arm_graph_wp(g):
    return _rewired_graph_preserve_params(g, 0)


# --------------------------------------------------------------------------
# Standalone edge_credit / edge_term under a FIXED mapping (faithful replica
# of the matcher's A-side credit loop + checkable denominator, mode/min_credit
# from the shipped config).
# --------------------------------------------------------------------------
def edge_term_under_mapping(a, b, mapping):
    """EXACT edge_credit / edge_term under a FIXED mapping.

    Faithful two-sided replica of the matcher's shipped ('checkable',
    min_credit=False, gamma=1.0) branch:
        edge_credit_total = A-side credit + B-side credit
        edge_term = edge_credit_total / (2*min(checkable_a - pres_a,
                                               checkable_b - pres_b))
    The matcher's own ``edge_term`` under this mode is TWO-SIDED, so a
    one-sided replica would not reproduce it.  Verified against the logged
    chosen-direction edge_term for the N arm (mapping == logged mapping).

    Returns (edge_credit_total, edge_term, matched_edges, n_edges_a,
    n_edges_b, checkable_a, checkable_b).
    """
    not_evidence = bool(getattr(P, "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE", False))
    min_credit = bool(getattr(P, "MATCH_EDGE_MIN_CREDIT", False))
    b_und = {(min(x, y), max(x, y), et) for x, y, et in b.edges
             if et in UNDIRECTED_EDGE_TYPES}
    b_dir = {(x, y, et) for x, y, et in b.edges if et not in UNDIRECTED_EDGE_TYPES}
    a_und = {(min(x, y), max(x, y), et) for x, y, et in a.edges
             if et in UNDIRECTED_EDGE_TYPES}
    a_dir = {(x, y, et) for x, y, et in a.edges if et not in UNDIRECTED_EDGE_TYPES}

    def _edge_weight(graph, x, y, et):
        p = _lookup_edge_params(getattr(graph, "edge_params", None), x, y, et)
        return float(getattr(p, "confidence", 1.0)) if p is not None else 1.0

    def edge_sim(x, y, u, v, et):
        return edge_pair_credit(a, b, x, y, u, v, et)

    credits = []
    graded = []
    presence_checkable_a = 0
    for x, y, et in sorted(a.edges, key=edge_key):
        u, v = mapping.get(x), mapping.get(y)
        if u is None or v is None:
            continue
        target = (min(u, v), max(u, v), et) if et in UNDIRECTED_EDGE_TYPES else (u, v, et)
        present = (target in b_und if et in UNDIRECTED_EDGE_TYPES else target in b_dir)
        if not present:
            continue
        p = _lookup_edge_params(a.edge_params, x, y, et)
        q = _lookup_edge_params(b.edge_params, u, v, et)
        if edge_params_incomparable(p, q):
            continue
        if edge_graded_axes(p, q) > 0:
            value = edge_sim(x, y, u, v, et)
            graded.append(value)
            if min_credit:
                credits.append(min(_edge_weight(a, x, y, et),
                                   _edge_weight(b, u, v, et)) * value)
            else:
                credits.append(value)
        else:
            raw = edge_sim(x, y, u, v, et)
            presence_checkable_a += 1
            if not_evidence and raw <= 0.0:
                pass
            else:
                if min_credit:
                    credits.append(min(_edge_weight(a, x, y, et),
                                       _edge_weight(b, u, v, et)) * raw)
                else:
                    credits.append(raw)
    edge_credit_a = math.fsum(credits)

    rev = {v: k for k, v in mapping.items()}
    checkable_a = sum(1 for x, y, _et in a.edges
                      if mapping.get(x) is not None and mapping.get(y) is not None)
    checkable_b = sum(1 for x, y, _et in b.edges
                      if rev.get(x) is not None and rev.get(y) is not None)

    # ---- B-side credit (the matcher reads it for every non-"all" mode) ----
    b_credits = []
    presence_checkable_b = 0
    for x, y, et in sorted(b.edges, key=edge_key):
        u, v = rev.get(x), rev.get(y)
        if u is None or v is None:
            continue
        target = (min(u, v), max(u, v), et) if et in UNDIRECTED_EDGE_TYPES else (u, v, et)
        present = (target in a_und if et in UNDIRECTED_EDGE_TYPES else target in a_dir)
        if not present:
            continue
        p = _lookup_edge_params(b.edge_params, x, y, et)
        q = _lookup_edge_params(a.edge_params, u, v, et)
        if edge_params_incomparable(p, q):
            continue
        if edge_graded_axes(p, q) > 0:
            b_credits.append(edge_sim(u, v, x, y, et))
        else:
            raw = edge_sim(u, v, x, y, et)
            presence_checkable_b += 1
            if not_evidence and raw <= 0.0:
                pass
            else:
                b_credits.append(raw)
    edge_credit_b = math.fsum(b_credits)
    edge_credit_total = edge_credit_a + edge_credit_b

    if min_credit:
        checkable_mass_a = math.fsum(
            _edge_weight(a, x, y, et) for x, y, et in a.edges
            if mapping.get(x) is not None and mapping.get(y) is not None)
        checkable_mass_b = math.fsum(
            _edge_weight(b, x, y, et) for x, y, et in b.edges
            if rev.get(x) is not None and rev.get(y) is not None)
        edge_mass = float(2 * min(checkable_mass_a, checkable_mass_b))
    else:
        edge_mass = float(2 * min(checkable_a - presence_checkable_a,
                                  checkable_b - presence_checkable_b))
    edge_mass = max(0.0, edge_mass)
    gamma = float(getattr(P, "MATCH_COVERAGE_EXPONENT_GAMMA", 1.0))
    if gamma != 1.0:
        edge_mass = float(edge_mass ** gamma) if edge_mass > 0.0 else 0.0
    edge_term = (edge_credit_total / edge_mass) if edge_mass else 0.0
    n_matched = len(credits) + len(b_credits)
    return (edge_credit_total, edge_term, n_matched, len(a.edges), len(b.edges),
            checkable_a, checkable_b)


# --------------------------------------------------------------------------
# logging harness
# --------------------------------------------------------------------------
_G = {}


def _load(cls, obj, view):
    inst = corpus._instances(cls)
    e = corpus._entry(cls, obj, view, inst[obj][view])
    return fload(build_cached(e)["cache"])


def _set_flags():
    for k, v in WINNER.items():
        setattr(P, k, v)
    P.MATCH_SYSTEMATICITY_WEIGHT = BETA
    P.BINDER_COMPLETE_POSITION_MAP = True


def _comp(res):
    return {
        "score": res.score, "node_credit": res.node_credit,
        "edge_credit": res.edge_credit, "node_term": res.node_term,
        "edge_term": res.edge_term, "denominator": res.denominator,
        "matched_edges": res.matched_edges, "graded_edges": res.graded_edges,
        "budget_exhausted": res.budget_exhausted,
    }


def _init_worker(refs_meta):
    _set_flags()
    _G["refs"] = []
    for (cls, obj) in refs_meta:
        _G["refs"].append((cls, obj, _load(cls, obj, REF_VIEW)))


def _process_query(qmeta):
    cls, obj, view = qmeta
    qg = _load(cls, obj, view)
    out = []
    for (rcls, robj, rg) in _G["refs"]:
        for arm in ("R", "N", "W"):
            ga = _arm_graph(qg, arm)
            gr = _arm_graph(rg, arm)
            fwd = _directional_joint_mcs(ga.subgraph, gr.subgraph, ga.sigs, gr.sigs)
            rev = _directional_joint_mcs(gr.subgraph, ga.subgraph, gr.sigs, ga.sigs)
            if fwd.score >= rev.score:
                chosen = "fwd"
                ch = fwd
            else:
                chosen = "rev"
                ch = rev
            rec = {
                "query": [cls, int(obj), view], "ref": [rcls, int(robj)],
                "arm": arm, "chosen": chosen,
                "fwd": _comp(fwd), "rev": _comp(rev),
                "n_nodes_q": len(ga.subgraph.nodes), "n_nodes_r": len(gr.subgraph.nodes),
                "n_edges_q": len(ga.subgraph.edges), "n_edges_r": len(gr.subgraph.edges),
            }
            if arm == "N":
                # store the shipped mapping (chosen direction) for 1b/1f
                mapping = dict(ch.mapping) if chosen == "fwd" else \
                    {v: k for k, v in rev.mapping.items()}
                rec["mapping"] = [[int(u), int(v)] for u, v in sorted(mapping.items())]
            out.append(rec)
    return out


def main() -> int:
    print("=== Part 1: component logging R/N/W (8 refs x 128 queries) ===", flush=True)
    _set_flags()
    refs_meta = []
    for cls in CLASSES:
        inst = corpus._instances(cls)
        o = sorted(inst)[0]
        refs_meta.append((cls, o))
    queries = []
    for cls in CLASSES:
        inst = corpus._instances(cls)
        o = sorted(inst)[0]
        for v in QUERY_VIEWS:
            queries.append((cls, o, v))
    print("refs=%d queries=%d" % (len(refs_meta), len(queries)), flush=True)

    from multiprocessing import Pool
    out_path = RESULTS_DIR / "t4" / "retrieval_components.jsonl"
    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as fh, Pool(2, initializer=_init_worker,
                                         initargs=(refs_meta,)) as pool:
        for recs in pool.imap_unordered(_process_query, queries):
            fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 16 == 0:
                print("  query %d/%d (%ds)" % (n_done, len(queries),
                                               int(time.time() - t0)), flush=True)
    print("done %d queries (%ds) -> %s" % (n_done, int(time.time() - t0), out_path),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
