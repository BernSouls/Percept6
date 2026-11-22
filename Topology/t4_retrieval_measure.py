"""Task 4 RETRIEVAL — measure R/N/W on the ETH-80 retrieval task (parallel).

Headline: 8 primary refs (rank-0/class, -090-000) x 128 queries (rank-0/class
x 16 cross-view).  Arms: R real, N no_edge, W rewired (seed 0).  beta=0.50
reference.  Repaired instrument.  2-worker multiprocessing.
"""
from __future__ import annotations
import json, os, sys, time
from dataclasses import replace
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402
from t4_shift import counterpart_map, matcher_accuracy  # noqa: E402
from tune import _rewire_edges  # noqa: E402

ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
REF_VIEW = "-090-000"
QUERY_VIEWS = ("-000-000", "-022-000", "-022-090", "-022-180", "-022-270",
               "-035-135", "-035-315", "-045-000", "-045-090", "-045-180",
               "-045-270", "-066-027", "-066-117", "-066-207", "-066-297",
               "-090-135")
CLASSES = ["apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato"]

WINNER = {
    "MATCH_UNMATCHED_COST": 0.15, "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1, "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
    "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
}
BETA = 0.50

_G = {}  # global graphs in workers


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _arm_graph(g, arm):
    sg = g.subgraph
    if arm == "R":
        return g
    if arm == "N":
        return replace(g, subgraph=replace(sg, edges=frozenset(), edge_params={}))
    if arm == "W":
        edges = _rewire_edges(sg.edges, 0)
        return replace(g, subgraph=replace(sg, edges=edges, edge_params={}))
    raise ValueError(arm)


def _run_matcher(q_g, r_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    P.MATCH_SYSTEMATICITY_WEIGHT = BETA
    P.BINDER_COMPLETE_POSITION_MAP = True
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(q_g.subgraph, r_g.subgraph, q_g.sigs, r_g.sigs)
    return res.score, res.mapping


def _init_worker(refs_meta, queries_meta):
    from frontend import load
    refs = []
    for m in refs_meta:
        g = load(m["cache"])
        refs.append({"class": m["class"], "object": m["object"],
                     "view": m["view"], "graph": g,
                     "pos": _pos_lens(g),
                     "arms": {a: _arm_graph(g, a) for a in ("R", "N", "W")}})
    _G["refs"] = refs
    _G["queries_meta"] = queries_meta


def _process_query(qi):
    from frontend import load
    mq = _G["queries_meta"][qi]
    g = load(mq["cache"])
    q = {"class": mq["class"], "object": mq["object"], "view": mq["view"],
         "graph": g, "pos": _pos_lens(g),
         "arms": {a: _arm_graph(g, a) for a in ("R", "N", "W")}}
    refs = _G["refs"]
    out = {"qi": qi, "scores": {}, "case_acc": {}, "d1": {}}
    token_kinds = set(str(q["graph"].sigs.get(n, ("?",))[0])
                      for n in q["graph"].subgraph.nodes)
    for ri, r in enumerate(refs):
        same_obj = (q["object"] == r["object"] and q["class"] == r["class"])
        same_cls = (q["class"] == r["class"])
        ptype = "same_obj_same_cls" if same_obj else (
            "same_cls_diff_obj" if same_cls else "diff_cls")
        cp, _, _ = counterpart_map(q["pos"], r["pos"], 0, 0, 8.0)
        for a in ("R", "N", "W"):
            sc, mp = _run_matcher(q["arms"][a], r["arms"][a])
            out["scores"]["%d_%d_%s" % (qi, ri, a)] = sc
            macc = matcher_accuracy(q["pos"], mp, cp, token_kinds)
            out["d1"]["%d_%d_%s" % (qi, ri, a)] = macc["acc"]
            tot = macc["total"] - macc["no_counterpart"]
            out["case_acc"]["%d_%d_%s" % (qi, ri, a)] = (
                macc["correct"] / tot if tot > 0 else None)
        out.setdefault("pair_type", []).append(ptype)
    return out


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def _boot(vals, seed=42, nboot=1000):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return {"mean": float(vals.mean()), "lo": lo, "hi": hi, "hw": (hi - lo) / 2,
            "n": n}


def _paired(a, b, seed=42, nboot=1000):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return None
    diffs = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return {"mean": float(diffs.mean()), "lo": lo, "hi": hi, "hw": (hi - lo) / 2,
            "n": n}


def _entries_meta():
    import corpus
    from frontend import build_cached
    refs, queries = [], []
    for cls in CLASSES:
        inst = corpus._instances(cls)
        ids = sorted(inst)
        obj0 = ids[0]
        e = corpus._entry(cls, obj0, REF_VIEW, inst[obj0][REF_VIEW])
        m = build_cached(e)
        refs.append({"class": cls, "object": obj0, "view": REF_VIEW,
                     "cache": m["cache"]})
        for v in QUERY_VIEWS:
            eq = corpus._entry(cls, obj0, v, inst[obj0][v])
            mq = build_cached(eq)
            queries.append({"class": cls, "object": obj0, "view": v,
                            "cache": mq["cache"]})
    return refs, queries


def main() -> int:
    print("=== RETRIEVAL: R/N/W (8 refs x 128 queries), 2 workers ===", flush=True)
    refs_meta, queries_meta = _entries_meta()
    print("refs=%d queries=%d" % (len(refs_meta), len(queries_meta)), flush=True)
    from multiprocessing import Pool
    t0 = time.perf_counter()
    results = []
    with Pool(2, initializer=_init_worker,
              initargs=(refs_meta, queries_meta)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_query,
                                                  range(len(queries_meta)))):
            results.append(r)
            if (i + 1) % 16 == 0:
                print("  query %d/%d (%.0fs)"
                      % (i + 1, len(queries_meta), time.perf_counter() - t0),
                      flush=True)
    print("done %d queries (%.0fs)" % (len(results), time.perf_counter() - t0),
          flush=True)

    # aggregate
    arms = ("R", "N", "W")
    nq = len(queries_meta)
    nr = len(refs_meta)
    to1 = {a: [] for a in arms}
    auc_cls = {a: [] for a in arms}
    ca = {a: [] for a in arms}
    d1v = {a: [] for a in arms}
    scores = {a: {} for a in arms}
    pair_type = {}
    for r in results:
        qi = r["qi"]
        for ri, ptype in enumerate(r["pair_type"]):
            pair_type[(qi, ri)] = ptype
        for a in arms:
            sc = [r["scores"]["%d_%d_%s" % (qi, ri, a)] for ri in range(nr)]
            scores[a][qi] = sc
            labels = [1 if pair_type[(qi, ri)] == "same_obj_same_cls" else 0
                      for ri in range(nr)]
            ci = [ri for ri in range(nr)
                  if pair_type[(qi, ri)] == "same_obj_same_cls"][0]
            rank = 1 + sum(1 for s in sc if s > sc[ci])
            to1[a].append(1.0 if rank == 1 else 0.0)
            auc_cls[a].append(_auc(sc, labels))
            ca[a].extend([v for k, v in r["case_acc"].items()
                          if k.endswith(a) and v is not None])
            d1v[a].extend([v for k, v in r["d1"].items() if k.endswith(a)])

    diffs = {}
    for (a, b) in [("R", "N"), ("R", "W"), ("N", "W")]:
        diffs["%s-%s_top1" % (a, b)] = _paired(to1[a], to1[b])
        diffs["%s-%s_auc" % (a, b)] = _paired(
            [x for x in auc_cls[a] if x is not None],
            [x for x in auc_cls[b] if x is not None])

    per_type = {}
    for ptype in ("same_obj_same_cls", "same_cls_diff_obj", "diff_cls"):
        idx = [(qi, ri) for (qi, ri), t in pair_type.items() if t == ptype]
        per_type[ptype] = {
            "n": len(idx),
            "R-N_score": _paired(
                [scores["R"][qi][ri] for qi, ri in idx],
                [scores["N"][qi][ri] for qi, ri in idx]),
            "R-W_score": _paired(
                [scores["R"][qi][ri] for qi, ri in idx],
                [scores["W"][qi][ri] for qi, ri in idx]),
        }

    out = {
        "headline_8ref": {
            "n_refs": nr, "n_queries": nq, "n_pairs": nq * nr,
            "top1": {a: _boot(to1[a]) for a in arms},
            "auc_class": {a: _boot([x for x in auc_cls[a] if x is not None])
                          for a in arms},
            "paired_diffs": diffs,
            "case_accuracy": {a: _boot(ca[a]) for a in arms},
            "d1": {a: _boot(d1v[a]) for a in arms},
            "per_pair_type": per_type,
        }
    }
    write_json(RESULTS_DIR / "t4" / "retrieval_8ref.json", out)
    print("\n=== 8-ref headline ===", flush=True)
    for a in arms:
        print("  %s: top1=%s auc=%s" % (a, out["headline_8ref"]["top1"][a],
              out["headline_8ref"]["auc_class"][a]), flush=True)
    for k, v in diffs.items():
        print("  %s: %s" % (k, v), flush=True)
    print("\n=== 8-ref complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
