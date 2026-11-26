"""MISSION 12 STEP 5 -- the retrieval shortlist: recall against K, and whether a
cheap structural prefilter can do the work of a big K.

Two scorers, both truth-free at match time:

    node similarity   mean over the query's nodes of the best same-lens similarity
                      to the candidate image (the alignment-free shortlist score)
    structure         place count, edge-type histogram and degree distribution of
                      the two graphs, compared as normalised histograms (L1 overlap)

and their blend.  Nothing is fitted on the check side: the blend weight is chosen
on the same FIT scenes the family definitions were developed on, and the table
reports recall at K = 8, 16, 32, 48 for each scorer, per gallery.

    python Topology/placement_m12_synth_shortlist.py --workers 2
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m12_shortlist.json")
KS = (8, 16, 32, 48)
#: Every Nth query of the FIT grid -- deterministic, and sparse enough that the
#: whole table is minutes rather than hours.
QUERY_STRIDE = 5
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def gallery() -> List[Any]:
    """The WHOLE M9 family set (24 scenes): the queries come from its FIT half, so
    the true reference is in the gallery and the rest of the gallery is the
    distractor pool -- twins included, which is what makes recall at K informative."""
    from Topology import placement_families_m9 as FM
    return list(FM.all_specs_irregular(12)) + list(FM.all_specs_oop(4))


def queries() -> List[Tuple[str, Any, str, Dict[str, Any]]]:
    from Topology import placement_m12_fit_w as FW
    return FW.queries()


def _image_key(qk: str) -> Tuple[str, str]:
    name, lab = qk.split("|")
    return name, lab


def prewarm() -> None:
    from Topology import placement_m12_fit_w as FW
    FW.prewarm()


def _features(graph: Any) -> Dict[str, Any]:
    """Place count, edge-type histogram, degree distribution -- from the graph."""
    sub = graph.subgraph
    nodes = sorted(int(n) for n in sub.nodes)
    deg = {n: 0 for n in nodes}
    types: Dict[str, int] = {}
    for (s, d, et) in sub.edges:
        s, d = int(s), int(d)
        deg[s] = deg.get(s, 0) + 1
        deg[d] = deg.get(d, 0) + 1
        name = getattr(et, "name", None) or str(et)
        types[name] = types.get(name, 0) + 1
    dvals = np.asarray([deg[n] for n in nodes], dtype=float) if nodes else np.zeros(1)
    hist, _edges = np.histogram(dvals, bins=[0, 1, 2, 4, 8, 1e9])
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    total_types = float(sum(types.values())) or 1.0
    return {"places": float(len(nodes)),
            "types": {k: v / total_types for k, v in sorted(types.items())},
            "deg_hist": hist}


def _struct_sim(fa: Dict[str, Any], fb: Dict[str, Any]) -> float:
    """L1 overlap of the histograms + a place-count agreement term, in [0, 1]."""
    keys = sorted(set(fa["types"]) | set(fb["types"]))
    ta = np.asarray([fa["types"].get(k, 0.0) for k in keys]) if keys else np.zeros(1)
    tb = np.asarray([fb["types"].get(k, 0.0) for k in keys]) if keys else np.zeros(1)
    type_sim = float(np.minimum(ta, tb).sum()) / max(1e-9, float(max(ta.sum(), tb.sum())) * 0.5)
    type_sim = min(1.0, type_sim)
    deg_sim = float(np.minimum(fa["deg_hist"], fb["deg_hist"]).sum())
    pa, pb = fa["places"], fb["places"]
    place_sim = 1.0 - abs(pa - pb) / max(1.0, max(pa, pb))
    return float((type_sim + deg_sim + place_sim) / 3.0)


_REF_CACHE: Dict[str, np.ndarray] = {}


def _ref_u8(name: str) -> np.ndarray:
    """The reference pixels of a gallery scene (rendered once per process)."""
    if name not in _REF_CACHE:
        from Topology import placement_families_m9 as FM
        specs = {s.name: s for s in gallery()}
        _REF_CACHE[name] = specs[name].scene.render().image_u8
    return _REF_CACHE[name]


def _job(args: Tuple[str, str, List[str]]) -> Dict[str, Any]:
    """One query: its node-similarity and structure scores against every gallery image."""
    qkey, q_img_key, names = args
    import guard
    guard.activate()
    from Topology import placement_census as CEN
    from Topology.placement_cache import get_or_build

    imgs = np.load(os.path.join(ROOT, ".cache", "m12_fit_w_images.npz"))
    q = get_or_build(imgs[qkey], "m12sl|" + qkey)
    qf = _features(q)
    out = {"q": qkey, "cand": {}}
    qnodes = sorted(int(n) for n in q.subgraph.nodes)
    for name in names:
        g = get_or_build(_ref_u8(name), "m12sl|" + name)
        rows = CEN.similarity_rows(q, g, qnodes,
                                   sorted(int(n) for n in g.subgraph.nodes))
        best = [max((s for (_v, s) in rows.get(u, [])), default=0.0) for u in qnodes]
        out["cand"][name] = {"node": float(np.mean(best)) if best else 0.0,
                             "struct": _struct_sim(qf, _features(g))}
    return out


def recall_table(rows: List[Dict[str, Any]], lam: float) -> Dict[str, Any]:
    per_k: Dict[str, float] = {}
    ranks: List[int] = []
    for r in rows:
        truth = r["q"].split("|")[0]
        if truth not in r["cand"]:
            continue
        order = sorted(r["cand"], key=lambda c: (-(r["cand"][c]["node"]
                                                  + lam * r["cand"][c]["struct"]), c))
        ranks.append(order.index(truth) + 1)
    for K in KS:
        per_k["K%d" % K] = float(np.mean([r <= K for r in ranks])) if ranks else float("nan")
    return {"recall": per_k, "n": len(ranks),
            "mrr": float(np.mean([1.0 / r for r in ranks])) if ranks else float("nan")}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    prewarm()
    gal = gallery()
    names = [s.name for s in gal]
    qs = queries()[::QUERY_STRIDE]
    jobs = [(qk, qk, names) for (qk, *_r) in qs if qk.split("|")[0] in set(names)]
    print("%d queries x %d gallery images" % (len(jobs), len(names)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_job, jobs))
    res = {"queries": len(rows), "gallery": len(names),
           "node_only": recall_table(rows, 0.0),
           "structure_only": None, "blend": {}}
    # structure alone: rank by the structural score only
    ranks = []
    for r in rows:
        truth = r["q"].split("|")[0]
        if truth not in r["cand"]:
            continue
        order = sorted(r["cand"], key=lambda c: (-r["cand"][c]["struct"], c))
        ranks.append(order.index(truth) + 1)
    res["structure_only"] = {"recall": {"K%d" % K: float(np.mean([x <= K for x in ranks]))
                                        for K in KS} if ranks else {},
                             "n": len(ranks),
                             "mrr": float(np.mean([1.0 / x for x in ranks])) if ranks else float("nan")}
    for lam in BLEND_WEIGHTS:
        res["blend"]["lam=%.2f" % lam] = recall_table(rows, lam)
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
