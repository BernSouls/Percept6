"""MISSION 10 STEP 3 -- scene retrieval with the fixed shortlist.

Run after ``placement_m10_gallery.shortlist`` has reported recall >= 0.95.  The
two arms differ ONLY in the pair score (defined in the gallery module's docstring
before the run):

  node-only   sum of the shortlist pair's best per-node similarities
  placement   sum over the pairs the deployed matcher MAPPED of the blended score
              it used, (1-beta)*sim + beta*term at the matcher's own context

Scoring the placement arm uses the same frozen bundle as the matcher, so a node
the matcher leaves unmapped contributes nothing -- a matcher that maps less is not
rewarded for it.

Checkpointed: every query is appended to ``m10_step3_progress.jsonl`` so a
multi-hour run can be resumed and reported partially without pretending.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PROGRESS = os.path.join(OUT, "m10_step3_progress.jsonl")


def _one_query(args: Tuple[str, List[str], List[float]]) -> Dict[str, Any]:
    qkey, refs, sims = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    from Topology.placement_m10_gallery import QUERIES, gallery_specs

    specs = {s.name: s for s in gallery_specs()}
    qname, label = qkey.split("|")
    sc, _pv = FM.pose_fn(specs[qname], dict(QUERIES)[label])
    rq = sc.render()
    gq = get_or_build(rq.image_u8, qkey)
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g, beta = float(b["_a_exist"]), float(b["_a_geom"]), float(b["beta"])
    scored: Dict[str, float] = {}
    info: Dict[str, Any] = {}
    for rname, node_only in zip(refs, sims):
        spec_r = specs[rname]
        rr = spec_r.scene.render()
        gr = get_or_build(rr.image_u8, rname + "|ref")
        side_q = PM.build_side(gq, rq.image_u8)
        side_r = PM.build_side(gr, rr.image_u8)
        P.PLACEMENT_MATCHER_ENABLED = True
        P.PLACEMENT_MATCHER_PRESENCE = False
        t = time.time()
        res = joint_mcs(gq.subgraph, gr.subgraph, gq.sigs, gr.sigs,
                        placement_images=(rq.image_u8, rr.image_u8))
        dt = time.time() - t
        P.PLACEMENT_MATCHER_ENABLED = False
        mapping = {int(k): int(v) for k, v in res.mapping.items()}
        nodes_q = sorted(int(n) for n in gq.subgraph.nodes)
        rows = similarity_rows(gq, gr, nodes_q,
                               sorted(int(n) for n in gr.subgraph.nodes))
        total = 0.0
        n_mapped = 0
        for u, v in mapping.items():
            row = rows.get(u) or []
            sim = None
            for (vv, s) in row:
                if int(vv) == int(v):
                    sim = float(s)
                    break
            if sim is None:
                continue
            term = L.plus_score(u, int(v), mapping, side_q.incident, side_q.index,
                                side_r.index, fit, pfit, a_e, a_g)
            total += (1.0 - beta) * sim + beta * float(term)
            n_mapped += 1
        scored[rname] = float(total)
        info[rname] = {"node_only": float(node_only), "mapped": int(n_mapped),
                       "match_s": float(dt)}
    keep = sorted(info, key=lambda r: -info[r]["node_only"])
    return {"query": qkey, "truth": qname, "placement": scored,
            "node_only": {r: info[r]["node_only"] for r in keep},
            "mapped": {r: info[r]["mapped"] for r in keep},
            "match_s": {r: info[r]["match_s"] for r in keep},
            "truth_shortlisted": qname in scored}


def _rank(scores: Dict[str, float], truth: str) -> int:
    order = sorted(scores.items(), key=lambda t: (-t[1], t[0]))
    names = [n for n, _ in order]
    return names.index(truth) + 1 if truth in names else 10 ** 6


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="m10_step3")
    args = ap.parse_args()

    with open(os.path.join(OUT, "m10_shortlist.json")) as fh:
        sl = json.load(fh)
    if float(sl["recall_at_K"]) < 0.95:
        print("REFUSED: shortlist recall@%d is %.3f, below the 0.95 rule"
              % (sl["K"], sl["recall_at_K"]))
        return 2
    print("shortlist recall@%d = %.3f -> judging" % (sl["K"], sl["recall_at_K"]))

    done: Dict[str, Any] = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[r["query"]] = r
                except Exception:
                    continue
    todo = []
    for qkey, v in sl["per_query"].items():
        if qkey in done:
            continue
        refs = list(v["top"])
        sims = [float(v["scores"][r]) for r in refs]
        todo.append((qkey, refs, sims))
    print("%d queries already done, %d to go" % (len(done), len(todo)), flush=True)

    t0 = time.time()
    with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_one_query, todo):
            done[r["query"]] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("  %-22s %.0fs" % (r["query"], time.time() - t0), flush=True)

    res: Dict[str, Any] = {"queries": len(done), "arms": {}}
    for arm in ("node_only", "placement"):
        ranks = {q: _rank(r[arm], r["truth"]) for q, r in done.items()}
        res["arms"][arm] = {
            "top1": float(np.mean([1.0 if k == 1 else 0.0 for k in ranks.values()])),
            "mrr": float(np.mean([1.0 / k for k in ranks.values()])),
            "recall": float(np.mean([1.0 if k <= sl["K"] else 0.0 for k in ranks.values()])),
            "ranks": ranks}
        cond = {q: k for q, k in ranks.items() if k <= sl["K"]}
        res["arms"][arm]["top1_conditional"] = (
            float(np.mean([1.0 if k == 1 else 0.0 for k in cond.values()]))
            if cond else float("nan"))
        res["arms"][arm]["mrr_conditional"] = (
            float(np.mean([1.0 / k for k in cond.values()])) if cond else float("nan"))
        res["arms"][arm]["n_conditional"] = len(cond)
    d = np.array([(1.0 if res["arms"]["placement"]["ranks"][q] == 1 else 0.0)
                  - (1.0 if res["arms"]["node_only"]["ranks"][q] == 1 else 0.0)
                  for q in done])
    per_scene: Dict[str, List[float]] = {}
    for q in done:
        per_scene.setdefault(q.split("|")[0], []).append(
            (1.0 if res["arms"]["placement"]["ranks"][q] == 1 else 0.0)
            - (1.0 if res["arms"]["node_only"]["ranks"][q] == 1 else 0.0))
    sm = {k: float(np.mean(v)) for k, v in per_scene.items()}
    arr = np.array(list(sm.values()))
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    res["placement_minus_node_top1"] = {
        "mean": float(d.mean()), "n_queries": int(d.size),
        "per_scene": sm, "se_scene": se, "mde": 2.8 * se,
        "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
    res["cost"] = {"matches": int(sum(len(r["placement"]) for r in done.values())),
                   "seconds": float(time.time() - t0)}
    path = os.path.join(OUT, args.tag + ".json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    for arm in ("node_only", "placement"):
        a = res["arms"][arm]
        print("%-10s top1 %.3f  mrr %.3f  |  conditional top1 %.3f  mrr %.3f (n %d)"
              % (arm, a["top1"], a["mrr"], a["top1_conditional"],
                 a["mrr_conditional"], a["n_conditional"]))
    print("placement - node_only on top-1: %+0.3f  se %.3f  mde %.3f"
          % (res["placement_minus_node_top1"]["mean"],
             res["placement_minus_node_top1"]["se_scene"],
             res["placement_minus_node_top1"]["mde"]))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
