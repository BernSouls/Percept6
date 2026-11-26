"""MISSION 10 STEP 3 -- scene retrieval, with the shortlist fixed first.

Mission 9 measured the shortlist and stopped there: on 64 scenes from one
generator the cheap node similarity plateaus at recall@8 = 0.875 because every
scene shares the same seven-element structure, and the rule says the shortlist is
repaired BEFORE either arm is judged.

The repair, measured in Mission 9 and used here:

  gallery   24 scenes of MIXED structure -- 16 irregular scenes (seven elements,
            two appearance-identical twin pairs) and 8 out-of-plane scenes (two
            bodies), all at reference pose,
  *  queries 8 of the irregular gallery scenes under TWO pose changes: in-plane
            12 degrees and out-of-plane y 20 degrees,
  *  one shortlist for BOTH arms, top K by cheap node similarity (no aligner),
  *  K = 16, which is the first setting at which the mixed gallery is perfect in
            Mission 9 (recall@16 = 1.000, recall@8 = 0.938).

Pair scores, fixed here before the run:

  node-only     sum over the shortlist pair's best per-node similarities -- the
                same "sum over matched pairs" shape as the placement arm, with
                raw similarity in place of the blended score, so the two arms
                differ only in the score and not in the aggregation.
  placement     sum over the pairs the deployed matcher's OWN mapping contains
                of the blended per-pair score it used: (1-beta)*sim + beta*term,
                evaluated at the matcher's context.  Pairs the matcher did not
                map contribute nothing, so a matcher that maps less is not
                rewarded for it.

Reported: shortlist recall@K (must reach 0.95 before anything is judged), top-1
and MRR overall and conditional on the truth being shortlisted, scene-clustered
intervals and MDE, and the compute actually spent.
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

K = 16
N_IRREGULAR = 16
N_QUERY_SCENES = 8

QUERIES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("q_inplane12", {"rot_deg": 12.0}),
    ("q_y20", {"rot_y_deg": 20.0}),
)


def gallery_specs() -> List[Any]:
    """The mixed 24-scene gallery; queries are the first eight irregular ones."""
    from Topology import placement_families_m9 as FM
    specs = []
    for s in FM.all_specs_irregular(12)[:N_IRREGULAR]:
        specs.append(s)
    for s in FM.all_specs_oop(4):          # 8 scenes: two per shape
        specs.append(s)
    return specs


def queries_of(spec: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Only the query scenes get rendered under a pose change."""
    if spec.family == "IRREGULAR" and int(spec.name.rsplit("_", 1)[-1]) < N_QUERY_SCENES:
        return list(QUERIES)
    return []


def _sim_pair(args: Tuple[str, str]) -> Tuple[str, str, float]:
    qkey, rname = args
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    specs = {s.name: s for s in gallery_specs()}
    qname, label = qkey.split("|")
    view = dict(QUERIES)[label]
    sc, _pv = FM.pose_fn(specs[qname], view)
    q = get_or_build(sc.render().image_u8, qkey)
    ref = get_or_build(specs[rname].scene.render().image_u8, rname + "|ref")
    rows = similarity_rows(q, ref, sorted(int(n) for n in q.subgraph.nodes),
                           sorted(int(n) for n in ref.subgraph.nodes))
    best = [r[0][1] for r in rows.values() if r]
    return qkey, rname, float(np.sum(best))      # "sum over matched pairs"


def shortlist(workers: int = 2) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    specs = gallery_specs()
    names = [s.name for s in specs]
    qkeys = ["%s|%s" % (s.name, lab) for s in specs if queries_of(s)
             for lab, _v in QUERIES]
    jobs = [(qk, rn) for qk in qkeys for rn in names]
    t0 = time.time()
    scores: Dict[str, Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for qk, rn, sc in ex.map(_sim_pair, jobs, chunksize=4):
            scores.setdefault(qk, {})[rn] = sc
    per: Dict[str, Any] = {}
    for qk, d in scores.items():
        order = sorted(d.items(), key=lambda t: (-t[1], t[0]))
        rank = [n for n, _ in order].index(qk.split("|")[0]) + 1
        per[qk] = {"rank": rank, "top": [n for n, _ in order[:K]],
                   "scores": {n: v for n, v in order[:K]}}
    out = {"K": K, "gallery": len(names), "queries": len(per),
           "recall_at_K": float(np.mean([1.0 if v["rank"] <= K else 0.0
                                         for v in per.values()])),
           "seconds": time.time() - t0, "per_query": per}
    for lab, _v in QUERIES:
        sub = [v for k, v in per.items() if k.endswith(lab)]
        out["recall_at_K|" + lab] = float(np.mean(
            [1.0 if v["rank"] <= K else 0.0 for v in sub]))
    with open(os.path.join(OUT, "m10_shortlist.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("SHORTLIST recall@%d = %.3f over %d queries (%.0fs)"
          % (K, out["recall_at_K"], len(per), out["seconds"]), flush=True)
    return out
