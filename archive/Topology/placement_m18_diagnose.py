"""MISSION 18 -- the two diagnostics every other number depends on.

(1) WHERE THE 15 deg TRUE RESIDUALS LIVE.  O1's "non-flat where it matters" is defined by
    this distribution, not by the pooled one: the pooled bank is dominated by 0 deg pairs
    whose median residual is 0.021 tol, and a table fitted to that is flat exactly where
    15 deg needs slope.  Reported per pose and per image: median, quartiles, q90, and the
    share of true residuals above the support of the M17 table (1.8 tol).

(2) WHERE THE TRUE PARTNER SITS IN THE SIMILARITY ROW (O3).  Median rank, the share
    outside the candidate cap, and the share that is rank 1.  If it is outside the cap no
    context can reach it, and the fix belongs in the descriptor or candidate generation.

Read off the CACHED M17 graphs (`m16|<tag>|<pose>|on`), with the M17 shipped
configuration (assignment context, deployed table) -- the same cells every other number
in this mission is measured on.

    python Topology/placement_m18_diagnose.py --workers 2
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m18_diagnose.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")


def _job(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, pose, spec = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build
    from Topology.placement_m17_eval import M16, _image
    from core import keypoint_classes as KPC

    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    P.PLACEMENT_LLR_TABLE_M17 = False            # the deployed table, the incumbent
    P.PLACEMENT_CONTEXT_ASSIGN = True            # the M17 shipped context
    P.PLACEMENT_CONTEXT_ASSIGN_LAMBDA = 0.5
    PM.reset_cache()
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()

    img = _image(tag)
    img1 = R._transform(img, spec)
    g0 = get_or_build(img, "m16|%s|orig|on" % tag)
    g1 = get_or_build(img1, "m16|%s|%s|on" % (tag, pose))
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    sq = PM.build_side_cached(g1, img1)
    sr = PM.build_side_cached(g0, img)
    truth = PL.lens_truth(g1, g0, spec, img.shape[:2])
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    tol_of = PM._spacing_tol_fn(sq, sr)
    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    cap = int(b["cand_cap"])
    triples = PM.top_candidates(matrix, nodes_b, cap)
    ctx = PM.context_mapping(triples, nodes_a, sq, sr)

    res_true: List[float] = []
    res_best_imp: List[float] = []
    ranks: List[int] = []
    n_row: List[int] = []
    outside_cap = 0
    n = 0
    ctx_ok = 0
    for i, u in enumerate(nodes_a):
        vt = truth.get(u)
        if vt is None:
            continue
        row = [(v, matrix[i][idx_b[v]]) for v in nodes_b if matrix[i][idx_b[v]] > 0.0]
        if not row:
            continue
        n += 1
        n_row.append(len(row))
        ctx_ok += 1 if ctx.get(u) == vt else 0
        order = sorted(row, key=lambda t: -t[1])
        rank = next((k for k, (v, _s) in enumerate(order) if v == vt), None)
        if rank is None:
            continue                    # not admissible at all: a different failure
        ranks.append(rank + 1)
        if rank + 1 > cap:
            outside_cap += 1
        slots = L.slots_for_candidate(u, vt, ctx, sq.incident, sq.index, sr.index,
                                      tol=(None if tol_of is None else tol_of(u, vt)))
        got = [float(x[2]) for x in slots if x[2] is not None]
        res_true.extend(got)
        best = order[0][0]
        if best != vt:
            sb = L.slots_for_candidate(u, best, ctx, sq.incident, sq.index, sr.index,
                                       tol=None if tol_of is None else tol_of(u, best))
            res_best_imp.extend([float(x[2]) for x in sb if x[2] is not None])
    return {"tag": tag, "pose": pose, "queries": n, "ctx_ok": ctx_ok,
            "res_true": res_true, "res_best_imp": res_best_imp,
            "ranks": ranks, "n_row": n_row, "outside_cap": outside_cap,
            "cap": cap}


def _q(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
            "q90": float(np.percentile(a, 90)),
            "share_gt_1.8": float((a > 1.8).mean()),
            "share_gt_0.3": float((a > 0.3).mean())}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--images", default=",".join(DEV))
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    tags = tuple(args.images.split(","))
    jobs = [(t, pose, spec) for (pose, spec) in POSES for t in tags]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-8s q %4d ctx %.3f  rank-median %.1f  outside-cap %d"
                  % (r["tag"], r["pose"], r["queries"],
                     r["ctx_ok"] / max(1, r["queries"]),
                     st.median(r["ranks"]) if r["ranks"] else -1, r["outside_cap"]),
                  flush=True)
    res: Dict[str, Any] = {"cells": cells, "cap": cells[0]["cap"] if cells else None,
                           "units": "residuals in tolunits; ranks are 1-based positions "
                                    "in the similarity row"}
    for pose in [p[0] for p in POSES]:
        rr = [c for c in cells if c["pose"] == pose]
        if not rr:
            continue
        rt = [x for c in rr for x in c["res_true"]]
        rb = [x for c in rr for x in c["res_best_imp"]]
        rk = [x for c in rr for x in c["ranks"]]
        res.setdefault("by_pose", {})[pose] = {
            "images": len(rr), "queries": sum(c["queries"] for c in rr),
            "ctx_accuracy": (sum(c["ctx_ok"] for c in rr)
                             / max(1, sum(c["queries"] for c in rr))),
            "true_residual": _q(rt), "best_impostor_residual": _q(rb),
            "rank": {"n": len(rk), "median": float(np.median(rk)) if rk else None,
                     "share_rank1": (sum(1 for x in rk if x == 1) / len(rk)) if rk else None,
                     "share_outside_cap": (sum(c["outside_cap"] for c in rr) / len(rk))
                     if rk else None,
                     "share_outside_10": (sum(1 for x in rk if x > 10) / len(rk))
                     if rk else None},
            "row_size": {"median": float(np.median([x for c in rr for x in c["n_row"]]))},
        }
        res.setdefault("per_image", {}).setdefault(pose, {})[
            "residual_median"] = {c["tag"]: (_q(c["res_true"]).get("median"))
                                  for c in rr}
        res["per_image"][pose]["ctx_accuracy"] = {c["tag"]: c["ctx_ok"] / max(1, c["queries"])
                                                  for c in rr}
        res["per_image"][pose]["rank_median"] = {
            c["tag"]: (float(np.median(c["ranks"])) if c["ranks"] else None) for c in rr}
        res["per_image"][pose]["outside_cap_share"] = {
            c["tag"]: (c["outside_cap"] / len(c["ranks"])) if c["ranks"] else None
            for c in rr}
    MF.write_result(RESULT, res)
    for pose, d in res["by_pose"].items():
        t = d["true_residual"]
        print("\n%-8s n=%d q=%d ctx %.3f | true res med %.3f [%.3f-%.3f] q90 %.3f "
              ">0.3 %.1f%% >1.8 %.1f%% | rank med %.0f  rank1 %.3f  outside cap %.3f"
              % (pose, d["images"], d["queries"], d["ctx_accuracy"], t["median"],
                 t["q25"], t["q75"], t["q90"], 100 * t["share_gt_0.3"],
                 100 * t["share_gt_1.8"], d["rank"]["median"], d["rank"]["share_rank1"],
                 d["rank"]["share_outside_cap"]))
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
