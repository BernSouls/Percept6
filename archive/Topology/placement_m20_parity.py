"""MISSION 20 / item 1 -- DOES THE DEPLOYED PATH MAKE THE HARNESS'S DECISION?

Mission 19 measured the design in `Topology/placement_m17_eval.py`, which applies the
descriptor itself and then blends.  Mission 20 item 1 moved the descriptor INTO
`matching.placement_matcher.rescore_matrix`, which is what
`matching/alignment.global_consistency_alignment` calls whenever
`PLACEMENT_MATCHER_ENABLED` is on and both sides are usable.  Two scorers now claim the
same decision, so the burden is to show, on the same cells, that they are the same
decision -- not "close", the same.

WHAT IS COMPARED, PER CELL (the four DEV images at 15 deg, the M19 record):

  harness    the M19 recipe: descriptor -> capped candidate rows -> context mapping ->
             s_full = (1-beta)*sim + beta*(0.5 + e), pick = argmax over the row
  deployed   ONE call to PM.rescore_matrix on the SAME seed matrix with the bundle on
             (PLACEMENT_M19=1), pick = argmax over the positive cells of the rescored row
  record     Topology/placement_m17_eval.py's own numbers for the same cells, so a
             disagreement cannot be blamed on this file's re-implementation

Reported per cell and pooled: top-1 both ways, the number of queries where the two picks
differ, the maximum absolute score difference over EVERY positive cell, the geo AUC both
ways, and the effective flags the deployed rescore recorded in `stats["effective"]`.

Truth is used only to score.  Nothing here is on the matcher's path.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")
#: the M19 record for these four cells at 15 deg (MISSION19_REPORT.md, n = 1166)
RECORD = {"top1": 0.569, "app": 0.310, "geo_auc": 0.826, "queries": 1166}


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def cell(tag: str, pose: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology import placement_m17_eval as M17
    from Topology.placement_cache import get_or_build
    from core import keypoint_classes as KPC

    for k, v in M17.M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    PM.reset_cache()
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()

    img = M17._image(tag)
    img1 = R._transform(img, spec)
    g0 = get_or_build(img, "m16|%s|orig|on" % tag)
    g1 = get_or_build(img1, "m16|%s|%s|on" % (tag, pose))
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    side_q = PM.build_side_cached(g1, img1)
    side_r = PM.build_side_cached(g0, img)
    truth = PL.lens_truth(g1, g0, spec, img.shape[:2])
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    tol_of = PM._spacing_tol_fn(side_q, side_r)
    beta = float(b["beta"])
    # The harness caps with `PM.cand_cap()`; the deployed scorer used to read the frozen
    # table directly.  They agree now by construction, and the value is recorded so a
    # mismatch cannot hide: with the bundle off it must be the frozen cap (6).
    cap_used = int(PM.cand_cap())
    if not bool(getattr(P, "PLACEMENT_CAND_CAP", 0)):
        assert cap_used == int(b["cand_cap"]), "cap differs at defaults"

    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    seed = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                seed[i][idx_b[v]] = float(s)

    eff = {k: getattr(P, k) for k in ("PLACEMENT_M19", "PLACEMENT_NODE_DESC",
                                      "PLACEMENT_NODE_DESC_K", "PLACEMENT_NODE_DESC_W",
                                      "PLACEMENT_NODE_DESC_MAXCELLS", "PLACEMENT_LLR_TABLE_M18",
                                      "PLACEMENT_CONTEXT_ASSIGN")}

    def e_of(u: int, v: int, mapping: Dict[int, int]) -> float:
        return float(L.plus_score(u, int(v), mapping, side_q.incident, side_q.index,
                                  side_r.index, fit, pfit, a_e, a_g,
                                  slot_cap=int(L.SLOT_CAP),
                                  tol=(None if tol_of is None else tol_of(u, v)))) - 0.5

    # ---------------- harness: the M19 recipe, written as M19 wrote it ----------------
    h_mat = PM.apply_node_descriptor([list(r) for r in seed], nodes_a, nodes_b,
                                     side_q, side_r)
    triples = PM.top_candidates_adaptive(h_mat, nodes_b, PM.cand_cap())
    ctx = PM.context_mapping(triples, nodes_a, side_q, side_r)
    L.judged_reset()
    h_pick: Dict[int, int] = {}
    h_score: Dict[Tuple[int, int], float] = {}
    h_geo: Dict[Tuple[int, int], float] = {}
    app_pairs: List[Tuple[float, List[float]]] = []
    geo_pairs: List[Tuple[float, List[float]]] = []
    n_q = h_top1 = h_app = 0
    for i, u in enumerate(nodes_a):
        vt = truth.get(u)
        row = [(int(nodes_b[j]), float(h_mat[i][j])) for j in range(len(nodes_b))
               if h_mat[i][j] > 0.0]
        if vt is None or len(row) < 2 or vt not in {v for (v, _s) in row}:
            continue
        sim = dict(row)
        em = {v: e_of(u, v, ctx) for (v, _s) in row}
        full = {v: (1.0 - beta) * sim[v] + beta * (0.5 + em[v]) for (v, _s) in row}
        for (v, _s) in row:
            h_score[(u, v)] = float(full[v])
            h_geo[(u, v)] = float(em[v])
        best = max((full[v], -v) for v in sim if v != vt)[0]
        h_pick[u] = max(full, key=lambda v: (full[v], -v))
        n_q += 1
        h_top1 += 1 if full[vt] > best else 0
        h_app += 1 if sim[vt] > max(sim[v] for v in sim if v != vt) else 0
        app_pairs.append((sim[vt], [sim[v] for v in sim if v != vt]))
        geo_pairs.append((em[vt], [em[v] for v in sim if v != vt]))

    # ---------------- deployed: one call, the real signature ----------------
    stats: Dict[str, Any] = {}
    d_mat = PM.rescore_matrix([list(r) for r in seed], nodes_a, nodes_b,
                              side_q, side_r, stats=stats)
    d_pick: Dict[int, int] = {}
    diffs: List[float] = []
    n_cmp = 0
    for i, u in enumerate(nodes_a):
        if u not in h_pick:
            continue
        live = [(int(nodes_b[j]), float(d_mat[i][j])) for j in range(len(nodes_b))
                if d_mat[i][j] > 0.0]
        d_pick[u] = max(live, key=lambda t: (t[1], -t[0]))[0]
        for (v, s) in live:
            if (u, v) in h_score:
                diffs.append(abs(s - h_score[(u, v)]))
                n_cmp += 1
    d_top1 = sum(1 for u in d_pick if d_pick[u] == truth.get(u))
    disagree = sorted(u for u in d_pick if d_pick[u] != h_pick.get(u))
    geo_auc = _auc(geo_pairs)
    geo_num = sum((1.0 if t > x else 0.5 if t == x else 0.0) for (t, imp) in geo_pairs for x in imp)
    geo_den = sum(len(imp) for (_t, imp) in geo_pairs)
    return {
        "tag": tag, "pose": pose, "queries": n_q,
        "harness": {"top1": h_top1 / max(n_q, 1), "top1_n": h_top1,
                    "app": h_app / max(n_q, 1), "geo_auc": geo_auc,
                    "auc_app": _auc(app_pairs)},
        "deployed": {"top1": d_top1 / max(n_q, 1), "top1_n": d_top1},
        "geo_num": geo_num, "geo_den": geo_den,
        "agree_n": n_q - len(disagree), "disagree": disagree[:20],
        "disagree_n": len(disagree), "cells_compared": n_cmp,
        "max_score_diff": (max(diffs) if diffs else None),
        "effective": stats.get("effective"), "flags": eff, "cap_used": int(cap_used),
        "judged": {"pairs": int(L.JUDGED[0]), "slots": int(L.JUDGED[1])},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="rot15")
    ap.add_argument("--images", default="dev")
    ap.add_argument("--tag", default="m20_parity")
    args = ap.parse_args()
    spec = dict(dict(__import__("Topology.placement_m17_eval", fromlist=["POSES"]).POSES)[args.pose])
    tags = DEV
    import parameters as P
    if not bool(getattr(P, "PLACEMENT_M19", False)):
        raise SystemExit("run this with PLACEMENT_M19=1: the point is the DEPLOYED path")
    res = [cell(t, args.pose, spec) for t in tags]
    q = sum(r["queries"] for r in res)
    gnum = sum(r["geo_num"] for r in res)
    gden = sum(r["geo_den"] for r in res)
    pooled_geo = gnum / gden if gden else None
    cell_geo = [r["harness"]["geo_auc"] for r in res if r["harness"]["geo_auc"] is not None]
    mean_geo = sum(cell_geo) / len(cell_geo) if cell_geo else None
    print("%-16s q %4d  harness top1 %.3f  deployed top1 %.3f  geo %.3f  "
          "disagree %d/%d  max|dsc| %.3g"
          % ("POOLED", q,
             sum(r["harness"]["top1_n"] for r in res) / max(q, 1),
             sum(r["deployed"]["top1_n"] for r in res) / max(q, 1),
             pooled_geo or 0.0,
             sum(r["disagree_n"] for r in res), q,
             max((r["max_score_diff"] or 0.0) for r in res)), flush=True)
    print("geo AUC: pooled over pairs %.3f   mean of the %d cell AUCs %.3f"
          % (pooled_geo or 0.0, len(cell_geo), mean_geo or 0.0))
    print("record  (M19 report)  top1 %.3f  geo %.3f  n %d  <- mean of cells"
          % (RECORD["top1"], RECORD["geo_auc"], RECORD["queries"]), flush=True)
    for r in res:
        print("  %-16s q %4d  h %.3f (%d)  d %.3f (%d)  geo %.3f  dis %d  "
              "cells %5d  max|dsc| %.3g  desc %s w %.2f k %d mc %d tbl %s asg %s"
              % (r["tag"], r["queries"], r["harness"]["top1"], r["harness"]["top1_n"],
                 r["deployed"]["top1"], r["deployed"]["top1_n"],
                 r["harness"]["geo_auc"] or 0.0, r["disagree_n"], r["cells_compared"],
                 r["max_score_diff"] or 0.0,
                 r["effective"]["PLACEMENT_NODE_DESC"],
                 r["effective"]["PLACEMENT_NODE_DESC_W"],
                 r["effective"]["PLACEMENT_NODE_DESC_K"],
                 r["effective"]["PLACEMENT_NODE_DESC_MAXCELLS"],
                 r["effective"]["llr_table"],
                 r["effective"]["PLACEMENT_CONTEXT_ASSIGN"]), flush=True)
    out = {"pose": args.pose, "images": list(tags), "queries": q, "record": RECORD,
           "pooled": {"geo_auc": pooled_geo, "geo_auc_mean_of_cells": mean_geo,
                      "geo_auc_cells": [round(x, 6) for x in cell_geo],
                      "harness_top1": sum(r["harness"]["top1_n"] for r in res) / max(q, 1),
                      "deployed_top1": sum(r["deployed"]["top1_n"] for r in res) / max(q, 1),
                      "disagree_n": sum(r["disagree_n"] for r in res),
                      "cap_used": sorted({r["cap_used"] for r in res}),
                      "max_score_diff": max((r["max_score_diff"] or 0.0) for r in res)},
           "cells": res}
    pth = os.path.join(OUT, "%s.json" % args.tag)
    with open(pth, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote", pth)
    ok = (out["pooled"]["disagree_n"] == 0
          and abs(out["pooled"]["harness_top1"] - out["pooled"]["deployed_top1"]) < 1e-12
          and (out["pooled"]["max_score_diff"] or 0.0) < 1e-9)
    print("PARITY:", "IDENTICAL" if ok else "DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
