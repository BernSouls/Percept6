"""MISSION 21 -- the analysis the report quotes, all of it from cached tables.

Four questions, four answers, none of them a re-run of a graph:

1. THE SIGMA GRID, on all 40 queries.  The widths were chosen on the fit half and the
   registered defaults were kept (the grid is flat and the argmax sat on a boundary).  This
   prints the whole grid on the full query set as a DIAGNOSTIC -- it is not a choice, and the
   report says so next to it.
2. BEFORE vs AFTER, PAIRED.  The same 40 queries under both rulers, so the difference is
   summarised as flips (McNemar's b and c) and not only as two rates.  With n = 40 one query
   is worth 0.025 and three queries are not a difference.
3. WHAT PADDING WOULD HAVE FABRICATED.  The distribution of slot support per place: how many
   places have fewer than K neighbours at all.  Mission 20 padded those by repeating the last
   entry; Mission 21 masks them.
4. THE OVERLAP THE SCORES ACTUALLY USE.  `n_overlap_mean` on a seeded sample of pairs, i.e.
   how many slots two places share, which is what the masked kernel divides by.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--overlap-sample", type=int, default=100)
    ap.add_argument("--tag", default="m21_analysis")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_prefilter as PF
    from Topology import placement_m21_terms as T
    from Topology.placement_m21_fit import ensure_bags, split_objects
    from Topology.placement_m21_shortlist import geo_matrix, score_tasks, appearance
    if not P.PLACEMENT_M19:
        raise SystemExit("set PLACEMENT_M19=1")
    objs = split_objects("FIT", a.per_class)
    tbl = T.build("FIT", a.per_class, 2)
    res: Dict[str, Any] = {}

    # 1. the grid, all queries (diagnostic only)
    grid = []
    for k, (sd, sa) in enumerate(tbl["grid"]):
        M, gal, _q = geo_matrix(tbl, objs, k)
        r = score_tasks(M, objs, gal, "T3", 8)
        grid.append({"sigma_d": sd, "sigma_a": sa, "recall8_T3": r["recall"], "n": r["n"]})
    res["sigma_grid_all_queries"] = grid
    res["sigma_grid_note"] = ("diagnostic only: the widths were chosen on the fit half and "
                              "the registered defaults were kept; picking the max of this "
                              "table would be fitting on the report set")

    # 2. paired before/after
    before_path = os.path.join(OUT, "m21_before_fit_pc%d.json" % a.per_class)
    after_path = os.path.join(OUT, "m21_shortlist_fit_pc%d.json" % a.per_class)
    before = json.load(open(before_path))
    after = json.load(open(after_path))
    paired = {}
    for task in ("T1", "T2", "T3"):
        paired[task] = {"before": before["tasks"][task]["recall"],
                        "after": after["channels"]["geo"][task]["recall"], "n": 40}
    # recompute the v1 hit vector for the McNemar table (a re-measure, not a re-quote)
    bags = ensure_bags("FIT", 2, a.per_class)
    gal = sorted({o["ref"] for o in objs})
    Mv1 = np.zeros((len(objs), len(gal)), dtype=np.float64)
    for i, o in enumerate(objs):
        for j, r in enumerate(gal):
            s = PF.score_bags(bags[o["query"]]["bag"], bags[r]["bag"]).get("score")
            Mv1[i, j] = 0.0 if s is None else float(s)
    k_def = [kk for kk, (sd, sa) in enumerate(tbl["grid"])
             if abs(sd - PF.load_params()["sigma_d"]) < 1e-9
             and abs(sa - PF.load_params()["sigma_a"]) < 1e-9][0]
    Mv2, gal2, _q = geo_matrix(tbl, objs, k_def)
    assert gal == gal2
    for task in ("T1", "T2", "T3"):
        rows1 = score_tasks(Mv1, objs, gal, task, 8)["rows"]
        rows2 = score_tasks(Mv2, objs, gal, task, 8)["rows"]
        h1 = {r["query"]: r["hit"] for r in rows1}
        h2 = {r["query"]: r["hit"] for r in rows2}
        both = sum(1 for q in h1 if h1[q] and h2[q])
        b = sum(1 for q in h1 if h1[q] and not h2[q])
        c = sum(1 for q in h1 if not h1[q] and h2[q])
        paired[task].update({"both_hit": both, "v1_only": b, "v2_only": c,
                             "neither": len(h1) - both - b - c,
                             "net": c - b})
    res["paired_before_after"] = paired

    # 3. slot support (what padding fabricated)
    support: List[int] = []
    for o in objs:
        M = bags[o["ref"]]["bag_v2"]["M"]
        support.extend(int(x) for x in M.sum(axis=1))
    s = np.asarray(support)
    res["slot_support"] = {"places": int(s.size), "K": int(PF.load_params()["K"]),
                           "mean": float(s.mean()),
                           "frac_lt_K": float((s < PF.load_params()["K"]).mean()),
                           "frac_lt_6": float((s < 6).mean()),
                           "frac_lt_2": float((s < 2).mean()),
                           "p05": float(np.percentile(s, 5)),
                           "p50": float(np.percentile(s, 50))}

    # 4. the overlap the masked kernel divides by
    rng = np.random.default_rng(20261118)
    paths = sorted({o["ref"] for o in objs})
    std = {p: PF.standardize(bags[p]["bag_v2"], PF.load_params()) for p in paths}
    idx = [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths))]
    sel = rng.choice(len(idx), size=min(a.overlap_sample, len(idx)), replace=False)
    ovs = []
    for t in sel:
        i, j = idx[int(t)]
        out = PF.score_bags_v2(std[paths[i]], std[paths[j]], PF.load_params())
        if out["n_overlap_mean"] is not None:
            ovs.append(out["n_overlap_mean"])
    res["overlap"] = {"n_pairs": len(ovs), "mean": float(np.mean(ovs)),
                      "min": float(np.min(ovs)), "max": float(np.max(ovs)),
                      "seed": 20261118}

    # 5. THE O2 CHANNEL GRID, extended upward.  The shortlist run stopped its weight grid at
    # 0.8 and the fit half chose 0.5, but the appearance channel ALONE scored higher than the
    # union at every weight tested; a mix that is beaten by one of its own ingredients is not
    # a union, so the grid is extended to the appearance-dominant end and the choice is made
    # on the fit half exactly as registered.
    app_g = np.asarray([appearance(p) for p in gal])
    app_q = np.asarray([appearance(o["query"]) for o in objs])
    A = app_q @ app_g.T
    W = (0.0, 0.15, 0.3, 0.5, 0.8, 1.5, 3.0, 8.0)
    even = [i for i in range(len(objs)) if i % 2 == 0]
    gridw = []
    for w in W:
        mat = Mv2 + w * A
        full = score_tasks(mat, objs, gal, "T3", 8)
        half = score_tasks(mat[even], [objs[i] for i in even], gal, "T3", 8)
        gridw.append({"w": w, "T3_all": full["recall"], "n_all": full["n"],
                      "T3_fit_half": half["recall"], "n_fit_half": half["n"]})
    best_w = max(gridw, key=lambda g: (g["T3_fit_half"] or 0, -g["w"]))
    mat_best = Mv2 + best_w["w"] * A
    per_class: Dict[str, Dict[str, int]] = {}
    for task in ("T1", "T2", "T3"):
        rr = score_tasks(mat_best, objs, gal, task, 8)
        res.setdefault("per_class_best", {})[task] = {}
        for row in rr["rows"]:
            cls = row["query"].split("/")[0]
            d = res["per_class_best"][task].setdefault(cls, {"n": 0, "hits": 0})
            d["n"] += 1
            d["hits"] += row["hit"]
    res["channel_grid"] = {"grid": gridw, "chosen_on_fit_half": best_w,
                           "note": "O2: the weight is chosen on the fit half; the all-query "
                                   "column is the report and is not used to choose"}
    res["appearance_alone"] = {"T3_all": score_tasks(A, objs, gal, "T3", 8)["recall"],
                               "T2_all": score_tasks(A, objs, gal, "T2", 8)["recall"],
                               "n": 40}

    # 6. the asymmetry, PAIRED: the same pairs under both rulers
    rng2 = np.random.default_rng(7)
    pairs = [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths))]
    sel2 = [pairs[int(x)] for x in rng2.choice(len(pairs), size=120, replace=False)]
    kk = k_def
    d2v, d1v = [], []
    for i, j in sel2:
        a_ = tbl["S"][[p for p in range(len(tbl["paths"])) if tbl["paths"][p] == paths[i]][0],
                      [p for p in range(len(tbl["paths"])) if tbl["paths"][p] == paths[j]][0],
                      kk]
        b_ = tbl["S"][[p for p in range(len(tbl["paths"])) if tbl["paths"][p] == paths[j]][0],
                      [p for p in range(len(tbl["paths"])) if tbl["paths"][p] == paths[i]][0],
                      kk]
        if np.isfinite(a_) and np.isfinite(b_):
            d2v.append(abs(float(a_) - float(b_)))
        s1 = PF.score_bags(bags[paths[i]]["bag"], bags[paths[j]]["bag"]).get("score")
        s2 = PF.score_bags(bags[paths[j]]["bag"], bags[paths[i]]["bag"]).get("score")
        if s1 is not None and s2 is not None:
            d1v.append(abs(float(s1) - float(s2)))
    res["asymmetry_paired"] = {
        "n_pairs": len(sel2),
        "v2_mean_abs": float(np.mean(d2v)) if d2v else None,
        "v1_mean_abs": float(np.mean(d1v)) if d1v else None,
        "v2_p95": float(np.percentile(d2v, 95)) if d2v else None,
        "v1_p95": float(np.percentile(d1v, 95)) if d1v else None,
        "v2_level": float(np.mean([tbl["S"][i, j, kk] for i, j in
                                   [(0, 1), (2, 3), (4, 5)] if np.isfinite(
                                       tbl["S"][i, j, kk])])),
        "v1_level": float(np.mean([PF.score_bags(bags[paths[i]]["bag"],
                                                 bags[paths[j]]["bag"]).get("score") or 0.0
                                   for i, j in [(0, 1), (2, 3), (4, 5)]])),
        "seed": 7}

    out = os.path.join(OUT, a.tag + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("grid on all 40 queries (diagnostic):",
          "%s" % {round(g["sigma_d"], 2): [g["recall8_T3"] for g in grid
                                           if abs(g["sigma_d"] - round(g["sigma_d"], 2)) < 1e-9]
                  for g in grid[:1]})
    for g in grid:
        print("  sd %.2f sa %.2f -> %.3f" % (g["sigma_d"], g["sigma_a"], g["recall8_T3"]))
    print("\npaired before/after (n=40):")
    for task in ("T1", "T2", "T3"):
        p = paired[task]
        if "v1_only" in p:
            print("  %s before %.3f after %.3f | both %d v1-only %d v2-only %d "
                  "neither %d net %+d"
                  % (task, p["before"] or 0, p["after"] or 0, p["both_hit"],
                     p["v1_only"], p["v2_only"], p["neither"], p["net"]))
        else:
            print("  %s before %.3f after %.3f" % (task, p["before"] or 0, p["after"] or 0))
    ss = res["slot_support"]
    print("\nslot support: %d places, mean %.2f, %.1f%% below K, %.1f%% below 6, p50 %d"
          % (ss["places"], ss["mean"], 100 * ss["frac_lt_K"], 100 * ss["frac_lt_6"],
             ss["p50"]))
    ov = res["overlap"]
    print("overlap: %d pairs, mean %.1f slots (min %.1f, max %.1f)"
          % (ov["n_pairs"], ov["mean"], ov["min"], ov["max"]))
    print("\nO2 channel grid (T3 recall@8, fit half is the choice):")
    for g in gridw:
        print("  w %.2f -> fit half %.3f (n=%d) | all %.3f (n=%d)"
              % (g["w"], g["T3_fit_half"] or 0, g["n_fit_half"], g["T3_all"] or 0,
                 g["n_all"]))
    print("  appearance alone: T3 %.3f T2 %.3f (n=40)"
          % (res["appearance_alone"]["T3_all"] or 0, res["appearance_alone"]["T2_all"] or 0))
    ap_ = res["asymmetry_paired"]
    print("asymmetry on the same %d pairs: v2 mean %.4f p95 %.4f | v1 mean %.4f p95 %.4f"
          % (ap_["n_pairs"], ap_["v2_mean_abs"] or 0, ap_["v2_p95"] or 0,
             ap_["v1_mean_abs"] or 0, ap_["v1_p95"] or 0))
    print("level: v2 %.3f vs v1 %.3f (the two rulers do not live on the same scale)"
          % (ap_["v2_level"], ap_["v1_level"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
