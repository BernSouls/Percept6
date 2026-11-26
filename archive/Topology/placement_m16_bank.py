"""MISSION 16 acceptance D (second half) -- refit the weight on the new bank.

The bank: for every query with a truth partner, the APPEARANCE similarity and the
blended evidence E = plus_score - 0.5 of every admissible candidate, under the
SHIPPED M16 arm (graded constitution, population off).  Read offline: score =
(1-beta)*sim + beta*(0.5 + w*E) at the frozen beta; report top-1 and
true-vs-impostor AUC per w, and whether the optimum is interior.

Graphs come from the cache the graph stage already built, so this is a read, not a
re-measurement.

    python Topology/placement_m16_bank.py
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
RESULT = os.path.join(OUT, "m16_bank.json")
GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _job(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, pose, spec = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from core import keypoint_classes as KPC
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build

    from Topology.placement_m16_graphs import M16
    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = False          # THE SHIPPED ARM (see the report)
    KPC.unregister()
    KPC.register(str(P.KP_GRADED_VARIANT))
    img = R._transform if False else None
    from Topology.placement_m16_graphs import _image
    img = _image(tag)
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
    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    triples = PM.top_candidates(matrix, nodes_b, int(b["cand_cap"]))
    ctx = PM.context_mapping(triples, nodes_a, side_q, side_r)
    bank: List[Dict[str, Any]] = []
    for i, u in enumerate(nodes_a):
        vt = truth.get(u)
        if vt is None:
            continue
        row = [(int(nodes_b[j]), float(matrix[i][j]))
               for j in range(len(nodes_b)) if matrix[i][j] > 0.0]
        if len(row) < 2:
            continue
        bank.append({"u": int(u), "v": [v for (v, _s) in row],
                     "sim": [s for (_v, s) in row],
                     "E": [float(L.plus_score(u, v, ctx, side_q.incident,
                                              side_q.index, side_r.index, fit, pfit,
                                              a_e, a_g, slot_cap=int(L.SLOT_CAP),
                                              tol=(None if tol_of is None
                                                   else tol_of(u, v))) - 0.5)
                           for (v, _s) in row],
                     "true": [1 if v == vt else 0 for (v, _s) in row]})
    return {"tag": tag, "pose": pose, "bank": bank, "queries": len(bank)}


def _auc(pairs):
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def profile(bank: List[Dict[str, Any]], beta: float) -> List[Dict[str, Any]]:
    out = []
    for w in GRID:
        hits = used = 0
        pairs = []
        for q in bank:
            sc = [(1 - beta) * s + beta * (0.5 + w * e)
                  for s, e in zip(q["sim"], q["E"])]
            tr = [x for x, t in zip(sc, q["true"]) if t == 1]
            im = [x for x, t in zip(sc, q["true"]) if t == 0]
            if not tr or not im:
                continue
            used += 1
            hits += 1 if max(tr) > max(im) else 0
            pairs.append((max(tr), im))
        out.append({"w": float(w), "top1": hits / float(used or 1),
                    "auc": _auc(pairs), "queries": used})
    return out


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep-bank", action="store_true",
                    help="also write the candidate-level bank (138 MB; do not commit)")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    from Topology.placement_m16_graphs import IMAGES, POSES

    jobs = [(t, pose, spec) for t in IMAGES for pose, spec in POSES]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-8s queries %d" % (r["tag"], r["pose"], r["queries"]),
                  flush=True)
    from matching.placement_matcher import frozen
    beta = float(frozen()["beta"])
    res: Dict[str, Any] = {"cells": cells, "beta": beta, "grid": list(GRID),
                           "arm": "graded (KP_GRADED_SIMILARITY on, population off)"}
    for split in (None, "syn", "real"):
        bank = [q for c in cells if split is None or c["tag"].startswith(split)
                for q in c["bank"]]
        if not bank:
            continue
        prof = profile(bank, beta)
        key = "all" if split is None else split
        res["profile_" + key] = {"profile": prof,
                                 "queries": sum(p["queries"] for p in prof) // len(GRID),
                                 "best_w": max(prof, key=lambda r: r["auc"] or 0)["w"],
                                 "interior": max(prof, key=lambda r: r["auc"] or 0)["w"]
                                 not in (GRID[0], GRID[-1])}
    # THE RAW BANK IS 138 MB AND GITHUB REFUSES IT (found by the pre-receive hook,
    # not by a rule).  What the report needs is the PROFILE; the rows travel only
    # when asked for, and then to a git-ignored path.
    if not args.keep_bank:
        for c in res["cells"]:
            c["n_candidates"] = sum(len(q["v"]) for q in c["bank"])
            c["n_truth_in_row"] = sum(sum(q["true"]) for q in c["bank"])
            c.pop("bank")
        res["note"] = ("per-cell rows hold COUNTS only; the candidate-level bank is "
                       "138 MB and is written with --keep-bank to a path outside git")
    MF.write_result(RESULT, res)
    for key in ("all", "syn", "real"):
        p = res.get("profile_" + key)
        if not p:
            continue
        print("\n== %s (%d queries) best_w %s interior %s"
              % (key, p["queries"], p["best_w"], p["interior"]))
        print("   w     " + " ".join("%6.3f" % r["w"] for r in p["profile"]))
        print("   top1  " + " ".join("%6.3f" % r["top1"] for r in p["profile"]))
        print("   auc   " + " ".join("%6.3f" % (r["auc"] or 0) for r in p["profile"]))
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
