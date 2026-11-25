"""MISSION 18 / O1 -- the residual bank the table is fitted from, per POSE.

M17's refit failed for a reason that had nothing to do with monotonicity: it was fitted on
a pool dominated by 0 deg pairs (median true residual 0.013 tol), and at 15 deg the TRUE
residuals are ten times larger (median 0.135, IQR [0.062, 0.230] -- `m18_diagnose.json`).
A table calibrated to the first regime puts a hard negative LLR exactly where the second
regime's true mass is: the typical 15 deg true pair (0.135 tol) falls in M17's bin 2 and
reads -0.89 nats, where the deployed table's 0.25-wide first bin reads +1.386.

So the bank is recorded PER POSE and PER IMAGE, as histograms on a fine grid (0.05 tol,
120 bins over 6.0), so any candidate table's binning and any pose weighting can be formed
from it without re-measuring.

    python Topology/placement_m18_bank.py --workers 2
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
RESULT = os.path.join(OUT, "m18_bank_cap24.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")
GRID_MAX = 6.0
GRID_BINS = 120
TOP_IMP = 6


def _job(args: Tuple[str, str, Dict[str, Any], int]) -> Dict[str, Any]:
    tag, pose, spec, cap = args
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
    P.PLACEMENT_LLR_TABLE_M17 = False
    P.PLACEMENT_CONTEXT_ASSIGN = True
    P.PLACEMENT_CONTEXT_ASSIGN_LAMBDA = 0.5
    P.PLACEMENT_CONTEXT_CONSENSUS = False
    P.PLACEMENT_CAND_CAP = int(cap)
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
    tol_of = PM._spacing_tol_fn(sq, sr)
    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    triples = PM.top_candidates(matrix, nodes_b, max(cap, TOP_IMP))
    ctx = PM.context_mapping(triples, nodes_a, sq, sr)

    ht = np.zeros(GRID_BINS, dtype=np.int64)
    hw = np.zeros(GRID_BINS, dtype=np.int64)
    n_q = 0
    edges = np.linspace(0.0, GRID_MAX, GRID_BINS + 1)

    def add(hist, res):
        for r in res:
            if not np.isfinite(r):
                continue
            k = int(np.clip(np.floor(r / (GRID_MAX / GRID_BINS)), 0, GRID_BINS - 1))
            hist[k] += 1

    for i, u in enumerate(nodes_a):
        vt = truth.get(u)
        if vt is None:
            continue
        row = [(v, matrix[i][idx_b[v]]) for v in nodes_b if matrix[i][idx_b[v]] > 0.0]
        if not row:
            continue
        n_q += 1
        slots = L.slots_for_candidate(u, vt, ctx, sq.incident, sq.index, sr.index,
                                      tol=(None if tol_of is None else tol_of(u, vt)))
        add(ht, [float(x[2]) for x in slots if x[2] is not None])
        imps = [v for (v, _s) in sorted(row, key=lambda t: -t[1]) if v != vt][:TOP_IMP]
        for v in imps:
            sl = L.slots_for_candidate(u, v, ctx, sq.incident, sq.index, sr.index,
                                       tol=(None if tol_of is None else tol_of(u, v)))
            add(hw, [float(x[2]) for x in sl if x[2] is not None])
    return {"tag": tag, "pose": pose, "queries": n_q,
            "hist_true": [int(x) for x in ht], "hist_wrong": [int(x) for x in hw],
            "n_true": int(ht.sum()), "n_wrong": int(hw.sum())}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--images", default=",".join(DEV))
    ap.add_argument("--cap", type=int, default=6)
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    tags = tuple(args.images.split(","))
    jobs = [(t, pose, spec, args.cap) for (pose, spec) in POSES for t in tags]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-8s q %4d true %6d wrong %6d"
                  % (r["tag"], r["pose"], r["queries"], r["n_true"], r["n_wrong"]),
                  flush=True)
    res = {"cells": cells, "grid_max": GRID_MAX, "grid_bins": GRID_BINS,
           "grid_width": GRID_MAX / GRID_BINS, "top_imp": TOP_IMP,
           "context": "assignment (M17 shipped), deployed table", "cap": int(args.cap),
           "units": "residuals in tolunits; histograms over [0, 6.0] in 0.05 bins"}
    MF.write_result(RESULT, res)
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
