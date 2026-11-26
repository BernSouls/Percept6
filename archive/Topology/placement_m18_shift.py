"""MISSION 18 / O2 -- WHOSE SHIFT IS IT?  The consensus, measured against the truth's own.

The rotation consensus helps three cells at 15 deg and destroys one (real|cell: top-1
0.516 -> 0.147 at cap 6).  A consensus that is CONFIDENT AND WRONG is the only mechanism
that can do that, so this script measures, per cell:

* ``true_shift`` -- the density mode of the shifts implied by the TRUE pairs alone (the
  shift the image rotation actually produced; diagnostic only, never used by the matcher);
* ``modes`` and ``best_weight_share`` -- what the candidate rows vote for, which is all
  the matcher is allowed to see;
* ``hit`` -- whether a consensus mode lands within the tolerance of the true shift;
* ``kept_share`` -- the share of TRUE rows whose candidate list still contains the truth
  partner after the restriction (a filter that drops the truth cannot be rescued later).

    python Topology/placement_m18_shift.py --workers 2 --cap 6
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot15", {"rot_deg": 15.0}),
)
DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")


def _job(args: Tuple[str, str, Dict[str, Any], int]) -> Dict[str, Any]:
    tag, pose, spec, cap = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_context as CX
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
    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    triples = PM.top_candidates(matrix, nodes_b, max(PM.cand_cap(), 1))
    full = {int(u): [(int(vid), float(s)) for (_j, vid, s) in triples[i]]
            for i, u in enumerate(nodes_a) if triples[i]}
    pop = sorted(full)

    st_all: Dict[str, Any] = {}
    modes = CX.consensus_shifts(pop, full, sq, sr, stats=st_all)
    # the truth's own shift: one vote per true pair, weight 1
    true_rows = {u: [(v, 1.0)] for u, v in truth.items() if u in full}
    st_true: Dict[str, Any] = {}
    tmodes = CX.consensus_shifts([u for u in true_rows], true_rows, sq, sr, stats=st_true)

    # does the restriction keep the truth partner?
    kept = 0
    rows = 0
    for u in pop:
        vt = truth.get(u)
        if vt is None:
            continue
        rows += 1
        ta, ax = CX.node_theta(sq, u)
        tb, axb = CX.node_theta(sr, vt)
        if ta is None or tb is None or not modes:
            kept += 1
            continue
        d = CX._shift(ta, tb, bool(ax or axb))
        if any(abs(d - m) <= CX.CONSENSUS_TOL for m in modes):
            kept += 1
    return {"tag": tag, "pose": pose, "cap": int(cap), "rows": rows,
            "kept_share": (kept / rows) if rows else float("nan"),
            "modes": modes, "n_votes": (st_all.get("consensus") or {}).get("votes"),
            "share": (st_all.get("consensus") or {}).get("best_weight_share"),
            "true_shift": tmodes[0] if tmodes else None,
            "true_n": (st_true.get("consensus") or {}).get("votes"),
            "hit": (bool(tmodes and modes and any(
                abs(m - tmodes[0]) <= CX.CONSENSUS_TOL for m in modes)))}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--out", default="m18_shift.json")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    jobs = [(t, pose, spec, args.cap) for (pose, spec) in POSES for t in DEV]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-6s cap %2d  rows %4d kept %.3f  share %.3f  mode %s  "
                  "true %s  hit %s"
                  % (r["tag"], r["pose"], r["cap"], r["rows"], r["kept_share"],
                     r["share"] or float("nan"),
                     ["%.2f" % m for m in r["modes"]],
                     "%.2f" % r["true_shift"] if r["true_shift"] is not None else "None",
                     r["hit"]), flush=True)
    MF.write_result(os.path.join(OUT, args.out),
                    {"cells": cells, "cap": args.cap,
                     "units": "shifts in radians, wrapped to [-pi, pi); tol 0.175 rad",
                     "note": "true_shift uses the truth and is DIAGNOSTIC ONLY"})
    print("\nwrote %s" % os.path.join(OUT, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
