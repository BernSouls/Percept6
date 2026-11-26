"""Mission 15 item 9 -- does the alignment refinement budget BIND?

The budget (`MATCH_ALIGN_MAX_PAIR_EVALS`) is a constant, while the work the refinement
can do grows with the number of admissible (positive-similarity) cells in the pair.  If
the budget binds, what the refinement settles on depends on the budget rather than on
the evidence, and the fix is to scale it with those cells
(`MATCH_ALIGN_BUDGET_SCALES_WITH_CELLS`, factor `MATCH_ALIGN_BUDGET_CELL_FACTOR`).

The measurement, per pair, is the alignment's own `budget_exhausted` flag, with the
number of positive cells beside it -- counted, not modelled.

    python Topology/placement_m15_budget.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m15_budget.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("shift8", {"shift": (8.0, 0.0)}),
    ("rot15", {"rot_deg": 15.0}),
)


def _job(args: Tuple[str, str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    tag, pose, spec = args
    import guard
    guard.activate()
    import parameters as P
    from matching import alignment as A
    from matching import m15_places as PL
    from matching import placement_matcher as PM
    from Topology import placement_census as CEN
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    from Topology.placement_cache import get_or_build

    for k, v in dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True,
                     FE_SUBPIXEL_PEAK_POS=True, FE_SUBPIXEL_GABOR_ANCHOR=True,
                     PLACEMENT_FRAME_KEYPOINT=True,
                     PLACEMENT_FRAME_KEYPOINT_SIGMA="largest",
                     PLACEMENT_FRAME_FALLBACK="none",
                     PLACEMENT_M15_PLACES=True).items():
        setattr(P, k, v)
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        img = [s for s in F.gallery() if s.name == name][0].scene.render().image_u8
    else:
        img = RP.images("validation")[tag.split("|", 1)[1]]
    from Topology import placement_m12_read_real as R
    img1 = R._transform(img, spec)
    g1 = get_or_build(img1, "m15geo|%s|%s|places" % (tag, pose))
    g0 = get_or_build(img, "m15geo|%s|orig|places" % tag)
    keep = set(PL.place_lenses())
    # PLACE NODES ONLY: the seed's Hungarian is O(n^3) and a full 2,600-node
    # photograph is not a matcher input in the first place.
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes
                     if str(g1.sigs.get(int(n), ("?",))[0]) in keep)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes
                     if str(g0.sigs.get(int(n), ("?",))[0]) in keep)
    rows = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    pos_cells = 0
    for i, u in enumerate(nodes_a):
        for (v, s) in rows.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
                pos_cells += 1
    out: List[Dict[str, Any]] = []
    for label in ("shipped", "scaled"):
        P.MATCH_ALIGN_BUDGET_SCALES_WITH_CELLS = (label == "scaled")
        al = A.global_consistency_alignment(
            nodes_a, nodes_b, lambda u, v: 0.0, sim=matrix,
            budget_refinement_only=True)
        out.append({"tag": tag, "pose": pose, "arm": label,
                    "positive_cells": int(pos_cells),
                    "nodes_a": len(nodes_a), "nodes_b": len(nodes_b),
                    "matched": len(al.mapping),
                    "budget_exhausted": bool(al.budget_exhausted)})
    return out


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="syn|m11ir_00,real|cell")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF

    tags = args.tags.split(",")
    jobs = [(t, pose, spec) for t in tags for pose, spec in POSES]
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=1) as ex:
        for got in ex.map(_job, jobs):
            rows.extend(got)
            for r in got:
                print("  %-16s %-7s %-8s cells %6d matched %4d exhausted %s"
                      % (r["tag"], r["pose"], r["arm"], r["positive_cells"],
                         r["matched"], r["budget_exhausted"]), flush=True)
    rep = MF.budget_report([(r["positive_cells"], type("A", (), {
        "mapping": range(r["matched"]), "budget_exhausted": r["budget_exhausted"]})())
        for r in rows])
    res = {"rows": rows, "report": rep,
           "flag": "MATCH_ALIGN_BUDGET_SCALES_WITH_CELLS",
           "factor": float(getattr(__import__("parameters"),
                                   "MATCH_ALIGN_BUDGET_CELL_FACTOR", 0.0))}
    MF.write_result(RESULT, res)
    print("\n%d pairs, %d exhausted (%.0f%%), by cell bucket %s"
          % (rep["pairs"], rep["exhausted"], 100.0 * (rep["share"] or 0.0),
             rep["by_cell_bucket"]))
    print("wrote", RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
