"""Mission 13 step 3 -- fix the DOMINANT term: anchor repeatability.

Step 2 names detection as the binding term (13 % of nodes re-detected within 1 px
under 5 deg, 10 % under 1.15x, against appearance-only accuracy 0.33/0.18 on the
pairs that do survive).  The MECHANISM is positional quantisation: the front end
places node anchors on an image-aligned lattice (8 px and integer in the shipped
graph: 39 % of synthetic anchors sit exactly on multiples of 8, 84 % on integers),
so a rotation or a scale move every physical point OFF the lattice and the nearest
re-detected node is a lattice step away -- median 3.6 px, which is exactly the
"detection collapse".

The lever the tree already has is SUB-PIXEL ANCHORING
(`FE_SUBPIXEL_PEAK_POS`, `FE_SUBPIXEL_GABOR_ANCHOR`): with them on, the share of
anchors on the 8-px lattice falls 0.39 -> 0.01 and the non-integer share rises
0.14 -> 0.45.  This harness measures what that buys:

* DETECTION curve (tolerance 0.5/1/2/3 px) under 5 deg, 15 deg, 1.15x;
* ARM A geometric correspondence: run the shipped aligner (placement OFF) on the
  transformed/reference pair and ask, for every reference node that is mapped,
  whether the partner's anchor lands within TOL px of the transformed anchor.
  This criterion is IDENTICAL for both flag states (it never uses the truth
  pairing the flags would change), so the comparison is fair;
* the same with placement ON (arm C) for context.

    python Topology/placement_m13_repeat.py --workers 2
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
RESULT = os.path.join(OUT, "m13_repeat.json")
POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot05", {"rot_deg": 5.0}), ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}))
TOLS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


def _job(args: Tuple[str, str, Dict[str, Any], bool]) -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from scipy.spatial import cKDTree
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology import placement_m13_decompose as D
    from Topology import placement_m12_read_real as R

    tag, label, spec, subpix = args
    P.FE_SUBPIXEL_PEAK_POS = bool(subpix)
    P.FE_SUBPIXEL_GABOR_ANCHOR = bool(subpix)
    imgs = D._images()
    img0 = imgs[tag]
    img1 = R._transform(img0, spec)
    g0 = get_or_build(img0, "m13rp|%s|orig|%d" % (tag, subpix))
    g1 = get_or_build(img1, "m13rp|%s|%s|%d" % (tag, label, subpix))
    sub0, sub1 = g0.subgraph, g1.subgraph
    n0 = [int(n) for n in sorted(sub0.nodes) if n in sub0.anchor_of_node]
    n1 = [int(n) for n in sorted(sub1.nodes) if n in sub1.anchor_of_node]
    a0 = np.asarray([sub0.anchor_of_node[n] for n in n0], float)
    a1 = np.asarray([sub1.anchor_of_node[n] for n in n1], float)
    h, w = img0.shape[:2]
    fwd = R._forward(spec, a0, (h, w))
    inside = np.asarray([(R.BORDER <= x < w - R.BORDER and R.BORDER <= y < h - R.BORDER)
                         for x, y in fwd], bool)
    tree = cKDTree(a1) if len(a1) else None
    det = {}
    if tree is not None and inside.any():
        d, _i = tree.query(fwd[inside], k=1)
        for tol in TOLS:
            det["%.1f" % tol] = float(np.mean(d <= tol))
    # ---- arm A (placement off): geometric correspondence -------------------
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    # DIRECTION: reference -> query, so `mapping[u]` is a QUERY node for every
    # reference node u and its anchor can be compared with the transformed anchor.
    res = joint_mcs(g0.subgraph, g1.subgraph, g0.sigs, g1.sigs)
    pos1 = {int(k): np.asarray(v, float) for k, v in sub1.anchor_of_node.items()}
    idx0 = {int(n): i for i, n in enumerate(n0)}
    geo: Dict[str, float] = {}
    for tol in (1.0, 1.5, 2.0):
        hit = tot = 0
        for u in n0:
            i = idx0[u]
            if not inside[i]:
                continue
            tot += 1
            v = res.mapping.get(int(u))
            if v is None or int(v) not in pos1:
                continue
            if float(np.hypot(*(pos1[int(v)] - fwd[i]))) <= tol:
                hit += 1
        geo["%.1f" % tol] = (hit / tot) if tot else float("nan")
    # accuracy conditional on being mapped (the "does it point at the right
    # physical place" reading, separated from coverage)
    cond: Dict[str, float] = {}
    for tol in (1.0, 1.5, 2.0):
        hit = tot = 0
        for u in n0:
            i = idx0[u]
            if not inside[i]:
                continue
            v = res.mapping.get(int(u))
            if v is None or int(v) not in pos1:
                continue
            tot += 1
            if float(np.hypot(*(pos1[int(v)] - fwd[i]))) <= tol:
                hit += 1
        cond["%.1f" % tol] = (hit / tot) if tot else float("nan")
    return {"tag": tag, "pose": label, "subpix": bool(subpix),
            "nodes0": len(n0), "nodes1": len(n1), "inside": int(inside.sum()),
            "mapped": int(len(res.mapping)),
            "detect": det, "arm_a_geo": geo, "arm_a_geo_if_mapped": cond}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    from Topology import placement_m13_decompose as D
    imgs = D._images()
    jobs = [(tag, label, spec, sp) for sp in (False, True)
            for tag in sorted(imgs) for label, spec in POSES]
    print("%d cells (%d images x %d poses x 2 flag states)"
          % (len(jobs), len(imgs), len(POSES)), flush=True)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            print("  %-14s %-9s sp=%-5s det1 %.3f  geo1.5 %.3f  cond1.5 %.3f  mapped %d/%d"
                  % (r["tag"], r["pose"], r["subpix"], r["detect"].get("1.0", float("nan")),
                     r["arm_a_geo"]["1.5"], r["arm_a_geo_if_mapped"]["1.5"],
                     r["mapped"], r["inside"]), flush=True)
    print("cells done in %.0fs" % (time.time() - t0), flush=True)
    agg: Dict[str, Any] = {}
    for sp in (False, True):
        for pose, _sp in POSES:
            for kind in ("syn", "real", "all"):
                sub = [r for r in rows if r["subpix"] == sp and r["pose"] == pose
                       and (kind == "all" or r["tag"].startswith(kind))]
                if not sub:
                    continue
                agg["%s|%s|subpix=%d" % (pose, kind, int(sp))] = {
                    "cells": len(sub),
                    **{"detect_%.1f" % t: float(np.nanmean([r["detect"].get("%.1f" % t, float("nan")) for r in sub])) for t in TOLS},
                    **{"geo_%.1f" % t: float(np.nanmean([r["arm_a_geo"]["%.1f" % t] for r in sub])) for t in (1.0, 1.5, 2.0)},
                    **{"cond_%.1f" % t: float(np.nanmean([r["arm_a_geo_if_mapped"]["%.1f" % t] for r in sub])) for t in (1.0, 1.5, 2.0)},
                }
    res = {"rows": rows, "aggregate": agg,
           "note": ("arm A = shipped aligner, placement OFF. geo_T = share of the "
                    "reference nodes inside the image whose mapped partner lands within "
                    "T px of the transformed anchor (coverage AND precision); cond_T = "
                    "the same among the nodes that were mapped at all (precision only)")}
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "cells"}
                      for k, v in agg.items()}, indent=1)[:2600])
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
