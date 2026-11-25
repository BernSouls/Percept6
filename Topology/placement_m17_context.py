"""MISSION 17 / O2 -- WHY the context is wrong, measured without guessing.

The O2 sweep (iters 0/2/5/8, margin, rival count, rival tolerance, mutual) moved the
geometry AUC at 15 deg by less than 0.005 and the top-1 by less than one standard
error.  So the question is not which damping constant to use, it is whether the mapping
the iteration starts from carries any information at all.

This measures CONTEXT ACCURACY -- the share of query places whose mapping is the truth
partner -- for three mechanisms, using truth ONLY to score the mapping, never to build
it:

    sim        argmax of the appearance similarity in the row      (what runs today)
    +config    argmax of appearance PLUS a ROTATION-INVARIANT configuration
               descriptor: the multiset of (log distance, direction-in-source-frame)
               to a place's nearest neighbours.  PlacementVector.direction is measured
               in the SOURCE node's frame (matching/placement.py:85-104), so the
               descriptor is invariant to a rotation of the image and to a change of
               scale -- exactly the two transforms the mission is about -- and it needs
               no correspondence, which is what makes it usable where the context is
               not yet known.
    assign     the same score, but as a ONE-TO-ONE assignment (scipy
               linear_sum_assignment) instead of a per-node argmax.

    python Topology/placement_m17_context.py --poses rot15 --images dev
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m17_context.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")
K = 6


def descriptor(side: Any, u: int, k: int = K) -> Optional[Tuple[List[float], List[float]]]:
    """(sorted log distances, sorted directions) to ``u``'s nearest neighbours."""
    from matching.placement import wrap_pi
    rows: List[Tuple[float, float]] = []
    for (other, is_source, name) in side.incident.get(int(u), ()):
        key = (int(u), int(other), name) if is_source else (int(other), int(u), name)
        pv = side.index.get(key)
        if pv is None:
            continue
        d = float(getattr(pv, "distance", float("nan")))
        if not math.isfinite(d) or d <= 0.0:
            continue
        th = float(getattr(pv, "direction", 0.0))
        rows.append((math.log(d), th if getattr(pv, "axial", False) is False
                     else _axial(th)))
    if len(rows) < 2:
        return None
    rows.sort()
    return ([r[0] for r in rows][:k], [r[1] for r in rows][:k])


def _axial(th: float) -> float:
    """Fold a direction into (-pi/2, pi/2]; the axial angle is mod pi."""
    x = (th + math.pi / 2.0) % math.pi - math.pi / 2.0
    return x


def config_similarity(a, b, sigma_d: float = 0.35, sigma_a: float = 0.5) -> float:
    """Rank-aligned agreement of two descriptors, in [0, 1]."""
    if a is None or b is None:
        return 0.0
    da, aa = a
    db, bb = b
    n = min(len(da), len(db))
    if n < 2:
        return 0.0
    sd = 0.0
    for i in range(n):
        sd += (da[i] - db[i]) ** 2
    sa = 0.0
    for i in range(n):
        d = (aa[i] - bb[i] + math.pi) % (2.0 * math.pi) - math.pi
        d = min(abs(d), abs(abs(d) - math.pi))       # axial: mod pi
        sa += d * d
    return float(math.exp(-0.5 * (sd / (n * sigma_d ** 2)
                                  + sa / (n * sigma_a ** 2))))


def _job(args: Tuple[str, str, Dict[str, Any], float]) -> Dict[str, Any]:
    tag, pose, spec, lam = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build
    from Topology.placement_m17_eval import _image, M16
    from core import keypoint_classes as KPC
    from scipy.optimize import linear_sum_assignment

    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
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

    desc_q = {u: descriptor(sq, u) for u in nodes_a}
    desc_r = {v: descriptor(sr, v) for v in nodes_b}
    cand: Dict[int, List[Tuple[int, float, float]]] = {}
    for u in nodes_a:
        row = [(v, float(s)) for (v, s) in rows_sim.get(u, ()) if s > 0.0]
        if not row:
            continue
        cand[u] = [(v, s, config_similarity(desc_q.get(u), desc_r.get(v)))
                   for (v, s) in row]
    qs = [u for u in nodes_a if u in truth and u in cand]
    base = {u: max(cand[u], key=lambda t: t[1])[0] for u in qs}
    pluscfg = {u: max(cand[u], key=lambda t: (1 - lam) * t[1] + lam * t[2])[0]
               for u in qs}
    # one-to-one assignment on the same score
    cols = sorted({v for u in qs for (v, _s, _c) in cand[u]})
    cmap = {v: j for j, v in enumerate(cols)}
    big = np.full((len(qs), len(cols)), -1e9, dtype=float)
    for i, u in enumerate(qs):
        for (v, s, c) in cand[u]:
            big[i, cmap[v]] = (1 - lam) * s + lam * c
    ri, ci = linear_sum_assignment(-big)
    assigned = {qs[i]: cols[j] for i, j in zip(ri, ci) if big[i, j] > -1e8}

    def acc(m: Dict[int, int]) -> float:
        return sum(1 for u in qs if m.get(u) == truth[u]) / float(len(qs) or 1)

    in_row = sum(1 for u in qs if any(v == truth[u] for (v, _s, _c) in cand[u]))
    return {"tag": tag, "pose": pose, "lam": lam, "queries": len(qs),
            "truth_in_row": in_row,
            "acc_sim": acc(base), "acc_cfg": acc(pluscfg), "acc_assign": acc(assigned),
            "agreement_cfg_vs_sim": (sum(1 for u in qs if pluscfg[u] == base[u])
                                     / float(len(qs) or 1))}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="rot00,rot05,rot15,scale115")
    ap.add_argument("--images", default="dev")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    from Topology.placement_m17_eval import IMAGE_SETS
    tags = IMAGE_SETS.get(args.images, tuple(args.images.split(",")))
    poses = [p for p in POSES if p[0] in args.poses.split(",")]
    jobs = [(t, pose, spec, args.lam) for (pose, spec) in poses for t in tags]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-8s q %4d in-row %4d  acc sim %.3f cfg %.3f assign %.3f"
                  % (r["tag"], r["pose"], r["queries"], r["truth_in_row"],
                     r["acc_sim"], r["acc_cfg"], r["acc_assign"]), flush=True)
    res = {"cells": cells, "lam": args.lam, "k": K,
           "units": "accuracy is a share of queries; distances are in source-scale "
                    "units (scale-free), directions in radians in the source frame"}
    MF.write_result(RESULT, res)
    import collections
    for pose in args.poses.split(","):
        rr = [c for c in cells if c["pose"] == pose]
        if not rr:
            continue
        q = sum(c["queries"] for c in rr)
        print("\n%s n=%d q=%d  sim %.3f  +config %.3f  assign %.3f"
              % (pose, len(rr), q,
                 sum(c["acc_sim"] * c["queries"] for c in rr) / q,
                 sum(c["acc_cfg"] * c["queries"] for c in rr) / q,
                 sum(c["acc_assign"] * c["queries"] for c in rr) / q))
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
