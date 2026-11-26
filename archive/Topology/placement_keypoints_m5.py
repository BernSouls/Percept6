"""MISSION 5 STAGE 3 -- precision of the true-keypoint layer, measured on FIT.

The detector in `lenses/keypoints.py` produces corners, blobs and boundary-arc
points with sub-pixel positions, orientation, coherence, scale and a per-axis
position variance.  This instrument asks the same question Stage 1 asked of the
front end's own nodes, so the two tables are directly comparable:

    carry a detection through its object's known pose (renderer truth, used
    only to score) and look for the nearest detection OF THE SAME KIND.

Reported per kind and overall: repeatability, localisation (median, p90),
orientation error on coherent detections, distinctness at the NMS separation,
and coverage (share of detections whose pixel sits on an object).

The gate's numbers are: localisation median <= 0.50 px, p90 <= 1.50 px,
repeatability >= 0.70, distinctness >= 0.95, direction median <= 3 deg.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
for _p in (ROOT, os.path.join(ROOT, "Topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")
RADIUS = 3.0
TIGHT = 1.5


def _stub(points: List[Any]) -> Any:
    """An ImageGraph-shaped object carrying keypoints, for `predict`."""
    pts = sorted(points, key=lambda p: (p.kind, p.x, p.y, p.scale))
    nodes = frozenset(range(len(pts)))
    anchors = {i: (p.x, p.y) for i, p in enumerate(pts)}
    sigs = {i: (p.lens, None) for i, p in enumerate(pts)}
    sub = SimpleNamespace(nodes=nodes, anchor_of_node=anchors)
    return SimpleNamespace(subgraph=sub, sigs=sigs), pts


def _wrap(a: float) -> float:
    """Axial wrap to (-pi/2, pi/2]: a direction has no sign."""
    a = float(a) % math.pi
    return a - math.pi if a > math.pi / 2 else a


def run(specs: List[Any], views: List[Any], tag: str) -> Dict[str, Any]:
    import guard
    guard.activate()
    from lenses.keypoints import MIN_SEPARATION_PX, detect
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import predict

    acc: Dict[str, Dict[str, List[float]]] = {}

    def bucket(k: str) -> Dict[str, List[float]]:
        return acc.setdefault(k, {"n": [], "repeat": [], "loc": [],
                                  "dir": [], "distinct": [], "cover": []})

    for spec in specs:
        r0 = spec.scene.render()
        k0 = detect(r0.image_u8)
        for label, view in views:
            pv_ = view_pose(spec, view)
            sc1 = posed(spec.scene, pv_["rot_deg"], tuple(pv_["shift"]),
                        pv_["scale"])
            r1 = sc1.render()
            k1 = detect(r1.image_u8)
            stub, pts = _stub(k0)
            pred = predict(spec.scene, r0, sc1, stub)
            rot = math.radians(float(pv_["rot_deg"]))

            by_kind: Dict[str, List[int]] = {}
            for i, p in enumerate(sorted(k1, key=lambda q: (q.kind, q.x, q.y,
                                                            q.scale))):
                by_kind.setdefault(p.kind, []).append(i)
            k1s = sorted(k1, key=lambda q: (q.kind, q.x, q.y, q.scale))

            # distinctness: detections with no neighbour inside the NMS radius
            for p in k1s:
                near = sum(1 for q in k1s
                           if q is not p
                           and abs(q.x - p.x) <= MIN_SEPARATION_PX
                           and abs(q.y - p.y) <= MIN_SEPARATION_PX)
                for b in ("all", p.kind):
                    bucket(b)["distinct"].append(1.0 if near == 0 else 0.0)

            for i, p in enumerate(pts):
                for b in ("all", p.kind):
                    bucket(b)["n"].append(1.0)
                pr = pred.get(i)
                if pr is None:
                    for b in ("all", p.kind):
                        bucket(b)["repeat"].append(0.0)
                    continue
                pool = by_kind.get(p.kind, [])
                if not pool:
                    for b in ("all", p.kind):
                        bucket(b)["repeat"].append(0.0)
                    continue
                px, py = pr
                d = [math.hypot(k1s[j].x - px, k1s[j].y - py) for j in pool]
                m = int(np.argmin(d))
                hit = d[m] <= RADIUS
                for b in ("all", p.kind):
                    bucket(b)["repeat"].append(1.0 if hit else 0.0)
                    bucket(b)["cover"].append(1.0)
                if hit:
                    q = k1s[pool[m]]
                    for b in ("all", p.kind):
                        bucket(b)["loc"].append(float(d[m]))
                    if p.coherence >= 0.5 and q.coherence >= 0.5:
                        err = abs(math.degrees(_wrap(q.theta - p.theta - rot)))
                        for b in ("all", p.kind):
                            bucket(b)["dir"].append(float(min(err, 90.0)))

    res: Dict[str, Any] = {"radius_px": RADIUS, "scenes": len(specs),
                           "kinds": {}}
    hdr = ("%-10s %6s %8s %10s %8s %9s %10s %9s"
           % ("kind", "n", "repeat", "loc med", "loc p90", "dir med",
              "distinct", "cov@1.5"))
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(acc, key=lambda x: (x != "all", x)):
        d = acc[k]
        if not d["n"]:
            continue
        row = {
            "n": int(len(d["n"])),
            "repeat": float(np.mean(d["repeat"])),
            "loc_median": float(np.median(d["loc"])) if d["loc"] else float("nan"),
            "loc_p90": float(np.percentile(d["loc"], 90)) if d["loc"] else float("nan"),
            "loc_p90_tight": (float(np.mean([1.0 if v <= TIGHT else 0.0
                                             for v in d["loc"]]))
                              if d["loc"] else float("nan")),
            "dir_median": float(np.median(d["dir"])) if d["dir"] else float("nan"),
            "distinct": float(np.mean(d["distinct"])) if d["distinct"] else float("nan"),
        }
        res["kinds"][k] = row
        print("%-10s %6d %8.3f %10.3f %8.3f %9.2f %10.3f %9.3f"
              % (k, row["n"], row["repeat"], row["loc_median"],
                 row["loc_p90"], row["dir_median"], row["distinct"],
                 row["loc_p90_tight"]))

    path = os.path.join(OUT, "%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="m5_stage3_keypoints")
    args = ap.parse_args()
    from Topology.placement_dev3 import VIEWS, all_specs, split
    fit, _ = split(all_specs())
    specs = fit[:args.limit]
    print("STAGE 3 keypoint precision -- FIT, %d scenes\n" % len(specs))
    run(specs, VIEWS, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
