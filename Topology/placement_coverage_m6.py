"""MISSION 6 STEP 3 -- attach, don't drop: coverage of the AMBIGUOUS population.

Mission 5 excluded the ridge lenses from the placement population, which
improved identity but removed those nodes from every placement edge.  The
mission's instruction is the opposite: a ridge or attribute node is not a place,
but it should ATTACH to one -- the nearest keypoint place -- and inherit that
place's support.

This measures whether that is possible at all, and on the population the
mission names: every object-anchored node of an attribute or ridge lens (the
nodes Mission 5 dropped), not just the placement nodes.

Reported per lens and overall:
    coverage@R   share of those nodes with a keypoint place within R px
    median dist  distance to the nearest keypoint place

R is stated before measuring: ATTACH_RADIUS = 4.0 px, which is the median
nearest-neighbour spacing of the lattice these nodes come from (the scale the
front end already uses for TARGET_RADIUS_SCALES = 4.0).

Truth is used only to decide which nodes sit on an object.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
for _p in (ROOT, os.path.join(ROOT, "Topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")
ATTACH_RADIUS = 4.0

#: The lenses Mission 5 dropped from the population: ridge/boundary tokens and
#: the attribute lenses.  These are the nodes that must ATTACH.
ATTACH_LENSES = ("ContourFragment", "TextureBoundary", "Texture", "Color",
                 "Polarity", "Surface", "Boundary", "BorderOwnership",
                 "Position", "RegionShape", "Orientation", "Onset")


def run(specs: List[Any], tag: str) -> Dict[str, Any]:
    import guard
    guard.activate()
    from lenses.keypoints import as_detections
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import lens_of, object_mask

    per_lens: Dict[str, List[float]] = defaultdict(list)
    dists: Dict[str, List[float]] = defaultdict(list)
    all_cov: List[float] = []
    all_dist: List[float] = []
    n_kp: List[float] = []

    for spec in specs:
        r = spec.scene.render()
        g = get_or_build(r.image_u8, spec.name + "|kpattach")
        on_obj = object_mask(g, r)
        _d, pos = as_detections(r.image_u8)
        kp = np.array([(v[0], v[1]) for v in pos.values()], dtype=float)
        n_kp.append(float(len(kp)))
        if not len(kp):
            continue
        for n in sorted(int(x) for x in g.subgraph.nodes):
            if not on_obj.get(n):
                continue
            L = lens_of(g, n)
            if L not in ATTACH_LENSES:
                continue
            a = g.subgraph.anchor_of_node.get(n)
            if a is None:
                continue
            d = float(np.min(np.hypot(kp[:, 0] - float(a[0]),
                                      kp[:, 1] - float(a[1]))))
            hit = 1.0 if d <= ATTACH_RADIUS else 0.0
            per_lens[L].append(hit)
            dists[L].append(d)
            all_cov.append(hit)
            all_dist.append(d)

    res: Dict[str, Any] = {"attach_radius_px": ATTACH_RADIUS,
                           "scenes": len(specs),
                           "mean_keypoints_per_graph": (float(np.mean(n_kp))
                                                        if n_kp else 0.0),
                           "by_lens": {}, "overall": {}}
    print("attach coverage of the ambiguous population -- radius %.1f px\n"
          % ATTACH_RADIUS)
    print("%-18s %6s %10s %12s" % ("lens", "n", "coverage", "median dist"))
    print("-" * 50)
    rows = []
    for L in sorted(per_lens):
        v = per_lens[L]
        row = {"n": len(v), "coverage": float(np.mean(v)),
               "median_dist": float(np.median(dists[L]))}
        res["by_lens"][L] = row
        rows.append((row["coverage"], L, row))
    for _c, L, row in sorted(rows):
        print("%-18s %6d %10.3f %12.2f"
              % (L, row["n"], row["coverage"], row["median_dist"]))
    res["overall"] = {
        "n": len(all_cov),
        "coverage": float(np.mean(all_cov)) if all_cov else float("nan"),
        "median_dist": float(np.median(all_dist)) if all_dist else float("nan"),
    }
    print("-" * 50)
    print("%-18s %6d %10.3f %12.2f"
          % ("ALL", res["overall"]["n"], res["overall"]["coverage"],
             res["overall"]["median_dist"]))
    print("\nkeypoints per graph: %.1f" % res["mean_keypoints_per_graph"])

    path = os.path.join(OUT, "%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s" % path)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--tag", default="m6_attach_coverage")
    args = ap.parse_args()
    from Topology.placement_dev3 import all_specs, split
    fit, _ = split(all_specs())
    specs = fit[:args.limit]
    print("STEP 3 attach coverage -- FIT, %d scenes\n" % len(specs))
    run(specs, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
