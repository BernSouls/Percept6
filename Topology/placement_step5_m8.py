"""MISSION 8 STEP 5 -- real images (ETH-80).  Non-gating, but required.

Synthetic DEV-5 scenes are one click away from the renderer that made them.
This step asks the questions only a real photograph can answer:

  * does the front end still build at all, on real images, with the flags on,
  * do the node counts and the place layer look like the synthetic ones (or did
    something collapse),
  * what does a graph cost in wall time,
  * does the placement population still ATTACH -- how many object nodes have a
    place within ATTACH_RADIUS px,
  * and as a sanity only: does the best variant retrieve the right object
    instance at least as often as N.

No tuning happens here.  Nothing here may change a frozen constant.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
ATTACH_RADIUS = 4.0
IMG_SIZE = 256


def load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--tag", default="m8_real")
    args = ap.parse_args()

    import guard
    guard.activate()
    from Topology import corpus
    from Topology.placement_cache import get_or_build
    from matching import placement_edges as PE
    from matching.placement_scorer import incident_index, payload_index
    from matching import placement_llr as L
    from matching import placement_context as Ctx
    from Topology.placement_census import lens_of
    from Topology import placement_arms2 as A2
    from core.place_key import place_lenses

    spl = corpus.split()
    refs = [e for e in spl["train"] if e["view"] == "-090-000"]
    queries = [e for e in spl["test"] if e["view"] == "-090-000"]
    by_cls: Dict[str, List[dict]] = defaultdict(list)
    for e in refs:
        by_cls[e["class"]].append(e)
    q_by_cls: Dict[str, List[dict]] = defaultdict(list)
    for e in queries:
        q_by_cls[e["class"]].append(e)

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    place_set = set(place_lenses())

    counts = {"off": [], "on": []}
    times = {"off": [], "on": []}
    attach: List[float] = []
    dists: List[float] = []
    per_lens_occ: Dict[str, List[int]] = defaultdict(list)
    n_place_nodes: List[int] = []
    build_cache: Dict[str, Any] = {}

    imgs: List[dict] = []
    for cls in corpus.CLASSES:
        imgs.extend(q_by_cls[cls][:args.per_class])
        imgs.extend(by_cls[cls][:args.per_class])

    print("ETH-80 real images: %d\n" % len(imgs))
    import eth80_frontend as F
    for e in imgs:
        u8 = load_u8(e["path"])
        key = corpus.name_of(e)
        # flags ON (the frozen configuration)
        t0 = time.time()
        g = get_or_build(u8, "real|on|" + key)
        dt = time.time() - t0
        build_cache[key] = g
        times["on"].append(dt)
        counts["on"].append((len(g.subgraph.nodes), len(g.subgraph.edges)))
        kp = 0
        for n in g.subgraph.nodes:
            Lm = str(lens_of(g, n))
            per_lens_occ[Lm].append(1)
            if Lm in place_set:
                kp += 1
        n_place_nodes.append(kp)
        # attach: object nodes without a place must find one within R px
        kps = np.array([(float(g.subgraph.anchor_of_node[n][0]),
                         float(g.subgraph.anchor_of_node[n][1]))
                        for n in g.subgraph.nodes
                        if n in g.subgraph.anchor_of_node
                        and str(lens_of(g, n)) in place_set], dtype=float)
        if len(kps):
            for n in g.subgraph.nodes:
                a = g.subgraph.anchor_of_node.get(n)
                if a is None or str(lens_of(g, n)) in place_set:
                    continue
                d = float(np.min(np.hypot(kps[:, 0] - float(a[0]),
                                          kps[:, 1] - float(a[1]))))
                attach.append(1.0 if d <= ATTACH_RADIUS else 0.0)
                dists.append(d)

    print("%-22s %8s %8s %10s %8s" % ("image", "nodes", "edges",
                                      "place nodes", "build s"))
    for e, c, t, k in zip(imgs, counts["on"], times["on"], n_place_nodes):
        print("  %-20s %8d %8d %10d %8.1f"
              % (corpus.name_of(e), c[0], c[1], k, t))

    res: Dict[str, Any] = {
        "images": len(imgs),
        "nodes": {"mean": float(np.mean([c[0] for c in counts["on"]])),
                  "min": int(min(c[0] for c in counts["on"])),
                  "max": int(max(c[0] for c in counts["on"]))},
        "edges": {"mean": float(np.mean([c[1] for c in counts["on"]]))},
        "place_nodes": {"mean": float(np.mean(n_place_nodes))},
        "build_seconds": {"mean": float(np.mean(times["on"])),
                          "max": float(np.max(times["on"]))},
        "attach_coverage": {
            "radius_px": ATTACH_RADIUS,
            "n": len(attach),
            "coverage": float(np.mean(attach)) if attach else float("nan"),
            "median_dist": float(np.median(dists)) if dists else float("nan")},
        "per_lens": {k: len(v) for k, v in sorted(per_lens_occ.items())},
        "collapsed": [k for k, v in per_lens_occ.items()
                      if len(v) < max(5, 0.02 * sum(len(x) for x in
                                                    per_lens_occ.values()))],
    }
    print("\nnodes %.1f  edges %.1f  place nodes %.1f  build %.1f s"
          % (res["nodes"]["mean"], res["edges"]["mean"],
             res["place_nodes"]["mean"], res["build_seconds"]["mean"]))
    print("attach coverage %.3f (n %d)  median %.2f px"
          % (res["attach_coverage"]["coverage"], res["attach_coverage"]["n"],
             res["attach_coverage"]["median_dist"]))
    print("lenses present: %d   collapsed: %s"
          % (len(res["per_lens"]), res["collapsed"] or "none"))

    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
