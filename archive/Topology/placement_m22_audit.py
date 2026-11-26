"""MISSION 22 / A.1 -- READ-ONLY: what a place actually carries.

The voting front door stands or falls on one question: does a place node carry a FRAME
(position, scale, orientation) and the CONTENT the M15 layer hangs on it?  This file answers it
by reading, before any index is built, and it is deliberately cheap: a handful of images, no
matcher, no scoring.

What it reports, per image:
  * places                 -- nodes that carry a placement payload
  * position               -- the anchor the place was read at (x, y)
  * scale                  -- the nearest-neighbour distance among places (`spacing_by_node`),
                              which is what a "how big is this place" transform needs
  * orientation            -- the local frame from `placement_edges.node_frames_keypoint`
                              (axial, radians) and how many places get one
  * content                -- the node's signature: its lens, the graded population vector the
                              similarity reads, and the QUANTISED bins that can serve as a
                              discrete content key
  * payload                -- the incident edges the M15 layer attached, by lens and class
  * the BUCKET SIZES a content key would produce over the whole FIT gallery: with 61k places
    pooled, a key that is too fine gives buckets of 1 (no votes) and one that is too coarse
    gives buckets of 10k (no speed).  This is the measurement that decides the key.

Usage:
    python Topology/placement_m22_audit.py --split FIT --images 6 --pool 40
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def audit_image(path: str) -> Dict[str, Any]:
    """Everything one image's place layer carries.  Read-only."""
    import guard
    guard.activate()
    from Topology.placement_m20_eth80 import _cell
    from matching import placement_matcher as PM
    from matching import placement_edges as PE
    img, g, side = _cell(path, "m22audit")
    sub = g.subgraph
    nodes = sorted(int(n) for n in sub.nodes)
    pay_nodes = sorted({int(k[0]) for k in side.payload}
                       | {int(k[1]) for k in side.payload})
    # the frame: axial orientation per node from the keypoint structure tensor
    frames, sigma_of = PE.node_frames_keypoint(sub, img)
    with_pos = [n for n in nodes if n in sub.anchor_of_node]
    with_scale = [n for n in pay_nodes if n in side.spacing_by_node]
    with_frame = [n for n in pay_nodes if n in frames]
    lenses: Dict[str, int] = {}
    for n in pay_nodes:
        lenses[side.lens.get(n, "?")] = lenses.get(side.lens.get(n, "?"), 0) + 1
    edge_classes: Dict[str, int] = {}
    for (s, d, et) in side.payload:
        nm = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        edge_classes[nm] = edge_classes.get(nm, 0) + 1
    # the signature: what a content key can be built from
    sig_sample = None
    for n in pay_nodes:
        s = g.sigs.get(n)
        if s is not None:
            sig = s[1]
            sig_sample = {"lens": s[0], "kind": sig.kind,
                          "n_quantized": len(getattr(sig, "quantized", ()) or ()),
                          "n_population_axes": len(getattr(sig, "population", ()) or ()),
                          "population_len": [len(x) for x in
                                             (getattr(sig, "population", ()) or ())][:4]}
            break
    return {"path": path, "nodes": len(nodes), "places_payload": len(pay_nodes),
            "places_with_position": len(with_pos), "places_with_scale": len(with_scale),
            "places_with_frame": len(with_frame), "lenses": lenses,
            "edge_classes": edge_classes, "signature_sample": sig_sample,
            "spacing_median": side.spacing_median,
            "scale_spread": ([float(np.percentile(
                [side.spacing_by_node[n] for n in pay_nodes if n in side.spacing_by_node], 5)),
                float(np.percentile(
                    [side.spacing_by_node[n] for n in pay_nodes if n in side.spacing_by_node],
                    95))] if with_scale else None),
            "frame_spread": ([float(np.percentile([frames[n] for n in with_frame], 5)),
                              float(np.percentile([frames[n] for n in with_frame], 95))]
                             if with_frame else None)}


def content_keys(path: str) -> Dict[str, Any]:
    """The candidate discrete content keys, and the bucket-size distribution each yields.

    A key is only useful if it is DISCRETE (so a pool can be indexed) and not too fine (so a
    bucket still contains something to vote with).  Three candidates, cheapest first:

      lens            the place's lens name alone -- as coarse as it gets
      quant           (lens, the signature's quantised bins) -- the bins are the coarse INDEX
                      the signature was designed around
      quant+config    (lens, quantised bins, the first four CONFIGURATION slots quantised),
                      i.e. content AND configuration together
    """
    import guard
    guard.activate()
    from Topology.placement_m20_eth80 import _cell
    from matching import placement_prefilter as PF
    img, g, side = _cell(path, "m22keys")
    params = PF.load_params()
    raw = PF.bag_v2(side, sorted(int(n) for n in side.payload and
                                 {int(k[0]) for k in side.payload} |
                                 {int(k[1]) for k in side.payload}), params["K"])
    std = PF.standardize(raw, params)
    out: Dict[str, List[Any]] = {"lens": [], "quant": [], "quant+config": []}
    for r, n in enumerate(raw["ids"]):
        s = g.sigs.get(n)
        if s is None:
            continue
        lens, sig = s
        q = tuple(float(x) for x in (getattr(sig, "quantized", ()) or ()))
        out["lens"].append((lens,))
        out["quant"].append((lens, q))
        d4 = tuple(int(round(float(std["D"][r, i]))) for i in range(min(4, params["K"]))
                   if std["M"][r, i] > 0)
        out["quant+config"].append((lens, q, d4))
    return {k: v for k, v in out.items()}, len(out["lens"])


def bucket_stats(keys: Dict[str, List[Any]]) -> Dict[str, Any]:
    res = {}
    for name, ks in keys.items():
        from collections import Counter
        c = Counter(ks)
        sizes = np.asarray(sorted(c.values()))
        res[name] = {"places": int(len(ks)), "buckets": int(len(c)),
                     "mean": float(sizes.mean()), "median": float(np.median(sizes)),
                     "p95": float(np.percentile(sizes, 95)), "max": int(sizes.max()),
                     "singletons": int((sizes == 1).sum())}
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--pool", type=int, default=40, help="objects pooled for the bucket stats")
    ap.add_argument("--out", default="m22_audit")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    from Topology.placement_m20_eth80 import objects
    objs = objects(a.split)
    res: Dict[str, Any] = {"split": a.split, "images": [], "note": "read-only"}
    t0 = time.time()
    for o in objs[:a.images]:
        res["images"].append(audit_image(o["ref"]))
    print("audited %d images in %.1fs" % (len(res["images"]), time.time() - t0))
    for r in res["images"]:
        print("  %-28s places %-5d pos %-5d scale %-5d frame %-5d lenses %d edges %d"
              % (os.path.basename(r["path"]), r["places_payload"], r["places_with_position"],
                 r["places_with_scale"], r["places_with_frame"], len(r["lenses"]),
                 sum(r["edge_classes"].values())))
    # the pooled bucket statistics over the whole gallery
    pooled: Dict[str, List[Any]] = {"lens": [], "quant": [], "quant+config": []}
    t0 = time.time()
    allkeys: Dict[str, List[Any]] = {}
    for o in objs[:a.pool]:
        ks, n = content_keys(o["ref"])
        for k in pooled:
            pooled[k].extend(ks[k])
    print("pooled %d objects (%d places) in %.1fs"
          % (min(a.pool, len(objs)), len(pooled["lens"]), time.time() - t0))
    res["buckets"] = bucket_stats(pooled)
    for name, b in res["buckets"].items():
        print("  key %-13s places %-6d buckets %-6d mean %-7.1f median %-6.1f p95 %-6.1f "
              "max %-6d singletons %d"
              % (name, b["places"], b["buckets"], b["mean"], b["median"], b["p95"],
                 b["max"], b["singletons"]))
    res["elapsed"] = time.time() - t0
    path = os.path.join(OUT, a.out + ".json")
    from matching import m15_manifest as MF
    MF.write_result(path, res)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
