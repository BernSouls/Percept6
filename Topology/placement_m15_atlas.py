"""Mission 15 item 1 -- THE REPEATABILITY ATLAS.

Re-detection is defined TWICE, and every number says which definition it uses:

  (i)  SAME LENS, and the node's anchor is within `tol` px of the transformed
       anchor of its partner in the other view.
  (ii) SAME LENS **and** the same QUANTISED SIGNATURE (i.e. the same node key:
       the tree mints a node from `(lens, signature_hash)`, so (ii) is "this view
       minted the same node", (i) is "this view put a node of the same kind in the
       same place").

A tiling lens can pass (i) and fail (ii) -- the same place re-detected by the same
lens, but with a signature that quantises into a different bin.  A lens that fails
(i) is not a place at all.

THE PLACE-LENS BAR USES (i) at 2 px **and (ii)**, in that order: a place must be
re-detectable at its own position (i), and its IDENTITY must survive (ii) or the
matcher cannot pair it (the class gate compares signatures).  Both are reported
per lens so the choice can be checked.

Ridge lenses are reported with the displacement decomposed into ALONG-ridge and
ACROSS-ridge components: an arc-length sample on a boundary is repeatable across
the boundary and undefined along it, so a ridge lens can be a place across-ridge
and not along it.

Images: 6 synthetic IRREGULAR scenes + 3 real photographs, every transform EXACT
(the transform is applied to the pixels and to the anchors by the same code).
Poses: an exact stride-multiple shift (8 px), a non-multiple shift (5 px), 5 deg,
15 deg, 1.15x.  Two configurations: the defaults, and the M15 configuration
(keypoint layer + place layer + sub-pixel anchors on).

    python Topology/placement_m15_atlas.py --workers 2
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m15_atlas.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("shift8", {"shift": (8.0, 0.0)}),
    ("shift5", {"shift": (5.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
TOLS = (1.0, 2.0)
#: lenses whose geometry is a RIDGE (a boundary), not a point
RIDGE_HINTS = ("CONTOUR", "BOUNDARY", "RIDGE", "EDGE")


_IMG: Dict[str, np.ndarray] = {}
_ANG: Dict[str, np.ndarray] = {}


def _tags(n: int = 6) -> List[str]:
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    return ["syn|" + x.name for x in [y for y in F.gallery()
                                      if y.family == "IRREGULAR"][:n]] + \
           ["real|" + k for k in sorted(RP.images("validation"))[:3]]


def _image(tag: str) -> np.ndarray:
    """ONE image, memoised in the worker: rendering all nine per cell dominated the
    run time (a scene render plus nine photograph decodes saved nothing)."""
    if tag in _IMG:
        return _IMG[tag]
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        for s in F.gallery():
            if s.name == name:
                _IMG[tag] = s.scene.render().image_u8
                break
    else:
        _IMG[tag] = RP.images("validation")[tag.split("|", 1)[1]]
    return _IMG[tag]


def _images(n: int = 6) -> Dict[str, np.ndarray]:
    return {t: _image(t) for t in _tags(n)}


def _job(args: Tuple[str, str, Dict[str, Any], bool]) -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from scipy.spatial import cKDTree
    from Topology.placement_cache import get_or_build
    from Topology import placement_m12_read_real as R

    tag, label, spec, m15 = args
    P.FE_KEYPOINTS = bool(m15)
    P.FE_PLACE_LAYER = bool(m15)
    P.FE_SUBPIXEL_PEAK_POS = bool(m15)
    P.FE_SUBPIXEL_GABOR_ANCHOR = bool(m15)
    P.PLACEMENT_FRAME_KEYPOINT = bool(m15)
    P.PLACEMENT_FRAME_KEYPOINT_SIGMA = "largest" if m15 else "smallest"
    P.PLACEMENT_FRAME_FALLBACK = "none" if m15 else "field"
    img0 = _image(tag)
    img1 = R._transform(img0, spec)
    k = "m15atlas|%s|%s|%d" % (tag, label, int(m15))
    g0 = get_or_build(img0, "m15atlas|%s|orig|%d" % (tag, int(m15)))
    g1 = get_or_build(img1, k)
    s0, s1 = g0.subgraph, g1.subgraph
    n0 = [int(n) for n in sorted(s0.nodes) if n in s0.anchor_of_node]
    n1 = [int(n) for n in sorted(s1.nodes) if n in s1.anchor_of_node]
    if not n0 or not n1:
        return {"tag": tag, "pose": label, "m15": m15, "empty": True}
    a0 = np.asarray([s0.anchor_of_node[n] for n in n0], float)
    a1 = np.asarray([s1.anchor_of_node[n] for n in n1], float)
    lens0 = [str(g0.sigs.get(n, ("?",))[0]) for n in n0]
    lens1 = [str(g1.sigs.get(n, ("?",))[0]) for n in n1]
    sig0 = [getattr(g0.sigs.get(n, (None, None))[1], "quantized", None) for n in n0]
    sig1 = [getattr(g1.sigs.get(n, (None, None))[1], "quantized", None) for n in n1]
    h, w = img0.shape[:2]
    fwd = R._forward(spec, a0, (h, w))
    inside = np.asarray([(R.BORDER <= x < w - R.BORDER and
                          R.BORDER <= y < h - R.BORDER) for x, y in fwd], bool)
    by_lens: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(n0)):
        by_lens[lens0[i]].append(i)
    # ridge tangent from the shipped field provider, at the source anchor
    if tag not in _ANG:
        from matching.placement_edges import field_angles
        _ANG[tag] = field_angles(img0)[0]
    ang = _ANG[tag]
    rows = []
    for lens, idxs in sorted(by_lens.items()):
        cand_idx = [j for j in range(len(n1)) if lens1[j] == lens]
        if not cand_idx:
            rows.append({"lens": lens, "n": len(idxs), "i1": 0.0, "i2": 0.0,
                         "ii1": 0.0, "ii2": 0.0, "no_counterpart": True})
            continue
        tree = cKDTree(a1[cand_idx])
        c1 = c2 = c11 = c12 = seen = 0
        along, across = [], []
        for i in idxs:
            if not inside[i]:
                continue
            seen += 1
            d, j = tree.query(fwd[i], k=1)
            jj = cand_idx[int(j)]
            if d <= TOLS[0]:
                c1 += 1
                if sig0[i] == sig1[jj]:
                    c11 += 1
            if d <= TOLS[1]:
                c2 += 1
                if sig0[i] == sig1[jj]:
                    c12 += 1
            if any(hh in lens.upper() for hh in RIDGE_HINTS):
                x, y = a0[i]
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < ang.shape[1] and 0 <= yi < ang.shape[0]:
                    th = float(ang[yi, xi])
                    v = a1[jj] - fwd[i]
                    t = np.array([np.cos(th), np.sin(th)])
                    nrm = np.array([-np.sin(th), np.cos(th)])
                    along.append(abs(float(v @ t)))
                    across.append(abs(float(v @ nrm)))
        rows.append({"lens": lens, "n": seen,
                     "i1": c1 / seen if seen else float("nan"),
                     "i2": c2 / seen if seen else float("nan"),
                     "ii1": c11 / seen if seen else float("nan"),
                     "ii2": c12 / seen if seen else float("nan"),
                     "ridge": bool(any(hh in lens.upper() for hh in RIDGE_HINTS)),
                     "along_median": float(np.median(along)) if along else None,
                     "across_median": float(np.median(across)) if across else None})
    return {"tag": tag, "pose": label, "m15": bool(m15), "nodes0": len(n0),
            "nodes1": len(n1), "inside": int(inside.sum()), "rows": rows}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    import guard
    guard.activate()
    cells: List[Dict[str, Any]] = []
    if not args.report:
        tags = _tags(6)
        jobs = [(t, lab, spec, m) for m in (False, True) for t in tags
                for lab, spec in POSES]
        print("%d cells (%d images x %d poses x 2 configurations)"
              % (len(jobs), len(tags), len(POSES)), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_job, jobs):
                cells.append(r)
                print("  %-22s %-9s m15=%-5s nodes %s -> %s"
                      % (r["tag"], r["pose"], r.get("m15"), r.get("nodes0"),
                         r.get("nodes1")), flush=True)
        with open(RESULT, "w") as fh:
            json.dump({"cells": cells, "poses": [p for p, _ in POSES],
                       "tols": list(TOLS)}, fh, indent=1, default=float)
    with open(RESULT) as fh:
        cells = json.load(fh)["cells"]

    # ---------------- aggregate: per lens, per configuration ----------------
    agg: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m15 in (False, True):
        for lens in sorted({r["lens"] for c in cells if c.get("m15") == m15
                            for r in c.get("rows", [])}):
            for pose in [p for p, _ in POSES] + ["ALL"]:
                vals = {"i1": [], "i2": [], "ii1": [], "ii2": [], "along": [],
                        "across": []}
                n = 0
                for c in cells:
                    if c.get("m15") != m15 or "rows" not in c:
                        continue
                    if pose != "ALL" and c["pose"] != pose:
                        continue
                    for r in c["rows"]:
                        if r["lens"] != lens or r.get("no_counterpart"):
                            continue
                        n += r["n"]
                        for kk in ("i1", "i2", "ii1", "ii2"):
                            if r[kk] == r[kk]:
                                vals[kk].append(float(r[kk]))
                        for kk in ("along_median", "across_median"):
                            if r.get(kk) is not None:
                                vals[kk].append(float(r[kk]))
                if not vals["i1"]:
                    continue
                agg.setdefault("m15=%d" % int(m15), {}).setdefault(pose, {})[lens] = {
                    "n_nodes": int(n), "images": len(vals["i1"]),
                    **{kk: float(np.mean(vv)) if vv else float("nan")
                       for kk, vv in vals.items()},
                    **{kk + "_min": float(np.min(vv)) if vv else float("nan")
                       for kk, vv in vals.items()},
                    **{kk + "_max": float(np.max(vv)) if vv else float("nan")
                       for kk, vv in vals.items()},
                    "ridge": bool(any(hh in lens.upper() for hh in RIDGE_HINTS)),
                }

    # ---------------- the written lists (item 1 "done") --------------------
    def _lists(m15: bool) -> Dict[str, List[str]]:
        key = "m15=%d" % int(m15)
        allp = agg.get(key, {}).get("ALL", {})
        place, ridge_across, attr = [], [], []
        for lens, v in sorted(allp.items()):
            if v["n_nodes"] < 20:
                continue
            if v["i2"] >= 0.5 and v["ii2"] >= 0.5:
                place.append(lens)
                if v["ridge"]:
                    ridge_across.append(lens)
            elif v["i2"] >= 0.5:
                # re-detected in place but its IDENTITY does not survive: an
                # attribute lens -- measurable, not matchable.
                attr.append(lens)
            else:
                attr.append(lens)
        return {"place_lenses": place, "ridge_places_across_only": ridge_across,
                "attribute_lenses": attr}

    res = {"cells": len(cells), "aggregate": agg,
           "lists": {"defaults": _lists(False), "m15": _lists(True)},
           "bar": "place = (i) at 2 px AND (ii) same quantised signature, on >= 20 nodes"}
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for cfg in ("m15=0", "m15=1"):
        print("\n== %s : per lens over ALL poses (6 synthetic + 3 real)" % cfg)
        print("%-28s %5s %6s %6s %6s %6s  %8s %8s %s"
              % ("lens", "n", "i@1", "i@2", "ii@1", "ii@2", "along", "across", "ridge"))
        for lens, v in sorted(agg.get(cfg, {}).get("ALL", {}).items(),
                              key=lambda kv: -kv[1]["i2"]):
            print("%-28s %5d %6.3f %6.3f %6.3f %6.3f  %8s %8s %s"
                  % (lens, v["n_nodes"], v["i1"], v["i2"], v["ii1"], v["ii2"],
                     "%.2f" % v["along"] if v["along"] == v["along"] else "-",
                     "%.2f" % v["across"] if v["across"] == v["across"] else "-",
                     "ridge" if v["ridge"] else ""))
    print("\nplace lenses (m15 config):", res["lists"]["m15"])
    print("place lenses (defaults):  ", res["lists"]["defaults"])
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
