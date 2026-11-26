"""Mission 13 step 2 -- decompose the rotation collapse into three terms.

The M12 read measured the collapse (arm A near-tie correspondence 0.758 under an
exact shift, 0.04-0.15 under 5-15 deg of rotation and 1.15x scale) and stopped at
"rotation intolerance".  This harness names the term, with numbers, on the SAME
kind of data: 6 irregular synthetic scenes + 6 real photographs, poses
{5 deg, 15 deg, 1.15x}, and for every pose:

**a. APPEARANCE.**  For every TRUTH pair (u in the transformed graph, v in the
   original), the signature similarity `sim(sig_u, sig_v)`.  At zero pose the pair
   is the same node and reads 1.0, so the appearance loss is `1 - sim`.  Reported
   per signature class, and, decisively, as an **appearance-only correspondence
   accuracy**: is `v` the argmax of `sim(sig_u, .)` over the whole reference
   graph?  That is the ceiling any matcher built on appearance alone can reach.

   The orientation-slot hypothesis is tested by RECOMPUTING the same two numbers
   with every ORIENTATION-BEARING AXIS canonicalised to the node's own frame: for
   each slot declared circular in `CIRCULAR_CYCLE_LENGTHS` whose name carries
   "theta", the quantised bin is rotated by `-round(theta_frame/pi * cycle)`, and
   a circular population axis of the same cycle is rolled by the same amount.  If
   the absolute orientation of the patches is the cause, the canonicalised
   similarity must recover what the raw one lost.

**b. DETECTION.**  Share of the original nodes whose TRANSFORMED anchor has a node
   of the transformed graph within `DETECT_TOL` px.  This is an upper bound on any
   correspondence: a node that is not re-detected cannot be matched.

**c. TRUTH.**  Share of the original nodes (transformed inside the image) that
   yield a truth pair under the M12 tight rule (nearest within 1 px, runner-up 2x
   further, mutual).  This bounds what the instrument can even ask about.

    python Topology/placement_m13_decompose.py --workers 2
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
RESULT = os.path.join(OUT, "m13_decompose.json")
POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    # THE CONTROL: an exact integer shift resamples nothing (order-1 on an integer
    # offset is the identity), so if the harness is sound it must show detection
    # ~1.0 and a nearest distance of ~0.  Without it, "detection collapses under
    # rotation" cannot be separated from "the harness' transform is wrong".
    ("trans8", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}), ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}))
DETECT_TOL = 1.0
#: The detection curve: a node is "re-detected" at a tolerance.  Reporting only one
#: tolerance confounds DETECTION with sub-pixel ANCHOR JITTER, and the two have
#: different fixes, so the whole curve is reported.
DETECT_TOLS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
SAMPLE_Q = 160          # query nodes sampled for the appearance-only accuracy


# ---------------------------------------------------------------------------
# the frame-canonicalised reading
# ---------------------------------------------------------------------------
def _canon(sig: Any, theta: Optional[float]) -> Any:
    """The signature with its ORIENTATION-BEARING axes expressed in the node's frame.

    No axis is invented or dropped: a circular "theta" slot is rotated by the
    node's own frame (`-round(theta/pi * cycle)` bins, mod cycle) and a circular
    population axis of the same cycle is rolled by the same number of bins.  With
    `theta is None` the signature is returned untouched.
    """
    from core.registries import CIRCULAR_CYCLE_LENGTHS, POPULATION_AXES
    from core.slots import slot_names
    from core.types import TypedSignature
    if theta is None or sig is None or getattr(sig, "kind", "") != "CANONICAL":
        return sig
    cycles = CIRCULAR_CYCLE_LENGTHS.get(sig.class_)
    if not cycles:
        return sig
    names = slot_names(sig.class_)
    q = list(sig.quantized)
    pop = [tuple(v) for v in (sig.population or ())]
    axes = POPULATION_AXES.get(sig.class_, ())
    moved = False
    for slot, cycle in sorted(cycles.items()):
        nm = names[slot] if slot < len(names) else ""
        if "theta" not in nm.lower():
            continue
        if slot >= len(q):
            continue
        s = int(round(((float(theta) % np.pi) / np.pi) * cycle)) % cycle
        if s == 0:
            continue
        try:
            q[slot] = float((int(q[slot]) - s) % cycle)
        except (TypeError, ValueError):
            continue
        for i, ax in enumerate(axes):
            if "theta" in ax.name.lower() and ax.circular and ax.n_bins == cycle \
                    and i < len(pop):
                pop[i] = tuple(np.roll(np.asarray(pop[i], dtype=float), -s).tolist())
        moved = True
    if not moved:
        return sig
    return TypedSignature(kind=sig.kind, class_=sig.class_, relation=sig.relation,
                          quantized=tuple(q), population=tuple(pop))


def _frames(g: Any, img: np.ndarray) -> Dict[int, float]:
    """node -> frame from the ADOPTED provider (keypoint, largest sigma)."""
    from matching import placement_edges as PE
    th, _sig = PE.node_frames_keypoint(g.subgraph, img, sigma_rule="largest")
    return {int(k): float(v) for k, v in th.items()}


def _transform_image(img: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    from Topology import placement_m12_read_real as R
    return R._transform(img, spec)


def _truth_and_detect(g0: Any, g1: Any, spec: Dict[str, Any],
                      shape: Tuple[int, int]) -> Dict[str, Any]:
    """(truth pairs, detection share, truth survival) under one pose."""
    from scipy.spatial import cKDTree
    from Topology import placement_m12_read_real as R
    sub0, sub1 = g0.subgraph, g1.subgraph
    n0 = [int(n) for n in sorted(sub0.nodes) if n in sub0.anchor_of_node]
    n1 = [int(n) for n in sorted(sub1.nodes) if n in sub1.anchor_of_node]
    a0 = np.asarray([sub0.anchor_of_node[n] for n in n0], dtype=float)
    a1 = np.asarray([sub1.anchor_of_node[n] for n in n1], dtype=float)
    h, w = shape
    fwd = R._forward(spec, a0, (h, w))
    inside = np.asarray([(R.BORDER <= x < w - R.BORDER
                          and R.BORDER <= y < h - R.BORDER) for x, y in fwd], bool)
    tree = cKDTree(a1) if len(a1) else None
    det = 0
    curve: Dict[str, float] = {}
    if tree is not None and inside.any():
        d, _i = tree.query(fwd[inside], k=1)
        det = int(np.sum(d <= DETECT_TOL))
        for tol in DETECT_TOLS:
            curve["%.1f" % tol] = float(np.mean(d <= tol))
        vec = fwd[inside] - a1[np.argmin(
            np.linalg.norm(a1[None, :, :] - fwd[inside][:, None, :], axis=2), axis=1)] \
            if len(a1) else np.zeros((0, 2))
        bias = np.median(vec, axis=0) if len(vec) else np.asarray([float("nan")] * 2)
    else:
        d = np.asarray([])
    truth = R._truth(g1, g0, spec, (h, w))
    return {"nodes0": len(n0), "nodes1": len(n1),
            "inside": int(inside.sum()), "detected": det,
            "detect_share": (det / int(inside.sum())) if inside.any() else float("nan"),
            "detect_curve": curve,
            "nearest_median_px": float(np.median(d)) if d.size else float("nan"),
            "bias_dx": float(bias[0]), "bias_dy": float(bias[1]),
            "nearest_p90_px": float(np.percentile(d, 90)) if d.size else float("nan"),
            "truth_pairs": len(truth),
            "truth_share": (len(truth) / int(inside.sum())) if inside.any() else float("nan"),
            "truth": truth}


# ---------------------------------------------------------------------------
def _job(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, label, spec = args
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology import placement_m13_decompose as D          # for _images()
    from Topology import placement_m12_read_real as R

    imgs = D._images()
    img0 = imgs[tag]
    img1 = _transform_image(img0, spec) if label != "zero" else img0
    g0 = get_or_build(img0, "m13dc|%s|orig" % tag)
    g1 = get_or_build(img1, "m13dc|%s|%s" % (tag, label))
    shape = (img0.shape[0], img0.shape[1])
    td = _truth_and_detect(g0, g1, spec if label != "zero" else {}, shape)
    truth = td.pop("truth")
    f0 = _frames(g0, img0)
    f1 = _frames(g1, img1)
    s0, s1 = g0.sigs, g1.sigs
    # ---- appearance: the true pair, raw and frame-canonicalised -------------
    per_class: Dict[str, Dict[str, List[float]]] = {}
    pairs = []
    for u, v in sorted(truth.items()):
        su, sv = s1.get(int(u)), s0.get(int(v))
        if su is None or sv is None:
            continue
        cls = str(getattr(su[1], "class_", "") or "?")
        raw = float(su[1].similarity(sv[1]))
        can = float(_canon(su[1], f1.get(int(u))).similarity(
            _canon(sv[1], f0.get(int(v)))))
        per_class.setdefault(cls, {"raw": [], "canon": []})
        per_class[cls]["raw"].append(raw)
        per_class[cls]["canon"].append(can)
        pairs.append((int(u), int(v), raw, can))
    # ---- appearance-only correspondence accuracy ----------------------------
    lens_by_class: Dict[str, List[int]] = {}
    for v in sorted(s0):
        lens_by_class.setdefault(str(getattr(s0[v][1], "class_", "") or "?"),
                                 []).append(int(v))
    rng = np.random.default_rng(20261119)
    qs = [u for (u, _v, _r, _c) in pairs]
    if len(qs) > SAMPLE_Q:
        qs = [int(x) for x in rng.choice(qs, size=SAMPLE_Q, replace=False)]
    truth_map = {u: v for (u, v, _r, _c) in pairs}
    hit_raw = hit_can = n_q = 0
    best_raw: List[float] = []
    for u in qs:
        su, su_c = s1.get(int(u)), None
        if su is None:
            continue
        cls = str(getattr(su[1], "class_", "") or "?")
        cand = lens_by_class.get(cls, [])
        if len(cand) < 2:
            continue
        n_q += 1
        target = truth_map[int(u)]
        su_c = _canon(su[1], f1.get(int(u)))
        best_r, best_c = -1.0, -1.0
        arg_r = arg_c = -1
        for v in cand:
            sv = s0[v]
            r = float(su[1].similarity(sv[1]))
            c = float(su_c.similarity(_canon(sv[1], f0.get(v))))
            if r > best_r:
                best_r, arg_r = r, v
            if c > best_c:
                best_c, arg_c = c, v
        hit_raw += int(arg_r == target)
        hit_can += int(arg_c == target)
        best_raw.append(best_r)
    return {"tag": tag, "pose": label, **td,
            "truth_pairs": len(pairs),
            "per_class": {k: {"n": len(v["raw"]),
                              "raw_mean": float(np.mean(v["raw"])),
                              "canon_mean": float(np.mean(v["canon"])),
                              "raw_median": float(np.median(v["raw"])),
                              "canon_median": float(np.median(v["canon"]))}
                          for k, v in per_class.items()},
            "appearance": {"n": n_q,
                           "acc_raw": (hit_raw / n_q) if n_q else float("nan"),
                           "acc_canon": (hit_can / n_q) if n_q else float("nan"),
                           "best_sim_mean": float(np.mean(best_raw)) if best_raw else float("nan")}}


_IMG: Dict[str, np.ndarray] = {}


def _images() -> Dict[str, np.ndarray]:
    if _IMG:
        return _IMG
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    for s in [x for x in F.gallery() if x.family == "IRREGULAR"][:6]:
        _IMG["syn|" + s.name] = s.scene.render().image_u8
    for name, img in sorted(RP.images("validation").items()):
        _IMG["real|" + name] = img
    return _IMG


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    imgs = _images()
    jobs = [(tag, label, spec) for tag in sorted(imgs) for label, spec in POSES]
    print("%d image x pose cells" % len(jobs), flush=True)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            print("  %-14s %-9s nodes %3d/%3d  det %.2f  truth %.2f  app %.3f -> %.3f"
                  % (r["tag"], r["pose"], r["nodes0"], r["nodes1"],
                     r["detect_share"], r["truth_share"],
                     r["appearance"]["acc_raw"], r["appearance"]["acc_canon"]),
                  flush=True)
    print("cells done in %.0fs" % (time.time() - t0), flush=True)
    # ---- aggregate by class of image and by pose ---------------------------
    agg: Dict[str, Any] = {}
    for pose, _sp in POSES:
        for kind in ("syn", "real", "all"):
            sub = [r for r in rows if r["pose"] == pose
                   and (kind == "all" or r["tag"].startswith(kind))]
            if not sub:
                continue
            agg["%s|%s" % (pose, kind)] = {
                "cells": len(sub),
                "detect_share": float(np.nanmean([r["detect_share"] for r in sub])),
                "detect_curve": {k: float(np.nanmean([r["detect_curve"].get(k, float("nan"))
                                                      for r in sub]))
                                 for k in ("0.5", "1.0", "2.0", "3.0")},
                "nearest_median_px": float(np.nanmedian([r["nearest_median_px"] for r in sub])),
                "bias_px": [float(np.nanmedian([r["bias_dx"] for r in sub])),
                            float(np.nanmedian([r["bias_dy"] for r in sub]))],
                "truth_share": float(np.nanmean([r["truth_share"] for r in sub])),
                "acc_raw": float(np.nanmean([r["appearance"]["acc_raw"] for r in sub])),
                "acc_canon": float(np.nanmean([r["appearance"]["acc_canon"] for r in sub])),
            }
    cls_agg: Dict[str, Any] = {}
    for pose, _sp in POSES:
        sub = [r for r in rows if r["pose"] == pose]
        acc: Dict[str, Dict[str, List[List[float]]]] = {}
        for r in sub:
            for cls, v in r["per_class"].items():
                acc.setdefault(cls, {"raw": [], "canon": [], "n": 0})
                acc[cls]["raw"].append(v["raw_mean"])
                acc[cls]["canon"].append(v["canon_mean"])
                acc[cls]["n"] += v["n"]
        cls_agg[pose] = {cls: {"n": v["n"],
                               "raw": float(np.mean(v["raw"])),
                               "canon": float(np.mean(v["canon"]))}
                         for cls, v in sorted(acc.items()) if v["n"] >= 20}
    res = {"poses": [p for p, _ in POSES], "rows": rows, "aggregate": agg,
           "by_class": cls_agg,
           "note": ("zero pose is the identity: the true pair is the same node, so "
                    "its similarity is 1.0 and the appearance loss is 1 - sim; "
                    "acc_raw/acc_canon are appearance-only correspondence accuracy "
                    "(is the true counterpart the argmax over the whole same-class "
                    "reference set)")}
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(agg, indent=1))
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
