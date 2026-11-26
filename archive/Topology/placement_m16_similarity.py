"""MISSION 16 -- does a place stay RECOGNISABLE across a rotation?

THE VERIFICATION STAGE (`--stage detections`, cheap, no graphs)

For each image and each exact transform it pairs the keypoint DETECTIONS the tree
sees (same lens, transformed position within TRUTH_TOL) and asks what the tree's own
`TypedSignature.similarity` says about each pair, under every arm:

    off              the shipped state (KP_GRADED_SIMILARITY off)
    graded-narrow    the constitution, sigma_theta = 1 bin = 15 deg
    graded-wide      the constitution, sigma_theta = 4 bins = 60 deg
    pop-heavy        population arm, axis weights (theta 8, coherence 1, scale 1)
    pop-balanced     population arm, axis weights (theta 4, coherence 1, scale 2)

and it also classifies WHICH BRANCH of `core/types.py::_similarity_gaussian` runs,
which is verification question 2: exact-equal -> 1.0, ranges-None -> 0.0, the graded
kernel, or the population overlap.

THE GRAPH STAGE (`--stage graphs`) builds the M15 cells under an arm and measures
acceptance B (candidate admissibility), C (geometry AUC at 15 deg) and D (appearance
ranking) on the same cells Mission 15 used.

    python Topology/placement_m16_similarity.py --stage detections
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
RESULT = os.path.join(OUT, "m16_similarity.json")

#: (label, graded, population, population axis weights, theta tuning width in bins)
ARMS: Tuple[Tuple[str, bool, bool, Optional[Tuple[float, float, float]], float], ...] = (
    ("off", False, False, None, 1.0),
    ("graded-narrow", True, False, None, 1.0),
    ("graded-wide", True, False, None, 1.0),
    ("pop-heavy", True, True, (8.0, 1.0, 1.0), 1.0),
    ("pop-balanced", True, True, (4.0, 1.0, 2.0), 1.0),
    ("pop-wide", True, True, (8.0, 1.0, 1.0), 2.0),
    ("pop-wide-balanced", True, True, (4.0, 1.0, 2.0), 2.0),
)
POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (0.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
TRUTH_TOL = 1.7009
BORDER = 16
IMAGES: Tuple[str, ...] = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")


def _image(tag: str) -> np.ndarray:
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        return [s for s in F.gallery() if s.name == name][0].scene.render().image_u8
    return RP.images("validation")[tag.split("|", 1)[1]]


def _pairs(p0: List[Any], p1: List[Any], spec: Dict[str, Any],
           shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Same-lens detection pairs whose transformed positions agree, injective."""
    from Topology import placement_m12_read_real as R
    from scipy.spatial import cKDTree
    h, w = shape
    a0 = np.asarray([[p.x, p.y] for p in p0], float)
    a1 = np.asarray([[p.x, p.y] for p in p1], float)
    fwd = R._inverse(spec, a1, (h, w)) if False else R._forward(spec, a0, (h, w))
    out: List[Tuple[float, int, int]] = []
    by_kind: Dict[str, List[int]] = {}
    for j, q in enumerate(p1):
        by_kind.setdefault(q.kind, []).append(j)
    for i, p in enumerate(p0):
        if not (BORDER <= fwd[i][0] < w - BORDER and BORDER <= fwd[i][1] < h - BORDER):
            continue
        idx = by_kind.get(p.kind, [])
        if not idx:
            continue
        d = np.hypot(a1[idx, 0] - fwd[i][0], a1[idx, 1] - fwd[i][1])
        k = int(np.argmin(d))
        if d[k] <= TRUTH_TOL:
            out.append((float(d[k]), i, idx[k]))
    used: Dict[int, int] = {}
    for _d, _i, j in out:
        used[j] = used.get(j, 0) + 1
    return [(i, j) for _d, i, j in out if used[j] == 1]


def _branch(a: Any, b: Any) -> str:
    """Verification question 2: which branch of _similarity_gaussian runs."""
    from core.registries import get_population_axes, get_similarity_ranges
    if a.class_ != b.class_:
        return "class_gate_0.0"
    axes = get_population_axes(a.class_)
    if a.population and b.population and axes is not None:
        return "population"
    if a.quantized == b.quantized:
        return "exact_equal_1.0"
    if get_similarity_ranges(a.class_) is None:
        return "ranges_none_0.0"
    return "graded_kernel"


def _arm_signature(p: Any, graded: bool, population: bool,
                   pop_weights: Optional[Tuple[float, float, float]],
                   theta_sigma: float = 1.0) -> Any:
    """Build the signature a keypoint WOULD carry under an arm (no graph needed)."""
    import parameters as P
    from core import keypoint_classes as KPC
    if pop_weights is not None:
        KPC.POPULATION_AXIS_SPEC = (
            ("theta", 12, True, float(theta_sigma), float(pop_weights[0])),
            ("coherence", 4, False, 1.0, float(pop_weights[1])),
            ("scale", 4, False, 1.0, float(pop_weights[2])),
        )
    P.KP_GRADED_SIMILARITY = bool(graded)
    P.KP_GRADED_POPULATION = bool(population)
    KPC.unregister()          # an arm starts from the SHIPPED registries, always
    from lenses.keypoints import signature_of
    return signature_of(p)


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def stage_detections() -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from lenses import keypoints as K
    from matching import m15_manifest as MF
    from Topology import placement_m12_read_real as R

    rows: List[Dict[str, Any]] = []
    for tag in IMAGES:
        img = _image(tag)
        shape = img.shape[:2]
        det0 = K.detect(img)
        for pose, spec in POSES:
            img1 = R._transform(img, spec)
            det1 = K.detect(img1)
            pairs = _pairs(det0, det1, spec, shape)
            rng = np.random.default_rng(12345)
            for arm, graded, population, popw, tsigma in ARMS:
                variant = "wide" if arm == "graded-wide" else "narrow"
                P.KP_GRADED_VARIANT = variant
                s0 = [_arm_signature(p, graded, population, popw, tsigma)
                      for p in det0]
                s1 = [_arm_signature(p, graded, population, popw, tsigma)
                      for p in det1]
                branches: Dict[str, int] = {}
                trues: List[float] = []
                pairs_auc: List[Tuple[float, List[float]]] = []
                for (i, j) in pairs:
                    s = float(s0[i].similarity(s1[j]))
                    trues.append(s)
                    br = _branch(s0[i], s1[j])
                    branches[br] = branches.get(br, 0) + 1
                    # impostors: same lens, sampled deterministically, not the partner
                    cand = [jj for jj in range(len(det1))
                            if det1[jj].kind == det0[i].kind and jj != j]
                    if len(cand) > 24:
                        cand = list(rng.choice(cand, 24, replace=False))
                    if cand:
                        pairs_auc.append((s, [float(s0[i].similarity(s1[jj]))
                                              for jj in cand]))
                rows.append({"tag": tag, "pose": pose, "arm": arm,
                             "pairs": len(pairs), "true_mean": (float(np.mean(trues))
                                                                if trues else None),
                             "true_median": (float(np.median(trues)) if trues else None),
                             "true_min": (float(np.min(trues)) if trues else None),
                             "true_zero_share": (float(np.mean([t == 0.0 for t in trues]))
                                                 if trues else None),
                             "auc": _auc(pairs_auc), "branches": branches})
                print("  %-14s %-8s %-14s pairs %4d true med %s zero %.2f auc %s"
                      % (tag, pose, arm, len(pairs),
                         "%.3f" % rows[-1]["true_median"] if rows[-1]["true_median"]
                         is not None else "-", rows[-1]["true_zero_share"] or 0.0,
                         "%.3f" % rows[-1]["auc"] if rows[-1]["auc"] else "-"), flush=True)
    # reset the arm state to the shipped one
    P.KP_GRADED_SIMILARITY = False
    P.KP_GRADED_POPULATION = False
    return {"stage": "detections", "rows": rows,
            "images": list(IMAGES), "poses": [p for p, _ in POSES],
            "truth_tol": TRUTH_TOL,
            "units": ("detection positions in PIXELS; similarity is unitless [0,1]")}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="detections", choices=("detections", "report"))
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    from matching import m15_manifest
    if args.stage == "report":
        d = json.load(open(RESULT))
    else:
        d = stage_detections()
        m15_manifest.write_result(RESULT, d)
    rows = d["rows"]
    print("\n== acceptance A: the matcher's own similarity on TRUE place pairs")
    print("%-14s %-8s %-14s %5s %8s %7s %7s %8s"
          % ("image", "pose", "arm", "pairs", "median", "zero", "min", "auc"))
    for r in rows:
        print("%-14s %-8s %-14s %5d %8s %7.2f %7s %8s"
              % (r["tag"], r["pose"], r["arm"], r["pairs"],
                 "%.3f" % r["true_median"] if r["true_median"] is not None else "-",
                 r["true_zero_share"] or 0.0,
                 "%.3f" % r["true_min"] if r["true_min"] is not None else "-",
                 "%.3f" % r["auc"] if r["auc"] else "-"))
    print("\n== verification 2: which branch runs (summed over images, per pose/arm)")
    for pose in [p for p, _ in POSES]:
        for arm, *_rest in ARMS:
            agg: Dict[str, int] = {}
            for r in rows:
                if r["pose"] == pose and r["arm"] == arm:
                    for k, v in r["branches"].items():
                        agg[k] = agg.get(k, 0) + v
            if agg:
                print("  %-8s %-14s %s" % (pose, arm, agg))
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
