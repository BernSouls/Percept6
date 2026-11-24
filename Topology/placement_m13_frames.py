"""Mission 13 step 1 -- the frame provider: fallback, sub-pixel sampling, adoption.

WHAT IS MEASURED, AND WHERE IT IS MEASURED.  Two criteria, both declared HERE, in
this file, BEFORE the first number is produced (the commit that adds this file is
the timestamp; the run that fills `m13_frames.json` is a later commit):

**E (equivariance).**  Rotate the image about its centre by delta, rotate the node
anchors by the SAME transform (the sign calibrated unanimously in M12: an anchor at
p sits at c + R(-delta)(p - c) in the rotated image, and the axial angle changes by
-delta), sample each provider's frame at the transformed anchor, and report the
SIGNED residual `theta_rot - theta_orig + delta` wrapped into (-90, 90].  Twelve
images (6 irregular synthetic scenes + 6 real photographs reserved for frame work),
delta in {5, 15}, so 24 cells.  The headline is the share of anchors within 3 deg,
pooled over cells; the paired comparison against the shipped field is a
scene-clustered bootstrap (4000 draws, clusters = the 12 images).

**C (coverage).**  The deployed matcher on the 8 hard FIT pairs of M12
(`gallery()[:8]`, irregular, second view), placement ON, everything else at the M12
setting: the mean over pairs of `placement_share_judged` (judged pairs / all pairs)
and of the judged count.  A provider that does not put frames on nodes cannot be
paid for anything it is not read on.

**THE ADOPTION RULE, declared before the run.**

1. A candidate is ELIGIBLE iff C(candidate) >= C(shipped field) (no loss of
   coverage: `share_judged` no lower, judged count no lower, both to 1e-12).
2. Among the eligible candidates, the ADOPTED provider is the one with the highest
   pooled E.
3. It is adopted only if its paired E gain over the shipped field has a 95 %
   scene-clustered bootstrap interval that EXCLUDES ZERO in its favour.
4. Otherwise nothing is adopted: the shipped single-7x7 field stays and every new
   flag stays off.

This replaces the M12 test (an 8-pair near-tie ACCURACY comparison whose interval
was [+0.000, +0.188] -- uninformative by construction, and withdrawn as a reason).

    python Topology/placement_m13_frames.py --equivariance --coverage --workers 2
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
RESULT = os.path.join(OUT, "m13_frames.json")
DELTAS = (5.0, 15.0)
BORDER = 8
NBOOT = 4000
SEED = 20261119

RULE = {
    "E": ("pooled share of anchors within 3 deg of the rotation-equivariant value, "
          "12 images x delta in {5, 15}; paired scene-clustered bootstrap vs the "
          "shipped single 7x7 field, 4000 draws"),
    "C": ("deployed matcher on the 8 hard FIT pairs: mean placement_share_judged "
          "and mean judged count"),
    "eligibility": "C(candidate) >= C(shipped field) on BOTH C numbers (1e-12)",
    "adoption": ("highest pooled E among eligible candidates, and only if the paired "
                 "E gain over the shipped field excludes 0 in its favour at 95 %"),
    "fallback": "otherwise nothing is adopted and every new flag stays off",
}

VARIANTS: Tuple[str, ...] = ("field_nearest", "field_subpixel", "multiscale",
                             "keypoint", "keypoint+field", "keypoint+multiscale")


#: [M13 / step 1, SECOND DECLARATION -- written before these candidates were
#: measured]  The first run showed the DEPLOYED keypoint provider (the SMALLEST
#: keypoint sigma whose coherence clears the bar) scoring 0.563 pooled, while M12
#: reported 0.803 for what it called "the keypoint provider".  It is not the same
#: configuration: the M12 harness sampled the tensor at the LARGEST sigma
#: (`max(KEYPOINT_SIGMAS)` = 4.0) and never at the smallest.  So the two
#: candidates below are the configuration M12 actually measured, and the same
#: E/C rule is applied to them, on their own terms, BEFORE they are run:
#:
#:   eligible  <=> coverage no worse than the shipped 7x7 field (both numbers)
#:   adopted   <=> highest pooled E among eligible, and the paired E gain over the
#:                 shipped field excludes 0 at 95 %
#:
#: They are reported beside the first decision, not folded into it: the first
#: decision's candidate set was declared and closed before any number existed.
EXTRA_RULE = {
    "why": ("M12 measured the keypoint estimator at the LARGEST sigma; the deployed "
            "provider takes the SMALLEST sigma that clears the bar; these candidates "
            "are the measured configuration"),
    "eligibility": "C(candidate) >= C(shipped field) on BOTH C numbers (1e-12)",
    "adoption": ("highest pooled E among eligible candidates, and only if the paired "
                 "E gain over the shipped field excludes 0 in favour at 95 %"),
}
EXTRA_VARIANTS: Tuple[str, ...] = ("keypoint_s4", "keypoint_s4+field")


# ---------------------------------------------------------------------------
# a stand-in "subgraph" whose anchors ARE the points to sample, so the harness
# calls the module's own provider code rather than a copy of it
# ---------------------------------------------------------------------------
class _Pts:
    def __init__(self, pts: np.ndarray) -> None:
        self.pts = np.asarray(pts, dtype=float)
        self.nodes = list(range(len(self.pts)))
        self.anchor_of_node = {i: (float(x), float(y))
                              for i, (x, y) in enumerate(self.pts)}


def frames_of(image_u8: np.ndarray, pts: np.ndarray, variant: str
              ) -> Dict[int, float]:
    """node -> frame, for one provider variant, at arbitrary sample points."""
    from matching import placement_edges as PE
    P = _Pts(pts)
    if variant == "field_nearest":
        ang, coh = PE.field_angles(image_u8)
        return PE.node_frames(P, ang, coh, subpixel=False)
    if variant == "field_subpixel":
        ang, coh = PE.field_angles(image_u8)
        return PE.node_frames(P, ang, coh, subpixel=True)
    if variant == "multiscale":
        # `node_frames_multiscale` has no subpixel argument: it reads the
        # `PLACEMENT_FRAME_SUBPIXEL` parameter, which is off in this run.
        return PE.node_frames_multiscale(P, PE.field_multiscale(image_u8))[0]
    if variant == "keypoint":
        return PE.node_frames_keypoint(P, image_u8)[0]
    if variant == "keypoint+field":
        return PE.node_frames_keypoint_then_field(P, image_u8, fallback="field")[0]
    if variant == "keypoint+multiscale":
        return PE.node_frames_keypoint_then_field(P, image_u8,
                                                  fallback="multiscale")[0]
    if variant in ("keypoint_s4", "keypoint_s4+field"):
        # the M12 configuration: the tensor at the LARGEST keypoint sigma,
        # bilinear at the anchor (exactly what m12_frames2.py sampled)
        img = np.asarray(image_u8, dtype=np.float64)
        if img.ndim == 3:
            img = img.mean(axis=2)
        gy, gx = np.gradient(img)
        ang, coh = PE._tensor_at_sigma(gx, gy, float(max(PE.KEYPOINT_SIGMAS)))
        out: Dict[int, float] = {}
        for i in range(len(P.pts)):
            x, y = float(P.pts[i][0]), float(P.pts[i][1])
            c = PE._bilinear(coh, x, y)
            if c is None or c < PE.FRAME_COHERENCE_MIN:
                continue
            a = PE._bilinear(ang, x, y)
            if a is None or not np.isfinite(a):
                continue
            out[i] = float(a)
        if variant == "keypoint_s4+field":
            f_ang, f_coh = PE.field_angles(image_u8)
            fall = PE.node_frames(P, f_ang, f_coh, subpixel=False)
            for i in range(len(P.pts)):
                if i not in out and i in fall:
                    out[i] = float(fall[i])
        return out
    raise SystemExit("unknown variant %r" % (variant,))


# ---------------------------------------------------------------------------
# E: equivariance
# ---------------------------------------------------------------------------
def _rotate(image_u8: np.ndarray, delta: float) -> np.ndarray:
    from scipy.ndimage import rotate
    a = np.asarray(image_u8, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return np.clip(rotate(a, float(delta), reshape=False, order=1,
                          mode="nearest"), 0, 255).astype(np.uint8)


def _move(pts: np.ndarray, delta: float, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    c = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
    th = np.deg2rad(-float(delta))          # the CALIBRATED sign (M12: 24/24)
    ct, st = float(np.cos(th)), float(np.sin(th))
    d = np.asarray(pts, dtype=float) - c
    return np.stack([ct * d[:, 0] - st * d[:, 1],
                     st * d[:, 0] + ct * d[:, 1]], axis=1) + c


def _wrap(a: float) -> float:
    d = float(a) % np.pi
    return d - np.pi if d > np.pi / 2.0 else d


def _images() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """(tag, image, anchors) for 6 synthetic scenes and 6 real photographs."""
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    from Topology.placement_cache import get_or_build
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    gal = [s for s in F.gallery() if s.family == "IRREGULAR"][:6]
    for i, s in enumerate(gal):
        img = s.scene.render().image_u8
        g = get_or_build(img, "m13fr|" + s.name)
        sub = g.subgraph
        pts = np.asarray([sub.anchor_of_node[n] for n in sorted(sub.nodes)
                          if n in sub.anchor_of_node], dtype=float)
        out.append(("syn|" + s.name, img, pts))
    for name, img in sorted(RP.images("validation").items()):
        g = get_or_build(img, "m13fr|" + name)
        sub = g.subgraph
        pts = np.asarray([sub.anchor_of_node[n] for n in sorted(sub.nodes)
                          if n in sub.anchor_of_node], dtype=float)
        out.append(("real|" + name, img, pts))
    return out


def equivariance(variants: Tuple[str, ...] = VARIANTS) -> Dict[str, Any]:
    import guard
    guard.activate()
    rows: List[Dict[str, Any]] = []
    for tag, img, pts in _images():
        h, w = np.asarray(img).shape[:2]
        keep = np.asarray([(BORDER <= x < w - BORDER and BORDER <= y < h - BORDER)
                           for x, y in pts], dtype=bool)
        base_pts = pts[keep]
        if len(base_pts) < 8:
            continue
        for delta in DELTAS:
            rot = _rotate(img, delta)
            moved = _move(base_pts, delta, (h, w))
            inside = np.asarray([(BORDER <= x < w - BORDER and BORDER <= y < h - BORDER)
                                 for x, y in moved], dtype=bool)
            a0, a1 = base_pts[inside], moved[inside]
            for v in variants:
                f0 = frames_of(img, a0, v)
                f1 = frames_of(rot, a1, v)
                res = []
                for i in range(len(a0)):
                    t0, t1 = f0.get(i), f1.get(i)
                    if t0 is None or t1 is None:
                        continue
                    res.append(float(np.degrees(_wrap(float(t1) - float(t0)
                                                      + np.deg2rad(delta)))))
                arr = np.asarray(res, dtype=float)
                rows.append({"tag": tag, "delta": delta, "variant": v,
                             "n_anchors": int(len(a0)), "n_framed_both": int(arr.size),
                             "frame_keep": (float(arr.size) / len(a0)) if len(a0) else 0.0,
                             "median_signed_deg": float(np.median(arr)) if arr.size else float("nan"),
                             "mad_deg": float(np.median(np.abs(arr - np.median(arr)))) if arr.size else float("nan"),
                             "share_within_3deg": float(np.mean(np.abs(arr) <= 3.0)) if arr.size else float("nan")})
    agg: Dict[str, Any] = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v and r["n_framed_both"] > 0]
        if not sub:
            continue
        agg[v] = {"cells": len(sub),
                  "share_within_3deg": float(np.mean([r["share_within_3deg"] for r in sub])),
                  "median_mad_deg": float(np.median([r["mad_deg"] for r in sub])),
                  "frame_keep": float(np.mean([r["frame_keep"] for r in sub])),
                  "framed_both": int(sum(r["n_framed_both"] for r in sub)),
                  "anchors": int(sum(r["n_anchors"] for r in sub))}
    return {"rows": rows, "aggregate": agg}


# ---------------------------------------------------------------------------
# C: coverage under the deployed matcher on the 8 FIT pairs
# ---------------------------------------------------------------------------
def _fit_pairs() -> List[Tuple[str, Any, Any]]:
    from Topology import placement_m11_families as F
    gal = [s for s in F.gallery() if s.family == "IRREGULAR"][:8]
    out = []
    for i, s in enumerate(gal):
        lab, view = F.VIEW_STEPS[i % 2]
        out.append((s.name + "|" + lab, s, view))
    return out


def _set_variant(v: str) -> None:
    import parameters as P
    if v in ("keypoint_s4", "keypoint_s4+field"):
        # `PLACEMENT_FRAME_KEYPOINT` selects the deployed smallest-sigma rule;
        # the largest-sigma configuration is reached through the M12 provider
        # switch, which is what the M13 code keeps for reproducibility.
        P.PLACEMENT_FRAME_KEYPOINT = True
        P.PLACEMENT_FRAME_MULTISCALE = False
        P.PLACEMENT_FRAME_SUBPIXEL = False
        P.PLACEMENT_FRAME_FALLBACK = ("field" if v.endswith("+field") else "none")
        P.PLACEMENT_FRAME_KEYPOINT_SIGMA = "largest"
        return
    P.PLACEMENT_FRAME_KEYPOINT = v.startswith("keypoint")
    P.PLACEMENT_FRAME_MULTISCALE = (v == "multiscale")
    P.PLACEMENT_FRAME_SUBPIXEL = (v == "field_subpixel")
    P.PLACEMENT_FRAME_KEYPOINT_SIGMA = "smallest"
    P.PLACEMENT_FRAME_FALLBACK = ("none" if v == "keypoint" else
                                  "multiscale" if v.endswith("multiscale") else "field")


def _coverage_job(job: Tuple[str, Any, Any, str]) -> Dict[str, Any]:
    tag, spec, view, v = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology import placement_m11_families as F
    from Topology.placement_cache import get_or_build
    _set_variant(v)
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.375
    P.PLACEMENT_E_NORMALISED = True
    sc1, _pv = F.pose_fn(spec, view)
    r0, r1 = spec.scene.render(), sc1.render()
    g0 = get_or_build(r0.image_u8, "m13fit|%s|ref" % spec.name)
    g1 = get_or_build(r1.image_u8, "m13fit|%s|%s" % (spec.name, tag.split("|")[1]))
    t0 = time.time()
    res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                    placement_images=(r1.image_u8, r0.image_u8))
    return {"variant": v, "pair": tag, "judged": int(res.placement_judged),
            "share": float(res.placement_share_judged),
            "mapped": int(len(res.mapping)), "sec": time.time() - t0}


def coverage(workers: int, variants: Tuple[str, ...] = VARIANTS) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    pairs = _fit_pairs()
    jobs = [(t, s, v, var) for var in variants for (t, s, v) in pairs]
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_coverage_job, jobs):
            rows.append(r)
            print("  %-22s %-22s judged %4d  share %.3f"
                  % (r["variant"], r["pair"], r["judged"], r["share"]), flush=True)
    agg = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        agg[v] = {"pairs": len(sub),
                  "share_judged": float(np.mean([r["share"] for r in sub])),
                  "judged_mean": float(np.mean([r["judged"] for r in sub])),
                  "mapped_mean": float(np.mean([r["mapped"] for r in sub]))}
    return {"rows": rows, "aggregate": agg}


# ---------------------------------------------------------------------------
def _boot(diffs: List[float], clusters: List[str]) -> Dict[str, Any]:
    by: Dict[str, List[float]] = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(d)
    keys = sorted(by)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(NBOOT):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        draws.append(float(np.mean([v for k in pick for v in by[k]])))
    draws.sort()
    return {"mean": float(np.mean(diffs)) if diffs else float("nan"),
            "n": len(diffs), "clusters": len(keys),
            "lo": float(draws[int(0.025 * len(draws))]) if draws else float("nan"),
            "hi": float(draws[int(0.975 * len(draws)) - 1]) if draws else float("nan")}


def decide(eq: Dict[str, Any], cov: Dict[str, Any]) -> Dict[str, Any]:
    agg_e, agg_c = eq["aggregate"], cov["aggregate"]
    base = "field_nearest"
    out: Dict[str, Any] = {"rule": RULE, "shipped": base, "candidates": {}}
    if base not in agg_e:
        out["adopted"] = None
        out["reason"] = "no shipped baseline row"
        return out
    # paired per-cell E differences against the shipped field
    rows = eq["rows"]
    cells = sorted({(r["tag"], r["delta"]) for r in rows})
    for v in VARIANTS:
        if v not in agg_e or v not in agg_c:
            continue
        diffs, clusters = [], []
        for tag, delta in cells:
            a = [r for r in rows if r["variant"] == base and r["tag"] == tag
                 and r["delta"] == delta]
            b = [r for r in rows if r["variant"] == v and r["tag"] == tag
                 and r["delta"] == delta]
            if not a or not b or a[0]["n_framed_both"] == 0 or b[0]["n_framed_both"] == 0:
                continue
            diffs.append(b[0]["share_within_3deg"] - a[0]["share_within_3deg"])
            clusters.append(tag)
        ci = _boot(diffs, clusters)
        c = agg_c[v]
        eligible = bool(c["share_judged"] >= agg_c[base]["share_judged"] - 1e-12
                        and c["judged_mean"] >= agg_c[base]["judged_mean"] - 1e-12)
        out["candidates"][v] = {"E": agg_e[v]["share_within_3deg"],
                                "E_gain_vs_shipped": ci,
                                "C_share_judged": c["share_judged"],
                                "C_judged_mean": c["judged_mean"],
                                "eligible": eligible}
    elig = [v for v in out["candidates"] if out["candidates"][v]["eligible"]]
    if not elig:
        out["adopted"] = None
        out["reason"] = "no candidate holds coverage; the shipped field stays"
        return out
    best = max(elig, key=lambda v: out["candidates"][v]["E"])
    gain = out["candidates"][best]["E_gain_vs_shipped"]
    out["best_eligible"] = best
    if best != base and gain["lo"] > 0.0:
        out["adopted"] = best
        out["reason"] = ("highest eligible equivariance, paired interval excludes 0 "
                         "in its favour")
    else:
        out["adopted"] = None
        out["reason"] = ("best eligible candidate (%s) does not beat the shipped field "
                         "with an interval that excludes 0 (lo=%r)" % (best, gain["lo"]))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--equivariance", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="measure the two candidates declared in EXTRA_RULE")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    res: Dict[str, Any] = {}
    if os.path.exists(RESULT):
        with open(RESULT) as fh:
            res = json.load(fh)
    if args.equivariance:
        t0 = time.time()
        res["equivariance"] = equivariance()
        print("equivariance %.0fs" % (time.time() - t0), flush=True)
        for v, a in res["equivariance"]["aggregate"].items():
            print("  %-22s share3 %.3f  mad %.3f  keep %.3f  n %d/%d"
                  % (v, a["share_within_3deg"], a["median_mad_deg"], a["frame_keep"],
                     a["framed_both"], a["anchors"]), flush=True)
    if args.coverage:
        t0 = time.time()
        res["coverage"] = coverage(args.workers)
        print("coverage %.0fs" % (time.time() - t0), flush=True)
        for v, a in res["coverage"]["aggregate"].items():
            print("  %-22s share_judged %.4f  judged %.1f  mapped %.1f"
                  % (v, a["share_judged"], a["judged_mean"], a["mapped_mean"]), flush=True)
    if args.extra:
        t0 = time.time()
        res["extra_equivariance"] = equivariance(EXTRA_VARIANTS)
        res["extra_coverage"] = coverage(args.workers, EXTRA_VARIANTS)
        eq = {"rows": res["equivariance"]["rows"] + res["extra_equivariance"]["rows"],
              "aggregate": dict(res["equivariance"]["aggregate"],
                                **res["extra_equivariance"]["aggregate"])}
        cov = {"aggregate": dict(res["coverage"]["aggregate"],
                                 **res["extra_coverage"]["aggregate"])}
        dec = decide(eq, cov)
        dec["rule"] = {"first": RULE, "extra": EXTRA_RULE}
        res["decision_all"] = dec
        print("EXTRA %.0fs" % (time.time() - t0), flush=True)
        for v in EXTRA_VARIANTS:
            a = res["extra_equivariance"]["aggregate"].get(v)
            c = res["extra_coverage"]["aggregate"].get(v)
            print("  %-18s E %.3f  C %.4f judged %.1f"
                  % (v, a["share_within_3deg"] if a else float("nan"),
                     c["share_judged"] if c else float("nan"),
                     c["judged_mean"] if c else float("nan")), flush=True)
        print("DECISION_ALL", dec.get("adopted"), "-", dec.get("reason"), flush=True)
    if "equivariance" in res and "coverage" in res:
        res["decision"] = decide(res["equivariance"], res["coverage"])
        print("DECISION", json.dumps(res["decision"].get("adopted")), "-",
              res["decision"].get("reason"), flush=True)
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
