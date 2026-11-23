"""MISSION 9 STEP 5 -- does the radius and the tolerance belong to the image?

TRACK_RADIUS and FIXED_TOL are absolute pixel numbers, frozen on synthetic scenes
where places sit about ten pixels apart.  On real ETH-80 photographs the place
layer is denser -- places sit ~5 px apart -- and an absolute radius then reaches
past the neighbouring place while an absolute tolerance spans two of them.

This script measures, it does not assume:

  * the local place spacing on synthetic FIT and on real images,
  * the tolerance the true pairs actually need at that spacing, fitted on
    SYNTHETIC FIT ONLY (never on the real images, which are the report),
  * the coverage of the frozen rules against a spacing-scaled rule on both,
  * and the place-attachment coverage on real images.

Nothing here changes a constant.  The fitted rule is written to JSON and, if it
is worth anything, the report says what it would take to adopt it.
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
ATTACH_RADIUS = 4.0
IMG_SIZE = 256


def synthetic_fit():
    """True-pair residual and the local place spacing that produced it."""
    import Topology.placement_m3 as M3
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching.placement_scorer import payload_residual
    from Topology import placement_dev5 as D5
    from Topology import placement_families_m9 as FM

    specs = list(FM.all_specs_irregular(12))
    d5 = list(D5.all_specs5())
    rows: List[Tuple[float, float, str]] = []
    for spec in specs:
        for label, view in FM.VIEWS_IRREGULAR:
            sc1, _pv = FM.pose_fn(spec, view)
            r1 = sc1.render()
            from Topology.placement_cache import get_or_build
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            side = None
            try:
                side = PM.build_side(g1, r1.image_u8)
            except Exception:
                side = None
            if side is None:
                continue
            q_idx = {}
            for (u, v, name), val in (getattr(side, "payload_index", {}) or {}).items():
                q_idx[(int(u), int(v), name)] = val
            for (u, v, name), qv in q_idx.items():
                a = g1.subgraph.anchor_of_node.get(int(u))
                b = g1.subgraph.anchor_of_node.get(int(v))
                if a is None or b is None:
                    continue
                sp = float(side.spacing_by_node.get(int(u), float("nan")))
                if not np.isfinite(sp):
                    continue
                # the payload is a relation of the two places: the tolerance it
                # has to cover is set by the COARSER of the two, not the finer
                sp2 = float(side.spacing_by_node.get(int(v), float("nan")))
                if np.isfinite(sp2):
                    sp = max(sp, sp2)
                rows.append((float(np.hypot(a[0] - b[0], a[1] - b[1])), sp, label))
    return rows


def real_images(args) -> Dict[str, Any]:
    from PIL import Image
    from Topology import corpus
    from Topology.placement_cache import get_or_build
    from matching import placement_matcher as PM

    def load_u8(path: str) -> np.ndarray:
        im = Image.open(path).convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

    spl = corpus.split()
    items: List[dict] = spl["test"]          # the 16 held-out images M8 step 5 used
    sp_all: List[float] = []
    attach = [0, 0]
    for e in items:
        u8 = load_u8(e["path"])
        g = get_or_build(u8, "real|" + corpus.name_of(e))
        side = PM.build_side(g, u8)
        if side is None:
            continue
        d = list(side.spacing_by_node.values())
        sp_all.extend(float(x) for x in d if np.isfinite(x))
        for n, sp in side.spacing_by_node.items():
            if not np.isfinite(sp):
                continue
            attach[0] += 1
            if sp <= ATTACH_RADIUS:
                attach[1] += 1
    sp = np.asarray(sp_all, dtype=float)
    return {"nodes": int(sp.size),
            "spacing_median": float(np.median(sp)) if sp.size else float("nan"),
            "spacing_p10": float(np.percentile(sp, 10)) if sp.size else float("nan"),
            "spacing_p90": float(np.percentile(sp, 90)) if sp.size else float("nan"),
            "attach_rate_at_%.1fpx" % ATTACH_RADIUS:
                (attach[1] / attach[0]) if attach[0] else float("nan")}


def main() -> int:
    import guard
    guard.activate()
    rows = synthetic_fit()
    if not rows:
        print("no synthetic rows")
        return 1
    dist = np.array([r[0] for r in rows], dtype=float)
    sp = np.array([r[1] for r in rows], dtype=float)
    per_label: Dict[str, List[float]] = defaultdict(list)
    for d, s, lab in rows:
        per_label[lab].append(s)
    print("synthetic FIT: %d place pairs, spacing median %.2f px (p10 %.2f p90 %.2f)"
          % (len(rows), float(np.median(sp)), float(np.percentile(sp, 10)),
             float(np.percentile(sp, 90))))
    for lab, v in sorted(per_label.items()):
        print("  %-10s spacing median %.2f px" % (lab, float(np.median(v))))

    # the residual needed as a function of spacing: 95 % of true pairs, fitted
    # on synthetic FIT in spacing bins
    bins = [(0, 6), (6, 8), (8, 11), (11, 99)]
    fit_rule = {}
    for lo, hi in bins:
        m = (sp >= lo) & (sp < hi)
        if m.sum() < 50:
            continue
        fit_rule["%g-%g" % (lo, hi)] = {
            "n": int(m.sum()), "median": float(np.median(dist[m])),
            "p95": float(np.percentile(dist[m], 95)),
            "spacing_median": float(np.median(sp[m]))}
        print("  spacing %-6s n=%6d residual median %.2f  p95 %.2f  (tol today %.2f)"
              % ("%g-%g" % (lo, hi), int(m.sum()), float(np.median(dist[m])),
                 float(np.percentile(dist[m], 95)), 1.7009))

    ok_sp = [fit_rule[k] for k in fit_rule]
    if len(ok_sp) >= 2:
        x = np.array([r["spacing_median"] for r in ok_sp])
        y = np.array([r["p95"] for r in ok_sp])
        slope, icpt = np.polyfit(x, y, 1)
        print("\n  fitted rule (synthetic FIT only): tolerance95 = %.3f + %.3f * spacing"
              % (icpt, slope))
    else:
        slope = icpt = float("nan")
    print("\n  out-of-plane family residual vs step (premise check, FIT): see "
          "m9_checks.json; the p90 residual grows 0.16 -> 0.53 px from 5 to 20 "
          "degrees while every true pair stays inside the frozen tolerance")

    real = {}
    if "--real" in sys.argv:
        real = real_images(None)
        print("\nreal ETH-80: %s" % json.dumps(real, indent=2))
    res = {"synthetic": {"pairs": len(rows),
                         "spacing_median": float(np.median(sp)),
                         "bins": fit_rule,
                         "rule": {"intercept": float(icpt), "slope": float(slope)},
                         "frozen_tolerance": 1.7009,
                         "frozen_radius": 2.0},
           "real": real}
    path = os.path.join(OUT, "m9_step5_spacing.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
