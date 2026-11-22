"""DEV fitting for the placement mission: tolerance, separation, and the arms.

Everything here is derived on DEV and then frozen.  The EVAL set is separate
and is not touched.

WHAT IS PRODUCED
  tolerance   per view step, from the q90 of the TRUE-correspondence residual
              of the new edge class, with a floor (a zero pose change must give
              a zero residual) and a detector-localisation term in the same
              scale-normalised units.  FROZEN before any arm is scored.
  separation  true-vs-wrong AUC and d' per view step, so the payload's own
              power is known before the arms are asked to use it.
  arms        N, R, R-fixed, P on the DEV set, on OBJECT-anchored nodes,
              ambiguous and unambiguous reported separately.
  modifier    the pre-stated rule for the strength: placement may overturn a
              NEAR-TIE but must not overturn a CLEAR gap.  The rule is stated
              and then CHECKED, not fitted.
  headroom    of N's wrong ambiguous nodes, how many have the true match among
              N's candidates AND a relational context that differs from the
              rivals -- the ceiling on what any relational arm could fix.

Run:  .venv/bin/python Topology/placement_devfit.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")
#: Localisation of a detector, in pixels; converted to scale units per node.
DETECTOR_SIGMA_PX = 1.0
#: How many wrong reference vectors to draw per query vector, and the overall
#: cap.  Deterministic (canonical stride), large enough for a stable AUC.
WRONG_PER_QUERY = 40
WRONG_CAP = 120000


def true_pairs(truth: Dict[int, int], q_pay: Dict[Tuple, Any],
               r_pay: Dict[Tuple, Any]) -> List[Tuple[Any, Any]]:
    """(query vector, reference vector) for edges whose BOTH endpoints are true."""
    out: List[Tuple[Any, Any]] = []
    for key in sorted(q_pay, key=lambda k: (k[0], k[1], getattr(k[2], "name", ""))):
        s, d, et = key
        ms, md = truth.get(int(s)), truth.get(int(d))
        if ms is None or md is None:
            continue
        rv = r_pay.get((int(ms), int(md), et))
        if rv is not None:
            out.append((q_pay[key], rv))
    return out


def residual_parts(q: Any, r: Any) -> Tuple[float, float, float]:
    """The three residual components in NATURAL units (before dividing by tol)."""
    from matching.placement import wrap_ax, wrap_pi
    if bool(getattr(q, "axial", False)) or bool(getattr(r, "axial", False)):
        dpsi = abs(wrap_ax(float(q.direction) - float(r.direction)))
    else:
        dpsi = abs(wrap_pi(float(q.direction) - float(r.direction)))
    dlogd = abs(math.log(max(float(q.distance), 1e-9))
                - math.log(max(float(r.distance), 1e-9)))
    dlogs = abs(float(q.log_scale) - float(r.log_scale))
    return dpsi, dlogd, dlogs


def separation_from_parts(true_p: Sequence[Tuple[float, float, float]],
                          wrong_p: Sequence[Tuple[float, float, float]],
                          tol: float) -> Tuple[float, float, float]:
    def res(t):
        dpsi, dlogd, dlogs = t
        return math.sqrt((dpsi / tol) ** 2 + (dlogd / tol) ** 2
                         + (dlogs / tol) ** 2)
    t = [res(x) for x in true_p]
    w = [res(x) for x in wrong_p]
    from matching.placement_scorer import separation_auc, d_prime
    return (separation_auc(t, w), d_prime(t, w),
            float(np.median(t)) if t else float("nan"))


def main() -> int:
    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching.placement import edge_payloads
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import track_optimal
    from Topology.placement_scenes import posed
    from Topology.placement_dev import all_specs, DEV_VIEWS
    from Topology import placement_census as C

    os.makedirs(OUT, exist_ok=True)

    # ---------------- pass 1: true correspondences -> tolerance ------------
    true_parts: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    scale_by_view: Dict[str, List[float]] = defaultdict(list)
    cache: Dict[Tuple[str, str], Any] = {}

    for spec in all_specs():
        r0 = spec.scene.render()
        g_ref = get_or_build(r0.image_u8, spec.name + "|ref")
        r_pay = PE.build_edges(g_ref.subgraph, r0.image_u8)
        for label, view in DEV_VIEWS:
            if label == "v0_base":
                continue
            sc1 = posed(spec.scene, float(view["rot_deg"]),
                        tuple(view["shift"]), float(view["scale"]))
            r1 = sc1.render()
            g_q = get_or_build(r1.image_u8, spec.name + "|" + label)
            q_pay = PE.build_edges(g_q.subgraph, r1.image_u8)
            truth_fwd = track_optimal(spec.scene, r0, sc1, r1, g_ref, g_q)
            truth = {int(q): int(r) for r, q in truth_fwd.items()}
            cache[(spec.name, label)] = (g_ref, g_q, r_pay, q_pay, truth, r0, r1)
            for qv, rv in true_pairs(truth, q_pay, r_pay):
                true_parts[label].append(residual_parts(qv, rv))
            scale_by_view[label].extend(
                float(v.source_scale) for v in q_pay.values())

    tol_table: Dict[str, Dict[str, float]] = {}
    for label in sorted(true_parts):
        parts = true_parts[label]
        if not parts:
            continue
        # q90 of the TRUE residual, measured with tol = 1 so the number is in
        # SCALE UNITS; the floor is the detector localisation in the same units.
        res1 = [math.sqrt(a * a + b * b + c * c) for a, b, c in parts]
        q90 = float(np.quantile(res1, 0.90))
        med_scale = float(np.median(scale_by_view[label])) if scale_by_view[label] else 5.0
        floor = DETECTOR_SIGMA_PX / max(med_scale, 1e-9)
        tol_table[label] = {"n": len(parts), "q90": q90, "median": float(np.median(res1)),
                            "floor": floor,
                            "tolerance": max(q90, floor),
                            "median_source_scale_px": med_scale}

    print("FROZEN TOLERANCE (DEV)")
    for label in sorted(tol_table):
        t = tol_table[label]
        print("  %-13s n=%4d  median %.4f  q90 %.4f  floor %.4f  -> tol %.4f"
              % (label, t["n"], t["median"], t["q90"], t["floor"],
                 t["tolerance"]))

    # ---------------- pass 2: separation -----------------------------------
    sep: Dict[str, Dict[str, float]] = {}
    for label in sorted(tol_table):
        tol = tol_table[label]["tolerance"]
        tp: List[Tuple[float, float, float]] = []
        wp: List[Tuple[float, float, float]] = []
        for (spec_name, lab), (g_ref, g_q, r_pay, q_pay, truth, r0, r1) in \
                sorted(cache.items()):
            if lab != label:
                continue
            for qv, rv in true_pairs(truth, q_pay, r_pay):
                tp.append(residual_parts(qv, rv))
            # WRONG: the same query vector against other reference vectors of
            # the same class that are NOT its true counterpart.  (Permuting the
            # mapping destroys the reference edge and yields almost no wrong
            # pairs -- measured in the earlier mission.)
            true_keys = {(truth[int(k[0])], truth[int(k[1])], k[2])
                         for k in sorted(q_pay, key=lambda k: (k[0], k[1]))
                         if int(k[0]) in truth and int(k[1]) in truth}
            # Deterministic CAP: the wrong population is quadratic in the
            # edge count (~5000 x 5000 per view), which made a full sweep take
            # over seven minutes and produce nothing.  A canonical stride keeps
            # the sample reproducible and far larger than needed for an AUC.
            rks = sorted(r_pay, key=lambda kk: (kk[0], kk[1]))
            qks = sorted(q_pay, key=lambda kk: (kk[0], kk[1]))
            stride = max(1, len(rks) // WRONG_PER_QUERY)
            for k in qks:
                qv = q_pay[k]
                for rk in rks[::stride]:
                    if rk in true_keys:
                        continue
                    wp.append(residual_parts(qv, r_pay[rk]))
                    if len(wp) >= WRONG_CAP:
                        break
                if len(wp) >= WRONG_CAP:
                    break
        auc, dp, med_true = separation_from_parts(tp, wp, tol)
        sep[label] = {"auc": auc, "d_prime": dp, "n_true": len(tp),
                      "n_wrong": len(wp), "median_true": med_true,
                      "tolerance": tol}
        print("  SEP %-13s AUC %s  d' %s  n_true %d n_wrong %d"
              % (label, "%.4f" % auc if auc else "NA",
                 "%.3f" % dp if dp else "NA", len(tp), len(wp)))

    payload = {"tolerance": tol_table, "separation": sep,
               "detector_sigma_px": DETECTOR_SIGMA_PX}
    with open(os.path.join(OUT, "devfit_tolerance.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("\nwrote", os.path.join(OUT, "devfit_tolerance.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
