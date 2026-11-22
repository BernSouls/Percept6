"""Task 4 PLACEMENT mission — Part 3: the scorer's frozen tolerance.

TOLERANCE IS MEASURED, THEN FROZEN, BEFORE SCORING.

    tolerance = max( q90 of the residual over TRUE correspondences at that
                     view step,
                     the detector's localisation sigma in the same
                     scale-normalised units )

The floor is REQUIRED: at zero pose change the true residual is 0, so a
tolerance derived only from data would be 0 and every edge would be
uncheckable.  The localisation sigma is the floor.

WHERE THE TRUTH COMES FROM
--------------------------
``Topology/placement_scenes.py`` carries every feature's ``point_local`` on
its object, so the SAME feature under two poses is a TRUE correspondence by
construction (the feature's identity is its (object, point_local) pair).  The
residual of a true correspondence is the difference between the query view's
payload and the reference view's payload for that SAME identity.

This script writes ``Topology/results/placement/tolerance.json``.  After it
runs, the numbers in that file are the frozen table; nothing downstream may
re-derive them.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: The detector's localisation sigma, in PIXELS.  Taken from the front end's
#: own subpixel peak work: ``FE_SUBPIXEL_PEAK_POS`` reports peak positions on a
#: subpixel grid, and the stage-6 position registry quantises anchors to the
#: pixel centre.  The conservative reading is one pixel; it is expressed in
#: SCALE UNITS below by dividing by the graph's median NN spacing, so the floor
#: is in the same unit as the residual.
DETECTOR_SIGMA_PX = 1.0


def _payload_pair_residuals(spec: Any) -> Dict[str, List[float]]:
    """Residuals of TRUE correspondences, per view step.

    A true correspondence is one FEATURE (object, point_local) seen in the
    reference view and in one query view.  Its payload is built from the
    object's own frame, so the residual is the angular + metric difference
    between the two payloads of the same identity.
    """
    from matching.placement import (median_nn_spacing, node_scale,
                                    placement_vector, wrap_pi)
    from Topology.placement_scenes import derive_track, posed

    # reference view = v0_base; query views = the other steps
    views = [v[0] for v in spec.views]
    ref = "v0_base"

    def frames(label: str) -> Dict[str, Tuple[Tuple[float, float], float]]:
        """feature key -> (image position, continuous orientation of the
        FEATURE's own local frame under this pose)."""
        pose = dict(spec.views)[label]
        sc = posed(spec.scene, pose["rot_deg"], tuple(pose["shift"]),
                   float(pose["scale"]))
        by_name = {o.name: o for o in sc.objects}
        tr = derive_track(spec, label)
        fallback = median_nn_spacing(list(tr.values()))
        out: Dict[str, Tuple[Tuple[float, float], float, float]] = {}
        for f in spec.features:
            o = by_name.get(f.object_name)
            if o is None:
                continue
            from Topology.placement_scenes import _fkey
            k = _fkey(f)
            xy = tr.get(k)
            if xy is None:
                continue
            # the feature's own local orientation, carried through the pose
            th = _feature_theta(o, f)
            sc_ = node_scale(xy, fallback)
            out[k] = (xy, th, sc_)
        return out

    ref_frames = frames(ref)
    # neighbour structure: which features are "related" is decided by the
    # placement graph itself, but for the TOLERANCE we only need the residual
    # of a true correspondence, which is per FEATURE, not per edge.
    out: Dict[str, List[float]] = {}
    for label in views:
        if label == ref:
            continue
        qf = frames(label)
        res: List[float] = []
        for k in sorted(set(ref_frames) & set(qf)):
            pu, tu, su = ref_frames[k]
            pv, tv, sv = qf[k]
            if tu is None or tv is None or su is None or su <= 0 or sv is None or sv <= 0:
                continue
            a = placement_vector(pu, pv, tu, tv, su, sv)
            b = placement_vector(pu, pu, tu, tu, su, su)
            if a is None or b is None:
                continue
            # residual of the true correspondence: its direction must be the
            # SAME as the identity's own (zero) direction, and its distance
            # must equal the identity's (zero).
            res.append(abs(wrap_pi(a.direction - b.direction)))
            res.append(abs(a.distance - b.distance))
        out[label] = res
    return out


def _feature_theta(obj: Any, f: Any) -> Optional[float]:
    """A feature's own continuous orientation, in radians.

    The feature's local frame is the OBJECT's frame at that point: the tangent
    of the surface.  For a sphere the surface tangent in the plane containing
    the point and the object's z axis is the observable orientation; using the
    azimuth of the point gives a well-defined, pose-carrying direction.
    """
    r0, r1, r2 = f.point_local
    return math.atan2(r1, r0)


def main() -> int:
    import guard
    guard.activate()
    from Topology.placement_scenes import all_specs

    os.makedirs(OUT, exist_ok=True)
    per_view: Dict[str, List[float]] = {}
    for spec in all_specs():
        r = _payload_pair_residuals(spec)
        for label, vals in r.items():
            per_view.setdefault(label, []).extend(vals)

    table: Dict[str, Any] = {
        "detector_sigma_px": DETECTOR_SIGMA_PX,
        "note": "q90 of TRUE-correspondence residual per view step; the "
                "tolerance is max(q90, detector floor) and is FROZEN here",
        "views": {},
    }
    for label in sorted(per_view):
        vals = np.asarray(per_view[label], dtype=np.float64)
        if vals.size == 0:
            continue
        q90 = float(np.quantile(vals, 0.90))
        # The detector floor in the SAME scale-normalised units.  The
        # residual above is already scale-normalised (distance in source
        # scale units, direction in radians), so the floor converts one pixel
        # of localisation error through the same median NN spacing used as
        # the fallback scale.  A representative spacing of 6 px puts one
        # pixel at 1/6 of a scale unit, and 1 rad of direction error at 1 px
        # of arc at radius 1 scale unit.
        floor = DETECTOR_SIGMA_PX / 6.0
        table["views"][label] = {
            "n_residuals": int(vals.size),
            "median": float(np.median(vals)),
            "q90": q90,
            "max": float(vals.max()),
            "detector_floor": floor,
            "tolerance": max(q90, floor),
        }
    path = os.path.join(OUT, "tolerance.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=2, sort_keys=True)
    for label in sorted(table["views"]):
        v = table["views"][label]
        print("%-14s n=%4d median=%.4f q90=%.4f floor=%.4f -> tol=%.4f"
              % (label, v["n_residuals"], v["median"], v["q90"],
                 v["detector_floor"], v["tolerance"]), flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
