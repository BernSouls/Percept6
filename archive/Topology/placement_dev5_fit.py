"""MISSION 7 STEP 4 -- the FIT repair loop (free reruns).

Scores the arms on the FIT half only, under different placement CONFIGURATIONS,
so a repair can be judged without spending a CHECK look.  The configuration is
not a weight: it decides which nodes may be a placement endpoint.

    all      every node may be an endpoint (the shipped rule)
    places   only place lenses may be an endpoint; attribute and ridge nodes
             attach to the nearest place within 4 px and inherit its support
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]

ATTACH_RADIUS_PX = 4.0


def pose_fn(spec: Any, view: Dict[str, Any]):
    from Topology.placement_dev5 import posed5
    pv = {"kind": view.get("kind"),
          "rot_deg": float(view.get("rot_deg", 0.0)),
          "shift": tuple(view.get("shift", (0.0, 0.0))),
          "scale": float(view.get("scale", 1.0))}
    return posed5(spec, view), pv


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--tag", default="m7_dev5_fit")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from core.place_key import place_lenses

    specs = all_specs5(per_family=args.per_family)
    fit_specs, _check = split5(specs)
    out: Dict[str, Any] = {}

    for name, lenses in (("all", ()), ("places", tuple(place_lenses()))):
        A2.ENDPOINT_LENSES = lenses
        print("\n=== FIT rerun, endpoints = %s (%d lenses)" %
              (name, len(lenses) or 0), flush=True)
        fit, pfit, meta, ratio_fit = M3.collect_fit(fit_specs, VIEWS5, None,
                                                    pose_fn=pose_fn)
        recs = M3.run_check(fit_specs, VIEWS5, fit, pfit, None, ratio_fit,
                            pose_fn=pose_fn)
        arms = {}
        for arm in ("N", "P", "P_presence", "P_shuffled", "W", "R_fixed"):
            try:
                arms[arm] = M3.scene_stats(recs, arm)["mean"]
            except Exception:
                arms[arm] = float("nan")
        c1 = M3.contrast(recs, "P", "P_presence")
        c2 = M3.contrast(recs, "P", "N")
        out[name] = {"arms": arms, "P_minus_P_presence": c1, "P_minus_N": c2,
                     "fit": {"n_true": meta.get("n_res_true"),
                             "n_wrong": meta.get("n_res_wrong"),
                             "mu_wrong": meta.get("mu_wrong")}}
        print("  N %.4f  P %.4f  P_presence %.4f  (P-Ppres %+.4f, MDE %.4f)"
              % (arms["N"], arms["P"], arms["P_presence"], c1["mean"], c1["mde"]))
        print("  P-N %+.4f (MDE %.4f)  n_records=%d"
              % (c2["mean"], c2["mde"], len(recs)))

    A2.ENDPOINT_LENSES = ()
    path = os.path.join(ROOT, "Topology", "results", "placement",
                        "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
