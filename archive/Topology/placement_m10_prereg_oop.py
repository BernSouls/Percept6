"""MISSION 10 STEP 2 -- pre-registration for the OUT-OF-PLANE confirmation.

Mission 9's out-of-plane read was underpowered: +0.0203 with MDE 0.0355 at six
scenes.  This registers a decision BEFORE the scenes exist, on a seed range that
has never been rendered, and the decision rule is fixed here:

  scenes       20 fresh out-of-plane scenes (seed base 30000, five per object
               type), rendered at reference pose and under the six y/x rotations
               of the family.  Registered while ZERO of their graphs exist; the
               script refuses if any does.
  bundle       whatever Step 1 froze.  If Step 1 kept the Mission-8 bundle, that
               is the bundle; the record names the version and its numbers.
  arms         deployed matcher, PLACEMENT_MATCHER_ENABLED on, geometry arm
               (a_geom as frozen) against presence arm (a_geom = 0).  Same
               machinery, same caps, same context iteration -- the two arms
               differ in one coefficient.
  primary      geometry - presence on the near-tie AMBIGUOUS bank, scene-clustered
               paired t over the 20 scenes, two-sided.  This set alone decides;
               Mission 9's six scenes are reported beside it and never pooled.
  no harm      the unambiguous bank must not fall by more than its own MDE.
  MDE          taken from FIT (the out-of-plane FIT half, all eight scenes) for
               the SAME contrast and rescaled to 20 scenes.  If the registered
               MDE exceeds the effect FIT suggests, that is stated in the record
               as a known risk rather than fixed after the fact.
  looks        one.  A second look at this family requires a new seed range and a
               new registration.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

N_SCENES = 20


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m10_prereg_oop")
    ap.add_argument("--fit-read", default="m10_oop_fit.json")
    args = ap.parse_args()

    import guard
    guard.activate()
    import parameters as P
    import eth80_frontend as F
    from matching import placement_matcher as PM
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import CACHE_DIR, key_for

    specs = FM.confirmation_specs_oop_m10(N_SCENES)
    if len(specs) != N_SCENES:
        print("scene generator produced %d, expected %d" % (len(specs), N_SCENES))
        return 2

    present: List[str] = []
    total = 0
    for spec in specs:
        total += 1
        if os.path.exists(os.path.join(CACHE_DIR, key_for(spec.scene.render().image_u8) + ".pkl")):
            present.append(spec.name + "|ref")
        for label, view in spec.views:
            sc, _pv = FM.pose_fn(spec, view)
            total += 1
            if os.path.exists(os.path.join(CACHE_DIR,
                                           key_for(sc.render().image_u8) + ".pkl")):
                present.append(spec.name + "|" + label)
    if present:
        print("REFUSED: %d of %d confirmation graphs already exist: %s"
              % (len(present), total, present[:6]))
        return 2
    print("confirmation graphs present: 0 of %d" % total)

    fit = json.load(open(os.path.join(OUT, args.fit_read)))
    c_amb = fit["contrasts"]["geom - pres (ambiguous)"]
    c_un = fit["contrasts"]["geom - pres (unambiguous)"]
    n_fit = int(c_amb["n_scenes"])
    se = float(c_amb["se_scene"]) * float(np.sqrt(n_fit / N_SCENES))
    bundle = PM.frozen()
    refit_path = os.path.join(OUT, "m10_refit.json")
    refit = json.load(open(refit_path)) if os.path.exists(refit_path) else None

    rec: Dict[str, Any] = {
        "mission": 10,
        "family": "OUT_OF_PLANE (third seed range)",
        "scenes": [s.name for s in specs],
        "view_labels": [v[0] for v in FM.VIEWS_OOP],
        "scenes_by_shape": {sh: sum(1 for s in specs if sh.lower() in s.name)
                            for sh in FM.OOP_SHAPES},
        "bundle": {"version": bundle.get("version"),
                   "plus": bundle["plus"], "beta": bundle["beta"],
                   "cand_cap": bundle["cand_cap"],
                   "iteration": bundle["iteration"],
                   "sha256": hashlib.sha256(json.dumps(
                       bundle, sort_keys=True, default=str).encode()).hexdigest()},
        "step1_refit_decision": (None if refit is None else
                                 {"adopted": refit.get("adopted"),
                                  "best": (refit.get("best") or {}).get("name")}),
        "arms": {"geometry": "PLACEMENT_MATCHER_ENABLED on, a_geom frozen",
                 "presence": "PLACEMENT_MATCHER_ENABLED on, a_geom = 0"},
        "primary": "geometry - presence, ambiguous bank, scene-clustered paired t",
        "no_harm": "unambiguous bank must not fall by more than its own MDE",
        "fit_estimate": {"effect": float(c_amb["mean"]), "se_scene": float(c_amb["se_scene"]),
                         "n_fit_scenes": n_fit,
                         "mde_if_same_spread": 2.8 * se,
                         "unambiguous_effect": float(c_un["mean"])},
        "mde_at_registration": 2.8 * se,
        "se_at_registration": se,
        "looks": {"planned": 1, "used": 0},
        "pooling": "none: this set alone decides",
        "flags": {"PLACEMENT_MATCHER_ENABLED": True,
                  "PLACEMENT_MATCHER_PRESENCE": "per arm",
                  "PLACEMENT_MATCHER_STRICT": False},
        "frontend_digest": F.frontend_source_digest(),
        "place_layer": bool(P.FE_PLACE_LAYER),
    }
    path = os.path.join(OUT, args.tag + ".json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    print("FIT: effect %+0.4f (SE %.4f over %d scenes) -> MDE at %d scenes %.4f"
          % (rec["fit_estimate"]["effect"], c_amb["se_scene"], n_fit,
             N_SCENES, rec["mde_at_registration"]))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
