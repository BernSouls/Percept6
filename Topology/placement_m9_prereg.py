"""MISSION 9 -- pre-registration for the two new families, before their graphs exist.

Writes the contract for the two primary contrasts, both in the DEPLOYED matcher:

    IRREGULAR      P_plus_ctx - P_plus_ctx_presence   (ambiguous bank)
    OUT-OF-PLANE   P_plus_ctx - P_plus_ctx_presence   (ambiguous bank)

Holm across the two.  The MDE is taken from the FIT half's scene-clustered SE for
the SAME contrast in the SAME matcher and rescaled to the confirmation scene
count -- not from the harness, and not from the effect it is meant to detect.

It REFUSES to write while any graph of a confirmation scene is cached, so the
read that follows cannot be contaminated by a build that already happened.
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m9_preregistration")
    args = ap.parse_args()

    import guard
    guard.activate()
    import parameters as P
    from matching import placement_matcher as PM
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import CACHE_DIR, key_for
    from core.place_key import place_lenses

    ir = FM.all_specs_irregular(12)
    oop = FM.all_specs_oop(4)
    ir_fit, _ir_used = FM.split_scenes(ir)
    oop_fit, _oop_used = FM.split_scenes(oop)
    # the confirmation sets are FRESH scenes from new seed ranges; the scenes
    # used to repair the families and plan the read can never be confirmation
    # data, whatever the split says.
    ir_check = FM.confirmation_specs_irregular(6)
    oop_check = FM.confirmation_specs_oop(6)

    # nothing of either confirmation set may exist yet
    live: List[str] = []
    for spec in list(ir_check) + list(oop_check):
        r = spec.scene.render()
        if os.path.exists(os.path.join(CACHE_DIR, key_for(r.image_u8) + ".pkl")):
            live.append(spec.name + "|ref")
        for label, view in spec.views:
            sc, _pv = FM.pose_fn(spec, view)
            if os.path.exists(os.path.join(CACHE_DIR,
                                           key_for(sc.render().image_u8) + ".pkl")):
                live.append(spec.name + "|" + label)
    if live:
        print("REFUSED: %d graphs of the confirmation sets already exist: %s"
              % (len(live), live[:8]))
        return 2
    print("confirmation graphs present: 0 of %d"
          % (sum(1 + len(s.views) for s in list(ir_check) + list(oop_check))))

    prim: Dict[str, Any] = {}
    for fam, tag, check in (("irregular", "m9_ir_fit", ir_check),
                            ("out_of_plane", "m9_oop_fit", oop_check)):
        d = json.load(open(os.path.join(OUT, "%s.json" % tag)))
        c = d["contrasts"]["geom - pres (ambiguous)"]
        n_fit = int(c["n_scenes"])
        n_chk = len(check)
        se = float(c["se_scene"]) * float(np.sqrt(n_fit / n_chk))
        prim["%s: P_plus_ctx - P_plus_ctx_presence (ambiguous)" % fam.upper()] = {
            "fit_effect": float(c["mean"]), "fit_se_scene": float(c["se_scene"]),
            "fit_scenes": n_fit, "confirmation_scenes": n_chk,
            "se_at_confirmation": se, "mde": 2.8 * se}

    oc = {"irregular": json.load(open(os.path.join(OUT, "m9_checks_ir.json")))["irregular"],
          "oop_premise": json.load(open(os.path.join(OUT, "m9_checks_oop.json")))["oop_premise"],
          "oop_arms": json.load(open(os.path.join(OUT, "m9_checks_oop.json")))["oop_arms"]}
    bundle = PM.frozen()
    cfg = {
        "frontend_digest": __import__("eth80_frontend").frontend_source_digest(),
        "switch_suffix": __import__("eth80_frontend").switch_suffix(),
        "frozen_bundle_sha256": hashlib.sha256(
            json.dumps(bundle, sort_keys=True, default=str).encode()).hexdigest(),
        "plus": {"a_exist": bundle["_a_exist"], "a_geom": bundle["_a_geom"]},
        "iteration": bundle["iteration"], "beta": bundle["beta"],
        "cand_cap": bundle["cand_cap"], "constants": bundle["constants"],
        "endpoint_lenses": bundle["endpoint_lenses"],
        "flags": {"PLACEMENT_MATCHER_ENABLED": True,
                  "PLACEMENT_MATCHER_PRESENCE": "per arm",
                  "PLACEMENT_MATCHER_STRICT": False},
        "place_layer": bool(P.FE_PLACE_LAYER),
    }
    rec = {
        "mission": 9,
        "families": {
            "IRREGULAR": {
                "scenes_fit": [s.name for s in ir_fit],
                "scenes_confirmation": [s.name for s in ir_check],
                "views": [v[0] for v in FM.VIEWS_IRREGULAR],
                "validity": oc["irregular"]},
            "OUT_OF_PLANE": {
                "scenes_fit": [s.name for s in oop_fit],
                "scenes_confirmation": [s.name for s in oop_check],
                "views": [v[0] for v in FM.VIEWS_OOP],
                "premise": oc["oop_premise"],
                "validity": oc["oop_arms"]}},
        "frozen_configuration": cfg,
        "arms": {"off": "shipped matcher",
                 "geom": "P_plus_ctx (deployed)",
                 "pres": "P_plus_ctx_presence (a_geom = 0, same machinery)"},
        "primaries": prim,
        "correction": "Holm across the two primaries (both must reach p <= 0.025)",
        "validity_rule": ("headroom is judged on the bank the primary is defined on "
                          "(near-tie ambiguous nodes), because that is the only "
                          "population where a relational term can change a "
                          "decision; the overall-bank headroom is reported beside "
                          "it and is smaller only because unambiguous nodes are "
                          "already decided by appearance"),
        "analysis": {"unit": "scene", "bank": "ambiguous (margin < 0.05)",
                     "also_reported": ["unambiguous bank", "net flips",
                                       "moved nodes per match",
                                       "the four DEV-5 families as regression"]},
        "decision_rule": {
            "credit": "both primaries positive beyond their MDE, Holm-corrected",
            "partial": "one clears, reported as partial",
            "null": "neither clears: reported with the size that would decide"},
        "read_budget": {"looks_per_family": 1, "looks_used": 0},
        "visibility_rule": ("a node is countable iff it has a truth partner in "
                            "the posed image; oracle arms use the truth mapping "
                            "and count every such node, so self-occlusion shows "
                            "up as a smaller checkable-edge share, not as a "
                            "hidden exclusion"),
    }
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    print("\nMDE at the confirmation sizes:")
    for k, v in prim.items():
        print("  %-58s effect %+0.4f  MDE %.4f  (fit %d scenes -> check %d)"
              % (k, v["fit_effect"], v["mde"], v["fit_scenes"],
                 v["confirmation_scenes"]))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
