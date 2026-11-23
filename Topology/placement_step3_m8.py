"""MISSION 8 STEP 3 -- the pre-registration, written before the graphs exist.

This file is the whole contract for the fresh read.  It is produced from the
FIT results committed as m8_step2.json plus the frozen front-end record; nothing
here looks at a single score from the confirmation set, and the script REFUSES
to write if a graph for any confirmation scene is already in the cache.

Contents, all frozen by this commit:

  * the exact front-end configuration (source digest, file hashes, every switch),
  * the endpoint whitelist, the context switch and the P_plus coefficients,
  * every threshold the arms use (FIXED_TOL, SLOT_CAP, LLR_CLIP, CTX_ITERS,
    CTX_MARGIN, CTX_MUTUAL, EDGE_BETA, CAND_CAP, the ambiguity margin),
  * the confirmation scenes themselves and why they are fresh,
  * the arms, the two primary contrasts, the correction, the decision rule and
    the MDE that the fresh set can actually resolve.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: Scenes are seeds `_SEED_BASE + 173 * family_index + i`; the FIT/CHECK split of
#: Mission 5-7 uses i = 0..7 and Mission 8 fits on the ODD i only.  The fresh set
#: is therefore the ranks that have never existed in any run: i >= 8.
FRESH_PER_FAMILY = 4
FRESH_FROM = 8


def fresh_specs() -> List[Any]:
    from Topology.placement_dev5 import all_specs5
    out = []
    for s in all_specs5(per_family=FRESH_FROM + FRESH_PER_FAMILY):
        i = int(s.name.rsplit("_", 1)[-1])
        if i >= FRESH_FROM:
            out.append(s)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2", default="m8_step2")
    ap.add_argument("--step0", default="m8_step0")
    ap.add_argument("--tag", default="m8_preregistration")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from matching import placement_context as Ctx
    from matching import placement_llr as L
    from matching import placement_scorer as S
    from Topology import placement_arms2 as A2
    from Topology.placement_cache import CACHE_DIR, key_for
    from Topology.placement_dev5 import VIEWS5, posed5, split5
    from core.place_key import place_lenses

    s2 = json.load(open(os.path.join(OUT, "%s.json" % args.step2)))
    freeze = json.load(open(os.path.join(OUT, "m8_frontend_freeze.json")))
    try:
        s0 = json.load(open(os.path.join(OUT, "%s.json" % args.step0)))
    except Exception:
        s0 = {}
    try:
        s0b = json.load(open(os.path.join(OUT, "m8_step0b.json")))
    except Exception:
        s0b = {}

    specs = fresh_specs()
    print("confirmation set: %d scenes" % len(specs))
    for s in specs:
        print("  %-18s %s" % (s.name, s.family))

    # NOTHING of this set may exist yet, or the read is not fresh.
    live = []
    for s in specs:
        r = s.scene.render()
        if os.path.exists(os.path.join(CACHE_DIR, key_for(r.image_u8) + ".pkl")):
            live.append(s.name + "|ref")
        for label, view in VIEWS5:
            r1 = posed5(s, view).render()
            if os.path.exists(os.path.join(CACHE_DIR,
                                           key_for(r1.image_u8) + ".pkl")):
                live.append(s.name + "|" + label)
    if live:
        print("REFUSED: %d graphs of the confirmation set already exist: %s"
              % (len(live), live[:6]))
        return 2
    print("  no graph of this set exists in the cache (%d graphs to build)"
          % (len(specs) * (1 + len(VIEWS5))))

    # MDE: scene-clustered SE on FIT, rescaled to the confirmation scene count.
    n_fit = None
    try:
        from Topology.placement_dev5 import all_specs5
        fit_specs, _ = split5(all_specs5(per_family=8))
        n_fit = len(fit_specs)
    except Exception:
        pass
    prim = ("P_plus_ctx - P_plus", "P_plus_ctx - N")
    mde = {}
    for k in prim:
        c = s2["contrasts"].get(k) or {}
        se = float(c.get("se_scene", float("nan")))
        if n_fit and np.isfinite(se):
            se_f = se * float(np.sqrt(n_fit / len(specs)))
        else:
            se_f = se
        mde[k] = {"fit_se_scene": se, "fit_n_scenes": n_fit,
                  "se_at_confirmation_size": se_f,
                  "mde": 2.8 * se_f if np.isfinite(se_f) else float("nan"),
                  "fit_effect": float(c.get("mean", float("nan")))}
    print("\nMDE at n=%d scenes (from FIT scene SE):" % len(specs))
    for k, v in mde.items():
        print("  %-24s effect %+0.4f  se %.4f -> MDE %.4f"
              % (k, v["fit_effect"], v["se_at_confirmation_size"], v["mde"]))

    rec = {
        "mission": 8,
        "confirmation_set": {
            "scenes": [{"name": s.name, "family": s.family,
                        "seed": s.notes} for s in specs],
            "n_scenes": len(specs), "n_graphs": len(specs) * (1 + len(VIEWS5)),
            "selection": ("DEV-5 ranks i >= %d of the same generator, one pose "
                          "list (VIEWS5: rot5, translate, rot15, scale).  No "
                          "scene of this set was fitted, scored or looked at in "
                          "Missions 5-8; the cache held none of its graphs when "
                          "this file was written." % FRESH_FROM),
            "views": [lbl for lbl, _v in VIEWS5],
            "justification_of_size": (
                "the mission caps the build at 128 scenes / 512 graphs; this is "
                "%d scenes / %d graphs, and the MDE below is what that buys."
                % (len(specs), len(specs) * (1 + len(VIEWS5)))),
        },
        "frozen_configuration": {
            "frontend": freeze["frontend"],
            "endpoint_whitelist": list(place_lenses()),
            "place_layer": (A2.ENDPOINT_LENSES is not None),
            "context": {"enabled": bool(M3.CONTEXT), "iters": M3.CTX_ITERS,
                        "margin": M3.CTX_MARGIN, "mutual": M3.CTX_MUTUAL,
                        "rival_tol": Ctx.RIVAL_TOL,
                        "max_rivals": Ctx.MAX_RIVALS},
            "plus": {"a_exist": s2["plus_fit"]["a_exist"],
                     "a_geom": s2["plus_fit"]["a_geom"],
                     "slot_cap": L.SLOT_CAP, "llr_clip": L.LLR_CLIP,
                     "fixed_tol": L.FIXED_TOL, "res_max": L.RES_MAX,
                     "n_bins": L.N_BINS, "pseudo": L._PSEUDO},
            "scorer": {"edge_beta": M3.EDGE_BETA, "cand_cap": M3.CAND_CAP,
                       "ambiguous_margin": A2.AMBIGUOUS_MARGIN},
            "oracle": {"used_in_read": False,
                       "note": "P_oracle / P_ratio are DEV diagnostics only and "
                               "are NOT contrasts in this read"},
        },
        "arms": ["N", "P", "P_presence", "P_plus", "P_plus_presence",
                 "P_plus_ctx", "P_shuffled", "P_oracle", "P_oracle_presence"],
        "primary_contrasts": list(prim),
        "secondary_contrasts": ["P_plus - P_plus_presence", "P - P_presence",
                                "P_plus - P", "P_plus_ctx - P_plus_presence"],
        "correction": ("two primary contrasts, both must reach p <= 0.025 "
                       "(Holm on two); the reported p values are also given "
                       "UNCORRECTED so nothing is hidden"),
        "decision_rule": {
            "credit": ("P_plus_ctx - P_plus > its MDE AND P_plus_ctx - N > its "
                       "MDE, both in the pre-registered direction, scene-"
                       "clustered"),
            "partial": ("one of the two clears its MDE and the other is "
                        "positive: reported as partial, not credited"),
            "null": "neither clears: reported as a null with the size that would decide",
        },
        "mde": mde,
        "read_budget": {"looks_on_this_set": 1, "looks_used": 0},
        "step0_context_audit": {
            "corrected": s0b,
            "note": ("the first audit in m8_step0.json compared query node ids "
                     "against reference node ids and is superseded by m8_step0b"),
            "filter": ("confident-context filter is optional by the >50% rule "
                       "(wrong share %.3f) and is kept anyway: the rival-count "
                       "refusal never relaxes"
                       % float(s0b.get("wrong_share", float("nan")))
                       if s0b else "not measured"),
        },
        "step0_oracle_gap": (s0.get("oracle_gap") if s0 else {}),
        "analysis": {
            "unit": "scene", "bank": "ambiguous (margin < AMBIGUOUS_MARGIN)",
            "also_reported": ["unambiguous bank", "per family", "net flips",
                              "both labels", "P_oracle diagnostics"],
        },
    }
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
