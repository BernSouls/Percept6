"""Freeze the Mission-8 FIT into a JSON bundle the deployed matcher can read.

A deployed matcher must not fit, refit, or look anything up at match time.  This
writes every number the scorer needs -- the residual LLR density, the presence
log-ratios, the two P_plus coefficients and the caps they were fitted under --
together with the frozen iteration constants, and it REFUSES to write if the
live caps have drifted from the ones the fit used.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
TARGET = os.path.join(ROOT, "matching", "placement_frozen_params.json")


def main() -> int:
    from matching import placement_llr as L
    from matching import placement_context as Ctx
    from core.place_key import place_lenses

    with open(os.path.join(OUT, "_fitcache", "m8_step2_fit.pkl"), "rb") as fh:
        F = pickle.load(fh)
    bundle = L.to_jsonbundle(F["fit"], F["pfit"], float(F["a_exist"]),
                             float(F["a_geom"]))
    bundle["iteration"] = {"iters": int(F["ctx_iters"]),
                           "margin": float(F["ctx_margin"]),
                           "mutual": bool(F["ctx_mutual"]),
                           "rival_tol": float(Ctx.RIVAL_TOL),
                           "max_rivals": int(Ctx.MAX_RIVALS)}
    bundle["beta"] = 0.5
    bundle["cand_cap"] = 6
    bundle["endpoint_lenses"] = list(place_lenses())
    bundle["provenance"] = {
        "fit": "m8_step2_fit.pkl (Mission 8 FIT, 16 scenes, 64 views)",
        "read": "m8_read.json credited P_plus_ctx on 16 fresh scenes",
        "note": "deployed scorer reads ONLY this file; nothing is fitted here",
    }
    cfg = json.load(open(os.path.join(OUT, "m8_preregistration.json")))
    want = cfg["frozen_configuration"]["plus"]
    for k, mine in (("a_exist", bundle["plus"]["a_exist"]),
                    ("a_geom", bundle["plus"]["a_geom"]),
                    ("slot_cap", bundle["constants"]["slot_cap"]),
                    ("llr_clip", bundle["constants"]["llr_clip"]),
                    ("fixed_tol", bundle["constants"]["fixed_tol"])):
        if abs(float(want[k]) - float(mine)) > 1e-12:
            raise SystemExit("REFUSED: %s differs from the pre-registered value "
                             "(%r vs %r)" % (k, mine, want[k]))
    with open(TARGET, "w") as fh:
        json.dump(bundle, fh, indent=1, sort_keys=True)
    print("wrote %s" % TARGET)
    print("  a_exist %.5f  a_geom %.5f  slot_cap %d  llr_clip %.2f  fixed_tol %.4f"
          % (bundle["plus"]["a_exist"], bundle["plus"]["a_geom"],
             bundle["constants"]["slot_cap"], bundle["constants"]["llr_clip"],
             bundle["constants"]["fixed_tol"]))
    print("  iteration iters %d margin %.3f mutual %s  beta %.2f  cand_cap %d"
          % (bundle["iteration"]["iters"], bundle["iteration"]["margin"],
             bundle["iteration"]["mutual"], bundle["beta"],
             bundle["cand_cap"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
