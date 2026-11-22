"""MISSION 7 STEP 3 -- the powered DEV-5 placement run.

32 scenes (8 per family) x 4 view steps, split by SCENE into FIT and CHECK
before anything is fitted, stratified by family.  The LLR scorer and the
presence rates are fitted on FIT only; every arm is then scored on CHECK.

Arms compared on the SAME graph, as required: N, R (frozen), R_fixed, P,
P_shuffled, W, P_presence, plus the DEV-only oracle-context arm.  Geometry is
credited only if P - P_presence AND P - N are both positive beyond the MDE,
scene-clustered.

CHECK looks are counted and printed.  This script is the one that spends them.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]


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
    ap.add_argument("--tag", default="m7_dev5")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5

    if args.no_oracle:
        M3.ORACLE = False

    specs = all_specs5(per_family=args.per_family)
    fit_specs, check_specs = split5(specs)
    print("DEV-5: %d scenes (%d families) -> FIT %d / CHECK %d, %d views each"
          % (len(specs), len({s.family for s in specs}), len(fit_specs),
             len(check_specs), len(VIEWS5)))
    print("FIT/CHECK split by scene, stratified by family, before any fitting.\n")

    fit, pfit, meta, ratio_fit = M3.collect_fit(fit_specs, VIEWS5, None,
                                                pose_fn=pose_fn)
    print("FIT: %d scenes, %d true / %d wrong residuals, mu_wrong %.4f"
          % (meta.get("scenes", 0), meta.get("n_res_true", 0),
             meta.get("n_res_wrong", 0), meta.get("mu_wrong", float("nan"))))

    records = M3.run_check(check_specs, VIEWS5, fit, pfit, None, ratio_fit,
                           pose_fn=pose_fn)
    print("\nCHECK LOOK 1 of 5 spent, correction 0.05/look -> 0.05\n")
    print("== CHECK, near-tie ambiguous nodes")
    res: Dict[str, Any] = {"fit": meta, "n_records": len(records),
                           "scenes": len({r["scene"] for r in records}),
                           "arms": {}, "contrasts": {}}
    arms_here = [a for a in list(M3.ARMS) + ["P_oracle", "P_oracle_presence",
                                             "P_ratio"]
                 if records and a in records[0]["ok"]]
    for arm in arms_here:
        s = M3.scene_stats(records, arm)
        res["arms"][arm] = s
        print("  %-12s %.4f  scene_se %.4f  n=%d scenes=%d"
              % (arm, s["mean"], s["scene_se"], s["n"], s["n_scenes"]))

    print("\n  contrast            effect   scene_se     MDE        CI           p_exact  verdict")
    pairs = [("P", "N"), ("P", "P_presence"), ("P", "P_shuffled"),
             ("P", "W"), ("P", "R_fixed"), ("R", "N"), ("R_fixed", "R"),
             ("P_presence", "N"), ("P_oracle", "P"), ("P_oracle", "P_presence")]
    for a, b in pairs:
        if a not in arms_here or b not in arms_here:
            continue
        c = M3.contrast(records, a, b)
        res["contrasts"]["%s-%s" % (a, b)] = c
        print("  %-18s %+.4f  %.4f    %.4f  [%+.4f,%+.4f]  %.4f  %s"
              % ("%s-%s" % (a, b), c["mean"], c["se_scene"], c["mde"],
                 c["ci"][0], c["ci"][1], c["p_exact"], M3.verdict(c)))

    # per-family breakdown
    print("\n  per family:  arm   mean")
    fams = sorted({r.get("family", "?") for r in records})
    res["by_family"] = {}
    for fam in fams:
        sub = [r for r in records if r.get("family") == fam]
        if not sub:
            continue
        row = {}
        for arm in ("N", "P", "P_presence"):
            if arm in sub[0]["ok"]:
                import numpy as np
                row[arm] = float(np.mean([r["ok"][arm] for r in sub]))
        res["by_family"][fam] = row
        print("  %-10s n=%3d  N %.4f  P %.4f  P_presence %.4f"
              % (fam, len(sub), row.get("N", float("nan")),
                 row.get("P", float("nan")), row.get("P_presence", float("nan"))))

    path = os.path.join(ROOT, "Topology", "results", "placement",
                        "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
