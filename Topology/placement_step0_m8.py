"""MISSION 8 STEP 0 -- free FIT reads: the oracle decomposition and a context audit.

(a) Report P_oracle_presence and P_ratio, and split the oracle ceiling into the
    part that PERFECT CONTEXT can buy (P_oracle - P_oracle_presence) and the
    part that perfect context cannot buy (P_oracle_presence - P_presence).  If
    the first is small the ceiling is INTRINSIC ambiguity, and no amount of
    context repair will close the gap; if the second is small the ceiling is
    context, and it will.

(b) Context audit on the SAME edges P uses: for every neighbour correspondence
    in the query payload, is it the one truth names?  Report the correct share
    and the residual distribution for right vs wrong context.

Everything runs on the FIT half.  FIT reruns are free.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def pose_fn(spec: Any, view: Dict[str, Any]):
    from Topology.placement_dev5 import posed5
    pv = {"kind": view.get("kind"),
          "rot_deg": float(view.get("rot_deg", 0.0)),
          "shift": tuple(view.get("shift", (0.0, 0.0))),
          "scale": float(view.get("scale", 1.0))}
    return posed5(spec, view), pv


def context_audit(specs: List[Any], views: List[Any]) -> Dict[str, Any]:
    """Is the neighbour context P is handed actually correct?"""
    import guard
    guard.activate()
    from matching import placement_edges as PE
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import node_objects, track_optimal

    n_ctx = n_right = 0
    res_right: List[float] = []
    res_wrong: List[float] = []
    per_family: Dict[str, List[float]] = defaultdict(list)
    for spec in specs:
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        for label, view in views:
            sc1, _pv = pose_fn(spec, view)
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            fwd = {int(r): int(q) for r, q
                   in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            q_pay = PE.build_edges(g1.subgraph, r1.image_u8,
                                   lens_map=PE.lens_map_of(g1))
            r_pay = PE.build_edges(g0.subgraph, r0.image_u8,
                                   lens_map=PE.lens_map_of(g0))
            for (u, v, et) in q_pay:
                if u not in fwd:
                    continue
                # the correspondence this edge OUGHT to have
                want = fwd[u]
                # does the reference payload hold the corresponding edge?
                r_edges = [w for (a, b, e) in r_pay if a == want
                           for w in (b,) if e == et]
                if not r_edges:
                    continue
                n_ctx += 1
                ok = int(v) in set(r_edges)
                n_right += 1 if ok else 0
                # residual in the source frame, right vs wrong context
                pv_q = q_pay[(u, v, et)]
                if ok:
                    pv_r = r_pay[(want, int(v), et)]
                    d = math.hypot(pv_q.distance * math.cos(pv_q.direction)
                                   - pv_r.distance * math.cos(pv_r.direction),
                                   pv_q.distance * math.sin(pv_q.direction)
                                   - pv_r.distance * math.sin(pv_r.direction))
                    res_right.append(float(d))
                else:
                    pv_r = r_pay[(want, int(r_edges[0]), et)]
                    d = math.hypot(pv_q.distance * math.cos(pv_q.direction)
                                   - pv_r.distance * math.cos(pv_r.direction),
                                   pv_q.distance * math.sin(pv_q.direction)
                                   - pv_r.distance * math.sin(pv_r.direction))
                    res_wrong.append(float(d))
                per_family[spec.family].append(1.0 if ok else 0.0)
    return {"n_context_edges": n_ctx,
            "correct_share": (n_right / n_ctx) if n_ctx else float("nan"),
            "wrong_share": (1.0 - n_right / n_ctx) if n_ctx else float("nan"),
            "res_right_median": float(np.median(res_right)) if res_right else float("nan"),
            "res_wrong_median": float(np.median(res_wrong)) if res_wrong else float("nan"),
            "by_family": {k: float(np.mean(v)) for k, v in per_family.items()}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--tag", default="m8_step0")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from core.place_key import place_lenses

    specs = all_specs5(per_family=args.per_family)
    fit_specs, _ = split5(specs)
    A2.ENDPOINT_LENSES = tuple(place_lenses())
    M3.ORACLE = True

    fit, pfit, meta, ratio_fit = M3.collect_fit(fit_specs, VIEWS5, None,
                                                pose_fn=pose_fn)
    recs = M3.run_check(fit_specs, VIEWS5, fit, pfit, None, ratio_fit,
                        pose_fn=pose_fn)
    print("STEP 0a -- FIT arms, places-only endpoints, %d scenes\n" % len(fit_specs))
    arms = {}
    for arm in ("N", "R", "R_fixed", "P", "P_presence", "P_shuffled", "W",
                "P_oracle", "P_oracle_presence", "P_ratio"):
        try:
            arms[arm] = M3.scene_stats(recs, arm)["mean"]
        except Exception:
            arms[arm] = float("nan")
        print("  %-20s %.4f" % (arm, arms[arm]))

    head_ctx = arms.get("P_oracle", float("nan")) - arms.get("P_oracle_presence", float("nan"))
    head_int = arms.get("P_oracle_presence", float("nan")) - arms.get("P_presence", float("nan"))
    print("\n  ORACLE DECOMPOSITION")
    print("    P_oracle - P_oracle_presence = %+0.4f   <- what PERFECT CONTEXT buys"
          % head_ctx)
    print("    P_oracle_presence - P_presence = %+0.4f <- what perfect context CANNOT buy"
          % head_int)

    # per-family oracle accuracy
    fam_rows = {}
    for fam in sorted({r.get("family") for r in recs}):
        sub = [r for r in recs if r.get("family") == fam]
        if not sub or "P_oracle" not in sub[0]["ok"]:
            continue
        fam_rows[fam] = {a: float(np.mean([r["ok"][a] for r in sub
                                           if a in r["ok"]]))
                         for a in ("N", "P", "P_presence", "P_oracle",
                                   "P_oracle_presence")}
    print("\n  per family / arm")
    print("    %-10s %8s %8s %10s %8s %16s" % ("family", "N", "P", "P_presence",
                                              "P_oracle", "P_oracle_presence"))
    for fam, row in fam_rows.items():
        print("    %-10s %8.4f %8.4f %10.4f %8.4f %16.4f"
              % (fam, row["N"], row["P"], row["P_presence"], row["P_oracle"],
                 row["P_oracle_presence"]))

    print("\nSTEP 0b -- context audit (is the neighbour context right?)\n")
    audit = context_audit(fit_specs, VIEWS5)
    print("  context edges: %d" % audit["n_context_edges"])
    print("  CORRECT share: %.3f   WRONG share: %.3f"
          % (audit["correct_share"], audit["wrong_share"]))
    print("  residual median  right context %.3f | wrong context %.3f"
          % (audit["res_right_median"], audit["res_wrong_median"]))
    print("  by family (correct share): %s"
          % {k: round(v, 3) for k, v in sorted(audit["by_family"].items())})
    print("\n  -> confident-context filter is %s before iteration"
          % ("MANDATORY" if audit["wrong_share"] > 0.5 else "optional"))

    res = {"arms": arms, "oracle_gap": {"context": head_ctx, "intrinsic": head_int},
           "by_family": fam_rows, "context_audit": audit}
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
