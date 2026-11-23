"""MISSION 8 STEP 4 -- the ONE fresh read of the pre-registered contrasts.

The confirmation set, the arms, the two primary contrasts, the correction and
the decision rule were all committed in m8_preregistration.json BEFORE any graph
of this set existed.  This script only reads them back, scores the set once and
prints the verdict.  It does not choose anything.

Nothing here refits: the LLR densities, the presence rates and the P_plus
coefficients are loaded from the pickle the FIT produced, and every threshold is
asserted against the frozen record before a single node is scored.
"""
from __future__ import annotations

import json
import os
import pickle
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


def bands(recs: List[Dict[str, Any]], arms: List[str],
          edges: List[float]) -> Dict[str, Any]:
    """Accuracy per similarity band: where in the ambiguity does what help?"""
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sub = [r for r in recs if r["bucket"] == "ambiguous"
               and lo <= float(r["top_sim"]) < hi]
        if not sub:
            continue
        out["%.2f-%.2f" % (lo, hi)] = {
            "n": len(sub),
            **{a: float(np.mean([r["ok"][a] for r in sub])) for a in arms}}
    return out


def flips(recs: List[Dict[str, Any]], a: str, b: str) -> Dict[str, int]:
    up = down = 0
    for r in recs:
        if r["bucket"] != "ambiguous":
            continue
        x, y = bool(r["ok"][a]), bool(r["ok"][b])
        if x and not y:
            up += 1
        elif y and not x:
            down += 1
    return {"gain": up, "loss": down, "net": up - down}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", default="m8_preregistration")
    ap.add_argument("--tag", default="m8_read")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from matching import placement_context as Ctx
    from matching import placement_llr as L
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import VIEWS5
    from core.place_key import place_lenses

    prereg = json.load(open(os.path.join(OUT, "%s.json" % args.prereg)))
    with open(os.path.join(OUT, "_fitcache", "m8_step2_fit.pkl"), "rb") as fh:
        F = pickle.load(fh)
    fit, pfit = F["fit"], F["pfit"]
    M3.A_E, M3.A_G = float(F["a_exist"]), float(F["a_geom"])
    M3.CTX_MARGIN = float(F["ctx_margin"])
    M3.CTX_ITERS = int(F["ctx_iters"])
    M3.CTX_MUTUAL = bool(F["ctx_mutual"])
    A2.ENDPOINT_LENSES = tuple(place_lenses())
    M3.ORACLE = True

    # The frozen record and the live code must agree, or the read is void.
    cfg = prereg["frozen_configuration"]
    checks = {
        "a_exist": abs(M3.A_E - float(cfg["plus"]["a_exist"])) < 1e-12,
        "a_geom": abs(M3.A_G - float(cfg["plus"]["a_geom"])) < 1e-12,
        "ctx_iters": M3.CTX_ITERS == int(cfg["context"]["iters"]),
        "ctx_margin": abs(M3.CTX_MARGIN - float(cfg["context"]["margin"])) < 1e-12,
        "ctx_mutual": bool(M3.CTX_MUTUAL) == bool(cfg["context"]["mutual"]),
        "edge_beta": abs(M3.EDGE_BETA - float(cfg["scorer"]["edge_beta"])) < 1e-12,
        "slot_cap": L.SLOT_CAP == int(cfg["plus"]["slot_cap"]),
        "llr_clip": abs(L.LLR_CLIP - float(cfg["plus"]["llr_clip"])) < 1e-12,
        "fixed_tol": abs(L.FIXED_TOL - float(cfg["plus"]["fixed_tol"])) < 1e-12,
        "endpoints": list(place_lenses()) == list(cfg["endpoint_whitelist"]),
    }
    if not all(checks.values()):
        print("REFUSED: frozen configuration does not match the live code:",
              {k: v for k, v in checks.items() if not v})
        return 2

    from Topology.placement_dev5 import all_specs5
    names = [s["name"] for s in prereg["confirmation_set"]["scenes"]]
    want = set(names)
    specs = [s for s in all_specs5(per_family=12)
             if s.name in want]
    if len(specs) != len(names):
        print("REFUSED: %d of %d confirmation scenes not generated"
              % (len(names) - len(specs), len(names)))
        return 2
    print("MISSION 8 FRESH READ -- %d scenes, %d graphs, one look\n"
          % (len(specs), len(specs) * (1 + len(VIEWS5))), flush=True)

    recs = M3.run_check(specs, VIEWS5, fit, pfit, None, F.get("ratio_fit"),
                        pose_fn=pose_fn)

    arms = ["N", "P", "P_presence", "P_plus", "P_plus_presence",
            "P_plus_ctx", "P_plus_ctx_presence", "P_shuffled", "P_oracle",
            "P_oracle_presence"]
    table = {}
    print("ARMS (ambiguous bank, scene-clustered)")
    for a in arms:
        try:
            s = M3.scene_stats(recs, a)
            table[a] = s["mean"]
            print("  %-18s %.4f   (n %d, scenes %d)"
                  % (a, s["mean"], s["n"], s["n_scenes"]))
        except Exception:
            table[a] = float("nan")
            print("  %-18s n/a" % a)

    prim = prereg["primary_contrasts"]
    sec = prereg["secondary_contrasts"]
    print("\nPRIMARY CONTRASTS (scene-clustered)")
    verdicts = {}
    for k in prim:
        a, b = [t.strip() for t in k.split("-", 1)]
        c = M3.contrast(recs, a, b)
        p_corr = min(1.0, 2.0 * float(c["p_exact"]))
        mde = prereg["mde"][k]["mde"]
        ok = (c["mean"] > mde) and (p_corr <= 0.025)
        verdicts[k] = {"contrast": c, "p_corrected": p_corr, "mde": mde,
                       "clears": bool(ok)}
        print("  %-24s %+0.4f  se %.4f  ci [%+0.4f,%+0.4f]  p %.4f (x2 %.4f)"
              "  MDE %.4f  %s"
              % (k, c["mean"], c["se_scene"], c["ci"][0], c["ci"][1],
                 c["p_exact"], p_corr, mde, "CLEARS" if ok else "under MDE"))

    print("\nSECONDARY CONTRASTS (scene-clustered, uncorrected)")
    secondary = {}
    for k in sec:
        a, b = [t.strip() for t in k.split("-", 1)]
        c = M3.contrast(recs, a, b)
        secondary[k] = c
        print("  %-30s %+0.4f  se %.4f  mde %.4f  p %.4f"
              % (k, c["mean"], c["se_scene"], c["mde"], c["p_exact"]))

    print("\nPER FAMILY (N / P / P_plus / P_plus_ctx / P_oracle)")
    fam = {}
    for f in sorted({r["family"] for r in recs}):
        sub = [r for r in recs if r["family"] == f and r["bucket"] == "ambiguous"]
        if not sub:
            continue
        fam[f] = {a: float(np.mean([r["ok"][a] for r in sub]))
                  for a in arms if a in sub[0]["ok"]}
        fam[f]["n"] = len(sub)
        print("  %-10s %.4f  %.4f  %.4f  %.4f  %.4f  (n %d)"
              % (f, fam[f].get("N", np.nan), fam[f].get("P", np.nan),
                 fam[f].get("P_plus", np.nan), fam[f].get("P_plus_ctx", np.nan),
                 fam[f].get("P_oracle", np.nan), fam[f]["n"]))

    print("\nBY VIEW LABEL (P_plus_ctx - P_plus)")
    byview = {}
    for lab in sorted({r["view"] for r in recs}):
        sub = [r for r in recs if r["view"] == lab and r["bucket"] == "ambiguous"]
        if not sub:
            continue
        byview[lab] = {a: float(np.mean([r["ok"][a] for r in sub]))
                       for a in ("N", "P_plus", "P_plus_ctx")}
        byview[lab]["n"] = len(sub)
        print("  %-10s N %.4f  P_plus %.4f  P_plus_ctx %.4f  (n %d)"
              % (lab, byview[lab]["N"], byview[lab]["P_plus"],
                 byview[lab]["P_plus_ctx"], byview[lab]["n"]))

    band = bands(recs, ["N", "P", "P_presence", "P_plus", "P_plus_ctx"],
                 [0.0, 0.4, 0.55, 0.7, 0.85, 1.01])
    print("\nSIMILARITY BANDS (top_sim)")
    for k, v in band.items():
        print("  %-10s n %4d  N %.3f  P %.3f  P_plus %.3f  ctx %.3f"
              % (k, v["n"], v["N"], v["P"], v["P_plus"], v["P_plus_ctx"]))

    fl = {"P_plus_ctx vs P_plus": flips(recs, "P_plus_ctx", "P_plus"),
          "P_plus_ctx vs N": flips(recs, "P_plus_ctx", "N"),
          "P_plus vs P_plus_presence": flips(recs, "P_plus",
                                             "P_plus_presence")}
    print("\nNET FLIPS (ambiguous nodes)")
    for k, v in fl.items():
        print("  %-28s +%d / -%d  net %+d" % (k, v["gain"], v["loss"], v["net"]))

    una = {}
    for a in ("N", "P_plus", "P_plus_ctx"):
        una[a] = M3.scene_stats(recs, a, bucket="unambiguous")["mean"]
    print("\nUNAMBIGUOUS BANK  N %.4f  P_plus %.4f  P_plus_ctx %.4f"
          % (una["N"], una["P_plus"], una["P_plus_ctx"]))
    c_una = M3.contrast(recs, "P_plus_ctx", "N", bucket="unambiguous")
    print("  P_plus_ctx - N on unambiguous: %+0.4f (mde %.4f)"
          % (c_una["mean"], c_una.get("mde", np.nan)))

    ctx = {"n_views": len(M3.CTX_STATS),
           "ctx_acc_before": float(np.mean([s["n_acc"] for s in M3.CTX_STATS]))
           if M3.CTX_STATS else float("nan"),
           "ctx_acc_after": float(np.mean([s["ctx_acc"] for s in M3.CTX_STATS]))
           if M3.CTX_STATS else float("nan"),
           "views_hurt": int(sum(1 for s in M3.CTX_STATS
                                 if s["ctx_acc"] < s["n_acc"])),
           "moves": int(sum(s["moved"] for s in M3.CTX_STATS)),
           "refused": int(sum(s["refused"] for s in M3.CTX_STATS))}
    print("\nCONTEXT ITERATION on this set: %.4f -> %.4f (views worse %d)"
          % (ctx["ctx_acc_before"], ctx["ctx_acc_after"], ctx["views_hurt"]))

    clears = [k for k, v in verdicts.items() if v["clears"]]
    if len(clears) == len(prim):
        verdict = "CREDITED"
    elif clears:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"
    print("\nVERDICT: %s" % verdict)
    for k, v in verdicts.items():
        print("  %-24s %+0.4f vs MDE %.4f -> %s"
              % (k, v["contrast"]["mean"], v["mde"],
                 "beyond MDE" if v["contrast"]["mean"] > v["mde"] else
                 "under MDE"))

    res = {"arms": table, "primary": verdicts, "secondary": secondary,
           "by_family": fam, "by_view": byview, "bands": band, "flips": fl,
           "unambiguous": una, "unambiguous_contrast": c_una,
           "context": ctx, "verdict": verdict,
           "n_scenes": len(specs), "n_nodes": len(recs),
           "checks": checks}
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
