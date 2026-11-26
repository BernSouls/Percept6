"""MISSION 9 STEP 2 -- validity and premise checks, BEFORE any fitting or read.

Both new families have to prove they can answer the question before they are
asked it.  Nothing here scores a deployed arm; nothing here is looked at twice.

IRREGULAR (validity, 2a):
  * how many nodes are in the near-tie band at all (the ambiguous population),
  * the share, and whether that share is big enough that a planning-size effect
    clears the MDE the confirmation set would have,
  * the ORACLE CEILING: what perfect context is worth here.  The mission's bar is
    (oracle - N) >= 0.05, and a family below it is repaired (texture, layout,
    scene count) rather than reported as a null.

OUT-OF-PLANE (premise, 2b):
  * the TRUE source-frame residual per rotation step: does the 2-D source-frame
    model still describe the change at 20 degrees about y?
  * a per-node paired test of the direction/frame provider: the same node's
    residual at in-plane vs out-of-plane steps,
  * VISIBILITY / CHECKABILITY: the share of true pairs whose payload edge exists
    on both sides under truth, per step -- the rule the oracle arms count by,
  * the per-edge tolerance as a function of step size, fitted on FIT.

The frozen Mission-8 densities are used for the oracle arms; that is stated
rather than glossed: the ceiling here is the ceiling the DEPLOYED scorer would
see, not a fresh fit on the family being judged.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def _load():
    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching.placement_scorer import payload_residual
    from Topology import placement_arms2 as A2
    from Topology import placement_families_m9 as FM
    from Topology.placement_dev5 import VIEWS5
    from Topology.placement_dev5 import posed5
    from core.place_key import place_lenses
    A2.ENDPOINT_LENSES = tuple(place_lenses())
    fit, pfit = PM.fit_objects()
    return M3, L, PM, payload_residual, A2, FM, VIEWS5, posed5, fit, pfit


def ambiguity_and_oracle(M3, L, A2, fit, pfit, specs, views, pose_fn,
                         tag: str) -> Dict[str, Any]:
    """Near-tie population, the arms, and the oracle ceiling -- on FIT."""
    n_nodes = n_amb = 0
    res_true: List[float] = []
    ok: Dict[str, List[float]] = defaultdict(list)
    a_e, a_g = 0.40625, 0.84375     # the frozen M8 coefficients
    from matching import placement_matcher as PM
    b = PM.frozen()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    for vw in M3.iter_views(specs, views, None, pose_fn):
        rows, truth, nmap = vw["rows"], vw["truth"], vw["nmap"]
        q_inc, q_idx, r_idx = vw["q_inc"], vw["q_idx"], vw["r_idx"]
        for u in vw["pop"]:
            row = rows.get(u) or []
            if not row:
                continue
            n_nodes += 1
            amb = (len(row) > 1 and (row[0][1] - row[1][1]) < A2.AMBIGUOUS_MARGIN)
            n_amb += 1 if amb else 0
            cand = row[:M3.CAND_CAP]
            tv = truth.get(u)

            def pick(mapping, geom=True):
                best_v, best_s = -1, -float("inf")
                for (v, s) in cand:
                    term = L.plus_score(u, int(v), mapping, q_inc, q_idx, r_idx,
                                        fit, pfit, a_e,
                                        (a_g if geom else 0.0))
                    val = (1.0 - M3.EDGE_BETA) * float(s) + M3.EDGE_BETA * term
                    if val > best_s:
                        best_s, best_v = val, int(v)
                return best_v

            for arm, mapping, geom in (("N", nmap, True),
                                       ("P_plus", nmap, True),
                                       ("P_plus_presence", nmap, False),
                                       ("P_oracle", truth, True),
                                       ("P_oracle_presence", truth, False)):
                key = arm + ("|amb" if amb else "|un")
                got = pick(mapping, geom)
                ok[key].append(1.0 if got == int(tv) else 0.0)
                if arm == "N":
                    ok["N" + ("|amb" if amb else "|un")] = ok[key]
    mean = {k: float(np.mean(v)) for k, v in ok.items() if v}
    def at(arm: str) -> float:
        vs = [mean.get(arm + "|amb", float("nan")), mean.get(arm + "|un", float("nan"))]
        return float(np.nanmean(vs))
    out = {"nodes": n_nodes, "ambiguous": n_amb,
           "ambiguous_share": (n_amb / n_nodes) if n_nodes else float("nan"),
           "arms_ambiguous": {a: mean.get(a + "|amb", float("nan"))
                              for a in ("N", "P_plus", "P_plus_presence",
                                        "P_oracle", "P_oracle_presence")},
           "arms_overall": {a: at(a) for a in ("N", "P_plus", "P_plus_presence",
                                              "P_oracle", "P_oracle_presence")}}
    out["headroom_oracle_minus_N"] = out["arms_overall"]["P_oracle"] - out["arms_overall"]["N"]
    out["headroom_amb"] = (out["arms_ambiguous"]["P_oracle"]
                           - out["arms_ambiguous"]["N"])
    out["valid"] = bool(out["headroom_oracle_minus_N"] >= 0.05 and n_amb >= 200)
    return out


def oop_premise(M3, L, payload_residual, specs, views, pose_fn) -> Dict[str, Any]:
    """Does the in-plane source-frame model still track out of plane?"""
    per_step: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for vw in M3.iter_views(specs, views, None, pose_fn):
        truth, q_inc, q_idx, r_idx = vw["truth"], vw["q_inc"], vw["q_idx"], vw["r_idx"]
        g1 = vw["g1"]
        for u in vw["pop"]:
            tv = truth.get(u)
            if tv is None:
                continue
            a = g1.subgraph.anchor_of_node.get(int(tv))
            if a is None:
                continue
            n_edges = 0
            n_checked = 0
            for (other, is_source, name) in q_inc.get(int(u), ()):
                n_edges += 1
                mo = truth.get(int(other))
                if mo is None:
                    continue
                # under TRUTH: does the reference payload hold this edge?
                if is_source:
                    qv = q_idx.get((int(u), int(other), name))
                    rv = r_idx.get((int(tv), int(mo), name))
                else:
                    qv = q_idx.get((int(other), int(u), name))
                    rv = r_idx.get((int(mo), int(tv), name))
                if qv is None or rv is None:
                    continue
                n_checked += 1
                per_step[vw["label"]]["res"].append(
                    float(abs(payload_residual(qv, rv, L.FIXED_TOL))))
            if n_edges:
                per_step[vw["label"]]["check_share"].append(n_checked / n_edges)
        # renderer-truth visibility: a node is countable iff it has a truth
        # partner AND that partner is visible in the posed image
        vis = sum(1 for u in vw["pop"] if truth.get(u) is not None)
        per_step[vw["label"]]["pop"].append(float(vis))
    out: Dict[str, Any] = {}
    for label, d in sorted(per_step.items()):
        res = np.asarray(d["res"], dtype=float)
        out[label] = {
            "n_true_edges": int(res.size),
            "res_median": float(np.median(res)) if res.size else float("nan"),
            "res_p90": float(np.percentile(res, 90)) if res.size else float("nan"),
            "checkable_share": float(np.mean(d["check_share"])) if d["check_share"] else float("nan"),
            "inside_fixed_tol": float(np.mean(res <= L.FIXED_TOL)) if res.size else float("nan"),
        }
    # tolerance as a function of step size, fitted on FIT: the ratio of the
    # step's residual median to the in-plane baseline.
    base = [out[k]["res_median"] for k in out if k.endswith("y5") or k.endswith("x5")]
    b0 = float(np.nanmean(base)) if base else float("nan")
    for k in out:
        out[k]["ratio_to_5deg"] = (out[k]["res_median"] / b0) if b0 and np.isfinite(b0) else float("nan")
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", type=int, default=12)
    ap.add_argument("--oop", type=int, default=4)
    ap.add_argument("--tag", default="m9_checks")
    ap.add_argument("--only", choices=("ir", "oop", "both"), default="both")
    args = ap.parse_args()

    (M3, L, PM, payload_residual, A2, FM, VIEWS5, posed5,
     fit, pfit) = _load()

    ir = FM.all_specs_irregular(args.ir)
    ir_fit, ir_check = FM.split_scenes(ir)
    oop = FM.all_specs_oop(args.oop)
    oop_fit, oop_check = FM.split_scenes(oop)
    print("FIT/CHECK by scene: irregular %d/%d, out-of-plane %d/%d\n"
          % (len(ir_fit), len(ir_check), len(oop_fit), len(oop_check)), flush=True)

    v_ir, p_oop, v_oop = None, None, None
    if args.only in ("ir", "both"):
        print("2a -- IRREGULAR validity (FIT only)", flush=True)
        v_ir = ambiguity_and_oracle(M3, L, A2, fit, pfit, ir_fit,
                                    FM.VIEWS_IRREGULAR, FM.pose_fn, "ir")
        print("  nodes %d  ambiguous %d (%.3f)"
              % (v_ir["nodes"], v_ir["ambiguous"], v_ir["ambiguous_share"]))
        for a, x in v_ir["arms_overall"].items():
            print("    %-20s %.4f  (ambiguous %.4f)"
                  % (a, x, v_ir["arms_ambiguous"][a]))
        print("  headroom (oracle - N): overall %+0.4f  ambiguous %+0.4f  -> %s"
              % (v_ir["headroom_oracle_minus_N"], v_ir["headroom_amb"],
                 "VALID" if v_ir["valid"] else "NOT VALID"), flush=True)

    if args.only in ("oop", "both"):
        print("\n2b -- OUT-OF-PLANE premise (FIT only)", flush=True)
        p_oop = oop_premise(M3, L, payload_residual, oop_fit, FM.VIEWS_OOP,
                            FM.pose_fn)
        print("  %-8s %8s %10s %10s %10s %10s"
              % ("step", "n edges", "res med", "res p90", "checkable", "in tol"))
        for k, d in p_oop.items():
            print("  %-8s %8d %10.3f %10.3f %10.3f %10.3f"
                  % (k, d["n_true_edges"], d["res_median"], d["res_p90"],
                     d["checkable_share"], d["inside_fixed_tol"]))
        print("  ratio of residual median to the 5-degree step:")
        for k, d in p_oop.items():
            print("    %-8s %.2f" % (k, d["ratio_to_5deg"]), flush=True)
        v_oop = ambiguity_and_oracle(M3, L, A2, fit, pfit, oop_fit,
                                     FM.VIEWS_OOP, FM.pose_fn, "oop")
        print("\n  OUT-OF-PLANE arms (FIT), overall and ambiguous:")
        for a, x in v_oop["arms_overall"].items():
            print("    %-20s %.4f  (ambiguous %.4f)"
                  % (a, x, v_oop["arms_ambiguous"][a]))
        print("  headroom (oracle - N): overall %+0.4f  ambiguous %+0.4f  -> %s"
              % (v_oop["headroom_oracle_minus_N"], v_oop["headroom_amb"],
                 "VALID" if v_oop["valid"] else "NOT VALID"))

    res = {"irregular": v_ir, "oop_arms": v_oop, "oop_premise": p_oop,
           "sizes": {"ir_fit": len(ir_fit), "ir_check": len(ir_check),
                     "oop_fit": len(oop_fit), "oop_check": len(oop_check)},
           "note": ("frozen Mission-8 densities are used for the oracle arms; no "
                    "fitting happens on either family here")}
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
