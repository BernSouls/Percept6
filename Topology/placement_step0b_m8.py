"""MISSION 8 STEP 0b (corrected) -- is the neighbour context N supplies right?

The first version of this audit compared a QUERY node id against REFERENCE node
ids, which is only meaningful where the two graphs happen to number their nodes
the same way; it reported a 0.027 "correct share" that was really an id-agreement
rate.  This version asks the question the scorer actually asks:

    for the candidate N picks for u (mu = N[u]), and for every neighbour `other`
    of u that N also maps (mo = N[other]), does the reference payload hold the
    edge between mu and mo?

and scores that correspondence against truth:

    correct  <=>  mu == truth[u]  and  mo == truth[other]

Only edges that EXIST under N's mapping are counted (the ones P can use), and
the residual distribution is reported for the correct ones against the wrong
ones, because a wrong neighbour image is worse than no neighbour image at all.

Truth is used only to label; nothing here feeds a matcher.
"""
from __future__ import annotations

import json
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


def audit(specs: List[Any], views: List[Any],
          limit: int = None) -> Dict[str, Any]:
    from matching import placement_llr as L
    from matching.placement_scorer import payload_residual
    from Topology import placement_m3 as M3

    n_ctx = n_right = 0
    res_right: List[float] = []
    res_wrong: List[float] = []
    per_family: Dict[str, List[float]] = defaultdict(list)
    per_lens: Dict[str, List[float]] = defaultdict(list)
    n_map_right = n_map_total = 0
    for vw in M3.iter_views(specs, views, limit, pose_fn):
        q_inc, q_idx, r_idx = vw["q_inc"], vw["q_idx"], vw["r_idx"]
        truth, nmap, pop = vw["truth"], vw["nmap"], vw["pop"]
        from Topology.placement_arms2 import lens_of
        g1 = vw["g1"]
        for u in pop:
            mu = nmap.get(u)
            if mu is None:
                continue
            n_map_total += 1
            if int(mu) == int(truth.get(u, -1)):
                n_map_right += 1
            for (other, is_source, name) in sorted(q_inc.get(int(u), ())):
                mo = nmap.get(int(other))
                if mo is None:
                    continue
                if is_source:
                    qv = q_idx.get((int(u), int(other), name))
                    rv = r_idx.get((int(mu), int(mo), name))
                else:
                    qv = q_idx.get((int(other), int(u), name))
                    rv = r_idx.get((int(mo), int(mu), name))
                if qv is None or rv is None:
                    continue          # no edge under N's mapping: not usable
                n_ctx += 1
                ok = (int(mo) == int(truth.get(int(other), -1))
                      and int(mu) == int(truth.get(int(u), -1)))
                n_right += 1 if ok else 0
                d = float(abs(payload_residual(qv, rv, L.FIXED_TOL)))
                (res_right if ok else res_wrong).append(d)
                per_family[vw["spec"].family].append(1.0 if ok else 0.0)
                per_lens[str(lens_of(g1, int(u)))].append(1.0 if ok else 0.0)
    return {
        "n_context_edges": n_ctx,
        "correct_share": (n_right / n_ctx) if n_ctx else float("nan"),
        "wrong_share": (1.0 - n_right / n_ctx) if n_ctx else float("nan"),
        "map_correct_share": (n_map_right / n_map_total) if n_map_total
        else float("nan"),
        "n_mapped_nodes": n_map_total,
        "res_right_median": float(np.median(res_right)) if res_right
        else float("nan"),
        "res_wrong_median": float(np.median(res_wrong)) if res_wrong
        else float("nan"),
        "res_right_mean": float(np.mean(res_right)) if res_right
        else float("nan"),
        "res_wrong_mean": float(np.mean(res_wrong)) if res_wrong
        else float("nan"),
        "by_family": {k: float(np.mean(v)) for k, v in per_family.items()},
        "by_lens": {k: {"n": len(v), "correct": float(np.mean(v))}
                    for k, v in sorted(per_lens.items())},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--tag", default="m8_step0b")
    args = ap.parse_args()

    import guard
    guard.activate()
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from core.place_key import place_lenses

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    specs = all_specs5(per_family=args.per_family)
    fit_specs, _ = split5(specs)
    a = audit(fit_specs, VIEWS5)
    print("STEP 0b -- context audit over the edges P actually uses")
    print("  mapped nodes: %d   N's mapping correct on %.3f"
          % (a["n_mapped_nodes"], a["map_correct_share"]))
    print("  usable neighbour edges under N's mapping: %d" % a["n_context_edges"])
    print("  CORRECT correspondence share: %.3f   WRONG: %.3f"
          % (a["correct_share"], a["wrong_share"]))
    print("  residual |.| median  right %.3f  wrong %.3f   (mean %.3f / %.3f)"
          % (a["res_right_median"], a["res_wrong_median"],
             a["res_right_mean"], a["res_wrong_mean"]))
    print("  by family: %s" % {k: round(v, 3)
                               for k, v in sorted(a["by_family"].items())})
    print("  -> confident-context filter is %s"
          % ("MANDATORY" if a["wrong_share"] > 0.5 else "optional"))
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(a, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
