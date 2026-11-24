"""Diagnostic for the Step-5 read: is the TRUTH clean, or is the matcher failing?

Reports, per (photo, transform): the back-projection distance of every truth pair
(median, 90th pct), how many truth pairs are MUTUAL nearest neighbours under the
inverse/forward map, and -- among the pairs the matcher actually answered -- the
accuracy of the mapping restricted to truth pairs whose back-projection is
tighter than 0.5 px (a truth set that cannot be blamed).

Not a read: it uses the graphs the read already built and the same truth function.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
PROG = os.path.join(ROOT, "Topology", "results", "placement", "m12_read_real_progress.jsonl")


def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology import placement_m12_read_real as R

    reg = R._registration()
    imgs = R._images()
    rows = [json.loads(l) for l in open(PROG) if l.strip()]
    out: List[Dict[str, Any]] = []
    for r in rows:
        name, lab = r["photo"], r["transform"]
        spec = {t["label"]: t["spec"] for t in reg["transforms"]}[lab]
        kq, kr = "m12real|%s|%s" % (name, lab), "m12real|%s|orig" % name
        q, g = get_or_build(imgs[kq], kq), get_or_build(imgs[kr], kr)
        subq, subr = q.subgraph, g.subgraph
        h, w = imgs[kr].shape[:2]
        qn = [int(n) for n in sorted(subq.nodes) if n in subq.anchor_of_node]
        rn = [int(n) for n in sorted(subr.nodes) if n in subr.anchor_of_node]
        qa = np.asarray([subq.anchor_of_node[n] for n in qn], float)
        ra = np.asarray([subr.anchor_of_node[n] for n in rn], float)
        back = R._inverse(spec, qa, (h, w))
        truth = R._truth(q, g, spec, (h, w))
        dist = []
        mutual = 0
        for u, v in truth.items():
            i = qn.index(u)
            j = rn.index(v)
            d = float(np.hypot(*(back[i] - ra[j])))
            dist.append(d)
            # mutual: v's forward image must find u as its nearest query anchor
            fwd = R._inverse({"rot_deg": -spec["rot_deg"]} if "rot_deg" in spec else spec,
                             ra[j][None, :], (h, w))[0]
            if "rot_deg" in spec:
                # undo: forward map = apply the transform to the reference anchor
                th = np.deg2rad(float(spec["rot_deg"]))
                c = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
                dd = ra[j] - c
                fwd = (np.array([np.cos(th) * dd[0] - np.sin(th) * dd[1],
                                 np.sin(th) * dd[0] + np.cos(th) * dd[1]]) + c)
            dd = np.hypot(qa[:, 0] - fwd[0], qa[:, 1] - fwd[1])
            if int(np.argmin(dd)) == i:
                mutual += 1
        P.PLACEMENT_MATCHER_ENABLED = False
        res = joint_mcs(q.subgraph, g.subgraph, q.sigs, g.sigs)
        tight = [u for u, v in truth.items()
                 if np.hypot(*(R._inverse(spec, qa[qn.index(u)][None, :], (h, w))[0]
                               - ra[rn.index(v)])) < 0.5]
        hit = sum(1 for u in tight if res.mapping.get(u, -1) == truth[u])
        out.append({"photo": name, "transform": lab, "truth_n": len(truth),
                    "dist_median": float(np.median(dist)) if dist else float("nan"),
                    "dist_p90": float(np.percentile(dist, 90)) if dist else float("nan"),
                    "mutual_share": (mutual / len(truth)) if truth else float("nan"),
                    "tight_n": len(tight),
                    "tight_acc": (hit / len(tight)) if tight else float("nan")})
        print("%-16s %-9s truth %3d  d_med %.2f  d_p90 %.2f  mutual %.2f  tight %3d acc %.3f"
              % (name, lab, len(truth), out[-1]["dist_median"], out[-1]["dist_p90"],
                 out[-1]["mutual_share"], len(tight), out[-1]["tight_acc"]), flush=True)
    by: Dict[str, List[Dict[str, Any]]] = {}
    for o in out:
        by.setdefault(o["transform"], []).append(o)
    print()
    for t, sub in sorted(by.items()):
        print("%-9s d_med %.2f  mutual %.2f  tight_n %4d  tight_acc %.3f"
              % (t, float(np.median([s["dist_median"] for s in sub])),
                 float(np.median([s["mutual_share"] for s in sub])),
                 sum(s["tight_n"] for s in sub),
                 (sum(s["tight_acc"] * s["tight_n"] for s in sub)
                  / max(1, sum(s["tight_n"] for s in sub)))))
    with open(os.path.join(ROOT, "Topology", "results", "placement",
                           "m12_read_real_truthcheck.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
