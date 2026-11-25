"""MISSION 18 / O3 -- what the candidate cap COSTS, measured in the mission's own harness.

The cap was the largest single change in the mission (top-1 at 15 deg 0.210 -> 0.434), and
it is four times the slot scoring of the deployed cap.  `placement_m12_perf.py` does not
take a cap, so the cost is measured here the same way it measures one: the same pairs, the
same `joint_mcs` call, arm C only (the placement path), cap 6 against cap 24.

    python Topology/placement_m18_capcost.py --pairs 2
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m18_capcost.json")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology import placement_m12_perf as PERF
    from Topology.placement_cache import get_or_build

    imgs = PERF._images()
    ps = PERF.pairs()[:args.pairs]
    rows: List[Dict[str, Any]] = []
    for (tag, qk, rk) in ps:
        q_img, r_img = imgs[qk], imgs[rk]
        q = get_or_build(q_img, "perf|" + qk)
        r = get_or_build(r_img, "perf|" + rk)
        rec: Dict[str, Any] = {"tag": tag, "q": qk, "r": rk, "sec": {}}
        for cap in (6, 24):
            P.PLACEMENT_CAND_CAP = cap
            P.PLACEMENT_MATCHER_ENABLED = True
            P.PLACEMENT_SCORE_EVIDENCE = True
            P.PLACEMENT_SCORE_WEIGHT = 0.375
            best = float("inf")
            for _ in range(args.repeats):
                t0 = time.time()
                res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                                placement_images=(q_img, r_img))
                best = min(best, time.time() - t0)
            rec["sec"]["cap%d" % cap] = best
            rec["judged_cap%d" % cap] = int(res.placement_judged)
        P.PLACEMENT_CAND_CAP = 0
        rows.append(rec)
        print("  %-14s cap6 %.3fs  cap24 %.3fs  ratio %.2f  judged %d -> %d"
              % (tag, rec["sec"]["cap6"], rec["sec"]["cap24"],
                 rec["sec"]["cap24"] / max(rec["sec"]["cap6"], 1e-9),
                 rec["judged_cap6"], rec["judged_cap24"]), flush=True)
    import statistics as st
    med6 = st.median([r["sec"]["cap6"] for r in rows])
    med24 = st.median([r["sec"]["cap24"] for r in rows])
    res_d = {"rows": rows, "median_cap6": med6, "median_cap24": med24,
             "ratio": med24 / med6, "repeats": args.repeats,
             "units": "seconds per placement call (best of %d), same cached graphs"
                      % args.repeats}
    with open(OUT, "w") as fh:
        json.dump(res_d, fh, indent=1, sort_keys=True)
    print("\nmedian cap6 %.3fs  cap24 %.3fs  ratio %.2fx" % (med6, med24, med24 / med6))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
