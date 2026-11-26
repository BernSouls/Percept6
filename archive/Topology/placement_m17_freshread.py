"""MISSION 17 / O6 -- the fresh read, scored by the PRE-REGISTERED rule.

Primary contrast: full placement (frozen beta, w) vs appearance-only (w = 0) on TOP-1 at
15 deg, place endpoints, image-clustered.  Unit of analysis: the IMAGE.

    n = 20 images (16 new-seed synthetic scenes + 4 photographs M15/M16 never touched)
    MDE = (t_{0.975,19} + t_{0.80,19}) * sd_diff / sqrt(20) = 0.660 * sd_diff
    planning sd 0.0438 -> MDE 0.0289  (registered before any fresh graph existed)

Verdict rule: if MDE > the planning effect the read is an ESTIMATE, never a detection.
It is a detection only if the observed paired difference also clears the MDE.

    python Topology/placement_m17_freshread.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
SRC = os.path.join(OUT, "m17_fresh.json")
DST = os.path.join(OUT, "m17_freshread.json")
MDE_FACTOR = 0.660           # (t_.975,19 + t_.80,19) / sqrt(20)
PLANNING_SD = 0.0438
PLANNING_EFFECT = 0.42


def main() -> int:
    d = json.load(open(SRC))
    rows = [r for r in d["rows"] if r["pose"] == "rot15"]
    per: List[Dict[str, Any]] = []
    for r in rows:
        c = r["counts"]
        q = c["queries"]
        if not q:
            continue
        per.append({"tag": r["tag"], "queries": q,
                    "top1_full": c["top1"] / q, "top1_app": c["app_ok"] / q,
                    "diff": (c["top1"] - c["app_ok"]) / q,
                    "nonzero_diff": int(c["top1"] != c["app_ok"]),
                    "truth_pairs": r.get("truth_pairs"),
                    "appearance_is_the_ceiling": int(c["app_ok"] == q)})
    diffs = [p["diff"] for p in per]
    n = len(diffs)
    sd = st.stdev(diffs) if n > 1 else float("nan")
    mde = MDE_FACTOR * sd / math.sqrt(n) if n > 1 else float("nan")
    mean = st.mean(diffs)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    # two-sided 95 % interval on the mean per-image difference
    tcrit = _tcrit975(n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    powered = mde <= PLANNING_EFFECT
    nonvacuous = {
        "images": n,
        "images_with_a_decided_row": sum(1 for p in per if p["nonzero_diff"] > 0),
        "images_where_appearance_is_already_perfect":
            sum(1 for p in per if p["appearance_is_the_ceiling"]),
        "queries": sum(p["queries"] for p in per),
        "min_queries_per_image": min((p["queries"] for p in per), default=0),
    }
    res = {
        "contrast": "full placement (w = 0.375, frozen beta) vs appearance-only (w = 0) "
                    "on top-1 at 15 deg, ties fail",
        "arm": rows[0]["arm"] if rows else None,
        "table": rows[0].get("table") if rows else None,
        "images": per,
        "mean_diff": mean, "sd_diff": sd, "se": se, "t": t,
        "ci95": [lo, hi], "mde": mde, "planning_sd": PLANNING_SD,
        "planning_effect": PLANNING_EFFECT, "powered": powered,
        "non_vacuity": nonvacuous,
        "verdict": ("DETECTION: paired mean difference %.4f clears the MDE %.4f"
                    % (mean, mde)) if (powered and abs(mean) >= mde and mean > 0)
                   else ("ESTIMATE (not powered): MDE %.4f against a planning effect of "
                         "%.4f" % (mde, PLANNING_EFFECT)) if not powered
                   else ("ESTIMATE: observed %.4f does not clear the MDE %.4f"
                         % (mean, mde)),
        "units": "top-1 is a share of queries within an image; differences are shares",
    }
    from matching import m15_manifest as MF
    MF.write_result(DST, res)
    print("images %d, queries %d" % (n, nonvacuous["queries"]))
    for p in per:
        print("  %-14s q %4d full %.3f app %.3f  diff %+0.3f"
              % (p["tag"], p["queries"], p["top1_full"], p["top1_app"], p["diff"]))
    print("\nmean %+0.4f  sd %.4f  se %.4f  t %.2f  95%% CI [%+0.4f, %+0.4f]"
          % (mean, sd, se, t, lo, hi))
    print("MDE %.4f (planning sd %.4f) vs planning effect %.2f -> powered %s"
          % (mde, PLANNING_SD, PLANNING_EFFECT, powered))
    print("non-vacuity %s" % (nonvacuous,))
    print("VERDICT: %s" % res["verdict"])
    print("\nwrote %s" % DST)
    return 0


def _tcrit975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145, 15: 2.131,
             16: 2.120, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 30: 2.042,
             40: 2.021, 60: 2.000}
    if df in table:
        return table[df]
    return 1.96 if df > 60 else table[max(k for k in table if k <= df)]


if __name__ == "__main__":
    raise SystemExit(main())
