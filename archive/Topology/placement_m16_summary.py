"""Turn `m16_graphs.json` into the acceptance table -- with the duplicate removed.

The graph harness appends one EQUIVALENCE build (the first cell, second pose, rebuilt
with the M16 flags off).  It has the same tag and pose as a real cell, so any mean
computed by grouping on (tag, pose) silently counts that cell twice.  That is exactly
the kind of quiet double-count this repository keeps being bitten by, so the exclusion
is done here, once, and the equivalence verdict is written down instead of implied.

    python Topology/placement_m16_summary.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from typing import Any, Dict, List

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
SRC = os.path.join(OUT, "m16_graphs.json")
DST = os.path.join(OUT, "m16_graphs_summary.json")
ARMS = ("before", "graded", "after")
POSES = ("rot00", "rot05", "rot15", "scale115")


def _m(xs: List[float]):
    xs = [x for x in xs if x is not None]
    return None if not xs else st.mean(xs)


def main() -> int:
    d = json.load(open(SRC))
    rows: List[Dict[str, Any]] = d["cells"]
    # the LAST row is the equivalence build (the harness appends it); drop it from
    # the acceptance cells and use it as the proof
    eq = rows[-1] if len(rows) > len(d["images"]) * len(POSES) else None
    cells = rows[:len(d["images"]) * len(POSES)] if eq else rows
    same = [c for c in cells if c["tag"] == eq["tag"] and c["pose"] == eq["pose"]][0]
    equiv = {
        "tag": eq["tag"], "pose": eq["pose"],
        "graph_key_digest_off": eq["graph_key_digest"],
        "graph_key_digest_on": same["graph_key_digest"],
        "digest_identical": eq["graph_key_digest"] == same["graph_key_digest"],
        "readings_identical": all(
            abs((eq[a]["geometry_auc"]["ctx_matcher"] or 0)
                - (same[a]["geometry_auc"]["ctx_matcher"] or 0)) < 1e-12
            and eq[a]["admissible"] == same[a]["admissible"]
            and abs((eq[a]["appearance_auc"] or 0) - (same[a]["appearance_auc"] or 0)) < 1e-12
            for a in ARMS),
        "note": "the graphs do not depend on the M16 flags; the arms are read off ONE "
                "build by re-registering the constitution, and this proves it",
    }
    table: Dict[str, Dict[str, Any]] = {}
    for pose in POSES:
        rr = [c for c in cells if c["pose"] == pose]
        if not rr:
            continue
        pose_row: Dict[str, Any] = {"images": len(rr),
                                    "truth_pairs": sum(r["truth_pairs"] for r in rr)}
        for arm in ARMS:
            v = [r[arm] for r in rr]
            pose_row[arm] = {
                "admissible": sum(x["admissible"] for x in v),
                "truth": sum(x["truth"] for x in v),
                "admissible_share": (sum(x["admissible"] for x in v)
                                     / float(sum(x["truth"] for x in v) or 1)),
                "geo_matcher": _m([x["geometry_auc"]["ctx_matcher"] for x in v]),
                "geo_matcher_min": min(x["geometry_auc"]["ctx_matcher"] for x in v),
                "geo_matcher_max": max(x["geometry_auc"]["ctx_matcher"] for x in v),
                "geo_truth": _m([x["geometry_auc"]["ctx_truth"] for x in v]),
                "appearance": _m([x["appearance_auc"] for x in v]),
                "appearance_min": min(x["appearance_auc"] for x in v),
                "appearance_max": max(x["appearance_auc"] for x in v),
                "top1": _m([x["appearance_top1"] for x in v]),
                "queries": int(_m([x["queries"] for x in v]) or 0),
            }
        for arm in ("before", "after"):
            pass
        pose_row["geo_gap_before"] = (pose_row["before"]["geo_truth"]
                                      - pose_row["before"]["geo_matcher"])
        pose_row["geo_gap_after"] = (pose_row["after"]["geo_truth"]
                                     - pose_row["after"]["geo_matcher"])
        pose_row["geo_gap_graded"] = (pose_row["graded"]["geo_truth"]
                                      - pose_row["graded"]["geo_matcher"])
        table[pose] = pose_row
    res = {"table": table, "equivalence": equiv, "arms": list(ARMS),
           "images": d["images"], "flags": d["flags"],
           "manifest": d["manifest"],
           "units": "AUC unitless; admissible in truth pairs"}
    from matching import m15_manifest as MF
    MF.write_result(DST, res)
    print(json.dumps(equiv, indent=1))
    for pose in POSES:
        t = table.get(pose)
        if not t:
            continue
        print("\n%-8s truth %4d  n=%d" % (pose, t["truth_pairs"], t["images"]))
        for arm in ARMS:
            r = t[arm]
            print("  %-7s adm %5.1f%%  geo %.3f [%.3f-%.3f]  ceil %.3f  gap %.3f  app %.3f  top1 %.3f  q %d"
                  % (arm, 100 * r["admissible_share"], r["geo_matcher"],
                     r["geo_matcher_min"], r["geo_matcher_max"], r["geo_truth"],
                     t["geo_gap_" + arm], r["appearance"], r["top1"], r["queries"]))
    print("\nwrote %s" % DST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
