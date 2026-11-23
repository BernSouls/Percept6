"""MISSION 10 STEP 4b -- ETH-80 aggregation, both pair scores, both intervals.

Reads the checkpointed per-query rows (`m10_eth80_progress.jsonl`) and reports, for
200+ real-image queries over a K = 8 same-class gallery:

  * node_only        -- sum of the best per-node similarities (the shortlist score),
  * placement_sum    -- the PRE-REGISTERED pair score: sum over the mapped pairs of
                        the blended score the deployed matcher used,
  * placement_mean   -- the same sum divided by the number of mapped pairs, i.e. the
                        average quality of the correspondences rather than how many
                        there are,
  * placement_rate   -- divided by mapped x attached, for the coverage cross-check.

The pre-registered score is the sum.  The mean is reported beside it because Step 3
measured what a sum over correspondences actually is on this kind of data -- a
coverage statistic -- and that finding applies here unchanged; the mean is a
post-hoc reading and is labelled as one.

Interval: per-INSTANCE (scene-clustered) bootstrap over instances, 4000 draws, and
the paired delta is computed inside the instance before resampling.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PROGRESS = os.path.join(OUT, "m10_eth80_progress.jsonl")


def rank(sc: Dict[str, float], truth: str) -> int:
    order = sorted(sc.items(), key=lambda t: (-t[1], t[0]))
    names = [k for k, _ in order]
    return names.index(truth) + 1 if truth in names else 99


ARMS = {
    "node_only": lambda x: x["node_only"],
    "placement_sum": lambda x: x["placement"],
    "placement_mean": lambda x: {k: x["placement"][k] / max(1.0, x["meta"][k]["mapped"])
                                 for k in x["placement"]},
    "placement_rate": lambda x: {k: x["placement"][k] /
                                 max(1.0, x["meta"][k]["mapped"] * x["meta"][k]["attach"])
                                 for k in x["placement"]},
}


def main() -> int:
    rows: List[Dict[str, Any]] = [json.loads(l) for l in open(PROGRESS)]
    # One row per query: the checkpoint can hold a repeated query after a restart.
    seen, uniq = set(), []
    for x in rows:
        if x["query"] in seen:
            continue
        seen.add(x["query"])
        uniq.append(x)
    rows = uniq
    rng = np.random.default_rng(0)
    res: Dict[str, Any] = {"queries": len(rows), "arms": {}}
    for name, f in ARMS.items():
        t1 = np.array([1.0 if rank(f(x), x["truth"]) == 1 else 0.0 for x in rows])
        rr = np.array([1.0 / rank(f(x), x["truth"]) for x in rows])
        res["arms"][name] = {"top1": float(t1.mean()), "mrr": float(rr.mean())}
        by_cls: Dict[str, Any] = {}
        for cls in sorted({x["cls"] for x in rows}):
            sel = [x for x in rows if x["cls"] == cls]
            by_cls[cls] = {
                "n": len(sel),
                "top1": float(np.mean([1.0 if rank(f(x), x["truth"]) == 1 else 0.0
                                       for x in sel])),
                "mrr": float(np.mean([1.0 / rank(f(x), x["truth"]) for x in sel]))}
        res["arms"][name]["by_class"] = by_cls
    # scene-clustered paired deltas vs node_only
    inst = {}
    for x in rows:
        inst.setdefault(x["inst"], []).append(x)
    for cand in ("placement_sum", "placement_mean", "placement_rate"):
        f, g = ARMS["node_only"], ARMS[cand]
        d = np.array([np.mean([1.0 if rank(g(x), x["truth"]) == 1 else 0.0 for x in v])
                      - np.mean([1.0 if rank(f(x), x["truth"]) == 1 else 0.0 for x in v])
                      for v in inst.values()])
        bs = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(4000)]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        res["arms"][cand]["delta_vs_node_only"] = {
            "delta": float(d.mean()), "se": se, "mde": float(2.8 * se),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "instances": int(len(d))}
    M = [m for x in rows for m in x["meta"].values()]
    res["coverage"] = {
        "mapped_mean": float(np.mean([m["mapped"] for m in M])),
        "attach_mean": float(np.mean([m["attach"] for m in M])),
        "match_s_mean": float(np.mean([m["match_s"] for m in M])),
        "counters": {k: int(np.sum([m["counters"].get(k, 0) for m in M]))
                     for k in ("rescored", "raised", "fell_back", "skipped")},
        "nodes_moved_by_flag": int(np.sum([m["counters"].get("raised", 0) for m in M])),
    }
    path = os.path.join(OUT, "m10_eth80_agg.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    for name in ARMS:
        a = res["arms"][name]
        print("%-15s top1 %.3f  mrr %.3f%s" % (
            name, a["top1"], a["mrr"],
            "" if "delta_vs_node_only" not in a else
            "  delta %+.3f se %.3f mde %.3f ci [%+.3f,%+.3f]" % (
                a["delta_vs_node_only"]["delta"], a["delta_vs_node_only"]["se"],
                a["delta_vs_node_only"]["mde"], a["delta_vs_node_only"]["ci"][0],
                a["delta_vs_node_only"]["ci"][1])))
    print("coverage", json.dumps(res["coverage"], sort_keys=True))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
