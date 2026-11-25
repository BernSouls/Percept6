"""MISSION 20 / item 8 -- WHY a query failed, from the rows already on disk.

Four tags, applied in this order so the counts are disjoint and reproducible.  A query is a
failure when the shipped arm (S0) does not put the positive first.

  SHORTLIST   the positive was never scored -- the stage-1 filter dropped it.  Nothing
              downstream can be blamed, and the recall@S of the run says how often.
  CORRESPONDENCE  the true pair mapped or judged materially FEWER places than the winner
              (less than 80 % of the winner's mapped count): the matcher did not find the
              places, so no aggregate could have ranked it.
  APPEARANCE  the true pair's shipped score is below the winner's while its mapped and
              judged counts are comparable: the evidence that existed did not favour it.
  AGGREGATION the true pair has the HIGHER shipped score of the two but a lower composite --
              the pieces were there and the arithmetic put them together wrongly.

T0 is the aggregation control: the query is the reference image itself, rotated 15 deg, so
correspondence and appearance are identical between the truth and any impostor of the same
object -- a T0 failure is aggregation by construction.  Counts are reported per task and per
class, with n, and never pooled across splits.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def attribute(tag: str, split: str, per_class: int, task: str) -> Dict[str, Any]:
    from Topology import placement_m20_eth80 as H
    from Topology.placement_m20_fit import objects_for
    objs = objects_for(split, per_class)
    key_of = {o["ref"]: o for o in objs}
    q_of = {o["query"]: o for o in objs}
    if task == "T0":
        q_of = {}
        for o in objs:
            q_of[H._rotated_png(o["ref"])] = o
    rows = [json.loads(x) for x in open(os.path.join(OUT, tag + "_pairs.jsonl")) if x.strip()]
    rows = [r for r in rows if "error" not in r]
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["q"], []).append(r)
    tags: Dict[str, int] = {"SHORTLIST": 0, "CORRESPONDENCE": 0, "APPEARANCE": 0,
                            "AGGREGATION": 0}
    by_class: Dict[str, Dict[str, int]] = {}
    n = 0
    for q, rs in sorted(by_q.items()):
        qo = q_of.get(q)
        if qo is None:
            continue
        n += 1
        cls = qo["cls"]
        by_class.setdefault(cls, {k: 0 for k in tags})
        by_class[cls]["n"] = by_class[cls].get("n", 0) + 1
        scored = sorted(rs, key=lambda r: (-float(r["B"]), r["r"]))
        truth = [r for r in rs if key_of[r["r"]]["key"] == qo["key"]]
        if not truth:
            tags["SHORTLIST"] += 1
            by_class[cls]["SHORTLIST"] += 1
            continue
        win = scored[0]
        if win is truth[0]:
            continue
        t = truth[0]
        if float(t["B"]) > float(win["B"]):
            k = "AGGREGATION"
        elif (t["mapped"] < 0.8 * win["mapped"]) or (t["judged"] < 0.8 * win["judged"]):
            k = "CORRESPONDENCE"
        else:
            k = "APPEARANCE"
        tags[k] += 1
        by_class[cls][k] += 1
    return {"tag": tag, "split": split, "task": task, "queries": n, "tags": tags,
            "by_class": by_class}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="m20_fit_t3g2:FIT:T3,m20_dev_t3g2:DEV:T3,"
                                     "m20_fit_t0g2:FIT:T0")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="m20_attrib")
    a = ap.parse_args(argv)
    res = []
    for spec in a.runs.split(","):
        if not spec.strip():
            continue
        tag, split, task = spec.split(":")
        if not os.path.exists(os.path.join(OUT, tag + "_pairs.jsonl")):
            print("missing", tag)
            continue
        res.append(attribute(tag, split, a.per_class, task))
    out = os.path.join(OUT, a.out + ".json")
    with open(out, "w") as fh:
        json.dump({"runs": res}, fh, indent=1, sort_keys=True)
    print("%-16s %-5s %-4s %-6s %-6s %-6s %-6s %-6s"
          % ("run", "split", "task", "n", "SHORT", "CORR", "APPEAR", "AGG"))
    for r in res:
        t = r["tags"]
        print("%-16s %-5s %-4s %-6d %-6d %-6d %-6d %-6d"
              % (r["tag"], r["split"], r["task"], r["queries"], t["SHORTLIST"],
                 t["CORRESPONDENCE"], t["APPEARANCE"], t["AGGREGATION"]))
        for cls, c in sorted(r["by_class"].items()):
            print("    %-7s n %-3d short %-2d corr %-2d app %-2d agg %-2d"
                  % (cls, c.get("n", 0), c["SHORTLIST"], c["CORRESPONDENCE"],
                     c["APPEARANCE"], c["AGGREGATION"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
