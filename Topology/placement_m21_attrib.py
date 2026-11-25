"""MISSION 21 / O4 -- make AGGREGATION reachable: the composite is computed IN the tagger.

WHAT WAS WRONG IN MISSION 20.  The tagger picked the winner by the shipped score `B` and then
asked whether the truth's `B` was above the winner's.  It cannot be: the winner is the
maximum of `B`.  The AGGREGATION tag was therefore unreachable for arm S0 by construction, and
a zero count was reported as if it were a measurement of the aggregation.

WHAT THIS FILE DOES INSTEAD.  Each arm's own composite is computed here, from the per-pair
components already on disk, and the WINNER IS THE ARGMAX OF THAT COMPOSITE:

    S0  B                                   the shipped score
    S1  B + w * E                           shipped + placement evidence, w from
                                            `matching/placement_score_params.json`
    S2  B + lambda * S3                     shipped + the coverage-neutral aggregate, lambda
                                            from the M20 fit (0.0 when the fit was at the
                                            boundary -- a boundary optimum is not a fit)

Then the tags are read off in this order, so they stay disjoint and reproducible:

    SHORTLIST       the positive is not in the run at all (stage 1 dropped it)
    AGGREGATION     the truth's SHIPPED score is ABOVE the winner's, but the arm's composite
                    puts the winner first: the pieces favoured the truth and the arithmetic
                    reversed them.  This is the tag M20 could not produce.
    CORRESPONDENCE  the truth mapped or judged materially fewer places than the winner
                    (< 80%): the matcher did not find the places
    APPEARANCE      counts comparable and the truth's shipped score below the winner's: the
                    evidence that existed did not favour it

`selftest()` builds rows where the shipped score ranks the truth first and the composite does
not, and asserts the tagger both picks the composite's winner and emits AGGREGATION -- a
tagger that ranked by the quantity it tests fails that assertion.

T0 is the aggregation control (same image, 15 deg), not evidence: a T0 failure is arithmetic
by construction.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
TAGS = ("SHORTLIST", "AGGREGATION", "CORRESPONDENCE", "APPEARANCE")


def arm_value(arm: str, r: Dict[str, Any], w_ev: float, lam_s2: float) -> float:
    if arm == "S0":
        return float(r["B"])
    if arm == "S1":
        return float(r["B"]) + w_ev * float(r.get("E") or 0.0)
    if arm == "S2":
        return float(r["B"]) + lam_s2 * float(r.get("S3") or 0.0)
    raise KeyError(arm)


def tag_query(rows: List[Dict[str, Any]], truth: Optional[Dict[str, Any]], arm: str,
              w_ev: float, lam_s2: float) -> str:
    """The tag for one query.  ``rows`` is every pair of that query in the run."""
    if truth is None:
        return "SHORTLIST"
    win = max(rows, key=lambda r: (arm_value(arm, r, w_ev, lam_s2), -float(r["B"])))
    if win is truth:
        return ""
    if float(truth["B"]) > float(win["B"]):
        return "AGGREGATION"
    if (truth["mapped"] < 0.8 * win["mapped"]) or (truth["judged"] < 0.8 * win["judged"]):
        return "CORRESPONDENCE"
    return "APPEARANCE"


def attribute(path: str, arm: str, w_ev: float, lam_s2: float,
              truth_of: Optional[Dict[str, Dict[str, Any]]] = None,
              key_of: Optional[Dict[str, Dict[str, Any]]] = None
              ) -> Dict[str, Any]:
    """Tag every query of one run for one arm, with per-class counts and n."""
    rows = [json.loads(x) for x in open(path) if x.strip()]
    rows = [r for r in rows if "error" not in r]
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["q"], []).append(r)
    counts = {k: 0 for k in TAGS}
    by_class: Dict[str, Dict[str, int]] = {}
    wins = 0
    n = 0
    for q, rs in sorted(by_q.items()):
        truth = None
        if truth_of is not None:
            t = truth_of.get(q)
            truth = next((r for r in rs if key_of[r["r"]]["key"] == t["key"]), None) \
                if t is not None else None
        cls = (truth_of.get(q) or {}).get("cls", "?") if truth_of else "?"
        by_class.setdefault(cls, {k: 0 for k in TAGS})
        by_class[cls]["n"] = by_class[cls].get("n", 0) + 1
        n += 1
        t = tag_query(rs, truth, arm, w_ev, lam_s2)
        if t == "":
            wins += 1
        else:
            counts[t] += 1
            by_class[cls][t] += 1
    return {"file": os.path.basename(path), "arm": arm, "queries": n, "top1": wins,
            "top1_rate": (wins / n) if n else None, "tags": counts, "by_class": by_class,
            "w_evidence": w_ev, "lambda_s2": lam_s2}


def selftest() -> Dict[str, Any]:
    """The tagger must rank by the composite, not by the quantity it tags."""
    truth = {"B": 0.90, "E": 0.0, "S3": -0.2, "mapped": 10, "judged": 5, "r": "truth"}
    other = {"B": 0.80, "E": 0.9, "S3": 0.4, "mapped": 10, "judged": 5, "r": "other"}
    rows = [truth, other]
    t_s1 = tag_query(rows, truth, "S1", 0.375, 0.0)
    t_s2 = tag_query(rows, truth, "S2", 0.375, 1.0)
    t_s0 = tag_query(rows, truth, "S0", 0.375, 1.0)
    ok = (t_s1 == "AGGREGATION" and t_s2 == "AGGREGATION" and t_s0 == "")
    return {"rows": rows, "tag_S0": t_s0, "tag_S1": t_s1, "tag_S2": t_s2, "ok": ok}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="m20_fit_t3g2:FIT:T3,m20_dev_t3g2:DEV:T3")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--lam-s2", type=float, default=None)
    ap.add_argument("--out", default="m21_attrib")
    a = ap.parse_args(argv)
    from Topology.placement_m20_fit import objects_for
    w_ev = float(json.load(open(os.path.join(ROOT, "matching",
                                             "placement_score_params.json")))["w"])
    lam_s2 = a.lam_s2
    if lam_s2 is None:
        f = os.path.join(OUT, "m20_fit_read.json")
        lam_s2 = float(json.load(open(f)).get("lambda_shipped", 0.0)) if os.path.exists(f) else 0.0
    st = selftest()
    print("selftest: S0 %r S1 %r S2 %r -> ok %s"
          % (st["tag_S0"], st["tag_S1"], st["tag_S2"], st["ok"]))
    res = [{"selftest": st}]
    for spec in a.runs.split(","):
        if not spec.strip():
            continue
        tag, split, task = spec.split(":")
        path = os.path.join(OUT, tag + "_pairs.jsonl")
        if not os.path.exists(path):
            print("missing", tag)
            continue
        objs = objects_for(split, a.per_class)
        truth_of = {o["query"]: o for o in objs}
        key_of = {o["ref"]: o for o in objs}
        if task == "T0":
            from Topology import placement_m20_eth80 as H
            truth_of = {H._rotated_png(o["ref"]): o for o in objs}
        for arm in ("S0", "S1", "S2"):
            res.append(attribute(path, arm, w_ev, lam_s2, truth_of, key_of))
    out = os.path.join(OUT, a.out + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, {"runs": res})
    print("%-18s %-4s %-4s %-4s %-6s %-6s %-6s %-6s %-6s"
          % ("file", "arm", "n", "top1", "SHORT", "AGG", "CORR", "APPEAR", "rate"))
    for r in res[1:]:
        t = r["tags"]
        print("%-18s %-4s %-4d %-4d %-6d %-6d %-6d %-6d %.3f"
              % (r["file"][:18], r["arm"], r["queries"], r["top1"], t["SHORTLIST"],
                 t["AGGREGATION"], t["CORRESPONDENCE"], t["APPEARANCE"],
                 r["top1_rate"] or 0))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
