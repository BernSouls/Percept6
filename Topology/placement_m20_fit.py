"""MISSION 20 / items 4, 6, 7 -- the four arms on FIT, the null, and the coverage control.

Reads the pair rows a run already produced (nothing is recomputed) and answers four
questions, each with its n:

  4a  the four arms' top-1 / MRR on T1, T2, T3 (T0 is a separate run: the control)
  4b  the NULL: how well each arm separates a true pair from an impostor pair (AUC over
      pairs), which is the calibration the raw top-1 hides
  4c  the coverage control: is the score a function of how MANY places were mapped?  Report
      the correlation of each arm with the mapped count, then re-rank inside a mapped-count
      MATCHED subsample (pairs within +-5 % of the median count) and show the top-1 there.
  6   the shipped arm's top-1 per task, against the registered [0.30, 0.85] band
  7   the aggregate weight: sweep lambda on FIT; if the optimum is at a boundary, say so and
      use 0 -- a boundary value is not a fitted value.

Usage:
    python Topology/placement_m20_fit.py --tag m20_fit_t3g2 --split FIT --per-class 2
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def load(tag: str) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    from Topology import placement_m20_eth80 as H
    objs = H.objects("FIT")
    per: Dict[str, List[Dict[str, Any]]] = {}
    for o in objs:
        per.setdefault(o["cls"], []).append(o)
    return objs, per


def objects_for(split: str, per_class: int) -> List[Dict[str, Any]]:
    from Topology import placement_m20_eth80 as H
    objs = H.objects(split)
    if not per_class:
        return objs
    per: Dict[str, List[Dict[str, Any]]] = {}
    for o in objs:
        per.setdefault(o["cls"], []).append(o)
    return [o for c in sorted(per) for o in per[c][:int(per_class)]]


def auc(pos: List[float], neg: List[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    num = den = 0.0
    for p in pos:
        for q in neg:
            den += 1.0
            num += 1.0 if p > q else 0.5 if p == q else 0.0
    return num / den


def spear(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m20_fit_t3g2")
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="m20_fit_read")
    a = ap.parse_args(argv)
    lam_ev = float(json.load(open(os.path.join(ROOT, "matching",
                                               "placement_score_params.json")))["w"])
    objs = objects_for(a.split, a.per_class)
    key_of = {o["ref"]: o for o in objs}
    q_of = {o["query"]: o for o in objs}
    rows = [json.loads(x) for x in
            open(os.path.join(OUT, a.tag + "_pairs.jsonl")) if x.strip()]
    rows = [r for r in rows if "error" not in r]
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_q.setdefault(r["q"], []).append(r)
    res: Dict[str, Any] = {"tag": a.tag, "split": a.split, "per_class": a.per_class,
                           "lam_evidence": lam_ev, "n_pairs": len(rows),
                           "n_queries": len(by_q), "arms": {}, "lambdas": []}

    def value(arm: str, r: Dict[str, Any], lam: float) -> Optional[float]:
        s3 = r.get("S3")
        if arm == "S0":
            return r["B"]
        if arm == "S1":
            return r["B"] + lam_ev * r["E"]
        if arm == "S2":
            return r["B"] + lam * (s3 if s3 is not None else 0.0)
        if arm == "S3":
            return s3
        raise KeyError(arm)

    def rank_task(task: str, arm: str, lam: float) -> Dict[str, Any]:
        ranks, rrs, n, pos = [], [], 0, 0
        for q, rs in sorted(by_q.items()):
            qo = q_of[q]
            if task == "T1":
                pool = [r for r in rs if key_of[r["r"]]["cls"] == qo["cls"]]
            else:
                pool = list(rs)
            scored = [(r, value(arm, r, lam)) for r in pool]
            scored = [(r, v) for (r, v) in scored if v is not None]
            if not scored:
                continue
            scored.sort(key=lambda t: (-t[1], t[0]["r"]))
            if task == "T2":
                ok = [j for j, (r, _v) in enumerate(scored)
                      if key_of[r["r"]]["cls"] == qo["cls"] and key_of[r["r"]]["key"] != qo["key"]]
            else:
                ok = [j for j, (r, _v) in enumerate(scored) if key_of[r["r"]]["key"] == qo["key"]]
            n += 1
            if ok:
                pos += 1
                ranks.append(ok[0] + 1)
                rrs.append(1.0 / (ok[0] + 1))
            else:
                ranks.append(0)
                rrs.append(0.0)
        if not n:
            return {"n": 0}
        return {"n": n, "rank1": sum(1 for r in ranks if r == 1) / n,
                "mrr": sum(rrs) / n, "recall_at_S": pos / n,
                "n_scored_mean": float(np.mean([len(by_q[q]) for q in by_q]))}

    for arm in ("S0", "S1", "S2", "S3"):
        res["arms"][arm] = {t: rank_task(t, arm, lam_ev) for t in ("T1", "T2", "T3")}

    # ---- item 7: the lambda sweep, on FIT, with the boundary rule stated ----
    best = None
    for lam in LAMBDAS:
        row = {t: rank_task(t, "S2", lam) for t in ("T1", "T2", "T3")}
        row["lambda"] = lam
        row["mean_rank1"] = float(np.mean([row[t]["rank1"] for t in ("T1", "T2", "T3")
                                           if row[t].get("n")]))
        res["lambdas"].append(row)
        if best is None or row["mean_rank1"] > best["mean_rank1"] + 1e-12:
            best = row
    res["lambda_fit"] = {"best": best["lambda"], "best_mean_rank1": best["mean_rank1"],
                         "boundary": best["lambda"] in (min(LAMBDAS), max(LAMBDAS)),
                         "grid": list(LAMBDAS)}
    # the registered rule: a boundary optimum is not a fit -- weight 0 and say so
    res["lambda_shipped"] = 0.0 if res["lambda_fit"]["boundary"] else best["lambda"]

    # ---- item 4b: the null (truth pair against every impostor pair of the same query) ----
    null = {}
    for arm in ("S0", "S1", "S2", "S3"):
        pos, neg = [], []
        for q, rs in by_q.items():
            qo = q_of[q]
            for r in rs:
                v = value(arm, r, res["lambda_shipped"] if arm == "S2" else lam_ev)
                if v is None:
                    continue
                if key_of[r["r"]]["key"] == qo["key"]:
                    pos.append(float(v))
                else:
                    neg.append(float(v))
        null[arm] = {"auc": auc(pos, neg), "n_pos": len(pos), "n_neg": len(neg),
                     "mean_pos": float(np.mean(pos)) if pos else None,
                     "mean_neg": float(np.mean(neg)) if neg else None}
    res["null"] = null

    # ---- item 4c: the coverage control ----
    counts = [r["mapped"] for r in rows]
    med = float(np.median(counts)) if counts else None
    cov = {}
    for arm in ("S0", "S1", "S2", "S3"):
        sc = [value(arm, r, res["lambda_shipped"] if arm == "S2" else lam_ev) for r in rows]
        keep = [(float(v), int(r["mapped"])) for v, r in zip(sc, rows) if v is not None]
        cov[arm] = {"spearman_score_vs_mapped": spear([k[0] for k in keep],
                                                      [k[1] for k in keep]),
                    "pearson_score_vs_mapped": (float(np.corrcoef(
                        [k[0] for k in keep], [k[1] for k in keep])[0, 1])
                        if len(keep) > 3 else None)}
    matched = {}
    if med is not None:
        band = (0.95 * med, 1.05 * med)
        for arm in ("S0", "S1", "S2", "S3"):
            ranks, n, pos = [], 0, 0
            for q, rs in sorted(by_q.items()):
                qo = q_of[q]
                pool = [r for r in rs if band[0] <= r["mapped"] <= band[1]]
                if len(pool) < 2:
                    continue
                scored = [(r, value(arm, r, res["lambda_shipped"] if arm == "S2" else lam_ev))
                          for r in pool]
                scored = [(r, v) for (r, v) in scored if v is not None]
                if len(scored) < 2:
                    continue
                scored.sort(key=lambda t: (-t[1], t[0]["r"]))
                ok = [j for j, (r, _v) in enumerate(scored) if key_of[r["r"]]["key"] == qo["key"]]
                n += 1
                if ok:
                    pos += 1
                    ranks.append(ok[0] + 1)
            matched[arm] = {"n": n, "rank1": (sum(1 for r in ranks if r == 1) / n) if n else None,
                            "band": [band[0], band[1]]}
    res["coverage"] = {"counts_median": med, "correlation": cov, "matched": matched}
    # judged vs the new denominator: the operationalisation is checked, not assumed
    over = sum(1 for r in rows if r["judged"] > r["s3_den"])
    res["denominator"] = {"pairs_with_judged_gt_den": over, "n_pairs": len(rows),
                          "den_values": sorted({r["s3_den"] for r in rows})[:5]}
    out = os.path.join(OUT, a.out + ".json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("%-4s %-28s %-28s %-28s" % ("arm", "T1 rank1/MRR/recall", "T2", "T3"))
    for arm, tasks in res["arms"].items():
        def f(t):
            d = tasks[t]
            return ("%.3f/%.3f/%.3f (n=%d)" % (d["rank1"], d["mrr"], d["recall_at_S"], d["n"])
                    if d.get("n") else "-")
        print("%-4s %-28s %-28s %-28s" % (arm, f("T1"), f("T2"), f("T3")))
    print("\nlambda sweep (mean rank1 over T1-T3):",
          " ".join("L%.2f=%.3f" % (r["lambda"], r["mean_rank1"]) for r in res["lambdas"]))
    print("lambda fit %r (boundary %s) -> shipped %r"
          % (res["lambda_fit"]["best"], res["lambda_fit"]["boundary"],
             res["lambda_shipped"]))
    print("null AUC:", {k: (None if v["auc"] is None else round(v["auc"], 3))
                        for k, v in null.items()})
    print("score-vs-mapped spearman:",
          {k: (None if v["spearman_score_vs_mapped"] is None
               else round(v["spearman_score_vs_mapped"], 3))
           for k, v in cov.items()})
    print("coverage-matched rank1:", {k: v["rank1"] for k, v in matched.items()})
    print("judged>den pairs %d/%d" % (over, len(rows)))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
