"""MISSION 11 STEP 4 -- ETH-80 aggregation, per gallery, complete queries only.

Reads the checkpointed rows (`m11_eth80_progress.jsonl`) and the shortlist
(`m11_eth80_shortlist.json`), and scores the three arms on:

    G1  same-class K = 8   (recall 1.0 by construction, chance 1/8)
    G2  all-class K = 16   (recall measured at shortlist time, chance 1/16)

A query is counted only when EVERY candidate of the gallery it is scored in has a
row: a partial query would be a different task.  The number of complete and partial
queries is printed, because a real-image read that cannot finish must say exactly how
far it got rather than reporting a mean over a convenience subset.

Primary contrast C - A, paired per query, clustered by INSTANCE, 4000-draw bootstrap.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PROGRESS = os.path.join(OUT, "m11_eth80_progress.jsonl")
SHORT = os.path.join(OUT, "m11_eth80_shortlist.json")


def load() -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    with open(SHORT) as fh:
        short = json.load(fh)
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rows[(d["q"], d["r"])] = d
    return short, rows


def _path_of(short: Dict[str, Any]) -> Dict[str, str]:
    from Topology.placement_m11_eth80 import catalog, _qkey, _rkey
    refs, queries = catalog()
    rmap = {_rkey(r): r["path"] for r in refs}
    qmap = {_qkey(e): e["path"] for e in queries}
    return {"refs": rmap, "queries": qmap}


def rank_of(cands: List[Dict[str, Any]], arm: str, truth: str) -> int:
    order = sorted(cands, key=lambda d: (-d[arm], d["r"]))
    names = [d["r"] for d in order]
    return (names.index(truth) + 1) if truth in names else 99


def aggregate(gallery: str, short: Dict[str, Any],
              rows: Dict[Tuple[str, str], Dict[str, Any]],
              pmap: Dict[str, str]) -> Dict[str, Any]:
    K = len(next(iter(short["per_query"].values()))[gallery])
    complete: Dict[str, List[Dict[str, Any]]] = {}
    partial = 0
    for qk, rec in short["per_query"].items():
        qpath = pmap["queries"][qk]
        want = [pmap["refs"][rk] for rk in rec[gallery]]
        if not all((qpath, rp) in rows for rp in want):
            partial += 1
            continue
        cands = []
        for rk, rp in zip(rec[gallery], want):
            d = dict(rows[(qpath, rp)])
            d["r"] = rk
            cands.append(d)
        complete[qk] = cands
    res: Dict[str, Any] = {"gallery": gallery, "K": K, "chance": 1.0 / K,
                           "queries_complete": len(complete), "queries_partial": partial,
                           "arms": {}}
    if not complete:
        return res
    rng = np.random.default_rng(0)
    for arm in ("A", "B", "C"):
        t1, rr = [], []
        for qk, cands in complete.items():
            truth = short["per_query"][qk]["truth"]
            r = rank_of(cands, arm, truth)
            t1.append(1.0 if r == 1 else 0.0)
            rr.append(1.0 / r)
        res["arms"][arm] = {"top1": float(np.mean(t1)), "mrr": float(np.mean(rr)),
                            "n": len(t1)}
    inst: Dict[str, List[str]] = {}
    for qk in complete:
        inst.setdefault(short["per_query"][qk]["inst"], []).append(qk)
    for tag, x, y in (("C-A", "C", "A"), ("B-A", "B", "A"), ("C-B", "C", "B")):
        d = []
        for i, qs in inst.items():
            dx = [1.0 if rank_of(complete[q], x, short["per_query"][q]["truth"]) == 1
                  else 0.0 for q in qs]
            dy = [1.0 if rank_of(complete[q], y, short["per_query"][q]["truth"]) == 1
                  else 0.0 for q in qs]
            d.append(float(np.mean(dx) - np.mean(dy)))
        d = np.array(d)
        if len(d) < 2:
            continue
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        bs = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(4000)]
        res.setdefault("contrasts", {})[tag] = {
            "delta": float(d.mean()), "se": se, "mde": float(2.8 * se),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "instances": int(len(d))}
    by_cls: Dict[str, Any] = {}
    for cls in sorted({short["per_query"][q]["cls"] for q in complete}):
        qs = [q for q in complete if short["per_query"][q]["cls"] == cls]
        by_cls[cls] = {"queries": len(qs)}
        for arm in ("A", "C"):
            by_cls[cls][arm] = float(np.mean(
                [1.0 if rank_of(complete[q], arm, short["per_query"][q]["truth"]) == 1
                 else 0.0 for q in qs]))
    res["by_class"] = by_cls
    M = [d for q in complete for d in complete[q]]
    res["coverage"] = {
        "judged_mean": float(np.mean([d["judged"] for d in M])),
        "mapped_mean": float(np.mean([d["mapped"] for d in M])),
        "share_mean": float(np.mean([d["share"] for d in M])),
        "evidence_mean": float(np.mean([d["E"] for d in M])),
        "sec_a": float(np.mean([d["sec_a"] for d in M])),
        "sec_c": float(np.mean([d["sec_c"] for d in M]))}
    return res


def main() -> int:
    import guard
    guard.activate()
    short, rows = load()
    pmap = _path_of(short)
    out: Dict[str, Any] = {"rows": len(rows),
                           "shortlist": {"recall_g1": short.get("recall_g1"),
                                         "recall_g2": short.get("recall_g2")}}
    for gallery in ("g1", "g2"):
        out[gallery] = aggregate(gallery, short, rows, pmap)
        g = out[gallery]
        print("== %s  K=%d chance %.3f  complete %d partial %d"
              % (gallery, g["K"], g["chance"], g["queries_complete"],
                 g["queries_partial"]))
        for arm, d in sorted(g.get("arms", {}).items()):
            print("   %s top1 %.3f mrr %.3f" % (arm, d["top1"], d["mrr"]))
        for tag, c in sorted(g.get("contrasts", {}).items()):
            print("   %s delta %+.3f se %.3f mde %.3f ci [%+.3f,%+.3f] n_inst %d"
                  % (tag, c["delta"], c["se"], c["mde"], c["ci"][0], c["ci"][1],
                     c["instances"]))
    with open(os.path.join(OUT, "m11_eth80.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print("wrote %s" % os.path.join(OUT, "m11_eth80.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
