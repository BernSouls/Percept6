"""M27 item 1a, second half: are the COMMITTED pair scores the arithmetic of this tree?

The proof run in `m27_profile.py --proof` recomputed three of M23's 621 committed pairs and did NOT
get the same numbers (+0.0052, +0.0016, +0.0099 on scores near 0.35, i.e. +1.5 % relative).  The
place records are not the cause: all 320 cells in this sandbox have the same place and node counts
as the committed M26 census.  So either the matcher's arithmetic changed after M23, or the scores
depend on something the pair-cache key does not cover.

This script measures the consequence, not the cause: for a sample of queries it recomputes the whole
committed shortlist and asks whether the ORDER changes -- because a cache that is off by 1.5 % is
harmless if it ranks the same way, and useless if it does not.

    python Topology/m27_drift.py --queries 5 --workers 2
    python Topology/m27_drift.py --report        # what the last run found
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement", "m27_drift.json")


def _committed_by_query() -> Dict[str, List[Dict[str, Any]]]:
    from Topology.m27_profile import committed_pairs
    by: Dict[str, List[Dict[str, Any]]] = {}
    p = os.path.join(ROOT, "Topology", "results", "placement", "m23_read_pairs.jsonl")
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(os.path.basename(str(r["q"])), []).append(
            {"q": str(r["q"]), "r": str(r["r"]), "B": float(r["B"])})
    return by


def _local(path: str) -> str:
    """Resolve a committed absolute path inside THIS checkout by basename and folder."""
    inst = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    cls_name = "".join(ch for ch in name.split("-")[0] if ch.isalpha())
    classdir = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
                "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}
    return os.path.join(ROOT, "data", "ETH-80-master", classdir[cls_name], inst, name)


def _one(job: Tuple[str, str]) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    q, r = job
    return {"q": q, "r": r, "B": float(FE.pair_score(q, r))}


def run(queries: int = 5, workers: int = 0) -> Dict[str, Any]:
    import multiprocessing as mp
    import matching.placement_frontend as FE
    from matching.placement_records import placed_workers
    FE.apply_recipe()
    workers = int(workers) or placed_workers()
    by = _committed_by_query()
    keys = sorted(by)[:queries]
    jobs: List[Tuple[str, str]] = []
    for k in keys:
        for row in by[k]:
            jobs.append((_local(row["q"]), _local(row["r"])))
    images = sorted({p for j in jobs for p in j})
    t0 = time.time()
    for p in images:
        FE.cell(p, "m27drift")
    t_warm = time.time() - t0
    t0 = time.time()
    ctx = mp.get_context("fork")
    if workers == 1:
        out = [_one(j) for j in jobs]
    else:
        with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
            out = list(pool.imap_unordered(_one, jobs, chunksize=1))
    el = time.time() - t0
    new = {(os.path.basename(o["q"]), os.path.basename(o["r"])): o["B"] for o in out}
    rows: List[Dict[str, Any]] = []
    order_same = 0
    for k in keys:
        old = sorted(by[k], key=lambda r: (-r["B"], os.path.basename(r["r"])))
        fresh = sorted(by[k], key=lambda r: (-new[(k, os.path.basename(r["r"]))],
                                             os.path.basename(r["r"])))
        d = [new[(k, os.path.basename(r["r"]))] - r["B"] for r in by[k]]
        same = [os.path.basename(r["r"]) for r in old] == [os.path.basename(r["r"]) for r in fresh]
        order_same += int(same)
        rows.append({"query": k, "n": len(by[k]), "order_identical": bool(same),
                     "mean_delta": float(np.mean(d)), "max_abs_delta": float(np.max(np.abs(d))),
                     "old_winner": os.path.basename(old[0]["r"]),
                     "new_winner": os.path.basename(fresh[0]["r"]),
                     "top1_identical": os.path.basename(old[0]["r"]) ==
                                       os.path.basename(fresh[0]["r"])})
    res = {"queries": len(keys), "pairs": len(jobs), "workers": workers,
           "seconds": round(el, 1), "seconds_per_pair": round(el / len(jobs), 2),
           "prewarm_seconds": round(t_warm, 1),
           "orders_identical": order_same, "top1_identical": sum(r["top1_identical"] for r in rows),
           "mean_delta": float(np.mean([r["mean_delta"] for r in rows])),
           "max_abs_delta": float(np.max([r["max_abs_delta"] for r in rows])), "rows": rows}
    print(json.dumps(res, indent=1))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.report:
        print(open(OUT).read() if os.path.exists(OUT) else "no run yet")
        return 0
    run(a.queries, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
