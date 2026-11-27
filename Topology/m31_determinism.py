"""M31 item 5: determinism at 1, 2 and 4 workers, with BLAS pinned to one thread.

    python Topology/m31_determinism.py --pairs 2

One query's pairs, scored by pools of 1, 2 and 4 worker processes (the read's own `_score_unit`,
i.e. the shipped pair score with the bundle's blend), compared to 1e-12.  Pairs are independent and
the matcher is deterministic, so the only way this can move is a shared mutable cache -- which is
exactly what M31 touched (the per-side colour cache), so it is worth pinning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m31_determinism.json")
GATE = 1e-12


def _jobs(n: int) -> List[Any]:
    from Topology import m26_records as REC
    from Topology.m30_read import door, _ref_path
    rows = door(True)
    out: List[Any] = []
    for r in rows[:2]:
        js = [(r["query"], _ref_path(j)) for j in r["order"][:n]]
        out.append((r["object"], js, True, 0.10))
    return out


def run(n: int = 2) -> Dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import multiprocessing as mp
    from Topology.m30_read import _score_unit
    units = _jobs(n)
    res: Dict[str, Any] = {"pairs_per_query": n, "queries": len(units), "workers": {},
                           "max_abs_diff": 0.0, "gate": GATE}
    base: Dict[str, float] = {}
    for w in (1, 2, 4):
        t0 = time.time()
        with mp.get_context("fork").Pool(processes=w) as pool:
            got = pool.map(_score_unit, units, chunksize=1)
        scores = {r["r"]: float(r["score"]) for chunk in got for r in chunk
                  if r.get("score") is not None}
        res["workers"][w] = {"pairs": len(scores), "seconds": round(time.time() - t0, 1)}
        if not base:
            base = dict(scores)
            res["workers"][w]["reference"] = True
        else:
            common = sorted(set(base) & set(scores))
            worst = max((abs(base[k] - scores[k]) for k in common), default=0.0)
            res["workers"][w]["compared"] = len(common)
            res["workers"][w]["max_abs_diff"] = worst
            res["max_abs_diff"] = max(res["max_abs_diff"], worst)
        print("  %d worker(s): %d pairs, %.0fs, max |diff| vs 1 worker %.3e"
              % (w, len(scores), res["workers"][w]["seconds"],
                 res["workers"][w].get("max_abs_diff", 0.0)), flush=True)
    res["passes"] = bool(res["max_abs_diff"] <= GATE)
    res["threads"] = os.environ.get("OPENBLAS_NUM_THREADS")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k != "workers"}, indent=1))
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2)
    a = ap.parse_args(list(argv) if argv is not None else None)
    run(a.pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
