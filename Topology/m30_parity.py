"""M30 parity: the pair scores the read produced with two workers must be reproduced exactly by a
single-threaded run with BLAS pinned to one thread.

    python Topology/m30_parity.py --pairs 3

Two things at once, which is why it is cheap:

* **determinism** -- the same pair, scored in a 2-worker pool (the read) and sequentially in this
  process with `OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = 1`, must agree to 1e-12;
* **the shipped path** -- the scores are read back from `m30_read_pairs.jsonl`, i.e. this compares
  against numbers produced by a different process, not against numbers produced here.

The blend is ON in both, at the same weight, because that is the arm the read scored.
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
PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m30_parity.json")
GATE = 1e-12


def run(n: int = 3) -> Dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rows: List[Dict[str, Any]] = []
    with open(PAIRS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("score") is not None:
                rows.append(r)
    # one pair per query, so the sample is not eight pairs of the same object
    seen, sample = set(), []
    for r in rows:
        if r["object"] in seen:
            continue
        seen.add(r["object"])
        sample.append(r)
        if len(sample) >= n:
            break
    import parameters as P
    import matching.placement_frontend as FE
    FE.apply_recipe()
    P.PLACEMENT_NODE_COLOUR = True
    P.PLACEMENT_NODE_COLOUR_W = 0.
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    for r in sample:
        P.PLACEMENT_NODE_COLOUR = False
        FE.cell(r["q"], "parity")
        FE.cell(r["r"], "parity")
        P.PLACEMENT_NODE_COLOUR = True
        P.PLACEMENT_NODE_COLOUR_W = float(r.get("colour_w", 0.10))
        s = float(FE.pair_score(r["q"], r["r"]))
        out.append({"object": r["object"], "read_score": r["score"], "here": s,
                    "diff": abs(s - r["score"])})
        print("  %-10s read %.12f  here %.12f  |diff| %.3e"
              % (r["object"], r["score"], s, abs(s - r["score"])), flush=True)
    worst = max((x["diff"] for x in out), default=0.0)
    res = {"pairs": out, "n": len(out), "max_abs_diff": worst, "gate": GATE,
           "passes": bool(worst <= GATE), "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "seconds": round(time.time() - t0, 1),
           "compared_against": "m30_read_pairs.jsonl (2-worker pool, separate process)"}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3)
    a = ap.parse_args(list(argv) if argv is not None else None)
    run(a.pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
