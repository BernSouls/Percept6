"""M27 item 1a proof + item 1c profile: what a pair costs, and whether the cache is real.

    python Topology/m27_profile.py --proof      # 3 committed pairs recomputed, must be identical
    python Topology/m27_profile.py --profile    # graph build vs matching, at 1 / 2 / 4 workers

``--proof`` is item 1a's requirement: three of the committed M23 pair scores are recomputed from
the shipped matcher and must come back identical, which is what licenses adopting 1,062 committed
pairs under the current digest.

``--profile`` separates the two costs a pair carries -- building (or loading) each image's graph,
and the matcher itself -- and measures throughput at 1, 2 and 4 workers so the default worker count
can be set to the physical cores rather than guessed.
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

from Topology import m26_records as REC  # noqa: E402

OUT_DIR = os.path.join(ROOT, "Topology", "results", "placement")


def _paths(name: str) -> Tuple[str, str]:
    """A committed pair's (query, reference) path, resolved in THIS checkout."""
    label_q, label_r = name
    base = os.path.join(ROOT, "data", "ETH-80-master")
    return label_q, label_r


def committed_pairs(limit: int = 0) -> List[Tuple[str, str, float]]:
    """(query path, reference path, committed score) from the M23 read, resolved locally."""
    p = os.path.join(OUT_DIR, "m23_read_pairs.jsonl")
    rows: List[Tuple[str, str, float]] = []
    classdir = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
                "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        q = str(r["q"])
        ref = str(r["r"])
        inst_q = os.path.basename(os.path.dirname(q))
        cls = os.path.basename(q).rstrip("0123456789-")[:0] or None      # noqa: F841
        name = os.path.basename(q).split("-")[0]
        cls_name = "".join(ch for ch in name if ch.isalpha())
        qp = os.path.join(ROOT, "data", "ETH-80-master", classdir[cls_name], inst_q,
                          os.path.basename(q))
        ref_inst = os.path.basename(os.path.dirname(ref))
        ref_name = os.path.basename(ref).split("-")[0]
        ref_cls = "".join(ch for ch in ref_name if ch.isalpha())
        rp = os.path.join(ROOT, "data", "ETH-80-master", classdir[ref_cls], ref_inst,
                          os.path.basename(ref))
        if os.path.exists(qp) and os.path.exists(rp):
            rows.append((qp, rp, float(r["B"])))
        if limit and len(rows) >= limit:
            break
    return rows


def proof(n: int = 3) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    FE.apply_recipe()
    from matching import placement_recognise as RC
    cache = RC.load_pair_cache()
    rows = committed_pairs(limit=0)
    out: List[Dict[str, Any]] = []
    for qp, rp, want in rows[:n]:
        key = RC._pair_key(qp, rp, 0.375)
        t0 = time.time()
        got = float(FE.pair_score(qp, rp))
        out.append({"query": os.path.basename(qp), "ref": os.path.basename(rp),
                    "committed": want, "recomputed": got,
                    "identical": abs(got - want) < 1e-12, "seconds": round(time.time() - t0, 2),
                    "cached": key in cache})
    res = {"pairs": out, "all_identical": all(o["identical"] for o in out),
           "adopted": dict(RC.ADOPTED)}
    print(json.dumps(res, indent=1))
    return res


def _one(arg: Tuple[str, str]) -> Tuple[str, str, float, float]:
    """(build seconds, match seconds) for one pair in a worker."""
    import matching.placement_frontend as FE
    qp, rp = arg
    t0 = time.time()
    FE.cell(qp, "m27profile")
    FE.cell(rp, "m27profile")
    t_build = time.time() - t0
    t0 = time.time()
    FE.pair_score(qp, rp)
    t_match = time.time() - t0
    return qp, rp, t_build, t_match


def profile(pairs: int = 8, workers_list: Sequence[int] = (1, 2, 4)) -> Dict[str, Any]:
    import multiprocessing as mp
    import matching.placement_frontend as FE
    from matching.placement_records import placed_workers
    FE.apply_recipe()
    rows = committed_pairs(limit=pairs)
    jobs = [(qp, rp) for qp, rp, _ in rows]
    uniq = sorted({p for j in jobs for p in j})

    # (0) PREWARM every unique graph once, under the tag the pair jobs will use
    t0 = time.time()
    for p in uniq:
        FE.cell(p, "m27profile")
    t_prewarm = time.time() - t0

    # (i) a COLD build: a tag nothing has ever been built with
    qp = jobs[0][0]
    FE.clear_cells()
    t0 = time.time()
    FE.cell(qp, "m27cold")
    t_cold = time.time() - t0

    # (ii) a WARM graph: the same image, the prewarmed tag, this process's memo dropped
    FE.clear_cells()
    t0 = time.time()
    FE.cell(qp, "m27profile")
    t_warm = time.time() - t0

    # (iii) matching only, both cells in memory
    qp, rp = jobs[0]
    FE.cell(qp, "m27profile")
    FE.cell(rp, "m27profile")
    t0 = time.time()
    FE.pair_score(qp, rp)
    t_match = time.time() - t0

    out: Dict[str, Any] = {"cold_graph_seconds": round(t_cold, 2),
                           "warm_graph_seconds": round(t_warm, 2),
                           "matching_only_seconds": round(t_match, 2),
                           "prewarm_seconds": round(t_prewarm, 1),
                           "prewarm_images": len(uniq),
                           "cores": os.cpu_count(), "default_workers": placed_workers(),
                           "throughput": {}}
    ctx = mp.get_context("fork")
    for w in workers_list:
        FE.clear_cells()
        t0 = time.time()
        if w == 1:
            for j in jobs:
                _one(j)
        else:
            with ctx.Pool(processes=w, maxtasksperchild=4) as pool:
                list(pool.imap_unordered(_one, jobs, chunksize=1))
        el = time.time() - t0
        out["throughput"]["%d" % w] = {"pairs": len(jobs), "seconds": round(el, 1),
                                       "seconds_per_pair": round(el / len(jobs), 2)}
        print("workers %d: %.1fs for %d pairs (%.2f s/pair)"
              % (w, el, len(jobs), el / len(jobs)), flush=True)
    print(json.dumps(out, indent=1))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--json", default=os.path.join(OUT_DIR, "m27_profile.json"))
    a = ap.parse_args(list(argv) if argv is not None else None)
    out: Dict[str, Any] = {}
    if a.proof:
        out["proof"] = proof(3)
    if a.profile:
        out["profile"] = profile(a.pairs)
    if out:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", a.json)
    if not out:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
