"""Task 4 PART 4d: matcher AUC read-out (NOT a gate).

Held-out ETH-80 pair-level AUC + top-1 with paired bootstrap intervals.
Arms: B0 and the best arm (F).  Modes: real and rewired (seed 0) only.

Full protocol: 24 references (ref ranks 0-2, 3 per class) x 48 queries
(held-out ranks 6-8, both views).  If the full protocol does not fit the
time budget, use the balanced 8x16 labelled SMOKE (1 ref per class, 2
queries per class) and do NOT shrink further.

Usage:
    python t4_measure_4d.py --best-arm F --workers 2 --mode full|smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
    "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
    "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
}
ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}
HOLDOUT_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"
REF_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "refs"


def _worker_init(spec):
    import guard
    guard.activate()
    import _bootstrap as b
    b.apply_switch_overrides(spec)


def _build_ref_one(args2):
    arm, entry = args2
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    g = build(entry)
    return arm, g


def _build_refs(arms, workers):
    """Build reference graphs for the given arms (ref ranks 0-2, -090-000)."""
    import corpus
    import multiprocessing as mp

    split = corpus.split_tuning()
    ref_entries = [e for e in split["references"]
                   if e["view"] == "-090-000"]
    REF_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(arm, e) for arm in arms for e in ref_entries]
    # skip if refs already built
    need = []
    for arm, e in tasks:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        if not (REF_DIR / ("%s_%s.pkl" % (arm, name))).is_file():
            need.append((arm, e))
    if not need:
        print("  refs already built (%d), skipping" % len(tasks), flush=True)
        return
    print("  building %d refs..." % len(need), flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=("",)) as pool:
        for i, (arm, g) in enumerate(pool.imap_unordered(_build_ref_one, need,
                                                         chunksize=1)):
            name = g.name
            path = REF_DIR / ("%s_%s.pkl" % (arm, name))
            with open(path, "wb") as fh:
                pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if (i + 1) % 10 == 0 or i + 1 == len(need):
                print("  refs %3d/%d" % (i + 1, len(need)), flush=True)


def _load_ref(arm, name):
    path = REF_DIR / ("%s_%s.pkl" % (arm, name))
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_query(arm, name, shift=0):
    path = HOLDOUT_DIR / ("%s_%s_%d.pkl" % (arm, name, shift))
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _rewired(g, seed=0):
    import random
    rng = random.Random(seed)
    edges = list(g.subgraph.edges)
    new_edges = []
    for (u, v, et) in edges:
        if rng.random() < 0.5:
            new_edges.append((v, u, et))
        else:
            new_edges.append((u, v, et))
    sg = replace(g.subgraph, edges=frozenset(new_edges), edge_params={})
    return replace(g, subgraph=sg)


def top1(scores, truth):
    pred = int(np.argmax(scores))
    return float(pred == truth), pred


def pair_auc(scores, truth):
    total, count = 0.0, 0
    for i, t in enumerate(truth):
        pos = scores[i, t]
        for d in range(scores.shape[1]):
            if d == t:
                continue
            neg = scores[i, d]
            total += 1.0 if pos > neg else (0.5 if pos == neg else 0.0)
            count += 1
    return total / count if count else float("nan")


def bootstrap_auc(scores, truth, reps=2000, seed=0):
    import statistics
    rng = np.random.default_rng(seed)
    n = len(truth)
    truth_arr = np.array(truth)
    values = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        values.append(pair_auc(scores[idx], truth_arr[idx]))
    values.sort()
    return {"mean": statistics.fmean(values),
            "lo": values[int(0.025 * reps)],
            "hi": values[int(0.975 * reps) - 1]}


def _paired_t1(a_correct, b_correct):
    """Paired bootstrap interval for top-1 difference."""
    n = len(a_correct)
    diffs = np.array(a_correct) - np.array(b_correct)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    return {"mean": float(diffs.mean()),
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-arm", default="F")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    args = ap.parse_args()
    best = args.best_arm
    arms = ["B0", best]

    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)

    # build references
    print("building references...", flush=True)
    _build_refs(arms, args.workers)

    import corpus
    split = corpus.split_tuning()
    # held-out queries: ranks 6-8, both views
    query_entries = [e for e in split["heldout"]]
    if args.mode == "smoke":
        # balanced 8x16: 1 ref per class, 2 queries per class
        by_class_q = defaultdict(list)
        for e in query_entries:
            by_class_q[e["class"]].append(e)
        query_entries = []
        for c in corpus.CLASSES:
            query_entries.extend(by_class_q[c][:2])

    query_names = sorted({"%s%d%s" % (e["class"], int(e["object"]),
                                      e["view"]) for e in query_entries})
    ref_names_by_arm = {}
    for arm in arms:
        names = sorted({
            p.stem.split("_", 1)[1]
            for p in REF_DIR.glob("%s_*.pkl" % arm)})
        if args.mode == "smoke":
            # balanced 8x16: 1 ref per class
            by_class = defaultdict(list)
            for n in names:
                cls = "".join(ch for ch in n.split("-")[0] if not ch.isdigit())
                by_class[cls].append(n)
            names = [by_class[c][0] for c in corpus.CLASSES
                     if by_class.get(c)]
        ref_names_by_arm[arm] = names
    print("queries:", len(query_names), "refs B0:", len(ref_names_by_arm["B0"]),
          "refs %s:" % best, len(ref_names_by_arm[best]), flush=True)

    # score each (arm, mode) query against all refs
    modes = [("B0", "real"), (best, "real"), (best, "rewired")]
    results = {}
    per_query = {}  # (arm, mode) -> (list of per-query AUC, list of per-query top1)
    for arm, mode in modes:
        print("mode %s/%s..." % (arm, mode), flush=True)
        t0 = time.perf_counter()
        from matching.joint_mcs import joint_mcs
        refs = [_load_ref(arm, n) for n in ref_names_by_arm[arm]]
        mat = np.full((len(query_names), len(corpus.CLASSES)),
                      -np.inf, dtype=np.float64)
        truth = []
        q_aucs = []
        q_t1s = []
        for qi, qname in enumerate(query_names):
            q = _load_query(arm, qname, 0)
            if mode == "rewired":
                q = _rewired(q, 0)
            for r in refs:
                cls_idx = corpus.CLASSES.index(r.class_)
                res = joint_mcs(r.subgraph, q.subgraph, r.sigs, q.sigs)
                if res.score > mat[qi, cls_idx]:
                    mat[qi, cls_idx] = res.score
            t = corpus.CLASSES.index(q.class_)
            truth.append(t)
            q_aucs.append(pair_auc(mat[qi:qi+1], [t]))
            q_t1s.append(top1(mat[qi], t)[0])
        auc = pair_auc(mat, truth)
        auc_ci = bootstrap_auc(mat, truth)
        t1 = float(np.mean(q_t1s))
        results[(arm, mode)] = {
            "auc": auc, "auc_ci": auc_ci, "top1": t1,
            "n_queries": len(query_names), "n_refs": len(refs),
            "seconds": time.perf_counter() - t0,
        }
        per_query[(arm, mode)] = (q_aucs, q_t1s)
        print("  AUC=%.4f [%.4f, %.4f] top1=%.4f (%.0fs)"
              % (auc, auc_ci["lo"], auc_ci["hi"], t1,
                 results[(arm, mode)]["seconds"]), flush=True)

    # paired contrasts from stored per-query values
    contrasts = {}
    b0_auc, b0_t1 = per_query[("B0", "real")]
    best_auc, best_t1 = per_query[(best, "real")]
    rev_auc, rev_t1 = per_query[(best, "rewired")]
    contrasts["auc_best_minus_b0"] = _paired_t1(best_auc, b0_auc)
    contrasts["auc_real_minus_rewired"] = _paired_t1(best_auc, rev_auc)
    contrasts["top1_best_minus_b0"] = _paired_t1(best_t1, b0_t1)
    contrasts["top1_real_minus_rewired"] = _paired_t1(best_t1, rev_t1)

    out = {"best_arm": best, "mode": args.mode, "winner": WINNER,
           "results": {"%s/%s" % k: v for k, v in results.items()},
           "contrasts": contrasts,
           "corpus": "ETH-80, split_tuning()"}
    write_json(RESULTS_DIR / "t4" / "measure_4d.json", out)
    print("=== 4d read-out ===", flush=True)
    for k, v in results.items():
        print("  %s/%s AUC=%.4f top1=%.4f" % (k[0], k[1], v["auc"], v["top1"]),
              flush=True)
    for k, v in contrasts.items():
        print("  %s: %.4f [%.4f, %.4f]" % (k, v["mean"], v["lo"], v["hi"]),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
