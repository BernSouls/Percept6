"""Task 3 PART C: end-to-end measurement with the project's own instruments.

Arms (same queries/references/scorer = production ``joint_mcs``):
    M0  everything OFF            (soft-OFF graphs; min-credit off; no address)
    M1  soft gates ON, weights ignored (soft-ON graphs; min-credit off)
    M2  soft gates ON, weights used    (soft-ON graphs; min-credit on)
    M3  M2 + address seed               (soft-ON graphs; min-credit on; addrs)

Matcher flags = the baseline winner's cell, unchanged:
    beta=0.75 unmatched=0.15 occlusion=0.5 pooled w=0.1 gamma=0.5
    denom=checkable isolated=downweight

Metrics: pair-level AUC + top-1 with paired bootstrap intervals (as in
``Topology/classify.py`` / ``tune.py``).  On the best arm only: the
real/rewired/no-edge control from the tuning report.  Also: node
correspondence (known-transform) for M0/M2/M3, edge-weight bands, edge
flicker, survival/0px/information-guard invariants, graph census, build
time per graph.

Usage:
    python t3_measure.py --arms M0,M1,M2,M3 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]

import numpy as np  # noqa: E402

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import corpus  # noqa: E402

TUNE = RESULTS_DIR / "tune"
TUNE.mkdir(parents=True, exist_ok=True)

#: the baseline winner's alignment + score settings (TUNING_REPORT), unchanged.
WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
    "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
}

#: arm -> (graph variant file, MATCH_EDGE_MIN_CREDIT, use address seed)
ARMS = {
    "M0": ("graphs_softoff.json", False, False),
    "M1": ("graphs.json", False, False),
    "M2": ("graphs.json", True, False),
    "M3": ("graphs.json", True, True),
}
ARM_ORDER = ("M0", "M1", "M2", "M3")


# --------------------------------------------------------------- metrics
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
    rng = np.random.default_rng(seed)
    n = len(truth)
    values = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        values.append(pair_auc(scores[idx], truth[idx]))
    values.sort()
    return {"mean": statistics.fmean(values),
            "lo": values[int(0.025 * reps)],
            "hi": values[int(0.975 * reps) - 1]}


# --------------------------------------------------------------- worker
_TRAIN = []
_ADDR_CACHE = {}
_ARM = None


def _worker_init(train_meta, arm_cfg, winner):
    global _TRAIN, _ARM
    import parameters as P
    for name, value in winner.items():
        setattr(P, name, value)
    setattr(P, "MATCH_EDGE_MIN_CREDIT", bool(arm_cfg["min_credit"]))
    _ARM = arm_cfg
    from frontend import load
    _TRAIN = [(m["class_"], m["name"], load(m["cache"])) for m in train_meta]


def _query_addrs(graph):
    """Compute addresses on the fly for the address-seed arm (M3)."""
    if not _ARM["use_addr"]:
        return None
    import parameters as P
    old = getattr(P, "FE_VECTOR_ADDRESS", False)
    P.FE_VECTOR_ADDRESS = True
    try:
        from frontend import _graph_addresses
        return _graph_addresses(graph.subgraph, graph.sigs)
    finally:
        P.FE_VECTOR_ADDRESS = old


def _run_test(item):
    from frontend import TEST_OFFSET, load, shift
    from matching.joint_mcs import joint_mcs

    _key, cache_path, true_class = item
    q = load(cache_path)
    q_addrs = _query_addrs(q)
    sub_b = shift(q.subgraph, TEST_OFFSET)
    sigs_b = {node + TEST_OFFSET: value for node, value in q.sigs.items()}
    b_addrs = None
    if q_addrs is not None:
        b_addrs = {k: {n + TEST_OFFSET: v for n, v in layer.items()}
                   for k, layer in q_addrs.items()}
    scores, stats, seconds = {}, {}, []
    for cls, name, train in _TRAIN:
        t0 = time.perf_counter()
        a_addrs = _query_addrs(train) if _ARM["use_addr"] else None
        result = joint_mcs(train.subgraph, sub_b, train.sigs, sigs_b,
                           addrs_a=a_addrs, addrs_b=b_addrs)
        seconds.append(time.perf_counter() - t0)
        if len(result.mapping) != len(set(result.mapping.values())):
            raise AssertionError("correspondence is not one-to-one")
        scores.setdefault(cls, []).append(result.score)
        stats.setdefault(cls, []).append({
            "train": name, "matched_nodes": len(result.mapping),
            "matched_edges": result.matched_edges,
            "graded_edges": result.graded_edges,
            "presence_only": result.presence_only_edges})
    return {"test": q.name, "class": true_class, "scores": scores,
            "stats": stats, "seconds": seconds,
            "total_seconds": math.fsum(seconds)}


def _run_arm(name, items, workers):
    arm_cfg = {"min_credit": ARMS[name][1], "use_addr": ARMS[name][2]}
    train_meta = items["train_meta"]
    tasks = items["tasks"]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_worker_init,
                  initargs=(train_meta, arm_cfg, WINNER)) as pool:
        for row in pool.imap_unordered(_run_test, tasks, chunksize=1):
            results.append(row)
            print("  %s %3d/%d  %s" % (name, len(results), len(tasks),
                                       row["test"]), flush=True)
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r["test"])

    cells = {}
    for row in results:
        for cls in corpus.CLASSES:
            cells.setdefault(row["class"], {}).setdefault(cls, []).extend(
                row["scores"][cls])
    predictions, correct = [], 0
    for row in results:
        scores = {c: max(row["scores"][c]) for c in corpus.CLASSES}
        predicted = max(corpus.CLASSES,
                        key=lambda c: (scores[c], -corpus.CLASSES.index(c)))
        correct += predicted == row["class"]
        predictions.append({"test": row["test"], "true": row["class"],
                            "predicted": predicted,
                            "correct": predicted == row["class"],
                            "scores": scores,
                            "scores_per_stored_graph": row["scores"],
                            "match_seconds": row["total_seconds"]})
    # stacked score matrix for AUC: (query, class)
    n = len(results)
    mat = np.full((n, len(corpus.CLASSES)), -np.inf, dtype=np.float64)
    truth = []
    for i, row in enumerate(results):
        for ci, cls in enumerate(corpus.CLASSES):
            mat[i, ci] = max(row["scores"][cls])
        truth.append(corpus.CLASSES.index(row["class"]))
    truth = np.array(truth)
    t1 = correct / n
    auc = pair_auc(mat, truth)
    auc_ci = bootstrap_auc(mat, truth)
    digest = hashlib.sha256(json.dumps(
        [[p["test"], p["predicted"],
          [float(v).hex() for v in p["scores"].values()]]
         for p in sorted(predictions, key=lambda p: p["test"])],
        sort_keys=True).encode()).hexdigest()
    return {
        "arm": name, "flags": {"min_credit": arm_cfg["min_credit"],
                               "use_addr": arm_cfg["use_addr"],
                               "winner": WINNER},
        "n": n, "top1": t1, "correct": correct,
        "auc": auc, "auc_ci": auc_ci,
        "determinism_sha256": digest,
        "predictions": predictions,
        "timing": {"wall_seconds": wall,
                   "per_test_image": statistics.fmean(
                       [r["total_seconds"] for r in results])},
    }


# --------------------------------------------------------------- control
def _rewire_edges(edges, seed):
    import random
    from core.types import UNDIRECTED_EDGE_TYPES
    rng = random.Random(seed)
    und = [(min(a, b), max(a, b), t) for a, b, t in edges
           if t in UNDIRECTED_EDGE_TYPES]
    direct = [(a, b, t) for a, b, t in edges if t not in UNDIRECTED_EDGE_TYPES]
    out = set(und)
    for _ in range(20 * max(1, len(und))):
        if len(und) < 2:
            break
        i, j = rng.sample(range(len(und)), 2)
        a, b, t1 = und[i]
        c, d, t2 = und[j]
        if t1 != t2 or a == b or c == d:
            continue
        cand = [(a, d, t1), (c, b, t2)]
        if any(x == y for x, y, _ in cand):
            continue
        if any((min(x, y), max(x, y), t) in out and
               (min(x, y), max(x, y), t) not in ((min(a, b), max(a, b), t1),
                                                 (min(c, d), max(c, d), t2))
               for x, y, t in cand):
            continue
        before = set(out)
        out.discard((a, b, t1))
        out.discard((c, d, t2))
        out.update(cand)
        if len(out) != len(before):
            out = before
            continue
        und[i] = cand[0]
        und[j] = cand[1]
    by_type = {}
    for a, b, t in direct:
        by_type.setdefault(t, []).append((a, b, t))
    re_direct = set()
    for t, group in by_type.items():
        dests = [b for _a, b, _t in group]
        for _ in range(100):
            rng.shuffle(dests)
            if all(a != d for (a, _b, _t), d in zip(group, dests)):
                break
        for (a, b, _t), d in zip(group, dests):
            re_direct.add((a, d if a != d else b, t))
    return frozenset(out | re_direct)


def _control_worker(args):
    qmeta, refs, winner, min_credit, use_addr = args
    from frontend import load, _graph_addresses
    import parameters as P
    for name, value in winner.items():
        setattr(P, name, value)
    setattr(P, "MATCH_EDGE_MIN_CREDIT", bool(min_credit))
    from matching.joint_mcs import joint_mcs
    from dataclasses import replace as _replace
    q = load(qmeta["cache"])
    q_addrs = _graph_addresses(q.subgraph, q.sigs) if use_addr else None
    out = {m: np.full(len(corpus.CLASSES), -np.inf, dtype=np.float64)
           for m in ("real", "rewired", "no_edge")}
    for ref_meta in refs:
        r = load(ref_meta["cache"])
        targets = {"real": r}
        import zlib
        targets["rewired"] = _replace(
            r, subgraph=_replace(r.subgraph, edges=_rewire_edges(
                r.subgraph.edges,
                zlib.crc32(ref_meta["name"].encode()) & 0xFFFF),
                edge_params={}))
        targets["no_edge"] = _replace(
            r, subgraph=_replace(r.subgraph, edges=frozenset(),
                                 edge_params={}))
        for mode in ("real", "rewired", "no_edge"):
            g = targets[mode]
            a_addrs = _graph_addresses(g.subgraph, g.sigs) if use_addr else None
            res = joint_mcs(g.subgraph, q.subgraph, g.sigs, q.sigs,
                            addrs_a=a_addrs, addrs_b=q_addrs)
            ci = corpus.CLASSES.index(ref_meta["class_"])
            out[mode][ci] = max(out[mode][ci], res.score)
    for mode, vec in out.items():
        if not np.isfinite(vec).all():
            raise RuntimeError("%s: %s unscored" % (qmeta["name"], mode))
    return qmeta["name"], {m: vec.tolist() for m, vec in out.items()}


def _run_control(name, items, workers):
    arm_cfg = {"min_credit": ARMS[name][1], "use_addr": ARMS[name][2]}
    refs = items["train_meta"]
    queries = items["tasks"]
    # tasks are (name, cache, class_) tuples; the control worker wants dicts
    queries = [{"name": q[0], "cache": q[1], "class_": q[2]}
               if isinstance(q, tuple) else q for q in queries]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    acc = {m: [] for m in ("real", "rewired", "no_edge")}
    truth = []
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        tasks = [(q, refs, WINNER, arm_cfg["min_credit"], arm_cfg["use_addr"])
                 for q in queries]
        for qname, out in pool.imap_unordered(_control_worker, tasks,
                                               chunksize=1):
            for m in acc:
                acc[m].append(out[m])
            truth.append(corpus.CLASSES.index(
                next(q["class_"] for q in queries if q["name"] == qname)))
    truth = np.array(truth)
    table = {}
    for m in ("real", "rewired", "no_edge"):
        mat = np.array(acc[m])
        pred = mat.argmax(axis=1)
        t1 = float((pred == truth).mean())
        table[m] = {"top1": t1, "auc": pair_auc(mat, truth),
                    "auc_ci": bootstrap_auc(mat, truth)}
    return {"arm": name, "control": table,
            "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------- main
def load_variant(variant_file):
    with open(TUNE / variant_file) as fh:
        return json.load(fh)


def balanced_subset(entries, per_class):
    """Take up to `per_class` entries per class, preserving order, so every
    class with a reference keeps one.  Used for the reduced-protocol run so
    the 8-class AUC is defined for every arm."""
    if per_class <= 0:
        return list(entries)
    by_class = defaultdict(list)
    for e in entries:
        by_class[e["class_"]].append(e)
    out = []
    for cls in corpus.CLASSES:
        out.extend(by_class.get(cls, [])[:per_class])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="M0,M1,M2,M3")
    ap.add_argument("--control-arm", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-refs", type=int, default=0,
                    help="per-class reference count (0 = all)")
    ap.add_argument("--max-heldout", type=int, default=0,
                    help="per-class held-out query count (0 = all)")
    ap.add_argument("--skip-arms", action="store_true",
                    help="skip arm computation; load existing t3_arms.json")
    args = ap.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    sub = (args.max_refs, args.max_heldout)

    report = {"protocol": "Task 3 PART C end-to-end measurement",
              "winner_cell": WINNER, "corpus": "ETH-80 instance-disjoint",
              "subset": {"max_refs": sub[0], "max_heldout": sub[1]}}

    # ---- graph census + build time (from the variant indexes) ----
    census = {}
    for name in arms:
        variant_file = ARMS[name][0]
        data = load_variant(variant_file)
        nodes = [g["nodes"] for g in data["graphs"]]
        edges = [g["edges"] for g in data["graphs"]]
        census[name] = {
            "variant": variant_file, "n_graphs": len(data["graphs"]),
            "mean_nodes": statistics.fmean(nodes),
            "mean_edges": statistics.fmean(edges),
            "min_edges": min(edges), "max_edges": max(edges),
            "build_seconds": data.get("build_seconds"),
        }
    report["graph_census"] = census

    results = {}
    if args.skip_arms and (TUNE / "t3_arms.json").exists():
        with open(TUNE / "t3_arms.json") as fh:
            results = json.load(fh).get("arms", {})
        print("loaded existing arm results:", sorted(results), flush=True)
    if not args.skip_arms:
        for name in arms:
            variant_file = ARMS[name][0]
            data = load_variant(variant_file)
            train_meta = data["references"]
            test_meta = data["heldout"]
            if sub[0]:
                train_meta = balanced_subset(train_meta, sub[0])
            if sub[1]:
                test_meta = balanced_subset(test_meta, sub[1])
            items = {"train_meta": train_meta,
                     "tasks": [(m["name"], m["cache"], m["class_"]) for m in test_meta]}
            print("arm", name, "variant", variant_file,
                  "refs", len(train_meta), "heldout", len(test_meta), flush=True)
            results[name] = _run_arm(name, items, args.workers)
            print("arm %-3s top-1 %.4f (%d/%d)  AUC %.4f"
                  % (name, results[name]["top1"], results[name]["correct"],
                     results[name]["n"], results[name]["auc"]), flush=True)
            write_json(TUNE / "t3_arms.json",
                       {"arms": results, "census": census, "winner": WINNER})

    # ---- control on the best arm ----
    if args.control_arm:
        best = args.control_arm
    else:
        best = max(results, key=lambda a: (results[a]["auc"],
                                           results[a]["top1"]))
    print("control arm:", best, flush=True)
    variant_file = ARMS[best][0]
    data = load_variant(variant_file)
    items = {"train_meta": data["references"],
             "tasks": [(m["name"], m["cache"], m["class_"])
                       for m in data["heldout"]]}
    if sub[0]:
        items["train_meta"] = balanced_subset(items["train_meta"], sub[0])
    if sub[1]:
        items["tasks"] = balanced_subset(
            [{"name": t[0], "cache": t[1], "class_": t[2]} for t in items["tasks"]],
            sub[1])
        items["tasks"] = [(t["name"], t["cache"], t["class_"]) for t in items["tasks"]]
    report["control"] = _run_control(best, items, args.workers)
    report["best_arm"] = best
    report["arms"] = results
    write_json(TUNE / "t3_measure.json", report)
    print("wrote", TUNE / "t3_measure.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
