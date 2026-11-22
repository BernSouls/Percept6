"""Tune the structural matcher on an instance-disjoint ETH-80 split.

Pipeline
--------
reference graphs (3 instances/class)  --stored-->
query graph  --global_consistency_alignment-->  cached correspondence
             --score formula (from the cache)-->  S(query, class)
             argmax = prediction; pair-level AUC over all class pairs.

The alignment is solved once per (pair, alignment cell) and the score formula
is swept from the cached facts, so the 6 480-cell score surface costs no
re-matching.  The grid and the selection rule are pre-registered in
``TUNING_PREREGISTRATION.md``.

Modes
-----
    python tune.py --build                # 120 graphs, cache
    python tune.py --sweep --workers 12   # dev surface
    python tune.py --combine              # stack dev scores
    python tune.py --report-dev           # surface CSV + top cells
    python tune.py --heldout --winner results/tune/winner.json
    python tune.py --det --winner ... --tag seed0
    python tune.py --compare-det A.npz B.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import corpus  # noqa: E402
import solver  # noqa: E402

TUNE = RESULTS_DIR / "tune"
TUNE.mkdir(parents=True, exist_ok=True)

BETAS = (0.0, 0.15, 0.3, 0.5, 0.75)
UNMATCHED = (0.0, 0.15, 0.3, "derived")
OCCLUSIONS = (0.0, 0.25, 0.5)
WS = (0.1, 0.25, 0.5, 0.75)
GAMMAS = (0.0, 0.5, 1.0)
DELTAS = (0.0, 0.5, 1.0)
DENOMS = ("all", "checkable", "per_side")
ISOLATED = ("keep", "downweight", "exclude")

ALIGN_CONFIGS = [(b, u, o) for b in BETAS for u in UNMATCHED for o in OCCLUSIONS]
SCORE_CONFIGS = [
    {"formula": "pooled", "w": w, "gamma": g, "delta": 0.0,
     "denom": d, "isolated": i, "presence": True, "reference": False,
     "name": "pooled_w%.2f_g%.1f_%s_%s" % (w, g, d, i)}
    for w in WS for g in GAMMAS for d in DENOMS for i in ISOLATED
] + [
    {"formula": "normalized", "w": w, "gamma": g, "delta": dlt,
     "denom": "per_side", "isolated": i, "presence": True,
     "reference": False,
     "name": "norm_w%.2f_g%.1f_d%.1f_%s" % (w, g, dlt, i)}
    for w in WS for g in GAMMAS for dlt in DELTAS for i in ISOLATED
] + [
    {"formula": "pooled", "w": 0.0, "gamma": 1.0, "delta": 0.0,
     "denom": "all", "isolated": "keep", "presence": True,
     "reference": True, "name": "node_only"},
    {"formula": "normalized", "w": 0.0, "gamma": 1.0, "delta": 0.0,
     "denom": "per_side", "isolated": "keep", "presence": True,
     "reference": True, "name": "norm_node_only_d0"},
    {"formula": "normalized", "w": 0.0, "gamma": 1.0, "delta": 1.0,
     "denom": "per_side", "isolated": "keep", "presence": True,
     "reference": True, "name": "norm_node_only_d1"},
    {"formula": "pooled", "w": 0.5, "gamma": 1.0, "delta": 0.0,
     "denom": "all", "isolated": "keep", "presence": False,
     "reference": True, "name": "presence_excluded"},
]

WORKER_INIT = None


def _init():
    guard.activate()


# --------------------------------------------------------------------- build
def _build_one(entry):
    from frontend import build_cached
    return build_cached(entry)


def build(workers: int):
    split = corpus.split_tuning()
    entries = split["references"] + split["dev"] + split["heldout"]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    metas = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(_build_one, entries, chunksize=1)):
            if isinstance(rec, dict) and "error" in rec:
                raise RuntimeError(rec["error"])
            metas.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(entries):
                print("  front end %3d/%d (%.0fs)" % (i + 1, len(entries),
                                                      time.perf_counter() - t0),
                      flush=True)
    metas.sort(key=lambda m: m["name"])
    by_name = {m["name"]: m for m in metas}
    refs = [by_name[corpus.name_of(e)] for e in split["references"]]
    dev = [by_name[corpus.name_of(e)] for e in split["dev"]]
    held = [by_name[corpus.name_of(e)] for e in split["heldout"]]
    out = {"split": split, "graphs": metas, "references": refs,
           "dev": dev, "heldout": held, "counts": split["counts"],
           "build_seconds": time.perf_counter() - t0}
    write_json(TUNE / "graphs.json", out)
    print("wrote", TUNE / "graphs.json", flush=True)
    return 0


# --------------------------------------------------------------------- sweep
def _load_graphs():
    data = json.loads((TUNE / "graphs.json").read_text(encoding="utf-8"))
    return data


def _ref_groups(metas):
    groups = {c: [] for c in corpus.CLASSES}
    for m in metas:
        groups[m["class_"]].append(m)
    return {c: sorted(v, key=lambda m: m["name"]) for c, v in groups.items()}


def _transpose(sim):
    return [list(row) for row in zip(*sim)]


def sweep_worker(args):
    qmeta, refs, align_cfgs, score_cfgs = args
    from frontend import load
    q = load(qmeta["cache"])
    # -inf, not NaN: Python's max(nan, x) keeps nan, so a NaN initialiser
    # would silently mask every class.  A class with no reference stays
    # -inf and the assertion below refuses to write the query.
    out = np.full((len(align_cfgs) * len(score_cfgs), len(corpus.CLASSES)),
                  -np.inf, dtype=np.float32)
    for ref_meta in refs:
        r = load(ref_meta["cache"])
        sim = solver.node_sim_matrix(r, q, r.sigs, q.sigs)
        sim_t = _transpose(sim)
        cache = {}
        for ai, (beta, uc, occ) in enumerate(align_cfgs):
            key = (beta, uc) if beta == 0.0 else (beta, uc, occ)
            if key not in cache:
                um = (solver.derived_unmatched_cost(sim)
                      if uc == "derived" else float(uc))
                fwd = solver.solve_direction(r, q, r.sigs, q.sigs, sim,
                                             beta=beta, unmatched_cost=um,
                                             unmapped_credit=occ)
                rev = solver.solve_direction(q, r, q.sigs, r.sigs, sim_t,
                                             beta=beta, unmatched_cost=um,
                                             unmapped_credit=occ)
                cache[key] = (solver.facts(r, q, r.sigs, q.sigs, sim, fwd),
                              solver.facts(q, r, q.sigs, r.sigs, sim_t, rev))
            fa, fb = cache[key]
            for si, sc in enumerate(score_cfgs):
                a = solver.score(fa, w=sc["w"], gamma=sc["gamma"],
                                 denom_mode=sc["denom"], isolated=sc["isolated"],
                                 presence_counts=sc["presence"],
                                 formula=sc["formula"], delta=sc["delta"])[0]
                b = solver.score(fb, w=sc["w"], gamma=sc["gamma"],
                                 denom_mode=sc["denom"], isolated=sc["isolated"],
                                 presence_counts=sc["presence"],
                                 formula=sc["formula"], delta=sc["delta"])[0]
                cell = ai * len(score_cfgs) + si
                ci = corpus.CLASSES.index(ref_meta["class_"])
                out[cell, ci] = max(out[cell, ci], max(a, b))
    present = {ref["class_"] for ref in refs}
    masked = [(ci, corpus.CLASSES[ci]) for ci in range(len(corpus.CLASSES))
              if corpus.CLASSES[ci] in present
              and not np.isfinite(out[:, ci]).all()]
    if masked:
        raise RuntimeError("%s: masked class cells %r (a class has a reference "
                           "but no finite score)" % (qmeta["name"], masked))
    for ci, cls in enumerate(corpus.CLASSES):
        if cls not in present and np.isfinite(out[:, ci]).any():
            raise RuntimeError("%s: class %s has a score but no reference"
                               % (qmeta["name"], cls))
    path = TUNE / "dev" / (qmeta["name"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, scores=out)
    tmp.replace(path)
    return qmeta["name"], float(out.max())


def sweep(workers: int, resume: bool = True):
    data = _load_graphs()
    refs = [m for g in _ref_groups(data["references"]).values() for m in g]
    refs.sort(key=lambda m: m["name"])
    queries = sorted(data["dev"], key=lambda m: m["name"])
    todo = [q for q in queries
            if not (resume and (TUNE / "dev" / (q["name"] + ".npz")).is_file())]
    print("dev queries: %d, to run: %d" % (len(queries), len(todo)), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init) as pool:
        tasks = [(q, refs, ALIGN_CONFIGS, SCORE_CONFIGS) for q in todo]
        for i, (name, hi) in enumerate(pool.imap_unordered(sweep_worker, tasks,
                                                           chunksize=1)):
            print("  %3d/%d %-18s (%.0fs)" % (i + 1, len(todo), name,
                                              time.perf_counter() - t0), flush=True)
    elapsed = time.perf_counter() - t0
    write_json(TUNE / "sweep_receipt.json",
               {"queries": len(todo), "align_cells": len(ALIGN_CONFIGS),
                "score_cells": len(SCORE_CONFIGS), "workers": workers,
                "seconds": elapsed,
                "pair_solves": len(todo) * len(refs) * len(ALIGN_CONFIGS),
                "score_evaluations": len(todo) * len(refs)
                * len(ALIGN_CONFIGS) * len(SCORE_CONFIGS)})
    print("sweep %.0fs" % elapsed, flush=True)
    return 0


def combine():
    data = _load_graphs()
    queries = sorted(data["dev"], key=lambda m: m["name"])
    stack = []
    for q in queries:
        with np.load(TUNE / "dev" / (q["name"] + ".npz")) as z:
            stack.append(z["scores"])
    scores = np.stack(stack, axis=1)          # (score cell, query, class)
    masked = ~np.isfinite(scores)
    if masked.any():
        detail = []
        for qi, q in enumerate(queries):
            classes = [corpus.CLASSES[ci] for ci in range(len(corpus.CLASSES))
                       if masked[:, qi, ci].any()]
            if classes:
                detail.append({"query": q["name"], "classes": classes,
                               "cells": int(masked[:, qi, :].sum())})
        write_json(TUNE / "masked_cells.json", detail)
        raise RuntimeError("%d masked query x class cells: %s"
                           % (int(masked.sum()), detail[:5]))
    np.savez_compressed(TUNE / "dev_scores.npz", scores=scores,
                        queries=np.array([q["name"] for q in queries]),
                        classes=np.array(corpus.CLASSES),
                        truth=np.array([corpus.CLASSES.index(q["class_"])
                                        for q in queries]))
    write_json(TUNE / "manifest.json",
               {"align": [list(c) for c in ALIGN_CONFIGS],
                "score": SCORE_CONFIGS,
                "queries": [q["name"] for q in queries],
                "truth": [q["class_"] for q in queries]})
    print("combined", scores.shape)
    return 0


# ------------------------------------------------------------------- metrics
def top1(scores, truth):
    pred = scores.argmax(axis=-1)
    return float((pred == truth).mean()), pred


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


def _surface():
    with np.load(TUNE / "dev_scores.npz") as z:
        scores = z["scores"]
        truth = z["truth"]
    rows = []
    # scores is indexed [align_cell * n_score + score_cell, query, class]
    for ai, (beta, uc, occ) in enumerate(ALIGN_CONFIGS):
        for ci, sc in enumerate(SCORE_CONFIGS):
            s = scores[ai * len(SCORE_CONFIGS) + ci]
            t1, _ = top1(s, truth)
            rows.append({"beta": beta, "unmatched": uc, "occlusion": occ,
                         "formula": sc["formula"], "w": sc["w"],
                         "gamma": sc["gamma"], "delta": sc["delta"],
                         "denom": sc["denom"], "isolated": sc["isolated"],
                         "presence": sc["presence"], "reference": sc["reference"],
                         "name": sc["name"],
                         "top1": t1, "auc": pair_auc(s, truth)})
    return rows


def report_dev():
    rows = _surface()
    import csv
    path = TUNE / "dev_surface.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = [r for r in rows if not r["reference"]]
    best = max(candidates, key=lambda r: (r["auc"], r["top1"]))
    print("rows", len(rows), "best", json.dumps(best, sort_keys=True))
    print("\nTop 15 by dev AUC:")
    for r in sorted(candidates, key=lambda r: (-r["auc"], -r["top1"]))[:15]:
        print("  auc %.4f top1 %.4f  beta %.2f uc %s occ %.2f %s w %.2f "
              "gamma %.1f delta %.1f %s %s"
              % (r["auc"], r["top1"], r["beta"], r["unmatched"],
                 r["occlusion"], r["formula"], r["w"], r["gamma"],
                 r["delta"], r["denom"], r["isolated"]))
    # paired bootstrap by query frame for the selected cell and the node-only row
    with np.load(TUNE / "dev_scores.npz") as z:
        scores, truth = z["scores"], z["truth"]

    def cell_scores(r):
        ai = ALIGN_CONFIGS.index((r["beta"], r["unmatched"], r["occlusion"]))
        si = next(i for i, sc in enumerate(SCORE_CONFIGS)
                  if sc["name"] == r["name"])
        return scores[ai * len(SCORE_CONFIGS) + si]

    best["auc_ci"] = bootstrap_auc(cell_scores(best), truth)
    write_json(TUNE / "winner.json", best)
    # node-only reference, best alignment cell
    node = [r for r in rows if r["name"] == "node_only"]
    best_node = max(node, key=lambda r: (r["auc"], r["top1"]))
    best_node["auc_ci"] = bootstrap_auc(cell_scores(best_node), truth)
    write_json(TUNE / "node_only.json", best_node)
    print("\nwinner:", json.dumps(best, sort_keys=True))
    print("node-only reference:", json.dumps(best_node, sort_keys=True))
    return 0


# ------------------------------------------------------------ winner-only run
def _rewire_edges(edges, seed):
    """Degree-preserving-ish rewiring: undirected double swaps / directed
    destination permutation.  Edge parameters are dropped by the caller."""
    from core.types import UNDIRECTED_EDGE_TYPES
    rng = random.Random(seed)
    # [DETERMINISM] `edges` is usually a frozenset, whose iteration order
    # depends on string hashing (salted PER PROCESS), so feeding it to a
    # seeded RNG made the rewiring -- and hence the W arm -- differ between
    # runs. Measured: the same ref produced 350 vs 351 edges and different
    # sets across processes. Sort canonically first so `seed` means something.
    _edges = sorted(edges, key=lambda e: (e[0], e[1], str(e[2])))
    und = [(min(a, b), max(a, b), t) for a, b, t in _edges
           if t in UNDIRECTED_EDGE_TYPES]
    direct = [(a, b, t) for a, b, t in _edges if t not in UNDIRECTED_EDGE_TYPES]
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
        if len(out) != len(before):       # a swap must not merge two edges
            out = before
            continue
        und[i] = cand[0]
        und[j] = cand[1]
    by_type = {}
    for a, b, t in direct:
        by_type.setdefault(t, []).append((a, b, t))
    re_direct = set()
    for t, group in by_type.items():
        original = [b for _a, b, _t in group]
        dests = list(original)
        for _ in range(100):
            rng.shuffle(dests)
            if all(a != d for (a, _b, _t), d in zip(group, dests)):
                break
        for (a, b, _t), d in zip(group, dests):
            # a self-loop has no other node to point at; keep the original
            # endpoint so the edge count (and therefore the ablation's mass)
            # is unchanged.
            re_direct.add((a, d if a != d else b, t))
    return frozenset(out | re_direct)


def _rewired_graph(g, seed):
    sg = g.subgraph
    edges = _rewire_edges(sg.edges, seed)
    return SimpleNamespace(subgraph=replace(sg, edges=edges, edge_params={}),
                           sigs=g.sigs)


def _without_edges(g):
    sg = g.subgraph
    return SimpleNamespace(subgraph=replace(sg, edges=frozenset(), edge_params={}),
                           sigs=g.sigs)


def winner_worker(args):
    qmeta, refs, winner, modes = args
    from frontend import load
    q = load(qmeta["cache"])
    out = {m: np.full(len(corpus.CLASSES), -np.inf, dtype=np.float32)
           for m in modes}
    beta = float(winner["beta"])
    uc = winner["unmatched"]
    occ = float(winner["occlusion"])
    w = float(winner["w"])
    gamma = float(winner["gamma"])
    delta = float(winner.get("delta", 0.0))
    formula = winner.get("formula", "pooled")
    denom = winner["denom"]
    isolated = winner["isolated"]
    presence = bool(winner["presence"])
    for ref_meta in refs:
        r = load(ref_meta["cache"])
        targets = {"real": r}
        if "rewired" in modes:
            targets["rewired"] = _rewired_graph(
                r, seed=zlib.crc32(ref_meta["name"].encode()) & 0xFFFF)
        if "no_edge" in modes:
            targets["no_edge"] = _without_edges(r)
        for mode in modes:
            # node_only is the real mapping scored with w=0, so it has no
            # graph of its own.
            g = targets["real"] if mode == "node_only" else targets[mode]
            sim = solver.node_sim_matrix(g, q, g.sigs, q.sigs)
            sim_t = _transpose(sim)
            um = (solver.derived_unmatched_cost(sim)
                  if uc == "derived" else float(uc))
            fwd = solver.solve_direction(g, q, g.sigs, q.sigs, sim,
                                         beta=beta, unmatched_cost=um,
                                         unmapped_credit=occ)
            rev = solver.solve_direction(q, g, q.sigs, g.sigs, sim_t,
                                         beta=beta, unmatched_cost=um,
                                         unmapped_credit=occ)
            fa = solver.facts(g, q, g.sigs, q.sigs, sim, fwd)
            fb = solver.facts(q, g, q.sigs, g.sigs, sim_t, rev)
            use_w = 0.0 if mode in ("node_only", "no_edge") else w
            a = solver.score(fa, w=use_w, gamma=gamma, denom_mode=denom,
                             isolated=isolated, presence_counts=presence,
                             formula=formula, delta=delta)[0]
            b = solver.score(fb, w=use_w, gamma=gamma, denom_mode=denom,
                             isolated=isolated, presence_counts=presence,
                             formula=formula, delta=delta)[0]
            out[mode][corpus.CLASSES.index(ref_meta["class_"])] = max(
                out[mode][corpus.CLASSES.index(ref_meta["class_"])], max(a, b))
    for mode, vec in out.items():
        if not np.isfinite(vec).all():
            raise RuntimeError("%s: %s has an unscored class (no reference for "
                               "it)" % (qmeta["name"], mode))
    path = TUNE / "winner" / (qmeta["name"] + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    return qmeta["name"]


def winner_run(split: str, workers: int, tag: str, modes=("real",)):
    data = _load_graphs()
    winner = json.loads((TUNE / "winner.json").read_text(encoding="utf-8"))
    refs = [m for g in _ref_groups(data["references"]).values() for m in g]
    refs.sort(key=lambda m: m["name"])
    queries = sorted(data[split], key=lambda m: m["name"])
    todo = [q for q in queries
            if not (TUNE / "winner" / (q["name"] + ".npz")).is_file()]
    print("%s queries: %d, to run: %d" % (split, len(queries), len(todo)),
          flush=True)
    # the winner's node-only row is the node_only score at w=0 on the real mapping
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init) as pool:
        tasks = [(q, refs, winner, modes) for q in todo]
        for i, name in enumerate(pool.imap_unordered(winner_worker, tasks,
                                                     chunksize=1)):
            print("  %s %3d/%d %s (%.0fs)" % (split, i + 1, len(queries), name,
                                              time.perf_counter() - t0),
                  flush=True)
    stack = {m: [] for m in modes}
    truth = []
    for q in queries:
        with np.load(TUNE / "winner" / (q["name"] + ".npz")) as z:
            for m in modes:
                stack[m].append(z[m])
        truth.append(corpus.CLASSES.index(q["class_"]))
    out = {m: np.stack(v) for m, v in stack.items()}
    out["truth"] = np.array(truth)
    path = TUNE / ("%s_%s.npz" % (split, tag))
    np.savez_compressed(path, **out)
    print("wrote", path)
    return 0


def report_winner(path, label):
    with np.load(path) as z:
        truth = z["truth"]
        table = {}
        for m in ("real", "node_only", "rewired", "no_edge"):
            if m in z:
                t1, _ = top1(z[m], truth)
                table[m] = {"top1": t1, "auc": pair_auc(z[m], truth),
                            "auc_ci": bootstrap_auc(z[m], truth)}
        real = z["real"]
    print(label, json.dumps(table, indent=2, sort_keys=True), flush=True)
    write_json(Path(path).with_name(Path(path).stem + "_metrics.json"), table)
    return table


# ------------------------------------------------------------------- compare
def _prediction_hash(path):
    with np.load(path) as z:
        real = z["real"]
    predicted = real.argmax(axis=1)
    payload = [[float(v).hex() for v in row] for row in real]
    return hashlib.sha256(json.dumps([payload, predicted.tolist()],
                                     sort_keys=True).encode()).hexdigest()


def compare_det(a, b):
    ha, hb = _prediction_hash(a), _prediction_hash(b)
    out = {"a": str(a), "b": str(b), "hash_a": ha, "hash_b": hb,
           "match": ha == hb}
    write_json(TUNE / "determinism.json", out)
    print(json.dumps(out, indent=2))
    return 0 if ha == hb else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--report-dev", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="seed0")
    ap.add_argument("--compare-det", nargs=2)
    ap.add_argument("--winner", default=str(TUNE / "winner.json"))
    args = ap.parse_args(argv)
    if args.compare_det:
        return compare_det(*args.compare_det)
    if args.build:
        return build(args.workers)
    if args.sweep:
        return sweep(args.workers)
    if args.combine:
        return combine()
    if args.report_dev:
        return report_dev()
    if args.heldout:
        rc = winner_run("heldout", args.workers, args.tag,
                        modes=("real", "node_only", "rewired", "no_edge"))
        report_winner(TUNE / ("heldout_%s.npz" % args.tag), "held-out")
        return rc
    if args.det:
        rc = winner_run("dev", args.workers, args.tag, modes=("real",))
        report_winner(TUNE / ("dev_%s.npz" % args.tag), "dev-det")
        return rc
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
