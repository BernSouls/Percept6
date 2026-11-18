"""Arm runner for structural-topology classification on the pinned ETH-80 split.

Pre-registration: ``work/topology/PREREGISTRATION.md``, written before any
accuracy number existed.

    image -> [this tree's front end] -> situation window
          -> attach_named_features (attributes as named features)
          -> select_perceptual_graph (512-edge structural budget)
          -> a stored graph                             <-- "training"

    test image -> the same build
    score(test, class) = MAX over that class's stored graphs of
                         joint_mcs(stored, test).score
    predict argmax.  Chance = 1/8.  No label reaches the matcher.

Usage:
    python classify.py --build                 # build/refresh the graph cache
    python classify.py --seed-name seed0       # all five arms
    python classify.py --compare A.json B.json
    python classify.py --report A.json
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
from pathlib import Path

import guard
from guard import RESULTS_DIR, write_json

import corpus

CHANCE = 1.0 / len(corpus.CLASSES)
TOP_K = 3

#: every arm is a value of the three matcher flags, nothing else.
ARMS = {
    "baseline": {"MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": False,
                 "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
                 "MATCH_SEPARATE_TERM_NORMALIZATION": False},
    "F1": {"MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
           "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
           "MATCH_SEPARATE_TERM_NORMALIZATION": False},
    "F3": {"MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": False,
           "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": True,
           "MATCH_SEPARATE_TERM_NORMALIZATION": False},
    "F1+F3": {"MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
              "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": True,
              "MATCH_SEPARATE_TERM_NORMALIZATION": False},
    "separate_normalization": {"MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": False,
                               "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
                               "MATCH_SEPARATE_TERM_NORMALIZATION": True},
}
ARM_ORDER = ("baseline", "F1", "F3", "F1+F3", "separate_normalization")

# ------------------------------------------------------------------ workers

_TRAIN = []
_PROVENANCE = {}
_TRIPWIRE = None


def _build_init():
    guard.activate()


def _build_one(entry):
    from frontend import build_cached
    return build_cached(entry)


def _worker_init(train_meta, flags):
    global _TRAIN, _PROVENANCE, _TRIPWIRE
    guard.activate()
    import parameters as P
    for name, value in flags.items():
        if not hasattr(P, name):
            raise RuntimeError("no such matcher parameter: %s" % name)
        setattr(P, name, bool(value))
    # the matcher imports _sig_similarity from memory.cores lazily; load it now
    # so that the banned-symbol scan below actually sees that module.
    guard.load("memory.cores")
    guard.load("matching.joint_mcs")
    paths = guard.assert_product_paths()
    banned = guard.assert_banned_absent()
    from frontend import load
    _TRAIN = [(m["class_"], m["name"], load(m["cache"])) for m in train_meta]
    resolved = {name: bool(getattr(P, name)) for name in ARMS["baseline"]}
    if resolved != {k: bool(v) for k, v in flags.items()}:
        raise RuntimeError("flags did not resolve: %r != %r" % (resolved, flags))
    _TRIPWIRE = guard.tripwire().__enter__()      # armed for the whole process
    _PROVENANCE = {
        "pid": os.getpid(),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "flags": resolved,
        "banned": banned,
        "armed": list(_TRIPWIRE.armed),
        "product_modules": len(paths),
        "digest": hashlib.sha256(json.dumps(
            {"paths": paths, "banned": banned, "flags": resolved},
            sort_keys=True).encode()).hexdigest(),
    }


def _run_test(item):
    """One test image against every stored graph."""
    from frontend import TEST_OFFSET, load, shift
    from matching.joint_mcs import joint_mcs

    _key, cache_path, true_class = item
    test_graph = load(cache_path)
    sub_b = shift(test_graph.subgraph, TEST_OFFSET)
    sigs_b = {node + TEST_OFFSET: value for node, value in test_graph.sigs.items()}
    scores, stats, seconds = {}, {}, []
    for cls, name, train in _TRAIN:
        t0 = time.perf_counter()
        result = joint_mcs(train.subgraph, sub_b, train.sigs, sigs_b)
        seconds.append(time.perf_counter() - t0)
        if len(result.mapping) != len(set(result.mapping.values())):
            raise AssertionError("correspondence is not one-to-one")
        scores.setdefault(cls, []).append(result.score)
        stats.setdefault(cls, []).append({
            "train": name, "matched_nodes": len(result.mapping),
            "matched_edges": result.matched_edges,
            "graded_edges": result.graded_edges,
            "graded_sum": result.graded_edge_similarity_sum,
            "presence_only": result.presence_only_edges,
            "budget_exhausted": bool(result.budget_exhausted)})
    return {"test": test_graph.name, "class": true_class,
            "object": test_graph.object_, "view": test_graph.view,
            "scores": scores, "stats": stats, "seconds": seconds,
            "total_seconds": math.fsum(seconds),
            "provenance": _PROVENANCE["digest"], "pid": os.getpid()}


# ------------------------------------------------------------- aggregation

def _quantiles(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    p95 = ordered[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    return {"n": n, "mean": statistics.fmean(values),
            "median": statistics.median(values), "p95": p95,
            "min": ordered[0], "max": ordered[-1],
            "total": math.fsum(values)}


def _predict(scores):
    """argmax over the class maximum; ties go to the earlier class name."""
    best = max(corpus.CLASSES,
               key=lambda c: (scores[c], -corpus.CLASSES.index(c)))
    return best


def _top_k_mean(values, k=TOP_K):
    return statistics.fmean(sorted(values, reverse=True)[:k])


def _run_arm(name, items, workers):
    flags = ARMS[name]
    train_meta = items["train_meta"]
    tasks = items["tasks"]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_worker_init,
                  initargs=(train_meta, flags)) as pool:
        for row in pool.imap_unordered(_run_test, tasks, chunksize=1):
            results.append(row)
            print("  %-22s %3d/%d  %s" % (name, len(results), len(tasks),
                                          row["test"]), flush=True)
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r["test"])

    cells, stats = {}, {}
    for row in results:
        for cls in corpus.CLASSES:
            cells.setdefault(row["class"], {}).setdefault(cls, []).extend(
                row["scores"][cls])
            stats.setdefault(row["class"], {}).setdefault(cls, []).extend(
                row["stats"][cls])
    predictions, per_class, confusion = [], {}, {c: {d: 0 for d in corpus.CLASSES}
                                                for c in corpus.CLASSES}
    correct = secondary = 0
    for row in results:
        scores = {c: max(row["scores"][c]) for c in corpus.CLASSES}
        predicted = _predict(scores)
        top3 = {c: _top_k_mean(row["scores"][c]) for c in corpus.CLASSES}
        predicted_3 = _predict(top3)
        correct += predicted == row["class"]
        secondary += predicted_3 == row["class"]
        confusion[row["class"]][predicted] += 1
        per_class.setdefault(row["class"], {"n": 0, "correct": 0})
        per_class[row["class"]]["n"] += 1
        per_class[row["class"]]["correct"] += predicted == row["class"]
        predictions.append({
            "test": row["test"], "true": row["class"], "predicted": predicted,
            "correct": predicted == row["class"],
            "predicted_top3": predicted_3,
            "scores": scores, "scores_per_stored_graph": row["scores"],
            "match_seconds": row["total_seconds"]})

    cell_means = {t: {c: statistics.fmean(cells[t][c]) for c in corpus.CLASSES}
                  for t in corpus.CLASSES}
    cell_stats = {}
    for t in corpus.CLASSES:
        cell_stats[t] = {}
        for c in corpus.CLASSES:
            rows = stats[t][c]
            graded = sum(r["graded_edges"] for r in rows)
            graded_sum = math.fsum(r["graded_sum"] for r in rows)
            cell_stats[t][c] = {
                "comparisons": len(rows),
                "mean_matched_nodes": statistics.fmean(r["matched_nodes"] for r in rows),
                "mean_matched_edges": statistics.fmean(r["matched_edges"] for r in rows),
                "graded_edges": graded,
                "mean_graded_edge_similarity": (graded_sum / graded) if graded else None,
                "presence_only_edges": sum(r["presence_only"] for r in rows),
                "budget_exhausted": sum(r["budget_exhausted"] for r in rows),
            }

    digest_rows = []
    for p in sorted(predictions, key=lambda p: p["test"]):
        per_cell = []
        for cls in corpus.CLASSES:
            per_cell.append([cls, [float(v).hex()
                                   for v in p["scores_per_stored_graph"][cls]]])
        digest_rows.append([p["test"], per_cell, p["predicted"]])
    return {
        "arm": name, "flags": flags, "n": len(results),
        "accuracy": correct / len(results), "correct": correct,
        "secondary_top3_accuracy": secondary / len(results),
        "chance": CHANCE,
        "confusion": confusion,
        "confusion_rows": "true class", "confusion_cols": "predicted class",
        "per_class": {c: {"n": v["n"], "correct": v["correct"],
                          "accuracy": v["correct"] / v["n"]}
                      for c, v in sorted(per_class.items())},
        "predictions": predictions,
        "score_cell_means": cell_means,
        "cell_stats": cell_stats,
        "determinism_sha256": hashlib.sha256(
            json.dumps(digest_rows, sort_keys=True).encode()).hexdigest(),
        "timing": {
            "wall_seconds": wall,
            "per_match": _quantiles([s for r in results for s in r["seconds"]]),
            "per_test_image": _quantiles([r["total_seconds"] for r in results]),
            "matches_per_test_image": len(train_meta),
        },
        "provenance_digests": sorted({r["provenance"] for r in results}),
        "worker_pids": sorted({r["pid"] for r in results}),
    }


# --------------------------------------------------------------------- build

def ensure_graphs(workers):
    split = corpus.split()
    entries = split["train"] + split["test"]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    metas = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_build_init) as pool:
        for i, record in enumerate(pool.imap_unordered(_build_one, entries,
                                                      chunksize=1)):
            if isinstance(record, dict) and "error" in record:
                raise RuntimeError(record["error"])
            metas.append(record)
            if (i + 1) % 10 == 0 or i + 1 == len(entries):
                print("  front end %3d/%d  (%.0fs)"
                      % (i + 1, len(entries), time.perf_counter() - t0), flush=True)
    metas.sort(key=lambda m: m["name"])
    by_name = {m["name"]: m for m in metas}
    train_meta = [by_name[corpus.name_of(e)] for e in split["train"]]
    test_meta = [by_name[corpus.name_of(e)] for e in split["test"]]
    if len(metas) != len(entries):
        raise RuntimeError("%d graphs for %d entries" % (len(metas), len(entries)))
    return split, train_meta, test_meta, metas


def _graph_summary(metas):
    def block(rows):
        return {"n": len(rows),
                "mean_nodes": statistics.fmean(r["nodes"] for r in rows),
                "mean_edges": statistics.fmean(r["edges"] for r in rows),
                "mean_feature_nodes": statistics.fmean(r["feature_nodes"] for r in rows),
                "mean_feature_count": statistics.fmean(r["feature_count"] for r in rows),
                "min_nodes": min(r["nodes"] for r in rows),
                "max_nodes": max(r["nodes"] for r in rows),
                "min_edges": min(r["edges"] for r in rows),
                "max_edges": max(r["edges"] for r in rows)}
    return {"all": block(metas),
            "frontend_seconds": _quantiles([r["frontend_seconds"] for r in metas]),
            "budget": metas[0]["budget"], "source_digest": metas[0]["source_digest"]}


def run(args):
    guard.activate()
    split, train_meta, test_meta, metas = ensure_graphs(args.workers)
    tasks = [(m["name"], m["cache"], m["class_"]) for m in test_meta]
    items = {"train_meta": train_meta, "tasks": tasks}
    path = RESULTS_DIR / ("arms-%s.json" % args.seed_name)
    seed = os.environ.get("PYTHONHASHSEED")
    arms = {}
    if args.resume and path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("hash_seed") != seed:
            raise SystemExit("cannot resume: %s was written under hash seed %r"
                             % (path, prior.get("hash_seed")))
        if prior.get("graphs", {}).get("source_digest") != metas[0]["source_digest"]:
            raise SystemExit("cannot resume: the graph cache changed")
        arms = prior.get("arms", {})
        print("resuming with arms already measured:", sorted(arms), flush=True)

    def snapshot(complete):
        measured = dict(arms)
        out = {
            "seed_name": args.seed_name,
            "hash_seed": seed,
            "chance": CHANCE,
            "top_k_secondary": TOP_K,
            "corpus": split["corpus"], "split": split["split"],
            "counts": split["counts"],
            "graphs": _graph_summary(metas),
            "graph_records": metas,
            "arms": measured,
            "best_arm_by_test_accuracy": max(
                measured, key=lambda a: (measured[a]["accuracy"],
                                         -ARM_ORDER.index(a))),
            "note": ("the best arm is selected on test accuracy: its confusion "
                     "matrix is therefore not an unbiased estimate"),
            "complete": complete,
            "python": sys.version,
            "tree": str(guard.TREE),
        }
        write_json(path, out)
        return out

    for name in (args.arms.split(",") if args.arms else ARM_ORDER):
        if name not in ARMS:
            raise SystemExit("unknown arm %r; known: %s" % (name, list(ARMS)))
        if name in arms:
            print("arm", name, "already measured; skipping", flush=True)
            continue
        print("arm", name, "flags", ARMS[name], flush=True)
        arms[name] = _run_arm(name, items, args.workers)
        print("arm %-22s top-1 %.4f (%d/%d)  chance %.4f"
              % (name, arms[name]["accuracy"], arms[name]["correct"],
                 arms[name]["n"], CHANCE), flush=True)
        snapshot(len(arms) == len(ARM_ORDER))
    out = snapshot(len(arms) == len(ARM_ORDER))
    print("wrote", path)
    return 0


# -------------------------------------------------------------------- modes

def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    rows, same = [], True
    for name in ARM_ORDER:
        if name not in a["arms"] or name not in b["arms"]:
            continue
        x, y = a["arms"][name], b["arms"][name]
        labels = all(p["predicted"] == q["predicted"] and p["test"] == q["test"]
                     for p, q in zip(x["predictions"], y["predictions"]))
        hashed = x["determinism_sha256"] == y["determinism_sha256"]
        same &= labels and hashed
        rows.append({"arm": name, "accuracy_a": x["accuracy"],
                     "accuracy_b": y["accuracy"],
                     "predictions_identical": labels,
                     "det_hash_a": x["determinism_sha256"],
                     "det_hash_b": y["determinism_sha256"],
                     "hash_match": hashed})
    out = {"a": str(a_path), "b": str(b_path),
           "hash_seed_a": a["hash_seed"], "hash_seed_b": b["hash_seed"],
           "arms": rows, "all_identical": same}
    write_json(guard.RESULTS_DIR / ("determinism-%s-%s.json"
                                    % (a["seed_name"], b["seed_name"])), out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if same else 1


def report(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = data["arms"]
    print("# Structural-topology classification -- measured tables\n")
    print("Corpus: %s\n" % data["corpus"])
    print("Split: %s" % json.dumps(data["split"], sort_keys=True))
    print("\nGraphs: %s\n" % json.dumps(data["graphs"], sort_keys=True))
    print("## Arms (top-1, chance %.4f)\n" % data["chance"])
    print("| arm | top-1 | correct/n | top-3 mean (secondary) |")
    print("|---|---:|---:|---:|")
    for name in ARM_ORDER:
        if name in arms:
            arm = arms[name]
            print("| %s | %.4f | %d/%d | %.4f |"
                  % (name, arm["accuracy"], arm["correct"], arm["n"],
                     arm["secondary_top3_accuracy"]))
    best = data["best_arm_by_test_accuracy"]
    print("\nBest arm selected on test accuracy: **%s** (not unbiased)\n" % best)
    arm = arms[best]
    print("## Confusion matrix, %s (rows true, cols predicted)\n" % best)
    print("| true \\ pred | " + " | ".join(corpus.CLASSES) + " |")
    print("|---|" + "---:|" * len(corpus.CLASSES))
    for true in corpus.CLASSES:
        print("| %s | " % true + " | ".join(str(arm["confusion"][true][p])
                                            for p in corpus.CLASSES) + " |")
    print("\n## Per-class accuracy\n")
    print("| class | " + " | ".join(
        (n if n in arms else "-") for n in ARM_ORDER) + " |")
    print("|---|" + "---:|" * len(ARM_ORDER))
    for cls in corpus.CLASSES:
        print("| %s | " % cls + " | ".join(
            ("%.3f" % arms[n]["per_class"][cls]["accuracy"]) if n in arms else "-"
            for n in ARM_ORDER) + " |")
    print("\n## Per-pair correspondence, %s\n" % best)
    for label, key in (("mean matched nodes", "mean_matched_nodes"),
                       ("mean matched edges", "mean_matched_edges"),
                       ("mean graded edge similarity", "mean_graded_edge_similarity")):
        print("### %s (rows test class, cols stored class)\n" % label)
        print("| test \\ stored | " + " | ".join(corpus.CLASSES) + " |")
        print("|---|" + "---:|" * len(corpus.CLASSES))
        for t in corpus.CLASSES:
            cells = []
            for c in corpus.CLASSES:
                value = arm["cell_stats"][t][c][key]
                cells.append("-" if value is None else "%.3f" % value)
            print("| %s | " % t + " | ".join(cells) + " |")
        print()
    print("## Runtime\n")
    print("```")
    print("per match         ", json.dumps(arm["timing"]["per_match"]))
    print("per test image    ", json.dumps(arm["timing"]["per_test_image"]))
    print("front end / graph ", json.dumps(data["graphs"]["frontend_seconds"]))
    print("```")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed-name", default="seed0")
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--report")
    ap.add_argument("--resume", action="store_true",
                    help="keep arms already written to the seed's results file")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.report:
        return report(args.report)
    if args.build:
        guard.activate()
        split, train_meta, test_meta, metas = ensure_graphs(args.workers)
        print(json.dumps(_graph_summary(metas), indent=2, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
