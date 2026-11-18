"""Correspondence detail for the selected configuration.

For each query/reference pair the winner's alignment is solved once and the
tree's own ``joint_mcs`` facts are aggregated: matched nodes, matched edges,
edge-preservation fraction and mean graded edge similarity.  The direction
with the better winner score is the one reported, exactly as ``joint_mcs``
selects it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import corpus  # noqa: E402
import solver  # noqa: E402
import tune  # noqa: E402


def _worker(args):
    qmeta, refs, winner = args
    from frontend import load
    q = load(qmeta["cache"])
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
    rows = []
    for ref_meta in refs:
        r = load(ref_meta["cache"])
        sim = solver.node_sim_matrix(r, q, r.sigs, q.sigs)
        sim_t = tune._transpose(sim)
        um = solver.derived_unmatched_cost(sim) if uc == "derived" else float(uc)
        fwd = solver.solve_direction(r, q, r.sigs, q.sigs, sim, beta=beta,
                                     unmatched_cost=um, unmapped_credit=occ)
        rev = solver.solve_direction(q, r, q.sigs, r.sigs, sim_t, beta=beta,
                                     unmatched_cost=um, unmapped_credit=occ)
        fa = solver.facts(r, q, r.sigs, q.sigs, sim, fwd)
        fb = solver.facts(q, r, q.sigs, r.sigs, sim_t, rev)
        sa = solver.score(fa, w=w, gamma=gamma, denom_mode=denom,
                          isolated=isolated, presence_counts=presence,
                          formula=formula, delta=delta)[0]
        sb = solver.score(fb, w=w, gamma=gamma, denom_mode=denom,
                          isolated=isolated, presence_counts=presence,
                          formula=formula, delta=delta)[0]
        f = fa if sa >= sb else fb
        present = sum(1 for r_ in f.edge_rows_a if r_[0] and r_[1]) + \
            sum(1 for r_ in f.edge_rows_b if r_[0] and r_[1])
        checkable = sum(1 for r_ in f.edge_rows_a if r_[0]) + \
            sum(1 for r_ in f.edge_rows_b if r_[0])
        graded = [r_[2] for r_ in (f.edge_rows_a + f.edge_rows_b)
                  if r_[0] and r_[1] and r_[3]]
        rows.append({"test": qmeta["name"], "ref": ref_meta["name"],
                     "class": ref_meta["class_"],
                     "matched_nodes": len(f.mapping),
                     "matched_edges": present,
                     "checkable_edges": checkable,
                     "edge_preservation": (present / checkable) if checkable else None,
                     "graded_edges": len(graded),
                     "mean_graded_similarity": statistics.fmean(graded) if graded else None,
                     "nodes_a": f.na, "nodes_b": f.nb,
                     "edges_a": f.ea, "edges_b": f.eb})
    return rows


def _summary(rows):
    def block(key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return {"n": 0}
        vals.sort()
        return {"n": len(vals), "mean": statistics.fmean(vals),
                "median": statistics.median(vals), "p95": vals[int(0.95 * len(vals)) - 1],
                "min": vals[0], "max": vals[-1]}
    return {"pairs": len(rows),
            "mean_matched_nodes": block("matched_nodes")["mean"],
            "mean_matched_edges": block("matched_edges")["mean"],
            "mean_edge_preservation": block("edge_preservation").get("mean"),
            "mean_graded_similarity": block("mean_graded_similarity").get("mean"),
            "matched_nodes": block("matched_nodes"),
            "matched_edges": block("matched_edges"),
            "edge_preservation": block("edge_preservation"),
            "graded_similarity": block("mean_graded_similarity")}


def run(split: str, workers: int):
    data = json.loads((tune.TUNE / "graphs.json").read_text(encoding="utf-8"))
    winner = json.loads((tune.TUNE / "winner.json").read_text(encoding="utf-8"))
    refs = [m for g in tune._ref_groups(data["references"]).values() for m in g]
    refs.sort(key=lambda m: m["name"])
    queries = sorted(data[split], key=lambda m: m["name"])
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    out = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=tune._init) as pool:
        for i, rows in enumerate(pool.imap_unordered(
                _worker, [(q, refs, winner) for q in queries], chunksize=1)):
            out.extend(rows)
            print("  %3d/%d (%.0fs)" % (i + 1, len(queries),
                                        time.perf_counter() - t0), flush=True)
    out.sort(key=lambda r: (r["test"], r["ref"]))
    result = {"split": split, "winner": winner,
              "summary": _summary(out), "pairs": out}
    write_json(tune.TUNE / ("details_%s.json" % split), result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    return run(args.split, args.workers)


if __name__ == "__main__":
    sys.exit(main())
