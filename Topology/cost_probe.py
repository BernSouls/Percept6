"""Compute probe: front-end cost and match cost.  NO accuracy is computed.

This exists only to size the pre-registered sample.  It builds four graphs and
runs four matches and prints the seconds; it never scores a label.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import guard

guard.activate()

import corpus


def main() -> int:
    from frontend import TEST_OFFSET, build, shift
    from matching.joint_mcs import joint_mcs

    split = corpus.split()
    entries = {("%s%d" % (e["class"], e["object"])): e
               for e in split["train"] + split["test"]}
    picked = [(c, o) for c in ("apple", "cup") for o in (1, 5)]
    graphs, rows = {}, []
    for cls, obj in picked:
        entry = entries["%s%d" % (cls, obj)]
        graph = build(entry)
        graphs[(cls, obj)] = graph
        print("front end %-8s %2d  nodes=%4d edges=%4d feat_nodes=%4d  %.2fs"
              % (graph.name, obj, graph.nodes, graph.edges,
                 graph.feature_nodes, graph.frontend_seconds), flush=True)
        rows.append({"name": graph.name, "nodes": graph.nodes,
                     "edges": graph.edges, "feature_nodes": graph.feature_nodes,
                     "feature_count": graph.feature_count,
                     "frontend_seconds": graph.frontend_seconds})

    pairs = [(("apple", 1), ("apple", 5)), (("apple", 1), ("cup", 5)),
             (("cup", 1), ("apple", 1)), (("cup", 1), ("cup", 5))]
    times = []
    for a, b in pairs:
        ga, gb = graphs[a], graphs[b]
        sub_b = shift(gb.subgraph, TEST_OFFSET)
        sigs_b = {n + TEST_OFFSET: v for n, v in gb.sigs.items()}
        t0 = time.perf_counter()
        result = joint_mcs(ga.subgraph, sub_b, ga.sigs, sigs_b)
        seconds = time.perf_counter() - t0
        times.append(seconds)
        print("match    %-10s %-10s score=%.6f nodes=%d edges=%d graded=%d  %.3fs"
              % (ga.name, gb.name, result.score, len(result.mapping),
                 result.matched_edges, result.graded_edges, seconds), flush=True)

    out = {
        "graphs": rows,
        "frontend_seconds": {"n": len(rows),
                             "mean": statistics.fmean(r["frontend_seconds"] for r in rows),
                             "max": max(r["frontend_seconds"] for r in rows)},
        "match_seconds": {"n": len(times), "mean": statistics.fmean(times),
                          "max": max(times)},
        "match_seconds_all": times,
        "note": "no label was read and no accuracy was computed by this probe",
    }
    path = guard.RESULTS_DIR / "cost-probe.json"
    guard.write_json(path, out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("graphs", "match_seconds_all")}, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
