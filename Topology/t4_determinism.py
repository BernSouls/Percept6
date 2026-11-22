"""Determinism check: identical output under PYTHONHASHSEED 0 and 1729.

Usage:
    python t4_determinism.py <seed> <outfile>

Builds one dev graph per arm, hashes the full graph structure + signatures +
positions, writes the result to <outfile>. Run twice with seeds 0 and 1729,
then compare.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}


def _graph_hash(g):
    h = hashlib.sha256()
    nodes = sorted(g.subgraph.nodes, key=lambda n: str(n))
    h.update(json.dumps([str(n) for n in nodes], sort_keys=True).encode())
    edges = sorted(g.subgraph.edges, key=lambda e: (str(e[0]), str(e[1]),
                                                     str(e[2])))
    h.update(json.dumps([[str(a), str(b), str(c)] for a, b, c in edges],
                        sort_keys=True).encode())
    sigs = {str(n): str(g.sigs.get(n, ("?",))) for n in nodes}
    h.update(json.dumps(sigs, sort_keys=True).encode())
    positions = {}
    for n in nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None:
            positions[str(n)] = [float(a[i]) if i < len(a) else None
                                 for i in range(max(2, len(a)))]
    h.update(json.dumps(positions, sort_keys=True).encode())
    return h.hexdigest()


def main() -> int:
    seed = int(sys.argv[1])
    outfile = sys.argv[2]
    out = {"seed": seed, "arms": {}}
    for arm in ARMS:
        import _bootstrap
        _bootstrap.apply_switch_overrides(ARMS[arm])
        import corpus
        from frontend import build
        split = corpus.split_tuning()
        dev = [e for e in split["dev"] if e["view"] == "-090-000"]
        by_class = {}
        for e in dev:
            c = e["class"]
            if c not in by_class or e["object"] < by_class[c]["object"]:
                by_class[c] = e
        entries = [by_class[c] for c in corpus.CLASSES]
        g = build(entries[0])
        out["arms"][arm] = {"hash": _graph_hash(g), "name": g.name}
        print("seed=%d %s: %s" % (seed, arm, out["arms"][arm]["hash"][:16]),
              flush=True)
    import json as _j
    with open(outfile, "w") as fh:
        _j.dump(out, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
