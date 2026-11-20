"""Smoke pin: graph-hash change for BINDER_COMPLETE_POSITION_MAP (ON vs OFF).

Builds ONE probe dev graph with BINDER_COMPLETE_POSITION_MAP=True (default)
and ONE with OFF, computes the graph hash and positioned-node count for each,
and reports the difference. This is the smoke pin requested but never reported.
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
from guard import RESULTS_DIR, write_json  # noqa: E402


def _graph_hash(g):
    """Deterministic hash of the subgraph structure (no positions)."""
    h = hashlib.sha256()
    h.update(repr(sorted(g.subgraph.nodes)).encode())
    h.update(repr(sorted((repr(a), repr(b), repr(e))
                         for a, b, e in g.subgraph.edges)).encode())
    return h.hexdigest()


def _positioned_count(g):
    return sum(1 for n in g.subgraph.nodes
               if g.subgraph.anchor_of_node.get(n) is not None)


def _content_positioned_count(g):
    from perception.front_end_mechanisms import CONTENT_TOKEN_LENSES
    cnt = 0
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None and len(a) >= 2:
            lens = str(g.sigs.get(n, ("?",))[0])
            if lens in CONTENT_TOKEN_LENSES:
                cnt += 1
    return cnt


def _build_probe(flag_value):
    import _bootstrap
    import corpus
    from frontend import build

    # Force the flag
    import parameters as P
    P.BINDER_COMPLETE_POSITION_MAP = flag_value
    _bootstrap.apply_switch_overrides("FE_L3_SPINE=false;FE_L4_JUNCTIONS=false")
    # Also set via env so _bootstrap sees it
    os.environ["PERCEPT_SET"] = "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false"

    split = corpus.split_tuning()
    entry = None
    for e in split["dev"]:
        if e["class"] == "apple" and e["view"] == "-090-000":
            entry = e
            break
    g = build(entry)
    return g


def main():
    out = {}
    results = {}
    for flag_val, label in [(True, "ON"), (False, "OFF")]:
        print("Building probe with BINDER_COMPLETE_POSITION_MAP=%s..." % label,
              flush=True)
        g = _build_probe(flag_val)
        gh = _graph_hash(g)
        pos_total = _positioned_count(g)
        pos_content = _content_positioned_count(g)
        results[label] = {
            "graph_hash": gh,
            "positioned_nodes": pos_total,
            "positioned_content_nodes": pos_content,
            "total_nodes": len(g.subgraph.nodes),
        }
        print("  %s: graph_hash=%s positioned=%d/%d content_positioned=%d"
              % (label, gh[:16], pos_total, len(g.subgraph.nodes), pos_content),
              flush=True)

    on_hash = results["ON"]["graph_hash"]
    off_hash = results["OFF"]["graph_hash"]
    out["smoke_pin"] = results
    out["graph_hash_change"] = {
        "on_hash": on_hash,
        "off_hash": off_hash,
        "hashes_differ": on_hash != off_hash,
        "positioned_delta": results["ON"]["positioned_nodes"]
        - results["OFF"]["positioned_nodes"],
        "content_positioned_delta":
            results["ON"]["positioned_content_nodes"]
            - results["OFF"]["positioned_content_nodes"],
        "interpretation": (
            "Graph hashes differ: BINDER_COMPLETE_POSITION_MAP changes the "
            "emitted graph structure (edge set / node set), not just positions."
            if on_hash != off_hash else
            "Graph hashes identical: the flag changes only the position map, "
            "not the graph structure."
        ),
    }
    print("\nGraph hash change: %s" % json.dumps(out["graph_hash_change"],
                                                  indent=2), flush=True)
    write_json(RESULTS_DIR / "t4" / "smoke_binder_pin.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
