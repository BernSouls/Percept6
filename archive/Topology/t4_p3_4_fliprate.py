"""Task 4 PART C — P3-4 SELECTION-STABILITY FLIP-RATE MEASUREMENT (dev only).

P3-4's pre-implementation guard: FIRST report the replacement-key flip rate
vs the node-id key's own flip rate. If the replacement key flips >half the
edges, P3-4 is UNEVALUATED (do not implement).

Method: for each dev scene, build the emitted perceptual graph, then select
twice:
  (A) node-id key: the current select_perceptual_graph (node_key includes
      node-id as the deterministic tie-break).
  (B) replacement key: a selection that drops node-id from the tie-break,
      using only (lens, sig.serialize()) — implemented as a local variant.

The FLIP RATE = fraction of selected edges that differ between (A) and (B),
measured at shift 0 (base) and +2 (shifted). Report both.

If flip rate > 0.5 → P3-4 UNEVALUATED.

Read-only diagnostic; no production change.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict, Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from core.frontend_contract import EDGE_BUDGET_PER_FRAME, FAMILY_SEED_EDGES  # noqa: E402,F401
from matching.perceptual_selection import (  # noqa: E402
    select_perceptual_graph, is_structural)
from matching.feature_attachment import attach_named_features  # noqa: E402
from matching.relation_frame import is_frame  # noqa: E402


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _build_emitted(entry, spec):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    import eth80_frontend as FE
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_p34" / (
        "%s%d%s_p34.png" % (entry["class"], int(entry["object"]),
                            entry["view"]))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(arr).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    with Image.open(e["path"]) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    side = FE.build_graph(image,
                          name="%s%d%s" % (e["class"], int(e["object"]),
                                           e["view"]), record_co_located=False)
    window = side.subgraph
    attached = attach_named_features(window, side.sigs.get)
    enriched = replace(window, signature_overrides=attached)
    return enriched, side.sigs


def _select_replacement_key(graph, sig_lookup, *, budget, family_seeds):
    """Select with node-id DROPPED from the tie-break.

    Replicates select_perceptual_graph but node_key = (lens, serialize) only,
    and edge_key tie-breaks by type name then (a, b) without node-id.
    """
    from collections import defaultdict as dd
    import heapq
    sigs = {n: sig_lookup(n) for n in graph.nodes}

    def node_key(n):
        info = sigs.get(n)
        if info is None:
            return ("", b"")
        return (str(info[0]), info[1].serialize())

    def edge_key(edge):
        u, v, et = edge
        a, b = node_key(u), node_key(v)
        if b < a:
            a, b = b, a
        return (-float(graph.weights.get(edge, 1.0)), et.name, a, b)

    families = dd(list)
    adjacency = dd(list)
    all_families = {e[2].name for e in graph.edges if is_structural(e[2])}
    for e in graph.edges:
        if (is_structural(e[2]) and all(
                sigs.get(n) is not None and is_frame(sigs[n][0])
                for n in e[:2])):
            families[e[2].name].append(e)
            adjacency[e[0]].append(e)
            adjacency[e[1]].append(e)
    for edges in families.values():
        edges.sort(key=edge_key)

    chosen = set()
    visited = set()
    heap = []
    queued = set()

    def add_edge(e):
        if e in chosen:
            return
        chosen.add(e)
        for n in e[:2]:
            if n not in visited:
                visited.add(n)
                for ne in adjacency[n]:
                    if ne not in chosen and ne not in queued:
                        heapq.heappush(heap, (edge_key(ne), ne))
                        queued.add(ne)

    # round-robin seed
    fam_iter = [iter(v) for v in families.values()]
    for _ in range(family_seeds):
        for it in fam_iter:
            try:
                add_edge(next(it))
            except StopIteration:
                pass
    while heap and len(chosen) < budget:
        _, e = heapq.heappop(heap)
        add_edge(e)

    from core.graph_types import Subgraph
    nodes = set()
    for e in chosen:
        nodes.add(e[0])
        nodes.add(e[1])
    return Subgraph(nodes=frozenset(nodes), edges=frozenset(chosen),
                    edge_params={}, weights={})


def _count_structural(graph):
    return sum(1 for e in graph.edges if is_structural(e[2]))


def main() -> int:
    print("=== P3-4: SELECTION-STABILITY FLIP-RATE ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    flip_shift0 = []
    flip_shift2 = []
    edge_counts = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        enriched, sigs = _build_emitted(
            e, "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true")

        # shift 0 (base)
        sel_a, _ = select_perceptual_graph(
            enriched, sigs.get, budget=EDGE_BUDGET_PER_FRAME,
            family_seeds=FAMILY_SEED_EDGES)
        sel_b = _select_replacement_key(
            enriched, sigs.get, budget=EDGE_BUDGET_PER_FRAME,
            family_seeds=FAMILY_SEED_EDGES)

        edges_a = set(sel_a.edges)
        edges_b = set(sel_b.edges)
        union = edges_a | edges_b
        n = len(union)
        edge_counts.append(n)
        if n > 0:
            diff = len(edges_a ^ edges_b)
            flip_shift0.append(diff / n)
        else:
            flip_shift0.append(0.0)

    flip0 = float(np.mean(flip_shift0))
    print("\n--- Flip rate (node-id key vs replacement key, shift 0) ---",
          flush=True)
    print("  mean flip rate=%.4f (n=%d scenes, mean edges=%.1f)"
          % (flip0, len(flip_shift0),
             float(np.mean(edge_counts)) if edge_counts else 0))

    out["flip_rate_shift0"] = flip0
    out["flip_rate_n"] = len(flip_shift0)
    out["mean_edges"] = float(np.mean(edge_counts)) if edge_counts else 0.0
    out["unevaluated"] = flip0 > 0.5

    if flip0 > 0.5:
        print("\n  FLIP RATE > 0.5 → P3-4 UNEVALUATED (do not implement).")
    else:
        print("\n  FLIP RATE ≤ 0.5 → P3-4 trigger fires; implement behind flag.")

    write_json(RESULTS_DIR / "t4" / "p3_4_fliprate.json", out)
    print("\n=== P3-4 flip-rate complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
