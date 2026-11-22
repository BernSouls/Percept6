"""Bounded, deterministic selection of perceptual structural relations.

This is deliberately separate from the cognitive situation-size cap. It
accepts a complete perceptual graph; callers must not apply it to an episode
or action graph. It preserves selected edge parameters and spatial metadata.
"""
from __future__ import annotations

from collections import defaultdict
import heapq

from core.frontend_contract import EDGE_BUDGET_PER_FRAME, FAMILY_SEED_EDGES
from core.graph_types import Subgraph
from core.types import REPRODUCIBLE_EDGE_TYPES
from matching.relation_frame import is_frame

_EXCLUDED = frozenset({"CONTACT", "CO_LOCATED", "PROXIMITY", "SURFACE_GROUPING"})


def is_structural(edge_type):
    return edge_type in REPRODUCIBLE_EDGE_TYPES and getattr(edge_type, "name", "") not in _EXCLUDED


def select_perceptual_graph(graph, sig_lookup, *, budget=EDGE_BUDGET_PER_FRAME,
                            family_seeds=FAMILY_SEED_EDGES):
    """Return (selected graph, diagnostics) using family seeds then BFS.

    Highest edge weight wins within each family and frontier, with typed
    endpoint identities as deterministic ties. Disconnected frame nodes are
    retained; other nodes enter only as endpoints of selected structural edges.
    No unselected induced edges are silently reintroduced.
    """
    if budget < 0 or family_seeds < 0:
        raise ValueError("selection budgets must be non-negative")
    sigs = {n: sig_lookup(n) for n in graph.nodes}

    def node_key(n):
        info = sigs.get(n)
        if info is None:
            return ("", b"", n)
        return (str(info[0]), info[1].serialize(), n)

    def edge_key(edge):
        u, v, et = edge
        a, b = node_key(u), node_key(v)
        if b < a:
            a, b = b, a
        # [0d] CANONICAL.  `node_key` ends in the node id, so the old key
        # broke signal ties by BUILD ORDER.  a[:2]/b[:2] are the endpoints'
        # CONTENT (lens type, signature bytes); the ids remain only as the
        # final total-order fallback for content-identical pairs.
        # Measured: 1109 structural edges compete for the 512 budget, so this
        # decider really does choose which edges the matcher ever sees.
        return (-float(graph.weights.get(edge, 1.0)), et.name,
                a[:2], b[:2], edge[0], edge[1])

    families = defaultdict(list)
    adjacency = defaultdict(list)
    all_families = {e[2].name for e in graph.edges if is_structural(e[2])}
    for e in graph.edges:
        if (is_structural(e[2]) and all(
                sigs.get(n) is not None and is_frame(sigs[n][0]) for n in e[:2])):
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
                for next_edge in adjacency[n]:
                    if next_edge not in chosen and next_edge not in queued:
                        heapq.heappush(heap, (edge_key(next_edge), next_edge))
                        queued.add(next_edge)

    # Round-robin over families ensures a budget smaller than the full seed
    # set does not allocate every seat to the lexically first family.
    for rank in range(family_seeds):
        for family in sorted(families):
            if len(chosen) >= budget:
                break
            edges = families[family]
            if rank < len(edges):
                add_edge(edges[rank])
        if len(chosen) >= budget:
            break

    while heap and len(chosen) < budget:
        _, e = heapq.heappop(heap)
        if e not in chosen:
            add_edge(e)

    for n, info in sigs.items():
        if info is not None and is_frame(info[0]):
            visited.add(n)
    edges = frozenset(chosen)
    params = {e: graph.edge_params[e] for e in edges if e in graph.edge_params}
    weights = {e: graph.weights[e] for e in edges if e in graph.weights}
    nodes = frozenset(visited)
    selected = Subgraph(nodes=nodes, edges=edges, edge_params=params, weights=weights,
                        node_weights={n: graph.node_weights[n] for n in nodes if n in graph.node_weights},
                        anchor_of_node={n: graph.anchor_of_node[n] for n in nodes if n in graph.anchor_of_node},
                        t_junctions=graph.t_junctions,
                        signature_overrides={n: graph.signature_overrides[n] for n in nodes
                                             if n in graph.signature_overrides},
                        perceptual_prepared=True)
    selected_families = {e[2].name for e in edges}
    diagnostics = {
        "budget": budget, "input_edges": len(graph.edges), "selected_edges": len(edges),
        "structural_families_present": tuple(sorted(all_families)),
        "structural_families_missing": {name: ("no_frame_endpoints" if name not in families
                                               else "edge_budget" if len(edges) >= budget
                                               else "no_seed_or_reachable_edge")
                                        for name in sorted(all_families - selected_families)},
        "dropped_unanchored_nodes": len(graph.nodes - nodes),
    }
    return selected, diagnostics


def prepare_perceptual_graph(graph, sig_lookup):
    """Attach observed attributes before selecting and removing their nodes."""
    from dataclasses import replace
    from matching.feature_attachment import attach_named_features
    if graph.perceptual_prepared:
        return graph, {"already_prepared": True, "selected_edges": len(graph.edges),
                       "feature_owners": len(graph.signature_overrides)}
    def lookup(n):
        return graph.signature_overrides.get(n) or sig_lookup(n)
    attached = dict(graph.signature_overrides)
    attached.update(attach_named_features(graph, lookup))
    enriched = replace(graph, signature_overrides=attached)

    def enriched_lookup(n):
        # [LEAD] THE SELECTION READS THE ENRICHED SIGNATURES.  The
        # override-aware lookup was built and used for the feature
        # attachment only; `select_perceptual_graph` was given the RAW
        # provider, so it never saw the overrides -- and a node whose raw
        # signature is missing but which carries an override had all of its
        # structural edges dropped from the selection (the edge test needs
        # the signature to know the lens type).
        return enriched.signature_overrides.get(n) or sig_lookup(n)

    selected, diagnostics = select_perceptual_graph(enriched, enriched_lookup)
    diagnostics["feature_owners"] = len(selected.signature_overrides)
    return selected, diagnostics
