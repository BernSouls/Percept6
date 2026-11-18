"""One image -> one stored graph.  No labels, no filtering, no cores.

The front end is this tree's own driver (``diagnostics/eth80_frontend.py``),
which runs the perception + binding half of one ``Pipeline.step`` and reads the
pipeline's situation window.  Nothing here re-implements perception.

The stored graph is:

  1. the situation window the front end returns;
  2. ``matching.feature_attachment.attach_named_features`` -- every measured
     attribute that has a registered owner travels as a ``NamedFeature`` beside
     its frame signature (the whole quantized tuple and population, untouched);
  3. ``matching.perceptual_selection.select_perceptual_graph`` under the
     declared 512-edge structural budget.

The whole selected graph is stored.  Nothing is filtered by class, by
discriminability, by frequency, or by any training-set statistic: the class
label is attached to the stored graph and is read only by the scoring loop.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Tuple

import guard
from guard import CACHE_DIR

#: the pre-registered structural edge budget (core.frontend_contract.EDGE_BUDGET_PER_FRAME)
BUDGET = 512

#: node ids of the test graph are moved up by this much so that the two graphs'
#: ids cannot collide in one correspondence (the matcher maps a-nodes to b-nodes)
TEST_OFFSET = 1 << 80

CACHE = CACHE_DIR / "graphs"


@dataclass(frozen=True)
class Graph:
    name: str
    class_: str
    object_: int
    view: str
    nodes: int
    edges: int
    feature_nodes: int
    feature_count: int
    frontend_seconds: float
    subgraph: Any
    sigs: Dict[int, Tuple[str, Any]]


def _key(entry: dict) -> str:
    digest = guard.source_digest()[:16]
    return "%s_%s_b%d" % (entry["sha256"][:16], digest, BUDGET)


def paths(entry: dict) -> Tuple[Path, Path]:
    key = _key(entry)
    return CACHE / (key + ".pkl"), CACHE / (key + ".json")


def build(entry: dict) -> Graph:
    """Run the tree's front end on one image and select the perceptual graph."""
    import numpy as np
    from PIL import Image

    import eth80_frontend as FE
    from core.frontend_contract import EDGE_BUDGET_PER_FRAME, FAMILY_SEED_EDGES
    from matching.feature_attachment import attach_named_features
    from matching.perceptual_selection import select_perceptual_graph

    if BUDGET != EDGE_BUDGET_PER_FRAME:
        raise RuntimeError("budget %r is not the declared EDGE_BUDGET_PER_FRAME %r"
                           % (BUDGET, EDGE_BUDGET_PER_FRAME))
    with Image.open(entry["path"]) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    name = "%s%d%s" % (entry["class"], int(entry["object"]), entry["view"])

    t0 = time.perf_counter()
    side = FE.build_graph(image, name=name, record_co_located=False)
    window = side.subgraph
    attached = attach_named_features(window, side.sigs.get)
    enriched = replace(window, signature_overrides=attached)
    selected, diag = select_perceptual_graph(
        enriched, side.sigs.get, budget=BUDGET, family_seeds=FAMILY_SEED_EDGES)
    seconds = time.perf_counter() - t0

    sigs: Dict[int, Tuple[str, Any]] = {}
    for node in selected.nodes:
        info = selected.signature_overrides.get(node) or side.sigs.get(node)
        if info is None:
            raise RuntimeError("%s: selected node %r has no signature" % (name, node))
        sigs[node] = info
    feature_nodes = sum(1 for _lens, sig in sigs.values()
                        if getattr(sig, "_named_features", ()))
    feature_count = sum(len(getattr(sig, "_named_features", ()))
                        for _lens, sig in sigs.values())
    if len(sigs) != len(selected.nodes):
        raise RuntimeError("%s: %d signatures for %d nodes"
                           % (name, len(sigs), len(selected.nodes)))
    return Graph(name=name, class_=entry["class"], object_=int(entry["object"]),
                 view=entry["view"], nodes=len(selected.nodes),
                 edges=len(selected.edges), feature_nodes=feature_nodes,
                 feature_count=feature_count, frontend_seconds=seconds,
                 subgraph=selected, sigs=sigs)


def meta(graph: Graph, source_digest: str) -> dict:
    return {"name": graph.name, "class_": graph.class_, "object_": graph.object_,
            "view": graph.view, "nodes": graph.nodes, "edges": graph.edges,
            "feature_nodes": graph.feature_nodes,
            "feature_count": graph.feature_count,
            "frontend_seconds": graph.frontend_seconds,
            "budget": BUDGET, "source_digest": source_digest}


def build_cached(entry: dict, force: bool = False) -> dict:
    """Build (once) and return the graph's metadata; the graph is on disk."""
    data, meta_path = paths(entry)
    if not force and data.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    graph = build(entry)
    data.parent.mkdir(parents=True, exist_ok=True)
    tmp = data.with_suffix(".tmp%d" % 0)
    with open(tmp, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(data)
    record = meta(graph, guard.source_digest())
    record["cache"] = str(data)
    meta_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return record


def load(cache_path) -> Graph:
    with open(cache_path, "rb") as fh:
        graph = pickle.load(fh)
    if not isinstance(graph, Graph):
        raise RuntimeError("%s did not hold a Graph" % cache_path)
    return graph


def shift(graph: Any, delta: int):
    """The same graph with every node id moved by `delta`."""
    def edge(e):
        return (e[0] + delta, e[1] + delta, e[2])
    return replace(
        graph,
        nodes=frozenset(n + delta for n in graph.nodes),
        edges=frozenset(edge(e) for e in graph.edges),
        edge_params={edge(k): v for k, v in graph.edge_params.items()},
        weights={edge(k): v for k, v in graph.weights.items()},
        node_weights={n + delta: v for n, v in graph.node_weights.items()},
        anchor_of_node={n + delta: v for n, v in graph.anchor_of_node.items()},
        signature_overrides={n + delta: v for n, v in
                             graph.signature_overrides.items()})
