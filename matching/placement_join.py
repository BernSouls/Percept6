"""[MISSION 36 / PART C] THE JOIN: dense local-difference nodes IN THE CELL, attached
to the sparse places, with every edge in the SOURCE node's own frame.

WHY THIS MODULE EXISTS.  Measured on the shipped cell (M36 audit, §1):
518 places, 518 of them at degree 0, holding 0 of the cell's 2,686 edges.  The
placement truth was computed (`matching/placement_edges.build_edges`) and written
into the matcher PAYLOAD, so no graph consumer -- no census, no door, no policy --
could see that a place had neighbours.  The dense field of Part B would have the
same fate if it were appended as a bag of points.

WHAT THE JOIN DOES, in three steps and nothing else:

  1. IDS.  Dense nodes get ids above every pattern node id (`DENSE_ID_BASE`), so a
     dense id can never collide with a graph id, and its anchors are the same
     unique tuples the lens hands out (`("dense", scale, i)`).
  2. OWNERSHIP (C1).  Every dense node is attached to exactly ONE sparse place:
     the nearest place, measured in the plane, with ties broken by the lower node
     id.  A dense node with no place to attach to is REFUSED and counted -- a
     dense node is carried by the place it belongs to, so it may not float.
  3. EDGES (C2).  `DENSE_IN_PLACE` place -> dense with the dense node's position in
     the PLACE's axial frame, and `DENSE_PROXIMITY` dense -> dense between the K
     nearest dense nodes that share a place, in the SOURCE's own frame.  A pair
     whose source has no measured orientation is refused and counted: no frame is
     invented.

The sparse place keeps its own identity and its own M15 repeatability (82-100 %
under shift); the dense node needs no view-repeatability of its own, and this
module does not claim any -- what the node covers is what the place carries.

FLAGS OFF BY DEFAULT.  `PLACEMENT_DENSE_FIELD` (the field exists in the record) and
`PLACEMENT_DENSE_JOIN` (its nodes are in the graph) are both False unless declared,
so every shipped number is produced by exactly the path it was produced by before.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Ids for dense nodes start here: above any pattern-node id a cell can mint.
DENSE_ID_BASE: int = 1 << 20

#: How many dense->dense neighbours each dense node links to INSIDE its own place.
#: Pre-registered, not tuned: 4 is the 4-neighbourhood the whole tree uses.
DENSE_LINK_K: int = 4

#: The parameters that switch the layer on, in the order they matter.
FLAG_FIELD: str = "PLACEMENT_DENSE_FIELD"
FLAG_JOIN: str = "PLACEMENT_DENSE_JOIN"


def _flag(name: str, default: bool = False) -> bool:
    try:
        import parameters as P
        return bool(getattr(P, name, default))
    except Exception:                     # pragma: no cover - parameters always imports
        return bool(default)


def field_on() -> bool:
    """Is the dense FIELD in the record (its nodes measured, carried, visible)?"""
    return _flag(FLAG_FIELD, False)


def join_on() -> bool:
    """Are the dense nodes in the GRAPH (so a graph reader sees them as nodes)?"""
    return _flag(FLAG_JOIN, False)


def place_nodes(subgraph: Any, sigs: Dict[int, Any], lenses: Optional[Sequence[str]] = None
                ) -> List[int]:
    """The sparse places of a cell: nodes whose lens is a place lens (M15's three)."""
    if lenses is None:
        try:
            import parameters as P
            lenses = P.PLACEMENT_M15_PLACE_LENSES
        except Exception:                 # pragma: no cover
            lenses = ("Blob", "BoundaryArc", "CornerPoint")
    want = set(str(x) for x in lenses)
    out: List[int] = []
    for n in sorted(int(k) for k in subgraph.nodes):
        s = sigs.get(n)
        if s is None:
            continue
        lens = s[0] if isinstance(s, (tuple, list)) and s else getattr(s, "class_", None)
        if str(lens) in want:
            out.append(n)
    return out


def _nearest_place_all(px: Any, py: Any, places: Sequence[int],
                       pos: Dict[int, Tuple[float, float]]
                       ) -> Any:
    """The nearest place for every dense node at once: index into `places`, -1 if none.

    Ties go to the LOWER node id, so the assignment is deterministic and independent
    of the order the places were discovered in.
    """
    import numpy as np
    keep = [i for i, n in enumerate(places) if n in pos]
    if not keep:
        return np.full(len(px), -1, dtype=np.int64)
    ids = np.asarray([places[i] for i in keep], dtype=np.int64)
    pp = np.asarray([pos[places[i]] for i in keep], dtype=np.float64)
    dx = np.asarray(px, dtype=np.float64)[:, None] - pp[None, :, 0]
    dy = np.asarray(py, dtype=np.float64)[:, None] - pp[None, :, 1]
    d2 = dx * dx + dy * dy
    # lexsort on (id, distance) picks the lowest id among the exact ties of a row
    order = np.lexsort((np.broadcast_to(ids, d2.shape), d2), axis=1)
    first = order[:, 0]
    return np.asarray([keep[int(j)] for j in first.tolist()], dtype=np.int64)


def degree_of(subgraph: Any) -> Dict[int, int]:
    """node -> degree in the subgraph's own edge set (both ends counted)."""
    deg: Dict[int, int] = {int(n): 0 for n in subgraph.nodes}
    for e in subgraph.edges:
        u, v = int(e[0]), int(e[1])
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return deg


def census(subgraph: Any, sigs: Dict[int, Any]) -> Dict[str, Any]:
    """The lens membership census of one cell's graph (C4): who is in the scene."""
    by_lens: Dict[str, int] = {}
    for n in sorted(int(k) for k in subgraph.nodes):
        s = sigs.get(n)
        lens = (s[0] if isinstance(s, (tuple, list)) and s else
                getattr(s, "class_", None) if s is not None else None)
        k = str(lens) if lens is not None else "NO_SIG"
        by_lens[k] = by_lens.get(k, 0) + 1
    deg = degree_of(subgraph)
    by_type: Dict[str, int] = {}
    for e in subgraph.edges:
        by_type[str(getattr(e[2], "name", e[2]))] = by_type.get(str(getattr(e[2], "name", e[2])), 0) + 1
    return {"n_nodes": int(len(subgraph.nodes)), "n_edges": int(len(subgraph.edges)),
            "by_lens": dict(sorted(by_lens.items())),
            "by_edge_type": dict(sorted(by_type.items())),
            "degree_zero": int(sum(1 for v in deg.values() if v == 0)),
            "degree_mean": (float(sum(deg.values())) / float(len(deg))) if deg else 0.0,
            "degree_max": int(max(deg.values())) if deg else 0}


def join(subgraph: Any, image: Any, sigs: Dict[int, Any], *,
         field_: Any = None, k: int = DENSE_LINK_K, lenses: Optional[Sequence[str]] = None,
         places: Optional[Sequence[int]] = None
         ) -> Tuple[Any, Dict[str, Any]]:
    """Return a NEW subgraph with the dense layer joined, and the join's own census.

    The input subgraph is never mutated: a caller that does not want the layer gets
    the graph it passed in.  The returned graph carries `edge_params` for every new
    edge, so a reader can recover the frame-relative vector rather than a bare pair.

    `places` names the sparse places a dense node may be attached to.  The cell passes
    the PAYLOAD places -- the ones the record and the matcher actually carry -- so a
    dense node is never owned by a place no reader will ever see.  Left as None, the
    join uses every node whose lens is a place lens (`place_nodes`).
    """
    from core.graph_types import Subgraph
    from core.types import EdgeType
    from matching.placement import PlacementVector
    from matching import placement_edges as PE
    import perception.local_difference as LD

    t0 = time.time()
    img = image
    f = field_ if field_ is not None else LD.dense_field(img)
    if places is None:
        places = place_nodes(subgraph, sigs, lenses)
    else:
        places = sorted(int(n) for n in places if int(n) in subgraph.nodes)

    pos = {int(n): (float(p[0]), float(p[1])) for n, p in subgraph.anchor_of_node.items()
           if p is not None}
    # The place's scale: its own detection scale where the graph carries one, else the
    # graph's median nearest-neighbour spacing -- the same rule `matching/placement.node_scale`
    # uses, so a place's units are not a number this module invented.
    from matching.placement import median_nn_spacing
    fb = median_nn_spacing([pos[n] for n in places if n in pos])
    scales = {int(n): float(s) for n, s in (PE.scale_map(subgraph, fallback=fb) or {}).items()}
    # The place's frame: the keypoint estimator where it has one, the FIELD fallback
    # where it does not -- `node_frames_keypoint_then_field`, the tree's own composition
    # ("the composition can only ADD frames").  The provider is reported per place, so a
    # reader can see which frames were measured and which fell back.
    if places:
        theta, frame_src = PE.node_frames_keypoint_then_field(subgraph, img)
        theta = dict(theta)
        frame_src = dict(frame_src)
    else:
        theta, frame_src = {}, {}

    nodes = set(int(n) for n in subgraph.nodes)
    anchors = dict(subgraph.anchor_of_node)
    new_sigs = dict(sigs)
    edges = set(subgraph.edges)
    params: Dict[Any, Any] = dict(getattr(subgraph, "edge_params", {}) or {})
    #: the EXACT vectors, keyed by the edge, in the source node's own frame (C2)
    vectors: Dict[Any, Any] = dict(getattr(subgraph, "dense_vectors", {}) or {})

    # ---- 1/2: ids and ownership -------------------------------------------------
    assign: Dict[int, int] = {}
    refused_no_place = 0
    refused_no_frame = 0
    n_dense = 0
    owner_of = _nearest_place_all([n.x for n in f.nodes], [n.y for n in f.nodes], places, pos)
    for i, node in enumerate(f.nodes):
        nid = int(DENSE_ID_BASE) + int(i)
        nodes.add(nid)
        anchors[nid] = (float(node.x), float(node.y))
        new_sigs[nid] = ("DENSE_FIELD", _sig_of(node))
        n_dense += 1
        oi = int(owner_of[i]) if i < len(owner_of) else -1
        if oi < 0:
            refused_no_place += 1
            continue
        owner = int(places[oi])
        assign[nid] = owner
        th = theta.get(owner)
        sc = scales.get(owner)
        if th is None or sc is None or not (sc > 0.0):
            refused_no_frame += 1
            continue
        vec = PE.placement_vector_from(_Proxy(anchors), owner, nid, float(th), float(sc))
        if vec is None:
            refused_no_frame += 1
            continue
        e = (owner, nid, EdgeType.DENSE_IN_PLACE)
        edges.add(e)
        params[e] = _dense_edge_params(vec)
        vectors[e] = vec

    # ---- 3a: dense <-> dense inside a place -------------------------------------
    by_place: Dict[int, List[Tuple[int, Any]]] = {}
    for i, node in enumerate(f.nodes):
        nid = int(DENSE_ID_BASE) + int(i)
        if nid in assign:
            by_place.setdefault(assign[nid], []).append((nid, node))
    prox = 0
    for owner, members in sorted(by_place.items()):
        members.sort(key=lambda t: t[0])
        for nid, node in members:
            th = float(node.theta)
            sc = float(node.support_px)
            if not (sc > 0.0) or th is None:
                continue
            others = [(m, p) for m, p in members if m != nid]
            others.sort(key=lambda t: (float(t[1].x) - float(node.x)) ** 2
                        + (float(t[1].y) - float(node.y)) ** 2)
            for m, p in others[:int(k)]:
                dx = float(p.x) - float(node.x)
                dy = float(p.y) - float(node.y)
                d = math.hypot(dx, dy)
                if d <= 1e-9:
                    continue
                e = (nid, m, EdgeType.DENSE_PROXIMITY)
                edges.add(e)
                vec = PlacementVector(
                    direction=PE.wrap_ax(math.atan2(dy, dx) - th),
                    distance=d / sc,
                    log_scale=float(math.log2(max(float(p.support_px), 1e-6) / sc)),
                    source_theta=th, source_scale=sc, tolerance=None, axial=True)
                params[e] = _dense_edge_params(vec)
                vectors[e] = vec
                prox += 1

    sub = Subgraph(nodes=frozenset(nodes), edges=frozenset(edges),
                   edge_params=params, weights=getattr(subgraph, "weights", {}) or {})
    # The subgraph's own lookups are attached, exactly as the builder attaches them.
    for name in ("anchor_of_node", "lens_of_node", "sig_of_node", "id_of_node"):
        if hasattr(subgraph, name):
            try:
                object.__setattr__(sub, name, anchors if name == "anchor_of_node"
                                   else getattr(subgraph, name))
            except Exception:             # pragma: no cover
                pass
    if not hasattr(sub, "anchor_of_node"):
        try:
            object.__setattr__(sub, "anchor_of_node", anchors)
        except Exception:                 # pragma: no cover
            pass
    try:
        object.__setattr__(sub, "dense_vectors", vectors)
    except Exception:                     # pragma: no cover
        pass
    # The dense signatures travel WITH the graph: `Subgraph` has no `sigs` field, so the
    # joined graph carries the merged map as an attribute, exactly the way the builder's
    # graph object carries `sigs` -- otherwise a reader with only the subgraph would see
    # dense NODES and no dense CONTENT.
    try:
        object.__setattr__(sub, "sigs", new_sigs)
    except Exception:                     # pragma: no cover
        pass

    src_counts: Dict[str, int] = {}
    owned = set(assign.values())
    for n in places:
        if int(n) not in owned:
            continue
        v = frame_src.get(int(n))
        # the provider map is `+sigma` for the keypoint estimator and `-window` for the
        # field fallback (`node_frames_keypoint_then_field`), so the sign IS the provider.
        k = "none" if v is None else ("keypoint" if float(v) > 0 else "field")
        src_counts[k] = src_counts.get(k, 0) + 1
    stats = {"n_dense": int(n_dense),
             "place_frame_sources": dict(sorted(src_counts.items())), "n_places": int(len(places)),
             "n_places_positioned": int(sum(1 for n in places if n in pos)),
             "n_dense_attached": int(len(assign)),
             "n_refused_no_place": int(refused_no_place),
             "n_refused_no_frame": int(refused_no_frame),
             "n_dense_in_place_edges": int(len(assign) - refused_no_frame),
             "n_dense_proximity_edges": int(prox),
             "place_degree_before": _place_degree(subgraph, places),
             "place_degree_after": _place_degree(sub, places),
             "secs": float(time.time() - t0)}
    stats["dense_per_place"] = (float(n_dense) / float(len(places))) if places else 0.0
    return sub, stats


#: The dense relation's registered edge class (core/schema.EDGE_SCHEMAS) and its bins.
DENSE_EDGE_CLASS: str = "EDGE_DENSE_PLACEMENT"


def _dense_edge_params(vec: Any) -> Any:
    """A measured `PlacementVector` as the graph's own `EdgeParams` (registered class).

    The bins are the door's coarse index; the exact floats travel alongside in
    `subgraph.dense_vectors`, because `EdgeParams.quantized` is a bin tuple and C2 asks
    for the VECTOR, not a bin of it.
    """
    from core.schema import EdgeParams
    bearing = float(vec.direction) % math.pi          # axial: the source's frame
    d = float(vec.distance)
    ls = float(vec.log_scale)
    b0 = min(11.0, max(0.0, math.floor(bearing / math.pi * 12.0)))
    b1 = min(7.0, max(0.0, math.floor(d)))
    b2 = min(7.0, max(0.0, math.floor((ls + 2.0) / 4.0 * 8.0)))
    return EdgeParams(class_=DENSE_EDGE_CLASS, quantized=(b0, b1, b2))


def _sig_of(node: Any) -> Any:
    """The registered signature of one dense node (the lens's own function, not a copy)."""
    from lenses.dense_field import signature_of
    return signature_of(node)


def _place_degree(subgraph: Any, places: Sequence[int]) -> Dict[str, float]:
    """Mean/min/max degree over the PLACES only (the census C1 is about)."""
    if not places:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "zero": 0.0, "n": 0.0}
    deg = degree_of(subgraph)
    vals = [deg.get(int(n), 0) for n in places]
    return {"mean": float(sum(vals)) / float(len(vals)), "min": float(min(vals)),
            "max": float(max(vals)), "zero": float(sum(1 for v in vals if v == 0)),
            "n": float(len(vals))}


class _Proxy:
    """A two-field view of an anchor map, for `placement_vector_from`."""

    __slots__ = ("anchor_of_node",)

    def __init__(self, anchors: Dict[int, Tuple[float, float]]) -> None:
        self.anchor_of_node = anchors


def dense_rows(field_: Any, subgraph: Any = None,
               keep: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """The dense nodes as record rows (D2): ids as they appear in the graph, content, place.

    `keep` is the set of place node ids the RECORD carries.  A dense node whose owner
    place was dropped by the record's own filters is written with `place = -1`: the
    record never names a place row that does not exist, and the count of such rows is
    reported by the caller (no silent dangling reference, no invented owner).
    """
    from core.registries import POPULATION_SLOTS
    names = [s.name for s in POPULATION_SLOTS["DENSE_FIELD"]]
    assign: Dict[int, int] = {}
    if subgraph is not None:
        for e in subgraph.edges:
            if getattr(e[2], "name", "") == "DENSE_IN_PLACE":
                assign[int(e[1])] = int(e[0])
    keep_ids = set(int(k) for k in keep) if keep is not None else None
    out: List[Dict[str, Any]] = []
    for i, n in enumerate(field_.nodes):
        nid = int(DENSE_ID_BASE) + int(i)
        owner = int(assign.get(nid, -1))
        if keep_ids is not None and owner not in keep_ids:
            owner = -1
        row = {"n": nid, "place": owner,
               "x": float(n.x), "y": float(n.y), "s": float(n.support_px),
               "activity": float(n.activity), "scale_index": int(n.scale_index)}
        for nm, v in zip(names, n.values):
            row[nm] = float(v)
        out.append(row)
    return out
