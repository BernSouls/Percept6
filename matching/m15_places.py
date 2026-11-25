"""Mission 15 items 3 and 4 -- what an endpoint is, and what a true pair is.

ITEM 3 (endpoints and context)
    With `PLACEMENT_M15_PLACES` on, the endpoint set is the ATLAS PLACE LENSES
    (`PLACEMENT_M15_PLACE_LENSES`, measured by `Topology/placement_m15_atlas.py`),
    not the eight-lens whitelist frozen in `placement_frozen_params.json`.  A ridge
    place is admitted ACROSS-ridge only.  Every node that is not a place becomes an
    ATTRIBUTE: it keeps its signature, loses its vote, and is attached to the nearest
    place node, which is what item 5's geometry term and item 10's probe then read.
    The keypoint layer and the adopted frame provider are required by
    `placement_edges._require_m15_configuration()`; this module only consumes them.

ITEM 4 (the truth rule)
    Two rules are computed side by side, always:
      OLD -- POSITIONAL truth: the exact transform puts the two anchors within
             `TRUTH_TOL` and the pairing is nearest, separated, injective and mutual.
             No lens is consulted, so an attribute node "is" whatever sits there.
      NEW -- LENS truth: same LENS, same position within `TRUTH_TOL`; for a ridge lens
             only the ACROSS-ridge distance counts (an arc-length sample is not
             repeatable along its own boundary).
    Every rate this mission reports is printed beside both counts.

Nothing here is truth-aware at match time: both rules are functions of the exact
transform and of the graph, and the matcher never sees either.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TRUTH_TOL = 1.7009          # same constant the shipped read uses
TRUTH_SEP = 1.6
RIDGE_HINTS = ("CONTOUR", "BOUNDARY", "RIDGE", "EDGE")


def is_ridge(lens: str) -> bool:
    return any(h in str(lens).upper() for h in RIDGE_HINTS)


def place_lenses(explicit: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """The endpoint lens list in force: explicit > the flag's atlas list > frozen."""
    if explicit:
        return tuple(explicit)
    import parameters as P
    if getattr(P, "PLACEMENT_M15_PLACES", False):
        got = tuple(getattr(P, "PLACEMENT_M15_PLACE_LENSES", ()) or ())
        if not got:
            raise RuntimeError(
                "PLACEMENT_M15_PLACES is on but PLACEMENT_M15_PLACE_LENSES is empty: "
                "fill it from Topology/results/placement/m15_atlas.json (item 1)")
        return got
    from matching.placement_matcher import frozen
    return tuple(frozen()["endpoint_lenses"])


def _nodes(graph: Any) -> Tuple[List[int], np.ndarray, List[str]]:
    sub = graph.subgraph
    ids = [int(n) for n in sorted(sub.nodes) if n in sub.anchor_of_node]
    pos = np.asarray([sub.anchor_of_node[n] for n in ids], float) if ids else \
        np.zeros((0, 2))
    lens = [str(graph.sigs.get(n, ("?",))[0]) for n in ids]
    return ids, pos, lens


def place_nodes(graph: Any, lenses: Optional[Sequence[str]] = None
                ) -> Tuple[List[int], np.ndarray, List[str]]:
    """(ids, anchors, lens) of the nodes whose lens is a place lens."""
    keep = set(place_lenses(lenses))
    ids, pos, lens = _nodes(graph)
    idx = [i for i, l in enumerate(lens) if l in keep]
    return [ids[i] for i in idx], pos[idx], [lens[i] for i in idx]


def attribute_attachment(graph: Any, lenses: Optional[Sequence[str]] = None
                         ) -> Dict[int, int]:
    """Every non-place node -> the id of the nearest place node (item 3).

    The attribute keeps its own signature; it simply cannot be a vote.  `{}` when
    there is no place node at all -- the caller must then say so rather than invent
    an attachment.
    """
    from scipy.spatial import cKDTree
    pids, ppos, _ = place_nodes(graph, lenses)
    if not pids:
        return {}
    tree = cKDTree(ppos)
    keep = set(place_lenses(lenses))
    ids, pos, lens = _nodes(graph)
    out: Dict[int, int] = {}
    for i, n in enumerate(ids):
        if lens[i] in keep:
            continue
        _d, j = tree.query(pos[i], k=1)
        out[n] = pids[int(j)]
    return out


def endpoint_report(graph: Any, lenses: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Counts before/after: the frozen whitelist against the atlas place list."""
    from matching.placement_matcher import frozen
    old = set(frozen()["endpoint_lenses"])
    new = set(place_lenses(lenses))
    ids, pos, lens = _nodes(graph)
    n_old = sum(1 for l in lens if l in old)
    n_new = sum(1 for l in lens if l in new)
    attached = attribute_attachment(graph, lenses)
    return {"nodes": len(ids), "endpoints_old_list": n_old,
            "endpoints_new_list": n_new, "attributes_attached": len(attached),
            "share_old": n_old / len(ids) if ids else None,
            "share_new": n_new / len(ids) if ids else None,
            "old_list": sorted(old), "new_list": sorted(new)}


# --------------------------------------------------------------------------- truth
def _forward(spec: Dict[str, Any], p: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    from Topology.placement_m12_read_real import _forward as F
    return F(spec, p, shape)


def _frame_vectors(angles: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """tangent (along-ridge) and normal (across-ridge) unit vectors at each anchor."""
    h, w = angles.shape[:2]
    xi = np.clip(np.round(pos[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pos[:, 1]).astype(int), 0, h - 1)
    th = angles[yi, xi]
    t = np.stack([np.cos(th), np.sin(th)], axis=1)
    n = np.stack([-np.sin(th), np.cos(th)], axis=1)
    return t, n


def lens_truth(graph_q: Any, graph_r: Any, spec: Dict[str, Any],
               shape: Tuple[int, int], *, lenses: Optional[Sequence[str]] = None,
               tol: float = TRUTH_TOL, angles: Optional[np.ndarray] = None
               ) -> Dict[int, int]:
    """NEW rule (item 4): same place lens, same position; ridges across-ridge only."""
    qids, qpos, qlens = _nodes(graph_q)
    rids, rpos, rlens = _nodes(graph_r)
    if not qids or not rids:
        return {}
    keep = set(place_lenses(lenses))
    fwd_q = _forward(spec, qpos, shape)                  # query anchor -> reference frame
    tang = nrm = None
    if angles is not None:
        tang, nrm = _frame_vectors(angles, qpos)
    best: Dict[int, List[Tuple[float, int]]] = {}
    for i, u in enumerate(qids):
        if qlens[i] not in keep:
            continue
        rows: List[Tuple[float, int]] = []
        for j, v in enumerate(rids):
            if rlens[j] != qlens[i]:
                continue
            d = rpos[j] - fwd_q[i]
            if tang is not None and is_ridge(qlens[i]):
                dd = abs(float(d @ nrm[i]))                  # ACROSS only
            else:
                dd = float(np.hypot(d[0], d[1]))
            rows.append((dd, v))
        rows.sort()
        if rows and rows[0][0] <= tol:
            best[u] = rows[:2]
    out: Dict[int, int] = {}
    for u, rows in best.items():
        if len(rows) > 1 and rows[1][0] < TRUTH_SEP * max(rows[0][0], 1e-9):
            continue                                          # not separated
        out[u] = rows[0][1]
    # injective
    counts: Dict[int, int] = {}
    for v in out.values():
        counts[v] = counts.get(v, 0) + 1
    return {u: v for u, v in out.items() if counts[v] == 1}


def positional_truth(graph_q: Any, graph_r: Any, spec: Dict[str, Any],
                     shape: Tuple[int, int], tol: float = TRUTH_TOL
                     ) -> Dict[int, int]:
    """OLD rule: the shipped positional truth, reused verbatim from the read code."""
    from Topology.placement_m12_read_real import _truth
    return _truth(graph_q, graph_r, spec, shape)


def both_truths(graph_q: Any, graph_r: Any, spec: Dict[str, Any],
                shape: Tuple[int, int], *, lenses: Optional[Sequence[str]] = None,
                angles: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """OLD and NEW side by side, on the same cell (item 4's reporting unit)."""
    old = positional_truth(graph_q, graph_r, spec, shape)
    new = lens_truth(graph_q, graph_r, spec, shape, lenses=lenses, angles=angles)
    agree = sum(1 for u, v in new.items() if old.get(u) == v)
    return {"old": len(old), "new": len(new), "new_confirmed_by_old": agree,
            "new_pairs": new, "old_pairs": old}
