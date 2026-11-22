"""PLACEMENT edges carried in the SOURCE node's own frame.

WHY A NEW CLASS
  Only 29 of 536 nodes could ever enter the shipped orientation spine, so an
  arm built on it was a no-op (measured: P bit-identical to N).  Widening the
  spine's LENS admission raised reach to 28 % but was refused on frames.  The
  frame question is now settled by the per-node paired frame test
  (``Topology/placement_frame_test.py``):

    provider            textured, in-plane 10/20/30 deg      plain control
    local orientation field   +0.14/+0.33/+0.14  TRACKS      NOISE
    every lens               54 STATIC, 16 NOISE, 4 TRACKS   --

  So the ONLY real frame carrier is the local orientation field at the node's
  ANCHOR, which exists for any node that has an anchor -- it does not depend on
  the lens's stored theta at all.  This module builds edges on that frame.

THE FRAME IS AN AXIS, NOT AN ARROW
  A structure tensor gives the dominant gradient direction, which is
  180-degree ambiguous.  The vector therefore carries ``axial=True`` and is
  scored modulo pi, so a flip of the source's axis is not scored as a different
  placement.  No sign is invented.

TARGETS
  Chosen by a VIEW-STABLE deterministic rule, never "the single nearest" (the
  nearest node flips between views): the k nearest nodes within a radius
  measured in the SOURCE's scale units, ordered by (distance, canonical key).
  Several targets per source, so some survive to the reference.

Only the SOURCE needs a frame; the target needs only a position.  Absolute
position never enters a payload -- only the relative vector, in the source's
frame.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matching.placement import (PlacementVector, median_nn_spacing, node_scale,
                                wrap_ax)

#: A node needs at least this much local coherence for its frame to be used.
#: Fixed below by the coherence census, before any arm is run.
FRAME_COHERENCE_MIN = 0.30
#: Structure-tensor window, pixels (odd).
FIELD_WINDOW = 7
#: How many targets one source keeps.
TARGET_K = 4
#: Radius, in the source's scale units (i.e. in median nearest-neighbour
#: spacings).  Large enough that several targets exist, small enough that the
#: vector is local.
TARGET_RADIUS_SCALES = 4.0


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------
def field_angles(image_u8: np.ndarray, window: int = FIELD_WINDOW
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel AXIAL orientation of the local orientation field.

    Returns ``(angle, coherence)``.  The structure tensor's dominant
    eigenvector is the direction of maximum gradient (the stripe normal), an
    axis on [0, pi).  ``coherence`` is the eigenvalue separation over the
    trace, in [0, 1]: 0 where the patch has no dominant direction, which is
    exactly where a frame must not be claimed.
    """
    img = np.asarray(image_u8, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gy, gx = np.gradient(img)
    h, w = img.shape
    r = window // 2
    pgx = np.pad(gx, r, mode="edge")
    pgy = np.pad(gy, r, mode="edge")

    def boxsum(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.cumsum(a, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        k = window
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    Jxx = boxsum(pgx * pgx)
    Jyy = boxsum(pgy * pgy)
    Jxy = boxsum(pgx * pgy)
    tr = Jxx + Jyy
    ang = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    coh = np.where(tr > 1e-9, disc / np.maximum(tr, 1e-9), 0.0)
    return ang.astype(np.float64), np.maximum(coh, 0.0).astype(np.float64)


def frame_at(ang: np.ndarray, coh: np.ndarray, x: float, y: float
             ) -> Optional[Tuple[float, float]]:
    h, w = ang.shape
    xi, yi = int(round(float(x))), int(round(float(y)))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    a, c = ang[yi, xi], coh[yi, xi]
    if not np.isfinite(a) or c < FRAME_COHERENCE_MIN:
        return None
    return float(a), float(c)


def node_frames(subgraph: Any, ang: np.ndarray, coh: np.ndarray
                ) -> Dict[int, float]:
    """node -> axial frame angle, for every node whose anchor carries one."""
    out: Dict[int, float] = {}
    for n in sorted(subgraph.nodes):
        a = subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        f = frame_at(ang, coh, float(a[0]), float(a[1]))
        if f is not None:
            out[int(n)] = f[0]
    return out


# ---------------------------------------------------------------------------
# deterministic target selection
# ---------------------------------------------------------------------------
def canonical_key(subgraph: Any, n: int) -> Tuple:
    a = subgraph.anchor_of_node.get(n)
    if a is None:
        return (float("inf"), float("inf"), int(n))
    return (float(a[0]), float(a[1]), int(n))


def select_targets(subgraph: Any, frames: Dict[int, float],
                   positions: Dict[int, Tuple[float, float]],
                   scales: Dict[int, float], k: int = TARGET_K,
                   radius_scales: float = TARGET_RADIUS_SCALES
                   ) -> Dict[int, List[int]]:
    """source -> its targets, by a view-stable deterministic rule.

    k nearest within ``radius_scales`` of the SOURCE's own scale, ordered by
    (distance, canonical key).  Ties are broken by the canonical key, never by
    iteration order, and never by "the single nearest" -- that flips between
    views and would leave nothing to check.
    """
    out: Dict[int, List[int]] = {}
    pool = sorted(positions)
    for u in sorted(frames):
        pu = positions.get(u)
        su = scales.get(u)
        if pu is None or not su or su <= 0.0:
            continue
        radius = float(radius_scales) * float(su)
        cand: List[Tuple[float, Tuple, int]] = []
        for v in pool:
            if v == u:
                continue
            pv = positions[v]
            d = math.hypot(pv[0] - pu[0], pv[1] - pu[1])
            if d <= radius and d > 1e-9:
                cand.append((d, canonical_key(subgraph, v), int(v)))
        cand.sort()
        sel = [v for _d, _k, v in cand[:k]]
        if sel:
            out[u] = sel
    return out


# ---------------------------------------------------------------------------
# payloads
# ---------------------------------------------------------------------------
def placement_vector_from(subgraph: Any, u: int, v: int, theta_u: float,
                          scale_u: float,
                          tolerance: Optional[float] = None) -> Optional[PlacementVector]:
    """The vector u -> v in u's OWN axial frame.  Absolute position never enters."""
    pu = subgraph.anchor_of_node.get(u)
    pv = subgraph.anchor_of_node.get(v)
    if pu is None or pv is None or not scale_u or scale_u <= 0.0:
        return None
    dx = float(pv[0]) - float(pu[0])
    dy = float(pv[1]) - float(pu[1])
    dist_px = math.hypot(dx, dy)
    if dist_px <= 1e-9:
        return None
    ang = math.atan2(dy, dx)
    return PlacementVector(direction=wrap_ax(ang - theta_u),
                           distance=dist_px / float(scale_u),
                           log_scale=0.0,
                           source_theta=float(theta_u),
                           source_scale=float(scale_u),
                           tolerance=tolerance,
                           axial=True)


def scale_map(subgraph: Any, fallback: Optional[float] = None
              ) -> Dict[int, float]:
    """node -> the divisor that turns pixels into that node's scale units."""
    pos = {int(n): (float(subgraph.anchor_of_node[n][0]),
                    float(subgraph.anchor_of_node[n][1]))
           for n in sorted(subgraph.nodes) if n in subgraph.anchor_of_node}
    fb = fallback if fallback is not None else median_nn_spacing(list(pos.values()))
    out: Dict[int, float] = {}
    for n in sorted(subgraph.nodes):
        det_scale = None
        sc = node_scale(pos.get(int(n)), fb, det_scale)
        if sc:
            out[int(n)] = float(sc)
    return out


def build_edges(subgraph: Any, image_u8: np.ndarray,
                tolerance: Optional[float] = None,
                k: int = TARGET_K,
                radius_scales: float = TARGET_RADIUS_SCALES,
                **sel_kw: Any) -> Dict[Tuple[int, int, Any], PlacementVector]:
    """The SOURCE_FRAME_PLACEMENT payloads for one graph."""
    from core.types import EdgeType
    ang, coh = field_angles(image_u8)
    frames = node_frames(subgraph, ang, coh)
    pos = {int(n): (float(subgraph.anchor_of_node[n][0]),
                    float(subgraph.anchor_of_node[n][1]))
           for n in sorted(subgraph.nodes) if n in subgraph.anchor_of_node}
    scales = scale_map(subgraph)
    targets = select_targets(subgraph, frames, pos, scales, k=k,
                             radius_scales=radius_scales, **sel_kw)
    out: Dict[Tuple[int, int, Any], PlacementVector] = {}
    for u in sorted(targets):
        for v in targets[u]:
            pv = placement_vector_from(subgraph, u, v, frames[u],
                                       scales.get(u, 0.0), tolerance)
            if pv is not None:
                out[(int(u), int(v), EdgeType.SOURCE_FRAME_PLACEMENT)] = pv
    return out


def attach(subgraph: Any, image_u8: np.ndarray, tolerance: Optional[float] = None
           ) -> Any:
    """A COPY of the subgraph with the placement edges and payloads added.

    The shipped edge set is left intact and untouched, so the frozen R arm
    (which filters on its own classes) sees exactly what it always saw.  Only
    the node set must be identical across arms, and this adds no nodes.
    """
    from dataclasses import replace
    pay = build_edges(subgraph, image_u8, tolerance)
    edges = frozenset(set(subgraph.edges) | set(pay))
    ep = dict(subgraph.edge_params or {})
    return replace(subgraph, edges=edges, edge_params=ep), pay
