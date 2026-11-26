"""The exact-transform truth rule, in core (Mission 24).

Moved verbatim from ``Topology/placement_m12_read_real.py``, which re-exports these names for the
Mission-12..19 harnesses that import them from there.  It is here because ``matching/m15_places.py``
-- a core module -- needs it and core may not import from ``Topology/``.

WHAT IT IS.  A synthetic scene is registered to a reference by an exact transform (rotation, uniform
scale, translation).  Under that transform a query anchor has exactly one counterpart in the
reference, so the truth rule can say which reference place a query place IS: nearest within
``TRUTH_TOL`` px, separated from the runner-up by ``TRUTH_SEP`` times the nearest distance,
injective, and MUTUAL (the reference anchor's forward image must pick the query anchor back).  A
pair the anchors cannot resolve is not truth, and is not guessed.

The constants are the Mission-12 ones, unchanged: changing them would change what "correct" means
in every mission that used this rule, so they travel with the code that defined them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

TRUTH_TOL = 1.0          # px
TRUTH_SEP = 2.0          # second nearest must be >= 2x the nearest
BORDER = 8               # px, the border the payload discards


def transform(img: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """The registration's own transform, verbatim (scipy, order 1, edge padding)."""
    from scipy.ndimage import rotate, zoom, shift as nd_shift
    a = img.astype(np.float64)
    if "rot_deg" in spec:
        a = rotate(a, float(spec["rot_deg"]), reshape=False, order=1, mode="nearest")
    if "scale" in spec:
        s = float(spec["scale"])
        a = zoom(a, (s, s, 1.0), order=1)
        h, w = a.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y0, x0 = int(round(cy - img.shape[0] / 2.0)), int(round(cx - img.shape[1] / 2.0))
        a = a[max(0, y0):y0 + img.shape[0], max(0, x0):x0 + img.shape[1]]
        a = np.pad(a, ((0, img.shape[0] - a.shape[0]),
                       (0, img.shape[1] - a.shape[1]), (0, 0)), mode="edge")
    if "shift" in spec:
        a = nd_shift(a, (float(k) for k in (spec["shift"][1], spec["shift"][0], 0.0)),
                     order=1, mode="nearest")
    return np.clip(a, 0, 255).astype(np.uint8)


def inverse(spec: Dict[str, Any], pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Query-image pixels -> original-image pixels, for the registration's spec.

    The rotation convention is the one the Step-2 frame test calibrated: with the image rotated by
    ``+delta``, an anchor at p in the original sits at ``c + R(-delta)(p - c)``, so the inverse is
    ``p = c + R(+delta)(q - c)``.
    """
    h, w = shape
    c = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
    p = np.asarray(pts, dtype=float)
    if "rot_deg" in spec:
        th = np.deg2rad(float(spec["rot_deg"]))
        ct, st = float(np.cos(th)), float(np.sin(th))
        d = p - c
        p = np.stack([ct * d[:, 0] - st * d[:, 1], st * d[:, 0] + ct * d[:, 1]],
                     axis=1) + c
    if "scale" in spec:
        s = float(spec["scale"])
        y0 = int(round((s * h) / 2.0 - h / 2.0))
        x0 = int(round((s * w) / 2.0 - w / 2.0))
        p = np.stack([(p[:, 0] + x0) / s, (p[:, 1] + y0) / s], axis=1)
    if "shift" in spec:
        p = np.stack([p[:, 0] - float(spec["shift"][0]),
                      p[:, 1] - float(spec["shift"][1])], axis=1)
    return p


def anchors(g: Any) -> np.ndarray:
    """Every node's anchor position, in node order."""
    sub = g.subgraph
    return np.asarray([(float(sub.anchor_of_node[n][0]), float(sub.anchor_of_node[n][1]))
                       for n in sorted(sub.nodes) if n in sub.anchor_of_node], dtype=float)


def forward(spec: Dict[str, Any], pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Original-image pixels -> query-image pixels (the inverse of ``inverse``)."""
    h, w = shape
    c = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
    p = np.asarray(pts, dtype=float)
    if "rot_deg" in spec:
        th = np.deg2rad(-float(spec["rot_deg"]))
        ct, st = float(np.cos(th)), float(np.sin(th))
        d = p - c
        p = np.stack([ct * d[:, 0] - st * d[:, 1], st * d[:, 0] + ct * d[:, 1]],
                     axis=1) + c
    if "scale" in spec:
        s = float(spec["scale"])
        y0 = int(round((s * h) / 2.0 - h / 2.0))
        x0 = int(round((s * w) / 2.0 - w / 2.0))
        p = np.stack([p[:, 0] * s - x0, p[:, 1] * s - y0], axis=1)
    if "shift" in spec:
        p = np.stack([p[:, 0] + float(spec["shift"][0]),
                      p[:, 1] + float(spec["shift"][1])], axis=1)
    return p


def truth(gq: Any, gr: Any, spec: Dict[str, Any],
          shape: Tuple[int, int]) -> Dict[int, int]:
    """query node -> the reference node the exact transform says it is.

    Nearest within ``TRUTH_TOL`` px, separated from the runner-up by ``TRUTH_SEP``, injective, and
    mutual.  A pair the anchors cannot resolve is not truth.
    """
    subq, subr = gq.subgraph, gr.subgraph
    qn = [int(n) for n in sorted(subq.nodes) if n in subq.anchor_of_node]
    rn = [int(n) for n in sorted(subr.nodes) if n in subr.anchor_of_node]
    if not qn or not rn:
        return {}
    qa = np.asarray([subq.anchor_of_node[n] for n in qn], dtype=float)
    ra = np.asarray([subr.anchor_of_node[n] for n in rn], dtype=float)
    h, w = shape
    back = inverse(spec, qa, (h, w))
    fwd = forward(spec, ra, (h, w))
    cand: List[Tuple[float, int, int]] = []
    for i, (x, y) in enumerate(back):
        if not (BORDER <= x < w - BORDER and BORDER <= y < h - BORDER):
            continue
        d = np.hypot(ra[:, 0] - x, ra[:, 1] - y)
        order = np.argsort(d)
        j = int(order[0])
        d1 = float(d[j])
        d2 = float(d[order[1]]) if d.size > 1 else float("inf")
        if d1 > TRUTH_TOL or d2 < TRUTH_SEP * max(d1, 1e-9):
            continue
        dd = np.hypot(qa[:, 0] - fwd[j, 0], qa[:, 1] - fwd[j, 1])
        if int(np.argmin(dd)) != i or float(dd.min()) > TRUTH_TOL:
            continue
        cand.append((d1, int(qn[i]), int(rn[j])))
    cand.sort()
    used_r: Dict[int, int] = {}
    for _d, u, v in cand:
        used_r[v] = used_r.get(v, 0) + 1
    return {u: v for _d, u, v in cand if used_r[v] == 1}


#: the private names the Mission-12 code used, kept so the moved call sites read the same
_transform, _inverse, _anchors, _forward, _truth = transform, inverse, anchors, forward, truth
