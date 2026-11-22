"""MISSION 5 STAGE 3 -- the true-keypoint frame layer (DARK).

The front end's placement endpoints are region centroids and integer Gabor
argmax points.  Neither is a feature a second view can be asked to reproduce:
a centroid is the average of a tile, and an argmax is a pixel.  Measured
node-to-node on 20 FIT scenes, localisation given the right node is 1.08-1.48 px
median against a 0.50 px gate.

This module detects the things a view CAN reproduce, analytically and
deterministically:

    corner  a Harris corner, sub-pixel by a quadratic fit of the response
    blob    a multi-scale LoG extremum, with its scale
    edge    a point on a boundary, sampled by ARC LENGTH and flagged 1-D --
            its position is reliable ACROSS the boundary and undefined along it

Every detection carries what a frame needs and what a tolerance needs:

    theta      dominant orientation (structure tensor, sub-bin)
    coherence  (l1 - l2) / (l1 + l2) of the structure tensor
    scale      the sigma at which the response peaked
    var_x/y    per-axis position variance from the fit curvature
    dim        2 for a point, 1 for a boundary arc

Nothing here reads truth: no masks, no depth, no tracks.  It takes pixels.

NMS is applied in (x, y, scale) and NEVER MOVES A POSITION: a suppressed
detection is dropped, the survivor keeps the position it measured.

The layer is DARK (`parameters.FE_KEYPOINTS` is False), and the flag is in
`eth80_frontend._FRONTEND_SWITCHES`, so a lit run cannot be served a dark graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: lens type registered for each detection kind.  All three are declared in
#: `matching.relation_frame.FRAME_LENS_TYPES`.
LENS_OF_KIND: Dict[str, str] = {
    "corner": "CornerPoint",
    "blob": "Blob",
    "edge": "BoundaryArc",
}

#: The minimum separation NMS enforces, in pixels (and in octaves of scale).
#: Stated here because the mission requires it to be stated: two detections
#: closer than this in space, AND within SCALE_OCTAVES of each other, cannot
#: both survive.
MIN_SEPARATION_PX: float = 2.0
SCALE_OCTAVES: float = 0.6          # 2 ** 0.6 = 1.52x

DEFAULT_SIGMAS: Tuple[float, ...] = (1.0, 1.6, 2.5, 4.0)


@dataclass(frozen=True)
class Keypoint:
    kind: str
    x: float
    y: float
    scale: float
    theta: float
    coherence: float
    resp: float
    var_x: float
    var_y: float
    dim: int

    @property
    def lens(self) -> str:
        return LENS_OF_KIND.get(self.kind, "Keypoint")

    def as_row(self) -> Dict[str, Any]:
        return {"kind": self.kind, "x": self.x, "y": self.y,
                "scale": self.scale, "theta": self.theta,
                "coherence": self.coherence, "resp": self.resp,
                "var_x": self.var_x, "var_y": self.var_y, "dim": self.dim,
                "lens": self.lens}


def _gray(image: Any) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if a.size and float(np.max(a)) > 1.5:
        a = a / 255.0
    return a


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma, mode="nearest")


def _parabola(c0: float, cm: float, cp: float) -> Tuple[float, float]:
    """Sub-pixel offset and its variance from a 3-point parabola.

    Returns (offset in [-0.5, 0.5], variance).  A flat or convex window has no
    peak, and then the offset is 0 and the variance is the maximum -- an honest
    statement that the fit says nothing, not a silent 0.5.
    """
    den = cm + cp - 2.0 * c0
    if den >= -1e-12:
        return 0.0, 1.0
    off = 0.5 * (cm - cp) / den
    off = max(-0.5, min(0.5, float(off)))
    var = 1.0 / (2.0 * abs(den))
    return off, float(max(1e-3, min(1.0, var)))


def _local_maxima(resp: np.ndarray, thr: float, cap: int) -> List[Tuple[float, int, int]]:
    """3x3 local maxima of `resp` above `thr`, strongest first, deterministic."""
    from scipy.ndimage import maximum_filter
    if resp.size == 0:
        return []
    m = maximum_filter(resp, size=3, mode="nearest")
    ys, xs = np.nonzero((resp >= m) & (resp > thr) & np.isfinite(resp))
    if ys.size == 0:
        return []
    vals = resp[ys, xs]
    order = np.lexsort((xs, ys, -vals))
    out: List[Tuple[float, int, int]] = []
    for i in order[:cap]:
        out.append((float(vals[i]), int(ys[i]), int(xs[i])))
    return out


def _orientation(jxx: float, jyy: float, jxy: float) -> Tuple[float, float]:
    """(theta, coherence) of a 2x2 structure tensor."""
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(0.0, tr * tr / 4.0 - det)
    root = float(np.sqrt(disc))
    l1 = tr / 2.0 + root
    l2 = tr / 2.0 - root
    coh = (l1 - l2) / (l1 + l2) if (l1 + l2) > 1e-12 else 0.0
    theta = 0.5 * float(np.arctan2(2.0 * jxy, jxx - jyy))
    return theta, float(max(0.0, min(1.0, coh)))


def _nms(points: List[Keypoint], min_sep: float) -> List[Keypoint]:
    keep: List[Keypoint] = []
    for p in sorted(points, key=lambda q: (-q.resp, q.kind, q.x, q.y, q.scale)):
        clash = False
        for q in keep:
            if (abs(p.x - q.x) <= min_sep and abs(p.y - q.y) <= min_sep
                    and abs(np.log2(p.scale / q.scale)) <= SCALE_OCTAVES):
                clash = True
                break
        if not clash:
            keep.append(p)
    return keep


def detect(image: Any, sigmas: Sequence[float] = DEFAULT_SIGMAS,
           harris_k: float = 0.05, min_sep: float = MIN_SEPARATION_PX,
           max_per_kind: int = 220, corner_quantile: float = 98.5,
           blob_quantile: float = 98.5, edge_quantile: float = 97.0
           ) -> List[Keypoint]:
    """Detect corners, blobs and boundary-arc points.  Deterministic."""
    from scipy.ndimage import gaussian_laplace
    g = _gray(image)
    if g.ndim != 2 or min(g.shape) < 5:
        return []
    out: List[Keypoint] = []

    for sigma in sigmas:
        gs = _blur(g, sigma)
        gy, gx = np.gradient(gs)
        w = sigma * 1.5
        jxx = _blur(gx * gx, w)
        jyy = _blur(gy * gy, w)
        jxy = _blur(gx * gy, w)

        # --- corners: Harris, sub-pixel from the response itself
        for scale_norm in (1.0,):
            har = (jxx * jyy - jxy * jxy) - harris_k * (jxx + jyy) ** 2
            har = scale_norm * har
            finite = har[np.isfinite(har)]
            if finite.size == 0:
                continue
            thr = float(np.percentile(finite, corner_quantile))
            if not np.isfinite(thr) or thr <= 0.0:
                thr = float(np.max(finite)) * 0.5
            for val, y, x in _local_maxima(har, thr, max_per_kind):
                ox, vx = _parabola(float(har[y, x]), float(har[y, x - 1]),
                                   float(har[y, x + 1])) if 0 < x < g.shape[1] - 1 else (0.0, 1.0)
                oy, vy = _parabola(float(har[y, x]), float(har[y - 1, x]),
                                   float(har[y + 1, x])) if 0 < y < g.shape[0] - 1 else (0.0, 1.0)
                theta, coh = _orientation(float(jxx[y, x]), float(jyy[y, x]),
                                          float(jxy[y, x]))
                out.append(Keypoint("corner", x + ox, y + oy, float(sigma),
                                    theta, coh, float(val), vx, vy, 2))

        # --- blobs: LoG, sub-pixel from the LoG response
        log = -float(sigma) ** 2 * gaussian_laplace(g, float(sigma),
                                                    mode="nearest")
        alog = np.abs(log)
        finite = alog[np.isfinite(alog)]
        if finite.size:
            thr = float(np.percentile(finite, blob_quantile))
            for val, y, x in _local_maxima(alog, thr, max_per_kind):
                ox, vx = _parabola(float(alog[y, x]), float(alog[y, x - 1]),
                                   float(alog[y, x + 1])) if 0 < x < g.shape[1] - 1 else (0.0, 1.0)
                oy, vy = _parabola(float(alog[y, x]), float(alog[y - 1, x]),
                                   float(alog[y + 1, x])) if 0 < y < g.shape[0] - 1 else (0.0, 1.0)
                theta, coh = _orientation(float(jxx[y, x]), float(jyy[y, x]),
                                          float(jxy[y, x]))
                out.append(Keypoint("blob", x + ox, y + oy, float(sigma),
                                    theta, coh, float(val), vx, vy, 2))

        # --- boundary arc points: gradient-magnitude ridge, flagged 1-D
        mag = np.hypot(gx, gy)
        finite = mag[np.isfinite(mag)]
        if finite.size:
            thr = float(np.percentile(finite, edge_quantile))
            for val, y, x in _local_maxima(mag, thr, max_per_kind):
                if not (0 < x < g.shape[1] - 1 and 0 < y < g.shape[0] - 1):
                    continue
                ox, vx = _parabola(float(mag[y, x]), float(mag[y, x - 1]),
                                   float(mag[y, x + 1]))
                oy, vy = _parabola(float(mag[y, x]), float(mag[y - 1, x]),
                                   float(mag[y + 1, x]))
                theta, coh = _orientation(float(jxx[y, x]), float(jyy[y, x]),
                                          float(jxy[y, x]))
                # ACROSS the boundary is reliable, ALONG it is not: the
                # variance is inflated along theta and tightened across it.
                across = 0.5 * (vx + vy)
                along = min(1.0, across + 4.0)
                out.append(Keypoint("edge", x + ox, y + oy, float(sigma),
                                    theta + np.pi / 2.0, coh, float(val),
                                    across, along, 1))

    return _nms(out, float(min_sep))


def detected(out: List[Keypoint]) -> Dict[str, int]:
    """Counts by kind, for a caller that only wants to know what fired."""
    c: Dict[str, int] = {}
    for p in out:
        c[p.kind] = c.get(p.kind, 0) + 1
    return c
