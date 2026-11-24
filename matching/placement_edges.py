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
#:
#: DERIVED ON DEV, from the per-node paired residual STRATIFIED BY COHERENCE
#: (in-plane rotations 10/20/30 on all four DEV scenes):
#:
#:   coherence   n     median |residual|   share within 15 deg
#:   0.30-0.50  129        14.68 deg            0.52
#:   0.50-0.70  264         8.08 deg            0.64
#:   0.70-0.90  453         6.44 deg            0.80
#:   0.90-1.01  1989        2.12 deg            0.97
#:
#: At 0.30-0.50 the frame is wrong far more often than the scorer's own 15-deg
#: band, so admitting it would be admitting noise.  The floor is set at 0.50:
#: the first band whose median residual sits inside that band.
FRAME_COHERENCE_MIN = 0.50
#: Structure-tensor window, pixels (odd).
FIELD_WINDOW = 7
#: How many targets one source keeps.
#:
#: DERIVED ON DEV (4 scenes x 5 views), coverage of OBJECT-anchored ambiguous
#: nodes under N's frozen mapping / pair survival:
#:
#:   k    edges    coverage(N)   survival
#:   4     1287        26.3 %      46.3 %
#:   8     2574        42.6 %      55.6 %
#:  12     3862        54.9 %      63.4 %
#:  16     5149        65.8 %      67.1 %
#:  24     7705        77.5 %      72.1 %
#:  32    10141        85.1 %      76.3 %
#:
#: 16 is taken: it clears the 50 % coverage target with margin while keeping
#: the degree near 7 per node.  Beyond that the returns flatten and every
#: neighbour adds a weaker constraint.
TARGET_K = 16
#: Radius, in the source's scale units (i.e. in median nearest-neighbour
#: spacings).  MEASURED NOT TO BIND on these graphs: k nearest are always found
#: inside it (sweeping 6/12/24/48 gave byte-identical edge counts, because the
#: node field is dense -- median NN spacing 5.0 px).  It is kept as a bound so
#: that a sparse graph cannot produce a long-range "placement" vector, and its
#: non-binding status is recorded rather than presented as a tuned value.
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


def _bilinear_axial(ang: np.ndarray, x: float, y: float) -> Optional[float]:
    """Interpolate an AXIAL angle field at a sub-pixel position.

    An axial angle lives on a circle with period pi, so interpolating the number
    itself would tear at the wrap (a 179-degree and a 1-degree neighbour would
    average to 90).  The interpolation is therefore done on the DOUBLED angle as
    a unit vector -- cos(2a), sin(2a) -- and the result is halved back:
    exact equivariance under a rotation of the image, and no wrap artefact.
    """
    h, w = ang.shape
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = float(x - x0), float(y - y0)
    ws = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
    cc = ss = 0.0
    for wt, xx, yy in zip(ws, (x0, x1, x0, x1), (y0, y0, y1, y1)):
        if wt == 0.0:
            continue
        a = float(ang[yy, xx])
        if not np.isfinite(a):
            return None
        cc += wt * np.cos(2.0 * a)
        ss += wt * np.sin(2.0 * a)
    if cc == 0.0 and ss == 0.0:
        return None
    return float(0.5 * np.arctan2(ss, cc))


def frame_at(ang: np.ndarray, coh: np.ndarray, x: float, y: float,
             subpixel: Optional[bool] = None) -> Optional[Tuple[float, float]]:
    """(angle, coherence) at (x, y), or None below the coherence bar.

    `subpixel` (default: the dark `PLACEMENT_FRAME_SUBPIXEL` parameter) samples the
    fields BILINEARLY at the anchor instead of rounding it to a pixel.  The old
    rounding is the shipped behaviour and stays the default: an anchor at x = 10.6
    was read at pixel 11, and under rotation the anchor moves by a fraction of a
    pixel, so the rounded provider can switch pixel and lose the frame.
    """
    if subpixel is None:
        subpixel = bool(_param("PLACEMENT_FRAME_SUBPIXEL", False))
    h, w = ang.shape
    if subpixel:
        if not (0.0 <= float(x) <= w - 1 and 0.0 <= float(y) <= h - 1):
            return None
        c = _bilinear(coh, float(x), float(y))
        if c is None or c < FRAME_COHERENCE_MIN:
            return None
        a = _bilinear_axial(ang, float(x), float(y))
        if a is None or not np.isfinite(a):
            return None
        return float(a), float(c)
    xi, yi = int(round(float(x))), int(round(float(y)))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    a, c = ang[yi, xi], coh[yi, xi]
    if not np.isfinite(a) or c < FRAME_COHERENCE_MIN:
        return None
    return float(a), float(c)


#: [MISSION 12 / step 3] THE KEYPOINT LAYER'S OWN FRAME.  The field providers read
#: the orientation off a BOX window of the gradient tensor and sample it at the
#: nearest pixel.  The keypoint layer (`lenses/keypoints.py`) reads it off a
#: GAUSSIAN-smoothed tensor at its own sigma ladder and at SUB-PIXEL positions, and
#: those two differences are exactly what the ambiguous (low-contrast) places need:
#: `frame_at` rounds a sub-pixel anchor to a neighbour pixel that may be below the
#: coherence bar while the keypoint's own (x, y) sits on the ridge.
#:
#: Ladder and bar are the keypoint layer's own constants; nothing is fitted here.
KEYPOINT_SIGMAS: Tuple[float, ...] = (1.0, 1.6, 2.5, 4.0)


def _tensor_at_sigma(gx: np.ndarray, gy: np.ndarray, sigma: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(angle, coherence) of the Gaussian-smoothed structure tensor at `sigma`."""
    from scipy.ndimage import gaussian_filter
    jxx = gaussian_filter(gx * gx, sigma, mode="nearest")
    jyy = gaussian_filter(gy * gy, sigma, mode="nearest")
    jxy = gaussian_filter(gx * gy, sigma, mode="nearest")
    tr = jxx + jyy
    ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coh = np.where(tr > 1e-9, disc / np.maximum(tr, 1e-9), 0.0)
    return ang.astype(np.float64), np.maximum(coh, 0.0).astype(np.float64)


def _bilinear(field: np.ndarray, x: float, y: float) -> Optional[float]:
    h, w = field.shape
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v = (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
         + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)
    return float(v)


def node_frames_keypoint(subgraph: Any, image_u8: np.ndarray,
                         coherence_min: float = FRAME_COHERENCE_MIN,
                         sigmas: Sequence[float] = KEYPOINT_SIGMAS,
                         sigma_rule: Optional[str] = None
                         ) -> Tuple[Dict[int, float], Dict[int, int]]:
    """node -> axial frame from the KEYPOINT estimator, and the sigma it came from.

    The smallest sigma whose SUB-PIXEL (bilinear) coherence at the node's anchor
    clears the bar wins.  No frame is invented: a node incoherent at every sigma
    gets none, exactly as every other provider behaves.
    """
    # [MISSION 13 / step 1] WHICH SIGMA WINS.  The deployed rule is the SMALLEST
    # sigma whose coherence clears the bar.  M12's frame test measured the LARGEST
    # sigma -- `max(KEYPOINT_SIGMAS)` -- and reported it as "the keypoint provider";
    # the two configurations differ by ~0.24 in equivariance (M13, node level), so
    # the rule is now explicit and configurable, and the harness can name either.
    if sigma_rule is None:
        sigma_rule = str(_param("PLACEMENT_FRAME_KEYPOINT_SIGMA", "smallest"))
    order = sorted((float(s) for s in sigmas),
                   reverse=(str(sigma_rule).lower() == "largest"))
    img = np.asarray(image_u8, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gy, gx = np.gradient(img)
    fields = [(float(sig),) + _tensor_at_sigma(gx, gy, float(sig))
              for sig in order]
    theta: Dict[int, float] = {}
    sig_of: Dict[int, int] = {}
    for n in sorted(subgraph.nodes):
        a = subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        x, y = float(a[0]), float(a[1])
        for sig, ang, coh in fields:
            c = _bilinear(coh, x, y)
            if c is None or c < coherence_min:
                continue
            v = _bilinear(ang, x, y)
            if v is None or not np.isfinite(v):
                continue
            theta[int(n)] = float(v)
            sig_of[int(n)] = int(round(sig * 10))
            break
    return theta, sig_of


def node_frames_keypoint_then_field(subgraph: Any, image_u8: np.ndarray,
                                    coherence_min: float = FRAME_COHERENCE_MIN,
                                    fallback: Optional[str] = None,
                                    sigma_rule: Optional[str] = None,
                                    ) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Keypoint frame where it exists, a FIELD fallback where it does not.

    [MISSION 13 / step 1a] THE FALLBACK USED TO BE THE MULTISCALE LADDER, and the
    ladder is measurably the WORST of the three frame sources under rotation
    (share within 3 deg: 0.531, against 0.626 for the single 7x7 field) because
    the window a node picks can CHANGE when the image is rotated -- a 23x23
    estimate at a pixel is not a 7x7 estimate at that pixel.  Adding it to the
    keypoint's misses therefore imported the deficiency it was meant to repair.
    The fallback is now the single 7x7 field by default (parameter
    `PLACEMENT_FRAME_FALLBACK`), and "multiscale" is kept only so the M12
    configuration stays reproducible.

    The composition can only ADD frames: a node the keypoint estimator gave a
    frame keeps exactly that frame.
    """
    if fallback is None:
        fallback = str(_param("PLACEMENT_FRAME_FALLBACK", "field"))
    theta, src = node_frames_keypoint(subgraph, image_u8, coherence_min,
                                      sigma_rule=sigma_rule)
    theta = dict(theta)
    src = dict(src)
    miss = [n for n in sorted(subgraph.nodes) if int(n) not in theta]
    if miss and str(fallback).lower() == "none":
        return theta, src        # keypoint only: no fallback at all
    if miss:
        if str(fallback).lower() == "multiscale":
            f_theta, wins = node_frames_multiscale(
                subgraph, field_multiscale(image_u8), coherence_min)
            for n in miss:
                if int(n) in f_theta:
                    theta[int(n)] = float(f_theta[int(n)])
                    src[int(n)] = -int(wins[int(n)])   # negative: the field provider
        else:
            ang, coh = field_angles(image_u8)
            f_theta = node_frames(subgraph, ang, coh, coherence_min=coherence_min)
            for n in miss:
                if int(n) in f_theta:
                    theta[int(n)] = float(f_theta[int(n)])
                    src[int(n)] = -FIELD_WINDOW
    return theta, src


def node_frames(subgraph: Any, ang: np.ndarray, coh: np.ndarray,
                coherence_min: Optional[float] = None,
                subpixel: Optional[bool] = None) -> Dict[int, float]:
    """node -> axial frame angle, for every node whose anchor carries one.

    `coherence_min` defaults to the module bar; the keypoint-then-field
    composition passes its own so the fallback uses the SAME bar as the primary
    estimator (otherwise the composition would silently change the bar).
    """
    out: Dict[int, float] = {}
    for n in sorted(subgraph.nodes):
        a = subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        f = frame_at(ang, coh, float(a[0]), float(a[1]), subpixel=subpixel)
        if f is None:
            continue
        if coherence_min is not None and f[1] < float(coherence_min):
            continue
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
    # [MISSION 10 / step 4] The radius is a multiple of the source's own scale,
    # which on a dense image can reach past the neighbouring place.  When the dark
    # PLACEMENT_SPACING_RADIUS flag is on, the radius is CAPPED by the local place
    # spacing (nearest neighbour among places).  It is only ever a cap: a scene
    # whose places are far apart keeps the scale-relative radius it had.
    spacing: Dict[int, float] = {}
    try:
        from matching import placement_spacing as _SP
        if _SP.enabled("radius"):
            for u in pool:
                pu0 = positions[u]
                best = float("inf")
                for v in pool:
                    if v == u:
                        continue
                    pv0 = positions[v]
                    d = math.hypot(pv0[0] - pu0[0], pv0[1] - pu0[1])
                    if 1e-9 < d < best:
                        best = d
                if math.isfinite(best):
                    spacing[u] = best
    except Exception:
        spacing = {}
    for u in sorted(frames):
        pu = positions.get(u)
        su = scales.get(u)
        if pu is None or not su or su <= 0.0:
            continue
        radius = float(radius_scales) * float(su)
        if spacing:
            from matching import placement_spacing as _SP
            cap = _SP.radius_cap(spacing.get(u))
            if cap is not None:
                radius = min(radius, float(cap))
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


# ---------------------------------------------------------------------------
# [MISSION 5 / STAGE 2a] THE POPULATION LEVER -- ridge tokens
# ---------------------------------------------------------------------------
#: Lenses whose position along their own structure is undefined (the aperture
#: problem).  Measured node-to-node on 20 FIT scenes (MISSION5_REPORT.md):
#: identity 0.213 (ContourFragment) and 0.212 (TextureBoundary), against
#: 0.71-0.95 for every other lens.  They are 86 of the 179 placement nodes, so
#: they dominate a population whose geometry is exactly the geometry they
#: cannot supply.  A ridge token knows WHERE THE LINE IS, not where on it.
RIDGE_LENSES: Tuple[str, ...] = ("ContourFragment", "TextureBoundary")


def _param(name: str, default: Any) -> Any:
    """A parameter if one has been declared, else the dark default."""
    try:
        import parameters as P
    except Exception:
        return default
    return getattr(P, name, default)


def lens_map_of(g: Any) -> Dict[int, str]:
    """node -> lens name, from an ImageGraph's signatures."""
    sigs = getattr(g, "sigs", None) or {}
    out: Dict[int, str] = {}
    for n in sorted(int(k) for k in sigs):
        s = sigs.get(n)
        if s is not None:
            out[int(n)] = str(s[0])
    return out


def build_edges(subgraph: Any, image_u8: np.ndarray,
                tolerance: Optional[float] = None,
                k: int = TARGET_K,
                radius_scales: float = TARGET_RADIUS_SCALES,
                object_of: Optional[Dict[int, int]] = None,
                intra_only: bool = False,
                lens_map: Optional[Dict[int, str]] = None,
                exclude_lenses: Optional[Sequence[str]] = None,
                ridge_across_only: Optional[bool] = None,
                collapse_places: Optional[bool] = None,
                endpoint_lenses: Optional[Sequence[str]] = None,
                **sel_kw: Any) -> Dict[Tuple[int, int, Any], PlacementVector]:
    """The SOURCE_FRAME_PLACEMENT payloads for one graph.

    ``intra_only`` drops every edge whose endpoints sit on DIFFERENT objects.
    That is the NEGATIVE-A control, and it belongs in the BUILDER: with
    intra-object edges only, the two identical objects contribute exactly the
    same geometry, so a correct score cannot tell the identity mapping from the
    swap -- any difference is a positional leak.  NEGATIVE-B is the same scene
    with this left False, where the swap reverses the cross-object vectors and
    identity must WIN.
    """
    from core.types import EdgeType
    # [MISSION 11 / step 5, DARK] The frame provider.  The single 7x7 window leaves
    # 59 % of real-image anchors below the coherence bar (measured, m11_coverage.json)
    # and a node without a frame earns no payload edge and no judged pair.  With
    # PLACEMENT_FRAME_MULTISCALE the smallest window that clears the bar wins, which
    # is the provider the module already ships for exactly this reason.  Off, this is
    # the old line.
    # [MISSION 12 / step 3, DARK] The keypoint estimator first (Gaussian tensor at
    # the keypoint layer's own sigmas, sub-pixel bilinear sampling), the multiscale
    # field where it has no frame.  It can only ADD frames, never move one the
    # keypoint estimator gave, so a node that had a frame keeps the same value when
    # the multiscale flag is also on.
    if bool(_param("PLACEMENT_FRAME_KEYPOINT", False)):
        frames, _wins = node_frames_keypoint_then_field(
            subgraph, image_u8,
            sigma_rule=_param("PLACEMENT_FRAME_KEYPOINT_SIGMA", "smallest"))
    elif bool(_param("PLACEMENT_FRAME_MULTISCALE", False)):
        frames, _wins = node_frames_multiscale(subgraph, field_multiscale(image_u8))
    else:
        ang, coh = field_angles(image_u8)
        frames = node_frames(subgraph, ang, coh)
    pos = {int(n): (float(subgraph.anchor_of_node[n][0]),
                    float(subgraph.anchor_of_node[n][1]))
           for n in sorted(subgraph.nodes) if n in subgraph.anchor_of_node}
    lens = dict(lens_map or {})
    excl = tuple(exclude_lenses if exclude_lenses is not None
                 else _param("PLACEMENT_EXCLUDE_LENSES", ()))
    ridge = bool(ridge_across_only if ridge_across_only is not None
                 else _param("PLACEMENT_RIDGE_ACROSS_ONLY", False))
    across_min = float(_param("PLACEMENT_RIDGE_ACROSS_MIN", 0.5))

    # [MISSION 5 / STAGE 2] ONE PLACE PER SITE.
    #
    # Census (m5_stage0_census.json): 179 placement nodes sit at 55 places,
    # max 16 at one site.  Edges already avoid same-place pairs (0 of 4384),
    # so the cost is not degenerate edges but DILUTION: a place with 16 nodes
    # contributes 16 candidates to every neighbour slot, and the extra copies
    # are near-duplicates whose vectors a rival can match as well as the true
    # partner can.  Collapsing picks ONE representative per site -- the
    # canonical-key minimum, never "the first seen" -- so the edge set is
    # place-to-place and the slot contents are distinct places.
    collapse = bool(collapse_places if collapse_places is not None
                    else _param("PLACEMENT_COLLAPSE_PLACES", False))
    if collapse and pos:
        grid = float(_param("PLACEMENT_PLACE_GRID_PX", 1.0))
        sites: Dict[Tuple[int, int], List[int]] = {}
        for u in sorted(pos):
            site_key = (int(round(pos[u][0] / grid)),
                        int(round(pos[u][1] / grid)))
            sites.setdefault(site_key, []).append(u)
        rep: Dict[int, int] = {}
        for site_key in sorted(sites):
            members = sites[site_key]
            r = min(members, key=lambda v: canonical_key(subgraph, v))
            for v in members:
                rep[v] = int(r)
        keep = sorted({rep[u] for u in pos})
        pos = {u: pos[u] for u in keep if u in pos}
        frames = {u: frames[u] for u in keep if u in frames}

    # [MISSION 7 / STEP 2] PLACEMENT ENDPOINTS ARE PLACES.
    #
    # A whitelist, not a blacklist: only lenses that are a PLACE can be an
    # endpoint.  An attribute or ridge node describes or bounds a place, so it
    # attaches to one (coverage 0.736 within 4 px, Mission 6) and inherits its
    # support; it does not get to be an endpoint itself.  Empty (the default)
    # means "no restriction", so the shipped behaviour is untouched.
    ends = tuple(endpoint_lenses if endpoint_lenses is not None
                 else _param("PLACEMENT_ENDPOINT_LENSES", ()))
    if ends and lens:
        pos = {u: p for u, p in pos.items() if lens.get(u) in ends}
        frames = {u: t for u, t in frames.items() if lens.get(u) in ends}

    if excl:
        # [STAGE 2a (i)] A ridge token is not a place.  Drop it as an endpoint
        # on BOTH sides -- source and target -- so no edge can hang off one.
        pos = {u: p for u, p in pos.items() if lens.get(u) not in excl}
        frames = {u: t for u, t in frames.items() if lens.get(u) not in excl}

    scales = scale_map(subgraph)
    targets = select_targets(subgraph, frames, pos, scales, k=k,
                             radius_scales=radius_scales, **sel_kw)
    out: Dict[Tuple[int, int, Any], PlacementVector] = {}
    for u in sorted(targets):
        seen_place: set = set()
        for v in targets[u]:
            v = rep.get(int(v), int(v)) if collapse else int(v)
            if v in seen_place:
                continue
            seen_place.add(v)
            if intra_only and object_of is not None:
                if object_of.get(int(u)) != object_of.get(int(v)):
                    continue
            pv = placement_vector_from(subgraph, u, v, frames[u],
                                       scales.get(u, 0.0), tolerance)
            if pv is None:
                continue
            if ridge and lens.get(int(u)) in RIDGE_LENSES:
                # [STAGE 2a (ii)] Keep the ridge token, score only the axis it
                # actually defines.  `direction` is axial-wrapped, so a
                # neighbour ALONG the ridge sits at direction ~ 0 (mod pi) and
                # |sin| ~ 0; ACROSS the ridge sits at |sin| ~ 1.
                if abs(math.sin(pv.direction)) < across_min:
                    continue
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

# ---------------------------------------------------------------------------
# multi-scale field: the frame provider has to work where the texture is smooth
# ---------------------------------------------------------------------------
#: Windows tried, smallest FIRST.  A node takes the smallest window whose
#: coherence clears FRAME_COHERENCE_MIN, so a finely textured region keeps the
#: local estimate while a smooth region still gets one.  Measured motivation:
#: with the single 7x7 window only 33.8 % of AMBIGUOUS nodes had a frame --
#: ambiguous nodes are the low-contrast ones, so the provider failed exactly
#: where the mission needs it.
FIELD_WINDOWS: Tuple[int, ...] = (7, 15, 23, 31)


def field_multiscale(image_u8: np.ndarray,
                     windows: Sequence[int] = FIELD_WINDOWS
                     ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    return [(int(w),) + field_angles(image_u8, w) for w in windows]


def node_frames_multiscale(subgraph: Any,
                           fields: Sequence[Tuple[int, np.ndarray, np.ndarray]],
                           coherence_min: float = FRAME_COHERENCE_MIN
                           ) -> Tuple[Dict[int, float], Dict[int, int]]:
    """node -> axial frame, and node -> the window it came from.

    Smallest window that clears the coherence bar wins.  No frame is invented:
    a node whose anchor is incoherent at EVERY scale gets none, and an edge with
    no frame is not emitted.
    """
    theta: Dict[int, float] = {}
    win: Dict[int, int] = {}
    for n in sorted(subgraph.nodes):
        a = subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        for w, ang, coh in fields:
            f = frame_at(ang, coh, float(a[0]), float(a[1]))
            if f is None:
                continue
            theta[int(n)] = f[0]
            win[int(n)] = int(w)
            break
    return theta, win

# ---------------------------------------------------------------------------
# signing the axis
# ---------------------------------------------------------------------------
def field_signed(image_u8: np.ndarray, window: int = FIELD_WINDOW
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """The orientation field as a SIGNED direction, plus coherence.

    The structure tensor alone gives an AXIS (180-degree ambiguous).  The
    mission allows either folding the score (which loses the flip) or ADDING A
    CUE THAT FIXES THE SIGN; the cue here is the sign of the mean intensity
    gradient along that axis: the frame points the way intensity INCREASES.
    That is a real, measured direction, not an invented one.

    Checked before adoption, per-node paired on DEV rotations:

        stimulus          delta   axial median |res|   signed median |res|
        sphere+box        10/20/30   1.46/3.67/4.09      2.46/4.45/4.79
        two identical     10/20/30   1.41/2.62/3.14      1.65/2.69/3.98

    i.e. the signed residual concentrates at ZERO (not at +/-delta, which is
    what a static provider would give), so the sign is carried.  It costs a
    couple of degrees of noise and buys the full turn.
    """
    img = np.asarray(image_u8, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gy, gx = np.gradient(img)
    ang, coh = field_angles(img, window)
    r = window // 2
    pgx = np.pad(gx, r, mode="edge")
    pgy = np.pad(gy, r, mode="edge")

    def boxsum(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.cumsum(a, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        k = window
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    mgx = boxsum(pgx)
    mgy = boxsum(pgy)
    proj = np.cos(ang) * mgx + np.sin(ang) * mgy
    signed = np.where(proj >= 0.0, ang, ang + math.pi)
    return signed.astype(np.float64), coh


def node_frames_signed(subgraph: Any, image_u8: np.ndarray,
                       coherence_min: float = FRAME_COHERENCE_MIN,
                       windows: Sequence[int] = FIELD_WINDOWS
                       ) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Signed frames, multi-scale: smallest window that clears the bar wins."""
    fields = []
    for w in windows:
        ang, coh = field_signed(image_u8, w)
        fields.append((int(w), ang, coh))
    return node_frames_multiscale(subgraph, fields, coherence_min)


def build_edges_signed(subgraph: Any, image_u8: np.ndarray,
                       tolerance: Optional[float] = None,
                       k: int = TARGET_K,
                       radius_scales: float = TARGET_RADIUS_SCALES,
                       coherence_min: float = FRAME_COHERENCE_MIN,
                       ) -> Dict[Tuple[int, int, Any], PlacementVector]:
    """SOURCE_FRAME_PLACEMENT payloads on the SIGNED field frame."""
    from core.types import EdgeType
    frames, _win = node_frames_signed(subgraph, image_u8, coherence_min)
    pos = {int(n): (float(subgraph.anchor_of_node[n][0]),
                    float(subgraph.anchor_of_node[n][1]))
           for n in sorted(subgraph.nodes) if n in subgraph.anchor_of_node}
    scales = scale_map(subgraph)
    targets = select_targets(subgraph, frames, pos, scales, k=k,
                             radius_scales=radius_scales)
    out: Dict[Tuple[int, int, Any], PlacementVector] = {}
    from matching.placement import wrap_pi
    for u in sorted(targets):
        pu = pos.get(u)
        su = scales.get(u, 0.0)
        if pu is None or su <= 0.0:
            continue
        for v in targets[u]:
            pv = pos.get(v)
            if pv is None:
                continue
            dx, dy = pv[0] - pu[0], pv[1] - pu[1]
            dpx = math.hypot(dx, dy)
            if dpx <= 1e-9:
                continue
            out[(int(u), int(v), EdgeType.SOURCE_FRAME_PLACEMENT)] = \
                PlacementVector(direction=wrap_pi(math.atan2(dy, dx) - frames[u]),
                                distance=dpx / su,
                                log_scale=0.0,
                                source_theta=float(frames[u]),
                                source_scale=float(su),
                                tolerance=tolerance,
                                axial=False)
    return out
