"""PLACEMENT: correspondence decided by where a node SITS, not by what it is.

THE IDEA
--------
A node is a feature.  An edge ``u -> v`` says "v sits THERE relative to u",
in **u's own frame**:

    direction   the SIGNED angle of (v - u) relative to u's CONTINUOUS
                orientation, over the full 2*pi -- never a quantized slot
    distance    |v - u| divided by u's SCALE (``node_scale`` below)
    log_scale   log2(scale_v / scale_u)
    tolerance   how much residual this edge is allowed (frozen, see below)

Mapping ``u -> u'`` is supported when ``v'`` sits where the edge says, measured
in ``u'``'s own frame.  **Never in image coordinates** -- that is the whole
point, and it is why the payload carries its own direction rather than
re-using ``EDGE_CURVE_CONTAINMENT.polar_angle`` (Part 0b measured that slot in
the IMAGE frame, which is the defect this module exists to avoid).

NO CHECKABLE EDGES = NO CREDIT AND NO PENALTY
---------------------------------------------
An edge whose neighbour is unmapped, or whose endpoints carry no measurable
frame, leaves BOTH the numerator and the denominator.  Presence is never
rewarded: an edge that cannot be checked contributes nothing rather than a
default.

IDENTITY STAYS CATEGORICAL
--------------------------
Nothing that is used for SCORING is quantized.  The node's lens type and
signature are used for identity, as they always were; the placement payload
is continuous and is compared as a continuous residual.

WHERE THE NODE FRAME COMES FROM
-------------------------------
``Orientation`` nodes carry ``theta`` in slot 0: 24 bins over 2*pi
(``parameters/_text.py:299 ORIENTATION_N_THETA = 24``, observed bins 0..23),
de-binned to the bin CENTRE.  A node with no orientation (``CurveComplex``,
``Contour``) has **no direction frame** and therefore cannot source a
placement direction; it can still be the FAR end of an edge sourced by an
oriented node.  That asymmetry is real and is reported, not smoothed over.

The scale is the node's ``Detection.scale`` when available, else the median
nearest-neighbour anchor spacing of its graph -- named ``node_scale`` and
documented at the read site.

DETERMINISM
-----------
Neighbours are visited in sorted order; the payload is a frozen dataclass with
a total-order key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: 24 bins over 2*pi == 15 degrees per bin (parameters/_text.py ORIENTATION_N_THETA)
THETA_N_THETA = 24

#: The two registered placement classes whose endpoints are ORIENTED and whose
#: geometry is frame-relative enough to carry a payload.  CURVE_CONTAINMENT is
#: excluded on purpose: its polar_angle is measured in the IMAGE frame
#: (Part 0b), so it cannot express "relative to u's own orientation".
PAYLOAD_CLASSES = ("ORIENTATION_CONTINUITY",)

#: Flags, all DARK until Part 6.
DEFAULT_FLAGS: Dict[str, Any] = {
    "PLACEMENT_PAYLOAD_ENABLED": False,
    "PLACEMENT_SCORER_ENABLED": False,
    "PLACEMENT_GATE_ENABLED": False,
    #: ONE pre-stated weight, no sweep (Part 4).
    "PLACEMENT_WEIGHT": 0.25,
    #: The frozen tolerance floor; see `freeze_tolerance`.
    "PLACEMENT_TOLERANCE": None,
    "PLACEMENT_MAX_NEIGHBOUR_CREDIT": 1.0,
}


# ---------------------------------------------------------------------------
# the payload itself
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlacementVector:
    """ONE edge's continuous relative placement, in the SOURCE node's frame.

    ``direction`` is signed over the full 2*pi, measured from the source
    node's continuous orientation: ``atan2(dy, dx) - theta_u``, wrapped to
    (-pi, pi].  ``distance`` is in source-scale units.  Neither is quantized.
    """

    direction: float          # radians in (-pi, pi], source frame
    distance: float           # |v - u| / source_scale
    log_scale: float          # log2(scale_v / scale_u); 0 == equal
    source_theta: float       # the source's continuous orientation (radians)
    source_scale: float       # the divisor used, in px
    tolerance: Optional[float] = None   # frozen per view step (Part 3)
    #: True when the source's frame is an AXIS rather than an arrow.  A
    #: structure-tensor field gives the dominant gradient direction, which is
    #: 180-degree ambiguous, so ``direction`` is then the AXIAL angle in
    #: (-pi/2, pi/2] and must be compared modulo pi.  Carrying the flag on the
    #: vector means the scorer cannot silently treat an axis as an arrow.
    axial: bool = False

    def key(self) -> Tuple[float, float, float]:
        return (self.direction, self.distance, self.log_scale)

    def reversed(self) -> "PlacementVector":
        """The SAME relation walked from the other end.

        Canonical direction is SOURCE -> TARGET with the source's own frame.
        Reversal is not the negation of ``direction`` alone: the far end has
        its own orientation, and the distance is re-normalised by the far
        end's scale.  The caller supplies the far end's frame, so reversal is
        a separate construction (``placement_vector``) rather than a flip.
        """
        raise NotImplementedError(
            "reversal needs the far end's own frame; call placement_vector() "
            "with (v, u) swapped instead of reversing the vector")


# ---------------------------------------------------------------------------
# node frames
# ---------------------------------------------------------------------------
def debin_theta(bin_value: float) -> Optional[float]:
    """Bin index -> the bin CENTRE in radians.

    ``ORIENTATION`` slot 0 is a 24-cycle over 2*pi with a ``floor`` quantiser
    (core/registries.py CIRCULAR_CYCLE_LENGTHS["ORIENTATION"] = {0: 24}).
    The de-binned centre is ``(b + 0.5) / 24 * 2*pi``, wrapped to (-pi, pi].
    ``None`` for a missing measurement.
    """
    if bin_value is None:
        return None
    try:
        b = float(bin_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(b) or b < 0.0:
        return None
    b = b % THETA_N_THETA
    a = (b + 0.5) * (2.0 * math.pi / THETA_N_THETA)
    return wrap_pi(a)


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    x = (a + math.pi) % (2.0 * math.pi)
    if x <= 0.0:
        x += 2.0 * math.pi
    return x - math.pi


def wrap_ax(x: float) -> float:
    """Fold an angle onto the AXIS interval (-pi/2, pi/2].

    Used by frames that are lines rather than arrows (see
    ``PlacementVector.axial``).  Folding here, once, is what keeps the scorer
    symmetric under a flip of an axial frame.
    """
    while x > math.pi / 2:
        x -= math.pi
    while x <= -math.pi / 2:
        x += math.pi
    return x


def node_theta(sig: Any) -> Optional[float]:
    """The node's CONTINUOUS orientation (bin centre), or None.

    Deliberately reads the bin CENTRE rather than a quantized slot value: the
    mission forbids scoring on a quantized quantity, so the de-binning happens
    once here and every downstream comparison is continuous.
    """
    q = getattr(sig, "quantized", None)
    if not q:
        return None
    return debin_theta(q[0])


def median_nn_spacing(positions: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Median nearest-neighbour distance over DISTINCT positions.

    Anchors are shared between nodes (measured in Part 1: 535 nodes on 284
    distinct positions), so a per-node nearest neighbour is 0 by construction
    and is not a scale.  Distinct positions are the right population.
    """
    uniq = sorted(set(positions))
    if len(uniq) < 2:
        return None
    P = np.asarray(uniq, dtype=np.float64)
    out: List[float] = []
    for i in range(len(P)):
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        d[i] = np.inf
        m = float(d.min())
        if math.isfinite(m):
            out.append(m)
    if not out:
        return None
    return float(np.median(out))


def node_scale(pos_px: Optional[Tuple[float, float]],
               fallback: Optional[float], det_scale: Optional[float] = None) -> Optional[float]:
    """The divisor that turns a pixel distance into SCALE UNITS.

    Named, in order of preference:

    1. ``det_scale`` -- the detection's own ``Detection.scale`` (the node's
       measured size in px), when the caller can supply it;
    2. ``fallback`` -- the graph's median nearest-neighbour anchor spacing
       (``median_nn_spacing``), the field's own length unit -- the same unit
       ``EDGE_ORIENTATION_CONTINUITY.dist`` is expressed in
       (``core/schema.py`` declares it in ``median_spacings``).

    ``None`` when neither exists: a node with no scale cannot source a
    normalised distance, and the edge is then uncheckable rather than defaulted.
    """
    if det_scale is not None and math.isfinite(float(det_scale)) and float(det_scale) > 0.0:
        return float(det_scale)
    if fallback is not None and math.isfinite(float(fallback)) and float(fallback) > 0.0:
        return float(fallback)
    return None


# ---------------------------------------------------------------------------
# building one edge's payload
# ---------------------------------------------------------------------------
def placement_vector(u_pos: Optional[Tuple[float, float]],
                     v_pos: Optional[Tuple[float, float]],
                     u_theta: Optional[float],
                     v_theta: Optional[float],
                     u_scale: Optional[float],
                     v_scale: Optional[float]) -> Optional[PlacementVector]:
    """The relative placement of ``v`` in ``u``'s own frame, or None.

    None means UNCHECKABLE (a missing position, orientation or scale), and the
    caller must then leave the edge out of BOTH numerator and denominator.

    Frame rule: the direction is the bearing of ``v`` from ``u`` MINUS ``u``'s
    own orientation.  Two nodes that rotate together therefore keep the same
    payload -- the invariance the mission's first control requires.
    """
    if u_pos is None or v_pos is None:
        return None
    if u_theta is None or u_scale is None:
        return None
    dx = float(v_pos[0]) - float(u_pos[0])
    dy = float(v_pos[1]) - float(u_pos[1])
    bearing = math.atan2(dy, dx)
    direction = wrap_pi(bearing - float(u_theta))
    dist = math.hypot(dx, dy) / float(u_scale)
    if v_scale is not None and u_scale > 0.0 and v_scale > 0.0:
        log_scale = math.log2(float(v_scale) / float(u_scale))
    else:
        log_scale = 0.0
    return PlacementVector(direction=direction, distance=dist,
                           log_scale=log_scale, source_theta=float(u_theta),
                           source_scale=float(u_scale))


def edge_payloads(g: Any, classes: Sequence[str] = PAYLOAD_CLASSES
                  ) -> Dict[Tuple[int, int, Any], PlacementVector]:
    """Every checkable placement edge's payload, keyed by the stored edge.

    Only ``classes`` edges with BOTH endpoints positioned are considered.  The
    scale is the graph's median nearest-neighbour anchor spacing -- one scale
    for the whole graph, so the payload is a property of the GEOMETRY and not
    of whichever detection happened to supply a size.
    """
    pos: Dict[int, Tuple[float, float]] = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None and len(a) >= 2:
            pos[int(n)] = (float(a[0]), float(a[1]))
    fallback = median_nn_spacing(list(pos.values()))
    out: Dict[Tuple[int, int, Any], PlacementVector] = {}
    allowed = {str(c) for c in classes}
    for s, d, et in sorted(g.subgraph.edges,
                           key=lambda e: (int(e[0]), int(e[1]),
                                          getattr(e[2], "name", str(e[2])))):
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        if name not in allowed:
            continue
        su = g.sigs.get(s)
        sv = g.sigs.get(d)
        if su is None or sv is None:
            continue
        pu, pv = pos.get(int(s)), pos.get(int(d))
        if pu is None or pv is None:
            continue
        vu = placement_vector(pu, pv, node_theta(su[1]), node_theta(sv[1]),
                              node_scale(pu, fallback),
                              node_scale(pv, fallback))
        if vu is not None:
            out[(int(s), int(d), et)] = vu
    return out


# ---------------------------------------------------------------------------
# TOGGLE TEST: moving or rotating every node together changes nothing
# ---------------------------------------------------------------------------
def payloads_under_global_transform(g: Any, dtheta: float = 0.0,
                                    dx: float = 0.0, dy: float = 0.0
                                    ) -> Dict[Tuple[int, int, Any], PlacementVector]:
    """Recompute the payloads after a GLOBAL rigid move of every node.

    Used by the Part 6 control "moving or rotating every node together leaves
    P's score unchanged".  Rotating every node by ``dtheta`` shifts every
    bearing by ``dtheta`` AND every source orientation by ``dtheta``, so the
    difference -- the payload -- must be invariant to machine precision.
    """
    pos: Dict[int, Tuple[float, float]] = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None and len(a) >= 2:
            pos[int(n)] = (float(a[0]), float(a[1]))
    if not pos:
        return {}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    c, s_ = math.cos(dtheta), math.sin(dtheta)
    moved: Dict[int, Tuple[float, float]] = {}
    for n, (x, y) in pos.items():
        rx, ry = x - cx, y - cy
        moved[n] = (rx * c - ry * s_ + cx + dx, rx * s_ + ry * c + cy + dy)
    fallback = median_nn_spacing(list(moved.values()))
    out: Dict[Tuple[int, int, Any], PlacementVector] = {}
    allowed = set(PAYLOAD_CLASSES)
    for k, vec in sorted(edge_payloads(g).items(),
                         key=lambda kv: (kv[0][0], kv[0][1],
                                         getattr(kv[0][2], "name", ""))):
        u, v, et = k
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        if name not in allowed:
            continue
        su = g.sigs.get(u)
        sv = g.sigs.get(v)
        if su is None or sv is None:
            continue
        tu, tv = node_theta(su[1]), node_theta(sv[1])
        if tu is not None:
            tu = wrap_pi(tu + dtheta)
        if tv is not None:
            tv = wrap_pi(tv + dtheta)
        pu, pv = moved.get(u), moved.get(v)
        if pu is None or pv is None:
            continue
        nv = placement_vector(pu, pv, tu, tv, node_scale(pu, fallback),
                              node_scale(pv, fallback))
        if nv is not None:
            out[k] = nv
    return out
