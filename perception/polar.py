"""perception/polar.py -- extracted from sensory_floor.py (Move 3a).

MOVE ONLY: whole top-level statements, verbatim. The re-export
shim in sensory_floor.py keeps `from sensory_floor import X`
working until the importer repoint retires it.
"""
from __future__ import annotations

import math

def _axial_wrap_sf(x: float) -> float:
    return (x + math.pi / 2.0) % math.pi - math.pi / 2.0

def signed_menger(p1, p2, p3, c) -> float:
    """Menger curvature 4A/(abc); + when the arc bends TOWARD reference point c.
    Traversal-direction independent. NOTE: c = object centroid is a shortcut for
    'object interior'; pass a figure-side reference (see inside_points) for
    non-convex objects."""
    a = math.hypot(p2[0]-p1[0], p2[1]-p1[1]); b = math.hypot(p3[0]-p2[0], p3[1]-p2[1])
    d = math.hypot(p3[0]-p1[0], p3[1]-p1[1])
    if a < 1e-9 or b < 1e-9 or d < 1e-9:
        return 0.0
    turn = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
    side = (p3[0]-p1[0])*(c[1]-p1[1]) - (p3[1]-p1[1])*(c[0]-p1[0])
    s = ((turn > 0) - (turn < 0)) * ((side > 0) - (side < 0))
    return 2.0 * abs(turn) / (a * b * d) * s

def _arc_split(poly, target):
    """Cut at exact host-arc distances, independent of vertex sampling.

    Fragment boundaries interpolate within segments, retaining interior
    vertices. The short terminal remainder follows the historical half-length
    admission rule. Each accepted piece has at least three points.
    """
    if len(poly) < 2 or target <= 0:
        return []
    frags, cur, acc = [], [poly[0]], 0.0
    for end in poly[1:]:
        start = cur[-1]
        remaining = math.hypot(end[0] - start[0], end[1] - start[1])
        while remaining > 1e-12 and acc + remaining >= target:
            fraction = (target - acc) / remaining
            cut = (start[0] + fraction * (end[0] - start[0]),
                   start[1] + fraction * (end[1] - start[1]))
            cur.append(cut)
            if len(cur) == 2:
                cur.insert(1, ((cur[0][0] + cut[0]) / 2, (cur[0][1] + cut[1]) / 2))
            frags.append(cur)
            cur, acc, start = [cut], 0.0, cut
            remaining = math.hypot(end[0] - start[0], end[1] - start[1])
        if remaining > 1e-12:
            cur.append(end)
            acc += remaining
    if acc >= 0.5 * target:
        if len(cur) == 2:
            cur.insert(1, ((cur[0][0] + cur[-1][0]) / 2,
                           (cur[0][1] + cur[-1][1]) / 2))
        frags.append(cur)
    return frags


def _arc_mid(fr):
    lengths = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(fr, fr[1:])]
    total = sum(lengths)
    half, travelled = total / 2.0, 0.0
    for a, b, length in zip(fr, fr[1:], lengths):
        if travelled + length >= half and length > 0:
            t = (half - travelled) / length
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), total
        travelled += length
    return fr[-1], total


def polar_boundary_fragments(contours, frag_frac=0.35, kappa_cap=8.0, min_points=12,
                             center=None, inside_points=None):
    """Pasupathy & Connor object-centred fragments from association-field contours.
    center: optional (x,y) overriding the vertex centroid C_obj.
    inside_points: optional {contour_idx: (x,y)} figure-side reference used ONLY
    for the curvature sign. Bins: kappa 0..8 (zero-centred, 4 = straight), theta 0..23
    (circular), r 0..7, tangent-vs-radial 0..7."""
    polys = []
    for ci, c in enumerate(contours):
        pl = c.get("polyline")
        if pl and len(pl) >= min_points:
            points = [(float(x), float(y)) for x, y in pl]
            if c.get("is_closed"):
                if points[0] == points[-1]:
                    points.pop()
                # A translation-equivariant, deterministic seam and direction.
                seam = min(range(len(points)), key=lambda i: points[i])
                points = points[seam:] + points[:seam]
                if points[-1] < points[1]:
                    points = [points[0]] + list(reversed(points[1:]))
                points.append(points[0])
            polys.append((ci, points))
    pts = [p for _, pl in polys for p in pl]
    if len(pts) < 2 * min_points:
        return [], {}
    cx, cy = center if center is not None else (sum(p[0] for p in pts)/len(pts),
                                                sum(p[1] for p in pts)/len(pts))
    dist = sorted(math.hypot(p[0]-cx, p[1]-cy) for p in pts)
    R = dist[int(0.9 * (len(dist) - 1))]
    if R < 1e-6:
        return [], {}
    target, S, out = max(4.0, frag_frac * R), math.asinh(kappa_cap), []
    for ci, pl in polys:
        ref = (inside_points or {}).get(ci, (cx, cy))
        # Fragment scale belongs to its host, not all the other contours in
        # the image. Adding a distant object must not repartition this host.
        hx = sum(p[0] for p in pl) / len(pl)
        hy = sum(p[1] for p in pl) / len(pl)
        radii = sorted(math.hypot(p[0] - hx, p[1] - hy) for p in pl)
        host_r = radii[int(0.9 * (len(radii) - 1))]
        host_target = max(4.0, frag_frac * host_r)
        arc_start = 0.0
        for k, fr in enumerate(_arc_split(pl, host_target)):
            (mid, arc), p0, p1 = _arc_mid(fr), fr[0], fr[-1]
            kappa_n = signed_menger(p0, mid, p1, ref) * R
            kc = max(-kappa_cap, min(kappa_cap, kappa_n))
            kb = float(max(0, min(8, int(round(math.asinh(kc) / S * 4.0)) + 4)))
            th = math.atan2(mid[1]-cy, mid[0]-cx) % (2.0 * math.pi)
            tb = float(int(round(th / (2.0 * math.pi) * 24)) % 24)
            rr = math.hypot(mid[0]-cx, mid[1]-cy) / R
            rb = float(min(7, max(0, int(min(rr, 1.0) * 8.0))))
            tvr = abs(_axial_wrap_sf(math.atan2(p1[1]-p0[1], p1[0]-p0[0]) - th))
            vb = float(min(7, int(tvr / (math.pi / 2.0) * 8.0)))
            # [STAGE 6 / R1] the chord direction itself (axial).  `tvr` below
            # keeps only its ABSOLUTE difference from the radial angle, so
            # theta and tvr leave two candidate tangents (theta +/- tvr); the
            # attachment relation needs the one that was measured.
            chord = math.atan2(p1[1]-p0[1], p1[0]-p0[0]) % math.pi
            out.append({"contour": ci, "frag": k, "mid": mid, "arc": arc, "kappa_n": kappa_n,
                        "tangent": chord,
                        "theta": th, "r": rr, "bins": (kb, tb, rb, vb),
                        # [GRADED POPULATION] The CONTINUOUS values of the
                        # four graded slots, so the emitter can build the
                        # population from the pre-quantisation measurement
                        # instead of echoing the bin: the asinh-compressed
                        # signed curvature (the same compression the level
                        # uses, so the bump and the bin agree), the polar
                        # angle, the normalised radius and the
                        # tangent-vs-radial angle.
                        "kappa_c": (math.asinh(kc) / S) if S > 0 else 0.0,
                        "theta_c": th,
                        "r_c": rr,
                        "tvr_c": tvr, "host_arc_start": arc_start,
                        "host_arc_end": arc_start + arc})
            arc_start += arc
    return out, {"center": (cx, cy), "R": R, "target_len": target}



# [ITEM 4] Moved verbatim from perception/contours.py (thin-shim split; nothing deleted).
def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference a − b, in (−π, π]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d

def _axial_angle_diff(a: float, b: float) -> float:
    """Absolute shortest distance between two axial angles (mod pi)."""
    d = abs((a % math.pi) - (b % math.pi))
    return min(d, math.pi - d)
