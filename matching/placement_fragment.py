"""M34: compose places into FRAGMENTS, described in the fragment's own registered frame.

THIS IS THE ONE LEVEL OF COMPOSITION THE TREE DID NOT HAVE.  Every descriptor before M34 has either
a single place as its support (`placement_content.features`, `placement_shape.features` -- both
per-place) or the whole image (`placement_frontdoor.appearance_feature`).  `config` (M19) pools one
place's star of neighbours, which is a neighbourhood, not a part.  A fragment is a SET of places.

THE GROUPING, AND WHAT THE MISSION ASKED FOR THAT DOES NOT EXIST.  M34's item 1 named four edge
classes -- EDGE_BOUNDARY_ATTACHMENT, EDGE_COCIRCULAR, EDGE_BORDER_OWNERSHIP, EDGE_BOUNDARY_SUCCESSOR
-- as the grouping.  Measured on a real graph (apple/1 at -035-045, 1,506 nodes, 2,686 edges): **no
edge of any class has a place node as an endpoint**; 0 of 2,686.  The front end emits two disjoint
node universes -- places (Blob / BoundaryArc / CornerPoint, 474 nodes) and region machinery
(Contour, ContourFragment, RegionShape, BorderOwnership, Surface) -- and the four named edge classes
live entirely inside the second.  So the mission's grouping rule is not available on this graph, and
pretending otherwise would be the exact defect the audit names: building on a lens nobody wired.

WHAT THIS MODULE DOES INSTEAD, registered here before any number is read: it WIRES the region
machinery to the places by the only relation the front end actually provides -- `nearest part
anchor`.  The graph's `ContourFragment` (fine) and `Contour` (coarse) nodes each carry an anchor
position; a place joins the fragment of its nearest anchor within a stated radius.  That is a
composition whose support is a set of places, and whose grouping is decided by machinery the shipped
path reads nowhere.  A purely geometric control (`proximity`: single-linkage clustering on place
positions at a stated radius) is also registered, so the difference between "the region machinery
decides" and "geometry decides" is measurable rather than assumed.

THE DESCRIPTOR, FIXED HERE BEFORE ANY NUMBER.  Everything is relative to the fragment's OWN frame:
centre = mean member position; orientation theta = principal axis of the member positions, mod pi;
scale = median member place scale.  21 numbers, all dimensionless:

    log_size      log1p(n places) / log1p(512)
    aspect        sqrt(lambda_minor / lambda_major) of the member positions, in [0, 1]
    spread        RMS member distance from the centre, in units of (scale * sqrt(n))
    d_mean        mean member distance from the centre / scale
    d_sd          sd of the same
    align1        mean |cos(phi)| over members, phi = member angle relative to theta
    align2        mean |cos(2 phi)|
    cross2        mean |sin(2 phi)|
    sharp_x_m, sharp_y_m, curv_x_m, curv_y_m       mean over members
    sharp_x_s, sharp_y_s, curv_x_s, curv_y_s       sd over members
    lab0..lab5    mean over members of the six CIELAB ring values

Similarity is the SAME form as the colour and shape channels (`exp(-0.5 * sum_j w_j (d_j/sigma_j)^2)`
with `sigma_j` the pool's own column spread), so a fragment is compared the way this tree compares
everything else.  An UNREGISTERED class returns 0.0, not a small number, and a test pins it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASS = "PLACE_FRAGMENT"

FIELDS: Tuple[str, ...] = (
    "log_size", "aspect", "spread", "d_mean", "d_sd", "align1", "align2", "cross2",
    "sharp_x_m", "sharp_y_m", "curv_x_m", "curv_y_m",
    "sharp_x_s", "sharp_y_s", "curv_x_s", "curv_y_s",
    "lab0", "lab1", "lab2", "lab3", "lab4", "lab5",
)
#: physical spans (measured on 6 objects x 2 views before the descriptor was written down; the
#: kernel standardises by the POOL's spread, so these are documentation + the registry's ranges)
RANGES: Tuple[float, ...] = (1.0, 1.0, 1.0, 24.0, 12.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 6.0, 6.0, 1.0, 1.0, 4.0, 4.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
WEIGHTS: Tuple[float, ...] = (1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
BINS: Tuple[int, ...] = (8,) * len(FIELDS)
SIGMAS: Tuple[float, ...] = (1.0,) * len(FIELDS)
CYCLE: Dict[int, int] = {}                       # every axis is already rotation-invariant summary

#: the grouping rules, registered before any number
GROUPINGS: Tuple[str, ...] = ("contour_fragment", "contour", "proximity")
#: the radius (in pixels) inside which a place may join an anchor; 256 x 256 images, mean place
#: spacing ~11 px, and a fragment is meant to be a PART, not a neighbourhood
ANCHOR_RADIUS = 48.0
PROXIMITY_RADIUS = 16.0

REFUSALS: Dict[str, str] = {}
_INSTALLED: set = set()


def _rows(places: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(p["x"]), float(p["y"]), float(p.get("s") or 1.0)] for p in places],
                      dtype=np.float64).reshape(-1, 3)


def group_by_anchor(places: Sequence[Dict[str, Any]], anchors: Sequence[Tuple[float, float]],
                    radius: float = ANCHOR_RADIUS) -> List[List[int]]:
    """Each place joins the nearest anchor's fragment; a place farther than ``radius`` joins none."""
    if not places or not anchors:
        return []
    P = _rows(places)[:, :2]
    A = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    d = np.hypot(P[:, None, 0] - A[None, :, 0], P[:, None, 1] - A[None, :, 1])
    nearest = np.argmin(d, axis=1)
    near = d[np.arange(P.shape[0]), nearest]
    groups: Dict[int, List[int]] = {}
    for i in range(P.shape[0]):
        if float(near[i]) <= float(radius):
            groups.setdefault(int(nearest[i]), []).append(int(i))
    return [sorted(v) for v in groups.values()]


def group_by_proximity(places: Sequence[Dict[str, Any]],
                       radius: float = PROXIMITY_RADIUS) -> List[List[int]]:
    """Single-linkage clustering of the places at ``radius`` -- the purely geometric control."""
    if not places:
        return []
    P = _rows(places)[:, :2]
    n = P.shape[0]
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in np.nonzero(d[i] <= float(radius))[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values()]


def describe(places: Sequence[Dict[str, Any]], members: Sequence[int],
             fields: Sequence[str] = FIELDS) -> np.ndarray:
    """The 21-number descriptor of one fragment, in the fragment's own registered frame."""
    idx = list(members)
    P = _rows([places[i] for i in idx])
    c = P[:, :2].mean(axis=0)
    rel = P[:, :2] - c[None, :]
    scale = max(float(np.median(P[:, 2])), 1e-6)
    n = max(len(idx), 1)
    # principal axis of the member positions (mod pi); a 2-point fragment has no aspect
    cov = np.cov(rel.T) if len(idx) > 1 else np.zeros((2, 2))
    w, V = np.linalg.eigh(cov)
    major = float(max(w[-1], 1e-12))
    minor = float(max(w[0], 0.0)) if len(idx) > 1 else 0.0
    axis = V[:, -1] if np.all(np.isfinite(V)) else np.asarray([1.0, 0.0])
    theta = math.atan2(float(axis[1]), float(axis[0]))
    r = np.hypot(rel[:, 0], rel[:, 1])
    phi = np.arctan2(rel[:, 1], rel[:, 0]) - theta
    # pooled curvature and colour, all already on the record
    def _col(name: str) -> np.ndarray:
        return np.asarray([float(places[i].get(name) or 0.0) for i in idx], dtype=np.float64)
    cur = {k: _col(k) for k in ("sharp_x", "sharp_y", "curv_x", "curv_y")}
    lab = np.asarray([[float(x) for x in (places[i].get("content") or [0.0] * 6)][:6]
                      for i in idx], dtype=np.float64).reshape(-1, 6)
    vals = np.asarray([
        math.log1p(n) / math.log1p(512.0),
        math.sqrt(minor / major) if major > 0 else 0.0,
        float(np.sqrt(np.mean(r ** 2)) / (scale * math.sqrt(n))),
        float(r.mean() / scale),
        float(r.std() / scale) if n > 1 else 0.0,
        float(np.mean(np.abs(np.cos(phi)))),
        float(np.mean(np.abs(np.cos(2.0 * phi)))),
        float(np.mean(np.abs(np.sin(2.0 * phi)))),
        float(cur["sharp_x"].mean()), float(cur["sharp_y"].mean()),
        float(cur["curv_x"].mean()), float(cur["curv_y"].mean()),
        float(cur["sharp_x"].std()), float(cur["sharp_y"].std()),
        float(cur["curv_x"].std()), float(cur["curv_y"].std()),
        *[float(lab[:, j].mean()) if lab.size else 0.0 for j in range(6)],
    ], dtype=np.float64)
    if len(vals) != len(fields):
        raise ValueError("fragment descriptor arity %d != %d" % (len(vals), len(fields)))
    return vals


def full_descriptor(places: Sequence[Dict[str, Any]], members: Sequence[int]) -> np.ndarray:
    """`describe` on exactly `FIELDS` -- the shipped arity, and the only one `similarity` reads."""
    return describe(places, members, FIELDS)


# --------------------------------------------------------------------------- the door's channel
def channel(gallery: Sequence[Sequence[Dict[str, Any]]],
            query: Sequence[Dict[str, Any]],
            gallery_anchors: Optional[Sequence[Any]] = None,
            query_anchors: Optional[Any] = None,
            grouping: str = "contour_fragment",
            sigma_pool: Optional[np.ndarray] = None) -> np.ndarray:
    """(n_query, n_gallery) fragment scores -- the channel `placement_fuse.fuse(fragment=...)` takes.

    ``gallery`` is the door's own gallery: one entry per (object, view) cell, each ``{"places": ...}``
    (a bare list of place rows is accepted too).  ``query`` is ONE cell's list of place rows.
    ``gallery_anchors``/``query_anchors`` are the `(n, 2)` pixel anchors of that cell's grouping
    (`matching.placement_anchors`); a cell with no anchors contributes nothing, it is not invented.

    The score is the same symmetric mean-of-best the FIT screen registered in M34: each of the
    query's parts is matched to the gallery part that explains it best, the means are taken in both
    directions and averaged, so parts the gallery cannot explain cost.  ``sigma_pool`` is the pool of
    fragment descriptors the kernel standardises by; passing None builds it from this gallery, which
    is what the shipped door does (the gallery IS the pool).
    """
    n_q, n_g = 1, len(gallery)                   # `query` is ONE cell's places -> one row
    out = np.zeros((n_q, n_g), dtype=np.float64)
    if not query or not n_g:
        return out
    q_groups = ([] if query_anchors is None
                else group_by_anchor(query, list(np.asarray(query_anchors, dtype=np.float64))))
    q_desc = np.asarray([full_descriptor(query, m) for m in q_groups],
                        dtype=np.float64).reshape(-1, len(FIELDS))
    g_desc: List[np.ndarray] = []
    for j in range(n_g):
        cell = gallery[j]["places"] if isinstance(gallery[j], dict) else gallery[j]
        anc = None if gallery_anchors is None else gallery_anchors[j]
        groups = ([] if anc is None or len(np.asarray(anc)) == 0
                  else group_by_anchor(cell, list(np.asarray(anc, dtype=np.float64))))
        g_desc.append(np.asarray([full_descriptor(cell, m) for m in groups],
                                 dtype=np.float64).reshape(-1, len(FIELDS)))
    if q_desc.shape[0] == 0:
        return out
    if sigma_pool is None:
        stack = [d for d in g_desc if d.shape[0]] + [q_desc]
        sigma_pool = np.vstack(stack) if stack else np.zeros((1, len(FIELDS)))
    sig = scales(sigma_pool)
    for j, G in enumerate(g_desc):
        if G.shape[0] == 0:
            continue
        block = similarity(q_desc, G, sig)                              # (query parts, gallery parts)
        out[0, j] = 0.5 * float(block.max(axis=1).mean()) \
            + 0.5 * float(block.max(axis=0).mean())
    return out


# --------------------------------------------------------------------------- registration
def _registries() -> Any:
    import core.registries as R
    return R


def register(strict: bool = True) -> Dict[str, Any]:
    """Install `PLACE_FRAGMENT` into the live registries.  Idempotent; `unregister` restores."""
    R = _registries()
    from core.population import PopAxis
    if CLASS in getattr(R, "CONSTITUTIONAL_RANGES"):
        return {"class": CLASS, "already": True}
    if not (len(RANGES) == len(WEIGHTS) == len(BINS) == len(SIGMAS) == len(FIELDS)):
        REFUSALS["register:arity"] = "ranges/weights/bins/sigmas disagree with FIELDS"
        if strict:
            raise ValueError("placement_fragment: arity mismatch")
        return {}
    for table, name, value in (
            ("CONSTITUTIONAL_RANGES", CLASS, tuple(float(x) for x in RANGES)),
            ("SLOT_WEIGHT_PRIORS", CLASS, tuple(float(x) for x in WEIGHTS)),
            ("CIRCULAR_CYCLE_LENGTHS", CLASS, dict(CYCLE)),
            ("POPULATION_AXES", CLASS, tuple(
                PopAxis(name=f, n_bins=int(b), circular=(i in CYCLE),
                        sigma_bins=float(s), weight=float(w))
                for i, (f, b, s, w) in enumerate(zip(FIELDS, BINS, SIGMAS, WEIGHTS))))):
        table_obj = getattr(R, table)
        table_obj[CLASS] = value
        _INSTALLED.add(table)
    return {"class": CLASS, "fields": list(FIELDS), "ranges": list(RANGES),
            "weights": list(WEIGHTS), "axes": len(FIELDS)}


def unregister() -> None:
    R = _registries()
    for table in list(_INSTALLED):
        getattr(R, table).pop(CLASS, None)
    _INSTALLED.clear()


def registered() -> Dict[str, Any]:
    R = _registries()
    return {"ranges": R.CONSTITUTIONAL_RANGES.get(CLASS),
            "weights": R.SLOT_WEIGHT_PRIORS.get(CLASS),
            "cycle": R.CIRCULAR_CYCLE_LENGTHS.get(CLASS),
            "axes": [a.name for a in (R.POPULATION_AXES.get(CLASS) or ())] or None}


def missing_fields() -> List[str]:
    r = registered()
    out: List[str] = []
    if not r["ranges"]:
        return list(FIELDS)
    if not r["weights"]:
        out.append("weights")
    if r["cycle"] is None:
        # an EMPTY dict is correct for a class with no circular axis; a MISSING entry is not
        out.append("cycle")
    if not r["axes"] or list(r["axes"]) != list(FIELDS):
        out.append("axes")
    return out


# --------------------------------------------------------------------------- similarity
def scales(pool: np.ndarray) -> np.ndarray:
    """Per-column spread of the POOL -- the sigma every distance is divided by."""
    P = np.atleast_2d(np.asarray(pool, dtype=np.float64))
    return np.maximum(P.std(axis=0), 1e-9)


def kernel_block(A: np.ndarray, B: np.ndarray, sig: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The tree's one similarity form: ``exp(-0.5 * sum_j w_j (d_j / sigma_j)^2)``."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    w = np.asarray(WEIGHTS if weights is None else weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    sg = np.maximum(np.abs(np.asarray(sig, dtype=np.float64)), 1e-9)
    d2 = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for j in range(A.shape[1]):
        if w[j] == 0.0:
            continue
        d = A[:, j][:, None] - B[:, j][None, :]
        d2 += w[j] * (d / sg[j]) ** 2
    return np.exp(-0.5 * d2)


def similarity(A: np.ndarray, B: np.ndarray, sig: Sequence[float]) -> np.ndarray:
    """Registered classes only: an UNREGISTERED `PLACE_FRAGMENT` returns 0.0, never a small number."""
    if CLASS not in getattr(_registries(), "CONSTITUTIONAL_RANGES"):
        REFUSALS["unregistered"] = "PLACE_FRAGMENT is not in CONSTITUTIONAL_RANGES"
        return np.zeros((np.atleast_2d(A).shape[0], np.atleast_2d(B).shape[0]))
    return kernel_block(A, B, sig)


try:                                            # pragma: no branch
    register(strict=False)
except Exception as _exc:                       # pragma: no cover - defensive
    REFUSALS["import"] = repr(_exc)
