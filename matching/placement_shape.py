"""M30 B: SHAPE as graded columns on the place record -- never as new key slots.

THE MECHANISM M29 MISSED.  M29 called the place key's utilisation ~1e-8 by comparing Blob's 172
signatures against 2^34.9, which is the MEAN nominal over all 67 classes.  Blob has five slots:
`lenses/keypoints.py` quantises (theta 12 bins, coherence 4, scale 4 sigmas, kind, dim) and kind and
dim are fixed by the lens, so the real grid is **12 x 4 x 4 = 192 cells** and Blob uses 192 of them.
BoundaryArc 182, CornerPoint 185 -- **every place lens saturates its grid**.  23,220 blobs in 192
cells is 121 places per cell: the bottleneck is RESOLUTION, and that is why two cups collide.

THE FIX, and why it is not new key slots.  The keypoint detector already computes `resp`, `var_x`
and `var_y` and throws them away (`Keypoint` carries them; `signature_of` keeps only the five
bins).  Putting them into `quantized` would change every node's hash, move every key and force a
1,824 s re-derivation.  So they ride on the record as GRADED COLUMNS, exactly as colour did in M28:
keys stay byte-identical, and the enrichment path (36.9 s) attaches the new fields.

FIVE FIELDS, fixed before any number:

    resp        the detector's response at the place, divided by the image's 99.5th percentile
    var_x       per-axis position variance of the sub-pixel parabola fit (1e-3 .. 1)
    var_y       idem on the other axis
    eigratio    lambda_min / lambda_max of the structure tensor at radius 1.5 s (corner-ness)
    rel_angle   the structure tensor's principal orientation minus the place's OWN theta, mod pi
                (rotation-normalised by the place's own axis, so two views of one corner agree)

`resp`/`var_x`/`var_y` come from the detector's own fit; `eigratio`/`rel_angle` are computed here
from the image at the place's own scale.  Nothing reads a label.

REGISTRATION (the M16 lesson: an unregistered class returns 0.0, not a small number).  `register()`
installs the class `PLACE_SHAPE` into the live registries -- constitutional ranges, slot weights, a
circular cycle on the angle slot, and population axes -- and `tests/test_m30_shape.py` pins that
every one of the five fields is registered before and after a record is built.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASS = "PLACE_SHAPE"

#: The five graded fields, in fixed order.  The names are the record's own field names.
FIELDS: Tuple[str, ...] = ("resp", "var_x", "var_y", "eigratio", "rel_angle",
                           # [M32] the curvature the clip destroyed.  `sharp_*` is |den|, the
                           # second difference of the response at the peak; `curv_*` is |den|/|c0|,
                           # the same measurement made tolerant to the response's own size.  They
                           # are GRADED COLUMNS: the keypoint key does not read them, so every key
                           # stays byte-identical and the enrichment path applies.
                           "sharp_x", "sharp_y", "curv_x", "curv_y")

#: Say what the similarity reads, in the tree's own vocabulary.
#: ranges: the axis span in the field's own unit; weights: >1 where shape lives (angle,
#: corner-ness); cycle: the angle slot is circular on 12 bins (mod pi, the tree's convention).
#: physical spans, measured (M32: 99th percentiles over 6 objects x 2 views for the new four)
RANGES: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 6.0, 0.12, 0.13, 1.4, 1.5)
WEIGHTS: Tuple[float, ...] = (2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0)
CYCLE: Dict[int, int] = {4: 12}
BINS: Tuple[int, ...] = (8, 6, 6, 8, 12, 8, 8, 8, 8)
SIGMAS: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

#: Structure-tensor window, in units of the place's own scale.
TENSOR_RADIUS = 1.5

_INSTALLED: Dict[str, set] = {}
_PRIOR: Dict[str, Dict[str, Any]] = {}
REFUSALS: Dict[str, str] = {}


def digest() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# --------------------------------------------------------------------------- registration
def _registries():
    import core.registries as R
    return R


def register(strict: bool = True) -> Dict[str, Any]:
    """Install `PLACE_SHAPE` into the live registries.  Idempotent; `unregister` restores."""
    R = _registries()
    from core.population import PopAxis
    if CLASS in getattr(R, "CONSTITUTIONAL_RANGES"):
        return {"class": CLASS, "already": True}
    if len(RANGES) != len(FIELDS) or len(WEIGHTS) != len(FIELDS) or len(BINS) != len(FIELDS):
        REFUSALS["register:arity"] = "ranges/weights/bins disagree with FIELDS"
        if strict:
            raise ValueError("placement_shape: %s, %s, %s vs %d fields"
                             % (RANGES, WEIGHTS, BINS, len(FIELDS)))
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
        _PRIOR.setdefault(table, {}).setdefault(CLASS, table_obj.get(CLASS))
        table_obj[CLASS] = value
        _INSTALLED.setdefault(table, set()).add(CLASS)
    return {"class": CLASS, "ranges": list(RANGES), "weights": list(WEIGHTS),
            "cycle": dict(CYCLE), "bins": list(BINS), "fields": list(FIELDS)}


def unregister() -> None:
    R = _registries()
    for table, classes in list(_INSTALLED.items()):
        obj = getattr(R, table)
        for cls in sorted(classes):
            was = _PRIOR.get(table, {}).get(cls)
            if was is None:
                obj.pop(cls, None)
            else:
                obj[cls] = was
        _INSTALLED[table] = set()


def registered() -> Dict[str, Any]:
    """What the similarity would find if it read a PLACE_SHAPE signature right now."""
    R = _registries()
    return {"ranges": R.CONSTITUTIONAL_RANGES.get(CLASS),
            "weights": R.SLOT_WEIGHT_PRIORS.get(CLASS),
            "cycle": R.CIRCULAR_CYCLE_LENGTHS.get(CLASS),
            "axes": [a.name for a in (R.POPULATION_AXES.get(CLASS) or ())] or None,
            "fields": list(FIELDS)}


def missing_fields() -> List[str]:
    """Every field whose registration is absent or mis-sized (empty == fully registered)."""
    r = registered()
    out: List[str] = []
    if not r["ranges"]:
        return list(FIELDS)
    if not r["weights"]:
        out.append("weights")
    if not r["cycle"]:
        out.append("cycle")
    if not r["axes"] or list(r["axes"]) != list(FIELDS):
        out.append("axes")
    return out


# --------------------------------------------------------------------------- the features
def _gray(image: Any) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if a.size and float(a.max()) > 1.5:
        a = a / 255.0
    return a


def _tensor_maps(gray: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The structure tensor's three smoothed products at ONE scale."""
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(gray)
    s = max(float(sigma), 0.6)
    return (gaussian_filter(gx * gx, s, mode="nearest"),
            gaussian_filter(gy * gy, s, mode="nearest"),
            gaussian_filter(gx * gy, s, mode="nearest"))


def _sample(a: np.ndarray, x: float, y: float) -> float:
    H, W = a.shape
    xi = int(round(min(max(x, 0), W - 1)))
    yi = int(round(min(max(y, 0), H - 1)))
    return float(a[yi, xi])


def _finite(v: Any) -> float:
    """A graded value as a finite float; a missing or non-finite one reads 0.0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def features(image: Any, places: Sequence[Dict[str, Any]],
             detections: Optional[Sequence[Any]] = None) -> np.ndarray:
    """The (n, len(FIELDS)) graded shape of every place, rows in ``places`` order.

    `detections` are the keypoint layer's own `Keypoint` objects for this image (whatever the
    record was built from); a place with no matching detection keeps `resp` = var = 0 and is
    still given the tensor fields, which need only the image and the frame.
    """
    gray = _gray(image)
    n = len(places)
    out = np.zeros((n, len(FIELDS)), dtype=np.float64)
    # the image's own response scale, from the detections this image actually produced
    resp_ref = 1.0
    if detections:
        vals = sorted(float(getattr(d, "resp", 0.0)) for d in detections)
        idx = min(len(vals) - 1, int(0.995 * (len(vals) - 1)))
        resp_ref = max(1e-9, vals[idx] if vals else 1.0)
    # one tensor map per distinct place scale (the detector's sigmas), built lazily
    maps: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, p in enumerate(places):
        s = max(float(p.get("s") or 1.0), 0.6)
        key = round(s, 3)
        if key not in maps:
            maps[key] = _tensor_maps(gray, TENSOR_RADIUS * s)
        Jxx, Jyy, Jxy = maps[key]
        x, y = float(p["x"]), float(p["y"])
        a = _sample(Jxx, x, y)
        b = _sample(Jyy, x, y)
        c = _sample(Jxy, x, y)
        tr = a + b
        det = a * b - c * c
        disc = math.sqrt(max(0.0, tr * tr / 4.0 - det))
        l1 = tr / 2.0 + disc
        l2 = tr / 2.0 - disc
        eig = float(l2 / l1) if l1 > 1e-12 else 0.0
        ang = 0.5 * math.atan2(2.0 * c, a - b)
        theta = 0.0 if p.get("theta") is None else float(p["theta"])
        rel = (ang - theta) % math.pi
        out[i, 0] = min(2.0, max(0.0, float(p.get("resp", 0.0)) / resp_ref))
        out[i, 1] = float(p.get("var_x") or 0.0)
        out[i, 2] = float(p.get("var_y") or 0.0)
        out[i, 3] = min(1.0, max(0.0, eig))
        out[i, 4] = rel
        # [M32] the curvature the clip threw away, as carried by the detection that made the place
        for j, name in enumerate(FIELDS[5:], start=5):
            out[i, j] = _finite(p.get(name))
    return out


def attach(record: Dict[str, Any], image: Any, detections: Optional[Sequence[Any]] = None
           ) -> Dict[str, Any]:
    """Put the five graded fields on every place, in place.  Keys are untouched."""
    f = features(image, record["places"], detections)
    for i, p in enumerate(record["places"]):
        for j, name in enumerate(FIELDS):
            p[name] = float(f[i, j])
    record["shape_kind"] = "graded"
    return record


# --------------------------------------------------------------------------- similarity
def graded_matrix(A: np.ndarray, B: np.ndarray,
                  weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The shape similarity the gauge reads: a graded cosine over the five axes.

    Each axis is standardised to [0, 1] by its own physical span, centred, and compared as a
    population code; the angle axis uses the circular form cos(2 d) (mod pi).  The weighted mean
    uses the SAME weights the registered kernel uses, so the vector form and the shipped kernel
    read one constitution.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    w = np.asarray(WEIGHTS if weights is None else weights, dtype=np.float64)
    # [M32] a caller may hand in a SUBSET of the columns (the four unclipped ones, say); the
    # constitution's weights still apply, restricted to the columns present.
    width = int(A.shape[1])
    w = w[:width]
    w = w / max(w.sum(), 1e-12)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for j, name in enumerate(FIELDS[:width]):
        a = A[:, j]
        b = B[:, j]
        if j in CYCLE:
            # circular: cos of twice the angular difference, on [0, pi)
            d = np.abs(a[:, None] - b[None, :])
            v = np.cos(2.0 * d)
        else:
            span = 1.0
            a01 = np.clip(a / span, 0.0, 1.0)
            b01 = np.clip(b / span, 0.0, 1.0)
            v = 1.0 - np.abs(a01[:, None] - b01[None, :])
        out += w[j] * v
    return out


def scales(pool: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Per-column spread of the POOL, the sigma every distance is divided by (M32)."""
    P = np.atleast_2d(np.asarray(pool, dtype=np.float64))
    s = P.std(axis=0)
    return np.maximum(s, floor)


def kernel_matrix(A: np.ndarray, B: np.ndarray, sig: Sequence[float],
                  weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The SAME distance form as the colour channel (M32 item 4b), not a cosine.

    ``d^2 = sum_j w_j * ((a_j - b_j) / sigma_j)^2`` and ``s = exp(-d^2 / 2)``, with ``sigma_j`` the
    pooled column spread.  The shipped graded form (`graded_matrix`) divides by an arbitrary
    physical span on a one-sided range, so every pair of places sits within a thousandth of 1.0 --
    M32 item 2 measured a truth/impostor gap of 0.000062 on exactly that form.  Standardising by
    the pool's own spread sets the scale to the data's, which is the fix for that mechanism.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    width = int(A.shape[1])
    w = np.asarray(WEIGHTS if weights is None else weights, dtype=np.float64)[:width]
    w = w / max(w.sum(), 1e-12)
    sg = np.maximum(np.abs(np.asarray(sig, dtype=np.float64))[:width], 1e-9)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for j, name in enumerate(FIELDS[:width]):
        if w[j] == 0.0:
            continue
        d = np.abs(A[:, j][:, None] - B[:, j][None, :])
        if j in CYCLE:
            # a circular axis of period L: compared ON THE CIRCLE, as `2 - 2cos(dphi)` scaled to
            # the same units as the linear axes, so that `12` and `0` are one place.  `embed`
            # computes exactly this value with two columns, hence the agreement to 1e-12.
            L = float(CYCLE[j])
            dc = np.minimum(d, np.abs(L - d))
            d = (L / (2.0 * math.pi)) * 2.0 * np.sin(math.pi * dc / L)
        out += w[j] * (d / sg[j]) ** 2
    return np.exp(-0.5 * out)


def embed(A: np.ndarray, sig: Sequence[float],
          weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The kernel's own embedding: ``s(a, b) = exp(-0.5 * (|x_a|^2 + |x_b|^2 - 2 x_a . x_b))``.

    Same numbers as `kernel_matrix`, computed as one Gram matrix instead of a (rows_a x rows_b x
    fields) difference -- which is what makes the 4,284-place gauge and the 80 x 80 colour cell
    affordable.  Linear axes contribute ``sqrt(w)/sigma`` times the value; the circular axis
    contributes its two components at ``1 / (2 sigma)`` because ``2 - 2cos(2d) = 4 sin^2(d)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    width = int(A.shape[1])
    w = np.asarray(WEIGHTS if weights is None else weights, dtype=np.float64)[:width]
    w = w / max(w.sum(), 1e-12)
    sg = np.maximum(np.abs(np.asarray(sig, dtype=np.float64))[:width], 1e-9)
    cols: List[np.ndarray] = []
    for j, name in enumerate(FIELDS[:width]):
        if w[j] == 0.0:
            continue
        if j in CYCLE:
            # period L: the smallest way round, mapped to a unit circle; `2 - 2cos(dphi)`
            # equals `4 sin^2(dphi / 2)`, which is `(2 pi d / L)^2` near zero -- hence the
            # `L / (2 pi)` factor that makes the embedded distance agree with `kernel_matrix`
            L = float(CYCLE[j])
            k = (L / (2.0 * math.pi)) * (float(np.sqrt(w[j])) / sg[j])
            phi = 2.0 * math.pi * A[:, j] / L
            cols.append(k * np.cos(phi))
            cols.append(k * np.sin(phi))
        else:
            cols.append((float(np.sqrt(w[j])) / sg[j]) * A[:, j])
        del name
    return np.stack(cols, axis=1) if cols else np.zeros((A.shape[0], 0))


def kernel_block(A: np.ndarray, B: np.ndarray, sig: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """`kernel_matrix` through `embed`: one Gram matrix, the same values to 1e-12."""
    X = embed(A, sig, weights)
    Y = embed(B, sig, weights)
    d2 = ((X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1)[None, :] - 2.0 * (X @ Y.T))
    return np.exp(-0.5 * np.maximum(d2, 0.0))


#: [M33 item 2] The channel: the four curvature columns, EXCLUDED from everything else.  The five
#: detection-strength columns are left out because M32 measured them DILUTING the four (all nine =
#: 0.558 same-class against the four alone at 0.596).
CURVATURE_FIELDS: Tuple[str, ...] = ("sharp_x", "sharp_y", "curv_x", "curv_y")


def curvature_matrix(gallery: Sequence[Dict[str, Any]], query: Sequence[Dict[str, Any]],
                     how: str = "object_mean",
                     weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """(n_query, n_gallery) curvature scores, OBJECT-LEVEL, from the place records alone.

    For one query object and one gallery object the score is the MEAN over the query's places of
    the best same-lens match in the gallery object, read through the kernel form with the pool's
    own column spread.  That is the aggregation that read G_same 0.671 in M32's gauge (against
    0.596 at place level); `max` is kept for comparison and reports the place-level form.

    ``gallery``/``query`` are sequences of ``{"places": [{"lens", "sharp_x", ...}]}`` in object
    order -- the same shape `placement_fuse.colour_channel` takes.
    """
    if how not in ("object_mean", "max"):
        raise ValueError("curvature aggregation %r is not object_mean or max" % (how,))
    idx = [FIELDS.index(f) for f in CURVATURE_FIELDS]
    n_g, n_q = len(gallery), len(query)
    out = np.zeros((n_q, n_g), dtype=np.float64)
    pool = np.asarray([[float(p.get(f) or 0.0) for f in CURVATURE_FIELDS]
                       for entry in gallery for p in entry.get("places", ())], dtype=np.float64)
    if pool.size == 0:
        return out
    sig = scales(pool)
    for lens in ("Blob", "BoundaryArc", "CornerPoint"):
        G: List[np.ndarray] = []
        for entry in gallery:
            rows = [[float(p.get(f) or 0.0) for f in CURVATURE_FIELDS]
                    for p in entry.get("places", ()) if p.get("lens") == lens]
            G.append(np.asarray(rows, dtype=np.float64).reshape(-1, len(CURVATURE_FIELDS)))
        if all(x.shape[0] == 0 for x in G):
            continue
        # gallery places stacked in object order, one reduceat bound per object
        stacked = np.vstack([g for g in G if g.shape[0]])
        owner = np.concatenate([np.full(g.shape[0], j, dtype=np.int64)
                                for j, g in enumerate(G) if g.shape[0]])
        bounds = np.searchsorted(owner, np.arange(n_g + 1))
        for qi, entry in enumerate(query):
            rows = [[float(p.get(f) or 0.0) for f in CURVATURE_FIELDS]
                    for p in entry.get("places", ()) if p.get("lens") == lens]
            if not rows:
                continue
            S = kernel_block(np.asarray(rows, dtype=np.float64), stacked, sig, weights)
            per_obj = np.maximum.reduceat(S, bounds[:-1], axis=1)     # (n_q_places, n_g)
            if how == "max":
                out[qi] = np.maximum(out[qi], per_obj.max(axis=0))
            else:
                out[qi] = np.maximum(out[qi], per_obj.mean(axis=0))
    del idx
    return out


# Register on import.  A class that is not in the registries returns 0.0 -- not a small number --
# so every reader of these five fields must have them registered before it compares anything.  The
# call is idempotent and never raises (a refusal is counted in REFUSALS and the fields simply read
# as unregistered, which `missing_fields()` and the M30 test make loud).
try:                                        # pragma: no branch
    register(strict=False)
except Exception as _exc:                   # pragma: no cover - defensive
    REFUSALS["import"] = repr(_exc)
