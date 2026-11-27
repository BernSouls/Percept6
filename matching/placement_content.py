"""The place's CONTENT beyond the keypoint class (Mission 28, item 6).

Today a place's content is `(theta, coherence, scale, kind, dim)` -- the keypoint signature.  M28's
gauge (item 4) measured what that buys for cross-view identity: same-class G = 0.496 at chance 0.5.
The audit (item 3) says why: the place carries orientation and scale and nothing else, while the
pixels at the place's own scale carry colour, orientation population and texture.

Item 5 built four candidates and fixed the rule before any number: a candidate proceeds only if its
same-class G beats the signature's by >= +0.10.  C1 (two concentric rings of CIELAB at the place's
own scale) measured G_same 0.745, the signature 0.496 -> +0.249; C4 (C1+C2+C3) 0.616 -> +0.120;
C2 orientation-population 0.563 (+0.067) and C3 texture 0.532 (+0.036) failed.  C1 is the higher
of the two that passed and is the one this module implements for the pipeline.

WHAT IS FIXED HERE, BEFORE THE DOOR NUMBERS:

    ring radii     0.5 s and 1.5 s, s the place's own scale (scale-normalised by construction)
    sampling       12 angular samples per ring, bilinear, mean per channel
    channels       CIELAB L / 100, a / 128, b / 128  -> 6 values
    rotation       none is needed: a ring mean is rotation-invariant by construction
    similarity     cosine of the 6-vector (unit-normalised)
    quantisation   for the counted door: L into 6 bins over [0, 100], a and b into 8 bins over
                   [-128, 128) each -> a 6-int key

Nothing here reads a label.  ``features`` takes the image and the record's own frames.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The bundle value `content.kind` may take.  ``keypoint`` is the shipped default: a place carries
#: the keypoint's quantised signature only (what every record before M28 carries).
KINDS: Tuple[str, ...] = ("keypoint", "colour_rings")
DEFAULT_KIND = "keypoint"

RINGS: Tuple[float, ...] = (0.5, 1.5)
RING_SAMPLES = 12
N_VALUES = 6
SIMILARITY = "cosine"

#: counted-door bins per channel, in the order L, a, b (then twice, one block per ring)
QUANT: Tuple[Tuple[float, float, int], ...] = (
    (0.0, 100.0, 6), (-128.0, 128.0, 8), (-128.0, 128.0, 8),
) * len(RINGS)

_EPS = 1e-9


def digest() -> str:
    """sha256 of this module's source: the content's own pin, recorded in every manifest."""
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def kind(kind_: Optional[str] = None, bundle: Any = None) -> str:
    """The effective kind: the argument, else the bundle's value, else the default.  Refuses
    anything not in ``KINDS`` -- there is no fallback."""
    k = kind_ or (getattr(bundle, "content_kind", None) if bundle is not None else None) \
        or DEFAULT_KIND
    if k not in KINDS:
        raise ValueError("content.kind %r is not one of %s" % (k, list(KINDS)))
    return str(k)


def lab(image_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIELAB (D65), float64.  Fixed math, no fitting."""
    rgb = image_u8.astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a (H, W[, C]) array at float coordinates, clamped at the border."""
    H, W = a.shape[0], a.shape[1]
    x = np.clip(x, 0.0, W - 1.001)
    y = np.clip(y, 0.0, H - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    w00, w01, w10, w11 = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
    if a.ndim == 3:
        w00, w01, w10, w11 = (w[..., None] for w in (w00, w01, w10, w11))
    return (a[y0, x0] * w00 + a[y0, x1] * w01 + a[y1, x0] * w10 + a[y1, x1] * w11)


def features(image_u8: np.ndarray, places: Sequence[Dict[str, Any]]) -> np.ndarray:
    """The (n, 6) content of every place of one image, rows in ``places`` order."""
    lab_img = lab(image_u8)
    n = len(places)
    out = np.zeros((n, N_VALUES), dtype=np.float64)
    ang = np.linspace(0.0, 2 * np.pi, RING_SAMPLES, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    for i, p in enumerate(places):
        s = max(float(p.get("s") or 0.0), 0.75)
        for r_i, r in enumerate(RINGS):
            v = bilinear(lab_img, float(p["x"]) + r * s * ca,
                         float(p["y"]) + r * s * sa).mean(axis=0)
            out[i, 3 * r_i:3 * r_i + 3] = (v[0] / 100.0, v[1] / 128.0, v[2] / 128.0)
    return unit(out)


def unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), _EPS)


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between (n, d) and (m, d) unit vectors -> (n, m)."""
    A = unit(np.atleast_2d(np.asarray(a, dtype=np.float64)))
    B = unit(np.atleast_2d(np.asarray(b, dtype=np.float64)))
    return A @ B.T


def quantise(v: Sequence[float]) -> Tuple[int, ...]:
    """The counted door's key for a content vector: one bin per channel, from ``QUANT``.

    The stored vector is already divided by 100 / 128 / 128, so the channels are scaled back to
    physical units before binning."""
    vals = (v[0] * 100.0, v[1] * 128.0, v[2] * 128.0, v[3] * 100.0, v[4] * 128.0, v[5] * 128.0)
    out: List[int] = []
    for x, (lo, hi, nb) in zip(vals, QUANT):
        b = int(math.floor((float(x) - lo) / (hi - lo) * nb))
        out.append(min(max(b, 0), nb - 1))
    return tuple(out)


def attach(record: Dict[str, Any], image_u8: np.ndarray, k: Optional[str] = None) -> Dict[str, Any]:
    """Put the content on a record's places, in place.  ``keypoint`` leaves the record untouched."""
    kk = kind(k)
    if kk == "keypoint":
        return record
    f = features(image_u8, record["places"])
    for i, p in enumerate(record["places"]):
        p["content"] = [float(x) for x in f[i]]
    record["content_kind"] = kk
    return record
