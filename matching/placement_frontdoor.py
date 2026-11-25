"""MISSION 22 / (A) + the appearance front door -- how a query finds its candidates.

THE FLOOR (appearance).  A fixed, unfitted appearance descriptor -- a 4x4x4 colour histogram
and a 12-bin gradient-orientation histogram, L2-normalised, cosine -- is the safety net.  It is
the M21 channel, moved here so both front doors share one implementation and one bin
convention.  It costs 0.6 ms per image and fits in one line of arithmetic.

THE GRAPH-NATIVE DOOR (place voting).  One correct place pairing already says where the whole
object sits: the two places' frames differ by one similarity transform, so the transform VOTES
for the object the pool place came from.  Retrieval is then a pooled lookup plus agreement --
no pair-by-pair matching, no per-pair loop:

    index   every gallery place is pooled once: (content key, frozen configuration row,
            frame) where the frame is (x, y) position, local scale (the place's nearest-
            neighbour distance) and axial orientation (the local structure-tensor frame).
            The CONTENT KEY is (lens, the signature's quantised bins) -- the coarse index the
            signature layer was designed around.  Measured over the FIT gallery, that key gives
            buckets with median 3 and p95 37 places; a finer key (adding configuration) drops
            36% of places into singleton buckets and would leave them unable to vote.
    query   each query place probes its content bucket, ranks the bucket by CONFIGURATION
            distance (the Mission-21 frozen ruler: per-slot mu/sd, overlap mask, no
            per-reference statistics), keeps its nearest ``n_hits`` pool places, and each of
            those proposes the transform between the two frames and votes for its object in a
            coarse histogram.
    score   the object's largest agreeing cluster, normalised by the query's place count and
            the object's place count: ``votes_max / sqrt(n_query_places * n_object_places)``.
            That normalisation is what stops a big object from winning on size alone.

Two histograms, because 13% of places have no orientation: every place votes on
(dx, dy, log2 scale); places that carry a frame ALSO vote on (dx, dy, log2 scale, dtheta).  The
object score is the best bin of either.  Both bin widths are stated in ``BINS`` below and are
part of the measurement, not tuned here.

THE COMBINED DOOR.  Rank fusion of the two score lists (reciprocal rank fusion, k = 60), which
has no weight to tune: a candidate's fused score is the sum over the two lists of 1/(60 + rank).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The transform histogram: translations in units of the query's median place spacing, scale
#: as log2, rotation in 30-degree bins.  Coarse on purpose: a vote says "the object sits about
#: here, about this big, about this turned", and agreement is what is counted.
BINS = {"xy_units_of_spacing": 1.0, "log2_scale": 0.5, "theta": math.pi / 6.0}

#: The local size ratio a pair may vote with, in octaves.  A pair whose places differ in local
#: size by more than 2x is voting with the wrong neighbourhood scale; clipping keeps its vote
#: finite instead of letting one bad pair throw the offset to infinity.
_SIGMA_CLIP = 1.0

#: Reciprocal-rank-fusion constant.  NOT tuned: 60 is the value the information-retrieval
#: literature uses and this mission forbids a tuned weight; the sensitivity is reported anyway.
RRF_K = 60


# --------------------------------------------------------------------------- the length unit
def local_scales(entry: Dict[str, Any]) -> Tuple[List[float], float]:
    """Each place's distance to its nearest DISTINCT neighbour, and the image's unit of length.

    ``Side.spacing_by_node`` is the nearest-neighbour distance including places that sit on the
    same anchor, so on these images its median is 0 (measured: 0.0 on most ETH-80 images, and
    the per-place values that survive are ~1 px).  A zero length unit makes every transform bin
    infinitely thin, so no two votes can ever agree -- that is a defect in the DOOR, not in the
    data, and it is fixed here: the distances are recomputed from the stored positions with
    co-located places excluded, and the unit is their median.  If that is degenerate too, the
    unit falls back to the occupied bounding box over sqrt(n) places, which is always positive.
    """
    rows = place_rows(entry)
    if not rows:
        return [], 1.0
    P = np.asarray([[r["x"], r["y"]] for r in rows], dtype=np.float64)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    D[D <= 1e-6] = np.inf                     # a co-located twin is not a neighbour
    nn = np.min(D, axis=1)
    nn = np.where(np.isfinite(nn), nn, np.nan)
    good = nn[np.isfinite(nn) & (nn > 0)]
    if good.size:
        unit = float(np.median(good))
    else:
        unit = 0.0
    if not np.isfinite(unit) or unit <= 0:
        span = float(np.hypot(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()))
        unit = span / max(1.0, math.sqrt(len(rows))) if span > 0 else 1.0
    filled = np.where(np.isfinite(nn) & (nn > 0), nn, unit)
    return [float(x) for x in filled], float(unit)


# --------------------------------------------------------------------------- appearance
def appearance_feature(path: str) -> np.ndarray:
    """Colour + gradient-orientation histograms, L2.  Fixed bins, nothing fitted."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    q = np.minimum((img * 4).astype(np.int64), 3)
    col = np.bincount((q[:, :, 0] * 16 + q[:, :, 1] * 4 + q[:, :, 2]).ravel(),
                      minlength=64).astype(np.float64).reshape(4, 4, 4)
    grey = img.mean(axis=2)
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    b = np.minimum((ang / np.pi * 12).astype(np.int64), 11)
    hist = np.bincount(b.ravel(), weights=mag.ravel(), minlength=12)
    v = np.concatenate([col.ravel() / max(1e-9, col.sum()), hist / max(1e-9, hist.sum())])
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def appearance_scores(query_paths: Sequence[str], gallery_paths: Sequence[str]
                      ) -> np.ndarray:
    """(n_query, n_gallery) cosine similarity.  The floor, and the safety net."""
    A = np.asarray([appearance_feature(p) for p in query_paths])
    B = np.asarray([appearance_feature(p) for p in gallery_paths])
    return A @ B.T


# --------------------------------------------------------------------------- the pool
def place_rows(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The pool rows of one gallery image, from the harness's place record.

    Every row is one payload place and carries everything the query needs:
    ``key`` (content), ``Dz``/``Mz`` (the frozen configuration row and its mask), ``x``/``y``
    (position), ``s`` (local scale) and ``theta`` (axial orientation, or None).
    """
    return entry["places"]


class PlaceIndex:
    """Every gallery place, pooled once.  Lookup is a dict probe, never a pairwise loop."""

    def __init__(self, entries: Sequence[Dict[str, Any]], obj_keys: Sequence[str],
                 keyfn: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self.obj_keys = list(obj_keys)
        self.by_key: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
        self.n_places: List[int] = []
        self.units: List[float] = []
        self._scales: List[Dict[int, float]] = []
        self._keyfn = keyfn or (lambda r: r["key"])
        for i, e in enumerate(entries):
            rows = place_rows(e)
            self.n_places.append(len(rows))
            sc, unit = local_scales(e)
            self.units.append(unit)
            self._scales.append({int(r["n"]): s for r, s in zip(rows, sc)})
            for r in rows:
                self.by_key.setdefault(self._keyfn(r), []).append((i, r))
        for k in self.by_key:
            self.by_key[k].sort(key=lambda t: (t[0], t[1]["n"]))

    # ---- the query side
    def vote(self, query_entry: Dict[str, Any], spacing: Optional[float] = None,
             n_hits: int = 3, use_rot: bool = True) -> Dict[str, Any]:
        """The per-object vote histogram for one query image.

        The translation is voted in units of the query's own length unit SCALED BY THE PAIR's
        local size ratio, so a change of apparent size does not smear the vote: the offset that
        is voted is the offset a similarity would have to undo, measured in reference-sized
        query units.  (The mission's operating range is a scale change, so this is the
        rotation/scale handling of the door, and it is part of the mechanism, not a tuned knob.)
        """
        q_scales, q_unit = local_scales(query_entry)
        unit = float(q_unit) if spacing is None else float(spacing)
        if not np.isfinite(unit) or unit <= 0:
            unit = float(q_unit)
        xy_bin = BINS["xy_units_of_spacing"] * max(1e-9, unit)
        ls_bin = BINS["log2_scale"]
        th_bin = BINS["theta"]
        buckets_no_rot: Dict[Tuple[int, int, int, int], int] = {}
        buckets_rot: Dict[Tuple[int, int, int, int, int], int] = {}
        used = 0
        hits_total = 0
        for qi, q in enumerate(place_rows(query_entry)):
            if not _live(q):
                continue
            pool = self.by_key.get(self._keyfn(q))
            if not pool:
                continue
            used += 1
            qs = float(q_scales[qi]) if qi < len(q_scales) else unit
            ranked = sorted(pool, key=lambda t: (cfg_distance(q, t[1]), t[0], t[1]["n"]))
            for (obj, r) in ranked[:n_hits]:
                if not _live(r):
                    continue
                hits_total += 1
                rs = float(self._scales[obj].get(int(r["n"]), self.units[obj]))
                rel = math.log2(max(1e-9, rs) / max(1e-9, qs))
                sigma = 2.0 ** max(-_SIGMA_CLIP, min(_SIGMA_CLIP, rel))
                dx = (r["x"] - q["x"]) / (xy_bin * sigma)
                dy = (r["y"] - q["y"]) / (xy_bin * sigma)
                ls = rel / ls_bin
                key = (obj, int(round(dx)), int(round(dy)), int(round(ls)))
                buckets_no_rot[key] = buckets_no_rot.get(key, 0) + 1
                if use_rot and q.get("theta") is not None and r.get("theta") is not None:
                    # axial: the angle lives on [0, pi), so the difference does too, and the
                    # bin that wraps past pi is the same bin as zero
                    dth = ((float(q["theta"]) - float(r["theta"])) % math.pi) / th_bin
                    n_bins = int(round(math.pi / th_bin))
                    key4 = key + (int(round(dth)) % n_bins,)
                    buckets_rot[key4] = buckets_rot.get(key4, 0) + 1
        best: Dict[int, int] = {}
        for key, c in buckets_no_rot.items():
            best[key[0]] = max(best.get(key[0], 0), c)
        for key, c in buckets_rot.items():
            best[key[0]] = max(best.get(key[0], 0), c)
        scores: Dict[int, float] = {}
        for obj, c in best.items():
            denom = math.sqrt(max(1, used) * max(1, self.n_places[obj]))
            scores[obj] = float(c) / denom
        return {"scores": scores, "places_used": used, "votes": hits_total,
                "bins": len(buckets_no_rot) + len(buckets_rot)}


def _live(row: Dict[str, Any]) -> bool:
    """A place can only vote if its configuration row has at least two live slots."""
    mz = row["Mz"]
    return int(np.count_nonzero(mz)) >= 2


def cfg_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Mean squared standardized configuration distance over the overlap, or +inf.

    The Mission-21 frozen ruler: both rows were standardized by the SAME file at build time, so
    this is a distance in one metric and no reference contributes a scale.
    """
    ma = np.asarray(a["Mz"], dtype=np.float64)
    mb = np.asarray(b["Mz"], dtype=np.float64)
    m = ma * mb
    n = float(m.sum())
    if n < 2:
        return float("inf")
    da = np.asarray(a["Dz"], dtype=np.float64)
    db = np.asarray(b["Dz"], dtype=np.float64)
    d2 = float((m * (da - db) ** 2).sum()) / n
    return d2


def voting_scores(index: PlaceIndex, query_entries: Sequence[Dict[str, Any]],
                  spacings: Optional[Sequence[float]] = None, n_hits: int = 3,
                  use_rot: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """(n_query, n_object) vote scores and the per-query diagnostics."""
    n_obj = len(index.obj_keys)
    out = np.zeros((len(query_entries), n_obj), dtype=np.float64)
    diag = []
    for i, qe in enumerate(query_entries):
        r = index.vote(qe, None if spacings is None else spacings[i],
                       n_hits=n_hits, use_rot=use_rot)
        for obj, s in r["scores"].items():
            out[i, obj] = s
        diag.append({"places_used": r["places_used"], "votes": r["votes"],
                     "bins": r["bins"], "n_objects_scored": len(r["scores"])})
    return out, diag


def count_scores(index: PlaceIndex, query_entries: Sequence[Dict[str, Any]],
                 n_hits: int = 8) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """THE COUNTED DOOR -- the same index and the same vote, without the geometric bin.

    Measured on FIT, the transform vote above cannot work on these images: for the pool places
    that DO pair with a query place on the correct object, the transform they propose has an IQR
    of 49 units in dx and 2.9 rad in dtheta (dtheta's IQR over [0, pi) is the whole range), so
    agreement is the noise and counting is the signal.  Taking the bin width to its limit turns
    "the largest agreeing cluster" into "every vote counted once": an object scores the number
    of the query's places that find a place in it, over the object's own place count.  That is
    the same pooled index, one dict probe and a sort per query place, no pairwise loop -- and it
    is the door this mission ships, because it is what works.  ``n_hits`` is the number of pool
    places one query place may count; the measured plateau is flat from 2 to 32 (report, (A)).
    """
    n_obj = len(index.obj_keys)
    out = np.zeros((len(query_entries), n_obj), dtype=np.float64)
    diag: List[Dict[str, Any]] = []
    for i, qe in enumerate(query_entries):
        used = 0
        cnt: Dict[int, int] = {}
        for q in place_rows(qe):
            if not _live(q):
                continue
            pool = index.by_key.get(index._keyfn(q))
            if not pool:
                continue
            used += 1
            ranked = sorted(pool, key=lambda t: (cfg_distance(q, t[1]), t[0], t[1]["n"]))
            for (obj, r) in ranked[:n_hits]:
                if _live(r):
                    cnt[obj] = cnt.get(obj, 0) + 1
        for obj, c in cnt.items():
            out[i, obj] = float(c) / math.sqrt(max(1, used) * max(1, index.n_places[obj]))
        diag.append({"places_used": used, "objects_scored": len(cnt),
                     "votes": int(sum(cnt.values()))})
    return out, diag


# --------------------------------------------------------------------------- fusion
def rank_fuse(a: np.ndarray, b: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion of two (n_query, n_object) score matrices.

    No weight: each candidate earns 1/(k + rank) from each list, ranks counted from 1.  Ties
    are broken by object index so the result is deterministic.
    """
    out = np.zeros_like(a, dtype=np.float64)
    for M, sign in ((a, 1.0), (b, 1.0)):
        for i in range(M.shape[0]):
            order = sorted(range(M.shape[1]), key=lambda j: (-sign * M[i, j], j))
            for r, j in enumerate(order):
                out[i, j] += 1.0 / (k + r + 1)
    return out


# --------------------------------------------------------------------------- shortlist
def shortlist(scores: np.ndarray, i: int, s: int) -> List[int]:
    """Top-``s`` object indices for query ``i``, deterministic tie-break by index."""
    return sorted(range(scores.shape[1]), key=lambda j: (-scores[i, j], j))[:s]
