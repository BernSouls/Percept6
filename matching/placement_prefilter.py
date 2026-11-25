"""[M20 / item 3] STAGE 1: a cheap image-pair score from per-node CONFIGURATIONS, no mapping.

Why this can exist at all: the configuration descriptor
(`placement_context.config_descriptor`) is a per-place, rotation-invariant summary of the
local frame -- sorted log neighbour distances and sorted source-frame directions.  It needs
no correspondence, so a whole image becomes a small bag of descriptors, and two images can
be compared in O(n log n + n*m) with no bijection, no context iteration and no LLR table.

WHAT THE SCORE IS.  For every query place take its best configuration similarity to ANY
reference place; the image score is the MEAN of those best similarities.  That is
deliberately the weakest possible aggregation -- it is a filter, not a decision -- but it
has the two properties a filter needs: it is deterministic (no sampling, no seed), and it
is truth-free (it reads a place's own geometry, never the transform, never the pixels of
the other image beyond its places).

WHY NOT APPEARANCE.  Mission 10's shortlist was built from appearance and reached recall@8
of 0.235 in a full gallery against 0.78 within class: appearance is exactly the quantity
that fails across instances of one class.  Stage 1 here is geometry-only, which is the
quantity that survives a change of instance, and its recall is measured per task before
anything is claimed about it.

FLAG.  `PLACEMENT_PREFILTER` (default off).  Nothing in the alignment path calls this
module; Mission 20 uses it harness-side and reports its recall next to every number, as
registered.  With the flag off `image_score` still computes (it is a pure function) -- the
flag only gates the deployed wiring that does not exist yet.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _flatten(d: Any, k: int) -> Optional[List[float]]:
    """(sorted log distances, sorted directions) -> a fixed-length feature row of 3K.

    Directions go in as (cos, sin) so the -pi/pi seam cannot split two equal angles, and a
    place with fewer than K graded neighbours repeats its last entry rather than being
    dropped: dropping them would keep only the well-connected places, a selection on
    exactly the quantity under test.
    """
    if d is None:
        return None
    ds, angs = d
    n = min(len(ds), len(angs))
    if n < 2:
        return None
    kk = int(k)
    D = [float(x) for x in ds[:n]]
    A = [float(x) for x in angs[:n]]
    D = (D + [D[-1]] * kk)[:kk]
    A = (A + [A[-1]] * kk)[:kk]
    return D + [math.cos(a) for a in A] + [math.sin(a) for a in A]


def _bag(side: Any, nodes: Sequence[int]) -> np.ndarray:
    """The image as a (n, 2K) array of configuration descriptors, rows by node id."""
    from matching import placement_context as Ctx
    k = int(getattr(Ctx, "ASSIGN_K", 8))
    rows: List[List[float]] = []
    for u in nodes:
        f = _flatten(Ctx.config_descriptor(side, int(u)), k)
        if f is not None:
            rows.append(f)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def bag_of(side: Any, nodes: Sequence[int]) -> np.ndarray:
    """The public form of `_bag`: what a caller caches per image (a few tens of KB)."""
    return _bag(side, nodes)


def score_bags(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """`image_score` on two precomputed bags, so a read does not rebuild descriptors."""
    if A.size == 0 or B.size == 0:
        return {"score": None, "n_places": int(A.shape[0])}
    sd = B.std(axis=0)
    keep = sd > 1e-9
    if not bool(keep.any()):
        return {"score": None, "n_places": int(A.shape[0])}
    mu = B[:, keep].mean(axis=0)
    Az = (A[:, keep] - mu) / sd[keep]
    Bz = (B[:, keep] - mu) / sd[keep]
    d2 = ((Az[:, None, :] - Bz[None, :, :]) ** 2).sum(axis=2)
    best = np.exp(-d2 / (2.0 * Az.shape[1]))
    return {"score": float(best.max(axis=1).mean()), "n_places": int(A.shape[0])}


def image_score(side_a: Any, side_b: Any,
                nodes_a: Optional[Sequence[int]] = None,
                nodes_b: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Mean over query places of the best configuration similarity to any reference place.

    Returns a dict so a caller cannot mistake a filter score for a probability:
    ``score`` is in [0, 1] (1 = every place finds its own configuration somewhere),
    ``n_places`` is the number of query places that contributed, and ``score`` is None
    (never inf, never NaN) when there is nothing to compare.
    """
    if nodes_a is None:
        nodes_a = sorted(int(n) for n in side_a.incident)
    if nodes_b is None:
        nodes_b = sorted(int(n) for n in side_b.incident)
    return score_bags(_bag(side_a, nodes_a), _bag(side_b, nodes_b))


def shortlist(scores: Dict[str, float], s: int) -> List[str]:
    """Top-``s`` keys by score, ties broken by key, so the list is a pure function."""
    keys = [k for k, v in scores.items() if v is not None and not math.isnan(v)]
    keys.sort(key=lambda k: (-scores[k], k))
    return keys[:int(s)]


def recall_at(ranked: Sequence[str], correct: str, s: int) -> Optional[float]:
    """1.0 / 0.0 if ``correct`` is in the top ``s``, None when it is not in the gallery."""
    if correct not in ranked:
        return None
    return 1.0 if ranked.index(correct) < int(s) else 0.0


# =========================================================================================
# [MISSION 21 / O1] STAGE 1, SECOND ATTEMPT: ONE SCALE FOR EVERY GALLERY ITEM.
#
# WHAT WAS WRONG.  `score_bags` standardized the query by EACH REFERENCE's own mean, std and
# kept dimensions (``sd = B.std(axis=0); keep = sd > 1e-9``).  That is a per-reference
# feature selection and a per-reference scale: score(A, B1) and score(A, B2) live on two
# different rulers, so a shortlist built by comparing them compares incomparable numbers.
# The defect is invisible inside one pair and decisive across a gallery, which is exactly
# where a shortlist lives.
#
# WHAT REPLACES IT.  A FIXED standardization -- mean, standard deviation and kept slots per
# slot index, fitted once on FIT and stored in ``matching/placement_prefilter_params.json``
# with its provenance -- applied identically to every image of every pair.  No reference ever
# contributes to the ruler again.
#
# THE DESCRIPTOR IS ALSO NOW ONE K.  The old bag used ``ASSIGN_K`` (6) while the matcher's
# configuration descriptor uses K = 24; the parameters file states K and the measurement that
# chose it.  There is no padding: a place keeps only the neighbours it has, the comparison is
# over the OVERLAP of the two places' slots (exactly the convention
# ``placement_context.config_similarity`` uses, ``n = min(len(a), len(b))``), and the size of
# the overlap is data rather than a fabricated last entry.  Slots a place does not have are
# masked out of the sum and out of the denominator, so a sparse place is compared on what it
# has, not on medians it was given.
#
# Determinism: pure numpy, no sampling, and the params file is read once (``load_params``).
# =========================================================================================

def descriptor_variant() -> str:
    """[M21 / O6] the descriptor shape, read at SCORING time (see `placement_context`)."""
    try:
        import parameters as P
        return str(getattr(P, "PLACEMENT_NODE_DESC_VARIANT", "sorted"))
    except Exception:
        return "sorted"


PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "placement_prefilter_params.json")
_PARAMS_CACHE: Optional[Dict[str, Any]] = None


def load_params(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The frozen standardization, or None when it has not been fitted yet."""
    global _PARAMS_CACHE
    if path is not None:
        with open(path) as fh:
            return json.load(fh)
    if _PARAMS_CACHE is None:
        if not os.path.exists(PARAMS_PATH):
            return None
        with open(PARAMS_PATH) as fh:
            _PARAMS_CACHE = json.load(fh)
    return _PARAMS_CACHE


def bag_v2(side: Any, nodes: Sequence[int], k: Optional[int] = None) -> Dict[str, Any]:
    """Raw per-place descriptors, variable length but stored at K with a mask.

    ``D`` is log nearest-neighbour distance, ``A`` the source-frame direction, ``M`` the mask
    (1 = this slot exists for this place).  Rows are in the order of ``nodes``, and the node
    id is carried so a caller can pair a place with truth without re-deriving the order.
    """
    from matching import placement_context as Ctx
    kk = int(k if k is not None else (load_params() or {}).get("K", 24))
    D = np.zeros((len(nodes), kk), dtype=np.float64)
    A = np.zeros((len(nodes), kk), dtype=np.float64)
    M = np.zeros((len(nodes), kk), dtype=np.float64)
    ids: List[int] = []
    for r, u in enumerate(nodes):
        d = Ctx.config_descriptor(side, int(u), kk)
        ids.append(int(u))
        if d is None:
            continue
        ds, angs = d
        n = min(len(ds), len(angs), kk)
        if n < 2:
            continue
        D[r, :n] = [float(x) for x in ds[:n]]
        A[r, :n] = [float(x) for x in angs[:n]]
        M[r, :n] = 1.0
    return {"D": D, "A": A, "M": M, "ids": ids}


def standardize(bag: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the FROZEN per-slot standardization.  Identical for every image, always."""
    muD = np.asarray(params["mu_D"], dtype=np.float64)
    sdD = np.asarray(params["sd_D"], dtype=np.float64)
    keepD = np.asarray(params["keep_D"], dtype=bool)
    muC = np.asarray(params["mu_cos"], dtype=np.float64)
    muS = np.asarray(params["mu_sin"], dtype=np.float64)
    sdC = np.asarray(params["sd_cos"], dtype=np.float64)
    sdS = np.asarray(params["sd_sin"], dtype=np.float64)
    D = (bag["D"] - muD) / sdD
    C = (np.cos(bag["A"]) - muC) / sdC
    S = (np.sin(bag["A"]) - muS) / sdS
    M = bag["M"] * keepD[None, :].astype(np.float64)
    return {"D": D, "C": C, "S": S, "M": M, "ids": bag["ids"]}


def _masked_ssq(X: np.ndarray, Y: np.ndarray, Mx: np.ndarray, My: np.ndarray):
    """sum_i Mx_i My_i (X_i - Y_i)^2 for every pair of rows, without materialising n*n*k."""
    sx = (Mx * X * X) @ My.T
    sy = Mx @ (My * Y * Y).T
    sxy = (Mx * X) @ (My * Y).T
    return sx + sy - 2.0 * sxy


def pair_terms(A: Dict[str, Any], B: Dict[str, Any]):
    """The sigma-FREE part of the similarity: per-overlap-slot mean squared standardized
    distance, the same for the angles, the live mask, and the overlap counts.

    Factored out so a fit computes the expensive part ONCE per pair and then evaluates every
    kernel width for free -- the standardization is frozen, so only the widths change.
    """
    Ma, Mb = A["M"], B["M"]
    if Ma.size == 0 or Mb.size == 0:
        return None
    n_ov = Ma @ Mb.T
    live = n_ov >= 2.0
    if not bool(live.any()):
        return None
    denom = np.maximum(n_ov, 1.0)
    d2 = _masked_ssq(A["D"], B["D"], Ma, Mb) / denom
    a2 = (_masked_ssq(A["C"], B["C"], Ma, Mb)
          + _masked_ssq(A["S"], B["S"], Ma, Mb)) / denom
    return ({"d2": np.where(live, d2, 0.0), "a2": np.where(live, a2, 0.0),
             "live": live}, n_ov)


def score_terms(terms: Dict[str, Any], sigma_d: float, sigma_a: float):
    """The image score from cached `pair_terms` at a given pair of kernel widths.

    The aggregation is the registered one: the mean over query places of the best reference
    place, with places that have no live partner contributing nothing.
    """
    live = terms["live"]
    sim = np.exp(-0.5 * (terms["d2"] / float(sigma_d) ** 2
                         + terms["a2"] / float(sigma_a) ** 2))
    sim = np.where(live, sim, 0.0)
    best = sim.max(axis=1)
    has = best > 0.0
    if not bool(has.any()):
        return None
    return float(best[has].mean())


def score_bags_v2(A: Dict[str, Any], B: Dict[str, Any],
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`image_score` on two STANDARDIZED bags (see `standardize`).

    Symmetric in its two arguments up to the aggregation direction: the kernel below is a
    function of the unordered pair of places, and the image score is the mean over query
    places of the best reference place.  ``asymmetry`` is reported by the caller.
    """
    p = params or load_params()
    if p is None:
        raise RuntimeError("placement_prefilter_params.json is missing: fit it on FIT first")
    Ma, Mb = A["M"], B["M"]
    if Ma.size == 0 or Mb.size == 0:
        return {"score": None, "n_places": 0, "n_overlap_mean": None}
    t = pair_terms(A, B)
    if t is None:
        return {"score": None, "n_places": int(Ma.shape[0]), "n_overlap_mean": None}
    terms, n_ov = t
    s = score_terms(terms, p["sigma_d"], p["sigma_a"])
    return {"score": s, "n_places": int(terms["live"].any(axis=1).sum()),
            "n_overlap_mean": float(n_ov[terms["live"]].mean())}


def image_score_v2(side_a: Any, side_b: Any,
                   nodes_a: Optional[Sequence[int]] = None,
                   nodes_b: Optional[Sequence[int]] = None,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole stage-1 score for one pair of images, under the frozen ruler."""
    p = params or load_params()
    if p is None:
        raise RuntimeError("placement_prefilter_params.json is missing: fit it on FIT first")
    if nodes_a is None:
        nodes_a = sorted(int(n) for n in side_a.incident)
    if nodes_b is None:
        nodes_b = sorted(int(n) for n in side_b.incident)
    A = standardize(bag_v2(side_a, nodes_a, p["K"]), p)
    B = standardize(bag_v2(side_b, nodes_b, p["K"]), p)
    return score_bags_v2(A, B, p)
