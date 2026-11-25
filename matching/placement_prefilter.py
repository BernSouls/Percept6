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

import math
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
