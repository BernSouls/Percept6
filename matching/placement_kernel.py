"""The registry's graded signature kernel, evaluated as a block (Mission 26).

WHY THIS FILE EXISTS.  Every place in the tree that compares two signatures one at a time calls
``core.types.TypedSignature.similarity``, which is correct and far too slow for a door: it builds
two Python objects per pair, dispatches through four layers (population, graded features, the
class' own slot rules, the kernel) and costs ~10 us a pair, i.e. minutes for one query against a
pool of 24,000 places.  On top of that, a keypoint signature whose population payload was dropped
(the place record keeps the quantised tuple only) trips
``core.types.record_population_payload_miss`` on every pair, which refuses in strict mode.

So the kernel is re-expressed here, without changing it: the SAME constitution
(``CONSTITUTIONAL_RANGES``), the SAME slot weights (``get_slot_weights``), the SAME circular cycle
lengths, the SAME absent-slot renormalisation and the SAME per-class slot overrides
(``core.slots.class_slot_distance``) -- evaluated with numpy over the whole ``(nq, np)`` block.

    sigma_i      = ranges[i] / 3   (or SIGMA_FALLBACK_FOR_RANGE0 when ranges[i] <= 0)
    d_i          = |a_i - b_i|, wrapped on cyclic slots (d = min(d, n - d)), then any override
    S            = exp(-sum_i (w_i / W) (d_i / sigma_i)^2)              no absent slots
    S            = exp(-sum_i w_i (d_i / sigma_i)^2 / sum_i w_i)        with absent slots
    absent pairs = 0.0 when every slot is absent

A class with NO registered ranges has no kernel: the reference returns 1.0 for an equal tuple and
0.0 otherwise, so this module does the same and COUNTS it (``FALLBACKS``), never silently.

WHAT IT IS NOT.  It does not do the population branch of ``_similarity_gaussian`` (the place
record does not carry populations -- that is Mission 26's item 1 census), and it does not do
``feature_similarity``.  Both are asserted by the parity test in ``tests/test_m26_kernel.py``,
which compares this block against ``TypedSignature.similarity`` on the quantised branch, with the
population payload absent, under the shipped recipe's flags.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: Values a comparison may carry that mean "not measured" (``core.types.ABSENT``).
ABSENT = -1000000000.0

#: Classes with no registered ranges, and how often each was asked for a block.  Reported, never
#: hidden: a class that silently degrades to equality is exactly the defect this mission audits.
FALLBACKS: Dict[str, int] = {}


def _spec(class_: str, context: Any = None) -> Dict[str, Any]:
    """Everything the kernel needs for one class, read from the live registries."""
    from core import registries as R
    from core import slots as S
    ranges = R.get_similarity_ranges(class_)
    n = len(ranges) if ranges else 0
    weights = R.get_slot_weights(class_, n) if ranges else None
    has_rule = S._CLASS_SLOT_RULES.get(class_) is not None
    return {"class": class_, "ranges": ranges, "weights": weights,
            "cycle": dict(R.CIRCULAR_CYCLE_LENGTHS.get(class_) or {}),
            "overrides": tuple(range(n)) if has_rule else ()}


def _sigma(ranges: Sequence[float], i: int) -> Tuple[float, bool]:
    import parameters as P
    if float(ranges[i]) > 0.0:
        return float(ranges[i]) / 3.0, True
    return float(getattr(P, "SIGMA_FALLBACK_FOR_RANGE0", 1e-6)), False


def _shim(class_: str, q: np.ndarray) -> Any:
    """A stand-in signature for ``class_slot_distance`` (which reads only these fields)."""
    from core.types import TypedSignature
    return TypedSignature(kind="CANONICAL", class_=class_,
                          quantized=tuple(float(x) for x in q), population=())


def equality_block(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """1.0 where the tuples are identical, else 0.0 -- the reference's no-ranges branch."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    eq = (A[:, None, :] == B[None, :, :]).all(axis=2)
    return eq.astype(np.float64)


def similarity_block(class_: str, A: Sequence[Sequence[float]], B: Sequence[Sequence[float]],
                     context: Any = None) -> np.ndarray:
    """The (nq, np) graded similarity block between two sets of quantised tuples of ONE class."""
    A = np.asarray(A, dtype=np.float64).reshape(-1, len(A[0]) if len(A) else 0)
    B = np.asarray(B, dtype=np.float64).reshape(-1, len(B[0]) if len(B) else 0)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    if A.shape[1] != B.shape[1]:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    spec = _spec(class_, context)
    ranges = spec["ranges"]
    if ranges is None or len(ranges) != A.shape[1]:
        FALLBACKS[class_] = FALLBACKS.get(class_, 0) + 1
        return equality_block(A, B)
    weights = spec["weights"]
    W_sum = float(sum(weights)) if weights else float(A.shape[1])
    if W_sum <= 0.0:
        W_sum = float(A.shape[1]) or 1.0
    cycle = spec["cycle"]
    overrides = spec["overrides"]

    # the absent mask, only materialised when some value is ABSENT
    absA = (A == ABSENT)
    absB = (B == ABSENT)
    any_absent = bool(absA.any() or absB.any())

    total = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    W_used = np.zeros_like(total) if any_absent else None
    for i in range(A.shape[1]):
        d = np.abs(A[:, None, i] - B[None, :, i])
        if i in cycle:
            n = float(cycle[i])
            d = np.minimum(d, n - d)
        sigma, _ok = _sigma(ranges, i)
        w = float(weights[i]) if (weights and i < len(weights)) else 1.0
        term = (d / sigma) ** 2
        if overrides and i in overrides:
            from core.slots import class_slot_distance
            for a in range(A.shape[0]):
                sa = _shim(class_, A[a])
                for b in range(B.shape[0]):
                    if absA[a, i] or absB[b, i]:
                        continue
                    o = class_slot_distance(class_, i, float(A[a, i]), float(B[b, i]),
                                            sa, _shim(class_, B[b]))
                    if o is not None:
                        term[a, b] = (float(o) / sigma) ** 2
        if not any_absent:
            total += (w / W_sum) * term
        else:
            live = ~(absA[:, None, i] | absB[None, :, i])
            total += w * term * live
            W_used += w * live
    if any_absent:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp(-np.where(W_used > 0.0, total / np.where(W_used > 0.0, W_used, 1.0),
                                   np.inf))
        return np.where(W_used > 0.0, out, 0.0)
    return np.exp(-total)


def similarity_pairs(class_: str, A: Sequence[Sequence[float]], B: Sequence[Sequence[float]],
                     context: Any = None) -> np.ndarray:
    """Row-wise similarity between paired tuples (``A[j]`` against ``B[j]``)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.diagonal(similarity_block(class_, A, B, context=context)).copy()


def report() -> Dict[str, Any]:
    """What the kernel is reading right now, and which classes fell back to equality."""
    from core import registries as R
    out: Dict[str, Any] = {"fallbacks": dict(FALLBACKS), "classes": {}}
    for cls in sorted(set(list(getattr(R, "CONSTITUTIONAL_RANGES", {}).keys()))):
        ranges = R.get_similarity_ranges(cls)
        if ranges is None:
            continue
        w = R.get_slot_weights(cls, len(ranges))
        out["classes"][cls] = {"ranges": list(ranges), "sigma": [ranges[i] / 3.0 for i in range(len(ranges))],
                               "weights": list(w) if w else None,
                               "cycle": dict(R.CIRCULAR_CYCLE_LENGTHS.get(cls) or {})}
    return out
