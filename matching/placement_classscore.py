"""MISSION 22 / (C) -- the class score: "same kind of thing, a different instance".

The repo's own registered next change.  A class is scored by pooling its instances' matcher
scores, so the question "is this the same kind of thing" is answered by the evidence for its
members rather than by a nearest single instance:

    primary   sum of the top TWO instance scores of that class
    arms      max of the class's instance scores, and their mean
              (fixed here, before any measurement; not tuned)

NON-CIRCULARITY, which is the whole point of the module's shape.  `class_scores` takes the
INSTANCE scores and a gallery table that maps a GALLERY item to its class.  It never receives,
and has no way to receive, the class of the QUERY: the aggregation is a function of evidence
only, so it cannot read the answer it is scored against.  `tests/test_m22_bundle.py` pins this
by rebuilding every class score after deleting the query's own class from the caller's table --
the numbers must not move.

Chance is the exact hypergeometric probability that a random top-S of ``S`` of the ``N``
gallery items contains at least one of the ``k`` other instances of the query's class.
"""
from __future__ import annotations

from math import comb
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def class_scores(instance_scores: Mapping[int, float], class_of: Mapping[int, str],
                 agg: str = "top2") -> Dict[str, float]:
    """class -> pooled score.  Reads instance scores and the GALLERY's classes only."""
    per: Dict[str, List[float]] = {}
    for j, s in instance_scores.items():
        c = class_of.get(int(j))
        if c is None:
            continue
        per.setdefault(str(c), []).append(float(s))
    out: Dict[str, float] = {}
    for c, vals in per.items():
        vals = sorted(vals, reverse=True)
        if agg == "top2":
            out[c] = float(sum(vals[:2]))
        elif agg == "max":
            out[c] = float(vals[0])
        elif agg == "mean":
            out[c] = float(sum(vals) / len(vals))
        else:
            raise KeyError(agg)
    return out


def class_top1(scores: Mapping[str, float], truth_class: str) -> int:
    """1 when the best-scoring class is the query's class.  This is where the label is read --
    once, at the reporting boundary, and nowhere in `class_scores`."""
    if not scores:
        return 0
    best = max(sorted(scores), key=lambda c: (scores[c], c))
    return 1 if best == truth_class else 0


def chance_top1(n_same_class: int, n_gallery: int) -> float:
    """P(the top-1 of a random ranking is of the query's class) = k / N.

    ``k`` counts every gallery item of the query's class, INCLUDING the query's own reference
    view -- that view is in the gallery and a random ranking can pick it, so it belongs in the
    chance.  (The earlier draft of this docstring called the argument ``n_other``; the caller
    has always passed k, and the arithmetic is the caller's.)
    """
    if n_gallery <= 0:
        return 0.0
    return float(n_same_class) / float(n_gallery)


def chance_at_s(n_same_class: int, n_gallery: int, s: int) -> float:
    """P(a random top-``s`` of ``n_gallery`` contains at least one of the ``k`` same-class ones)."""
    if n_same_class <= 0 or n_gallery <= 0:
        return 0.0
    if n_gallery - n_same_class < s:
        return 1.0
    return 1.0 - comb(n_gallery - n_same_class, s) / float(comb(n_gallery, s))
