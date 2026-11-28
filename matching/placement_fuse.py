"""M30 A1: the door's fusion, with an optional third channel for place colour.

The shipped door is ``rrf(counted, appearance)`` -- two channels, equal weight, ``k = 60``
(``matching/placement_frontdoor.rank_fuse``).  Mission 28 measured a graded colour channel on the
same records (FIT +0.125 at ``-035-045`` and +0.050 at ``-045-180``) and could not ship it only
because M28's rule was a conjunction that missed one view by one object of forty.  Mission 30 fuses
it as a THIRD channel, at a DISCOUNT fixed before any number:

    w_counted = 1.0        (incumbent, unchanged)
    w_appearance = 1.0     (incumbent, unchanged)
    w_colour = 0.5         (BELOW both incumbents, by construction)

and the bar replaces M28's conjunction: the AVERAGE recall@8 gain across the two FIT views must be
>= +0.050, and DEV+FRESH must not fall below the shipped door by more than 0.025.

WHY A DISCOUNT AND NOT ANOTHER EQUAL VOTE.  Rank fusion throws away magnitudes; a third equal
channel would quietly take a third of the vote from the two channels the shipped door was measured
with, and colour has never been read in the shipped path.  A half-vote says: colour may reorder the
head, it may not outvote the configuration.

THE COLOUR CHANNEL'S OWN SCORE, fixed here: for one query object and one gallery object, the
**maximum cosine similarity** between the query's places and that object's places of the SAME LENS
(the M28 ``graded_content`` channel, which is the variant that measured +0.125).  Colour is only
ever compared where the content exists, and a place without content contributes nothing.

`fuse` with the colour channel absent returns the shipped arithmetic BIT-IDENTICALLY: with two
channels the sum per (query, object) is the same two terms added in the same order as
``rank_fuse``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The shipped door's own channel weights.  Colour's default is 0.0 -- OFF.
#: [M33] `shape` is the fourth channel slot.  Its default weight is 0.0 -- OFF -- so a door that
#: does not ask for it computes exactly the arithmetic it computed before (identity tested).
DEFAULT_WEIGHTS: Dict[str, float] = {"counted": 1.0, "appearance": 1.0, "colour": 0.0,
                                    "shape": 0.0,
                                    #: [M34] the fragment channel's slot.  Default 0.0 -- OFF -- so a
                                    #: door that does not ask for it computes exactly the arithmetic
                                    #: it computed before (identity tested in tests/test_m34_door.py).
                                    "fragment": 0.0}

#: A1's stated discount, used when a caller asks for the colour channel.
COLOUR_WEIGHT = 0.5
#: [M33] the same discount for the curvature channel, fixed before any number: a channel that has
#: never been read in the shipped path may reorder the head, it may not outvote the configuration.
SHAPE_WEIGHT = 0.5
#: [M34] the same discount for the fragment channel: measured at this weight, shipped at this weight
#: (`Topology/m34_door.py`; +0.0625 mean FIT recall@8, object-clustered CI [0.0125, 0.1125]).
FRAGMENT_WEIGHT = 0.5
RRF_K = 60
LENSES: Tuple[str, ...] = ("Blob", "BoundaryArc", "CornerPoint")


def _ranks(M: np.ndarray) -> List[List[int]]:
    """Rank order per row, best first, ties broken by object index (the shipped rule)."""
    return [sorted(range(M.shape[1]), key=lambda j: (-float(M[i, j]), j))
            for i in range(M.shape[0])]


def rrf_weighted(mats: Sequence[np.ndarray], weights: Sequence[float],
                 k: int = RRF_K) -> np.ndarray:
    """``sum_c w_c / (k + rank_c)`` over the channels, ranks counted from 1."""
    base = np.asarray(mats[0], dtype=np.float64)
    out = np.zeros_like(base)
    for M, w in zip(mats, weights):
        if float(w) == 0.0:
            continue
        A = np.asarray(M, dtype=np.float64)
        for i, order in enumerate(_ranks(A)):
            for r, j in enumerate(order):
                out[i, j] += float(w) / (k + r + 1)
    return out


def colour_channel(gallery: Sequence[Dict[str, Any]], query: Sequence[Dict[str, Any]],
                   n_obj: Optional[int] = None) -> np.ndarray:
    """Per (query object, gallery object) the max same-lens cosine of the place content.

    ``gallery`` and ``query`` are the door's own entries: ``{"places": [{"lens", "content"}]}``.
    An object with no content anywhere scores 0 for every query, which is what the channel says
    when it has nothing to say.
    """
    g = len(gallery)
    q = len(query)
    out = np.zeros((q, g), dtype=np.float64)
    for lens in LENSES:
        G: List[np.ndarray] = []
        for entry in gallery:
            rows = [p["content"] for p in entry.get("places", ())
                    if p.get("lens") == lens and p.get("content") is not None]
            G.append(np.asarray(rows, dtype=np.float64).reshape(-1, 6))
        if all(x.shape[0] == 0 for x in G):
            continue
        for qi, entry in enumerate(query):
            rows = [p["content"] for p in entry.get("places", ())
                    if p.get("lens") == lens and p.get("content") is not None]
            if not rows:
                continue
            Q = np.asarray(rows, dtype=np.float64)
            Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
            for gi, mat in enumerate(G):
                if mat.shape[0] == 0:
                    continue
                Gb = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
                v = float(np.max(Q @ Gb.T))
                if v > out[qi, gi]:
                    out[qi, gi] = v
    return out


def fuse(counted: np.ndarray, appearance: np.ndarray,
         colour: Optional[np.ndarray] = None,
         weights: Optional[Dict[str, float]] = None, k: int = RRF_K,
         shape: Optional[np.ndarray] = None,
         fragment: Optional[np.ndarray] = None) -> np.ndarray:
    """The door's fused score.  No colour, no shape, no fragment, no weights -> shipped arithmetic."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({str(a): float(b) for a, b in weights.items()})
    mats = [np.asarray(counted, dtype=np.float64), np.asarray(appearance, dtype=np.float64)]
    ws = [w["counted"], w["appearance"]]
    if colour is not None and float(w.get("colour", 0.0)) != 0.0:
        mats.append(np.asarray(colour, dtype=np.float64))
        ws.append(float(w["colour"]))
    if shape is not None and float(w.get("shape", 0.0)) != 0.0:
        mats.append(np.asarray(shape, dtype=np.float64))
        ws.append(float(w["shape"]))
    if fragment is not None and float(w.get("fragment", 0.0)) != 0.0:
        mats.append(np.asarray(fragment, dtype=np.float64))
        ws.append(float(w["fragment"]))
    return rrf_weighted(mats, ws, k=k)


def effective_weights(bundle: Any = None) -> Dict[str, float]:
    """The weights in force: the bundle's `fusion` block if it names them, else the defaults.

    A negative or non-finite weight is refused -- a fusion is a sum of positive votes.
    """
    w = dict(DEFAULT_WEIGHTS)
    block = getattr(bundle, "fusion", None) if bundle is not None else None
    if isinstance(block, dict):
        for key in ("counted", "appearance", "colour", "shape", "fragment"):
            if key in block:
                w[key] = float(block[key])
    for key, v in w.items():
        if not np.isfinite(v) or v < 0.0:
            raise ValueError("fusion weight %r = %r is not a non-negative vote" % (key, v))
    return w
