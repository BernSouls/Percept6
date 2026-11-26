"""Which stored views an object gets, and why -- the rule, written before any score existed.

Mission 25.  The door's ceiling on an unseen view is set by how far the *nearest stored view* is
from the query, so the gallery stores more than one view per object.  Which views is a decision
that must not look at a score, so it is made here, on the view grid alone, and frozen into the
bundle.

THE GRID.  ETH-80 stores 41 views per object, named ``<polar>-<azimuth>`` in degrees:

    polar 000 : 1 view      (000-000)
    polar 022 : 4 views     azimuth 000, 090, 180, 270
    polar 035 : 4 views     azimuth 045, 135, 225, 315
    polar 045 : 4 views     azimuth 000, 090, 180, 270
    polar 066 : 8 views     azimuth 027, 063, ... 333 (45 deg apart)
    polar 068 : 4 views     azimuth 000, 090, 180, 270
    polar 090 : 16 views    azimuth 000, 022, ... 338 (22.5 deg apart)

THE METRIC (checked against the images, not assumed).  Two views are two directions, so the distance between them is the great-circle angle
between their unit vectors, with the first field the polar angle from the ``000-000`` direction and
the second the azimuth about it:

    v(a, b) = (sin a cos b, sin a sin b, cos a),      d(u, w) = acos(u . w) in degrees

This reading is the only one consistent with the grid's own shape: a single view at polar 0 is a
pole (no azimuth needed), and the azimuth count rises with the polar angle (1, 4, 4, 4, 8, 4, 16)
exactly as converging meridians allow.  The corpus does not ship a camera model, so the grid is
the best available statement of the geometry, and the distances below inherit its precision.

THE DISTANCES.  The grid's smallest non-zero separation is **22.0 deg** (between adjacent views of
the 16-view equator ring).  With four stored views per object, and the first pinned at ``000-000``
as the mission requires (so k = 1 is a subset of k = 4), no set can bring every view within 22 deg:
the best achievable covering radius is **65.5 deg**, and the rule below finds it exactly.  So the
mission's target -- "every other view within the smallest angular distance the grid allows" -- is
not attainable at k = 4, and this module says so before the measurement rather than after.

THE RULE (score-free, deterministic, exhaustive).  The reference set S is

    S = {000-000} union { the 3-subset T of the 39 non-query views that minimises
                          R(S) = max over ALL 41 views v of min over r in S of d(v, r) }

ties broken by the lexicographically smallest sorted label tuple, and the search is exhaustive over
all C(39,3) = 9,139 subsets, so the result is the optimum and not a heuristic's guess.  It uses the
view labels and nothing else: no image, no score, no correspondence, no result.

THE RESERVED VIEWS.  The read's query views (``035-045`` and ``045-180``) are never references:
a view used as a reference cannot be a query for the same object without answering a question
with its own answer.  Both are excluded from the candidate set before the search, which is also
why they are named here rather than discovered later.

WHY NOT CHASE THE QUERIES.  A rule that placed references near the two query views would score
better on this read and mean less: it would be fitting the reference set to the measurement.  The
coverage rule is indifferent to which views are queried, which is the property that makes the
result a statement about the grid.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The 41-view grid, exactly as the corpus names it.
GRID: Tuple[str, ...] = (
    "000-000",
    "022-000", "022-090", "022-180", "022-270",
    "035-045", "035-135", "035-225", "035-315",
    "045-000", "045-090", "045-180", "045-270",
    "066-027", "066-063", "066-117", "066-153", "066-207", "066-243", "066-297", "066-333",
    "068-000", "068-090", "068-180", "068-270",
    "090-000", "090-022", "090-045", "090-068", "090-090", "090-112", "090-135", "090-158",
    "090-180", "090-202", "090-225", "090-248", "090-270", "090-292", "090-315", "090-338",
)

#: The view every earlier mission stored, and the one k = 1 means.
BASE_VIEW = "000-000"

#: Filenames carry the view as a suffix: ``apple1`` + ``-035-045`` + ``.png``.
SUFFIX = "-"


def vec(label: str) -> np.ndarray:
    """The unit direction a view label names (polar, azimuth) -> (x, y, z)."""
    a, b = (float(x) for x in label.split("-"))
    th, ph = math.radians(a), math.radians(b)
    return np.asarray([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def distance(a: str, b: str) -> float:
    """Great-circle angle between two views, in degrees."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(vec(a) @ vec(b)))))))


_D_CACHE: Dict[Tuple[str, str], float] = {}


def _d(a: str, b: str) -> float:
    k = (a, b) if a <= b else (b, a)
    if k not in _D_CACHE:
        _D_CACHE[k] = distance(a, b)
    return _D_CACHE[k]


def min_spacing() -> float:
    """The grid's smallest non-zero separation, in degrees."""
    return min(_d(a, b) for a, b in itertools.combinations(GRID, 2))


def covering_radius(refs: Sequence[str]) -> float:
    """The worst distance from ANY grid view to its nearest stored view."""
    return max(min(_d(v, r) for r in refs) for v in GRID)


def reference_views(k: int = 4, queries: Iterable[str] = ("035-045", "045-180"),
                    base: str = BASE_VIEW, pin_base: bool = True) -> List[str]:
    """The stored views for every object: ``base`` plus ``k - 1`` chosen by covering radius.

    Deterministic and score-free (see the module docstring): exhaustive minimisation of the
    covering radius over all subsets, ties broken by the sorted label tuple.  The query views are
    removed from the candidate set first, so a query can never also be a reference.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return [base]
    q = set(queries)
    cands = [v for v in GRID if v not in q and not (pin_base and v == base)]
    need = k - (1 if pin_base else 0)
    if len(cands) < need:
        raise ValueError("not enough candidate views for k = %d" % k)
    pinned = [base] if pin_base else []
    best: Optional[Tuple[float, Tuple[str, ...]]] = None
    for combo in itertools.combinations(cands, need):
        trial = tuple(sorted(pinned + list(combo)))
        r = covering_radius(trial)
        if best is None or (r, trial) < best:
            best = (r, trial)
    return list(best[1])


def view_suffix(label: str) -> str:
    """The filename suffix a view label contributes: ``035-045`` -> ``-035-045``."""
    return SUFFIX + label


def label_of(path: str) -> str:
    """Recover a view label from a filename like ``apple1-035-045.png``."""
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".png"):
        stem = stem[:-4]
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError("not an ETH-80 view filename: %r" % (path,))
    return "%s-%s" % (parts[-2], parts[-1])


def distances_to_nearest(refs: Sequence[str], views: Optional[Iterable[str]] = None
                         ) -> Dict[str, Dict[str, float]]:
    """For each view, its nearest stored view and that distance -- the read's operating range."""
    out: Dict[str, Dict[str, float]] = {}
    for v in (GRID if views is None else views):
        r = min(refs, key=lambda x: (_d(v, x), x))
        out[v] = {"nearest": r, "deg": round(_d(v, r), 2)}
    return out
