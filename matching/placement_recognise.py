"""THE ENTRY POINT: recognise an object against a gallery of known objects (Mission 24).

    from matching.placement_recognise import build_gallery, recognise

    gallery = build_gallery([("apple/1", "apple1-000-000.png"),
                             ("apple/2", "apple2-000-000.png"), ...])
    result  = recognise("apple3-035-045.png", gallery)

    result.ranking     every gallery index, best first (the shortlist re-ordered by the matcher,
                       then everything else in the door's order -- the re-order never drops)
    result.shortlist   the S = 8 the matcher scored
    result.door_order  the fused door's ranking of the whole gallery
    result.scores      {gallery index: shipped matcher score} for the scored candidates
    result.unscored    shortlisted candidates the matcher could not score (a refused pair, not a
                       zero and not a fallback -- they stay in the ranking at their door rank)
    result.seconds     {"appearance", "door", "matcher", "total"}

WHAT IT IS.  The content-count door -- every query place probes a pooled index of gallery places
by content key, keeps its nearest few by the frozen Mission-21 ruler and COUNTS which object they
belong to -- fused by reciprocal rank fusion (k = 60) with an unfitted colour+gradient histogram,
shortlist S = 8, then the M19 matcher scores those 8 and the final order is its plain re-order.
Mission 23 measured this configuration and it is what ships; nothing here re-tunes it.

WHAT IT DOES NOT DO.  It does not estimate pose, it does not recognise a new instance of a class
reliably (0.500 against 0.103 chance), and it degrades at yaw beyond ~20 degrees (pitch is fine).

CONFIGURATION.  One frozen file, ``matching/placement_bundle.json`` (see ``placement_bundle``).
No environment variable is needed and none is read: the entry point applies the bundle internally,
inside a scope that restores every switch it touched, so the legacy env-flag path is exactly as it
was before the call (``tests/test_m24_core.py`` asserts that).

WHY THE 8 PAIR SCORES RUN IN PARALLEL.  They are independent -- one (query, gallery) pair each --
and each costs ~15 s on this box.  They are scored in a process pool and collected IN SHORTLIST
ORDER, so the output is identical to the serial run (asserted by a test).  The pool is created per
call because a worker must not inherit a half-applied flag scope.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matching import placement_bundle as PB

__all__ = ["Gallery", "Result", "build_gallery", "recognise", "order_plain", "PlacementBundle"]


def _library() -> Tuple[Any, Any, Any]:
    """(frontend, frontdoor, gallery module) -- imported lazily, in that order."""
    from matching import placement_frontend as FE
    from matching import placement_frontdoor as FD
    from matching import placement_gallery as GL
    return FE, FD, GL


Gallery = Any                                       # matching.placement_gallery.Gallery
PlacementBundle = PB.PlacementBundle


@dataclass
class Result:
    """What ``recognise`` returns.  Every field is deterministic given (query, gallery, bundle)."""

    query: str
    ranking: List[int]
    shortlist: List[int]
    door_order: List[int]
    scores: Dict[int, float]
    unscored: List[int]
    seconds: Dict[str, float]
    bundle: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> int:
        return self.ranking[0] if self.ranking else -1

    def key_of(self, gallery: Gallery) -> str:
        return gallery.keys[self.top] if self.ranking else ""

    def as_dict(self) -> Dict[str, Any]:
        return {"query": os.path.basename(self.query), "ranking": self.ranking,
                "shortlist": self.shortlist, "door_order": self.door_order,
                "scores": {str(k): v for k, v in sorted(self.scores.items())},
                "unscored": self.unscored, "seconds": self.seconds, "bundle": self.bundle}


# --------------------------------------------------------------------------- the flag scope
def _validate(bundle: PlacementBundle) -> None:
    """Refuse a bundle this code cannot honour.  No silent fallback anywhere."""
    FE, _FD, _GL = _library()
    if dict(bundle.recipe) != dict(FE.RECIPE):
        raise PB.BundleRefusal(
            "the bundle's front-end recipe is not this tree's recipe: %d key(s) differ.  A "
            "different recipe builds different graphs, so its numbers are not these numbers."
            % len({k for k in set(bundle.recipe) | set(FE.RECIPE)
                   if bundle.recipe.get(k) != FE.RECIPE.get(k)}))
    if dict(bundle.recipe_extra) != dict(FE.RECIPE_EXTRA):
        raise PB.BundleRefusal("the bundle's extra pre-switches are not this tree's")
    if bundle.rerank != "plain":
        raise PB.BundleRefusal("the shipped order is the matcher's plain re-order; this code does "
                               "not implement %r" % (bundle.rerank))
    if int(bundle.shortlist) < 1 or int(bundle.rrf_k) < 1:
        raise PB.BundleRefusal("bundle shortlist/rrf_k are not usable: %r / %r"
                               % (bundle.shortlist, bundle.rrf_k))
    if abs(float(bundle.matcher_weight) - PB.shipped_weight()) > 1e-12:
        raise PB.BundleRefusal("the bundle's matcher weight %.6f is not the M19 record's %.6f"
                               % (bundle.matcher_weight, PB.shipped_weight()))


class _FlagScope:
    """Set switches for the duration of a call and put every one of them back.

    The legacy path reads these as ordinary ``parameters`` module attributes at call time, so a
    scope is all that is needed -- but it must be a scope: a caller that leaves
    ``PLACEMENT_SCORE_WEIGHT`` at 0.375 changes every later measurement in the process.
    """

    def __init__(self, values: Dict[str, Any]):
        self.values = dict(values)
        self.old: Dict[str, Any] = {}
        self.had: Dict[str, bool] = {}

    def __enter__(self) -> "_FlagScope":
        import parameters as P
        for k, v in self.values.items():
            self.had[k] = hasattr(P, k)
            self.old[k] = getattr(P, k, None)
            setattr(P, k, v)
        return self

    def __exit__(self, *exc: Any) -> bool:
        import parameters as P
        for k, v in self.old.items():
            if self.had[k]:
                setattr(P, k, v)
            else:                                          # pragma: no cover - defensive
                try:
                    delattr(P, k)
                except AttributeError:
                    pass
        return False


def _bundle_flags(bundle: PlacementBundle) -> Dict[str, Any]:
    FE, _FD, _GL = _library()
    vals: Dict[str, Any] = {}
    vals.update(bundle.recipe)
    vals.update(bundle.recipe_extra)
    vals.update(FE.matcher_flags(bundle.matcher_weight))
    return vals


def order_plain(cands: Sequence[int], scores: Dict[int, float]) -> List[int]:
    """THE SHIPPED ORDER: the matcher's plain re-order of the candidates, dropping none.

    Sorting by the shipped score, ties broken by gallery index, so the order is deterministic.
    A candidate with no score sorts last (by index) -- it is not removed, and it is not given a
    fabricated score.
    """
    return sorted(cands, key=lambda j: (-float(scores.get(j, 0.0)), j))


# --------------------------------------------------------------------------- the two entry points
def build_gallery(items: Iterable[Any], *, bundle: Optional[PlacementBundle] = None,
                  workers: int = 2, use_cache: bool = True,
                  cache_dir: Optional[str] = None) -> Gallery:
    """Build (or load) the gallery for ``items``: (key, path) pairs or dicts with key and path.

    The bundle is loaded (and validated) first, because its ``code_digest`` is half of the cache
    key: a gallery built by different code is not this gallery.
    """
    FE, _FD, GL = _library()
    b = bundle or PB.PlacementBundle.load()
    _validate(b)
    with _FlagScope(_bundle_flags(b)):
        g = GL.build_gallery(items, code_digest=b.code_digest, workers=workers,
                             use_cache=use_cache, cache_dir=cache_dir)
    g.seconds["total"] = g.seconds.get("total", 0.0)
    return g


def _score_one(job: Tuple[str, str, float]) -> Optional[float]:
    """One pair, in a worker.  Returns None when the matcher refuses the pair."""
    from matching import placement_frontend as FE
    qpath, rpath, weight = job
    FE.apply_recipe()
    try:
        return FE.pair_score(qpath, rpath, weight=weight)
    except Exception:                                     # noqa: BLE001 - a refusal is data
        return None


def _worker_init(bundle_fields: Dict[str, Any]) -> None:  # pragma: no cover - runs in a worker
    with _FlagScope(bundle_fields):
        pass


def recognise(query: Any, gallery: Gallery, *, bundle: Optional[PlacementBundle] = None,
              workers: int = 2, s: Optional[int] = None, matcher: bool = True) -> Result:
    """Rank ``gallery`` for ``query``.  ``query`` is a path or a (key, path) pair.

    ``matcher=False`` stops after the door: the result carries the door's ranking and the
    shortlist, and ``unscored`` is the whole shortlist (nothing was scored, nothing is claimed).
    That is what a caller who only wants candidates should use -- it costs no part of the ~15 s a
    pair score takes.
    """
    FE, FD, _GL = _library()
    b = bundle or PB.PlacementBundle.load()
    _validate(b)
    qpath = str(query[1]) if isinstance(query, (tuple, list)) else str(query)
    s_eff = int(b.shortlist if s is None else s)
    t_all = time.time()
    with _FlagScope(_bundle_flags(b)):
        t0 = time.time()
        q = FD.appearance_feature(qpath)
        app = np.asarray([q], dtype=np.float64) @ gallery.appearance.T
        t_app = time.time() - t0
        t0 = time.time()
        cnt, _diag = FD.count_scores(gallery.index, [FE.place_record(qpath, tag="query")],
                                     n_hits=int(b.door_n_hits))
        fused = FD.rank_fuse(app, cnt, k=int(b.rrf_k))
        door_order = [j for j in sorted(range(len(gallery)), key=lambda j: (-fused[0, j], j))]
        t_door = time.time() - t0
        short = door_order[:s_eff]
        t0 = time.time()
        jobs = [(qpath, gallery.paths[j], float(b.matcher_weight)) for j in short]
        if not matcher:
            vals = [None] * len(jobs)
        elif workers and int(workers) > 1 and len(jobs) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            try:
                ctx = mp.get_context("fork")
            except ValueError:                             # pragma: no cover - non-fork platforms
                ctx = mp.get_context()
            with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
                vals = list(ex.map(_score_one, jobs, chunksize=1))
        else:
            vals = [_score_one(j) for j in jobs]
        t_mat = time.time() - t0
    scores: Dict[int, float] = {}
    unscored: List[int] = []
    for j, v in zip(short, vals):                          # collected in SHORTLIST order
        if v is None:
            unscored.append(j)
        else:
            scores[j] = float(v)
    ordered = order_plain(short, scores)
    rest = [j for j in door_order if j not in set(short)]
    ranking = ordered + rest
    return Result(query=qpath, ranking=ranking, shortlist=short, door_order=door_order,
                  scores=scores, unscored=unscored,
                  seconds={"appearance": t_app, "door": t_door, "matcher": t_mat,
                           "total": time.time() - t_all},
                  bundle=b.as_dict())


# --------------------------------------------------------------------------- operator CLI
def _main(argv: Optional[Sequence[str]] = None) -> int:   # pragma: no cover - operator action
    import argparse
    import json
    # Run it as `python -m matching.placement_recognise` from the repository root: that is what
    # puts the tree's top-level modules (eth80_frontend, parameters) on sys.path.  Nothing here
    # touches sys.path -- a library that rearranges the import path for its caller is a library
    # that changes what the caller's own imports mean.
    ap = argparse.ArgumentParser(description="the shipped placement recogniser")
    ap.add_argument("--freeze", action="store_true",
                    help="write matching/placement_bundle.json for this tree")
    ap.add_argument("--describe", action="store_true", help="print the frozen bundle")
    ap.add_argument("--date", default="")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.freeze:
        d = PB.freeze(date=a.date or time.strftime("%Y-%m-%d"),
                      provenance={"measured_by": "MISSION23_REPORT.md section 5 (the read)",
                                  "entry_point": "matching/placement_recognise.py",
                                  "note": "frozen from the configuration Mission 23 shipped"})
        print("froze %s\n  code_digest %s\n  shortlist %d, rrf_k %d, rerank %s, door n_hits %d"
              % (PB.BUNDLE_PATH, d["code_digest"][:16], d["shortlist"], d["rrf_k"],
                 d["rerank"], d["door"]["n_hits"]))
        return 0
    b = PB.PlacementBundle.load()
    print(json.dumps(b.as_dict() | {"door_n_hits": b.door_n_hits,
                                    "matcher_weight": b.matcher_weight,
                                    "appearance": b.appearance}, indent=1))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(_main())
