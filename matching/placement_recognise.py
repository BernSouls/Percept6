"""THE ENTRY POINT: recognise an object against a gallery of known objects.

    from matching.placement_recognise import build_gallery, recognise

    gallery = build_gallery([
        ("apple/1", {"000-000": "apple1-000-000.png", "045-000": "apple1-045-000.png", ...}),
        ("apple/2", {"000-000": "apple2-000-000.png", "045-000": "apple2-045-000.png", ...}),
    ])                                     # one entry per OBJECT, k stored views each
    result = recognise("apple3-035-045.png", gallery)

    result.ranking      every gallery OBJECT, best first (the shortlist re-ordered by the matcher,
                        then everything else in the door's order -- the re-order never drops)
    result.shortlist    the S = 8 OBJECTS the matcher scored
    result.door_order   the fused door's ranking of every object
    result.best_view    {object index: the stored view the door chose for this query}
    result.scores       {object index: shipped matcher score}, scored on that best view
    result.unscored     shortlisted objects the matcher could not score (a refused pair, not a zero)
    result.seconds      {"appearance", "door", "matcher", "total"}

WHAT IT IS.  For every stored view of every object the content-count door scores the query's place
probes -- each query place probes a pooled index of gallery places by content key, keeps its
nearest few by the frozen Mission-21 ruler, and COUNTS which view they belong to; a view scores
``count / sqrt(query places x view places)``.  An OBJECT scores the MAXIMUM over its stored views,
and likewise the appearance channel.  The two object score lists are fused by reciprocal rank
fusion (k = 60) and the shortlist is S = 8 OBJECTS -- the same S as before, so recall is comparable
with chance 8 / N_objects.  The M19 matcher scores each shortlisted object on the ONE stored view
the door chose for it, and the final order is its plain re-order.  At one stored view per object
every number here is exactly the Mission-24 number, because a max over a single view is that view.

WHAT IT DOES NOT DO.  It does not estimate pose; it does not recognise a new instance of a class
reliably (0.500 against 0.103 chance); and it degrades at yaw beyond ~20 degrees, which is why more
stored views help only in so far as one of them is CLOSE in yaw to the query.

CONFIGURATION.  One frozen file, ``matching/placement_bundle.json`` (see ``placement_bundle``),
which also names the stored views.  No environment variable is needed and none is read: the entry
point applies the bundle internally, inside a scope that restores every switch it touched, so the
legacy env-flag path is exactly as it was before the call (``tests/test_m24_core.py`` asserts it).

CACHES.  Graph builds (content-hashed, ``matching/placement_cache``), gallery records (keyed on the
items and the bundle's code digest, ``matching/placement_gallery``) and PAIR SCORES (keyed on the
two image paths, the M19 weight and a digest of the matcher's own parameters, in
``.cache/placement/pairs.jsonl``).  The pair cache is what makes a re-read free: the matcher is the
expensive stage, and a pair that has been scored is never scored twice.  A pair the matcher refuses
is never cached as a number.

WHY PAIRS RUN IN PARALLEL.  The S pair scores are independent -- one (query, stored view) pair each
-- and each costs ~15 s on this box.  They run in a process pool and are collected IN SHORTLIST
ORDER, so the output is identical to the serial run (asserted by a test).  The pool is created per
call because a worker must not inherit a half-applied flag scope.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matching import placement_bundle as PB

__all__ = ["Gallery", "Result", "build_gallery", "recognise", "order_plain", "PlacementBundle",
           "matcher_digest", "pair_cache_path"]

Gallery = Any                                       # matching.placement_gallery.Gallery
PlacementBundle = PB.PlacementBundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_CACHE = os.environ.get("PLACEMENT_PAIR_CACHE") or os.path.join(
    ROOT, ".cache", "placement", "pairs.jsonl")


def _library() -> Tuple[Any, Any, Any]:
    """(frontend, frontdoor, gallery module) -- imported lazily, in that order."""
    from matching import placement_frontend as FE
    from matching import placement_frontdoor as FD
    from matching import placement_gallery as GL
    return FE, FD, GL


@dataclass
class Result:
    """What ``recognise`` returns.  Deterministic given (query, gallery, bundle)."""

    query: str
    ranking: List[int]
    shortlist: List[int]
    door_order: List[int]
    best_view: Dict[int, str]
    scores: Dict[int, float]
    unscored: List[int]
    seconds: Dict[str, float]
    views: List[str] = field(default_factory=list)
    bundle: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> int:
        return self.ranking[0] if self.ranking else -1

    def key_of(self, gallery: Gallery) -> str:
        return gallery.keys[self.top] if self.ranking else ""

    def as_dict(self) -> Dict[str, Any]:
        return {"query": os.path.basename(self.query), "ranking": self.ranking,
                "shortlist": self.shortlist, "door_order": self.door_order,
                "best_view": {str(k): v for k, v in sorted(self.best_view.items())},
                "scores": {str(k): v for k, v in sorted(self.scores.items())},
                "unscored": self.unscored, "seconds": self.seconds, "views": self.views,
                "bundle": self.bundle}


# --------------------------------------------------------------------------- the flag scope
def _validate(bundle: PlacementBundle) -> None:
    """Refuse a bundle this code cannot honour.  No silent fallback anywhere."""
    FE, _FD, _GL = _library()
    from matching import placement_views as V
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
    if str(getattr(bundle, "door_kind", "counted")) == "bag":
        from matching import placement_bag as BAG
        BAG.load_edges()                     # a bag door without its reference ruler is a refusal
        if not bundle.colour_rings:
            raise PB.BundleRefusal(
                "the bundle ships door.kind = bag but content.kind = %r: the descriptor it reads "
                "is the place colour content, so a bag door requires colour_rings"
                % (bundle.content_kind,))
    if str(getattr(bundle, "door_kind", "counted")) == "fused3":
        if not bundle.colour_rings:
            raise PB.BundleRefusal(
                "the bundle ships door.kind = fused3 but content.kind = %r: the colour channel "
                "reads the place content, so fused3 requires colour_rings"
                % (bundle.content_kind,))
        wcol = float(bundle.fusion_weight)
        if not (0.0 <= wcol <= 1.0):
            raise PB.BundleRefusal("fused3 colour weight %r is not a vote in [0, 1] -- a negative "
                                   "or unbounded rank weight is not a fusion (M30 shipped 0.5; 0 "
                                   "is the two-channel shipped door, pinned by test)" % (wcol,))
    blend = dict(getattr(bundle, "matcher_blend", {}) or {})
    if blend:
        wc = float(blend.get("w_cfg", 0.0))
        wl = float(blend.get("w_col", 0.0))
        if wl < 0.0 or wc < 0.0 or wc + wl > 1.0:
            raise PB.BundleRefusal("matcher blend w_cfg %r + w_col %r is not a convex blend"
                                   % (wc, wl))
    if int(bundle.shortlist) < 1 or int(bundle.rrf_k) < 1:
        raise PB.BundleRefusal("bundle shortlist/rrf_k are not usable: %r / %r"
                               % (bundle.shortlist, bundle.rrf_k))
    if abs(float(bundle.matcher_weight) - PB.shipped_weight()) > 1e-12:
        raise PB.BundleRefusal("the bundle's matcher weight %.6f is not the M19 record's %.6f"
                               % (bundle.matcher_weight, PB.shipped_weight()))
    unknown = [v for v in bundle.reference_views if v not in V.GRID]
    if unknown:
        raise PB.BundleRefusal("the bundle names stored views that are not on the grid: %r"
                               % (unknown,))
    if bundle.reserved and set(bundle.reserved) & set(bundle.reference_views):
        raise PB.BundleRefusal("a view cannot be both a stored reference and a read query: %r"
                               % sorted(set(bundle.reserved) & set(bundle.reference_views)))


class _FlagScope:
    """Set switches for the duration of a call and put every one of them back."""

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
    blend = dict(getattr(bundle, "matcher_blend", {}) or {})       # M31 item 1c
    if blend:
        vals["PLACEMENT_NODE_COLOUR"] = float(blend.get("w_col", 0.0)) > 0.0
        vals["PLACEMENT_NODE_COLOUR_W"] = float(blend.get("w_col", 0.0))
    return vals


# --------------------------------------------------------------------------- the pair cache
def matcher_digest() -> str:
    """Digest of the MATCHER's own configuration: the M19 record plus the flags it runs under.

    Deliberately narrower than the bundle's code digest: a pair score depends on the matcher and
    its weight, not on the door or the gallery code, so an edit to the door must not throw a
    perfectly good pair score away.  An edit to the matcher MUST.
    """
    from matching import placement_frontend as FE
    h = hashlib.sha256()
    with open(PB.SCORE_PARAMS, "rb") as fh:
        h.update(fh.read())
    h.update(repr(sorted(FE.MATCHER_FLAGS.items())).encode())
    h.update(repr(PB.shipped_weight()).encode())
    return h.hexdigest()[:16]


def pair_env() -> str:
    """Digest of the GRAPH PIPELINE that produced a score -- the thing a pair cache must pin.

    ``matcher_digest`` covers the matcher's configuration (parameters file, flags, weight).  It does
    not cover the code that turns pixels into graphs and the switches that code reads, and a pair
    score depends on both.  Mission 27 measured the consequence: three of M23's committed pairs
    recomputed at +0.0052, +0.0016 and +0.0099 on scores near 0.35, while the place RECORDS of the
    same cells are identical.

    The digest folds the front end's live switch values (that is what its own cache key does), so it
    must be taken under the SHIPPED recipe and the caller's values restored afterwards --
    otherwise the same environment fingerprints differently depending on what ran before it, which
    is a cache that misses when it should hit.  Mission 28 item 1b fixed exactly that.
    """
    import hashlib
    import parameters as P
    from matching import placement_cache as GC
    from matching import placement_frontend as FE
    saved = {k: getattr(P, k) for k in list(FE.RECIPE) + list(FE.RECIPE_EXTRA) if hasattr(P, k)}
    try:
        FE.apply_recipe()
        parts = (GC._source_digest(), GC._switch_digest())
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode())
    for mod in ("numpy", "scipy"):
        try:
            m = __import__(mod)
            h.update(("%s=%s" % (mod, getattr(m, "__version__", "?"))).encode())
        except Exception:
            h.update(("%s=missing" % mod).encode())
    return h.hexdigest()[:16]


def pair_cache_path() -> str:
    return PAIR_CACHE


def _pair_key(qpath: str, rpath: str, weight: float) -> str:
    return "%s|%s|%s|%.12g" % (matcher_digest(), os.path.basename(qpath),
                               os.path.basename(rpath), float(weight))


#: Pair scores COMMITTED by earlier missions, adopted under the current matcher digest.  They were
#: scored with the shipped flags and the shipped weight (0.375), which is what the digest covers,
#: so a fresh sandbox does not have to buy them again.  Keys are built from BASENAMES: a different
#: checkout prefix must not turn a cached pair into a miss (Mission 27, item 1a).
LEGACY_PAIR_FILES: Tuple[str, ...] = ("m23_read_pairs.jsonl", "m22_assemble_fit_pairs.jsonl",
                                      "m22_assemble_dev_pairs.jsonl")
LEGACY_PAIR_DIR = os.path.join(ROOT, "Topology", "results", "placement")
ADOPTED: Dict[str, int] = {}


def _legacy_pair_rows(path: str) -> List[Tuple[str, str, float]]:
    """(query name, reference name, score) for every row of a committed pair file."""
    rows: List[Tuple[str, str, float]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "B" not in r or not r.get("q") or not r.get("r"):
            continue
        rows.append((os.path.basename(str(r["q"])), os.path.basename(str(r["r"])),
                     float(r["B"])))
    return rows


def load_pair_cache(path: Optional[str] = None, *, legacy: bool = True,
                    same_env: bool = False) -> Dict[str, float]:
    """The pair scores on disk, keyed by BASENAME so a different checkout prefix still hits.

    Two kinds of row end up here:

    * rows this tree wrote, each carrying ``d`` (the matcher digest) and ``env`` (the graph-pipeline
      digest, see ``pair_env``).  A row is used when ``d`` matches; with ``same_env=True`` its
      ``env`` must match too, which is what a DECISION run needs: a score is only this tree's score
      if the pipeline that produced it is this tree's pipeline.
    * the committed M22/M23 files, adopted under the current matcher digest and counted in
      ``ADOPTED``.  They carry no ``env`` (they predate it), so ``same_env=True`` drops them --
      that is the honest reading of "the environment is the arithmetic", and Mission 27 measured
      the size of the difference (+0.004 mean, max +0.010, on scores near 0.35).
    """
    from matching.placement_bundle import shipped_weight
    p = path or PAIR_CACHE
    out: Dict[str, float] = {}
    ADOPTED.clear()
    env = pair_env() if same_env else ""
    if legacy and not same_env:
        w = float(shipped_weight())
        for name in LEGACY_PAIR_FILES:
            lp = os.path.join(LEGACY_PAIR_DIR, name)
            rows = _legacy_pair_rows(lp)
            n = 0
            for qn, rn, b in rows:
                key = "%s|%s|%s|%.12g" % (matcher_digest(), qn, rn, w)
                if key not in out:
                    out[key] = b
                    n += 1
            if rows:
                ADOPTED[name] = n
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue                                        # a corrupt line is a miss
        if r.get("d") != matcher_digest() or "B" not in r:
            continue
        if same_env and str(r.get("env") or "") != env:
            continue
        out["%s|%s|%s|%.12g" % (r["d"], r["q"], r["r"], float(r["w"]))] = float(r["B"])
    return out


def append_pair_cache(rows: Sequence[Dict[str, Any]], path: Optional[str] = None) -> None:
    """Append scored pairs (``{"q", "r", "w", "B"}`` with basenames) to the cache file."""
    p = path or PAIR_CACHE
    if not rows:
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = matcher_digest()
    env = pair_env()
    with open(p, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"d": d, "env": str(r.get("env") or env),
                                 "q": os.path.basename(str(r["q"])),
                                 "r": os.path.basename(str(r["r"])), "w": float(r["w"]),
                                 "B": float(r["B"])}, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- the two entry points
def order_plain(cands: Sequence[int], scores: Dict[int, float]) -> List[int]:
    """THE SHIPPED ORDER: the matcher's plain re-order of the candidates, dropping none.

    Sorting by the shipped score, ties broken by gallery index, so the order is deterministic.
    A candidate with no score sorts last (by index) -- it is not removed, and it is not given a
    fabricated score.
    """
    return sorted(cands, key=lambda j: (-float(scores.get(j, 0.0)), j))


def build_gallery(items: Iterable[Any], *, bundle: Optional[PlacementBundle] = None,
                  workers: int = 2, use_cache: bool = True,
                  cache_dir: Optional[str] = None, views: Optional[Sequence[str]] = None
                  ) -> Gallery:
    """Build (or load) the gallery for ``items`` (one entry per object).

    ``items`` are ``(key, path)``, ``(key, {view: path})`` or ``(key, [(view, path), ...])``.
    ``views`` overrides the stored views the bundle names (the read uses this to compare k = 1
    against k = 4 on the same code); everything else comes from the frozen bundle.
    """
    _FE, _FD, GL = _library()
    b = bundle or PB.PlacementBundle.load()
    _validate(b)
    vs = list(b.reference_views if views is None else views)
    with _FlagScope(_bundle_flags(b)):
        # [M32] the content kind is part of the shipped CONFIGURATION: a gallery built with the
        # default keypoint records under a bundle that promises colour_rings is refused later
        # ("the colour door over keypoint records has no colour to read"), so the entry point has
        # to carry the bundle's kind, not the library's default.
        gal = GL.build_gallery(items, code_digest=b.code_digest, views=vs, workers=workers,
                               use_cache=use_cache, cache_dir=cache_dir,
                               content=getattr(b, "content_kind", None))
    # [M34] the fragment channel is part of the frozen door: a gallery that can carry its cells'
    # anchors carries them (a cache hit returns the same Gallery object, so attach AFTER the call).
    if float(b.fragment_weight) != 0.0:
        from matching import placement_anchors as PA
        GL.attach_fragment_anchors(gal, PA.load())
    # [M35] the bag door reads the OBJECT descriptor: build it here, from the records the gallery
    # already carries, so `recognise` does not rebuild 160 descriptors per query.  A cache hit
    # returns the same Gallery object, so attach AFTER the call.
    if str(getattr(b, "door_kind", "counted")) == "bag":
        from matching import placement_bag as BAG
        from matching import placement_shape as PS
        edges = BAG.load_edges()
        gal.bag_features = []
        for r, p in zip(gal.flat_records(), gal.flat_paths()):
            if any("eigratio" not in x for x in r["places"]):
                PS.attach(r, _FE.load_image(p))       # the same enrichment a live query gets
            gal.bag_features.append(BAG.describe(r, edges))
    return gal


def _score_one(job: Tuple[str, str, float]) -> Optional[float]:
    """One pair, in a worker.  Returns None when the matcher refuses the pair."""
    from matching import placement_frontend as FE
    qpath, rpath, weight = job
    FE.apply_recipe()
    try:
        return FE.pair_score(qpath, rpath, weight=weight)
    except Exception:                                     # noqa: BLE001 - a refusal is data
        return None


def _shortlist_size(scores: np.ndarray, order: Sequence[int], b: PlacementBundle,
                    s: int) -> int:
    """ADAPTIVE S (M26 item 6D), fixed on the FIT cells: widen when the top-s margin is not clear.

    ``margin = (score[s-1] - score[s]) / score[0]`` on the fused door; the bundle holds the wide
    size and the threshold the FIT-object cells fixed.  ``shortlist_rule = "fixed"`` keeps the
    shipped S = 8 everywhere, which is what M24/M25 measured.
    """
    rule = str(getattr(b, "shortlist_rule", "fixed"))
    if rule != "adaptive":
        return int(s)
    wide = int(getattr(b, "shortlist_wide", 16))
    tau = float(getattr(b, "margin_tau", 0.0))
    v = np.asarray([scores[j] for j in order], dtype=np.float64)
    top = float(v[0]) if v.size else 0.0
    if v.size <= s or top <= 0.0:
        return int(s)
    margin = float(v[s - 1] - v[s]) / top
    return wide if margin < tau else int(s)


def _max_over_views(flat: np.ndarray, n_obj: int, n_views: int) -> np.ndarray:
    """(n, n_obj * n_views) object-major -> (n, n_obj) maximum over each object's views."""
    return flat.reshape(flat.shape[0], n_obj, n_views).max(axis=2)


def _argmax_over_views(flat: np.ndarray, n_obj: int, n_views: int) -> List[int]:
    """Which view of each object scored highest; ties go to the lowest view index."""
    m = flat.reshape(flat.shape[0], n_obj, n_views)[0]
    return [int(np.argmax(m[j])) if n_views > 1 else 0 for j in range(n_obj)]


def recognise(query: Any, gallery: Gallery, *, bundle: Optional[PlacementBundle] = None,
              workers: int = 2, s: Optional[int] = None, matcher: bool = True,
              use_pair_cache: bool = True, record: Optional[Dict[str, Any]] = None,
              appearance: Optional[np.ndarray] = None,
              anchors: Optional[np.ndarray] = None) -> Result:
    """Rank ``gallery``'s OBJECTS for ``query``.  ``query`` is a path or a (key, path) pair.

    [M31] ``record`` and ``appearance`` let a caller hand in the query's own place record and
    appearance vector instead of rebuilding them from ``query``.  That is what makes the door's
    output reproducible from the committed records -- the whole 80 x 2 equality is then a
    records-only computation -- and it changes nothing else: the record is the same object
    ``place_record`` returns, read from the npz rather than from a graph.

    ``matcher=False`` stops after the door: the result carries the door's object ranking and the
    shortlist, and ``unscored`` is the whole shortlist (nothing was scored, nothing is claimed).

    [M34] ``anchors`` is the QUERY cell's `(n, 2)` fragment anchors (`matching/placement_anchors`).
    A bundle whose `fusion.fragment` is non-zero needs them, and needs the gallery's, and is REFUSED
    without them: the fragment channel is part of the arithmetic the bundle froze, so a run that
    cannot compute it is not the shipped configuration.
    """
    FE, FD, _GL = _library()
    b = bundle or PB.PlacementBundle.load()
    _validate(b)
    qpath = str(query[1]) if isinstance(query, (tuple, list)) else str(query)
    s_eff = int(b.shortlist if s is None else s)
    n_obj, n_views = gallery.n, gallery.n_views
    t_all = time.time()
    _dk = str(getattr(b, "door_kind", "counted"))
    with _FlagScope(_bundle_flags(b)):
        t0 = time.time()
        if _dk == "bag":                       # [M35] the bag door reads records, not appearance
            app_flat = np.zeros((1, n_obj * n_views), dtype=np.float64)
        else:
            q_app = (np.asarray(appearance, dtype=np.float64) if appearance is not None
                     else FD.appearance_feature(qpath))
            app_flat = np.asarray([q_app], dtype=np.float64) @ gallery.appearance.T
        t_app = time.time() - t0
        t0 = time.time()
        ckind = "colour_rings" if b.colour_rings else None
        qrec = (dict(record) if record is not None
                else FE.place_record(qpath, tag="query", content=ckind))   # built once
        if ckind is not None and str(gallery.content_kind) != ckind:
            raise PB.BundleRefusal(
                "the bundle reads content.kind = %r but the gallery was built with %r: rebuild the "
                "gallery through build_gallery(content=...), because a colour door over keypoint "
                "records has no colour to read" % (ckind, gallery.content_kind))
        t_rec = time.time() - t0
        t0 = time.time()
        dkind = str(getattr(b, "door_kind", "counted"))
        bag_flat = None
        if dkind == "bag":
            from matching import placement_bag as BAG
            from matching import placement_shape as PS
            edges = BAG.load_edges()
            need = [p for p in qrec["places"] if "eigratio" not in p]
            if need:                       # a live path: attach the graded columns from the image
                PS.attach(qrec, FE.load_image(qpath))
            q_feat = BAG.describe(qrec, edges)
            gal_feats = getattr(gallery, "bag_features", None)
            if gal_feats is None or len(gal_feats) != n_obj * n_views:
                gal_feats = [BAG.describe(r, edges) for r in gallery.flat_records()]
                gallery.bag_features = gal_feats
            fused = BAG.door_scores(q_feat, gal_feats, n_obj, n_views)[None, :]
            cnt_flat, _diag = np.zeros((1, n_obj * n_views)), {}
            colour_flat = None
            fragment_flat = None
        elif dkind == "graded":
            cnt_flat, _diag = FD.graded_scores(gallery.index, [qrec],
                                               **dict(getattr(b, "door_params", {}) or {}))
        else:
            cnt_flat, _diag = FD.count_scores(gallery.index, [qrec], n_hits=int(b.door_n_hits))
        colour_flat = None
        fragment_flat = None
        if dkind == "fused3":
            from matching import placement_fuse as FU
            colour_flat = np.asarray(FU.colour_channel(gallery.flat_records(), [qrec]),
                                     dtype=np.float64)
            if colour_flat.shape != (1, n_obj * n_views):
                raise PB.BundleRefusal("the colour channel returned %r for %d cells"
                                       % (colour_flat.shape, n_obj * n_views))
            if float(b.fragment_weight) != 0.0:
                from matching import placement_fragment as PF
                if anchors is None:                    # the shipped query is a PATH: resolve it
                    from matching import placement_anchors as PA
                    anchors = PA.for_path(qpath)
                    if anchors is None:                # an image the file does not cover: compute it
                        anchors = PA.anchors_from_image(qpath)
                gal_anch = getattr(gallery, "fragment_anchors", None)
                if gal_anch is None:
                    from matching import placement_anchors as PA2
                    gal_anch = [PA2.for_path(p) for p in gallery.flat_paths()]
                    gallery.fragment_anchors = gal_anch
                if any(a is None for a in gal_anch):
                    from matching import placement_anchors as PA3
                    gal_anch = [(a if a is not None else PA3.anchors_from_image(p))
                                for a, p in zip(gal_anch, gallery.flat_paths())]
                    gallery.fragment_anchors = gal_anch
                if anchors is None:
                    raise PB.BundleRefusal(
                        "the bundle fuses the fragment channel at weight %.3f but %s anchors were "
                        "supplied: pass the query's anchors and build the gallery with "
                        "`placement_anchors` (the channel cannot be recomputed from the records "
                        "alone)" % (float(b.fragment_weight),
                                    "no query" if anchors is None else "no gallery"))
                fragment_flat = PF.channel([{"places": r["places"]} for r in gallery.flat_records()],
                                           qrec["places"], list(gal_anch), anchors)
                if fragment_flat.shape != (1, n_obj * n_views):
                    raise PB.BundleRefusal("the fragment channel returned %r for %d cells"
                                           % (fragment_flat.shape, n_obj * n_views))
        t_door = time.time() - t0
    app = _max_over_views(app_flat, n_obj, n_views)
    cnt = _max_over_views(cnt_flat, n_obj, n_views)
    which = _argmax_over_views(cnt_flat, n_obj, n_views)
    if str(getattr(b, "door_kind", "counted")) == "bag":
        pass                                        # `fused` is already the bag door's score
    elif str(getattr(b, "door_kind", "counted")) == "fused3":
        from matching import placement_fuse as FU
        colour = _max_over_views(colour_flat, n_obj, n_views)
        frag = (_max_over_views(fragment_flat, n_obj, n_views)
                if fragment_flat is not None else None)
        fused = np.asarray(FU.fuse(cnt, app, colour, fragment=frag,
                                   weights={"colour": float(b.fusion_weight),
                                            "fragment": float(b.fragment_weight)},
                                   k=int(b.rrf_k)), dtype=np.float64)
    else:
        fused = FD.rank_fuse(app, cnt, k=int(b.rrf_k))
    door_order = [j for j in sorted(range(n_obj), key=lambda j: (-fused[0, j], j))]
    s_used = _shortlist_size(fused[0], door_order, b, s_eff)
    short = door_order[:s_used]
    best_view = {j: gallery.views[which[j]] for j in short}
    t0 = time.time()
    jobs = [(qpath, gallery.paths[j][which[j]], float(b.matcher_weight)) for j in short]
    if not matcher:
        vals: List[Optional[float]] = [None] * len(jobs)
    else:
        vals = [None] * len(jobs)
        cached = load_pair_cache() if use_pair_cache else {}
        todo: List[int] = []
        fresh: List[Dict[str, Any]] = []
        for i, (qp, rp, w) in enumerate(jobs):
            hit = cached.get(_pair_key(qp, rp, w))
            if hit is not None:
                vals[i] = hit
            else:
                todo.append(i)
        if todo:
            sub = [jobs[i] for i in todo]
            if workers and int(workers) > 1 and len(sub) > 1:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                try:
                    ctx = mp.get_context("fork")
                except ValueError:                         # pragma: no cover - non-fork
                    ctx = mp.get_context()
                with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as ex:
                    got = list(ex.map(_score_one, sub, chunksize=1))
            else:
                got = [_score_one(j) for j in sub]
            for i, v in zip(todo, got):
                vals[i] = v
                if v is not None:
                    fresh.append({"q": jobs[i][0], "r": jobs[i][1], "w": jobs[i][2], "B": v})
            if use_pair_cache:
                append_pair_cache(fresh)
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
    return Result(query=qpath, ranking=ordered + rest, shortlist=short, door_order=door_order,
                  best_view=best_view, scores=scores, unscored=unscored,
                  seconds={"appearance": t_app, "query_record": t_rec, "door": t_door,
                           "matcher": t_mat, "total": time.time() - t_all},
                  views=list(gallery.views), bundle=b.as_dict())


# --------------------------------------------------------------------------- the class output
def recognise_class(query: Any, gallery: Gallery, *, bundle: Optional[PlacementBundle] = None,
                    record: Optional[Dict[str, Any]] = None, top: int = 2,
                    exclude: Optional[str] = None,
                    arm: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Rank the gallery's CLASSES for ``query``: the mean of a class's top-`top` instance scores.

    [M35 item 4] The descriptor is the object bag at the ALL arm -- every block, equal block weight
    (`matching.placement_bag.ARMS["ALL"]`) -- so the class output does not re-register a descriptor
    that item 2 did not measure.  The class of an object is its gallery key's first path component
    (`"apple/1"` -> `"apple"`); the query's own object is EXCLUDED when the query is a `(key, path)`
    pair or `exclude` names it, because "is the nearest class the right class" must not be answered
    by the query itself (its own descriptor scores 1.0 and would pin its own class).

    Returns ``{"classes": [(class, score), ...], "top": class, "scores": {...},
    "objects": (n_obj,) instance scores, "excluded": key-or-None}``.
    """
    from matching import placement_bag as BAG
    b = bundle or PB.PlacementBundle.load()
    _validate(b)
    qkey = None
    if isinstance(query, (tuple, list)) and len(query) >= 2:
        qkey = str(query[0])
    qpath = str(query[1]) if isinstance(query, (tuple, list)) else str(query)
    ex = exclude or qkey
    with _FlagScope(_bundle_flags(b)):
        edges = BAG.load_edges()
        qrec = (dict(record) if record is not None
                else _library()[0].place_record(qpath, tag="classquery",
                                                content=("colour_rings" if b.colour_rings else None)))
        gal_feats = getattr(gallery, "bag_features", None)
        if gal_feats is None or not gal_feats:
            gal_feats = [BAG.describe(r, edges) for r in gallery.flat_records()]
            gallery.bag_features = gal_feats
        q_feat = BAG.describe(qrec, edges)
        arm_used = tuple(arm or BAG.ARMS["ALL"])
        obj = BAG.door_scores(q_feat, gal_feats, gallery.n, gallery.n_views, arm=arm_used,
                              weighting=(BAG.WEIGHTING.get("ALL") if arm is None else None))
    classes = [str(k).split("/")[0] for k in gallery.keys]
    if ex is not None and ex in gallery.keys:
        obj = np.array(obj, dtype=np.float64)
        obj[gallery.keys.index(ex)] = np.nan
    keep = np.isfinite(obj)
    ranked = BAG.class_scores(obj[keep], np.asarray(classes, dtype=object)[keep], top=top)
    return {"classes": ranked, "top": ranked[0][0] if ranked else "",
            "scores": {c: v for c, v in ranked}, "objects": obj.tolist(),
            "excluded": ex, "top_n": int(top), "arm": list(arm_used),
            "bundle": b.as_dict()}


# --------------------------------------------------------------------------- operator CLI
def _main(argv: Optional[Sequence[str]] = None) -> int:   # pragma: no cover - operator action
    import argparse
    # Run it as `python -m matching.placement_recognise` from the repository root: that is what
    # puts the tree's top-level modules (eth80_frontend, parameters) on sys.path.  Nothing here
    # touches sys.path -- a library that rearranges the import path for its caller is a library
    # that changes what the caller's own imports mean.
    ap = argparse.ArgumentParser(description="the shipped placement recogniser")
    ap.add_argument("--freeze", action="store_true",
                    help="write matching/placement_bundle.json for this tree")
    ap.add_argument("--describe", action="store_true", help="print the frozen bundle")
    ap.add_argument("--date", default="")
    ap.add_argument("--k", type=int, default=0,
                    help="with --freeze: the default number of stored views per object")
    ap.add_argument("--door", default="counted", help="counted | graded (M26)")
    ap.add_argument("--shortlist", type=int, default=8,
                    help="the shipped shortlist size S (M27: 8 or 16)")
    ap.add_argument("--shortlist-rule", default="fixed", help="fixed | adaptive (M26 6D)")
    ap.add_argument("--margin-tau", type=float, default=0.0,
                    help="with --shortlist-rule adaptive: the FIT-fixed top-s margin threshold")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.freeze:
        d = PB.freeze(date=a.date or time.strftime("%Y-%m-%d"), k=(a.k or None),
                      door_kind=str(a.door or "counted"), shortlist=int(a.shortlist or 8),
                      shortlist_rule=str(a.shortlist_rule or "fixed"),
                      margin_tau=float(a.margin_tau or 0.0),
                      provenance={"measured_by": "MISSION23_REPORT.md section 5 (the read); "
                                                 "MISSION25_REPORT.md section 4 (the k = 1 vs "
                                                 "k = 4 read)",
                                  "entry_point": "matching/placement_recognise.py",
                                  "note": "frozen from the configuration Mission 25 shipped"})
        print("froze %s\n  code_digest %s\n  shortlist %d objects, rrf_k %d, rerank %s, "
              "door n_hits %d\n  stored views (k=%d): %s"
              % (PB.BUNDLE_PATH, d["code_digest"][:16], d["shortlist"], d["rrf_k"], d["rerank"],
                 d["door"]["n_hits"], d["references"]["k"], d["references"]["views"]))
        print("  door kind %s, shortlist rule %s (wide %d, margin tau %.4f)"
              % (d["door"]["kind"], d["shortlist_rule"], d["shortlist_wide"], d["margin_tau"]))
        return 0
    b = PB.PlacementBundle.load()
    print(json.dumps(b.as_dict() | {"door_kind": b.door_kind, "door_n_hits": b.door_n_hits,
                                    "matcher_weight": b.matcher_weight,
                                    "views": b.reference_views,
                                    "appearance": b.appearance}, indent=1))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(_main())
