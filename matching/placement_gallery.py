"""The gallery: photographs of known objects, turned into the thing the door can search.

``build_gallery`` takes items -- one per OBJECT, each with one or more stored views -- and returns
a ``Gallery`` holding

* ``records[object][view]`` -- one place record per stored view (``placement_frontend.place_record``):
  the content key, the frozen configuration row and the frame of every payload place;
* ``index`` -- the pooled door index over every stored VIEW, as one ``PlaceIndex`` (a flat list of
  view entries, object-major, with ``obj_keys`` the flat view ids);
* ``appearance`` -- the unfitted colour + gradient histograms of every stored view, one row per
  flat view, in the same object-major order;
* ``digest`` -- what the cache file is keyed on.

Mission 25 gave the gallery more than one view per object: the door's ceiling on an unseen view is
set by how far the nearest STORED view is, so an object can store several.  Which views is decided
in ``matching/placement_views.py`` (a score-free rule on the view grid) and frozen in the bundle.
At one view per object every number here is exactly the Mission-24 number, because a max over a
single view is that view.

The records are the expensive part (~13 s of graph build per image) and they are cached ON DISK,
keyed on the frozen bundle's code digest, the item keys, the view labels and the image files'
sizes.  The cache is working state: ``.cache/`` is gitignored and ``PLACEMENT_GALLERY_CACHE``
moves it.  No path in this module is hard-coded: the root comes from ``__file__`` and every path
comes from the items.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("PLACEMENT_GALLERY_CACHE") or os.path.join(ROOT, ".cache", "placement")


def _norm_items(items: Iterable[Any], views: Optional[Sequence[str]] = None
                ) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Normalise items to ``[(key, [(view, path), ...]), ...]``, sorted by key.

    Accepted forms, because a caller should not have to build a dataclass to store two pictures:

        (key, path)                     one view, its label read from the filename
        (key, {view: path, ...})        several views, explicit labels
        (key, [(view, path), ...])      several views, explicit labels
        {"key": ..., "path"/"views": ...}

    If ``views`` is given, every item must supply exactly those labels: a gallery missing a stored
    view is not a smaller gallery, it is a different one.
    """
    from matching import placement_views as V
    out: List[Tuple[str, List[Tuple[str, str]]]] = []
    for it in items:
        if isinstance(it, dict):
            key = str(it["key"])
            vv = it.get("views") or it.get("path")
        else:
            key, vv = str(it[0]), it[1]
        if isinstance(vv, str):
            pairs = [(V.label_of(vv), vv)]
        elif isinstance(vv, dict):
            pairs = [(str(k), str(p)) for k, p in vv.items()]
        else:
            pairs = [(str(a), str(b)) for a, b in vv]
        pairs.sort(key=lambda t: t[0])
        if views:
            want = list(views)
            have = [v for v, _p in pairs]
            if sorted(have) != sorted(want):
                raise ValueError("item %r supplies views %r, the bundle says %r"
                                 % (key, have, want))
            pairs = [(v, dict(pairs)[v]) for v in want]
        out.append((key, pairs))
    keys = [k for k, _p in out]
    if len(set(keys)) != len(keys):
        raise ValueError("gallery keys must be unique")
    out.sort(key=lambda t: t[0])
    if not out:
        raise ValueError("build_gallery needs at least one item")
    n_views = {len(p) for _k, p in out}
    if len(n_views) != 1:
        raise ValueError("every gallery item must supply the same number of views, got %r"
                         % (sorted(n_views),))
    return out


def gallery_digest(items: Sequence[Tuple[str, List[Tuple[str, str]]]], code_digest: str,
                   content: str = "keypoint") -> str:
    """Key for the on-disk cache: item keys, view labels, image sizes, the code digest, and
    (M31) the place CONTENT kind -- a colour gallery is not the keypoint gallery."""
    h = hashlib.sha256()
    h.update(code_digest.encode())
    h.update(b"content:" + str(content).encode())
    for k, pairs in items:
        h.update(k.encode())
        for v, p in pairs:
            h.update(v.encode())
            h.update(b"\0")
            h.update(os.path.basename(p).encode())
            try:
                h.update(str(os.path.getsize(p)).encode())
            except OSError:
                h.update(b"missing")
    return h.hexdigest()


@dataclass
class Gallery:
    """A built gallery: per-object, per-view records; a flat pooled index; appearance."""

    keys: List[str]                                     # objects, sorted
    views: List[str]                                    # stored views, one list for all objects
    paths: List[List[str]]                              # [object][view]
    records: List[List[Dict[str, Any]]]                 # [object][view]
    index: Any                                          # PlaceIndex over flat (object, view) entries
    appearance: np.ndarray                              # (n_flat, 76), object-major
    digest: str
    seconds: Dict[str, float] = field(default_factory=dict)
    content_kind: str = "keypoint"                      # M31: what each place carries
    #: [M34] per stored cell, that cell's `(n, 2)` fragment anchors, object-major like
    #: `flat_records()`.  None means "this gallery carries no anchors", and a bundle that fuses the
    #: fragment channel then refuses rather than scoring the channel as 0.
    fragment_anchors: Optional[List[Any]] = None

    @property
    def n(self) -> int:
        return len(self.keys)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_flat(self) -> int:
        return len(self.keys) * len(self.views)

    def flat_records(self) -> List[Dict[str, Any]]:
        """Every stored view's record, object-major: flat index = object * n_views + view."""
        return [r for per_obj in self.records for r in per_obj]

    def flat_paths(self) -> List[str]:
        return [p for per_obj in self.paths for p in per_obj]

    def index_of(self, key: str) -> int:
        return self.keys.index(key)

    def view_path(self, obj: int, view: str) -> str:
        return self.paths[obj][self.views.index(view)]

    def __len__(self) -> int:
        return self.n


def _record_job(job: Any) -> Dict[str, Any]:
    """One image -> its place record.  ``job`` is a path or a (path, content kind) pair (M31)."""
    from matching import placement_frontend as FE
    if isinstance(job, (tuple, list)):
        path, content = job[0], job[1]
    else:
        path, content = job, None
    return FE.place_record(str(path), tag="gallery", content=content)


def _assemble(items: Sequence[Tuple[str, List[Tuple[str, str]]]],
              records: Sequence[Sequence[Dict[str, Any]]], digest: str,
              seconds: Dict[str, float], content_kind: str = "keypoint") -> Gallery:
    from matching import placement_frontdoor as FD
    keys = [k for k, _p in items]
    views = [v for v, _p in items[0][1]]
    paths = [[p for _v, p in pairs] for _k, pairs in items]
    flat = [r for per_obj in records for r in per_obj]
    flat_ids = ["%s|%s" % (k, v) for k, _p in items for v in views]
    index = FD.PlaceIndex(flat, flat_ids)
    appear = np.asarray([FD.appearance_feature(p) for p_vec in paths for p in p_vec],
                        dtype=np.float64)
    if content_kind == "colour_rings":
        # a gallery that promised colour content and does not carry it is a REFUSAL, not a zero
        lacking = []
        for (k, _p), per_view in zip(items, records):
            for v, rec in zip(views, per_view):
                if not any(p.get("content") is not None for p in rec.get("places", ())):
                    lacking.append("%s|%s" % (k, v))
        if lacking:
            from matching.placement_bundle import BundleRefusal
            raise BundleRefusal(
                "%d of %d gallery cells carry no colour content (e.g. %s) while the bundle names "
                "content.kind = colour_rings: refusing rather than scoring the channel as 0.0"
                % (len(lacking), len(items) * len(views), ", ".join(lacking[:3])))
    gal = Gallery(keys=keys, views=views, paths=paths, records=[list(r) for r in records],
                  index=index, appearance=appear, digest=digest, seconds=seconds,
                  content_kind=str(content_kind))
    # [M34] a gallery that can find its cells' fragment anchors carries them: the channel is part of
    # the frozen door now, and a gallery without them forces the door to refuse.  A file that does
    # not cover a cell leaves that cell None -- refused later, never invented as an empty list.
    try:
        from matching import placement_anchors as PA
        attach_fragment_anchors(gal, PA.load())
    except (OSError, KeyError):                       # pragma: no cover - file absent
        pass
    return gal


def attach_fragment_anchors(gallery: "Gallery", table: Dict[str, Any],
                            grouping: str = "contour_fragment") -> "Gallery":
    """[M34] Hand a built gallery its fragment anchors, from `matching.placement_anchors`.

    Kept separate from `_assemble` on purpose: the graph of a stored view is not rebuilt for a
    records-only read, and the anchors are the ONE thing a fragment needs that the record does not
    carry.  A cell with no anchors stays None -- refused by the door, never invented as [].
    """
    from matching import placement_anchors as PA
    out: List[Any] = []
    for k in gallery.keys:
        for v in gallery.views:
            try:
                out.append(PA.anchors_of(table, k, v, grouping))
            except KeyError:
                out.append(None)
    gallery.fragment_anchors = out
    return gallery


def from_records(records_path: str, objects: Sequence[Dict[str, Any]], view: str,
                 content: str = "colour_rings") -> Gallery:
    """[M31] A gallery assembled from COMMITTED place records instead of rebuilt graphs.

    The records are the door's own input (`matching/placement_records.load_cells`), so a read or a
    test over them is the same computation `build_gallery` performs after its graphs are built --
    it is the cache, not a different path.  A cell with no colour content under
    `content="colour_rings"` is refused by `_assemble`, never scored as 0.
    """
    from matching import placement_records as PR
    keys = [str(o["key"]) for o in objects]
    cells = PR.load_cells(records_path, [view])
    missing = [k for k in keys if (k, view) not in cells]
    if missing:
        from matching.placement_bundle import BundleRefusal
        raise BundleRefusal(
            "the records at %s carry no cell for %d of %d objects at %s (e.g. %s): a gallery cannot "
            "be assembled from records that do not cover it"
            % (os.path.basename(records_path), len(missing), len(keys), view, ", ".join(missing[:3])))
    items = [(k, [(view, _path_of(o, view))]) for k, o in zip(keys, objects)]
    recs = [[cells[(k, view)]] for k in keys]
    digest = "records|%s|%s|%s" % (os.path.basename(records_path), view, content)
    return _assemble(items, recs, digest, {"build": 0.0}, content_kind=str(content))


def _path_of(obj: Dict[str, Any], view: str) -> str:
    """The image path of one object's stored view, from the object table (no harness import).

    Every object row carries one full path (``ref``); the view label is the file name's trailing
    ``-<az>-<el>``, so the path for any other stored view is that substitution and nothing else.
    """
    base = str(obj.get("ref") or obj.get("path") or "")
    if not base:
        raise ValueError("object row %r carries no image path" % (obj.get("key"),))
    stem = base[:-4] if base.lower().endswith(".png") else base
    parts = stem.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError("object path %r does not end in -<az>-<el>" % (base,))
    return "%s-%s.png" % (parts[0], str(view))


def build_gallery(items: Iterable[Any], *, code_digest: str, views: Optional[Sequence[str]] = None,
                  workers: int = 2, use_cache: bool = True,
                  cache_dir: Optional[str] = None,
                  content: str = "keypoint") -> Gallery:
    """Build (or load) the gallery for ``items``.

    ``code_digest`` comes from the frozen bundle: it is what makes a cached gallery invalid the
    moment the shipped code changes.  ``views`` (optional) fixes the stored view labels and the
    order they appear in; ``workers`` parallelises the graph builds and the records are collected
    in item order, so the result never depends on worker scheduling.
    """
    import time
    t0 = time.time()
    its = _norm_items(items, views)
    ckind = str(content or "keypoint")
    digest = gallery_digest(its, code_digest, ckind)
    cdir = cache_dir or CACHE_DIR
    cpath = os.path.join(cdir, "gallery-%s.pkl" % digest)
    records: Optional[List[List[Dict[str, Any]]]] = None
    if use_cache and os.path.exists(cpath):
        try:
            with open(cpath, "rb") as fh:
                records = [list(r) for r in pickle.load(fh)["records"]]
        except Exception:
            records = None                                  # a corrupt entry is a miss
    if records is None:
        t_build = time.time()
        jobs: List[Any] = [(p, (ckind if ckind != "keypoint" else None))
                           for _k, pairs in its for _v, p in pairs]
        if workers and int(workers) > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                flat = list(ex.map(_record_job, jobs, chunksize=2))
        else:
            flat = [_record_job(p) for p in jobs]
        nv = len(its[0][1])
        records = [flat[i * nv:(i + 1) * nv] for i in range(len(its))]
        if use_cache:
            os.makedirs(cdir, exist_ok=True)
            tmp = cpath + ".tmp%d" % os.getpid()
            with open(tmp, "wb") as fh:
                pickle.dump({"items": its, "records": records, "code_digest": code_digest},
                            fh, protocol=4)
            os.replace(tmp, cpath)
        sec = time.time() - t_build
    else:
        sec = 0.0
    return _assemble(its, records, digest, {"build": sec, "total": time.time() - t0},
                     content_kind=ckind)
