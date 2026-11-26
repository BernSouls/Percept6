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


def gallery_digest(items: Sequence[Tuple[str, List[Tuple[str, str]]]], code_digest: str) -> str:
    """Key for the on-disk cache: item keys, view labels, image sizes, and the code digest."""
    h = hashlib.sha256()
    h.update(code_digest.encode())
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


def _record_job(path: str) -> Dict[str, Any]:
    from matching import placement_frontend as FE
    return FE.place_record(path, tag="gallery")


def _assemble(items: Sequence[Tuple[str, List[Tuple[str, str]]]],
              records: Sequence[Sequence[Dict[str, Any]]], digest: str,
              seconds: Dict[str, float]) -> Gallery:
    from matching import placement_frontdoor as FD
    keys = [k for k, _p in items]
    views = [v for v, _p in items[0][1]]
    paths = [[p for _v, p in pairs] for _k, pairs in items]
    flat = [r for per_obj in records for r in per_obj]
    flat_ids = ["%s|%s" % (k, v) for k, _p in items for v in views]
    index = FD.PlaceIndex(flat, flat_ids)
    appear = np.asarray([FD.appearance_feature(p) for p_vec in paths for p in p_vec],
                        dtype=np.float64)
    return Gallery(keys=keys, views=views, paths=paths, records=[list(r) for r in records],
                   index=index, appearance=appear, digest=digest, seconds=seconds)


def build_gallery(items: Iterable[Any], *, code_digest: str, views: Optional[Sequence[str]] = None,
                  workers: int = 2, use_cache: bool = True,
                  cache_dir: Optional[str] = None) -> Gallery:
    """Build (or load) the gallery for ``items``.

    ``code_digest`` comes from the frozen bundle: it is what makes a cached gallery invalid the
    moment the shipped code changes.  ``views`` (optional) fixes the stored view labels and the
    order they appear in; ``workers`` parallelises the graph builds and the records are collected
    in item order, so the result never depends on worker scheduling.
    """
    import time
    t0 = time.time()
    its = _norm_items(items, views)
    digest = gallery_digest(its, code_digest)
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
        jobs = [p for _k, pairs in its for _v, p in pairs]
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
    return _assemble(its, records, digest, {"build": sec, "total": time.time() - t0})
