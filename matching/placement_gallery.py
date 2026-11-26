"""The gallery: one photograph per known object, turned into the thing the door can search.

``build_gallery`` takes (key, path) items and returns a ``Gallery`` holding

* ``records`` -- one place record per item (``placement_frontend.place_record``): the content key,
  the frozen configuration row and the frame of every payload place;
* ``index`` -- the pooled door index over those places (``placement_frontdoor.PlaceIndex``);
* ``appearance`` -- the unfitted colour + gradient histograms, L2-normalised, one row per item;
* ``digest`` -- what the cache file is keyed on.

The records are the expensive part (~13 s of graph build per image) and they are cached ON DISK,
keyed on the frozen bundle's code digest, the item keys and the image files' sizes, so a gallery
built by one process is not rebuilt by the next.  The cache is working state: ``.cache/`` is
gitignored and ``PLACEMENT_GALLERY_CACHE`` moves it.

Nothing in this module reads a hard-coded path: the repository root comes from ``__file__`` and
every path in a gallery comes from its items.
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


def _norm_items(items: Iterable[Any]) -> List[Tuple[str, str]]:
    """Accept ``(key, path)`` pairs or ``{"key":..., "path":...}`` dicts, in a fixed order."""
    out: List[Tuple[str, str]] = []
    for it in items:
        if isinstance(it, dict):
            out.append((str(it["key"]), str(it["path"])))
        elif isinstance(it, (tuple, list)) and len(it) == 2:
            out.append((str(it[0]), str(it[1])))
        else:
            raise TypeError("gallery items are (key, path) pairs or {'key','path'} dicts, got %r"
                            % (type(it),))
    keys = [k for k, _p in out]
    if len(set(keys)) != len(keys):
        raise ValueError("gallery keys must be unique")
    return sorted(out, key=lambda t: t[0])


def gallery_digest(items: Sequence[Tuple[str, str]], code_digest: str) -> str:
    """Key for the on-disk cache: items, their sizes, and the frozen code digest."""
    h = hashlib.sha256()
    h.update(code_digest.encode())
    for k, p in items:
        h.update(k.encode())
        h.update(b"\0")
        h.update(os.path.basename(p).encode())
        try:
            h.update(str(os.path.getsize(p)).encode())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


@dataclass
class Gallery:
    """A built gallery: records, the pooled index and the appearance matrix."""

    keys: List[str]
    paths: List[str]
    records: List[Dict[str, Any]]
    index: Any
    appearance: np.ndarray
    digest: str
    seconds: Dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.keys)

    def index_of(self, key: str) -> int:
        return self.keys.index(key)

    def __len__(self) -> int:
        return self.n


def _record_job(path: str) -> Dict[str, Any]:
    from matching import placement_frontend as FE
    return FE.place_record(path, tag="gallery")


def _assemble(items: Sequence[Tuple[str, str]], records: Sequence[Dict[str, Any]],
              digest: str, seconds: Dict[str, float]) -> Gallery:
    from matching import placement_frontdoor as FD
    keys = [k for k, _p in items]
    paths = [p for _k, p in items]
    index = FD.PlaceIndex(records, keys)
    appear = np.asarray([FD.appearance_feature(p) for p in paths], dtype=np.float64)
    return Gallery(keys=keys, paths=paths, records=list(records), index=index,
                   appearance=appear, digest=digest, seconds=seconds)


def build_gallery(items: Iterable[Any], *, code_digest: str, workers: int = 2,
                  use_cache: bool = True, cache_dir: Optional[str] = None) -> Gallery:
    """Build (or load) the gallery for ``items``.

    ``code_digest`` comes from the frozen bundle: it is what makes a cached gallery invalid the
    moment the shipped code changes.  ``workers`` parallelises the per-image graph builds; the
    records are collected in item order, so the result does not depend on worker scheduling.
    """
    import time
    t0 = time.time()
    its = _norm_items(items)
    if not its:
        raise ValueError("build_gallery needs at least one item")
    digest = gallery_digest(its, code_digest)
    cdir = cache_dir or CACHE_DIR
    cpath = os.path.join(cdir, "gallery-%s.pkl" % digest)
    records: Optional[List[Dict[str, Any]]] = None
    if use_cache and os.path.exists(cpath):
        try:
            with open(cpath, "rb") as fh:
                records = list(pickle.load(fh)["records"])
        except Exception:
            records = None                                     # a corrupt entry is a miss
    if records is None:
        t_build = time.time()
        if workers and int(workers) > 1 and len(its) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                records = list(ex.map(_record_job, [p for _k, p in its], chunksize=2))
        else:
            records = [_record_job(p) for _k, p in its]
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
    g = _assemble(its, records, digest, {"build": sec, "total": time.time() - t0})
    return g
