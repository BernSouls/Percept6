"""Content-hash cache for built graphs -- the core home (Mission 24).

This is the cache that used to live only in ``Topology/placement_cache.py``.  The shipped path
needs it (a graph costs ~13 s to build), and core may not import from ``Topology/``, so the code
lives here and the Topology module re-exports it.  Nothing about the KEY changed, so a cache entry
built by a Mission-20/21/22 harness is still valid here and vice versa -- the key is the pixels,
the front-end switch digest and the front-end source closure digest, and none of those mention the
import path.

What DID change, and only this: the cache directory.  It used to be
``<repo>/Topology/results/placement/_graphcache``; it is now ``<repo>/.cache/graphcache`` because a
graph cache is working state, not a result, and Mission 24 archives result directories.
``PLACEMENT_GRAPH_CACHE`` overrides the directory; there is no other configuration and no
hard-coded home directory anywhere in this file.

Determinism: an entry is the pickled graph, restored exactly.  A fresh process therefore returns
the same object or rebuilds it from the same bytes -- either way the graph is bit-identical.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, Optional

#: The repository root, from this file's own location.  No absolute path is written down.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Where the cached graphs live.  Working state: `.cache/` is gitignored.
CACHE_DIR = os.environ.get("PLACEMENT_GRAPH_CACHE") or os.path.join(ROOT, ".cache", "graphcache")


def _switch_digest() -> str:
    """The front end's own settings digest (covers every placement flag)."""
    import eth80_frontend as F
    fn = getattr(F, "switch_suffix", None)
    if fn is None:
        return "no-switch-digest"
    try:
        return str(fn())
    except Exception as exc:                                  # pragma: no cover - defensive
        return "switch-error:%r" % (exc,)


def _source_digest() -> str:
    """The front-end source closure digest (see ``eth80_frontend``)."""
    import eth80_frontend as F
    fn = getattr(F, "frontend_source_digest", None)
    if fn is None:
        return "no-source-digest"
    try:
        return str(fn())
    except Exception as exc:                                  # pragma: no cover - defensive
        return "source-error:%r" % (exc,)


def key_for(image_u8: Any) -> str:
    """Pixels + switch values + the source tree the graph was built from.

    The switch digest covers parameter VALUES, never the source that reads them, and the source
    digest covers the import closure of the build path.  Together they make the entry and the code
    that produced it move together (the full history is in the Mission-5 report, ``[0d]``).
    """
    import numpy as np
    a = np.ascontiguousarray(image_u8)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    h.update(_switch_digest().encode())
    h.update(_source_digest().encode())
    return h.hexdigest()


def get_or_build(image_u8: Any, tag: str = "", **stats: Dict[str, Any]) -> Any:
    """Return the graph for these exact pixels, building it at most once."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    k = key_for(image_u8)
    path = os.path.join(CACHE_DIR, k + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                g = pickle.load(fh)
            stats.setdefault("cache", []).append(tag)         # type: ignore[attr-defined]
            return g
        except Exception:
            pass                                              # a corrupt entry is a miss
    from eth80_frontend import build_graph
    g = build_graph(image_u8, tag or k[:12])
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as fh:
        pickle.dump(g, fh, protocol=4)
    os.replace(tmp, path)
    return g


def cache_dir() -> str:
    """Where entries are being written (read the module attribute, not a copy of it)."""
    return CACHE_DIR
