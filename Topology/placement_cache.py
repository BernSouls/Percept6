"""Content-hash cache for built graphs.

A graph takes ~13 s to build, and the DEV loop rebuilds the same images many
times.  The cache key is the SHA-256 of

    the exact rendered pixel bytes  ++  the front-end switch digest

so a cache entry can never be served for a different image, and never across a
different parameter setting: the switch digest already covers every PLACEMENT
flag (asserted in `eth80_frontend._FRONTEND_SWITCHES`).  The caller supplies a
`tag` for nothing but human-readable provenance; two different tags with the
same pixels and switches share one entry, which is the point.

Determinism: the cache stores the pickled graph and restores it exactly.  A
fresh-process rerun therefore returns the same object or rebuilds it from the
same bytes -- either way the graph is bit-identical, which is what the mission's
determinism control asks for.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "Topology", "results", "placement", "_graphcache")


def _switch_digest() -> str:
    """The front-end's own settings digest (covers every placement flag)."""
    sys.path.insert(0, ROOT)
    import eth80_frontend as F
    fn = getattr(F, "switch_suffix", None)
    if fn is None:
        return "no-switch-digest"
    try:
        return str(fn())
    except Exception as exc:      # pragma: no cover - defensive
        return "switch-error:%r" % (exc,)


def _source_digest() -> str:
    """The front-end source closure digest (see eth80_frontend)."""
    sys.path.insert(0, ROOT)
    import eth80_frontend as F
    fn = getattr(F, "frontend_source_digest", None)
    if fn is None:
        return "no-source-digest"
    try:
        return str(fn())
    except Exception as exc:      # pragma: no cover - defensive
        return "source-error:%r" % (exc,)


def key_for(image_u8: Any) -> str:
    """Pixels + switch values + THE SOURCE TREE the graph was built from.

    [MISSION 5 / 0d] The switch digest covers parameter VALUES, never the
    source that reads them.  A change to `lenses/base.py` therefore left every
    key untouched while changing every graph -- measured: the sub-pixel Gabor
    anchor was added to a tree whose cache entries were all still live, so a
    run would have been served graphs made by code that no longer existed.
    `frontend_source_digest()` covers the import closure of the build path
    (84 files, checked by Topology/placement_cacheclosure_m5.py), so folding it
    in makes the entry and the code that produced it move together.
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
            stats.setdefault("cache", []).append(tag)  # type: ignore[attr-defined]
            return g
        except Exception:
            pass  # a corrupt entry is a cache miss, never an error
    from eth80_frontend import build_graph
    g = build_graph(image_u8, tag or k[:12])
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as fh:
        pickle.dump(g, fh, protocol=4)
    os.replace(tmp, path)
    return g
