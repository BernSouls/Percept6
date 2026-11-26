"""The graph cache, kept at this path for the harnesses that import it (Mission 24).

The implementation MOVED to ``matching/placement_cache.py``: the shipped path needs a content-hash
graph cache and a core package may not import from ``Topology/``.  This module is the old name,
re-exporting the same objects, so every ``from Topology.placement_cache import get_or_build`` in a
Mission-5..23 harness keeps working unchanged -- and, deliberately, keeps sharing ONE cache:
``get_or_build`` here IS core's, so a graph built by a legacy harness is the graph the recogniser
loads.

What changed for the harnesses: the entries now live in ``<repo>/.cache/graphcache`` rather than
``<repo>/Topology/results/placement/_graphcache``.  The KEY is unchanged (pixels + front-end switch
digest + front-end source-closure digest), so no entry is invalidated by the move -- only its
directory.  ``PLACEMENT_GRAPH_CACHE`` overrides the directory.
"""
from __future__ import annotations

from matching.placement_cache import (CACHE_DIR, cache_dir, get_or_build, key_for,  # noqa: F401
                                      _source_digest, _switch_digest)

__all__ = ["CACHE_DIR", "cache_dir", "get_or_build", "key_for"]
