"""Reconstructed ``action.decode`` (export shim).

The executive action decoder was not part of this export.  Its contract in
the shipped callers is ``decode_action_sequence(G, concept_graph) ->
Optional[List[Any]]``; returning ``None`` is the documented "no decoded
sequence" outcome that makes the caller fall back to
``_extract_actions_from_subgraph``.  The image -> match-graph front end
never calls it (it runs one perception step on a fresh graph, with no
Action nodes), so this stub only exists to resolve the import graph.

Reconstructed by the front-end fix session; see work/topology/README.md.
"""
from __future__ import annotations

from typing import Any, Optional


def decode_action_sequence(G: Any, concept_graph: Any) -> Optional[Any]:
    """No decoded sequence in a fresh-graph front end; use the fallback."""
    return None
