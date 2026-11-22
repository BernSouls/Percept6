"""Canonical, process-independent tie-breaking for edge selection.

WHY THIS EXISTS
---------------
Two places decide which edges survive a budget by falling back on the edge's
own node ids when the signal ties:

  * ``orchestration/binder.py::_channel_process`` caps a channel with
    ``nlargest(cap, candidates, key=lambda c: (c[2], _neg_bytes(c[3])))``,
    where ``c[3]`` is the channel's key tuple, which CONTAINS the node ids.
  * ``matching/perceptual_selection.py::edge_key`` ends its sort key with
    ``edge[0], edge[1]`` -- the two node ids.

Node ids are not a property of the scene. They are the order in which the
binder happened to mint nodes, so using them as the tie decider makes the
selected edge SET depend on build order. That is the same class of defect as
frozenset iteration being process-salted, which already corrupted one
measurement on this branch: the result looks deterministic and is not.

WHAT REPLACES IT
----------------
A canonical key built only from what the edge IS:

    (endpoint node keys ascending, edge type name)

where a node key is ``(lens type, signature bytes)`` -- the same identity
``perceptual_selection`` already uses for its primary ordering. Two edges
that tie on signal are then separated by their own content, and only edges
that are content-identical fall through to the id, which cannot happen for
distinct endpoints in practice and is retained purely to keep the key total.

Measured effect is reported in ``Topology/results/placement/part0_answers.md``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple


def node_content_key(sigs: Any, nid: int) -> Tuple[Any, ...]:
    """A node's identity WITHOUT its id: ``(lens type, signature bytes)``.

    Falls back to the id alone when the node has no signature, so the key
    stays total.
    """
    info = None
    if sigs is not None:
        try:
            info = sigs(nid) if callable(sigs) else sigs.get(nid)
        except Exception:
            info = None
    if info is None:
        return ("", b"", int(nid))
    try:
        return (str(info[0]), info[1].serialize(), int(nid))
    except Exception:
        return (str(info[0]), b"", int(nid))


def canonical_edge_key(sigs: Any, u: int, v: int, et: Any) -> Tuple[Any, ...]:
    """The canonical, order-independent tie-break key of one edge.

    ``(min endpoint key, max endpoint key, edge type name, u, v)``.  The
    endpoints are sorted so the key does not depend on which way the edge was
    stored; the trailing ids are a last resort that keeps the key total.
    """
    a = node_content_key(sigs, u)
    b = node_content_key(sigs, v)
    if b < a:
        a, b = b, a
    name = getattr(et, "name", None) or str(et)
    return (a[0], a[1], b[0], b[1], name, int(u), int(v))


def canonical_channel_key_bytes(sigs: Any, u: int, v: int) -> bytes:
    """The tie-break payload for the binder's channel cap.

    ``_channel_process`` compares with ``_neg_bytes(c[3])`` so that LOWER
    bytes win.  Feeding it the canonical endpoint keys instead of the
    channel's id-bearing key tuple removes the build-order decider while
    keeping that comparison rule intact.
    """
    from orchestration.binder import serialize_tuple, serialize_value
    a = node_content_key(sigs, u)
    b = node_content_key(sigs, v)
    if b < a:
        a, b = b, a
    return serialize_tuple([serialize_value(a[0]), serialize_value(a[1]),
                            serialize_value(b[0]), serialize_value(b[1])])
