"""M34: the FRAGMENT ANCHORS a fragment channel needs, as a committed data file.

WHY THIS FILE EXISTS AT ALL.  The door's channels are computed from the committed place
*records* -- that is what makes an 80-object read a records-only computation.  A fragment is a
group of places, and the grouping M34 registered (a place joins the fragment of its nearest
`ContourFragment` anchor) needs one thing the record does not carry: **where the contour part nodes
were**.  Those anchors come from the front end's graph.

A channel the shipped door cannot compute from the committed data is not shippable, and asking the
door to rebuild 80 graphs per read would undo the record file.  So the anchors are exported ONCE,
as data, exactly as the records are: `data/placement_anchors_m34.npz`, one `(n, 2)` pixel array per
(object, view) per grouping, keyed `"<object>__<view>__<grouping>"`.

This is the M34 report's §7 finding turned on the mission itself: the mechanism may not be left
outside the artifact the decision reads.  `matching/placement_recognise.recognise` refuses a bundle
that asks for the fragment channel and is handed no anchors, rather than silently dropping the
channel -- a silent drop would be a door that no longer computes what its bundle says.

    python -m matching.placement_anchors --export     # rebuild the npz from the front end
    python -m matching.placement_anchors --check      # the file against the cache: 0 diffs
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "placement_anchors_m34.npz")
#: the groupings whose anchors are exported.  `proximity` needs none (it is positions only).
GROUPINGS: Tuple[str, ...] = ("contour_fragment", "contour")
#: the node class each grouping takes its anchors from
ANCHOR_CLASS: Dict[str, str] = {"contour_fragment": "ContourFragment", "contour": "Contour"}


#: the two non-cell arrays: the file stems of the cells they were exported from, so the door can
#: find a cell's anchors from an image PATH (the shipped query is a path, not a key).
STEMS = "stems"
STEM_KEYS = "stem_keys"


def _is_cell(name: str) -> bool:
    return "__" in name and name.rsplit("__", 1)[-1] in GROUPINGS


def key(object_key: str, view: str, grouping: str) -> str:
    return "%s__%s__%s" % (object_key.replace("/", "-"), view, grouping)


def anchors_from_graph(graph: Any, grouping: str) -> np.ndarray:
    """`(n, 2)` pixel anchors of the grouping's node class, sorted for determinism."""
    want = ANCHOR_CLASS[grouping]
    out: List[Tuple[float, float]] = []
    for n in graph.subgraph.nodes:
        pair = graph.sigs.get(n)
        if pair and pair[0] == want:
            a = graph.subgraph.anchor_of_node.get(n)
            if a is not None and len(a) >= 2:
                out.append((float(a[0]), float(a[1])))
    return np.asarray(sorted(out), dtype=np.float64).reshape(-1, 2)


def anchors_from_image(image_path: str, grouping: str = "contour_fragment") -> np.ndarray:
    """The anchors of ONE image, computed from the front end.

    Used when the committed file does not cover the image: the door then pays one graph build rather
    than inventing an empty part list or zeroing the channel.  The export lives in
    The bulk exporter is a mission harness: core never imports one.
    """
    import matching.placement_frontend as FE
    _img, g, _s = FE.cell(str(image_path), "m34anchors")
    return anchors_from_graph(g, grouping)


def load(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """The cells only: {(object, view, grouping) key -> (n, 2) anchors}."""
    p = path or PATH
    z = np.load(p)
    return {k: z[k] for k in z.files if _is_cell(k)}


def stems(path: Optional[str] = None) -> Dict[str, str]:
    """image stem (no extension) -> "<object>|<view>"."""
    z = np.load(path or PATH)
    return {str(s): str(k) for s, k in zip(z[STEMS], z[STEM_KEYS])}


def for_path(image_path: str, grouping: str = "contour_fragment",
             path: Optional[str] = None) -> Optional[np.ndarray]:
    """The anchors of the CELL this image path names, or None if the file does not cover it.

    The shipped query is a path.  The record file covers the cells it covers; an image outside the
    exported set has no anchors, and the door REFUSES rather than inventing parts.
    """
    stem = os.path.basename(str(image_path)).rsplit(".", 1)[0]
    ref = stems(path).get(stem)
    if ref is None:
        return None
    object_key, view = ref.rsplit("|", 1)
    try:
        return anchors_of(load(path), object_key, view, grouping)
    except KeyError:
        return None


def anchors_of(table: Dict[str, np.ndarray], object_key: str, view: str,
               grouping: str = "contour_fragment") -> np.ndarray:
    k = key(object_key, view, grouping)
    if k not in table:
        raise KeyError("no %s anchors for %s at %s" % (grouping, object_key, view))
    return table[k]


def digest(path: Optional[str] = None) -> str:
    h = hashlib.sha256()
    with open(path or PATH, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit("the exporter is a mission harness script: core never imports one")
