"""The place key: one node per physical place, not one per signature class.

[MISSION 7] A pattern node is keyed by ``(lens_type, signature_hash)``, so every
detection that quantises to the same signature folds into ONE node with ONE
position (first write wins).  Measured on one control scene: Orientation 105
detections -> 32 nodes, Texture 135 -> 27, Color 110 -> 7, Onset 64 -> 1,
keypoints 405 -> 96.  A node was therefore a SIGNATURE CLASS, and "one place per
physical feature" was not representable at all.

The repair is to SALT the signature hash with the detection's own anchor -- a
region/cell tuple for the tiling lenses, ``("kp", kind, idx)`` for the keypoint
layer.  Each detection of a place lens then gets its own node, and because the
position is carried on the node (never in the signature), matching is unchanged:
the stored `TypedSignature` stays position-free.

NO GRID ROUNDING is used anywhere here.  Rounding would merge two genuinely
distinct detections that happen to land in one cell and split one that straddles
a boundary; the anchor is the thing the lens actually emitted, so it is what the
key is salted with.

Determinism: the salt is a sha256 over the anchor's canonical serialization, so
it is stable across processes and independent of dict order.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Tuple

#: Lenses minted per PLACE.  Everything else keeps the signature-class key,
#: which is correct for it: an attribute lens describes what a place is like, so
#: folding its detections is the intended behaviour.
DEFAULT_PLACE_LENSES: Tuple[str, ...] = (
    "Orientation",
    "Texture",
    "Color",
    "Contour",
    "Boundary",
    # the keypoint layer (Mission 6)
    "CornerPoint",
    "Blob",
    "BoundaryArc",
)


def place_lenses() -> Tuple[str, ...]:
    """The place lenses, from the parameter if declared, else the default."""
    try:
        import parameters as P
    except Exception:
        return DEFAULT_PLACE_LENSES
    v = getattr(P, "PLACE_LAYER_LENSES", None)
    return tuple(v) if v else DEFAULT_PLACE_LENSES


def place_layer_on() -> bool:
    """The dark flag."""
    try:
        import parameters as P
    except Exception:
        return False
    return bool(getattr(P, "FE_PLACE_LAYER", False))


def is_place_lens(lens: str) -> bool:
    return str(lens) in place_lenses()


def _anchor_bytes(anchor: Any) -> bytes:
    """Canonical bytes for an anchor, independent of dict/iteration order."""
    if isinstance(anchor, tuple):
        return b"T(" + b"|".join(_anchor_bytes(a) for a in anchor) + b")"
    if isinstance(anchor, (list,)):
        return b"L(" + b"|".join(_anchor_bytes(a) for a in anchor) + b")"
    if isinstance(anchor, bool):
        return b"b" + (b"1" if anchor else b"0")
    if isinstance(anchor, int):
        return b"i%d" % int(anchor)
    if isinstance(anchor, float):
        return b"f%.6f" % float(anchor)
    return b"s" + str(anchor).encode()


def salted_sig_hash(lens: str, sig_hash: bytes, anchor: Any) -> bytes:
    """The node key's second element, salted with the place's anchor.

    Flag off, or a non-place lens, returns `sig_hash` UNCHANGED, so the default
    graph cannot move.
    """
    if not place_layer_on() or not is_place_lens(lens):
        return sig_hash
    h = hashlib.sha256()
    h.update(b"place|")
    h.update(bytes(sig_hash))
    h.update(b"|")
    h.update(_anchor_bytes(anchor))
    return h.digest()[:16]


def node_key(lens: str, sig_hash: bytes, anchor: Any) -> Tuple[str, bytes]:
    """The key a mint site should use."""
    return (str(lens), salted_sig_hash(lens, sig_hash, anchor))


def sig_hash_of(det: Any) -> bytes:
    """The salted signature hash for a DETECTION (the one call sites should use).

    Every place that builds a `(lens, hash)` pair -- a mint, a lookup, a
    presence set, an onset table -- must agree, or a salted node gets an
    unsalted twin that nothing can find.  Measured: salting only the mint sites
    made the graph LOSE its Orientation/Texture/Color/Boundary nodes entirely
    (657 -> 496), because every downstream lookup was still unsalted.
    """
    from core.hashing import _node_signature_hash
    lens = getattr(det, "k", "")
    return salted_sig_hash(lens, _node_signature_hash(getattr(det, "sigma", None)),
                           getattr(det, "alpha", None))
