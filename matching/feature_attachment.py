"""Attach registered local attributes to physical frame signatures.

This module constructs an explicit provider overlay rather than changing the
identity or hash of a pattern node. Input attribute signatures and their full
populations remain available to callers; context/event channels are excluded.
"""
from __future__ import annotations

from collections import defaultdict
from core.graded_features import NamedFeature, with_features
from matching.relation_frame import FRAME_LENS_TYPES, FRAME_LENS_EXTENSIONS

# Existing source classes and units remain authoritative. Attachment is only
# through an observed CO_LOCATED relation; no proximity-based owner is guessed.
FEATURE_ATTACHMENT_REGISTRY = {
    "Color": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 0),
    "Texture": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 0),
    "Boundary": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 0),
    "Polarity": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 2),
    "TextureBoundary": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 2),
    "BorderOwnership": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 2),
    "Surface": (FRAME_LENS_TYPES | FRAME_LENS_EXTENSIONS, 2),
}


def attach_named_features(graph, sig_lookup):
    """Return node-id → (lens, signature) overrides for measured attachments.

    Both edge orientations are accepted. Multiple measurements of the same
    name stay separate, including distinct populations with identical bins.
    Nodes without a valid, registered owner are not assigned speculatively.
    """
    infos = {n: sig_lookup(n) for n in graph.nodes}
    attached = defaultdict(list)

    # Contour content's historical summary drops mean_theta and circ_var.
    # Carry those registered measurements, including their graded populations,
    # as one named feature on the contour and on each recorded parent part.
    from core.registries import ABSENT
    from core.slots import slot_names
    from core.types import TypedSignature
    directions = {}
    for n, info in infos.items():
        if not info or info[0] != "Contour" or info[1].class_ != "CONTOUR":
            continue
        sig = info[1]
        names = slot_names("CONTOUR")
        indices = {names.index(name) for name in ("mean_theta", "circ_var") if name in names}
        if not indices or max(indices) >= len(sig.quantized):
            continue
        q = tuple(value if i in indices else ABSENT for i, value in enumerate(sig.quantized))
        # Missing population axes are skipped using the ABSENT bins; retaining
        # the source vectors avoids fabricating or re-quantising evidence.
        direction = TypedSignature(kind=sig.kind, class_=sig.class_, relation=sig.relation,
                                   quantized=q, population=sig.population)
        feature = NamedFeature("contour_direction", "Contour", direction,
                               (("relation", "CURVE_CONTAINMENT"), ("tier", 1)))
        directions[n] = feature
        attached[n].append(feature)

    for src, dst, edge_type in graph.edges:
        if (getattr(edge_type, "name", "") == "CURVE_CONTAINMENT"
                and infos.get(src) and infos[src][0] == "CurveComplex" and dst in directions):
            attached[src].append(directions[dst])
        if getattr(edge_type, "name", "") != "CO_LOCATED":
            continue
        for owner, attr in ((src, dst), (dst, src)):
            oi, ai = infos.get(owner), infos.get(attr)
            if not oi or not ai or ai[0] not in FEATURE_ATTACHMENT_REGISTRY:
                continue
            owners, tier = FEATURE_ATTACHMENT_REGISTRY[ai[0]]
            if oi[0] not in owners or ai[1].kind != "CANONICAL":
                continue
            attached[owner].append(NamedFeature(
                name=ai[0], source_lens=ai[0], signature=ai[1],
                attachment=(("relation", "CO_LOCATED"), ("tier", tier))))
    return {n: (infos[n][0], with_features(infos[n][1], features))
            for n, features in attached.items()}
