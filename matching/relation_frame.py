"""matching/relation_frame.py -- the FRAME / ATTRIBUTE partition of a graph.

[STAGE 5 / C1]  A graph is read in two parts.

  FRAME      nodes that are pieces of SHAPE: a contour, a corner, a junction,
             an oriented edge element, a curvature peak.  Their relations
             join two different places (continuity, co-circularity, a
             junction and its arms, a chain and its peaks).  In the design
             this partition was written for, correspondence would be
             decided here, by relations.

  ATTRIBUTE  nodes that say what a place IS LIKE: its colour, its texture,
             its position, its boundary statistics, its polarity.  In that
             design they would attach to the frame after it is matched and
             never decide a correspondence.

WHY AN EXPLICIT LIST AND NOT `source_layer`.  `Detection.source_layer` is the
tier that PRODUCED a detection (0 = the V1 floor, 2 = a higher lens), not the
kind of thing it describes, and the two cut across each other on both sides:

  * Orientation and ContourFragment are tier 0 and are shape: Orientation
    carries the ORIENTATION_CONTINUITY / BEND / JUNCTION relations, which are
    the densest structural relations in a graph.
  * Polarity, Surface, TextureBoundary and BorderOwnership are higher-tier
    lenses and are attributes: a contrast sign, a surface's colour and
    texture, a texture step, a side assignment.

Splitting on the tier would put the first group among the attributes and the
second in the frame.  The list below is therefore written out, by lens type,
and pinned (tests_matcher/test_relation_frame_partition.py): adding a lens
type to the front end reddens the pin until someone decides which side it is
on.  A lens type in neither list is treated as ATTRIBUTE -- it can then never
drive growth, which is the safe side to be wrong on.

Nothing here reads a switch or a matcher; the module is a declaration plus
three small readers.

**HISTORY (corrected 2026-11-12).** This module used to carry the sentence
"NO RELATION-GROWN MATCHER EXISTS: the census fired its pre-registered stop
rule (the reachable set is 16 % of the frame at the median, against a floor of
25 %)".  That was true of the census of stage 5, which measured the reach of
the relations available AT THAT TIME.  It is no longer true of the tree: a
relation-grown matcher exists -- ``matching/relation_walker.py`` -- and it
grows along exactly this partition (only FRAME nodes are grown; ATTRIBUTE
nodes are attached afterwards through CO_LOCATED edges).  The stage-5 census
remains the record of that measurement; what it gated (building the matcher on
the stage-5 relations) was superseded by the stage-6 relations, and the
matcher's own measurement is in ``diagnostics/results/phase6_walker_synthetic.txt``.

The frame declaration itself is unchanged and still pinned by
``tests/matcher/test_relation_frame_partition.py``.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Tuple

from core.schema import (
    EDGE_IDENTITY_SLOTS,
    EDGE_UNIT_TABLE,
    UNKNOWN,
    U30_EDGE_IDENTITY_SLOTS,
    U30_EDGE_UNIT_TABLE,
    EdgeParamsMixture,
    _is_u30_edge_params,
)

#: lens types whose nodes are pieces of shape
FRAME_LENS_TYPES: FrozenSet[str] = frozenset({
    "Contour",            # a chain of rim elements, with its curvature profile
    "ContourFragment",    # a curved fragment: curvature, tangent, radius
    "Corner",             # a turn between two arms
    "Junction",           # a meeting of arms (also the L / T / Y aliases)
    "T",
    "Y",
    "CurveComplex",       # a bundle of contours and junctions
    "Orientation",        # an oriented edge element
    "CONTOUR_PEAK",       # a peak of the contour field
    "PEAK_CHAIN",         # a chain of field peaks
    "EndStopped",         # a line end
    "IllusoryContour",    # a completed contour between two inducers
    "RegionShape",        # the outline shape of a region
})

#: lens types whose nodes say what a place is like
ATTRIBUTE_LENS_TYPES: FrozenSet[str] = frozenset({
    "Color",
    "Texture",
    "Position",
    "Boundary",
    "Polarity",
    "TextureBoundary",    # a texture step: an attribute contrast, not a shape
    "BorderOwnership",    # which side owns a border
    "Surface",            # a surface patch's colour / texture summary
    "Onset",              # a temporal event
    # [PHASE 2] THE REGISTRATION GAP.  These four lens types are
    # DECLARED by the front end (lenses/base.py::_LENS_NAMES) but were
    # in NEITHER partition, so nothing said what they are: they were
    # "not frame" by the tree's own rule AND unclassified by the
    # attribute set.  MEASURED: they emit 0 nodes on the four ETH-80
    # frames (the gap is latent, not active), but a frame where they
    # fire would produce unclassified nodes.
    #
    # THE CHOICE (stated, because it is a judgement): each is a
    # descriptor of a PLACE, not a piece of shape, and each is emitted
    # PER ANCHOR (the field tiling).  OrientationPol is the polar axis
    # of a region blob -- the same substrate as Size -- and it tiles
    # the field, so registering it as FRAME would put thousands of
    # nodes into the correspondence's domain.  If it is ever promoted
    # to a frame primitive it must first stop tiling the field.
    "Size",               # a region's radius and elongation
    "Motion",             # a temporal motion descriptor
    "EffectPresent",      # an event marker
    "OrientationPol",     # a region blob's polar axis
})

#: types that are NOT perceptual lens types at all: a motor concept
#: that leaks into the pattern-node space.  They must never enter a
#: correspondence, so they are named explicitly rather than left to
#: fall through "not frame" by accident.
NON_LENS_TYPES: FrozenSet[str] = frozenset({
    "Action",
})


# The U20 literal remains pinned; new physical kinds are explicit extensions.
from core.junction_families import JUNCTION_FAMILY_EXTENSIONS
FRAME_LENS_EXTENSIONS = frozenset(JUNCTION_FAMILY_EXTENSIONS)


def is_frame(lens_type: str) -> bool:
    """True for the historical frame or an explicitly registered extension."""
    return lens_type in FRAME_LENS_TYPES or lens_type in FRAME_LENS_EXTENSIONS


def _axis_sets(ep: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The (identity, lookup-key) slot indices of one edge's parameters.

    The two sets are kept apart because they exclude different things: a
    graded axis is neither an identity (NAP) slot nor a lookup key, while a
    CATEGORICAL axis is an identity slot -- and a lookup key that is NOT an
    identity slot is the class's bare membership index, which is neither.
    """
    if ep is None:
        return (), ()
    if isinstance(ep, EdgeParamsMixture):
        if not ep.protos:
            return (), ()
        ep = ep.dominant
    quantized = getattr(ep, "quantized", None)
    if not quantized:
        return (), ()
    u30 = _is_u30_edge_params(ep)
    if u30:
        categorical = set(U30_EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
        specs = U30_EDGE_UNIT_TABLE.get(ep.class_, ())
    else:
        ids = getattr(ep, "identity_slots", None)
        categorical = set(ids if ids is not None
                          else EDGE_IDENTITY_SLOTS.get(ep.class_, ()))
        specs = EDGE_UNIT_TABLE.get(ep.class_, ())
    lookup = set()
    if len(specs) == len(quantized):
        lookup = {i for i, s in enumerate(specs)
                  if s.quantiser == "lookup_key"}
    return tuple(sorted(categorical)), tuple(sorted(lookup))


def measured_axes(ep: Any) -> Tuple[int, ...]:
    """Slot indices of ONE edge's parameters that hold a graded measurement.

    A slot counts when it is not UNKNOWN, not an identity (NAP) slot and not
    a lookup key: the same three exclusions `core.schema.edge_graded_axes`
    applies to a pair, applied to one side.  A mixture is read through its
    DOMINANT prototype (`EdgeParamsMixture.dominant`: the heaviest, ties to
    the earliest).
    """
    if ep is None:
        return ()
    if isinstance(ep, EdgeParamsMixture):
        if not ep.protos:
            return ()
        ep = ep.dominant
    quantized = getattr(ep, "quantized", None)
    if not quantized:
        return ()
    categorical, lookup = _axis_sets(ep)
    excluded = set(categorical) | set(lookup)
    return tuple(
        i for i, v in enumerate(quantized)
        if v != UNKNOWN and i not in excluded)


def categorical_axes(ep: Any) -> Tuple[int, ...]:
    """Slot indices that hold a measured CATEGORICAL (lookup) value.

    Three-way evidence: graded axes are physics, categorical axes are
    organization when they agree, and a bare membership flag is neither.

    [FIX] A BARE MEMBERSHIP FLAG IS NOT EVIDENCE.  A lookup-key slot that is
    NOT also an identity (NAP) slot is the class's index -- the literal
    containment = 1.0 of EDGE_CONTOUR_ORIENTATION, EDGE_REGION_CONTENT and
    EDGE_JUNCTION_ORIENTATION.  It carries the same value on every edge of
    the class, so agreeing on it says nothing about the two relations: it is
    a count, not organization.  A lookup key that IS an identity slot (the
    topology_class of a junction, the side of a containment, the relation of
    an occlusion) discriminates, and it counts -- which is the rule
    `core.schema.edge_categorical_axes` follows for a PAIR.
    """
    if ep is None:
        return ()
    if isinstance(ep, EdgeParamsMixture):
        if not ep.protos:
            return ()
        ep = ep.dominant
    quantized = getattr(ep, "quantized", None)
    if not quantized:
        return ()
    identity, _lookup = _axis_sets(ep)
    identity_set = set(identity)
    return tuple(i for i, v in enumerate(quantized)
                 if v != UNKNOWN and i in identity_set)


def evidence_axes(ep: Any) -> Tuple[int, ...]:
    """All measured axes that can carry evidence: graded + categorical."""
    return tuple(sorted(set(measured_axes(ep)) | set(categorical_axes(ep))))


def has_measured_axis(ep: Any) -> bool:
    """True when the edge carries graded OR categorical evidence."""
    return bool(evidence_axes(ep))
