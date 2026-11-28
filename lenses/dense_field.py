"""[MISSION 36 / PART B3] THE DENSE-FIELD LENS.

Wraps `perception.local_difference.dense_field` as detections, exactly the way the
keypoint lens wraps its frame (`lenses/keypoints.as_detections`): each node hands
the builder its own anchor and position, so a caller that moves the detection set
into a graph gets a positioned node, and a caller that never does pays nothing.

The signature is REGISTERED (`core.registries`: `POPULATION_SLOTS["DENSE_FIELD"]`,
`POPULATION_AXES`, `CONSTITUTIONAL_RANGES`, `CIRCULAR_CYCLE_LENGTHS`), so the tree's
own kernel compares it.  An unregistered class returns 0.0 from
`core.types._similarity_gaussian`, which `tests/test_m36_dense.py` pins by asserting
the class IS registered and that corrupting a node's content moves its similarity.

THE LENS DOES NOT DECIDE WHAT AN OBJECT IS.  It thresholds a local difference
measurement and keeps its local maxima.  No region, no patch, no merge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from perception.local_difference import DENSE_CLASS, DenseField, DenseNode, dense_field

#: The lens name a dense detection carries.  The same string as the registered class.
LENS_NAME = DENSE_CLASS

#: Field identity, for the anchor: a dense anchor must never collide with a region id,
#: a keypoint anchor or a peak anchor (see `lenses/_anchors.py` for those bands).
DENSE_ANCHOR_TAG = "dense"


def anchor_of(node: DenseNode, idx: int) -> Tuple[Any, ...]:
    """A unique, hashable anchor: never an int, so it cannot be read as a region id."""
    return (DENSE_ANCHOR_TAG, int(node.scale_index), int(idx))


def signature_of(node: DenseNode) -> Any:
    """The registered signature of one dense node: bins for the key, population for the kernel."""
    from core.registries import population_from_values
    from core.types import TypedSignature
    pop = population_from_values(DENSE_CLASS, node.values)
    if not pop:
        # A layout disagreement is never mixed into a population: raise, which the
        # caller counts, rather than emit a bin-only signature that the kernel would
        # compare as if it were complete.
        raise ValueError("DENSE_FIELD has no usable population layout")
    return TypedSignature(kind="CANONICAL", class_=DENSE_CLASS,
                          relation="LOCAL_DIFFERENCE",
                          quantized=tuple(float(v) for v in node.slots),
                          population=pop)


def field_of(image: Any, **kw: Any) -> DenseField:
    """The field of one image (the perception module's own entry point)."""
    return dense_field(image, **kw)


def as_detections(image: Any, t: int = 0, field_: Optional[DenseField] = None,
                  **kw: Any) -> Tuple[List[Any], Dict[Any, Tuple[float, float, float]]]:
    """(detections, anchor -> position) for one image.

    Same contract as the keypoint lens: the position map is what makes the node
    real to a writer that looks a node's position up by its anchor.
    """
    from core.types import Detection
    f = field_ if field_ is not None else dense_field(image, **kw)
    dets: List[Any] = []
    pos: Dict[Any, Tuple[float, float, float]] = {}
    for i, n in enumerate(f.nodes):
        a = anchor_of(n, i)
        pos[a] = (float(n.x), float(n.y), 0.0)
        d = Detection(k=LENS_NAME, source_layer=0, scale=int(n.scale_index),
                      sigma=signature_of(n), alpha=a, z=float(n.activity),
                      t=int(t), rho=())
        object.__setattr__(d, "_place_pos", (float(n.x), float(n.y)))
        dets.append(d)
    return dets, pos


def count(image: Any, **kw: Any) -> int:
    """How many dense nodes an image produces (used by the coverage reports and tests)."""
    return len(dense_field(image, **kw).nodes)
