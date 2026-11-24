"""MISSION 11 -- FRESH synthetic scenes for the retrieval read (new seed ranges).

The family builders are the repaired Mission-9 ones (seven-element irregular scenes
with TWO appearance-identical twin pairs, and two-body out-of-plane scenes); only
the SEEDS are new, so the scenes did not exist when any earlier fit, read or report
was made.

Seed ranges already spent:
    9925 + i        Mission 8/9 development scenes ("ir_%02d", "oop_*_%02d")
    24200 / 26200   Mission 9 confirmation scenes ("irc_", "oopc_")
    44000-ish       Mission 10 out-of-plane confirmation ("oop_m10_")
    52000 / 56000   THIS mission ("m11ir_", "m11oop_")
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

SEED_M11_IRREGULAR = 52000
SEED_M11_OOP = 56000

#: Query pose steps.  Both are INSIDE the range the aligner is deployed for
#: (Mission 9/10 read out-of-plane at 20 degrees; the DEV-5 gates only apply inside
#: 5 degrees, which this read is not claiming).  15 and 25 are chosen so the shipped
#: arm has HEADROOM on a synthetic family: at 5-12 degrees the node-only baseline is
#: already near the ceiling and a null would say nothing about the score.
VIEW_STEPS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("q_inplane15", {"rot_deg": 15.0}),
    ("q_y25", {"rot_y_deg": 25.0}),
)

N_IRREGULAR = 16
N_OOP_PER_SHAPE = 2


def fresh_irregular(n: int = N_IRREGULAR) -> List[Any]:
    from Topology import placement_families_m9 as FM
    from Topology.placement_scenes import Scene, SceneSpec
    out: List[Any] = []
    for i in range(int(n)):
        seed = SEED_M11_IRREGULAR + i
        rng = np.random.default_rng(seed)
        sc = Scene(objects=FM._irregular_objects(rng), **FM._base_kwargs(rng))
        out.append(SceneSpec(name="m11ir_%02d" % i, family="IRREGULAR", scene=sc,
                             features=FM._features_for(sc, rng),
                             views=FM.VIEWS_IRREGULAR, notes="fresh seed %d" % seed))
    return out


def fresh_oop(per_shape: int = N_OOP_PER_SHAPE) -> List[Any]:
    from Topology import placement_families_m9 as FM
    from Topology.placement_scenes import SceneSpec
    out: List[Any] = []
    for i in range(int(per_shape)):
        for fi, shape in enumerate(FM.OOP_SHAPES):
            seed = SEED_M11_OOP + 173 * fi + i
            rng = np.random.default_rng(seed)
            sc = FM._oop_scene(shape, rng)
            out.append(SceneSpec(name="m11oop_%s_%02d" % (shape.lower(), i),
                                 family="OUT_OF_PLANE", scene=sc,
                                 features=FM._features_for(sc, rng),
                                 views=FM.VIEWS_OOP,
                                 notes="fresh seed %d shape %s" % (seed, shape)))
    return out


def gallery() -> List[Any]:
    """The mixed gallery: 16 irregular (twins) + 8 two-body out-of-plane scenes."""
    return list(fresh_irregular(N_IRREGULAR)) + list(fresh_oop(N_OOP_PER_SHAPE))


def query_steps(spec: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Every gallery scene is queried at both steps (in-plane and out-of-plane)."""
    return list(VIEW_STEPS)


def pose_fn(spec: Any, view: Dict[str, Any]):
    from Topology import placement_families_m9 as FM
    return FM.pose_fn(spec, view)
