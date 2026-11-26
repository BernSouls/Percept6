"""MISSION 17 / O6 -- the FRESH image set, seeds fixed by the pre-registration.

`MISSION17_PREREGISTRATION.md` fixed the composition BEFORE any of these graphs
existed: 16 irregular scenes at seeds 71000 + i (a range never used by any fit, read
or report) plus the four photographs M15/M16 never touched.  Nothing here is tuned;
the builder is the Mission-9 one, unchanged.

    from Topology.placement_m17_fresh import fresh_gallery, FRESH_REAL
"""
from __future__ import annotations

from typing import Any, List, Tuple

SEED_M17 = 71000
N_FRESH = 16

#: photographs NOT used by M15 (which used `cell`, `grass` on the places arm)
FRESH_REAL: Tuple[str, ...] = ("page", "gravel", "moon", "retina")


def fresh_gallery(n: int = N_FRESH) -> List[Any]:
    import numpy as np
    from Topology import placement_families_m9 as FM
    from Topology.placement_scenes import Scene, SceneSpec
    out: List[Any] = []
    for i in range(int(n)):
        seed = SEED_M17 + i
        rng = np.random.default_rng(seed)
        sc = Scene(objects=FM._irregular_objects(rng), **FM._base_kwargs(rng))
        out.append(SceneSpec(name="m17ir_%02d" % i, family="IRREGULAR", scene=sc,
                             features=FM._features_for(sc, rng),
                             views=FM.VIEWS_IRREGULAR, notes="fresh seed %d" % seed))
    return out


def tags(n: int = N_FRESH) -> Tuple[str, ...]:
    return tuple("syn|m17ir_%02d" % i for i in range(int(n))) + tuple(
        "real|%s" % p for p in FRESH_REAL)
