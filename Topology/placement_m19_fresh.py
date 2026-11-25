"""MISSION 19 / O5 -- the FRESH image set, fixed by `MISSION19_PREREGISTRATION.md` §5.

The pre-registration fixed the composition BEFORE any of these graphs existed: 16 irregular
scenes at seeds **91000 + i** (a range no fit, read or report has used) built by the
unchanged Mission-9 builder, plus FOUR photographs no read has touched -- `brick` (texture),
`coins` (an object with many similar parts), `camera` (a hard-edged object) and `hubble`
(a scene with almost no texture).  The M15/M16 DEV pairs are `cell`/`grass`; M17/M18's read
set was `page`/`gravel`/`moon`/`retina`.  Everything else in the photo manifest has never
been built into a graph.

    from Topology.placement_m19_fresh import fresh_gallery, tags
"""
from __future__ import annotations

from typing import Any, List, Tuple

SEED_M19 = 91000
N_FRESH = 16

#: photographs no previous read or fit has used
FRESH_REAL_M19: Tuple[str, ...] = ("brick", "coins", "camera", "hubble")


def fresh_gallery(n: int = N_FRESH) -> List[Any]:
    import numpy as np
    from Topology import placement_families_m9 as FM
    from Topology.placement_scenes import Scene, SceneSpec
    out: List[Any] = []
    for i in range(int(n)):
        seed = SEED_M19 + i
        rng = np.random.default_rng(seed)
        sc = Scene(objects=FM._irregular_objects(rng), **FM._base_kwargs(rng))
        out.append(SceneSpec(name="m19ir_%02d" % i, family="IRREGULAR", scene=sc,
                             features=FM._features_for(sc, rng),
                             views=FM.VIEWS_IRREGULAR, notes="fresh seed %d" % seed))
    return out


def tags(n: int = N_FRESH) -> Tuple[str, ...]:
    return tuple("syn|m19ir_%02d" % i for i in range(int(n))) + tuple(
        "real|%s" % p for p in FRESH_REAL_M19)
