"""MISSION 9 STEP 3 -- scene retrieval: a gallery, a shortlist, two arms.

The design fixed before any of it is run (and before the graphs exist):

  * GALLERY: 64 scenes at reference pose, from fresh seeds, all distinct.
  * QUERIES: the same scenes under one in-plane step (12 degrees) and one
    out-of-plane step (y 20 degrees).  The other 63 gallery scenes are the
    distractors; nothing is held back, because a shortlist of 8 out of 64 is
    already the question.
  * ONE SHORTLIST for BOTH arms: top 8 by cheap node similarity (no aligner).
    A shortlist per arm would let the arms differ in a way that has nothing to do
    with placement.
  * PAIR SCORE, placement arm: sum over MATCHED pairs of the blended per-pair
    score produced by the deployed matcher (raw similarity for the six context
    candidates that were not re-scored, which is what the contract says).
  * PAIR SCORE, node-only arm: sum of the matched pairs' RAW similarity -- the
    same pairs, the same definition of "sum", without the term.
  * Shortlist recall@8 is printed BEFORE any judgement; if it is below 0.95 the
    rule is to enlarge K or shrink the gallery, not to report a retrieval number.

Cost, measured on this machine: 18-25 s per rescored pair.  64 gallery builds and
128 query builds at ~16 s each.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
SEED_BASE_GALLERY = 28000
GALLERY = 64
K = 8

QUERIES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("q_inplane12", {"rot_deg": 12.0}),
    ("q_y20", {"rot_y_deg": 20.0}),
)


def gallery_specs(n: int = GALLERY) -> List[Any]:
    from Topology import placement_families_m9 as FM
    from Topology.placement_scenes import SceneSpec
    out = []
    for i in range(int(n)):
        seed = SEED_BASE_GALLERY + i
        rng = np.random.default_rng(seed)
        from perception.synthetic3d import Scene
        sc = Scene(objects=FM._irregular_objects(rng), **FM._base_kwargs(rng))
        out.append(SceneSpec(name="gal_%02d" % i, family="GALLERY", scene=sc,
                             features=FM._features_for(sc, rng), views=QUERIES,
                             notes="seed %d" % seed))
    return out


def build(n: int = GALLERY) -> None:
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    specs = gallery_specs(n)
    t0 = time.time()
    k = 0
    for spec in specs:
        get_or_build(spec.scene.render().image_u8, spec.name + "|ref")
        k += 1
        for label, view in QUERIES:
            sc, _pv = FM.pose_fn(spec, view)
            get_or_build(sc.render().image_u8, spec.name + "|" + label)
            k += 1
        print("  %-8s %4d graphs %6.0fs" % (spec.name, k, time.time() - t0),
              flush=True)
    print("GALLERY_BUILD_DONE %d graphs %.0fs" % (k, time.time() - t0))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build(int(sys.argv[sys.argv.index("--build") + 1])
              if len(sys.argv) > sys.argv.index("--build") + 1 else GALLERY)
    else:
        print(__doc__)
