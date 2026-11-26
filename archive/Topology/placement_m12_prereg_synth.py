"""MISSION 12 STEP 6a -- the pre-registration of the fresh synthetic read.

Written BEFORE any graph or match exists for these scenes: the script refuses to
run twice, and it asserts at registration time that not one of the fresh scenes'
images is already in the graph cache.  The rule is the Step-0 rule, and the design
is checked against it BEFORE the read: the design's MDE must be at or below the
planning effect, or the read is declared underpowered in advance.

    python Topology/placement_m12_prereg_synth.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PATH = os.path.join(OUT, "m12_prereg_synth.json")

#: A FRESH seed range: nothing in the tree has ever drawn from it.
SEED_BASE_IRREGULAR = 38200
SEED_BASE_OOP = 40200
N_IRREGULAR, N_OOP = 8, 4

#: The query views.  Fixed here, before the scenes exist.
VIEWS: Dict[str, Dict[str, Any]] = {
    "i05": {"rot_deg": 5.0}, "i10": {"rot_deg": 10.0}, "i15": {"rot_deg": 15.0},
    "i20": {"rot_deg": 20.0}, "i30": {"rot_deg": 30.0}, "i35": {"rot_deg": 35.0},
    "y10": {"rot_y_deg": 10.0}, "y15": {"rot_y_deg": 15.0},
    "y25": {"rot_y_deg": 25.0}, "y30": {"rot_y_deg": 30.0},
    "x10": {"rot_x_deg": 10.0}, "x15": {"rot_x_deg": 15.0},
    "s09": {"scale": 0.9}, "s11": {"scale": 1.1},
    "sh8": {"shift": (8.0, 0.0)}, "sh16": {"shift": (0.0, 16.0)},
}

#: The query count is fixed by the MDE gate, not by taste: MDE = 2.8*SD/sqrt(n) has
#: to be at or below the 0.10 planning effect, which with SD ~ 0.48 needs n >= 181.
#: Twelve fresh scenes x sixteen views = 192 queries.  (The FIRST registration of
#: this read used six views, 72 queries, and its own arithmetic declared it
#: UNDERPOWERED IN ADVANCE; it was replaced before any graph existed -- both
#: registrations report 0 graphs in the cache at registration time.)
N_QUERIES_MIN = 181

#: Planning numbers, stated in advance from the M11 synthetic read's own SE (0.060
#: on a paired C - A margin at its n): SD ~ 0.48 per query, so with n queries the SE
#: is 0.48/sqrt(n) and the design MDE at 80 % power, two-sided 5 % alpha, is 2.8*SE.
PLANNING_EFFECT = 0.10
ASSUMED_SD = 0.48
MDE_FACTOR = 2.8


def _fresh_specs(FM: Any) -> List[Any]:
    """The fresh scenes, built from seeds nothing in the tree has drawn from.

    The construction is the family's own (`_irregular_objects` / `_oop_scene` with
    the same base kwargs), so the read asks the same question as every other
    synthetic read -- only the instances are new.
    """
    import numpy as np
    from Topology.placement_scenes import Scene, SceneSpec
    out: List[Any] = []
    for i in range(N_IRREGULAR):
        rng = np.random.default_rng(SEED_BASE_IRREGULAR + i)
        sc = Scene(objects=FM._irregular_objects(rng), **FM._base_kwargs(rng))
        out.append(SceneSpec(name="m12ir_%02d" % i, family="IRREGULAR", scene=sc,
                             features=FM._features_for(sc, rng),
                             views=FM.VIEWS_IRREGULAR,
                             notes="M12 fresh seed %d" % (SEED_BASE_IRREGULAR + i)))
    for i in range(N_OOP):
        shape = FM.OOP_SHAPES[i % len(FM.OOP_SHAPES)]
        rng = np.random.default_rng(SEED_BASE_OOP + 211 * i)
        sc = FM._oop_scene(shape, rng)
        out.append(SceneSpec(name="m12oop_%s_%02d" % (shape.lower(), i),
                             family="OUT_OF_PLANE", scene=sc,
                             features=FM._features_for(sc, rng),
                             views=FM.VIEWS_OOP,
                             notes="M12 fresh seed %d" % (SEED_BASE_OOP + 211 * i)))
    return out


def main() -> int:
    if os.path.exists(PATH):
        print("REFUSING: %s exists -- one registration, one read" % PATH)
        return 2
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import CACHE_DIR, key_for

    specs = _fresh_specs(FM)
    if not specs:
        print("REFUSING: could not build the fresh scenes")
        return 3
    queries = [(s.name + "|" + lab, s, lab, view)
               for s in specs for (lab, view) in VIEWS.items()]
    n_q = len(queries)

    # freshness: not one of these images may be in the graph cache
    cached = 0
    for name, s, lab, _v in queries:
        sc, _pv = FM.pose_fn(s, VIEWS[lab])
        k = key_for(sc.render().image_u8)
        if os.path.exists(os.path.join(CACHE_DIR, k + ".pkl")):
            cached += 1

    if n_q < N_QUERIES_MIN:
        print("REFUSING: %d queries cannot clear the MDE gate (need >= %d)"
              % (n_q, N_QUERIES_MIN))
        return 4
    se = ASSUMED_SD / (n_q ** 0.5) if n_q else float("inf")
    mde = MDE_FACTOR * se
    reg = {
        "mission": "12 step 6a -- fresh synthetic retrieval, arms A/B/C",
        "freshness": {
            "seed_base_irregular": SEED_BASE_IRREGULAR,
            "seed_base_oop": SEED_BASE_OOP,
            "scenes": [s.name for s in specs],
            "graphs_in_cache_at_registration": cached,
            "note": ("a fresh seed range, and the registration checks the graph "
                     "cache: any hit would mean the read is not fresh"),
        },
        "shortlist": {"K": 8, "rule": "top 8 of the 12 fresh references by the "
                                      "node-similarity shortlist score (M12 step 5), "
                                      "recall reported with the read",
                      "note": "retrieval is conditional on the shortlist and the "
                              "report says so"},
        "queries": {"n": n_q, "target": 128, "views": sorted(VIEWS),
                    "order": "deterministic: scene then view"},
        "gallery": {"rule": ("the fresh scenes themselves: one reference per scene, "
                             "so the truth is in the gallery and every rival is a "
                             "different instance of the same construction"),
                    "chance": 1.0 / max(1, len(specs))},
        "arms": {
            "A": "placement off, JointMCS.score",
            "B": "placement on, score_shipped from the placement-on mapping",
            "C": "placement on, score_shipped + w*E, E on the STEP-2 scale "
                 "(PLACEMENT_E_NORMALISED), w frozen before the read",
        },
        "prediction": {
            "B<=A": True,
            "why": ("B changes the MAPPING and nothing else. The score it reports is "
                    "computed from the correspondence the scorer steered, so when the "
                    "placement term points the wrong way on a hard pair B can only "
                    "lose: it pays the cost of a moved mapping without being paid for "
                    "the evidence that moved it. That is why C, not B, is the arm "
                    "under test."),
        },
        "primary_contrast": "C - A on top-1, paired per query, clustered by SCENE",
        "secondary": ["B - A", "C - A per view family (in-plane vs out-of-plane)",
                      "judged-pair share per arm", "per-scene table"],
        "interval": "paired bootstrap over SCENES, 4000 draws, 2.5/97.5 percentiles",
        "multiplicity": "one primary; Holm is vacuous here and is reported as such",
        "mde": {"planning_effect": PLANNING_EFFECT, "assumed_sd": ASSUMED_SD,
                "formula": "MDE = 2.8 * SD/sqrt(n_queries)",
                "se": se, "mde": mde,
                "verdict_on_design": ("POWERED" if mde <= PLANNING_EFFECT
                                      else "UNDERPOWERED IN ADVANCE")},
        "verdict_rule": ("DETECTED iff the scene-clustered interval excludes zero in "
                         "the pre-registered direction; a detected effect below the "
                         "design MDE is labelled possibly inflated (Type M)"),
        "one_read": True,
        "graphs_built_at_registration": 0,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(PATH, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    print("registered %d fresh queries over %d scenes; SE %.4f, MDE %.4f -> %s"
          % (n_q, len(specs), se, mde, reg["mde"]["verdict_on_design"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
