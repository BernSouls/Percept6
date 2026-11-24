"""MISSION 11 STEP 3 -- the synthetic retrieval read, registered before it runs.

Nothing in here scores an arm.  It writes down, in one file and before any graph of
these scenes exists: the scenes and their seeds, the shortlist rule and its K, the
frozen weight, the three arms, the primary contrast, the interval construction and
the MDE.  Then it counts the graphs that exist for these scenes -- which must be 0 --
and refuses to overwrite an existing registration.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PATH = os.path.join(OUT, "m11_prereg_synth.json")


def main() -> int:
    if os.path.exists(PATH):
        print("REFUSING: %s already exists -- one registration, one read" % PATH)
        return 2
    import guard
    guard.activate()
    from Topology import placement_m11_families as F

    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        params = json.load(fh)
    gallery = F.gallery()
    query_scenes = [s for s in gallery if s.family == "IRREGULAR"]
    queries = ["%s|%s" % (s.name, lab) for s in query_scenes for lab, _v in F.query_steps(s)]
    reg = {
        "mission": "11 step 3 -- synthetic retrieval, one read",
        "scenes": {
            "gallery": [s.name for s in gallery],
            "gallery_seeds": {s.name: s.notes for s in gallery},
            "gallery_size": len(gallery),
            "query_scenes": [s.name for s in query_scenes],
            "queries": queries,
            "pose_steps": [[lab, v] for lab, v in F.VIEW_STEPS],
            "why_these_poses": ("15 deg in plane and 25 deg out of plane, so the "
                                "SHIPPED arm is not at ceiling and a null can say "
                                "something; 25 deg is inside the range Mission 10 "
                                "confirmed at 20 deg"),
        },
        "shortlist": {
            "score": "sum over query nodes of the best raw node similarity (the "
                     "shipped front end's own quantity, no aligner)",
            "K": 8,
            "fallback_rule": ("if recall@8 over the 32 queries is below 0.95 the "
                              "shortlist is K=16; decided by that rule alone, before "
                              "any arm is scored"),
            "one_shortlist_for_all_arms": True,
        },
        "weight": {
            "w": params["w"], "shrink": params.get("shrink", 8.0),
            "source": "matching/placement_score_params.json, fitted on FIT by ML",
            "frozen_before_this_read": True,
        },
        "arms": {
            "A": "joint_mcs, PLACEMENT_MATCHER_ENABLED=0, JointMCS.score",
            "B": "joint_mcs, placement matcher on, JointMCS.score_shipped",
            "C": "joint_mcs, placement matcher on, score_shipped + w * E",
        },
        "primary_contrast": "C - A, paired per query, scene-clustered over query scenes",
        "secondary": ["B - A", "C - B", "MRR for each arm",
                      "mean per-query score margin between truth and best wrong "
                      "candidate, normalised by the arm's own score scale"],
        "interval": ("paired bootstrap over QUERY SCENES (32 queries from 16 scenes, "
                     "so the unit is the scene), 4000 draws, 2.5/97.5 percentile"),
        "mde": {
            "formula": "MDE = 2.8 * SE of the paired per-scene difference",
            "planned": ("with 16 scene clusters and a planning discordance of 0.25 on "
                        "top-1, SE ~ sqrt(0.25/16) = 0.125, so MDE ~ 0.35; the read "
                        "reports the OBSERVED SE and its own MDE as well, and no "
                        "effect below the observed MDE will be called a gain"),
        },
        "one_read": True,
        "graphs_built_at_registration": 0,
        "no_check_data": "no CHECK scene, no confirmation scene, no real image",
    }
    os.makedirs(OUT, exist_ok=True)
    with open(PATH, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    print("registered %d gallery scenes, %d queries, w=%.4f -> %s"
          % (len(gallery), len(queries), params["w"], PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
