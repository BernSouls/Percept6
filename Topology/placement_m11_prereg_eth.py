"""MISSION 11 STEP 4 -- the ETH-80 read, registered before any of its graphs exist.

Writes scenes, weights, galleries, arms, primary contrast, interval and MDE to
`m11_prereg_eth.json`, counts the graphs that exist for these images (must be 0), and
refuses to overwrite an existing registration.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PATH = os.path.join(OUT, "m11_prereg_eth.json")


def main() -> int:
    if os.path.exists(PATH):
        print("REFUSING: %s exists -- one registration, one read" % PATH)
        return 2
    import guard
    guard.activate()
    from Topology import corpus
    from Topology.placement_m11_eth80 import (FRESH_VIEWS, K1, K2, N_QUERIES,
                                              catalog)
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        params = json.load(fh)
    refs, queries = catalog()
    prior = [r["query"] for r in (
        json.loads(l) for l in open(os.path.join(OUT, "m10_eth80_progress.jsonl")))]
    prior_views = sorted({p.split("|")[1] for p in prior})
    clash = [q for q in queries if q["key"] + "|" + q["view"] in set(prior)]
    reg = {
        "mission": "11 step 4 -- ETH-80 view-to-view retrieval, fresh queries",
        "corpus": "data/ETH-80-master (%d classes x 10 instances x 41 views)"
                  % len(corpus.CLASSES),
        "freshness": {
            "m10_query_views": prior_views,
            "m11_query_views": list(FRESH_VIEWS),
            "overlap_with_mission10_queries": len(clash),
            "note": "a (instance, view) pair is what Mission 10 scored; these views "
                    "had none",
        },
        "gallery": {
            "references": len(refs),
            "reference_view": list(corpus.REF_VIEWS),
            "instances": "ranks 0-7 of each class (the same eight the Mission-10 "
                         "gallery used, so the two missions ask the same question)",
            "G1": {"K": K1, "rule": "the eight instances of the query's own class",
                   "recall": "1.0 BY CONSTRUCTION (K equals the gallery)",
                   "chance": 1.0 / K1},
            "G2": {"K": K2, "rule": "top 16 of all 64 references by node-similarity "
                                    "sum on a fixed 60 x 120 node subsample",
                   "chance": "1/16 given the truth is shortlisted; recall measured"},
        },
        "queries": {"n": len(queries), "target": N_QUERIES,
                    "order": "deterministic: class, instance, view"},
        "weight": {"w": params["w"], "w_ml": params.get("w_ml"),
                   "fit": "FIT only; ML degenerate at the bound, declared tie-break "
                          "= FIT top-1, ties to the smaller weight",
                   "frozen_before_this_read": True},
        "arms": {
            "A": "placement off, JointMCS.score",
            "B": "placement on, JointMCS.score_shipped (mapping only)",
            "C": "placement on, score_shipped + w*E",
        },
        "primary_contrast": "C - A, paired per query, clustered by INSTANCE",
        "secondary": ["B - A", "C - A on the all-class gallery", "per-class C - A",
                      "coverage: share of mapped pairs judged, per arm and class"],
        "interval": "paired bootstrap over instances, 4000 draws, 2.5/97.5",
        "mde": {"formula": "MDE = 2.8 * SE of the paired per-instance difference",
                "planned": "200 queries over 64 instances; at a planning discordance "
                           "of 0.20 on top-1, SE ~ sqrt(0.20/64) = 0.056, MDE ~ 0.16"},
        "compute": {
            "measured": "~14 s per match, ~11 s per build (Mission 10 + this session)",
            "G1": "200 queries x 8 candidates x 2 arms ~ 6 h on two cores",
            "G2": "the extra candidates of a 16-way all-class shortlist ~ 7 h more",
            "staging": "G1 pairs first for all 200 queries, then the extra G2 pairs; "
                       "one row per (query, reference) so nothing is recomputed",
        },
        "one_read": True,
        "graphs_built_at_registration": 0,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(PATH, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    print("registered %d references, %d fresh queries (views %s), w=%.4f"
          % (len(refs), len(queries), ",".join(FRESH_VIEWS), params["w"]))
    print("overlap with Mission 10 queries: %d" % len(clash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
