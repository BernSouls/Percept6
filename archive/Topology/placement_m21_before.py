"""MISSION 21 / O1 -- the BEFORE number, on the same gallery and the same tasks.

A change is only reportable against the number it replaces, and the number it replaces was
measured with a different ruler.  This file recomputes the Mission-20 stage-1 score
(`score_bags`: per-reference mean, std and kept dimensions) on the SAME 40-item gallery, the
SAME tasks and the SAME S = 8, so the before/after pair differs in the ruler and in nothing
else.  It is a re-measurement, not a re-quote: the numbers on disk from M20 were taken on a
16-item gallery.

Also reported here: the per-reference scale drift that motivates the change, measured as the
spread of the score level across references -- for a fixed query, the standard deviation of
the 40 gallery scores under each convention.  A ruler that depends on the reference shows up
as a wider, reference-driven spread.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="m21_before")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_prefilter as PF
    from Topology.placement_m21_fit import ensure_bags, split_objects
    from Topology.placement_m21_shortlist import score_tasks
    if not P.PLACEMENT_M19:
        raise SystemExit("set PLACEMENT_M19=1")
    objs = split_objects(a.split, a.per_class)
    bags = ensure_bags(a.split, a.workers, a.per_class)
    gal = sorted({o["ref"] for o in objs})
    M = np.zeros((len(objs), len(gal)), dtype=np.float64)
    t0 = time.time()
    for i, o in enumerate(objs):
        for j, r in enumerate(gal):
            s = PF.score_bags(bags[o["query"]]["bag"], bags[r]["bag"]).get("score")
            M[i, j] = 0.0 if s is None else float(s)
    sec = (time.time() - t0) / (len(objs) * len(gal))
    res: Dict[str, Any] = {"split": a.split, "per_class": a.per_class, "gallery": len(gal),
                           "S": 8, "ruler": "M20 score_bags (per-reference mean/std/keep)",
                           "sec_per_pair": sec, "tasks": {}}
    for task in ("T1", "T2", "T3"):
        res["tasks"][task] = score_tasks(M, objs, gal, task, 8)
    np.save(os.path.join(OUT, "m21_v1_matrix_%s_pc%d.npy" % (a.split.lower(), a.per_class)), M)
    res["matrix"] = "m21_v1_matrix_%s_pc%d.npy" % (a.split.lower(), a.per_class)
    res["level_spread_across_references"] = float(np.mean(np.std(M, axis=1)))
    res["level_mean"] = float(np.mean(M))
    print("%s: gallery %d, S=8, %.4f s/pair" % (a.split, len(gal), sec))
    print("%-6s %-8s %-6s %-8s %-8s" % ("task", "recall8", "n", "chance", "lift"))
    for task in ("T1", "T2", "T3"):
        r = res["tasks"][task]
        print("%-6s %-8.3f %-6d %-8.3f %-8.2f" % (task, r["recall"] or 0, r["n"],
                                                  r["chance"] or 0, r["lift"] or 0))
    out = os.path.join(OUT, a.tag + "_" + a.split.lower() + "_pc%d.json" % a.per_class)
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
