"""MISSION 21 / O1 -- the sigma-free stage-1 pair table, built ONCE per split.

Why this file exists: the standardized bags do not depend on the kernel widths (the
standardization is frozen), so recomputing a similarity for every width is wasted work with
the same answer.  One pass computes, for every ordered pair of images, the per-slot overlap
statistics; every width is then a vectorised exponential away.  That is what makes a 25-point
grid over 80 images affordable, and it is why the O1 fit can afford to fit on one half of FIT
and report on the other.

The table is directional on purpose.  ``S[i, j, k]`` is the score of image ``j`` as seen by
image ``i`` (mean over i's places of the best j place), so the asymmetry the mission asks
for is a difference of two entries of the same tensor, not a second run.

Usage (as a library, from the fit and the shortlist):
    from Topology.placement_m21_terms import build
    tbl = build("FIT", per_class=5, workers=2)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
SIGMA_D = (0.20, 0.35, 0.50, 0.75, 1.00)
SIGMA_A = (0.20, 0.35, 0.50, 0.75, 1.00)
GRID = tuple((sd, sa) for sd in SIGMA_D for sa in SIGMA_A)

_G: Dict[str, Any] = {}


def _init(rows: Dict[str, Any]) -> None:
    from matching import placement_prefilter as PF
    _G["pf"] = PF
    _G["params"] = rows["params"]
    _G["paths"] = rows["paths"]
    _G["std"] = {p: PF.standardize(rows["bags"][p]["bag_v2"], rows["params"])
                 for p in rows["paths"]}


def _job(ij: Tuple[int, int]) -> Tuple[int, int, List[float]]:
    """(i, j, 2 * len(GRID) scores): i->j for every width, then j->i for every width."""
    PF = _G["pf"]
    i, j = ij
    a = _G["std"][_G["paths"][i]]
    b = _G["std"][_G["paths"][j]]
    t = PF.pair_terms(a, b)
    if t is None:
        return i, j, [float("nan")] * (2 * len(GRID))
    terms, _n = t
    rev = {"d2": terms["d2"].T.copy(), "a2": terms["a2"].T.copy(),
           "live": terms["live"].T.copy()}
    out: List[float] = []
    for sd, sa in GRID:
        s = PF.score_terms(terms, sd, sa)
        out.append(float("nan") if s is None else float(s))
    for sd, sa in GRID:
        s = PF.score_terms(rev, sd, sa)
        out.append(float("nan") if s is None else float(s))
    return i, j, out


def build(split: str, per_class: int = 5, workers: int = 2,
          force: bool = False) -> Dict[str, Any]:
    """The pair table for every image of the split (references AND queries)."""
    from concurrent.futures import ProcessPoolExecutor
    from Topology.placement_m21_fit import ensure_bags
    from matching import placement_prefilter as PF
    tag = "m21_terms_%s_pc%d.npz" % (split.lower(), per_class)
    path = os.path.join(OUT, tag)
    params = PF.load_params()
    if params is None:
        raise SystemExit("run Topology/placement_m21_fit.py --stage params first")
    import parameters as P
    from Topology import placement_m20_eth80 as H
    objs = H.objects(split)
    if per_class:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for o in objs:
            per.setdefault(o["cls"], []).append(o)
        objs = [o for c in sorted(per) for o in per[c][:int(per_class)]]
    paths = sorted({o["ref"] for o in objs} | {o["query"] for o in objs})
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=True)
        if list(z["paths"]) == paths:
            return {"paths": paths, "S": z["S"], "grid": GRID, "cached": True}
    bags = ensure_bags(split, workers, per_class=per_class)
    n = len(paths)
    S = np.full((n, n, len(GRID)), np.nan, dtype=np.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print("terms: %d pairs over %d images" % (len(pairs), n), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=({"bags": bags, "params": params,
                                        "paths": paths},)) as ex:
        for k, (i, j, vals) in enumerate(ex.map(_job, pairs, chunksize=4)):
            S[i, j, :] = vals[:len(GRID)]
            S[j, i, :] = vals[len(GRID):]
            if (k + 1) % 200 == 0:
                print("  %d/%d %.0fs" % (k + 1, len(pairs), time.time() - t0), flush=True)
    print("terms done %.0fs" % (time.time() - t0), flush=True)
    np.savez_compressed(path, paths=np.asarray(paths, dtype=object), S=S,
                        grid=np.asarray(GRID, dtype=np.float64))
    return {"paths": paths, "S": S, "grid": GRID, "cached": False}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t = build(a.split, a.per_class, a.workers, a.force)
    print("paths %d, S %s, cached %s" % (len(t["paths"]), t["S"].shape, t["cached"]))
