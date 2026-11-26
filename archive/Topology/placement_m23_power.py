"""MISSION 23 / item 5 -- the power arithmetic for the one confirmatory read, before it is run.

The read: 80 ETH-80 objects, three unseen view pairs each (240 queries), paired per query, and
clustered by object, tested with a paired bootstrap over the 80 objects (10,000 resamples, seed
20261118).  The question is a paired binary one -- "is the assembled system's top-1 higher than
the deployed matcher's" -- so the arithmetic is not a two-sample normal approximation and not the
exact sign test of Mission 22 either: three queries share an object, so the unit of resampling is
the object, and the quantity tested is the mean over objects of the per-object hit-rate
difference.

The simulation model, stated so it can be argued with:

    for each object o:   u_o ~ N(0, 1)                      the object's difficulty, shared by its
                                                            three queries (this is the clustering)
    for query i, arm a:  x_ia = u_o * w + e_ia,  e_ia ~ N(0, 1)
                         correct  <=>  x_ia > z_{1 - p_a}

``w`` is how much of the outcome is the object rather than the query.  w = 0 is independent
queries (the most favourable case, and the one a naive per-query power calculation assumes);
w = 1 makes an object's three queries exchangeable with a shared difficulty (the pessimistic
case, and the one that matches "one photograph of this object is easy, another is not").

``mde`` is the smallest true difference in top-1 whose bootstrap CI excludes zero with 80 %
probability, at alpha 0.05 two-sided.  The registered planning effect is 0.375 (assembled 0.550
against deployed 0.175 on FIT).  The read runs only if the MDE is below it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

OUT = "/home/user/Percept6/Topology/results/placement"
Z95 = 1.959963984540054


def _simulate(p_a: float, p_d: float, w: float, n_obj: int, per_obj: int,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """(n_obj, per_obj) boolean hit matrices for the assembled and the deployed arm."""
    u = rng.standard_normal(n_obj)[:, None]
    out = []
    for p in (p_a, p_d):
        z = -np.sqrt(1.0 + w * w) * _norm_ppf(1.0 - p)      # keep the marginal rate at p
        x = u * w + rng.standard_normal((n_obj, per_obj))
        out.append(x > z)
    return out[0], out[1]


def _norm_ppf(q: float) -> float:
    from math import erf, sqrt
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _boot_p(a: np.ndarray, d: np.ndarray, n_boot: int, seed: int) -> float:
    """Two-sided bootstrap p for the mean per-object difference (percentile, 10k resamples)."""
    diff = a.mean(axis=1) - d.mean(axis=1)
    n = diff.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    obs = float(diff.mean())
    # p = 2 x the share of resamples on the other side of zero (never 0, so it can be compared)
    p = 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    if obs == 0.0:
        return 1.0
    return max(p, 1.0 / n_boot)


def power_at(delta: float, w: float, n_obj: int = 80, per_obj: int = 3,
             p_d: float = 0.175, reps: int = 200, n_boot: int = 400,
             seed: int = 20261118) -> float:
    """Share of simulated datasets in which the paired bootstrap rejects zero."""
    rng = np.random.default_rng(seed + int(round(delta * 1e4)) + int(w * 100))
    hits = 0
    for r in range(reps):
        a, d = _simulate(p_d + delta, p_d, w, n_obj, per_obj, rng)
        if _boot_p(a, d, n_boot, seed + r) < 0.05:
            hits += 1
    return hits / reps


def mde(w: float, lo: float = 0.02, hi: float = 0.60, **kw) -> float:
    if power_at(lo, w, **kw) >= 0.8:
        return float(lo)
    if power_at(hi, w, **kw) < 0.8:
        return float("nan")
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if power_at(mid, w, **kw) >= 0.8:
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args(argv)
    res: Dict[str, Any] = {"test": "paired bootstrap over objects, 10,000 resamples at read time",
                           "alpha": 0.05, "power_target": 0.8, "n_objects": 80, "per_object": 3,
                           "planning_effect": 0.375, "p_deployed": 0.175, "cells": {}}
    print("%-8s %-10s %-10s %s" % ("w", "MDE", "< 0.375 ?", "power at 0.375"))
    for w in (0.0, 0.5, 1.0):
        m = mde(w, reps=a.reps, n_boot=a.n_boot)
        p375 = power_at(0.375, w, reps=a.reps, n_boot=a.n_boot)
        res["cells"]["w=%.1f" % w] = {"mde": m, "power_at_planning": p375}
        print("%-8.1f %-10s %-10s %.2f" % (w, m, "yes" if (m == m and m < 0.375) else "NO", p375))
        # a per-object-count curve for the pessimistic cell, so a smaller read can be sized
    # how a smaller read behaves: the same 80 objects with fewer queries each (a view dropped)
    res["per_object_curve"] = {}
    for per_obj in (1, 2, 3):
        row = {}
        for w in (0.5, 1.0):
            row["w=%.1f" % w] = mde(w, reps=max(60, a.reps // 2), n_boot=a.n_boot, per_obj=per_obj)
        res["per_object_curve"][str(per_obj)] = row
        print("per_object %d (n_objects 80): MDE %s" % (per_obj, row))
    res["n_curve"] = {}
    for n_obj in (40, 60, 80):
        m = mde(1.0, reps=max(60, a.reps // 2), n_boot=a.n_boot, n_obj=n_obj)
        res["n_curve"][str(n_obj)] = m
        print("n_objects %-4d MDE(w=1) %s" % (n_obj, m))
    with open(os.path.join(OUT, "m23_power.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", os.path.join(OUT, "m23_power.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
