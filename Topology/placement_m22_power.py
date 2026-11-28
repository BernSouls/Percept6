"""MISSION 22 -- the power arithmetic the registration needs, written as code so it can be checked.

The one evaluation's primaries are PAIRED on objects (each object contributes one query to the
assembled arm and one to the appearance arm).  So the right arithmetic is McNemar's exact
binomial over the DISCORDANT objects, not a two-sample normal approximation.  For a target
difference ``delta`` and a discordance rate ``pi_d``:

    p_better = (1 + delta / pi_d) / 2          the share of discordant objects the better arm wins
    power    = P(X >= k | n_d, p_better),  X ~ Binomial(n_d, p_better)
    k        = smallest k with P(X >= k | n_d, 0.5) <= alpha / 2      (exact, two-sided)

``mde`` is the smallest delta whose power reaches the target, maximised over the discordance
rate (the best case is every object discordant, ``pi_d = 1``, which is the most informative and
therefore the most favourable bound).  A read whose MDE EXCEEDS its planning effect is not run:
that is the standing rule, and this file is how the rule gets applied instead of asserted.
"""
from __future__ import annotations

import os

import json
import math
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(ROOT, 'Topology', 'results', 'placement', 'm22_power.json')


def _pmf(n: int, p: float) -> List[float]:
    return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


def power(n_d: int, delta: float, pi_d: float = 1.0, alpha: float = 0.05) -> float:
    if n_d <= 0 or delta <= 0:
        return alpha
    p = (1.0 + delta / max(1e-9, pi_d)) / 2.0
    p = min(0.999, max(0.001, p))
    tail = _pmf(n_d, 0.5)
    k = None
    acc = 0.0
    for j in range(n_d, -1, -1):
        acc += tail[j]
        if acc <= alpha / 2.0:
            k = j
    if k is None:
        return 0.0
    return float(sum(_pmf(n_d, p)[k:]))


def mde(n_d: int, want: float = 0.8, alpha: float = 0.05) -> float:
    lo, hi = 0.0, 1.0
    if power(n_d, hi, alpha=alpha) < want:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if power(n_d, mid, alpha=alpha) >= want:
            hi = mid
        else:
            lo = mid
    return hi


def table(obj_n: int, deltas=(0.2, 0.3, 0.4, 0.5, 0.6)) -> Dict[str, object]:
    """Per-object power at a given number of objects, discordance 1.0 (best case) and 0.6."""
    rows = {}
    for pi_d in (1.0, 0.6):
        n_d = max(1, int(round(obj_n * pi_d)))
        rows[str(pi_d)] = {"n_discordant": n_d,
                           "power": {str(d): round(power(n_d, d, pi_d), 3) for d in deltas},
                           "mde": round(mde(n_d), 3)}
    rows["objects"] = obj_n
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="16,24,40,60")
    a = ap.parse_args(argv)
    out = {"alpha": 0.05, "power_target": 0.8, "test": "exact McNemar (sign) on discordant objects",
           "splits": {}}
    for n in [int(x) for x in a.objects.split(",")]:
        out["splits"][str(n)] = table(n)
        t = out["splits"][str(n)]
        print("objects %-3d | all-discordant: n_d %-3d power@0.5 %.2f MDE %.3f | "
              "60%%-discordant: n_d %-3d power@0.5 %.2f MDE %.3f"
              % (n, t["1.0"]["n_discordant"], t["1.0"]["power"]["0.5"], t["1.0"]["mde"],
                 t["0.6"]["n_discordant"], t["0.6"]["power"]["0.5"], t["0.6"]["mde"]))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
