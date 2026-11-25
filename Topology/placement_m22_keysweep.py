"""MISSION 22 / (A) diagnosis -- which content key can vote?

The shipped key `(lens, quantised signature bins)` scores **1.000 top-1 on a SELF query** (an
image against itself) and **at chance across views**: the quantised bins are measured in the
image frame, so any pose change moves them and the key pairs the wrong places.  This script
measures the candidate keys on the SAME cached place records (no graphs rebuilt) and reports,
per key: its bucket statistics, the recall@8 / top-1 of the vote, and the VOTE PEAKEDNESS --
how much of the winning object's vote sits in its best bin.  Peakiness is the direct,
label-free sign that correspondences are real: a correct pairing votes the same transform
again and again, a wrong one scatters.

    for each key:   index  -> vote (n_hits) -> recall@8, top-1, peak share
"""
from __future__ import annotations

import math
import pickle
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

sys.path[:0] = ["/home/user/Percept6", "/home/user/Percept6/Topology"]
OUT = "/home/user/Percept6/Topology/results/placement"
BINS = {"xy": 1.0, "ls": 0.5, "th": math.pi / 6.0}


def _mz(row) -> np.ndarray:
    return np.asarray(row["Mz"], dtype=np.float64)


def _dz(row) -> np.ndarray:
    return np.asarray(row["Dz"], dtype=np.float64)


def _bands(row, n_slots: int, edges: Sequence[float]) -> Tuple[int, ...]:
    """How many of the place's first n_slots live neighbours fall in each distance band."""
    m = _mz(row)[:n_slots]
    d = _dz(row)[:n_slots]
    out = []
    for lo, hi in zip([-1e9] + list(edges), list(edges) + [1e9]):
        out.append(int(np.sum((m > 0) & (d >= lo) & (d < hi))))
    return tuple(out)


def key_shipped(r):
    return r["key"]


def key_head(r, n):
    k = r["key"][1]
    return (r["lens"], k[:n])


def key_radial(r, n_slots=12, edges=(-0.5, 0.5)):
    return (r["lens"], _bands(r, n_slots, edges))


def key_radial4(r, n_slots=12, edges=(-1.0, 0.0, 1.0)):
    return (r["lens"], _bands(r, n_slots, edges))


def key_sortedD(r, n=4, step=1.0):
    m = _mz(r)
    d = _dz(r)
    live = sorted(float(x) for x, k in zip(d, m) if k > 0)[:n]
    return (r["lens"], tuple(int(round(x / step)) for x in live))


def key_lens(r):
    return (r["lens"],)


def key_radial_nolens(r):
    return _bands(r, 12, (-0.5, 0.5))


#: The keys swept.  ``lens only`` is DEFINED above but NOT RUN: on this gallery one bucket holds
#: every place of a lens (~20k places), so one query costs minutes and the key cannot discriminate
#: content at all by construction.  It is recorded here as constructed-but-not-run rather than
#: dropped silently, and the report says so.
KEYS: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = [
    ("shipped bins", key_shipped),
    ("bins[:8]", lambda r: key_head(r, 8)),
    ("bins[:4]", lambda r: key_head(r, 4)),
    ("radial 3x12", key_radial),
    ("radial 4x12", key_radial4),
    ("sortedD4", key_sortedD),
    ("radial no-lens", key_radial_nolens),
]


def build(entries: Sequence[Dict[str, Any]], keyfn) -> Dict[Any, List[Tuple[int, Dict[str, Any]]]]:
    by: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, e in enumerate(entries):
        for r in e["places"]:
            if int(np.count_nonzero(_mz(r))) < 2:
                continue
            by.setdefault(keyfn(r), []).append((i, r))
    return by


def vote(by, n_places, query_entry, spacing, n_hits, keyfn):
    xy_bin = BINS["xy"] * max(1e-9, float(spacing))
    no_rot: Dict[Tuple[int, int, int, int], int] = {}
    rot: Dict[Tuple[int, int, int, int, int], int] = {}
    used = 0
    n_votes = 0
    for q in query_entry["places"]:
        if int(np.count_nonzero(_mz(q))) < 2:
            continue
        pool = by.get(keyfn(q))
        if not pool:
            continue
        used += 1
        qd, qm = _dz(q), _mz(q)
        best = []
        for (obj, r) in pool:
            rd, rm = _dz(r), _mz(r)
            mm = qm * rm
            cnt = float(mm.sum())
            if cnt < 2:
                continue
            dist = float((mm * (qd - rd) ** 2).sum()) / cnt
            best.append((dist, obj, r))
        best.sort(key=lambda t: (t[0], t[1], t[2]["n"]))
        for _d, obj, r in best[:n_hits]:
            n_votes += 1
            dx = (r["x"] - q["x"]) / xy_bin
            dy = (r["y"] - q["y"]) / xy_bin
            ls = math.log2(max(1e-9, r["s"]) / max(1e-9, q["s"])) / BINS["ls"]
            k3 = (obj, int(round(dx)), int(round(dy)), int(round(ls)))
            no_rot[k3] = no_rot.get(k3, 0) + 1
            if q.get("theta") is not None and r.get("theta") is not None:
                dth = ((float(q["theta"]) - float(r["theta"])) % math.pi) / BINS["th"]
                nb = int(round(math.pi / BINS["th"]))
                k4 = k3 + (int(round(dth)) % nb,)
                rot[k4] = rot.get(k4, 0) + 1
    best_bin: Dict[int, int] = {}
    for k, c in list(no_rot.items()) + list(rot.items()):
        best_bin[k[0]] = max(best_bin.get(k[0], 0), c)
    scores = np.zeros(len(n_places))
    for obj, c in best_bin.items():
        scores[obj] = c / math.sqrt(max(1, used) * max(1, n_places[obj]))
    return scores, {"used": used, "votes": n_votes, "cells": len(no_rot) + len(rot),
                    "peak": max(best_bin.values()) if best_bin else 0}


def run(store, objs, gal, keyfn, n_hits, self_query=False):
    entries = [store[p] for p in gal]
    by = build(entries, keyfn)
    n_places = [len(e["places"]) for e in entries]
    S = np.zeros((len(objs), len(gal)))
    diag = []
    for i, o in enumerate(objs):
        qe = store[o["ref"] if self_query else o["query"]]
        # the door's own length unit; `Side.spacing_median` is 0.0 on these images and a zero
        # unit makes every vote its own cell, which measures the divisor (see the report, (A))
        from matching import placement_frontdoor as FD
        s, d = vote(by, n_places, qe, FD.local_scales(qe)[1], n_hits, keyfn)
        S[i] = s
        diag.append(d)
    key_idx = {p: j for j, p in enumerate(gal)}
    ranks = [sorted(range(len(gal)), key=lambda j: (-S[i, j], j)).index(key_idx[o["ref"]]) + 1
             for i, o in enumerate(objs)]
    n = len(ranks)
    sizes = sorted(len(v) for v in by.values())
    return {"recall8": sum(1 for r in ranks if r <= 8) / n,
            "top1": sum(1 for r in ranks if r == 1) / n,
            "n": n, "buckets": len(by),
            "median_bucket": float(np.median(sizes)) if sizes else 0.0,
            "p95_bucket": float(np.percentile(sizes, 95)) if sizes else 0.0,
            "peak_share": float(np.mean([d["peak"] / max(1, d["votes"]) for d in diag])),
            "cells_per_vote": float(np.mean([d["cells"] / max(1, d["votes"]) for d in diag]))}


def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    P.PLACEMENT_M19 = True
    from Topology import placement_m22_eth80 as H
    store = pickle.load(open(H.places_path("FIT", 0), "rb"))
    objs = H.objects("FIT")
    gal = sorted({o["ref"] for o in objs})
    rows = []
    for name, fn in KEYS:
        for n_hits in (3, 8):
            # the self query ("does the key find the same image") is a sanity check, so it is
            # run once per key instead of once per hit count -- it is by far the slowest arm
            self_r = (run(store, objs, gal, fn, 3, self_query=True) if n_hits == 3
                      else dict(self_r))
            cross = run(store, objs, gal, fn, n_hits)
            rows.append({"key": name, "n_hits": n_hits, "self": self_r, "cross": cross})
            print("%-15s hits %-3d buckets %-6d med %-5.0f p95 %-6.0f | "
                  "self top1 %.3f | cross recall@8 %.3f top1 %.3f | peak %.3f cells/vote %.2f"
                  % (name, n_hits, cross["buckets"], cross["median_bucket"],
                     cross["p95_bucket"], self_r["top1"], cross["recall8"], cross["top1"],
                     cross["peak_share"], cross["cells_per_vote"]), flush=True)
    from matching import m15_manifest as MF
    MF.write_result(OUT + "/m22_keysweep.json",
                    {"split": "FIT", "n": len(objs), "gallery": len(gal), "rows": rows})
    print("wrote", OUT + "/m22_keysweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
