"""M28 item 6: the TWO door variants over the content, fixed before any number.

Both read the SAME cached cells (`data/placement_records_m28.npz`, view -000-000 as the gallery),
both rank objects only (door-only, no matcher, no fusion).

  (i)  `counted_content`   the counted door over a QUANTISED version of the content: a place's
       content key is (lens, 6 bins) from `placement_content.quantise` -- L into 6 bins over
       [0, 100], a and b into 8 bins each over [-128, 128), per ring.  For every query place, the
       objects holding a place with the SAME key count one hit each (the shipped `n_hits` limit is
       irrelevant here: equality is exact, so a place contributes to exactly one key).  An object's
       score is its total hits over all the query's places.  This is the shipped counted door's own
       mechanism, with the key extended by the content.

  (ii) `graded_content`    per object, the MAXIMUM cosine similarity between any of the query's
       places and any of that object's places OF THE SAME LENS -- the candidate's own fixed
       similarity (`placement_content.SIMILARITY`), no threshold, no kernel, no fitting.

THE RULE, fixed before the numbers (mission item 6, same as M27):
    FIT recall@8 must beat the shipped door's by >= 0.075 at BOTH -035-045 and -045-180, and
    DEV+FRESH (the non-FIT objects) must not be below the shipped door's by more than 0.025.

The shipped door's own cells are recomputed here as a check: the counted door's shortlists must
reproduce M27's `a_counted` numbers (same records, same rule), which also proves the enrichment did
not disturb a single frozen key.

    python Topology/m28_door.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                                  # noqa: E402
from Topology.m26_oracle import metrics                                 # noqa: E402

REF = "000-000"
VIEWS = ("035-045", "045-180")
RECORDS = os.path.join(ROOT, "data", "placement_records_m28.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m28_door.json")

GAIN_REQUIRED = 0.075
CONFIRM_TOLERANCE = -0.025

#: M27's measured shipped-door cells on the same records (MISSION27_REPORT.md item 4, row
#: "shipped"): FIT / dev+fresh recall@8 for the two views, in VIEWS order.
SHIPPED_M27 = {"035-045": (0.550, 0.550), "045-180": (0.550, 0.475)}


def _split(objs: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    return ([i for i, o in enumerate(objs) if o["split"] == "FIT"],
            [i for i, o in enumerate(objs) if o["split"] != "FIT"])


_LOADED: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}


def _cells(view: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One decompression of the file, both views (a per-cell `load_cell` re-reads 32 MB each time)."""
    from matching import placement_records as PR
    if "cells" not in _LOADED:
        _LOADED["cells"] = PR.load_cells(RECORDS, [REF] + list(VIEWS))
    cells = _LOADED["cells"]
    keys = [o["key"] for o in REC.objects()]
    return [cells[(k, REF)] for k in keys], [cells[(k, view)] for k in keys]


def counted_content(gal: Sequence[Dict[str, Any]], qs: Sequence[Dict[str, Any]]) -> np.ndarray:
    from matching import placement_content as PC
    index: Dict[Tuple[str, Tuple[int, ...]], List[int]] = {}
    for oi, rec in enumerate(gal):
        for p in rec["places"]:
            c = p.get("content")
            if c is None:
                continue
            index.setdefault((p["lens"], PC.quantise(c)), []).append(oi)
    S = np.zeros((len(qs), len(gal)), dtype=np.float64)
    for qi, rec in enumerate(qs):
        for p in rec["places"]:
            c = p.get("content")
            if c is None:
                continue
            for oi in index.get((p["lens"], PC.quantise(c)), ()):
                S[qi, oi] += 1.0
    return S


def graded_content(gal: Sequence[Dict[str, Any]], qs: Sequence[Dict[str, Any]]) -> np.ndarray:
    from matching import placement_content as PC
    lanes = ("Blob", "BoundaryArc", "CornerPoint")
    S = np.zeros((len(qs), len(gal)), dtype=np.float64)
    for lane in lanes:
        G = []
        for oi, rec in enumerate(gal):
            rows = [p["content"] for p in rec["places"] if p["lens"] == lane
                    and p.get("content") is not None]
            G.append(np.asarray(rows, dtype=np.float64).reshape(-1, PC.N_VALUES))
        for qi, rec in enumerate(qs):
            rows = [p["content"] for p in rec["places"] if p["lens"] == lane
                    and p.get("content") is not None]
            if not rows:
                continue
            Q = np.asarray(rows, dtype=np.float64)
            for oi, g in enumerate(G):
                if g.shape[0] == 0:
                    continue
                S[qi, oi] = max(S[qi, oi], float(np.max(PC.similarity(Q, g))))
    return S


def counted_shipped(gal: Sequence[Dict[str, Any]], qs: Sequence[Dict[str, Any]]
                    ) -> np.ndarray:
    from matching import placement_frontdoor as FD
    from matching import placement_bundle as PB
    keys = [o["key"] for o in REC.objects()]
    n_hits = int(PB.PlacementBundle.load().door_n_hits)
    S, _ = FD.count_scores(FD.PlaceIndex(list(gal), keys), list(qs), n_hits=n_hits)
    return S


def report(S: np.ndarray, fit: Sequence[int], conf: Sequence[int],
           objs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from Topology.m26_oracle import by_class
    truth = list(range(S.shape[0]))
    cls = [o["cls"] for o in objs]
    return {"fit": metrics(S[fit], [truth[i] for i in fit]),
            "conf": metrics(S[conf], [truth[i] for i in conf]),
            "by_class_fit": by_class(S[fit], [truth[i] for i in fit],
                                     [cls[i] for i in fit])}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    objs = REC.objects()
    fit, conf = _split(objs)
    res: Dict[str, Any] = {"records": os.path.basename(RECORDS), "views": {},
                           "rule": {"gain_required_fit": GAIN_REQUIRED,
                                    "confirm_tolerance": CONFIRM_TOLERANCE,
                                    "shipped_m27": SHIPPED_M27},
                           "n": {"fit": len(fit), "conf": len(conf), "gallery": len(objs)}}
    t0 = time.time()
    for view in VIEWS:
        gal, qs = _cells(view)
        t: Dict[str, float] = {}
        v: Dict[str, Any] = {}
        for name, fn in (("a_counted_shipped", counted_shipped),
                         ("i_counted_content", counted_content),
                         ("ii_graded_content", graded_content)):
            t0v = time.time()
            S = fn(gal, qs)
            t[name] = round(time.time() - t0v, 1)
            v[name] = report(S, fit, conf, objs)
            print("  %-20s %s  FIT %.3f (%2d/%d)  conf %.3f (%2d/%d)"
                  % (view, name, v[name]["fit"]["recall_s"], v[name]["fit"]["recall_hits"],
                     v[name]["fit"]["n"], v[name]["conf"]["recall_s"],
                     v[name]["conf"]["recall_hits"], v[name]["conf"]["n"]), flush=True)
        v["seconds"] = t
        v["shipped_m27"] = {"fit": SHIPPED_M27[view][0], "conf": SHIPPED_M27[view][1]}
        v["reproduces_m27_counted"] = abs(v["a_counted_shipped"]["fit"]["recall_s"]
                                          - SHIPPED_M27[view][0]) <= 1e-9
        res["views"][view] = v
    verdict: Dict[str, Any] = {}
    for name in ("i_counted_content", "ii_graded_content"):
        gains = {view: res["views"][view][name]["fit"]["recall_s"]
                 - SHIPPED_M27[view][0] for view in VIEWS}
        drops = {view: res["views"][view][name]["conf"]["recall_s"]
                 - SHIPPED_M27[view][1] for view in VIEWS}
        ok = all(g >= GAIN_REQUIRED for g in gains.values()) and \
             all(d >= CONFIRM_TOLERANCE for d in drops.values())
        verdict[name] = {"fit_gain": gains, "conf_delta": drops, "passes": bool(ok)}
        print("  %-20s FIT gain %s  conf delta %s -> %s"
              % (name, {k: round(x, 3) for k, x in gains.items()},
                 {k: round(x, 3) for k, x in drops.items()}, "PASS" if ok else "fail"), flush=True)
    res["verdict"] = verdict
    res["any_passes"] = any(v["passes"] for v in verdict.values())
    res["seconds"] = round(time.time() - t0, 1)
    res["manifest"] = {"environment": _env(), "records_bytes": os.path.getsize(RECORDS)}
    with open(a.json, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({"any_passes": res["any_passes"], "seconds": res["seconds"]}, indent=1))
    return 0


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


if __name__ == "__main__":
    raise SystemExit(main())
