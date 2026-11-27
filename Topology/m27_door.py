"""M27 item 4: the two cheap door variants, on cached cells, against the registration.

    python Topology/m27_door.py                 # both views, both variants, the verdict
    python Topology/m27_door.py --json out.json

Cells and rule are the ones in `MISSION27_PREREGISTRATION.md` section 4: all 80 objects, query
views -035-045 and -045-180, 80-object gallery at -000-000, FIT / DEV+FRESH reported separately,
recall@8 with chance beside it.  A variant ships only if it beats the shipped door by >= 0.075 on
BOTH FIT cells and is not below it by more than 0.025 on either confirmation cell.

Nothing here is fitted: the narrow constitution is the one registered in
`core/keypoint_classes.py`, tau is 0.5 and 0.7 (registered before the run), the neighbour weight is
0.5 and the RRF constant is the shipped 60.
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

from Topology import m26_records as REC           # noqa: E402
from Topology.m26_oracle import by_class, metrics, rrf   # noqa: E402

REF = "000-000"
VIEWS = ("035-045", "045-180")
SELECT_VIEW, CONFIRM_VIEW = VIEWS
NARROW_TAUS = (0.5, 0.7)
NEIGHBOUR_WEIGHT = 0.5
GAIN_REQUIRED = 0.075
CONFIRM_TOLERANCE = -0.025


def _split(objs: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    return ([i for i, o in enumerate(objs) if o["split"] == "FIT"],
            [i for i, o in enumerate(objs) if o["split"] != "FIT"])


def cell(objs: Sequence[Dict[str, Any]], view: str) -> Dict[str, np.ndarray]:
    """Every door variant's score matrix for one query view (one index, one appearance pass)."""
    from matching import placement_frontdoor as FD
    from matching import placement_kernel as K
    gal = [REC.load(o["key"], REF) for o in objs]
    qs = [REC.load(o["key"], view) for o in objs]
    index = FD.PlaceIndex(gal, [o["key"] for o in objs])
    t: Dict[str, float] = {}
    out: Dict[str, np.ndarray] = {}

    t0 = time.time()
    counted, _ = FD.count_scores(index, qs, n_hits=8)
    t["counted"] = time.time() - t0
    t0 = time.time()
    app = FD.appearance_scores([REC.path_of(o, view) for o in objs],
                               [REC.path_of(o, REF) for o in objs])
    t["appearance"] = time.time() - t0
    out["a_counted"] = counted
    out["c_shipped"] = rrf(counted, app)

    t0 = time.time()
    relaxed, diag = FD.relaxed_scores(index, qs, n_hits=8, neighbour_weight=NEIGHBOUR_WEIGHT)
    t["relaxed"] = time.time() - t0
    out["k_relaxed"] = relaxed
    out["l_rrf_relaxed_plus_shipped"] = rrf(relaxed, out["c_shipped"])

    # ---- item 4a: the graded kernel in the NARROW constitution
    from core import keypoint_classes as KP
    variant = KP.register("narrow", strict=False)
    try:
        for tau in NARROW_TAUS:
            t0 = time.time()
            g, gd = FD.graded_scores(index, qs, tau=float(tau))
            t["graded_narrow_tau%.1f" % tau] = time.time() - t0
            out["f_narrow_tau%.1f" % tau] = g
            out["_df_tau%.1f" % tau] = float(np.mean([d["mean_df"] for d in gd
                                                      if d.get("mean_df")]))
    finally:
        KP.unregister()
    info = {"seconds": {k: round(v, 2) for k, v in t.items()},
            "seconds_per_query": {k: round(v / max(1, len(qs)), 3) for k, v in t.items()},
            "narrow_registration": {k: (v if not isinstance(v, dict) else sorted(v))
                                    for k, v in (variant or {}).items()},
            "kernel_fallbacks": dict(K.FALLBACKS),
            "relaxed_diag_mean_objects": float(np.mean([d["objects_scored"] for d in diag]))}
    return out, info


NAMES = {"a_counted": "a counted door", "c_shipped": "c shipped door (RRF counted+appearance)",
         "k_relaxed": "k relaxed partition", "l_rrf_relaxed_plus_shipped":
         "l RRF(relaxed, shipped c)", "f_narrow_tau0.5": "f graded NARROW tau 0.5",
         "f_narrow_tau0.7": "f graded NARROW tau 0.7"}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "Topology", "results", "placement",
                                                   "m27_door.json"))
    ap.add_argument("--limit", type=int, default=0, help="first N objects (smoke test)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    objs = REC.objects()
    if a.limit:
        objs = objs[:int(a.limit)]
    fit, conf = _split(objs)
    truth = list(range(len(objs)))
    res: Dict[str, Any] = {"n": len(objs), "chance": 8.0 / len(objs), "views": {}}
    for view in VIEWS:
        mats, info = cell(objs, view)
        table: Dict[str, Any] = {}
        for name, S in mats.items():
            if name.startswith("_"):
                continue
            m = metrics(S, truth)
            m["fit"] = {k: v for k, v in metrics(S[fit], [truth[i] for i in fit]).items()}
            m["conf"] = {k: v for k, v in metrics(S[conf], [truth[i] for i in conf]).items()}
            m["by_class_fit"] = by_class(S[fit], [truth[i] for i in fit],
                                         [objs[i]["cls"] for i in fit])
            table[name] = m
        res["views"][view] = {"variants": table, "info": info}
        print("\n== view %s (n=%d, chance recall@8 %.3f) ==" % (view, len(objs), 0.1))
        for name in NAMES:
            if name in table:
                f, c = table[name]["fit"], table[name]["conf"]
                print("  %-34s FIT %.3f (%2d/%d)  conf %.3f (%2d/%d)"
                      % (NAMES[name], f["recall_s"], f["recall_hits"], f["n"],
                         c["recall_s"], c["recall_hits"], c["n"]), flush=True)

    # ---- the verdict, by the registered rule
    verdict: Dict[str, Any] = {}
    for name in ("f_narrow_tau0.5", "f_narrow_tau0.7", "k_relaxed",
                 "l_rrf_relaxed_plus_shipped"):
        up = []
        for view in VIEWS:
            fit_c = res["views"][view]["variants"]["c_shipped"]["fit"]["recall_s"]
            fit_v = res["views"][view]["variants"][name]["fit"]["recall_s"]
            up.append(fit_v - fit_c)
        confirm = []
        for view in VIEWS:
            c = res["views"][view]["variants"]["c_shipped"]["conf"]["recall_s"]
            v = res["views"][view]["variants"][name]["conf"]["recall_s"]
            confirm.append(v - c)
        ok = all(d >= GAIN_REQUIRED - 1e-12 for d in up) and all(
            d >= CONFIRM_TOLERANCE - 1e-12 for d in confirm)
        verdict[name] = {"fit_gain": [round(x, 4) for x in up],
                         "conf_delta": [round(x, 4) for x in confirm], "passes": bool(ok)}
    res["verdict"] = verdict
    res["door"] = ("frozen: counted door" if not any(v["passes"] for v in verdict.values())
                   else "one variant passes: " + ",".join(
                       k for k, v in verdict.items() if v["passes"]))
    print("\n== verdict (rule fixed in the pre-registration) ==")
    for k, v in verdict.items():
        print("  %-30s FIT gain %s  conf delta %s  -> %s"
              % (k, v["fit_gain"], v["conf_delta"], "PASS" if v["passes"] else "fail"))
    print("  door:", res["door"])
    with open(a.json, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
