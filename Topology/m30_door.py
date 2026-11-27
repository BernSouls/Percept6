"""M30 A1: fuse colour into the shipped door as a discounted third channel.

    python Topology/m30_door.py            # both views, FIT / DEV+FRESH, per class, the verdict

Reads the committed records (no rebuild, no matcher).  The rule, stated before the numbers:

    average recall@8 gain over the two FIT views  >= +0.050      (replaces M28's conjunction)
    DEV+FRESH recall@8 not below the shipped door by more than   0.025   (per view)

Shipped door = RRF(counted, appearance), k = 60, equal weights.  The colour channel is the
per-object max same-lens cosine of the M28 content, fused at weight 0.5 against 1.0 + 1.0.
Chance is 8/80 = 0.100, n = 40 FIT and 40 DEV+FRESH per view.
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

from Topology import m26_records as REC                     # noqa: E402
from Topology.m26_oracle import by_class, metrics           # noqa: E402

REF = "000-000"
VIEWS = ("035-045", "045-180")
RECORDS = os.path.join(ROOT, "data", "placement_records_m28.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m30_door.json")

GAIN_REQUIRED = 0.050
CONFIRM_TOLERANCE = -0.025
RRF_K = 60
COLOUR_WEIGHT = 0.5

_LOADED: Dict[Tuple[str, str], Dict[str, Any]] = {}


def cells(views: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not _LOADED:
        from matching import placement_records as PR
        _LOADED.update(PR.load_cells(RECORDS, [REF] + list(views)))
    return _LOADED


def _split(objs):
    return ([i for i, o in enumerate(objs) if o["split"] == "FIT"],
            [i for i, o in enumerate(objs) if o["split"] != "FIT"])


def channels(view: str) -> Dict[str, np.ndarray]:
    """counted, appearance and colour score matrices for one query view."""
    from matching import placement_frontdoor as FD
    from matching import placement_fuse as FU
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    C = cells(VIEWS)
    gal = [C[(k, REF)] for k in keys]
    qs = [C[(k, view)] for k in keys]
    index = FD.PlaceIndex(gal, keys)
    counted, _ = FD.count_scores(index, qs, n_hits=8)
    app = FD.appearance_scores([REC.path_of(o, view) for o in objs],
                               [REC.path_of(o, REF) for o in objs])
    colour = FU.colour_channel(gal, qs)
    shipped = FU.fuse(counted, app)
    fused = FU.fuse(counted, app, colour, weights={"colour": COLOUR_WEIGHT})
    return {"counted": counted, "appearance": app, "colour": colour,
            "shipped": shipped, "fused": fused}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=OUT)
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    objs = REC.objects()
    fit, conf = _split(objs)
    cls = [o["cls"] for o in objs]
    truth = list(range(len(objs)))
    res: Dict[str, Any] = {"rule": {"gain_required_avg_fit": GAIN_REQUIRED,
                                    "confirm_tolerance": CONFIRM_TOLERANCE,
                                    "rrf_k": RRF_K, "colour_weight": COLOUR_WEIGHT,
                                    "incumbent_weights": {"counted": 1.0, "appearance": 1.0}},
                           "n": {"fit": len(fit), "conf": len(conf), "gallery": len(objs)},
                           "views": {}}
    t0 = time.time()
    for view in VIEWS:
        S = channels(view)
        v: Dict[str, Any] = {}
        for name in ("counted", "appearance", "colour", "shipped", "fused"):
            v[name] = {"fit": metrics(S[name][fit], [truth[i] for i in fit]),
                       "conf": metrics(S[name][conf], [truth[i] for i in conf])}
        v["fit_by_class_shipped"] = by_class(S["shipped"][fit], [truth[i] for i in fit],
                                            [cls[i] for i in fit])
        v["fit_by_class_fused"] = by_class(S["fused"][fit], [truth[i] for i in fit],
                                          [cls[i] for i in fit])
        # how often the fusion keeps the true object in the top 8 that the shipped door lost
        gained, lost = [], []
        for i in fit:
            rs = sorted(range(len(objs)), key=lambda j: (-S["shipped"][i, j], j)).index(i)
            rf = sorted(range(len(objs)), key=lambda j: (-S["fused"][i, j], j)).index(i)
            if rf < 8 <= rs:
                gained.append(objs[i]["key"])
            if rs < 8 <= rf:
                lost.append(objs[i]["key"])
        v["flips"] = {"gained": gained, "lost": lost}
        res["views"][view] = v
        print("  %s  shipped FIT %.3f (%2d/40) conf %.3f (%2d/40) | fused FIT %.3f (%2d/40) "
              "conf %.3f (%2d/40) | colour alone FIT %.3f"
              % (view, v["shipped"]["fit"]["recall_s"], v["shipped"]["fit"]["recall_hits"],
                 v["shipped"]["conf"]["recall_s"], v["shipped"]["conf"]["recall_hits"],
                 v["fused"]["fit"]["recall_s"], v["fused"]["fit"]["recall_hits"],
                 v["fused"]["conf"]["recall_s"], v["fused"]["conf"]["recall_hits"],
                 v["colour"]["fit"]["recall_s"]), flush=True)
        print("     top-1 shipped %.3f fused %.3f | gained %s lost %s"
              % (v["shipped"]["fit"]["top1"], v["fused"]["fit"]["top1"],
                 gained[:4], lost[:4]), flush=True)
    gains = {view: res["views"][view]["fused"]["fit"]["recall_s"]
             - res["views"][view]["shipped"]["fit"]["recall_s"] for view in VIEWS}
    deltas = {view: res["views"][view]["fused"]["conf"]["recall_s"]
              - res["views"][view]["shipped"]["conf"]["recall_s"] for view in VIEWS}
    avg = float(sum(gains.values()) / len(gains))
    res["fit_gain"] = gains
    res["conf_delta"] = deltas
    res["fit_gain_average"] = avg
    res["passes"] = bool(avg >= GAIN_REQUIRED and all(d >= CONFIRM_TOLERANCE
                                                      for d in deltas.values()))
    res["seconds"] = round(time.time() - t0, 1)
    res["manifest"] = {"environment": _env(), "records": os.path.basename(RECORDS),
                       "records_bytes": os.path.getsize(RECORDS)}
    with open(a.json, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print("\nFIT gain per view %s  average %+.3f (need >= +0.050); DEV+FRESH delta %s "
          "(floor -0.025) -> %s"
          % ({k: round(x, 3) for k, x in gains.items()}, avg,
             {k: round(x, 3) for k, x in deltas.items()},
             "PASS" if res["passes"] else "fail"))
    return 0


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


if __name__ == "__main__":
    raise SystemExit(main())
