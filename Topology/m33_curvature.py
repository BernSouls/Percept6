"""M33 item 2: the curvature channel at OBJECT level, as a door channel -- registered, then read.

REGISTERED BEFORE ANY NUMBER (this docstring is the registration; the file's git history is the
timestamp):

  CHANNEL   the four curvature columns only -- ``sharp_x``, ``sharp_y``, ``curv_x``, ``curv_y`` --
            read through the kernel form with the POOL's own column spread (`PS.kernel_block`), and
            aggregated at OBJECT level: for a query object and a gallery object, the MEAN over the
            query's places of the best same-lens match in that gallery object
            (`PS.curvature_matrix(how="object_mean")`).  That is the form that read G_same 0.671.
  EXCLUDED  the five detection-strength columns (resp, var_x, var_y, eigratio, rel_angle), because
            M32 measured all nine together at 0.558 same-class against the four alone at 0.596:
            they DILUTE.
  DOOR      the channel enters the fused door as a FOURTH channel at the same half-vote colour
            carries: ``rrf(counted 1.0, appearance 1.0, colour 0.5, shape 0.5)``, k = 60.  A
            channel that has never been read in the shipped path may reorder the head; it may not
            outvote the configuration.
  CELLS     gallery = all 80 objects at -000-000; queries = FIT objects at -035-045 and -045-180
            choose; DEV + FRESH confirm.  Cached records only, no matcher.
  BAR       mean FIT fused3 recall@8 gain over the CURRENT shipped door >= +0.050, object-clustered
            paired bootstrap CI (10,000 resamples, seed 20261118) EXCLUDING ZERO, and DEV+FRESH not
            below the shipped door by more than 0.025.  Up/down/tied beside every rate; ties ->
            fewer changes.

    python Topology/m33_curvature.py --channels   # cache the door channels (records only)
    python Topology/m33_curvature.py --arm        # the channel, the fused door, the bootstrap
    python Topology/m33_curvature.py --refine     # item 3, ONLY if item 2 passes
    python Topology/m33_curvature.py --gauge-ci   # the gauge gain's own CI (measurement, not a gate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                       # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m33_curvature.json")
CH = os.path.join(ROOT, "Topology", "results", "placement", "m33_channels.npz")
REF = "000-000"
VIEWS = {"fit_a": "035-045", "fit_b": "045-180"}
S = 8
K = 60
W_COLOUR = 0.5
W_SHAPE = 0.5
GAIN_REQUIRED = 0.050
CONFIRM_TOLERANCE = -0.025
RESAMPLES = 10000
SEED = 20261118


def _split() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(REC.objects()):
        out.setdefault(str(o["split"]), []).append(i)
    return out


def channels(views: Sequence[str]) -> Dict[str, np.ndarray]:
    """counted + appearance for each view (cached from M32 when the records have not moved)."""
    old = os.path.join(ROOT, "Topology", "results", "placement", "m32_colour_channels.npz")
    need = ["counted|%s" % v for v in views] + ["appearance|%s" % v for v in views]
    if os.path.exists(old):
        z = np.load(old)
        have = {k: z[k] for k in z.files}
        if all(k in have for k in need):
            print("  reusing M32's counted/appearance channels", flush=True)
            return have
    from matching import placement_frontdoor as FD
    from matching import placement_gallery as GL
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    cells = PR.load_cells(RECORDS, [REF, *views])
    out: Dict[str, np.ndarray] = {}
    for v in views:
        qs = [cells[(k, v)] for k in keys]
        counted, _i = FD.count_scores(gal.index, qs, n_hits=8)
        app = FD.appearance_scores([REC.path_of(o, v) for o in objs],
                                   [REC.path_of(o, REF) for o in objs])
        out["counted|%s" % v] = np.asarray(counted, dtype=np.float64)
        out["appearance|%s" % v] = np.asarray(app, dtype=np.float64)
    np.savez(CH, **out)
    return out


def _rank(M: np.ndarray) -> np.ndarray:
    return np.argsort(-M, axis=1, kind="stable")


def _hit(M: np.ndarray, i: int) -> bool:
    return bool(i in _rank(M)[i][:S])


def metrics(M: np.ndarray, label: str) -> Dict[str, Any]:
    n = M.shape[0]
    order = _rank(M)
    ranks = np.asarray([int(np.nonzero(order[i] == i)[0][0]) for i in range(n)])
    off = M[~np.eye(n, dtype=bool)]
    return {"label": label, "top1": float((ranks == 0).mean()),
            "recall@8": float((ranks < S).mean()), "truth_rank_median": float(np.median(ranks)),
            "share_ge_0.99": float((off >= 0.99).mean()),
            "share_ge_0.999": float((off >= 0.999).mean()),
            "truth_median": float(np.median(np.diag(M))),
            "impostor_median": float(np.median(off)),
            "gap_medians": float(np.median(np.diag(M)) - np.median(off)),
            "distinct": int(np.unique(np.round(off, 12)).size)}


def _paired(base: np.ndarray, new: np.ndarray, idx: Sequence[int]) -> Dict[str, int]:
    up = down = tie = 0
    for i in idx:
        a, b = _hit(base, i), _hit(new, i)
        if b and not a:
            up += 1
        elif a and not b:
            down += 1
        else:
            tie += 1
    return {"up": up, "down": down, "tied": tie}


def _boot_ci(base: np.ndarray, new: np.ndarray, idx: Sequence[int],
             resamples: int = RESAMPLES, seed: int = SEED) -> Dict[str, Any]:
    """Object-clustered paired bootstrap of the recall@8 difference (resample OBJECTS)."""
    idx = list(idx)
    d = np.asarray([float(_hit(new, i)) - float(_hit(base, i)) for i in idx])
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(idx), size=(int(resamples), len(idx)))
    means = d[draw].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point": float(d.mean()), "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0), "resamples": int(resamples),
            "seed": int(seed)}


def arm() -> Dict[str, Any]:
    from matching import placement_fuse as FU
    from matching import placement_records as PR
    from matching import placement_shape as PS
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    split = _split()
    ch = channels(list(VIEWS.values()))
    cells = PR.load_cells(RECORDS, [REF, *VIEWS.values()])
    gal_entries = [cells[(k, REF)] for k in keys]
    res: Dict[str, Any] = {"registration": {"channel": list(PS.CURVATURE_FIELDS),
                                            "aggregation": "object_mean", "w_shape": W_SHAPE,
                                            "w_colour": W_COLOUR, "k": K},
                           "split": {k: len(v) for k, v in split.items()}, "cells": {}}
    for cell, view in VIEWS.items():
        counted = ch["counted|%s" % view]
        app = ch["appearance|%s" % view]
        q = [cells[(k, view)] for k in keys]
        colour = np.asarray(FU.colour_channel(gal_entries, q), dtype=np.float64)
        shape = PS.curvature_matrix(gal_entries, q, how="object_mean")
        base = np.asarray(FU.fuse(counted, app, colour, weights={"colour": W_COLOUR}, k=K))
        new = np.asarray(FU.fuse(counted, app, colour, weights={"colour": W_COLOUR,
                                                               "shape": W_SHAPE}, k=K,
                                 shape=shape))
        fit = split["FIT"]
        conf = split["DEV"] + split["FRESH"]
        hits = {i: (float(_hit(base, i)), float(_hit(new, i))) for i in range(len(keys))}
        res["cells"][cell] = {
            "view": view,
            "shape_channel": metrics(shape, "curvature object_mean"),
            "shipped": metrics(base, "fused3 (shipped)"),
            "fused4": metrics(new, "fused3 + curvature 0.5"),
            "paired_fit": _paired(base, new, fit),
            "paired_confirm": _paired(base, new, conf),
            "fit_ci": _boot_ci(base, new, fit),
            "confirm_ci": _boot_ci(base, new, conf),
            "hits": {str(i): hits[i] for i in fit},
        }
        c = res["cells"][cell]
        print("  %-6s channel r@8 %.3f top1 %.3f share>=0.99 %.3f | fused4 r@8 %.3f vs %.3f "
              "| FIT %s CI %s" % (cell, c["shape_channel"]["recall@8"], c["shape_channel"]["top1"],
                                  c["shape_channel"]["share_ge_0.99"], c["fused4"]["recall@8"],
                                  c["shipped"]["recall@8"], c["paired_fit"],
                                  [round(x, 3) for x in c["fit_ci"]["ci"]]), flush=True)
    gains = [res["cells"][c]["fused4"]["recall@8"] - res["cells"][c]["shipped"]["recall@8"]
             for c in VIEWS]
    conf_gains = [res["cells"][c]["fused4"]["recall@8"] - res["cells"][c]["shipped"]["recall@8"]
                  for c in VIEWS]
    # the registered bar: the pooled FIT difference, bootstrapped over the 40 FIT OBJECTS, each
    # object contributing the mean of its two cells' paired hits
    pooled = _pooled_ci(res, split["FIT"])
    passes = bool(float(np.mean(gains)) >= GAIN_REQUIRED
                  and pooled["excludes_zero"]
                  and float(np.mean(conf_gains)) >= CONFIRM_TOLERANCE)
    res["decision"] = {"fit_gain_mean": float(np.mean(gains)),
                       "confirm_gain_mean": float(np.mean(conf_gains)),
                       "pooled_fit_ci": pooled, "required_gain": GAIN_REQUIRED,
                       "confirm_tolerance": CONFIRM_TOLERANCE, "passes": passes}
    not_flat = all(res["cells"][c]["shape_channel"]["share_ge_0.99"] < 0.5 for c in VIEWS)
    res["decision"]["channel_not_flat"] = bool(not_flat)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res["decision"], indent=1))
    print("wrote", OUT)
    return res


def _pooled_ci(res: Dict[str, Any], idx: Sequence[int],
               resamples: int = RESAMPLES, seed: int = SEED) -> Dict[str, Any]:
    """The registered CI: pool the two FIT cells per OBJECT, then bootstrap over the objects."""
    per: Dict[int, List[float]] = {}
    for cell in VIEWS:
        for k, (a, b) in res["cells"][cell]["hits"].items():
            per.setdefault(int(k), []).append(float(b) - float(a))
    objs = [i for i in sorted(per) if i in set(idx)]
    d = np.asarray([float(np.mean(per[i])) for i in objs])
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(objs), size=(int(resamples), len(objs)))
    means = d[draw].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    up = int(sum(1 for i in objs if per[i][0] > 0 or (len(per[i]) > 1 and np.mean(per[i]) > 0)))
    down = int(sum(1 for i in objs if np.mean(per[i]) < 0))
    return {"point": float(d.mean()), "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0.0 or hi < 0.0), "objects": len(objs),
            "objects_up": up, "objects_down": down, "objects_tied": len(objs) - up - down,
            "resamples": int(resamples), "seed": int(seed)}


def gauge_ci() -> Dict[str, Any]:
    """Item 2's also-report: the paired object-clustered CI on the gauge gain M32 measured +0.099."""
    import importlib
    mod = importlib.import_module("Topology.m32_shape")
    from matching import placement_records as PR
    keys = [o["key"] for o in REC.objects()]
    cells = PR.load_cells(RECORDS, [REF, *VIEWS.values()])
    qk = [k for k in mod.SAMPLE if k in keys]
    base = mod.gauge("signature", keys, qk, cells)
    new = mod.gauge("unclipped", keys, qk, cells)
    # per-object gauge values, so the difference can be bootstrapped over objects
    per: Dict[str, Tuple[List[float], List[float]]] = {}
    for variant, store in (("signature", 0), ("unclipped", 1)):
        for q in qk:
            r = mod.gauge(variant, keys, [q], cells)
            per.setdefault(q, ([], []))[store].append(float(r["G_same"]))
    diffs = [np.mean(per[q][1]) - np.mean(per[q][0]) for q in qk]
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))
    means = np.asarray(diffs)[draw].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out = {"baseline_G_same": base["G_same"], "unclipped_G_same": new["G_same"],
           "delta": float(new["G_same"] - base["G_same"]),
           "per_object_delta": [float(x) for x in diffs], "objects": qk,
           "delta_ci": [float(lo), float(hi)], "excludes_zero": bool(lo > 0.0 or hi < 0.0),
           "distinguishable_from_0.10": bool(not (lo <= 0.10 <= hi)),
           "resamples": RESAMPLES, "seed": SEED,
           "note": "8 objects: the interval is wide by construction, which is the measurement"}
    with open(OUT.replace("m33_curvature.json", "m33_gauge_ci.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(out, indent=1))
    return out


def refine() -> Dict[str, Any]:
    """Item 3: weight within the four (2:1, 1:2, equal) and per-lens, chosen on FIT only."""
    from matching import placement_fuse as FU
    from matching import placement_records as PR
    from matching import placement_shape as PS
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    split = _split()
    ch = channels(list(VIEWS.values()))
    cells = PR.load_cells(RECORDS, [REF, *VIEWS.values()])
    gal_entries = [cells[(k, REF)] for k in keys]
    # weight variants inside the four: sharpness (sharp_x, sharp_y) vs relative curvature (curv_*)
    variants = {"equal": (1.0, 1.0, 1.0, 1.0), "sharp2:1": (2.0, 2.0, 1.0, 1.0),
                "curv2:1": (1.0, 1.0, 2.0, 2.0)}
    out: Dict[str, Any] = {"variants": {}, "per_lens": {}}
    for name, w in variants.items():
        gains = []
        for cell, view in VIEWS.items():
            counted = ch["counted|%s" % view]
            app = ch["appearance|%s" % view]
            q = [cells[(k, view)] for k in keys]
            colour = np.asarray(FU.colour_channel(gal_entries, q), dtype=np.float64)
            shape = PS.curvature_matrix(gal_entries, q, how="object_mean", weights=w)
            fused = np.asarray(FU.fuse(counted, app, colour,
                                       weights={"colour": W_COLOUR, "shape": W_SHAPE}, k=K,
                                       shape=shape))
            base = np.asarray(FU.fuse(counted, app, colour, weights={"colour": W_COLOUR}, k=K))
            gains.append(float(np.mean([_hit(fused, i) for i in split["FIT"]])
                               - np.mean([_hit(base, i) for i in split["FIT"]])))
        out["variants"][name] = {"fit_gain_mean": float(np.mean(gains)), "per_cell": gains}
        print("  weights %-8s FIT gain %+.4f" % (name, float(np.mean(gains))), flush=True)
    # per lens: the channel's recall@8 restricted to places of one lens
    for lens in ("Blob", "BoundaryArc", "CornerPoint"):
        per = {}
        for cell, view in VIEWS.items():
            counted = ch["counted|%s" % view]
            q = [cells[(k, view)] for k in keys]
            sub = [{**e, "places": [p for p in e["places"] if p.get("lens") == lens]}
                   for e in q]
            subg = [{**e, "places": [p for p in e["places"] if p.get("lens") == lens]}
                    for e in gal_entries]
            M = PS.curvature_matrix(subg, sub, how="object_mean")
            per[cell] = metrics(M, "curvature %s only" % lens)
            per[cell]["query_places"] = int(sum(len(e["places"]) for e in sub))
        out["per_lens"][lens] = per
        print("  lens %-12s recall@8 %s" % (lens, {k: round(v["recall@8"], 3)
                                                   for k, v in per.items()}), flush=True)
    total_places = int(sum(len(e["places"]) for e in
                           [cells[(k, VIEWS["fit_a"])] for k in keys]))
    out["coverage"] = {lens: out["per_lens"][lens]["fit_a"]["query_places"] / max(total_places, 1)
                       for lens in out["per_lens"]}
    with open(OUT.replace("m33_curvature.json", "m33_refine.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps({"coverage": out["coverage"]}, indent=1))
    return out


def mechanism() -> Dict[str, Any]:
    """Item 2 failed: is it the AGGREGATION or the COLUMNS?  Three numbers decide it.

    1. The channel's own recall@8 under BOTH aggregations on the same cells.  If `max` (the
       place-level form, which keeps one place's score) is markedly better than `object_mean`, the
       aggregation is the limit; if both are poor, the columns are.
    2. The intraclass correlation of each column: between-object variance over total.  A column
       whose variation is mostly WITHIN an object cannot identify the object under any aggregation.
    3. The cross-view correlation of each column's per-object profile: does the measurement
       transfer from -000-000 to the query view at all?
    """
    import math
    from matching import placement_records as PR
    from matching import placement_shape as PS
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cells = PR.load_cells(RECORDS, [REF, *VIEWS.values()])
    gal = [cells[(k, REF)] for k in keys]
    out: Dict[str, Any] = {"aggregation": {}, "icc": {}, "cross_view": {}}
    for cell, view in VIEWS.items():
        q = [cells[(k, view)] for k in keys]
        for how in ("object_mean", "max"):
            M = PS.curvature_matrix(gal, q, how=how)
            out["aggregation"].setdefault(how, {})[cell] = metrics(M, "curvature %s" % how)
            m = out["aggregation"][how][cell]
            print("  %-11s %-6s r@8 %.3f top1 %.3f share>=0.99 %.3f rank_med %.1f"
                  % (how, cell, m["recall@8"], m["top1"], m["share_ge_0.99"],
                     m["truth_rank_median"]), flush=True)
    # per-column variance decomposition, per view
    for f in PS.CURVATURE_FIELDS:
        per_view: Dict[str, Any] = {}
        for v in (REF, *VIEWS.values()):
            means: List[float] = []
            within: List[float] = []
            allv: List[float] = []
            for k in keys:
                x = np.asarray([float(p.get(f) or 0.0) for p in cells[(k, v)]["places"]])
                if x.size < 2:
                    continue
                means.append(float(x.mean()))
                within.append(float(x.var(ddof=1)))
                allv.extend(x.tolist())
            if len(means) < 3:
                continue
            between = float(np.var(means, ddof=1))
            wmean = float(np.mean(within))
            per_view[v] = {"between_obj_var": between, "within_obj_var": wmean,
                           "icc": between / (between + wmean) if between + wmean > 0 else 0.0,
                           "objects": len(means)}
        out["icc"][f] = per_view
        print("  ICC %-8s %s" % (f, {v: round(x["icc"], 4) for v, x in per_view.items()}),
              flush=True)
    # cross-view correlation of the per-object profile (10 deciles of the column)
    for f in PS.CURVATURE_FIELDS:
        pair: Dict[str, Any] = {}
        for cell, view in VIEWS.items():
            prof: Dict[str, List[float]] = {}
            for v in (REF, view):
                for k in keys:
                    x = np.asarray([float(p.get(f) or 0.0) for p in cells[(k, v)]["places"]])
                    prof.setdefault(v, {})[k] = (np.quantile(x, np.linspace(0.05, 0.95, 10))
                                                 if x.size else np.zeros(10))
            A = np.asarray([prof[REF][k] for k in keys])
            B = np.asarray([prof[view][k] for k in keys])
            cs = []
            for q10 in range(A.shape[1]):
                a, b = A[:, q10], B[:, q10]
                if a.std() > 0 and b.std() > 0:
                    cs.append(float(np.corrcoef(a, b)[0, 1]))
            pair[cell] = {"decile_corr_mean": float(np.mean(cs)) if cs else None,
                          "deciles": [round(c, 3) for c in cs]}
            print("  cross-view %-8s %-6s decile corr %.3f"
                  % (f, cell, pair[cell]["decile_corr_mean"] or float("nan")), flush=True)
        out["cross_view"][f] = pair
    with open(OUT.replace("m33_curvature.json", "m33_mechanism.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--gauge-ci", action="store_true")
    ap.add_argument("--mechanism", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.channels:
        channels(list(VIEWS.values()))
        return 0
    if a.arm:
        arm()
        return 0
    if a.refine:
        refine()
        return 0
    if a.mechanism:
        mechanism()
        return 0
    if a.gauge_ci:
        gauge_ci()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
