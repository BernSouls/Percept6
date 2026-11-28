"""M30 B1/B3: is the shape grid saturated, and does the graded shape carry identity?

    python Topology/m30_shape.py --saturation    # B1: grid, keys used, places per cell, I(key;object)
    python Topology/m30_shape.py --information    # B3: I(key+shape;object) per lens
    python Topology/m30_shape.py --gauge          # B3: same-class cross-view gauge, 8 objects

Reads the committed records only (`data/placement_records_m30.npz`).  Chance 0.500 for the gauge.
The bar, fixed before the numbers: **same-class gauge gain >= +0.10 over the current signature**.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m30_shape.json")
REF = "000-000"
VIEW = "035-045"
SAMPLE = ("apple/1", "car/1", "cow/1", "cup/1", "dog/1", "horse/1", "pear/1", "tomato/1")
LENSES = ("Blob", "BoundaryArc", "CornerPoint")
BAR = 0.10

#: the deliverable grid of each place lens (M30's correction of M29): theta 12 x coherence 4 x
#: scale 4; kind and dim are fixed by the lens.
GRID = {"Blob": 12 * 4 * 4, "BoundaryArc": 12 * 4 * 4, "CornerPoint": 12 * 4 * 4}

_CELLS: Dict[Tuple[str, str], Dict[str, Any]] = {}


def cells(views: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not _CELLS:
        from matching import placement_records as PR
        _CELLS.update(PR.load_cells(RECORDS, list(views)))
    return _CELLS


def _entropy(counts: Sequence[int]) -> float:
    n = float(sum(counts))
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def _cond_entropy(joint: Dict[Tuple[Any, Any], int], totals: Dict[Any, int]) -> float:
    n = float(sum(joint.values()))
    if n <= 0:
        return 0.0
    h = 0.0
    by_y: Dict[Any, List[int]] = defaultdict(list)
    for (x, y), c in joint.items():
        by_y[y].append(c)
    for y, xs in by_y.items():
        h += (totals[y] / n) * _entropy(xs)
    return h


def shape_bins(p: Dict[str, Any]) -> Tuple[int, ...]:
    """The five graded fields, quantised by the registered bins (the index view of the shape)."""
    from matching import placement_shape as PS
    out: List[int] = []
    vals = (p.get("resp", 0.0), p.get("var_x", 0.0), p.get("var_y", 0.0),
            p.get("eigratio", 0.0), p.get("rel_angle", 0.0))
    for v, nb, (lo, hi) in zip(vals, PS.BINS, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
                                               (0.0, 1.0), (0.0, math.pi))):
        t = (float(v) - lo) / (hi - lo)
        out.append(int(min(max(int(t * nb), 0), nb - 1)))
    return tuple(out)


# --------------------------------------------------------------------------- B1 / B3 information
def information() -> Dict[str, Any]:
    from Topology import m26_records as REC
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    t0 = time.time()
    out: Dict[str, Any] = {"n_objects": len(keys), "H_object_bits": round(math.log2(len(keys)), 2),
                           "views": {}}
    for view in (REF, VIEW):
        C = cells([REF, VIEW])
        per_lens: Dict[str, Dict[str, Any]] = {}
        for lens in LENSES:
            key_counts: Counter = Counter()
            joint_key: Dict[Tuple[Any, Any], int] = defaultdict(int)
            joint_both: Dict[Tuple[Any, Any], int] = defaultdict(int)
            totals: Dict[str, int] = defaultdict(int)
            for k in keys:
                for p in C[(k, view)]["places"]:
                    if p["lens"] != lens:
                        continue
                    q = tuple(p["key"][1])
                    key_counts[q] += 1
                    joint_key[(q, k)] += 1
                    joint_both[((q, shape_bins(p)), k)] += 1
                    totals[k] += 1
            if not key_counts:
                continue
            H = _entropy(list(key_counts.values()))
            Hc = _cond_entropy(joint_key, totals)
            both: Counter = Counter()
            for (b, _k), c in joint_both.items():
                both[b] += c
            Hb = _entropy(list(both.values()))
            Hbc = _cond_entropy(joint_both, totals)
            occupied = len(key_counts)
            per_lens[lens] = {
                "places": int(sum(key_counts.values())),
                "grid_cells": GRID.get(lens),
                "distinct_keys": occupied,
                "grid_filled_frac": round(occupied / max(1, GRID.get(lens, 1)), 4),
                "places_per_occupied_cell": round(sum(key_counts.values()) / max(1, occupied), 1),
                "max_places_in_one_cell": int(max(key_counts.values())),
                "H_key_bits": round(H, 3),
                "I_key_object_bits": round(H - Hc, 3),
                "distinct_key_plus_shape": len(both),
                "H_key_plus_shape_bits": round(Hb, 3),
                "I_key_plus_shape_object_bits": round(Hb - Hbc, 3),
                "identity_gain_bits": round((Hb - Hbc) - (H - Hc), 3),
            }
        # pooled over the three lenses
        tot_places = sum(v["places"] for v in per_lens.values())
        w_key = sum(v["I_key_object_bits"] * v["places"] for v in per_lens.values()) / max(1, tot_places)
        w_both = (sum(v["I_key_plus_shape_object_bits"] * v["places"]
                      for v in per_lens.values()) / max(1, tot_places))
        out["views"][view] = {"by_lens": per_lens,
                              "pooled_places_weighted_I_key": round(w_key, 3),
                              "pooled_places_weighted_I_key_plus_shape": round(w_both, 3)}
        print("  %s: %s" % (view, {l: (v["distinct_keys"], v["places_per_occupied_cell"],
                                      v["I_key_object_bits"], v["I_key_plus_shape_object_bits"])
                                  for l, v in per_lens.items()}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- B3 the gauge
def _auc(best: np.ndarray, truth: int, restrict: Sequence[int] = ()) -> float | None:
    s_true = best[truth]
    if not np.isfinite(s_true):
        return None
    idx = [j for j in range(len(best)) if j != truth and np.isfinite(best[j])
           and (not restrict or j in restrict)]
    if not idx:
        return None
    s_other = best[np.asarray(idx)]
    return float(np.mean((s_true > s_other) + 0.5 * (s_true == s_other)))


def gauge(kind: str, chunk: int = 192, aggregate: str = "max") -> Dict[str, Any]:
    """Same-class cross-view AUC for one similarity on the M28 sample of 8 objects."""
    from Topology import m26_records as REC
    from matching import placement_shape as PS
    from matching import placement_kernel as K
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls_of = {k: k.split("/")[0] for k in keys}
    idx_of = {k: i for i, k in enumerate(keys)}
    C = cells([REF, VIEW])
    any_scores: List[float] = []
    same_scores: List[float] = []
    obj_any: List[float] = []
    obj_same: List[float] = []
    for qk in SAMPLE:
        truth = idx_of[qk]
        same = [idx_of[k] for k in keys if cls_of[k] == cls_of[qk] and k != qk]
        per_obj = np.full(len(keys), -np.inf)
        for lens in LENSES:
            pool_rows: List[Tuple[int, np.ndarray, Tuple[float, ...]]] = []
            for oi, k in enumerate(keys):
                for p in C[(k, REF)]["places"]:
                    if p["lens"] != lens:
                        continue
                    pool_rows.append((oi, np.asarray([p.get(f, 0.0) or 0.0
                                                      for f in PS.FIELDS], dtype=np.float64),
                                      tuple(p["key"][1])))
            q_rows: List[Tuple[np.ndarray, Tuple[float, ...]]] = [
                (np.asarray([p.get(f, 0.0) or 0.0 for f in PS.FIELDS], dtype=np.float64),
                 tuple(p["key"][1]))
                for p in C[(qk, VIEW)]["places"] if p["lens"] == lens]
            if not pool_rows or not q_rows:
                continue
            P = np.asarray([r[1] for r in pool_rows])
            Pobj = np.asarray([r[0] for r in pool_rows], dtype=np.int64)
            Pkey = [r[2] for r in pool_rows]
            S = PS.graded_matrix(np.asarray([r[0] for r in q_rows]), P)
            if kind in ("key", "both"):
                # the key's own graded kernel, row by row (cheap: 5 slots)
                Kb = np.asarray(Pkey, dtype=np.float64)
                for i in range(S.shape[0]):
                    kq = np.asarray(q_rows[i][1], dtype=np.float64)
                    kk = K.similarity_block(lens, [kq], Kb)[0]
                    if kind == "key":
                        S[i] = kk
                    else:
                        S[i] = np.sqrt(np.maximum(kk, 0.0) * np.maximum(S[i], 0.0))
            for i in range(S.shape[0]):
                best = np.full(len(keys), -np.inf)
                np.maximum.at(best, Pobj, S[i])
                a = _auc(best, truth)
                s = _auc(best, truth, same)
                if a is not None:
                    any_scores.append(a)
                if s is not None:
                    same_scores.append(s)
                per_obj = np.maximum(per_obj, best)
        # object-level aggregation: the best place score, summed over lenses (the door's own read)
        if kind == "both":
            obj_any.append(_auc(per_obj, truth) or 0.0)
            obj_same.append(_auc(per_obj, truth, same) or 0.0)
    return {"kind": kind, "queries": list(SAMPLE), "query_places": len(any_scores),
            "G_any": float(np.mean(any_scores)) if any_scores else None,
            "G_same": float(np.mean(same_scores)) if same_scores else None,
            "object_level_G_any": float(np.mean(obj_any)) if obj_any else None,
            "object_level_G_same": float(np.mean(obj_same)) if obj_same else None,
            "chance": 0.5}


def field_gauge(field: str, chunk: int = 256) -> Dict[str, Any]:
    """Same-class cross-view AUC using ONE shape field alone (the mechanism, field by field)."""
    from Topology import m26_records as REC
    from matching import placement_shape as PS
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls_of = {k: k.split("/")[0] for k in keys}
    idx_of = {k: i for i, k in enumerate(keys)}
    C = cells([REF, VIEW])
    j = PS.FIELDS.index(field)
    any_s: List[float] = []
    same_s: List[float] = []
    for qk in SAMPLE:
        truth = idx_of[qk]
        same = [idx_of[k] for k in keys if cls_of[k] == cls_of[qk] and k != qk]
        for lens in LENSES:
            pool = [(idx_of[k], float(p[field])) for k in keys for p in C[(k, REF)]["places"]
                    if p["lens"] == lens]
            qv = [float(p[field]) for p in C[(qk, VIEW)]["places"] if p["lens"] == lens]
            if not pool or not qv:
                continue
            Pobj = np.asarray([o for o, _v in pool], dtype=np.int64)
            Pv = np.asarray([v for _o, v in pool], dtype=np.float64)
            Q = np.asarray(qv, dtype=np.float64)
            if field in ("rel_angle",):
                D = np.cos(2.0 * np.abs(Q[:, None] - Pv[None, :]))
            else:
                D = 1.0 - np.abs(Q[:, None] - Pv[None, :])
            for i in range(D.shape[0]):
                best = np.full(len(keys), -np.inf)
                np.maximum.at(best, Pobj, D[i])
                a = _auc(best, truth)
                s = _auc(best, truth, same)
                if a is not None:
                    any_s.append(a)
                if s is not None:
                    same_s.append(s)
    return {"field": field, "G_any": float(np.mean(any_s)) if any_s else None,
            "G_same": float(np.mean(same_s)) if same_s else None,
            "n": len(any_s), "chance": 0.5}


def corners(quantiles: Sequence[float] = (90.0, 95.0, 97.0, 98.5)) -> Dict[str, Any]:
    """B4: sweep `corner_quantile` on FIT only; pick by TOTAL identity carried."""
    from lenses import keypoints as KP
    from matching.placement_frontend import load_image
    from Topology import m26_records as REC
    objs = [o for o in REC.objects() if o["split"] == "FIT"]
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for q in quantiles:
        counts: Counter = Counter()
        joint: Dict[Tuple[Any, Any], int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)
        for o in objs:
            img = load_image(REC.path_of(o, REF))
            for p in KP.detect(img, corner_quantile=float(q)):
                sig = KP.signature_of(p)
                key = (sig.class_, tuple(sig.quantized))
                counts[sig.class_] += 1
                joint[(key, o["key"])] += 1
                totals[o["key"]] += 1
        H = _entropy(list(Counter({k: c for (k, _o), c in joint.items()
                                   }).values()))
        Hc = _cond_entropy(dict(joint), dict(totals))
        per_place = H - Hc
        n_places = int(sum(totals.values()))
        corner_n = counts.get("CornerPoint", 0)
        rows.append({"corner_quantile": float(q), "places": n_places,
                     "corner_places": corner_n,
                     "corner_frac": round(corner_n / max(1, n_places), 4),
                     "by_class": dict(counts),
                     "H_key_bits": round(H, 3), "I_per_place_bits": round(per_place, 3),
                     "I_total_bits": round(per_place * n_places, 1)})
        print("  quantile %.1f: %d places (%d corners, %.1f%%), I %.3f bits/place, total %.0f bits"
              % (q, n_places, corner_n, 100.0 * corner_n / max(1, n_places), per_place,
                 per_place * n_places), flush=True)
    best = max(rows, key=lambda r: r["I_total_bits"])
    out = {"fit_objects": len(objs), "view": REF, "rows": rows,
           "chosen": best["corner_quantile"],
           "chosen_by": "I_total_bits (total identity carried, FIT only)",
           "shipped": 98.5, "seconds": round(time.time() - t0, 1)}
    print("  chosen by total identity: quantile %.1f (%.0f bits vs shipped %.0f bits)"
          % (out["chosen"], best["I_total_bits"],
             [r for r in rows if r["corner_quantile"] == 98.5][0]["I_total_bits"]))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saturation", action="store_true")
    ap.add_argument("--information", action="store_true")
    ap.add_argument("--gauge", action="store_true")
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--corners", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    from matching import placement_shape as PS
    PS.register()
    data: Dict[str, Any] = {}
    if os.path.exists(OUT):
        data = json.load(open(OUT))
    t0 = time.time()
    if a.saturation or a.information:
        data["information"] = information()
    if a.gauge:
        g: Dict[str, Any] = {}
        for kind in ("key", "shape", "both"):
            g[kind] = gauge(kind)
            print("  gauge %-6s G_any %.3f  G_same %.3f  (object-level same %.3f), n %d"
                  % (kind, g[kind]["G_any"] or 0, g[kind]["G_same"] or 0,
                     g[kind]["object_level_G_same"] or 0, g[kind]["query_places"]), flush=True)
        base = g["key"]["G_same"]
        g["bar"] = {"baseline_key_same_class": base, "gain_required": BAR,
                    "both_gain": (g["both"]["G_same"] - base) if base else None,
                    "shape_gain": (g["shape"]["G_same"] - base) if base else None}
        g["passes"] = bool(g["bar"]["both_gain"] is not None
                           and g["bar"]["both_gain"] >= BAR)
        data["gauge"] = g
        print("  bar: both_gain %+0.3f (need >= +0.10) -> %s"
              % (g["bar"]["both_gain"] or 0.0, "PASS" if g["passes"] else "fail"), flush=True)
    if a.fields:
        f: Dict[str, Any] = {}
        for name in PS.FIELDS:
            f[name] = field_gauge(name)
            print("  field %-10s G_any %.3f  G_same %.3f" % (name, f[name]["G_any"] or 0,
                                                             f[name]["G_same"] or 0), flush=True)
        data["fields"] = f
    if a.corners:
        data["corners"] = corners()
    data["registration"] = {"class": PS.CLASS, "digest": PS.digest(), **PS.registered(),
                            "missing": PS.missing_fields()}
    data["seconds"] = round(time.time() - t0, 1)
    data["manifest"] = {"environment": _env(), "records": os.path.basename(RECORDS),
                        "records_bytes": os.path.getsize(RECORDS)}
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True, default=float)
    print("wrote", OUT)
    return 0


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


if __name__ == "__main__":
    raise SystemExit(main())
