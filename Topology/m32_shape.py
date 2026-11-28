"""M32 item 4c: the M28 subset gauge, read through the unclipped columns.

    python Topology/m32_shape.py --gauge

The gauge is M28's, unchanged: for every query place of the two reserved views against the pool of
base-view places, `s_true` is the best match to the TRUE object and `s_other` the best match to
another object; G = P(s_true > s_other), ties 0.5, reported against ANY other object and against a
SAME-CLASS other object.  Chance 0.500.  Comparisons stay inside a lens.  Sample: one object per
class (M28's SAMPLE), 8 objects x 2 views, ~4,284 query places.

Four variants, all read on the SAME places:

    signature   the shipped graded signature kernel on the 5-slot quantised tuple (the baseline)
    m30         the five live graded columns through `graded_matrix` (the shipped shape form)
    unclipped   sharp_x, sharp_y, curv_x, curv_y through `kernel_block` (the same distance form
                as the colour channel: standardised by the pool's own spread, Gaussian)
    combined    all nine columns through `kernel_block`

BAR (written before the numbers): same-class gauge gain >= +0.10 over the baseline's 0.494.
If it fails, name whether the COLUMN (no variance across views) or the SIMILARITY (variance
present, not separated) is the limit, with distinct-value counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                       # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m32_shape.json")
REF = "000-000"
VIEWS = ("035-045", "045-180")
SAMPLE = ("apple/1", "car/1", "cow/1", "cup/1", "dog/1", "horse/1", "pear/1", "tomato/1")
LENSES = ("Blob", "BoundaryArc", "CornerPoint")
GAIN_REQUIRED = 0.10
BASELINE_CLAIMED = 0.494

VARIANTS = {
    "signature": {"kind": "sig", "fields": None},
    "m30": {"kind": "graded", "fields": ("resp", "var_x", "var_y", "eigratio", "rel_angle")},
    "unclipped": {"kind": "kernel", "fields": ("sharp_x", "sharp_y", "curv_x", "curv_y")},
    "combined": {"kind": "kernel", "fields": ("resp", "var_x", "var_y", "eigratio", "rel_angle",
                                              "sharp_x", "sharp_y", "curv_x", "curv_y")},
}


def _places(records: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(records["places"])


def _stack(places: Sequence[Dict[str, Any]], fields: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(p.get(f) or 0.0) for f in fields] for p in places],
                      dtype=np.float64)


def _auc(best: np.ndarray, truth: int, restrict: Optional[Sequence[int]] = None) -> Optional[float]:
    s_true = best[truth]
    if not np.isfinite(s_true):
        return None
    idx = [j for j in range(len(best)) if j != truth and np.isfinite(best[j])
           and (restrict is None or j in restrict)]
    if not idx:
        return None
    s_other = best[np.asarray(idx)]
    return float(np.mean((s_true > s_other) + 0.5 * (s_true == s_other)))


def gauge(variant: str, pool_keys: Sequence[str], query_keys: Sequence[str],
          cells: Dict[Tuple[str, str], Any], chunk: int = 512) -> Dict[str, Any]:
    from matching import placement_kernel as K
    from matching import placement_shape as PS
    cfg = VARIANTS[variant]
    fields = cfg["fields"]
    cls_of = {k: k.split("/")[0] for k in pool_keys}
    idx_of = {k: i for i, k in enumerate(pool_keys)}
    any_s: List[float] = []
    same_s: List[float] = []
    obj_any: List[float] = []
    obj_same: List[float] = []
    places_by_lens: Dict[str, int] = {}
    for view in VIEWS:
        for lens in LENSES:
            pool = [p for k in pool_keys for p in cells[(k, REF)]["places"]
                    if str(p.get("lens")) == lens]
            if not pool:
                continue
            pobj = np.asarray([idx_of[k] for k in pool_keys for p in cells[(k, REF)]["places"]
                               if str(p.get("lens")) == lens], dtype=np.int64)
            n_obj = len(pool_keys)
            # per (query object, lens) place rows
            for qk in query_keys:
                qi = idx_of[qk]
                qplaces = [p for p in cells[(qk, view)]["places"] if str(p.get("lens")) == lens]
                if not qplaces or len(qplaces) < 2:
                    continue
                same = [j for j in range(n_obj) if cls_of[pool_keys[j]] == cls_of[qk]
                        and pool_keys[j] != qk]
                per_obj_best: List[np.ndarray] = []
                for start in range(0, len(qplaces), chunk):
                    part = qplaces[start:start + chunk]
                    if cfg["kind"] == "sig":
                        Qk = [np.asarray(p["key"][1], dtype=np.float64) for p in part]
                        Pk = [np.asarray(p["key"][1], dtype=np.float64) for p in pool]
                        S = K.similarity_block(lens, np.asarray(Qk), np.asarray(Pk))
                    else:
                        Q = _stack(part, fields)
                        P = _stack(pool, fields)
                        if cfg["kind"] == "graded":
                            sub = PS.graded_matrix(Q, P)
                        else:
                            sig = PS.scales(P)
                            sub = PS.kernel_block(Q, P, sig)
                        S = sub
                    # best score per pool object, then per query place
                    bounds = np.searchsorted(pobj, np.arange(n_obj + 1))
                    per_obj_best.append(np.maximum.reduceat(S, bounds[:-1], axis=1))
                B = np.vstack(per_obj_best)                     # (n_q, n_obj)
                for r in range(B.shape[0]):
                    a = _auc(B[r], qi)
                    sm = _auc(B[r], qi, restrict=same)
                    if a is not None:
                        any_s.append(a)
                        places_by_lens[lens] = places_by_lens.get(lens, 0) + 1
                    if sm is not None:
                        same_s.append(sm)
                # object level: the query object's own mean place score against every object
                mean_rows = B.mean(axis=0)
                a = _auc(mean_rows, qi)
                sm = _auc(mean_rows, qi, restrict=same)
                if a is not None:
                    obj_any.append(a)
                if sm is not None:
                    obj_same.append(sm)
    return {"variant": variant, "query_places": len(any_s), "places_by_lens": places_by_lens,
            "G_any": float(np.mean(any_s)) if any_s else None,
            "G_same": float(np.mean(same_s)) if same_s else None,
            "n_same": len(same_s),
            "G_any_object": float(np.mean(obj_any)) if obj_any else None,
            "G_same_object": float(np.mean(obj_same)) if obj_same else None,
            "n_object_pairs": len(obj_any), "chance": 0.5}


def columns(keys: Sequence[str], cells: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
    """Distinct values per column and per VIEW -- the test of whether a column can separate."""
    from matching import placement_records as PR
    out: Dict[str, Any] = {}
    for f in PR.SHAPE_FIELDS:
        per_view = {}
        for v in (REF, *VIEWS):
            vals = np.asarray([p.get(f) for k in keys for p in cells[(k, v)]["places"]
                               if p.get(f) is not None], dtype=np.float64)
            per_view[v] = {"places": int(vals.size),
                           "distinct": int(np.unique(np.round(vals, 9)).size),
                           "std": float(vals.std()) if vals.size else None}
        out[f] = per_view
    return out


def run() -> Dict[str, Any]:
    from matching import placement_records as PR
    keys = [o["key"] for o in REC.objects()]
    cells = PR.load_cells(RECORDS, [REF, *VIEWS])
    qk = [k for k in SAMPLE if k in keys]
    res: Dict[str, Any] = {"sample": qk, "views": list(VIEWS), "pool_objects": len(keys),
                           "columns": columns(keys, cells), "results": {}}
    for v in VARIANTS:
        r = gauge(v, keys, qk, cells)
        res["results"][v] = r
        print("  %-10s G_any %.3f  G_same %.3f  (object: %.3f / %.3f)  places %d"
              % (v, r["G_any"], r["G_same"], r["G_any_object"] or float("nan"),
                 r["G_same_object"] or float("nan"), r["query_places"]), flush=True)
    base = res["results"]["signature"]["G_same"]
    dec: Dict[str, Any] = {}
    for v in VARIANTS:
        if v == "signature":
            continue
        g = res["results"][v]["G_same"]
        dec[v] = {"delta_same_class": float(g - base), "passes": bool(g - base >= GAIN_REQUIRED)}
    res["decision"] = {"baseline_G_same": base, "claimed_baseline": BASELINE_CLAIMED,
                       "required_gain": GAIN_REQUIRED, "by_variant": dec,
                       "winner": max([k for k in dec if dec[k]["passes"]],
                                     key=lambda k: dec[k]["delta_same_class"], default=None)}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res["decision"], indent=1))
    print("wrote", OUT)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", action="store_true")
    ap.parse_args(list(argv) if argv is not None else None)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
