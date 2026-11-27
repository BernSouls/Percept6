"""M31 items 4a/4b/4c: what the two "dead" shape columns actually are, and what shape is worth
once they stop voting.

    python Topology/m31_shape.py --columns     # 4a: distinct-value counts, both views, all 80 objects
    python Topology/m31_shape.py --gauge2      # 4b: the M28-subset gauge with var_x/var_y weighted 0
    python Topology/m31_shape.py --anisotropy  # 4c: rotation-normalised columns, same gauge, once

4a is the correction M30 needs: M30 reported `var_x`/`var_y` at exactly 0.500 cross-view and read
that as "image-axis variances rotate with the viewpoint".  They do not rotate -- they are pinned at
the cap of `lenses/keypoints.py::_parabola` (`1 / (2|den|)` clipped to [1e-3, 1]) because the
normalised response has a tiny second difference, so the column carries one or two values and the
gauge is chance for want of variance, not for want of invariance.  This script counts the values.

4b re-measures the gauge with those two columns weighted 0.  No rebuild: they are columns, not key
slots, so the registered key is untouched; only the weight vector handed to `graded_matrix` changes.
Bar, fixed before the run: **gain >= +0.10** over the current signature (the same bar as M30).

4c runs only if 4b fails: two rotation-normalised columns replace them,
`eigratio` (already rotation invariant) gaining two companions --
`aniso_along` = structure-tensor eigenvalue ratio sampled along the place's own frame, and
`var_norm` = the scale-normalised per-axis response variance (`resp / |den|`), computed here and
NOT registered (the mission asks for one measurement, not a rebuild).
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

from Topology import m26_records as REC                       # noqa: E402
from Topology.m30_shape import LENSES, SAMPLE, VIEW, REF, _auc  # noqa: E402

OUT = os.path.join(ROOT, "Topology", "results", "placement", "m31_shape.json")
BAR = 0.10


def _places(view: str, keys: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Live place records for ``keys`` at ``view`` (the detector and the lens split, cached)."""
    from lenses import keypoints as KP
    from matching.placement_frontend import load_image
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k in keys:
        o = next(x for x in REC.objects() if x["key"] == k)
        img = load_image(REC.path_of(o, view))
        rows = []
        for p in KP.detect(img):
            sig = KP.signature_of(p)
            rows.append({"lens": sig.class_, "key": (sig.class_, tuple(sig.quantized)),
                         "x": float(p.x), "y": float(p.y), "s": float(p.scale),
                         "resp": float(p.resp), "var_x": float(p.var_x), "var_y": float(p.var_y)})
        out[k] = rows
    return out


def columns() -> Dict[str, Any]:
    """4a: how many distinct values does each shape column actually carry?"""
    keys = [o["key"] for o in REC.objects()]
    res: Dict[str, Any] = {"objects": len(keys), "views": {}}
    for view in (REF, VIEW):
        C = _places(view, keys)
        per: Dict[str, Any] = {}
        for f in ("resp", "var_x", "var_y"):
            vals = np.asarray([p[f] for rows in C.values() for p in rows], dtype=np.float64)
            u = np.unique(np.round(vals, 6))
            counts = {str(float(v)): int((np.abs(vals - v) < 1e-6).sum()) for v in u[:6]}
            per[f] = {"places": int(vals.size), "distinct": int(u.size),
                      "min": float(vals.min()), "max": float(vals.max()),
                      "top_values": counts}
        # the response's own second difference, which `_parabola` divides by
        per["parabola_note"] = ("var = 1/(2|den|) clipped to [1e-3, 1]; a column at the cap means "
                               "|den| <= 0.5 for every place")
        res["views"][view] = per
        print("  %s: var_x distinct %d of %d places (min %.4f max %.4f), var_y distinct %d"
              % (view, per["var_x"]["distinct"], per["var_x"]["places"],
                 per["var_x"]["min"], per["var_x"]["max"], per["var_y"]["distinct"]), flush=True)
    return res


def _gauge(weights: Sequence[float], kind: str = "both", extra: str = "") -> Dict[str, Any]:
    """The M30 gauge, with an explicit weight vector over the five fields."""
    from matching import placement_shape as PS
    from matching import placement_kernel as K
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls_of = {k: k.split("/")[0] for k in keys}
    idx_of = {k: i for i, k in enumerate(keys)}
    C = {v: _records(v, keys) for v in (REF, VIEW)}
    any_s: List[float] = []
    same_s: List[float] = []
    for qk in SAMPLE:
        truth = idx_of[qk]
        same = [idx_of[k] for k in keys if cls_of[k] == cls_of[qk] and k != qk]
        per_obj = np.full(len(keys), -np.inf)
        for lens in LENSES:
            pool = [(oi, np.asarray([p.get(f, 0.0) or 0.0 for f in PS.FIELDS]), tuple(p["key"][1]))
                    for oi, k in enumerate(keys) for p in C[REF][k] if p["lens"] == lens]
            q_rows = [(np.asarray([p.get(f, 0.0) or 0.0 for f in PS.FIELDS]), tuple(p["key"][1]))
                      for p in C[VIEW][qk] if p["lens"] == lens]
            if not pool or not q_rows:
                continue
            P = np.asarray([r[1] for r in pool])
            Pobj = np.asarray([r[0] for r in pool], dtype=np.int64)
            S = PS.graded_matrix(np.asarray([r[0] for r in q_rows]), P, weights=weights)
            if kind in ("key", "both"):
                Kb = np.asarray([r[2] for r in pool], dtype=np.float64)
                for i in range(S.shape[0]):
                    kq = np.asarray(q_rows[i][1], dtype=np.float64)
                    kk = K.similarity_block(lens, [kq], Kb)[0]
                    S[i] = kk if kind == "key" else np.sqrt(np.maximum(kk, 0.0)
                                                            * np.maximum(S[i], 0.0))
            for i in range(S.shape[0]):
                best = np.full(len(keys), -np.inf)
                np.maximum.at(best, Pobj, S[i])
                a = _auc(best, truth)
                s = _auc(best, truth, same)
                if a is not None:
                    any_s.append(a)
                if s is not None:
                    same_s.append(s)
                per_obj = np.maximum(per_obj, best)
        if kind == "both":
            any_s.append(_auc(per_obj, truth) or 0.0)
    return {"kind": kind, "weights": list(weights), "extra": extra,
            "G_any": float(np.mean(any_s)), "G_same": float(np.mean(same_s)),
            "n": len(same_s), "chance": 0.5}


_REC_CACHE: Dict[Tuple[str, int], Dict[str, List[Dict[str, Any]]]] = {}


def _records(view: str, keys: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Place records WITH the shape columns, read from the committed npz (no rebuild)."""
    from matching import placement_records as PR
    tag = (view, len(keys))
    if tag in _REC_CACHE:
        return _REC_CACHE[tag]
    from matching import placement_shape as PS
    cells = PR.load_cells(os.path.join(ROOT, "data", "placement_records_m30.npz"), [view])
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k in keys:
        c = cells.get((k, view))
        rows = []
        if c:
            for p in c["places"]:
                q = dict(p)
                if not any(f in q and q[f] is not None for f in PS.FIELDS):
                    q.update({f: 0.0 for f in PS.FIELDS})
                rows.append(q)
        out[k] = rows
    _REC_CACHE[tag] = out
    return out


def gauge2() -> Dict[str, Any]:
    """4b: the two pinned columns weighted 0, against the M30 baseline."""
    from matching import placement_shape as PS
    w0 = list(PS.WEIGHTS)
    w = list(w0)
    w[PS.FIELDS.index("var_x")] = 0.0
    w[PS.FIELDS.index("var_y")] = 0.0
    t0 = time.time()
    base_key = _gauge(PS.WEIGHTS, kind="key")
    print("  key only           G_any %.3f  G_same %.3f" % (base_key["G_any"],
                                                           base_key["G_same"]), flush=True)
    base = _gauge(PS.WEIGHTS, kind="both")
    print("  shape all five     G_any %.3f  G_same %.3f" % (base["G_any"], base["G_same"]),
          flush=True)
    fixed = _gauge(w, kind="both")
    print("  shape, 2 pinned 0  G_any %.3f  G_same %.3f" % (fixed["G_any"], fixed["G_same"]),
          flush=True)
    live = _gauge(w, kind="shape")
    print("  live columns only  G_any %.3f  G_same %.3f" % (live["G_any"], live["G_same"]),
          flush=True)
    gain = fixed["G_same"] - base_key["G_same"]
    out = {"bar": {"gain_required": BAR, "measured": gain,
                   "baseline_key_same": base_key["G_same"],
                   "both_same": fixed["G_same"], "passes": bool(gain >= BAR)},
           "key": base_key, "both_all_five": base, "both_pinned_zero": fixed,
           "live_only": live, "weights_pinned_zero": w, "weights_shipped": w0,
           "query_places": fixed["n"], "seconds": round(time.time() - t0, 1)}
    return out


def anisotropy() -> Dict[str, Any]:
    """4c: rotation-normalised columns in place of the pinned ones -- measured ONCE, no fitting.

    Four columns, each invariant to a rotation about the place's own frame by construction:
    ``eigr`` (structure-tensor eigenvalue ratio), ``rel`` (principal direction minus the place's
    own theta, mod pi), ``coh`` (the detector's own coherence -- computed and discarded in M30),
    and ``along`` (gradient energy along the place's frame direction over the energy across it).
    Weights are fixed at (2, 3, 2, 2) before the measurement.
    """
    from lenses import keypoints as KP
    from matching.placement_frontend import load_image
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    idx_of = {k: i for i, k in enumerate(keys)}
    cls_of = {k: k.split("/")[0] for k in keys}
    W = [2.0, 3.0, 2.0, 2.0]

    def cols(view: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            o = next(x for x in objs if x["key"] == k)
            img = load_image(REC.path_of(o, view))
            g = np.asarray(img, dtype=np.float64)
            while g.ndim > 2:
                g = g.mean(axis=-1)
            g = g / 255.0
            gy, gx = np.gradient(g)
            rows = []
            for p in KP.detect(img):
                r = int(max(1, round(1.5 * max(p.scale, 1e-6))))
                x0, y0 = int(round(p.x)), int(round(p.y))
                sl = (slice(max(0, y0 - r), min(g.shape[0], y0 + r + 1)),
                      slice(max(0, x0 - r), min(g.shape[1], x0 + r + 1)))
                Jxx = float(np.mean(gx[sl] ** 2))
                Jyy = float(np.mean(gy[sl] ** 2))
                Jxy = float(np.mean(gx[sl] * gy[sl]))
                tr = Jxx + Jyy
                disc = float(np.hypot(Jxx - Jyy, 2.0 * Jxy))
                eigr = disc / (tr + 1e-9)
                arm = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
                th = p.theta if p.theta is not None else 0.0
                rel = float((arm - 0.5 * th) % np.pi)
                ct, st = np.cos(0.5 * th), np.sin(0.5 * th)
                along = (Jxx * ct * ct + 2.0 * Jxy * ct * st + Jyy * st * st)
                across = (Jxx * st * st - 2.0 * Jxy * ct * st + Jyy * ct * ct)
                an = float(np.log((along + 1e-9) / (across + 1e-9)))
                rows.append((eigr, rel, float(p.coherence), an))
            out[k] = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
        return out

    C = {v: cols(v) for v in (REF, VIEW)}
    any_s: List[float] = []
    same_s: List[float] = []
    for qk in SAMPLE:
        truth = idx_of[qk]
        same = [idx_of[k] for k in keys if cls_of[k] == cls_of[qk] and k != qk]
        pool = np.vstack([C[REF][k] for k in keys])
        Pobj = np.repeat(np.arange(len(keys)), [C[REF][k].shape[0] for k in keys])
        Q = C[VIEW][qk]
        if not Q.size:
            continue
        S = _graded(Q, pool, W)
        per_obj = np.full(len(keys), -np.inf)
        for i in range(S.shape[0]):
            best = np.full(len(keys), -np.inf)
            np.maximum.at(best, Pobj, S[i])
            a = _auc(best, truth)
            s = _auc(best, truth, same)
            if a is not None:
                any_s.append(a)
            if s is not None:
                same_s.append(s)
            per_obj = np.maximum(per_obj, best)
        any_s.append(_auc(per_obj, truth) or 0.0)
    out = {"fields": ["eigr", "rel", "coherence", "aniso_along"], "weights": W,
           "G_any": float(np.mean(any_s)), "G_same": float(np.mean(same_s)),
           "n": len(same_s), "chance": 0.5, "key_baseline_same": 0.494}
    print("  rotation-normalised  G_any %.3f  G_same %.3f  (key 0.494)"
          % (out["G_any"], out["G_same"]), flush=True)
    return out


def _graded(A: np.ndarray, B: np.ndarray, w: Sequence[float]) -> np.ndarray:
    ww = np.asarray(w, dtype=np.float64)
    ww = ww / max(ww.sum(), 1e-12)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for j, wj in enumerate(ww):
        if wj == 0.0:
            continue
        if j == 1:
            d = np.abs(A[:, j][:, None] - B[:, j][None, :])
            v = np.cos(2.0 * d)
        else:
            span = max(float(np.percentile(np.abs(np.concatenate([A[:, j], B[:, j]])), 99)), 1e-6)
            a01 = np.clip(A[:, j] / span, -1.0, 1.0)
            b01 = np.clip(B[:, j] / span, -1.0, 1.0)
            v = 1.0 - np.abs(a01[:, None] - b01[None, :]) / 2.0
        out += wj * v
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", action="store_true")
    ap.add_argument("--gauge2", action="store_true")
    ap.add_argument("--anisotropy", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    data: Dict[str, Any] = {}
    if os.path.exists(OUT):
        try:
            data = json.load(open(OUT))
        except Exception:
            data = {}
    if a.columns:
        data["columns"] = columns()
    if a.gauge2:
        data["gauge2"] = gauge2()
        if data["gauge2"]["bar"]["passes"]:
            data["decision"] = "shape enters the door and matcher below colour's weight"
        else:
            data["decision"] = "shape ceiling named; 4c runs once"
    if a.anisotropy:
        data["anisotropy"] = anisotropy()
    data["seconds"] = data.get("seconds", 0)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True, default=float)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
