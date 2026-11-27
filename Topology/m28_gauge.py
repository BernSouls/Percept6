"""M28 items 4 and 5: the gauge, and the candidate place content.

THE GAUGE (item 4).  For every query place of view -035-045 against the pool of places at
-000-000, let

    s_true(q)  = max over the TRUE object's places of similarity(q, p)
    s_other(q) = max over another object's places of the same

and G = P(s_true > s_other) over the (query place, other object) pairs, ties counting 0.5.  G is
reported twice: against ANY other object, and against a SAME-CLASS other object (the failure the
mission's item 3 found: 22 of 34 losses are to another instance of the same class).  Chance 0.5.
Comparisons are within lens, for the baseline and for every candidate alike, because a Blob's
content is not comparable with a CornerPoint's.

THE CANDIDATES (item 5).  All graded population codes, rotation-normalised by the place's own
orientation, scale-normalised by its own scale, nothing read from a label.  Constants fixed here,
before any number:

    C1 colour    two concentric rings at radii 0.5 s and 1.5 s, 12 angular samples each,
                 mean CIELAB L (divided by 100) and a, b (divided by 128)  -> 6 values
    C2 orientation  Gabor energies at 8 orientations x 2 spatial frequencies taken from the
                 EXISTING bank (`perception/filters.py` kernel builder and its fractions, the
                 2nd and 4th of its 5 geo-spaced frequencies), averaged over a disc of radius
                 2 s, bins ROLLED so index 0 is the place's own axis, L1-normalised  -> 16 values
    C3 texture   from the same responses: roughness = std of the orientation marginal, and
                 log total energy                                          -> 2 values
    C4 = C1 + C2 + C3, each block L2-normalised, concatenated with equal weights, re-normalised.

Similarity, for every candidate, is the COSINE of those unit vectors.  The baseline's similarity is
the shipped graded signature kernel (`matching/placement_kernel.similarity_block`, the registered
wide constitution) on the place's 5-slot quantised tuple -- i.e. exactly the content that is in the
record today.

THE RULE (written before the numbers): a candidate PROCEEDS only if its same-class G beats the
baseline's same-class G by >= +0.10.

    python Topology/m28_gauge.py --baseline     # item 4, from the cached records, all 80 objects
    python Topology/m28_gauge.py --candidates   # item 5, 8 query objects x 4 candidates
    python Topology/m28_gauge.py --report       # one table
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

REF = "000-000"
VIEW = "035-045"
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m28_gauge.json")

#: item 5's sample: one object per class, in class order (fixed here, before any number).
SAMPLE = ("apple/1", "car/1", "cow/1", "cup/1", "dog/1", "horse/1", "pear/1", "tomato/1")
PROCEED_GAIN = 0.10

# ---- constants of the candidate content (fixed here, before any number) -------------------
RINGS = (0.5, 1.5)              # radii as multiples of the place's own scale
RING_SAMPLES = 12               # angular samples per ring
ORIENT_N = 8                    # Gabor orientations
ORIENT_SF_INDEX = (1, 3)        # of the existing bank's 5 geo-spaced frequencies
WINDOW_RADIUS = 2.0             # disc radius, in units of the place's own scale
LENSES = ("Blob", "BoundaryArc", "CornerPoint")

_EPS = 1e-9


# --------------------------------------------------------------------------- images and places
def _places(view: str, objs: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Places per object at one view, from the committed export (no rebuild)."""
    from matching import placement_records as PR
    from Topology import m26_records as REC
    out: Dict[str, Dict[str, Any]] = {}
    for k in objs:
        out[k] = REC.load(k, view)
    return out


def _lab(image_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIELAB (D65), float64.  Fixed math, no fitting."""
    rgb = image_u8.astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _features_foreground(path: str, places: Sequence[Dict[str, Any]],
                         mask_path: str) -> np.ndarray:
    """DIAGNOSTIC ONLY (not a candidate): C1's rings sampled on FOREGROUND pixels only.

    The mission's rule was applied to C1 as specified; this variant answers a different question,
    asked after the fact -- is the colour signal the OBJECT's surface or the scene's colour near the
    place?  ET
H-80 ships a per-pixel part map beside every image (`maps/<name>-<view>-map.png`), background
    exactly 0, so a ring sample that lands on the background is dropped."""
    from PIL import Image
    from matching.placement_frontend import load_image
    from matching.placement_content import bilinear, lab, unit
    img = load_image(path)
    lab_img = lab(img)
    m = np.asarray(Image.open(mask_path).convert("RGB"))
    fg = (m.max(axis=2) > 0)
    n = len(places)
    out = np.zeros((n, 6), dtype=np.float64)
    ang = np.linspace(0.0, 2 * np.pi, RING_SAMPLES, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    for i, p in enumerate(places):
        s = max(float(p.get("s") or 0.0), 0.75)
        for r_i, r in enumerate(RINGS):
            px = float(p["x"]) + r * s * ca
            py = float(p["y"]) + r * s * sa
            ok = bilinear(fg.astype(np.float64), px, py) > 0.5
            if int(ok.sum()) < 2:
                continue
            v = bilinear(lab_img, px[ok], py[ok]).mean(axis=0)
            out[i, 3 * r_i:3 * r_i + 3] = (v[0] / 100.0, v[1] / 128.0, v[2] / 128.0)
    return unit(out)


def _gabor_bank(scale: float = 1.0) -> List[Tuple[float, float, np.ndarray]]:
    """8 orientations x 2 frequencies, built by the EXISTING bank's own kernel and fractions."""
    import parameters as P
    from perception.filters import _gabor_kernel
    sfs = np.geomspace(float(P.GABOR_SF_MIN) / scale, float(P.GABOR_SF_MAX) / scale,
                       int(P.GABOR_N_SF))
    out: List[Tuple[float, float, np.ndarray]] = []
    for sf in [float(sfs[i]) for i in ORIENT_SF_INDEX]:
        sx = float(P.GABOR_SIGMA_X_FRAC) / sf
        sy = sx / float(P.GABOR_ASPECT)
        size = int(min(int(P.GABOR_KERNEL_MAX), max(7, 8.0 * max(sx, sy) + 1)))
        size += (1 - size % 2)          # odd: _conv2d's same-size convention
        for it in range(ORIENT_N):
            theta = math.pi * it / ORIENT_N
            out.append((theta, sf, _gabor_kernel(theta, sf, 0.0, sx, sy, size)))
    return out


def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a (H, W[, C]) array at float coordinates, clamped at the border."""
    H, W = a.shape[0], a.shape[1]
    x = np.clip(x, 0.0, W - 1.001)
    y = np.clip(y, 0.0, H - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    a00 = a[y0, x0]
    a01 = a[y0, x1]
    a10 = a[y1, x0]
    a11 = a[y1, x1]
    w00 = (1 - fx) * (1 - fy)
    w01 = fx * (1 - fy)
    w10 = (1 - fx) * fy
    w11 = fx * fy
    if a.ndim == 3:
        w00, w01, w10, w11 = (w[..., None] for w in (w00, w01, w10, w11))
    return a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11


def _features(path: str, places: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """C1, C2, C3, C4 for every place of one image (rows follow ``places``)."""
    from matching.placement_frontend import load_image
    from perception.filters import _conv2d
    img = load_image(path)
    lab = _lab(img)
    H, W = img.shape[:2]
    n = len(places)
    xs = np.asarray([p["x"] for p in places], dtype=np.float64)
    ys = np.asarray([p["y"] for p in places], dtype=np.float64)
    ss = np.asarray([p["s"] for p in places], dtype=np.float64)
    ts = np.asarray([0.0 if p.get("theta") is None else float(p["theta"]) for p in places])
    ss = np.maximum(ss, 0.75)

    # ---- C1: two concentric rings of CIELAB
    C1 = np.zeros((n, 6), dtype=np.float64)
    ang = np.linspace(0.0, 2 * np.pi, RING_SAMPLES, endpoint=False)
    for i in range(n):
        for r_i, r in enumerate(RINGS):
            rr = r * ss[i]
            sx = xs[i] + rr * np.cos(ang)
            sy = ys[i] + rr * np.sin(ang)
            v = _bilinear(lab, sx, sy).mean(axis=0)
            C1[i, 3 * r_i:3 * r_i + 3] = (v[0] / 100.0, v[1] / 128.0, v[2] / 128.0)

    # ---- C2/C3: Gabor energies in the place's own window, rotated to its own axis
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)
    resp = []
    for _theta, _sf, ker in _gabor_bank(1.0):
        r = np.asarray(_conv2d(gray, ker), dtype=np.float64)
        resp.append(r[:H, :W] if r.shape[0] > H or r.shape[1] > W
                    else np.pad(r, ((0, H - r.shape[0]), (0, W - r.shape[1])),
                                mode='edge'))
    R = np.asarray(resp).reshape(ORIENT_N, len(ORIENT_SF_INDEX), H, W)
    C2 = np.zeros((n, ORIENT_N * len(ORIENT_SF_INDEX)), dtype=np.float64)
    C3 = np.zeros((n, 2), dtype=np.float64)
    # a fixed sample pattern inside the disc: 8 angles x 3 radii
    disc = [(r * WINDOW_RADIUS, a) for r in (0.5, 1.0, 1.6)
            for a in np.linspace(0.0, 2 * np.pi, 8, endpoint=False)]
    for i in range(n):
        px = np.asarray([xs[i] + r * ss[i] * np.cos(a) for r, a in disc])
        py = np.asarray([ys[i] + r * ss[i] * np.sin(a) for r, a in disc])
        e = np.zeros((ORIENT_N, len(ORIENT_SF_INDEX)), dtype=np.float64)
        for oi in range(ORIENT_N):
            for si in range(len(ORIENT_SF_INDEX)):
                e[oi, si] = np.abs(_bilinear(R[oi, si], px, py)).mean()
        # roll so that bin 0 is the place's own axis
        shift = int(round((ts[i] % np.pi) / (np.pi / ORIENT_N))) % ORIENT_N
        e = np.roll(e, -shift, axis=0)
        flat = e.reshape(-1)
        tot = float(flat.sum())
        C2[i] = flat / tot if tot > _EPS else flat
        C3[i] = (float(np.std(C2[i].reshape(ORIENT_N, -1).sum(axis=1))), math.log1p(tot))

    def _unit(a: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(nrm, _EPS)

    C4 = _unit(np.concatenate([_unit(C1), _unit(C2), _unit(C3)], axis=1))
    return {"C1": _unit(C1), "C2": _unit(C2), "C3": _unit(C3), "C4": C4}


# --------------------------------------------------------------------------- the gauge
def _pool(records: Dict[str, Dict[str, Any]], obj_keys: Sequence[str], feature: str,
          feats: Optional[Dict[str, np.ndarray]] = None
          ) -> Dict[str, Dict[str, Any]]:
    """Per lens: the pool matrix, its object index, and the object order."""
    mats: Dict[str, List[np.ndarray]] = {l: [] for l in LENSES}
    objs: Dict[str, List[int]] = {l: [] for l in LENSES}
    if feature == "baseline":
        keys: Dict[str, List[Tuple[float, ...]]] = {l: [] for l in LENSES}
        for oi, k in enumerate(obj_keys):
            for p in records[k]["places"]:
                if p["lens"] in keys:
                    keys[p["lens"]].append(tuple(float(x) for x in p["key"][1]))
                    objs[p["lens"]].append(oi)
        return {"kind": "sig", "keys": keys, "obj": {l: np.asarray(objs[l], dtype=np.int64)
                                                     for l in LENSES},
                "n_obj": len(obj_keys)}
    for oi, k in enumerate(obj_keys):
        f = feats[k][feature] if isinstance(feats[k], dict) else feats[k]
        lens = [p["lens"] for p in records[k]["places"]]
        order = np.arange(len(lens))
        for l in LENSES:
            sel = order[[i for i, x in enumerate(lens) if x == l]]
            if sel.size:
                mats[l].append(f[sel])
                objs[l].extend([oi] * int(sel.size))
    return {"kind": "vec", "mats": {l: (np.concatenate(mats[l], axis=0) if mats[l]
                                        else np.zeros((0, 1))) for l in LENSES},
            "obj": {l: np.asarray(objs[l], dtype=np.int64) for l in LENSES},
            "n_obj": len(obj_keys)}


def _best_per_object(sim: np.ndarray, obj: np.ndarray, n_obj: int) -> np.ndarray:
    out = np.full(n_obj, -np.inf)
    np.maximum.at(out, obj, sim)
    return out


def _gauge_one_lens(qlens: np.ndarray, qobj: np.ndarray, qrows: List[np.ndarray],
                    pool: Dict[str, Any], lens: str, feature: str,
                    chunk: int = 256) -> List[Tuple[Optional[float], Optional[float]]]:
    """AUC rows for the query places of ONE lens: block similarity, in chunks."""
    from matching import placement_kernel as K
    if pool["kind"] == "sig":
        P = np.asarray(pool["keys"][lens], dtype=np.float64)
    else:
        P = pool["mats"][lens]
    if P.shape[0] == 0:
        return [(None, None)] * len(qrows)
    pobj = pool["obj"][lens]
    out: List[Tuple[Optional[float], Optional[float]]] = []
    same_of = pool.get("same", {})
    for start in range(0, len(qrows), chunk):
        part = qrows[start:start + chunk]
        Q = np.asarray(part, dtype=np.float64)
        S = (K.similarity_block(lens, Q, P) if pool["kind"] == "sig" else Q @ P.T)
        for r in range(Q.shape[0]):
            obji = int(qobj[start + r])
            best = _best_per_object(S[r], pobj, pool["n_obj"])
            out.append((_auc(best, obji), _auc(best, obji, restrict=same_of.get(obji, []))))
    return out


def _auc(best: np.ndarray, truth: int, restrict: Optional[Sequence[int]] = None
         ) -> Optional[float]:
    s_true = best[truth]
    if not np.isfinite(s_true):
        return None
    idx = [j for j in range(len(best)) if j != truth
           and np.isfinite(best[j]) and (restrict is None or j in restrict)]
    if not idx:
        return None
    s_other = best[np.asarray(idx)]
    return float(np.mean((s_true > s_other) + 0.5 * (s_true == s_other)))


def gauge(query_keys: Sequence[str], pool_keys: Sequence[str], feature: str,
          feats: Optional[Dict[str, np.ndarray]] = None,
          qfeats: Optional[Dict[str, np.ndarray]] = None,
          queries: Optional[Dict[str, Dict[str, Any]]] = None,
          poolrecs: Optional[Dict[str, Dict[str, Any]]] = None
          ) -> Dict[str, Any]:
    """G_any and G_same for one feature kind over the given query objects."""
    from matching import placement_kernel as K
    qrecs = queries if queries is not None else _places(VIEW, query_keys)
    precs = poolrecs if poolrecs is not None else _places(REF, pool_keys)
    pool = _pool(precs, pool_keys, feature, feats)
    cls_of = {k: k.split("/")[0] for k in pool_keys}
    idx_of = {k: i for i, k in enumerate(pool_keys)}
    pool["same"] = {i: [idx_of[k] for k in pool_keys
                        if cls_of[k] == cls_of[qk] and k != qk]
                    for i, qk in ((idx_of[k], k) for k in query_keys)}
    any_scores: List[float] = []
    same_scores: List[float] = []
    per_lens_counts: Dict[str, int] = {}
    qrecs = {k: qrecs[k] for k in query_keys}
    for k in query_keys:
        by_lens: Dict[str, List[int]] = {}
        for j, p in enumerate(qrecs[k]["places"]):
            if p["lens"] in LENSES:
                by_lens.setdefault(p["lens"], []).append(j)
        for lens, rows in by_lens.items():
            if pool["kind"] == "sig":
                Q = [np.asarray(qrecs[k]["places"][j]["key"][1], dtype=np.float64) for j in rows]
            else:
                src = qfeats if qfeats is not None else feats
                fk = src[k][feature] if isinstance(src[k], dict) else src[k]
                Q = [fk[j] for j in rows]
            res = _gauge_one_lens(None, np.full(len(rows), idx_of[k], dtype=np.int64), Q,
                                  pool, lens, feature)
            for a, sm in res:
                if a is not None:
                    any_scores.append(a)
                    per_lens_counts[lens] = per_lens_counts.get(lens, 0) + 1
                if sm is not None:
                    same_scores.append(sm)
    return {"feature": feature, "queries": list(query_keys), "pool": len(pool_keys),
            "query_places": int(len(any_scores)),
            "G_any": float(np.mean(any_scores)) if any_scores else None,
            "G_same": float(np.mean(same_scores)) if same_scores else None,
            "n_same_pairs": int(len(same_scores)),
            "places_by_lens": per_lens_counts, "chance": 0.5}


# --------------------------------------------------------------------------- the two items
def baseline() -> Dict[str, Any]:
    from Topology import m26_records as REC
    objs = [o["key"] for o in REC.objects()]
    t0 = time.time()
    all80 = gauge(objs, objs, "baseline")
    sub = gauge(list(SAMPLE), objs, "baseline")
    out = {"item": 4, "all_objects": all80, "sample_of_8": sub,
           "seconds": round(time.time() - t0, 1),
           "manifest": {"environment": _env()}}
    print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=1))
    return out


def candidates() -> Dict[str, Any]:
    from Topology import m26_records as REC
    from matching import placement_records as PR
    objs = [o["key"] for o in REC.objects()]
    t0 = time.time()
    feats: Dict[str, Dict[str, np.ndarray]] = {}
    for k in objs:
        feats[k] = _features(REC.path_of(REC.objects()[objs.index(k)], REF),
                             REC.load(k, REF)["places"])
    qfeats: Dict[str, Dict[str, np.ndarray]] = {}
    for k in SAMPLE:
        qfeats[k] = _features(REC.path_of(REC.objects()[objs.index(k)], VIEW),
                              REC.load(k, VIEW)["places"])
    build = time.time() - t0
    res: Dict[str, Any] = {"item": 5, "sample": list(SAMPLE), "constants": {
        "rings": RINGS, "ring_samples": RING_SAMPLES, "orient_n": ORIENT_N,
        "orient_sf_index": ORIENT_SF_INDEX, "window_radius": WINDOW_RADIUS},
        "build_seconds": round(build, 1), "results": {}}
    base = gauge(list(SAMPLE), objs, "baseline")
    res["results"]["baseline"] = base
    for kind in ("C1", "C2", "C3", "C4"):
        res["results"][kind] = gauge(list(SAMPLE), objs, kind, feats=feats, qfeats=qfeats)
        r = res["results"][kind]
        print("  %-8s G_any %.3f  G_same %.3f" % (kind, r["G_any"], r["G_same"]), flush=True)
    b = res["results"]["baseline"]["G_same"]
    verdict: Dict[str, Any] = {}
    for kind in ("C1", "C2", "C3", "C4"):
        g = res["results"][kind]["G_same"]
        verdict[kind] = {"delta_same_class": None if (g is None or b is None) else g - b,
                         "proceeds": bool(g is not None and b is not None
                                          and (g - b) >= PROCEED_GAIN)}
    res["verdict"] = verdict
    res["baseline_same_class"] = b
    res["any_proceeds"] = any(v["proceeds"] for v in verdict.values())
    res["seconds"] = round(time.time() - t0, 1)
    res["manifest"] = {"environment": _env()}
    print("\nbaseline G_same %.3f; verdict %s" % (b, json.dumps(verdict)), flush=True)
    return res


def diagnostic() -> Dict[str, Any]:
    """POST-HOC DIAGNOSTIC, not a registered candidate: is C1 the object or the background?"""
    from Topology import m26_records as REC
    objs = [o["key"] for o in REC.objects()]
    t0 = time.time()
    feats: Dict[str, np.ndarray] = {}
    for k in objs:
        o = REC.objects()[objs.index(k)]
        img_path = REC.path_of(o, REF)
        mask = os.path.join(os.path.dirname(img_path), "maps",
                            os.path.basename(img_path).replace(".png", "-map.png"))
        feats[k] = _features_foreground(img_path, REC.load(k, REF)["places"], mask)
    qfeats: Dict[str, np.ndarray] = {}
    for k in SAMPLE:
        o = REC.objects()[objs.index(k)]
        img_path = REC.path_of(o, VIEW)
        mask = os.path.join(os.path.dirname(img_path), "maps",
                            os.path.basename(img_path).replace(".png", "-map.png"))
        qfeats[k] = _features_foreground(img_path, REC.load(k, VIEW)["places"], mask)
    r = gauge(list(SAMPLE), objs, "C1f", feats=feats, qfeats=qfeats)
    base = gauge(list(SAMPLE), objs, "baseline")
    out = {"item": "5 diagnostic (post-hoc, NOT a candidate)", "C1_foreground_only": r,
           "baseline": base,
           "delta_same_class_vs_signature": r["G_same"] - base["G_same"],
           "seconds": round(time.time() - t0, 1),
           "manifest": {"environment": _env()}}
    print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=1))
    return out


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    data: Dict[str, Any] = {}
    path = OUT
    if a.baseline or a.candidates or a.diagnostic:
        if os.path.exists(path):
            data = json.load(open(path))
        if a.baseline:
            data["baseline"] = baseline()
        if a.candidates:
            data["candidates"] = candidates()
        if a.diagnostic:
            data["diagnostic_foreground"] = diagnostic()
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        print("wrote", path)
    if not (a.baseline or a.candidates or a.diagnostic):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
