"""MISSION 12 STEP 2 -- the frame validation, done properly, and the provider chosen
by the DEPLOYED MATCHER rather than by an angle.

THE OLD TEST WAS WRONG and the mission says so: it sampled the provider at the SAME
pixel anchors in the original and in the rotated image, so it measured "does the
field change when the picture moves", never "does the frame move WITH the picture".
A static provider would have scored perfectly on it.

THE TEST HERE
  * the image is rotated about its centre by delta (scipy.ndimage.rotate, order 1,
    mode nearest);
  * the node ANCHORS are rotated by the SAME transform about the same centre, and
    the SIGN CONVENTION is CALIBRATED on the shipped field provider: the script runs
    both signs and keeps the one where the field's signed residual is centred on
    zero, and it records which sign that was;
  * the provider is sampled at the TRANSFORMED anchor (each provider with its own
    sampling: nearest for the box providers, bilinear for the keypoint one);
  * the residual is SIGNED and wrapped to the axial range:
        r = wrap(angle_rotated(anchor') - angle_original(anchor) - delta)
  * anchors whose transformed position falls outside the image or inside a border
    of BORDER px are EXCLUDED (a border sample reads padded edge, which is not the
    photograph);
  * reported per provider and per structure class: median signed residual, MAD, and
    the share within 3 degrees -- for delta = 5 and 15 degrees, on synthetic scenes
    and on the six REAL PHOTOGRAPHS reserved for exactly this.

THEN the provider is chosen by the deployed matcher on synthetic FIT: correspondence
accuracy on the near-tie bank and on the unambiguous bank (frozen rules), the
judged-pair share, paired per scene with a bootstrap interval.  A provider is adopted
only if that interval excludes zero in its favour; otherwise the shipped provider
stays and the result says so.  Everything stays dark.

    python Topology/placement_m12_frames2.py --workers 1
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m12_frames2.json")
DELTAS = (5.0, 15.0)
BORDER = 8
IMG = 256
N_SYNTH = 6
NBOOT, SEED = 4000, 20261118


# ---------------------------------------------------------------------------
# the transform, shared by the image and the anchors
# ---------------------------------------------------------------------------
def _rotate_image(img: np.ndarray, delta_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate
    a = np.asarray(img, dtype=np.float64)
    out = np.stack([rotate(a[..., c], delta_deg, reshape=False, order=1,
                           mode="nearest") for c in range(a.shape[2])], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rotate_anchors(pts: np.ndarray, delta_deg: float, sign: float) -> np.ndarray:
    """Rotate (x, y) pixels about the image centre by `sign * delta_deg`.

    CONVENTION, stated once and used by both sides: the image rotation is
    ``scipy.ndimage.rotate(img, +delta)``, which rotates the ARRAY (row, col) by the
    standard rotation matrix; the anchor map that goes with it is therefore
    ``(x, y) -> c + R(-sign*delta) (p - c)`` in (col, row) order and the sign is
    CALIBRATED on the shipped field provider rather than argued about here.
    """
    c = (IMG - 1) / 2.0
    th = np.deg2rad(sign * float(delta_deg))
    ct, st = float(np.cos(th)), float(np.sin(th))
    d = np.asarray(pts, dtype=float) - c
    x = ct * d[:, 0] - st * d[:, 1]
    y = st * d[:, 0] + ct * d[:, 1]
    return np.stack([x + c, y + c], axis=1)


def _wrap_axial(a: float) -> float:
    """Wrap an angle difference into (-pi/2, pi/2] (frames are axial)."""
    d = float(a) % np.pi
    if d > np.pi / 2.0:
        d -= np.pi
    return d


def _sample(field: np.ndarray, x: float, y: float, bilinear: bool) -> Any:
    h, w = field.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    if not bilinear:
        return float(field[int(round(y)), int(round(x))])
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    return float(field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
                 + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


# ---------------------------------------------------------------------------
# providers as FIELDS (the same estimators the payload builder uses)
# ---------------------------------------------------------------------------
def _fields(image_u8: np.ndarray) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, bool]]]:
    """provider -> LAYERS of (angle field, coherence field, bilinear?), smallest scale first.

    * ``field``      : the deployed 7x7 box-window tensor (`field_angles`), nearest sampling.
    * ``multiscale`` : the DEPLOYED multi-window ladder (`field_multiscale` over
      `FIELD_WINDOWS`); at each anchor the smallest window whose coherence clears
      `FRAME_COHERENCE_MIN` wins, exactly as `node_frames_multiscale` does.
    * ``keypoint``   : the Gaussian tensor at the largest keypoint sigma, bilinear.
    """
    from matching import placement_edges as PE
    ang, coh = PE.field_angles(image_u8)
    ms = [(a, c) for (_w, a, c) in PE.field_multiscale(image_u8)]
    img = np.asarray(image_u8, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gy, gx = np.gradient(img)
    sig = max(PE.KEYPOINT_SIGMAS)
    kp_ang, kp_coh = PE._tensor_at_sigma(gx, gy, float(sig))
    return {"field": [(ang, coh, False)],
            "multiscale": [(a, c, False) for (a, c) in ms],
            "keypoint": [(kp_ang, kp_coh, True)]}


def _pick_layer(layers, x0, y0, x1, y1, cmin):
    """Smallest layer where BOTH anchors clear the coherence bar (deployment rule)."""
    for i, (a, c, bil) in enumerate(layers):
        v0 = _sample(a, x0, y0, bil); v1 = _sample(a, x1, y1, bil)
        k0 = _sample(c, x0, y0, bil); k1 = _sample(c, x1, y1, bil)
        if None in (v0, v1, k0, k1):
            continue
        if k0 >= cmin and k1 >= cmin:
            return i, float(v0), float(v1)
    return None, None, None


def _anchor_residuals(image_u8: np.ndarray, anchors: np.ndarray, delta: float,
                      sign: float) -> Dict[str, Dict[str, Any]]:
    """Per provider: the SIGNED residuals over the anchors that survive the transform."""
    from matching import placement_edges as PE
    cmin = float(PE.FRAME_COHERENCE_MIN)
    base = _fields(image_u8)
    rot = _fields(_rotate_image(image_u8, delta))
    moved = _rotate_anchors(anchors, delta, sign)
    th = np.deg2rad(float(delta))
    out: Dict[str, Dict[str, Any]] = {}
    for name, layers0 in base.items():
        layers1 = rot[name]
        res: List[float] = []
        used: Dict[int, int] = {}
        for (x0, y0), (x1, y1) in zip(anchors, moved):
            if not (BORDER <= x0 < IMG - BORDER and BORDER <= y0 < IMG - BORDER):
                continue
            if not (BORDER <= x1 < IMG - BORDER and BORDER <= y1 < IMG - BORDER):
                continue
            i, v0, _ = _pick_layer(layers0, float(x0), float(y0), float(x1), float(y1), cmin)
            if i is None:
                continue
            j, _, v1 = _pick_layer(layers1, float(x0), float(y0), float(x1), float(y1), cmin)
            if j is None:
                continue
            r = _wrap_axial(float(v1) - float(v0) - sign * th)
            res.append(float(np.degrees(r)))
            used[i] = used.get(i, 0) + 1
        arr = np.asarray(res, dtype=float)
        out[name] = {"n": int(arr.size),
                     "median_signed_deg": float(np.median(arr)) if arr.size else float("nan"),
                     "mad_deg": float(np.median(np.abs(arr - np.median(arr)))) if arr.size else float("nan"),
                     "share_within_3deg": float(np.mean(np.abs(arr) <= 3.0)) if arr.size else float("nan"),
                     "layer_hist": {str(k): int(v) for k, v in sorted(used.items())}}
    return out


def _calibrate(image_u8: np.ndarray, anchors: np.ndarray, delta: float) -> Tuple[float, Dict[str, Any]]:
    """Which anchor-rotation sign makes the SHIPPED field provider track?"""
    best = None
    detail = {}
    for sign in (+1.0, -1.0):
        r = _anchor_residuals(image_u8, anchors, delta, sign)
        med = r["field"]["median_signed_deg"]
        detail["sign%+.0f" % sign] = med
        if best is None or abs(med) < abs(best[0]):
            best = (med, sign)
    return best[1], detail


# ---------------------------------------------------------------------------
# graphs, for both image families
# ---------------------------------------------------------------------------
def _synth_images() -> List[Tuple[str, np.ndarray, Any]]:
    from Topology import placement_m11_families as F
    from Topology.placement_cache import get_or_build
    gal = [s for s in F.gallery() if s.family == "IRREGULAR"][:N_SYNTH]
    out = []
    for i, s in enumerate(gal):
        lab, view = F.VIEW_STEPS[i % 2]
        sc, _pv = F.pose_fn(s, view)
        img = sc.render().image_u8
        g = get_or_build(img, s.name + "|" + lab)
        out.append(("syn|" + s.name, img, g))
    return out


def _real_images() -> List[Tuple[str, np.ndarray, Any]]:
    from Topology import placement_m12_realphotos as RP
    from Topology.placement_cache import get_or_build
    out = []
    for name, img in sorted(RP.images("validation").items()):
        g = get_or_build(img, "m12frame|" + name)
        out.append(("real|" + name, img, g))
    return out


def _anchors(g: Any) -> np.ndarray:
    sub = g.subgraph
    pts = [(float(sub.anchor_of_node[n][0]), float(sub.anchor_of_node[n][1]))
           for n in sorted(sub.nodes) if n in sub.anchor_of_node]
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def _validation() -> Dict[str, Any]:
    imgs = _synth_images() + _real_images()
    rows: List[Dict[str, Any]] = []
    calib: List[Dict[str, Any]] = []
    for tag, img, g in imgs:
        anchors = _anchors(g)
        if len(anchors) < 8:
            continue
        for delta in DELTAS:
            sign, detail = _calibrate(img, anchors, delta)
            calib.append({"tag": tag, "delta": delta, "chosen_sign": sign,
                          "field_median_by_sign": detail})
            r = _anchor_residuals(img, anchors, delta, sign)
            rows.append({"tag": tag, "delta": delta, "sign": sign,
                         "class": ("synthetic" if tag.startswith("syn")
                                   else ("real:" + tag.split("|")[1])),
                         "providers": r})
    agg: Dict[str, Any] = {}
    for prov in ("field", "multiscale", "keypoint"):
        for kind in ("synthetic", "real", "all"):
            sub = [x for x in rows if kind == "all" or x["class"].startswith(kind)]
            med = [x["providers"][prov]["median_signed_deg"] for x in sub
                   if x["providers"][prov]["n"] > 0]
            mad = [x["providers"][prov]["mad_deg"] for x in sub
                   if x["providers"][prov]["n"] > 0]
            w3 = [x["providers"][prov]["share_within_3deg"] for x in sub
                  if x["providers"][prov]["n"] > 0]
            if not med:
                continue
            agg["%s|%s" % (prov, kind)] = {
                "images": len(med),
                "median_signed_deg": float(np.median(med)),
                "median_mad_deg": float(np.median(mad)),
                "mean_share_within_3deg": float(np.mean(w3)),
            }
    # per STRUCTURE CLASS: one line per scene / photograph, deltas pooled
    by_struct: Dict[str, Any] = {}
    for tag in sorted({x["tag"] for x in rows}):
        sub = [x for x in rows if x["tag"] == tag]
        ent: Dict[str, Any] = {"deltas": [x["delta"] for x in sub],
                               "class": sub[0]["class"]}
        for prov in ("field", "multiscale", "keypoint"):
            med = [x["providers"][prov]["median_signed_deg"] for x in sub
                   if x["providers"][prov]["n"] > 0]
            w3 = [x["providers"][prov]["share_within_3deg"] for x in sub
                  if x["providers"][prov]["n"] > 0]
            mad = [x["providers"][prov]["mad_deg"] for x in sub
                   if x["providers"][prov]["n"] > 0]
            n = int(sum(x["providers"][prov]["n"] for x in sub))
            if not med:
                continue
            ent[prov] = {"n": n,
                         "median_abs_signed_deg": float(np.median(np.abs(med))),
                         "median_mad_deg": float(np.median(mad)),
                         "mean_share_within_3deg": float(np.mean(w3))}
        by_struct[tag] = ent
    return {"rows": rows, "calibration": calib, "aggregate": agg,
            "by_structure": by_struct}


# ---------------------------------------------------------------------------
# the provider choice: the DEPLOYED MATCHER on synthetic FIT
# ---------------------------------------------------------------------------
def _fit_pairs() -> List[Tuple[str, Any, Any]]:
    from Topology import placement_m11_families as F
    gal = [s for s in F.gallery() if s.family == "IRREGULAR"][:8]
    out = []
    for i, s in enumerate(gal):
        lab, view = F.VIEW_STEPS[i % 2]
        out.append((s.name + "|" + lab, s, view))
    return out


def _matcher_job(args: Tuple[str, Any, Any, bool]) -> Dict[str, Any]:
    """One FIT pair under one frame provider: near-tie and unambiguous accuracy."""
    tag, spec, view, keypoint = args
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology import placement_census as CEN
    from Topology import placement_m11_families as F
    from Topology.placement_cache import get_or_build
    from Topology.placement_gate import appearance_ambiguous
    from Topology.placement_truth import node_objects, track_optimal

    sc1, _pv = F.pose_fn(spec, view)
    r0, r1 = spec.scene.render(), sc1.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    g1 = get_or_build(r1.image_u8, spec.name + "|" + tag.split("|")[1])
    truth = {int(q): int(r)
             for r, q in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    o0, o1 = node_objects(r0, g0), node_objects(r1, g1)
    refc = [v for v in sorted(int(n) for n in g0.subgraph.nodes) if o0.get(v, -1) >= 0]
    pop = [u for u in sorted(int(n) for n in g1.subgraph.nodes) if o1.get(u, -1) >= 0]
    rows = CEN.similarity_rows(g1, g0, pop, refc)
    amb = appearance_ambiguous(g1.sigs, g0.sigs, sims={u: rows[u] for u in rows})

    P.PLACEMENT_FRAME_KEYPOINT = bool(keypoint)
    P.PLACEMENT_E_NORMALISED = True
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                    placement_images=(r1.image_u8, r0.image_u8))
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_FRAME_KEYPOINT = False
    mp = {int(u): int(v) for u, v in res.mapping.items()}

    def acc(nodes: List[int]) -> Tuple[int, int]:
        hit = tot = 0
        for u in nodes:
            if u not in truth:
                continue
            tot += 1
            hit += 1 if mp.get(u) == truth[u] else 0
        return hit, tot

    h_amb, n_amb = acc([u for u in pop if amb.get(u)])
    h_un, n_un = acc([u for u in pop if not amb.get(u)])
    return {"tag": tag, "keypoint": bool(keypoint),
            "amb_hit": h_amb, "amb_n": n_amb, "un_hit": h_un, "un_n": n_un,
            "judged": int(res.placement_judged), "share": float(res.placement_share_judged),
            "mapped": int(len(res.mapping))}


def _provider_decision(workers: int) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    pairs = _fit_pairs()
    jobs = [(t, s, v, kp) for (t, s, v) in pairs for kp in (False, True)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_matcher_job, jobs))

    def summarise(kp: bool) -> Dict[str, float]:
        sub = [r for r in rows if r["keypoint"] is kp]
        return {"amb_acc": float(sum(r["amb_hit"] for r in sub) / max(1, sum(r["amb_n"] for r in sub))),
                "un_acc": float(sum(r["un_hit"] for r in sub) / max(1, sum(r["un_n"] for r in sub))),
                "amb_n": sum(r["amb_n"] for r in sub), "un_n": sum(r["un_n"] for r in sub),
                "judged_share": float(np.mean([r["share"] for r in sub])),
                "judged_mean": float(np.mean([r["judged"] for r in sub]))}

    base, kp = summarise(False), summarise(True)
    diffs = []
    scenes = []
    for t, _s, _v in pairs:
        a = [r for r in rows if r["tag"] == t and not r["keypoint"]][0]
        b = [r for r in rows if r["tag"] == t and r["keypoint"]][0]
        if a["amb_n"]:
            diffs.append(b["amb_hit"] / b["amb_n"] - a["amb_hit"] / a["amb_n"])
            scenes.append(t.split("|")[0])
    ci = {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else float("nan")}
    if diffs:
        rng = np.random.default_rng(SEED)
        draws = sorted(float(np.mean([diffs[i] for i in rng.integers(0, len(diffs), len(diffs))]))
                       for _ in range(NBOOT))
        ci["lo"] = draws[int(0.025 * len(draws))]
        ci["hi"] = draws[int(0.975 * len(draws)) - 1]
    adopt = bool(ci.get("lo", -1) > 0.0)
    return {"fit_pairs": len(pairs), "shipped_field": base, "keypoint_field": kp,
            "amb_paired_diff": ci, "adopted": adopt,
            "rule": ("adopt the keypoint provider only if the paired FIT interval for "
                     "near-tie correspondence accuracy excludes zero in its favour")}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    import guard
    guard.activate()
    val = _validation()
    dec = _provider_decision(args.workers)
    res = {"validation": val, "provider_decision": dec,
           "note": ("residuals are SIGNED: angle in the rotated image minus angle in "
                    "the original minus the rotation, wrapped to the axial range; "
                    "anchors outside the image or inside the %d px border are "
                    "excluded; the anchor-rotation sign is calibrated on the shipped "
                    "field provider per image and delta" % BORDER)}
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps({"aggregate": val["aggregate"],
                      "provider_decision": dec}, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
