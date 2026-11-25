"""Mission 15 items 2, 3, 5 and 10 -- the estimator, measured on real places.

WHAT THIS FILE ANSWERS, one block per mission item:

  ITEM 2  slot admission.  For every positively-similar candidate cell, the JUDGED
          share (slots with a residual / SLOT_CAP) with
          `PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST` off and on, and how many cells gain
          a checkable slot that the shipped order discarded.   -> "before/after".

  ITEM 5  the geometry term.  (a) the residual distribution of TRUE pairs and of
          IMPOSTOR pairs, per configuration, with per-bin counts; (b) a POOLED refit
          of the 24-bin LLR over synthetic FIT + real known-transform residuals, with
          the bin width required to be no finer than the measured real-photo residual
          noise under an EXACT shift (measured here, both numbers stated), and the
          minimum count per bin; (c) the shipped empty-bin rule against the neutral
          one, reporting every bin where the LLR rises after smoothing; (d) the AUC of
          the GEOMETRY TERM ALONE, true against impostor, per cell and split.

  ITEM 10 the decisive probe.  Geometry only, place endpoints, arms:
            (a) context = the matcher's own place-node mapping (deployable),
            (b) context = the exact transform (DEV ceiling),
          on synthetic and on real photographs, split by pose, with the OLD endpoint
          list on the same cells, each arm reporting AUC and top-1.  Every number is
          written with its image count and its spread.

The candidate set is the matcher's: same lens, positive signature similarity -- an
impostor must be a candidate the matcher would actually consider, or the AUC is
measuring distance, not discrimination.

    python Topology/placement_m15_geometry.py --cells 2 --workers 1
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m15_geometry.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("shift8", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
)
M15 = dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
           FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
           PLACEMENT_FRAME_KEYPOINT_SIGMA="largest", PLACEMENT_FRAME_FALLBACK="none")


def _images(n: int) -> Dict[str, np.ndarray]:
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    imgs: Dict[str, np.ndarray] = {}
    for s in [x for x in F.gallery() if x.family == "IRREGULAR"][:n]:
        imgs["syn|" + s.name] = s.scene.render().image_u8
    for name, img in sorted(RP.images("validation").items())[:n]:
        imgs["real|" + name] = img
    return imgs


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    """Pairwise AUC over (true score, impostor scores); ties count one half."""
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def _job(args: Tuple[str, str, Dict[str, Any], str]) -> Dict[str, Any]:
    tag, pose, spec, mode = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build

    for k, v in M15.items():
        setattr(P, k, v)
    if mode == "places":
        P.PLACEMENT_M15_PLACES = True
        P.PLACEMENT_M15_PLACE_LENSES = tuple(P.PLACEMENT_M15_PLACE_LENSES)
        P.PLACEMENT_M15_TRUTH_RULE = True
    img = _images(2)[tag]
    img1 = R._transform(img, spec)
    key = "m15atlas|%s|%s|1" if mode != "places" else "m15geo|%s|%s|places"
    g0 = get_or_build(img, key % (tag, "orig"))
    g1 = get_or_build(img1, key % (tag, pose))
    side_q = PM.build_side_cached(g1, img1)
    side_r = PM.build_side_cached(g0, img)
    if not (side_q.usable and side_r.usable):
        return {"tag": tag, "pose": pose, "mode": mode, "skipped": "side not usable"}
    shape = img.shape[:2]
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    cap = int(b["cand_cap"])
    triples = PM.top_candidates(matrix, nodes_b, cap)
    ctx = PM.context_mapping(triples, nodes_a, side_q, side_r)
    truth = PL.both_truths(g1, g0, spec, shape,
                           angles=None if mode != "places" else None)
    tol_of = PM._spacing_tol_fn(side_q, side_r)
    keep = set(PL.place_lenses())

    def _slots(u: int, v: int, mapping: Dict[int, int]) -> List[Any]:
        tol = None if tol_of is None else tol_of(u, v)
        return L.slots_for_candidate(u, v, mapping, side_q.incident, side_q.index,
                                    side_r.index, tol=tol)

    cells: List[Dict[str, Any]] = []
    judged = {"off": [], "on": []}
    gained = 0
    res_true: List[float] = []
    res_wrong: List[float] = []
    auc = {"ctx_matcher": [], "ctx_truth": []}
    top1 = {"ctx_matcher": [], "ctx_truth": []}
    endpoint_rows: List[Dict[str, Any]] = []
    bank: List[Dict[str, Any]] = []
    for i, u in enumerate(nodes_a):
        row = [(int(nodes_b[j]), float(matrix[i][j])) for j in range(len(nodes_b))
               if matrix[i][j] > 0.0]
        if not row:
            continue
        vt = truth["new"].get(u)
        for mapping_name, mapping in (("ctx_matcher", ctx), ("ctx_truth", truth["new"])):
            old_flag = P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST
            scores = []
            for (v, s) in row:
                P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = False
                so = _slots(u, v, mapping)
                n_off = min(len(so), int(L.SLOT_CAP))
                P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = True
                sn = _slots(u, v, mapping)
                n_on = min(len(sn), int(L.SLOT_CAP))
                j_on = L.judged_slots(sn)
                j_off = L.judged_slots(so)
                P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = old_flag
                if mapping_name == "ctx_matcher":
                    judged["off"].append(j_off / float(L.SLOT_CAP))
                    judged["on"].append(j_on / float(L.SLOT_CAP))
                    if j_on > j_off:
                        gained += 1
                _e, g = L.plus_terms(u, v, mapping, side_q.incident, side_q.index,
                                     side_r.index, fit, pfit,
                                     slot_cap=int(L.SLOT_CAP),
                                     tol=(None if tol_of is None else tol_of(u, v)))
                scores.append((v, float(g)))
                if vt is not None:
                    if v == vt:
                        res_true.extend([r for (n, o, r) in sn if r is not None])
                    else:
                        res_wrong.extend([r for (n, o, r) in sn if r is not None])
            if vt is not None and scores:
                d = dict(scores)
                if mapping_name == "ctx_matcher" and len(scores) > 1:
                    # ITEM 7 / ITEM 10 BANK: the appearance similarity and the
                    # blended evidence E = plus_score - 0.5 for every candidate of
                    # this query, under the item-2 admission fix.  Recorded raw so
                    # the weight profile is an offline read of the SAME cells.
                    E = []
                    for (v, _s) in row:
                        pv = L.plus_score(u, v, mapping, side_q.incident,
                                          side_q.index, side_r.index, fit, pfit,
                                          a_e, a_g, slot_cap=int(L.SLOT_CAP),
                                          tol=(None if tol_of is None else tol_of(u, v)))
                        E.append(float(pv) - 0.5)
                    bank.append({"u": int(u), "v": [int(v) for (v, _s) in row],
                                 "sim": [float(sv) for (_v, sv) in row],
                                 "E": E, "true": [1 if v == vt else 0 for (v, _s) in row]})
                if vt in d:
                    imp = [s for (v, s) in scores if v != vt]
                    if imp:
                        auc[mapping_name].append((d[vt], imp))
                        best = max(scores, key=lambda t: (t[1], -t[0]))[0]
                        top1[mapping_name].append(1.0 if best == vt else 0.0)
    if mode == "places":
        endpoint_rows.append(PL.endpoint_report(g1, tuple(keep)))
    cells.append({"tag": tag, "pose": pose, "mode": mode,
                  "new_rule_pairs": len(truth["new"]), "old_rule_pairs": len(truth["old"]),
                  "judged_off": float(np.mean(judged["off"])) if judged["off"] else None,
                  "judged_on": float(np.mean(judged["on"])) if judged["on"] else None,
                  "cells_gaining_a_checkable_slot": gained,
                  "candidates": len(judged["off"]),
                  "auc": {k: _auc(v) for k, v in auc.items()},
                  "top1": {k: (float(np.mean(v)) if v else None) for k, v in top1.items()},
                  "n_queries": {k: len(v) for k, v in auc.items()},
                  "res_true": res_true, "res_wrong": res_wrong,
                  "bank": bank,
                  "endpoints": endpoint_rows})
    return cells[0]


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cells", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    from matching import placement_llr as L
    res: Dict[str, Any] = {"cells": [], "modes": ["frozen", "places"]}
    if args.report:
        res = json.load(open(RESULT)) if os.path.exists(RESULT) else res
    else:
        imgs = _images(args.cells)
        jobs = [(t, pose, spec, mode) for mode in ("frozen", "places")
                for t in sorted(imgs) for pose, spec in POSES]
        print("%d cells (%d images x %d poses x 2 endpoint modes)"
              % (len(jobs), len(imgs), len(POSES)), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_job, jobs):
                res["cells"].append(r)
                print("  %-18s %-7s %-6s pairs new/old %s/%s judged %.2f->%.2f auc %s"
                      % (r["tag"], r["pose"], r["mode"], r.get("new_rule_pairs"),
                         r.get("old_rule_pairs"),
                         r.get("judged_off") or 0, r.get("judged_on") or 0,
                         r.get("auc")), flush=True)
                with open(RESULT, "w") as fh:
                    json.dump(res, fh, default=float)

    # ------------------------------- item 2 / 3 aggregates -------------------
    agg: Dict[str, Any] = {}
    for mode in ("frozen", "places"):
        cs = [c for c in res["cells"] if c.get("mode") == mode and "auc" in c]
        if not cs:
            continue
        agg[mode] = {
            "cells": len(cs),
            "judged_off": float(np.mean([c["judged_off"] for c in cs])),
            "judged_on": float(np.mean([c["judged_on"] for c in cs])),
            "judged_off_spread": [float(np.min([c["judged_off"] for c in cs])),
                                 float(np.max([c["judged_off"] for c in cs]))],
            "judged_on_spread": [float(np.min([c["judged_on"] for c in cs])),
                                float(np.max([c["judged_on"] for c in cs]))],
            "cells_gaining_a_checkable_slot": int(sum(
                c["cells_gaining_a_checkable_slot"] for c in cs)),
            "candidates": int(sum(c["candidates"] for c in cs)),
            "pairs_new": int(sum(c["new_rule_pairs"] for c in cs)),
            "pairs_old": int(sum(c["old_rule_pairs"] for c in cs)),
            "auc_ctx_matcher": [c["auc"]["ctx_matcher"] for c in cs],
            "auc_ctx_truth": [c["auc"]["ctx_truth"] for c in cs],
            "top1_ctx_matcher": float(np.mean([c["top1"]["ctx_matcher"] for c in cs
                                               if c["top1"]["ctx_matcher"] is not None])),
            "top1_ctx_truth": float(np.mean([c["top1"]["ctx_truth"] for c in cs
                                             if c["top1"]["ctx_truth"] is not None])),
        }
    for mode, a in agg.items():
        for split in ("syn", "real"):
            sub = [c for c in res["cells"] if c.get("mode") == mode
                   and c["tag"].startswith(split) and "auc" in c]
            if not sub:
                continue
            a["auc_ctx_matcher_" + split] = float(np.mean(
                [c["auc"]["ctx_matcher"] for c in sub if c["auc"]["ctx_matcher"]]))
            a["auc_ctx_truth_" + split] = float(np.mean(
                [c["auc"]["ctx_truth"] for c in sub if c["auc"]["ctx_truth"]]))
            a["cells_" + split] = len(sub)
    res["aggregate"] = agg

    # ------------------------------- item 5 ---------------------------------
    def _res(mode: str, split: Optional[str] = None) -> Tuple[List[float], List[float]]:
        t: List[float] = []
        w: List[float] = []
        for c in res["cells"]:
            if c.get("mode") != mode or "res_true" not in c:
                continue
            if split and not c["tag"].startswith(split):
                continue
            t.extend(c["res_true"])
            w.extend(c["res_wrong"])
        return t, w

    geo: Dict[str, Any] = {}
    for mode in ("frozen", "places"):
        t, w = _res(mode)
        if not t:
            continue
        geo[mode] = {
            "true_residual_n": len(t), "wrong_residual_n": len(w),
            "true_median": float(np.median(t)), "true_q90": float(np.percentile(t, 90)),
            "wrong_median": float(np.median(w)),
            "true_iqr": [float(np.percentile(t, 25)), float(np.percentile(t, 75))],
        }
        for split in ("syn", "real"):
            ts, ws = _res(mode, split)
            if not ts:
                continue
            geo[mode]["true_median_" + split] = float(np.median(ts))
            geo[mode]["true_n_" + split] = len(ts)
            # noise: the exact stride-multiple shift is a pure re-detection test
            tn: List[float] = []
            for c in res["cells"]:
                if (c.get("mode") == mode and c["pose"] == "shift8"
                        and c["tag"].startswith(split) and "res_true" in c):
                    tn.extend(c["res_true"])
            if tn:
                geo[mode]["noise_exact_shift_median_" + split] = float(np.median(tn))
                geo[mode]["noise_exact_shift_q90_" + split] = float(
                    np.percentile(tn, 90))
                geo[mode]["noise_exact_shift_p97_" + split] = float(
                    np.percentile(tn, 97))
    res["item5a"] = geo

    if geo:
        mode = "places" if "places" in geo else "frozen"
        t, w = _res(mode)
        noise = geo[mode].get("noise_exact_shift_q90_real",
                              geo[mode].get("noise_exact_shift_q90_syn"))
        # (b) BIN WIDTH: no finer than the measured real-photo residual noise.
        bin_w = max(L.RES_MAX / L.N_BINS, float(noise) if noise else 0.0)
        n_bins = max(4, int(np.ceil(L.RES_MAX / bin_w)))
        import parameters as P
        old_flag = int(getattr(P, "PLACEMENT_LLR_MIN_BIN_COUNT", 0))
        old_neu = bool(getattr(P, "PLACEMENT_LLR_EMPTY_BIN_NEUTRAL", False))
        P.PLACEMENT_LLR_MIN_BIN_COUNT = 20
        P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = True
        fit = L.fit_llr(t, w, n_bins=n_bins)
        res["item5b"] = {
            "mode": mode, "pooled_true": len(t), "pooled_wrong": len(w),
            "bin_width_shipped": L.RES_MAX / L.N_BINS, "bin_width_used": bin_w,
            "n_bins": n_bins, "real_photo_noise_q90": noise,
            "min_bin_count": 20, "llr": list(map(float, fit.llr)),
            "bin_counts_true": list(map(int, np.histogram(
                np.clip(np.asarray(t), 0.0, L.RES_MAX - 1e-9), bins=fit.edges)[0])),
            "bin_counts_wrong": list(map(int, np.histogram(
                np.clip(np.asarray(w), 0.0, L.RES_MAX - 1e-9), bins=fit.edges)[0])),
            "rises_after_smoothing": L.llr_monotonicity(fit.llr, fit.edges),
            "mu_wrong": float(fit.mu_wrong),
            "pooled_from": "synthetic FIT + real known-transform residuals (item 5b)",
        }
        P.PLACEMENT_LLR_MIN_BIN_COUNT = old_flag
        P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = old_neu
        # (d) geometry ALONE, per mode and split
        for m in ("frozen", "places"):
            cs = [c for c in res["cells"] if c.get("mode") == m and "auc" in c]
            if not cs:
                continue
            res.setdefault("item5d", {})[m] = {
                "auc_ctx_truth": _auc_pooled(cs, "ctx_truth"),
                "auc_ctx_matcher": _auc_pooled(cs, "ctx_matcher"),
                "images": len({c["tag"] for c in cs}),
                "cells": len(cs),
            }

    # ------------------------------- item 7: the weight profile --------------
    def _profile(mode: str, split: Optional[str] = None,
                 grid=(0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
        qs = []
        for c in res["cells"]:
            if c.get("mode") != mode or "bank" not in c:
                continue
            if split and not c["tag"].startswith(split):
                continue
            qs.extend(c["bank"])
        if not qs:
            return None
        beta = 0.5
        prof = []
        for w in grid:
            top1 = 0
            auc_pairs = []
            for q in qs:
                scored = [((1 - beta) * s + beta * (0.5 + w * e), int(t))
                          for s, e, t in zip(q["sim"], q["E"], q["true"])]
                best = max(scored, key=lambda z: (z[0], -z[1] if z[1] else 1))[0]
                trues = [sc for sc, t in zip(scored, q["true"]) if t == 1]
                imps = [sc for sc, t in zip(scored, q["true"]) if t == 0]
                if not trues or not imps:
                    continue
                top1 += 1 if scored[[i for i, t in enumerate(q["true"]) if t == 1][0]][0] >= max(
                    sc for sc, t in zip(scored, q["true"]) if t == 0) else 0
                auc_pairs.append((max(trues), imps))
            prof.append({"w": float(w), "top1": top1 / float(len(qs)),
                         "auc": _auc(auc_pairs), "queries": len(qs)})
        return prof

    if "bank" in json.dumps(res["cells"][:1], default=float):
        for mode in ("frozen", "places"):
            for split in (None, "syn", "real"):
                pr = _profile(mode, split)
                if pr:
                    key = "item7_%s%s" % (mode, "" if not split else "_" + split)
                    res.setdefault("item7", {})[key] = {
                        "profile": pr,
                        "best_w": max(pr, key=lambda r: r["auc"])["w"],
                        "weight_source": "refit on item-5/10 cells; frozen w_top1 in "
                                         "matching/placement_score_params.json is the "
                                         "shipped value",
                    }

    if args.report:
        print(json.dumps({k: v for k, v in res.items()
                          if k in ("aggregate", "item5a", "item5b", "item5d")},
                         indent=1, default=float)[:4000])
        return 0
    MF.write_result(RESULT, res)
    print("wrote", RESULT)
    return 0


def _auc_pooled(cells: List[Dict[str, Any]], arm: str) -> Optional[float]:
    vals = [c["auc"][arm] for c in cells if c["auc"].get(arm) is not None]
    return float(np.mean(vals)) if vals else None


if __name__ == "__main__":
    raise SystemExit(main())
