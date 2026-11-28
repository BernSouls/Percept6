"""MISSION 21 / O1 + O2 + O5 -- the shortlist on ONE ruler, a second channel chosen on recall,
and every rate reported against its own chance.

O1.  `score_bags` standardized the query by each REFERENCE's own mean, std and kept
dimensions, so two gallery items scored the same query on two rulers and the shortlist
compared incomparable numbers.  Here every score comes from the frozen ruler
(`matching/placement_prefilter_params.json`), applied identically to every image: no
reference contributes to the scale, no per-reference feature selection exists, and a sparse
place is compared on the slots it has instead of on a repeated last entry.  The score stays
directional -- it is a mean over the query's places -- so that asymmetry is REPORTED rather
than assumed away.

O2.  A second channel is added only if it raises recall@8 on the 40-item mixed gallery, with
its weight chosen on the fit half of the queries.  The appearance channel is fixed and
unfitted (a 4x4x4 colour histogram plus a 12-bin gradient-orientation histogram, L2, cosine).

O5.  Chance is stated with every rate and on its own units:
      T1  the truth is one item of a same-class gallery            chance 1/|gallery|
      T3  the truth is one item of the mixed gallery               chance 1/|gallery|
      T2  the truth is the CLASS, so the first same-class item is scored and chance is the
          hypergeometric probability that a random top-S holds one of the other instances.
     Raw rates are never compared across different chance; the lift is what is compared.

Nothing here builds a graph: the pair table comes from the cached bags.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
S = 8
W_GRID = (0.0, 0.15, 0.3, 0.5, 0.8)


# --------------------------------------------------------------------------- appearance
def appearance(path: str) -> np.ndarray:
    """A fixed, unfitted appearance descriptor: colour and gradient-orientation histograms.

    Nothing is fitted: the bins are fixed, the normalisation is L2, and two images are
    compared by a dot product, so the channel cannot carry an ETH-80 label into the score.
    """
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    q = np.minimum((img * 4).astype(np.int64), 3)
    col = np.bincount((q[:, :, 0] * 16 + q[:, :, 1] * 4 + q[:, :, 2]).ravel(),
                      minlength=64).astype(np.float64).reshape(4, 4, 4)
    grey = img.mean(axis=2)
    gy, gx = np.gradient(grey)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    b = np.minimum((ang / np.pi * 12).astype(np.int64), 11)
    hist = np.bincount(b.ravel(), weights=mag.ravel(), minlength=12)
    v = np.concatenate([col.ravel() / max(1e-9, col.sum()), hist / max(1e-9, hist.sum())])
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- chance + tasks
def _hyper_ge(n_same: int, n_pool: int, k: int) -> float:
    """P(at least one of ``n_same`` same-class items is in a random top-``k``)."""
    if n_same <= 0 or n_pool <= 0:
        return 0.0
    if n_pool - n_same < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n_pool - n_same, k) / float(comb(n_pool, k))


def _cls_of(path: str, cls_by_ref: Dict[str, str]) -> str:
    return cls_by_ref[path]


def score_tasks(mat: np.ndarray, objs: List[Dict[str, Any]],
                gal_paths: List[str], task: str, depth: int = S) -> Dict[str, Any]:
    """Recall@depth and chance for one task.

    ``mat`` is (n_queries, n_gallery) in the order of ``objs``' queries and ``gal_paths``.
    """
    g_idx = {p: j for j, p in enumerate(gal_paths)}
    cls_by_ref = {o["ref"]: str(o["cls"]) for o in objs}
    hits: List[int] = []
    chances: List[float] = []
    per_image: List[Dict[str, Any]] = []
    for i, o in enumerate(objs):
        truth = g_idx.get(o["ref"])
        if truth is None:
            continue
        if task == "T1":
            cand = [j for j, p in enumerate(gal_paths) if cls_by_ref[p] == o["cls"]]
            order = sorted(cand, key=lambda j: (-mat[i, j], j))
            hit = 1 if truth in order[:depth] else 0
            chance = 1.0 / len(cand)
        elif task == "T3":
            cand = list(range(len(gal_paths)))
            order = sorted(cand, key=lambda j: (-mat[i, j], j))
            hit = 1 if truth in order[:depth] else 0
            chance = 1.0 / len(cand)
        else:                                            # T2: the class is the truth
            cand = [j for j in range(len(gal_paths)) if j != truth]
            order = sorted(cand, key=lambda j: (-mat[i, j], j))
            same = [j for j in cand if cls_by_ref[gal_paths[j]] == o["cls"]]
            hit = 1 if any(j in same for j in order[:depth]) else 0
            chance = _hyper_ge(len(same), len(cand), depth)
        hits.append(hit)
        chances.append(chance)
        per_image.append({"query": o["key"], "hit": hit, "chance": chance})
    n = len(hits)
    rec = (sum(hits) / n) if n else None
    ch = float(np.mean(chances)) if chances else None
    return {"task": task, "depth": depth, "recall": rec, "n": n, "chance": ch,
            "lift": (rec / ch) if (rec is not None and ch) else None,
            "n_hits": int(sum(hits)), "rows": per_image}


# --------------------------------------------------------------------------- the table
def index_maps(paths: List[str], objs: List[Dict[str, Any]]) -> Tuple[Dict[str, int],
                                                                     List[str], List[str]]:
    q_idx = {o["query"]: i for i, o in enumerate(objs)}
    gal = sorted({o["ref"] for o in objs})
    return q_idx, gal, [o["query"] for o in objs]


def geo_matrix(tbl: Dict[str, Any], objs: List[Dict[str, Any]], k: int
               ) -> Tuple[np.ndarray, List[str], List[str]]:
    """(query x gallery stage-1 scores at grid point ``k``, gallery paths, query paths)."""
    pos = {p: i for i, p in enumerate(tbl["paths"])}
    gal = sorted({o["ref"] for o in objs})
    qs = [o["query"] for o in objs]
    M = np.zeros((len(qs), len(gal)), dtype=np.float64)
    for i, q in enumerate(qs):
        for j, r in enumerate(gal):
            v = tbl["S"][pos[q], pos[r], k]
            M[i, j] = 0.0 if not np.isfinite(v) else float(v)
    return M, gal, qs


def evaluate_table(tbl: Dict[str, Any], objs: List[Dict[str, Any]], depth: int,
                   k_only: Optional[int] = None) -> Dict[str, Any]:
    """recall@8 over the whole sigma grid, on the FIT fit-half only (the choice half)."""
    half = [o for i, o in enumerate(objs) if i % 2 == 0]
    grid = []
    for k, (sd, sa) in enumerate(tbl["grid"]):
        if k_only is not None and k != k_only:
            continue
        M_all, gal, _qs = geo_matrix(tbl, objs, k)
        keep = {o["query"] for o in half}
        idx = [i for i, o in enumerate(objs) if o["query"] in keep]
        M = M_all[idx]
        sub = [objs[i] for i in idx]
        r = score_tasks(M, sub, gal, "T3", depth)
        grid.append({"sigma_d": float(sd), "sigma_a": float(sa), "k": k,
                     "recall8_T3": r["recall"], "n": r["n"],
                     "chance": r["chance"]})
    best = max(grid, key=lambda g: ((g["recall8_T3"] or 0), -g["sigma_d"], -g["sigma_a"]))
    return {"grid": grid, "best": best, "n_queries": best["n"],
            "objective": "recall@8 on the 40-item mixed gallery, fit-half queries only"}


def asymmetry(tbl: Dict[str, Any], k: int, seed: int = 0) -> Dict[str, Any]:
    """score(A,B) against score(B,A) over every ordered pair of the split's images."""
    n = len(tbl["paths"])
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            a = tbl["S"][i, j, k]
            b = tbl["S"][j, i, k]
            if np.isfinite(a) and np.isfinite(b):
                d.append(abs(float(a) - float(b)))
    arr = np.asarray(d)
    return {"n_pairs": int(arr.size), "mean_abs": float(arr.mean()),
            "p95_abs": float(np.percentile(arr, 95)), "max_abs": float(arr.max()),
            "frac_gt_0.05": float((arr > 0.05).mean())}


def asymmetry_v1(bags: Dict[str, Any], paths: List[str], sample: int = 300,
                 seed: int = 0) -> Dict[str, Any]:
    """The same statistic under the M20 convention, on a seeded sample of pairs."""
    from matching import placement_prefilter as PF
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths))]
    if len(pairs) > sample:
        sel = rng.choice(len(pairs), size=sample, replace=False)
        pairs = [pairs[int(x)] for x in sel]
    d = []
    for i, j in pairs:
        ba, bb = bags[paths[i]]["bag"], bags[paths[j]]["bag"]
        a = PF.score_bags(ba, bb).get("score")
        b = PF.score_bags(bb, ba).get("score")
        if a is not None and b is not None:
            d.append(abs(float(a) - float(b)))
    arr = np.asarray(d)
    return {"n_pairs": int(arr.size), "n_sampled_from": int(len(paths) * (len(paths) - 1) / 2),
            "mean_abs": float(arr.mean()), "p95_abs": float(np.percentile(arr, 95)),
            "max_abs": float(arr.max()), "frac_gt_0.05": float((arr > 0.05).mean()),
            "seed": int(seed)}


# --------------------------------------------------------------------------- report
def report(tbl: Dict[str, Any], objs: List[Dict[str, Any]], depth: int = S,
           w_fixed: Optional[float] = None) -> Dict[str, Any]:
    from matching import placement_prefilter as PF
    from Topology.placement_m21_fit import ensure_bags
    params = PF.load_params()
    k = None
    for kk, (sd, sa) in enumerate(tbl["grid"]):
        if abs(sd - params["sigma_d"]) < 1e-9 and abs(sa - params["sigma_a"]) < 1e-9:
            k = kk
    if k is None:
        raise SystemExit("chosen sigma is not on the table's grid: rerun --stage choose")
    M, gal, qs = geo_matrix(tbl, objs, k)
    app_q = np.asarray([appearance(p) for p in qs])
    app_g = np.asarray([appearance(p) for p in gal])
    A = app_q @ app_g.T
    even = [i for i, _o in enumerate(objs) if i % 2 == 0]
    odd = [i for i, _o in enumerate(objs) if i % 2 == 1]
    grid = []
    for w in W_GRID:
        r = score_tasks(M[even] + w * A[even], [objs[i] for i in even], gal, "T3", depth)
        grid.append({"w": w, "recall8_T3_fit_half": r["recall"], "n": r["n"]})
    best = max(grid, key=lambda g: (g["recall8_T3_fit_half"] or 0, -g["w"]))
    w = best["w"] if w_fixed is None else float(w_fixed)
    res: Dict[str, Any] = {"split": tbl.get("split"), "gallery": len(gal), "S": depth,
                           "sigma_d": params["sigma_d"], "sigma_a": params["sigma_a"],
                           "K": params["K"], "w_grid": grid, "w_chosen": w,
                           "w_source": ("chosen on this split's fit half" if w_fixed is None
                                        else "FIXED at the FIT choice, not re-chosen here"),
                           "channels": {}, "splits": {"n_fit": len(even), "n_rep": len(odd)}}
    for name, mat in (("geo", M), ("app", A), ("geo+w*app", M + w * A)):
        res["channels"][name] = {}
        for task in ("T1", "T2", "T3"):
            res["channels"][name][task] = score_tasks(mat, objs, gal, task, depth)
        res["channels"][name]["T3_fit_half"] = score_tasks(
            mat[even], [objs[i] for i in even], gal, "T3", depth)
        res["channels"][name]["T3_rep_half"] = score_tasks(
            mat[odd], [objs[i] for i in odd], gal, "T3", depth)
    print("%s: gallery %d, S=%d, sigma %.2f/%.2f, K %d, w=%.2f"
          % ("FIT" if tbl.get("split") is None else tbl["split"], len(gal), depth,
             params["sigma_d"], params["sigma_a"], params["K"], w))
    print("%-10s %-24s %-24s %-24s" % ("channel", "T1 same-class", "T2 class",
                                       "T3 object"))
    for name in ("geo", "app", "geo+w*app"):
        cells = []
        for task in ("T1", "T2", "T3"):
            r = res["channels"][name][task]
            cells.append("%.3f n=%d ch=%.3f" % (r["recall"] or 0, r["n"], r["chance"] or 0))
        print("%-10s %-24s %-24s %-24s" % (name, *cells))
    print("lift over chance (same channel order):")
    for name in ("geo", "app", "geo+w*app"):
        print("  %-10s T1 x%.2f  T2 x%.2f  T3 x%.2f"
              % (name, *[res["channels"][name][t]["lift"] or 0
                         for t in ("T1", "T2", "T3")]))
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--v1-sample", type=int, default=300)
    ap.add_argument("--w", type=float, default=None,
                    help="fix the appearance weight (a split other than FIT must not choose)")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_prefilter as PF
    from Topology import placement_m21_terms as T
    from Topology.placement_m21_fit import ensure_bags, split_objects
    if not P.PLACEMENT_M19:
        raise SystemExit("set PLACEMENT_M19=1")
    params = PF.load_params()
    if params is None:
        raise SystemExit("no frozen params: run Topology/placement_m21_fit.py --stage params")
    tbl = T.build(a.split, a.per_class, a.workers)
    tbl["split"] = a.split
    objs = split_objects(a.split, a.per_class)
    res = report(tbl, objs, S, a.w)
    k = [kk for kk, (sd, sa) in enumerate(tbl["grid"])
         if abs(sd - params["sigma_d"]) < 1e-9 and abs(sa - params["sigma_a"]) < 1e-9][0]
    if a.split == "FIT":
        bags = ensure_bags(a.split, a.workers, a.per_class)
        res["asymmetry_v2"] = asymmetry(tbl, k)
        res["asymmetry_v1"] = asymmetry_v1(bags, tbl["paths"], a.v1_sample)
        print("asymmetry |s(A,B)-s(B,A)|: v2 mean %.4f p95 %.4f (n=%d) | "
              "v1 mean %.4f p95 %.4f (n=%d)"
              % (res["asymmetry_v2"]["mean_abs"], res["asymmetry_v2"]["p95_abs"],
                 res["asymmetry_v2"]["n_pairs"], res["asymmetry_v1"]["mean_abs"],
                 res["asymmetry_v1"]["p95_abs"], res["asymmetry_v1"]["n_pairs"]))
    res["per_class"] = a.per_class
    res["params"] = os.path.relpath(PF.PARAMS_PATH, ROOT)
    out = os.path.join(OUT, (a.tag or ("m21_shortlist_%s_pc%d" % (a.split.lower(),
                                                                 a.per_class))) + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)          # every number carries its EFFECTIVE flags
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
