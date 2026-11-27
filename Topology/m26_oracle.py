"""M26 item 4: the oracle table -- every cheap channel, the same cells, one table.

Reads only the cached place records (``Topology/m26_records.py``) and the images (for the
appearance floor).  Nothing here builds a graph or runs the matcher.

    python Topology/m26_oracle.py --out Topology/results/placement/m26_oracle.json

Cells (fixed in ``MISSION26_PREREGISTRATION.md``):
    selection     40 FIT objects x views 035-045, 045-180   (80 query cells)
    saturation    the same 40 x view 022-000                (reported, not used for selection)
    confirmation  40 DEV + FRESH objects x the same two views

Variants (ids as in the registration):
    a counted door (shipped)     b appearance        c RRF(a, b)  (the shipped door)
    d graded signature only      e configuration only f the graded door
    g f without distinctiveness  h f with pooled top-8
    i graph-native global layout j RRF(b,f), RRF(b,f,i), RRF(b,a,f)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC  # noqa: E402

VIEWS_SEL = ("035-045", "045-180")
VIEW_SAT = "022-000"
REF = "000-000"


# --------------------------------------------------------------------------- objects
def objects() -> List[Dict[str, Any]]:
    return REC.objects()


def path_of(o: Dict[str, Any], view: str) -> str:
    return REC.path_of(o, view)


# --------------------------------------------------------------------------- variant i
def layout_vector(rec: Dict[str, Any]) -> np.ndarray:
    """The graph-native global layout of one image, from its place record only.

    Four fixed blocks, each L2-normalised, then the whole vector L2-normalised:
      lens shares (3) | log2 local-scale histogram (6 bins over [-1, 3]) |
      extent-normalised radial histogram (6 bins of r / r_max) | axial theta histogram (6).
    """
    rows = rec["places"]
    v = np.zeros(3 + 6 + 6 + 6, dtype=np.float64)
    if not rows:
        return v
    lens_names = ("Blob", "BoundaryArc", "CornerPoint")
    for r in rows:
        if r["lens"] in lens_names:
            v[lens_names.index(r["lens"])] += 1.0
    v[:3] /= max(1e-9, v[:3].sum())
    sc = np.asarray([np.log2(max(1e-9, float(r["s"]))) for r in rows])
    hs, _ = np.histogram(np.clip(sc, -1.0, 3.0), bins=6, range=(-1.0, 3.0))
    v[3:9] = hs / max(1.0, hs.sum())
    P = np.asarray([[r["x"], r["y"]] for r in rows], dtype=np.float64)
    c = P.mean(axis=0)
    r = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    rmax = float(np.percentile(r, 95)) or 1.0
    hr, _ = np.histogram(np.clip(r / rmax, 0.0, 1.0), bins=6, range=(0.0, 1.0))
    v[9:15] = hr / max(1.0, hr.sum())
    th = np.asarray([float(r_["theta"]) if r_.get("theta") is not None else np.nan for r_ in rows])
    good = th[np.isfinite(th)]
    if good.size:
        ht, _ = np.histogram(good % np.pi, bins=6, range=(0.0, np.pi))
        v[15:21] = ht / max(1.0, ht.sum())
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- fusion
def rrf(*mats: np.ndarray, k: int = 60) -> np.ndarray:
    """Sum over lists of 1 / (k + rank).  No weight, ties by object index."""
    out = np.zeros_like(mats[0], dtype=np.float64)
    for M in mats:
        for i in range(M.shape[0]):
            order = sorted(range(M.shape[1]), key=lambda j: (-M[i, j], j))
            for r, j in enumerate(order):
                out[i, j] += 1.0 / (k + r + 1)
    return out


# --------------------------------------------------------------------------- metrics
def metrics(S: np.ndarray, truth: Sequence[int], s: int = 8) -> Dict[str, Any]:
    """recall@8, top-1, the rank of the truth, and the counts behind them."""
    n = S.shape[0]
    hits, top1, ranks = 0, 0, []
    for i in range(n):
        order = sorted(range(S.shape[1]), key=lambda j: (-S[i, j], j))
        r = order.index(int(truth[i])) + 1
        ranks.append(r)
        hits += int(r <= s)
        top1 += int(r == 1)
    return {"n": n, "recall_s": hits / n, "recall_hits": hits, "top1": top1 / n, "top1_hits": top1,
            "rank_median": float(np.median(ranks)), "rank_mean": float(np.mean(ranks)),
            "chance_recall": s / S.shape[1], "chance_top1": 1.0 / S.shape[1]}


def by_class(S: np.ndarray, truth: Sequence[int], classes: Sequence[str], s: int = 8
             ) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in sorted(set(classes)):
        idx = [i for i, cc in enumerate(classes) if cc == c]
        sub = S[idx]
        t = [int(truth[i]) for i in idx]
        m = metrics(sub, t, s)
        out[c] = {"n": m["n"], "recall_s": m["recall_s"], "recall_hits": m["recall_hits"],
                  "top1": m["top1"], "top1_hits": m["top1_hits"]}
    return out


# --------------------------------------------------------------------------- the sweep
def build_variants(gal_objs: List[Dict[str, Any]], q_objs: List[Dict[str, Any]],
                   view: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    from matching import placement_frontdoor as FD
    from matching import placement_kernel as K
    n_obj = len(gal_objs)
    gal = [REC.load(o["key"], REF) for o in gal_objs]
    qs = [REC.load(o["key"], view) for o in q_objs]
    gal_index = {o["key"]: i for i, o in enumerate(gal_objs)}
    truth = [gal_index[o["key"]] for o in q_objs]      # the query objects ARE gallery objects
    t: Dict[str, float] = {}
    out: Dict[str, np.ndarray] = {}

    index = FD.PlaceIndex(gal, [o["key"] for o in gal_objs])
    t0 = time.time(); out["a_counted"], _ = FD.count_scores(index, qs, n_hits=8); t["a"] = time.time() - t0
    t0 = time.time()
    A = FD.appearance_scores([path_of(o, view) for o in q_objs],
                             [path_of(o, REF) for o in gal_objs])
    t["b"] = time.time() - t0
    out["b_appearance"] = A
    out["c_rrf_ab"] = rrf(out["a_counted"], A)
    t0 = time.time(); out["d_sig_only"], _ = FD.graded_scores(index, qs, config=False); t["d"] = time.time() - t0
    t0 = time.time(); out["e_cfg_only"], _ = FD.graded_scores(index, qs, signature=False,
                                                             lens_gate=False); t["e"] = time.time() - t0
    t0 = time.time(); out["f_graded"], diag_f = FD.graded_scores(index, qs); t["f"] = time.time() - t0
    t0 = time.time(); out["g_noidf"], _ = FD.graded_scores(index, qs, idf=False); t["g"] = time.time() - t0
    t0 = time.time(); out["h_pool8"], _ = FD.graded_scores(index, qs, pooled_top=8); t["h"] = time.time() - t0
    t0 = time.time()
    Lg = np.asarray([layout_vector(r) for r in gal])
    Lq = np.asarray([layout_vector(r) for r in qs])
    out["i_layout"] = Lq @ Lg.T
    t["i"] = time.time() - t0
    out["j_rrf_bf"] = rrf(A, out["f_graded"])
    out["j_rrf_bfi"] = rrf(A, out["f_graded"], out["i_layout"])
    out["j_rrf_baf"] = rrf(A, out["a_counted"], out["f_graded"])
    info = {"timings_sec": {k: round(v, 2) for k, v in t.items()},
            "seconds_per_query": {k: round(v / max(1, len(q_objs)), 3) for k, v in t.items()},
            "kernel_fallbacks": dict(K.FALLBACKS),
            "graded_diag_mean_df": float(np.mean([d["mean_df"] for d in diag_f
                                                  if d.get("mean_df")])),
            "truth": truth}
    return out, info


MATS: Dict[str, np.ndarray] = {}

VARIANT_NAMES = {"a_counted": "a counted door (shipped)", "b_appearance": "b appearance",
                 "c_rrf_ab": "c RRF(a,b) -- the shipped door", "d_sig_only": "d graded signature only",
                 "e_cfg_only": "e configuration only", "f_graded": "f the graded door",
                 "g_noidf": "g f without distinctiveness", "h_pool8": "h f with pooled top-8",
                 "i_layout": "i global layout", "j_rrf_bf": "j RRF(b,f)", "j_rrf_bfi": "j RRF(b,f,i)",
                 "j_rrf_baf": "j RRF(b,a,f)"}


def cell_table(gal: List[Dict[str, Any]], qs: List[Dict[str, Any]], view: str,
               splits: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    variants, info = build_variants(gal, qs, view)
    truth = info["truth"]
    cls = [o["cls"] for o in qs]
    table: Dict[str, Any] = {}
    oracle = np.zeros(len(qs), dtype=bool)
    for name, S in variants.items():
        m = metrics(S, truth)
        m["by_class"] = by_class(S, truth, cls)
        table[name] = m
        if name not in ("j_rrf_bf", "j_rrf_bfi", "j_rrf_baf"):
            order = [sorted(range(S.shape[1]), key=lambda j: (-S[i, j], j))[:8] for i in range(S.shape[0])]
            oracle |= np.asarray([int(truth[i]) in order[i] for i in range(len(qs))], dtype=bool)
    counts = [len(REC.load(o["key"], REF)["places"]) for o in gal]
    med = float(np.median(counts))
    thin = [counts[t] < 0.5 * med for t in truth]
    fail_a = [i for i in range(len(qs))
              if truth[i] not in sorted(range(len(gal)),
                                        key=lambda j: (-variants["a_counted"][i, j], j))[:8]]
    return ({"view": view, "splits": list(splits), "n": len(qs),
            "chance": 8.0 / len(gal), "variants": table,
            "union_oracle": {"recall_s": float(oracle.mean()), "hits": int(oracle.sum()),
                             "n": len(qs)},
            "thin_objects": {"median_places": med,
                             "share_of_failures_thin_a": (float(np.mean([thin[i] for i in fail_a]))
                                                          if fail_a else None),
                             "failures_a": len(fail_a),
                             "share_of_queries_thin": float(np.mean(thin))},
            "info": info}, variants)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "Topology", "results", "placement",
                                                  "m26_oracle.json"))
    ap.add_argument("--views", default=",".join(VIEWS_SEL))
    ap.add_argument("--splits", default="FIT,DEV,FRESH")
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    views = [v.strip().lstrip("-") for v in a.views.split(",") if v.strip()]
    want = [s.strip() for s in a.splits.split(",") if s.strip()]
    objs = objects()
    gal = objs                                   # gallery: all 80 at the reference view
    missing = [o["key"] for o in gal if not REC.have(o["key"], REF)]
    if missing:
        print("missing %d reference records (e.g. %s) -- run m26_records.py --build"
              % (len(missing), missing[0]))
        return 2
    MATS.clear()
    out: Dict[str, Any] = {"gallery": len(gal), "ref_view": REF, "views": views,
                           "cells": {}, "records": REC.__name__}
    for split_name, splits in (("selection", ["FIT"]), ("confirmation", ["DEV", "FRESH"])):
        qs = [o for o in objs if o["split"] in splits]
        for v in views + ([VIEW_SAT] if split_name == "selection" else []):
            key = "%s|%s" % (split_name, v)
            miss = [o["key"] for o in qs if not REC.have(o["key"], v)]
            if miss:
                print("%s: %d records missing -- skipped" % (key, len(miss)))
                continue
            t0 = time.time()
            table, mats = cell_table(gal, qs, v, splits)
            out["cells"][key] = table
            for name, M in mats.items():
                MATS["%s|%s|%s" % (split_name, v, name)] = M
            print("%-22s n=%3d  %.1fs" % (key, len(qs), time.time() - t0), flush=True)
            tab = out["cells"][key]["variants"]
            for name in VARIANT_NAMES:
                if name in tab:
                    print("    %-30s recall@8 %.3f (%2d/%2d)  top1 %.3f  rank med %4.1f"
                          % (VARIANT_NAMES[name], tab[name]["recall_s"], tab[name]["recall_hits"],
                             tab[name]["n"], tab[name]["top1"], tab[name]["rank_median"]))
            u = out["cells"][key]["union_oracle"]
            print("    %-30s recall@8 %.3f (%d/%d)" % ("UNION ORACLE a..i", u["recall_s"],
                                                       u["hits"], u["n"]))
            print("    timings (s/query):", out["cells"][key]["info"]["seconds_per_query"], flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("wrote", a.out)
    if MATS:
        npz = os.path.splitext(a.out)[0] + ".npz"
        np.savez_compressed(npz, **MATS)
        print("wrote", npz, "(matrices, gitignored: *.npz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
