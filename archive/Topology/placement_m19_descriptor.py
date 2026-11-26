"""MISSION 19 / O1 and O2 -- WHY the ordering fails, and what breaks the tie.

O1's acceptance is about ORDER (appearance-only top-1, median true rank, outside-cap share),
so the first instrument measures the order directly and never needs the matcher: given the
similarity rows, re-rank them by a candidate descriptor and read the true partner's rank.
That is cheap, it is deterministic, and it is the same quantity the matcher's appearance arm
reports, so the expensive instrument is run once, on the candidate that survives here.

Three measurements:

1. OCCUPANCY (the mechanism, not the excuse).  The keypoint signature is gated by five
   quantised slots (theta/12, coherence/4, scale, kind, dim).  For every query place this
   counts how many reference places share its FULL quantised key (class, relation, slots),
   per image: if many places share a key by construction, no five-slot similarity can order
   them, and the share of the row that is distinguishable is a ceiling on rank-1.
2. THE CONFIGURATION DESCRIPTOR as a node descriptor: AUC of `config_similarity(true)` vs
   `config_similarity(best impostor)`, and the rank statistics of the re-ranked rows for the
   registered candidates R1 (top-K) and R2 (whole row), K in {24, 72}, w in {0.25, 0.5}.
3. O2's three statistics, per image: texture period (px), the coherence distribution, and the
   absolute place spacing in px (`distance` is dimensionless source-scale units, so px =
   distance * source_scale).

    python Topology/placement_m19_descriptor.py --workers 2
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m19_descriptor.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot15", {"rot_deg": 15.0}),
)
DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")
#: the registered candidates
CANDIDATES: Tuple[Tuple[int, float], ...] = ((24, 0.25), (24, 0.5), (72, 0.25), (72, 0.5))


def _quant_key(sig: Any) -> Tuple[Any, ...]:
    return (str(getattr(sig, "class_", "?")), str(getattr(sig, "relation", "")),
            tuple(float(x) for x in (getattr(sig, "quantized", ()) or ())))


def _ranks(row: List[Tuple[int, float]], truth: int) -> Optional[int]:
    for i, (v, _s) in enumerate(row):
        if v == truth:
            return i + 1
    return None


def _job(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, pose, spec = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_context as CX
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build
    from Topology.placement_m17_eval import M16, _image
    from core import keypoint_classes as KPC

    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    P.PLACEMENT_LLR_TABLE_M18 = True
    P.PLACEMENT_CONTEXT_ASSIGN = True
    P.PLACEMENT_CONTEXT_ASSIGN_LAMBDA = 0.5
    P.PLACEMENT_CONTEXT_CONSENSUS = False
    P.PLACEMENT_CAND_CAP = 24
    PM.reset_cache()
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()

    img = _image(tag)
    img1 = R._transform(img, spec)
    g0 = get_or_build(img, "m16|%s|orig|on" % tag)
    g1 = get_or_build(img1, "m16|%s|%s|on" % (tag, pose))
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    sq = PM.build_side_cached(g1, img1)
    sr = PM.build_side_cached(g0, img)
    truth = PL.lens_truth(g1, g0, spec, img.shape[:2])
    rows = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)

    # 1. occupancy of the quantised key, per lens
    key_b: Dict[Tuple[Any, ...], int] = {}
    for v in nodes_b:
        sv = g0.sigs.get(v)
        if sv is None:
            continue
        key_b[_quant_key(sv[1])] = key_b.get(_quant_key(sv[1]), 0) + 1
    occ, occ_true, distinct = [], [], []
    for u in nodes_a:
        su = g1.sigs.get(u)
        row = rows.get(u) or []
        if su is None or not row:
            continue
        k = _quant_key(su[1])
        occ.append(key_b.get(k, 0))
        distinct.append(len({_quant_key(g0.sigs[v][1]) for (v, _s) in row
                             if g0.sigs.get(v) is not None}))
        vt = truth.get(u)
        if vt is not None:
            sv = g0.sigs.get(vt)
            occ_true.append(key_b.get(_quant_key(sv[1]), 0) if sv is not None else 0)

    # 2. the configuration descriptor as a node descriptor
    desc_q = {u: CX.config_descriptor(sq, u) for u in nodes_a}
    desc_r = {v: CX.config_descriptor(sr, v) for v in nodes_b}
    cfg: Dict[Tuple[int, int], float] = {}

    def _cfg(u: int, v: int) -> float:
        k = (int(u), int(v))
        got = cfg.get(k)
        if got is None:
            got = CX.config_similarity(desc_q.get(u), desc_r.get(v))
            cfg[k] = got
        return got

    auc_pairs: List[Tuple[float, float]] = []
    rank_base: List[int] = []
    rank_cand: Dict[str, List[int]] = {"%d/%.2f" % c: [] for c in CANDIDATES}
    rank_r2: Dict[str, List[int]] = {"whole/%.2f" % w: [] for (_K, w) in CANDIDATES}
    outside_base = 0
    n_truth = 0
    for u in nodes_a:
        vt = truth.get(u)
        row = rows.get(u) or []
        if vt is None or not row:
            continue
        n_truth += 1
        rb = _ranks(row, vt)
        if rb is None:
            continue
        rank_base.append(rb)
        if rb > 24:
            outside_base += 1
        # the hardest impostor: the best-scoring candidate that is not the truth
        for (v, _s) in row:
            if v != vt:
                auc_pairs.append((_cfg(u, vt), _cfg(u, v)))
                break
        sim = {v: s for (v, s) in row}
        for (K, w) in CANDIDATES:
            head = row[:K]
            rest = row[K:]
            head_keys = {v for (v, _s) in head}
            rr = sorted(head, key=lambda t: (-((1.0 - w) * sim[t[0]] + w * _cfg(u, t[0])),
                                             t[0]))
            # keep the tail in its original order behind the re-ranked head
            tail = [(v, s) for (v, s) in rest if v not in head_keys]
            r = _ranks(rr + tail, vt)
            if r is not None:
                rank_cand["%d/%.2f" % (K, w)].append(r)
            rr2 = sorted(row, key=lambda t: (-((1.0 - w) * sim[t[0]] + w * _cfg(u, t[0])),
                                             t[0]))
            r2 = _ranks(rr2, vt)
            if r2 is not None:
                rank_r2["whole/%.2f" % w].append(r2)

    # 3. per-image statistics
    lum = img.astype(float).mean(axis=2)
    lum = lum - lum.mean()
    F = np.fft.rfft2(lum)
    ac = np.fft.irfft2(F * np.conj(F), s=lum.shape)
    ac = ac / max(float(ac[0, 0]), 1e-9)
    prof = ac[: lum.shape[0] // 2, 0]
    prof[0] = -1.0
    period = int(np.argmax(prof)) if len(prof) else 0
    coh = []
    scale = []
    for v in nodes_b:
        sv = g0.sigs.get(v)
        if sv is None:
            continue
        q = getattr(sv[1], "quantized", ()) or ()
        if len(q) >= 2:
            coh.append(float(q[1]))
        if len(q) >= 3:
            scale.append(float(q[2]))
    dist_px = []
    for (u, src, name) in sq.incident.get(nodes_a[0], ())[:0]:
        pass
    for u in nodes_a:
        for (other, is_source, name) in sq.incident.get(u, ()):
            if not is_source:
                continue
            pv = sq.index.get((u, other, name))
            if pv is None:
                continue
            d = float(getattr(pv, "distance", float("nan")))
            sc = float(getattr(pv, "source_scale", float("nan")))
            if math.isfinite(d) and math.isfinite(sc) and sc > 0:
                dist_px.append(d * sc)

    def _auc(pairs: List[Tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        win = sum(1.0 if a > b else 0.5 if a == b else 0.0 for (a, b) in pairs)
        return win / len(pairs)

    def _stats(v: List[int]) -> Dict[str, Any]:
        if not v:
            return {"n": 0}
        return {"n": len(v), "median": float(st.median(v)),
                "rank1": sum(1 for x in v if x == 1) / len(v),
                "outside6": sum(1 for x in v if x > 6) / len(v),
                "outside24": sum(1 for x in v if x > 24) / len(v)}

    return {"tag": tag, "pose": pose, "queries": n_truth,
            "row_size_median": float(st.median([len(rows[u]) for u in rows])) if rows else 0,
            "occupancy": {"median": float(st.median(occ)) if occ else 0,
                          "share_gt1": (sum(1 for x in occ if x > 1) / len(occ)) if occ else 0.0,
                          "median_true": float(st.median(occ_true)) if occ_true else 0,
                          "distinct_keys_median": float(st.median(distinct)) if distinct else 0,
                          "n": len(occ)},
            "config_auc": _auc(auc_pairs), "config_pairs": len(auc_pairs),
            "rank_base": _stats(rank_base),
            "rank_r1": {k: _stats(v) for k, v in rank_cand.items()},
            "rank_r2": {k: _stats(v) for k, v in rank_r2.items()},
            "outside_cap24_base": (outside_base / max(len(rank_base), 1)),
            "image": {"texture_period_px": period,
                      "coherence_median": float(st.median(coh)) if coh else None,
                      "coherence_q1": float(np.percentile(coh, 25)) if coh else None,
                      "coherence_q3": float(np.percentile(coh, 75)) if coh else None,
                      "place_spacing_px_median": float(st.median(dist_px)) if dist_px else None,
                      "place_spacing_px_n": len(dist_px),
                      "places_a": len(nodes_a), "places_b": len(nodes_b)}}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    jobs = [(t, pose, spec) for (pose, spec) in POSES for t in DEV]
    cells: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            cells.append(r)
            print("  %-14s %-6s q %4d row %4.0f occ %4.0f(>1 %.2f) cfgAUC %s rank "
                  "%.1f->%s"
                  % (r["tag"], r["pose"], r["queries"], r["row_size_median"],
                     r["occupancy"]["median"], r["occupancy"]["share_gt1"],
                     "None" if r["config_auc"] is None else "%.3f" % r["config_auc"],
                     r["rank_base"].get("median", float("nan")),
                     " ".join("%s:%.1f" % (k, v.get("median", float("nan")))
                              for k, v in r["rank_r1"].items())), flush=True)
    MF.write_result(RESULT, {"cells": cells, "candidates": [list(c) for c in CANDIDATES],
                             "units": "ranks in places; period and spacing in px; AUC "
                                      "unitless; coherence in coarse slot units",
                             "note": "truth is read only for MEASUREMENT here; the "
                                     "candidates themselves are truth-free"})
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
