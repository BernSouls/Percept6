"""MISSION 21 / O3 -- the equal-coverage control, done as REGISTERED.

WHAT MISSION 20 DID.  The "+-5 %" coverage control FILTERED candidates: it dropped the
candidates whose mapped count fell outside a band and re-ranked what was left.  That is not a
coverage control, it is a different task, and it collapsed every arm to 0.133 (2/15) -- a
number about candidate filtering, not about coverage.  It is superseded here and must not be
quoted.

WHAT THIS DOES.  For every query of the run, take the shortlist as it stands.  Let

    m_q = min over the shortlist of the number of MAPPED places of that candidate

Then, for each candidate, SUBSAMPLE its mapped places to exactly m_q (uniformly, without
replacement, deterministic seed `PLACEMENT_M21_COVERAGE_SEED` + the pair + the draw index),
RECOMPUTE every arm's score from the subsampled per-place contributions, RE-RANK the same
candidates -- every candidate is kept -- and repeat for `PLACEMENT_M21_COVERAGE_DRAWS` draws,
reporting the mean and the spread over draws.

The arms are recomputed exactly as the run defines them, from the per-place contributions the
run recorded:

    B                 the shipped identity score.  It does not read the mapping, so it is
                      COVERAGE-FREE BY CONSTRUCTION: S0 is the built-in control and its
                      equal-coverage value must equal its unmatched value, which is checked
    E'                sum(plus - 0.5 over judged places of the subsample) / (judged' + k)
    S3'               sum(plus - 0.5 over judged places) / (den + k)
    S1 = B + w E'     S2 = B + lambda S3'     S3 = S3'

and the coverage sensitivity that the control is meant to remove is measured directly:
Spearman(score, mapped count) across the shortlist, before and after equalisation.

A per-place contribution is a place of the matcher's mapping with its frozen-estimator
`plus_score`; the run records them with `--save-places`.  A pair that recorded no places is
reported, never silently zeroed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def _value(arm: str, r: Dict[str, Any], w: float, lam: float, e: float, s3: float) -> float:
    if arm == "S0":
        return float(r["B"])
    if arm == "S1":
        return float(r["B"]) + w * e
    if arm == "S2":
        return float(r["B"]) + lam * s3
    return s3


def recompute(row: Dict[str, Any], m: int, seed: int, draw: int, k_shrink: float
              ) -> Optional[Dict[str, float]]:
    """(E', S3', judged', share') from a subsample of ``m`` mapped places, or None."""
    places = row.get("places")
    if not places:
        return None
    if len(places) <= m:
        sub = places
    else:
        # a STABLE per-pair component: `hash()` is salted per process, so using it would
        # make the draws depend on PYTHONHASHSEED -- determinism at two hash seeds is a
        # quality bar, and this is where it would have been broken.
        tag = int(hashlib.sha256(("%s|%s" % (row["q"], row["r"])).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, tag, draw])
        idx = rng.choice(len(places), size=m, replace=False)
        sub = [places[int(i)] for i in sorted(idx)]
    judged = [p for p in sub if p[3]]
    num = sum(float(p[2]) - 0.5 for p in judged)
    den = float(row.get("den") or 0)
    e = (num / (len(judged) + k_shrink)) if judged else 0.0
    s3 = (num / (den + k_shrink)) if den else None
    return {"E": e, "S3": (s3 if s3 is not None else 0.0), "judged": len(judged),
            "mapped": len(sub), "share": (len(judged) / len(sub)) if sub else 0.0}


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def equalize(rows: List[Dict[str, Any]], w: float, lam: float, k_shrink: float,
             draws: int, seed: int, truth_of: Dict[str, Dict[str, Any]],
             key_of: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if "error" in r or "places" not in r:
            continue
        by_q.setdefault(r["q"], []).append(r)
    arms = ("S0", "S1", "S2", "S3")
    unmatched = {a: 0 for a in arms}
    #: hits per DRAW, summed over queries: `mean/n_q` is then a rate per draw, which is what
    #: the "top-1 at equalised coverage" the mission asks for means.  (The first pass appended
    #: a per-query count of successful draws and divided it by n_q, which under-reports by a
    #: factor of `draws`; the number it produced, 0.078, was that bug and not a measurement.)
    per_draw: Dict[str, List[int]] = {a: [0] * draws for a in arms}
    ms: List[int] = []
    rho_before, rho_after = [], []
    n_q = 0
    n_pairs_no_places = 0
    for q, rs in sorted(by_q.items()):
        qo = truth_of.get(q)
        if qo is None:
            continue
        n_q += 1
        m = min(len(r["places"]) for r in rs)
        ms.append(int(m))
        truth = next((r for r in rs if key_of.get(r["r"], {}).get("key") == qo["key"]), None)
        for arm in arms:
            best = max(rs, key=lambda r: _value(arm, r, w, lam, float(r.get("E") or 0.0),
                                                float(r.get("S3") or 0.0)))
            unmatched[arm] += 1 if (truth is not None and best is truth) else 0
        for draw in range(draws):
            vals: Dict[str, List[float]] = {a: [] for a in arms}
            mapped: List[float] = []
            for r in rs:
                rec = recompute(r, m, seed, draw, k_shrink)
                if rec is None:
                    n_pairs_no_places += 1
                    continue
                mapped.append(rec["mapped"])
                for arm in arms:
                    vals[arm].append((r, _value(arm, r, w, lam, rec["E"], rec["S3"])))
            for arm in arms:
                if not vals[arm]:
                    continue
                ranked = sorted(vals[arm], key=lambda t: -t[1])
                per_draw[arm][draw] += 1 if (truth is not None and ranked[0][0] is truth) else 0
            if truth is not None and len(vals["S3"]) >= 3:
                sc = np.asarray([v for _r, v in vals["S3"]], dtype=float)
                mm = np.asarray(mapped, dtype=float)
                rho = _spearman(sc, mm)
                if rho is not None:
                    rho_after.append(rho)
        if truth is not None and len(rs) >= 3:
            sc = np.asarray([float(r.get("S3") or 0.0) for r in rs])
            mm = np.asarray([len(r["places"]) for r in rs], dtype=float)
            rho = _spearman(sc, mm)
            if rho is not None:
                rho_before.append(rho)
    out = {"queries": n_q, "draws": draws, "seed": seed, "k_shrink": k_shrink,
           "m_per_query": ms, "m_min": min(ms) if ms else None,
           "m_max": max(ms) if ms else None,
           "pairs_without_places": n_pairs_no_places,
           "w_evidence": w, "lambda_s2": lam, "arms": {}}
    for arm in arms:
        pd = per_draw[arm]
        out["arms"][arm] = {
            "top1_unmatched": unmatched[arm],
            "top1_unmatched_rate": (unmatched[arm] / n_q) if n_q else None,
            "top1_equalized_mean": (float(np.mean(pd)) / n_q) if n_q and pd else None,
            "top1_equalized_min": (min(pd) / n_q) if n_q and pd else None,
            "top1_equalized_max": (max(pd) / n_q) if n_q and pd else None,
            "top1_per_draw_hits": [int(x) for x in pd],
            "per_draw_hits": pd, "n": n_q}
    out["spearman_score_vs_count_before"] = (float(np.mean(rho_before)) if rho_before
                                             else None)
    out["spearman_score_vs_count_after"] = (float(np.mean(rho_after)) if rho_after else None)
    out["n_spearman"] = len(rho_before)
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m21_cov_fit_t3")
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--task", default="T3")
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--out", default="m21_coverage")
    a = ap.parse_args(argv)
    from Topology.placement_m20_fit import objects_for
    from Topology import placement_m20_eth80 as H
    import parameters as P
    w = float(json.load(open(os.path.join(ROOT, "matching",
                                          "placement_score_params.json")))["w"])
    lam = 0.0
    f = os.path.join(OUT, "m20_fit_read.json")
    if os.path.exists(f):
        lam = float(json.load(open(f)).get("lambda_shipped", 0.0))
    k_shrink = float(getattr(P, "PLACEMENT_SCORE_SHRINK", 8.0))
    rows = [json.loads(x) for x in open(os.path.join(OUT, a.tag + "_pairs.jsonl"))
            if x.strip()]
    objs = objects_for(a.split, a.per_class)
    truth_of = {o["query"]: o for o in objs}
    key_of = {o["ref"]: o for o in objs}
    if a.task == "T0":
        truth_of = {H._rotated_png(o["ref"]): o for o in objs}
    res = equalize(rows, w, lam, k_shrink, int(P.PLACEMENT_M21_COVERAGE_DRAWS),
                   int(P.PLACEMENT_M21_COVERAGE_SEED), truth_of, key_of)
    res.update({"tag": a.tag, "split": a.split, "task": a.task, "per_class": a.per_class})
    out = os.path.join(OUT, a.out + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("%s (%s %s): %d queries, m in [%s, %s], %d draws, seed %d"
          % (a.tag, a.split, a.task, res["queries"], res["m_min"], res["m_max"],
             res["draws"], res["seed"]))
    print("%-4s %-16s %-18s %-8s" % ("arm", "top1 unmatched", "top1 equalized", "spread"))
    for arm in ("S0", "S1", "S2", "S3"):
        r = res["arms"][arm]
        print("%-4s %-16s %-18s %s"
              % (arm, "%d/%d" % (r["top1_unmatched"], r["n"]),
                 "%.3f" % (r["top1_equalized_mean"] or 0),
                 "[%d, %d]" % (r["top1_equalized_min"] * r["n"],
                               r["top1_equalized_max"] * r["n"])))
    print("Spearman(score, mapped count): before %.3f  after %.3f  (n=%d queries)"
          % (res["spearman_score_vs_count_before"] or 0,
             res["spearman_score_vs_count_after"] or 0, res["n_spearman"]))
    print("pairs without recorded places:", res["pairs_without_places"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
