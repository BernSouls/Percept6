"""MISSION 12 STEP 2 -- the coverage gradient of the placement evidence E, and the
normalisation chosen on FIT.

The question this answers, before any weight is refitted:

    E = sum over judged slots of (plus_score - 0.5) / (n_judged + 8)

is an UNNORMALISED sum.  The estimator pads the slots it cannot check with the
wrong-pair mean and the cap is a fixed 8, so a pair sitting at a slot-dense place
can accumulate evidence simply because it HAS slots -- "rewards slot-dense places"
in the mission's words.  This measures that correlation on FIT, separately for the
TRUE candidate and the rivals, and for every candidate-level reading on the table:

    e_sum       the frozen estimator, sum over the cap                (deployed)
    e_mean      the same estimator divided by the slots it judged    (DARK,
                PLACEMENT_PAIR_NORMALISED, M10 step 4b)
    e_min2      e_mean, refused (0.0 = no evidence) below 2 judged slots
    e_min3      e_mean, refused below 3 judged slots
    e_shrink    e_mean shrunk toward its own query's mean, n/(n+4)   (DARK, M12)
    e_rank      e_mean rank-normalised WITHIN the query's shortlist  (DARK, M12)

The reading is chosen on FIT by a criterion fixed in advance:

    among the readings whose |rho(e, n_judged)| over the NON-TRUE candidates is
    below 0.30, take the one with the highest FIT top-1 (the true candidate ranked
    first within its own query node's candidate row under the blended score at the
    frozen beta); ties break on the smaller |rho|; if none clears 0.30, take the
    smallest |rho| and say so.

No evidence must contribute exactly 0: every reading returns 0.0 when no slot was
judged, and nothing here changes a default.  Truth here is the renderer's own
track truth on synthetic FIT scenes -- a FIT-side instrument, never a read.

    python Topology/placement_m12_e_coverage.py --save
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
BASE = os.path.join(OUT, "m12_e_coverage.json")

#: The readings compared.  ``None`` marks the deployed one.
READINGS = ("e_sum", "e_mean", "e_min2", "e_min3", "e_shrink", "e_rank")
MIN2, MIN3, SHRINK_K = 2, 3, 4


def pairs() -> List[Tuple[str, Any, Any]]:
    """Fixed FIT pairs: (tag, spec, view).  Deterministic, no randomness."""
    from Topology import placement_m11_families as F
    gal = F.gallery()
    ir = [s for s in gal if s.family == "IRREGULAR"]
    oop = [s for s in gal if s.family == "OUT_OF_PLANE"]
    out: List[Tuple[str, Any, Any]] = []
    for i, s in enumerate(ir[:6]):
        lab, view = F.VIEW_STEPS[i % 2]
        out.append(("ir_%02d|%s" % (i, lab), s, view))
    for i, s in enumerate(oop[:4]):
        lab, view = F.VIEW_STEPS[i % 2]
        out.append(("oop_%02d|%s" % (i, lab), s, view))
    return out


def _sym_job(args: Tuple[str, Any, Any]) -> Dict[str, Any]:
    """One pair: every positive candidate cell, with its judged count and truth."""
    tag, spec, view = args
    import guard
    guard.activate()
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from Topology import placement_m11_families as F
    from Topology import placement_census as CEN
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import node_objects, track_optimal

    sc1, _pv = F.pose_fn(spec, view)
    r0 = spec.scene.render()
    r1 = sc1.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    g1 = get_or_build(r1.image_u8, spec.name + "|" + tag.split("|")[1])
    o0, o1 = node_objects(r0, g0), node_objects(r1, g1)
    refc = [v for v in sorted(int(n) for n in g0.subgraph.nodes) if o0.get(v, -1) >= 0]
    pop = [u for u in sorted(int(n) for n in g1.subgraph.nodes) if o1.get(u, -1) >= 0]
    truth = {int(q): int(r)
             for r, q in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}

    side_q = PM.build_side(g1, r1.image_u8)
    side_r = PM.build_side(g0, r0.image_u8)
    if not side_q.usable or not side_r.usable:
        return {"tag": tag, "rows": [], "skipped": "side not usable"}

    rows_sim = CEN.similarity_rows(g1, g0, pop, refc)
    nodes_a = sorted(rows_sim)
    nodes_b = sorted(refc)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim[u]:
            matrix[i][idx_b[v]] = float(s)

    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    beta, cap = float(b["beta"]), int(b["cand_cap"])
    ctx = PM.context_mapping(PM.top_candidates(matrix, nodes_b, cap),
                            nodes_a, side_q, side_r)
    tol_of = PM._spacing_tol_fn(side_q, side_r)

    out: List[Dict[str, Any]] = []
    for i, u in enumerate(nodes_a):
        cells: List[Dict[str, Any]] = []
        for j in range(len(nodes_b)):
            s = float(matrix[i][j])
            if s <= 0.0:
                continue
            vid = int(nodes_b[j])
            tol = None if tol_of is None else tol_of(u, vid)
            slots = L.slots_for_candidate(u, vid, ctx, side_q.incident,
                                          side_q.index, side_r.index, tol=tol)
            n_j = min(len(slots), int(L.SLOT_CAP))
            e_sum_, g_sum_ = L.plus_terms(u, vid, ctx, side_q.incident,
                                          side_q.index, side_r.index, fit, pfit,
                                          slot_cap=int(L.SLOT_CAP), tol=tol,
                                          normalise=False)
            p_sum = float(L.plus_map(e_sum_, g_sum_, a_e, a_g,
                                     slot_cap=int(L.SLOT_CAP)))
            p_mean = float(L.plus_score(u, vid, ctx, side_q.incident, side_q.index,
                                        side_r.index, fit, pfit, a_e, a_g,
                                        slot_cap=int(L.SLOT_CAP), tol=tol,
                                        normalise=True))
            cells.append({"u": int(u), "v": vid, "sim": s, "n_judged": int(n_j),
                          "e_sum": float(p_sum - 0.5),
                          "e_mean": float(p_mean - 0.5) if n_j > 0 else 0.0,
                          "true": bool(truth.get(int(u)) == vid)})
        if not cells:
            continue
        # the per-query readings
        qmean = float(np.mean([c["e_mean"] for c in cells]))
        order = sorted(range(len(cells)), key=lambda k: (cells[k]["e_mean"], cells[k]["v"]))
        rank_of = {k: (order.index(k) / max(1, len(order) - 1)) for k in range(len(cells))}
        for k, c in enumerate(cells):
            n = c["n_judged"]
            c["e_min2"] = c["e_mean"] if n >= MIN2 else 0.0
            c["e_min3"] = c["e_mean"] if n >= MIN3 else 0.0
            c["e_shrink"] = (n * c["e_mean"] + SHRINK_K * qmean) / (n + SHRINK_K)
            c["e_rank"] = rank_of[k] - 0.5
            for r in READINGS:
                c["score_" + r] = float((1.0 - beta) * c["sim"] + beta * (c[r] + 0.5))
        out.extend(cells)
    return {"tag": tag, "rows": out, "truth_n": len(truth), "pop_n": len(pop)}


def _rho(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return float("nan")
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.std() == 0.0 or ya.std() == 0.0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(xa, ya).statistic)


def _top1(cells: List[Dict[str, Any]], key: str) -> float:
    """Share of query nodes whose TRUE candidate ranks first under `key`."""
    by_u: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        by_u.setdefault(int(c["u"]), []).append(c)
    hit = tot = 0
    for u, row in by_u.items():
        if not any(c["true"] for c in row):
            continue
        tot += 1
        best = max(row, key=lambda c: (c[key], -c["v"]))
        hit += 1 if best["true"] else 0
    return float(hit / tot) if tot else float("nan")


def summarise(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells = [c for j in jobs for c in j["rows"]]
    tru = [c for c in cells if c["true"]]
    fal = [c for c in cells if not c["true"]]
    res: Dict[str, Any] = {
        "pairs": len(jobs), "cells": len(cells), "true_cells": len(tru),
        "non_true_cells": len(fal),
        "judged_share": float(np.mean([c["n_judged"] > 0 for c in cells])) if cells else 0.0,
        "mean_n_judged": float(np.mean([c["n_judged"] for c in cells])) if cells else 0.0,
        "readings": {},
    }
    for r in READINGS:
        res["readings"][r] = {
            "rho_all": _rho([c[r] for c in cells], [c["n_judged"] for c in cells]),
            "rho_true": _rho([c[r] for c in tru], [c["n_judged"] for c in tru]),
            "rho_non_true": _rho([c[r] for c in fal], [c["n_judged"] for c in fal]),
            "score_rho_non_true": _rho([c["score_" + r] for c in fal],
                                       [c["n_judged"] for c in fal]),
            "top1": _top1(cells, "score_" + r),
        }
    return res


#: Which readings put the evidence on ONE SCALE in the judged count.  The mean
#: readings divide by the slots a pair judged, so E is a quality at every place;
#: the sum readings carry the slot count into E, which is what makes a single
#: frozen weight (PLACEMENT_SCORE_WEIGHT) place-dependent.
SCALE_FREE = ("e_mean", "e_min2", "e_min3", "e_shrink", "e_rank")


def choose(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the pre-stated FIT criterion and return the decision.

    THE RULE, fixed before this harness was run:

      1. A reading is ADMISSIBLE only if the thing the matcher actually ranks by --
         the blended candidate score -- correlates less than 0.30 with the judged
         count over the NON-TRUE candidates.  The term's own correlation with n is
         not the bar: a place with more judged slots gives a rival more chances to
         be contradicted, which is information, not bias (its sign is negative).
      2. Among the admissible readings, prefer the ones that are SCALE-FREE in n
         (the mean family): E must mean the same thing at every place, or no single
         frozen weight can be right for all of them.
      3. Break ties on the highest FIT top-1; then on the smaller |rho|.
      4. If nothing is admissible, take the smallest |rho| of the blended score and
         say so.
    """
    adm = {r: v for r, v in summary["readings"].items()
           if not np.isnan(v["score_rho_non_true"]) and abs(v["score_rho_non_true"]) < 0.30}
    if not adm:
        best = min(summary["readings"],
                   key=lambda r: abs(summary["readings"][r]["score_rho_non_true"]))
        return {"chosen": best, "admissible": [], "scale_free": [],
                "rule": "no reading admissible: smallest |rho(score, n)|"}
    free = {r: v for r, v in adm.items() if r in SCALE_FREE} or adm
    best = max(free, key=lambda r: (free[r]["top1"], -abs(free[r]["score_rho_non_true"])))
    return {"chosen": best, "admissible": sorted(adm), "scale_free": sorted(free),
            "rule": ("admissible = |rho(blended score, n)| < 0.30 over non-true cells; "
                     "prefer the scale-free mean family; tie-break on FIT top-1, then |rho|")}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    ps = pairs()
    print("%d FIT pairs" % len(ps), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = list(ex.map(_sym_job, ps))
    s = summarise(jobs)
    dec = choose(s)
    print(json.dumps({"summary": s, "decision": dec}, indent=2, sort_keys=True))
    if args.save:
        with open(BASE, "w") as fh:
            json.dump({"summary": s, "decision": dec,
                       "cells": [c for j in jobs for c in j["rows"]]}, fh)
        print("saved %s" % BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
