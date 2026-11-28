"""MISSION 21 / O1 -- fit the FROZEN standardization for stage 1, on FIT, once.

The Mission-20 stage-1 score standardized the query by each REFERENCE's own mean, std and
kept dimensions.  Two gallery items therefore scored the same query on two different rulers,
and a shortlist that compares ``score(A, B1)`` with ``score(A, B2)`` was comparing
incomparable numbers.  This file fits one ruler on FIT and writes it down:

    matching/placement_prefilter_params.json

  mu_D / sd_D                        per-slot mean and sd of the log nearest-neighbour
                                     distances
  mu_cos / sd_cos, mu_sin / sd_sin   per-slot mean and sd of the direction's cos and sin
  keep_D                             slots with enough support and non-zero spread; other
                                     slots are MASKED OUT (no feature is chosen per
                                     reference, and no slot is fabricated by padding)
  sigma_d, sigma_a                   the kernel widths, chosen on the FIT fit-half
  K                                  the descriptor length, 24: the SAME K the matcher's
                                     configuration descriptor has always used.  The Mission-20
                                     bag used ASSIGN_K = 6 while its docstring and the mission
                                     text said 8; there is now one stated K and the file says
                                     which one and why.

PROVENANCE AND THE HALF-SPLIT.  The parameters are fitted on the FIT objects at even index,
the widths are chosen on those same even-index queries, and every reported number is on the
odd-index objects, which are never fitted.  DEV is untouched here.

Stages (run in this order; each is resumable):
    --stage params   fit the standardization, write the file (widths provisional)
    --stage terms    build the sigma-free pair table for the split (cached)
    --stage choose   choose sigma_d/sigma_a on the fit half, rewrite the file
    --stage report   recall@8 per task on both halves, with n and spread
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PARAMS = os.path.join(ROOT, "matching", "placement_prefilter_params.json")
S = 8


def code_digest() -> str:
    h = hashlib.sha256()
    for rel in ("matching/placement_prefilter.py", "matching/placement_context.py"):
        with open(os.path.join(ROOT, rel), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def bags_path(split: str, per_class: int = 0) -> str:
    return os.path.join(OUT, "m21_%s_pc%d_bags.pkl" % (split.lower(), per_class))


def ensure_bags(split: str, workers: int = 2, per_class: int = 0) -> Dict[str, Any]:
    """Every bag this split can need, built once and cached.  Reused by every later stage."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    from Topology import placement_m20_eth80 as H
    p = bags_path(split, per_class)
    bags = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    objs = H.objects(split)
    if per_class:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for o in objs:
            per.setdefault(o["cls"], []).append(o)
        objs = [o for c in sorted(per) for o in per[c][:int(per_class)]]
    want = sorted({o["ref"] for o in objs} | {o["query"] for o in objs})
    need = [x for x in want if x not in bags or "bag_v2" not in bags[x]]
    if need:
        print("bags: building %d of %d images" % (len(need), len(want)), flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(H.bag_job, need, chunksize=1)):
                bags[r["path"]] = r
                if (i + 1) % 10 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(need), time.time() - t0), flush=True)
        with open(p, "wb") as fh:
            pickle.dump(bags, fh, protocol=4)
        print("bags done %.0fs" % (time.time() - t0), flush=True)
    missing = [x for x in want if x not in bags]
    if missing:
        raise SystemExit("bags missing after build: %d" % len(missing))
    return bags


def split_objects(split: str, per_class: int = 0) -> List[Dict[str, Any]]:
    from Topology import placement_m20_eth80 as H
    objs = H.objects(split)
    if per_class:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for o in objs:
            per.setdefault(o["cls"], []).append(o)
        objs = [o for c in sorted(per) for o in per[c][:int(per_class)]]
    return objs


def fit_standardization(bags: Dict[str, Any], paths: List[str], k: int = 24) -> Dict[str, Any]:
    """Per-slot mean, sd and support over every place of the given images.

    A slot is KEPT when it has support and spread; otherwise it is masked out everywhere,
    identically for every image.  That is the whole difference from Mission 20: the kept set
    is a property IF the corpus, not of the reference a query happens to be compared with.
    """
    Ds, As = [], []
    for p in paths:
        b = bags[p]["bag_v2"]
        m = b["M"] > 0
        if not bool(m.any()):
            continue
        Ds.append(np.where(m, b["D"], np.nan))
        As.append(np.where(m, b["A"], np.nan))
    D = np.concatenate(Ds, axis=0) if Ds else np.zeros((0, k))
    A = np.concatenate(As, axis=0) if As else np.zeros((0, k))
    C, Sn = np.cos(A), np.sin(A)

    def stats(X: np.ndarray) -> Tuple[List[float], List[float], List[int]]:
        mu, sd, cnt = [], [], []
        for i in range(X.shape[1]):
            col = X[:, i]
            col = col[~np.isnan(col)]
            cnt.append(int(col.size))
            mu.append(float(col.mean()) if col.size else 0.0)
            sd.append(float(col.std(ddof=1)) if col.size > 1 else 0.0)
        return mu, sd, cnt

    muD, sdD, cntD = stats(D)
    muC, sdC, cntC = stats(C)
    muS, sdS, cntS = stats(Sn)
    keep = [1 if (cntD[i] >= 20 and sdD[i] > 1e-6 and sdC[i] > 1e-6 and sdS[i] > 1e-6)
            else 0 for i in range(k)]
    guard = lambda v: [x if x > 1e-9 else 1.0 for x in v]      # noqa: E731
    return {"K": int(k), "mu_D": muD, "sd_D": guard(sdD), "mu_cos": muC, "sd_cos": guard(sdC),
            "mu_sin": muS, "sd_sin": guard(sdS), "keep_D": keep,
            "support_D": cntD, "support_cos": cntC, "support_sin": cntS,
            "n_places_fitted": int(D.shape[0]), "n_images_fitted": len(paths)}


def recall_from() -> None:                     # pragma: no cover - documented, see shortlist
    """Every recall number is computed by `Topology.placement_m21_shortlist.evaluate`."""
    raise NotImplementedError("use placement_m21_shortlist.evaluate")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="params",
                    choices=("params", "terms", "choose", "report", "all"))
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=PARAMS)
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_prefilter as PF
    from Topology import placement_m21_terms as T
    from Topology import placement_m21_shortlist as SL
    if not P.PLACEMENT_M19:
        raise SystemExit("set PLACEMENT_M19=1: this is the deployed configuration's stage 1")
    objs = split_objects(a.split, a.per_class)
    if a.stage in ("params", "all"):
        bags = ensure_bags(a.split, a.workers, a.per_class)
        even = [o for i, o in enumerate(objs) if i % 2 == 0]
        fit_paths = sorted({o["ref"] for o in even} | {o["query"] for o in even})
        st = fit_standardization(bags, fit_paths)
        st.update({"sigma_d": 0.35, "sigma_a": 0.5,
                   "path": os.path.relpath(a.out, ROOT),
                   "fitted_on": {"split": a.split, "per_class": a.per_class or "all",
                                 "rule": "even-index objects of the split; the odd-index "
                                         "objects are the report half and are never fitted",
                                 "n_images": len(fit_paths), "images": fit_paths},
                   "chosen_by": {"objective": "recall@8 on the split's mixed gallery, "
                                              "queries restricted to the fit half",
                                 "note": "filled in by --stage choose"},
                   "code_digest": code_digest(), "date": "2026-11-18",
                   "mission": "M21 O1"})
        with open(a.out, "w") as fh:
            json.dump(st, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("%s: fitted on %d images, kept %d of %d slots, %d places"
              % (a.split, len(fit_paths), sum(st["keep_D"]), st["K"],
                 st["n_places_fitted"]))
        print("wrote", a.out)
    if a.stage in ("terms", "all"):
        t = T.build(a.split, a.per_class, a.workers, force=(a.stage == "all"))
        print("terms: %d images, cached=%s" % (len(t["paths"]), t["cached"]))
    if a.stage in ("choose", "all"):
        t = T.build(a.split, a.per_class, a.workers)
        res = SL.evaluate_table(t, objs, S)
        best = res["best"]
        params = PF.load_params()
        params["sigma_d"], params["sigma_a"] = best["sigma_d"], best["sigma_a"]
        params["chosen_by"] = {"objective": res["objective"],
                               "grid": res["grid"], "n_queries": best["n"],
                               "fit_half_recall8_T3": best["recall8_T3"],
                               "note": "widths chosen on the FIT fit half only"}
        with open(a.out, "w") as fh:
            json.dump(params, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("chosen sigma_d %.2f sigma_a %.2f (fit-half recall@8 %.3f, n=%d)"
              % (best["sigma_d"], best["sigma_a"], best["recall8_T3"], best["n"]))
    if a.stage in ("report", "all"):
        t = T.build(a.split, a.per_class, a.workers)
        SL.report(t, objs, S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
