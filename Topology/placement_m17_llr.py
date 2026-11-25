"""MISSION 17 / O1 -- a DEFENSIBLE LLR table, as data, without moving a default.

The deployed table (`matching/placement_frozen_params.json`) was fitted on 2,574 true /
4,684 wrong residuals, carries NO per-bin counts, rises 5 times, has 16 of 24 bins
positive and pays +0.597 in the tail -- the Laplace prior of an empty bin read as
support.  Mission 16 added the two rules that make a table shippable
(`PLACEMENT_LLR_EMPTY_BIN_NEUTRAL`, `PLACEMENT_LLR_MONOTONE`) but could not apply them
to the deployed file without moving a default, and could not project the deployed table
honestly because the counts were not stored.

This writes a SECOND bundle, `matching/placement_frozen_params_m17.json`, selected only
when `PLACEMENT_LLR_TABLE_M17` is on:

  * fitted on the POOLED places-mode residuals of `m15_geometry.json`
    (13,578 true / 16,871 wrong, 4 images, three poses -- the M15 item-5b bank),
  * support 1.8 tol units in 36 bins of 0.05 tol (M15 measured the real-photo residual
    noise at 0.0235 tol; 0.05 is the coarsest bin that still resolves the term, and
    99.98 % of true residuals are inside the support),
  * EMPTY_BIN_NEUTRAL + MONOTONE on, MIN_BIN_COUNT 20,
  * WITH the per-bin true/wrong counts, so the freeze guard can check the weight
    profile and a later projection can be done honestly.

    python Topology/placement_m17_llr.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m17_llr.json")
BUNDLE_OUT = os.path.join(ROOT, "matching", "placement_frozen_params_m17.json")
BUNDLE_IN = os.path.join(ROOT, "matching", "placement_frozen_params.json")
RES_MAX = 1.8
N_BINS = 36
MIN_COUNT = 20


def pooled() -> tuple:
    d = json.load(open(os.path.join(OUT, "m15_geometry.json")))
    cells = [c for c in d["cells"] if c.get("mode") == "places"]
    T = np.array([x for c in cells for x in c.get("res_true", [])], dtype=float)
    W = np.array([x for c in cells for x in c.get("res_wrong", [])], dtype=float)
    return T, W, cells


def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import m15_manifest as MF

    T, W, cells = pooled()
    old = json.load(open(BUNDLE_IN))
    old_llr = np.array(old["fit"]["llr"], dtype=float)
    old_edges = np.array(old["fit"]["edges"], dtype=float)

    # ---- the exposure: what the tail is worth, measured
    exposure = {
        "wrong_n": int(W.size), "true_n": int(T.size),
        "wrong_beyond_3_5": int((W > 3.5).sum()),
        "wrong_beyond_3_5_share": float((W > 3.5).mean()),
        "wrong_beyond_2_75": int((W > 2.75).sum()),
        "wrong_beyond_2_75_share": float((W > 2.75).mean()),
        "true_beyond_1_8": int((T > RES_MAX).sum()),
        "true_beyond_1_8_share": float((T > RES_MAX).mean()),
        "true_median": float(np.median(T)), "true_q90": float(np.percentile(T, 90)),
        "wrong_median": float(np.median(W)),
        "units": "tolunits (payload_residual is tolerance-normalised and dimensionless)",
    }
    # what the DEPLOYED table pays a candidate sitting in the tail
    _old_fit = L.LLRFit(edges=old_edges, llr=old_llr, mu_wrong=0.0, n_true=0,
                        n_wrong=0)
    exposure["deployed_llr_at_5_0"] = float(_old_fit.of(5.0))
    exposure["deployed_llr_at_1_5"] = float(_old_fit.of(1.5))

    # ---- the refit, with the two rules ON
    P.PLACEMENT_LLR_EMPTY_BIN_NEUTRAL = True
    P.PLACEMENT_LLR_MONOTONE = True
    P.PLACEMENT_LLR_MIN_BIN_COUNT = MIN_COUNT
    fit = L.fit_llr(T, W, res_max=RES_MAX, n_bins=N_BINS)
    mono = L.llr_monotonicity(fit.llr, fit.edges)
    edges = np.asarray(fit.edges, dtype=float)
    ct = np.histogram(np.clip(T, 0.0, RES_MAX - 1e-9), bins=edges)[0]
    cw = np.histogram(np.clip(W, 0.0, RES_MAX - 1e-9), bins=edges)[0]
    empty_pos = [int(i) for i in range(N_BINS)
                 if (ct[i] + cw[i]) == 0 and float(fit.llr[i]) > 0.0]

    # ---- the bundle: the shipped one, with the new fit and its counts
    b = json.loads(json.dumps(old))
    from matching import placement_llr as _L
    b["fit"] = _L._fit_to_dict(fit)
    b["fit"]["bin_counts_true"] = [int(x) for x in ct]
    b["fit"]["bin_counts_wrong"] = [int(x) for x in cw]
    b["fit"]["weight_profile_digest"] = _profile_digest(ct, cw)
    b["constants"] = dict(slot_cap=int(L.SLOT_CAP), llr_clip=float(L.LLR_CLIP),
                          fixed_tol=float(L.FIXED_TOL), res_max=float(RES_MAX),
                          n_bins=int(N_BINS),
                          table="m17", monotone=True, empty_bin_neutral=True,
                          min_bin_count=int(MIN_COUNT))
    b["provenance"] = {
        "table": "m17", "fitted_on": "m15_geometry.json places-mode pooled residuals",
        "true_n": int(T.size), "wrong_n": int(W.size), "res_max": RES_MAX,
        "n_bins": N_BINS, "bin_width": RES_MAX / N_BINS,
        "units": "tolunits", "rises_after": mono["n_rises"],
        "empty_bins_with_positive_credit": empty_pos,
    }
    with open(BUNDLE_OUT, "w") as fh:
        json.dump(b, fh, indent=1, sort_keys=True)

    res: Dict[str, Any] = {
        "old": {"llr": [float(x) for x in old_llr],
                "edges": [float(x) for x in old_edges],
                "rises": L.llr_monotonicity(old_llr, old_edges)["n_rises"],
                "positive_bins": int((old_llr > 0).sum()),
                "n_bins": int(len(old_llr)),
                "res_max": float(old_edges[-1]),
                "counts_stored": "bin_counts_true" in old["fit"],
                "tail_llr": float(old_llr[-1])},
        "new": {"llr": [float(x) for x in fit.llr],
                "edges": [float(x) for x in edges],
                "rises": mono["n_rises"], "positive_bins": int((fit.llr > 0).sum()),
                "empty_bins_with_positive_credit": empty_pos,
                "bin_counts_true": [int(x) for x in ct],
                "bin_counts_wrong": [int(x) for x in cw],
                "n_bins": N_BINS, "res_max": RES_MAX,
                "isotonic": fit.extra.get("isotonic", {})},
        "exposure": exposure, "cells_pooled": len(cells),
        "bundle": os.path.relpath(BUNDLE_OUT, ROOT),
        "units": "residuals in tolunits; llr in nats",
    }
    MF.write_result(RESULT, res)

    print("old: %d bins over %.2f, %d positive, %d rises, tail llr %+.3f, counts stored %s"
          % (len(old_llr), old_edges[-1], int((old_llr > 0).sum()),
             res["old"]["rises"], old_llr[-1], res["old"]["counts_stored"]))
    print("new: %d bins over %.2f, %d positive, %d rises, %d empty-with-credit"
          % (N_BINS, RES_MAX, int((fit.llr > 0).sum()), mono["n_rises"],
             len(empty_pos)))
    print("exposure: wrong>3.5 %d (%.4f%%), wrong>2.75 %d (%.3f%%), true>1.8 %d (%.3f%%)"
          % (exposure["wrong_beyond_3_5"], 100 * exposure["wrong_beyond_3_5_share"],
             exposure["wrong_beyond_2_75"], 100 * exposure["wrong_beyond_2_75_share"],
             exposure["true_beyond_1_8"], 100 * exposure["true_beyond_1_8_share"]))
    print("deployed pays %+.3f at 5.0 tol and %+.3f at 1.5 tol"
          % (exposure["deployed_llr_at_5_0"], exposure["deployed_llr_at_1_5"]))
    print("wrote %s and %s" % (RESULT, BUNDLE_OUT))
    return 0


def _profile_digest(ct, cw) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps([int(x) for x in ct] + [int(x) for x in cw]).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
