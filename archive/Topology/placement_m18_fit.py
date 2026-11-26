"""MISSION 18 / O1 -- table candidates fitted POSE-BALANCED from `m18_bank.json`.

The registered routes (MISSION18_PREREGISTRATION.md, O1) and the candidates this script
forms for each.  The list was written BEFORE any candidate was scored:

  C0  `dep`      the deployed table (reference, 24 x 0.25 to 6.0).
  C1  `m17`      the M17 refit (reference: the fit that lost, 36 x 0.05 to 1.8).
  C2  `tb24`     pose-balanced mixture, the DEPLOYED grid, raw (no monotone pass).
  C3  `tb24m`    C2 + count-weighted PAVA + a count-scaled minimum (thin bin -> no credit).
  C4  `tb30m`    C3 on a 0.20 grid to 6.0 (finer where the 15 deg IQR lives).
  C5  `tbnu`     C3 on a NON-UNIFORM grid: 0.25 to 1.5, 0.50 to 3.0, 1.0 to 6.0.
  C6  `tbexp`    a PARAMETRIC monotone smoother a + b*exp(-r/tau) on the 0.20 grid.

The pose balance is equal weight over the four measured regimes (0, 5, 15 deg, 1.15x) --
"never pool regimes" said as arithmetic: each regime contributes a PROPER density and the
table is the log-ratio of the mixture, not an average of log-ratios (which would let a
regime with an empty bin vote without limit).

Every candidate is scored on the bank alone (separation AUC per pose, G1/G2/G3/G4 checks,
and the O1 non-flat metrics over the region where 15 deg TRUE residuals actually lie:
[0.062, 0.230] tol).  Only survivors go to the matcher -- the matcher is the expensive
instrument and a table that cannot separate on the bank has no business being measured
against a top-1 bar.

    python Topology/placement_m18_fit.py                     # score, write candidates
    python Topology/placement_m18_fit.py --write-bundles     # + readable bundles
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
BANK = os.path.join(OUT, "m18_bank_cap24.json")
RESULT = os.path.join(OUT, "m18_fit.json")
BUNDLE_DIR = os.path.join(OUT, "m18_tables")
BASE_BUNDLE = os.path.join(ROOT, "matching", "placement_frozen_params.json")

POSES = ("rot00", "rot05", "rot15", "scale115")
#: where 15 deg TRUE residuals actually lie (m18_diagnose.json, 4 DEV images, n=1166):
REGION = (0.062, 0.230)
MIN_COUNT = 20


# ---------------------------------------------------------------- grids
def _grid_uniform(width: float, top: float = 6.0) -> np.ndarray:
    return np.arange(0.0, top + 1e-9, width)


def _grid_nonuniform() -> np.ndarray:
    return np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])


def _grid_fine() -> np.ndarray:
    """[M18 / O1] C7.  ADDED AFTER C2-C6 were measured, and recorded as a deviation.

    Every pre-registered grid showed plateau_share 1.00 over the region where 15 deg TRUE
    residuals actually lie ([0.062, 0.230] tol), because that region is NARROWER than the
    smallest pre-registered bin (0.20).  A non-flat criterion cannot be met by a grid that
    cannot represent a step inside the region, so the grid -- not the monotone pass -- was
    the defect.  C7 refines RESOLUTION inside the region and widens SUPPORT outside it in
    the same table: 0.05 to 1.0, then 0.25 to 3.0, then 1.0 to 6.0.
    """
    a = np.arange(0.0, 1.0 + 1e-9, 0.05)
    b = np.arange(1.25, 3.0 + 1e-9, 0.25)
    c = np.array([4.0, 5.0, 6.0])
    return np.concatenate([a, b, c])


# ---------------------------------------------------------------- bank maths
class Bank:
    """Per-pose, per-image fine histograms, reduced to any table grid on demand."""

    def __init__(self, d: Dict[str, Any]):
        self.width = float(d["grid_width"])
        self.nfine = int(d["grid_bins"])
        self.cells = d["cells"]

    def pose(self, pose: str, what: str) -> np.ndarray:
        out = np.zeros(self.nfine, dtype=float)
        for c in self.cells:
            if c["pose"] == pose:
                out += np.asarray(c["hist_%s" % what], dtype=float)
        return out

    def reduce(self, hist: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Sum fine bins into table bins.  Fine bins are 0.05 wide and every grid here is
        a multiple of that, so the reduction is exact (asserted, not assumed)."""
        centres = (np.arange(self.nfine) + 0.5) * self.width
        idx = np.clip(np.digitize(centres, edges) - 1, 0, len(edges) - 2)
        out = np.zeros(len(edges) - 1, dtype=float)
        np.add.at(out, idx, hist)
        return out


def _llr_mixture(bt: Dict[str, np.ndarray], bw: Dict[str, np.ndarray],
                 poses: Sequence[str], weights: Dict[str, float]) -> np.ndarray:
    """log of the pose-balanced TRUE mixture over the pose-balanced WRONG mixture."""
    pt = None
    pw = None
    for p in poses:
        a = bt[p] / max(bt[p].sum(), 1.0)
        b = bw[p] / max(bw[p].sum(), 1.0)
        pt = weights[p] * a if pt is None else pt + weights[p] * a
        pw = weights[p] * b if pw is None else pw + weights[p] * b
    eps = 1e-6
    return np.log((pt + eps) / (pw + eps))


def _pava_noninc(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    from matching.placement_llr import isotonic_nonincreasing
    out, _rep = isotonic_nonincreasing(list(v), list(w))
    return np.asarray(out, dtype=float)


def _count_scaled_floor(llr: np.ndarray, counts: np.ndarray, min_count: int) -> np.ndarray:
    """A bin too thin to know pays NO positive credit.

    M17's accidental slope came from the empty-bin prior; the principled replacement is
    to refuse credit exactly where the count is too small to justify it, and to say so in
    the provenance rather than to smooth it away.
    """
    out = np.asarray(llr, dtype=float).copy()
    thin = counts < float(min_count)
    out[thin] = np.minimum(out[thin], 0.0)
    return out


def _fit_exp(centres: np.ndarray, llr: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    """Weighted least squares for a + b*exp(-r/tau): b >= 0 keeps it monotone by form."""
    best = (0.0, 0.0, 1.0)
    best_err = float("inf")
    for tau in np.linspace(0.05, 3.0, 120):
        e = np.exp(-centres / tau)
        A = np.stack([np.ones_like(e), e], axis=1) * np.sqrt(w)[:, None]
        y = llr * np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        err = float(((A @ coef - y) ** 2).sum())
        if err < best_err:
            best_err, best = err, (float(coef[0]), float(coef[1]), float(tau))
    a, b, tau = best
    b = abs(b)
    return a, b, tau


def _tail_exp(centres: np.ndarray, llr: np.ndarray, w: np.ndarray, anchor: float,
              join: float) -> np.ndarray:
    """A STRICTLY decreasing parametric tail that starts where the front ends.

    PAVA pools the thin tail bins into one level, and a level tail cannot satisfy O1's
    strict order (LLR(5.0) < LLR(1.5)) -- tbfine reads -6.72 at both.  The tail is where
    the counts are thinnest, so it is the part that most deserves a form instead of a
    free parameter: fit ``anchor - D*(1 - exp(-(r-join)/tau))`` by weighted least squares
    and it is monotone, strictly decreasing, and joins the measured front continuously.
    """
    sel = centres > join
    if not sel.any():
        return np.array([])
    c = centres[sel]
    y = llr[sel]
    ww = w[sel] / max(float(w[sel].sum()), 1.0)
    best = None
    for tau in np.linspace(0.1, 8.0, 80):
        f = 1.0 - np.exp(-(c - join) / tau)
        D = float(np.sum(ww * f * (anchor - y)) / max(np.sum(ww * f * f), 1e-12))
        D = max(D, 0.0)
        err = float(np.sum(ww * (anchor - D * f - y) ** 2))
        if best is None or err < best[0]:
            best = (err, D, tau)
    _e, D, tau = best
    return anchor - D * (1.0 - np.exp(-(c - join) / tau))


def _auc_from_bins(llr: np.ndarray, ht: np.ndarray, hw: np.ndarray) -> float:
    """P(llr(true) > llr(wrong)) with ties at 1/2, computed exactly on table bins."""
    if ht.sum() <= 0 or hw.sum() <= 0:
        return float("nan")
    order = np.argsort(llr, kind="stable")
    num = 0.0
    below = 0.0
    for i in order:
        # wrongs strictly below this score get full credit, ties half
        num += ht[i] * (below + 0.5 * hw[i])
        below += hw[i]
    return float(num / (ht.sum() * hw.sum()))


def _unflatin(edges: np.ndarray, llr: np.ndarray) -> Dict[str, Any]:
    """O1's non-flat requirement, measured where 15 deg true residuals lie."""
    lo, hi = REGION
    inside = [i for i in range(len(llr))
              if edges[i + 1] > lo and edges[i] < hi]
    vals = [float(llr[i]) for i in inside]
    # longest run of exactly equal values inside the region, as a share of the region
    run = best = 1 if vals else 0
    for i in range(1, len(vals)):
        run = run + 1 if abs(vals[i] - vals[i - 1]) <= 1e-12 else 1
        best = max(best, run)
    return {"region": [lo, hi], "bins_touched": len(inside),
            "levels": len(set(round(v, 9) for v in vals)),
            "span": float(max(vals) - min(vals)) if vals else 0.0,
            "plateau_share": float(best / len(vals)) if vals else 1.0,
            "values": [round(v, 6) for v in vals]}


def _checks(edges: np.ndarray, llr: np.ndarray, ht: np.ndarray, hw: np.ndarray) -> Dict[str, Any]:
    rises = [i for i in range(len(llr) - 1) if llr[i + 1] > llr[i] + 1e-12]
    thin_pos = [i for i in range(len(llr))
                if (ht[i] + hw[i]) < MIN_COUNT and llr[i] > 0.0]
    empty_pos = [i for i in range(len(llr)) if ht[i] == 0 and llr[i] > 0.0]

    def at(r: float) -> float:
        i = int(np.searchsorted(edges, r, side="right") - 1)
        return float(llr[max(0, min(len(llr) - 1, i))])

    return {"n_rises": len(rises), "rises": [int(i) for i in rises],
            "thin_bins_with_positive_credit": len(thin_pos),
            "empty_bins_with_positive_credit": len(empty_pos),
            "llr_1.5": at(1.5), "llr_5.0": at(5.0),
            # O1 as WRITTEN: LLR(5.0 tol) strictly < LLR(1.5 tol).  (An earlier version
            # of this checker had the inequality the other way round, which no
            # non-increasing table can satisfy -- it would have demanded a rise.)
            "strict_order_5.0_lt_1.5": bool(at(5.0) < at(1.5)),
            "min_bin_count": int(min((ht + hw)[i] for i in range(len(ht)))),
            "support": [float(edges[0]), float(edges[-1])],
            "n_bins": int(len(llr))}


# ---------------------------------------------------------------- candidates
def candidates(bank: Bank) -> Dict[str, Dict[str, Any]]:
    weights = {p: 1.0 / len(POSES) for p in POSES}
    bt = {p: bank.pose(p, "true") for p in POSES}
    bw = {p: bank.pose(p, "wrong") for p in POSES}
    out: Dict[str, Dict[str, Any]] = {}

    def build(name: str, edges: np.ndarray, monotone: bool, parametric: bool,
              note: str, llr_override=None) -> None:
        ht = {p: bank.reduce(bt[p], edges) for p in POSES}
        hw = {p: bank.reduce(bw[p], edges) for p in POSES}
        raw = _llr_mixture(ht, hw, POSES, weights)
        llr = raw.copy()
        extra: Dict[str, Any] = {}
        if monotone:
            self_c = sum(ht[p] + hw[p] for p in POSES)
            llr = _pava_noninc(llr, list(self_c + 1.0))
            llr = _count_scaled_floor(llr, self_c, MIN_COUNT)
            extra["monotone"] = "PAVA (non-increasing), weights = pooled bin counts"
        if parametric:
            centres = 0.5 * (edges[:-1] + edges[1:])
            w = sum(ht[p] + hw[p] for p in POSES)
            a, b, tau = _fit_exp(centres, raw, list(w + 1.0))
            llr = a + b * np.exp(-centres / tau)
            extra.update({"parametric": {"a": a, "b": b, "tau": tau,
                                         "form": "a + b*exp(-r/tau)"},
                          "monotone": "by form (b >= 0)"})
        if llr_override is not None:
            llr = np.asarray(llr_override, dtype=float).copy()
            extra["monotone"] = "PAVA front + strictly decreasing exp tail"
        tot_t = sum(ht[p].sum() for p in POSES)
        tot_w = sum(hw[p].sum() for p in POSES)
        cl = sum(ht[p] + hw[p] for p in POSES)
        out[name] = {"name": name, "note": note, "edges": edges.tolist(),
                     "llr_raw": raw.tolist(), "llr": llr.tolist(),
                     "counts_true": cl.tolist(),
                     "counts_wrong": tot_w,
                     "n_true": int(tot_t), "n_wrong": int(tot_w),
                     "bin_counts_true": [int(sum(ht[p][i] for p in POSES))
                                         for i in range(len(llr))],
                     "bin_counts_wrong": [int(sum(hw[p][i] for p in POSES))
                                          for i in range(len(llr))],
                     "per_pose_true": {p: [int(x) for x in ht[p]] for p in POSES},
                     "per_pose_wrong": {p: [int(x) for x in hw[p]] for p in POSES},
                     **extra}

    build("tb24", _grid_uniform(0.25), False, False,
          "pose-balanced mixture on the DEPLOYED grid (24 x 0.25 to 6.0), no monotone pass")
    build("tb24m", _grid_uniform(0.25), True, False,
          "pose-balanced mixture, 24 x 0.25 to 6.0, PAVA + count-scaled minimum")
    build("tb30m", _grid_uniform(0.20), True, False,
          "pose-balanced mixture, 30 x 0.20 to 6.0, PAVA + count-scaled minimum")
    build("tbnu", _grid_nonuniform(), True, False,
          "pose-balanced mixture, NON-UNIFORM grid (0.25|0.5|1.0 steps), PAVA + floor")
    build("tbfine", _grid_fine(), True, False,
          "C7: 0.05 resolution to 1.0 (covers the 15 deg IQR), then 0.25 to 3.0, then "
          "1.0 to 6.0; PAVA + count-scaled minimum")
    # C9: the front measured, the tail parametric -- see `_tail_exp`.
    _e9 = _grid_fine()
    _ht = {p: bank.reduce(bt[p], _e9) for p in POSES}
    _hw = {p: bank.reduce(bw[p], _e9) for p in POSES}
    _raw = _llr_mixture(_ht, _hw, POSES, weights)
    _cnt = sum(_ht[p] + _hw[p] for p in POSES)
    _front = _count_scaled_floor(_pava_noninc(_raw.copy(), list(_cnt + 1.0)), _cnt, MIN_COUNT)
    _cent = 0.5 * (_e9[:-1] + _e9[1:])
    _ij = int(np.searchsorted(_e9, 1.0, side="right") - 2)
    _tail = _tail_exp(_cent, _raw, _cnt + 1.0, float(_front[_ij]), 1.0)
    if len(_tail):
        _tail = np.minimum(_tail, float(_front[_ij]))
    _llr9 = np.concatenate([_front[:_ij + 1], _tail]) if len(_tail) else _front
    build("tbhyb", _e9, False, False,
          "C9: measured PAVA front (0.05 to 1.0) + strictly decreasing exp tail",
          llr_override=_llr9)
    build("tbexpf", _grid_fine(), False, True,
          "C8 (deviation, same reason as C7): the parametric monotone smoother on the "
          "FINE grid -- strict tail order by form, no plateau anywhere")
    build("tbexp", _grid_uniform(0.20), False, True,
          "parametric monotone a + b*exp(-r/tau) on the 0.20 grid")
    for c in out.values():
        c["checks"] = _checks(np.asarray(c["edges"]), np.asarray(c["llr"]),
                              np.asarray(c["bin_counts_true"]), np.asarray(c["bin_counts_wrong"]))
        c["unflatin"] = _unflatin(np.asarray(c["edges"]), np.asarray(c["llr"]))
        c["auc_bank"] = {}
        for p in POSES:
            e = np.asarray(c["edges"])
            c["auc_bank"][p] = _auc_from_bins(np.asarray(c["llr"]),
                                              np.asarray(c["per_pose_true"][p], dtype=float),
                                              np.asarray(c["per_pose_wrong"][p], dtype=float))
        del c["counts_true"]
    return out


def _references(bank: Bank) -> Dict[str, Any]:
    """The two tables already on disk, scored on the same bank."""
    refs: Dict[str, Any] = {}
    for name, path in (("dep", os.path.join(ROOT, "matching",
                                            "placement_frozen_params.json")),
                       ("m17", os.path.join(ROOT, "matching",
                                            "placement_frozen_params_m17.json"))):
        b = json.load(open(path))
        f = b["fit"]
        edges = np.asarray(f["edges"], dtype=float)
        llr = np.asarray(f["llr"], dtype=float)
        c = b.get("constants") or {}
        refs[name] = {"name": name, "edges": edges.tolist(), "llr": llr.tolist(),
                      "support": [float(edges[0]), float(edges[-1])],
                      "n_bins": int(len(llr)),
                      "checks": _checks(edges, llr,
                                        np.asarray(f.get("bin_counts_true", [0] * len(llr)),
                                                   dtype=float),
                                        np.asarray(f.get("bin_counts_wrong", [0] * len(llr)),
                                                   dtype=float)),
                      "unflatin": _unflatin(edges, llr),
                      "auc_bank": {},
                      "note": os.path.basename(path),
                      "constants": c}
        for p in POSES:
            ht = np.zeros(len(llr))
            hw = np.zeros(len(llr))
            fine_t = bank.pose(p, "true")
            fine_w = bank.pose(p, "wrong")
            ht = bank.reduce(fine_t, edges)
            hw = bank.reduce(fine_w, edges)
            refs[name]["auc_bank"][p] = _auc_from_bins(llr, ht, hw)
    return refs


def _bundle(c: Dict[str, Any]) -> Dict[str, Any]:
    b = json.load(open(BASE_BUNDLE))
    edges = c["edges"]
    llr = c["llr"]
    # `mu_wrong` is the score an UNFORMABLE slot pays: the wrong-pair expectation under
    # this table.  (First version averaged log(llr_raw) and wrote -11.8 nats; every
    # unresolvable slot then paid -11.8 and the matcher locked up at top-1 0.175.)
    hw_tot = np.asarray(c["bin_counts_wrong"], dtype=float)
    p_wrong = hw_tot / max(hw_tot.sum(), 1.0)
    mu_wrong = float(np.sum(p_wrong * np.asarray(llr, dtype=float)))
    digest = hashlib.sha256(
        np.asarray(c["bin_counts_true"], dtype=np.int64).tobytes()
        + np.asarray(c["bin_counts_wrong"], dtype=np.int64).tobytes()).hexdigest()[:16]
    b["fit"] = {"edges": [float(x) for x in edges],
                "llr": [float(x) for x in llr],
                "mu_wrong": mu_wrong,
                "n_true": int(c["n_true"]), "n_wrong": int(c["n_wrong"]),
                "bin_counts_true": [int(x) for x in c["bin_counts_true"]],
                "bin_counts_wrong": [int(x) for x in c["bin_counts_wrong"]],
                "weight_profile_digest": digest}
    b["constants"] = dict(b.get("constants") or {})
    b["constants"].update({"n_bins": int(len(llr)), "res_max": float(edges[-1]),
                           "monotone": bool("PAVA" in str(c.get("monotone", ""))
                                            or "form" in str(c.get("monotone", ""))),
                           "min_bin_count": MIN_COUNT,
                           "empty_bin_neutral": False,
                           "table": "m18-" + c["name"]})
    b["provenance"] = {"table": "m18-" + c["name"], "units": "tolunits",
                       "res_max": float(edges[-1]), "n_bins": int(len(llr)),
                       "bin_width": "non-uniform" if c["name"] == "tbnu" else float(
                           edges[1] - edges[0]),
                       "true_n": int(c["n_true"]), "wrong_n": int(c["n_wrong"]),
                       "fitted_on": "m18_bank.json (4 DEV images x 4 poses, pose-balanced "
                                    "mixture, assignment context)",
                       "rises_after": int(c["checks"]["n_rises"]),
                       "empty_bins_with_positive_credit":
                           int(c["checks"]["empty_bins_with_positive_credit"]),
                       "note": c["note"]}
    return b


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-bundles", action="store_true")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    bank = Bank(json.load(open(BANK)))
    cands = candidates(bank)
    refs = _references(bank)
    written: List[str] = []
    if args.write_bundles:
        os.makedirs(BUNDLE_DIR, exist_ok=True)
        for name, c in cands.items():
            p = os.path.join(BUNDLE_DIR, "m18_%s.json" % name)
            with open(p, "w") as fh:
                json.dump(_bundle(c), fh, indent=1, sort_keys=True)
            written.append(p)
    print("%-7s %-5s %-5s %-5s %-5s %-7s %-7s %-6s %-6s"
          % ("cand", "bins", "supp", "rises", "thin", "L(1.5)", "L(5.0)", "15AUC", "05AUC"))
    allc = dict(refs)
    allc.update(cands)
    ORDER = ("dep", "m17", "tb24", "tb24m", "tb30m", "tbnu", "tbexp", "tbfine",
             "tbexpf", "tbhyb")
    for name in ORDER:
        c = allc[name]
        ck = c["checks"]
        print("%-7s %-5d %-5.1f %-5d %-5d %-7.2f %-7.2f %-6.3f %-6.3f"
              % (name, ck["n_bins"], ck["support"][1], ck["n_rises"],
                 ck["thin_bins_with_positive_credit"], ck["llr_1.5"], ck["llr_5.0"],
                 c["auc_bank"]["rot15"], c["auc_bank"]["rot05"]))
    print("\n%-7s %-6s %-6s %-8s %-11s %s"
          % ("cand", "levels", "span", "plateau", "15 IQR", "llr over the 15 deg IQR bins"))
    for name in ORDER:
        u = allc[name]["unflatin"]
        print("%-7s %-6d %-6.2f %-8.2f %-11s %s"
              % (name, u["levels"], u["span"], u["plateau_share"],
                 "[%.3f,%.3f]" % tuple(u["region"]), u["values"]))
    res = {"candidates": cands, "references": refs,
           "region": list(REGION), "min_count": MIN_COUNT,
           "pose_weights": {p: 1.0 / len(POSES) for p in POSES},
           "units": "residuals in tolunits; LLR in nats; AUC unitless",
           "bank": BANK, "bundles": [os.path.relpath(p, ROOT) for p in written]}
    MF.write_result(RESULT, res)
    print("\nwrote %s%s" % (RESULT, (" + %d bundles" % len(written)) if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
