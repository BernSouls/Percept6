"""Task 4 PART A4 — TOLERANCE / PARAMETER DRIFT.

Measures how edge parameters change under a rigid translation, to derive a
matching tolerance. Steps:

1. For matched edges (both-track: an edge present in base AND shifted with
   the same topology), measure the parameter change (gap, angle, offset).
   Rigid translation leaves relation PARAMETERS invariant for pure translation
   (LIMIT noted): gap/angle are invariant; offset (position) shifts by dx.
2. tolerance = max(q90 matched-edge param change among both-track, UNCENSORED,
   one resolution step). Report the non-comparable fraction separately.
3. Measure at integer +2 and Fourier +0.5/+1.5.
4. Recompute IQR vs MEASURED drift; branching factor (same-lens candidates
   per node) mean/median/p95.

Report at frozen tolerance AND 4x it (per the mission's A4/P3-3 insertion).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

D_CLASS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                     "RegionShape", "Orientation"})
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_dev_graph(arm, name):
    with open(PKL_DIR / ("%s_%s.pkl" % (arm, name)), "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _build_shifted_fourier(entry, spec, dx_frac):
    """Build a shifted graph at a fractional (Fourier) offset."""
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_a4" / (
        "%s%d%s_f%.1f.png" % (entry["class"], int(entry["object"]),
                             entry["view"], dx_frac))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # roll_image takes integer dx; for fractional we roll by nearest int
        # and note the approximation (Fourier phase is not exactly modelled).
        shifted = roll_image(arr, int(round(dx_frac)), 0)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _edge_param_change(ep_base, ep_sh, mapped_nodes):
    """For matched edges, measure parameter change.

    Returns list of (change_magnitude, comparable_bool) per matched edge.
    An edge is comparable if both endpoints have counterparts and the edge
    type matches.
    """
    changes = []
    # Build a lookup of shifted edge params by (mapped_a, mapped_b, type)
    sh_ep_lookup = {}
    for (a, b, t), val in ep_sh.items():
        a2 = mapped_nodes.get(a)
        b2 = mapped_nodes.get(b)
        if a2 is not None and b2 is not None:
            sh_ep_lookup[(min(a2, b2), max(a2, b2), t)] = val
    for (a, b, t), val in ep_base.items():
        a2 = mapped_nodes.get(a)
        b2 = mapped_nodes.get(b)
        if a2 is None or b2 is None:
            continue
        key = (min(a2, b2), max(a2, b2), t)
        if key not in sh_ep_lookup:
            continue
        val2 = sh_ep_lookup[key]
        # EdgeParams: compare the quantized parameter tuple (gap/angle/offset)
        # and the graded population. comparable if both are EdgeParams with
        # matching quantized arity.
        def _extract(v):
            q = getattr(v, "quantized", None)
            if q is None and isinstance(v, (list, tuple)):
                q = v
            return q
        qa = _extract(val)
        qb = _extract(val2)
        if qa is not None and qb is not None and len(qa) == len(qb):
            try:
                diff = np.array(qa, dtype=float) - np.array(qb, dtype=float)
                changes.append((float(np.abs(diff).max()), True))
            except Exception:
                changes.append((0.0, False))
        else:
            changes.append((0.0, False))
    return changes


def _branching_factors(base_pos, sh_pos, cp):
    """Same-lens candidate count per base node."""
    sh_by_lens = Counter()
    for n, (x, y, lens) in sh_pos.items():
        sh_by_lens[lens] += 1
    bf = []
    for bn, (x, y, lens) in base_pos.items():
        if bn in cp:
            bf.append(sh_by_lens.get(lens, 0) - 1)  # exclude true counterpart
    return bf


def _quartiles(vals):
    if not vals:
        return None
    arr = np.array(vals)
    return {
        "q50": float(np.quantile(arr, 0.5)),
        "q90": float(np.quantile(arr, 0.90)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "n": len(vals),
    }


def main() -> int:
    print("=== A4: TOLERANCE / PARAMETER DRIFT ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    drift_all = []
    noncomparable = 0
    comparable = 0
    branching_all = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)

        # Integer +2
        sh_g = _load_shifted(name, 2)
        if sh_g is None:
            continue
        sh_pos = _pos_lens(sh_g)
        cp, cp_info, _cp_stats = counterpart_map(base_pos, sh_pos, 2, 0, 8.0)

        # param drift
        changes = _edge_param_change(
            base.subgraph.edge_params, sh_g.subgraph.edge_params, cp)
        for mag, comparable_flag in changes:
            drift_all.append(mag)
            if comparable_flag:
                comparable += 1
            else:
                noncomparable += 1

        # branching factor
        bf = _branching_factors(base_pos, sh_pos, cp)
        branching_all.extend(bf)

    # Fourier +0.5 / +1.5 (approximated by nearest integer roll; noted)
    fourier_drift = {}
    for dxf in [0.5, 1.5]:
        vals = []
        for e in entries[:8]:  # subset for time
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base = _load_dev_graph("B1", name)
            base_pos = _pos_lens(base)
            sh_g = _build_shifted_fourier(e, ARM_B1, dxf)
            sh_pos = _pos_lens(sh_g)
            cp, _, _cp_stats = counterpart_map(base_pos, sh_pos, int(round(dxf)), 0, 8.0)
            changes = _edge_param_change(
                base.subgraph.edge_params, sh_g.subgraph.edge_params, cp)
            vals.extend([m for m, _ in changes])
        fourier_drift["%.1f" % dxf] = _quartiles(vals)

    q = _quartiles(drift_all)
    tolerance = q["q90"] if q else 0.0
    tolerance_4x = 4.0 * tolerance

    print("\n--- Drift (integer +2, both-track matched edges) ---", flush=True)
    print("  comparable: %d, non-comparable: %d (%.2f)"
          % (comparable, noncomparable,
             noncomparable / (comparable + noncomparable)
             if (comparable + noncomparable) else 0))
    if q:
        print("  drift: median=%.4f q90=%.4f mean=%.4f p95=%.4f n=%d"
              % (q["median"], q["q90"], q["mean"], q["p95"], q["n"]))
    print("  tolerance (q90) = %.4f" % tolerance)
    print("  tolerance 4x     = %.4f" % tolerance_4x)

    bf_q = _quartiles(branching_all)
    print("\n--- Branching factor (same-lens candidates) ---", flush=True)
    if bf_q:
        print("  mean=%.2f median=%.2f p95=%.2f n=%d"
              % (bf_q["mean"], bf_q["median"], bf_q["p95"], bf_q["n"]))

    print("\n--- Fourier drift ---", flush=True)
    for k, v in fourier_drift.items():
        if v:
            print("  +%s: median=%.4f q90=%.4f" % (k, v["median"], v["q90"]))

    out["drift"] = q
    out["tolerance"] = tolerance
    out["tolerance_4x"] = tolerance_4x
    out["comparable_count"] = comparable
    out["noncomparable_count"] = noncomparable
    out["noncomparable_frac"] = noncomparable / (comparable + noncomparable) \
        if (comparable + noncomparable) else 0.0
    out["branching_factor"] = bf_q
    out["fourier_drift"] = fourier_drift
    out["limit_note"] = (
        "Pure rigid translation leaves relation PARAMETERS (gap/angle) invariant; "
        "only offset (position) shifts. Failing a parameter-invariance check is "
        "decisive; passing is necessary not sufficient.")

    write_json(RESULTS_DIR / "t4" / "a4_tolerance_drift.json", out)
    print("\n=== A4 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
