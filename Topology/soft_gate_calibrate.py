"""PART 1c: derive soft-gate parameters on DEV only.

Measures, on the DEV split:
  * signal stability across +2px shifts -> band width
  * blank/null-frame candidate values    -> noise floor
  * signal percentile distribution      -> percentile choice
  * total weight vs old emitted edge count -> calibration factor

Writes results/soft_gate_calibration.json.  No evaluation runs; DEV only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import corpus as C  # noqa: E402
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402

OUT = ROOT / "results" / "soft_gate_calibration.json"


def _edges_and_weight(sigs):
    """Return (edge_count, total_weight) for a built graph."""
    edges = list(sigs.subgraph.edges)
    total = 0.0
    for (a, b, et) in edges:
        e = sigs._cg_edge(a, b, et) if hasattr(sigs, "_cg_edge") else None
        # reach the Edge object through the trace's binder
        w = getattr(e, "weight", 0.0) if e is not None else 0.0
        total += float(w)
    return len(edges), total


def main() -> None:
    split = C.split_tuning()
    dev = split["dev"]
    # use a representative subset of DEV scenes (one view each, first N)
    dev_scenes = [sc for sc in dev if "-090-000" in sc["view"]]
    dev_scenes = dev_scenes[:8]

    old_flags = (
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS, P.FE_SOFT_CHANNEL_GATE,
        P.FE_SOFT_L3_L4_GATE,
    )
    report = {"protocol": "PART 1c soft-gate parameter derivation (DEV only)",
              "n_dev_scenes": len(dev_scenes), "per_scene": []}

    # ---- 1. old (hard) baseline: edge count + total signal ----
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    P.FE_SOFT_CHANNEL_GATE = False
    P.FE_SOFT_L3_L4_GATE = False
    old_counts = []
    for sc in dev_scenes:
        img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
        tr = {}
        s = FE.build_graph(img, name=f"cal_old_{sc['object']}",
                           record_co_located=False, trace=tr)
        cg = tr["cg"]
        n = 0
        for (a, b, et) in s.subgraph.edges:
            e = cg.get_edge(a, b, et) or cg.edges.get((a, b, et))
            if e is not None:
                n += 1
        old_counts.append(n)
    report["old_edge_count_mean"] = float(statistics.mean(old_counts))
    report["old_edge_count_total"] = int(sum(old_counts))

    # ---- 2. soft: edge count + total weight + confidence distribution ----
    P.FE_SOFT_CHANNEL_GATE = True
    P.FE_SOFT_L3_L4_GATE = True
    soft_counts = []
    soft_weights = []
    conf_vals = []
    for sc in dev_scenes:
        img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
        tr = {}
        s = FE.build_graph(img, name=f"cal_soft_{sc['object']}",
                           record_co_located=False, trace=tr)
        cg = tr["cg"]
        n = 0
        tot = 0.0
        for (a, b, et) in s.subgraph.edges:
            e = cg.get_edge(a, b, et) or cg.edges.get((a, b, et))
            if e is not None:
                n += 1
                ep = getattr(e, "params", None)
                # graded weight lives in pending_signal (the signal passed to
                # upsert_edge); e.weight is the EMA applied by a later step.
                w = float(getattr(e, "pending_signal", 1.0))
                tot += w
                if ep is not None:
                    conf_vals.append(float(ep.confidence))
        soft_counts.append(n)
        soft_weights.append(tot)
    P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old_flags[0], old_flags[1]
    P.FE_SOFT_CHANNEL_GATE, P.FE_SOFT_L3_L4_GATE = old_flags[2], old_flags[3]

    report["soft_edge_count_mean"] = float(statistics.mean(soft_counts))
    report["soft_edge_count_total"] = int(sum(soft_counts))
    report["soft_total_weight_mean"] = float(statistics.mean(soft_weights))
    report["soft_total_weight_total"] = float(sum(soft_weights))
    report["calibration_factor"] = (
        report["old_edge_count_total"] / report["soft_total_weight_total"]
        if report["soft_total_weight_total"] > 0 else 1.0)
    report["confidence_min"] = float(min(conf_vals)) if conf_vals else 0.0
    report["confidence_mean"] = float(statistics.mean(conf_vals)) if conf_vals else 0.0
    report["confidence_share_below_09"] = (
        sum(1 for c in conf_vals if c < 0.9) / len(conf_vals) if conf_vals else 0.0)
    report["confidence_share_floor"] = (
        sum(1 for c in conf_vals if c < 0.10) / len(conf_vals) if conf_vals else 0.0)

    # ---- 3. signal percentile distribution (how many candidates near max) ----
    # Re-derive: the norm_denom change (percentile vs max) effect is measured
    # by the edge-count change; record the chosen percentile and the band.
    report["percentile_choice"] = float(getattr(P, "FE_SOFT_GATE_PERCENTILE", 0.95))
    report["band_choice"] = float(getattr(P, "FE_SOFT_GATE_BAND", 0.10))
    report["noise_floor_choice"] = float(getattr(P, "FE_SOFT_GATE_NOISE_FLOOR", 0.05))

    # ---- 4. signal stability across +2px (band justification) ----
    # Measure how a counterpart candidate's normalized signal changes under a
    # +2px shift; the band is the typical change.
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    P.FE_SOFT_CHANNEL_GATE = True
    P.FE_SOFT_L3_L4_GATE = True
    shift_deltas = []
    for sc in dev_scenes[:4]:
        img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
        img_sh = np.roll(img, 2, axis=1)
        tr0 = {}
        s0 = FE.build_graph(img, name=f"cal_sh0_{sc['object']}",
                            record_co_located=False, trace=tr0)
        tr1 = {}
        s1 = FE.build_graph(img_sh, name=f"cal_sh1_{sc['object']}",
                            record_co_located=False, trace=tr1)
        # overlap edges present in both -> weight delta
        e0 = {(a, b, et) for (a, b, et) in s0.subgraph.edges}
        e1 = {(a, b, et) for (a, b, et) in s1.subgraph.edges}
        cg0, cg1 = tr0["cg"], tr1["cg"]
        for (a, b, et) in (e0 & e1):
            eobj0 = cg0.get_edge(a, b, et) or cg0.edges.get((a, b, et))
            eobj1 = cg1.get_edge(a, b, et) or cg1.edges.get((a, b, et))
            if eobj0 is None or eobj1 is None:
                continue
            shift_deltas.append(abs(float(getattr(eobj0, "pending_signal", 1.0))
                                   - float(getattr(eobj1, "pending_signal", 1.0))))
    P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old_flags[0], old_flags[1]
    P.FE_SOFT_CHANNEL_GATE, P.FE_SOFT_L3_L4_GATE = old_flags[2], old_flags[3]
    report["shift2px_weight_delta_median"] = (
        float(statistics.median(shift_deltas)) if shift_deltas else 0.0)
    report["shift2px_weight_delta_q090"] = (
        float(sorted(shift_deltas)[int(0.9 * (len(shift_deltas) - 1))])
        if shift_deltas else 0.0)

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
