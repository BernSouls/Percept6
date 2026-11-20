"""Task 4 PART 2d — Arm selection: F vs B1 at Fourier shifts.

F beats B1 only if the paired interval is above zero.
Ties go to the simpler arm (B1).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import fourier_shift, expected_position, counterpart_map  # noqa: E402

ARMS = {
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}
WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
    "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
    "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
}


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _build(entry, arm, dx, dy):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    if dx == 0 and dy == 0:
        return build(entry)
    tmp = Path(tempfile.gettempdir()) / "t4_2d" / (
        "%s%d%s_%s_%d_%d.png" % (entry["class"], int(entry["object"]),
                                 entry["view"], arm, dx, dy))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        shifted = fourier_shift(arr, dx, dy)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _paired_interval(a_vals, b_vals):
    n = len(a_vals)
    if n == 0:
        return None
    diffs = np.array(a_vals) - np.array(b_vals)
    mean = float(diffs.mean())
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    return {"mean": mean,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}


def main() -> int:
    print("=== 2d: Arm selection (F vs B1 at Fourier shifts) ===", flush=True)
    import corpus
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["view"] == "-090-000"]
    by_class = {}
    for e in dev:
        c = e["class"]
        if c not in by_class or e["object"] < by_class[c]["object"]:
            by_class[c] = e
    entries = [by_class[c] for c in corpus.CLASSES]

    SHIFTS = [(0, 0.5), (0, 1.5)]
    out = {"shifts": []}

    for dx, dy in SHIFTS:
        print("\n--- shift (dx=%s, dy=%s) ---" % (dx, dy), flush=True)
        b1_vals = []
        f_vals = []
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            # B1
            base_b1 = _build(e, "B1", 0, 0)
            sh_b1 = _build(e, "B1", dx, dy)
            bp_b1 = _pos_lens(base_b1)
            sp_b1 = _pos_lens(sh_b1)
            cp_b1, _ = counterpart_map(bp_b1, sp_b1, dx, dy, 8.0)
            map_b1 = _run_matcher(base_b1, sh_b1)
            d1_b1 = sum(1 for n in bp_b1 if n in cp_b1 and
                        map_b1.get(n) == cp_b1.get(n)) / len(bp_b1) if bp_b1 else 0
            b1_vals.append(d1_b1)

            # F
            base_f = _build(e, "F", 0, 0)
            sh_f = _build(e, "F", dx, dy)
            bp_f = _pos_lens(base_f)
            sp_f = _pos_lens(sh_f)
            cp_f, _ = counterpart_map(bp_f, sp_f, dx, dy, 8.0)
            map_f = _run_matcher(base_f, sh_f)
            d1_f = sum(1 for n in bp_f if n in cp_f and
                       map_f.get(n) == cp_f.get(n)) / len(bp_f) if bp_f else 0
            f_vals.append(d1_f)
            print("  %s: B1=%.4f F=%.4f" % (name, d1_b1, d1_f), flush=True)

        interval = _paired_interval(f_vals, b1_vals)
        out["shifts"].append({
            "shift": [dx, dy], "b1_mean": float(np.mean(b1_vals)),
            "f_mean": float(np.mean(f_vals)),
            "f_minus_b1": interval,
        })
        if interval:
            print("  F-B1: %.4f [%.4f, %.4f] hw=%.4f"
                  % (interval["mean"], interval["lo"], interval["hi"],
                     (interval["hi"] - interval["lo"]) / 2), flush=True)

    # Decision
    all_above_zero = all(
        s["f_minus_b1"] is not None and s["f_minus_b1"]["lo"] > 0
        for s in out["shifts"])
    out["decision"] = "F" if all_above_zero else "B1 (simpler; F does not beat B1)"
    print("\nDecision: %s" % out["decision"], flush=True)
    write_json(RESULTS_DIR / "t4" / "part2d_armselect.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
