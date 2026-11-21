"""Task 4 PART 1 — DIAGNOSE (direct probes).

1a: Per token kind AND per partition (content vs sheet-derived), one-to-one
    counterpart rate at various shifts and radii, with offset histogram.
1b: For non-equivariant kinds, probe candidate mechanisms.
1c: Sub-pixel diagnostics.
1d: Position-wiring fix across ALL dev graphs.

Uses the repaired shift instrument (t4_shift.py).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import (roll_image, fourier_shift, expected_position,
                      counterpart_map, primary_radius)  # noqa: E402

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}

# Sheet-derived lens types (static by design)
SHEET_LENSES = {"Orientation", "RegionShape"}
# Content lens types (expected to be equivariant)
CONTENT_LENSES = {"CONTOUR_PEAK", "Contour", "ContourFragment", "Corner",
                  "CurveComplex", "EndStopped", "IllusoryContour",
                  "Junction", "PEAK_CHAIN"}


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _partition(lens):
    if lens in SHEET_LENSES:
        return "sheet"
    return "content"


def _build_shifted(entry, arm, dx, dy, use_fourier=False):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    if dx == 0 and dy == 0:
        return build(entry)
    tmp = Path(tempfile.gettempdir()) / "t4_part1" / (
        "%s%d%s_%d_%d_%s.png" % (entry["class"], int(entry["object"]),
                                 entry["view"], dx, dy,
                                 "f" if use_fourier else "r"))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if use_fourier:
            shifted = fourier_shift(arr, dx, dy)
        else:
            shifted = roll_image(arr, dx, dy)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["view"] == "-090-000"]
    by_class = {}
    for e in dev:
        c = e["class"]
        if c not in by_class or e["object"] < by_class[c]["object"]:
            by_class[c] = e
    return [by_class[c] for c in corpus.CLASSES]


def _counterpart_rate(base_pos, shift_pos, dx, dy, radius):
    """One-to-one counterpart rate at a given radius.

    Returns: rate, offsets (list of (dx_err, dy_err) for matched pairs),
    ambiguous_count, total_content, total_sheet.
    """
    mapping, info, _cp_stats = counterpart_map(base_pos, shift_pos, dx, dy, radius)
    offsets = []
    ambiguous = 0
    for n, (x0, y0, l) in base_pos.items():
        if n in mapping:
            bn = mapping[n]
            bx, by = shift_pos[bn][0], shift_pos[bn][1]
            ex, ey = expected_position(x0, y0, dx, dy)
            offsets.append((bx - ex, by - ey, l))
            if info.get(n, {}).get("ambiguous", False):
                ambiguous += 1
    total = len(base_pos)
    rate = len(mapping) / total if total else 0.0
    return rate, offsets, ambiguous, total


def _per_kind_stats(base_pos, shift_pos, dx, dy, radius):
    """Per-kind, per-partition one-to-one counterpart rate."""
    mapping, info, _cp_stats = counterpart_map(base_pos, shift_pos, dx, dy, radius)
    by_kind = defaultdict(lambda: {"total": 0, "matched": 0, "offsets": [],
                                   "ambiguous": 0})
    for n, (x0, y0, l) in base_pos.items():
        part = _partition(l)
        key = (l, part)
        by_kind[key]["total"] += 1
        if n in mapping:
            bn = mapping[n]
            bx, by = shift_pos[bn][0], shift_pos[bn][1]
            ex, ey = expected_position(x0, y0, dx, dy)
            by_kind[key]["matched"] += 1
            by_kind[key]["offsets"].append((bx - ex, by - ey))
            if info.get(n, {}).get("ambiguous", False):
                by_kind[key]["ambiguous"] += 1
    result = {}
    for (l, part), d in sorted(by_kind.items()):
        result["%s|%s" % (l, part)] = {
            "total": d["total"], "matched": d["matched"],
            "rate": d["matched"] / d["total"] if d["total"] else 0.0,
            "ambiguous": d["ambiguous"],
            "offset_mean": [float(np.mean([o[0] for o in d["offsets"]]))
                            if d["offsets"] else 0.0,
                            float(np.mean([o[1] for o in d["offsets"]]))
                            if d["offsets"] else 0.0],
            "offset_rms": float(np.sqrt(np.mean([o[0]**2 + o[1]**2
                                                 for o in d["offsets"]])))
            if d["offsets"] else 0.0,
        }
    return result


def main() -> int:
    print("=== PART 1: DIAGNOSE ===", flush=True)
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # 0c radius table (computed on B0 dev, all 8 images)
    print("\n--- Computing primary radius (B0, all dev) ---", flush=True)
    all_positions = defaultdict(list)
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS["B0"])
    from frontend import build
    for e in entries:
        g = build(e)
        for n, (x, y, l) in _pos_lens(g).items():
            all_positions[l].append((x, y))
    radius_table = primary_radius(all_positions)
    for l, r in sorted(radius_table.items()):
        n = len(all_positions.get(l, []))
        print("  %-20s n=%d radius=%s" % (l, n,
              "%.3f" % r if r else "ABSENT"), flush=True)

    # 1a: Per-kind equivariance at various shifts
    # Shifts: vertical +2, +3; horizontal +2; Fourier +0.5, +1.5
    shifts_to_test = [
        ("vert+2", 0, 2, False),
        ("vert+3", 0, 3, False),
        ("horiz+2", 2, 0, False),
        ("fourier+0.5", 0, 0.5, True),
        ("fourier+1.5", 0, 1.5, True),
    ]

    # Use B1 as the primary diagnostic arm (L3/L4 ON)
    arm = "B1"
    out = {"arm": arm, "radius_table": radius_table, "shifts": {}}

    for shift_name, dx, dy, use_fourier in shifts_to_test:
        print("\n--- 1a: %s (dx=%s, dy=%s) ---" % (shift_name, dx, dy),
              flush=True)
        shift_results = {}
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base_g = _build_shifted(e, arm, 0, 0)
            sh_g = _build_shifted(e, arm, dx, dy, use_fourier)
            base_pos = _pos_lens(base_g)
            sh_pos = _pos_lens(sh_g)
            # Use a generous radius for the rate (8px) to see the raw rate
            rate, offsets, amb, total = _counterpart_rate(
                base_pos, sh_pos, dx, dy, 8.0)
            per_kind = _per_kind_stats(base_pos, sh_pos, dx, dy, 8.0)
            shift_results[name] = {
                "rate_8px": rate, "total": total, "ambiguous": amb,
                "per_kind": per_kind,
            }
            print("  %s: rate@8px=%.4f (%d/%d) amb=%d"
                  % (name, rate, len(offsets), total, amb), flush=True)

        # Aggregate per kind across all scenes
        agg_kind = defaultdict(lambda: {"total": 0, "matched": 0,
                                        "ambiguous": 0})
        for name, r in shift_results.items():
            for k, d in r["per_kind"].items():
                agg_kind[k]["total"] += d["total"]
                agg_kind[k]["matched"] += d["matched"]
                agg_kind[k]["ambiguous"] += d["ambiguous"]
        agg_summary = {}
        for k, d in sorted(agg_kind.items()):
            agg_summary[k] = {
                "total": d["total"], "matched": d["matched"],
                "rate": d["matched"] / d["total"] if d["total"] else 0.0,
                "ambiguous": d["ambiguous"],
            }
            print("  %-30s rate=%.4f (%d/%d) amb=%d"
                  % (k, agg_summary[k]["rate"], d["matched"], d["total"],
                     d["ambiguous"]), flush=True)
        out["shifts"][shift_name] = {"per_scene": shift_results,
                                     "aggregate": agg_summary}

    write_json(RESULTS_DIR / "t4" / "part1_diagnose.json", out)
    print("\n=== PART 1 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
