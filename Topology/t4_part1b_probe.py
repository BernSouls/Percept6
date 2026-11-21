"""Task 4 PART 1b/1c/1d — Probes for non-equivariant kinds, sub-pixel,
position-wiring.

1b: Probe candidate mechanisms for non-equivariant kinds (Contour, CurveComplex):
    anchor_allocator, greedy NMS, border discard/roll wrap, bin flips.
1c: Sub-pixel diagnostics: raw parabolic offsets, rejection fractions,
    ridge vs end-stop/curvature, quantization floor.
1d: Position-wiring fix across ALL dev graphs: per-lens anchor fraction
    before/after, graph-hash change.
"""
from __future__ import annotations

import hashlib
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

from t4_shift import roll_image, fourier_shift, expected_position  # noqa: E402

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


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


def _build(entry, arm, dx=0, dy=0, use_fourier=False):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    if dx == 0 and dy == 0:
        return build(entry)
    tmp = Path(tempfile.gettempdir()) / "t4_1b" / (
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


def probe_1b(arm="B1"):
    """1b: Probe mechanisms for non-equivariant kinds."""
    print("=== 1b: Probing non-equivariant kinds (arm=%s) ===" % arm,
          flush=True)
    entries = _dev_entries()
    out = {}

    # Check: for each base node of Contour/CurveComplex, is there a
    # same-kind node in the shifted graph at the expected position (loose tol)?
    # If not, check if it was dropped (no node at any nearby position)
    for shift_name, dx, dy, uf in [
            ("vert+2", 0, 2, False), ("horiz+2", 2, 0, False)]:
        print("\n--- %s ---" % shift_name, flush=True)
        kind_stats = defaultdict(lambda: {"base": 0, "found_loose": 0,
                                          "found_strict": 0, "no_nearby": 0})
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base_g = _build(e, arm)
            sh_g = _build(e, arm, dx, dy, uf)
            base_pos = _pos_lens(base_g)
            sh_pos = _pos_lens(sh_g)
            # Build loose and strict counterpart maps
            by_lens_sh = defaultdict(list)
            for n, (x, y, l) in sh_pos.items():
                by_lens_sh[l].append((n, x, y))
            for n, (x0, y0, l) in base_pos.items():
                if l not in ("Contour", "CurveComplex"):
                    continue
                key = "%s|%s" % (l, name)
                kind_stats[key]["base"] += 1
                ex, ey = expected_position(x0, y0, dx, dy)
                # loose: within 8px
                found_loose = False
                best_d = 8.0
                for nsh, xsh, ysh in by_lens_sh.get(l, ()):
                    d = math.hypot(xsh - ex, ysh - ey)
                    if d < best_d:
                        best_d = d
                        found_loose = True
                if found_loose:
                    kind_stats[key]["found_loose"] += 1
                    if best_d < 1.0:
                        kind_stats[key]["found_strict"] += 1
                else:
                    kind_stats[key]["no_nearby"] += 1
        # Aggregate
        agg = defaultdict(lambda: {"base": 0, "found_loose": 0,
                                   "found_strict": 0, "no_nearby": 0})
        for k, d in kind_stats.items():
            kind = k.split("|")[0]
            agg[kind]["base"] += d["base"]
            agg[kind]["found_loose"] += d["found_loose"]
            agg[kind]["found_strict"] += d["found_strict"]
            agg[kind]["no_nearby"] += d["no_nearby"]
        for kind, d in sorted(agg.items()):
            if d["base"] == 0:
                continue
            print("  %-15s base=%d loose=%d (%.1f%%) strict=%d no_nearby=%d (%.1f%%)"
                  % (kind, d["base"], d["found_loose"],
                     100 * d["found_loose"] / d["base"],
                     d["found_strict"], d["no_nearby"],
                     100 * d["no_nearby"] / d["base"]), flush=True)
        out[shift_name] = {k: dict(v) for k, v in agg.items()}

    return out


def probe_1c(arm="F"):
    """1c: Sub-pixel diagnostics."""
    print("\n=== 1c: Sub-pixel diagnostics (arm=%s) ===" % arm, flush=True)
    entries = _dev_entries()
    out = {}

    # For arm F, check CONTOUR_PEAK positions: integer vs sub-pixel
    # and the quantization floor under Fourier shifts
    for shift_name, dx, dy in [("fourier+0.5", 0, 0.5),
                               ("fourier+1.5", 0, 1.5)]:
        print("\n--- %s ---" % shift_name, flush=True)
        int_errs = []  # RMS error for integer-snapped tokens
        sub_errs = []  # RMS error for sub-pixel tokens
        int_count = sub_count = 0
        for e in entries:
            base_g = _build(e, arm)
            sh_g = _build(e, arm, dx, dy, use_fourier=True)
            base_pos = _pos_lens(base_g)
            sh_pos = _pos_lens(sh_g)
            # Match CONTOUR_PEAK tokens
            by_lens = defaultdict(list)
            for n, (x, y, l) in sh_pos.items():
                by_lens[l].append((n, x, y))
            for n, (x0, y0, l) in base_pos.items():
                if l != "CONTOUR_PEAK":
                    continue
                ex, ey = expected_position(x0, y0, dx, dy)
                best_d, best_n = 8.0, None
                for nsh, xsh, ysh in by_lens.get(l, ()):
                    d = math.hypot(xsh - ex, ysh - ey)
                    if d < best_d:
                        best_d = d
                        best_n = nsh
                if best_n is None:
                    continue
                bx, by = sh_pos[best_n][0], sh_pos[best_n][1]
                err_x = bx - ex
                err_y = by - ey
                # Determine if base token is integer-snapped or sub-pixel
                is_integer = (abs(x0 - round(x0)) < 1e-9 and
                              abs(y0 - round(y0)) < 1e-9)
                if is_integer:
                    int_errs.append(err_x**2 + err_y**2)
                    int_count += 1
                else:
                    sub_errs.append(err_x**2 + err_y**2)
                    sub_count += 1
        int_rms = math.sqrt(sum(int_errs) / len(int_errs)) if int_errs else 0
        sub_rms = math.sqrt(sum(sub_errs) / len(sub_errs)) if sub_errs else 0
        # Quantization floor for f=0.5: sqrt(0.5*0.5) = 0.5 on shifted axis
        floor = 0.5
        print("  integer-snapped: n=%d RMS=%.4f (floor=%.2f)"
              % (int_count, int_rms, floor), flush=True)
        print("  sub-pixel:       n=%d RMS=%.4f (floor=%.2f)"
              % (sub_count, sub_rms, floor), flush=True)
        out[shift_name] = {
            "integer_count": int_count, "integer_rms": int_rms,
            "subpixel_count": sub_count, "subpixel_rms": sub_rms,
            "quantization_floor": floor,
            "subpixel_beats_floor": sub_rms < floor if sub_count else None,
        }
    return out


def probe_1d():
    """1d: Position-wiring fix across ALL dev graphs."""
    print("\n=== 1d: Position-wiring (all dev graphs) ===", flush=True)
    entries = _dev_entries()
    out = {"arms": {}}
    for arm in ["B0", "B1", "F"]:
        import _bootstrap
        _bootstrap.apply_switch_overrides(ARMS[arm])
        from frontend import build
        per_lens = defaultdict(lambda: {"total": 0, "with_anchor": 0})
        graph_hashes = []
        for e in entries:
            g = build(e)
            # Graph hash
            h = hashlib.sha256()
            nodes = sorted(g.subgraph.nodes, key=str)
            h.update(json.dumps([str(n) for n in nodes]).encode())
            h.update(json.dumps(sorted(
                [str(x) for x in g.subgraph.edges],
                key=lambda x: (x[0], x[1]))).encode())
            graph_hashes.append(h.hexdigest()[:16])
            for n in g.subgraph.nodes:
                lens = str(g.sigs.get(n, ("?",))[0])
                per_lens[lens]["total"] += 1
                if n in g.subgraph.anchor_of_node:
                    per_lens[lens]["with_anchor"] += 1
        out["arms"][arm] = {
            "per_lens": {k: {"total": d["total"],
                             "with_anchor": d["with_anchor"],
                             "frac": d["with_anchor"] / d["total"]
                             if d["total"] else 0}
                         for k, d in sorted(per_lens.items())},
            "graph_hashes": graph_hashes,
        }
        total = sum(d["total"] for d in per_lens.values())
        with_a = sum(d["with_anchor"] for d in per_lens.values())
        print("  %s: %d/%d nodes have anchors (%.2f%%)"
              % (arm, with_a, total, 100 * with_a / total if total else 0),
              flush=True)
        for lens, d in sorted(per_lens.items()):
            if d["total"] > 0:
                print("    %-20s %d/%d (%.1f%%)"
                      % (lens, d["with_anchor"], d["total"],
                         100 * d["with_anchor"] / d["total"]), flush=True)
    return out


def main() -> int:
    out = {}
    out["1b"] = probe_1b("B1")
    out["1c"] = probe_1c("F")
    out["1d"] = probe_1d()
    write_json(RESULTS_DIR / "t4" / "part1b_probe.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
