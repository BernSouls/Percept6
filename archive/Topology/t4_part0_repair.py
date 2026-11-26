"""Task 4 PART 0 — REPAIR THE INSTRUMENT (0a-0e).

0a: ONE shift helper (dx, dy explicit) in t4_shift.py — strict < at radius.
0b: Tests with numbers.
0c: One-to-one counterparts, primary position radius, parameter tolerance.
0d: Mark previous artifacts INVALID.
0e: Provenance.

This script runs 0b(i) synthetic, 0b(iii) Fourier, 0b(iv) provenance.
0b(ii) cross-check with known_transform.py is reported separately.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import (roll_image, fourier_shift, expected_position,
                      counterpart_map, primary_radius,
                      matcher_accuracy)  # noqa: E402

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}


def _pos_lens(g):
    """node -> (x, y, lens) from anchor_of_node + sigs."""
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _position_stores(g):
    """List every post-build position store on the graph."""
    stores = {}
    sg = g.subgraph
    if hasattr(sg, "anchor_of_node"):
        stores["subgraph.anchor_of_node"] = dict(sg.anchor_of_node)
    if getattr(g, "addrs", None):
        stores["graph.addrs"] = g.addrs
    return stores


def _shift_store(store_val, dx, dy):
    """Shift a position store by (dx, dy)."""
    if store_val is None:
        return None
    # anchor_of_node: {node: (x, y)}
    if isinstance(store_val, dict) and store_val and \
        isinstance(next(iter(store_val.values())), (tuple, list)):
        return {k: (float(v[0]) + dx, float(v[1]) + dy)
                for k, v in store_val.items()}
    return store_val


def _shift_graph_positions(g, dx, dy, only_anchor=False):
    """Return a copy of g with all position stores shifted by (dx, dy)."""
    sg = g.subgraph
    new_anchor = {n: (float(a[0]) + dx, float(a[1]) + dy)
                  for n, a in sg.anchor_of_node.items()}
    new_sg = replace(sg, anchor_of_node=new_anchor)
    new_g = replace(g, subgraph=new_sg)
    if not only_anchor and g.addrs:
        # shift addrs too
        new_addrs = {}
        for k, node_map in g.addrs.items():
            new_addrs[k] = {n: arr + np.array([dx, dy])
                            for n, arr in node_map.items()}
        new_g = replace(new_g, addrs=new_addrs)
    return new_g


def _run_matcher(base_g, shift_g):
    import parameters as P
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
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _build_one(arm, cls_idx=0):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    import corpus
    from frontend import build
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["view"] == "-090-000"]
    by_class = {}
    for e in dev:
        c = e["class"]
        if c not in by_class or e["object"] < by_class[c]["object"]:
            by_class[c] = e
    entries = [by_class[c] for c in corpus.CLASSES]
    return build(entries[cls_idx])


def test_0b_i_synthetic(arm="B0"):
    """0b(i): Synthetic shift test."""
    print("=== 0b(i) Synthetic shift test (arm=%s) ===" % arm, flush=True)
    g = _build_one(arm)
    base_pos = _pos_lens(g)
    token_kinds = {l for (_x, _y, l) in base_pos.values()}

    # 0px ceiling: matcher mapping accuracy at zero shift
    mapping0 = _run_matcher(g, g)
    identity_cp = {n: n for n in base_pos}
    acc0 = matcher_accuracy(base_pos, mapping0, identity_cp, token_kinds)
    print("  0px ceiling: D1=%.4f (correct=%d/%d, no_cp=%d)"
          % (acc0["acc"], acc0["correct"], acc0["total"],
             acc0["no_counterpart"]), flush=True)

    # Position stores
    stores = _position_stores(g)
    print("  Position stores: %s" % list(stores.keys()), flush=True)

    # Shift by (dx, dy) = (2, 0) — horizontal
    dx, dy = 2.0, 0.0
    shifted_g = _shift_graph_positions(g, dx, dy, only_anchor=False)
    shifted_pos = _pos_lens(shifted_g)

    # (a) every base node finds same-kind node at expected position within 1e-9
    by_lens = defaultdict(list)
    for n, (x, y, l) in shifted_pos.items():
        by_lens[l].append((n, x, y))
    a_total = a_correct = 0
    same_kind_at_pos = 0
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = expected_position(x0, y0, dx, dy)
        a_total += 1
        found = False
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            if abs(ex - xsh) < 1e-9 and abs(ey - ysh) < 1e-9:
                found = True
                same_kind_at_pos += 1
                break
        if found:
            a_correct += 1
    print("  (a) position match: %d/%d (%.4f)"
          % (a_correct, a_total, a_correct / a_total if a_total else 0),
          flush=True)

    # (b) unique-position nodes: recovered mapping = identity
    positions = [(x, y) for (x, y, _l) in base_pos.values()]
    from collections import Counter
    pos_counts = Counter(positions)
    unique_nodes = [n for n, (x, y, _l) in base_pos.items()
                    if pos_counts[(x, y)] == 1]
    b_total = b_correct = 0
    for n in unique_nodes:
        x0, y0, _l = base_pos[n]
        ex, ey = expected_position(x0, y0, dx, dy)
        b_total += 1
        # find shifted node at (ex, ey)
        for nsh, (xsh, ysh, _l2) in shifted_pos.items():
            if abs(ex - xsh) < 1e-9 and abs(ey - ysh) < 1e-9:
                if nsh == n:
                    b_correct += 1
                break
    print("  (b) unique-position identity: %d/%d (%.4f)"
          % (b_correct, b_total, b_correct / b_total if b_total else 0),
          flush=True)

    # (c) matcher mapping accuracy on shifted copy = 0px ceiling
    mapping_s = _run_matcher(g, shifted_g)
    # counterpart = identity (node n maps to n, since we shifted positions
    # but kept node ids)
    acc_s = matcher_accuracy(base_pos, mapping_s,
                             {n: n for n in base_pos}, token_kinds)
    print("  (c) matcher acc on shifted copy: %.4f (ceiling %.4f)"
          % (acc_s["acc"], acc0["acc"]), flush=True)

    # Variant: shift ONLY anchor_of_node
    shifted_g2 = _shift_graph_positions(g, dx, dy, only_anchor=True)
    shifted_pos2 = _pos_lens(shifted_g2)
    a2_total = a2_correct = 0
    by_lens2 = defaultdict(list)
    for n, (x, y, l) in shifted_pos2.items():
        by_lens2[l].append((n, x, y))
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = expected_position(x0, y0, dx, dy)
        a2_total += 1
        found = False
        for nsh, xsh, ysh in by_lens2.get(l0, ()):
            if abs(ex - xsh) < 1e-9 and abs(ey - ysh) < 1e-9:
                found = True
                break
        if found:
            a2_correct += 1
    print("  Variant (only anchor_of_node): (a) %d/%d (%.4f)"
          % (a2_correct, a2_total, a2_correct / a2_total if a2_total else 0),
          flush=True)
    # Check if results differ
    differs = (a_correct != a2_correct) or (acc_s["acc"] != acc_s["acc"])
    # Re-run matcher on only-anchor variant
    mapping_s2 = _run_matcher(g, shifted_g2)
    acc_s2 = matcher_accuracy(base_pos, mapping_s2,
                              {n: n for n in base_pos}, token_kinds)
    matcher_differs = acc_s["acc"] != acc_s2["acc"]
    print("  Matcher differs between all-stores and only-anchor: %s"
          % matcher_differs, flush=True)
    if matcher_differs:
        print("  -> some consumer reads another store: check addrs",
              flush=True)

    return {
        "0px_ceiling": acc0,
        "position_stores": list(stores.keys()),
        "shift": (dx, dy),
        "a_position_match": a_correct / a_total if a_total else 0,
        "b_unique_identity": b_correct / b_total if b_total else 0,
        "c_matcher_acc_shifted": acc_s,
        "only_anchor_a": a2_correct / a2_total if a2_total else 0,
        "only_anchor_matcher_acc": acc_s2,
        "matcher_differs": matcher_differs,
    }


def test_0b_iii_fourier():
    """0b(iii): Fourier shift on analytic sinusoid."""
    print("=== 0b(iii) Fourier shift on sinusoid ===", flush=True)
    # Analytic sinusoid: sin(2*pi*f*x) with f = 0.1 cycles/pixel
    nx, ny = 256, 256
    x = np.arange(nx)
    y = np.arange(ny)
    XX, YY = np.meshgrid(x, y)
    f = 0.1
    sinusoid = np.sin(2 * np.pi * f * XX)

    results = {}
    for shift in [0.5, 1.5]:
        shifted = fourier_shift(sinusoid, shift, 0)
        # Verify: shifted[x] should equal original[x - shift] (cyclic)
        # Check at several x positions away from the wrap boundary
        errors = []
        for x in range(20, nx - 20):
            expected_val = sinusoid[128, int(round((x - shift) % nx))]
            actual_val = shifted[128, x]
            errors.append(abs(actual_val - expected_val))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        results[shift] = {"rms_error": rms, "shift": shift}
        print("  shift=%s: RMS error vs expected = %.6f"
              % (shift, rms), flush=True)

    # Token-count change vs shift 0 (this is on synthetic; real token counts
    # reported in Part 1)
    print("  (token-count change vs shift 0 reported in Part 1)", flush=True)
    return results


def test_0b_iv_provenance():
    """0b(iv): Roll-axis provenance for reused held-out graphs."""
    print("=== 0b(iv) Roll-axis provenance ===", flush=True)
    HOLDOUT = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"
    results = {}
    # Check a few held-out pickles: compute centroid displacement of content
    # tokens vs shift-0 graph
    for arm in ["B0", "F"]:
        for name in ["apple7-090-000", "car9-090-000"]:
            base_p = HOLDOUT / ("%s_%s_0.pkl" % (arm, name))
            shift_p = HOLDOUT / ("%s_%s_2.pkl" % (arm, name))
            if not base_p.is_file() or not shift_p.is_file():
                continue
            with open(base_p, "rb") as fh:
                g0 = pickle.load(fh)
            with open(shift_p, "rb") as fh:
                g2 = pickle.load(fh)
            p0 = _pos_lens(g0)
            p2 = _pos_lens(g2)
            # content tokens = non-sheet
            content_lenses = {"CONTOUR_PEAK", "Contour", "ContourFragment",
                              "Corner", "CurveComplex", "EndStopped",
                              "Orientation", "PEAK_CHAIN", "RegionShape",
                              "IllusoryContour", "Junction"}
            # centroid of content tokens
            def centroid(pos, lenses):
                xs, ys = [], []
                for n, (x, y, l) in pos.items():
                    if l in lenses:
                        xs.append(x)
                        ys.append(y)
                if not xs:
                    return (0, 0)
                return (np.mean(xs), np.mean(ys))
            c0 = centroid(p0, content_lenses)
            c2 = centroid(p2, content_lenses)
            disp = (c2[0] - c0[0], c2[1] - c0[1])
            # Vertical roll: (0, ±s). Horizontal: (±s, 0)
            if abs(disp[0]) < 0.1 and abs(disp[1]) > 1.0:
                convention = "vertical"
            elif abs(disp[1]) < 0.1 and abs(disp[0]) > 1.0:
                convention = "horizontal"
            elif abs(disp[0]) < 0.1 and abs(disp[1]) < 0.1:
                convention = "static (no shift detected)"
            else:
                convention = "mixed"
            results["%s_%s" % (arm, name)] = {
                "displacement": disp, "convention": convention}
            print("  %s/%s shift=2: disp=(%.2f, %.2f) -> %s"
                  % (arm, name, disp[0], disp[1], convention), flush=True)
    return results


def test_0c_radius_and_tolerance(arm="B0"):
    """0c: Primary position radius per kind, parameter tolerance."""
    print("=== 0c Primary position radius + parameter tolerance (arm=%s) ==="
          % arm, flush=True)
    # Use dev graphs to compute per-kind NN spacing
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    import corpus
    from frontend import build
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["view"] == "-090-000"]
    by_class = {}
    for e in dev:
        c = e["class"]
        if c not in by_class or e["object"] < by_class[c]["object"]:
            by_class[c] = e
    entries = [by_class[c] for c in corpus.CLASSES]

    # Collect all positions per lens across dev
    all_positions = defaultdict(list)
    for e in entries:
        g = build(e)
        for n, (x, y, l) in _pos_lens(g).items():
            all_positions[l].append((x, y))

    radius_table = primary_radius(all_positions)
    for lens, r in sorted(radius_table.items()):
        n = len(all_positions.get(lens, []))
        if r is None:
            print("  %-20s n=%d -> ABSENT (<20 nodes)" % (lens, n),
                  flush=True)
        else:
            print("  %-20s n=%d -> radius=%.3f px" % (lens, n, r),
                  flush=True)
    return radius_table


def main() -> int:
    out = {}
    out["0b_i"] = test_0b_i_synthetic("B0")
    out["0b_iii"] = test_0b_iii_fourier()
    out["0b_iv"] = test_0b_iv_provenance()
    out["0c_radius"] = test_0c_radius_and_tolerance("B0")
    write_json(RESULTS_DIR / "t4" / "part0_repair.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
