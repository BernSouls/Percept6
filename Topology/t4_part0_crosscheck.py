"""0b(ii) Cross-check: known_transform.py vs repaired harness on the same image.

Runs the repaired harness (t4_shift helpers) on the same image at TOL 8 px
with its own truth functions (continuous and cell), and reports side by side
with known_transform_corrected.json.
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

from t4_shift import roll_image, expected_position  # noqa: E402

ARMS = {"B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false"}
SHIFTS = [(0, 0), (2, 0), (-2, 0), (3, 0), (-3, 0)]
TOL = 8.0
PITCH = 16.0


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _build(entry, dx, dy):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS["B0"])
    import eth80_frontend as FE
    with Image.open(entry["path"]) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if dx != 0 or dy != 0:
        image = roll_image(image, dx, dy)
    trace = {}
    g = FE.build_graph(image, name="xcheck_%d_%d" % (dx, dy),
                       record_co_located=False, trace=trace)
    return g


def _match_continuous(base_pos, other_pos, dx, dy, tol):
    """One-to-one nearest same-lens match of expected positions (strict <).

    Appends ALL candidates within tol, sorts by distance, greedy one-to-one.
    """
    by_lens = defaultdict(list)
    for n, (x, y, l) in other_pos.items():
        by_lens[l].append((n, x, y))
    candidates = []
    for u, (x0, y0, lu) in base_pos.items():
        ex, ey = expected_position(x0, y0, dx, dy)
        for v, px, py in by_lens.get(lu, ()):
            d = math.hypot(px - ex, py - ey)
            if d < tol:
                candidates.append((d, u, v))
    # greedy one-to-one
    candidates.sort()
    used_u, used_v, pairs = set(), set(), {}
    for d, u, v in candidates:
        if u in used_u or v in used_v:
            continue
        used_u.add(u)
        used_v.add(v)
        pairs[u] = v
    return pairs


def _match_cells(base_pos, other_pos, dx, dy, pitch):
    """Historical cell instrument: same integer cell after shift."""
    cells_base = {}
    for u, (x0, y0, _l) in base_pos.items():
        ex, ey = expected_position(x0, y0, dx, dy)
        cells_base[u] = (int(math.floor(ex / pitch)),
                         int(math.floor(ey / pitch)))
    cells_other = defaultdict(list)
    for v, (px, py, _l) in other_pos.items():
        cells_other[(int(math.floor(px / pitch)),
                    int(math.floor(py / pitch)))].append(v)
    pairs = {}
    for u, cell in sorted(cells_base.items()):
        for v in cells_other.get(cell, ()):
            pairs[u] = v
            break
    return pairs, len(cells_base)


def _run_matcher(base_g, other_g):
    import parameters as P
    for k, v in {
        "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
        "MATCH_UNMATCHED_COST": 0.15,
        "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
        "MATCH_EDGE_WEIGHT": 0.1,
        "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
        "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
        "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
        "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
    }.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, other_g.subgraph,
                    base_g.sigs, other_g.sigs)
    return res.mapping


def main() -> int:
    import corpus
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["class"] == "apple"
           and e["view"] == "-090-000" and e["object"] == 4]
    entry = dev[0]
    print("image:", entry["path"], flush=True)

    # Build base (shift 0)
    base_g = _build(entry, 0, 0)
    base_pos = _pos_lens(base_g)
    print("base nodes:", len(base_pos), flush=True)

    rows = []
    for dx, dy in SHIFTS:
        other_g = _build(entry, dx, dy)
        other_pos = _pos_lens(other_g)
        # continuous truth
        cont_pairs = _match_continuous(base_pos, other_pos, dx, dy, TOL)
        # cell truth
        cell_pairs, cell_n = _match_cells(base_pos, other_pos, dx, dy, PITCH)
        # matcher
        if dx == 0 and dy == 0:
            matcher = {u: u for u in base_pos}
        else:
            matcher = _run_matcher(base_g, other_g)
        # score
        cont_hits = sum(1 for u, v in matcher.items()
                        if u in cont_pairs and matcher[u] == cont_pairs[u])
        cell_hits = sum(1 for u, v in matcher.items()
                        if u in cell_pairs and matcher[u] == cell_pairs[u])
        jac = None
        if dx != 0 or dy != 0:
            common = set(matcher.values()) & set(cont_pairs.values())
            union = set(matcher.values()) | set(cont_pairs.values())
            jac = len(common) / len(union) if union else 0.0
        rows.append({
            "shift": [dx, dy],
            "truth_frac": len(cont_pairs) / len(base_pos) if base_pos else 0,
            "truth_nodes": len(cont_pairs),
            "matcher_frac": cont_hits / len(base_pos) if base_pos else 0,
            "matcher_nodes": len(matcher),
            "anchor_cell_frac": len(cell_pairs) / cell_n if cell_n else 0,
            "jaccard": jac,
            "nodes": len(other_pos),
        })
        print("shift=(%+d,%+d): truth=%.4f matcher=%.4f cell=%.4f jaccard=%s"
              % (dx, dy, rows[-1]["truth_frac"], rows[-1]["matcher_frac"],
                 rows[-1]["anchor_cell_frac"],
                 "%.4f" % jac if jac is not None else "n/a"), flush=True)

    # Load known_transform results for side-by-side
    kt = json.load(open(RESULTS_DIR / "known_transform_corrected.json"))
    kt_rows = {tuple(r["shift"]): r for r in kt["rows"]}
    comparison = []
    for r in rows:
        s = tuple(r["shift"])
        ktr = kt_rows.get(s, {})
        comparison.append({
            "shift": list(s),
            "repaired_truth": r["truth_frac"],
            "kt_truth": ktr.get("truth_frac"),
            "repaired_matcher": r["matcher_frac"],
            "kt_matcher": ktr.get("matcher_frac"),
            "repaired_cell": r["anchor_cell_frac"],
            "kt_cell": ktr.get("anchor_cell_frac"),
        })
    out = {"image": entry["path"], "tol_px": TOL, "pitch": PITCH,
           "rows": rows, "comparison": comparison,
           "warp": "roll_image(arr, dx, dy) = np.roll(arr, (dy, dx), axis=(0,1))"}
    write_json(RESULTS_DIR / "t4" / "part0_crosscheck.json", out)
    print("=== Side by side ===", flush=True)
    for c in comparison:
        print("shift=%s: truth rep=%.4f kt=%s | matcher rep=%.4f kt=%s | cell rep=%.4f kt=%s"
              % (c["shift"], c["repaired_truth"],
                 "%.4f" % c["kt_truth"] if c["kt_truth"] is not None else "n/a",
                 c["repaired_matcher"],
                 "%.4f" % c["kt_matcher"] if c["kt_matcher"] is not None else "n/a",
                 c["repaired_cell"],
                 "%.4f" % c["kt_cell"] if c["kt_cell"] is not None else "n/a"),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
