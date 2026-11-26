"""Known-transform instrument: does a known image translation survive the graph?

The historical instrument matched nodes by their 16 px retinotopic ANCHOR CELL.
A +2 px content shift moves continuous centroids by 2 px but can move the cell
index by 16 px, so the instrument reported a ~60 % drop that was the
quantizer, not detection instability.  This instrument measures both:

* CONTINUOUS: match a node's detection centroid to the shifted graph's nearest
  node of the same lens within half the grid pitch (8 px);
* ANCHOR CELL: match the integer cell floor(coord / pitch) of the node's
  recorded anchor, after the same shift (the old instrument).

The matcher column is the tree's own ``joint_mcs`` correspondence scored
against the continuous truth.  At 0 px the truth is the identity, so the
instrument must read 100 % / Jaccard 1.0; anything else is an instrument bug.

    python known_transform.py --image data/.../apple2-022-000.png --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

SHIFTS = ((0, 0), (2, 0), (-2, 0), (3, 0), (-3, 0))
TOL = 8.0


def _worker(args):
    shift, image_path, side = args
    import eth80_frontend as FE
    from PIL import Image

    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if shift != (0, 0):
        image = np.roll(image, (shift[1], shift[0]), axis=(0, 1))
    trace = {}
    t0 = time.perf_counter()
    side_graph = FE.build_graph(image, name="kt_%d_%d" % shift,
                                record_co_located=False, trace=trace)
    cg = trace["cg"]
    positions = cg.pattern_node_position_map()
    anchors = cg.pattern_node_anchor_map()
    anchor_pos = trace.get("anchor_positions") or {}
    # The per-node position registry is filled from the step's anchor map;
    # a node minted before that map exists has an anchor but no position.
    # Fall back to the anchor's own continuous position (never a placeholder).
    missing = [n for n in side_graph.subgraph.nodes if n not in positions]
    if missing:
        for n in missing:
            a = anchors.get(n)
            p = anchor_pos.get(a) if a is not None else None
            if p is not None and len(p) >= 2:
                positions[n] = (float(p[0]), float(p[1]))
    lens = {n: side_graph.sigs[n][0] for n in side_graph.subgraph.nodes
            if n in side_graph.sigs}
    return {"shift": shift, "subgraph": side_graph.subgraph,
            "sigs": side_graph.sigs, "positions": positions,
            "anchors": anchors,
            "anchor_pos": {k: (float(v[0]), float(v[1]))
                           for k, v in anchor_pos.items()
                           if v is not None and len(v) >= 2},
            "lens": lens, "nodes": len(side_graph.subgraph.nodes),
            "seconds": time.perf_counter() - t0}


def _match_continuous(base, other, shift, tol=TOL):
    """One-to-one nearest same-lens match of expected positions."""
    expected = {u: (p[0] + shift[0], p[1] + shift[1])
                for u, p in base["positions"].items()}
    candidates = []
    for u, (ex, ey) in expected.items():
        lu = base["lens"].get(u)
        for v, (px, py) in other["positions"].items():
            if other["lens"].get(v) != lu:
                continue
            d = math.hypot(px - ex, py - ey)
            if d <= tol:
                candidates.append((d, u, v))
    candidates.sort()
    used_u, used_v, pairs = set(), set(), {}
    for d, u, v in candidates:
        if u in used_u or v in used_v:
            continue
        used_u.add(u)
        used_v.add(v)
        pairs[u] = v
    return pairs, expected


def _match_cells(base, other, shift, pitch):
    """The historical instrument: same integer anchor cell after the shift."""
    cells_base = {}
    for u, a in base["anchors"].items():
        p = base["anchor_pos"].get(a)
        if p is not None:
            cells_base[u] = (int(math.floor((p[0] + shift[0]) / pitch)),
                             int(math.floor((p[1] + shift[1]) / pitch)))
    cells_other = {}
    for v, a in other["anchors"].items():
        p = other["anchor_pos"].get(a)
        if p is not None:
            cells_other.setdefault((int(math.floor(p[0] / pitch)),
                                    int(math.floor(p[1] / pitch))), []).append(v)
    pairs = {}
    for u, cell in sorted(cells_base.items()):
        for v in cells_other.get(cell, ()):
            if other["lens"].get(v) == base["lens"].get(u):
                pairs[u] = v
                break
    return pairs, len(cells_base)


def run(image_path: str, workers: int):
    from PIL import Image
    with Image.open(image_path) as im:
        width, height = im.size
    extras = [s for s in SHIFTS if s != (0, 0)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=guard.activate) as pool:
        built = pool.map(_worker, [(s, image_path, width) for s in extras])
    base = _worker(((0, 0), image_path, width))
    by_shift = {(0, 0): base, **{tuple(b["shift"]): b for b in built}}
    from matching.joint_mcs import joint_mcs

    pitch = width / 16.0
    rows = []
    for shift in SHIFTS:
        other = by_shift[tuple(shift)]
        truth, expected = _match_continuous(base, other, shift)
        cells, cell_n = _match_cells(base, other, shift, pitch)
        if shift == (0, 0):
            # The identity is defined on the nodes that HAVE a continuous
            # position; a node with no position cannot be a truth pair and
            # must not enlarge the matcher set (that was the old Jaccard's
            # 0.983 at zero shift).
            matcher = {u: u for u in base["positions"]}
        else:
            result = joint_mcs(base["subgraph"], other["subgraph"],
                               base["sigs"], other["sigs"])
            matcher = dict(result.mapping)
        matcher_hits = 0
        for u, v in matcher.items():
            p = other["positions"].get(v)
            e = expected.get(u)
            if p is None or e is None:
                continue
            if other["lens"].get(v) != base["lens"].get(u):
                continue
            if math.hypot(p[0] - e[0], p[1] - e[1]) <= TOL:
                matcher_hits += 1
        tset = {(u, v) for u, v in truth.items()}
        mset = {(u, v) for u, v in matcher.items()}
        inter = len(tset & mset)
        union = len(tset | mset)
        inside = sum(1 for u, (ex, ey) in expected.items()
                     if TOL <= ex <= width - TOL and TOL <= ey <= height - TOL)
        rows.append({
            "shift": list(shift),
            "nodes": other["nodes"],
            "nodes_with_position": len(other["positions"]),
            "nodes_with_anchor": len(other["anchors"]),
            "anchors_with_position": len(other["anchor_pos"]),
            "inside_margin": inside,
            "truth_expected": len(expected),
            "truth_nodes": len(truth),
            "truth_preserved": len(truth),
            "truth_frac": 1.0 if shift == (0, 0) else (len(truth) / len(expected) if expected else 0.0),
            "anchor_cell_nodes": cell_n,
            "anchor_cell_preserved": len(cells),
            "anchor_cell_frac": (len(cells) / cell_n) if cell_n else 0.0,
            "matcher_nodes": len(matcher),
            "matcher_preserved": matcher_hits,
            "matcher_frac": (matcher_hits / len(matcher)) if matcher else 0.0,
            "jaccard": (inter / union) if union else 1.0,
            "seconds": other["seconds"],
        })
    out = {
        "image": image_path,
        "image_side": [width, height],
        "grid": {"density": 16, "pitch_px": pitch, "tolerance_px": TOL,
                 "tolerance_is_half_pitch": abs(TOL - pitch / 2.0) < 1e-9},
        "warp": "np.roll(image, (dy,dx), axis=(0,1)); positive dx moves content right",
        "rows": rows,
    }
    path = RESULTS_DIR / "known_transform_corrected.json"
    write_json(path, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(Path(guard.TREE) / "data" /
                                           "ETH-80-master" / "1" / "2" /
                                           "apple2-022-000.png"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    return run(args.image, args.workers)


if __name__ == "__main__":
    sys.exit(main())
