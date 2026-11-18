"""Survival / matcher precision / relation preservation on renderer-truth-v1.

For each scene, at 0px / +2px / +3px, and for both modes (full-frame, and
object-local = union of the renderer-truth object masks rolled like the
image):

  survival           |truth| / |expected|        (must be 1.0 at 0px)
  rel_preserved_true fraction of base edges whose both endpoints are mapped
                     and whose images exist in the shifted graph, TRUE
                     mapping               (must be 1.0 at 0px)
  joint_mcs / relation_walker:
    recall           |M ∩ truth| / |truth|
    prec_surv        correct / (matcher pairs whose base node has a truth)
    prec_all         |M ∩ truth| / |M|
    rel_frac         relation preservation under M
  truth definitions: greedy nearest same-lens (the known-transform rule) vs
  a Hungarian global 1-1 on the same candidate set (reported alongside).

relation_walker is run under TWO regimes:
  relation_walker        the tree's default (SEED_TOPOLOGICAL=False:
                         measurement-bin seeds; observed to find ZERO
                         admissible seeds on these scenes)
  relation_walker_tokens the D1 token-seed arm registered in the walker,
                         selected by the registered switch only -- a
                         diagnostic, NOT a selected configuration

Matcher sub-100% precision at 0px is signature tie-breaking on identical
(lens, signature) nodes; the 0px errors sharing the true partner's full
signature are counted per row.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import object_local_kt as OL  # noqa: E402   (CroppedSg, union masks)

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = ((0, 0), (2, 0), (3, 0))
TOL = 8.0


def _worker(args):
    scene, shift = args
    import eth80_frontend as FE
    from PIL import Image

    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    mask = OL._union(scene)
    if shift != (0, 0):
        image = np.roll(image, (shift[1], shift[0]), axis=(0, 1))
        mask = np.roll(mask, (shift[1], shift[0]), axis=(0, 1))

    def inside(p):
        y, x = int(round(p[1])), int(round(p[0]))
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])

    trace = {}
    t0 = time.perf_counter()
    g = FE.build_graph(image, name="st_%s_%d_%d" % (scene["name"], shift[0], shift[1]),
                       record_co_located=False, trace=trace)
    cg = trace["cg"]
    positions = cg.pattern_node_position_map()
    anchors = cg.pattern_node_anchor_map()
    anchor_pos = trace.get("anchor_positions") or {}
    for n in [n for n in g.subgraph.nodes if n not in positions]:
        a = anchors.get(n)
        p = anchor_pos.get(a) if a is not None else None
        if p is not None and len(p) >= 2:
            positions[n] = (float(p[0]), float(p[1]))
    keep = {n for n in g.subgraph.nodes
            if n in positions and inside(positions[n])}
    cropped = OL.CroppedSg(g.subgraph, keep,
                            len(getattr(g.subgraph, "t_junctions", ()) or ()))
    sigs = g.sigs
    lens = {n: sigs[n][0] for n in g.subgraph.nodes if n in sigs}

    def view(edges, edge_params, anchor_of_node, t_junctions, node_set, positions):
        return {
            "nodes": set(node_set),
            "edges": list(edges),
            "edge_params": dict(edge_params or {}),
            "anchor_of_node": dict(anchor_of_node or {}),
            "t_junctions": list(t_junctions or ()),
            "positions": {n: p for n, p in positions.items() if n in node_set},
            "sigs": {n: sigs[n] for n in node_set if n in sigs},
            "lens": {n: lens[n] for n in node_set if n in lens},
        }

    full_obj = g.subgraph
    full = view(full_obj.edges, getattr(full_obj, "edge_params", None),
                getattr(full_obj, "anchor_of_node", None),
                getattr(full_obj, "t_junctions", None),
                set(full_obj.nodes), positions)
    object_view = view(cropped.edges, cropped.edge_params, cropped.anchor_of_node,
                       cropped.t_junctions, keep, positions)
    grid = {
        "full_nodes": len(full_obj.nodes),
        "full_edges": len(full_obj.edges),
        "kept_nodes": len(object_view["nodes"]),
        "kept_edges": len(object_view["edges"]),
        "seconds": time.perf_counter() - t0,
    }
    return {"name": scene["name"], "split": scene["split"], "shift": tuple(shift),
            "grid": grid, "full": full, "object": object_view}


class ViewSg:
    def __init__(self, v):
        self.nodes = frozenset(v["nodes"])
        self.edges = list(v["edges"])
        self.edge_params = v["edge_params"]
        self.anchor_of_node = v["anchor_of_node"]
        self.t_junctions = v["t_junctions"]


def _truth_greedy(a, b, shift):
    expected = {u: (p[0] + shift[0], p[1] + shift[1]) for u, p in a["positions"].items()}
    cands = []
    for u, (ex, ey) in expected.items():
        lu = a["lens"].get(u)
        for v, (px, py) in b["positions"].items():
            if b["lens"].get(v) != lu:
                continue
            d = math.hypot(px - ex, py - ey)
            if d <= TOL:
                cands.append((d, u, v))
    cands.sort()
    uu, vv, truth = set(), set(), {}
    amb = 0
    for i, (d, u, v) in enumerate(cands):
        if u in uu or v in vv:
            continue
        uu.add(u); vv.add(v); truth[u] = v
    # ambiguous ties: same-lens pair with identical distance to another candidate
    shared = {}
    for d, u, v in cands:
        shared.setdefault(u, []).append(d)
    for u, ds in shared.items():
        if len(ds) >= 2 and abs(ds[0] - ds[1]) < 1e-9:
            amb += 1
    return truth, expected, amb


def _truth_hungarian(a, b, shift):
    expected = {u: (p[0] + shift[0], p[1] + shift[1]) for u, p in a["positions"].items()}
    from scipy.optimize import linear_sum_assignment
    na = sorted(expected); nb = sorted(b["positions"])
    if not na or not nb:
        return {}, expected
    big = 1e9
    cost = np.full((len(na), len(nb)), big)
    for i, u in enumerate(na):
        lu = a["lens"].get(u)
        ex, ey = expected[u]
        for j, v in enumerate(nb):
            if b["lens"].get(v) != lu:
                continue
            p = b["positions"][v]
            d = math.hypot(p[0] - ex, p[1] - ey)
            if d <= TOL:
                cost[i, j] = d
    ri, ci = linear_sum_assignment(cost)
    return ({na[i]: nb[j] for i, j in zip(ri, ci) if cost[i, j] <= TOL}, expected)


def _preserved_edges(a_edges, b_edges, mapping):
    from core.types import UNDIRECTED_EDGE_TYPES

    def key(s, d, et):
        if et in UNDIRECTED_EDGE_TYPES:
            return (min(s, d), max(s, d), et)
        return (s, d, et)

    ou = {key(s, d, et) for (s, d, et) in b_edges}
    eligible = preserved = 0
    for (s, d, et) in a_edges:
        u, v = mapping.get(s), mapping.get(d)
        if u is None or v is None:
            continue
        eligible += 1
        if key(u, v, et) in ou:
            preserved += 1
    return eligible, preserved


def _matcher_stats(a, matcher_map, truth):
    inter = sum(1 for (u, v) in matcher_map.items() if truth.get(u) == v)
    in_surv = sum(1 for u in matcher_map if u in truth)
    correct = sum(1 for u in matcher_map if u in truth and matcher_map[u] == truth[u])
    return {"mapped": len(matcher_map), "intersection": inter,
            "recall": inter / len(truth) if truth else 0.0,
            "prec_surv": (correct / in_surv) if in_surv else 0.0,
            "prec_all": inter / len(matcher_map) if matcher_map else 0.0}


def _run_matchers(a, b):
    from matching.joint_mcs import joint_mcs
    import matching.relation_walker as RWS
    a_sg, b_sg = ViewSg(a), ViewSg(b)
    out = {"joint_mcs": dict(joint_mcs(a_sg, b_sg, a["sigs"], b["sigs"]).mapping)}
    try:
        got = RWS.walk(a_sg, b_sg, a["sigs"], b["sigs"])
        out["relation_walker"] = dict(got.mapping)
    except Exception as exc:  # noqa: BLE001
        out["relation_walker"] = {}
        out["relation_walker_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        old = RWS.SEED_TOPOLOGICAL
        try:
            RWS.SEED_TOPOLOGICAL = True
            got = RWS.walk(a_sg, b_sg, a["sigs"], b["sigs"])
            out["relation_walker_tokens"] = dict(got.mapping)
        finally:
            RWS.SEED_TOPOLOGICAL = old
    except Exception as exc:  # noqa: BLE001
        out["relation_walker_tokens"] = {}
        out["relation_walker_tokens_error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def _grid_diagnostics(a):
    pos = list(a["positions"].values())
    intx = sum(1 for p in pos if abs(p[0] - round(p[0])) < 1e-6)
    inty = sum(1 for p in pos if abs(p[1] - round(p[1])) < 1e-6)
    best_chime, best_pitch = 0.0, 0
    for pitch in (2, 3, 4, 6, 8, 12, 16, 24, 32):
        alx = sum(1 for p in pos if abs(p[0] / pitch - round(p[0] / pitch)) < 1e-6)
        aly = sum(1 for p in pos if abs(p[1] / pitch - round(p[1] / pitch)) < 1e-6)
        chime = (alx + aly) / (2 * max(1, len(pos)))
        if chime > best_chime:
            best_chime, best_pitch = chime, pitch
    return {"positions": len(pos), "integral_x": intx,
            "integral_y": inty,
            "best_grid_pitch": best_pitch, "best_grid_chime": round(best_chime, 4)}


def _row(args):
    scene, mode, shift, a, b = args
    t0 = time.perf_counter()
    truth, expected, amb = _truth_greedy(a, b, shift)
    truth_h, _ = _truth_hungarian(a, b, shift)
    h_delta = len(set(truth.items()) ^ set(truth_h.items()))
    grid = _grid_diagnostics(a)
    matchers = _run_matchers(a, b)
    rel_t = _preserved_edges(a["edges"], b["edges"], truth)

    def _common(mmap):
        common = [e for e in a["edges"]
                  if truth.get(e[0]) is not None and truth.get(e[1]) is not None
                  and mmap.get(e[0]) is not None and mmap.get(e[1]) is not None]
        if not common:
            return 0, 0, 0
        pt = _preserved_edges(common, b["edges"], truth)
        pm = _preserved_edges(common, b["edges"], mmap)
        return len(common), pt[1], pm[1]

    mrow = {}
    for mname in ("joint_mcs", "relation_walker", "relation_walker_tokens"):
        mmap = matchers.get(mname, {})
        err = matchers.get(mname + "_error")
        if err:
            mrow[mname] = {"error": err}
            continue
        mstats = _matcher_stats(a, mmap, truth)
        rel_m = _preserved_edges(a["edges"], b["edges"], mmap)
        tie_0 = 0
        if shift == (0, 0):
            rev_sig = a["sigs"]
            for u, v in mmap.items():
                if u in truth and v != truth[u]:
                    if rev_sig.get(u) is not None and rev_sig.get(truth[u]) == rev_sig.get(u):
                        tie_0 += 1
        cb = _common(mmap)
        mrow[mname] = {**mstats,
                       "rel_preserved": rel_m[1], "rel_eligible": rel_m[0],
                       "rel_frac": (rel_m[1] / rel_m[0]) if rel_m[0] else 0.0,
                       "rel_common_eligible": cb[0],
                       "rel_common_true_preserved": cb[1],
                       "rel_common_matcher_preserved": cb[2],
                       "zero_px_same_signature_errors": tie_0}
    broken = False
    if shift == (0, 0):
        broken = (len(truth) != len(expected)) or (rel_t[0] != rel_t[1])
    return {
        "scene": scene, "mode": mode, "shift": list(shift),
        "grid": grid,
        "expected_nodes": len(expected),
        "truth_nodes": len(truth),
        "survival": (len(truth) / len(expected)) if expected else 0.0,
        "truth_ambiguous_ties": amb,
        "truth_hungarian": len(truth_h),
        "truth_delta": h_delta,
        "rel_true_preserved": rel_t[1], "rel_true_eligible": rel_t[0],
        "rel_true_frac": (rel_t[1] / rel_t[0]) if rel_t[0] else 0.0,
        "matchers": mrow,
        "instrument_broken": bool(broken),
        "seconds": time.perf_counter() - t0,
    }


def _fmt(v, nd=3):
    return ("%.3f" % v) if isinstance(v, (int, float)) else str(v)


def _print_row(r):
    jm, rw, rwt = (r["matchers"].get(k, {}) for k in
                   ("joint_mcs", "relation_walker", "relation_walker_tokens"))
    def col(m, k):
        return "%.3f" % m[k] if isinstance(m.get(k), (int, float)) else "  err"
    def dots(m):
        if "error" in m:
            return "  err   -     -     -     -     -   "
        return "%-3d %.3f %.3f %.3f %3d/%-3d" % (
            m["mapped"], m["recall"], m["prec_surv"], m["prec_all"],
            m["rel_preserved"], m["rel_eligible"])
    print("%-20s %-6s %+d  exp %4d truth %4d surv %.3f hun %4d dlt %3d | JM %s | RW %s | RWT %s%s" % (
        r["scene"], r["mode"], r["shift"][0], r["expected_nodes"], r["truth_nodes"],
        r["survival"], r["truth_hungarian"], r["truth_delta"],
        dots(jm), dots(rw), dots(rwt),
        "  [INSTRUMENT-BROKEN]" if r["instrument_broken"] else ""), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "survival_table.json"))
    args = ap.parse_args(argv)
    data = json.loads(TRUTH_PATH.read_text())
    scenes = data["scenes"]
    if args.split != "all":
        scenes = [s for s in scenes if s["split"] == args.split]
    if args.scenes:
        scenes = [s for s in scenes if s["name"] in set(args.scenes)]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=guard.activate) as pool:
        built = pool.map(_worker, [(s, shift) for s in scenes for shift in SHIFTS],
                         chunksize=1)
    by = {(b["name"], b["shift"]): b for b in built}
    print("[built %d scene-shift graphs in %.1fs]" %
          (len(built) * 2, time.perf_counter() - t0), flush=True)

    row_tasks = []
    for scene in scenes:
        b0 = by[(scene["name"], (0, 0))]
        for shift in SHIFTS:
            other = by[(scene["name"], shift)]
            for mode in ("full", "object"):
                row_tasks.append((scene["name"], mode, shift, b0[mode], other[mode]))
    t1 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=guard.activate) as pool:
        futures = [pool.apply_async(_row, (t,)) for t in row_tasks]
        rows = []
        try:
            for f in futures:
                rows.append(f.get())
        except KeyboardInterrupt:
            pool.terminate()
            raise
    for r in rows:
        _print_row(r)
    print("[rows computed in %.1fs]" % (time.perf_counter() - t1))

    # aggregate: per mode/shift
    agg = {}
    for r in rows:
        k = (r["mode"], tuple(r["shift"]))
        a = agg.setdefault(k, {"expected": 0, "truth": 0, "edges": 0, "eligible": 0,
                               "matchers": {"joint_mcs": [0, 0], "relation_walker": [0, 0],
                                            "relation_walker_tokens": [0, 0]}})
        a["expected"] += r["expected_nodes"]
        a["truth"] += r["truth_nodes"]
        a["edges"] += r["rel_true_preserved"]
        a["eligible"] += r["rel_true_eligible"]
    print("\nAGGREGATE:")
    for (mode, shift), a in sorted(agg.items()):
        print("  %-6s %+d  expected %5d truth %5d survival %.4f  true-rel %d/%d" % (
            mode, shift[0], a["expected"], a["truth"],
            (a["truth"] / a["expected"]) if a["expected"] else 0.0,
            a["edges"], a["eligible"]))

    out = {"protocol": "survival / precision / relation preservation, renderer-truth-v1, "
                       "full-frame vs object-local mask union",
           "tolerance_px": TOL,
           "shifts": [list(s) for s in SHIFTS],
           "rows": rows}
    write_json(Path(args.out), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
