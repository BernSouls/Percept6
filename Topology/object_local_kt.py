"""Object-local known-transform on the renderer-truth scenes.

The full-frame known-transform caps at ~0.60 Jaccard.  Hypothesis: background
texture edges dominate the correspondence, or the front-end breaks the
object'\''s own edges under translation.  This instrument crops the graph to the
pixel-perfect object mask provided by the renderer corpus and re-runs the
same known-transform: keep only nodes whose continuous anchor position falls
inside the (rolled) object mask; keep only edges with BOTH endpoints kept.

Correlated with the mask: the rolled image moves content, so the same roll is
applied to the mask; a node of the shifted graph is kept iff its position lies
inside the ROLLED mask.  Nodes without a continuous position cannot be
verified inside -> dropped (same conservative rule as the full-frame
instrument'\''s truth, which uses positions).

Report per scene and aggregate at 0, +2, -2, +3, -3 px:
  truth_frac, matcher_frac, Jaccard, mapped-node count, preserved edges.

No ETH-80 is measured.  No parameter is tuned.
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

import known_transform as KT  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = ((0, 0), (2, 0), (-2, 0), (3, 0), (-3, 0))
TOL = 8.0


def _masks(scene):
    out = {}
    for name, data in scene["masks"].items():
        a = np.asarray(data, dtype=bool)
        if a.shape != (192, 192):
            raise ValueError("unexpected mask shape %s for %s/%s" % (a.shape, scene["name"], name))
        out[name] = a
    return out


def _union(scene):
    gets = _masks(scene)
    u = None
    for a in gets.values():
        u = a.copy() if u is None else (u | a)
    return u if u is not None else np.zeros((192, 192), dtype=bool)


class CroppedSg:
    """A duck-typed subgraph: nodes / edges / edge_params / anchor_of_node
    / t_junctions, exactly what matching.joint_mcs reads."""

    def __init__(self, sg, keep, dropped_junctions):
        self.nodes = frozenset(keep)
        self.edges = [(s, d, et) for (s, d, et) in sg.edges
                      if s in keep and d in keep]
        ep = getattr(sg, "edge_params", None) or {}
        self.edge_params = {k: v for k, v in ep.items()
                            if _endpoint_keep(k, keep)}
        aof = getattr(sg, "anchor_of_node", None) or {}
        self.anchor_of_node = {n: p for n, p in aof.items() if n in keep}
        # [conservative] junctions that reference any dropped node are not
        # carried over; empty in the captured traces of these scenes.
        self.t_junctions = []


def _endpoint_keep(key, keep):
    try:
        s, d = key[0], key[1]
    except (TypeError, KeyError):
        return False
    return (s in keep) and (d in keep)


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    full = image
    if shift != (0, 0):
        image = np.roll(image, (shift[1], shift[0]), axis=(0, 1))
    mask = _union(scene)
    if shift != (0, 0):
        mask = np.roll(mask, (shift[1], shift[0]), axis=(0, 1))

    def inside(p):
        y, x = int(round(p[1])), int(round(p[0]))
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])

    trace = {}
    g = FE.build_graph(image, name="rt_%s_%d_%d" % (scene["name"], shift[0], shift[1]),
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
    dropped_junctions = len(getattr(g.subgraph, "t_junctions", ()) or ())
    cropped = CroppedSg(g.subgraph, keep, dropped_junctions)
    full_pos = len(positions)
    inside_ct = sum(1 for p in positions.values() if inside(p))
    mask_px = int(mask.sum())
    total_px = int(mask.size)
    return {
        "name": scene["name"],
        "split": scene["split"],
        "shift": tuple(shift),
        "full_nodes": len(g.subgraph.nodes),
        "full_edges": len(g.subgraph.edges),
        "kept_nodes": len(keep),
        "kept_edges": len(cropped.edges),
        "full_positions": full_pos,
        "full_inside": inside_ct,
        "full_outside": full_pos - inside_ct,
        "keep_frac_of_full_pos": (inside_ct / full_pos) if full_pos else 0.0,
        "mask_pixels": mask_px,
        "density_inside": (inside_ct / mask_px) if mask_px else 0.0,
        "density_outside": ((full_pos - inside_ct) / (total_px - mask_px))
        if total_px > mask_px else 0.0,
        "cropped": cropped,
        "sigs": {n: g.sigs[n] for n in keep if n in g.sigs},
        "positions": {n: p for n, p in positions.items() if n in keep},
        "lens": {n: g.sigs[n][0] for n in keep if n in g.sigs},
        "dropped_junctions": dropped_junctions,
        "seconds": 0.0,
    }


def _worker(args):
    scene, shift = args
    t0 = time.perf_counter()
    out = _build(scene, shift)
    out["seconds"] = time.perf_counter() - t0
    return out


def _match_continuous_from(base_pos, base_lens, other_pos, other_lens, shift, tol=TOL):
    expected = {u: (p[0] + shift[0], p[1] + shift[1]) for u, p in base_pos.items()}
    cands = []
    for u, (ex, ey) in expected.items():
        lu = base_lens.get(u)
        for v, (px, py) in other_pos.items():
            if other_lens.get(v) != lu:
                continue
            d = math.hypot(px - ex, py - ey)
            if d <= tol:
                cands.append((d, u, v))
    cands.sort()
    uu, vv, truth = set(), set(), {}
    for d, u, v in cands:
        if u in uu or v in vv:
            continue
        uu.add(u); vv.add(v); truth[u] = v
    return truth, expected


def _preserved(a, b, mapping):
    from core.types import UNDIRECTED_EDGE_TYPES

    def key(s, d, et):
        if et in UNDIRECTED_EDGE_TYPES:
            return (min(s, d), max(s, d), et)
        return (s, d, et)

    ou = {key(s, d, et) for (s, d, et) in b.edges}
    eligible = preserved = 0
    for (s, d, et) in a.edges:
        u, v = mapping.get(s), mapping.get(d)
        if u is None or v is None:
            continue
        eligible += 1
        if key(u, v, et) in ou:
            preserved += 1
    return eligible, preserved


def run(scenes, workers, only_split=None):
    import multiprocessing as mp
    tasks = [(s, shift) for s in scenes for shift in SHIFTS if shift != (0, 0)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=guard.activate) as pool:
        built = pool.map(_worker, tasks)
    built_by = {(b["name"], tuple(b["shift"])): b for b in built}
    rows = []
    from matching.joint_mcs import joint_mcs
    for scene in scenes:
        base = _worker((scene, (0, 0)))
        for shift in SHIFTS:
            other = base if shift == (0, 0) else built_by[(scene["name"], shift)]
            truth, expected = _match_continuous_from(base["positions"], base["lens"],
                                                     other["positions"], other["lens"],
                                                     shift)
            tset = {(u, v) for u, v in truth.items()}
            if shift == (0, 0):
                matcher = {u: u for u in base["positions"]}
                score = 1.0
            else:
                res = joint_mcs(base["cropped"], other["cropped"], base["sigs"], other["sigs"])
                matcher = dict(res.mapping)
                score = res.score
            hits = 0
            for u, v in matcher.items():
                p = other["positions"].get(v); e = expected.get(u)
                if p is None or e is None:
                    continue
                if other["lens"].get(v) != base["lens"].get(u):
                    continue
                if math.hypot(p[0] - e[0], p[1] - e[1]) <= TOL:
                    hits += 1
            mset = set(matcher.items())
            inter = len(tset & mset); union = len(tset | mset)
            leg, pres = _preserved(base["cropped"], other["cropped"], matcher)
            rows.append({
                "scene": scene["name"], "split": scene["split"],
                "shift": list(shift),
                "full_nodes": other["full_nodes"], "kept_nodes": other["kept_nodes"],
                "kept_edges": other["kept_edges"],
                "full_positions": other["full_positions"],
                "full_inside": other["full_inside"],
                "full_outside": other["full_outside"],
                "keep_frac_of_full_pos": other["keep_frac_of_full_pos"],
                "mask_pixels": other["mask_pixels"],
                "mask_frac_of_image": other["mask_pixels"] / float(192 * 192),
                "density_inside": other["density_inside"],
                "density_outside": other["density_outside"],
                "density_ratio": (other["density_inside"] / other["density_outside"])
                if other["density_outside"] else 0.0,
                "truth_expected": len(expected), "truth_nodes": len(truth),
                "truth_frac": 1.0 if shift == (0, 0) else (len(truth) / len(expected) if expected else 0.0),
                "matcher_nodes": len(matcher), "matcher_preserved": hits,
                "matcher_frac": (hits / len(matcher)) if matcher else 0.0,
                "jaccard": (inter / union) if union else 1.0,
                "edge_eligible": leg, "edge_preserved": pres,
                "edge_preserved_frac": (pres / leg) if leg else 0.0,
                "dropped_junctions": other["dropped_junctions"],
                "score": float(score), "build_seconds": other["seconds"],
            })
            print("%-22s %+d  keep %3d/%3d edges %3d  truth %3d  match %3d  frac %.3f  jaccard %.4f  dens in/out %.4f/%.4f" %
                  (scene["name"], shift[0], other["kept_nodes"], other["full_nodes"], other["kept_edges"],
                   len(truth), len(matcher), rows[-1]["matcher_frac"], rows[-1]["jaccard"],
                   other["density_inside"], other["density_outside"]), flush=True)
    # aggregate: is the detector density uniform across the mask boundary?
    agg = {}
    for r in rows:
        s_key = str(tuple(r["shift"]))
        agg.setdefault(s_key, []).append(r)
    print("\nAGGREGATE (all scenes):")
    summary = {}
    for shiftkey, rs in sorted(agg.items()):
        mi = sum(r["mask_pixels"] for r in rs)
        fi = sum(r["full_inside"] for r in rs)
        fo = sum(r["full_outside"] for r in rs)
        ni_frac = sum(r["density_inside"] / max(1e-9, r["density_outside"]) for r in rs) / len(rs)
        summary[shiftkey] = {"inside_density": fi / mi, "outside_density": fo / (len(rs) * 192 * 192 - mi),
                             "mean_ratio": ni_frac, "scenes": len(rs)}
        print("  shift %-6s density inside %.5f outside %.5f mean ratio %.3f" % (
            shiftkey, summary[shiftkey]["inside_density"], summary[shiftkey]["outside_density"],
            ni_frac))
    out = {"protocol": "object-local known-transform, renderer-truth-v1, union object mask, rolled with the image",
           "rows": rows, "aggregate": summary}
    path = RESULTS_DIR / "object_local_known_transform.json"
    write_json(path, out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="restrict to these scene names")
    args = ap.parse_args(argv)
    data = json.loads(TRUTH_PATH.read_text())
    scenes = data["scenes"]
    if args.split != "all":
        scenes = [s for s in scenes if s["split"] == args.split]
    if args.scenes:
        scenes = [s for s in scenes if s["name"] in set(args.scenes)]
    return run(scenes, args.workers)


if __name__ == "__main__":
    sys.exit(main())
