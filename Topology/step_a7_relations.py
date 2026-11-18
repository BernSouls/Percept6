"""STEP A7 — the decisive split: relation preservation by stream and by
endpoint type, and a node-free relation-family histogram distance.

Builds the 15 renderer scenes at 0/+2/+3 (transient, current tree).

For each base edge (s, d, et) under the wrap-aware greedy truth mapping:
  endpoint streams (sheet / content / attribute_higher / unregistered)
  preserved = the mapped endpoints form an edge of the same type in the
  shifted graph (undirected types canonicalised)

Node-free family check: for each edge type, the multiset of quantized
parameter tuples in the base graph vs the shifted graph (no mapping), as a
total-variation histogram distance; and the same distance between two
DIFFERENT scenes (the null).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = (0, 2, 3)
TOL = 8.0

SHEET_LENSES = {"Boundary", "Color", "Motion", "Size", "Texture", "Position",
                "Onset", "Orientation", "EffectPresent", "Polarity",
                "OrientationPol"}
CONTENT_LENSES = {"CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
                  "RegionShape", "Corner", "Contour", "CurveComplex",
                  "IllusoryContour", "Junction", "T", "Y"}
HIGHER_ATTR = {"BorderOwnership", "TextureBoundary", "Surface"}


def _stream(lens):
    if lens in SHEET_LENSES:
        return "sheet"
    if lens in CONTENT_LENSES:
        return "content"
    if lens in HIGHER_ATTR:
        return "attribute_higher"
    return "unregistered"


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="a7_%s_%d" % (scene["name"], shift),
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
    lens = {n: s[0] for n, s in g.sigs.items()}
    qparams = {}
    for k, v in (g.subgraph.edge_params or {}).items():
        qparams[k] = tuple(v.quantized) if getattr(v, "quantized", None) else ()
    return {"name": scene["name"], "shift": shift,
            "positions": positions, "lens": lens,
            "edges": list(g.subgraph.edges),
            "qparams": qparams}


def _worker(args):
    return _build(*args)


def _truth(a, b):
    shift = b["shift"]
    cands = []
    for u, p in a["positions"].items():
        lu = a["lens"].get(u)
        ex = (p[0] + shift) % 192.0
        for v, (px, py) in b["positions"].items():
            if b["lens"].get(v) != lu:
                continue
            d = math.hypot(px - ex, py - p[1])
            if d <= TOL:
                cands.append((d, u, v))
    cands.sort()
    uu, vv, truth = set(), set(), {}
    for d, u, v in cands:
        if u in uu or v in vv:
            continue
        uu.add(u); vv.add(v); truth[u] = v
    return truth


def _edge_key(e):
    s, d, et = e
    return (min(s, d), max(s, d), et)


def _family_hist(graph):
    h = defaultdict(Counter)
    for (s, d, et) in graph["edges"]:
        q = graph["qparams"].get((s, d, et))
        if q is None:
            q = graph["qparams"].get((d, s, et))
        h[et][q if q is not None else ()] += 1
    return h


def _tv(h1, h2):
    keys = set(h1) | set(h2)
    n1 = sum(h1.values()) or 1
    n2 = sum(h2.values()) or 1
    return 0.5 * sum(abs(h1.get(k, 0) / n1 - h2.get(k, 0) / n2) for k in keys)


def _analyze(scene, graphs):
    base = graphs[0]
    out = []
    for shift in SHIFTS:
        g = graphs[shift]
        truth = _truth(base, g)
        rev = {v: u for u, v in truth.items()}
        other_edges = {_edge_key(e) for e in g["edges"]}
        pair_counts = Counter()
        pair_pres = Counter()
        fam_counts = Counter()
        fam_pres = Counter()
        for e in base["edges"]:
            s, d, et = e
            st_s = _stream(base["lens"].get(s, "?"))
            st_d = _stream(base["lens"].get(d, "?"))
            pair = "-".join(sorted([st_s, st_d]))
            pair_counts[pair] += 1
            fam_counts[et] += 1
            u, v = truth.get(s), truth.get(d)
            if u is None or v is None:
                continue
            ok = _edge_key((u, v, et)) in other_edges
            if ok:
                pair_pres[pair] += 1
                fam_pres[et] += 1
        hb = _family_hist(base)
        hs = _family_hist(g)
        fam_tv = {str(k): _tv(hb.get(k, Counter()), hs.get(k, Counter()))
                  for k in set(hb) | set(hs)}
        out.append({
            "shift": shift,
            "truth_pairs": len(truth),
            "endpoint_pairs": {k: {"n": pair_counts[k],
                                   "preserved": pair_pres[k],
                                   "frac": pair_pres[k] / pair_counts[k]
                                   if pair_counts[k] else 0.0}
                               for k in sorted(pair_counts)},
            "families": {str(k): {"n": fam_counts[k],
                                  "preserved": fam_pres[k],
                                  "frac": fam_pres[k] / fam_counts[k]
                                  if fam_counts[k] else 0.0,
                                  "tv_same_scene": fam_tv.get(str(k))}
                          for k in sorted(fam_counts, key=str)},
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "step_a7_relations.json"))
    args = ap.parse_args(argv)
    data = json.loads(TRUTH_PATH.read_text())
    scenes = data["scenes"]
    if args.split != "all":
        scenes = [s for s in scenes if s["split"] == args.split]
    if args.scenes:
        scenes = [s for s in scenes if s["name"] in set(args.scenes)]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks = [(s, shift) for s in scenes for shift in SHIFTS]
    with ctx.Pool(args.workers, initializer=guard.activate) as pool:
        built = pool.map(_worker, tasks, chunksize=1)
    by = {(b["name"], b["shift"]): b for b in built}

    rows = []
    for s in scenes:
        graphs = {sh: by[(s["name"], sh)] for sh in SHIFTS}
        r = _analyze(s, graphs)
        rows.append({"scene": s["name"], "rows": r})
        for rr in r:
            ep = rr["endpoint_pairs"]
            cc = ep.get("content-content", {})
            ss = ep.get("sheet-sheet", {})
            mix = ep.get("content-sheet", {})
            print("%-20s %+d truth %4d | content-content %d/%d %.3f | sheet-sheet %d/%d %.3f | mixed %d/%d %.3f" % (
                s["name"], rr["shift"], rr["truth_pairs"],
                cc.get("preserved", 0), cc.get("n", 0), cc.get("frac", 0.0),
                ss.get("preserved", 0), ss.get("n", 0), ss.get("frac", 0.0),
                mix.get("preserved", 0), mix.get("n", 0), mix.get("frac", 0.0)),
                flush=True)

    # cross-scene null for the family TV
    nulls = {}
    if len(scenes) >= 2:
        a = by[(scenes[0]["name"], 0)]
        b = by[(scenes[1]["name"], 0)]
        ha, hb = _family_hist(a), _family_hist(b)
        nulls = {str(k): _tv(ha.get(k, Counter()), hb.get(k, Counter()))
                 for k in set(ha) | set(hb)}
    out = {"protocol": "A7 relation preservation by stream/endpoint type, renderer-truth-v1",
           "rows": rows, "cross_scene_family_tv_null": nulls}
    write_json(Path(args.out), out)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
