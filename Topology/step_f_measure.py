"""Step F measurement: graded/bin identity with both flags off, each on, BOTH on.

Same graphs as STEP A/PART 0 (transient rebuilds, current tree).
Reports, per scene/shift and aggregated:
  - frame graded identity vs truth pairs (mean sim, frac above pooled null p95)
  - bin identity (population excluded), full identity
  - population-only change rate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import matching.relation_frame as RF  # noqa: E402
import parameters as P  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = (0, 2, 3)
TOL = 8.0


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="f_%s_%d" % (scene["name"], shift),
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
    sigs = g.sigs
    lens = {n: s[0] for n, s in sigs.items()}
    sig_flat = {n: (s[1].class_, s[1].relation, tuple(s[1].quantized),
                    tuple(s[1].population)) for n, s in sigs.items()}
    sig_obj = {n: s[1] for n, s in sigs.items()}
    return {"name": scene["name"], "shift": shift,
            "positions": positions, "lens": lens,
            "sig": sig_flat, "sig_obj": sig_obj,
            "nodes": len(g.subgraph.nodes)}


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


def _eval(base, other, flags):
    from memory.cores import _sig_similarity
    for k, v in flags.items():
        setattr(P, k, v)
    truth = _truth(base, other)
    sims, bins, fulls, poponly, bins_only = [], [], [], [], []
    for u, v in sorted(truth.items()):
        if not RF.is_frame(base["lens"].get(u)):
            continue
        su = base["sig_obj"].get(u)
        sv = other["sig_obj"].get(v)
        if su is None or sv is None:
            continue
        sims.append(_sig_similarity(su, sv))
        qa = (base["sig"][u][0], base["sig"][u][1], base["sig"][u][2])
        qb = (other["sig"][v][0], other["sig"][v][1], other["sig"][v][2])
        bins.append(1 if qa == qb else 0)
        fulls.append(1 if base["sig"][u] == other["sig"][v] else 0)
        poponly.append(1 if qa == qb and base["sig"][u][3] != other["sig"][v][3] else 0)
    n = max(1, len(sims))
    return {
        "pairs": len(sims),
        "frame_graded_mean": float(np.mean(sims)) if sims else 0.0,
        "frame_graded_min": float(np.min(sims)) if sims else 0.0,
        "bin_identity": sum(bins) / n,
        "full_identity": sum(fulls) / n,
        "population_only": sum(poponly) / n,
    }


CONFIGS = {
    "off": {"MATCH_NEIGHBOR_BIN_CREDIT": False, "MATCH_POPULATION_EMD": False},
    "neighbor": {"MATCH_NEIGHBOR_BIN_CREDIT": True, "MATCH_POPULATION_EMD": False},
    "emd": {"MATCH_NEIGHBOR_BIN_CREDIT": False, "MATCH_POPULATION_EMD": True},
    "both": {"MATCH_NEIGHBOR_BIN_CREDIT": True, "MATCH_POPULATION_EMD": True},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "step_f_measure.json"))
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
    saved = {k: getattr(P, k) for k in ("MATCH_NEIGHBOR_BIN_CREDIT", "MATCH_POPULATION_EMD")}
    try:
        for scene in scenes:
            graphs = {s: by[(scene["name"], s)] for s in SHIFTS}
            for shift in SHIFTS:
                entry = {"scene": scene["name"], "split": scene["split"], "shift": shift}
                for name, flags in CONFIGS.items():
                    entry[name] = _eval(graphs[0], graphs[shift], flags)
                rows.append(entry)
                off, both = entry["off"], entry["both"]
                print("%-20s %+d  pairs %3d | graded off %.3f both %.3f | bin off %.3f both %.3f | full off %.3f both %.3f" % (
                    scene["name"], shift, off["pairs"],
                    off["frame_graded_mean"], both["frame_graded_mean"],
                    off["bin_identity"], both["bin_identity"],
                    off["full_identity"], both["full_identity"]), flush=True)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
    write_json(Path(args.out), {"protocol": "Step F flag effect, frame pairs, renderer-truth-v1",
                                "rows": rows})
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
