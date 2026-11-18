"""PART 0 — FIX THE RULER.  Diagnostic report (no fixes, no product edits).

The graphs are rebuilt transiently from the CURRENT tree (same builds as
STEP 1 / survival_table) because the renderer-truth scenes are not in the
topology cache; no product code is changed and rebuilds are deterministic.

Part 0 never runs a matcher.  It only uses memory.cores._sig_similarity as a
similarity FUNCTION against the true pairs and against random same-lens
null samples.

Definitions:
  expected node       any base-graph node with a continuous position
  same-lens counterpart  nearest same-lens detection within TOL=8px to
                      (pos + (shift mod W))   (wrap-aware, as the np.roll
                      warp demands)
  tracked              same-lens counterpart with |dx - shift| <= 1.5px
  static               same-lens counterpart with |dx| <= 1.5px
  neither              same-lens counterpart neither
  lost                 no same-lens counterpart within TOL
  identity (graded)    _sig_similarity above the pooled null's 95th percentile
  identity (bin)       signature equal EXCLUDING the population field
  identity (full)      signature equal (with population)
  population-only      everything equal except population
  true duplicate       0px: same-lens pair closer than 1px, graded >= .98,
                       signature equal ignoring population
  density neighbor     0px: pair closer than 4px, not a true duplicate
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

import matching.relation_frame as RF  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = (0, 2, 3)
TOL = 8.0
TRACK = 1.5


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image

    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    t0 = time.perf_counter()
    g = FE.build_graph(image, name="p0_%s_%d" % (scene["name"], shift),
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
    sigs = g.sigs  # n -> (lens, TypedSignature)
    lens = {n: s[0] for n, s in sigs.items()}
    sig_flat = {}
    sig_obj = {}
    for n, s in sigs.items():
        sig = s[1]
        sig_flat[n] = (sig.class_, sig.relation, tuple(sig.quantized),
                       tuple(sig.population))
        sig_obj[n] = sig
    return {"name": scene["name"], "split": scene.get("split"), "shift": shift,
            "nodes": len(g.subgraph.nodes), "edges": len(g.subgraph.edges),
            "positions": positions, "lens": lens,
            "sig": sig_flat, "sig_obj": sig_obj,
            "seconds": time.perf_counter() - t0}


def _worker(args):
    return _build(*args)


def _truth(a, b):
    """known-transform greedy, wrap-aware, same as before; returns {u: v}."""
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


def _nearest(a, b, u):
    p = a["positions"][u]
    shift = b["shift"]
    ex = (p[0] + shift) % 192.0
    best = None
    for v, (px, py) in b["positions"].items():
        if b["lens"].get(v) != a["lens"].get(u):
            continue
        d0 = math.hypot(px - ex, py - p[1])
        if d0 <= TOL and (best is None or d0 < best[0]):
            best = (d0, v)
    return best


def _percentiles(xs, ps):
    if not xs:
        return {str(p): float("nan") for p in ps}
    return {str(p): float(np.percentile(np.asarray(xs), p)) for p in ps}


def _run_scene_shift(base, other):
    """One (scene, shift) row."""
    from memory.cores import _sig_similarity
    import random
    rng = random.Random(1729)

    shift = other["shift"]
    truth = _truth(base, other)

    lens_stats = defaultdict(lambda: defaultdict(int))
    static_reported = {"mod8": Counter(), "mod16": Counter(),
                       "seam": 0, "identical_full": 0, "loc": 0,
                       "lenses": Counter()}
    # graded identity accumulation
    sim_true, max_null, abv_p95 = [], [], 0
    binid, fullid, poponly, binchg = [], [], [], []
    popid_frame = []
    pairs_done = 0

    WIDTH = HEIGHT = 192

    # tolerance sweep accumulator
    surv = {}
    for exact in (1.5, 2.5, 4.0):
        keep = 0
        for u, p in base["positions"].items():
            ex = (p[0] + shift) % WIDTH
            lu = base["lens"].get(u)
            for v, (px, py) in other["positions"].items():
                if other["lens"].get(v) != lu:
                    continue
                if math.hypot(px - ex, py - p[1]) <= exact:
                    keep += 1
                    break
        surv[exact] = keep / max(1, len(base["positions"]))

    for u in base["positions"]:
        lu = base["lens"].get(u, "?")
        st = lens_stats[lu]
        st["expected"] += 1
        best = _nearest(base, other, u)
        if best is None:
            st["gone"] += 1
            continue
        d0, v = best
        dxv = other["positions"][v][0] - base["positions"][u][0]
        st["detections"] += 1
        if abs(dxv - shift) <= TRACK:
            st["track"] += 1
        if abs(dxv) <= TRACK:
            st["static"] += 1
            if shift:
                x = other["positions"][v][0]
                y = other["positions"][v][1]
                static_reported["loc"] += 1
                static_reported["mod8"][int(round(x)) % 8] += 1
                static_reported["mod16"][int(round(x)) % 16] += 1
                if min(x, y, WIDTH - x, HEIGHT - y) <= TOL:
                    static_reported["seam"] += 1
                static_reported["lenses"][lu] += 1
                if base["sig"][u] == other["sig"][v]:
                    static_reported["identical_full"] += 1
        else:
            st["neither"] += 1

    # per-lens track/static/neither are subset sums of detections with gone separate
    lens_rows = []
    for lens, st in sorted(lens_stats.items(),
                           key=lambda kv: -kv[1].get("expected", 0)):
        lens_rows.append({
            "lens": lens,
            "split": "frame" if RF.is_frame(lens) else "attribute",
            "expected": st.get("expected", 0),
            "gone": st.get("gone", 0),
            "detections": st.get("detections", 0),
            "track": st.get("track", 0),
            "static": st.get("static", 0),
            "neither": st.get("neither", 0),
        })

    # graded identity over truth pairs
    by_lens = defaultdict(list)
    for v, sv in other["sig_obj"].items():
        by_lens[other["lens"][v]].append(v)

    for u, v in sorted(truth.items()):
        su = base["sig_obj"].get(u)
        sv = other["sig_obj"].get(v)
        if su is None or sv is None:
            continue
        pairs_done += 1
        score = _sig_similarity(su, sv)
        pool = by_lens.get(base["lens"].get(u), [])
        cand = [w for w in pool if w != v]
        rng.shuffle(cand)
        samp = cand[:10]
        null = [_sig_similarity(su, other["sig_obj"][w]) for w in samp]
        mx = max(null) if null else 0.0
        sim_true.append(score)
        max_null.append(mx)
        binid.append(1 if base["sig"][u][:3] == other["sig"][v][:3] else 0)
        fullid.append(1 if base["sig"][u] == other["sig"][v] else 0)
        poponly.append(1 if base["sig"][u][:3] == other["sig"][v][:3]
                       and base["sig"][u][3] != other["sig"][v][3] else 0)
        binchg.append(1 if base["sig"][u][:3] != other["sig"][v][:3] else 0)
        if RF.is_frame(base["lens"].get(u)):
            popid_frame.append(score)

    null_all = list(max_null)
    np95 = float(np.percentile(max_null, 95)) if max_null else 0.0
    pooled_p95 = np95
    graded = {
        "pairs": pairs_done,
        "mean_sim_true": float(np.mean(sim_true)) if sim_true else 0.0,
        "true_sim_min": float(np.min(sim_true)) if sim_true else 0.0,
        "true_sim_p50": _percentiles(sim_true, [50])["50"],
        "true_sim_p10": _percentiles(sim_true, [10])["10"],
        "null_max_median": float(np.median(null_all)) if null_all else 0.0,
        "null_max_p95": np95,
        "identity_above_null_p95": (sum(1 for x, mx in zip(sim_true, max_null) if x > mx)
                                    / max(1, len(sim_true))),
        "frac_above_pooled_null_p95": (sum(1 for x in sim_true if x > pooled_p95)
                                       / max(1, len(sim_true))),
        "bin_identity_frac": sum(binid) / max(1, len(binid)),
        "full_identity_frac": sum(fullid) / max(1, len(fullid)),
        "population_only_frac": sum(poponly) / max(1, len(poponly)),
        "bin_changed_frac": sum(binchg) / max(1, len(binchg)),
    }

    # frame-only graded identity
    f_t, f_mx = [], []
    for u, v in sorted(truth.items()):
        if not RF.is_frame(base["lens"].get(u)):
            continue
        su, sv = base["sig_obj"].get(u), other["sig_obj"].get(v)
        if su is None or sv is None:
            continue
        score = _sig_similarity(su, sv)
        pool = [w for w in by_lens.get(base["lens"].get(u), []) if w != v]
        rng.shuffle(pool)
        samp = pool[:10]
        null = [_sig_similarity(su, other["sig_obj"][w]) for w in samp]
        mx = max(null) if null else 0.0
        f_t.append(score)
        f_mx.append(mx)
    graded_frame = {
        "pairs": len(f_t),
        "mean_sim": float(np.mean(f_t)) if f_t else 0.0,
        "frac_above_null_p95": (sum(1 for x, mx in zip(f_t, f_mx) if x > mx)
                                / max(1, len(f_t))),
        "bin_identity_frac": float(np.mean([1.0 if base["sig"][u][:3] == other["sig"][truth[u]][:3] else 0.0
                                            for u in truth if RF.is_frame(base["lens"].get(u))]))
        if len(f_t) else 0.0,
    }

    # ---------- (e) true duplicates at 0px ----------
    dup_report = {"true_dups": [], "neighbors": []}
    if shift == 0:
        nodes = sorted(base["positions"])
        from memory.cores import _sig_similarity
        for i, u in enumerate(nodes):
            su = base["sig_obj"].get(u)
            for v in nodes[i + 1:]:
                if abs(base["positions"][u][0] - base["positions"][v][0]) >= 4.0:
                    continue
                d = math.hypot(base["positions"][u][0] - base["positions"][v][0],
                               base["positions"][u][1] - base["positions"][v][1])
                if d > 4.0:
                    continue
                if base["lens"].get(u) != base["lens"].get(v):
                    continue
                sv = base["sig_obj"].get(v)
                if su is None or sv is None:
                    continue
                sim_dup = _sig_similarity(su, sv)
                q_eq = base["sig"][u][:3] == base["sig"][v][:3]
                is_dup = d < 1.0 and sim_dup >= 0.98 and q_eq
                if is_dup:
                    dup_report["true_dups"].append({"a": int(u), "b": int(v),
                                                    "lens": base["lens"][u],
                                                    "d": round(d, 4),
                                                    "graded": round(sim_dup, 4)})
                else:
                    dup_report["neighbors"].append({"a": int(u), "b": int(v),
                                                    "lens": base["lens"][u],
                                                    "d": round(d, 4)})
    dup_count = Counter(x["lens"] for x in dup_report["true_dups"])
    dup_objects = {
        "true_dups_total": len(dup_report["true_dups"]),
        "density_neighbors_total": len(dup_report["neighbors"]),
        "true_dups_per_lens": dict(dup_count),
        # representative pairs for the report (first 12)
        "examples": dup_report["true_dups"][:12],
    }

    return {
        "scene": base["name"], "split": base["split"], "shift": shift,
        "expected": len(base["positions"]),
        "truth_pairs": len(truth),
        "nodes": other["nodes"], "edges": other["edges"],
        "survival_1_5": surv[1.5], "survival_2_5": surv[2.5],
        "survival_4_0": surv[4.0],
        "lens_report": lens_rows,
        "static": {
            "tracked_static_count": static_reported["loc"],
            "seam_band_statics": static_reported["seam"],
            "static_identical_full_count": static_reported["identical_full"],
            "static_lenses": dict(static_reported["lenses"]),
            "mod8": dict(static_reported["mod8"]),
            "mod16": dict(static_reported["mod16"]),
        },
        "graded_identity_all": graded,
        "graded_identity_frame": graded_frame,
        "duplicates": dup_objects if shift == 0 else {},
        "seconds": base.get("seconds"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "part0_ruler.json"))
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
    for scene in scenes:
        graphs = {s: by[(scene["name"], s)] for s in SHIFTS}
        for shift in SHIFTS:
            row = _run_scene_shift(graphs[0], graphs[shift])
            rows.append(row)
            print("%-20s %+d  exp %4d truth %4d | exists(1.5) %.3f (2.5) %.3f (4.0) %.3f | "
                  "statics %4d | all-id>null %.3f bin %.3f full %.3f pop-only %.3f | frame-id %.3f" % (
                      scene["name"], shift, row["expected"], row["truth_pairs"],
                      row["survival_1_5"], row["survival_2_5"], row["survival_4_0"],
                      row["static"]["tracked_static_count"],
                      row["graded_identity_all"]["identity_above_null_p95"],
                      row["graded_identity_all"]["bin_identity_frac"],
                      row["graded_identity_all"]["full_identity_frac"],
                      row["graded_identity_all"]["population_only_frac"],
                      row["graded_identity_frame"]["frac_above_null_p95"]), flush=True)

    write_json(Path(args.out), {"protocol": "part0 ruler check, full frame",
                                "shifts": list(SHIFTS), "rows": rows})
    return 0


if __name__ == "__main__":
    sys.exit(main())
