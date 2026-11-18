"""STEP A8 — the selected graph the matcher sees, with attachments.

For each renderer scene at 0/+2/+3:
  * situation-window graph (eth80_frontend.build_graph) and the SELECTED
    graph (matching.perceptual_selection.select_perceptual_graph, budget
    512) actually used by the classification harness;
  * node/edge census before/after selection;
  * graded + bin identity on the selected graph (same truth rule);
  * per-feature-name attachment identity: for each frame node, the
    attribute features attached by CO_LOCATED edges; the feature's
    cross-view graded similarity vs the host's, per feature lens;
  * frame bin identity split by whether the frame node carries a
    sheet-derived feature.
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
from matching.perceptual_selection import select_perceptual_graph  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = (0, 2, 3)
TOL = 8.0
BUDGET = 512

SHEET_LENSES = {"Boundary", "Color", "Motion", "Size", "Texture", "Position",
                "Onset", "Orientation", "EffectPresent", "Polarity",
                "OrientationPol"}


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="a8_%s_%d" % (scene["name"], shift),
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
    return {"name": scene["name"], "shift": shift, "graph": g,
            "positions": positions,
            "lens": {n: s[0] for n, s in g.sigs.items()},
            "sig_obj": {n: s[1] for n, s in g.sigs.items()}}


def _worker(args):
    return _build(*args)


def _select(graph_obj, sigs):
    sg = graph_obj.subgraph
    weights = getattr(sg, "weights", None)
    if weights is None:
        class _G:
            pass
        g2 = _G()
        g2.nodes = sg.nodes
        g2.edges = sg.edges
        g2.weights = {}
        sg = g2
    sel, diag = select_perceptual_graph(sg, lambda n: sigs.get(n),
                                        budget=BUDGET)
    return sel, diag


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


def _analyze(scene, graphs):
    from memory.cores import _sig_similarity
    base = graphs[0]
    out = []
    for shift in SHIFTS:
        g = graphs[shift]
        sel_b, diag_b = _select(base["graph"], base["graph"].sigs)
        sel_s, diag_s = _select(g["graph"], g["graph"].sigs)
        # truth on the SELECTED node set (positions still available)
        pos_b = {n: base["positions"][n] for n in sel_b.nodes if n in base["positions"]}
        pos_s = {n: g["positions"][n] for n in sel_s.nodes if n in g["positions"]}
        lens_b = {n: base["lens"][n] for n in sel_b.nodes if n in base["lens"]}
        lens_s = {n: g["lens"][n] for n in sel_s.nodes if n in g["lens"]}
        # remap truth onto selected ids (they are the same node ids)
        sub = {"name": "b", "shift": 0, "positions": pos_b, "lens": lens_b}
        sub2 = {"name": "s", "shift": shift, "positions": pos_s, "lens": lens_s}
        # truth computed with width 192 wrap
        cands = []
        for u, p in pos_b.items():
            lu = lens_b.get(u)
            ex = (p[0] + shift) % 192.0
            for v, (px, py) in pos_s.items():
                if lens_s.get(v) != lu:
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
        # graded identity on selected
        sims, bins, fulls = [], [], []
        for u, v in truth.items():
            su = base["sig_obj"].get(u)
            sv = g["sig_obj"].get(v)
            if su is None or sv is None:
                continue
            sims.append(_sig_similarity(su, sv))
            cb, rb, qb, pb = (su.class_, su.relation, tuple(su.quantized), tuple(su.population))
            cs, rs, qs, ps = (sv.class_, sv.relation, tuple(sv.quantized), tuple(sv.population))
            bins.append(1 if (cb, rb, qb) == (cs, rs, qs) else 0)
            fulls.append(1 if (cb, rb, qb, pb) == (cs, rs, qs, ps) else 0)
        # attachments: frame node -> CO_LOCATED attribute features
        attach = defaultdict(list)
        for (s, d, et) in sel_b.edges:
            if str(et).endswith("CO_LOCATED") or et.name == "CO_LOCATED":
                for host, feat in ((s, d), (d, s)):
                    if RF.is_frame(lens_b.get(host, "")) and not RF.is_frame(lens_b.get(feat, "")):
                        attach[host].append(feat)
        feat_stats = defaultdict(lambda: {"n": 0, "sim": [], "host_sim": []})
        for host, feats in attach.items():
            for f in feats:
                fl = lens_b.get(f, "?")
                feat_stats[fl]["n"] += 1
                # feature cross-view similarity: nearest same-lens in shifted
                best = None
                ex = (pos_b[host][0] + shift) % 192.0
                for v, (px, py) in pos_s.items():
                    if lens_s.get(v) != fl:
                        continue
                    d = math.hypot(px - pos_b[f][0] - shift if False else px - ((pos_b[f][0] + shift) % 192.0),
                                   py - pos_b[f][1])
                    if d <= TOL and (best is None or d < best[0]):
                        best = (d, v)
                if best is not None:
                    sv = g["sig_obj"].get(best[1])
                    sf = base["sig_obj"].get(f)
                    if sv is not None and sf is not None:
                        feat_stats[fl]["sim"].append(_sig_similarity(sf, sv))
                if host in truth:
                    sv = g["sig_obj"].get(truth[host])
                    sh = base["sig_obj"].get(host)
                    if sv is not None and sh is not None:
                        feat_stats[fl]["host_sim"].append(_sig_similarity(sh, sv))
        # frame bin identity with/without sheet-derived features
        fr_with, fr_without = [], []
        for u, v in truth.items():
            if not RF.is_frame(lens_b.get(u, "")):
                continue
            su = base["sig_obj"].get(u)
            sv = g["sig_obj"].get(v)
            if su is None or sv is None:
                continue
            same = (su.class_, su.relation, tuple(su.quantized)) == \
                   (sv.class_, sv.relation, tuple(sv.quantized))
            has_sheet_feat = any(lens_b.get(f, "") in SHEET_LENSES for f in attach.get(u, ()))
            (fr_with if has_sheet_feat else fr_without).append(1 if same else 0)
        out.append({
            "shift": shift,
            "situation_nodes": len(base["graph"].subgraph.nodes),
            "situation_edges": len(base["graph"].subgraph.edges),
            "selected_nodes": len(sel_b.nodes),
            "selected_edges": len(sel_b.edges),
            "selection_diag": {k: v for k, v in (diag_b or {}).items()
                               if isinstance(v, (int, float))},
            "truth_pairs_selected": len(truth),
            "graded_mean_selected": float(np.mean(sims)) if sims else 0.0,
            "bin_identity_selected": float(np.mean(bins)) if bins else 0.0,
            "full_identity_selected": float(np.mean(fulls)) if fulls else 0.0,
            "attachments": {k: {"n": v["n"],
                                "feature_sim_mean": float(np.mean(v["sim"])) if v["sim"] else 0.0,
                                "host_sim_mean": float(np.mean(v["host_sim"])) if v["host_sim"] else 0.0}
                            for k, v in feat_stats.items()},
            "frame_bin_with_sheet_feature": float(np.mean(fr_with)) if fr_with else None,
            "frame_bin_without_sheet_feature": float(np.mean(fr_without)) if fr_without else None,
            "frame_nodes_with_sheet_feature": len(fr_with),
            "frame_nodes_without": len(fr_without),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "step_a8_selected.json"))
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
            print("%-20s %+d sit %d/%d sel %d/%d truth %d bin %.3f graded %.3f | fr+sheet %.3f (%d) fr-sheet %.3f (%d)" % (
                s["name"], rr["shift"], rr["situation_nodes"], rr["situation_edges"],
                rr["selected_nodes"], rr["selected_edges"], rr["truth_pairs_selected"],
                rr["bin_identity_selected"], rr["graded_mean_selected"],
                rr["frame_bin_with_sheet_feature"] or -1, rr["frame_nodes_with_sheet_feature"],
                rr["frame_bin_without_sheet_feature"] or -1, rr["frame_nodes_without"]),
                flush=True)
    write_json(Path(args.out), {"protocol": "A8 selected graph with attachments",
                                "budget": BUDGET, "rows": rows})
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
