"""Step 1 diagnosis: WHERE DID THE NODES GO at +dx shifts.

No matcher runs.  Two front-end conventions are reported:
  NO-WRAP  (the historical known-transform rule): expected pos = p + (dx,dy),
            never reduced mod W; nodes whose expected pos leaves the frame
            are unmatchable by construction.
  WRAP     (the np.roll warp actually used): expected pos = (p + (dx,dy)) mod
            (W,H) -- every pixel exists in both views.

Per expected node (base graph with a continuous position):
  exists           >=1 same-lens detection within 1.5px (WRAP frame)
  same identity    >=1 same-lens detection within 1.5px whose FULL quantized
                   signature (class_, relation, quantized, population) matches
  cause (WRAP):
    legitimate     warp-overflow of the expected position (would leave frame)
                   -- counted, and excluded from the headline when noted
    gone           exists == 0
    split          >=2 same-lens detections within 1.5px
    merged         this node's single within-1.5px detection also claims
                   another expected node (many-to-one collapse)
    identity       exactly one within-1.5px detection, signature differs

Also reports: per-lens lost share at +2px; axes that flip identity
(class_, relation, each quantized axis, population); survival vs shift for
the 0..16 sweep with even/odd stride split; seam-band exclusion.
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
SHIFTS = (0, 1, 2, 3, 4, 5, 6, 8, 16)
EXACT = 1.5
TOL = 8.0


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene["name"] + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift != 0:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="loss_%s_%d" % (scene["name"], shift),
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
    lens = {n: g.sigs[n][0] for n in g.subgraph.nodes if n in g.sigs}
    sig = {n: (g.sigs[n][1].class_, g.sigs[n][1].relation,
               tuple(g.sigs[n][1].quantized),
               tuple(g.sigs[n][1].population))
           for n in g.sigs}
    return {"name": scene["name"], "shift": shift, "nodes": len(g.subgraph.nodes),
            "edges": len(g.subgraph.edges),
            "positions": positions, "lens": lens, "sig": sig}


def _worker(args):
    scene, shift = args
    t0 = time.perf_counter()
    out = _build(scene, shift)
    out["seconds"] = time.perf_counter() - t0
    return out


def _truth(a, b, shift, width, height, wrap):
    def expected_of(p):
        if wrap:
            return ((p[0] + shift) % width, p[1])
        return (p[0] + shift, p[1])

    expected = {u: expected_of(p) for u, p in a["positions"].items()}
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
    for d, u, v in cands:
        if u in uu or v in vv:
            continue
        uu.add(u); vv.add(v); truth[u] = v
    return truth, expected


def _axes_mismatch(sig_a, sig_b):
    ca, ra, qa, pa = sig_a
    cb, rb, qb, pb = sig_b
    axes = []
    if ca != cb:
        axes.append("class_")
    if ra != rb:
        axes.append("relation")
    qlen = max(len(qa), len(qb))
    for i in range(qlen):
        va = qa[i] if i < len(qa) else None
        vb = qb[i] if i < len(qb) else None
        if va != vb:
            axes.append("quantized[%d]" % i)
    if pa != pb:
        axes.append("population")
    return axes


def _analyze(scene, graphs, shift, exact=EXACT):
    rows = []
    a = graphs[0]
    width, height = 192, 192
    truth_w, exp_w = _truth(a, graphs[shift], shift, width, height, True)
    truth_nw, exp_nw = _truth(a, graphs[shift], shift, width, height, False)

    # frame-edge band (margin TOL) excluded variant
    def in_band(p, margin=TOL):
        return p[0] < margin or p[0] >= width - margin or p[1] < margin or p[1] >= height - margin

    expected_u = list(a["positions"])
    base_lens = a["lens"]

    # nearest-detection givers per detection (for merged)
    claimer = defaultdict(list)
    nearest_within = {}
    for u in expected_u:
        ex, ey = (a["positions"][u][0] + shift) % width, a["positions"][u][1]
        lu = base_lens.get(u)
        hits = [(math.hypot(px - ex, py - ey), v)
                 for v, (px, py) in graphs[shift]["positions"].items()
                 if graphs[shift]["lens"].get(v) == lu]
        hits = [h for h in hits if h[0] <= exact]
        hits.sort()
        nearest_within[u] = hits
        for d, v in hits:
            claimer[v].append(u)

    out = {
        "scene": scene["name"], "split": scene["split"], "shift": shift,
        "exists_window_px": exact,
        "expected_nodes": len(expected_u),
        "survival_wrap": len(truth_w) / max(1, len(exp_w)),
        "survival_nowrap": len(truth_nw) / max(1, len(exp_nw)),
        "survival_nowrap_ex_overflow": 0.0,
        "truth_wrap": len(truth_w), "truth_nowrap": len(truth_nw),
        "exists": 0, "same_identity": 0,
        "legit_wrap_overflow": 0, "wrap_recovered": 0,
        "gone": 0, "split": 0, "merged_victims": 0,
        "identity_changed": 0, "identity_same_but_lost": 0,
        "seam_band_nodes": sum(1 for u in expected_u if in_band(a["positions"][u])),
        "per_lens_lost": Counter(),
        "per_lens_expected": Counter(base_lens.get(u, "?") for u in expected_u),
        "axes_mismatch": Counter(),
        "split_examples": [], "merged_examples": [],
        "identity_changed_examples": [],
        "seconds": None,
    }
    for u in expected_u:
        lu = base_lens.get(u, "?")
        p = a["positions"][u]
        raw = (p[0] + shift, p[1])
        if raw[0] >= width or raw[0] < 0 or raw[1] >= height or raw[1] < 0:
            # out of frame under the NO-WRAP rule (the historical rule): a
            # legitimate loss for that metric.  The roll actually contains
            # the pixels at the opposite edge, so count how many of these the
            # WRAP rule recovers.
            out["legit_wrap_overflow"] += 1
            out["per_lens_lost"][lu] += 1
            exw, eyw = (p[0] + shift) % width, p[1]
            w_hits = w_hits if False else [(math.hypot(px - exw, py - eyw), v)
                      for v, (px, py) in graphs[shift]["positions"].items()
                      if graphs[shift]["lens"].get(v) == lu]
            w_hits = [h for h in w_hits if h[0] <= exact]
            if w_hits:
                out["wrap_recovered"] += 1
            continue
        ex, ey = raw
        hits = nearest_within[u]
        if not hits:
            out["gone"] += 1
            out["per_lens_lost"][lu] += 1
            continue
        if len(hits) >= 2:
            out["split"] += 1
            out["per_lens_lost"][lu] += 1
            if len(out["split_examples"]) < 6:
                out["split_examples"].append(
                    {"node": int(u), "lens": lu, "expected": [round(ex, 1), round(ey, 1)],
                     "detections": [(int(v), round(d, 3)) for d, v in hits]})
            continue
        det_v = hits[0][1]
        if len(claimer[det_v]) >= 2:
            out["merged_victims"] += 1
            out["per_lens_lost"][lu] += 1
            if len(out["merged_examples"]) < 6:
                out["merged_examples"].append(
                    {"node": int(u), "lens": lu,
                     "expected": [round(ex, 1), round(ey, 1)],
                     "claims": [int(x) for x in sorted(claimer[det_v])]})
            continue
        # exactly one detection within 1.5px; is it identity-identical?
        if a["sig"].get(u) == graphs[shift]["sig"].get(det_v):
            out["same_identity"] += 1
        else:
            out["identity_changed"] += 1
            axes = _axes_mismatch(a["sig"].get(u), graphs[shift]["sig"].get(det_v))
            for ax in axes:
                out["axes_mismatch"][ax] += 1
            if len(out["identity_changed_examples"]) < 8:
                out["identity_changed_examples"].append(
                    {"node": int(u), "lens": lu,
                     "sig_base": a["sig"].get(u),
                     "sig_shift": graphs[shift]["sig"].get(det_v),
                     "axes": axes})
    # POSITION-ERROR FORENSICS: for each expected node, where did its
    # same-lens detection land?  Distinguishes detection jitter from a
    # systematic stride/phase error.
    errors = []
    zero_x = exact_x = zero_y = exact_y = 0
    for u in expected_u:
        p = a["positions"][u]
        raw = (p[0] + shift, p[1])
        if raw[0] >= width or raw[0] < 0:
            continue
        lu = base_lens.get(u)
        bx, by = p
        dmin = TOL + 1
        best_v = None
        for v, (px, py) in graphs[shift]["positions"].items():
            if graphs[shift]["lens"].get(v) != lu:
                continue
            d = math.hypot(px - raw[0], py - raw[1])
            if d <= TOL and d < dmin:
                dmin, best_v = d, v
        if best_v is None:
            continue
        dxv = graphs[shift]["positions"][best_v][0] - bx
        dyv = graphs[shift]["positions"][best_v][1] - p[1]
        errors.append((dxv, dyv, dmin))
        zero_x += abs(dxv) < 0.5
        exact_x += abs(dxv - shift) < 0.5
        zero_y += abs(dyv) < 0.5
        exact_y += abs(dyv) < 0.5
    xs = [e[0] for e in errors]
    ys = [e[1] for e in errors]

    def _pctl(v, q):
        if not v:
            return float("nan")
        return float(np.percentile(np.asarray(v), q))

    hist_bins = [-1e9, -6, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1.5, 2, 2.5, 3, 4, 6, 1e9]
    hist = Counter()
    for x in xs:
        for i in range(len(hist_bins) - 1):
            if hist_bins[i] <= x < hist_bins[i + 1]:
                hist[i] += 1
                break
    out["pos_err_n"] = len(errors)
    out["pos_err_dx"] = {
        "dx_min": _pctl(xs, 0), "dx_p10": _pctl(xs, 10),
        "dx_med": _pctl(xs, 50), "dx_p90": _pctl(xs, 90), "dx_max": _pctl(xs, 100),
        "zero_x_frac": (zero_x / len(errors)) if errors else 0.0,
        "exact_shift_frac": (exact_x / len(errors)) if errors else 0.0,
    }
    out["pos_err_dy"] = {
        "zero_y_frac": (zero_y / len(errors)) if errors else 0.0,
        "exact_y_frac": (exact_y / len(errors)) if errors else 0.0,
    }
    out["pos_err_dx_hist"] = {str(i): hist.get(i, 0) for i in range(len(hist_bins) - 1)}

    out["exists"] = (len(expected_u) - out["gone"] - out["legit_wrap_overflow"])
    legit = out["legit_wrap_overflow"]
    out["survival_nowrap_ex_overflow"] = (len(truth_nw)
        / max(1, len(expected_u) - legit))
    lost_total = sum(v for k, v in out["per_lens_lost"].items())
    out["lost_total"] = lost_total
    out["saved_identity"] = out["same_identity"]
    out["per_lens_lost"] = dict(out["per_lens_lost"])
    out["per_lens_expected"] = dict(out["per_lens_expected"])
    out["axes_mismatch"] = dict(out["axes_mismatch"])
    out["truth_wrap_pairs"] = len(truth_w)
    # survival without seam band nodes
    band = {u for u in expected_u if in_band(a["positions"][u])}
    tfrac = sum(1 for u in truth_nw if u not in band)
    out["survival_nowrap_no_band"] = (tfrac / len([u for u in expected_u
                                                  if u not in band])) if expected_u else 0.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "loss_diagnosis.json"))
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
            for exact in (EXACT, 4.0):
                rows.append(_analyze(scene, graphs, shift, exact))
                r = rows[-1]
                print("%-22s %+2d win%.1f  exp %4d  survW %.3f survNW %.3f | exists %4d sameid %4d | gone %3d split %3d merged %3d ident %3d legit %3d wrec %3d" % (
                    r["scene"], r["shift"], r["exists_window_px"], r["expected_nodes"], r["survival_wrap"],
                    r["survival_nowrap"],
                    r["exists"], r["same_identity"],
                    r["gone"], r["split"], r["merged_victims"], r["identity_changed"],
                    r["legit_wrap_overflow"], r["wrap_recovered"]), flush=True)
    out = {"protocol": "loss diagnosis, full frame (object-local off the pipeline), wrap-aware + no-wrap truth",
           "shifts": list(SHIFTS), "exact_px": EXACT, "tolerance_px": TOL,
           "rows": rows}
    write_json(Path(args.out), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
