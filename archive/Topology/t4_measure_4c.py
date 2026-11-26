"""Task 4 PART 4c: PRIMARY ORGANIZATION TEST.

Node correspondence on held-out at shifts 0, ±2, ±3, production joint_mcs.
Shifts are cyclic np.roll of the whole image (translation, not object motion).

Counterpart = nearest shifted-graph node of same kind within 2 px of the
expected shifted position (primary). A node is correct iff the matcher maps
it to its counterpart.

Denominators:
  D1 = all base nodes of token kinds present in every arm; no-counterpart
       counted incorrect.
  D2 = nodes that have a counterpart (counterpart rate per arm).
  D3 = F's CONTOUR_PEAK event tokens, reported separately.

Arms B0, B1, F at all shifts.  Controls (real, real-params-dropped,
rewired, params-shuffled, no-edge) and seeded controls at shift +2 (the
primary contrast shift) and shift 0 (the 0px=1.000 check).

Contrasts (paired intervals, resampled BY IMAGE; seeded modes per seed then
averaged): headline real-rewired; topology real-params-dropped-rewired;
parameter presence real-real-params-dropped; parameter correctness
real-params-shuffled.

Usage:
    python t4_measure_4c.py --best-arm F
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

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
SHIFTS = (0, 2, -2, 3, -3)
TOL_PRIMARY = 2.0
HOLDOUT_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _counterpart_map(base_pos, shift_pos, shift_px, tol):
    by_lens = defaultdict(list)
    for n, (x, y, l) in shift_pos.items():
        by_lens[l].append((n, x, y))
    out = {}
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = x0 + shift_px, y0
        best_n, best_d = None, tol
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            d = math.hypot(ex - xsh, ey - ysh)
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is not None:
            out[n] = best_n
    return out


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _match_accuracy(base_pos, mapping, counterpart, token_kinds):
    total = correct = no_c = 0
    for n, (x, y, l) in base_pos.items():
        if l not in token_kinds:
            continue
        total += 1
        cp = counterpart.get(n)
        if cp is None:
            no_c += 1
            continue
        if mapping.get(n) == cp:
            correct += 1
    return {"total": total, "correct": correct, "no_counterpart": no_c,
            "acc": correct / total if total else 0.0}


def _paired_interval(a_vals, b_vals):
    n = len(a_vals)
    if n == 0:
        return None
    diffs = np.array(a_vals) - np.array(b_vals)
    mean = float(diffs.mean())
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    return {"mean": mean,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975))}


def _rewire_edges(edges, seed):
    from core.types import UNDIRECTED_EDGE_TYPES
    rng = random.Random(seed)
    und = [(min(a, b), max(a, b), t) for a, b, t in edges
           if t in UNDIRECTED_EDGE_TYPES]
    direct = [(a, b, t) for a, b, t in edges if t not in UNDIRECTED_EDGE_TYPES]
    out = set(und)
    for _ in range(20 * max(1, len(und))):
        if len(und) < 2:
            break
        i, j = rng.sample(range(len(und)), 2)
        a, b, t1 = und[i]
        c, d, t2 = und[j]
        if t1 != t2 or a == b or c == d:
            continue
        cand = [(a, d, t1), (c, b, t2)]
        if any(x == y for x, y, _ in cand):
            continue
        if any((min(x, y), max(x, y), t) in out and
               (min(x, y), max(x, y), t) not in ((min(a, b), max(a, b), t1),
                                                 (min(c, d), max(c, d), t2))
               for x, y, t in cand):
            continue
        before = set(out)
        out.discard((a, b, t1))
        out.discard((c, d, t2))
        out.update(cand)
        if len(out) != len(before):
            out = before
            continue
        und[i] = cand[0]
        und[j] = cand[1]
    return out | set(direct)


def _rewired_base(base_g, seed):
    new_edges = _rewire_edges(base_g.subgraph.edges, seed)
    sg = replace(base_g.subgraph, edges=frozenset(new_edges),
                 edge_params={})
    return replace(base_g, subgraph=sg)


def _params_shuffled_base(base_g, seed):
    rng = random.Random(seed)
    ep = dict(base_g.subgraph.edge_params)
    by_fam = defaultdict(list)
    for k in ep:
        by_fam[str(getattr(k[2], "value", k[2]))].append(k)
    new_ep = {}
    for fam, keys in by_fam.items():
        vals = [ep[k] for k in keys]
        rng.shuffle(vals)
        for k, v in zip(keys, vals):
            new_ep[k] = v
    sg = replace(base_g.subgraph, edge_params=new_ep)
    return replace(base_g, subgraph=sg)


def _params_dropped_base(base_g):
    sg = replace(base_g.subgraph, edge_params={})
    return replace(base_g, subgraph=sg)


def _no_edge_base(base_g):
    sg = replace(base_g.subgraph, edges=frozenset(), edge_params={})
    return replace(base_g, subgraph=sg)


def _load_all():
    """Load every held-out graph into memory, keyed by (arm, name, shift)."""
    graphs = {}
    for p in HOLDOUT_DIR.glob("*.pkl"):
        parts = p.stem.rsplit("_", 1)
        arm_name = parts[0]
        shift = int(parts[1])
        arm, name = arm_name.split("_", 1)
        with open(p, "rb") as fh:
            graphs[(arm, name, shift)] = pickle.load(fh)
    return graphs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-arm", default="F")
    args = ap.parse_args()
    best = args.best_arm

    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)

    print("loading graphs...", flush=True)
    t0 = time.perf_counter()
    graphs = _load_all()
    print("  loaded %d graphs (%.0fs)" % (len(graphs), time.perf_counter() - t0),
          flush=True)

    img_names = sorted({n for (_a, n, _s) in graphs})
    arms = ("B0", "B1", "F")

    # token kinds from B0's nodes (present in every arm)
    b0_pos = _pos_lens(graphs[("B0", img_names[0], 0)])
    token_kinds = {l for (_x, _y, l) in b0_pos.values()}
    print("token kinds:", sorted(token_kinds), flush=True)

    # ---- main arms: all shifts ----
    print("main arms...", flush=True)
    t0 = time.perf_counter()
    rows = {}
    for arm in arms:
        for name in img_names:
            base = graphs[(arm, name, 0)]
            base_pos = _pos_lens(base)
            for shift in SHIFTS:
                sh_g = graphs[(arm, name, shift)]
                sh_pos = _pos_lens(sh_g)
                counterpart = _counterpart_map(base_pos, sh_pos, shift,
                                               TOL_PRIMARY)
                mapping = _run_matcher(base, sh_g)
                d1 = _match_accuracy(base_pos, mapping, counterpart,
                                     token_kinds)
                have_c = sum(1 for n in base_pos if n in counterpart)
                d2 = have_c / len(base_pos) if base_pos else 0.0
                d3 = None
                if arm == "F":
                    cp_nodes = [n for n, (x, y, l) in base_pos.items()
                                if l == "CONTOUR_PEAK"]
                    if cp_nodes:
                        cp_c = sum(1 for n in cp_nodes
                                   if n in counterpart
                                   and mapping.get(n) == counterpart.get(n))
                        d3 = {"total": len(cp_nodes), "correct": cp_c,
                              "acc": cp_c / len(cp_nodes)}
                rows[(arm, name, shift)] = {
                    "d1": d1, "d2": d2, "d3": d3, "n_base": len(base_pos)}
        print("  %s done (%.0fs)" % (arm, time.perf_counter() - t0), flush=True)

    # ---- aggregate per arm/shift ----
    agg = {}
    for arm in arms:
        agg[arm] = {}
        for shift in SHIFTS:
            accs = [rows[(arm, n, shift)]["d1"]["acc"] for n in img_names]
            d2s = [rows[(arm, n, shift)]["d2"] for n in img_names]
            agg[arm][str(shift)] = {
                "d1_acc_mean": float(np.mean(accs)),
                "d1_acc_per_image": accs,
                "d2_rate_mean": float(np.mean(d2s)),
            }
    d3s = [rows[("F", n, 2)]["d3"]["acc"] for n in img_names
           if rows[("F", n, 2)]["d3"]]
    agg["F"]["d3_acc_mean_shift2"] = float(np.mean(d3s)) if d3s else None

    # ---- controls at shift +2 and 0 ----
    print("controls...", flush=True)
    t0 = time.perf_counter()
    ctrl_shifts = (0, 2)
    controls = {}
    for mode in ("real", "real_params_dropped", "rewired",
                 "params_shuffled", "no_edge"):
        controls[mode] = {}
        for shift in ctrl_shifts:
            accs = []
            for name in img_names:
                base = graphs[(best, name, 0)]
                sh_g = graphs[(best, name, shift)]
                base_pos = _pos_lens(base)
                counterpart = _counterpart_map(base_pos, _pos_lens(sh_g),
                                               shift, TOL_PRIMARY)
                if mode == "real":
                    b = base
                elif mode == "real_params_dropped":
                    b = _params_dropped_base(base)
                elif mode == "rewired":
                    b = _rewired_base(base, 0)
                elif mode == "params_shuffled":
                    b = _params_shuffled_base(base, 0)
                elif mode == "no_edge":
                    b = _no_edge_base(base)
                mapping = _run_matcher(b, sh_g)
                d1 = _match_accuracy(base_pos, mapping, counterpart,
                                     token_kinds)
                accs.append(d1["acc"])
            controls[mode][str(shift)] = {
                "d1_acc_mean": float(np.mean(accs)),
                "d1_acc_per_image": accs}
    print("  controls done (%.0fs)" % (time.perf_counter() - t0), flush=True)

    # ---- seeded controls at shift +2 ----
    print("seeded controls...", flush=True)
    t0 = time.perf_counter()
    shift = 2

    def _seeded_per_image(mode):
        per_img_seeds = [[] for _ in img_names]
        for seed in (0, 1, 2):
            for i, name in enumerate(img_names):
                base = graphs[(best, name, 0)]
                sh_g = graphs[(best, name, shift)]
                base_pos = _pos_lens(base)
                counterpart = _counterpart_map(base_pos, _pos_lens(sh_g),
                                               shift, TOL_PRIMARY)
                if mode == "rewired":
                    b = _rewired_base(base, seed)
                else:
                    b = _params_shuffled_base(base, seed)
                mapping = _run_matcher(b, sh_g)
                d1 = _match_accuracy(base_pos, mapping, counterpart,
                                     token_kinds)
                per_img_seeds[i].append(d1["acc"])
        return [float(np.mean(v)) for v in per_img_seeds]

    rewired_per_img = _seeded_per_image("rewired")
    rpd_per_img = _seeded_per_image("params_shuffled")
    print("  seeded done (%.0fs)" % (time.perf_counter() - t0), flush=True)

    # ---- contrasts at shift +2 ----
    real_vals = controls["real"][str(shift)]["d1_acc_per_image"]
    real_pd_vals = controls["real_params_dropped"][str(shift)]["d1_acc_per_image"]
    contrasts = {
        "headline_real_minus_rewired": _paired_interval(real_vals,
                                                        rewired_per_img),
        "topology_realpd_minus_rewired": _paired_interval(real_pd_vals,
                                                          rewired_per_img),
        "param_presence_real_minus_realpd": _paired_interval(real_vals,
                                                             real_pd_vals),
        "param_correct_real_minus_shuffled": _paired_interval(real_vals,
                                                             rpd_per_img),
    }

    out = {
        "best_arm": best, "winner": WINNER, "shifts": list(SHIFTS),
        "tol_primary_px": TOL_PRIMARY, "token_kinds": sorted(token_kinds),
        "images": img_names, "arms_agg": agg, "controls": controls,
        "seeded_per_image_shift2": {"rewired": rewired_per_img,
                                    "params_shuffled": rpd_per_img},
        "contrasts_shift2": contrasts,
    }
    write_json(RESULTS_DIR / "t4" / "measure_4c.json", out)
    print("=== 4c ORGANIZATION TEST ===", flush=True)
    for arm in arms:
        for shift in SHIFTS:
            a = agg[arm][str(shift)]
            print("  %s shift=%+d  D1=%.4f  D2=%.4f"
                  % (arm, shift, a["d1_acc_mean"], a["d2_rate_mean"]),
                  flush=True)
    print("  F D3 (CONTOUR_PEAK, +2):", agg["F"]["d3_acc_mean_shift2"],
          flush=True)
    for k, v in contrasts.items():
        if v:
            print("    %-40s %.4f [%.4f, %.4f]"
                  % (k, v["mean"], v["lo"], v["hi"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
