"""Task 4: FINAL HELD-OUT COMPARISON (after Part 3).

Arms: B0, B1, FINAL (= B1 since no Part 3 steps were implemented).
Node correspondence at 0, ±2, ±3 with partition- and class-correct scoring.
Controls on FINAL (real, real-params-dropped, rewired seeds 0,1,2,
params-shuffled seeds 0,1,2, no-edge) with all contrasts and half-widths.
D1 on both denominators (pooled and class-filtered).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

HOLDOUT_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
})
EQUIVARIANT_KINDS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                               "RegionShape"})
STATIC_KINDS = frozenset({"Orientation"})
# Kinds included in class-filtered denominator
CLASS_FILTERED_KINDS = EQUIVARIANT_KINDS | STATIC_KINDS

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


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_heldout():
    graphs = {}
    for p in HOLDOUT_DIR.glob("*.pkl"):
        parts = p.stem.rsplit("_", 1)
        arm_name = parts[0]
        shift = int(parts[1])
        arm, name = arm_name.split("_", 1)
        with open(p, "rb") as fh:
            graphs[(arm, name, shift)] = pickle.load(fh)
    return graphs


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _d1(base_pos, mapping, cp, kind_filter=None):
    if not base_pos:
        return 0.0, 0
    n_match = 0
    n_den = 0
    for n, (x, y, lens) in base_pos.items():
        if kind_filter is not None and lens not in kind_filter:
            continue
        n_den += 1
        if n in cp and mapping.get(n) == cp.get(n):
            n_match += 1
    return (n_match / n_den if n_den else 0.0), n_den


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
            "hi": float(np.quantile(boots, 0.975)),
            "hw": (float(np.quantile(boots, 0.975))
                   - float(np.quantile(boots, 0.025))) / 2}


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


def _rewired_base(g, seed):
    new_edges = _rewire_edges(g.subgraph.edges, seed)
    sg = replace(g.subgraph, edges=frozenset(new_edges), edge_params={})
    return replace(g, subgraph=sg)


def _params_dropped_base(g):
    sg = replace(g.subgraph, edge_params={})
    return replace(g, subgraph=sg)


def _params_shuffled_base(g, seed):
    rng = random.Random(seed)
    ep = dict(g.subgraph.edge_params)
    by_fam = {}
    for k in ep:
        fam = str(getattr(k[2], "value", k[2]))
        if fam not in by_fam:
            by_fam[fam] = []
        by_fam[fam].append(k)
    new_ep = {}
    for fam, keys in by_fam.items():
        vals = [ep[k] for k in keys]
        rng.shuffle(vals)
        for k, v in zip(keys, vals):
            new_ep[k] = v
    sg = replace(g.subgraph, edge_params=new_ep)
    return replace(g, subgraph=sg)


def _no_edge_base(g):
    sg = replace(g.subgraph, edges=frozenset(), edge_params={})
    return replace(g, subgraph=sg)


def main() -> int:
    print("=== FINAL HELD-OUT COMPARISON ===", flush=True)
    out = {}
    graphs = _load_heldout()
    img_names = sorted({n for (_a, n, _s) in graphs})
    print("loaded %d graphs, %d images" % (len(graphs), len(img_names)),
          flush=True)

    SHIFTS = (0, 2, -2, 3, -3)
    arms = ["B0", "B1"]

    # ---- Node correspondence ----
    print("\n--- Node correspondence ---", flush=True)
    corr = {}
    for arm in arms:
        corr[arm] = {}
        for shift in SHIFTS:
            d1_pooled = []
            d1_class = []
            d2_vals = []
            n_pooled_den = []
            n_class_den = []
            for name in img_names:
                base = graphs[(arm, name, 0)]
                sh_g = graphs[(arm, name, shift)]
                base_pos = _pos_lens(base)
                sh_pos = _pos_lens(sh_g)
                cp, _, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
                mapping = _run_matcher(base, sh_g)
                d1_p, den_p = _d1(base_pos, mapping, cp, None)
                d1_c, den_c = _d1(base_pos, mapping, cp, CLASS_FILTERED_KINDS)
                d1_pooled.append(d1_p)
                d1_class.append(d1_c)
                n_pooled_den.append(den_p)
                n_class_den.append(den_c)
                have_c = sum(1 for n in base_pos if n in cp)
                d2_vals.append(have_c / len(base_pos) if base_pos else 0)
            corr[arm][shift] = {
                "d1_pooled_mean": float(np.mean(d1_pooled)),
                "d1_pooled_per_image": d1_pooled,
                "d1_class_mean": float(np.mean(d1_class)),
                "d1_class_per_image": d1_class,
                "d2_rate_mean": float(np.mean(d2_vals)),
                "n_pooled_den_mean": float(np.mean(n_pooled_den)),
                "n_class_den_mean": float(np.mean(n_class_den)),
            }
            print("  %s s=%+d: D1_pooled=%.4f D1_class=%.4f D2=%.4f"
                  % (arm, shift, corr[arm][shift]["d1_pooled_mean"],
                     corr[arm][shift]["d1_class_mean"],
                     corr[arm][shift]["d2_rate_mean"]), flush=True)
    # FINAL = B1
    corr["FINAL"] = corr["B1"]
    out["correspondence"] = corr

    # ---- Controls on FINAL (=B1, shift +2) ----
    print("\n--- Controls on FINAL (shift +2) ---", flush=True)
    controls = {}
    best = "B1"
    shift = 2
    real_vals_p = []
    real_vals_c = []
    for mode in ("real", "real_params_dropped", "rewired",
                 "params_shuffled", "no_edge"):
        vals_p = []
        vals_c = []
        for name in img_names:
            base = graphs[(best, name, 0)]
            sh_g = graphs[(best, name, shift)]
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh_g)
            cp, _, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
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
            d1_p, _ = _d1(base_pos, mapping, cp, None)
            d1_c, _ = _d1(base_pos, mapping, cp, CLASS_FILTERED_KINDS)
            vals_p.append(d1_p)
            vals_c.append(d1_c)
        controls[mode] = {
            "d1_pooled_mean": float(np.mean(vals_p)),
            "d1_pooled_per_image": vals_p,
            "d1_class_mean": float(np.mean(vals_c)),
            "d1_class_per_image": vals_c,
        }
        print("  %-20s D1_pooled=%.4f D1_class=%.4f"
              % (mode, controls[mode]["d1_pooled_mean"],
                 controls[mode]["d1_class_mean"]), flush=True)

    # Rewired seeds 1, 2
    for seed in [1, 2]:
        vals_p = []
        vals_c = []
        for name in img_names:
            base = graphs[(best, name, 0)]
            sh_g = graphs[(best, name, shift)]
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh_g)
            cp, _, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
            b = _rewired_base(base, seed)
            mapping = _run_matcher(b, sh_g)
            d1_p, _ = _d1(base_pos, mapping, cp, None)
            d1_c, _ = _d1(base_pos, mapping, cp, CLASS_FILTERED_KINDS)
            vals_p.append(d1_p)
            vals_c.append(d1_c)
        controls["rewired_s%d" % seed] = {
            "d1_pooled_mean": float(np.mean(vals_p)),
            "d1_pooled_per_image": vals_p,
            "d1_class_mean": float(np.mean(vals_c)),
            "d1_class_per_image": vals_c,
        }

    # Params shuffled seeds 1, 2
    for seed in [1, 2]:
        vals_p = []
        vals_c = []
        for name in img_names:
            base = graphs[(best, name, 0)]
            sh_g = graphs[(best, name, shift)]
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh_g)
            cp, _, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
            b = _params_shuffled_base(base, seed)
            mapping = _run_matcher(b, sh_g)
            d1_p, _ = _d1(base_pos, mapping, cp, None)
            d1_c, _ = _d1(base_pos, mapping, cp, CLASS_FILTERED_KINDS)
            vals_p.append(d1_p)
            vals_c.append(d1_c)
        controls["params_shuffled_s%d" % seed] = {
            "d1_pooled_mean": float(np.mean(vals_p)),
            "d1_pooled_per_image": vals_p,
            "d1_class_mean": float(np.mean(vals_c)),
            "d1_class_per_image": vals_c,
        }

    out["controls"] = controls

    # Contrasts on both denominators
    real_p = controls["real"]["d1_pooled_per_image"]
    real_c = controls["real"]["d1_class_per_image"]
    contrasts = {}
    for mode in controls:
        if mode == "real":
            continue
        contrasts["real_minus_%s_pooled" % mode] = _paired_interval(
            real_p, controls[mode]["d1_pooled_per_image"])
        contrasts["real_minus_%s_class" % mode] = _paired_interval(
            real_c, controls[mode]["d1_class_per_image"])
    for k, v in contrasts.items():
        if v:
            print("  %-50s %.4f [%.4f, %.4f] hw=%.4f"
                  % (k, v["mean"], v["lo"], v["hi"], v["hw"]), flush=True)
    out["contrasts"] = contrasts

    write_json(RESULTS_DIR / "t4" / "final_heldout_compare.json", out)
    print("\n=== FINAL HELD-OUT COMPARISON complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
