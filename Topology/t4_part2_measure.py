"""Task 4 PART 2 — REDO THE MEASUREMENTS (repaired instrument).

2a: Arms B0 and B1 on dev: mean degree, zero-relation share, per-family count,
    stability vs null, branching at frozen tolerance, median NN arc gap.
    Content and sheet partitions separately.
2b: Node correspondence on held-out at 0, ±2, ±3 (repaired), plus Fourier
    shifts on dev. 0px ceiling decomposition. Two identifiability ceilings.
2c: Controls on the best arm: real, real-params-dropped, rewired,
    params-shuffled, no-edge. Paired intervals by image.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import (roll_image, fourier_shift, expected_position,
                      counterpart_map, matcher_accuracy,
                      primary_radius)  # noqa: E402

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}
SHEET_LENSES = {"Orientation", "RegionShape"}
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
HOLDOUT_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _partition(lens):
    return "sheet" if lens in SHEET_LENSES else "content"


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    dev = [e for e in split["dev"] if e["view"] == "-090-000"]
    by_class = {}
    for e in dev:
        c = e["class"]
        if c not in by_class or e["object"] < by_class[c]["object"]:
            by_class[c] = e
    return [by_class[c] for c in corpus.CLASSES]


def _build_dev(entry, arm, dx=0, dy=0, use_fourier=False):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    if dx == 0 and dy == 0:
        return build(entry)
    tmp = Path(tempfile.gettempdir()) / "t4_part2" / (
        "%s%d%s_%d_%d_%s.png" % (entry["class"], int(entry["object"]),
                                 entry["view"], dx, dy,
                                 "f" if use_fourier else "r"))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if use_fourier:
            shifted = fourier_shift(arr, dx, dy)
        else:
            shifted = roll_image(arr, dx, dy)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


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
    by_fam = defaultdict(list)
    for k in ep:
        by_fam[str(getattr(k[2], "value", k[2]))].append(k)
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
    print("=== PART 2: REDO MEASUREMENTS ===", flush=True)
    out = {}

    # ---- 2a: Dev measurements for B0, B1 ----
    print("\n--- 2a: Dev measurements ---", flush=True)
    entries = _dev_entries()
    dev_out = {}
    for arm in ["B0", "B1"]:
        arm_out = {"per_scene": {}, "per_family": Counter(),
                   "mean_degree": 0.0, "zero_relation_share": 0.0,
                   "median_nn_gap": 0.0}
        total_nodes = total_edges = total_zero = 0
        all_nn_gaps = []
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            g = _build_dev(e, arm)
            pos = _pos_lens(g)
            nodes = len(g.subgraph.nodes)
            edges = len(g.subgraph.edges)
            # mean degree
            deg = 2 * edges / nodes if nodes else 0.0
            # per-family counts
            for (u, v, et) in g.subgraph.edges:
                fam = str(getattr(et, "value", et))
                arm_out["per_family"][fam] += 1
            # zero-relation share: nodes with no edges
            deg_count = Counter()
            for (u, v, _et) in g.subgraph.edges:
                deg_count[u] += 1
                deg_count[v] += 1
            zero = sum(1 for n in g.subgraph.nodes if deg_count.get(n, 0) == 0)
            # NN arc gap for CONTOUR_PEAK
            cp_pts = [(x, y) for (x, y, l) in pos.values() if l == "CONTOUR_PEAK"]
            if len(cp_pts) > 1:
                for i, (x, y) in enumerate(cp_pts):
                    best = float("inf")
                    for j, (x2, y2) in enumerate(cp_pts):
                        if i == j:
                            continue
                        d = math.hypot(x - x2, y - y2)
                        if d < best:
                            best = d
                    if best < float("inf"):
                        all_nn_gaps.append(best)
            arm_out["per_scene"][name] = {
                "nodes": nodes, "edges": edges, "degree": deg,
                "zero_nodes": zero,
            }
            total_nodes += nodes
            total_edges += edges
            total_zero += zero
        n_scenes = len(entries)
        arm_out["mean_degree"] = total_edges * 2 / total_nodes if total_nodes else 0
        arm_out["zero_relation_share"] = total_zero / total_nodes if total_nodes else 0
        arm_out["median_nn_gap"] = float(np.median(all_nn_gaps)) if all_nn_gaps else 0
        arm_out["per_family"] = dict(arm_out["per_family"])
        dev_out[arm] = arm_out
        print("  %s: mean_degree=%.4f zero_share=%.4f median_nn_gap=%.3f"
              % (arm, arm_out["mean_degree"], arm_out["zero_relation_share"],
                 arm_out["median_nn_gap"]), flush=True)
    out["2a"] = dev_out

    # ---- 2b: Node correspondence on held-out ----
    print("\n--- 2b: Held-out correspondence ---", flush=True)
    graphs = _load_heldout()
    img_names = sorted({n for (_a, n, _s) in graphs})
    print("  loaded %d graphs, %d images" % (len(graphs), len(img_names)),
          flush=True)

    # Token kinds from B0
    b0_pos = _pos_lens(graphs[("B0", img_names[0], 0)])
    token_kinds = {l for (_x, _y, l) in b0_pos.values()}
    print("  token kinds:", sorted(token_kinds), flush=True)

    # Compute primary radius per kind from dev (B0)
    all_positions = defaultdict(list)
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS["B0"])
    from frontend import build
    for e in entries:
        g = build(e)
        for n, (x, y, l) in _pos_lens(g).items():
            all_positions[l].append((x, y))
    radius_table = primary_radius(all_positions)

    # 2b: held-out correspondence at shifts 0, ±2, ±3
    arms = ["B0", "B1", "F"]
    SHIFTS = (0, 2, -2, 3, -3)
    corr_out = {}
    for arm in arms:
        corr_out[arm] = {}
        for shift in SHIFTS:
            d1_vals = []
            d2_vals = []
            for name in img_names:
                base = graphs[(arm, name, 0)]
                sh_g = graphs[(arm, name, shift)]
                base_pos = _pos_lens(base)
                sh_pos = _pos_lens(sh_g)
                # Use a uniform radius of 2.0 px for the primary test
                # (per-kind radii vary; 2.0 is the frozen tolerance)
                cp, _info, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
                mapping = _run_matcher(base, sh_g)
                d1 = matcher_accuracy(base_pos, mapping, cp, token_kinds)
                have_c = sum(1 for n in base_pos if n in cp)
                d2 = have_c / len(base_pos) if base_pos else 0.0
                d1_vals.append(d1["acc"])
                d2_vals.append(d2)
            corr_out[arm][shift] = {
                "d1_acc_mean": float(np.mean(d1_vals)),
                "d1_per_image": d1_vals,
                "d2_rate_mean": float(np.mean(d2_vals)),
                "d2_per_image": d2_vals,
            }
            print("  %s shift=%+d: D1=%.4f D2=%.4f"
                  % (arm, shift, corr_out[arm][shift]["d1_acc_mean"],
                     corr_out[arm][shift]["d2_rate_mean"]), flush=True)
    out["2b_heldout"] = corr_out
    out["2b_radius_table"] = radius_table

    # 0px ceiling decomposition
    print("\n  0px ceiling decomposition:", flush=True)
    decomp = {}
    for arm in arms:
        name = img_names[0]
        base = graphs[(arm, name, 0)]
        sh_g = graphs[(arm, name, 0)]
        base_pos = _pos_lens(base)
        sh_pos = _pos_lens(sh_g)
        mapping = _run_matcher(base, sh_g)
        identity_cp = {n: n for n in base_pos}
        unmapped = 0
        mapped_ident = 0
        mapped_diff = 0
        for n in base_pos:
            if n not in mapping:
                unmapped += 1
            elif mapping[n] == n:
                mapped_ident += 1
            else:
                mapped_diff += 1
        total = len(base_pos)
        decomp[arm] = {
            "unmapped": unmapped, "mapped_ident": mapped_ident,
            "mapped_diff": mapped_diff, "total": total,
            "ident_frac": mapped_ident / total if total else 0,
            "diff_frac": mapped_diff / total if total else 0,
            "unmapped_frac": unmapped / total if total else 0,
        }
        print("    %s: unmapped=%d (%.3f) ident=%d (%.3f) diff=%d (%.3f)"
              % (arm, unmapped, unmapped / total, mapped_ident,
                 mapped_ident / total, mapped_diff, mapped_diff / total),
              flush=True)
    out["2b_0px_decomp"] = decomp

    # ---- 2c: Controls on best arm (F) at shift +2 ----
    print("\n--- 2c: Controls (arm F, shift +2) ---", flush=True)
    best = "F"
    shift = 2
    controls = {}
    for mode in ("real", "real_params_dropped", "rewired",
                 "params_shuffled", "no_edge"):
        vals = []
        for name in img_names:
            base = graphs[(best, name, 0)]
            sh_g = graphs[(best, name, shift)]
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh_g)
            cp, _info, _cp_stats = counterpart_map(base_pos, sh_pos, shift, 0, 2.0)
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
            d1 = matcher_accuracy(base_pos, mapping, cp, token_kinds)
            vals.append(d1["acc"])
        controls[mode] = {"mean": float(np.mean(vals)), "per_image": vals}
        print("  %-20s D1=%.4f" % (mode, controls[mode]["mean"]), flush=True)

    # Paired contrasts
    real_vals = controls["real"]["per_image"]
    contrasts = {
        "real_minus_no_edge": _paired_interval(real_vals,
                                               controls["no_edge"]["per_image"]),
        "real_minus_rewired": _paired_interval(real_vals,
                                               controls["rewired"]["per_image"]),
        "topology_realpd_minus_rewired": _paired_interval(
            controls["real_params_dropped"]["per_image"],
            controls["rewired"]["per_image"]),
        "param_presence_real_minus_realpd": _paired_interval(
            real_vals, controls["real_params_dropped"]["per_image"]),
        "param_correct_real_minus_shuffled": _paired_interval(
            real_vals, controls["params_shuffled"]["per_image"]),
    }
    for k, v in contrasts.items():
        if v:
            print("    %-40s %.4f [%.4f, %.4f] hw=%.4f"
                  % (k, v["mean"], v["lo"], v["hi"],
                     (v["hi"] - v["lo"]) / 2), flush=True)
    out["2c_controls"] = controls
    out["2c_contrasts"] = contrasts

    write_json(RESULTS_DIR / "t4" / "part2_measure.json", out)
    print("\n=== PART 2 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
