"""Task 4 PART 1 — EDGE DIAGNOSIS (on dev).

1a: Edge stability per family (preserved / lost-endpoint / lost-both / added).
1b: Identifiability (unique-colour fraction: signature only, +1WL real, +1WL rewired).
1c: Aperture check (along-tangent vs across-tangent error; accuracy by hops to events).
1d: Parameter informativeness (between-edge IQR vs +2px drift; branching factor).
1e: Edge-effect contrasts (paired, content/sheet, shifts +2/-2, arm F).
1f: Credit decomposition (joint_mcs instrumentation, read-only).
1g: Edge-weight bands for arm F.
1h: Seed-mapping pre-check for P3-3.
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
                      counterpart_map)  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
})
EVENT_KINDS = frozenset({"Corner", "EndStopped", "Junction", "T", "Y"})

ARM_F = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true"
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"

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


def _partition(lens):
    return "content" if lens in CONTENT_TOKEN_LENSES else "non_content"


def _load_dev_graph(arm, name):
    pkl = PKL_DIR / ("%s_%s.pkl" % (arm, name))
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    # Use -090-000 view only (24 scenes) for consistency with Part 0
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _build_shifted(entry, spec, dx, dy, use_fourier=False):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    if dx == 0 and dy == 0:
        return build(entry)
    tmp = Path(tempfile.gettempdir()) / "t4_p1" / (
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


def _edge_family(et):
    return str(getattr(et, "value", et))


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


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def main() -> int:
    print("=== PART 1: EDGE DIAGNOSIS ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # ---- 1a: Edge stability per family ----
    print("\n--- 1a: Edge stability per family ---", flush=True)
    edge_stability = {}
    for arm, spec in [("B1", ARM_B1), ("F", ARM_F)]:
        fam_stats = defaultdict(lambda: {"preserved": 0, "lost_ep": 0,
                                          "lost_both": 0, "added_new": 0,
                                          "added_surv": 0, "total_base": 0})
        part_pair_stats = defaultdict(lambda: {"preserved": 0, "lost_ep": 0,
                                                "lost_both": 0})
        class_pair_stats = defaultdict(lambda: {"preserved": 0, "lost_ep": 0,
                                                 "lost_both": 0})
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base = _load_dev_graph(arm, name)
            sh = _build_shifted(e, spec, 2, 0)
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh)
            cp, _ = counterpart_map(base_pos, sh_pos, 2, 0, 8.0)

            sh_edges = {(min(a, b), max(a, b), c)
                        for (a, b, c) in sh.subgraph.edges}
            inv_cp = {v: k for k, v in cp.items()}

            for (u, v, et) in base.subgraph.edges:
                fam = _edge_family(et)
                stats = fam_stats[fam]
                stats["total_base"] += 1
                eu, ev = cp.get(u), cp.get(v)
                pu = _partition(str(base.sigs.get(u, ("?",))[0]))
                pv = _partition(str(base.sigs.get(v, ("?",))[0]))
                pp_key = "%s-%s" % (pu, pv)
                cu = str(base.sigs.get(u, ("?",))[0])
                cv = str(base.sigs.get(v, ("?",))[0])
                ckey = "%s-%s" % (cu, cv)

                if eu is None or ev is None:
                    stats["lost_ep"] += 1
                    part_pair_stats[pp_key]["lost_ep"] += 1
                    class_pair_stats[ckey]["lost_ep"] += 1
                else:
                    key = (min(eu, ev), max(eu, ev), et)
                    if key in sh_edges:
                        stats["preserved"] += 1
                        part_pair_stats[pp_key]["preserved"] += 1
                        class_pair_stats[ckey]["preserved"] += 1
                    else:
                        stats["lost_both"] += 1
                        part_pair_stats[pp_key]["lost_both"] += 1
                        class_pair_stats[ckey]["lost_both"] += 1

            for (u, v, et) in sh.subgraph.edges:
                fam = _edge_family(et)
                stats = fam_stats[fam]
                bu, bv = inv_cp.get(u), inv_cp.get(v)
                if bu is None or bv is None:
                    stats["added_new"] += 1
                else:
                    key = (min(bu, bv), max(bu, bv), et)
                    base_edges = {(min(a, b), max(a, b), c)
                                  for (a, b, c) in base.subgraph.edges}
                    if key not in base_edges:
                        stats["added_surv"] += 1

        edge_stability[arm] = {
            "per_family": {k: dict(v) for k, v in fam_stats.items()},
            "partition_pairs": {k: dict(v) for k, v in part_pair_stats.items()},
            "class_pairs": {k: dict(v) for k, v in class_pair_stats.items()},
        }
        total_p = sum(v["preserved"] for v in fam_stats.values())
        total_ep = sum(v["lost_ep"] for v in fam_stats.values())
        total_lb = sum(v["lost_both"] for v in fam_stats.values())
        print("  %s: preserved=%d lost_ep=%d lost_both=%d"
              % (arm, total_p, total_ep, total_lb), flush=True)
    out["1a_edge_stability"] = edge_stability

    # ---- 1d: Parameter informativeness ----
    print("\n--- 1d: Parameter informativeness ---", flush=True)
    param_info = {}
    for arm, spec in [("B1", ARM_B1)]:
        fam_axes = defaultdict(lambda: defaultdict(list))
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base = _load_dev_graph(arm, name)
            for (u, v, et) in base.subgraph.edges:
                fam = _edge_family(et)
                ep = base.subgraph.edge_params.get((u, v, et))
                if ep is None:
                    continue
                # Gap, angle, offset parameters
                for axis in ["gap", "angle", "offset_x", "offset_y"]:
                    val = getattr(ep, axis, None)
                    if val is not None:
                        fam_axes[fam][axis].append(float(val))

        fam_out = {}
        for fam, axes in fam_axes.items():
            axes_out = {}
            for axis, vals in axes.items():
                if len(vals) < 2:
                    axes_out[axis] = {"n": len(vals), "status": "n<2"}
                    continue
                arr = np.array(vals)
                iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
                drift = 2.0  # +2px shift drift
                axes_out[axis] = {
                    "n": len(vals),
                    "iqr": iqr,
                    "drift_2px": drift,
                    "iqr_gt_drift": iqr > drift,
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                }
            # Branching factor at frozen tolerance (use 2px as proxy)
            total_edges = sum(len(v) for v in axes.values())
            fam_out[fam] = {
                "axes": axes_out,
                "total_edges": total_edges,
            }
        param_info[arm] = fam_out
        # Print families with IQR > drift on at least one axis
        for fam, info in fam_out.items():
            has_iqr = any(
                a["iqr_gt_drift"] for a in info["axes"].values()
                if "iqr_gt_drift" in a)
            if has_iqr:
                print("  %s: IQR > 2px drift on some axis" % fam, flush=True)
    out["1d_param_informativeness"] = param_info

    # ---- 1g: Edge-weight bands for arm F ----
    print("\n--- 1g: Edge-weight bands for arm F ---", flush=True)
    weight_out = {}
    all_weights = []
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        g = _load_dev_graph("F", name)
        for (u, v, et) in g.subgraph.edges:
            ep = g.subgraph.edge_params.get((u, v, et))
            w = 1.0
            if ep is not None:
                w = getattr(ep, "weight", getattr(ep, "pending_signal", 1.0))
            all_weights.append(float(w))
    if all_weights:
        arr = np.array(all_weights)
        weight_out = {
            "n_edges": len(arr),
            "share_below_0.5": float(np.mean(arr < 0.5)),
            "share_below_0.95": float(np.mean(arr < 0.95)),
            "max_weight": float(np.max(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
        }
        print("  n=%d share<0.5=%.3f share<0.95=%.3f max=%.3f median=%.3f"
              % (len(arr), weight_out["share_below_0.5"],
                 weight_out["share_below_0.95"], weight_out["max_weight"],
                 weight_out["median"]), flush=True)
    out["1g_edge_weights"] = weight_out

    # ---- 1e: Edge-effect contrasts (paired, shifts +2/-2, arm F) ----
    print("\n--- 1e: Edge-effect contrasts (paired, arm F) ---", flush=True)
    contrasts_out = {}
    for shift_dx in [2, -2]:
        print("  shift=%+d" % shift_dx, flush=True)
        # Build F shifted graphs
        real_vals = []
        no_edge_vals = []
        rewired_vals_list = [[] for _ in range(3)]
        content_content_vals = []
        equiv_content_vals = []
        sheet_vals = []
        event_edge_vals = []
        non_event_edge_vals = []

        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base = _load_dev_graph("F", name)
            sh = _build_shifted(e, ARM_F, shift_dx, 0)
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh)
            cp, _ = counterpart_map(base_pos, sh_pos, shift_dx, 0, 8.0)

            # Real
            mapping = _run_matcher(base, sh)
            d1_real = sum(1 for n in base_pos if n in cp and
                          mapping.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            real_vals.append(d1_real)

            # No-edge
            base_ne = replace(base.subgraph, edges=frozenset(), edge_params={})
            g_ne = replace(base, subgraph=base_ne)
            mapping_ne = _run_matcher(g_ne, sh)
            d1_ne = sum(1 for n in base_pos if n in cp and
                        mapping_ne.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            no_edge_vals.append(d1_ne)

            # Rewired (3 seeds)
            for s in range(3):
                base_rw = _rewired_base(base, s)
                mapping_rw = _run_matcher(base_rw, sh)
                d1_rw = sum(1 for n in base_pos if n in cp and
                            mapping_rw.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
                rewired_vals_list[s].append(d1_rw)

            # Content-content only
            base_cc = replace(
                base.subgraph,
                edges=frozenset((u, v, et) for (u, v, et) in base.subgraph.edges
                                if _partition(str(base.sigs.get(u, ("?",))[0])) == "content"
                                and _partition(str(base.sigs.get(v, ("?",))[0])) == "content"),
                edge_params={})
            g_cc = replace(base, subgraph=base_cc)
            mapping_cc = _run_matcher(g_cc, sh)
            d1_cc = sum(1 for n in base_pos if n in cp and
                        mapping_cc.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            content_content_vals.append(d1_cc)

            # Content-content, both endpoints in EQUIVARIANT kinds
            # EQUIVARIANT kinds from Part 0: CONTOUR_PEAK (need to load from part0_corrections)
            equiv_kinds = {"CONTOUR_PEAK"}  # will update from part0_corrections if available
            base_ec = replace(
                base.subgraph,
                edges=frozenset((u, v, et) for (u, v, et) in base.subgraph.edges
                                if str(base.sigs.get(u, ("?",))[0]) in equiv_kinds
                                and str(base.sigs.get(v, ("?",))[0]) in equiv_kinds),
                edge_params={})
            g_ec = replace(base, subgraph=base_ec)
            mapping_ec = _run_matcher(g_ec, sh)
            d1_ec = sum(1 for n in base_pos if n in cp and
                        mapping_ec.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            equiv_content_vals.append(d1_ec)

            # Sheet-involving only
            base_sh = replace(
                base.subgraph,
                edges=frozenset((u, v, et) for (u, v, et) in base.subgraph.edges
                                if _partition(str(base.sigs.get(u, ("?",))[0])) == "non_content"
                                or _partition(str(base.sigs.get(v, ("?",))[0])) == "non_content"),
                edge_params={})
            g_sh = replace(base, subgraph=base_sh)
            mapping_sh = _run_matcher(g_sh, sh)
            d1_sh = sum(1 for n in base_pos if n in cp and
                        mapping_sh.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            sheet_vals.append(d1_sh)

            # Event edges vs non-event edges
            base_ev = replace(
                base.subgraph,
                edges=frozenset((u, v, et) for (u, v, et) in base.subgraph.edges
                                if str(base.sigs.get(u, ("?",))[0]) in EVENT_KINDS
                                or str(base.sigs.get(v, ("?",))[0]) in EVENT_KINDS),
                edge_params={})
            g_ev = replace(base, subgraph=base_ev)
            mapping_ev = _run_matcher(g_ev, sh)
            d1_ev = sum(1 for n in base_pos if n in cp and
                        mapping_ev.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            event_edge_vals.append(d1_ev)

            base_nev = replace(
                base.subgraph,
                edges=frozenset((u, v, et) for (u, v, et) in base.subgraph.edges
                                if str(base.sigs.get(u, ("?",))[0]) not in EVENT_KINDS
                                and str(base.sigs.get(v, ("?",))[0]) not in EVENT_KINDS),
                edge_params={})
            g_nev = replace(base, subgraph=base_nev)
            mapping_nev = _run_matcher(g_nev, sh)
            d1_nev = sum(1 for n in base_pos if n in cp and
                         mapping_nev.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
            non_event_edge_vals.append(d1_nev)

        contrasts = {
            "real_minus_no_edge": _paired_interval(real_vals, no_edge_vals),
            "real_minus_rewired_s0": _paired_interval(real_vals, rewired_vals_list[0]),
            "real_minus_rewired_s1": _paired_interval(real_vals, rewired_vals_list[1]),
            "real_minus_rewired_s2": _paired_interval(real_vals, rewired_vals_list[2]),
            "content_content_minus_no_edge": _paired_interval(content_content_vals, no_edge_vals),
            "equiv_content_minus_no_edge": _paired_interval(equiv_content_vals, no_edge_vals),
            "sheet_minus_no_edge": _paired_interval(sheet_vals, no_edge_vals),
            "event_edges_minus_no_edge": _paired_interval(event_edge_vals, no_edge_vals),
            "non_event_edges_minus_no_edge": _paired_interval(non_event_edge_vals, no_edge_vals),
        }
        contrasts_out["shift_%+d" % shift_dx] = {
            "real_mean": float(np.mean(real_vals)),
            "no_edge_mean": float(np.mean(no_edge_vals)),
            "contrasts": contrasts,
        }
        for k, v in contrasts.items():
            if v:
                print("    %-40s %.4f [%.4f, %.4f] hw=%.4f"
                      % (k, v["mean"], v["lo"], v["hi"], v["hw"]), flush=True)

    out["1e_contrasts"] = contrasts_out

    # ---- 1h: Seed-mapping pre-check for P3-3 ----
    print("\n--- 1h: Seed-mapping pre-check ---", flush=True)
    # Toggle the vector-address seed on dev, report mapping change fraction and D1 delta
    # Using EXISTING address, no new code.
    seed_out = {}
    try:
        import parameters as P
        # Check current FE_VECTOR_ADDRESS value
        current = P.FE_VECTOR_ADDRESS
        seed_out["current_FE_VECTOR_ADDRESS"] = current
        if current:
            # Turn OFF and compare
            P.FE_VECTOR_ADDRESS = False
            d1_off_vals = []
            mapping_off_sets = []
            for e in entries[:8]:  # probe on 8 scenes
                name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
                base = _load_dev_graph("B1", name)
                sh = _build_shifted(e, ARM_B1, 2, 0)
                base_pos = _pos_lens(base)
                sh_pos = _pos_lens(sh)
                cp, _ = counterpart_map(base_pos, sh_pos, 2, 0, 8.0)
                mapping = _run_matcher(base, sh)
                d1 = sum(1 for n in base_pos if n in cp and
                         mapping.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
                d1_off_vals.append(d1)
                mapping_off_sets.append(mapping)
            P.FE_VECTOR_ADDRESS = True
            d1_on_vals = []
            mapping_on_sets = []
            for e in entries[:8]:
                name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
                base = _load_dev_graph("B1", name)
                sh = _build_shifted(e, ARM_B1, 2, 0)
                base_pos = _pos_lens(base)
                sh_pos = _pos_lens(sh)
                cp, _ = counterpart_map(base_pos, sh_pos, 2, 0, 8.0)
                mapping = _run_matcher(base, sh)
                d1 = sum(1 for n in base_pos if n in cp and
                         mapping.get(n) == cp.get(n)) / len(base_pos) if base_pos else 0
                d1_on_vals.append(d1)
                mapping_on_sets.append(mapping)

            # Compare mappings
            changed = 0
            total = 0
            for mo, mn in zip(mapping_off_sets, mapping_on_sets):
                for k in set(mo.keys()) | set(mn.keys()):
                    total += 1
                    if mo.get(k) != mn.get(k):
                        changed += 1
            seed_out["mapping_change_fraction"] = changed / total if total else 0
            seed_out["d1_delta"] = _paired_interval(d1_on_vals, d1_off_vals)
            seed_out["n_scenes"] = 8
            seed_out["interpretation"] = (
                "If no mapping changes, the seed pathway is inert and P3-3 is "
                "unimplementable at this configuration."
            )
            print("  mapping change fraction: %.4f (%d/%d)"
                  % (seed_out["mapping_change_fraction"], changed, total),
                  flush=True)
        else:
            seed_out["status"] = (
                "FE_VECTOR_ADDRESS is OFF. The seed pathway is not active. "
                "P3-3 trigger requires the seed pathway to be toggleable. "
                "Report: seed pathway inert at current configuration; P3-3 "
                "unimplementable unless FE_VECTOR_ADDRESS is turned ON first."
            )
            print("  FE_VECTOR_ADDRESS is OFF; seed pathway inert", flush=True)
    except Exception as ex:
        seed_out["status"] = "error: %s" % ex
        print("  error: %s" % ex, flush=True)
    out["1h_seed_precheck"] = seed_out

    write_json(RESULTS_DIR / "t4" / "part1_edge_diag.json", out)
    print("\n=== PART 1 EDGE DIAGNOSIS complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
