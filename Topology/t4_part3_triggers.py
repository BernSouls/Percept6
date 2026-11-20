"""Task 4: P3-0 and P3-1 TRIGGER EVALUATION on dev (optimized).

Pre-computes real and no_edge D1s once per scene per shift (family-independent),
then for each family only computes the without_family D1. This reduces matcher
runs from 3x to 1x per family per scene per shift.

P3-0 TRIGGER: leave-one-family-out shows removing a family improves the
    contrast with no_edge (interval above zero at corrected level).
    Corrected level: (1 - 0.05/k) where k = number of families tested.

P3-1 TRIGGER: paired interval (host-kind-only edges minus all edges,
    contrast vs no_edge) is above zero at corrected level.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
})
EQUIVARIANT_KINDS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                               "RegionShape"})

ARM_F = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true"

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


def _load_dev_graph(arm, name):
    pkl = PKL_DIR / ("%s_%s.pkl" % (arm, name))
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _build_shifted(entry, spec, dx, dy):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    tmp = Path(tempfile.gettempdir()) / "t4_p3" / (
        "%s%d%s_%d_%d.png" % (entry["class"], int(entry["object"]),
                              entry["view"], dx, dy))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        from t4_shift import roll_image
        from PIL import Image
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        shifted = roll_image(arr, dx, dy)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _d1(base_pos, mapping, cp):
    if not base_pos:
        return 0.0
    return sum(1 for n in base_pos if n in cp and
               mapping.get(n) == cp.get(n)) / len(base_pos)


def _paired_interval(a_vals, b_vals, conf=0.95):
    n = len(a_vals)
    if n == 0:
        return None
    diffs = np.array(a_vals) - np.array(b_vals)
    mean = float(diffs.mean())
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    alpha = 1.0 - conf
    return {"mean": mean,
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)),
            "hw": (float(np.quantile(boots, 1 - alpha / 2))
                   - float(np.quantile(boots, alpha / 2))) / 2}


def _graph_without_family(g, fam):
    kept = [(u, v, et) for (u, v, et) in g.subgraph.edges
            if str(getattr(et, "value", et)) != fam]
    sg = replace(g.subgraph, edges=frozenset(kept), edge_params={})
    return replace(g, subgraph=sg)


def _graph_host_only(g):
    kept = [(u, v, et) for (u, v, et) in g.subgraph.edges
            if str(g.sigs.get(u, ("?",))[0]) in EQUIVARIANT_KINDS
            and str(g.sigs.get(v, ("?",))[0]) in EQUIVARIANT_KINDS]
    sg = replace(g.subgraph, edges=frozenset(kept), edge_params={})
    return replace(g, subgraph=sg)


def _graph_no_edge(g):
    sg = replace(g.subgraph, edges=frozenset(), edge_params={})
    return replace(g, subgraph=sg)


def main() -> int:
    print("=== P3-0 / P3-1 TRIGGER EVALUATION (optimized) ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # Collect all families from F dev graphs
    all_families = set()
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        g = _load_dev_graph("F", name)
        for (u, v, et) in g.subgraph.edges:
            all_families.add(str(getattr(et, "value", et)))
    families = sorted(all_families)
    print("families (%d): %s" % (len(families), families), flush=True)

    # Pre-build shifted graphs and pre-compute family-independent D1s
    print("\nPre-computing shifted graphs and base D1s...", flush=True)
    # For each scene, store: base_pos, cp_pos[+2], cp_neg[-2],
    # sh_pos[+2], sh_neg[-2], and base graph
    scene_data = {}
    real_vals = {2: [], -2: []}
    no_edge_vals = {2: [], -2: []}

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        base = _load_dev_graph("F", name)
        base_pos = _pos_lens(base)

        for dx in [2, -2]:
            sh = _build_shifted(e, ARM_F, dx, 0)
            sh_pos = _pos_lens(sh)
            cp, _ = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)

            # Real D1
            mapping_real = _run_matcher(base, sh)
            d1_real = _d1(base_pos, mapping_real, cp)
            real_vals[dx].append(d1_real)

            # No-edge D1
            g_ne = _graph_no_edge(base)
            mapping_ne = _run_matcher(g_ne, sh)
            d1_ne = _d1(base_pos, mapping_ne, cp)
            no_edge_vals[dx].append(d1_ne)

            scene_data[(name, dx)] = {
                "base": base, "base_pos": base_pos,
                "sh": sh, "cp": cp,
            }
        if len(scene_data) % 4 == 0:
            print("  %d/%d scenes pre-computed" % (len(scene_data) // 2,
                                                   len(entries)),
                  flush=True)

    print("Pre-computation done", flush=True)

    # ---- P3-0: Leave-one-family-out ----
    print("\n--- P3-0: Leave-one-family-out ---", flush=True)
    k = len(families)
    corrected_conf = 1.0 - 0.05 / k
    print("corrected confidence: %.6f (k=%d)" % (corrected_conf, k),
          flush=True)

    loo_results = {}
    for idx, fam in enumerate(families):
        wo_vals = {2: [], -2: []}
        for (name, dx), sd in scene_data.items():
            base = sd["base"]
            base_pos = sd["base_pos"]
            sh = sd["sh"]
            cp = sd["cp"]
            g_wo = _graph_without_family(base, fam)
            mapping_wo = _run_matcher(g_wo, sh)
            d1_wo = _d1(base_pos, mapping_wo, cp)
            wo_vals[dx].append(d1_wo)

        interval_pos = _paired_interval(wo_vals[2], no_edge_vals[2],
                                        conf=corrected_conf)
        interval_neg = _paired_interval(wo_vals[-2], no_edge_vals[-2],
                                        conf=corrected_conf)
        loo_results[fam] = {
            "contrast_plus2": interval_pos,
            "contrast_minus2": interval_neg,
        }
        fires = (interval_pos and interval_pos["lo"] > 0) or \
                (interval_neg and interval_neg["lo"] > 0)
        print("  fam=%-4s +2: %s  -2: %s  fires=%s"
              % (fam,
                 "%.4f [%.4f, %.4f]" % (interval_pos["mean"],
                                        interval_pos["lo"], interval_pos["hi"])
                 if interval_pos else "N/A",
                 "%.4f [%.4f, %.4f]" % (interval_neg["mean"],
                                         interval_neg["lo"], interval_neg["hi"])
                 if interval_neg else "N/A",
                 fires), flush=True)

    fired_families = [f for f, r in loo_results.items()
                      if (r["contrast_plus2"] and r["contrast_plus2"]["lo"] > 0)
                      or (r["contrast_minus2"] and r["contrast_minus2"]["lo"] > 0)]
    out["P3-0_trigger"] = {
        "k_families": k,
        "corrected_confidence": corrected_conf,
        "fired_families": fired_families,
        "all_results": loo_results,
        "trigger_fired": len(fired_families) > 0,
    }
    print("\nP3-0 trigger fired: %s (families: %s)"
          % (len(fired_families) > 0, fired_families), flush=True)

    # ---- P3-1: Host-kind-only edges ----
    print("\n--- P3-1: Host-kind-only edges ---", flush=True)
    host_vals = {2: [], -2: []}
    for (name, dx), sd in scene_data.items():
        base = sd["base"]
        base_pos = sd["base_pos"]
        sh = sd["sh"]
        cp = sd["cp"]
        g_host = _graph_host_only(base)
        mapping_host = _run_matcher(g_host, sh)
        d1_host = _d1(base_pos, mapping_host, cp)
        host_vals[dx].append(d1_host)

    # Contrast: D1(host) - D1(all)
    interval_pos = _paired_interval(host_vals[2], real_vals[2], conf=0.95)
    interval_neg = _paired_interval(host_vals[-2], real_vals[-2], conf=0.95)
    # Also contrast vs no_edge
    host_vs_ne_pos = _paired_interval(host_vals[2], no_edge_vals[2], conf=0.95)
    host_vs_ne_neg = _paired_interval(host_vals[-2], no_edge_vals[-2],
                                      conf=0.95)
    all_vs_ne_pos = _paired_interval(real_vals[2], no_edge_vals[2], conf=0.95)
    all_vs_ne_neg = _paired_interval(real_vals[-2], no_edge_vals[-2],
                                     conf=0.95)

    out["P3-1_trigger"] = {
        "host_kinds": sorted(EQUIVARIANT_KINDS),
        "contrast_host_minus_all_pos2": interval_pos,
        "contrast_host_minus_all_neg2": interval_neg,
        "host_vs_no_edge_pos2": host_vs_ne_pos,
        "host_vs_no_edge_neg2": host_vs_ne_neg,
        "all_vs_no_edge_pos2": all_vs_ne_pos,
        "all_vs_no_edge_neg2": all_vs_ne_neg,
        "trigger_fired": (interval_pos and interval_pos["lo"] > 0)
                         or (interval_neg and interval_neg["lo"] > 0),
    }
    if interval_pos:
        print("  host-all +2: %.4f [%.4f, %.4f] hw=%.4f"
              % (interval_pos["mean"], interval_pos["lo"], interval_pos["hi"],
                 interval_pos["hw"]), flush=True)
    if interval_neg:
        print("  host-all -2: %.4f [%.4f, %.4f] hw=%.4f"
              % (interval_neg["mean"], interval_neg["lo"], interval_neg["hi"],
                 interval_neg["hw"]), flush=True)
    if host_vs_ne_pos:
        print("  host-no_edge +2: %.4f [%.4f, %.4f]"
              % (host_vs_ne_pos["mean"], host_vs_ne_pos["lo"],
                 host_vs_ne_pos["hi"]), flush=True)
    if all_vs_ne_pos:
        print("  all-no_edge +2: %.4f [%.4f, %.4f]"
              % (all_vs_ne_pos["mean"], all_vs_ne_pos["lo"],
                 all_vs_ne_pos["hi"]), flush=True)
    print("\nP3-1 trigger fired: %s" % out["P3-1_trigger"]["trigger_fired"],
          flush=True)

    write_json(RESULTS_DIR / "t4" / "part3_triggers.json", out)
    print("\n=== TRIGGER EVALUATION complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
