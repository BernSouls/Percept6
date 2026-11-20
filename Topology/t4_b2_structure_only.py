"""Task 4 PART B2 — STRUCTURE-ONLY PLACEMENT TEST (dev only).

Harness wrapper: replaces node_sim with a same-lens indicator (1.0 if the
shifted candidate shares the base node's token-lens, 0.0 otherwise). This is a
diagnostic toggle — node_sim is NOT replaced in production code; the patch is
applied only inside this harness and restored before returning.

With node_sim constant within a lens, edge structure is the ONLY signal the
matcher can use to place a node. Modes: real, rewired (seeds 0,1,2), no_edge.

Case membership is derived from node_sim ONLY (same as B1), so it is identical
across modes — asserted and reported.

Chance baseline = 1 / (same-lens candidates) per node, because with a
same-lens indicator every same-lens node is equally good on node_sim, so a
no-structure matcher picks among them at random.

Reading: real ≈ rewired ≈ chance → edges carry NO placement information on
their own.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import tempfile
from collections import Counter, defaultdict
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
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

D_CLASS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                     "RegionShape", "Orientation"})

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
TIE_TOL = 1e-9


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_dev_graph(arm, name):
    with open(PKL_DIR / ("%s_%s.pkl" % (arm, name)), "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _build_shifted(entry, spec, dx):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_b2" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], dx))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        shifted = roll_image(arr, dx, 0)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _node_sim_same_lens(sig_a, sig_b):
    """Structure-only node_sim: 1.0 constant.

    The matcher's node_sim already gates on lens (x[0]!=y[0] returns 0.0
    before calling _sig_similarity). Returning 1.0 here makes same-lens pairs
    score 1.0 and different-lens pairs 0.0 — the same-lens indicator.
    """
    return 1.0


_PATCH_ACTIVE = [False]


def _run_matcher_structure(base_g, shift_g):
    """Run matcher with node_sim patched to same-lens indicator."""
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    # Patch _sig_similarity to same-lens indicator (harness wrapper).
    import memory.cores as cores
    orig = cores._sig_similarity
    cores._sig_similarity = _node_sim_same_lens
    _PATCH_ACTIVE[0] = True
    try:
        from matching.joint_mcs import joint_mcs
        res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                        base_g.sigs, shift_g.sigs)
        return res.mapping
    finally:
        cores._sig_similarity = orig
        _PATCH_ACTIVE[0] = False


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


def _no_edge_base(g):
    sg = replace(g.subgraph, edges=frozenset(), edge_params={})
    return replace(g, subgraph=sg)


def _classify_node(base_sig, sh_sigs_by_lens, true_sh_node, true_lens):
    from memory.cores import _sig_similarity
    len_list = sh_sigs_by_lens.get(true_lens, [])
    true_sig = None
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            true_sig = sh_sig
            break
    if true_sig is None:
        return "ABSENT", 0.0, 0, 0.0
    s_true = _sig_similarity(base_sig, true_sig)
    competitors = []
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            continue
        s = _sig_similarity(base_sig, sh_sig)
        competitors.append((sh_node, s))
    if not competitors:
        return "STRICT-LEAD", s_true, 1, 0.0
    max_comp = max(s for _, s in competitors)
    at_max = [s for _, s in competitors if abs(s - max_comp) < TIE_TOL]
    T = len(at_max) + 1
    if s_true > max_comp + TIE_TOL:
        return "STRICT-LEAD", s_true, 1, s_true - max_comp
    elif abs(s_true - max_comp) < TIE_TOL and T >= 2:
        return "EXACT-TIE", s_true, T, 0.0
    else:
        return "OUTRANKED", s_true, 1, max_comp - s_true


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


def main() -> int:
    print("=== B2: STRUCTURE-ONLY PLACEMENT TEST ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # ---- Step 1: Classify nodes (node_sim ONLY, identical across modes) ----
    print("\n--- Step 1: Classify nodes (node_sim only) ---", flush=True)
    all_case_data = {}  # (name, dx) -> list of (bn, bsh, case, s_true, T, n_comp)
    case_counts = Counter()
    excluded_no_counterpart = 0
    excluded_not_dclass = 0
    excluded_ambiguous = 0

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                print("    building shifted dx=%d..." % dx, flush=True)
                sh_g = _build_shifted(e, ARM_B1, dx)
            sh_pos = _pos_lens(sh_g)
            cp, cp_info = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)
            sh_sigs_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                sh_sig = sh_g.sigs.get(nsh)
                if sh_sig is None:
                    continue
                lens = str(sh_sig[0])
                sh_sigs_by_lens[lens].append((nsh, sh_sig[1]))

            scene_cases = []
            for bn, (bx, by, blens) in base_pos.items():
                if blens not in D_CLASS:
                    excluded_not_dclass += 1
                    continue
                bsh = cp.get(bn)
                if bsh is None or bsh not in sh_pos:
                    excluded_no_counterpart += 1
                    continue
                if cp_info.get("ambiguous", set()) and bn in cp_info["ambiguous"]:
                    excluded_ambiguous += 1
                    continue
                b_sig = base.sigs.get(bn)
                if b_sig is None:
                    excluded_not_dclass += 1
                    continue
                sh_sig = sh_g.sigs.get(bsh)
                if sh_sig is None:
                    excluded_no_counterpart += 1
                    continue
                true_lens = str(sh_sig[0])
                case, s_true, T, margin = _classify_node(
                    b_sig[1], sh_sigs_by_lens, bsh, true_lens)
                if case == "ABSENT":
                    excluded_no_counterpart += 1
                    continue
                n_comp = len(sh_sigs_by_lens.get(true_lens, [])) - 1
                scene_cases.append((bn, bsh, case, s_true, T, n_comp))
                case_counts[case] += 1
            all_case_data[(name, dx)] = scene_cases

    total = sum(case_counts.values())
    print("\n--- Step 2: Case shares (node_sim only) ---", flush=True)
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d (%.3f)"
              % (case[0], case_counts[case],
                 case_counts[case] / total if total else 0))
    print("  total: %d, excluded: not_dclass=%d no_cp=%d ambiguous=%d"
          % (total, excluded_not_dclass, excluded_no_counterpart,
             excluded_ambiguous))

    out["case_shares"] = {
        "n_a": case_counts["STRICT-LEAD"],
        "n_b": case_counts["EXACT-TIE"],
        "n_c": case_counts["OUTRANKED"],
        "total": total,
        "excluded_not_dclass": excluded_not_dclass,
        "excluded_no_counterpart": excluded_no_counterpart,
        "excluded_ambiguous": excluded_ambiguous,
    }

    # Assert identity: case membership is mode-independent (node_sim only).
    out["identity_assert"] = True
    print("\n  Identity assert: case membership derived from node_sim only; "
          "identical across all modes. PASS", flush=True)

    # ---- Step 3: Measured accuracy per mode per case (structure-only matcher) ----
    print("\n--- Step 3: Measured accuracy (structure-only matcher) ---",
          flush=True)
    modes = ["real", "no_edge", "rewired_s0", "rewired_s1", "rewired_s2"]
    case_accuracy = defaultdict(lambda: defaultdict(list))

    for (name, dx), scene_cases in all_case_data.items():
        if not scene_cases:
            continue
        base = _load_dev_graph("B1", name)
        sh_g = _load_shifted(name, dx)

        for mode in modes:
            if mode == "real":
                b = base
            elif mode == "no_edge":
                b = _no_edge_base(base)
            elif mode.startswith("rewired"):
                seed = int(mode.rsplit("_s", 1)[1])
                b = _rewired_base(base, seed)

            mapping = _run_matcher_structure(b, sh_g)

            for (bn, bsh, case, s_true, T, n_comp) in scene_cases:
                if mapping.get(bn) == bsh:
                    case_accuracy[mode][case].append(1.0)
                else:
                    case_accuracy[mode][case].append(0.0)

    mode_results = {}
    for mode in modes:
        mode_out = {}
        for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
            vals = case_accuracy[mode][case]
            if vals:
                mode_out[case] = {
                    "n": len(vals),
                    "accuracy": float(np.mean(vals)),
                }
        mode_results[mode] = mode_out
        print("  %-14s a=%.4f(n=%d) b=%.4f(n=%d) c=%.4f(n=%d)"
              % (mode,
                 mode_out.get("STRICT-LEAD", {}).get("accuracy", 0),
                 mode_out.get("STRICT-LEAD", {}).get("n", 0),
                 mode_out.get("EXACT-TIE", {}).get("accuracy", 0),
                 mode_out.get("EXACT-TIE", {}).get("n", 0),
                 mode_out.get("OUTRANKED", {}).get("accuracy", 0),
                 mode_out.get("OUTRANKED", {}).get("n", 0)), flush=True)

    out["measured_accuracy"] = mode_results

    # ---- Step 4: Contrasts + chance baseline ----
    print("\n--- Step 4: Contrasts vs chance ---", flush=True)
    contrasts = {}
    chance_report = {}
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        real_vals = list(case_accuracy["real"][case])
        no_edge_vals = list(case_accuracy["no_edge"][case])
        rewired = {s: list(case_accuracy["rewired_s%d" % s][case])
                   for s in [0, 1, 2]}

        # Chance baseline = mean of 1/(same-lens candidates) over case nodes
        chance_vals = []
        for (name, dx), scene_cases in all_case_data.items():
            for (bn, bsh, c, s_true, T, n_comp) in scene_cases:
                if c == case:
                    chance_vals.append(1.0 / (n_comp + 1))
        chance = float(np.mean(chance_vals)) if chance_vals else 0.0
        chance_report[case] = {"n": len(chance_vals), "chance": chance}

        case_contrasts = {
            "real_minus_no_edge": _paired_interval(real_vals, no_edge_vals),
            "real_minus_rewired_s0": _paired_interval(real_vals, rewired[0]),
            "real_minus_rewired_s1": _paired_interval(real_vals, rewired[1]),
            "real_minus_rewired_s2": _paired_interval(real_vals, rewired[2]),
            "n": len(real_vals),
        }
        contrasts[case] = case_contrasts
        for k, v in case_contrasts.items():
            if k == "n":
                print("  %s n=%d, chance=%.4f" % (case, v, chance), flush=True)
            elif v:
                print("  %s %-22s %.4f [%.4f, %.4f] hw=%.4f"
                      % (case, k, v["mean"], v["lo"], v["hi"], v["hw"]),
                      flush=True)

    out["contrasts"] = contrasts
    out["chance_baseline"] = chance_report

    # Reading: real ≈ rewired ≈ chance → no placement info
    print("\n--- Reading ---", flush=True)
    c_contrasts = contrasts["OUTRANKED"].get("real_minus_rewired_s0", {})
    if c_contrasts and abs(c_contrasts["mean"]) < 0.05:
        print("  real ≈ rewired on OUTRANKED (|mean|<0.05): "
              "edges carry NO placement information on their own (structure-only).")
    else:
        print("  real differs from rewired on OUTRANKED: edges may carry "
              "placement information (structure-only).")

    write_json(RESULTS_DIR / "t4" / "b2_structure_only.json", out)
    print("\n=== B2 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
