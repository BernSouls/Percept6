"""Task 4 PART B1 — THREE-CASE PLACEMENT TEST (dev only; the primary test).

Unit: base nodes of arm B1 in D_class that have an UNAMBIGUOUS counterpart
(mutual-nearest, no duplicate position).
Competitors: ALL same-lens nodes in the shifted graph except the true counterpart.
node_sim: the matcher's raw node_sim (_sig_similarity), isolated down-weight applied
          at score time, NOT inside node_sim (state it).

Cases (float tolerance 1e-9):
  (a) STRICT-LEAD: s_true exceeds every competitor.
  (b) EXACT-TIE:   s_true equals the max and T >= 2 nodes share it.
  (c) OUTRANKED:   some competitor exceeds s_true.

Descriptive dose: margin terciles within (a) and (c), frozen from dev.

Predicted node-only accuracy = share(a) + share(b) * mean_b(1/T).
Report predicted vs measured no_edge accuracy overall and per case.
D1 per case for real, no_edge, rewired (seeds 0,1,2), params-shuffled (seeds 0,1,2).
real - no_edge and real - rewired per case; share-weighted net effect.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
})
# D_class = EQUIVARIANT + STATIC-WHEN-PRESENT from Part 0
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
    pkl = PKL_DIR / ("%s_%s.pkl" % (arm, name))
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _save_shifted(name, dx, g):
    SHIFTED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SHIFTED_DIR / ("%s_%d.pkl" % (name, dx)), "wb") as fh:
        pickle.dump(g, fh)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _build_shifted(entry, spec, dx):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_b1" / (
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


def _node_sim(sig_a, sig_b):
    """The matcher's raw node_sim. Isolated down-weight NOT applied here."""
    from memory.cores import _sig_similarity
    return _sig_similarity(sig_a, sig_b)


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


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


def _classify_node(base_sig, sh_sigs_by_lens, true_sh_node, true_lens):
    """Classify a base node into (a), (b), (c) using node_sim.

    Returns (case, s_true, T, competitors_info).
    """
    len_list = sh_sigs_by_lens.get(true_lens, [])
    # Find the true counterpart's signature
    true_sig = None
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            true_sig = sh_sig
            break
    if true_sig is None:
        return "ABSENT", 0.0, 0, 0.0

    s_true = _node_sim(base_sig, true_sig)

    competitors = []  # list of (sh_node, sim)
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            continue
        s = _node_sim(base_sig, sh_sig)
        competitors.append((sh_node, s))

    if not competitors:
        return "STRICT-LEAD", s_true, 1, 0.0  # no competitors = strict lead

    max_comp = max(s for _, s in competitors)
    # Count nodes at the max (including true)
    at_max = [s for _, s in competitors if abs(s - max_comp) < TIE_TOL]
    T = len(at_max) + 1  # +1 for true

    if s_true > max_comp + TIE_TOL:
        return "STRICT-LEAD", s_true, 1, s_true - max_comp
    elif abs(s_true - max_comp) < TIE_TOL and T >= 2:
        return "EXACT-TIE", s_true, T, 0.0
    else:
        return "OUTRANKED", s_true, 1, max_comp - s_true


def main() -> int:
    print("=== B1: THREE-CASE PLACEMENT TEST ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # ---- Step 1: Build shifted graphs and classify nodes ----
    print("\n--- Step 1: Build shifted graphs and classify ---", flush=True)
    all_case_data = {}  # (name, dx) -> list of (base_node, case, s_true, T, margin)
    case_counts = Counter()
    case_total = 0
    excluded_no_counterpart = 0
    excluded_not_dclass = 0
    excluded_ambiguous = 0
    sim_range = []
    margins_a = []
    margins_c = []

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
                _save_shifted(name, dx, sh_g)
            sh_pos = _pos_lens(sh_g)

            # Counterpart map (mutual-nearest, one-to-one)
            cp, cp_info = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)

            # Build sh_sigs_by_lens
            sh_sigs_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                sh_sig = sh_g.sigs.get(nsh)
                if sh_sig is None:
                    continue
                lens = str(sh_sig[0])
                sh_sigs_by_lens[lens].append((nsh, sh_sig[1]))

            # Classify each base node in D_class with an unambiguous counterpart
            scene_cases = []
            for bn, (bx, by, blens) in base_pos.items():
                if blens not in D_CLASS:
                    excluded_not_dclass += 1
                    continue
                bsh = cp.get(bn)
                if bsh is None:
                    excluded_no_counterpart += 1
                    continue
                # Check ambiguous (second-nearest within 2x nearest per one-to-one rule)
                if cp_info.get(bn, {}).get("ambiguous", False):
                    excluded_ambiguous += 1
                    continue

                base_sig = base.sigs.get(bn)
                if base_sig is None:
                    continue
                case, s_true, T, margin = _classify_node(
                    base_sig[1], sh_sigs_by_lens, bsh, blens)
                sim_range.append(s_true)
                if case == "STRICT-LEAD":
                    margins_a.append(margin)
                elif case == "OUTRANKED":
                    margins_c.append(margin)
                scene_cases.append((bn, bsh, case, s_true, T, margin, blens))
                case_counts[case] += 1
                case_total += 1

            all_case_data[(name, dx)] = scene_cases

        print("    cases: a=%d b=%d c=%d (total=%d)"
              % (case_counts["STRICT-LEAD"], case_counts["EXACT-TIE"],
                 case_counts["OUTRANKED"], case_total), flush=True)

    # ---- Report shares ----
    n_a = case_counts["STRICT-LEAD"]
    n_b = case_counts["EXACT-TIE"]
    n_c = case_counts["OUTRANKED"]
    total = n_a + n_b + n_c
    print("\n--- Step 2: Case shares ---", flush=True)
    print("  (a) STRICT-LEAD:  n=%d (%.3f)" % (n_a, n_a / total if total else 0))
    print("  (b) EXACT-TIE:    n=%d (%.3f)" % (n_b, n_b / total if total else 0))
    print("  (c) OUTRANKED:    n=%d (%.3f)" % (n_c, n_c / total if total else 0))
    print("  total: %d, excluded: not_dclass=%d no_cp=%d ambiguous=%d"
          % (total, excluded_not_dclass, excluded_no_counterpart,
             excluded_ambiguous))

    # Margin terciles
    if margins_a:
        ma = np.array(margins_a)
        print("  (a) margin: median=%.4f p25=%.4f p75=%.4f"
              % (np.median(ma), np.quantile(ma, 0.25), np.quantile(ma, 0.75)))
    if margins_c:
        mc = np.array(margins_c)
        print("  (c) margin: median=%.4f p25=%.4f p75=%.4f"
              % (np.median(mc), np.quantile(mc, 0.25), np.quantile(mc, 0.75)))
    if sim_range:
        sr = np.array(sim_range)
        print("  node_sim range: [%.4f, %.4f] perfect=1.0"
              % (sr.min(), sr.max()))

    # Predicted no_edge accuracy
    mean_b_invT = 0.0
    if n_b > 0:
        invTs = [1.0 / T for bn, bsh, case, s_true, T, margin, blens
                 in [(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
                    for cs in all_case_data.values() for c in cs
                    if c[2] == "EXACT-TIE"]
        ]
        mean_b_invT = float(np.mean(invTs)) if invTs else 0.0
    predicted_no_edge = (n_a / total if total else 0) + \
        (n_b / total if total else 0) * mean_b_invT
    print("  predicted no_edge accuracy: %.4f (a=%.4f + b=%.4f * %.4f)"
          % (predicted_no_edge, n_a / total if total else 0,
             n_b / total if total else 0, mean_b_invT))

    out["case_shares"] = {
        "n_a": n_a, "n_b": n_b, "n_c": n_c, "total": total,
        "share_a": n_a / total if total else 0,
        "share_b": n_b / total if total else 0,
        "share_c": n_c / total if total else 0,
        "excluded_not_dclass": excluded_not_dclass,
        "excluded_no_counterpart": excluded_no_counterpart,
        "excluded_ambiguous": excluded_ambiguous,
        "mean_b_1_over_T": mean_b_invT,
        "predicted_no_edge_accuracy": predicted_no_edge,
        "node_sim_range": [float(sr.min()), float(sr.max())] if sim_range else None,
    }

    # ---- Step 3: Measure accuracy per case for each mode ----
    print("\n--- Step 3: Measured accuracy per case ---", flush=True)
    modes = ["real", "no_edge"]
    for seed in [0, 1, 2]:
        modes.extend(["rewired_s%d" % seed, "params_shuffled_s%d" % seed])

    # Per-case accuracy: for each case, among nodes in that case, fraction where
    # the matcher maps the base node to its true counterpart.
    case_accuracy = defaultdict(lambda: defaultdict(list))  # mode -> case -> list
    case_n = defaultdict(int)  # case -> count

    for (name, dx), scene_cases in all_case_data.items():
        if not scene_cases:
            continue
        base = _load_dev_graph("B1", name)
        sh_g = _load_shifted(name, dx)
        base_pos = _pos_lens(base)

        for mode in modes:
            if mode == "real":
                b = base
            elif mode == "no_edge":
                b = _no_edge_base(base)
            elif mode.startswith("rewired"):
                seed = int(mode.rsplit("_s", 1)[1])
                b = _rewired_base(base, seed)
            elif mode.startswith("params_shuffled"):
                seed = int(mode.rsplit("_s", 1)[1])
                b = _params_shuffled_base(base, seed)

            mapping = _run_matcher(b, sh_g)

            for (bn, bsh, case, s_true, T, margin, blens) in scene_cases:
                case_n[case] += 1
                if mapping.get(bn) == bsh:
                    case_accuracy[mode][case].append(1.0)
                else:
                    case_accuracy[mode][case].append(0.0)

    # Aggregate
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
        print("  %-20s a=%.4f(n=%d) b=%.4f(n=%d) c=%.4f(n=%d)"
              % (mode,
                 mode_out.get("STRICT-LEAD", {}).get("accuracy", 0),
                 mode_out.get("STRICT-LEAD", {}).get("n", 0),
                 mode_out.get("EXACT-TIE", {}).get("accuracy", 0),
                 mode_out.get("EXACT-TIE", {}).get("n", 0),
                 mode_out.get("OUTRANKED", {}).get("accuracy", 0),
                 mode_out.get("OUTRANKED", {}).get("n", 0)),
              flush=True)

    out["measured_accuracy"] = mode_results

    # ---- Contrasts per case (reuses Step 3 per-node binary values; no
    # matcher re-runs) ----
    print("\n--- Step 4: Contrasts per case ---", flush=True)
    contrasts = {}
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        real_vals = list(case_accuracy["real"][case])
        no_edge_vals = list(case_accuracy["no_edge"][case])
        rewired_vals = {s: list(case_accuracy["rewired_s%d" % s][case])
                        for s in [0, 1, 2]}
        ps_vals = {s: list(case_accuracy["params_shuffled_s%d" % s][case])
                   for s in [0, 1, 2]}

        case_contrasts = {
            "real_minus_no_edge": _paired_interval(real_vals, no_edge_vals),
            "real_minus_rewired_s0": _paired_interval(real_vals, rewired_vals[0]),
            "real_minus_rewired_s1": _paired_interval(real_vals, rewired_vals[1]),
            "real_minus_rewired_s2": _paired_interval(real_vals, rewired_vals[2]),
            "real_minus_params_shuffled_s0": _paired_interval(real_vals, ps_vals[0]),
            "real_minus_params_shuffled_s1": _paired_interval(real_vals, ps_vals[1]),
            "real_minus_params_shuffled_s2": _paired_interval(real_vals, ps_vals[2]),
            "n": len(real_vals),
        }
        contrasts[case] = case_contrasts
        for k, v in case_contrasts.items():
            if k == "n":
                print("  %s n=%d" % (case, v), flush=True)
            elif v:
                print("  %s %-30s %.4f [%.4f, %.4f] hw=%.4f"
                      % (case, k, v["mean"], v["lo"], v["hi"], v["hw"]),
                      flush=True)

    # Share-weighted net effect
    net_real_no_edge = 0.0
    for case, ct in contrasts.items():
        if ct.get("real_minus_no_edge"):
            share = case_counts[case] / total if total else 0
            net_real_no_edge += share * ct["real_minus_no_edge"]["mean"]
    print("\n  Share-weighted net effect (real - no_edge): %.4f"
          % net_real_no_edge, flush=True)
    out["contrasts"] = contrasts
    out["net_effect_real_minus_no_edge"] = net_real_no_edge

    write_json(RESULTS_DIR / "t4" / "b1_placement.json", out)
    print("\n=== B1 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
