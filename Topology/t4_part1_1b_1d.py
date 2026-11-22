"""Task 4 PART 1 — 1b (truth-free ambiguity gradients) + 1d (params contrasts).

Dev, PRIMARY, repaired map, per-kind radius. Shifts +2, -2.

1b: P(case c) by K_top (candidates within eps of best node_sim), margin
terciles, and K_self (same-lens nodes in its own graph with node_sim >= 1-eps).
eps from the q90 shortfall of true counterparts. Usable gradient: top bin's
(c)-rate >= 2x bottom bin's, with n. Stability across +2/-2.

1d: real - params_shuffled and real - real_params_dropped per case, intervals.
"""
from __future__ import annotations
import json, os, pickle, random, sys, tempfile
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
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
PRIMARY_KINDS = frozenset({"CONTOUR_PEAK", "PEAK_CHAIN", "RegionShape", "Corner"})
WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75, "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5, "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5, "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
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
    tmp = Path(tempfile.gettempdir()) / "t4_p1" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], dx))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(roll_image(arr, dx, 0)).save(tmp)
    e = dict(entry); e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _node_sim(sig_a, sig_b):
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


def _params_shuffled_base(g, seed):
    rng = random.Random(seed)
    ep = dict(g.subgraph.edge_params)
    by_fam = {}
    for k in ep:
        fam = str(getattr(k[2], "value", k[2]))
        by_fam.setdefault(fam, []).append(k)
    new_ep = {}
    for fam, keys in by_fam.items():
        vals = [ep[k] for k in keys]
        rng.shuffle(vals)
        for k, v in zip(keys, vals):
            new_ep[k] = v
    return replace(g, subgraph=replace(g.subgraph, edge_params=new_ep))


def _real_params_dropped_base(g):
    return replace(g, subgraph=replace(g.subgraph, edge_params={}))


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
    return {"mean": mean, "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
            "hw": (float(np.quantile(boots, 0.975))
                   - float(np.quantile(boots, 0.025))) / 2}


def main() -> int:
    print("=== PART 1: 1b ambiguity gradients + 1d params contrasts ===",
          flush=True)
    out = {}
    radius_by_lens = json.load(open(RESULTS_DIR / "t4" / "primary_radius_dev.json"))
    radius_by_lens = {k: (v if v is not None else 8.0) for k, v in radius_by_lens.items()}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # ---- 1b: ambiguity gradients ----
    print("\n--- 1b: truth-free ambiguity gradients ---", flush=True)
    # eps from q90 shortfall of true counterparts (1 - node_sim_true)
    shortfalls = []
    ktop_bins = defaultdict(lambda: Counter())  # bin -> case counts
    margin_tercile_cases = defaultdict(lambda: defaultdict(list))
    kself_bins = defaultdict(Counter)
    kself_vals = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s (1b)..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, ARM_B1, dx)
            sh_pos = _pos_lens(sh_g)
            cp, cp_info, _ = counterpart_map(
                base_pos, sh_pos, dx, 0, 8.0, radius_by_lens=radius_by_lens)
            sh_sigs_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                sh_sig = sh_g.sigs.get(nsh)
                if sh_sig is None:
                    continue
                sh_sigs_by_lens[str(sh_sig[0])].append((nsh, sh_sig[1]))
            # base same-lens nodes for K_self
            base_sigs_by_lens = defaultdict(list)
            for bn in base.subgraph.nodes:
                bs = base.sigs.get(bn)
                if bs is None:
                    continue
                base_sigs_by_lens[str(bs[0])].append((bn, bs[1]))

            for bn, (bx, by, blens) in base_pos.items():
                if blens not in PRIMARY_KINDS:
                    continue
                bsh = cp.get(bn)
                if bsh is None or bsh not in sh_pos:
                    continue
                b_sig = base.sigs.get(bn)
                sh_sig = sh_g.sigs.get(bsh)
                if b_sig is None or sh_sig is None:
                    continue
                true_lens = str(sh_sig[0])
                # node_sim to all same-lens shifted candidates
                cand_sims = [(nsh, _node_sim(b_sig[1], sv))
                             for nsh, sv in sh_sigs_by_lens.get(true_lens, [])]
                s_true = _node_sim(b_sig[1], sh_sig[1])
                shortfalls.append(1.0 - s_true)
                # case
                comp = [s for nsh, s in cand_sims if nsh != bsh]
                if not comp:
                    case = "STRICT-LEAD"
                else:
                    mx = max(comp)
                    T = sum(1 for s in comp if abs(s - mx) < TIE_TOL) + 1
                    if s_true > mx + TIE_TOL:
                        case = "STRICT-LEAD"
                    elif abs(s_true - mx) < TIE_TOL and T >= 2:
                        case = "EXACT-TIE"
                    else:
                        case = "OUTRANKED"
                # K_top: candidates within eps of best
                eps = 1e-3
                best = max(s for _, s in cand_sims)
                k_top = sum(1 for _, s in cand_sims if s >= best - eps)
                bin_k = min(k_top, 5)  # bucket 1..5+
                ktop_bins[bin_k][case] += 1
                # margin tercile
                if comp:
                    mx = max(comp)
                    margin = s_true - mx
                    margins_all = [s_true - s for s in comp]
                else:
                    margin = 1.0
                    margins_all = []
                # tercile of margin within this scene's margins (approx: global)
                margin_tercile_cases[case]["margin"].append(margin)
                # K_self: same-lens base nodes with node_sim >= 1-eps
                k_self = sum(1 for nb2, sv2 in base_sigs_by_lens.get(true_lens, [])
                             if nb2 != bn and _node_sim(b_sig[1], sv2) >= 1.0 - eps)
                kself_bins[min(k_self, 5)][case] += 1
                kself_vals.append(k_self)

    eps = float(np.quantile(shortfalls, 0.90)) if shortfalls else 1e-3
    print("  eps (q90 shortfall of true counterparts) = %.4f" % eps, flush=True)

    # P(c) by K_top bin
    print("\n  P(case c) by K_top bin:", flush=True)
    ktop_report = {}
    for bin_k in sorted(ktop_bins):
        c = ktop_bins[bin_k]
        tot = sum(c.values())
        pc = c["OUTRANKED"] / tot if tot else 0.0
        ktop_report[str(bin_k)] = {"n": tot, "P_c": pc,
                                   "counts": dict(c)}
        print("    K_top=%s: n=%d P(c)=%.4f" % (bin_k, tot, pc))
    # usable gradient: top bin P(c) >= 2x bottom bin
    top_bin = max(ktop_bins)
    bot_bin = min(ktop_bins)
    top_pc = ktop_report[str(top_bin)]["P_c"]
    bot_pc = ktop_report[str(bot_bin)]["P_c"]
    usable = top_pc >= 2 * bot_pc and ktop_report[str(top_bin)]["n"] >= 200
    print("  usable K_top gradient: top(%s)=%.4f bot(%s)=%.4f -> %s"
          % (top_bin, top_pc, bot_bin, bot_pc, usable))

    # margin terciles
    margins_all_vals = [v for case in margin_tercile_cases
                        for v in margin_tercile_cases[case]["margin"]]
    if margins_all_vals:
        t1, t2 = np.quantile(margins_all_vals, [1/3, 2/3])
        print("\n  margin terciles: t1=%.4f t2=%.4f" % (t1, t2), flush=True)

    # K_self
    print("\n  P(case c) by K_self bin:", flush=True)
    kself_report = {}
    for bin_k in sorted(kself_bins):
        c = kself_bins[bin_k]
        tot = sum(c.values())
        pc = c["OUTRANKED"] / tot if tot else 0.0
        kself_report[str(bin_k)] = {"n": tot, "P_c": pc, "counts": dict(c)}
        print("    K_self=%s: n=%d P(c)=%.4f" % (bin_k, tot, pc))
    ks_top = max(kself_bins); ks_bot = min(kself_bins)
    ks_usable = (kself_report[str(ks_top)]["P_c"] >= 2 * kself_report[str(ks_bot)]["P_c"]
                 and kself_report[str(ks_top)]["n"] >= 200)
    print("  usable K_self gradient: %s" % ks_usable)

    out["1b"] = {"eps": eps, "ktop": ktop_report, "ktop_usable": usable,
                 "kself": kself_report, "kself_usable": ks_usable,
                 "kself_mean": float(np.mean(kself_vals)) if kself_vals else 0}

    # ---- 1d: params contrasts ----
    print("\n--- 1d: real - params_shuffled, real - real_params_dropped ---",
          flush=True)
    case_accuracy = defaultdict(lambda: defaultdict(list))
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s (1d)..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, ARM_B1, dx)
            sh_pos = _pos_lens(sh_g)
            cp, cp_info, _ = counterpart_map(
                base_pos, sh_pos, dx, 0, 8.0, radius_by_lens=radius_by_lens)
            modes = {
                "real": base,
                "params_shuffled": _params_shuffled_base(base, 0),
                "real_params_dropped": _real_params_dropped_base(base),
            }
            for mode, b in modes.items():
                mapping = _run_matcher(b, sh_g)
                for bn, (bx, by, blens) in base_pos.items():
                    if blens not in PRIMARY_KINDS:
                        continue
                    bsh = cp.get(bn)
                    if bsh is None or bsh not in sh_pos:
                        continue
                    case_accuracy[mode]["all"].append(
                        1.0 if mapping.get(bn) == bsh else 0.0)
                    # recompute case for this node
                    b_sig = base.sigs.get(bn)
                    sh_sig = sh_g.sigs.get(bsh)
                    if b_sig is None or sh_sig is None:
                        continue
                    true_lens = str(sh_sig[0])
                    sh_sigs_by_lens = defaultdict(list)
                    for nsh in sh_g.subgraph.nodes:
                        ss = sh_g.sigs.get(nsh)
                        if ss is None:
                            continue
                        sh_sigs_by_lens[str(ss[0])].append((nsh, ss[1]))
                    len_list = sh_sigs_by_lens.get(true_lens, [])
                    s_true = _node_sim(b_sig[1], sh_sig[1])
                    comp = [_node_sim(b_sig[1], sv) for nsh, sv in len_list
                            if nsh != bsh]
                    if not comp:
                        case = "STRICT-LEAD"
                    else:
                        mx = max(comp)
                        T = sum(1 for s in comp if abs(s - mx) < TIE_TOL) + 1
                        if s_true > mx + TIE_TOL:
                            case = "STRICT-LEAD"
                        elif abs(s_true - mx) < TIE_TOL and T >= 2:
                            case = "EXACT-TIE"
                        else:
                            case = "OUTRANKED"
                    case_accuracy[mode][case].append(
                        1.0 if mapping.get(bn) == bsh else 0.0)

    contrasts = {}
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED", "all"]:
        real_vals = case_accuracy["real"][case]
        ps_vals = case_accuracy["params_shuffled"][case]
        rpd_vals = case_accuracy["real_params_dropped"][case]
        contrasts[case] = {
            "real_minus_params_shuffled": _paired_interval(real_vals, ps_vals),
            "real_minus_real_params_dropped": _paired_interval(real_vals, rpd_vals),
            "n": len(real_vals),
        }
        print("  %s n=%d: real-ps=%s real-rpd=%s" %
              (case, len(real_vals),
               contrasts[case]["real_minus_params_shuffled"],
               contrasts[case]["real_minus_real_params_dropped"]), flush=True)
    out["1d"] = contrasts

    write_json(RESULTS_DIR / "t4" / "part1_1b_1d.json", out)
    print("\n=== PART 1 (1b+1d) complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
