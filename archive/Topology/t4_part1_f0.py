"""Task 4 PART 1 baseline + PART 2 F0 arms on PRIMARY (dev).

For beta in {0.25, 0.50, 0.75} (0.50 = unmodified reference):
  - real matcher (with edges) vs no_edge matcher (edges removed) per case
  - case accuracy (correct among nodes with counterpart), D1 = matcher_accuracy
  - share-weighted net (real - no_edge), bootstrap interval
  - case (c) OUTRANKED lift (real - no_edge) + interval (ABSENT if n<200)

F0 = MATCH_SYSTEMATICITY_WEIGHT (beta), pre-registered three fixed values.
"""
from __future__ import annotations
import json, os, pickle, sys, tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402
from t4_shift import counterpart_map, matcher_accuracy  # noqa: E402
PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
PRIMARY_KINDS = frozenset({"CONTOUR_PEAK", "PEAK_CHAIN", "RegionShape", "Corner"})
TIE_TOL = 1e-9

WINNER = {
    "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
    "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
}
BETAS = [0.25, 0.50, 0.75]


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_dev_graph(name):
    with open(PKL_DIR / ("B1_%s.pkl" % name), "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _build_shifted(entry, dx):
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARM_B1)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_f0" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]), entry["view"], dx))
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


def _no_edge_base(g):
    sg = replace(g.subgraph, edges=frozenset(), edge_params={})
    return replace(g, subgraph=sg)


def _run_matcher(base_g, shift_g, beta):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    P.MATCH_SYSTEMATICITY_WEIGHT = float(beta)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _classify(b_sig, sh_sig, len_list, bsh):
    s_true = _node_sim(b_sig[1], sh_sig[1])
    comp = [_node_sim(b_sig[1], sv) for nsh, sv in len_list if nsh != bsh]
    if not comp:
        return "STRICT-LEAD", s_true
    mx = max(comp)
    T = sum(1 for s in comp if abs(s - mx) < TIE_TOL) + 1
    if s_true > mx + TIE_TOL:
        return "STRICT-LEAD", s_true
    elif abs(s_true - mx) < TIE_TOL and T >= 2:
        return "EXACT-TIE", s_true
    else:
        return "OUTRANKED", s_true


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
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return {"mean": mean, "lo": lo, "hi": hi, "hw": (hi - lo) / 2}


def main() -> int:
    print("=== PART 1 baseline + F0 arms on PRIMARY ===", flush=True)
    out = {}
    radius_by_lens = json.load(open(RESULTS_DIR / "t4" / "primary_radius_dev.json"))
    radius_by_lens = {k: (v if v is not None else 8.0) for k, v in radius_by_lens.items()}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # Gather per-node truth: (scene, shift, node) -> (case, is_cp, base_pos, shift_pos)
    # We store per beta the real/no_edge correctness per node.
    records = []  # (name, dx, bn, bsh, case, is_cp)
    sh_sigs_cache = {}  # (name, dx) -> (sh_g, cp, base_pos)
    base_pos_cache = {}
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        base = _load_dev_graph(name)
        base_pos = _pos_lens(base)
        base_pos_cache[name] = base
        print("  %s..." % name, flush=True)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, dx)
            sh_pos = _pos_lens(sh_g)
            sh_sigs_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                ss = sh_g.sigs.get(nsh)
                if ss is None:
                    continue
                sh_sigs_by_lens[str(ss[0])].append((nsh, ss[1]))
            cp, cp_info, _ = counterpart_map(
                base_pos, sh_pos, dx, 0, 8.0, radius_by_lens=radius_by_lens)
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
                case, _ = _classify(b_sig, sh_sig,
                                    sh_sigs_by_lens.get(true_lens, []), bsh)
                records.append((name, dx, bn, bsh, case, True))
            sh_sigs_cache[(name, dx)] = (sh_g, cp, base_pos)

    print("  PRIMARY nodes with counterpart: %d" % len(records), flush=True)

    case_share = defaultdict(int)
    for r in records:
        case_share[r[4]] += 1
    shares = {c: case_share[c] / len(records) for c in case_share}

    # For each beta: real and no_edge mappings per (name, dx)
    for beta in BETAS:
        print("\n--- beta=%.2f ---" % beta, flush=True)
        real_corr = defaultdict(list)
        ne_corr = defaultdict(list)
        d1_real = {}
        d1_ne = {}
        rec_index = {}
        for idx, (name, dx, bn, bsh, case, is_cp) in enumerate(records):
            rec_index.setdefault((name, dx), []).append((bn, bsh, case))
        for (name, dx), node_list in rec_index.items():
            base = base_pos_cache[name]
            sh_g, cp, base_pos = sh_sigs_cache[(name, dx)]
            real_map = _run_matcher(base, sh_g, beta)
            ne_base = _no_edge_base(base)
            ne_map = _run_matcher(ne_base, sh_g, beta)
            # D1 = matcher_accuracy (no-counterpart counts wrong) over base nodes
            d1_real[(name, dx)] = matcher_accuracy(base_pos, real_map, cp, PRIMARY_KINDS)["acc"]
            d1_ne[(name, dx)] = matcher_accuracy(base_pos, ne_map, cp, PRIMARY_KINDS)["acc"]
            for bn, bsh, case in node_list:
                real_corr[case].append(1.0 if real_map.get(bn) == bsh else 0.0)
                ne_corr[case].append(1.0 if ne_map.get(bn) == bsh else 0.0)

        # per-case intervals + pooled
        per_case = {}
        net_share = 0.0
        for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
            rv = real_corr[case]; nv = ne_corr[case]
            ci = _paired_interval(rv, nv)
            per_case[case] = {"n": len(rv), "real_mean": float(np.mean(rv)) if rv else None,
                             "no_edge_mean": float(np.mean(nv)) if nv else None,
                             "real_minus_no_edge": ci, "share": shares.get(case, 0)}
            if ci is not None:
                net_share += shares.get(case, 0) * ci["mean"]
            print("    %s n=%d: real=%.4f no_edge=%.4f delta=%s"
                  % (case, len(rv),
                     per_case[case]["real_mean"] or 0,
                     per_case[case]["no_edge_mean"] or 0, ci), flush=True)
        # pooled over all PRIMARY nodes
        all_real = [v for case in real_corr for v in real_corr[case]]
        all_ne = [v for case in ne_corr for v in ne_corr[case]]
        pooled = _paired_interval(all_real, all_ne)
        d1_real_vals = list(d1_real.values())
        d1_ne_vals = list(d1_ne.values())
        d1_delta = _paired_interval(d1_real_vals, d1_ne_vals)
        print("    POOLED n=%d delta=%s" % (len(all_real), pooled), flush=True)
        print("    D1 real=%.4f no_edge=%.4f delta=%s"
              % (np.mean(d1_real_vals), np.mean(d1_ne_vals), d1_delta), flush=True)
        net_ci = _paired_interval(all_real, all_ne)
        out["beta_%.2f" % beta] = {
            "per_case": per_case, "pooled_delta": pooled,
            "share_weighted_net_mean": net_share,
            "d1_real_mean": float(np.mean(d1_real_vals)),
            "d1_no_edge_mean": float(np.mean(d1_ne_vals)),
            "d1_delta": d1_delta,
        }

    out["reference_beta"] = 0.50
    write_json(RESULTS_DIR / "t4" / "part1_f0_baseline.json", out)
    print("\n=== baseline + F0 arms complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
