"""Task 4 PART B5 — CREDIT DECOMPOSITION (dev only; read-only instrumentation).

For each base node, decompose the matcher's credit into:
  - node_sim: the raw _sig_similarity to the TRUE counterpart and to the
    MAPPED counterpart.
  - edge_credit: the edge-alignment contribution (number of matching edges
    incident to the mapped pair, vs the true pair).

Report per case (a/b/c): mean node_sim(true), mean node_sim(mapped), mean
edge-credit(mapped), mean edge-credit(true), and the "denominator penalty" on
case (a) = whether edges reduce the effective node credit (the P3-5 trigger
input: denominator penalty on (a)).

Read-only: the matcher is run as-is; no production change.
"""
from __future__ import annotations

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
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

D_CLASS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                     "RegionShape", "Orientation"})
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


def _edge_degree(g, node):
    """Number of edges incident to node (edge credit proxy)."""
    return sum(1 for e in g.subgraph.edges if node in e[:2])


def _case_of(b_sig, sh_sig, sh_sigs_by_lens, true_sh_node, true_lens):
    len_list = sh_sigs_by_lens.get(true_lens, [])
    true_sig = None
    for sh_node, sh_sig_v in len_list:
        if sh_node == true_sh_node:
            true_sig = sh_sig_v
            break
    if true_sig is None:
        return "ABSENT"
    s_true = _node_sim(b_sig, true_sig)
    comp = [(sh_node, s) for sh_node, sv in len_list
            if sh_node != true_sh_node
            for s in [_node_sim(b_sig, sv)]]
    if not comp:
        return "STRICT-LEAD"
    mx = max(s for _, s in comp)
    T = sum(1 for _, s in comp if abs(s - mx) < TIE_TOL) + 1
    if s_true > mx + TIE_TOL:
        return "STRICT-LEAD"
    elif abs(s_true - mx) < TIE_TOL and T >= 2:
        return "EXACT-TIE"
    else:
        return "OUTRANKED"


def main() -> int:
    print("=== B5: CREDIT DECOMPOSITION ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    acc = defaultdict(lambda: defaultdict(list))  # case -> field -> vals

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)

        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                continue
            sh_pos = _pos_lens(sh_g)
            cp, cp_info = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)
            mapping = _run_matcher(base, sh_g)

            sh_sigs_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                sh_sig = sh_g.sigs.get(nsh)
                if sh_sig is None:
                    continue
                sh_sigs_by_lens[str(sh_sig[0])].append((nsh, sh_sig[1]))

            for bn, (bx, by, blens) in base_pos.items():
                if blens not in D_CLASS:
                    continue
                bsh = cp.get(bn)
                if bsh is None or bsh not in sh_pos:
                    continue
                b_sig = base.sigs.get(bn)
                sh_sig = sh_g.sigs.get(bsh)
                if b_sig is None or sh_sig is None:
                    continue
                true_lens = str(sh_sig[0])
                case = _case_of(b_sig[1], sh_sig[1], sh_sigs_by_lens,
                                bsh, true_lens)
                if case == "ABSENT":
                    continue
                mapped = mapping.get(bn)
                # node_sim to true
                s_true = _node_sim(b_sig[1], sh_sig[1])
                # node_sim to mapped
                s_mapped = 0.0
                if mapped is not None:
                    ms_sig = sh_g.sigs.get(mapped)
                    if ms_sig is not None and ms_sig[0] == b_sig[0]:
                        s_mapped = _node_sim(b_sig[1], ms_sig[1])
                # edge credit: incident edges of mapped vs true
                ec_mapped = _edge_degree(sh_g, mapped) if mapped is not None else 0
                ec_true = _edge_degree(sh_g, bsh)

                acc[case]["node_sim_true"].append(s_true)
                acc[case]["node_sim_mapped"].append(s_mapped)
                acc[case]["edge_credit_mapped"].append(ec_mapped)
                acc[case]["edge_credit_true"].append(ec_true)
                acc[case]["correct"].append(1.0 if mapped == bsh else 0.0)

    print("\n--- Credit decomposition per case ---", flush=True)
    report = {}
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        if case not in acc or not acc[case]["node_sim_true"]:
            continue
        r = {}
        for field in ["node_sim_true", "node_sim_mapped",
                      "edge_credit_mapped", "edge_credit_true", "correct"]:
            vals = acc[case][field]
            r[field] = {"mean": float(np.mean(vals)), "n": len(vals)}
        # denominator penalty on (a): node_sim_mapped < node_sim_true means
        # edges pulled the match to a lower-node-sim node (penalty).
        ns_true = r["node_sim_true"]["mean"]
        ns_map = r["node_sim_mapped"]["mean"]
        r["denominator_penalty"] = ns_map - ns_true
        r["denominator_penalty_n"] = r["node_sim_true"]["n"]
        report[case] = r
        print("  %s (n=%d):" % (case, r["node_sim_true"]["n"]))
        print("    node_sim_true=%.4f node_sim_mapped=%.4f (penalty=%+.4f)"
              % (ns_true, ns_map, r["denominator_penalty"]))
        print("    edge_credit_mapped=%.2f edge_credit_true=%.2f"
              % (r["edge_credit_mapped"]["mean"],
                 r["edge_credit_true"]["mean"]))
        print("    correct=%.4f" % r["correct"]["mean"])

    # P3-5 trigger input: denominator penalty on (a)
    pen_a = report.get("STRICT-LEAD", {}).get("denominator_penalty")
    out["decomposition"] = report
    out["denominator_penalty_on_a"] = pen_a
    if pen_a is not None and pen_a < 0:
        out["p3_5_condition_denominator_penalty"] = True
        print("\n  P3-5 trigger: denominator penalty on (a) present "
              "(edges pull to lower-node-sim node).")
    else:
        out["p3_5_condition_denominator_penalty"] = False
        print("\n  P3-5 trigger: no denominator penalty on (a).")

    write_json(RESULTS_DIR / "t4" / "b5_credit.json", out)
    print("\n=== B5 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
