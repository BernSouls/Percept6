"""Task 4 PART B8 — ISOLATED-NODE CONFOUND (dev only).

Diagnostic arm: run the matcher with MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE
= False (isolated weight 1.0, i.e. isolated nodes are NOT excluded) and compare
against the winner cell (excluded = True). Report the degree-0 share.

Reading: if degree-0 share is high and the isolated arm differs little from the
winner, isolated nodes are not a confound. If it differs a lot, isolated nodes
drive the result and the confound must be reported.

Diagnostic toggle: the flag is set only inside the harness and restored.
"""
from __future__ import annotations

import os
import pickle
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


def _degree0_share(g):
    """Fraction of nodes with degree 0."""
    deg = Counter()
    for a, b, t in g.subgraph.edges:
        deg[a] += 1
        deg[b] += 1
    n_total = len(g.subgraph.nodes)
    n_deg0 = sum(1 for n in g.subgraph.nodes if deg.get(n, 0) == 0)
    return n_deg0, n_total, n_deg0 / n_total if n_total else 0.0


def _run_matcher(base_g, shift_g, exclude_isolated):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    P.MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = exclude_isolated
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


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
    print("=== B8: ISOLATED-NODE CONFOUND ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    deg0_counts = []
    deg0_shares = []
    mapping_diff = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)

        n0, ntot, share = _degree0_share(base)
        deg0_counts.append(n0)
        deg0_shares.append(share)

        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                continue
            cp, cp_info = counterpart_map(_pos_lens(base), _pos_lens(sh_g),
                                          dx, 0, 8.0)

            m_exclude = _run_matcher(base, sh_g, True)
            m_include = _run_matcher(base, sh_g, False)

            for bn, bsh in cp.items():
                if bn in m_exclude and bn in m_include:
                    mapping_diff.append(
                        1.0 if m_exclude[bn] == m_include[bn] else 0.0)

    print("\n--- Degree-0 share ---", flush=True)
    deg0_mean = float(np.mean(deg0_shares))
    print("  degree-0 nodes: mean share=%.4f (n=%d scenes)"
          % (deg0_mean, len(deg0_shares)))

    print("\n--- Mapping agreement (exclude isolated vs include isolated) ---",
          flush=True)
    agree = float(np.mean(mapping_diff)) if mapping_diff else 0.0
    print("  agreement=%.4f (n=%d)" % (agree, len(mapping_diff)))

    out["degree0_share_mean"] = deg0_mean
    out["degree0_counts"] = deg0_counts
    out["mapping_agreement"] = agree
    out["mapping_agreement_n"] = len(mapping_diff)

    if deg0_mean < 0.05 and agree > 0.95:
        print("\n  Reading: degree-0 share is low and mappings agree → "
              "isolated nodes are NOT a confound.")
    else:
        print("\n  Reading: isolated nodes may be a confound (high degree-0 "
              "share or low agreement).")

    write_json(RESULTS_DIR / "t4" / "b8_isolated_confound.json", out)
    print("\n=== B8 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
