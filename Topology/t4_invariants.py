"""Task 4 INVARIANTS — 0px matcher precision, coverage, determinism note.

Computes on dev (arm B1, shift 0 = identity):
  - 0px matcher precision = fraction of base nodes mapped to THEMSELVES by the
    matcher at shift 0. <1.000 is a finding, not a pass (A8 insertion).
  - Coverage (not deletion): fraction of content-partition base nodes that
    survive into the mapping, vs frozen ids. Dual denominator (pooled +
    class-filtered) and separate new-token count.
  - D1(0) ceiling (identity mapping accuracy).

Determinism (PYTHONHASHSEED 0 vs 1729) was measured in the prior pass and is
cited, not recomputed here.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
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
CLASS_FILTERED_KINDS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
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


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def main() -> int:
    print("=== INVARIANTS: 0px precision, coverage ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    identity_fracs = []      # fraction mapped to self
    coverage_pooled = []     # content nodes in mapping / content nodes total
    coverage_class = []
    new_token_counts = []
    d1_0 = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        # shift 0 = identity: match base against itself
        mapping = _run_matcher(base, base)
        base_pos = _pos_lens(base)

        n_total = len(base_pos)
        n_self = sum(1 for n in base_pos if mapping.get(n) == n)
        identity_fracs.append(n_self / n_total if n_total else 0.0)

        # content nodes
        content_nodes = [n for n in base_pos
                         if base_pos[n][2] in CONTENT_TOKEN_LENSES]
        content_in_map = sum(1 for n in content_nodes if n in mapping)
        coverage_pooled.append(
            content_in_map / len(content_nodes) if content_nodes else 0.0)

        class_nodes = [n for n in base_pos
                       if base_pos[n][2] in CLASS_FILTERED_KINDS]
        class_in_map = sum(1 for n in class_nodes if n in mapping)
        coverage_class.append(
            class_in_map / len(class_nodes) if class_nodes else 0.0)

        # new-token count: mapped nodes not in base (should be 0 at shift 0)
        new_tokens = sum(1 for v in mapping.values() if v not in base_pos)
        new_token_counts.append(new_tokens)

        # D1(0): fraction of content nodes mapped to their identity counterpart
        if content_nodes:
            correct = sum(1 for n in content_nodes if mapping.get(n) == n)
            d1_0.append(correct / len(content_nodes))

    print("\n--- 0px matcher precision (mapped to self) ---", flush=True)
    prec = float(np.mean(identity_fracs))
    print("  precision=%.4f (n=%d scenes)" % (prec, len(identity_fracs)))
    if prec < 1.0:
        print("  FINDING: precision < 1.000 — the matcher does NOT map every "
              "node to itself at shift 0 (not a pass).")

    print("\n--- Coverage (not deletion) ---", flush=True)
    cov_p = float(np.mean(coverage_pooled))
    cov_c = float(np.mean(coverage_class))
    print("  pooled content coverage=%.4f (n=%d)" % (cov_p, len(coverage_pooled)))
    print("  class-filtered coverage=%.4f (n=%d)" % (cov_c, len(coverage_class)))
    print("  new-token count mean=%.2f (n=%d)"
          % (float(np.mean(new_token_counts)), len(new_token_counts)))

    print("\n--- D1(0) ceiling (identity accuracy, content) ---", flush=True)
    d10 = float(np.mean(d1_0)) if d1_0 else 0.0
    print("  D1(0)=%.4f (n=%d)" % (d10, len(d1_0)))

    out["matcher_precision_0px"] = prec
    out["matcher_precision_0px_n"] = len(identity_fracs)
    out["coverage_pooled"] = cov_p
    out["coverage_class"] = cov_c
    out["new_token_count_mean"] = float(np.mean(new_token_counts))
    out["d1_0_content"] = d10
    out["precision_below_1_finding"] = prec < 1.0

    write_json(RESULTS_DIR / "t4" / "invariants.json", out)
    print("\n=== INVARIANTS complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
