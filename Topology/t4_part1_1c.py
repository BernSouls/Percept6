"""Task 4 PART 1 — 1c JITTER CONTROL (canonical-hash tie-order envelope).

Relabel base+shift graphs with random node-id/order permutations (the 0e
mechanism that changes 3.06% of mappings), re-measure the pooled real-no_edge
delta on PRIMARY, and report the envelope (min/median/max) for the reference
beta=0.50 and the best arm beta=0.25. Bounds how much the headline delta moves
under tie-order jitter.
"""
from __future__ import annotations
import json, os, pickle, random, sys, tempfile
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
from t4_part0e_order import _relabel  # noqa: E402
PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
PRIMARY_KINDS = frozenset({"CONTOUR_PEAK", "PEAK_CHAIN", "RegionShape", "Corner"})
TIE_TOL = 1e-9
WINNER = {
    "MATCH_UNMATCHED_COST": 0.15, "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1, "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
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


def _load_dev_graph(name):
    with open(PKL_DIR / ("B1_%s.pkl" % name), "rb") as fh:
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


def _no_edge_base(g):
    sg = replace(g.subgraph, edges=frozenset(), edge_params={})
    return replace(g, subgraph=sg)


def _run_matcher(base_g, shift_g, beta):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    P.MATCH_SYSTEMATICITY_WEIGHT = float(beta)
    from matching.joint_mcs import joint_mcs
    return joint_mcs(base_g.subgraph, shift_g.subgraph,
                     base_g.sigs, shift_g.sigs).mapping


def main() -> int:
    print("=== PART 1 — 1c jitter envelope ===", flush=True)
    out = {}
    radius_by_lens = json.load(open(RESULTS_DIR / "t4" / "primary_radius_dev.json"))
    radius_by_lens = {k: (v if v is not None else 8.0) for k, v in radius_by_lens.items()}
    entries = _dev_entries()[:8]  # subset for the envelope
    seeds = [42, 7, 1234]
    print("scenes: %d, seeds: %s" % (len(entries), seeds), flush=True)

    envelope = {}
    for beta in [0.50, 0.25]:
        deltas = []
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            base = _load_dev_graph(name)
            for dx in [2, -2]:
                sh_g = _load_shifted(name, dx)
                if sh_g is None:
                    continue
                # truth positions on the ORIGINAL graphs
                base_pos = _pos_lens(base)
                sh_pos = _pos_lens(sh_g)
                cp, _, _ = counterpart_map(
                    base_pos, sh_pos, dx, 0, 8.0, radius_by_lens=radius_by_lens)
                nodes = sorted(set(base.subgraph.nodes) | set(sh_g.subgraph.nodes))
                for seed in seeds:
                    rng = random.Random(seed)
                    perm = {n: i for i, n in enumerate(rng.sample(nodes, len(nodes)))}
                    b_p, _ = _relabel(base, perm)
                    s_p, _ = _relabel(sh_g, perm)
                    # recompute cp on relabeled positions
                    b_pos_p = _pos_lens(b_p)
                    s_pos_p = _pos_lens(s_p)
                    cp_p, _, _ = counterpart_map(
                        b_pos_p, s_pos_p, dx, 0, 8.0, radius_by_lens=radius_by_lens)
                    # nodes to score: PRIMARY kinds with counterpart
                    score = []
                    for bn, (bx, by, blens) in b_pos_p.items():
                        if blens not in PRIMARY_KINDS:
                            continue
                        bsh = cp_p.get(bn)
                        if bsh is None or bsh not in s_pos_p:
                            continue
                        score.append((bn, bsh))
                    if not score:
                        continue
                    real_map = _run_matcher(b_p, s_p, beta)
                    ne_map = _run_matcher(_no_edge_base(b_p), s_p, beta)
                    rv = [1.0 if real_map.get(bn) == bsh else 0.0 for bn, bsh in score]
                    nv = [1.0 if ne_map.get(bn) == bsh else 0.0 for bn, bsh in score]
                    deltas.append(float(np.mean(rv) - np.mean(nv)))
        arr = np.array(deltas) if deltas else np.array([0.0])
        envelope["beta_%.2f" % beta] = {
            "n_runs": int(len(arr)),
            "min": float(arr.min()), "median": float(np.median(arr)),
            "max": float(arr.max()), "mean": float(arr.mean()),
            "stdev": float(arr.std()),
        }
        print("  beta=%.2f: runs=%d min=%.4f med=%.4f max=%.4f (stdev %.4f)"
              % (beta, len(arr), arr.min(), np.median(arr), arr.max(), arr.std()),
              flush=True)
    out["1c_jitter_envelope"] = envelope
    write_json(RESULTS_DIR / "t4" / "part1_1c_jitter.json", out)
    print("\n=== 1c complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
