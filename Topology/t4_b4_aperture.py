"""Task 4 PART B4 — APERTURE ON MIS-MAPPED CONTENT NODES (dev only).

For content-class base nodes that the matcher MIS-mapped (mapped to a non-
true counterpart), measure:
  - the tangent (direction) from chain tokens: whether the mismapped node's
    tangent aligns ALONG the shift direction or ACROSS it.
  - accuracy by hops-to-event: graph distance from the node to an anchor
    (event) token, bucketed, and the mapping accuracy per bucket.

Reading: if mismapped nodes concentrate at certain apertures or hops, the
aperture/event-distance is a predictor of mis-mapping.

Uses cached dev + shifted graphs; matcher run once per scene (real mode).
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


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _bfs_hops(nodes, edges, source):
    """BFS hop distance from source to each node."""
    from collections import deque
    adj = defaultdict(set)
    for a, b, t in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _tangent_align(pos_a, pos_b, dx):
    """Classify the displacement as along (parallel to shift) or across.

    pos_a, pos_b are (x, y). Shift is horizontal (dx, 0). The displacement
    vector (dx_b - dx_a, dy_b - dy_a): along if |dy| < |dx|, else across.
    """
    dxv = pos_b[0] - pos_a[0]
    dyv = pos_b[1] - pos_a[1]
    if abs(dxv) > abs(dyv):
        return "along"
    return "across"


def main() -> int:
    print("=== B4: APERTURE ON MIS-MAPPED CONTENT NODES ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    along_correct = []
    along_total = []
    across_correct = []
    across_total = []
    hop_buckets = defaultdict(lambda: [0, 0])  # hop -> [correct, total]
    mis_total = 0
    mis_along = 0
    mis_across = 0

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)

        # BFS from an anchor (first content node) for hop distance
        content_nodes = [n for n in base_pos if base_pos[n][2] in D_CLASS]
        if not content_nodes:
            continue
        anchor = content_nodes[0]
        hops = _bfs_hops(base.subgraph.nodes, base.subgraph.edges, anchor)

        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                continue
            sh_pos = _pos_lens(sh_g)
            cp, cp_info = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)
            mapping = _run_matcher(base, sh_g)

            for bn, (bx, by, blens) in base_pos.items():
                if blens not in D_CLASS:
                    continue
                bsh = cp.get(bn)
                if bsh is None or bsh not in sh_pos:
                    continue
                correct = (mapping.get(bn) == bsh)
                align = _tangent_align(
                    (base_pos[bn][0], base_pos[bn][1]),
                    (sh_pos[bsh][0], sh_pos[bsh][1]), dx)
                if align == "along":
                    along_total.append(1)
                    along_correct.append(1.0 if correct else 0.0)
                else:
                    across_total.append(1)
                    across_correct.append(1.0 if correct else 0.0)
                if not correct:
                    mis_total += 1
                    if align == "along":
                        mis_along += 1
                    else:
                        mis_across += 1
                h = hops.get(bn, -1)
                if h < 0:
                    h = "inf"
                hop_buckets[h][1] += 1
                if correct:
                    hop_buckets[h][0] += 1

    print("\n--- Aperture (along vs across) accuracy ---", flush=True)
    along_acc = float(np.mean(along_correct)) if along_correct else 0.0
    across_acc = float(np.mean(across_correct)) if across_correct else 0.0
    print("  along:   acc=%.4f n=%d" % (along_acc, len(along_correct)))
    print("  across:  acc=%.4f n=%d" % (across_acc, len(across_correct)))

    print("\n--- Mis-mapped aperture split ---", flush=True)
    print("  mis-mapped total=%d, along=%d (%.3f), across=%d (%.3f)"
          % (mis_total, mis_along,
             mis_along / mis_total if mis_total else 0,
             mis_across, mis_across / mis_total if mis_total else 0))

    print("\n--- Accuracy by hops-to-event ---", flush=True)
    hop_report = {}
    for h, (c, t) in sorted(hop_buckets.items(),
                            key=lambda x: (str(x[0]) == "inf",
                                           str(x[0]))):
        acc = c / t if t else 0.0
        hop_report[str(h)] = {"correct": c, "total": t, "accuracy": acc}
        print("  hop=%s: acc=%.4f (%d/%d)" % (h, acc, c, t))

    out["along_accuracy"] = along_acc
    out["along_n"] = len(along_correct)
    out["across_accuracy"] = across_acc
    out["across_n"] = len(across_correct)
    out["mis_total"] = mis_total
    out["mis_along"] = mis_along
    out["mis_across"] = mis_across
    out["hop_buckets"] = hop_report

    write_json(RESULTS_DIR / "t4" / "b4_aperture.json", out)
    print("\n=== B4 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
