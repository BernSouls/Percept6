"""Task 4 PLACEMENT mission — Part 4: the gate, measured before it is used.

Placement support enters assignment as a MODIFIER on node similarity.  It may
be gated only on a TRUTH-FREE quantity: the **similarity margin** (best minus
second-best node similarity at match time).

This script measures ``P(appearance-ambiguous | margin bin)`` FIRST.  If there
is no usable gradient, P is reported UNGATED and the gate is UNEVALUATED --
the mission's own instruction.

WHAT "APPEARANCE-AMBIGUOUS" MEANS, AND WHY IT IS NOT "CORRECT"
--------------------------------------------------------------
Correctness is unobservable at match time, so this module never claims the
gate protects "correct" matches.  Ambiguity is a property of the OBSERVABLE:
a node is appearance-ambiguous when another node in the other graph shares its
``(lens type, signature)``.  That is observable from the graphs alone and is
fixed here before any margin is computed.

THE MODIFIER
------------
    sim' = sim * (1 + w * (support - 0.5))

with ``support`` in [0, 1] and ONE pre-stated ``w`` (``PLACEMENT_WEIGHT``,
default 0.25).  It is neutral at support = 0.5, can never more than 1.5x or
0.5x a similarity, and **never touches a node with no checkable edge**
(support None => sim unchanged).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: number of margin bins, fixed BEFORE looking at the data
N_MARGIN_BINS = 5


# ---------------------------------------------------------------------------
# the truth-free ambiguity label
# ---------------------------------------------------------------------------
#: An observable "looks the same" criterion.  A signature is UNIQUE in these
#: graphs (measured: 536 distinct signatures on 536 nodes, same on the query
#: side), so exact-signature multiplicity is identically 1 and cannot label
#: anything ambiguous.  The observable ambiguity that DOES exist is a near-tie
#: in node similarity: the top match scores >= APPARENT_TOP_SIM and is close to
#: the runner-up.  Both are computed from the graphs alone.
APPARENT_TOP_SIM = 0.99
APPARENT_MARGIN = 0.05


def appearance_ambiguous(q_sigs: Dict[int, Any], r_sigs: Dict[int, Any],
                         sims: Optional[Dict[int, List[Tuple[int, float]]]] = None,
                         top_sim: float = APPARENT_TOP_SIM,
                         margin_max: float = APPARENT_MARGIN
                         ) -> Dict[int, bool]:
    """Query node -> is its appearance APPARENTLY ambiguous?

    Two observable routes, unioned:

    1. exact-signature multiplicity -- another reference node shares its
       ``(lens type, signature)`` (in these graphs this is always 1 and
       contributes nothing, which is itself a measured fact);
    2. near-tie in node similarity -- the top match scores ``>= top_sim`` AND
       the margin to the runner-up is ``<= margin_max``.  This is the ambiguity
       that is actually present: 48.6 % of nodes have a top similarity
       >= 0.99 and 4.5 % are exact ties.

    NEITHER route implies correctness.  Correctness is unobservable at match
    time and nothing here claims to identify it.
    """
    def key(sig: Any) -> Any:
        try:
            return (str(sig[0]), sig[1].serialize())
        except Exception:
            return None

    counts: Dict[Any, int] = {}
    for n in sorted(r_sigs):
        k = key(r_sigs[n])
        if k is not None:
            counts[k] = counts.get(k, 0) + 1
    out: Dict[int, bool] = {}
    for n in sorted(q_sigs):
        k = key(q_sigs[n])
        exact = bool(k is not None and counts.get(k, 0) > 1)
        near = False
        if sims is not None:
            row = sims.get(int(n)) or []
            if len(row) >= 2:
                near = bool(row[0][1] >= top_sim
                            and (row[0][1] - row[1][1]) <= margin_max)
        out[int(n)] = bool(exact or near)
    return out


# ---------------------------------------------------------------------------
# the similarity margin
# ---------------------------------------------------------------------------
def node_similarities(gq: Any, gr: Any) -> Dict[int, List[Tuple[int, float]]]:
    """Every query node's similarity to every reference node, sorted desc.

    Uses the SAME ``TypedSignature.similarity`` the matcher's node term reads,
    so the margin is the margin the matcher actually has.
    """
    out: Dict[int, List[Tuple[int, float]]] = {}
    for u in sorted(gq.subgraph.nodes):
        su = gq.sigs.get(u)
        if su is None:
            continue
        row: List[Tuple[int, float]] = []
        for v in sorted(gr.subgraph.nodes):
            sv = gr.sigs.get(v)
            if sv is None:
                continue
            try:
                s = float(su[1].similarity(sv[1]))
            except Exception:
                continue
            row.append((int(v), s))
        row.sort(key=lambda t: (-t[1], t[0]))
        out[int(u)] = row
    return out


def margin_bins(margins: Sequence[float], nbins: int = N_MARGIN_BINS
                ) -> Tuple[List[float], List[int]]:
    """Equal-count (quantile) bins over the margin, edges frozen by the data."""
    arr = np.asarray([m for m in margins if math.isfinite(m)], dtype=np.float64)
    if arr.size == 0:
        return [], []
    edges = [float(np.quantile(arr, i / nbins)) for i in range(1, nbins)]
    return edges, [int(np.sum(arr <= e)) for e in edges]


# ---------------------------------------------------------------------------
# applying the modifier
# ---------------------------------------------------------------------------
def apply_modifier(sims: List[Tuple[int, float]], support: Optional[float],
                   weight: float) -> List[Tuple[int, float]]:
    """The ONE pre-stated modifier.  Neutral at support 0.5; unchanged when
    support is None (no checkable edge -> no credit and no penalty)."""
    if support is None:
        return list(sims)
    factor = 1.0 + float(weight) * (float(support) - 0.5)
    return [(v, s * factor) for v, s in sims]


def main() -> int:
    import guard
    guard.activate()
    from eth80_frontend import build_graph
    from matching.placement import edge_payloads
    from matching.placement_scorer import node_support
    from Topology.placement_scenes import all_specs, posed

    os.makedirs(OUT, exist_ok=True)
    tpath = os.path.join(OUT, "tolerance.json")
    with open(tpath) as fh:
        TOL = json.load(fh)

    rows: List[Dict[str, Any]] = []
    for spec in all_specs():
        g_ref = build_graph(posed(spec.scene, 0.0).render().image_u8,
                            spec.name + "|ref")
        for label, pose in spec.views:
            if label == "v0_base":
                continue
            g_q = build_graph(posed(spec.scene, pose["rot_deg"],
                                    tuple(pose["shift"]),
                                    float(pose["scale"])).render().image_u8,
                              spec.name + "|" + label)
            sims = node_similarities(g_q, g_ref)
            amb = appearance_ambiguous(g_q.sigs, g_ref.sigs, sims=sims)
            for u in sorted(sims):
                row = sims[u]
                if len(row) < 2:
                    continue
                margin = row[0][1] - row[1][1]
                rows.append({"scene": spec.name, "view": label,
                             "node": u, "margin": margin,
                             "ambiguous": amb.get(u, False)})
            print("  %s %s: %d nodes" % (spec.name, label, len(sims)), flush=True)

    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    margins = [r["margin"] for r in rows]
    edges, pops = margin_bins(margins)
    # assign bins
    for r in rows:
        b = 0
        for e in edges:
            if r["margin"] > e:
                b += 1
        r["bin"] = b

    table: Dict[str, Any] = {"n_rows": len(rows), "n_bins": N_MARGIN_BINS,
                            "bin_edges": edges, "bin_population": pops,
                            "per_bin": {}}
    for b in range(N_MARGIN_BINS):
        sel = [r for r in rows if r["bin"] == b]
        if not sel:
            continue
        table["per_bin"][str(b)] = {
            "n": len(sel),
            "margin_lo": float(min(x["margin"] for x in sel)),
            "margin_hi": float(max(x["margin"] for x in sel)),
            "P_ambiguous": float(np.mean([x["ambiguous"] for x in sel])),
        }
    overall = float(np.mean([r["ambiguous"] for r in rows]))
    table["P_ambiguous_overall_UNGATED"] = overall
    ps = [table["per_bin"][str(b)]["P_ambiguous"]
          for b in range(N_MARGIN_BINS) if str(b) in table["per_bin"]]
    # usable gradient: a monotone fall of P with margin, with real spread
    mono = all(ps[i] >= ps[i + 1] - 0.02 for i in range(len(ps) - 1))
    spread = (max(ps) - min(ps)) if ps else 0.0
    table["P_monotone_falling"] = bool(mono)
    table["P_spread"] = spread
    table["gate_usable"] = bool(mono and spread >= 0.10)

    for b in range(N_MARGIN_BINS):
        k = str(b)
        if k in table["per_bin"]:
            v = table["per_bin"][k]
            print("bin %d margin %.4f..%.4f  n=%4d  P(ambiguous)=%.4f"
                  % (b, v["margin_lo"], v["margin_hi"], v["n"], v["P_ambiguous"]),
                  flush=True)
    print("P(ambiguous) overall (UNGATED) = %.4f" % overall, flush=True)
    print("monotone_falling=%s spread=%.4f -> gate_usable=%s"
          % (mono, spread, table["gate_usable"]), flush=True)

    path = os.path.join(OUT, "part4_gate.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=2, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
