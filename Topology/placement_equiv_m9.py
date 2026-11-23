"""MISSION 9 STEP 1 -- equivalence: do the deployed path and the harness CHOOSE alike?

The deployed scorer is a candidate-level re-scoring of the matcher's own seed
matrix.  Before it may be believed it has to reproduce, exactly, the decision the
harness made under the SAME candidates and the SAME frozen numbers: for every FIT
node with candidates, argmax over ((1-b)*sim + b*P_plus_ctx) must pick the node
the harness picked.  Any disagreement is a wiring bug, not a modelling result --
the candidates, the mapping and the coefficients are supposed to be identical.

Three things are compared per node:
  harness     Topology/placement_m3.run_check's P_plus_ctx pick
  deployed    matching.placement_matcher.argmax_mapping  (same rows, same sims)
  matrix      the matcher's seed matrix after rescore_matrix (row argmax)

Reported: agreement, and the first disagreements with their score gaps.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def pose_fn(spec: Any, view: Dict[str, Any]):
    from Topology.placement_dev5 import posed5
    pv = {"kind": view.get("kind"),
          "rot_deg": float(view.get("rot_deg", 0.0)),
          "shift": tuple(view.get("shift", (0.0, 0.0))),
          "scale": float(view.get("scale", 1.0))}
    return posed5(spec, view), pv


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="m9_equiv")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import placement_edges as PE
    from matching.placement_scorer import incident_index, payload_index
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from core.place_key import place_lenses
    from types import SimpleNamespace
    import pickle

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    M3.ORACLE = False
    with open(os.path.join(OUT, "_fitcache", "m8_step2_fit.pkl"), "rb") as fh:
        F = pickle.load(fh)
    fit, pfit = F["fit"], F["pfit"]
    M3.A_E, M3.A_G = float(F["a_exist"]), float(F["a_geom"])
    M3.CTX_MARGIN = float(F["ctx_margin"])
    M3.CTX_ITERS = int(F["ctx_iters"])
    M3.CTX_MUTUAL = bool(F["ctx_mutual"])

    # the deployed module must be reading the SAME numbers
    b = PM.frozen()
    assert abs(b["_a_exist"] - M3.A_E) < 1e-12 and abs(b["_a_geom"] - M3.A_G) < 1e-12
    assert b["iteration"]["iters"] == M3.CTX_ITERS
    assert abs(b["iteration"]["margin"] - M3.CTX_MARGIN) < 1e-12
    assert bool(b["iteration"]["mutual"]) == M3.CTX_MUTUAL
    assert abs(float(b["beta"]) - M3.EDGE_BETA) < 1e-12
    assert int(b["cand_cap"]) == M3.CAND_CAP
    print("deployed frozen bundle == harness constants (%d checks)\n" % 7, flush=True)

    specs = all_specs5(per_family=args.per_family)
    fit_specs, _ = split5(specs)
    n_nodes = n_harness = n_deploy = n_matrix = 0
    bad: List[Dict[str, Any]] = []
    for vw in M3.iter_views(fit_specs, VIEWS5, args.limit, pose_fn):
        g0, g1 = vw["g0"], vw["g1"]
        rows, truth = vw["rows"], vw["truth"]
        r_pay = PE.build_edges(g0.subgraph, vw["r0"].image_u8,
                               lens_map=PE.lens_map_of(g0),
                               endpoint_lenses=tuple(place_lenses()))
        side_r = PM.build_side(g0, vw["r0"].image_u8)
        side_q = PM.build_side(g1, vw["r1"].image_u8)
        pop = vw["pop"]
        cands = {u: [(int(v), float(s)) for (v, s) in (rows.get(u) or [])[:M3.CAND_CAP]]
                 for u in pop if rows.get(u)}
        nodes_q = sorted(cands)
        if not nodes_q:
            continue
        nodes_r = sorted({v for cl in cands.values() for (v, _s) in cl})
        idx_r = {v: k for k, v in enumerate(nodes_r)}
        # the matcher's own seed matrix: rows over (nodes_q, nodes_r), zeros off
        matrix = [[0.0] * len(nodes_r) for _ in nodes_q]
        for i, u in enumerate(nodes_q):
            for (v, s) in cands[u]:
                matrix[i][idx_r[v]] = s
        stats: Dict[str, Any] = {}
        scored = PM.rescore_matrix(matrix, nodes_q, nodes_r, side_q, side_r,
                                   stats=stats)
        triples = [[(idx_r[v], int(v), s) for (v, s) in cands[u]]
                   for u in nodes_q]
        deploy = PM.argmax_mapping(triples, nodes_q, nodes_r, side_q, side_r)
        # harness pick, node by node
        ctx = PM.context_mapping(triples, nodes_q, side_q, side_r)
        for i, u in enumerate(nodes_q):
            n_nodes += 1
            best_v, best_val = -1, -float("inf")
            for (v, s) in cands[u]:
                term = L.plus_score(u, int(v), ctx, side_q.incident,
                                    side_q.index, side_r.index, fit, pfit,
                                    float(b["_a_exist"]), float(b["_a_geom"]))
                val = (1.0 - M3.EDGE_BETA) * float(s) + M3.EDGE_BETA * float(term)
                if val > best_val:
                    best_val, best_v = val, int(v)
            rowmax = max(range(len(nodes_r)), key=lambda j: scored[i][j])
            if best_v == int(deploy.get(u, -1)):
                n_deploy += 1
            if best_v == int(nodes_r[rowmax]):
                n_matrix += 1
            else:
                bad.append({"scene": vw["spec"].name, "view": vw["label"],
                            "node": int(u), "harness": int(best_v),
                            "deployed": int(deploy.get(u, -1)),
                            "matrix": int(nodes_r[rowmax]),
                            "gap": float(best_val - scored[i][rowmax])})
    agree_d = n_deploy / n_nodes if n_nodes else float("nan")
    agree_m = n_matrix / n_nodes if n_nodes else float("nan")
    print("nodes compared: %d" % n_nodes)
    print("argmax_mapping agrees with the harness: %.6f" % agree_d)
    print("rescore_matrix row argmax agrees    : %.6f  (contract allows a cell "
          "outside the candidate set to keep its raw similarity)" % agree_m)
    if bad:
        print("\nfirst disagreements:")
        for r in bad[:12]:
            print("  %s %s u=%d harness=%d deployed=%d matrix=%d gap %.6f"
                  % (r["scene"], r["view"], r["node"], r["harness"],
                     r["deployed"], r["matrix"], r["gap"]))
    # The DECISION rule the mission asks for is argmax over the same score, i.e.
    # argmax_mapping.  A row of the re-scored MATRIX may still hand the win to a
    # cell outside the top-CANDIDATE set, because the contract says those cells
    # keep their raw similarity; that is reported, not treated as a bug.
    verdict = ("EQUIVALENT" if agree_d == 1.0 else "MISMATCH")
    print("\nVERDICT: %s" % verdict)
    res = {"nodes": n_nodes, "agree_argmax": agree_d, "agree_matrix": agree_m,
           "n_disagree": len(bad), "verdict": verdict,
           "examples": bad[:20]}
    with open(os.path.join(OUT, "%s.json" % args.tag), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s.json" % args.tag)
    return 0 if verdict == "EQUIVALENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
