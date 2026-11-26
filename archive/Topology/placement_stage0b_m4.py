"""STAGE 0b -- is the absence penalty swamping weak geometry?

The Mission-3 estimator pads every uncheckable slot with ``mu_wrong``: an edge
that is simply MISSING (about a third of them vanish between views) is charged
as if it were a wrong relation.  If that penalty dominates, geometry can never
beat presence, because presence is the term that *rewards* the edges that did
survive.

This compares four arms on the FIT half (free reruns -- no CHECK look is spent):

  P_mu / P_neutral               residual LLR, padded at mu_wrong / at 0
  P_presence_mu / _neutral       the same two paddings for the presence term

and reports accuracy on near-tie ambiguous nodes, scene-clustered.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
for p in (ROOT, os.path.join(ROOT, "Topology")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching import placement_llr as L
    from matching.placement_scorer import incident_index, payload_index
    from Topology.placement_arms2 import is_ambiguous, shipped_edges
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import VIEWS, all_specs, split, view_pose
    from Topology.placement_m3 import CAND_CAP, EDGE_BETA, _payload, _rows_for
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, track_optimal
    from Topology.placement_m3 import collect_fit

    fit_specs, _check = split(all_specs())
    fit_specs = fit_specs[:args.limit]
    fit, pfit, meta, _rf = collect_fit(fit_specs, VIEWS, None)
    print("FIT: wrong-pair mean %.4f (this is what an ABSENT edge is charged)"
          % meta["mu_wrong"])

    per: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    vanish_num = vanish_den = 0
    for spec in fit_specs:
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        refc = [v for v in sorted(int(n) for n in g0.subgraph.nodes)
                if o0.get(v, -1) >= 0]
        intra = (spec.family == "NEGATIVE-A")
        r_pay = _payload(PE, g0, r0.image_u8, o0, intra)
        r_idx = payload_index(r_pay)
        for label, view in VIEWS:
            pv = view_pose(spec, view)
            sc1 = posed(spec.scene, pv["rot_deg"], tuple(pv["shift"]),
                        pv["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            q_pay = _payload(PE, g1, r1.image_u8, o1, intra)
            q_idx, q_inc = payload_index(q_pay), incident_index(q_pay)
            truth = {int(q): int(r) for r, q
                     in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
            rows = _rows_for(g1, g0, pop, refc)
            nmap = {u: rows[u][0][0] for u in pop if rows.get(u)}
            for u in pop:
                row = rows.get(u) or []
                if not row:
                    continue
                cand = row[:CAND_CAP]
                tv = truth.get(u)
                # how many of u's slots are uncheckable under its TRUE candidate
                sl = L.slots_for_candidate(u, int(nmap.get(u, -1)), nmap,
                                           q_inc, q_idx, r_idx)
                vanish_num += sum(1 for (_n, _o, r) in sl if r is None)
                vanish_den += len(sl)
                amb = is_ambiguous(row)
                if not amb:
                    continue
                for arm, pad in (("P_mu", None), ("P_neutral", 0.0)):
                    best, bs = -1, -float("inf")
                    for v, s in cand:
                        t = L.score_candidate(u, v, nmap, q_inc, q_idx, r_idx,
                                              fit, pad=pad)
                        val = (1 - EDGE_BETA) * s + EDGE_BETA * t
                        if val > bs:
                            bs, best = val, v
                    per[arm][spec.name].append(float(best == tv))
                for arm, pad in (("P_presence_mu", None),
                                 ("P_presence_neutral", 0.0)):
                    best, bs = -1, -float("inf")
                    for v, s in cand:
                        t = L.presence_score(u, v, nmap, q_inc, q_idx, r_idx,
                                             pfit, pad=pad)
                        val = (1 - EDGE_BETA) * s + EDGE_BETA * t
                        if val > bs:
                            bs, best = val, v
                    per[arm][spec.name].append(float(best == tv))

    print("uncheckable slot share under the top candidate: %.3f"
          % (vanish_num / max(1, vanish_den)))
    res: Dict[str, Any] = {"uncheckable_share": vanish_num / max(1, vanish_den),
                           "mu_wrong": meta["mu_wrong"], "arms": {}}
    print("\n%-22s %8s %8s %8s" % ("arm", "mean", "scene_se", "n"))
    for arm in ("P_mu", "P_neutral", "P_presence_mu", "P_presence_neutral"):
        d = per[arm]
        keys = sorted(d)
        means = np.array([float(np.mean(d[k])) for k in keys])
        se = float(means.std(ddof=1) / np.sqrt(len(keys))) if len(keys) > 1 else float("nan")
        n = sum(len(d[k]) for k in keys)
        res["arms"][arm] = {"mean": float(means.mean()), "scene_se": se,
                            "n": n, "n_scenes": len(keys)}
        print("%-22s %8.4f %8.4f %8d" % (arm, means.mean(), se, n))

    a = res["arms"]
    res["contrasts"] = {
        "P_mu-P_presence_mu": a["P_mu"]["mean"] - a["P_presence_mu"]["mean"],
        "P_neutral-P_presence_neutral": (a["P_neutral"]["mean"]
                                         - a["P_presence_neutral"]["mean"]),
        "P_neutral-P_mu": a["P_neutral"]["mean"] - a["P_mu"]["mean"],
    }
    print("\nP - P_presence under mu_wrong padding : %+.4f"
          % res["contrasts"]["P_mu-P_presence_mu"])
    print("P - P_presence under NEUTRAL padding  : %+.4f"
          % res["contrasts"]["P_neutral-P_presence_neutral"])
    print("P_neutral - P_mu                      : %+.4f"
          % res["contrasts"]["P_neutral-P_mu"])

    path = os.path.join(OUT, "m4_stage0b_padding.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s (FIT only; no CHECK look spent)" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
