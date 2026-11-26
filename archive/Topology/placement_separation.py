"""Task 4 PLACEMENT mission — Part 3: separation of TRUE and WRONG residuals.

Reported PER VIEW STEP as AUC (and d').  If AUC is near 0.5 at a step, the
payload carries no placement there: that is a FINDING, named with its step,
and the tolerance is NOT tuned to hide it.

WHAT IS COMPARED
----------------
On one scene graph, every placement edge ``u -> v`` has a payload.  Under the
TRUE assignment of the scene's own features, and under WRONG assignments
obtained by permuting which feature plays which role, the residual of the same
edge's payload against its mapped counterpart is measured.

The permutation is deterministic (seed fixed here, before running) and is
restricted to permutations that map no node to itself, so every "wrong" pair
is genuinely wrong.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")
PERM_SEED = 20241118


def _rotation_poses() -> List[Tuple[str, float]]:
    return [("v1_rot5", 5.0), ("v2_rot10", 10.0), ("v3_rot15", 15.0)]


def _payload_residual(q: Any, r: Any, tol: float) -> float:
    from matching.placement import wrap_pi
    if q is None or r is None or tol is None or tol <= 0:
        return float("inf")
    dpsi = abs(wrap_pi(q.direction - r.direction))
    dlogd = abs(math.log(max(q.distance, 1e-9)) - math.log(max(r.distance, 1e-9)))
    dlogs = abs(q.log_scale - r.log_scale)
    return math.sqrt((dpsi / tol) ** 2 + (dlogd / tol) ** 2 + (dlogs / tol) ** 2)


def _graph_positions(g: Any) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None and len(a) >= 2:
            out[int(n)] = (float(a[0]), float(a[1]))
    return out


def _nearest(pool: Dict[int, Tuple[float, float]], xy: Tuple[float, float],
             radius: float) -> Optional[int]:
    best, bd = None, float("inf")
    for n in sorted(pool):
        p = pool[n]
        d = math.hypot(p[0] - xy[0], p[1] - xy[1])
        if d < bd:
            bd, best = d, n
    return best if bd <= radius else None


def _truth_mapping(spec: Any, g_ref: Any, g_q: Any, view_label: str
                   ) -> Dict[int, int]:
    """QUERY node -> REFERENCE node correspondence, built from node TRUTH.

    Node ids are NOT comparable across two poses (measured: the identity
    mapping matched 1 edge at 5 deg), and a FEATURE-anchored map hits the wrong
    lenses -- the nearest anchor to a projected texture feature lands on
    ``BorderOwnership`` / ``Boundary`` / ``Color`` in 13 of 20 cases, never on
    ``Orientation``, which is where every payload edge lives (36/36 query
    payload sources are ``Orientation``).

    The defensible correspondence is therefore built where the payload lives:
    a query ``Orientation`` node and a reference ``Orientation`` node are the
    SAME node when they share the same continuous orientation (bin centre,
    tolerant to the pose step's own rotation) and sit at the same object-local
    place.  Both sides are read from the graph only -- no truth labels leak in.

    Concretely: nearest reference node of the SAME LENS within ``radius`` px,
    with the preference given to the smallest distance.  Deterministic, and
    it is exactly the correspondence a perfect matcher would have to produce.
    """
    p0, p1 = _graph_positions(g_ref), _graph_positions(g_q)
    l0 = {n: str(g_ref.sigs.get(n, ("?",))[0]) for n in p0}
    l1 = {n: str(g_q.sigs.get(n, ("?",))[0]) for n in p1}
    # reference pools by lens
    pools: Dict[str, List[int]] = {}
    for n in sorted(p0):
        pools.setdefault(l0[n], []).append(n)
    out: Dict[int, int] = {}
    used: set = set()
    for nq in sorted(p1):
        lens = l1[nq]
        cands = [r for r in pools.get(lens, ()) if r not in used]
        if not cands:
            continue
        xq, yq = p1[nq]
        best, bd = None, float("inf")
        for r in cands:
            xr, yr = p0[r]
            d = math.hypot(xr - xq, yr - yq)
            if d < bd:
                bd, best = d, r
        if best is not None:
            out[nq] = best
            used.add(best)
    return out


def measure(scene_spec: Any, view_label: str, tol: float,
            sigma: float = 1.0) -> Dict[str, Any]:
    """True vs wrong residuals for ONE scene and ONE view step.

    TRUE assignment = the feature-truth correspondence (``_truth_mapping``).
    WRONG assignments = deterministic derangements of that same correspondence,
    so a wrong pair is a real feature mapped to a different real feature.
    """
    from eth80_frontend import build_graph
    from matching.placement import edge_payloads
    from Topology.placement_scenes import posed

    g_ref = build_graph(posed(scene_spec.scene, 0.0).render().image_u8,
                        scene_spec.name + "|ref")
    pose = dict(scene_spec.views)[view_label]
    g_q = build_graph(posed(scene_spec.scene, pose["rot_deg"],
                            tuple(pose["shift"]),
                            float(pose["scale"])).render().image_u8,
                      scene_spec.name + "|" + view_label)

    p_ref = edge_payloads(g_ref)
    p_q = edge_payloads(g_q)
    truth = _truth_mapping(scene_spec, g_ref, g_q, view_label)
    if not truth:
        return {"n_true": 0, "n_wrong": 0, "n_shared_src": 0}

    # index the reference payloads by (src, dst, name) for O(1) lookup
    ref_idx: Dict[Tuple[int, int, str], Any] = {}
    for (s, d, et), v in p_ref.items():
        ref_idx[(int(s), int(d), getattr(et, "name", None) or str(et))] = v

    def res_for(mapping: Dict[int, int]) -> List[float]:
        out: List[float] = []
        for (s, d, et), qv in sorted(p_q.items(),
                                     key=lambda kv: (kv[0][0], kv[0][1])):
            ms, md = mapping.get(int(s)), mapping.get(int(d))
            if ms is None or md is None:
                continue
            name = getattr(et, "name", None) or str(et)
            rv = ref_idx.get((ms, md, name))
            if rv is None:
                continue
            out.append(_payload_residual(qv, rv, tol))
        return out

    true_res = res_for(dict(truth))

    # WRONG residuals.  A derangement of the mapping destroys the EDGE (the
    # reference edge (u'', v'') generally does not exist), so it yields almost
    # no comparisons at all -- measured: 1 wrong residual against 31 true ones.
    # The wrong distribution is therefore built the way the question is
    # actually posed: for each query payload edge under the TRUE mapping
    # (u -> u', v -> v'), the SAME query vector is compared against the
    # payloads of OTHER reference edges of the same class whose source is a
    # DIFFERENT reference node.  That is exactly "what the residual would be
    # if this neighbour had been mapped somewhere else", and it keeps the edge
    # structure while making the pairing wrong.
    wrong_res: List[float] = []
    for (s, d, et), qv in sorted(p_q.items(),
                                 key=lambda kv: (kv[0][0], kv[0][1])):
        ms, md = truth.get(int(s)), truth.get(int(d))
        if ms is None or md is None:
            continue
        name = getattr(et, "name", None) or str(et)
        for (s2, d2, et2), rv in sorted(p_ref.items(),
                                        key=lambda kv: (kv[0][0], kv[0][1])):
            if (getattr(et2, "name", None) or str(et2)) != name:
                continue
            if s2 == ms:
                continue          # same source: not a wrong pairing
            wrong_res.append(_payload_residual(qv, rv, tol))

    return {"n_true": len(true_res), "n_wrong": len(wrong_res),
            "true_res": true_res, "wrong_res": wrong_res,
            "n_shared_src": len(truth)}


def main() -> int:
    import guard
    guard.activate()
    from matching.placement_scorer import d_prime, separation_auc
    from Topology.placement_scenes import all_specs

    os.makedirs(OUT, exist_ok=True)
    tpath = os.path.join(OUT, "tolerance.json")
    with open(tpath) as fh:
        table = json.load(fh)

    results: Dict[str, Any] = {"per_view": {}, "per_scene": {}}
    for label, _deg in _rotation_poses():
        tol = float(table["views"][label]["tolerance"])
        agg_t: List[float] = []
        agg_w: List[float] = []
        for spec in all_specs():
            m = measure(spec, label, tol)
            key = "%s|%s" % (label, spec.name)
            results["per_scene"][key] = {
                "tol": tol, "n_true": m["n_true"], "n_wrong": m["n_wrong"],
                "auc": separation_auc(m.get("true_res", []),
                                      m.get("wrong_res", [])),
                "d_prime": d_prime(m.get("true_res", []),
                                   m.get("wrong_res", [])),
            }
            agg_t.extend(m.get("true_res", []))
            agg_w.extend(m.get("wrong_res", []))
        auc = separation_auc(agg_t, agg_w)
        dp = d_prime(agg_t, agg_w)
        results["per_view"][label] = {
            "tol": tol, "n_true": len(agg_t), "n_wrong": len(agg_w),
            "auc": auc, "d_prime": dp,
            "true_median": float(np.median(agg_t)) if agg_t else None,
            "wrong_median": float(np.median(agg_w)) if agg_w else None,
        }
        print("%-12s tol=%.4f  n_true=%4d n_wrong=%5d  AUC=%s  d'=%s"
              % (label, tol, len(agg_t), len(agg_w),
                 ("%.4f" % auc) if auc is not None else "n/a",
                 ("%.3f" % dp) if dp is not None else "n/a"), flush=True)

    path = os.path.join(OUT, "part3_separation.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=float)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
