"""MISSION 20 / item 5 -- where the shipped configuration breaks, out of plane.

The registered M19 read is an IN-PLANE rotation at 15 deg.  Recognition of a real object
across viewpoints is out of plane, and Mission 10's generator can make that: `rot_y` and
`rot_x` tilt the camera off the object's axis and re-render, so the place-to-place truth is
exact (`track_optimal`) rather than assumed.

WHAT IS MEASURED, per (scene, view):
  node accuracy   the deployed scorer's pick is the true partner for each query place
  geo AUC         the geometry term alone, true partner against its impostors
  bank split      the same two, on Mission 10's near-tie ambiguity bank and off it -- a
                  probe on a hard bank is only readable if the easy bank is reported too

20 deg is one of Mission 10's registered views (`o3_y20`, `o6_x20`).  35 deg is NOT: it was
never registered, no earlier read has touched it, and the graphs for it are built here.  So
the 20 deg rows are "cached-graph probe, previously registered view" and the 35 deg rows are
"new view, probe only".  They are reported as separate rows and never pooled.

The scorer is the DEPLOYED one (bundle on: descriptor w 0.25 / K 24 / whole row, M18 table,
assignment start, candidate cap 24), because the question is what the shipped configuration
does out of plane, not what the harness can be made to do.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: (label, view) -- the first two are Mission 10's registered 20 deg views, the last two the
#: 35 deg probe views that no registration covers
PROBE_VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("y20", {"rot_y_deg": 20.0}),
    ("x20", {"rot_x_deg": 20.0}),
    ("y35", {"rot_y_deg": 35.0}),
    ("x35", {"rot_x_deg": 35.0}),
)


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def cell(job: Tuple[str, str]) -> Dict[str, Any]:
    scene_name, label = job
    import guard
    guard.activate()
    import parameters as P
    from core.place_key import place_lenses
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from Topology import placement_arms2 as A2
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    from Topology.placement_truth import node_objects, track_optimal

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    _recipe()
    spec = {s.name: s for s in FM.confirmation_specs_oop_m10(20)}[scene_name]
    view = dict(PROBE_VIEWS)[label]
    t0 = time.time()
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    sc1, _pv = FM.pose_fn(spec, view)
    r1 = sc1.render()
    g1 = get_or_build(r1.image_u8, spec.name + "|m20" + label)
    truth = {int(q): int(r) for r, q in
             track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    o1 = node_objects(r1, g1)
    pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
    if not pop:
        return {"scene": scene_name, "view": label, "queries": 0, "reason": "no truth"}
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    side_q = PM.build_side_cached(g1, r1.image_u8)
    side_r = PM.build_side_cached(g0, r0.image_u8)
    rows = similarity_rows(g1, g0, pop, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    seed = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows.get(u, ()):
            if s > 0.0:
                seed[i][idx_b[v]] = float(s)
    stats: Dict[str, Any] = {}
    scored = PM.rescore_matrix(seed, nodes_a, nodes_b, side_q, side_r, stats=stats)
    # the geometry term alone, with the mapping the DEPLOYED scorer started from
    h_mat = PM.apply_node_descriptor([list(r) for r in seed], nodes_a, nodes_b,
                                    side_q, side_r)
    ctx = PM.context_mapping(PM.top_candidates_adaptive(h_mat, nodes_b, PM.cand_cap()),
                             nodes_a, side_q, side_r)
    fit, pfit = PM.fit_objects()
    b = PM.frozen()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    tol_of = PM._spacing_tol_fn(side_q, side_r)
    amb = set()
    for u in pop:
        cl = rows.get(u) or []
        t_0 = cl[0][1] if cl else 0.0
        t_1 = cl[1][1] if len(cl) > 1 else 0.0
        if (t_0 - t_1) < A2.AMBIGUOUS_MARGIN:
            amb.add(u)
    idx_a = {u: i for i, u in enumerate(nodes_a)}
    hit = {"amb": [], "un": []}
    geo: List[Tuple[float, List[float]]] = []
    for u in pop:
        i = idx_a.get(u)
        if i is None:
            continue
        live = [(nodes_b[j], float(scored[i][j])) for j in range(len(nodes_b))
                if scored[i][j] > 0.0]
        if len(live) < 2:
            continue
        pick = max(live, key=lambda t: (t[1], -t[0]))[0]
        (hit["amb"] if u in amb else hit["un"]).append(1.0 if pick == truth.get(u) else 0.0)
        em = {}
        for (v, _s) in live:
            em[v] = float(L.plus_score(u, int(v), ctx, side_q.incident, side_q.index,
                                       side_r.index, fit, pfit, a_e, a_g,
                                       tol=(None if tol_of is None else tol_of(u, v)))) - 0.5
        vt = truth.get(u)
        if vt in em:
            geo.append((em[vt], [em[v] for v in em if v != vt]))
    return {"scene": scene_name, "view": label, "shape": scene_name.split("_")[1],
            "queries": len(pop), "amb_n": len(hit["amb"]), "un_n": len(hit["un"]),
            "acc": (float(np.mean(hit["amb"] + hit["un"]))
                    if (hit["amb"] or hit["un"]) else None),
            "acc_amb": float(np.mean(hit["amb"])) if hit["amb"] else None,
            "acc_un": float(np.mean(hit["un"])) if hit["un"] else None,
            "geo_auc": _auc(geo), "geo_n": len(geo),
            "sec": time.time() - t0,
            "effective": {k: stats.get("effective", {}).get(k) for k in
                          ("PLACEMENT_NODE_DESC", "PLACEMENT_NODE_DESC_W",
                           "PLACEMENT_NODE_DESC_K", "PLACEMENT_NODE_DESC_MAXCELLS",
                           "PLACEMENT_LLR_TABLE_M18", "PLACEMENT_CONTEXT_ASSIGN")},
            "cap": stats.get("effective", {}).get("cand_cap_resolved")}


def _recipe() -> None:
    import parameters as P
    for k, v in dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
                     FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
                     PLACEMENT_FRAME_KEYPOINT_SIGMA="largest",
                     PLACEMENT_FRAME_FALLBACK="none", PLACEMENT_M15_PLACES=True,
                     KP_GRADED_SIMILARITY=True).items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    from core import keypoint_classes as KPC
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()


def main(argv=None) -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--tag", default="m20_oop")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from Topology import placement_families_m9 as FM
    if not P.PLACEMENT_M19:
        raise SystemExit("run with PLACEMENT_M19=1: this reads the deployed configuration")
    specs = FM.confirmation_specs_oop_m10(a.scenes)[:a.scenes]
    jobs = [(s.name, lab) for s in specs for lab, _v in PROBE_VIEWS]
    prog = os.path.join(OUT, a.tag + "_progress.jsonl")
    done: List[Dict[str, Any]] = []
    if os.path.exists(prog):
        done = [json.loads(x) for x in open(prog) if x.strip()]
    seen = {(d["scene"], d["view"]) for d in done}
    todo = [j for j in jobs if j not in seen]
    print("%d cells done, %d to go" % (len(done), len(todo)), flush=True)
    t0 = time.time()
    with open(prog, "a") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(cell, todo, chunksize=1)):
            fh.write(json.dumps(r, sort_keys=True) + "\n")
            done.append(r)
            if (i + 1) % 10 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(todo), time.time() - t0), flush=True)
    res = {"views": [v for v, _ in PROBE_VIEWS], "scenes": len(specs), "cells": done}
    out = os.path.join(OUT, a.tag + ".json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("\n%-5s %-6s %-6s %-6s %-6s %-6s %-8s" %
          ("view", "cells", "q", "acc", "acc_amb", "acc_un", "geo_auc"))
    for lab, _v in PROBE_VIEWS:
        rs = [d for d in done if d["view"] == lab and d.get("queries")]
        if not rs:
            continue
        q = sum(d["queries"] for d in rs)
        acc = sum((d["acc"] or 0) * d["queries"] for d in rs) / q
        am = [d["acc_amb"] for d in rs if d["acc_amb"] is not None]
        un = [d["acc_un"] for d in rs if d["acc_un"] is not None]
        gq = [d["geo_auc"] for d in rs if d["geo_auc"] is not None]
        print("%-5s %-6d %-6d %-6.3f %-6s %-6s %-8s"
              % (lab, len(rs), q, acc,
                 ("%.3f" % (sum(am) / len(am))) if am else "-",
                 ("%.3f" % (sum(un) / len(un))) if un else "-",
                 ("%.3f" % (sum(gq) / len(gq))) if gq else "-"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
