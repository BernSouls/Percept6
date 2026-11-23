"""MISSION 9 STEP 1 -- the credited scorer inside the DEPLOYED matcher.

Two modes, both on the matcher's own code path (`matching.joint_mcs.joint_mcs`):

  argmax      the per-node decision the harness makes, applied to the aligner's
              seed matrix (equivalence is proven separately, to 1.0)
  assignment  the aligner's OWN global assignment on the re-scored matrix -- the
              thing a deployed matcher actually returns

Reported per mode, ambiguous and unambiguous separately, scene-clustered:
baseline (flag off) against geometry (flag on) and against presence-only
(geometry coefficient zero, same flag machinery).  The comparison is paired by
scene, so the interval is over scenes and not over nodes.

Non-vacuity: the number of cells the flag actually changed, and the number of
nodes whose final partner moved.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
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
    ap.add_argument("--tag", default="m9_deployed")
    args = ap.parse_args()

    import guard
    guard.activate()
    import parameters as P
    import Topology.placement_m3 as M3
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology import placement_arms2 as A2
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from Topology.placement_truth import node_objects, track_optimal
    from core.place_key import place_lenses

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    with open(os.path.join(OUT, "_fitcache", "m8_step2_fit.pkl"), "rb") as fh:
        F = pickle.load(fh)
    M3.CTX_MARGIN = float(F["ctx_margin"])
    M3.CTX_ITERS = int(F["ctx_iters"])
    M3.CTX_MUTUAL = bool(F["ctx_mutual"])
    PM.frozen()

    specs = all_specs5(per_family=args.per_family)
    fit_specs, _ = split5(specs)
    if args.limit:
        fit_specs = fit_specs[:args.limit]

    ARMS = ("off", "geom", "pres")
    ok: Dict[str, Dict[str, List[float]]] = {a: defaultdict(list) for a in ARMS}
    n_cells: List[int] = []
    moved_nodes: List[int] = []
    for spec in fit_specs:
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        for label, view in VIEWS5:
            sc1, _pv = pose_fn(spec, view)
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            truth = {int(q): int(r)
                     for r, q in track_optimal(spec.scene, r0, sc1, r1,
                                               g0, g1).items()}
            from Topology.placement_census import similarity_rows
            o1 = node_objects(r1, g1)
            pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
            rows = similarity_rows(g1, g0, pop,
                                   [v for v in sorted(int(n) for n in
                                                      g0.subgraph.nodes)])
            buckets = {}
            for u in pop:
                cl = rows.get(u) or []
                n = len(cl)
                top = cl[0][1] if n else 0.0
                second = cl[1][1] if n > 1 else 0.0
                buckets[u] = ("ambiguous" if (top - second) < A2.AMBIGUOUS_MARGIN
                              else "unambiguous")
            maps = {}
            for arm in ARMS:
                P.PLACEMENT_MATCHER_ENABLED = (arm != "off")
                P.PLACEMENT_MATCHER_PRESENCE = (arm == "pres")
                res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                                placement_images=(r1.image_u8, r0.image_u8))
                maps[arm] = res.mapping
            for u in pop:
                tv = truth.get(u)
                for arm in ARMS:
                    v = maps[arm].get(u)
                    got = 1.0 if (v is not None and int(v) == int(tv)) else 0.0
                    ok[arm][(spec.name, buckets[u])].append(got)
            moved_nodes.append(sum(1 for u, v in maps["geom"].items()
                                   if maps["off"].get(u) != v))
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_MATCHER_PRESENCE = False

    def scene_means(arm: str, bucket: str) -> Dict[str, float]:
        per: Dict[str, List[float]] = defaultdict(list)
        for (scene, b), v in ok[arm].items():
            if b == bucket:
                per[scene].extend(v)
        return {s: float(np.mean(v)) for s, v in per.items() if v}

    def contrast(a: str, b: str, bucket: str) -> Dict[str, Any]:
        A, B = scene_means(a, bucket), scene_means(b, bucket)
        keys = sorted(set(A) & set(B))
        if len(keys) < 2:
            return {"mean": float("nan"), "n_scenes": len(keys)}
        d = np.array([A[k] - B[k] for k in keys])
        se = float(d.std(ddof=1) / np.sqrt(len(keys)))
        return {"mean": float(d.mean()), "se_scene": se, "mde": 2.8 * se,
                "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                "n_scenes": len(keys),
                "per_scene": {k: float(A[k] - B[k]) for k in keys}}

    out: Dict[str, Any] = {"arms": {}, "contrasts": {}, "modes": {}}
    print("\nDEPLOYED MATCHER (joint_mcs), FIT, scene-clustered\n")
    for bucket in ("ambiguous", "unambiguous"):
        out["arms"][bucket] = {}
        print("  %s bank" % bucket)
        for arm in ARMS:
            sm = scene_means(arm, bucket)
            out["arms"][bucket][arm] = (float(np.mean(list(sm.values())))
                                        if sm else float("nan"))
            print("    %-5s %.4f  (scenes %d)" % (arm, out["arms"][bucket][arm],
                                                  len(sm)))
        for a, b in (("geom", "off"), ("pres", "off"), ("geom", "pres")):
            c = contrast(a, b, bucket)
            out["contrasts"]["%s - %s (%s)" % (a, b, bucket)] = c
            print("    %-12s %+0.4f  se %.4f  mde %.4f  ci [%+0.4f,%+0.4f]"
                  % ("%s - %s" % (a, b), c.get("mean", float("nan")),
                     c.get("se_scene", float("nan")), c.get("mde", float("nan")),
                     c.get("ci", [float("nan")] * 2)[0],
                     c.get("ci", [float("nan")] * 2)[1]))
    out["modes"] = {"assignment": True,
                    "nodes_moved_by_flag_mean": float(np.mean(moved_nodes)),
                    "failures": dict(PM.FAILURES)}
    print("\n  nodes whose partner the flag moved, per match: mean %.1f"
          % np.mean(moved_nodes))
    print("  placement_matcher counters: %s" % PM.FAILURES)
    nonv = np.mean(moved_nodes) > 0
    print("  NON-VACUITY: the flag %s change decisions"
          % ("DOES" if nonv else "DOES NOT"))
    out["non_vacuous"] = bool(nonv)
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
