"""MISSION 9 -- the deployed matcher on the two new families, FIT then CHECK.

One driver for both halves, because the FIT half is what the pre-registration
reads its MDE from and the CHECK half is the ONE read.  Arms are the deployed
matcher's own switches, so what is measured is what a caller would get:

  off    PLACEMENT_MATCHER_ENABLED = False   (the shipped matcher)
  geom   enabled, geometry coefficient as frozen   (P_plus_ctx)
  pres   enabled, geometry coefficient ZERO        (P_plus_ctx_presence)

Ambiguous and unambiguous are reported separately, scene-clustered, and the
partner the flag actually moved is counted so the contrast cannot be vacuous.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def specs_for(family: str, n: int, ir: int = 12, oop: int = 4):
    from Topology import placement_families_m9 as FM
    if family == "ir":
        specs = FM.all_specs_irregular(ir)
        views = FM.VIEWS_IRREGULAR
    else:
        specs = FM.all_specs_oop(oop)
        views = FM.VIEWS_OOP
    fit, check = FM.split_scenes(specs)
    return (fit if n == 0 else fit)[:], check, views, FM.pose_fn


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("ir", "oop"), required=True)
    ap.add_argument("--split", choices=("fit", "check"), required=True)
    ap.add_argument("--scenes", type=int, default=0, help="0 = all in the split")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="the pre-registered CONFIRMATION scenes (never built "
                         "or scored before the pre-registration)")
    ap.add_argument("--no-off", action="store_true",
                    help="skip the flag-off arm (the FIT half already has it)")
    args = ap.parse_args()

    import guard
    guard.activate()
    import parameters as P
    import Topology.placement_m3 as M3
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology import placement_arms2 as A2
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    from Topology.placement_truth import node_objects, track_optimal
    from core.place_key import place_lenses

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    PM.frozen()
    if args.family == "ir":
        specs = (FM.confirmation_specs_irregular(6) if args.fresh
                 else FM.all_specs_irregular(12))
        views = FM.VIEWS_IRREGULAR
    else:
        specs = (FM.confirmation_specs_oop(6) if args.fresh
                 else FM.all_specs_oop(4))
        views = FM.VIEWS_OOP
    if args.fresh:
        fit, check = [], specs
    else:
        fit, check = FM.split_scenes(specs)
    chosen = fit if args.split == "fit" else check
    if args.scenes:
        chosen = chosen[:args.scenes]
    tag = args.tag or ("m9_%s_%s" % (args.family, args.split))

    ARMS = ("geom", "pres") if args.no_off else ("off", "geom", "pres")
    ok: Dict[str, Dict[str, List[float]]] = {a: defaultdict(list) for a in ARMS}
    moved: List[int] = []
    print("M9 %s %s -- %d scenes, %d views\n"
          % (args.family.upper(), args.split.upper(), len(chosen), len(views)),
          flush=True)
    for spec in chosen:
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        for label, view in views:
            sc1, _pv = FM.pose_fn(spec, view)
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            truth = {int(q): int(r)
                     for r, q in track_optimal(spec.scene, r0, sc1, r1,
                                               g0, g1).items()}
            o1 = node_objects(r1, g1)
            pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
            rows = similarity_rows(g1, g0, pop,
                                   [v for v in sorted(int(n) for n in
                                                      g0.subgraph.nodes)])
            bucket: Dict[int, str] = {}
            for u in pop:
                cl = rows.get(u) or []
                top = cl[0][1] if cl else 0.0
                second = cl[1][1] if len(cl) > 1 else 0.0
                bucket[u] = ("ambiguous" if (top - second) < A2.AMBIGUOUS_MARGIN
                             else "unambiguous")
            maps: Dict[str, Dict[int, int]] = {}
            for arm in ARMS:
                P.PLACEMENT_MATCHER_ENABLED = (arm != "off")
                P.PLACEMENT_MATCHER_PRESENCE = (arm == "pres")
                res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                                placement_images=(r1.image_u8, r0.image_u8))
                maps[arm] = {int(k): int(v) for k, v in res.mapping.items()}
            for u in pop:
                tv = truth.get(u)
                for arm in ARMS:
                    v = maps[arm].get(u)
                    ok[arm][(spec.name, bucket[u])].append(
                        1.0 if (v is not None and int(v) == int(tv)) else 0.0)
            base = maps.get("off") or maps.get("pres") or {}
            moved.append(sum(1 for u, v in maps["geom"].items()
                             if base.get(u) != v))
            print("  %-22s %-8s done (moved %d)" % (spec.name, label, moved[-1]),
                  flush=True)
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_MATCHER_PRESENCE = False

    def scene_means(arm: str, b: str) -> Dict[str, float]:
        per: Dict[str, List[float]] = defaultdict(list)
        for (scene, bb), v in ok[arm].items():
            if bb == b:
                per[scene].extend(v)
        return {s: float(np.mean(v)) for s, v in per.items() if v}

    def contrast(a: str, b: str, bucket: str) -> Dict[str, Any]:
        A, B = scene_means(a, bucket), scene_means(b, bucket)
        keys = sorted(set(A) & set(B))
        if len(keys) < 2:
            return {"mean": float("nan"), "n_scenes": len(keys),
                    "se_scene": float("nan"), "mde": float("nan")}
        d = np.array([A[k] - B[k] for k in keys])
        se = float(d.std(ddof=1) / np.sqrt(len(keys)))
        return {"mean": float(d.mean()), "se_scene": se, "mde": 2.8 * se,
                "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                "n_scenes": len(keys),
                "per_scene": {k: float(A[k] - B[k]) for k in keys}}

    out: Dict[str, Any] = {"family": args.family, "split": args.split,
                           "scenes": [s.name for s in chosen], "arms": {},
                           "contrasts": {}, "moved": float(np.mean(moved))}
    for bucket in ("ambiguous", "unambiguous"):
        out["arms"][bucket] = {}
        print("\n  %s bank" % bucket)
        for arm in ARMS:
            sm = scene_means(arm, bucket)
            out["arms"][bucket][arm] = (float(np.mean(list(sm.values())))
                                        if sm else float("nan"))
            print("    %-5s %.4f  (scenes %d, nodes %d)"
                  % (arm, out["arms"][bucket][arm], len(sm),
                     len(ok[arm] and [1 for (s, b) in ok[arm] if b == bucket])))
        for a, b in [("geom", "off"), ("pres", "off"), ("geom", "pres")]:
            if a not in ARMS or b not in ARMS:
                continue
            c = contrast(a, b, bucket)
            out["contrasts"]["%s - %s (%s)" % (a, b, bucket)] = c
            print("    %-12s %+0.4f  se %.4f  mde %.4f  ci [%+0.4f,%+0.4f]"
                  % ("%s - %s" % (a, b), c.get("mean", float("nan")),
                     c.get("se_scene", float("nan")), c.get("mde", float("nan")),
                     c.get("ci", [float("nan")] * 2)[0],
                     c.get("ci", [float("nan")] * 2)[1]))
    print("\n  nodes the flag moved per match: mean %.1f" % out["moved"])
    print("  counters: %s" % PM.FAILURES)
    path = os.path.join(OUT, "%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
