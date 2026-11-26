"""MISSION 10 STEP 2 -- the out-of-plane confirmation read (one look).

Twenty fresh scenes, registered before their graphs existed, two arms through the
deployed matcher.  The primary is geometry - presence on the near-tie ambiguous
bank; the unambiguous bank is the no-harm check.  Scene-clustered paired t.

Nothing here is tuned, and nothing is pooled with Mission 9's six scenes: the
record for those is `m9_oop_check.json` and it is reported beside this one.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PROGRESS = os.path.join(OUT, "m10_oop_check_progress.jsonl")


def _view_job(job):
    scene_name, label = job
    import guard
    guard.activate()
    import parameters as P
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
    spec = {s.name: s for s in FM.confirmation_specs_oop_m10(20)}[scene_name]
    view = dict(FM.VIEWS_OOP)[label]
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    sc1, _pv = FM.pose_fn(spec, view)
    r1 = sc1.render()
    g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
    truth = {int(q): int(r) for r, q in
             track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    o1 = node_objects(r1, g1)
    pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
    rows = similarity_rows(g1, g0, pop, sorted(int(n) for n in g0.subgraph.nodes))
    amb = set()
    for u in pop:
        cl = rows.get(u) or []
        t0 = cl[0][1] if cl else 0.0
        t1 = cl[1][1] if len(cl) > 1 else 0.0
        if (t0 - t1) < A2.AMBIGUOUS_MARGIN:
            amb.add(u)
    out = {}
    for arm, presence in (("geom", False), ("pres", True)):
        P.PLACEMENT_MATCHER_ENABLED = True
        P.PLACEMENT_MATCHER_PRESENCE = bool(presence)
        t = time.time()
        res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                        placement_images=(r1.image_u8, r0.image_u8))
        dt = time.time() - t
        P.PLACEMENT_MATCHER_ENABLED = False
        P.PLACEMENT_MATCHER_PRESENCE = False
        mapping = {int(k): int(v) for k, v in res.mapping.items()}
        out[arm] = {
            "amb": [1.0 if mapping.get(u) == truth.get(u) else 0.0 for u in sorted(amb)],
            "un": [1.0 if mapping.get(u) == truth.get(u) else 0.0
                   for u in sorted(set(pop) - amb)],
            "sec": dt, "counters": dict(PM.FAILURES)}
    return {"scene": scene_name, "view": label, "arms": out,
            "n_amb": len(amb), "n_un": len(pop) - len(amb)}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="m10_oop_check")
    args = ap.parse_args()

    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    specs = FM.confirmation_specs_oop_m10(20)
    jobs = [(s.name, lab) for s in specs for lab, _v in FM.VIEWS_OOP]
    done: List[Dict[str, Any]] = []
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    done.append(json.loads(line))
                except Exception:
                    continue
    seen = {(d["scene"], d["view"]) for d in done}
    todo = [j for j in jobs if j not in seen]
    print("%d of %d view-pairs done, %d to go" % (len(done), len(jobs), len(todo)),
          flush=True)
    t0 = time.time()
    with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_view_job, todo):
            done.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("  %-22s %-8s %.0fs" % (r["scene"], r["view"], time.time() - t0),
                  flush=True)

    res: Dict[str, Any] = {"scenes": sorted({d["scene"] for d in done}),
                           "view_pairs": len(done), "arms": {}}
    for bucket in ("amb", "un"):
        per_scene: Dict[str, Dict[str, List[float]]] = {}
        for d in done:
            per_scene.setdefault(d["scene"], {"geom": [], "pres": []})
            for arm in ("geom", "pres"):
                per_scene[d["scene"]][arm].extend(d["arms"][arm][bucket])
        means = {arm: {s: float(np.mean(v[arm])) for s, v in per_scene.items()}
                 for arm in ("geom", "pres")}
        keys = sorted(means["geom"])
        d_vals = np.array([means["geom"][s] - means["pres"][s] for s in keys])
        se = float(d_vals.std(ddof=1) / np.sqrt(len(keys))) if len(keys) > 1 else float("nan")
        n_nodes = sum(len(v["geom"]) + len(v["pres"]) for v in per_scene.values())
        res["arms"][bucket] = {
            "geom": float(np.mean(list(means["geom"].values()))),
            "pres": float(np.mean(list(means["pres"].values()))),
            "delta": float(d_vals.mean()), "se_scene": se, "mde": 2.8 * se,
            "ci": [float(d_vals.mean() - 1.96 * se), float(d_vals.mean() + 1.96 * se)],
            "n_scenes": len(keys), "nodes": int(n_nodes // 2),
            "per_scene": {s: float(means["geom"][s] - means["pres"][s]) for s in keys}}
        # paired t
        if len(keys) > 2 and se > 0:
            t = float(d_vals.mean() / se)
            from scipy import stats as _st
            p = float(2.0 * _st.t.sf(abs(t), len(keys) - 1))
            res["arms"][bucket].update({"t": t, "p": p})
    res["seconds"] = float(sum(d["arms"]["geom"]["sec"] + d["arms"]["pres"]["sec"]
                               for d in done))
    res["counters"] = {k: int(sum(d["arms"][a]["counters"].get(k, 0) for d in done
                                  for a in ("geom", "pres")))
                       for k in ("rescored", "raised", "fell_back", "skipped")}
    path = os.path.join(OUT, args.tag + ".json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    for b in ("amb", "un"):
        a = res["arms"][b]
        print("%s: geom %.4f pres %.4f  delta %+0.4f  se %.4f  mde %.4f  "
              "ci [%+0.4f,%+0.4f]  p %.4g  n_scenes %d"
              % (b, a["geom"], a["pres"], a["delta"], a["se_scene"], a["mde"],
                 a["ci"][0], a["ci"][1], a.get("p", float("nan")), a["n_scenes"]))
    print("counters %s ; %.0f s of matching" % (res["counters"], res["seconds"]))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
