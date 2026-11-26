"""MISSION 10 STEP 2 (second half) -- the four DEV-5 families at the DEPLOYED
matcher, on ALL FIT scenes.  Report-only regression; nothing here is fitted,
nothing here decides anything.

Mission 9 could only report one FIT scene for this regression.  One scene cannot
say "no regression": it can only say "no crash".  This runs every FIT scene of
DEV-5 (families SPHERE, BOX, CYLINDER, OCCLUSION, stratified split, so all four
families land at all four pose steps) through the same two arms as the
confirmation read -- geometry (placement on) and presence (existence only) -- and
scores the same ambiguous bank, per family and per pose step.

The question is narrow and stated in advance: does the deployed path lose
accuracy ANYWHERE in the DEV-5 families relative to existence-only, on scenes that
were FIT in Mission 8 and are not part of any confirmation set?  Accuracy is
reported per family and per step with its node counts; a family whose ambiguous
population is too small to say anything is reported as unresolved rather than as
a pass.
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
PROGRESS = os.path.join(OUT, "m10_dev5_fit_progress.jsonl")


def _specs():
    from Topology.placement_dev5 import all_specs5, split5
    fit, _check = split5(all_specs5(6))
    return fit


def _view_job(job):
    scene_name, family, label = job
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology import placement_arms2 as A2
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    from Topology.placement_dev5 import VIEWS5, posed5
    from Topology.placement_truth import node_objects, track_optimal
    from core.place_key import place_lenses

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    PM.frozen()
    spec = {s.name: s for s in _specs()}[scene_name]
    view = dict(VIEWS5)[label]
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    sc1 = posed5(spec, view)
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
    return {"scene": scene_name, "family": family, "view": label, "arms": out,
            "n_amb": len(amb), "n_un": len(pop) - len(amb)}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import guard
    guard.activate()
    from Topology.placement_dev5 import VIEWS5
    specs = _specs()
    jobs = [(s.name, s.family, lab) for s in specs for lab, _v in VIEWS5]
    done: List[Dict[str, Any]] = []
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    done.append(json.loads(line))
                except Exception:
                    continue
    seen = {(d["scene"], d["view"]) for d in done}
    todo = [j for j in jobs if (j[0], j[2]) not in seen]
    print("%d of %d view-pairs done, %d to go" % (len(done), len(jobs), len(todo)),
          flush=True)
    t0 = time.time()
    with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_view_job, todo):
            done.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("  %-18s %-8s %5.0fs" % (r["scene"], r["view"], time.time() - t0),
                  flush=True)

    res: Dict[str, Any] = {"scenes": sorted({d["scene"] for d in done}),
                           "families": sorted({d["family"] for d in done}),
                           "view_pairs": len(done)}
    # per family x view step, both buckets
    table: Dict[str, Any] = {}
    for fam in res["families"]:
        table[fam] = {}
        for lab, _v in VIEWS5:
            sel = [d for d in done if d["family"] == fam and d["view"] == lab]
            if not sel:
                continue
            row: Dict[str, Any] = {"view_pairs": len(sel)}
            for bucket in ("amb", "un"):
                for arm in ("geom", "pres"):
                    vals = [x for d in sel for x in d["arms"][arm][bucket]]
                    row["%s_%s_n" % (bucket, arm)] = len(vals)
                    row["%s_%s_acc" % (bucket, arm)] = (float(np.mean(vals))
                                                        if vals else float("nan"))
            d_vals = [np.mean([x for d in sel for x in d["arms"]["geom"]["amb"]]) -
                      np.mean([x for d in sel for x in d["arms"]["pres"]["amb"]])
                      if sel else 0.0][0]
            row["amb_delta_geom_minus_pres"] = float(d_vals)
            table[fam][lab] = row
    res["table"] = table
    # family level, pooled over the four steps
    famtbl: Dict[str, Any] = {}
    for fam in res["families"]:
        sel = [d for d in done if d["family"] == fam]
        row = {"view_pairs": len(sel)}
        for bucket in ("amb", "un"):
            for arm in ("geom", "pres"):
                vals = [x for d in sel for x in d["arms"][arm][bucket]]
                row["%s_%s_n" % (bucket, arm)] = len(vals)
                row["%s_%s_acc" % (bucket, arm)] = (float(np.mean(vals))
                                                    if vals else float("nan"))
        famtbl[fam] = row
    res["by_family"] = famtbl
    # scene-clustered paired delta over all FIT scenes
    per_scene = {}
    for d in done:
        e = per_scene.setdefault(d["scene"], {"geom": [], "pres": []})
        for arm in ("geom", "pres"):
            e[arm].extend(d["arms"][arm]["amb"])
    keys = [k for k in sorted(per_scene) if per_scene[k]["geom"]]
    if len(keys) > 1:
        dv = np.array([np.mean(per_scene[k]["geom"]) - np.mean(per_scene[k]["pres"])
                       for k in keys])
        se = float(dv.std(ddof=1) / np.sqrt(len(keys)))
        res["amb_delta"] = {"delta": float(dv.mean()), "se": se, "scenes": len(keys),
                            "mde": float(2.8 * se),
                            "geom": float(np.mean([np.mean(per_scene[k]["geom"]) for k in keys])),
                            "pres": float(np.mean([np.mean(per_scene[k]["pres"]) for k in keys]))}
    res["counters"] = {k: int(sum(d["arms"][a]["counters"].get(k, 0) for d in done
                                 for a in ("geom", "pres")))
                       for k in ("rescored", "raised", "fell_back", "skipped")}
    res["sec_total"] = float(sum(d["arms"][a]["sec"] for d in done for a in ("geom", "pres")))
    path = os.path.join(OUT, "m10_dev5_fit.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote %s" % path)
    if "amb_delta" in res:
        a = res["amb_delta"]
        print("amb geom %.4f vs pres %.4f  delta %+.4f se %.4f mde %.4f n %d scenes"
              % (a["geom"], a["pres"], a["delta"], a["se"], a["mde"], a["scenes"]))
    for fam, row in sorted(famtbl.items()):
        print("  %-9s amb geom %.3f (n %d) pres %.3f (n %d)" % (
            fam, row["amb_geom_acc"], row["amb_geom_n"],
            row["amb_pres_acc"], row["amb_pres_n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
