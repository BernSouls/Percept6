"""MISSION 10 STEP 1 -- refit the credited coefficients under the DEPLOYED objective.

The Mission-8 bundle was fitted for a per-node choice among six candidates: an
argmax that only changes when the term reorders that node's own top two.  The
deployed matcher  does something else with the same numbers -- it re-scores the
pairwise matrix and a GLOBAL assignment reads it -- and Mission 9 showed the two
objectives are not the same: on irregular FIT the argmax arms show P_plus == N
while the aligner moves 0.4569 -> 0.5794.

So the coefficients are refitted here against the objective that is deployed.

PROCEDURE, fixed before the first evaluation (this docstring is the contract):

  FIT pool        the six IRREGULAR FIT scenes (names ending in an odd rank) at
                  two of their four views, i1_rot5 and i3_rot15 -- 12 pairs,
                  6 independent scenes.  Chosen for cost, fixed here, never
                  extended after a result is seen.
  Objective       pooled accuracy of the aligner's OWN mapping on the near-tie
                  ambiguous bank (margin < 0.05), computed from the deployed
                  path only.  Unambiguous accuracy and the number of nodes the
                  flag moved are reported beside it, never optimised.
  Search          coordinate ascent, fixed order and grid:
                    beta      in {0.35, 0.50*, 0.65}
                    a_exist   in {0.25, 0.40625*, 0.55}
                    a_geom    in {0.60, 0.84375*, 1.10}
                    cand_cap  in {6*, 12}
                  (* = the Mission-8 bundle.)  One pass; the best value of each
                  coordinate is carried to the next.  No CHECK data of any kind.
  Adoption        a variant REPLACES the bundle only if its FIT gain over the
                  Mission-8 bundle exceeds the FIT MDE (2.8 x the scene-clustered
                  SE of the paired per-scene difference).  Otherwise the bundle
                  is kept and the decision is frozen with the number that made it.

The split of the deployed gain into presence / geometry / context iteration is a
SEPARATE, fixed five-arm run on the same pool:

    N                    flag off
    presence             a_geom = 0, context iterated
    presence, no context a_geom = 0, iters = 0  (the mapping is N's)
    geometry             a_geom = frozen, context iterated
    geometry, no context a_geom = frozen, iters = 0

Truth is used to SCORE these runs and to pick the ambiguous bank.  It is never
passed to the matcher: `joint_mcs` has no truth argument.
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

FIT_VIEWS = ("i1_rot5", "i3_rot15")

#: the fixed grid; None means "the Mission-8 value"
GRID: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("beta", (0.35, None, 0.65)),
    ("a_exist", (0.25, None, 0.55)),
    ("a_geom", (0.60, None, 1.10)),
    ("cand_cap", (None, 12)),
)


def _base_bundle() -> Dict[str, Any]:
    from matching import placement_matcher as PM
    import copy
    return copy.deepcopy(PM.frozen())


def variant_bundle(base: Dict[str, Any], **kw) -> Dict[str, Any]:
    """A candidate bundle: copy of the frozen one with the named fields moved."""
    import copy
    b = copy.deepcopy(base)
    for k, v in kw.items():
        if v is None:
            continue
        if k == "a_exist":
            b["plus"]["a_exist"] = float(v)
        elif k == "a_geom":
            b["plus"]["a_geom"] = float(v)
        elif k == "beta":
            b["beta"] = float(v)
        elif k == "cand_cap":
            b["cand_cap"] = int(v)
        elif k == "iters":
            b["iteration"]["iters"] = int(v)
        else:
            raise KeyError(k)
    return b


# --- worker ----------------------------------------------------------------
_POOL: Dict[str, Any] = {}


def _pair_job(job: Tuple[str, str, str, Dict[str, Any], bool, bool]) -> Dict[str, Any]:
    """One (scene, view) pair through the deployed matcher with one bundle."""
    scene_name, view_label, tag, bundle_kw, presence, use_flag = job
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
    spec = {s.name: s for s in FM.all_specs_irregular(12)}[scene_name]
    view = dict(FM.VIEWS_IRREGULAR)[view_label]
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    sc1, _pv = FM.pose_fn(spec, view)
    r1 = sc1.render()
    g1 = get_or_build(r1.image_u8, spec.name + "|" + view_label)
    truth = {int(q): int(r) for r, q in
             track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    o1 = node_objects(r1, g1)
    pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
    rows = similarity_rows(g1, g0, pop, sorted(int(n) for n in g0.subgraph.nodes))
    amb = set()
    for u in pop:
        cl = rows.get(u) or []
        top = cl[0][1] if cl else 0.0
        sec = cl[1][1] if len(cl) > 1 else 0.0
        if (top - sec) < A2.AMBIGUOUS_MARGIN:
            amb.add(u)
    PM.override(variant_bundle(_base_bundle(), **bundle_kw))
    P.PLACEMENT_MATCHER_ENABLED = bool(use_flag)
    P.PLACEMENT_MATCHER_PRESENCE = bool(presence)
    t = time.time()
    res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                    placement_images=(r1.image_u8, r0.image_u8))
    dt = time.time() - t
    mapping = {int(k): int(v) for k, v in res.mapping.items()}
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_MATCHER_PRESENCE = False
    hit_amb = [1.0 if mapping.get(u) == truth.get(u) else 0.0 for u in sorted(amb)]
    hit_un = [1.0 if mapping.get(u) == truth.get(u) else 0.0
              for u in sorted(set(pop) - amb)]
    # the N mapping, for the "nodes moved" count -- only where it is meaningful,
    # because a second full match per job would double the cost of the search
    if tag == "base" or tag == "N":
        P.PLACEMENT_MATCHER_ENABLED = False
        res_n = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                          placement_images=(r1.image_u8, r0.image_u8))
        map_n = {int(k): int(v) for k, v in res_n.mapping.items()}
        moved = sum(1 for u, v in mapping.items() if map_n.get(u) != v)
    else:
        moved = -1
    return {"scene": scene_name, "view": view_label, "tag": tag,
            "amb": float(np.mean(hit_amb)) if hit_amb else float("nan"),
            "n_amb": len(hit_amb),
            "un": float(np.mean(hit_un)) if hit_un else float("nan"),
            "moved": int(moved), "sec": dt,
            "counters": dict(PM.FAILURES)}


def _run(jobs: List[Any], workers: int = 2) -> List[Dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor
    out: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_pair_job, jobs, chunksize=1):
            out.append(r)
    return out


def fit_pairs() -> List[Tuple[str, str]]:
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    fit, _check = FM.split_scenes(FM.all_specs_irregular(12))
    return [(s.name, v) for s in fit for v in FIT_VIEWS]


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_scene: Dict[str, List[float]] = {}
    for r in rows:
        per_scene.setdefault(r["scene"], []).append(r["amb"])
    scenes = sorted(per_scene)
    sm = {s: float(np.mean(per_scene[s])) for s in scenes}
    return {"amb": float(np.mean([r["amb"] for r in rows])),
            "un": float(np.nanmean([r["un"] for r in rows])),
            "moved": float(np.mean([r["moved"] for r in rows])),
            "sec": float(np.mean([r["sec"] for r in rows])),
            "per_scene": sm, "scenes": scenes}


def paired(base: Dict[str, Any], var: Dict[str, Any]) -> Dict[str, float]:
    keys = [s for s in base["scenes"] if s in var["per_scene"]]
    d = np.array([var["per_scene"][s] - base["per_scene"][s] for s in keys])
    if len(d) < 2:
        return {"mean": float("nan"), "se": float("nan"), "mde": float("nan")}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean": float(d.mean()), "se": se, "mde": 2.8 * se,
            "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "n_scenes": len(d)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("search", "split", "all"), default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="m10_refit")
    args = ap.parse_args()

    t0 = time.time()
    base = _base_bundle()
    pairs = fit_pairs()
    print("FIT pool: %d pairs over %d scenes, views %s"
          % (len(pairs), len({s for s, _ in pairs}), list(FIT_VIEWS)), flush=True)
    record: Dict[str, Any] = {"fit_pairs": [list(p) for p in pairs],
                              "base_bundle_plus": base["plus"],
                              "beta": base["beta"], "cand_cap": base["cand_cap"],
                              "stage": args.stage}

    if args.stage in ("search", "all"):
        cur: Dict[str, Any] = {}
        outcomes: Dict[str, Any] = {}
        base_rows = _run([(s, v, "base", dict(cur), False, True) for s, v in pairs],
                         args.workers)
        base_sum = summarise(base_rows)
        print("\nreference (Mission-8 bundle): amb %.4f  un %.4f  moved %.0f  %.1fs/pair"
              % (base_sum["amb"], base_sum["un"], base_sum["moved"], base_sum["sec"]),
              flush=True)
        outcomes["reference"] = {**base_sum, "kw": dict(cur)}
        best = base_sum
        best_kw = dict(cur)
        for coord, grid in GRID:
            for val in grid:
                kw = dict(best_kw)
                if val is None and coord not in best_kw:
                    continue           # the frozen value is the reference
                kw[coord] = val
                if kw == best_kw:
                    continue
                rows = _run([(s, v, coord, dict(kw), False, True) for s, v in pairs],
                            args.workers)
                sm = summarise(rows)
                d = paired(best, sm)
                print("  %-9s %-8s amb %.4f  un %.4f  moved %3.0f  d(amb) %+0.4f "
                      "mde %.4f  %.1fs/pair"
                      % (coord, val, sm["amb"], sm["un"], sm["moved"], d["mean"],
                         d["mde"], sm["sec"]), flush=True)
                outcomes["%s=%s" % (coord, val)] = {**sm, "kw": dict(kw),
                                                    "vs_prev": d}
                if sm["amb"] > best["amb"]:
                    best, best_kw = sm, kw
            print("  -> %s carried as %s" % (coord, best_kw.get(coord, "frozen")),
                  flush=True)
        # the final variant against the promised reference
        final = outcomes.get("reference")
        cand = None
        for k, v in outcomes.items():
            if v["kw"] == best_kw and k != "reference":
                cand = (k, v)
        d = paired(outcomes["reference"], cand[1] if cand else base_sum)
        adopt = bool(cand and d["mean"] > max(d["mde"], 0.0))
        print("\nbest variant %s: amb %.4f vs reference %.4f -> delta %+0.4f "
              "(MDE %.4f) => %s"
              % (cand[0] if cand else "none", best["amb"], outcomes["reference"]["amb"],
                 d["mean"], d["mde"], "ADOPT" if adopt else "KEEP the bundle"))
        record.update({"search": {k: {kk: vv for kk, vv in v.items() if kk != "per_scene"}
                                  for k, v in outcomes.items()},
                       "best": {"name": cand[0] if cand else None,
                                "kw": best_kw, "kw_delta": best_kw,
                                "amb": best["amb"], "vs_reference": d},
                       "adopted": adopt,
                       "decision_rule": "adopt only if delta > FIT MDE (2.8 x scene SE)",
                       "elapsed_s": time.time() - t0})
        with open(os.path.join(OUT, args.tag + ".json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True, default=float)
        print("wrote %s.json" % args.tag, flush=True)

    if args.stage in ("split", "all"):
        arms = [("N", dict({}), False, False),
                ("presence", dict({}), True, True),
                ("presence_noctx", dict({"iters": 0}), True, True),
                ("geometry", dict({}), False, True),
                ("geometry_noctx", dict({"iters": 0}), False, True)]
        sp: Dict[str, Any] = {}
        ref = None
        for name, kw, pres, flag in arms:
            rows = _run([(s, v, name, dict(kw), pres, flag) for s, v in pairs],
                        args.workers)
            sm = summarise(rows)
            sp[name] = {k: v for k, v in sm.items()}
            if ref is None:
                ref = sm
            print("  %-16s amb %.4f  un %.4f  moved %3.0f  %.1fs/pair"
                  % (name, sm["amb"], sm["un"], sm["moved"], sm["sec"]), flush=True)
        record["split"] = sp
        if "presence" in sp and "N" in sp:
            record["split_reading"] = {
                "context_iteration": sp["presence"]["amb"] - sp["presence_noctx"]["amb"],
                "geometry_beyond_presence": sp["geometry"]["amb"] - sp["presence"]["amb"],
                "presence_over_N": sp["presence"]["amb"] - sp["N"]["amb"],
                "total": sp["geometry"]["amb"] - sp["N"]["amb"]}
            print("\n  split: presence %+0.4f  context %+0.4f  geometry %+0.4f  "
                  "total %+0.4f" % (record["split_reading"]["presence_over_N"],
                                    record["split_reading"]["context_iteration"],
                                    record["split_reading"]["geometry_beyond_presence"],
                                    record["split_reading"]["total"]))
        with open(os.path.join(OUT, args.tag + ".json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True, default=float)
        print("wrote %s.json" % args.tag, flush=True)

    print("\ntotal %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
