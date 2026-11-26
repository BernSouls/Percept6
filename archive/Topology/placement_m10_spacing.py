"""MISSION 10 STEP 4a -- measure the spacing, fit the rule, adopt it or not.

Three stages, all on SYNTHETIC FIT ONLY (the real images are the report):

  measure   the true-pair residual against the local place spacing, in spacing
            bins, and the ratio of the builder's scale-relative neighbour radius
            to that spacing.  This is the regression the mission asks for: it
            says how much of the residual is explained by how close together the
            places are, which is what decides the SHAPE of a scaling rule.

  fit       two 1-D searches, each evaluated by the deployed matcher's own
            ambiguous-bank accuracy on the irregular FIT pool:
              tolerance  gamma in {0.0*, 0.5, 1.0}   tol = base*(sp/ref)**gamma
              radius     factor in {None*, 1.0, 1.5, 2.5}   radius <= factor*sp
            (* = the shipped behaviour.)  A rule is ADOPTED only if its FIT gain
            over the shipped behaviour exceeds the FIT MDE; otherwise the flag
            stays dark and the number that kept it dark is recorded.

  params    writes matching/placement_spacing_params.json -- the file the dark
            flags read.  Written even when nothing is adopted, with the fitted
            numbers marked as such, so the rule exists to be tested by the
            report's own numbers rather than being retyped later.

Nothing here touches CHECK data, and nothing here is fitted on a real image.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PARAMS = os.path.join(ROOT, "matching", "placement_spacing_params.json")

FIT_VIEWS = ("i1_rot5", "i3_rot15")


# ---------------------------------------------------------------------------
def measure() -> Dict[str, Any]:
    """True-pair residual vs local place spacing, on synthetic FIT."""
    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from matching import placement_llr as L
    from matching.placement_scorer import payload_residual
    from Topology import placement_arms2 as A2
    from Topology import placement_families_m9 as FM
    from core.place_key import place_lenses

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    fit, _chk = FM.split_scenes(FM.all_specs_irregular(12))
    views = [(lbl, dict(FM.VIEWS_IRREGULAR)[lbl]) for lbl in FIT_VIEWS]
    pairs: List[Tuple[float, float]] = []          # (residual, spacing)
    radius_ratio: List[float] = []
    for vw in M3.iter_views(fit, views, None, FM.pose_fn):
        truth, q_inc, q_idx, r_idx = (vw["truth"], vw["q_inc"], vw["q_idx"],
                                      vw["r_idx"])
        g1 = vw["g1"]
        # spacing of the places themselves
        places = {}
        for n in sorted(g1.subgraph.nodes):
            a = g1.subgraph.anchor_of_node.get(n)
            if a is not None:
                places[int(n)] = (float(a[0]), float(a[1]))
        sp: Dict[int, float] = {}
        arr_nodes = sorted(places)
        if len(arr_nodes) > 1:
            arr = np.array([places[n] for n in arr_nodes], dtype=float)
            # distinct places only: duplicates are the same place, not a neighbour
            uniq, inv = np.unique(np.round(arr, 6), axis=0, return_inverse=True)
            if len(uniq) > 1:
                d = np.hypot(uniq[:, None, 0] - uniq[None, :, 0],
                             uniq[:, None, 1] - uniq[None, :, 1])
                np.fill_diagonal(d, np.inf)
                nearest = np.min(d, axis=1)
                for k, n in enumerate(arr_nodes):
                    sp[n] = float(nearest[int(inv[k])])
        for u in vw["pop"]:
            tv = truth.get(u)
            if tv is None:
                continue
            for (other, is_source, name) in q_inc.get(int(u), ()):
                mo = truth.get(int(other))
                if mo is None:
                    continue
                if is_source:
                    qv = q_idx.get((int(u), int(other), name))
                    rv = r_idx.get((int(tv), int(mo), name))
                else:
                    qv = q_idx.get((int(other), int(u), name))
                    rv = r_idx.get((int(mo), int(tv), name))
                if qv is None or rv is None:
                    continue
                s = max(sp.get(int(u), float("nan")), sp.get(int(other), float("nan")))
                if not np.isfinite(s):
                    continue
                pairs.append((float(payload_residual(qv, rv, L.FIXED_TOL)), float(s)))
        for u in vw["pop"]:
            su = sp.get(int(u))
            sc = None
            a = g1.subgraph.anchor_of_node.get(int(u))
            if su and su > 0 and a is not None:
                radius_ratio.append(float(su))
    res = np.array([p[0] for p in pairs], dtype=float)
    sps = np.array([p[1] for p in pairs], dtype=float)
    bins = [(0, 3), (3, 4.5), (4.5, 6), (6, 9), (9, 99)]
    table = {}
    for lo, hi in bins:
        m = (sps >= lo) & (sps < hi)
        if m.sum() < 40:
            continue
        table["%g-%g" % (lo, hi)] = {
            "n": int(m.sum()), "spacing_median": float(np.median(sps[m])),
            "res_median": float(np.median(res[m])),
            "res_p95": float(np.percentile(res[m], 95)),
            "res_p99": float(np.percentile(res[m], 99))}
    xs = np.array([v["spacing_median"] for v in table.values()])
    ys = np.array([v["res_p95"] for v in table.values()])
    gamma = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if len(xs) >= 2 else 1.0
    lin = np.polyfit(xs, ys, 1)
    spacings = np.array(sorted(sp for sp in (v["spacing_median"] for v in table.values())))
    out = {"pairs": int(res.size), "spacing_median": float(np.median(sps)),
           "res_median": float(np.median(res)), "res_p95": float(np.percentile(res, 95)),
           "bins": table, "power_law_gamma": gamma,
           "linear": {"c0": float(lin[1]), "c1": float(lin[0])},
           "frozen_tolerance": float(L.FIXED_TOL),
           "radius_to_spacing": {
               "median_px": float(np.median(radius_ratio)) if radius_ratio else float("nan"),
               "note": ("radius_ratio is the local place spacing in px; the "
                        "builder's radius is TARGET_RADIUS_SCALES x the node's "
                        "own scale")}}
    path = os.path.join(OUT, "m10_spacing_measure.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("spacing regression: %d true pairs, spacing median %.2f px"
          % (out["pairs"], out["spacing_median"]))
    for k, v in table.items():
        print("  spacing %-6s n=%5d  res median %.3f  p95 %.3f  p99 %.3f"
              % (k, v["n"], v["res_median"], v["res_p95"], v["res_p99"]))
    print("  power-law exponent %.3f, linear c0 %.3f c1 %.3f" % (gamma, lin[1], lin[0]))
    print("wrote %s" % path)
    return out


# ---------------------------------------------------------------------------
# the deployed-objective search (one worker, pairs through the real matcher)
def _eval(kind: str, value: Any, workers: int = 2) -> Dict[str, Any]:
    import guard
    guard.activate()
    from concurrent.futures import ProcessPoolExecutor
    from Topology import placement_families_m9 as FM
    fit, _c = FM.split_scenes(FM.all_specs_irregular(12))
    jobs = [(s.name, v, kind, value) for s in fit for v in FIT_VIEWS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_worker, jobs, chunksize=1):
            rows.append(r)
    per: Dict[str, List[float]] = {}
    for r in rows:
        per.setdefault(r["scene"], []).append(r["amb"])
    scenes = sorted(per)
    sm = {s: float(np.mean(per[s])) for s in scenes}
    return {"amb": float(np.mean([r["amb"] for r in rows])),
            "per_scene": sm, "scenes": scenes,
            "moved": float(np.mean([r["moved"] for r in rows]))}


def _worker(job):
    scene_name, view_label, kind, value = job
    import guard
    guard.activate()
    import importlib
    import parameters as P
    P = importlib.reload(P)
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
        t0 = cl[0][1] if cl else 0.0
        t1 = cl[1][1] if len(cl) > 1 else 0.0
        if (t0 - t1) < A2.AMBIGUOUS_MARGIN:
            amb.add(u)
    if kind == "tol":
        P.PLACEMENT_SPACING_TOL = True
        _write_params({"tol": {"gamma": float(value)}})
    else:
        P.PLACEMENT_SPACING_RADIUS = True
        _write_params({"radius": {"factor": float(value)}})
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_MATCHER_PRESENCE = False
    res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                    placement_images=(r1.image_u8, r0.image_u8))
    mapping = {int(k): int(v) for k, v in res.mapping.items()}
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SPACING_TOL = False
    P.PLACEMENT_SPACING_RADIUS = False
    hit = [1.0 if mapping.get(u) == truth.get(u) else 0.0 for u in sorted(amb)]
    return {"scene": scene_name, "view": view_label,
            "amb": float(np.mean(hit)) if hit else float("nan"),
            "moved": 0}


def _write_params(update: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> None:
    """Merge an update into the spacing params file (used by the search workers)."""
    cur = base
    if cur is None:
        if os.path.exists(PARAMS):
            with open(PARAMS) as fh:
                cur = json.load(fh)
        else:
            cur = {"version": 1, "spacing_ref": 6.324555320336759,
                   "tol": {"gamma": 1.0, "lo": 0.5, "hi": 2.0},
                   "radius": {"factor": 1.5, "min_px": 1.0},
                   "fitted_on": "synthetic FIT only"}
    for k, v in update.items():
        if isinstance(v, dict):
            cur.setdefault(k, {}).update(v)
        else:
            cur[k] = v
    with open(PARAMS, "w") as fh:
        json.dump(cur, fh, indent=2, sort_keys=True)


def paired(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    keys = [s for s in a["scenes"] if s in b["per_scene"]]
    d = np.array([b["per_scene"][s] - a["per_scene"][s] for s in keys])
    if len(d) < 2:
        return {"mean": float("nan"), "se": float("nan"), "mde": float("nan")}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean": float(d.mean()), "se": se, "mde": 2.8 * se}


def fit(workers: int = 2) -> Dict[str, Any]:
    _write_params({})            # make sure the file exists with defaults
    base = _eval("tol", 0.0, workers)      # gamma 0 == the shipped tolerance
    print("shipped: amb %.4f" % base["amb"], flush=True)
    out: Dict[str, Any] = {"shipped": base, "tol": {}, "radius": {}}
    for g in (0.5, 1.0):
        r = _eval("tol", g, workers)
        d = paired(base, r)
        out["tol"]["%.2f" % g] = {**r, "vs_shipped": d}
        print("  tol gamma %.1f: amb %.4f  d %+0.4f  mde %.4f"
              % (g, r["amb"], d["mean"], d["mde"]), flush=True)
    for f in (1.0, 1.5, 2.5):
        r = _eval("radius", f, workers)
        d = paired(base, r)
        out["radius"]["%.2f" % f] = {**r, "vs_shipped": d}
        print("  radius factor %.1f: amb %.4f  d %+0.4f  mde %.4f"
              % (f, r["amb"], d["mean"], d["mde"]), flush=True)
    best_tol = max(out["tol"].items(), key=lambda kv: kv[1]["amb"])
    best_rad = max(out["radius"].items(), key=lambda kv: kv[1]["amb"])
    adopt_tol = best_tol[1]["vs_shipped"]["mean"] > max(
        best_tol[1]["vs_shipped"]["mde"], 0.0)
    adopt_rad = best_rad[1]["vs_shipped"]["mean"] > max(
        best_rad[1]["vs_shipped"]["mde"], 0.0)
    out["decision"] = {
        "tol": {"value": best_tol[0], "adopted": bool(adopt_tol),
                "delta": best_tol[1]["vs_shipped"]},
        "radius": {"value": best_rad[0], "adopted": bool(adopt_rad),
                   "delta": best_rad[1]["vs_shipped"]},
        "rule": "adopt only if FIT gain exceeds FIT MDE; flags stay dark either way"}
    _write_params({"tol": {"gamma": float(best_tol[0])},
                   "radius": {"factor": float(best_rad[0])},
                   "fitted_on": "synthetic FIT only (irregular FIT, rot5/rot15)"})
    path = os.path.join(OUT, "m10_spacing_fit.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=float)
    print("decision: tol gamma %s adopted=%s ; radius factor %s adopted=%s"
          % (best_tol[0], adopt_tol, best_rad[0], adopt_rad))
    print("wrote %s" % path)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("measure", "fit", "both"), default="both")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.stage in ("measure", "both"):
        measure()
    if args.stage in ("fit", "both"):
        fit(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
