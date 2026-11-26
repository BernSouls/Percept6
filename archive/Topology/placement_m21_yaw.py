"""MISSION 21 / O6 -- the yaw wall: measure the descriptor's own invariance, then attack it.

WHAT IS KNOWN FROM M20.  The deployed configuration reaches node accuracy 0.332 at 20 deg
PITCH but only 0.208 at 20 deg YAW, and 0.126 at 35 deg yaw with geo AUC 0.537.  The claim
to test is the mechanism: out-of-plane rotation of a surface about a vertical axis
foreshortens image distances along the horizontal ANISOTROPICALLY, so the sorted vector of
log neighbour distances changes SHAPE, while a uniform scale would only change it by a
constant -- which the estimator is blind to by construction.

PART A -- DIAGNOSIS, no matcher, on the raw descriptor.  For every query place with a true
partner, compare the query descriptor with the partner's reference descriptor and decompose
the difference into:

    shift     mean(d_query - d_reference) over the overlap slots  -- a pure log-scale change
    shape     sd of (d_query - d_reference) after removing that mean -- what the estimator
              actually sees, because `config_similarity` differences the vectors
    angle     mean circular |dtheta| over the overlap slots

and report the self similarity, the best impostor similarity, and the gap, per view.  The
uniform-scale question is MEASURED, not assumed: `config_similarity` differences the vectors,
so a constant c added to every slot of one of them does NOT cancel -- it contributes
n c^2 to the numerator.  Measured: the same descriptor at 1.15x scores 0.923 against itself,
which is a systematic scale penalty, and "ratio" (centring each place) is the variant that
removes it.

PART B -- VARIANTS, cheapest first: `ratio` (centred distances, i.e. scale-free),
`rank` (distance order only, spacing discarded), `angles` (directions only), and `sorted`
(the shipped descriptor).  Each is measured through the
DEPLOYED blend (descriptor weight 0.25, K 24, cap 24, assignment start) so the number is the
one the configuration would actually get, and the sigma grid is chosen on the first half of
the scenes and reported on the second half.

Every cell is a renderer scene with exact truth (`track_optimal`), so this is a probe on
cached graphs, never pooled with ETH-80.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: the views of the diagnosis: two guardrails (identity, in-plane 15 deg, uniform scale) and
#: the four out-of-plane views of the M20 probe
VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("id00", {}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
    ("y20", {"rot_y_deg": 20.0}),
    ("x20", {"rot_x_deg": 20.0}),
    ("y35", {"rot_y_deg": 35.0}),
    ("x35", {"rot_x_deg": 35.0}),
)
VARIANTS = ("sorted", "ratio", "rank", "angles", "seed_only")

#: "seed_only" is not a descriptor variant: it is the DEPLOYED blend with the descriptor's
#: weight set to 0, i.e. the 0.75-weight graded-similarity channel on its own.  It exists to
#: NAME the breaking term: if the seed alone is much worse at yaw than in-plane, the wall is in
#: that channel and not in the configuration descriptor (whose variants are all flat).

#: deterministic caps on the diagnosis (see `diagnose`): places per cell and impostors per
#: place.  They bound a quadratic comparison; the resulting n is printed with every row.
QUERY_CAP = 250
REF_CAP = 80

#: the views the variant stage runs: the two guardrails, the uniform-scale control and the
#: yaw view the acceptance names.  x20/x35 belong to the diagnosis, not to the variant sweep.
VAR_VIEWS = ("id00", "rot15", "scale115", "y20")
SIGMAS = (0.20, 0.35)


def _pool(workers: int):
    """A pool that RECYCLES its workers.

    A cell builds two graphs and two payload sides; the in-process caches do not shrink, and a
    640-cell run died twice with a BrokenProcessPool (RSS, not a bug in a cell).  Recycling
    every 25 cells bounds each worker's lifetime; a cell that raises is recorded, not fatal.
    """
    import multiprocessing as mp
    return mp.Pool(processes=workers, maxtasksperchild=25)


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


def _scene_pair(scene_name: str, label: str):
    """(query graph/side/image, reference graph/side/image, truth, places) for one cell."""
    import parameters as P
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import node_objects, track_optimal
    from matching import placement_matcher as PM
    _recipe()
    spec = {s.name: s for s in FM.confirmation_specs_oop_m10(20)}[scene_name]
    view = dict(VIEWS)[label]
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|m21ref")
    sc1, _pv = FM.pose_fn(spec, view)
    r1 = sc1.render()
    g1 = get_or_build(r1.image_u8, spec.name + "|m21" + label)
    truth = {int(q): int(r) for r, q in
             track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    o1 = node_objects(r1, g1)
    pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
    side_q = PM.build_side_cached(g1, r1.image_u8)
    side_r = PM.build_side_cached(g0, r0.image_u8)
    return g1, side_q, r1.image_u8, g0, side_r, r0.image_u8, truth, pop


def diagnose(job: Tuple[str, str]) -> Dict[str, Any]:
    """PART A: per-node descriptor change under one view, and the self/impostor gap."""
    scene_name, label = job
    import guard
    guard.activate()
    from matching import placement_context as Ctx
    from matching import placement_matcher as PM
    K = 24
    g1, side_q, _i1, g0, side_r, _i0, truth, pop = _scene_pair(scene_name, label)
    ref_nodes = sorted(int(n) for n in g0.subgraph.nodes)
    # the impostor set is CAPPED, deterministically (every k-th place), because a full
    # 1500 x 1500 comparison per cell is quadratic in places and this is a diagnosis of the
    # descriptor's geometry, not a matching run.  The cap and the resulting n are recorded.
    stride_r = max(1, len(ref_nodes) // REF_CAP)
    ref_nodes = ref_nodes[::stride_r]
    ref_desc = {v: Ctx.config_descriptor(side_r, v, K) for v in ref_nodes}
    ref_live = [d for d in ref_desc.values() if d is not None]
    shifts, shapes, angs, selfs, imps, aucs = [], [], [], [], [], []
    var_shift, var_res = [], []
    stride_q = max(1, len(pop) // QUERY_CAP)
    for u in pop[::stride_q]:
        v = truth.get(u)
        dq = Ctx.config_descriptor(side_q, u, K)
        dr = ref_desc.get(v)
        if dq is None or dr is None:
            continue
        n = min(len(dq[0]), len(dr[0]))
        if n < 2:
            continue
        dd = [dq[0][i] - dr[0][i] for i in range(n)]
        da = [abs((dq[1][i] - dr[1][i] + math.pi) % (2.0 * math.pi) - math.pi)
              for i in range(n)]
        da = [min(x, math.pi - x) for x in da]
        mu = sum(dd) / n
        res = [x - mu for x in dd]
        shifts.append(sum(abs(x) for x in dd) / n)
        shapes.append(math.sqrt(sum(x * x for x in res) / n))
        angs.append(sum(da) / n)
        var_shift.append(math.sqrt(sum(x * x for x in dd) / n))
        var_res.append(math.sqrt(sum(x * x for x in res) / n))
        selfs.append(Ctx.config_similarity(dq, dr))
        # SUPERSEDED METRIC, kept only so the defect is visible: the first pass of this probe
        # recorded the MAX impostor similarity over a set that included the truth, which makes
        # "impostor" an order statistic of 80 draws and not comparable with a single self
        # value -- the identity row read self 1.000 / impostor 1.000 and every rotated row read
        # impostor ABOVE self.  The column is replaced by the MEAN impostor and by the AUC
        # (the fraction of true impostors below the self value, truth excluded), which is the
        # quantity Mission 19 reported; neither was re-measured before this file stopped.
        _imps = [Ctx.config_similarity(dq, other) for other in ref_live]
        imps.append(float(np.mean(_imps)) if _imps else 0.0)
        aucs.append(float(np.mean([1.0 if s_ < selfs[-1] else 0.0 for s_ in _imps]))
                    if _imps else 0.0)
    _ = PM
    return {"scene": scene_name, "view": label, "n": len(selfs),
            "n_ref_places": len(ref_live), "n_query_places": len(pop),
            "shift": float(np.mean(shifts)) if shifts else None,
            "shape": float(np.mean(shapes)) if shapes else None,
            "angle": float(np.mean(angs)) if angs else None,
            "rms_delta": float(np.mean(var_shift)) if var_shift else None,
            "rms_residual": float(np.mean(var_res)) if var_res else None,
            "self": float(np.mean(selfs)) if selfs else None,
            "impostor_mean": float(np.mean(imps)) if imps else None,
            "auc": float(np.mean(aucs)) if aucs else None,
            "superseded": True}


def accuracy(job: Tuple[str, str, str, float]) -> Dict[str, Any]:
    """PART B: node accuracy of the DEPLOYED blend under one descriptor variant."""
    try:
        return _accuracy(job)
    except Exception as exc:                                  # noqa: BLE001
        return {"scene": job[0], "view": job[1], "variant": job[2],
                "sigma_d": float(job[3]), "n": 0, "acc": None,
                "error": repr(exc)[:200]}


def _accuracy(job: Tuple[str, str, str, float]) -> Dict[str, Any]:
    scene_name, label, variant, sigma_d = job
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_context as Ctx
    from matching import placement_matcher as PM
    from Topology.placement_census import similarity_rows
    P.PLACEMENT_NODE_DESC_VARIANT = "sorted" if variant == "seed_only" else variant
    old = Ctx.SIGMA_D, Ctx.SIGMA_A
    old_w = getattr(P, "PLACEMENT_NODE_DESC_W", None)
    if variant == "seed_only":
        P.PLACEMENT_NODE_DESC_W = 0.0
    Ctx.SIGMA_D = float(sigma_d)
    try:
        g1, side_q, i1, g0, side_r, i0, truth, pop = _scene_pair(scene_name, label)
        nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
        nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
        rows = similarity_rows(g1, g0, nodes_a, nodes_b)
        idx_b = {v: j for j, v in enumerate(nodes_b)}
        seed = [[0.0] * len(nodes_b) for _ in nodes_a]
        for i, u in enumerate(nodes_a):
            for (v, s) in rows.get(u, ()):
                if s > 0.0:
                    seed[i][idx_b[v]] = float(s)
        if variant == "seed_only":
            # THE 0.75 CHANNEL ON ITS OWN.  No rescore: the seed IS the graded-similarity
            # contribution, and with the descriptor's weight at 0 the shipped blend ranks by
            # exactly this matrix.  Computing it directly is the same number for a hundredth
            # of the cost, which is what makes naming the breaking term affordable.
            idx_a = {u: i for i, u in enumerate(nodes_a)}
            hit = n = 0
            for u in pop:
                i = idx_a.get(u)
                if i is None:
                    continue
                live = [(nodes_b[j], float(seed[i][j])) for j in range(len(nodes_b))
                        if seed[i][j] > 0.0]
                if len(live) < 2:
                    continue
                pick = max(live, key=lambda t: (t[1], -t[0]))[0]
                n += 1
                hit += 1 if pick == truth.get(u) else 0
            return {"scene": scene_name, "view": label, "variant": variant,
                    "sigma_d": float(sigma_d), "n": n, "acc": (hit / n) if n else None}
        stats: Dict[str, Any] = {}
        scored = PM.rescore_matrix(seed, nodes_a, nodes_b, side_q, side_r, stats=stats)
        idx_a = {u: i for i, u in enumerate(nodes_a)}
        hit = n = 0
        for u in pop:
            i = idx_a.get(u)
            if i is None:
                continue
            live = [(nodes_b[j], float(scored[i][j])) for j in range(len(nodes_b))
                    if scored[i][j] > 0.0]
            if len(live) < 2:
                continue
            pick = max(live, key=lambda t: (t[1], -t[0]))[0]
            n += 1
            hit += 1 if pick == truth.get(u) else 0
        return {"scene": scene_name, "view": label, "variant": variant,
                "sigma_d": float(sigma_d), "n": n, "acc": (hit / n) if n else None}
    finally:
        P.PLACEMENT_NODE_DESC_VARIANT = "sorted"
        Ctx.SIGMA_D, Ctx.SIGMA_A = old
        if old_w is None:
            try:
                delattr(P, "PLACEMENT_NODE_DESC_W")
            except AttributeError:
                pass
        else:
            P.PLACEMENT_NODE_DESC_W = old_w


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="diag",
                    choices=("diag", "variants", "summary", "all"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--tag", default="m21_yaw")
    # [M23] READ-ONLY extension: the mission allowed the missing PITCH cell to be added to the
    # accuracy harness and nothing else.  Both defaults are the M21 values, so every M21 number
    # reproduces bit-for-bit; a caller passes e.g. --views y20,x20 --variants sorted to measure
    # pitch and yaw side by side without touching M21's own files (a separate --tag).
    ap.add_argument("--views", default="")
    ap.add_argument("--variants", default="")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from Topology import placement_families_m9 as FM
    if not P.PLACEMENT_M19:
        raise SystemExit("run with PLACEMENT_M19=1: this reads the deployed configuration")
    if a.stage == "summary":
        r = summary()
        print("summary: %d diagnosis cells, %d variant cells"
              % (r["n_diag_cells"], r["n_variant_cells"]))
        for v, row in r["diagnosis"].items():
            print("  %-9s n %-5d shift %.3f shape %.3f angle %.3f"
                  % (v, row["n"], row["shift"], row["shape"], row["angle"]))
        for var, views in r["variants"].items():
            print("  %-9s %s" % (var, {k: round(x["acc"], 3) for k, x in views.items()}))
        return 0
    specs = FM.confirmation_specs_oop_m10(a.scenes)[:a.scenes]
    names = [s.name for s in specs]
    res: Dict[str, Any] = {"views": [v for v, _ in VIEWS], "scenes": len(names),
                           "recipe": "M16", "variant_default": "sorted"}
    prog = os.path.join(OUT, a.tag + "_diag.jsonl")
    if a.stage in ("diag", "all"):
        done = [json.loads(x) for x in open(prog) if x.strip()] if os.path.exists(prog) else []
        seen = {(d["scene"], d["view"]) for d in done}
        todo = [(s, v) for s in names for v, _x in VIEWS if (s, v) not in seen]
        print("diagnosis: %d done, %d to go" % (len(done), len(todo)), flush=True)
        t0 = time.time()
        if todo:
            with open(prog, "a") as fh, _pool(a.workers) as ex:
                for i, r in enumerate(ex.imap_unordered(diagnose, todo, chunksize=1)):
                    r.pop("superseded", None)
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                    done.append(r)
                    if (i + 1) % 10 == 0:
                        print("  %d/%d %.0fs" % (i + 1, len(todo), time.time() - t0),
                              flush=True)
        res["diagnosis"] = done
        # the uniform-scale guardrail: a constant shift in log distances must be invisible
        from matching import placement_context as Ctx
        d = ([1.2, 1.9, 2.4, 3.0], [0.1, 0.9, 2.0, -1.4])
        c = math.log(1.15)
        shifted = ([x + c for x in d[0]], list(d[1]))
        res["scale_invariance"] = {"similarity": Ctx.config_similarity(d, shifted),
                                   "shift": c}
        print("\n%-9s %-5s %-6s %-6s %-6s %-6s %-6s %-6s"
              % ("view", "n", "self", "impost", "gap", "shift", "shape", "angle"))
        for v, _x in VIEWS:
            rs = [d for d in done if d["view"] == v and d.get("n")]
            if not rs:
                continue
            n = sum(r["n"] for r in rs)
            f = lambda k: sum((r[k] or 0) * r["n"] for r in rs) / n
            print("%-9s %-5d %-6.3f %-6.3f %-6.3f %-6.3f %-6.3f %-6.3f"
                  % (v, n, f("self"), f("impostor"), f("self") - f("impostor"),
                     f("shift"), f("shape"), f("angle")))
        print("scale invariance: %.17f (shift %.4f)"
              % (res["scale_invariance"]["similarity"], c))
    if a.stage in ("variants", "all"):
        prog2 = os.path.join(OUT, a.tag + "_variants.jsonl")
        done2 = [json.loads(x) for x in open(prog2) if x.strip()] if os.path.exists(prog2) else []
        seen2 = {(d["scene"], d["view"], d["variant"], round(d["sigma_d"], 3))
                 for d in done2}
        # sigma is chosen on the FIRST HALF of the scenes and reported on the second half
        half = names[: len(names) // 2]
        var_views = tuple(x for x in a.views.split(",") if x) or VAR_VIEWS
        var_variants = tuple(x for x in a.variants.split(",") if x) or VARIANTS
        todo2 = [(s, v, var, sg) for s in names for v in var_views
                 for var in var_variants for sg in SIGMAS
                 if (s, v, var, round(sg, 3)) not in seen2]
        print("\nvariants: %d to go" % len(todo2), flush=True)
        t0 = time.time()
        if todo2:
            with open(prog2, "a") as fh, _pool(a.workers) as ex:
                for i, r in enumerate(ex.imap_unordered(accuracy, todo2, chunksize=1)):
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                    done2.append(r)
                    if (i + 1) % 20 == 0:
                        print("  %d/%d %.0fs" % (i + 1, len(todo2), time.time() - t0),
                              flush=True)
        # choose sigma per variant on the fit half, at y20
        chosen = {}
        for var in VARIANTS:
            best = None
            for sg in SIGMAS:
                rs = [d for d in done2 if d["variant"] == var and d["view"] == "y20"
                      and abs(d["sigma_d"] - sg) < 1e-9 and d["scene"] in half and d["n"]]
                if not rs:
                    continue
                q = sum(r["n"] for r in rs)
                acc = sum((r["acc"] or 0) * r["n"] for r in rs) / q
                if best is None or acc > best[1]:
                    best = (sg, acc, q)
            chosen[var] = {"sigma_d": best[0], "fit_half_acc_y20": best[1],
                           "fit_half_n": best[2]} if best else None
        table = {}
        other = [s for s in names if s not in half]
        for var, ch in chosen.items():
            if not ch:
                continue
            for v in ("id00", "rot15", "scale115", "y20", "x20"):
                rs = [d for d in done2 if d["variant"] == var and d["view"] == v
                      and abs(d["sigma_d"] - ch["sigma_d"]) < 1e-9 and d["scene"] in other
                      and d["n"]]
                if not rs:
                    continue
                q = sum(r["n"] for r in rs)
                table.setdefault(var, {})[v] = {
                    "acc": sum((r["acc"] or 0) * r["n"] for r in rs) / q, "n": q,
                    "cells": len(rs),
                    "spread": [min(r["acc"] for r in rs), max(r["acc"] for r in rs)]}
        res["variants"] = {"chosen": chosen, "report_half": table,
                           "fit_half_scenes": half, "report_scenes": other}
        print("\nreport half (%d scenes), sigma chosen on the fit half:" % len(other))
        print("%-8s %-8s %-8s %-8s %-8s" % ("variant", "id00", "rot15", "scale", "y20"))
        for var, row in sorted(table.items()):
            print("%-8s %-8s %-8s %-8s %-8s"
                  % (var, *[("%.3f" % row[v]["acc"]) if v in row else "-"
                            for v in ("id00", "rot15", "scale115", "y20")]))
    out = os.path.join(OUT, a.tag + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("\nwrote", out)
    return 0

def summary(out_path: Optional[str] = None) -> Dict[str, Any]:
    """The combined artifact: the diagnosis table and the variant table, from the row files.

    Written because the two stages of this file write the same JSON path in turn, and the
    diagnosis (140 cells) was overwritten by the variant stage (800 cells).  The rows were kept
    (`--` per-cell rows are a quality bar), so the summary is derived from them and nothing had
    to be re-run.
    """
    import numpy as np
    diag = []
    p1 = os.path.join(OUT, "m21_yaw_diag.jsonl")
    if os.path.exists(p1):
        diag = [json.loads(x) for x in open(p1) if x.strip()]
    var = []
    p2 = os.path.join(OUT, "m21_yaw_variants.jsonl")
    if os.path.exists(p2):
        var = [json.loads(x) for x in open(p2) if x.strip()]
    table = {}
    for v, _x in VIEWS:
        rs = [d for d in diag if d["view"] == v and d.get("n")]
        if not rs:
            continue
        n = sum(r["n"] for r in rs)
        f = lambda k: sum((r.get(k) or 0) * r["n"] for r in rs) / n
        table[v] = {"n": n, "cells": len(rs), "shift": f("shift"), "shape": f("shape"),
                    "angle": f("angle"), "self": f("self")}
    acc: Dict[str, Dict[str, Any]] = {}
    for r in var:
        if not r.get("n"):
            continue
        acc.setdefault(r["variant"], {}).setdefault(r["view"], []).append(r)
    variants = {}
    for variant, views in sorted(acc.items()):
        variants[variant] = {}
        for view, rs in views.items():
            n = sum(r["n"] for r in rs)
            variants[variant][view] = {
                "acc": sum((r["acc"] or 0) * r["n"] for r in rs) / n, "n": n,
                "cells": len(rs), "spread": [min(r["acc"] for r in rs),
                                             max(r["acc"] for r in rs)]}
    res = {"diagnosis": table, "variants": variants,
           "diag_scenes": len({d["scene"] for d in diag}),
           "n_diag_cells": len(diag), "n_variant_cells": len(var),
           "caps": {"query_places": QUERY_CAP, "impostors": REF_CAP},
           "views": [v for v, _x in VIEWS], "var_views": list(VAR_VIEWS)}
    from matching import m15_manifest as MF
    MF.write_result(out_path or os.path.join(OUT, "m21_yaw_summary.json"), res)
    return res

if __name__ == "__main__":
    raise SystemExit(main())
