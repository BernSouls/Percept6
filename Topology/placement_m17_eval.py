"""MISSION 17 -- the evaluation harness for O1, O2 and O3.

Everything here is read off the CACHED M16 graphs (`m16|<tag>|<pose>|on`), which were
built with the graded constitution ON and the population payload attached.  No graph is
rebuilt, so the readings differ from M16's only by what this mission changes: the LLR
table and the context iteration constants.

ARMS are (table, context) pairs:
    base         deployed table, frozen context constants   <- the incumbent
    llr17        the refitted table, frozen context
    any          levers set explicitly through `context_mapping(..., iters=, margin=,
                 max_rivals=, rival_tol=, mutual=)`, which M17 extended to accept the
                 last two (the frozen path is unchanged when they are None)

PER QUERY (place lens, true partner admissible, >=1 impostor):
    appearance top-1     max(sim|true) > max(sim|imp)
    top-1                max(score|true) > max(score|imp), score = (1-b)sim + b*plus
    geometry AUC         the geometry term alone, matcher context and truth context
    failure attribution  which term put the impostor first: APPEARANCE / GEOMETRY /
                         CONTEXT (geometry right under the truth context but wrong under
                         the matcher's) / BLEND (both terms right, the blend wrong)

    python Topology/placement_m17_eval.py --arm base --poses rot15 --images syn|m11ir_00
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
IMAGE_SETS: Dict[str, Tuple[str, ...]] = {
    "dev": ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass"),
    "fresh": ("syn|m17ir_00",),
}
#: the PRE-REGISTERED fresh set: 16 new-seed synthetic scenes + the four photographs
#: M15/M16 never touched.  Built by `placement_m17_fresh`, whose docstring records that
#: the composition was fixed before any of these graphs existed.
from Topology.placement_m17_fresh import tags as _fresh_tags  # noqa: E402
IMAGE_SETS["fresh20"] = _fresh_tags()
M16 = dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
           FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
           PLACEMENT_FRAME_KEYPOINT_SIGMA="largest", PLACEMENT_FRAME_FALLBACK="none",
           PLACEMENT_M15_PLACES=True, KP_GRADED_SIMILARITY=True)

#: name -> dict(table=..., iters=, margin=, max_rivals=, rival_tol=, mutual=)
ARMS: Dict[str, Dict[str, Any]] = {
    "base": {"table": "m16"},
    "llr17": {"table": "m17"},
}


def arm(name: str) -> Dict[str, Any]:
    if name in ARMS:
        return dict(ARMS[name])
    # ctx-it2-m0-r1-rt02-mut0  style slugs
    import re
    m = re.match(r"^asg-l([0-9.]+)-it(\d+)$", name)
    if m:
        return {"table": "m17", "assign": True, "lam": float(m.group(1)),
                "assign_iters": int(m.group(2))}
    m = re.match(r"^ctx-it(\d+)-m([0-9.]+)-r(\d+)-rt([0-9.]+)-mut([01])$", name)
    if not m:
        raise KeyError("unknown arm %r" % (name,))
    return {"table": "m17", "iters": int(m.group(1)), "margin": float(m.group(2)),
            "max_rivals": int(m.group(3)), "rival_tol": float(m.group(4)),
            "mutual": bool(int(m.group(5)))}


def _image(tag: str) -> np.ndarray:
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    from Topology import placement_m17_fresh as FR
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        if name.startswith("m17ir_"):
            return [s for s in FR.fresh_gallery() if s.name == name][0].scene.render().image_u8
        return [s for s in F.gallery() if s.name == name][0].scene.render().image_u8
    return RP.images("validation")[tag.split("|", 1)[1]]


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def _job(args: Tuple[str, str, Dict[str, Any], str]) -> Dict[str, Any]:
    tag, pose, spec, arm_name = args[:4]
    table_override = args[4] if len(args) > 4 else None
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build
    from core import keypoint_classes as KPC

    cfg = arm(arm_name)
    if table_override:
        cfg["table"] = table_override
    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True           # the payload is in these cached graphs
    P.PLACEMENT_LLR_TABLE_M17 = (cfg.get("table") == "m17")
    P.PLACEMENT_CONTEXT_M17 = False         # levers are passed explicitly below
    P.PLACEMENT_CONTEXT_ASSIGN = bool(cfg.get("assign", False))
    if cfg.get("assign"):
        P.PLACEMENT_CONTEXT_ASSIGN_LAMBDA = float(cfg.get("lam", 0.5))
        P.PLACEMENT_CONTEXT_ASSIGN_ITERS = int(cfg.get("assign_iters", 0))
    PM.reset_cache()
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()

    img = _image(tag)
    img1 = R._transform(img, spec)
    g0 = get_or_build(img, "m16|%s|orig|on" % tag)
    g1 = get_or_build(img1, "m16|%s|%s|on" % (tag, pose))
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    side_q = PM.build_side_cached(g1, img1)
    side_r = PM.build_side_cached(g0, img)
    truth = PL.lens_truth(g1, g0, spec, img.shape[:2])
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    tol_of = PM._spacing_tol_fn(side_q, side_r)
    beta = float(b["beta"])

    rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
    idx_b = {v: j for j, v in enumerate(nodes_b)}
    matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
    for i, u in enumerate(nodes_a):
        for (v, s) in rows_sim.get(u, ()):
            if s > 0.0:
                matrix[i][idx_b[v]] = float(s)
    triples = PM.top_candidates(matrix, nodes_b, int(b["cand_cap"]))
    levers = {k: cfg[k] for k in ("iters", "margin", "max_rivals", "rival_tol",
                                  "mutual") if k in cfg}
    ctx = PM.context_mapping(triples, nodes_a, side_q, side_r, **levers)
    ctx_truth = dict(truth)

    def e_of(u: int, v: int, mapping: Dict[int, int]) -> float:
        return float(L.plus_score(u, int(v), mapping, side_q.incident, side_q.index,
                                  side_r.index, fit, pfit, a_e, a_g,
                                  slot_cap=int(L.SLOT_CAP),
                                  tol=(None if tol_of is None else tol_of(u, v)))) - 0.5

    app_pairs: List[Tuple[float, List[float]]] = []
    geo_m: List[Tuple[float, List[float]]] = []
    geo_t: List[Tuple[float, List[float]]] = []
    counts = {"queries": 0, "top1": 0, "top1_app": 0, "top1_geo_true": 0,
              "app_ok": 0, "geo_ok_matcher": 0, "geo_ok_truth": 0}
    fails: Dict[str, int] = {"APPEARANCE": 0, "GEOMETRY": 0, "CONTEXT": 0, "BLEND": 0}
    query_rows: List[Dict[str, Any]] = []
    for i, u in enumerate(nodes_a):
        vt = truth.get(u)
        if vt is None:
            continue
        row = [(int(nodes_b[j]), float(matrix[i][j])) for j in range(len(nodes_b))
               if matrix[i][j] > 0.0]
        if len(row) < 2 or vt not in {v for (v, _s) in row}:
            continue
        imps = [v for (v, _s) in row if v != vt]
        if not imps:
            continue
        sim = dict(row)
        em = {v: e_of(u, v, ctx) for (v, _s) in row}
        et = {v: e_of(u, v, ctx_truth) for (v, _s) in row}
        s_full = {v: (1.0 - beta) * sim[v] + beta * (0.5 + em[v]) for (v, _s) in row}
        # appearance-only arm: w = 0 is exactly `sim`
        best_app = max(sim[v] for v in imps)
        best_full = max(s_full[v] for v in imps)
        best_em = max(em[v] for v in imps)
        best_et = max(et[v] for v in imps)
        counts["queries"] += 1
        counts["app_ok"] += 1 if sim[vt] > best_app else 0
        counts["geo_ok_matcher"] += 1 if em[vt] > best_em else 0
        counts["geo_ok_truth"] += 1 if et[vt] > best_et else 0
        ok = s_full[vt] > best_full
        counts["top1"] += 1 if ok else 0
        counts["top1_app"] += 1 if sim[vt] > best_app else 0
        counts["top1_geo_true"] += 1 if et[vt] > best_et else 0
        app_pairs.append((sim[vt], [sim[v] for v in imps]))
        geo_m.append((em[vt], [em[v] for v in imps]))
        geo_t.append((et[vt], [et[v] for v in imps]))
        if not ok:
            # ---- O3: WHERE the loss is
            if sim[vt] <= best_app and et[vt] <= best_et:
                fails["APPEARANCE"] += 1
            elif em[vt] <= best_em and et[vt] > best_et:
                fails["CONTEXT"] += 1
            elif et[vt] <= best_et and em[vt] > best_em:
                fails["GEOMETRY"] += 1          # right under matcher, wrong under truth
            elif sim[vt] > best_app and et[vt] > best_et and em[vt] > best_em:
                fails["BLEND"] += 1
            elif sim[vt] <= best_app:
                fails["APPEARANCE"] += 1
            elif et[vt] <= best_et:
                fails["GEOMETRY"] += 1
            else:
                fails["BLEND"] += 1
        query_rows.append({"u": int(u), "vt": int(vt), "top1": int(ok),
                           "sim_true": float(sim[vt]), "sim_best_imp": float(best_app),
                           "e_true": float(em[vt]), "e_best_imp": float(best_em),
                           "et_true": float(et[vt]), "et_best_imp": float(best_et),
                           "ctx_ok": int(ctx.get(u, -1) == vt),
                           "n_imp": len(imps)})
    return {"tag": tag, "pose": pose, "arm": arm_name, "config": cfg,
            "table": os.path.basename(PM.active_path()),
            "counts": counts, "fails": fails,
            "auc_appearance": _auc(app_pairs), "auc_geo_matcher": _auc(geo_m),
            "auc_geo_truth": _auc(geo_t),
            "queries": query_rows,
            "context_stats": None}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,llr17")
    ap.add_argument("--poses", default="rot15")
    ap.add_argument("--images", default="dev")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--table", default="", choices=("", "m16", "m17"),
                    help="force the LLR table for every arm")
    ap.add_argument("--result", default="")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF

    tags = IMAGE_SETS.get(args.images, tuple(args.images.split(",")))
    poses = [p for p in POSES if p[0] in args.poses.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    jobs = [(t, pose, spec, a, args.table) for a in arms for pose, spec in poses
            for t in tags]
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
            c = r["counts"]
            print("  %-14s %-8s %-28s q %4d top1 %.3f app %.3f geo %.3f/%.3f"
                  % (r["tag"], r["pose"], r["arm"], c["queries"],
                     c["top1"] / max(1, c["queries"]), c["app_ok"] / max(1, c["queries"]),
                     r["auc_geo_matcher"] or 0, r["auc_geo_truth"] or 0), flush=True)
    res: Dict[str, Any] = {"rows": rows, "arms": arms, "images": list(tags),
                           "poses": [p[0] for p in poses],
                           "units": "residuals in tolunits; AUC unitless; top-1 a share"}
    if args.result:
        MF.write_result(os.path.join(OUT, args.result), res)
        print("wrote %s" % args.result)
    # summary
    print("\n== summary (pooled over cells, per arm)")
    for a in arms:
        rr = [r for r in rows if r["arm"] == a]
        q = sum(r["counts"]["queries"] for r in rr)
        if not q:
            continue
        t1 = sum(r["counts"]["top1"] for r in rr) / q
        ap_ = sum(r["counts"]["app_ok"] for r in rr) / q
        print("  %-28s q %5d top1 %.3f app %.3f" % (a, q, t1, ap_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
