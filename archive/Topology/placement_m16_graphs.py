"""MISSION 16 -- acceptance B, C and D on the SAME cells Mission 15 used.

B  THE CANDIDATE ROW STOPS EMPTYING: of the truth pairs (place lenses, same lens,
   transformed anchor within TRUTH_TOL), how many have their true partner as an
   ADMISSIBLE candidate (`similarity_rows` returns it with sim > 0)?  Mission 15's
   probe had 55 such queries at 15 deg out of 1,198 at 0 deg.

C  GEOMETRY CLOSES ITS GAP: true-vs-impostor AUC of the geometry term alone, under
   the MATCHER's context (a) and the exact-transform context (b), at 0 / 5 / 15 deg
   and 1.15x.  The a-b gap is reported before and after.

D  APPEARANCE RANKS: AUC of the appearance similarity alone on the same candidates.
   Mission 15 measured exactly 0.500 there because every same-bin candidate tied.

ARMS.  One graph is built per cell under the shipped M16 configuration (population
attached).  The arms are then read off the SAME graphs by changing the registries,
which only the similarity function reads:

    before   registries untouched  -> the shipped 0/1 reading
    graded   constitution, wide     -> the graded per-bin kernel
    after    constitution+population-> the shipped M16 reading

The graph content does not depend on any of them (the population is excluded from
the signature's bytes and hash, and the ranges are not part of the signature); the
harness PROVES that rather than assuming it (`graph_equivalence`: node-key sets and
the before-arm AUC must be identical when the graphs are built with the flags off).

    python Topology/placement_m16_graphs.py --workers 2
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
RESULT = os.path.join(OUT, "m16_graphs.json")

POSES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot00", {"shift": (8.0, 0.0)}),
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
)
IMAGES: Tuple[str, ...] = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")
M16 = dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
           FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
           PLACEMENT_FRAME_KEYPOINT_SIGMA="largest", PLACEMENT_FRAME_FALLBACK="none",
           PLACEMENT_M15_PLACES=True, KP_GRADED_SIMILARITY=True,
           KP_GRADED_POPULATION=True)
ARMS: Tuple[str, ...] = ("before", "graded", "after")


def _image(tag: str) -> np.ndarray:
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        return [s for s in F.gallery() if s.name == name][0].scene.render().image_u8
    return RP.images("validation")[tag.split("|", 1)[1]]


def _set_arm(arm: str) -> None:
    """Point the registries at one arm.  This is the ONLY thing an arm changes."""
    import parameters as P
    from core import keypoint_classes as KPC
    KPC.unregister()
    if arm == "before":
        return
    if arm == "graded":
        KPC.register("wide")
        return
    KPC.register(str(getattr(P, "KP_GRADED_VARIANT", "wide")))
    KPC.register_population()


def _auc(pairs: List[Tuple[float, List[float]]]) -> Optional[float]:
    num = den = 0.0
    for t, imp in pairs:
        for s in imp:
            den += 1.0
            num += 1.0 if t > s else 0.5 if t == s else 0.0
    return (num / den) if den else None


def _job(args: Tuple[str, str, Dict[str, Any], bool]) -> Dict[str, Any]:
    tag, pose, spec, flags_off = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching import m15_places as PL
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_cache import get_or_build

    for k, v in M16.items():
        setattr(P, k, v)
    if flags_off:
        P.KP_GRADED_SIMILARITY = False
        P.KP_GRADED_POPULATION = False
    img = _image(tag)
    img1 = R._transform(img, spec)
    key = "m16|%s|%s|%s" % (tag, pose, "off" if flags_off else "on")
    g0 = get_or_build(img, "m16|%s|orig|%s" % (tag, "off" if flags_off else "on"))
    g1 = get_or_build(img1, key)
    nodes_a = sorted(int(n) for n in g1.subgraph.nodes)
    nodes_b = sorted(int(n) for n in g0.subgraph.nodes)
    keys_a = {g1.sigs[n] for n in nodes_a if n in g1.sigs}
    keys_b = {g0.sigs[n] for n in nodes_b if n in g0.sigs}
    side_q = PM.build_side_cached(g1, img1)
    side_r = PM.build_side_cached(g0, img)
    shape = img.shape[:2]
    truth = PL.lens_truth(g1, g0, spec, shape)
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    tol_of = PM._spacing_tol_fn(side_q, side_r)

    out: Dict[str, Any] = {"tag": tag, "pose": pose, "truth_pairs": len(truth),
                           "nodes_a": len(nodes_a), "nodes_b": len(nodes_b),
                           "graph_key_digest": hash(tuple(sorted(
                               (str(s[0]), repr(getattr(s[1], "quantized", None)))
                               for s in list(keys_a)[:200])))}
    for arm in ARMS:
        _set_arm(arm)
        rows_sim = CEN.similarity_rows(g1, g0, nodes_a, nodes_b)
        idx_b = {v: j for j, v in enumerate(nodes_b)}
        matrix = [[0.0] * len(nodes_b) for _ in nodes_a]
        for i, u in enumerate(nodes_a):
            for (v, s) in rows_sim.get(u, ()):
                if s > 0.0:
                    matrix[i][idx_b[v]] = float(s)
        triples = PM.top_candidates(matrix, nodes_b, int(b["cand_cap"]))
        ctx = PM.context_mapping(triples, nodes_a, side_q, side_r)
        # ---- B: is the true partner an admissible candidate?
        admissible = 0
        for u, v in truth.items():
            if any(vv == v and s > 0.0 for (vv, s) in rows_sim.get(u, ())):
                admissible += 1
        geo = {"ctx_matcher": [], "ctx_truth": []}
        app = []
        top1_app = 0
        q_app = 0
        bank: List[Dict[str, Any]] = []
        for i, u in enumerate(nodes_a):
            vt = truth.get(u)
            if vt is None:
                continue
            row = [(int(nodes_b[j]), float(matrix[i][j]))
                   for j in range(len(nodes_b)) if matrix[i][j] > 0.0]
            if len(row) < 2:
                continue
            sims = [s for (_v, s) in row]
            trues_s = [s for (v, s) in row if v == vt]
            imps_s = [s for (v, s) in row if v != vt]
            if trues_s and imps_s:
                app.append((trues_s[0], imps_s))
                q_app += 1
                top1_app += 1 if max(trues_s) >= max(imps_s) else 0
            if arm == "after":
                # ITEM D's second half: the WEIGHT needs appearance AND evidence on
                # the same candidate rows, so the bank is recorded raw.
                bank.append({"u": int(u), "v": [int(v) for (v, _s) in row],
                             "sim": [float(sv) for (_v, sv) in row],
                             "E": [float(L.plus_score(
                                 u, v, ctx, side_q.incident, side_q.index,
                                 side_r.index, fit, pfit, a_e, a_g,
                                 slot_cap=int(L.SLOT_CAP),
                                 tol=(None if tol_of is None else tol_of(u, v))) - 0.5)
                                 for (v, _s) in row],
                             "true": [1 if v == vt else 0 for (v, _s) in row]})
            for name, mapping in (("ctx_matcher", ctx), ("ctx_truth", truth)):
                sc = []
                for (v, _s) in row:
                    _e, g = L.plus_terms(u, v, mapping, side_q.incident,
                                         side_q.index, side_r.index, fit, pfit,
                                         slot_cap=int(L.SLOT_CAP),
                                         tol=(None if tol_of is None else tol_of(u, v)))
                    sc.append((v, float(g)))
                d = dict(sc)
                if vt in d:
                    imps = [s for (v, s) in sc if v != vt]
                    if imps:
                        geo[name].append((d[vt], imps))
        out[arm] = {"admissible": admissible, "truth": len(truth),
                    "admissible_share": (admissible / len(truth)) if truth else None,
                    "geometry_auc": {k: _auc(v) for k, v in geo.items()},
                    "appearance_auc": _auc(app),
                    "appearance_top1": (top1_app / q_app) if q_app else None,
                    "queries": q_app,
                    "bank": bank if arm == "after" else None}
    _set_arm("after")
    return out


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--images", default=",".join(IMAGES))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    if args.report:
        d = json.load(open(RESULT))
    else:
        tags = args.images.split(",")
        jobs = [(t, pose, spec, False) for t in tags for pose, spec in POSES]
        # THE EQUIVALENCE PROOF, on the first cell only: the graphs do not depend
        # on the M16 flags, so an arm may be read off one build.
        jobs.append((tags[0], POSES[1][0], POSES[1][1], True))
        rows: List[Dict[str, Any]] = []
        print("%d cells (%d images x %d poses) + 1 equivalence build"
              % (len(jobs) - 1, len(tags), len(POSES)), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_job, jobs):
                rows.append(r)
                if r["tag"].startswith("__"):
                    continue
                print("  %-14s %-8s truth %4d adm %s  auc geo %s app %s"
                      % (r["tag"], r["pose"], r["truth_pairs"],
                         {a: r[a]["admissible"] for a in ARMS},
                         {a: (round(r[a]["geometry_auc"]["ctx_matcher"], 3)
                              if r[a]["geometry_auc"]["ctx_matcher"] else None)
                          for a in ARMS},
                         {a: (round(r[a]["appearance_auc"], 3)
                              if r[a]["appearance_auc"] else None) for a in ARMS}),
                      flush=True)
        d = {"cells": rows, "arms": list(ARMS), "poses": [p for p, _ in POSES],
             "images": tags, "flags": dict(M16)}
        MF.write_result(RESULT, d)
    cells = d["cells"]
    print("\n== acceptance B/C/D, mean over %d images" % len(d["images"]))
    print("%-8s %-7s %9s %7s %7s %7s %7s %7s %7s"
          % ("pose", "arm", "admiss%", "geo a", "geo b", "app", "top1", "queries", "truth"))
    for pose, _spec in POSES:
        for arm in ARMS:
            cs = [c for c in cells if c["pose"] == pose and arm in c]
            if not cs:
                continue
            def m(fn):
                xs = [fn(c) for c in cs if fn(c) is not None]
                return float(np.mean(xs)) if xs else float("nan")
            print("%-8s %-7s %9.1f %7.3f %7.3f %7.3f %7.3f %7.0f %7.0f"
                  % (pose, arm,
                     100.0 * m(lambda c: c[arm]["admissible_share"]),
                     m(lambda c: c[arm]["geometry_auc"]["ctx_matcher"]),
                     m(lambda c: c[arm]["geometry_auc"]["ctx_truth"]),
                     m(lambda c: c[arm]["appearance_auc"]),
                     m(lambda c: c[arm]["appearance_top1"]),
                     m(lambda c: c[arm]["queries"]),
                     m(lambda c: c[arm]["truth"])))
    eq = [c for c in cells if c.get("_equivalence")]
    print("\nwrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
