"""MISSION 23 / item 2 -- the geometric vote, settled on TRUE correspondences.

Mission 22 measured the transform vote on the correspondences the SHIPPED MATCHER chose, which are
mostly wrong pairs, and concluded that place voting cannot cluster.  That conclusion is only as
good as the correspondences it was measured on.  Here the truth is exact: the synthetic renderer
scenes carry `track_optimal`, so every query place's partner in the reference is known by geometry,
not guessed by the matcher.

Three questions, one script:

  1. DO TRUE CORRESPONDENCES AGREE?  For every true pair, the transform it proposes.  Under a
     SIMILARITY that is (dx, dy, log2 scale, axial dtheta) -- the shipped vote's own vote.  Under
     an AFFINE, one pair is not enough (an affine has six parameters and a point pair carries two
     constraints), so the affine a pair proposes is estimated from that pair plus its two nearest
     true-partner neighbours.  "Agreement" is the share of pairs whose transform lands in the
     histogram's modal bin, and the null is the same statistic with the pairing shuffled.

  2. HOW TIGHTLY CAN THEY BE FITTED?  One similarity and one affine are least-squares-fitted to
     all the true pairs of a cell and the RMS residual is reported, in query place-spacings,
     beside the same fit on a shuffled pairing.

  3. THE DECISION, WRITTEN BEFORE THE RUN (see DECISION below).

Cells: identity (the control -- if the method cannot find agreement here, nothing it says about
the other cells means anything), in-plane 15 deg, uniform scale 1.15, pitch 20 deg, yaw 20 deg,
on the 20 renderer scenes.  n is reported per cell, as the number of true pairs used.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: THE DECISION RULE, written before the run.  If either transform agrees at ten times its
#: shuffled null on yaw 20, that transform gets built into a door arm and is measured once on FIT
#: (and ships only if it beats the shipped door).  Otherwise the transform vote is closed: the
#: README says geometric voting does not work on out-of-plane views, the transform-histogram door
#: leaves the shipped options, and the counted door is renamed for what it is.
DECISION = {"statistic": "modal-bin share of the pair-proposed transform",
            "threshold_ratio": 10.0, "cell": "yaw 20",
            "if_met": "build that transform as a door arm and measure once on FIT recall@8",
            "if_not_met": "close geometric voting; ship the content-count door under its own name"}

#: the histogram, identical to the shipped door's (`matching/placement_frontdoor.py::BINS`)
BIN_XY, BIN_LS, BIN_TH = 1.0, 0.5, math.pi / 6.0
BIN_AFF = 0.25

VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("id00", {}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
    ("x20", {"rot_x_deg": 20.0}),
    ("y20", {"rot_y_deg": 20.0}),
)


def _recipe() -> None:
    import parameters as P
    for k, v in dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
                     FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
                     PLACEMENT_FRAME_KEYPOINT_SIGMA="largest", PLACEMENT_FRAME_FALLBACK="none",
                     PLACEMENT_M15_PLACES=True, KP_GRADED_SIMILARITY=True).items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    from core import keypoint_classes as KPC
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()


def _modal_share(cells: Sequence[Tuple[int, ...]]) -> float:
    if not cells:
        return float("nan")
    counts: Dict[Tuple[int, ...], int] = {}
    for c in cells:
        counts[c] = counts.get(c, 0) + 1
    return max(counts.values()) / float(len(cells))


def _fit_similarity(P: np.ndarray, Q: np.ndarray) -> float:
    """RMS residual of the best similarity mapping P onto Q (points are rows, 2 columns)."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    A, B = P - mp, Q - mq
    num = float((A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1]).sum())
    den = float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum())
    aa = float((A * A).sum())
    if aa <= 0:
        return float("nan")
    c, s = num / aa, den / aa
    R = np.array([[c, -s], [s, c]])
    R = (P - mp) @ R.T + mq
    return float(np.sqrt(((R - Q) ** 2).sum(axis=1).mean()))


def _fit_affine(P: np.ndarray, Q: np.ndarray) -> float:
    """RMS residual of the best affine mapping P onto Q, with the intercept."""
    A = np.hstack([P, np.ones((P.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, Q, rcond=None)
    R = A @ coef
    return float(np.sqrt(((R - Q) ** 2).sum(axis=1).mean()))


def _affine_of(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    A = np.hstack([p, np.ones((p.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    return coef                                     # (3, 2)


def cell_job(job: Tuple[str, str]) -> Dict[str, Any]:
    """One (scene, view) cell -> the agreement statistics on its true correspondences."""
    scene, label = job
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import track_optimal
    from matching import placement_matcher as PM
    from matching import placement_frontdoor as FD
    from matching import placement_edges as PE
    t0 = time.time()
    _recipe()
    spec = {s.name: s for s in FM.confirmation_specs_oop_m10(20)}[scene]
    view = dict(VIEWS)[label]
    r0 = spec.scene.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|m23geo_ref")
    sc1, _pv = FM.pose_fn(spec, view)
    r1 = sc1.render()
    g1 = get_or_build(r1.image_u8, spec.name + "|m23geo_" + label)
    truth = {int(q): int(r) for r, q in
             track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
    fr_q, _a = PE.node_frames_keypoint(g1.subgraph, r1.image_u8)
    fr_r, _b = PE.node_frames_keypoint(g0.subgraph, r0.image_u8)
    anc_q = g1.subgraph.anchor_of_node
    anc_r = g0.subgraph.anchor_of_node
    side_q = PM.build_side_cached(g1, r1.image_u8)
    side_r = PM.build_side_cached(g0, r0.image_u8)
    # the length unit: the door's own median nearest-distinct-neighbour distance, in the query
    unit, _ = _unit_from_anchors([anc_q[n] for n in sorted(anc_q)])
    rows: List[Dict[str, Any]] = []
    for u, v in sorted(truth.items()):
        if u not in anc_q or v not in anc_r:
            continue
        sq = side_q.spacing_by_node.get(u)
        sr = side_r.spacing_by_node.get(v)
        if not sq or not sr or sq <= 0 or sr <= 0:
            continue
        rows.append({"u": int(u), "v": int(v),
                     "p": np.asarray(anc_q[u], dtype=np.float64),
                     "q": np.asarray(anc_r[v], dtype=np.float64),
                     "ls": math.log2(float(sr) / float(sq)),
                     "th": (None if (u not in fr_q or v not in fr_r)
                            else float((fr_r[v] - fr_q[u]) % math.pi))})
    n = len(rows)
    if n < 8:
        return {"scene": scene, "view": label, "n": n, "skipped": "too few true pairs",
                "sec": time.time() - t0}
    P = np.asarray([r["p"] for r in rows])
    Q = np.asarray([r["q"] for r in rows])
    sim_cells: List[Tuple[int, ...]] = []
    for r in rows:
        dx = (r["q"][0] - r["p"][0]) / max(1e-9, unit)
        dy = (r["q"][1] - r["p"][1]) / max(1e-9, unit)
        ls = r["ls"] / BIN_LS
        th = 0.0 if r["th"] is None else r["th"] / BIN_TH
        sim_cells.append((int(round(dx / BIN_XY)), int(round(dy / BIN_XY)),
                          int(round(ls)), int(round(th))))
    # affine per pair: that pair plus its two nearest true-partner neighbours in the query
    aff_cells: List[Tuple[int, ...]] = []
    for i, r in enumerate(rows):
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        d[i] = np.inf
        near = [j for j in np.argsort(d)[:2] if np.isfinite(d[j])]
        if len(near) < 2:
            continue
        idx = [i] + near
        coef = _affine_of(P[idx], Q[idx])
        a = coef[:2, :].T                                   # 2x2
        t = coef[2, :] / max(1e-9, unit)
        if not np.all(np.isfinite(a)):
            continue
        aff_cells.append(tuple(int(round(x / BIN_AFF))
                               for x in (a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
                         + (int(round(t[0] / BIN_XY)), int(round(t[1] / BIN_XY))))
    # the shuffled null: the same query pairs against a random permutation of the partners
    rng = np.random.default_rng(20261118 + sum(ord(c) for c in scene + label))
    perm = rng.permutation(n)
    sim_null: List[Tuple[int, ...]] = []
    aff_null: List[Tuple[int, ...]] = []
    Qs = Q[perm]
    for i, r in enumerate(rows):
        dx = (Qs[i][0] - r["p"][0]) / max(1e-9, unit)
        dy = (Qs[i][1] - r["p"][1]) / max(1e-9, unit)
        sim_null.append((int(round(dx / BIN_XY)), int(round(dy / BIN_XY)),
                         int(round(r["ls"] / BIN_LS)),
                         int(round((0.0 if r["th"] is None else r["th"]) / BIN_TH))))
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        d[i] = np.inf
        near = [j for j in np.argsort(d)[:2] if np.isfinite(d[j])]
        if len(near) < 2:
            continue
        idx = [i] + near
        coef = _affine_of(P[idx], Qs[idx])
        a = coef[:2, :].T
        t = coef[2, :] / max(1e-9, unit)
        if not np.all(np.isfinite(a)):
            continue
        aff_null.append(tuple(int(round(x / BIN_AFF))
                              for x in (a[0, 0], a[0, 1], a[1, 0], a[1, 1]))
                        + (int(round(t[0] / BIN_XY)), int(round(t[1] / BIN_XY))))
    out = {"scene": scene, "view": label, "n": n, "unit_px": float(unit),
           "n_affine": len(aff_cells),
           "agree_sim": _modal_share(sim_cells), "agree_sim_null": _modal_share(sim_null),
           "agree_aff": _modal_share(aff_cells), "agree_aff_null": _modal_share(aff_null),
           "rms_sim": _fit_similarity(P, Q), "rms_sim_null": _fit_similarity(P, Qs),
           "rms_aff": _fit_affine(P, Q), "rms_aff_null": _fit_affine(P, Qs),
           "sec": time.time() - t0}
    for k in ("sim", "aff"):
        a, b = out["agree_%s" % k], out["agree_%s_null" % k]
        out["ratio_%s" % k] = (a / b) if (b and b == b and b > 0) else float("nan")
    return out


def _unit_from_anchors(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0:
        return 1.0, 0.0
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    D[D <= 1e-6] = np.inf
    nn = np.min(D, axis=1)
    good = nn[np.isfinite(nn) & (nn > 0)]
    if good.size:
        return float(np.median(good)), float(np.median(good))
    span = float(np.hypot(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()))
    u = span / max(1.0, math.sqrt(len(P)))
    return (u if u > 0 else 1.0), u


def main(argv=None) -> int:
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--views", default=",".join(v for v, _ in VIEWS))
    ap.add_argument("--tag", default="m23_geovote")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    if not getattr(P, "PLACEMENT_M19", False):
        raise SystemExit("set PLACEMENT_M19=1")
    from Topology import placement_families_m9 as FM
    scenes = [s.name for s in FM.confirmation_specs_oop_m10(a.scenes)]
    views = [v for v in a.views.split(",") if v]
    jobs = [(s, v) for s in scenes for v in views]
    out_path = os.path.join(OUT, a.tag + ".jsonl")
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done[(r["scene"], r["view"])] = r
    need = [j for j in jobs if j not in done]
    print("geovote: %d cells (%d cached)" % (len(need), len(done)), flush=True)
    t0 = time.time()
    if need:
        pool = mp.Pool(processes=a.workers, maxtasksperchild=8)
        with open(out_path, "a") as fh:
            for k, r in enumerate(pool.imap_unordered(_cell_star, need)):
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                fh.flush()
                done[(r["scene"], r["view"])] = r
                if (k + 1) % 10 == 0:
                    print("  %d/%d %.0fs" % (k + 1, len(need), time.time() - t0), flush=True)
        pool.close()
        pool.join()
    rows = [done[j] for j in jobs if j in done]
    summary: Dict[str, Any] = {"decision_rule": DECISION, "scenes": len(scenes),
                               "cells": {}, "sec": time.time() - t0,
                               "n_queries": len(jobs)}
    print("\n%-10s %-5s %-9s %-9s %-8s %-9s %-9s %-8s %-8s %-8s"
          % ("view", "n", "agree_sim", "null_sim", "xnull", "agree_aff", "null_aff", "xnull",
             "rms_sim", "rms_aff"))
    for v in views:
        here = [r for r in rows if r["view"] == v and r.get("n", 0) >= 8]
        if not here:
            continue
        f = lambda key: float(np.mean([r[key] for r in here if r.get(key) == r.get(key)]))
        cell = {"cells": len(here), "n_pairs": int(sum(r["n"] for r in here)),
                "agree_sim": f("agree_sim"), "agree_sim_null": f("agree_sim_null"),
                "agree_aff": f("agree_aff"), "agree_aff_null": f("agree_aff_null"),
                "rms_sim": f("rms_sim"), "rms_sim_null": f("rms_sim_null"),
                "rms_aff": f("rms_aff"), "rms_aff_null": f("rms_aff_null")}
        for k in ("sim", "aff"):
            b = cell["agree_%s_null" % k]
            cell["ratio_%s" % k] = (cell["agree_%s" % k] / b) if b > 0 else float("nan")
        summary["cells"][v] = cell
        print("%-10s %-5d %-9.3f %-9.3f %-8.2f %-9.3f %-9.3f %-8.2f %-8.2f %-8.2f"
              % (v, cell["n_pairs"], cell["agree_sim"], cell["agree_sim_null"],
                 cell["ratio_sim"], cell["agree_aff"], cell["agree_aff_null"],
                 cell["ratio_aff"], cell["rms_sim"], cell["rms_aff"]))
    yaw = summary["cells"].get("y20", {})
    met = {k: bool(yaw.get("ratio_%s" % k, 0) >= DECISION["threshold_ratio"]) for k in ("sim", "aff")}
    summary["decision"] = {"yaw_ratios": {k: yaw.get("ratio_%s" % k) for k in ("sim", "aff")},
                           "met": met, "outcome": ("build " + ",".join(
                               k for k, v in met.items() if v)) if any(met.values())
                           else "close geometric voting"}
    print("\nyaw 20 ratios: sim %.2f, aff %.2f -> %s"
          % (yaw.get("ratio_sim", float("nan")), yaw.get("ratio_aff", float("nan")),
             summary["decision"]["outcome"]))
    from matching import m15_manifest as MF
    MF.write_result(os.path.join(OUT, a.tag + ".json"), summary)
    print("wrote", os.path.join(OUT, a.tag + ".json"))
    return 0


def _cell_star(job: Tuple[str, str]) -> Dict[str, Any]:
    try:
        return cell_job(job)
    except Exception as exc:                                   # noqa: BLE001
        return {"scene": job[0], "view": job[1], "n": 0, "error": repr(exc)[:300]}


if __name__ == "__main__":
    raise SystemExit(main())
