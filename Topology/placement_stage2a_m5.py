"""MISSION 5 STAGE 2a -- the population lever, measured on FIT.

Stage 1 showed the placement population is dominated by two ridge/boundary lens
types whose along-structure position is undefined:

    ContourFragment  identity 0.213
    TextureBoundary  identity 0.212
    every other lens 0.71 - 0.95

Between them they are 86 of 179 placement nodes.  This script applies both
repairs to `matching.placement_edges.build_edges` (both dark by default) and
measures the same graph under three populations:

    A  baseline        the shipped population
    B  exclude_ridge   ridge lenses dropped as placement endpoints (both sides)
    C  across_only     ridge lenses kept, but only edges ACROSS their ridge
                       (|sin direction| >= 0.5, i.e. >= 30 deg off the axis)

`build_edges` runs per call, so nothing is rebuilt at the front end: the same
cached graphs serve all three variants.

Reported per variant: identity, repeatability, localisation, distinctness,
checkable share (coverage of the population by a checkable edge), pair survival,
and an UNFITTED geometric separation AUC (true pair vs best rival pair).

Truth is used only to score.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
for _p in (ROOT, os.path.join(ROOT, "Topology")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")
REPEAT_RADIUS = 3.0

RIDGE = ("ContourFragment", "TextureBoundary")

VARIANTS = {
    "A_baseline": dict(),
    "B_exclude_ridge": dict(exclude_lenses=RIDGE),
    "C_across_only": dict(ridge_across_only=True),
    # [STAGE 2] the same three populations with co-sited nodes collapsed to one
    # PLACE per site, so an edge connects two places and a slot's rivals are
    # distinct places rather than near-duplicates of the true partner.
    "D_collapse": dict(collapse_places=True),
    "E_excl_collapse": dict(exclude_lenses=RIDGE, collapse_places=True),
}


def _vec(pv: Any) -> Tuple[float, float]:
    return (pv.distance * math.cos(pv.direction),
            pv.distance * math.sin(pv.direction))


def _auc(pos: List[float], neg: List[float]) -> float:
    """P(a true residual is smaller than a wrong residual), ties = 1/2."""
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos, dtype=float)
    b = np.asarray(neg, dtype=float)
    gt = (b[None, :] < a[:, None]).mean()
    eq = (b[None, :] == a[:, None]).mean()
    return float(gt + 0.5 * eq)


def run(specs: List[Any], views: List[Any], tag: str) -> Dict[str, Any]:
    import guard
    guard.activate()
    from matching import placement_edges as PE
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import (checkable_edges, index_payloads,
                                           lens_of, object_mask,
                                           similarity_rows)
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, predict, track_optimal

    acc: Dict[str, Dict[str, List[float]]] = {
        v: defaultdict(list) for v in VARIANTS}

    for spec in specs:
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        for label, view in views:
            pv_ = view_pose(spec, view)
            sc1 = posed(spec.scene, pv_["rot_deg"], tuple(pv_["shift"]),
                        pv_["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            pred = predict(spec.scene, r0, sc1, g0)
            fwd = {int(r): int(q) for r, q
                   in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            qpos = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                             float(g1.subgraph.anchor_of_node[n][1]))
                    for n in g1.subgraph.nodes
                    if n in g1.subgraph.anchor_of_node}

            by_lens: Dict[str, List[int]] = defaultdict(list)
            for q in sorted(qpos):
                by_lens[lens_of(g1, q)].append(q)

            # per-site distinctness of the population
            sites = defaultdict(int)
            for q in sorted(qpos):
                sites[(round(qpos[q][0] * 2), round(qpos[q][1] * 2))] += 1

            for name, kw in VARIANTS.items():
                excl = tuple(kw.get("exclude_lenses", ()))
                coll = bool(kw.get("collapse_places", False))
                pop = [int(n) for n in sorted(pred)
                       if o0.get(int(n), -1) >= 0
                       and lens_of(g0, int(n)) not in excl]
                if not pop:
                    continue
                # --- repeatability / location / identity (Stage-1 method)
                for r in pop:
                    L = lens_of(g0, r)
                    cands = [q for q in by_lens.get(L, [])
                             if lens_of(g1, q) not in excl]
                    acc[name]["n_nodes"].append(1.0)
                    if not cands:
                        acc[name]["repeat"].append(0.0)
                        continue
                    px, py = pred[r]
                    arr = np.array([qpos[q] for q in cands], dtype=float)
                    d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                    k = int(np.argmin(d))
                    hit = d[k] <= REPEAT_RADIUS
                    acc[name]["repeat"].append(1.0 if hit else 0.0)
                    if hit:
                        q = cands[k]
                        acc[name]["loc"].append(float(d[k]))
                        acc[name]["ident"].append(
                            1.0 if fwd.get(r) == q else 0.0)
                # --- distinctness: population nodes that are alone at a site
                acc[name]["distinct"].append(
                    float(np.mean([1.0 if sites[(round(qpos[q][0] * 2),
                                                 round(qpos[q][1] * 2))] == 1
                                   else 0.0
                                   for q in qpos
                                   if o1.get(q, -1) >= 0
                                   and lens_of(g1, q) not in excl] or [0.0])))

                # --- payload-level metrics
                r_pay = PE.build_edges(g0.subgraph, r0.image_u8,
                                       lens_map=PE.lens_map_of(g0), **kw)
                q_pay = PE.build_edges(g1.subgraph, r1.image_u8,
                                       lens_map=PE.lens_map_of(g1), **kw)
                q_by_src, q_by_dst = index_payloads(q_pay)
                rows = similarity_rows(g1, g0, sorted(qpos), sorted(pred))
                nmap = {u: rows[u][0][0] for u in rows if rows[u]}

                q_pop = [q for q in sorted(qpos) if o1.get(q, -1) >= 0
                         and lens_of(g1, q) not in excl]
                chk = sum(1 for u in q_pop
                          if checkable_edges(u, nmap, q_pay, r_pay,
                                             q_by_src, q_by_dst))
                acc[name]["checkable"].append(chk / max(1, len(q_pop)))

                num = den = 0
                for (sv, dv, et) in sorted(r_pay, key=lambda k: (k[0], k[1])):
                    if int(sv) not in fwd or int(dv) not in fwd:
                        continue
                    den += 1
                    if (fwd[int(sv)], fwd[int(dv)], et) in q_pay:
                        num += 1
                if den:
                    acc[name]["survival"].append(num / den)
                    acc[name]["survival_den"].append(float(den))

                # --- unfitted separation: true pair vs best same-lens rival
                pos_err: List[float] = []
                neg_err: List[float] = []
                rng = np.random.default_rng(20261118)
                pos_r: List[float] = []
                neg_r: List[float] = []
                for (sv, dv, et) in sorted(r_pay, key=lambda k: (k[0], k[1])):
                    u, v = int(sv), int(dv)
                    if u not in fwd or v not in fwd:
                        continue
                    a = q_pay.get((fwd[u], fwd[v], et))
                    if a is None:
                        continue
                    ref, obs = _vec(r_pay[(u, v, et)]), _vec(a)
                    pos_err.append(math.hypot(ref[0] - obs[0],
                                              ref[1] - obs[1]))
                    # rival: the nearest same-lens query node that is NOT the
                    # true partner, standing in v's place
                    L = lens_of(g1, fwd[v])
                    best = None
                    for w in by_lens.get(L, ()):
                        if w == fwd[v] or lens_of(g1, w) in excl:
                            continue
                        b = q_pay.get((fwd[u], w, et))
                        if b is None:
                            continue
                        o2 = _vec(b)
                        e = math.hypot(ref[0] - o2[0], ref[1] - o2[1])
                        if best is None or e < best:
                            best = e
                    if best is not None:
                        neg_err.append(best)
                    # [AUC FIX] The nearest same-lens node is often a
                    # CO-SITED DUPLICATE, i.e. the same physical place, which
                    # legitimately has the same displacement.  Using it as the
                    # negative makes the metric measure duplication, not
                    # separation -- measured: AUC 0.33-0.44 in EVERY variant,
                    # below the 0.5 a coin gives.  The standard definition is a
                    # WRONG PAIR (a permuted partner), so this draws a seeded
                    # random same-lens node that is not the true partner and is
                    # not at the true partner's site.
                    pool = [w for w in by_lens.get(L, ())
                            if w != fwd[v] and lens_of(g1, w) not in excl]
                    if pool:
                        w = pool[int(rng.integers(0, len(pool)))]
                        b = q_pay.get((fwd[u], w, et))
                        if b is not None:
                            o2 = _vec(b)
                            pos_r.append(math.hypot(ref[0] - obs[0],
                                                    ref[1] - obs[1]))
                            neg_r.append(math.hypot(ref[0] - o2[0],
                                                    ref[1] - o2[1]))
                if pos_err and neg_err:
                    acc[name]["auc_true"].extend(pos_err)
                    acc[name]["auc_wrong"].extend(neg_err)
                    acc[name]["auc"].append(_auc(pos_err, neg_err))
                    if pos_r and neg_r:
                        acc[name]["auc_rand"].append(_auc(pos_r, neg_r))
                        acc[name]["auc_rand_pool"].append(float(len(pos_r)))
                    acc[name]["n_payload"].append(float(len(r_pay)))

    res: Dict[str, Any] = {"radius_px": REPEAT_RADIUS,
                           "scenes": len(specs), "variants": {}}
    hdr = ("%-16s %6s %8s %8s %9s %8s %9s %7s %8s %9s"
           % ("variant", "nodes", "repeat", "ident", "loc med", "distinct",
              "checkable", "surviv", "AUCmin", "AUCrand"))
    print(hdr)
    print("-" * len(hdr))
    for name in VARIANTS:
        d = acc[name]
        if not d["n_nodes"]:
            continue
        row = {
            "n_nodes": int(len(d["n_nodes"])),
            "repeat": float(np.mean(d["repeat"])) if d["repeat"] else float("nan"),
            "identity": float(np.mean(d["ident"])) if d["ident"] else float("nan"),
            "loc_median": float(np.median(d["loc"])) if d["loc"] else float("nan"),
            "distinct": float(np.mean(d["distinct"])) if d["distinct"] else float("nan"),
            "checkable": float(np.mean(d["checkable"])) if d["checkable"] else float("nan"),
            "survival": float(np.mean(d["survival"])) if d["survival"] else float("nan"),
            "auc": float(np.mean(d["auc"])) if d["auc"] else float("nan"),
            "auc_rand": (float(np.mean(d["auc_rand"]))
                         if d["auc_rand"] else float("nan")),
            "n_payload": int(np.sum(d["n_payload"])) if d["n_payload"] else 0,
        }
        res["variants"][name] = row
        print("%-16s %6d %8.3f %8.3f %9.3f %8.3f %9.3f %7.3f %8.3f %9.3f"
              % (name, row["n_nodes"], row["repeat"], row["identity"],
                 row["loc_median"], row["distinct"], row["checkable"],
                 row["survival"], row["auc"], row["auc_rand"]))

    path = os.path.join(OUT, "%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="m5_stage2a_population")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    from Topology.placement_dev3 import VIEWS, all_specs, split
    fit, check = split(all_specs())
    specs = (check if args.check else fit)[:args.limit]
    print("STAGE 2a population lever -- %s half, %d scenes\n"
          % ("CHECK" if args.check else "FIT", len(specs)))
    run(specs, VIEWS, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
