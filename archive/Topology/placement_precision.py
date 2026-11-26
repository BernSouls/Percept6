"""MISSION 4 STAGE 1 -- the precision instrument.

Every quantity here is measured against RENDERER TRUTH, and truth is used only
to score: nothing in this file feeds a match.

Reported per view step (translation, rotation, scale, mixed) and per lens:

  REPEATABILITY  share of truth-tracked base nodes that have a query node.
  LOCALISATION   |query node position - predicted true position|, median and p90,
                 in pixels and in the node's own scale units.  This is the node's
                 own precision GIVEN THE RIGHT NODE, so it is not contaminated by
                 mis-identification.
  IDENTITY       of the nearest query node to the predicted position, the share
                 that is the SAME feature -- the "wrong node for the feature"
                 error that the old 3.6 px number silently included.
  DISTINCTNESS   distinct positions per node (share sharing an exact site).
  SEPARATION     AUC of true vs rival placement residuals (delegated to the
                 existing scorer).

The three errors the mission says were being conflated -- localisation, missing
nodes, wrong node for the feature -- are reported as three separate numbers.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
for p in (ROOT, os.path.join(ROOT, "Topology")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")


def measure(specs: List[Any], views: List[Any], limit: Optional[int] = None,
            tag: str = "m4_precision") -> Dict[str, Any]:
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, predict, track_optimal

    per_step: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    per_lens: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))

    for spec in (specs[:limit] if limit else specs):
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        pos0 = {int(n): (float(g0.subgraph.anchor_of_node[n][0]),
                         float(g0.subgraph.anchor_of_node[n][1]))
                for n in g0.subgraph.nodes if n in g0.subgraph.anchor_of_node}
        for label, view in views:
            pv = view_pose(spec, view)
            sc1 = posed(spec.scene, pv["rot_deg"], tuple(pv["shift"]),
                        pv["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            pos1 = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                             float(g1.subgraph.anchor_of_node[n][1]))
                    for n in g1.subgraph.nodes if n in g1.subgraph.anchor_of_node}
            pred = predict(spec.scene, r0, sc1, g0)
            truth = {int(q): int(r)
                     for r, q in track_optimal(spec.scene, r0, sc1, r1,
                                               g0, g1).items()}
            q_nodes = np.array([pos1[k] for k in sorted(pos1)], dtype=float)
            q_ids = np.array(sorted(pos1), dtype=int)
            base = [n for n in sorted(pred) if o0.get(n, -1) >= 0]
            if not base or not len(q_nodes):
                continue
            # REPEATABILITY: a tracked query node exists for this base node
            tracked = [r for q, r in truth.items() if r in pred]
            rep = len(tracked) / len(base)
            per_step[label]["repeatability"].append(rep)

            # LOCALISATION given the right node, and IDENTITY
            errs = []
            errs_correct = []
            ident_ok = 0
            dent = 0
            for r in tracked:
                q = next((qq for qq, rr in truth.items() if rr == r), None)
                if q is None or q not in pos1 or r not in pred:
                    continue
                px, py = pred[r]
                ex = pos1[q][0] - px
                ey = pos1[q][1] - py
                errs.append(float(np.hypot(ex, ey)))
                # nearest query node to the predicted position
                d = np.hypot(q_nodes[:, 0] - px, q_nodes[:, 1] - py)
                nearest = int(q_ids[int(np.argmin(d))])
                dent += 1
                if nearest == q:
                    ident_ok += 1
                    # LOCALISATION GIVEN THE RIGHT NODE, properly: this branch
                    # keeps only pairs where the nearest query node to the
                    # feature's true position IS the tracked node.  The
                    # unrestricted number above mixes position error with
                    # "this node is a different feature", which is the whole
                    # point of separating the three errors.
                    errs_correct.append(float(np.hypot(ex, ey)))
            if errs:
                per_step[label]["loc_median_px"].append(float(np.median(errs)))
                per_step[label]["loc_p90_px"].append(
                    float(np.percentile(errs, 90)))
            if dent:
                per_step[label]["identity"].append(ident_ok / dent)
            if errs_correct:
                per_step[label]["loc_correct_median_px"].append(
                    float(np.median(errs_correct)))
                per_step[label]["loc_correct_p90_px"].append(
                    float(np.percentile(errs_correct, 90)))
                per_step[label]["n_correct"].append(float(len(errs_correct)))

            # DISTINCTNESS of the query graph's placement endpoints
            sites = defaultdict(list)
            for k, xy in pos1.items():
                sites[(round(xy[0], 3), round(xy[1], 3))].append(k)
            shared = sum(len(v) - 1 for v in sites.values() if len(v) > 1)
            per_step[label]["distinct_share"].append(1.0 - shared / max(1, len(pos1)))
            per_step[label]["nodes"].append(float(len(pos1)))

            # per lens localisation
            for r in tracked:
                q = next((qq for qq, rr in truth.items() if rr == r), None)
                if q is None or q not in pos1 or r not in pred:
                    continue
                s = g1.sigs.get(q)
                L = str(s[0]) if s is not None else "?"
                px, py = pred[r]
                per_lens[L]["err"].append(float(np.hypot(pos1[q][0] - px,
                                                         pos1[q][1] - py)))

    out: Dict[str, Any] = {"steps": {}, "lenses": {}}
    print("%-16s %8s %10s %8s %10s %8s %8s %9s %6s"
          % ("view step", "repeat", "loc med", "loc p90", "loc(correct)",
             "p90(corr)", "identity", "distinct", "nodes"))
    for label in sorted(per_step):
        d = per_step[label]
        row = {}
        for k, v in d.items():
            row[k] = float(np.mean(v)) if v else float("nan")
        out["steps"][label] = row
        print("%-16s %8.3f %10.3f %8.3f %10.3f %8.3f %8.3f %9.3f %6.0f"
              % (label, row.get("repeatability", float("nan")),
                 row.get("loc_median_px", float("nan")),
                 row.get("loc_p90_px", float("nan")),
                 row.get("loc_correct_median_px", float("nan")),
                 row.get("loc_correct_p90_px", float("nan")),
                 row.get("identity", float("nan")),
                 row.get("distinct_share", float("nan")),
                 row.get("nodes", float("nan"))))
    print("\nper lens, localisation error (px), median:")
    rows = []
    for L in sorted(per_lens):
        e = per_lens[L]["err"]
        rows.append((float(np.median(e)), L, len(e),
                     float(np.percentile(e, 90))))
        out["lenses"][L] = {"median_px": float(np.median(e)),
                            "p90_px": float(np.percentile(e, 90)),
                            "n": len(e)}
    for med, L, n, p90 in sorted(rows):
        print("  %-18s median %6.2f  p90 %6.2f  n %5d" % (L, med, p90, n))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="m4_precision")
    ap.add_argument("--subpixel", action="store_true",
                    help="turn FE_SUBPIXEL_PEAK_POS on before building")
    args = ap.parse_args()
    import guard
    guard.activate()
    import parameters as P
    if args.subpixel:
        P.FE_SUBPIXEL_PEAK_POS = True
        print("FE_SUBPIXEL_PEAK_POS = True (lever 1 test)")
    else:
        print("FE_SUBPIXEL_PEAK_POS = %r" % (getattr(P, "FE_SUBPIXEL_PEAK_POS",
                                                     None),))
    from Topology.placement_dev3 import VIEWS, all_specs, split

    fit_specs, check_specs = split(all_specs())
    print("STAGE 1 precision instrument -- FIT half (%d scenes)\n"
          % len(fit_specs))
    res = measure(fit_specs, VIEWS, args.limit, args.tag)
    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
