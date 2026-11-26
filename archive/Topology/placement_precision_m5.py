"""MISSION 5 STAGE 1 -- NODE-TO-NODE truth, per lens and per view step.

Mission 4 measured localisation against arbitrary Fibonacci surface points --
points that no detector was ever asked to find.  A node cannot be blamed for
missing a feature that does not exist.  This instrument instead asks a question
a detector CAN fail:

    take a node in the base view, back-project its anchor through the depth
    buffer to the surface, carry it through its object's known pose and
    re-project it into the query view.  That is where THIS detection should
    land.  Then look for the nearest node OF THE SAME LENS.

Reported per lens and per view step:

    REPEATABILITY  share of base nodes with a same-lens query node within
                   REPEAT_RADIUS px of the predicted position.
    LOCALISATION   distance from the predicted position to that node, given the
                   node was found (median, p90).
    IDENTITY       of those, the share where the matched node is the one the
                   optimal truth assignment chose (i.e. not merely nearby).

Truth is used only to score.  Nothing here feeds a matcher.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = "/home/user/Percept6"
for p in (ROOT, os.path.join(ROOT, "Topology")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")
REPEAT_RADIUS = 3.0


def run(specs: List[Any], views: List[Any], limit: Optional[int],
        tag: str) -> Dict[str, Any]:
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, predict, track_optimal

    lens_steps: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    steps: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))

    for spec in (specs[:limit] if limit else specs):
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        for label, view in views:
            pv = view_pose(spec, view)
            sc1 = posed(spec.scene, pv["rot_deg"], tuple(pv["shift"]),
                        pv["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            pred = predict(spec.scene, r0, sc1, g0)
            truth = {int(q): int(r) for r, q
                     in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            qpos = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                             float(g1.subgraph.anchor_of_node[n][1]))
                    for n in g1.subgraph.nodes
                    if n in g1.subgraph.anchor_of_node}
            # same-lens index of the query graph
            by_lens: Dict[str, List[int]] = defaultdict(list)
            for q in sorted(qpos):
                s = g1.sigs.get(q)
                if s is not None:
                    by_lens[str(s[0])].append(q)

            base = [int(n) for n in sorted(pred) if o0.get(int(n), -1) >= 0]
            for r in base:
                s = g0.sigs.get(r)
                if s is None:
                    continue
                L = str(s[0])
                cands = by_lens.get(L, [])
                if not cands:
                    lens_steps[L]["n"].append(1.0)
                    lens_steps[L]["repeat"].append(0.0)
                    continue
                px, py = pred[r]
                arr = np.array([qpos[q] for q in cands], dtype=float)
                d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                k = int(np.argmin(d))
                lens_steps[L]["n"].append(1.0)
                hit = d[k] <= REPEAT_RADIUS
                lens_steps[L]["repeat"].append(1.0 if hit else 0.0)
                steps[label]["repeat"].append(1.0 if hit else 0.0)
                if hit:
                    lens_steps[L]["loc"].append(float(d[k]))
                    steps[label]["loc"].append(float(d[k]))
                    q = cands[k]
                    lens_steps[L]["ident"].append(
                        1.0 if truth.get(q) == r else 0.0)
                    steps[label]["ident"].append(
                        1.0 if truth.get(q) == r else 0.0)

    res: Dict[str, Any] = {"radius_px": REPEAT_RADIUS, "steps": {}, "lenses": {}}
    print("%-16s %10s %12s %12s %10s"
          % ("view step", "repeat", "loc median", "loc p90", "identity"))
    for label in sorted(steps):
        d = steps[label]
        row = {"repeat": float(np.mean(d["repeat"])),
               "loc_median": float(np.median(d["loc"])) if d["loc"] else float("nan"),
               "loc_p90": float(np.percentile(d["loc"], 90)) if d["loc"] else float("nan"),
               "identity": float(np.mean(d["ident"])) if d["ident"] else float("nan"),
               "n": len(d["repeat"])}
        res["steps"][label] = row
        print("%-16s %10.3f %12.3f %12.3f %10.3f"
              % (label, row["repeat"], row["loc_median"], row["loc_p90"],
                 row["identity"]))

    print("\nper lens:")
    print("%-18s %6s %8s %10s %8s %8s"
          % ("lens", "n", "repeat", "loc median", "loc p90", "identity"))
    rows = []
    for L, d in lens_steps.items():
        n = len(d["repeat"])
        rep = float(np.mean(d["repeat"])) if n else float("nan")
        med = float(np.median(d["loc"])) if d["loc"] else float("nan")
        p90 = float(np.percentile(d["loc"], 90)) if d["loc"] else float("nan")
        idn = float(np.mean(d["ident"])) if d["ident"] else float("nan")
        rows.append((med if med == med else 1e9, L, n, rep, med, p90, idn))
        res["lenses"][L] = {"n": n, "repeat": rep, "loc_median": med,
                            "loc_p90": p90, "identity": idn}
    for _m, L, n, rep, med, p90, idn in sorted(rows):
        print("%-18s %6d %8.3f %10.2f %8.2f %8.3f" % (L, n, rep, med, p90, idn))

    path = os.path.join(OUT, "%s.json" % tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return res


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="m5_precision_nodetonode")
    ap.add_argument("--check", action="store_true",
                    help="use the CHECK half instead of FIT")
    args = ap.parse_args()
    from Topology.placement_dev3 import VIEWS, all_specs, split

    fit_specs, check_specs = split(all_specs())
    specs = check_specs if args.check else fit_specs
    print("node-to-node precision -- %s half (%d scenes)\n"
          % ("CHECK" if args.check else "FIT", len(specs)))
    run(specs, VIEWS, args.limit, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
