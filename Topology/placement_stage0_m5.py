"""MISSION 5 STAGE 0 -- position-writer map, flags, and nodes-per-place census.

Everything here is read off a REAL graph and its own provenance; nothing is
assumed from the reports.  Truth is used only to say which nodes are
object-anchored (a population definition), never to choose or move a position.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
for p in (ROOT, os.path.join(ROOT, "Topology")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(ROOT, "Topology", "results", "placement")


def lens_of(g: Any, n: int) -> str:
    s = g.sigs.get(n)
    return str(s[0]) if s is not None else "?"


def census(g: Any, pop: List[int], label: str) -> Dict[str, Any]:
    """Nodes per PLACE, per lens, and how positions are written."""
    a = g.subgraph.anchor_of_node
    sites: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for n, xy in a.items():
        sites[(round(float(xy[0]), 3), round(float(xy[1]), 3))].append(int(n))

    node_lens = {int(n): lens_of(g, int(n)) for n in g.subgraph.nodes}
    by_lens: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "distinct": 0, "shared": 0, "integer": 0,
                 "pop": 0, "pop_shared": 0})
    for site, nodes in sites.items():
        for n in nodes:
            L = node_lens.get(n, "?")
            d = by_lens[L]
            d["n"] += 1
            d["distinct"] += 1 if len(nodes) == 1 else 0
            d["shared"] += 1 if len(nodes) > 1 else 0
            d["integer"] += 1 if (float(site[0]) == int(site[0])
                                  and float(site[1]) == int(site[1])) else 0
            if n in pop:
                d["pop"] += 1
                d["pop_shared"] += 1 if len(nodes) > 1 else 0

    # which cross-lens combinations co-occupy a site
    combos: Counter = Counter()
    for site, nodes in sites.items():
        if len(nodes) > 1:
            combos[tuple(sorted({node_lens.get(n, "?") for n in nodes}))] += 1

    pop_sites = Counter()
    for n in pop:
        for site, nodes in sites.items():
            if n in nodes:
                pop_sites[len(nodes)] += 1
                break

    return {"label": label, "nodes": len(a), "distinct": len(sites),
            "share_shared": sum(len(v) - 1 for v in sites.values()
                                if len(v) > 1) / max(1, len(a)),
            "max_at_one_site": max((len(v) for v in sites.values()), default=0),
            "nodes_per_place_hist": dict(sorted(pop_sites.items())),
            "by_lens": {k: dict(v) for k, v in sorted(by_lens.items())},
            "top_cross_lens": [[list(k), v] for k, v in combos.most_common(8)]}


def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev import scene_resolvable
    from Topology.placement_dev3 import all_specs
    from Topology.placement_truth import node_objects

    print("STAGE 0 -- position writers, flags, nodes per place\n")
    print("flags: FE_L3_SPINE=%r FE_L4_JUNCTIONS=%r FE_SUBPIXEL_PEAK_POS=%r "
          "ORIENTATION_PIXEL_ANCHORS=%r"
          % (getattr(P, "FE_L3_SPINE", None), getattr(P, "FE_L4_JUNCTIONS", None),
             getattr(P, "FE_SUBPIXEL_PEAK_POS", None),
             getattr(P, "ORIENTATION_PIXEL_ANCHORS", None)))

    spec = all_specs()[0]
    r = spec.scene.render()
    g = get_or_build(r.image_u8, spec.name + "|s0m5")
    obj = node_objects(r, g)
    pop = [n for n in sorted(obj) if obj[n] >= 0]
    c = census(g, pop, spec.name)

    print("\n%s: %d nodes, %d distinct places, %.3f share a place, max %d/place"
          % (c["label"], c["nodes"], c["distinct"], c["share_shared"],
             c["max_at_one_site"]))
    print("placement-population nodes per place: %s"
          % c["nodes_per_place_hist"])

    print("\n%-18s %5s %8s %7s %8s %6s %9s"
          % ("lens", "n", "distinct", "shared", "integer", "pop", "pop_shared"))
    for L, d in c["by_lens"].items():
        print("%-18s %5d %8d %7d %8d %6d %9d"
              % (L, d["n"], d["distinct"], d["shared"], d["integer"],
                 d["pop"], d["pop_shared"]))

    print("\ncross-lens places (most common lens combinations at one site):")
    for combo, k in c["top_cross_lens"]:
        print("  %3d sites: %s" % (k, " + ".join(combo)))

    res = {"flags": {k: getattr(P, k, None) for k in
                     ("FE_L3_SPINE", "FE_L4_JUNCTIONS", "FE_SUBPIXEL_PEAK_POS",
                      "ORIENTATION_PIXEL_ANCHORS")},
           "scene": c}
    path = os.path.join(OUT, "m5_stage0_census.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
