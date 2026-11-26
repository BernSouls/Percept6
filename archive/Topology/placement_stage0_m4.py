"""MISSION 4 STAGE 0 -- cheap checks on the front end's positions.

Answers, on one scene, with and without FE_SUBPIXEL_PEAK_POS:

  0a  was the flag on, and does the consumer path read the sub-pixel position?
  0c  how many nodes share an exact position; per lens; lattice vs content;
      what share of AMBIGUOUS OBJECT nodes are anchored to a shared site.

Truth is used only to say which nodes are object-anchored (scoring a population),
never to choose a position.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
for p in (ROOT, os.path.join(ROOT, "Topology")):
    if p not in sys.path:
        sys.path.insert(0, p)


def census(g: Any, label: str) -> Dict[str, Any]:
    a = g.subgraph.anchor_of_node
    pos: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for n, xy in a.items():
        pos[(round(float(xy[0]), 3), round(float(xy[1]), 3))].append(int(n))
    n = len(a)
    shared = sum(len(v) - 1 for v in pos.values() if len(v) > 1)
    lens = {}
    for node in g.subgraph.nodes:
        s = g.sigs.get(node)
        lens[int(node)] = str(s[0]) if s is not None else "?"
    by_lens: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "shared": 0})
    for site, nodes in pos.items():
        for node in nodes:
            L = lens.get(node, "?")
            by_lens[L]["n"] += 1
            if len(nodes) > 1:
                by_lens[L]["shared"] += 1
    cross = 0
    for site, nodes in pos.items():
        if len({lens.get(x, "?") for x in nodes}) > 1:
            cross += len(nodes)
    integral = sum(1 for x, y in a.values()
                   if float(x) == int(x) and float(y) == int(y))
    return {"label": label, "nodes": n, "distinct": len(pos),
            "share_shared": shared / max(1, n), "max_at_one_site": max(
                (len(v) for v in pos.values()), default=0),
            "share_integer": integral / max(1, n),
            "share_on_cross_lens_site": cross / max(1, n),
            "by_lens": {k: {"n": v["n"], "shared": v["shared"],
                            "share": v["shared"] / max(1, v["n"])}
                        for k, v in sorted(by_lens.items())},
            "sites": pos}


def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import all_specs
    from Topology.placement_truth import node_objects

    print("STAGE 0a/0c -- front-end position census\n")
    print("FE_SUBPIXEL_PEAK_POS default = %r" % (getattr(
        P, "FE_SUBPIXEL_PEAK_POS", None),))

    spec = all_specs()[0]
    r = spec.scene.render()

    out = {}
    for flag in (False, True):
        if bool(getattr(P, "FE_SUBPIXEL_PEAK_POS", False)) != flag:
            setattr(P, "FE_SUBPIXEL_PEAK_POS", flag)
        g = get_or_build(r.image_u8, "%s|s0|sub%d" % (spec.name, int(flag)))
        c = census(g, "subpixel=%s" % flag)
        out[flag] = c
        obj = node_objects(r, g)
        pop = [n for n in sorted(obj) if obj[n] >= 0]
        sites = c["sites"]
        site_of = {}
        for site, nodes in sites.items():
            for x in nodes:
                site_of[x] = site
        shared_pop = sum(1 for x in pop if len(sites[site_of[x]]) > 1)
        print("%-16s nodes %4d  distinct %4d  shared %.3f  max/site %2d  "
              "integer %.3f  cross-lens %.3f"
              % (c["label"], c["nodes"], c["distinct"], c["share_shared"],
                 c["max_at_one_site"], c["share_integer"],
                 c["share_on_cross_lens_site"]))
        print("%-16s object-anchored %3d, of which on a SHARED site %.3f"
              % (c["label"], len(pop), shared_pop / max(1, len(pop))))

    print("\nper lens (share of that lens's nodes sitting on a shared site):")
    keys = sorted(set(out[False]["by_lens"]) | set(out[True]["by_lens"]))
    print("%-18s %6s %8s %6s %8s" % ("lens", "n(off)", "shared", "n(on)",
                                     "shared"))
    for k in keys:
        a = out[False]["by_lens"].get(k, {"n": 0, "share": float("nan")})
        b = out[True]["by_lens"].get(k, {"n": 0, "share": float("nan")})
        print("%-18s %6d %8.3f %6d %8.3f" % (k, a["n"], a["share"], b["n"],
                                             b["share"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
