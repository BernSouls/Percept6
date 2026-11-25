"""Mission 15 item 8 -- WHO WRITES A NODE'S POSITION, and does a finer writer exist?

The tree records a node's position ONCE, at the step that minted it
(`ConceptGraph.note_pattern_node_position`, first-write-wins), from the binder's
step map (`Binder._step_anchor_positions`) -- a TILING position: the centroid of the
cell of the lattice that fired.  A node minted from a tiling lens therefore has no
finer writer at all: its position is a lattice point, and asking for "the exact
position of this texture patch" has no answer beyond the tile.

Other lenses do have finer writers, and they exist as front-end switches:
`FE_PLACE_LAYER` (the Gabor place layer, `lenses/base.py:step` -> `gabor_place_positions`,
which returns a sub-pixel peak) and `FE_SUBPIXEL_PEAK_POS` / `FE_SUBPIXEL_GABOR_ANCHOR`
(the keypoint anchor refinement in `matching/placement_edges.node_frames_keypoint`).

So "prefer precision over first-write-wins" is admissible EXACTLY for the lenses whose
anchor measurably moves when a finer writer is on, and inadmissible for the tiling
lenses, which are attributes (mission item 3), not places.

This file measures it: one image, four front-end configurations, and for every lens
the share of nodes whose anchor moves, under which switch.

    python Topology/placement_m15_writers.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m15_position_writers.json")

#: (label, the switches that make the writer exist, code reference)
WRITERS: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("defaults", {}, "Binder._step_anchor_positions -> note_pattern_node_position "
                     "(tiling centroid, first-write-wins)"),
    ("place_layer", {"FE_PLACE_LAYER": True},
     "lenses/base.py:step -> gabor_place_positions (sub-pixel peak)"),
    ("subpixel", {"FE_SUBPIXEL_PEAK_POS": True},
     "FE_SUBPIXEL_PEAK_POS -> matching/placement_edges.node_frames_keypoint"),
    ("both", {"FE_PLACE_LAYER": True, "FE_SUBPIXEL_PEAK_POS": True,
              "FE_SUBPIXEL_GABOR_ANCHOR": True},
     "both finer writers"),
)
TILING = ("Texture", "Color", "Orientation")


def _job(args: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, label, flags = args
    import guard
    guard.activate()
    import parameters as P
    from Topology import placement_m11_families as F
    from Topology import placement_m12_realphotos as RP
    from Topology.placement_cache import get_or_build

    for k, v in flags.items():
        setattr(P, k, v)
    if tag.startswith("syn|"):
        name = tag.split("|", 1)[1]
        img = [s for s in F.gallery() if s.name == name][0].scene.render().image_u8
    else:
        img = RP.images("validation")[tag.split("|", 1)[1]]
    key = "m15atlas|%s|orig|%d" % (tag, 0 if not flags else 2) + "|" + label
    g = get_or_build(img, key)
    sub = g.subgraph
    rows = []
    for n in sorted(sub.nodes):
        if n not in sub.anchor_of_node:
            continue
        lens = str(g.sigs.get(n, ("?",))[0])
        x, y = sub.anchor_of_node[n]
        rows.append({"n": int(n), "lens": lens, "x": float(x), "y": float(y)})
    return {"tag": tag, "label": label, "nodes": rows}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="syn|m11ir_00")
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF

    jobs = [(args.tag, lab, fl) for lab, fl, _ref in WRITERS]
    got: Dict[str, Dict[int, Dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=1) as ex:
        for r in ex.map(_job, jobs):
            got[r["label"]] = {n["n"]: n for n in r["nodes"]}
            print("  %-12s %d nodes" % (r["label"], len(r["nodes"])), flush=True)
    base = got["defaults"]
    per_lens: Dict[str, Dict[str, Any]] = {}
    for lens in sorted({v["lens"] for v in base.values()}):
        ids = [n for n, v in base.items() if v["lens"] == lens]
        row: Dict[str, Any] = {"nodes": len(ids), "tiling": lens in TILING,
                               "writers": {}}
        for lab in ("place_layer", "subpixel", "both"):
            moves = []
            for n in ids:
                a, b = base.get(n), got.get(lab, {}).get(n)
                if not a or not b:
                    continue
                moves.append(float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])))
            row["writers"][lab] = {
                "shared": len(moves),
                "moved_share": (float(np.mean([m > 0.5 for m in moves]))
                                if moves else None),
                "median_move": float(np.median(moves)) if moves else None,
                "max_move": float(np.max(moves)) if moves else None,
            }
        finer = [lab for lab in ("place_layer", "subpixel")
                 if (row["writers"][lab]["moved_share"] or 0.0) > 0.25]
        row["finer_writer_exists"] = bool(finer)
        row["finer_writer"] = (", ".join(
            {"place_layer": "FE_PLACE_LAYER (gabor_place_positions)",
             "subpixel": "FE_SUBPIXEL_PEAK_POS (node_frames_keypoint)"}[f]
            for f in finer) or None)
        row["verdict"] = (
            "ATTRIBUTE, no finer writer: position is the tiling centroid, "
            "first-write-wins" if not finer else
            "PLACE-capable: a finer writer exists (%s), so precision may be preferred "
            "over first-write-wins" % row["finer_writer"])
        per_lens[lens] = row
    res = {"tag": args.tag, "writers": {lab: ref for lab, _f, ref in WRITERS},
           "tiling_lenses_assumed": list(TILING), "per_lens": per_lens}
    MF.write_result(RESULT, res)
    print("\n%-16s %6s %8s %8s %8s  %s"
          % ("lens", "nodes", "place_ly", "subpix", "both", "verdict"))
    for lens, r in sorted(per_lens.items()):
        def _f(lab):
            v = r["writers"][lab]["moved_share"]
            return "-" if v is None else "%.2f" % v
        print("%-16s %6d %8s %8s %8s  %s"
              % (lens, r["nodes"], _f("place_layer"), _f("subpixel"), _f("both"),
                 r["verdict"][:60]))
    print("\nwrote", RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
