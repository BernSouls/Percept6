"""Task 4 PART E6: flicker per family + graph census + build time.

For each dev scene and arm (B0, B1, F), at shift +2:
  base edges preserved      both endpoints survive AND edge present in shifted
  lost_endpoint_vanished    at least one endpoint has no counterpart
  lost_both_survive         both endpoints survive but edge absent in shifted
  added_new_node            shifted edge with a node that has no base counterpart
  added_surviving           shifted edge between two surviving nodes, not in base

Graph census: nodes, edges, per-family counts, build time.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _counterpart(base_pos, sh_pos, shift_px, tol=2.0):
    by_lens = defaultdict(list)
    for n, (x, y, l) in sh_pos.items():
        by_lens[l].append((n, x, y))
    out = {}
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = x0 + shift_px, y0
        best_n, best_d = None, tol
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            d = math.hypot(ex - xsh, ey - ysh)
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is not None:
            out[n] = best_n
    return out


def _load_dev(arm):
    idx = json.load(open(RESULTS_DIR / "t4" / "dev_graphs_%s.json" % arm))
    return idx


def main() -> int:
    # We need shifted graphs for B0/B1/F at +2.  Rebuild them here (cached
    # from the part0 dev run's rolled PNGs).
    import _bootstrap
    import corpus
    from frontend import build
    from PIL import Image

    split = corpus.split_tuning()
    by_class = {}
    for e in split["dev"]:
        if e["view"] != "-090-000":
            continue
        cls = e["class"]
        if cls not in by_class or e["object"] < by_class[cls]["object"]:
            by_class[cls] = e
    entries = [by_class[c] for c in corpus.CLASSES]

    out = {"arms": {}, "shift": 2, "tol_px": 2.0}
    for arm, spec in ARMS.items():
        os.environ["PERCEPT_SET"] = spec
        _bootstrap.apply_switch_overrides(spec)
        arm_out = {"per_scene": {}, "census": {}}
        fam_flicker = Counter()
        total_flicker = Counter()
        for e in entries:
            name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
            # base
            base = build(e)
            # shifted
            tmp = Path("/tmp/t4_roll") / ("%s%d%s_2.png"
                                          % (e["class"], int(e["object"]),
                                             e["view"]))
            if not tmp.is_file():
                tmp.parent.mkdir(parents=True, exist_ok=True)
                arr = np.asarray(Image.open(e["path"]).convert("RGB"),
                                 dtype=np.uint8)
                Image.fromarray(np.roll(arr, (2, 0), axis=(0, 1))).save(tmp)
            e2 = dict(e)
            e2["path"] = str(tmp)
            sh = build(e2)

            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh)
            cp = _counterpart(base_pos, sh_pos, 2)

            # base edges classification
            preserved = 0
            lost_ep = 0
            lost_both = 0
            base_fam = Counter()
            for (u, v, et) in base.subgraph.edges:
                fam = str(getattr(et, "value", et))
                base_fam[fam] += 1
                eu, ev = cp.get(u), cp.get(v)
                if eu is None or ev is None:
                    lost_ep += 1
                    fam_flicker[fam + "|lost_ep"] += 1
                else:
                    key = (min(eu, ev), max(eu, ev), et)
                    sh_edges = {(min(a, b), max(a, b), c)
                                for (a, b, c) in sh.subgraph.edges}
                    if key in sh_edges:
                        preserved += 1
                        fam_flicker[fam + "|preserved"] += 1
                    else:
                        lost_both += 1
                        fam_flicker[fam + "|lost_both"] += 1

            # shifted edges classification
            added_new = 0
            added_surv = 0
            sh_fam = Counter()
            for (u, v, et) in sh.subgraph.edges:
                fam = str(getattr(et, "value", et))
                sh_fam[fam] += 1
                # reverse map
                inv = {v2: k2 for k2, v2 in cp.items()}
                bu, bv = inv.get(u), inv.get(v)
                if bu is None or bv is None:
                    added_new += 1
                else:
                    key = (min(bu, bv), max(bu, bv), et)
                    base_edges = {(min(a, b), max(a, b), c)
                                  for (a, b, c) in base.subgraph.edges}
                    if key not in base_edges:
                        added_surv += 1

            arm_out["per_scene"][name] = {
                "nodes": len(base.subgraph.nodes),
                "edges": len(base.subgraph.edges),
                "base_families": dict(base_fam),
                "sh_families": dict(sh_fam),
                "preserved": preserved,
                "lost_endpoint": lost_ep,
                "lost_both_survive": lost_both,
                "added_new_node": added_new,
                "added_surviving": added_surv,
                "build_seconds_base": base.frontend_seconds,
                "build_seconds_sh": sh.frontend_seconds,
            }
            total_flicker["preserved"] += preserved
            total_flicker["lost_ep"] += lost_ep
            total_flicker["lost_both"] += lost_both
            total_flicker["added_new"] += added_new
            total_flicker["added_surv"] += added_surv

        arm_out["totals"] = dict(total_flicker)
        arm_out["fam_flicker"] = dict(fam_flicker)
        arm_out["census"] = {
            "mean_nodes": float(np.mean([
                r["nodes"] for r in arm_out["per_scene"].values()])),
            "mean_edges": float(np.mean([
                r["edges"] for r in arm_out["per_scene"].values()])),
            "mean_build_s": float(np.mean([
                r["build_seconds_base"]
                for r in arm_out["per_scene"].values()])),
        }
        out["arms"][arm] = arm_out
        print("arm %s: preserved=%d lost_ep=%d lost_both=%d "
              "added_new=%d added_surv=%d"
              % (arm, total_flicker["preserved"], total_flicker["lost_ep"],
                 total_flicker["lost_both"], total_flicker["added_new"],
                 total_flicker["added_surv"]), flush=True)

    write_json(RESULTS_DIR / "t4" / "e6_flicker.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
