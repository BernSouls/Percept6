"""Task 4: build ALL 48 dev graphs for arms B0 and B1.

Dev = all 48 scenes in corpus.split_tuning()["dev"] (24 at -090-000,
24 at -035-135).  Each graph is pickled to results/t4/dev_pkls/ so the
graphs survive cache resets.  JSON index written to results/t4/.

Usage:
    python t4_build_dev_all.py --arm B0
    python t4_build_dev_all.py --arm B1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}
PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"


def _dev_entries():
    import guard
    guard.activate()
    import corpus
    split = corpus.split_tuning()
    # Use -090-000 view only (24 scenes)
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()

    spec = ARMS[args.arm]
    os.environ["PERCEPT_SET"] = spec
    import _bootstrap
    applied = _bootstrap.apply_switch_overrides(spec)
    print("arm", args.arm, "applied:", applied, flush=True)

    import guard
    guard.activate()
    import corpus
    from frontend import build

    entries = _dev_entries()
    print("dev entries: %d scenes" % len(entries), flush=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    metas = []
    t0 = time.perf_counter()
    for i, e in enumerate(entries):
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        pkl_path = PKL_DIR / ("%s_%s.pkl" % (args.arm, name))
        if pkl_path.is_file():
            with open(pkl_path, "rb") as fh:
                g = pickle.load(fh)
        else:
            g = build(e)
            with open(pkl_path, "wb") as fh:
                pickle.dump(g, fh)
        metas.append({
            "name": name,
            "class_": e["class"],
            "object_": int(e["object"]),
            "view": e["view"],
            "nodes": len(g.subgraph.nodes),
            "edges": len(g.subgraph.edges),
            "pkl": str(pkl_path),
            "frontend_seconds": getattr(g, "frontend_seconds", 0.0),
        })
        if (i + 1) % 8 == 0 or i + 1 == len(entries):
            print("  %3d/%d (%.0fs)" % (i + 1, len(entries),
                                        time.perf_counter() - t0),
                  flush=True)

    metas.sort(key=lambda m: m["name"])
    out = {
        "arm": args.arm,
        "switch_spec": spec,
        "applied": applied,
        "n_scenes": len(metas),
        "graphs": metas,
        "build_seconds": time.perf_counter() - t0,
    }
    path = Path(ROOT) / "Topology" / "results" / "t4" / ("dev_graphs_all_%s.json" % args.arm)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
