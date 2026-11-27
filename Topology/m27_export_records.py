"""M27 item 1b: export the cached place records to the compact committed file.

    python Topology/m27_export_records.py --build     # write data/placement_records_m27.npz
    python Topology/m27_export_records.py --check     # the loader reproduces the live door

The live records are the pickles ``Topology/m26_records.py`` writes; the export is what a fresh
sandbox gets instead of a 2,338 s rebuild.  ``--check`` proves the substitution: the door's
shortlist from the npz records must equal the door's shortlist from the live records, cell by cell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC  # noqa: E402

OUT = os.path.join(ROOT, "data", "placement_records_m27.npz")


def export(path: str = OUT) -> Dict[str, Any]:
    from matching import placement_records as PR
    t0 = time.time()
    objs = REC.objects()
    cells = []
    for o in objs:
        for v in REC.VIEWS:
            p = REC.cache_path(o["key"], v)
            if os.path.exists(p):
                cells.append((o["key"], v, REC.load(o["key"], v)))
    arrays = PR.export(cells, [o["key"] for o in objs], list(REC.VIEWS))
    PR.save(path, arrays)
    info = PR.peek(path)
    info["cells"] = len(cells)
    info["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in info.items() if k != "shapes"}, indent=1))
    return info


def check(path: str = OUT, view: str = "035-045") -> Dict[str, Any]:
    """The npz loader must reproduce the live door's shortlists, 80/80."""
    from matching import placement_records as PR
    from matching import placement_frontdoor as FD
    from matching import placement_bundle as PB
    import matching.placement_frontend as FE
    FE.apply_recipe()
    b = PB.PlacementBundle.load()
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    live = [REC.load(k, REC.REF_VIEW) for k in keys]
    npz = PR.load_cells(path, [REC.REF_VIEW])
    packed = [npz[(k, REC.REF_VIEW)] for k in keys]
    same_places = sum(1 for a, c in zip(live, packed) if len(a["places"]) == len(c["places"]))
    idx_live = FD.PlaceIndex(live, keys)
    idx_npz = FD.PlaceIndex(packed, keys)
    agree = 0
    diffs: List[Dict[str, Any]] = []
    for o in objs:
        q = REC.load(o["key"], view)
        a, _ = FD.count_scores(idx_live, [q], n_hits=int(b.door_n_hits))
        c, _ = FD.count_scores(idx_npz, [q], n_hits=int(b.door_n_hits))
        la = sorted(range(len(keys)), key=lambda j: (-a[0, j], j))[:8]
        lc = sorted(range(len(keys)), key=lambda j: (-c[0, j], j))[:8]
        agree += int(la == lc)
        if la != lc:
            diffs.append({"object": o["key"], "live": la, "npz": lc})
    out = {"cells": len(keys), "place_counts_identical": same_places,
           "shortlists_identical": agree, "view": view, "differences": diffs[:5],
           "max_abs_score_diff": None}
    worst = 0.0
    for o in objs[:20]:
        q = REC.load(o["key"], view)
        a, _ = FD.count_scores(idx_live, [q], n_hits=int(b.door_n_hits))
        c, _ = FD.count_scores(idx_npz, [q], n_hits=int(b.door_n_hits))
        worst = max(worst, float(np.max(np.abs(a - c))))
    out["max_abs_score_diff"] = worst
    print(json.dumps(out, indent=1))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json")
    a = ap.parse_args(list(argv) if argv is not None else None)
    res: Dict[str, Any] = {}
    if a.build:
        res["export"] = export(a.out)
    if a.check:
        res["check"] = check(a.out)
    if a.json and res:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print("wrote", a.json)
    if not (a.build or a.check):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
