"""M34: export and check `data/placement_anchors_m34.npz` (the fragment anchors the door reads).

Thin harness, by the project's rule: the loading/geometry is in `matching/placement_anchors.py`, the
graph building is `matching.placement_frontend`, and nothing but this script knows about the
corpus list.

    python Topology/m34_anchors.py --export     # rebuild the file from the front end (resumable)
    python Topology/m34_anchors.py --check      # every cell against a fresh graph: 0 diffs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402


def export() -> Dict[str, Any]:
    from matching import placement_anchors as PA
    import matching.placement_frontend as FE
    payload: Dict[str, np.ndarray] = {}
    if os.path.exists(PA.PATH):
        z = np.load(PA.PATH)
        payload = {k: z[k] for k in z.files if PA._is_cell(k)}
    objs = REC.objects()
    vs = list(REC.VIEWS)
    done = 0
    for o in objs:
        for v in vs:
            ks = [PA.key(o["key"], v, g) for g in PA.GROUPINGS]
            if all(k in payload for k in ks):
                continue
            _img, g, _s = FE.cell(REC.path_of(o, v), "m34anchors")
            for gname, k in zip(PA.GROUPINGS, ks):
                payload[k] = PA.anchors_from_graph(g, gname)
            done += 1
    stems: List[str] = []
    stem_keys: List[str] = []
    for o in objs:
        for v in vs:
            stems.append(os.path.basename(REC.path_of(o, v))[:-4])
            stem_keys.append("%s|%s" % (o["key"], v))
    payload[PA.STEMS] = np.asarray(stems)
    payload[PA.STEM_KEYS] = np.asarray(stem_keys)
    np.savez(PA.PATH, **payload)
    return {"file": os.path.relpath(PA.PATH, ROOT), "cells": len(payload) - 2,
            "rebuilt_this_run": done, "size_kb": round(os.path.getsize(PA.PATH) / 1e3, 1),
            "digest": PA.digest()[:16]}


def check() -> Dict[str, Any]:
    from matching import placement_anchors as PA
    import matching.placement_frontend as FE
    table = PA.load()
    diffs = 0
    cells = 0
    for o in REC.objects():
        for v in REC.VIEWS:
            _img, g, _s = FE.cell(REC.path_of(o, v), "m34anchors")
            for gname in PA.GROUPINGS:
                want = PA.anchors_from_graph(g, gname)
                have = PA.anchors_of(table, o["key"], v, gname)
                cells += 1
                if want.shape != have.shape or not np.array_equal(want, have):
                    diffs += 1
    return {"cells": cells, "diffs": diffs, "digest": PA.digest()[:16]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.export:
        print(json.dumps(export(), indent=1))
        return 0
    if a.check:
        r = check()
        print(json.dumps(r, indent=1))
        return 0 if r["diffs"] == 0 else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
