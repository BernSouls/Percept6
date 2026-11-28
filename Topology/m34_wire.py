"""M34 item 6: prove the SHIPPED door now computes the number M34 measured.

The item-4 read (`Topology/m34_door.py`) measured the five-channel door directly from the records.
That is a measurement, not the shipped path.  This file closes the gap: it runs
`matching.placement_recognise.recognise(matcher=False)` -- the entry point the bundle's configuration
actually drives -- on the same objects and compares the door's object ORDER against the item-4
arithmetic.  Two things are checked:

  IDENTITY   the shipped door (with the fragment channel at the bundle's weight) returns the same
             order as the item-4 fused matrix, for every object it is run on;
  RESOLUTION the shipped query is a PATH: with no anchors handed in, the door finds that cell's
             anchors by itself and returns the SAME order as when they are passed explicitly.  A
             door that quietly dropped the channel would return a different order, and a door that
             zeroed it would too.

    python Topology/m34_wire.py --verify
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

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
CHANNELS = os.path.join(ROOT, "Topology", "results", "placement", "m32_colour_channels.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m34_wire.json")
REF = "000-000"
VIEW = "035-045"
K = 60
N_QUERIES = 5


def _item4_row(keys: List[str], qi: int) -> np.ndarray:
    """The measure's fused row for one query, exactly as `m34_door.py` builds it."""
    from matching import placement_fuse as F
    from matching import placement_records as PR
    import Topology.m34_door as D
    cells = PR.load_cells(RECORDS, [REF, VIEW])
    gal = [{"places": cells[(k, REF)]["places"]} for k in keys]
    qs = [{"places": cells[(k, VIEW)]["places"]} for k in keys]
    z = np.load(CHANNELS)
    colour = F.colour_channel(gal, qs)
    frag = D.fragment_matrix(VIEW)
    fused = F.rrf_weighted([z["counted|%s" % VIEW], z["appearance|%s" % VIEW], colour, frag],
                           [1.0, 1.0, 0.5, 0.5], k=K)
    return fused[qi]


def verify(n: int = N_QUERIES) -> Dict[str, Any]:
    from matching import placement_anchors as PA
    from matching import placement_gallery as GL
    from matching import placement_recognise as PR_REC
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    table = PA.load()
    gallery = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    GL.attach_fragment_anchors(gallery, table)
    cells = PR.load_cells(RECORDS, [VIEW])
    out: Dict[str, Any] = {"queries": [], "identity": 0, "resolved": 0, "n": 0}
    for o in objs[:n]:
        k = o["key"]
        qi = keys.index(k)
        want = _item4_row(keys, qi)
        want_order = [j for j in np.argsort(-want, kind="stable")][:8]
        # the shipped path, with the bundle's own weights and the anchors it needs
        res = PR_REC.recognise((k, REC.path_of(o, VIEW)), gallery, matcher=False, workers=1,
                               record=cells[(k, VIEW)], anchors=PA.anchors_of(table, k, VIEW))
        got = list(res.shortlist)[:8]
        same = [j for j in got] == [int(j) for j in want_order]
        # the resolution: the same bundle, NO anchors handed in -- the door must find them
        resolved = False
        try:
            r2 = PR_REC.recognise((k, REC.path_of(o, VIEW)), gallery, matcher=False, workers=1,
                                  record=cells[(k, VIEW)], anchors=None)
            resolved = list(r2.shortlist)[:8] == [int(j) for j in want_order]
        except Exception as exc:                             # noqa: BLE001 - a failure is data
            print("     resolved call failed: %r" % (exc,))
        out["queries"].append({"object": k, "same_order": bool(same),
                               "resolved_from_path": bool(resolved),
                               "shipped_top8": [int(j) for j in got],
                               "measured_top8": [int(j) for j in want_order]})
        out["identity"] += int(same)
        out["resolved"] += int(resolved)
        out["n"] += 1
        print("  %-12s order identical %s   resolved from the path %s"
              % (k, same, resolved), flush=True)
    out["verdict"] = ("wired" if out["identity"] == out["n"] and out["resolved"] == out["n"]
                      else "NOT the measured arithmetic")
    print(json.dumps({k: v for k, v in out.items() if k != "queries"}, indent=1))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=N_QUERIES)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.verify:
        r = verify(a.n)
        return 0 if r["verdict"] == "wired" else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
