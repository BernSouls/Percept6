"""M31 item 1d: the product's door against the numbers M30 measured, on all 80 objects and both views.

    python Topology/m31_wire.py --equalities      # the two equalities, 80 queries x 2 views
    python Topology/m31_wire.py --gallery         # build (or load) the shipped colour gallery

The product is ``matching.placement_recognise.recognise``: this script adds nothing to it.  What it
does is drive it with the committed place records (``data/placement_records_m30.npz``) as the
query's record and the gallery's records, so the entire comparison is a records-only computation
that touches no graph and no image beyond the appearance vector.

Two equalities, both at S = 8, on all 80 objects, at both read views:

1. **the fused door** -- ``recognise(matcher=False)`` over the shipped bundle (door.kind "fused3",
   colour weight 0.5, rrf_k 60, shortlist 8, content "colour_rings") must reproduce the M30 door's
   shortlist, cell for cell.  The reference is computed here from the same records through
   ``matching.placement_fuse.fuse`` (the function M30's A1 harness called).
2. **the shipped door** -- ``recognise`` with the bundle's colour weight set to 0 must reproduce the
   two-channel door (``placement_frontdoor.rank_fuse(counted, appearance)``), which is what M27's
   read consumed.  M27's json stores hit COUNTS, not per-object shortlists, so the reference is the
   shipped door's own arithmetic re-derived from the committed records -- the same arithmetic, from
   the same records, that produced those counts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m31_wire.json")
VIEWS = ("035-045", "045-180")
REF = "000-000"
S = 8


def gallery(records_path: str = RECORDS, view: str = REF) -> Any:
    """The shipped gallery, built from the committed records (no graph, no image but appearance)."""
    from matching import placement_gallery as GL
    return GL.from_records(records_path, REC.objects(), view, content="colour_rings")


def _query(obj: Dict[str, Any], view: str, cells: Dict[Any, Any]) -> Tuple[Dict[str, Any], str]:
    rec = cells.get((obj["key"], view))
    if rec is None:
        raise SystemExit("records carry no cell for %s at %s" % (obj["key"], view))
    return rec, REC.path_of(obj, view)


def equalities(records_path: str = RECORDS, limit: int = 0) -> Dict[str, Any]:
    from matching import placement_bundle as PB
    from matching import placement_fuse as FU
    from matching import placement_frontdoor as FD
    from matching import placement_records as PR
    from matching import placement_recognise as PR2
    import parameters as P

    b = PB.PlacementBundle.load()
    gal = gallery(records_path)
    objs = REC.objects()
    if limit:
        objs = objs[:int(limit)]
    cells = PR.load_cells(records_path, [REF] + list(VIEWS))
    gal_cells = [cells[(k, REF)] for k in gal.keys]
    counts = {"cells": 0, "fused_match": 0, "shipped_match": 0, "mismatch": []}
    t0 = time.time()
    for view in VIEWS:
        for o in objs:
            qrec, qpath = _query(o, view, cells)
            app = FD.appearance_feature(qpath)
            counts["cells"] += 1
            # 1. the product's door under the shipped bundle
            r = PR2.recognise(qpath, gal, bundle=b, matcher=False, record=qrec, appearance=app,
                              s=S, use_pair_cache=False)
            product = list(r.shortlist)
            # reference A: the M30 fused door, from the same records
            cnt, _d = FD.count_scores(gal.index, [qrec], n_hits=int(b.door_n_hits))
            ref_app = np.asarray([app], dtype=np.float64) @ gal.appearance.T
            colour = FU.colour_channel(gal.flat_records(), [qrec])
            fused = FU.fuse(cnt, ref_app, np.asarray(colour, dtype=np.float64),
                            weights={"colour": float(b.fusion_weight)}, k=int(b.rrf_k))
            order = sorted(range(gal.n), key=lambda j: (-fused[0, j], j))[:S]
            if product == order:
                counts["fused_match"] += 1
            elif len(counts["mismatch"]) < 8:
                counts["mismatch"].append({"object": o["key"], "view": view, "kind": "fused",
                                           "product": product, "reference": order})
            # 2. the shipped door: colour weight 0
            import dataclasses
            b0 = dataclasses.replace(b, door_params={**dict(b.door_params), "colour_weight": 0.0})
            r0 = PR2.recognise(qpath, gal, bundle=b0, matcher=False, record=qrec, appearance=app,
                               s=S, use_pair_cache=False)
            shipped = FD.rank_fuse(ref_app, cnt, k=int(b.rrf_k))
            order0 = sorted(range(gal.n), key=lambda j: (-shipped[0, j], j))[:S]
            if list(r0.shortlist) == order0:
                counts["shipped_match"] += 1
            elif len(counts["mismatch"]) < 8:
                counts["mismatch"].append({"object": o["key"], "view": view, "kind": "shipped",
                                           "product": list(r0.shortlist), "reference": order0})
            if counts["cells"] % 20 == 0:
                print("  %d cells  fused %d  shipped %d  %.0fs"
                      % (counts["cells"], counts["fused_match"], counts["shipped_match"],
                         time.time() - t0), flush=True)
    out = {"cells": counts["cells"], "S": S, "views": list(VIEWS), "objects": len(objs),
           "fused_matches": counts["fused_match"], "shipped_matches": counts["shipped_match"],
           "fused_identical": counts["fused_match"] == counts["cells"],
           "shipped_identical": counts["shipped_match"] == counts["cells"],
           "mismatches": counts["mismatch"], "seconds": round(time.time() - t0, 1),
           "bundle": {"door": b.door_kind, "colour_weight": b.fusion_weight, "rrf_k": b.rrf_k,
                      "shortlist": b.shortlist, "content": b.content_kind,
                      "blend": dict(b.matcher_blend), "code_digest": b.code_digest[:16]}}
    print(json.dumps({k: v for k, v in out.items() if k != "mismatches"}, indent=1))
    return out


def doorclaim(records_path: str = RECORDS, s: int = S) -> Dict[str, Any]:
    """The paired claim: the fused door against the shipped door, ALL 80 objects, both read views.

    Recall@S and top-1 per arm, the per-object difference, an object-clustered bootstrap interval on
    that difference (10,000 resamples, seed 20261118, resampling OBJECTS and recomputing both arms on
    the same resample), and the up/down/tied counts.  Records only: no graph, no matcher.
    """
    from matching import placement_frontdoor as FD
    from matching import placement_fuse as FU
    from matching import placement_records as PR
    from Topology.m27_read import RESAMPLES, SEED
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    gi = {k: i for i, k in enumerate(keys)}
    gal = gallery(records_path)
    cells = PR.load_cells(records_path, [REF] + list(VIEWS))
    gal_cells = [cells[(k, REF)] for k in keys]
    out: Dict[str, Any] = {"S": s, "objects": len(objs), "chance_recall": s / len(objs),
                           "chance_top1": 1.0 / len(objs), "views": {}}
    for view in VIEWS:
        qs = [cells[(k, view)] for k in keys]
        cnt, _i = FD.count_scores(gal.index, qs, n_hits=8)
        app = FD.appearance_scores([REC.path_of(o, view) for o in objs],
                                   [REC.path_of(o, REF) for o in objs])
        colour = np.asarray(FU.colour_channel(gal_cells, qs), dtype=np.float64)
        fused = np.asarray(FU.fuse(cnt, app, colour, weights={"colour": 0.5}, k=60))
        shipped = np.asarray(FD.rank_fuse(app, cnt, k=60))
        rec_f = np.zeros(len(keys))
        rec_s = np.zeros(len(keys))
        top_f = np.zeros(len(keys))
        top_s = np.zeros(len(keys))
        for i, o in enumerate(objs):
            t = gi[o["key"]]
            of = sorted(range(len(keys)), key=lambda j: (-fused[i, j], j))
            os_ = sorted(range(len(keys)), key=lambda j: (-shipped[i, j], j))
            rec_f[i] = float(t in of[:s])
            rec_s[i] = float(t in os_[:s])
            top_f[i] = float(of[0] == t)
            top_s[i] = float(os_[0] == t)
        d = rec_f - rec_s
        rng = np.random.default_rng(SEED)
        boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(RESAMPLES)])
        dt = top_f - top_s
        boott = np.asarray([dt[rng.integers(0, len(dt), len(dt))].mean()
                            for _ in range(RESAMPLES)])
        out["views"][view] = {
            "recall": {"shipped": {"hits": int(rec_s.sum()), "rate": float(rec_s.mean())},
                       "fused": {"hits": int(rec_f.sum()), "rate": float(rec_f.mean())},
                       "delta": float(d.mean()),
                       "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       "up": int((d > 0).sum()), "down": int((d < 0).sum()),
                       "tied": int((d == 0).sum()),
                       "objects_gained": [objs[i]["key"] for i in np.where(d > 0)[0]],
                       "objects_lost": [objs[i]["key"] for i in np.where(d < 0)[0]]},
            "top1": {"shipped": {"hits": int(top_s.sum()), "rate": float(top_s.mean())},
                     "fused": {"hits": int(top_f.sum()), "rate": float(top_f.mean())},
                     "delta": float(dt.mean()),
                     "ci95": [float(np.percentile(boott, 2.5)),
                              float(np.percentile(boott, 97.5))]},
            "resamples": RESAMPLES, "seed": SEED,
        }
        r = out["views"][view]["recall"]
        print("  %s recall@%d: shipped %d/%d %.3f -> fused %d/%d %.3f  delta %+.3f "
              "[%.3f, %.3f]  up %d / down %d / tied %d"
              % (view, s, r["shipped"]["hits"], len(keys), r["shipped"]["rate"],
                 r["fused"]["hits"], len(keys), r["fused"]["rate"], r["delta"],
                 r["ci95"][0], r["ci95"][1], r["up"], r["down"], r["tied"]), flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equalities", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--doorclaim", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    data: Dict[str, Any] = {}
    if os.path.exists(OUT):
        try:
            data = json.load(open(OUT))
        except Exception:
            data = {}
    if a.equalities:
        data["equalities"] = equalities(limit=a.limit)
    if a.doorclaim:
        data["doorclaim"] = doorclaim()
    if a.equalities or a.doorclaim:
        with open(OUT, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True, default=float)
        print("wrote", OUT)
    if not (a.equalities or a.doorclaim):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
