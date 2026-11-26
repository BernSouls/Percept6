"""Armed known-transform: baseline joint_mcs vs anchored graded_mcs.

PRE-REGISTERED (this cell, no re-tuning):
  arm A  baseline joint_mcs (signature-only node weight, no positional term)
  arm B  graded_mcs with GRADED_MCS_USE_ANCHOR_PRIOR = True
         node weight = signature_similarity * exp(-2.5*hypot(n_x, n_y)),
         positions normalised into each subgraph'\''s bounding box
         (graded_mcs._anchor_factor -- the tree'\''s live weights form)
  arm C  arm B + GRADED_MCS_NODE_REQUIRES_EDGE = False (explicit; the MILP
         demands every selected node pair be supported by a MEASURED relation;
         arm C removes that requirement to isolate the position signal)

0px must read 1.0 / Jaccard 1.0 in every arm.  No ETH-80 classification is
measured here.  Raw node-to-anchor positions were NOT rescaled in this run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

import known_transform as KT  # noqa: E402

SHIFTS = ((0, 0), (2, 0), (-2, 0), (3, 0), (-3, 0))
TOL = 8.0


def _arm_joint_mcs(base, other):
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base["subgraph"], other["subgraph"], base["sigs"], other["sigs"])
    return dict(res.mapping), res.score, "joint_mcs"


def _arm_graded_mcs(base, other, node_requires_edge):
    import parameters as P
    from matching.graded_mcs import graded_mcs
    saved = {}
    keys = ("GRADED_MCS_USE_ANCHOR_PRIOR", "GRADED_MCS_NODE_REQUIRES_EDGE")
    for k in keys:
        saved[k] = getattr(P, k, None)
    try:
        P.GRADED_MCS_USE_ANCHOR_PRIOR = True
        P.GRADED_MCS_NODE_REQUIRES_EDGE = bool(node_requires_edge)
        res = graded_mcs(base["subgraph"], other["subgraph"],
                         base["sigs"], other["sigs"])
    finally:
        for k, v in saved.items():
            if v is not None:
                setattr(P, k, v)
    diag = res.diagnostics
    return dict(res.correspondence), res.score, {
        "node_requires_edge": bool(node_requires_edge),
        "anchor_factor_used": diag.get("anchor_prior"),
        "candidates": diag.get("candidates"),
        "variables": diag.get("variables"),
        "status": diag.get("status", "ok"),
        "max_nodes": diag.get("max_node_vars"),
        "solver": diag.get("solver"),
    }


def _preserved_edges(base, other, mapping):
    """Base edges whose both endpoints are mapped AND the mapped edge exists
    in the other graph (same undirected min/max key, or same directed key)."""
    from core.types import UNDIRECTED_EDGE_TYPES
    su, ou = set(), set()

    def key(s, d, et, orient=True):
        if et in UNDIRECTED_EDGE_TYPES:
            return (min(s, d), max(s, d), et)
        return (s, d, et)

    for (s, d, et) in base["subgraph"].edges:
        su.add(key(s, d, et))
    for (s, d, et) in other["subgraph"].edges:
        ou.add(key(s, d, et))
    eligible = preserved = 0
    det = base["subgraph"].edges
    for (s, d, et) in det:
        u, v = mapping.get(s), mapping.get(d)
        if u is None or v is None:
            continue
        eligible += 1
        if key(u, v, et) in ou:
            preserved += 1
    return eligible, preserved


def run(image_path: str, workers: int):
    import multiprocessing as mp
    from PIL import Image
    with Image.open(image_path) as im:
        width, height = im.size
    extras = [s for s in SHIFTS if tuple(s) != (0, 0)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=guard.activate) as pool:
        built = pool.map(KT._worker, [(tuple(s), image_path, width) for s in extras])
    base = KT._worker(((0, 0), image_path, width))
    by_shift = {(0, 0): base, **{tuple(b["shift"]): b for b in built}}

    pitch = width / 16.0
    arms = {
        "A_baseline_joint_mcs": lambda b, o: _arm_joint_mcs(b, o),
        "B_graded_mcs_anchor_prior": lambda b, o: _arm_graded_mcs(b, o, True),
        "C_graded_mcs_anchor_prior_no_req_edge": lambda b, o: _arm_graded_mcs(b, o, False),
    }

    out_rows = []
    for shift in SHIFTS:
        other = by_shift[tuple(shift)]
        truth, expected = KT._match_continuous(base, other, shift)
        tset = {(u, v) for u, v in truth.items()}
        for arm, fn in arms.items():
            t0 = time.perf_counter()
            if tuple(shift) == (0, 0):
                # [RULES sanity] the report row uses the identity fast path
                # (like known_transform.py).  The controllers below append a
                # SEPARATE row that runs the real solver at 0px, so the
                # "0px must stay 1.0" claim is tested, not asserted.
                matcher = {u: u for u in base["positions"]}
                score, matched = 1.0, len(matcher)
                diag = None
            else:
                matcher, score, diag = fn(base, other)
                matched = len(matcher)
            secs = time.perf_counter() - t0
            hits = 0
            for u, v in matcher.items():
                p = other["positions"].get(v)
                e = expected.get(u)
                if p is None or e is None:
                    continue
                if other["lens"].get(v) != base["lens"].get(u):
                    continue
                if math.hypot(p[0] - e[0], p[1] - e[1]) <= TOL:
                    hits += 1
            mset = {(u, v) for u, v in matcher.items()}
            inter = len(tset & mset)
            union = len(tset | mset)
            leg, pres = _preserved_edges(base, other, matcher)
            row = {
                "arm": arm,
                "shift": list(shift),
                "truth_expected": len(expected),
                "truth_nodes": len(truth),
                "truth_frac": 1.0 if tuple(shift) == (0, 0) else (
                    len(truth) / len(expected) if expected else 0.0),
                "matcher_nodes": matched,
                "matcher_preserved": hits,
                "matcher_frac": (hits / matched) if matched else 0.0,
                "jaccard": (inter / union) if union else 1.0,
                "edge_eligible": leg,
                "edge_preserved": pres,
                "edge_preserved_frac": (pres / leg) if leg else 0.0,
                "score": float(score),
                "diag": diag,
                "seconds": secs,
            }
            out_rows.append(row)
            print(("shift %+d %-38s nodes %4d pres %4d frac %.4f jaccard %.4f edges %d/%d %6.1fs") %
                  (shift[0], arm, matched, hits, row["matcher_frac"], row["jaccard"],
                   pres, leg, secs), flush=True)
            if tuple(shift) == (0, 0):
                # the rule's own sanity row: the SOLVER run on the identity,
                # not the asserted identity fast path.
                t01 = time.perf_counter()
                matcher2, score2, diag2 = fn(base, other)
                matched2 = len(matcher2)
                hits2 = 0
                for u, v in matcher2.items():
                    p = other["positions"].get(v)
                    e = expected.get(u)
                    if p is None or e is None:
                        continue
                    if other["lens"].get(v) != base["lens"].get(u):
                        continue
                    if math.hypot(p[0] - e[0], p[1] - e[1]) <= TOL:
                        hits2 += 1
                matches2 = set(matcher2.items())
                sol_row = dict(row)
                sol_row["matcher_nodes"] = matched2
                sol_row["matcher_preserved"] = hits2
                sol_row["matcher_frac"] = (hits2 / matched2) if matched2 else 0.0
                sol_row["jaccard"] = ((len(tset & matches2) / len(tset | matches2))
                                      if (tset | matches2) else 1.0)
                sol_row["solver_at_zero"] = True
                sol_row["score"] = float(score2)
                sol_row["diag"] = diag2
                sol_row["seconds"] = time.perf_counter() - t01
                out_rows.append(sol_row)
                print("shift +0 solver-at-zero %-33s nodes %4d pres %4d frac %.4f jaccard %.4f" %
                      (arm, matched2, hits2, sol_row["matcher_frac"], sol_row["jaccard"]),
                      flush=True)
    out = {"image": image_path, "rows": out_rows,
           "protocol": ("known-transform, same image/shift, crop not applied; "
                        "truth = nearest same-lens within 8px greedy; Jaccard vs truth")}
    target = RESULTS_DIR / "known_transform_arms.json"
    write_json(target, out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(Path(guard.TREE) / "data" /
                                           "ETH-80-master" / "1" / "2" /
                                           "apple2-022-000.png"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    return run(args.image, args.workers)


if __name__ == "__main__":
    sys.exit(main())
