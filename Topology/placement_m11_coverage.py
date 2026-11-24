"""MISSION 11 STEP 5 -- why real images judge so few pairs, measured not guessed.

The Mission-10 ETH-80 read found only ~18 % of query nodes carrying an attached
place and, in this mission's fit, ~14 % of MAPPED pairs having a single slot the
frozen estimator could check.  The pair-score arm can only add evidence where a pair
is judgeable, so this measures the chain that produces a judged pair, on real
photographs, one link at a time:

    nodes
      -> nodes whose anchor carries an ORIENTATION FRAME      (frame provider)
      -> nodes with at least one PAYLOAD EDGE                 (target selection)
      -> mapped pairs                                         (the aligner)
      -> mapped pairs with >= 1 JUDGED SLOT                   (the estimator)

It then repeats the frame link with the MULTI-SCALE provider already in
`matching/placement_edges.py` (`field_multiscale` + `node_frames_multiscale`, whose
own note records that the single 7x7 window left 66 % of the low-contrast nodes
without a frame).  If that is the limiting link, the fix is a window choice, not a
tolerance.

Nothing here fits anything and nothing here scores an arm.  Absolute position is
never scored; the mapping is the deployed matcher's own.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
IMG = 256
N_PAIRS = 8


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def pairs() -> List[Tuple[dict, dict]]:
    """Deterministic real pairs: reference view vs a fresh query view."""
    from Topology import corpus
    from Topology.placement_m11_eth80 import catalog
    refs, queries = catalog()
    by_key = {}
    for r in refs:
        by_key.setdefault(r["key"], r)
    out: List[Tuple[dict, dict]] = []
    seen = set()
    for q in queries:
        if q["key"] in seen:
            continue
        seen.add(q["key"])
        out.append((q, by_key[q["key"]]))
        if len(out) >= N_PAIRS:
            break
    return out


def _one(job: Tuple[str, str]) -> Dict[str, Any]:
    qpath, rpath = job
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_edges as PE
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build

    q_img, r_img = _load_u8(qpath), _load_u8(rpath)
    q = get_or_build(q_img, "eth|" + qpath)
    r = get_or_build(r_img, "eth|" + rpath)

    out: Dict[str, Any] = {"q": qpath, "r": rpath,
                           "nodes_q": len(list(q.subgraph.nodes)),
                           "nodes_r": len(list(r.subgraph.nodes))}
    # 1. the frame link, single window and multi-scale
    ang, coh = PE.field_angles(q_img)
    frames1 = PE.node_frames(q.subgraph, ang, coh)
    fields = PE.field_multiscale(q_img)
    frames_ms, wins = PE.node_frames_multiscale(q.subgraph, fields)
    out["frame_single"] = len(frames1) / max(1, out["nodes_q"])
    out["frame_multiscale"] = len(frames_ms) / max(1, out["nodes_q"])
    out["window_hist"] = {str(w): int(sum(1 for v in wins.values() if v == w))
                          for w in sorted(set(wins.values()))}
    cohs = [float(coh[int(round(q.subgraph.anchor_of_node[n][1])),
                      int(round(q.subgraph.anchor_of_node[n][0]))])
            for n in q.subgraph.nodes if n in q.subgraph.anchor_of_node]
    out["coherence_at_anchor_median"] = float(np.median(cohs)) if cohs else float("nan")
    out["coherence_below_bar"] = (float(np.mean([1.0 if c < PE.FRAME_COHERENCE_MIN else 0.0
                                                for c in cohs])) if cohs else float("nan"))
    # 2. the payload link
    side_q = PM.build_side(q, q_img)
    side_r = PM.build_side(r, r_img)
    deg = {}
    for (u, v, _et) in side_q.index.keys():
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    out["payload_edges_q"] = len(side_q.index)
    out["nodes_with_edge"] = len(deg) / max(1, out["nodes_q"])
    out["incident_mean"] = float(np.mean(list(deg.values()))) if deg else 0.0
    sp = [v for v in side_q.spacing_by_node.values() if v > 0.0]
    out["spacing_median"] = float(np.median(sp)) if sp else float("nan")
    out["places"] = len(sp)
    # 3. the mapping link, and 4. the judged link
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    t0 = time.time()
    res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                    placement_images=(q_img, r_img))
    out["match_s"] = time.time() - t0
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    out["mapped"] = len(res.mapping)
    out["judged"] = res.placement_judged
    out["share_judged"] = res.placement_share_judged
    out["mapped_share"] = len(res.mapping) / max(1, out["nodes_q"])
    # edge visibility: payload edges of q whose both endpoints are mapped AND whose
    # target edge exists on r (measured, not assumed)
    m = res.mapping
    seen = 0
    for (u, v, _et2) in side_q.index.keys():
        if u in m and v in m:
            seen += 1
    out["edges_both_mapped"] = seen / max(1, len(side_q.index))
    return out


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    ps = pairs()
    jobs = [(q["path"], r["path"]) for q, r in ps]
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_one, jobs):
            rows.append(r)
            print("  %-28s frame1 %.2f framems %.2f edges %.2f judged %.3f  %.0fs"
                  % (os.path.basename(r["q"]), r["frame_single"],
                     r["frame_multiscale"], r["nodes_with_edge"],
                     r["share_judged"], time.time() - t0), flush=True)

    def agg(key):
        vals = [r[key] for r in rows if r.get(key) == r.get(key)]
        return float(np.mean(vals)) if vals else float("nan")

    res = {"pairs": len(rows), "seconds": time.time() - t0,
           "mean": {k: agg(k) for k in ("frame_single", "frame_multiscale",
                                        "coherence_at_anchor_median",
                                        "coherence_below_bar", "nodes_with_edge",
                                        "incident_mean", "spacing_median",
                                        "mapped_share", "share_judged",
                                        "edges_both_mapped", "match_s")},
           "per_pair": rows,
           "constants": {"FRAME_COHERENCE_MIN": None, "FIELD_WINDOWS": None}}
    from matching import placement_edges as PE
    res["constants"]["FRAME_COHERENCE_MIN"] = PE.FRAME_COHERENCE_MIN
    res["constants"]["FIELD_WINDOWS"] = list(PE.FIELD_WINDOWS)
    res["constants"]["TARGET_K"] = PE.TARGET_K
    res["constants"]["TARGET_RADIUS_SCALES"] = PE.TARGET_RADIUS_SCALES
    path = os.path.join(OUT, "m11_coverage.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res["mean"], indent=1, sort_keys=True))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
