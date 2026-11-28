"""MISSION 23 -- the closing harness: T2 done properly, the fused rerank, and the one read.

Stages, each resumable and each writing what it measured:

  t2     item 1.  The class score on a gallery with the QUERY'S OWN OBJECT REMOVED (39 of 40),
         drawn by the shipped door, scored by the shipped matcher, aggregated three ways.
         `matching/placement_classscore.py::t2_gallery` is the only place the removal happens and
         `class_scores_t2` raises if a caller forgets it.
  fused  item 3.  Three final orders over the same door shortlist: the door alone, the matcher's
         plain re-order (Mission 22 B), and the rank fusion of the two (RRF, k = 60).  FIT and
         DEV; the choice is made on FIT.
  read   item 5.  All 80 objects, three unseen view pairs per object (240 queries), the gallery
         the 80 references.  Every arm is a shortlist of 8 plus the shipped matcher; the primary
         is assembled vs the deployed prefilter, paired per query, clustered by object.

Two arms are built from ONE graph per image (`image_job`): the deployed stage-1 prefilter needs
`bag_v2` over every subgraph node, the door needs the place rows.  Building both from one `_cell`
is what keeps a 320-image read inside this box.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
S_DEFAULT = 8
VIEWS = ("-022-000", "-035-045", "-045-180")
REF_VIEW = "-000-000"


# --------------------------------------------------------------------------- objects
def all_objects() -> List[Dict[str, Any]]:
    """Every ETH-80 object with its reference and the three registered query views."""
    from Topology.placement_m20_eth80 import objects as m20_objects
    out: Dict[str, Dict[str, Any]] = {}
    for split in ("FIT", "DEV", "FRESH"):
        for o in m20_objects(split):
            base = os.path.dirname(o["ref"])
            label = os.path.basename(o["ref"]).split("-")[0]
            q = {v: os.path.join(base, "%s%s.png" % (label, v)) for v in VIEWS}
            e = dict(o)
            e["views"] = q
            e["split"] = split
            out[o["key"]] = e
    return [out[k] for k in sorted(out)]


def split_objects(split: str) -> List[Dict[str, Any]]:
    return [o for o in all_objects() if o["split"] == split]


# --------------------------------------------------------------------------- one image
def image_job(path: str) -> Dict[str, Any]:
    """One image -> the two things every stage needs, from ONE graph build."""
    import guard
    guard.activate()
    from Topology.placement_m20_eth80 import _cell
    from matching import placement_prefilter as PF
    from matching import placement_edges as PE
    t0 = time.time()
    img, g, side = _cell(path, "m23")
    params = PF.load_params()
    all_nodes = sorted(int(n) for n in g.subgraph.nodes)
    pay = sorted({int(k[0]) for k in side.payload} | {int(k[1]) for k in side.payload})
    bag_all = PF.bag_v2(side, all_nodes, params["K"])
    std_all = PF.standardize(bag_all, params)
    raw = PF.bag_v2(side, pay, params["K"])
    std = PF.standardize(raw, params)
    frames, _sig = PE.node_frames_keypoint(g.subgraph, img)
    sub = g.subgraph
    rows: List[Dict[str, Any]] = []
    dropped = {"no_sig": 0, "no_config": 0, "no_position": 0, "no_scale": 0}
    for r, n in enumerate(raw["ids"]):
        s = g.sigs.get(n)
        if s is None:
            dropped["no_sig"] += 1
            continue
        if float(std["M"][r].sum()) < 2:
            dropped["no_config"] += 1
            continue
        anchor = sub.anchor_of_node.get(n) if hasattr(sub, "anchor_of_node") else None
        if anchor is None:
            dropped["no_position"] += 1
            continue
        sp = side.spacing_by_node.get(n)
        if sp is None or not np.isfinite(sp) or sp <= 0:
            dropped["no_scale"] += 1
            continue
        lens, sig = s
        rows.append({"n": int(n), "lens": str(lens),
                     "key": (str(lens), tuple(float(x) for x in
                                              (getattr(sig, "quantized", ()) or ()))),
                     "Dz": [float(x) for x in std["D"][r]],
                     "Mz": [float(x) for x in std["M"][r]],
                     "x": float(anchor[0]), "y": float(anchor[1]), "s": float(sp),
                     "theta": (None if n not in frames else float(frames[n]))})
    return {"path": path, "places": rows, "dropped": dropped,
            "n_payload": len(pay), "n_all_nodes": len(all_nodes),
            # EVERY channel `standardize` produces, because `score_bags_v2` differences all
            # three of D, C and S; storing only D and M cost one crashed read to learn
            # (KeyError: 'C', 06:37).  float32 has the range and precision these scores use.
            "std_all": {k: np.asarray(std_all[k], dtype=np.float32)
                        for k in ("D", "C", "S", "M")}
            | {"ids": np.asarray(std_all["ids"], dtype=np.int64)},
            "sec": time.time() - t0}


def _image_star(path: str) -> Dict[str, Any]:
    return image_job(path)


def ensure_images(paths: Sequence[str], cache: str, workers: int = 2) -> Dict[str, Any]:
    """Build (or load) the per-image records for `paths`."""
    from concurrent.futures import ProcessPoolExecutor
    p = os.path.join(OUT, "m23_%s_images.pkl" % cache)
    store = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    need = [x for x in sorted(set(paths)) if x not in store]
    if need:
        print("images: %d of %d to build (%s)" % (len(need), len(set(paths)), cache), flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_image_star, need, chunksize=2)):
                store[r["path"]] = r
                if (i + 1) % 12 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(need), time.time() - t0), flush=True)
        with open(p, "wb") as fh:
            pickle.dump(store, fh, protocol=4)
        print("images done %.0fs" % (time.time() - t0), flush=True)
    missing = [x for x in sorted(set(paths)) if x not in store]
    if missing:
        raise SystemExit("image records missing: %d" % len(missing))
    return store


# --------------------------------------------------------------------------- the two shortlists
def deployed_ranking(store: Dict[str, Any], qpath: str, refs: Sequence[str],
                     params: Dict[str, Any]) -> List[int]:
    """The DEPLOYED shortlist: the M21 AFTER stage-1 score over every gallery item."""
    from matching import placement_prefilter as PF
    qa = store[qpath]["std_all"]
    sc = np.zeros(len(refs), dtype=np.float64)
    for j, rp in enumerate(refs):
        ra = store[rp]["std_all"]
        sc[j] = float(PF.score_bags_v2(qa, ra, params)["score"])
    return sorted(range(len(refs)), key=lambda j: (-sc[j], j))


def door_ranking(store: Dict[str, Any], refs: Sequence[str], ref_keys: Sequence[str],
                 query_paths: Sequence[str]) -> List[List[int]]:
    """The SHIPPED door's ranking of every gallery item, fused (counted vote + appearance)."""
    from matching import placement_frontdoor as FD
    index = FD.PlaceIndex([store[r] for r in refs], ref_keys)
    app = FD.appearance_scores(list(query_paths), list(refs))
    cnt, _diag = FD.count_scores(index, [store[p] for p in query_paths],
                                 n_hits=int(_n("PLACEMENT_COUNT_HITS")))
    fused = FD.rank_fuse(app, cnt)
    return [[j for j in sorted(range(len(refs)), key=lambda j: (-fused[i, j], j))]
            for i in range(len(query_paths))]


def _n(name: str) -> int:
    import parameters as P
    return int(getattr(P, name, 3))


# --------------------------------------------------------------------------- pairs
def _pair_star(job: Tuple[str, str]) -> Dict[str, Any]:
    from Topology.placement_m22_eth80 import pair_score_job
    return pair_score_job(job[0], job[1])


def ensure_pairs(jobs: Sequence[Tuple[str, str]], tag: str, workers: int = 2) -> Dict[Tuple[str, str], float]:
    """The shipped matcher's score for each (query, reference), cached in a JSONL."""
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(OUT, "m23_%s_pairs.jsonl" % tag)
    done: Dict[Tuple[str, str], float] = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[(r["q"], r["r"])] = float(r["B"])
    need = [(q, r) for (q, r) in sorted(set(jobs)) if (q, r) not in done]
    print("pairs[%s]: %d to score (%d cached)" % (tag, len(need), len(done)), flush=True)
    if need:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (job, r) in enumerate(zip(need, ex.map(_pair_star, need, chunksize=3))):
                if "error" not in r:
                    done[(r["q"], r["r"])] = float(r["B"])
                with open(path, "a") as fh:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                if (k + 1) % 10 == 0:
                    print("  %d/%d %.0fs (%.1f s/pair)"
                          % (k + 1, len(need), time.time() - t0,
                             (time.time() - t0) / (k + 1)), flush=True)
        print("pairs[%s] done %.0fs" % (tag, time.time() - t0), flush=True)
    return done


# --------------------------------------------------------------------------- final orders
def order_plain(cands: Sequence[int], scores: Dict[int, float]) -> List[int]:
    """Matcher re-order (Mission 22 B): sort the candidates by the shipped score, drop nothing."""
    return sorted(cands, key=lambda j: (-float(scores.get(j, 0.0)), j))


def order_fused(cands: Sequence[int], scores: Dict[int, float], k: int = 60) -> List[int]:
    """RRF of the door's order and the matcher's order over the SAME candidates.

    Both lists are rankings of the same set, so no candidate can be added or dropped by the
    fusion; k is fixed at 60 in advance (no tuned weight).
    """
    door_rank = {j: r for r, j in enumerate(cands)}
    mat_rank = {j: r for r, j in enumerate(order_plain(cands, scores))}
    out = []
    for j in cands:
        out.append((1.0 / (k + door_rank[j] + 1) + 1.0 / (k + mat_rank[j] + 1), j))
    return [j for _s, j in sorted(out, key=lambda t: (-t[0], t[1]))]


# --------------------------------------------------------------------------- statistics
def bootstrap_paired(per_object: Dict[str, Tuple[float, float]], n_boot: int = 10000,
                     seed: int = 20261118) -> Dict[str, Any]:
    """Paired bootstrap over OBJECTS of the mean top-1 difference (assembled - deployed).

    Each object contributes the mean over its queries of (assembled hit - deployed hit), so the
    resampling unit is the object and three queries share one draw.
    """
    keys = sorted(per_object)
    d = np.asarray([per_object[k][0] - per_object[k][1] for k in keys], dtype=np.float64)
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    return {"n_objects": int(n), "diff": obs, "ci95": [float(lo), float(hi)],
            "p_bootstrap": float(min(1.0, max(p, 1.0 / n_boot))),
            "assembled": float(np.mean([per_object[k][0] for k in keys])),
            "deployed": float(np.mean([per_object[k][1] for k in keys]))}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("t2", "fused", "read", "all"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shortlist", type=int, default=S_DEFAULT)
    ap.add_argument("--rerank", default="fused", choices=("plain", "fused"))
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop the read cleanly after this long and summarise what ran")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke runs ONLY: cut the object list to this many")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    if not getattr(P, "PLACEMENT_M19", False):
        raise SystemExit("set PLACEMENT_M19=1: this reads the deployed configuration")
    from matching import m15_manifest as MF
    out: Dict[str, Any] = {}
    if a.stage in ("t2", "all"):
        from Topology.placement_m23_stages import stage_t2
        out["t2"] = stage_t2(a.workers, a.shortlist, a.limit)
        MF.write_result(os.path.join(OUT, "m23_t2.json"), out["t2"])
    if a.stage in ("fused", "all"):
        from Topology.placement_m23_stages import stage_fused
        out["fused"] = stage_fused(a.workers, a.shortlist, a.limit)
        MF.write_result(os.path.join(OUT, "m23_fused.json"), out["fused"])
    if a.stage in ("read", "all"):
        from Topology.placement_m23_stages import stage_read
        out["read"] = stage_read(a.workers, a.shortlist, a.rerank,
                                 tuple(v for v in a.views.split(",") if v), a.limit,
                                 a.max_minutes)
        MF.write_result(os.path.join(OUT, "m23_read.json"), out["read"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
