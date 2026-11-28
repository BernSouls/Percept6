"""MISSION 11 STEP 4 -- ETH-80 view-to-view retrieval, FRESH queries, both galleries.

FRESH means: every (instance, view) query here was NOT used by Mission 10, whose
query views were -000-000, -022-000, -022-090 and -022-180.  This read uses
-022-270, -035-045, -045-180 and -090-045 -- four views per instance that no earlier
placement read has scored.  The gallery is unchanged in kind: one reference image
(`corpus.REF_VIEWS` = -090-000) per object instance.

Two galleries, one shortlist each, computed from node similarity only (the shipped
front end's own quantity, no aligner):

    G1  same-class K = 8   the eight instances of the query's own class.  K = 8 is
                           that whole gallery, so recall@8 = 1.0 BY CONSTRUCTION
                           (stated, not claimed) and chance is 1/8.
    G2  all-class K = 16   top 16 of the 64 references of all eight classes; chance
                           1/16 given the truth is shortlisted, and the recall is
                           measured and reported because it need not be 1.0.

Arms, on the same pairs (one row per (query, reference) serves both galleries):

    A  placement matcher off, JointMCS.score
    B  placement matcher on, JointMCS.score_shipped  (mapping only)
    C  placement matcher on, score_shipped + w * E   (w frozen from FIT)

Primary contrast: C - A, paired per query, clustered by INSTANCE.

BUDGET, stated up front because it shapes the run: a real image pair costs ~14 s to
match with the shipped aligner (Mission 10 measured 13.4 s off / 15.2 s on, ~11 s to
build).  200 queries x 8 candidates x 2 arms is ~12 CPU-hours; the all-class gallery
adds roughly as much again.  So the run is staged and checkpointed per pair: G1
first for all 200 queries (a complete primary read exists the moment it finishes),
then the extra G2 pairs.  Every row is kept, so nothing is recomputed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PROGRESS = os.path.join(OUT, "m11_eth80_progress.jsonl")
SHORTLIST = os.path.join(OUT, "m11_eth80_shortlist.json")
RESULT = os.path.join(OUT, "m11_eth80.json")
IMG = 256
K1, K2 = 8, 16
N_QUERIES = 200
SUBSAMPLE_Q = 60
SUBSAMPLE_R = 120
RANKS = (0, 1, 2, 3, 4, 5, 6, 7)

#: Views Mission 10 did NOT query.  Chosen before this run and never changed.
FRESH_VIEWS = ("-022-270", "-035-045", "-045-180", "-090-045")


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def catalog() -> Tuple[List[dict], List[dict]]:
    """(64 references at the reference view, 200 fresh queries), deterministic."""
    from Topology import corpus
    refs: List[dict] = []
    queries: List[dict] = []
    for cls in corpus.CLASSES:
        insts = corpus._instances(cls)
        for inst in sorted(insts):
            rank = sorted(insts).index(inst)
            if rank not in RANKS:
                continue
            for view in sorted(insts[inst]):
                e = {"key": "%s/%d" % (cls, inst), "cls": cls,
                     "inst": "%s%d" % (cls, inst), "view": view,
                     "path": insts[inst][view]}
                if view in corpus.REF_VIEWS:
                    refs.append(e)
                elif view in FRESH_VIEWS:
                    queries.append(e)
    refs.sort(key=lambda e: (e["cls"], e["key"]))
    queries.sort(key=lambda e: (e["cls"], e["key"], e["view"]))
    # [M11] The public mirror this sandbox fetched carries ONE truncated file
    # (tomato4-068-000.png; 3279 of 3280 images load).  Validating here and dropping
    # the unreadable ones is deterministic and keeps a partial read from dying
    # half-way through 12,800 pairs.
    refs = [e for e in refs if _readable(e["path"])]
    queries = [e for e in queries if _readable(e["path"])]
    return refs, queries[:N_QUERIES]


def _readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def _qkey(e: dict) -> str:
    return e["key"] + "|" + e["view"]


def _rkey(e: dict) -> str:
    return e["key"] + "|" + e["view"]


def _sub(nodes: List[int], n: int) -> List[int]:
    """Deterministic subsample: evenly spaced by node id, never random."""
    if len(nodes) <= n:
        return nodes
    step = len(nodes) / float(n)
    return [nodes[int(i * step)] for i in range(n)]


def _sim_job(args: Tuple[str, str]) -> Tuple[str, str, float]:
    qpath, rpath = args
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    try:
        q = get_or_build(_load_u8(qpath), "eth|" + qpath)
        r = get_or_build(_load_u8(rpath), "eth|" + rpath)
    except Exception:
        # One image in the public mirror is truncated.  A pair that cannot be read
        # scores -1 and is reported, so a partial mirror cannot silently kill a
        # 12,800-pair shortlist half-way through.
        return qpath, rpath, -1.0
    qn = _sub(sorted(int(n) for n in q.subgraph.nodes), SUBSAMPLE_Q)
    rn = _sub(sorted(int(n) for n in r.subgraph.nodes), SUBSAMPLE_R)
    rows = similarity_rows(q, r, qn, rn)
    best = [row[0][1] for row in rows.values() if row]
    return qpath, rpath, float(np.sum(best))


def stage_shortlist(workers: int) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    refs, queries = catalog()
    if os.path.exists(SHORTLIST):
        with open(SHORTLIST) as fh:
            return json.load(fh)
    jobs = [(e["path"], r["path"]) for e in queries for r in refs]
    print("shortlist: %d pairs" % len(jobs), flush=True)
    t0 = time.time()
    scores: Dict[str, Dict[str, float]] = {}
    by_path = {r["path"]: r for r in refs}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (qp, rp, sc) in enumerate(ex.map(_sim_job, jobs, chunksize=8)):
            scores.setdefault(qp, {})[rp] = sc
            if (i + 1) % 200 == 0:
                print("  %d/%d %.0fs" % (i + 1, len(jobs), time.time() - t0), flush=True)
    out = {"per_query": {}, "seconds": time.time() - t0}
    for e in queries:
        d = scores.get(e["path"], {})
        order = sorted(d.items(), key=lambda t: (-t[1], t[0]))
        own = [r for r in refs if r["key"] == e["key"]][0]
        rank_all = [p for p, _ in order].index(own["path"]) + 1
        same = [r for r in refs if r["cls"] == e["cls"]]
        same.sort(key=lambda r: (-d.get(r["path"], 0.0), r["path"]))
        out["per_query"][_qkey(e)] = {
            "cls": e["cls"], "inst": e["inst"], "truth": _rkey(own),
            "rank_all": rank_all,
            "g1": [_rkey(r) for r in same[:K1]],
            "g2": [_rkey(by_path[p]) for p, _ in order[:K2]],
            "node_only": {_rkey(by_path[p]): v for p, v in order[:max(K1, K2)]},
        }
    out["queries"] = len(out["per_query"])
    out["recall_g1"] = float(np.mean([1.0 if v["truth"] in v["g1"] else 0.0
                                      for v in out["per_query"].values()]))
    out["recall_g2"] = float(np.mean([1.0 if v["truth"] in v["g2"] else 0.0
                                      for v in out["per_query"].values()]))
    with open(SHORTLIST, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print("SHORTLIST recall@8(same-class) %.3f  recall@16(all-class) %.3f  %.0fs"
          % (out["recall_g1"], out["recall_g2"], out["seconds"]), flush=True)
    return out


def stage_build(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    refs, queries = catalog()
    # the union of the reference paths the two galleries can need is all 64 refs
    paths = sorted({r["path"] for r in refs} | {e["path"] for e in queries})
    done_marker = os.path.join(OUT, "m11_eth80_build_done")
    if os.path.exists(done_marker):
        print("build already done", flush=True)
        return
    t0 = time.time()

    def _one(p):
        import guard
        guard.activate()
        from Topology.placement_cache import get_or_build
        get_or_build(_load_u8(p), "eth|" + p)
        return p

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(_one, paths, chunksize=4)):
            if (i + 1) % 20 == 0:
                print("  built %d/%d %.0fs" % (i + 1, len(paths), time.time() - t0),
                      flush=True)
    with open(done_marker, "w") as fh:
        fh.write(str(len(paths)))
    print("ETH80_BUILD_DONE %d graphs %.0fs" % (len(paths), time.time() - t0), flush=True)


def _pair_job(args: Tuple[str, str, str]) -> Dict[str, Any]:
    qpath, rpath, tag = args
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        w = float(json.load(fh)["w"])
    q_img = _load_u8(qpath)
    r_img = _load_u8(rpath)
    q = get_or_build(q_img, "eth|" + qpath)
    r = get_or_build(r_img, "eth|" + rpath)

    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_MATCHER_PRESENCE = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    ta = time.time()
    a = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs)
    sa = time.time() - ta

    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = w
    tc = time.time()
    c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                  placement_images=(q_img, r_img))
    sc_ = time.time() - tc
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    return {"q": qpath, "r": rpath, "tag": tag,
            "A": float(a.score), "B": float(c.score_shipped), "C": float(c.score),
            "E": float(c.placement_evidence), "judged": int(c.placement_judged),
            "mapped": len(c.mapping), "share": float(c.placement_share_judged),
            "sec_a": sa, "sec_c": sc_}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("shortlist", "build", "read", "all"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gallery", default="g1", choices=("g1", "g2", "both"))
    args = ap.parse_args()

    import guard
    guard.activate()
    t_all = time.time()
    if args.stage == "build":
        stage_build(args.workers)
        return 0
    if args.stage in ("shortlist", "all"):
        short = stage_shortlist(args.workers)
    else:
        with open(SHORTLIST) as fh:
            short = json.load(fh)
    if args.stage == "shortlist":
        return 0

    refs, queries = catalog()
    by_qkey = {_qkey(e): e for e in queries}
    ref_path = {_rkey(r): r["path"] for r in refs}
    want: Dict[str, List[str]] = {}
    for qk, e in by_qkey.items():
        rec = short["per_query"][qk]
        gals = []
        if gallery_wanted(args.gallery, "g1"):
            gals.append("g1")
        if gallery_wanted(args.gallery, "g2"):
            gals.append("g2")
        keys = []
        for g in gals:                      # G1 first, so a complete primary read
            for rk in rec[g]:
                keys.append(rk)             # exists before G2 starts
        want[qk] = keys
    jobs = []
    seen = set()
    for qk, keys in want.items():
        for rk in keys:
            k = (qk, rk)
            if k in seen:
                continue
            seen.add(k)
            jobs.append((by_qkey[qk]["path"], ref_path[rk],
                         "g1" if rk in short["per_query"][qk]["g1"] else "g2"))
    done: List[Dict[str, Any]] = []
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    done.append(json.loads(line))
                except Exception:
                    continue
    seen_pairs = {(d["q"], d["r"]) for d in done}
    todo = [j for j in jobs if (j[0], j[1]) not in seen_pairs]
    print("read: %d of %d pairs done, %d to go" % (len(done), len(jobs), len(todo)),
          flush=True)
    t0 = time.time()
    if todo:
        with open(PROGRESS, "a") as fh, ProcessPoolExecutor(
                max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_pair_job, todo, chunksize=1)):
                done.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i + 1) % 10 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(todo), time.time() - t0),
                          flush=True)
    return 0


def gallery_wanted(g: str, name: str) -> bool:
    return g == "both" or g == name


if __name__ == "__main__":
    raise SystemExit(main())
