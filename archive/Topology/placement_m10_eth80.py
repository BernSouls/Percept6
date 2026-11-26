"""MISSION 10 STEP 4b -- ETH-80 view-to-view retrieval, 200 queries, two arms.

BUDGET FIRST, because it decides the design:
  * a real image pair costs ~11 s to build and ~13 s to match with the shipped
    aligner, plus ~2 s for the placement re-scoring (measured on a held-out pair:
    build 22.2 s for two images, match 13.4 s flag off, 15.2 s flag on);
  * 200 queries x K = 8 is 1600 placement-arm matches -- about 3.4 h on two
    cores, which is what this run costs.

DESIGN (fixed here, before the run):
  gallery     one reference image per object: 8 classes x 3 reference instances
              (REF_RANKS) at REF_VIEWS -- 24 references;
  queries     200 images: 6 non-reference instances per class (DEV_RANKS and
              HELDOUT_RANKS) at as many of their views as needed, never at a
              reference view.  Deterministic order, first 200 kept;
  shortlist   ONE for both arms: top K = 8 by node similarity, computed on a
              FIXED, deterministic subsample (120 query nodes x 200 reference
              nodes, by node id) so that 200 x 24 pairs are affordable.  The
              subsample is the same for both arms and for every candidate;
  node-only   sum of the best per-node similarities over the query's nodes;
  placement   sum over the pairs the deployed matcher MAPPED of the blended score
              it used.  Same frozen bundle, same caps.

Reported: shortlist recall@8 (the rule first), top-1 and MRR overall and
conditional on the truth being shortlisted, coverage (share of query nodes whose
place attached), nodes the flag moved, build and match time, and a
scene-clustered interval and MDE for the placement-minus-node contrast -- a scene
here is an object instance.

Checkpointed per query in ``m10_eth80_progress.jsonl``.
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
PROGRESS = os.path.join(OUT, "m10_eth80_progress.jsonl")
IMG = 256
K = 8
N_QUERIES = 200
SUBSAMPLE_Q = 60
SUBSAMPLE_R = 120


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


#: [MISSION 10 / measured] The gallery is the EIGHT instances of the query's own
#: class.  Measured on the full-80 gallery the shortlist is weak -- recall@8 =
#: 0.235, recall@64 = 0.93 -- and on a same-class 10-instance gallery recall@8 is
#: 0.78.  The rule says shrink the gallery rather than judge below 0.95, so the
#: gallery is eight same-class references at the reference view and K = 8 IS that
#: gallery: recall@8 = 1.0 by construction, stated rather than claimed as a
#: measurement.  The task is then within-class instance identification under a
#: view change, which is the ETH-80 question that has a right answer.
RANKS = (0, 1, 2, 3, 4, 5, 6, 7)


def catalog() -> Tuple[List[dict], List[dict]]:
    """(gallery references, queries), deterministic, via the corpus's own parser.

    [MISSION 10 / corrected] The first version took only the three REFERENCE
    instances into the gallery and drew queries from the dev/held-out instances --
    which are DIFFERENT OBJECTS, so the "truth" could never be in the shortlist and
    recall came out 0.000.  View-to-view retrieval asks whether a second view of an
    object finds that object, so the gallery is one reference view of EVERY
    instance and the queries are other views of the SAME instances.
    """
    from Topology import corpus
    refs: List[dict] = []
    queries: List[dict] = []
    for cls in corpus.CLASSES:
        insts = corpus._instances(cls)
        for inst in sorted(insts):
            for view in sorted(insts[inst]):
                e = {"key": "%s/%d" % (cls, inst), "cls": cls,
                     "inst": "%s%d" % (cls, inst), "view": view,
                     "path": insts[inst][view]}
                if view in corpus.REF_VIEWS and RANKS is not None:
                    rank = sorted(insts).index(inst)
                    if rank in RANKS:
                        refs.append(e)
                else:
                    rank = sorted(insts).index(inst)
                    if rank in RANKS:
                        queries.append(e)
    refs.sort(key=lambda e: (e["cls"], e["key"]))
    queries.sort(key=lambda e: (e["cls"], e["key"], e["view"]))
    out: List[dict] = []
    per_inst: Dict[str, int] = {}
    for e in queries:
        if len(out) >= N_QUERIES:
            break
        c = per_inst.get(e["key"], 0)
        if c >= 4:                 # at most four query views per instance
            continue
        per_inst[e["key"]] = c + 1
        out.append(e)
    return refs, out


def build(refs: List[dict], queries: List[dict]) -> None:
    from Topology.placement_cache import get_or_build
    t0 = time.time()
    n = 0
    for e in refs:
        get_or_build(_load_u8(e["path"]), "eth|" + e["key"] + "|" + e["view"])
        n += 1
        if n % 4 == 0:
            print("  refs %d/%d %.0fs" % (n, len(refs), time.time() - t0), flush=True)
    for e in queries:
        get_or_build(_load_u8(e["path"]), "eth|" + e["path"])
        n += 1
        if n % 20 == 0:
            print("  total %d/%d %.0fs" % (n, len(refs) + len(queries),
                                           time.time() - t0), flush=True)
    print("ETH80_BUILD_DONE %d graphs %.0fs" % (n, time.time() - t0), flush=True)


def _qkey(e: dict) -> str:
    return "eth|" + e["path"]


def _rkey(e: dict) -> str:
    return "eth|" + e["key"] + "|" + e["view"]


def _shortlist_one(job: Tuple[dict, List[dict]]) -> Dict[str, Any]:
    q, refs = job
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    u8 = _load_u8(q["path"])
    gq = get_or_build(u8, _qkey(q))
    nodes_q = sorted(int(n) for n in gq.subgraph.nodes)[:SUBSAMPLE_Q]
    scores = {}
    for r in refs:
        u8r = _load_u8(r["path"])
        gr = get_or_build(u8r, _rkey(r))
        nodes_r = sorted(int(n) for n in gr.subgraph.nodes)[:SUBSAMPLE_R]
        rows = similarity_rows(gq, gr, nodes_q, nodes_r)
        best = [row[0][1] for row in rows.values() if row]
        scores[r["key"]] = float(np.sum(best))
    order = sorted(scores.items(), key=lambda t: (-t[1], t[0]))
    return {"query": q["key"] + "|" + q["view"], "path": q["path"],
            "cls": q["cls"], "inst": q["inst"], "truth": q["key"],
            "node_only": dict(scores), "top": [k for k, _ in order[:K]]}


def shortlist(workers: int = 1) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    refs, queries = catalog()
    by_cls: Dict[str, List[dict]] = {}
    for r in refs:
        by_cls.setdefault(r["cls"], []).append(r)
    jobs = [(q, by_cls[q["cls"]]) for q in queries]
    t0 = time.time()
    out: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_shortlist_one, jobs):
            out.append(r)
            if len(out) % 10 == 0:
                print("  shortlist %d/%d %.0fs" % (len(out), len(jobs),
                                                   time.time() - t0), flush=True)
    rec = float(np.mean([1.0 if r["truth"] in r["top"] else 0.0 for r in out]))
    res = {"K": K, "gallery": len(refs), "queries": len(out),
           "recall_at_K": rec, "seconds": time.time() - t0,
           "per_query": {r["query"]: r for r in out}}
    with open(os.path.join(OUT, "m10_eth80_shortlist.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("ETH80 shortlist recall@%d = %.3f over %d queries (%.0fs)"
          % (K, rec, len(out), res["seconds"]), flush=True)
    return res


def _match_one(args: Tuple[str, str, str]) -> Dict[str, Any]:
    qpath, rpath, rkey = args
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_llr as L
    from matching import placement_matcher as PM
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows

    u8q, u8r = _load_u8(qpath), _load_u8(rpath)
    gq = get_or_build(u8q, "eth|" + qpath)
    gr = get_or_build(u8r, "eth|" + rpath)
    side_q = PM.build_side(gq, u8q)
    side_r = PM.build_side(gr, u8r)
    b = PM.frozen()
    fit, pfit = PM.fit_objects()
    a_e, a_g, beta = float(b["_a_exist"]), float(b["_a_geom"]), float(b["beta"])
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_MATCHER_PRESENCE = False
    t = time.time()
    res = joint_mcs(gq.subgraph, gr.subgraph, gq.sigs, gr.sigs,
                    placement_images=(u8q, u8r))
    dt = time.time() - t
    counters = dict(PM.FAILURES)
    P.PLACEMENT_MATCHER_ENABLED = False
    mapping = {int(k): int(v) for k, v in res.mapping.items()}
    nodes_q = sorted(int(n) for n in gq.subgraph.nodes)
    # [MISSION 10 / cost] Only the MAPPED pairs need a similarity, and the mapping
    # is one-to-one, so the similarity is one call per pair instead of a full
    # |q| x |r| matrix: the same number, ~1.7 s cheaper per candidate pair.
    total = 0.0
    for u, v in mapping.items():
        su, sv = gq.sigs.get(u), gr.sigs.get(v)
        if su is None or sv is None or su[0] != sv[0]:
            continue
        sim = float(su[1].similarity(sv[1]))
        term = L.plus_score(u, int(v), mapping, side_q.incident,
                            side_q.index, side_r.index, fit, pfit, a_e, a_g)
        total += (1.0 - beta) * sim + beta * float(term)
    attached = sum(1 for n in nodes_q if side_q.spacing_by_node.get(n, 0.0) > 0.0)
    return {"rkey": rkey, "placement": float(total),
            "mapped": len(mapping), "attach": attached / max(1, len(nodes_q)),
            "match_s": float(dt), "counters": counters}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("build", "shortlist", "match"), required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="m10_eth80")
    args = ap.parse_args()
    if args.stage == "build":
        refs, queries = catalog()
        print("gallery %d, queries %d" % (len(refs), len(queries)))
        build(refs, queries)
        return 0
    if args.stage == "shortlist":
        shortlist(args.workers)
        return 0
    # ---- match: resume from the progress log
    from concurrent.futures import ProcessPoolExecutor
    refs_by_key = {r["key"]: r for r in catalog()[0]}
    with open(os.path.join(OUT, "m10_eth80_shortlist.json")) as fh:
        sl = json.load(fh)
    print("shortlist recall@8 = %.3f" % sl["recall_at_K"])
    done: Dict[str, Any] = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[r["query"]] = r
                except Exception:
                    continue
    todo = [q for q in sl["per_query"] if q not in done]
    print("%d done, %d to go" % (len(done), len(todo)), flush=True)
    t0 = time.time()
    with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for q in todo:
            job = sl["per_query"][q]
            args_list = []
            for rk in job["top"]:
                r = refs_by_key[rk]
                args_list.append((job["path"], r["path"], rk))
            scored: Dict[str, float] = {}
            meta: Dict[str, Any] = {}
            for r in ex.map(_match_one, args_list):
                scored[r["rkey"]] = r["placement"]
                meta[r["rkey"]] = r
            rec = {"query": q, "truth": job["truth"], "cls": job["cls"],
                   "inst": job["inst"], "placement": scored,
                   "node_only": {k: job["node_only"][k] for k in job["top"]},
                   "top": job["top"], "meta": meta}
            done[q] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("  %-40s %5.0fs" % (q, time.time() - t0), flush=True)
    _report(done, sl, args.tag)
    return 0


def _rank(scores: Dict[str, float], truth: str) -> int:
    order = sorted(scores.items(), key=lambda t: (-t[1], t[0]))
    names = [n for n, _ in order]
    return names.index(truth) + 1 if truth in names else 10 ** 6


def _report(done: Dict[str, Any], sl: Dict[str, Any], tag: str) -> None:
    res: Dict[str, Any] = {"queries": len(done), "K": K,
                           "shortlist_recall": sl["recall_at_K"], "arms": {}}
    for arm in ("node_only", "placement"):
        ranks = {q: _rank(r[arm], r["truth"]) for q, r in done.items()}
        cond = {q: k for q, k in ranks.items() if k <= K}
        res["arms"][arm] = {
            "top1": float(np.mean([1.0 if k == 1 else 0.0 for k in ranks.values()])),
            "mrr": float(np.mean([1.0 / k for k in ranks.values()])),
            "top1_conditional": float(np.mean([1.0 if k == 1 else 0.0
                                               for k in cond.values()])) if cond else float("nan"),
            "mrr_conditional": float(np.mean([1.0 / k for k in cond.values()])) if cond else float("nan"),
            "n_conditional": len(cond), "ranks": ranks}
    per_inst: Dict[str, List[float]] = {}
    for q, r in done.items():
        per_inst.setdefault(r["inst"], []).append(
            (1.0 if _rank(r["placement"], r["truth"]) == 1 else 0.0)
            - (1.0 if _rank(r["node_only"], r["truth"]) == 1 else 0.0))
    sm = {k: float(np.mean(v)) for k, v in per_inst.items()}
    arr = np.array(list(sm.values()))
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    d = np.array([np.mean(v) for v in per_inst.values()])
    res["placement_minus_node_top1"] = {
        "mean": float(arr.mean()), "se_scene": se, "mde": 2.8 * se,
        "ci": [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)],
        "n_instances": len(arr), "per_instance": sm}
    meta = [m for r in done.values() for m in r["meta"].values()]
    res["coverage"] = {
        "attach_mean": float(np.mean([m["attach"] for m in meta])) if meta else float("nan"),
        "mapped_mean": float(np.mean([m["mapped"] for m in meta])) if meta else float("nan"),
        "match_s_mean": float(np.mean([m["match_s"] for m in meta])) if meta else float("nan"),
        "rescored_total": int(sum(m["counters"].get("rescored", 0) for m in meta)),
        "raised_total": int(sum(m["counters"].get("raised", 0) for m in meta)),
        "fell_back_total": int(sum(m["counters"].get("fell_back", 0) for m in meta)),
        "skipped_total": int(sum(m["counters"].get("skipped", 0) for m in meta))}
    path = os.path.join(OUT, tag + ".json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    for arm in ("node_only", "placement"):
        a = res["arms"][arm]
        print("%-10s top1 %.3f mrr %.3f | cond top1 %.3f mrr %.3f (n %d)"
              % (arm, a["top1"], a["mrr"], a["top1_conditional"],
                 a["mrr_conditional"], a["n_conditional"]))
    print("placement - node_only top1 %+0.3f  se %.3f  mde %.3f"
          % (res["placement_minus_node_top1"]["mean"],
             res["placement_minus_node_top1"]["se_scene"],
             res["placement_minus_node_top1"]["mde"]))
    print("coverage %s" % json.dumps(res["coverage"]))
    print("wrote %s" % path)


if __name__ == "__main__":
    raise SystemExit(main())
