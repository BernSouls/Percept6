"""MISSION 11 STEP 3 -- the synthetic retrieval read (one read, pre-registered).

Stages, in order, all checkpointed:

    prewarm    render the 24 gallery scenes and the 32 query views once
    shortlist  one shortlist for every arm: top K by node-similarity sum
    read       arms A (placement off), B (placement on, shipped score) and
               C (placement on, score + w*E) on that same shortlist

Registered in `Topology/results/placement/m11_prereg_synth.json` BEFORE any of these
graphs existed (K rule, w, arms, primary contrast, interval, MDE).  Nothing here
tunes anything: w is read from `matching/placement_score_params.json`.
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
IMAGES = os.path.join(ROOT, ".cache", "m11_synth_images.npz")
SHORT_PROGRESS = os.path.join(OUT, "m11_synth_shortlist.json")
READ_PROGRESS = os.path.join(OUT, "m11_synth_read.jsonl")
RESULT = os.path.join(OUT, "m11_synth.json")


def _scene_of(name: str):
    from Topology import placement_m11_families as F
    return {s.name: s for s in F.gallery()}[name]


def _image_key(qkey: str) -> str:
    return qkey


def prewarm() -> None:
    if os.path.exists(IMAGES):
        return
    import guard
    guard.activate()
    from Topology import placement_m11_families as F
    out: Dict[str, Any] = {}
    for spec in F.gallery():
        out[spec.name + "|ref"] = spec.scene.render().image_u8
        for lab, view in F.query_steps(spec):
            sc, _pv = F.pose_fn(spec, view)
            out["%s|%s" % (spec.name, lab)] = sc.render().image_u8
    np.savez_compressed(IMAGES, **out)
    print("prewarmed %d images" % len(out), flush=True)


def _shortlist_job(args: Tuple[str, str]) -> Tuple[str, str, float]:
    qkey, rname = args
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    from Topology.placement_census import similarity_rows
    with np.load(IMAGES) as z:
        q_img = np.array(z[qkey])
        r_img = np.array(z[rname + "|ref"])
    q = get_or_build(q_img, qkey)
    r = get_or_build(r_img, rname + "|ref")
    rows = similarity_rows(q, r, sorted(int(n) for n in q.subgraph.nodes),
                           sorted(int(n) for n in r.subgraph.nodes))
    best = [row[0][1] for row in rows.values() if row]
    return qkey, rname, float(np.sum(best))


def stage_shortlist(workers: int) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    from Topology import placement_m11_families as F
    names = [s.name for s in F.gallery()]
    qkeys = ["%s|%s" % (s.name, lab) for s in F.gallery() if s.family == "IRREGULAR"
             for lab, _v in F.query_steps(s)]
    jobs = [(qk, rn) for qk in qkeys for rn in names]
    scores: Dict[str, Dict[str, float]] = {}
    if os.path.exists(SHORT_PROGRESS):
        with open(SHORT_PROGRESS) as fh:
            scores = json.load(fh)["scores"]
    todo = [j for j in jobs if str(j[1]) not in scores.get(j[0], {})]
    print("shortlist: %d of %d pairs to compute" % (len(todo), len(jobs)), flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (qk, rn, sc) in enumerate(ex.map(_shortlist_job, todo, chunksize=8)):
                scores.setdefault(qk, {})[rn] = sc
                if (i + 1) % 100 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(todo), time.time() - t0),
                          flush=True)
    with open(SHORT_PROGRESS, "w") as fh:
        json.dump({"scores": scores, "seconds": time.time() - t0}, fh)
    return {"scores": scores, "seconds": time.time() - t0}


def recall_at(scores: Dict[str, Dict[str, float]], K: int) -> Dict[str, Any]:
    ranks = {}
    for qk, d in scores.items():
        order = sorted(d.items(), key=lambda t: (-t[1], t[0]))
        names = [n for n, _ in order]
        ranks[qk] = names.index(qk.split("|")[0]) + 1
    by_step: Dict[str, List[float]] = {}
    for qk, r in ranks.items():
        bytep = qk.split("|")[1]
        by_step.setdefault(bytep, []).append(1.0 if r <= K else 0.0)
    return {"K": K,
            "recall": float(np.mean([1.0 if r <= K else 0.0 for r in ranks.values()])),
            "by_step": {k: float(np.mean(v)) for k, v in sorted(by_step.items())},
            "ranks": ranks}


def _read_job(args: Tuple[str, str]) -> Dict[str, Any]:
    qkey, rname = args
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        w = float(json.load(fh)["w"])
    with np.load(IMAGES) as z:
        q_img = np.array(z[qkey])
        r_img = np.array(z[rname + "|ref"])
    q = get_or_build(q_img, qkey)
    r = get_or_build(r_img, rname + "|ref")

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
    return {"q": qkey, "truth": qkey.split("|")[0], "cand": rname,
            "A": float(a.score), "B": float(c.score_shipped), "C": float(c.score),
            "E": float(c.placement_evidence), "judged": int(c.placement_judged),
            "share": float(c.placement_share_judged), "mapped": len(c.mapping),
            "sec_a": sa, "sec_c": sc_}


def rank_of(cands: List[Dict[str, Any]], arm: str, truth: str) -> int:
    order = sorted(cands, key=lambda d: (-d[arm], d["cand"]))
    return [d["cand"] for d in order].index(truth) + 1


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("prewarm", "shortlist", "read", "all"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import guard
    guard.activate()
    with open(os.path.join(OUT, "m11_prereg_synth.json")) as fh:
        reg = json.load(fh)
    K = int(reg["shortlist"]["K"])
    t_all = time.time()

    if args.stage in ("prewarm", "all"):
        prewarm()
    short = None
    if args.stage in ("shortlist", "all"):
        short = stage_shortlist(args.workers)
        rec8 = recall_at(short["scores"], 8)
        print("SHORTLIST recall@8 = %.3f  %s" %
              (rec8["recall"], json.dumps(rec8["by_step"], sort_keys=True)), flush=True)
        if rec8["recall"] < 0.95:
            K = 16
            print("fallback rule triggered: K = 16", flush=True)
            rec16 = recall_at(short["scores"], 16)
            print("SHORTLIST recall@16 = %.3f" % rec16["recall"], flush=True)
        if args.stage == "shortlist":
            return 0
    if short is None:
        with open(SHORT_PROGRESS) as fh:
            short = json.load(fh)

    # one shortlist for every arm
    picks: Dict[str, List[str]] = {}
    for qk, d in short["scores"].items():
        order = sorted(d.items(), key=lambda t: (-t[1], t[0]))
        picks[qk] = [n for n, _ in order[:K]]
    jobs = [(qk, rn) for qk, rns in sorted(picks.items()) for rn in rns]
    done: List[Dict[str, Any]] = []
    if os.path.exists(READ_PROGRESS):
        with open(READ_PROGRESS) as fh:
            for line in fh:
                try:
                    done.append(json.loads(line))
                except Exception:
                    continue
    seen = {(d["q"], d["cand"]) for d in done}
    todo = [j for j in jobs if j not in seen]
    print("read: %d of %d pairs done, %d to go (K=%d)" %
          (len(done), len(jobs), len(todo), K), flush=True)
    t0 = time.time()
    if todo:
        with open(READ_PROGRESS, "a") as fh, ProcessPoolExecutor(
                max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_read_job, todo, chunksize=1)):
                done.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i + 1) % 20 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(todo), time.time() - t0),
                          flush=True)

    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for d in done:
        by_q.setdefault(d["q"], []).append(d)
    res: Dict[str, Any] = {"K": K, "queries": len(by_q), "pairs": len(done),
                           "seconds": time.time() - t_all, "arms": {}}
    for arm in ("A", "B", "C"):
        t1 = []
        rr = []
        margin = []
        for qk, cands in sorted(by_q.items()):
            truth = cands[0]["truth"]
            r = rank_of(cands, arm, truth)
            t1.append(1.0 if r == 1 else 0.0)
            rr.append(1.0 / r)
            ts = [d[arm] for d in cands if d["cand"] == truth][0]
            ws = max([d[arm] for d in cands if d["cand"] != truth] or [ts])
            margin.append(ts - ws)
        res["arms"][arm] = {"top1": float(np.mean(t1)), "mrr": float(np.mean(rr)),
                            "margin_mean": float(np.mean(margin)),
                            "n": len(t1)}
        by_step: Dict[str, List[float]] = {}
        for qk, cands in by_q.items():
            by_step.setdefault(qk.split("|")[1], []).append(
                1.0 if rank_of(cands, arm, cands[0]["truth"]) == 1 else 0.0)
        res["arms"][arm]["top1_by_step"] = {k: float(np.mean(v))
                                            for k, v in sorted(by_step.items())}
    # scene-clustered paired contrasts
    rng = np.random.default_rng(0)
    for tag, x, y in (("C-A", "C", "A"), ("B-A", "B", "A"), ("C-B", "C", "B")):
        per_scene: Dict[str, List[float]] = {}
        per_scene_m: Dict[str, List[float]] = {}
        for qk, cands in by_q.items():
            scene = qk.split("|")[0]
            truth = cands[0]["truth"]
            per_scene.setdefault(scene, []).append(
                (1.0 if rank_of(cands, x, truth) == 1 else 0.0)
                - (1.0 if rank_of(cands, y, truth) == 1 else 0.0))
            mx = [d[x] for d in cands if d["cand"] == truth][0] - max(
                [d[x] for d in cands if d["cand"] != truth] or [0.0])
            my = [d[y] for d in cands if d["cand"] == truth][0] - max(
                [d[y] for d in cands if d["cand"] != truth] or [0.0])
            per_scene_m.setdefault(scene, []).append(mx - my)
        d = np.array([np.mean(v) for v in per_scene.values()])
        dm = np.array([np.mean(v) for v in per_scene_m.values()])
        bs = [float(np.mean(rng.choice(d, size=len(d), replace=True)))
              for _ in range(4000)]
        bsm = [float(np.mean(rng.choice(dm, size=len(dm), replace=True)))
               for _ in range(4000)]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        sem = float(dm.std(ddof=1) / np.sqrt(len(dm)))
        res.setdefault("contrasts", {})[tag] = {
            "delta": float(d.mean()), "se": se, "mde": float(2.8 * se),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "scenes": int(len(d)),
            "margin_delta": float(dm.mean()), "margin_se": sem,
            "margin_ci": [float(np.percentile(bsm, 2.5)),
                          float(np.percentile(bsm, 97.5))]}
    M = [d for d in done]
    res["coverage"] = {
        "judged_mean": float(np.mean([d["judged"] for d in M])),
        "mapped_mean": float(np.mean([d["mapped"] for d in M])),
        "share_mean": float(np.mean([d["share"] for d in M])),
        "evidence_mean": float(np.mean([d["E"] for d in M])),
        "sec_a": float(np.mean([d["sec_a"] for d in M])),
        "sec_c": float(np.mean([d["sec_c"] for d in M])),
    }
    res["seconds_total"] = time.time() - t_all
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    for arm, d in sorted(res["arms"].items()):
        print("%s top1 %.3f mrr %.3f margin %+.4f  %s" % (
            arm, d["top1"], d["mrr"], d["margin_mean"],
            json.dumps(d["top1_by_step"], sort_keys=True)))
    for tag, c in sorted(res.get("contrasts", {}).items()):
        print("%s delta %+.3f se %.3f mde %.3f ci [%+.3f,%+.3f]  margin %+.4f (se %.4f)"
              % (tag, c["delta"], c["se"], c["mde"], c["ci"][0], c["ci"][1],
                 c["margin_delta"], c["margin_se"]))
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
