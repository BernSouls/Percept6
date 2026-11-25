"""MISSION 22 -- the harness for the three builds, and the one evaluation.

Stages (each resumable, each writing what it measured):

  places    one record per image: every payload place's CONTENT KEY (lens + the signature's
            quantised bins), its FROZEN configuration row (Mission-21 ruler), and its FRAME
            (position, local scale, axial orientation).  Built once per split.
  frontdoor the three front doors on a split: appearance, place voting, and their rank fusion.
            Recall@8 and top-1 against the EXACT chance for the gallery (S/N and 1/N -- for a
            recall@S the chance is S/N, not 1/N; see the report's correction of Mission 21).
  assemble  the assembled system: appearance shortlist S = 8 -> the SHIPPED matcher scores those
            8 -> re-order.  Nothing is removed, so recall@8 is the front door's.
  class     the class score (C) from the assembled instance scores, against the exact
            hypergeometric chance.
  all       places + frontdoor + assemble + class.

The gallery is the split's reference views and the queries are its other view (`-022-000`), the
M20/M21 convention, so every number here is comparable with the record.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
S_DEFAULT = 8


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def objects(split: str, per_class: int = 0) -> List[Dict[str, Any]]:
    from Topology.placement_m20_eth80 import objects as m20_objects
    objs = m20_objects(split)
    if per_class:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for o in objs:
            per.setdefault(o["cls"], []).append(o)
        objs = [o for c in sorted(per) for o in per[c][:int(per_class)]]
    return objs


# --------------------------------------------------------------------------- place records
def place_job(path: str) -> Dict[str, Any]:
    """One image -> its pool record.  Content key + frozen configuration row + frame."""
    import guard
    guard.activate()
    from Topology.placement_m20_eth80 import _cell
    from matching import placement_prefilter as PF
    from matching import placement_edges as PE
    t0 = time.time()
    img, g, side = _cell(path, "m22places")
    params = PF.load_params()
    nodes = sorted({int(k[0]) for k in side.payload} | {int(k[1]) for k in side.payload})
    raw = PF.bag_v2(side, nodes, params["K"])
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
                     "x": float(anchor[0]), "y": float(anchor[1]),
                     "s": float(sp),
                     "theta": (None if n not in frames else float(frames[n]))})
    return {"path": path, "places": rows, "dropped": dropped,
            "spacing_median": float(side.spacing_median),
            "n_payload_places": len(nodes), "sec": time.time() - t0}


def pair_score_job(qpath: str, rpath: str) -> Dict[str, Any]:
    """The SHIPPED pair score for one (query, gallery) pair -- and nothing else.

    The Mission-20 ``_pair_job`` also builds the equal-coverage aggregate, the per-place
    attribution and the arm-A control; none of them is part of "assembled", and together they
    roughly double the cost of a pair.  This job sets the SAME deployed flags, calls the SAME
    ``joint_mcs`` entry point and returns the SAME ``score_shipped``, so the assembled ordering
    is the shipped scorer's ordering.  That is checked, not asserted: ``--stage verify`` re-scores a
    deterministic sample of the cached pairs with the full Mission-20 pair job and requires the two
    scores to agree to 1e-12 (`m22_assemble_*_verify.json`).
    """
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_m20_eth80 import _cell, _recipe
    t0 = time.time()
    _recipe()
    q_img, q, _sq = _cell(qpath, "m22pair")
    r_img, r, _sr = _cell(rpath, "m22pair")
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = _shipped_weight()
    P.PLACEMENT_MATCHER_STRICT = True
    try:
        c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs, placement_images=(q_img, r_img))
        b = float(c.score_shipped)
    finally:
        P.PLACEMENT_SCORE_EVIDENCE = False
        P.PLACEMENT_SCORE_WEIGHT = 0.0
        P.PLACEMENT_MATCHER_STRICT = False
    return {"q": qpath, "r": rpath, "B": b, "sec_c": time.time() - t0, "lite": 1}


def _pair_lite(job: Tuple[str, str]) -> Dict[str, Any]:
    """`pair_score_job(q, r)` for `ProcessPoolExecutor.map` (one tuple per job)."""
    return pair_score_job(job[0], job[1])


def _shipped_weight() -> float:
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        return float(json.load(fh)["w"])


_SHIPPED_W = None


def places_path(split: str, per_class: int) -> str:
    return os.path.join(OUT, "m22_%s_pc%d_places.pkl" % (split.lower(), per_class))


def ensure_places(split: str, per_class: int = 0, workers: int = 2) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    p = places_path(split, per_class)
    store = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    want = sorted({o["ref"] for o in objects(split, per_class)} |
                  {o["query"] for o in objects(split, per_class)})
    need = [x for x in want if x not in store]
    if need:
        print("places: building %d of %d images" % (len(need), len(want)), flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(place_job, need, chunksize=1)):
                store[r["path"]] = r
                if (i + 1) % 8 == 0:
                    print("  %d/%d %.0fs" % (i + 1, len(need), time.time() - t0), flush=True)
        with open(p, "wb") as fh:
            pickle.dump(store, fh, protocol=4)
        print("places done %.0fs" % (time.time() - t0), flush=True)
    missing = [x for x in want if x not in store]
    if missing:
        raise SystemExit("places missing: %d" % len(missing))
    return store


# --------------------------------------------------------------------------- tests
def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for b vs c discordant objects (binomial, p = 0.5)."""
    from math import comb
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(comb(n, j) for j in range(0, k + 1)) / float(2 ** n)
    return float(min(1.0, 2.0 * tail))


# --------------------------------------------------------------------------- evaluation
def ranks_metrics(scores: np.ndarray, objs: Sequence[Dict[str, Any]],
                  gallery_keys: Sequence[str], s: int) -> Dict[str, Any]:
    """Recall@s, top-1, per-image rank and per-class rates, with the EXACT chance.

    For recall@s on a gallery of N the chance is s/N: a random shortlist of s of N items
    contains the truth with probability s/N.  The top-1 chance is 1/N.  Mission 21's chance
    column for recall@8 was the top-1 chance; it is corrected here, and the report says so.
    """
    key_idx = {k: j for j, k in enumerate(gallery_keys)}
    N = len(gallery_keys)
    hit8 = hit1 = 0
    ranks: List[int] = []
    by_class: Dict[str, Dict[str, Any]] = {}
    for i, o in enumerate(objs):
        truth = key_idx.get(o["key"])
        if truth is None:
            continue
        order = sorted(range(N), key=lambda j: (-scores[i, j], j))
        r = order.index(truth) + 1
        ranks.append(r)
        hit8 += 1 if r <= s else 0
        hit1 += 1 if r == 1 else 0
        d = by_class.setdefault(o["cls"], {"n": 0, "hit8": 0, "hit1": 0})
        d["n"] += 1
        d["hit8"] += 1 if r <= s else 0
        d["hit1"] += 1 if r == 1 else 0
    n = len(ranks)
    return {"n": n, "gallery": N, "S": s,
            "recall_s": (hit8 / n) if n else None, "chance_recall_s": s / N,
            "top1": (hit1 / n) if n else None, "chance_top1": 1.0 / N,
            "rank_median": float(np.median(ranks)) if ranks else None,
            "ranks": ranks,
            "per_class": {c: {"n": d["n"], "recall_s": d["hit8"] / d["n"],
                              "top1": d["hit1"] / d["n"]} for c, d in by_class.items()}}


def frontdoor_stage(split: str, per_class: int, s: int, workers: int,
                    tag: str = "") -> Dict[str, Any]:
    from matching import placement_frontdoor as FD
    store = ensure_places(split, per_class, workers)
    objs = objects(split, per_class)
    gal = sorted({o["ref"] for o in objs})
    key_of_ref = {o["ref"]: o["key"] for o in objs}
    gal_keys = [key_of_ref[p] for p in gal]
    gal_entries = [store[p] for p in gal]
    q_entries = [store[o["query"]] for o in objs]
    t0 = time.time()
    app = FD.appearance_scores([o["query"] for o in objs], gal)
    t_app = (time.time() - t0) / max(1, len(objs))
    index = FD.PlaceIndex(gal_entries, gal_keys)
    import parameters as _P
    t1 = time.time()
    vote, diag = FD.voting_scores(index, q_entries, None,
                                  n_hits=int(_P.PLACEMENT_VOTE_HITS),
                                  use_rot=bool(_P.PLACEMENT_VOTE_ROT))
    t_vote = (time.time() - t1) / max(1, len(objs))
    t2 = time.time()
    counted, cdiag = FD.count_scores(index, q_entries,
                                     n_hits=int(_P.PLACEMENT_COUNT_HITS))
    t_count = (time.time() - t2) / max(1, len(objs))
    fused = FD.rank_fuse(app, vote)
    fused_count = FD.rank_fuse(app, counted)
    res: Dict[str, Any] = {"split": split, "per_class": per_class, "gallery": len(gal),
                           "objects": len(objs), "S": s,
                           "sec_per_query": {"appearance": t_app, "voting": t_vote,
                                            "count": t_count},
                           "pool": {"objects": len(gal),
                                    "places": int(sum(len(e["places"]) for e in gal_entries)),
                                    "buckets": len(index.by_key),
                                    "dropped": {k: int(sum(e["dropped"][k] for e in gal_entries))
                                                for k in ("no_sig", "no_config", "no_position",
                                                          "no_scale")}},
                           "count_diag": {"places_used_mean": float(np.mean(
                               [d["places_used"] for d in cdiag])),
                               "objects_scored_mean": float(np.mean(
                                   [d["objects_scored"] for d in cdiag]))},
                           "vote_diag": {"places_used_mean": float(np.mean(
                               [d["places_used"] for d in diag])),
                               "votes_mean": float(np.mean([d["votes"] for d in diag])),
                               "bins_mean": float(np.mean([d["bins"] for d in diag])),
                               "objects_scored_mean": float(np.mean(
                                   [d["n_objects_scored"] for d in diag]))},
                           "frontdoors": {}}
    for name, mat in (("appearance", app), ("voting", vote), ("count", counted),
                      ("fused", fused), ("fused_count", fused_count)):
        res["frontdoors"][name] = ranks_metrics(mat, objs, gal_keys, s)
        np.save(os.path.join(OUT, "m22_%s_%s_scores.npy"
                             % (tag or split.lower(), name)), mat)
    print("%s: %d objects, gallery %d, pool %d places in %d buckets"
          % (split, len(objs), len(gal), res["pool"]["places"], res["pool"]["buckets"]))
    print("%-11s %-9s %-9s %-8s %-9s %-9s %s"
          % ("front door", "recall@%d" % s, "chance", "top-1", "chance", "s/query", "n"))
    for name in ("appearance", "voting", "count", "fused", "fused_count"):
        m = res["frontdoors"][name]
        print("%-11s %-9.3f %-9.3f %-8.3f %-9.3f %-9.4f %d"
              % (name, m["recall_s"], m["chance_recall_s"], m["top1"],
                 m["chance_top1"],
                 res["sec_per_query"][{"fused": "voting", "fused_count": "count"}.get(name, name)],
                 m["n"]))
    return res


# --------------------------------------------------------------------------- assemble + class
def assemble_stage(split: str, per_class: int, s: int, workers: int,
                   front: str = "appearance", tag: str = "") -> Dict[str, Any]:
    """The assembled system on one split.  Resumable through its pairs file."""
    from matching import placement_assemble as AS
    from matching import placement_frontdoor as FD
    api = __import__("parameters")
    if not bool(getattr(api, "PLACEMENT_ASSEMBLED", False)):
        print("note: PLACEMENT_ASSEMBLED is off; running the stage anyway (harness is explicit)")
    tag = tag or "m22_assemble_%s" % split.lower()
    objs = objects(split, per_class)
    gal = sorted({o["ref"] for o in objs})
    key_of_ref = {o["ref"]: o["key"] for o in objs}
    gal_keys = [key_of_ref[p] for p in gal]
    # the shortlist comes from the named front door's saved score matrix (frontdoor stage)
    _names = {"appearance": "appearance", "count": "count", "voting": "voting",
              "fused": "fused_count", "fused_voting": "fused"}
    if front not in _names:
        raise KeyError("unknown front door: %s" % front)
    app = np.load(os.path.join(OUT, "m22_%s_%s_scores.npy" % (split.lower(), _names[front])))
    # ONE pairs cache per split: the assembled arms share the same (query, gallery) scores, so a
    # second arm built over a different shortlist only pays for the pairs it has not seen.
    pairs_path = os.path.join(OUT, "m22_assemble_%s_pairs.jsonl" % split.lower())
    done = {}
    if os.path.exists(pairs_path):
        for line in open(pairs_path):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[(r["q"], r["r"])] = r
    plan: List[Tuple[Dict[str, Any], List[int]]] = []
    for i, o in enumerate(objs):
        cands = FD.shortlist(app, i, s)
        plan.append((o, cands))
    jobs = []
    for o, candss in plan:
        for j in candss:
            if (o["query"], gal[j]) not in done:
                jobs.append((o["query"], gal[j]))
    print("assemble %s: %d pairs to score (%d cached)" % (split, len(jobs), len(done)),
          flush=True)
    t0 = time.time()
    if jobs:
        from concurrent.futures import ProcessPoolExecutor
        jobs.sort()                       # grouped by query: a chunk reuses that query's graph
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, (job, r) in enumerate(zip(jobs, ex.map(_pair_lite, jobs, chunksize=4))):
                if "error" not in r:
                    done[(r["q"], r["r"])] = r
                with open(pairs_path, "a") as fh:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
                if (k + 1) % 8 == 0:
                    print("  %d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    # the assembled order, per query, and the class score from the same scores
    from matching import placement_classscore as CS
    res: Dict[str, Any] = {"split": split, "per_class": per_class, "S": s, "front": front,
                           "pairs_scored": len(done), "objects": len(objs)}
    key_idx = {k: j for j, k in enumerate(gal_keys)}
    app_order: List[List[int]] = []
    asm_order: List[List[int]] = []
    asm_top1 = app_top1 = 0
    n_missing = 0
    inst_scores: List[Dict[int, float]] = []
    asm_rank: List[Any] = []
    asm_by_class: Dict[str, Dict[str, int]] = {}
    for i, (o, candss) in enumerate(plan):
        truth = key_idx[o["key"]]
        app_order.append(list(candss))
        scores = {}
        missing = 0
        for j in candss:
            r = done.get((o["query"], gal[j]))
            if r is None:
                missing += 1
                continue
            scores[j] = float(r["B"])
        ordered, sc = AS.assemble(candss, lambda j: scores[j])
        asm_order.append(ordered)
        inst_scores.append({j: scores.get(j, 0.0) for j in candss})
        app_top1 += 1 if candss[0] == truth else 0
        asm_top1 += 1 if ordered[0] == truth else 0
        n_missing += missing
        # the truth's position in the assembled order (None = the door never offered it)
        asm_rank.append((ordered.index(truth) + 1) if truth in ordered else None)
        d = asm_by_class.setdefault(o["cls"], {"n": 0, "top1": 0, "in_S": 0})
        d["n"] += 1
        d["top1"] += 1 if ordered[0] == truth else 0
        d["in_S"] += 1 if truth in ordered else 0
        if missing:
            res.setdefault("queries_with_missing", []).append(o["key"])
    n = len(objs)
    res["appearance_top1"] = app_top1 / n
    res["assembled_top1"] = asm_top1 / n
    res["recall_S_frontdoor"] = sum(1 for i, (o, c) in enumerate(plan)
                                    if key_idx[o["key"]] in c) / n
    res["chance_top1"] = 1.0 / len(gal)
    res["chance_recall_S"] = s / len(gal)
    # paired, clustered by object (one observation per object, so clustering is the object)
    flips = {"app_only": 0, "asm_only": 0, "both": 0, "neither": 0}
    for i, (o, candss) in enumerate(plan):
        truth = key_idx[o["key"]]
        a = 1 if candss[0] == truth else 0
        b = 1 if asm_order[i][0] == truth else 0
        flips["both" if (a and b) else "app_only" if a else "asm_only" if b else "neither"] += 1
    res["paired"] = flips
    res["errors"] = int(n_missing)
    res["mcnemar_p"] = _mcnemar_exact(flips["asm_only"], flips["app_only"])
    res["assembled_ranks"] = asm_rank
    res["assembled_rank_median"] = (float(np.median([r for r in asm_rank if r is not None]))
                                    if any(r is not None for r in asm_rank) else None)
    res["assembled_per_class"] = {c: {"n": d["n"], "top1": d["top1"] / d["n"],
                                      "in_S": d["in_S"] / d["n"]}
                                  for c, d in asm_by_class.items()}
    # class score on the assembled instance scores, primary = sum of top-2, arms = max, mean
    cls_by_ref = {o["ref"]: o["cls"] for o in objs}
    class_of = {j: cls_by_ref[gal[j]] for j in range(len(gal))}
    k_of_class = {c: sum(1 for j in range(len(gal)) if class_of[j] == c)
                  for c in set(class_of.values())}
    res["class"] = {}
    for agg in ("top2", "max", "mean"):
        hits = 0
        n_used = 0
        chances: List[float] = []
        for i, (o, candss) in enumerate(plan):
            cs = CS.class_scores(inst_scores[i], class_of, agg=agg)
            if not cs:
                continue
            n_used += 1
            hits += CS.class_top1(cs, o["cls"])
            # exact chance: a random top-1 gallery item belongs to the query's class with
            # probability k/N, k = the number of gallery items of that class
            chances.append(CS.chance_top1(k_of_class[o["cls"]], len(gal)))
        res["class"][agg] = {"top1": (hits / n_used) if n_used else None, "n": n_used,
                             "chance": float(np.mean(chances)) if chances else None,
                             "k_per_class": k_of_class}
    out = os.path.join(OUT, tag + "_read.json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("%-10s appearance top-1 %.3f | assembled top-1 %.3f | chance %.3f | n %d"
          % (split, res["appearance_top1"], res["assembled_top1"], res["chance_top1"], n))
    print("  paired: both %d, appearance-only %d, assembled-only %d, neither %d"
          % (flips["both"], flips["app_only"], flips["asm_only"], flips["neither"]))
    for agg, m in res["class"].items():
        print("  class(%s): top-1 %.3f (n=%d, chance %.3f)"
              % (agg, m["top1"] or 0, m["n"], m["chance"]))
    print("wrote", out)
    return res


def verify_stage(split: str, n_pairs: int, workers: int) -> Dict[str, Any]:
    """Re-score cached pairs with the FULL Mission-20 pair job and compare to the lite score.

    The assembled arms are scored by `pair_score_job`, which calls the same `joint_mcs` with the
    same deployed flags but skips the equal-coverage aggregate, the attribution and the arm-A
    control.  That is a claim about identity, so it is checked: the same (query, gallery) pairs are
    re-run through `_pair_job` and `score_shipped` must agree.  Pairs are taken in sorted order so
    the check is deterministic and re-runnable.
    """
    from concurrent.futures import ProcessPoolExecutor
    from Topology.placement_m20_eth80 import _pair_job
    rows = {}
    path = os.path.join(OUT, "m22_assemble_%s_pairs.jsonl" % split.lower())
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                rows[(r["q"], r["r"])] = r
    keys = sorted(rows)[:int(n_pairs)]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_pair_job, q, r, False) for (q, r) in keys]
        for (q, r), f in zip(keys, futs):
            full = f.result()
            lite = rows[(q, r)]["B"]
            out.append({"q": q, "r": r, "lite": float(lite), "full": float(full["B"]),
                        "diff": abs(float(full["B"]) - float(lite))})
    res = {"split": split, "pairs": out, "n": len(out),
           "max_abs_diff": max([x["diff"] for x in out], default=None),
           "identical_1e12": all(x["diff"] <= 1e-12 for x in out)}
    print("verify %s: %d pairs, max |lite - full| = %s, identical(1e-12): %s"
          % (split, len(out), res["max_abs_diff"], res["identical_1e12"]))
    for x in out:
        print("  %-28s lite %.15f full %.15f diff %.3e"
              % (os.path.basename(x["q"]), x["lite"], x["full"], x["diff"]))
    from matching import m15_manifest as MF
    MF.write_result(os.path.join(OUT, "m22_assemble_%s_verify.json" % split.lower()), res)
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--stage", default="all",
                    choices=("places", "frontdoor", "assemble", "verify", "all"))
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--shortlist", type=int, default=S_DEFAULT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--front", default="appearance")
    ap.add_argument("--tag", default="")
    ap.add_argument("--verify-pairs", type=int, default=6)
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    if not getattr(P, "PLACEMENT_M19", False):
        raise SystemExit("set PLACEMENT_M19=1: this reads the deployed configuration")
    if a.stage in ("places",):
        store = ensure_places(a.split, a.per_class, a.workers)
        print("%d images in the place store" % len(store))
        return 0
    if a.stage == "verify":
        verify_stage(a.split, a.verify_pairs, a.workers)
        return 0
    res = {}
    if a.stage in ("frontdoor", "all"):
        res["frontdoor"] = frontdoor_stage(a.split, a.per_class, a.shortlist, a.workers,
                                           a.tag)
    if a.stage in ("assemble", "all"):
        res["assemble"] = assemble_stage(a.split, a.per_class, a.shortlist, a.workers,
                                         a.front, a.tag)
    out = os.path.join(OUT, (a.tag or ("m22_%s" % a.split.lower())) + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
