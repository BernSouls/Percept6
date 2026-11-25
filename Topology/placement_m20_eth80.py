"""MISSION 20 -- the ETH-80 read harness: stage-1 shortlist, stage-2 placement, four arms.

One pass over pairs serves every task and every arm, because the expensive thing is the
pair, not the arithmetic on it.  For a pair (query image, gallery image) the run stores:

    stage1   the cheap configuration score (no mapping)                      [item 3]
    A        joint_mcs with the placement matcher OFF (the shipped system before M19)
    B        joint_mcs with the matcher ON  -> .score_shipped                [arm S0]
    C        the same pair with the evidence term on -> .score               [arm S1]
    E        .placement_evidence, the number judged, the number mapped       [item 4]
    S3       the NEW coverage-neutral aggregate, computed here                [item 4 / S2-S3]
    sec_a/sec_c  per-pair seconds, so s/query is measured, not guessed        [item 3]

THE NEW AGGREGATE (S3), written down before it was measured:

    num = sum over MAPPED query places u whose pair (u, mapping[u]) the frozen estimator
          can judge, of (plus_score(u, mapping[u]) - 0.5)
    den = number of query places that CARRY a graded incident relation -- a property of
          the two graphs, NOT of the mapping
    S3  = num / (den + 8)                     S2 = score_shipped + lam * S3

Why this and not the shipped E: E divides by ``judged + 8``, and ``judged`` counts the
pairs the mapping produced.  A matcher that maps only the easy places therefore scores a
mean over a thin, self-selected base -- the count of correspondences leaks back in through
the denominator, which is exactly what "coverage-neutral" is supposed to stop.  Here the
denominator is what COULD have been checked, so mapping five places out of eighty scores
five places out of eighty, and adding places with no checkable evidence changes neither
the numerator nor the denominator.

Tasks (registered in M20_TASKS_PREREGISTRATION.md, not redefined here):

    T0  query = the reference image rotated 15 deg; truth = its own object; node-level too
    T1  query = view -022-000, gallery = the query's own class at -000-000
    T2  query = view -022-000, gallery = every object at -000-000; truth = the CLASS
        (the query's own instance is not a positive -- this task has no place truth)
    T3  query = view -022-000, gallery = every object at -000-000; truth = the object

Every number is reported with its n.  A shortlist number is never printed without its
recall.  Truth is used only to score, never inside the matcher.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
SPLITS = os.path.join(OUT, "m20_splits.json")
IMG = 256
REF_VIEW = "-000-000"
QUERY_VIEW = "-022-000"
CLASS_DIR = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
             "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG, IMG):
        im = im.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def objects(split: str) -> List[Dict[str, Any]]:
    d = json.load(open(SPLITS))
    out = []
    for o in d["splits"][split]["objects"]:
        base = os.path.join(ROOT, "data", "ETH-80-master", CLASS_DIR[o["class"]], o["instance"])
        out.append({"key": "%s/%s" % (o["class"], o["instance"]), "cls": o["class"],
                    "label": o["label"],
                    "ref": os.path.join(base, "%s%s.png" % (o["label"], REF_VIEW)),
                    "query": os.path.join(base, "%s%s.png" % (o["label"], QUERY_VIEW))})
    out.sort(key=lambda e: e["key"])
    return out


#: the front end every M20 number is measured under (the M16 recipe, which is what the
#: M19 record used).  The graph cache keys on pixels + the switch digest + the source
#: closure, so setting it here cannot silently reuse an older recipe's graphs.
RECIPE = dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
              FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
              PLACEMENT_FRAME_KEYPOINT_SIGMA="largest", PLACEMENT_FRAME_FALLBACK="none",
              PLACEMENT_M15_PLACES=True, KP_GRADED_SIMILARITY=True)


def _recipe() -> None:
    import parameters as P
    for k, v in RECIPE.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True          # the payload lives in these graphs
    from core import keypoint_classes as KPC
    KPC.unregister()
    KPC.register("wide")
    KPC.register_population()


#: per-process memo of (image, graph, side).  WHY IT EXISTS: `get_or_build` unpickles a
#: FRESH graph object on every call, and `build_side_cached` keys on `id(graph)`, so without
#: this the side cache never hits and grows to 4,096 entries of dead graphs.  Measured
#: before the fix: +200-400 MB per query image, two pool workers killed at 208 of 1,600
#: pairs on a 3.9 GB box.  Four entries is enough for stage 2 (one query against a gallery)
#: and keeps the footprint at a few hundred MB.
_CELLS: Dict[str, Any] = {}
_CELLS_MAX = 4


def _cell(path: str, tag: str):
    """(image, graph, side) for one ETH-80 image, memoised per process."""
    ent = _CELLS.get(path)
    if ent is not None:
        return ent
    import guard
    guard.activate()
    from matching import placement_matcher as PM
    from Topology.placement_cache import get_or_build
    _recipe()
    img = _load_u8(path)
    g = get_or_build(img, "eth|" + path)
    ent = (img, g, PM.build_side_cached(g, img))
    if len(_CELLS) >= _CELLS_MAX:
        _CELLS.clear()
        PM._SIDE_CACHE.clear()
    _CELLS[path] = ent
    return ent


def bag_job(path: str) -> Dict[str, Any]:
    """One image -> its configuration bag (tens of KB), so stage 1 never rebuilds a graph."""
    import guard
    guard.activate()
    from matching import placement_prefilter as PF
    t0 = time.time()
    img, g, side = _cell(path, "bag")
    nodes = sorted(int(n) for n in g.subgraph.nodes)
    return {"path": path, "bag": PF.bag_of(side, nodes),
            "n_nodes": len(nodes), "sec": time.time() - t0}


def pair_job(args: Tuple[str, str, bool]) -> Dict[str, Any]:
    """One pair, four arms.  This is the expensive call.

    A pair that raises is recorded, not fatal: 640 pairs on a two-core box must not lose
    the other 639 because one image is odd, and a failure rate is itself a measurement.
    """
    qpath, rpath, want_a = args
    try:
        return _pair_job(qpath, rpath, want_a)
    except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
        return {"q": qpath, "r": rpath, "error": repr(exc)[:200]}


def _pair_job(qpath: str, rpath: str, want_a: bool = True) -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_matcher as PM
    from matching import placement_llr as L
    from matching.joint_mcs import joint_mcs
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        w_ev = float(json.load(fh)["w"])
    q_img, q, side_q = _cell(qpath, "q")          # memoised: see _CELLS
    r_img, r, side_r = _cell(rpath, "r")

    # arm A (matcher off) is NOT one of the four arms; it is only the "no placement" control
    # and it doubles the cost of a pair, so a run may skip it and say so in its rows.
    arm_a = None
    sec_a = 0.0
    if want_a:
        P.PLACEMENT_MATCHER_ENABLED = False
        P.PLACEMENT_SCORE_EVIDENCE = False
        ta = time.time()
        arm_a = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs)
        sec_a = time.time() - ta

    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = w_ev
    P.PLACEMENT_MATCHER_STRICT = True          # a fallback here is a wiring bug, not a result
    tc = time.time()
    c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                  placement_images=(q_img, r_img))
    sec_c = time.time() - tc
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    P.PLACEMENT_MATCHER_STRICT = False

    # ---- S3: the coverage-neutral aggregate, from the same mapping ----
    fit, pfit = PM.fit_objects()
    b = PM.frozen()
    a_e, a_g = float(b["_a_exist"]), float(b["_a_geom"])
    mapping = dict(c.mapping)
    num = 0.0
    judged_places = 0
    for u, v in sorted(mapping.items()):
        slots = L.slots_for_candidate(u, v, mapping, side_q.incident, side_q.index,
                                      side_r.index)
        if not slots:
            continue
        judged_places += 1
        num += float(L.plus_score(u, v, mapping, side_q.incident, side_q.index,
                                  side_r.index, fit, pfit, a_e, a_g)) - 0.5
    # the denominator: query places that CARRY a graded incident relation, mapping or not
    den = sum(1 for u in sorted(int(n) for n in q.subgraph.nodes)
              if side_q.incident.get(int(u)))
    s3 = num / (den + 8.0) if den else None
    return {"q": qpath, "r": rpath,
            "A": (None if arm_a is None else float(arm_a.score)),
            "B": float(c.score_shipped), "C": float(c.score),
            "E": float(c.placement_evidence), "judged": int(c.placement_judged),
            "mapped": len(mapping), "share": float(c.placement_share_judged),
            "s3_num": float(num), "s3_den": int(den), "S3": s3,
            "judged_places": int(judged_places),
            "sec_a": sec_a, "sec_c": sec_c,
            "falls_back": int(__import__("matching.placement_matcher",
                                         fromlist=["FAILURES"]).FAILURES["fell_back"])}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def build_plan(split: str, task: str,
               per_class: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(queries, gallery).  The gallery is the same list for T0/T2/T3; T1 filters by class.

    ``per_class`` keeps the first N objects of each class (by sorted key, deterministic) as
    the gallery AND as the queries: for an instance-level task the truth must be a gallery
    item, so the two sets coincide.  Gallery size is a FIT decision (item 6).
    """
    objs = objects(split)
    if per_class:
        per: Dict[str, List[Dict[str, Any]]] = {}
        for o in objs:
            per.setdefault(o["cls"], []).append(o)
        objs = [o for c in sorted(per) for o in per[c][:int(per_class)]]
    gallery = [{"key": o["key"], "cls": o["cls"], "path": o["ref"]} for o in objs]
    queries = []
    for o in objs:
        if task == "T0":
            # the aggregation control: the reference image itself, rotated 15 deg.  Written
            # to a file so the whole pipeline (bags, shortlist, pairs) is keyed by a path,
            # exactly as it is for a real image; the file is deterministic and lives in /tmp.
            queries.append({"key": o["key"], "cls": o["cls"],
                            "path": _rotated_png(o["ref"])})
        else:
            queries.append({"key": o["key"], "cls": o["cls"], "path": o["query"]})
    return queries, gallery


def _rotated_png(path: str, deg: float = 15.0) -> str:
    """The image at ``path`` rotated by ``deg``, cached as a PNG next to nothing in git."""
    import hashlib
    from PIL import Image
    d = "/tmp/m20_rot"
    os.makedirs(d, exist_ok=True)
    tag = hashlib.sha256(("%s|%.3f" % (path, deg)).encode()).hexdigest()[:16]
    out = os.path.join(d, tag + ".png")
    if not os.path.exists(out):
        Image.fromarray(_transform_u8(_load_u8(path), deg)).save(out)
    return out


def _transform_u8(img: np.ndarray, deg: float) -> np.ndarray:
    from Topology import placement_m12_read_real as R
    return R._transform(img, {"rot_deg": float(deg)})


def rotated(path: str, deg: float = 15.0) -> np.ndarray:
    from Topology import placement_m12_read_real as R
    return R._transform(_load_u8(path), {"rot_deg": float(deg)})


def norm(ranked: List[str], correct: List[str]) -> Tuple[int, float]:
    """(rank of the best correct item, reciprocal rank); rank is inf-free: 0 when absent."""
    for i, k in enumerate(ranked):
        if k in correct:
            return i + 1, 1.0 / (i + 1)
    return 0, 0.0


def main(argv=None) -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="FIT")
    ap.add_argument("--task", default="T3", choices=("T0", "T1", "T2", "T3"))
    ap.add_argument("--stage", default="all",
                    choices=("bags", "shortlist", "pairs", "read", "all"))
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=0,
                    help="gallery (=queries) objects per class; 0 means all")
    ap.add_argument("--no-arm-a", action="store_true",
                    help="skip the matcher-off arm (it is not one of the four arms)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    tag = a.tag or ("m20_%s_%s" % (a.split.lower(), a.task.lower()))
    p1 = os.path.join(OUT, tag + "_stage1.jsonl")
    p2 = os.path.join(OUT, tag + "_pairs.jsonl")
    queries, gallery = build_plan(a.split, a.task, a.per_class)
    if a.limit:
        queries = queries[:a.limit]
    print("%s %s: %d queries, %d gallery items, shortlist %d"
          % (a.split, a.task, len(queries), len(gallery), a.shortlist), flush=True)

    # ---- stage 1a: one bag per image (the only part that touches a graph) ----
    bags_path = os.path.join(OUT, tag + "_bags.pkl")
    if a.stage in ("bags", "shortlist", "all"):
        if os.path.exists(bags_path):
            with open(bags_path, "rb") as fh:
                bags = pickle.load(fh)
        else:
            bags = {}
        want = sorted({q["path"] for q in queries} | {g["path"] for g in gallery})
        need = [p for p in want if p not in bags]
        print("bags: %d images, %d missing" % (len(want), len(need)), flush=True)
        t0 = time.time()
        if need:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for i, r in enumerate(ex.map(bag_job, need, chunksize=1)):
                    bags[r["path"]] = r
                    if (i + 1) % 10 == 0:
                        print("  bags %d/%d %.0fs" % (i + 1, len(need), time.time() - t0),
                              flush=True)
                    with open(bags_path + ".tmp", "wb") as fh:
                        pickle.dump(bags, fh, protocol=4)
                    os.replace(bags_path + ".tmp", bags_path)
        print("bags done %.1fs (%d images)" % (time.time() - t0, len(bags)), flush=True)
        if a.stage == "bags":
            return 0

    # ---- stage 1b: the shortlist, pure numpy over bags ----
    if a.stage in ("shortlist", "all"):
        from matching import placement_prefilter as PF
        t0 = time.time()
        n = 0
        with open(p1, "w") as fh:
            for q in queries:
                for g in gallery:
                    out = PF.score_bags(bags[q["path"]]["bag"], bags[g["path"]]["bag"])
                    fh.write(json.dumps({"q": q["path"], "r": g["path"],
                                         "score": out["score"],
                                         "n_places": out["n_places"]},
                                        sort_keys=True) + "\n")
                    n += 1
        print("stage1: %d pairs in %.1fs (%.1f us/pair)"
              % (n, time.time() - t0, 1e6 * (time.time() - t0) / max(1, n)), flush=True)
    if a.stage == "shortlist":
        return 0

    # ---- the shortlist per query, then stage 2 on it ----
    s1 = {(r["q"], r["r"]): r for r in _read_jsonl(p1)}
    plan: List[Tuple[Dict[str, Any], List[str]]] = []
    for q in queries:
        sc = {g["path"]: (s1.get((q["path"], g["path"])) or {}).get("score")
              for g in gallery}
        from matching import placement_prefilter as PF
        ranked = PF.shortlist(sc, a.shortlist)
        if a.task == "T1":
            keep = {g["path"] for g in gallery if g["cls"] == q["cls"]}
            ranked = [p for p in ranked if p in keep] + \
                     [p for p in sorted(keep) if p not in ranked]
        plan.append((q, ranked[:a.shortlist]))
    done2 = {(r["q"], r["r"]) for r in _read_jsonl(p2)}
    jobs = [(q["path"], p, not a.no_arm_a) for (q, ps) in plan for p in ps
            if (q["path"], p) not in done2]
    print("stage2: %d pairs to match (%d cached)" % (len(jobs), len(done2)), flush=True)
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(pair_job, jobs, chunksize=1)):
                _append(p2, r)
                if (i + 1) % 10 == 0:
                    print("  stage2 %d/%d %.0fs" % (i + 1, len(jobs), time.time() - t0),
                          flush=True)
    print("stage2 done %.1fs" % (time.time() - t0), flush=True)
    if a.stage == "pairs":
        return 0

    # ---- read ----
    rows = {(r["q"], r["r"]): r for r in _read_jsonl(p2)}
    lam = float(json.load(open(os.path.join(ROOT, "matching",
                                            "placement_score_params.json")))["w"])
    key_of = {g["path"]: g for g in gallery}

    def correct_keys(q: Dict[str, Any]) -> List[str]:
        if a.task == "T2":
            return [g["key"] for g in gallery
                    if g["cls"] == q["cls"] and g["key"] != q["key"]]
        return [q["key"]]

    def value(arm: str, r: Dict[str, Any]) -> float:
        if arm == "S0":
            return float(r["B"])
        if arm == "S1":
            return float(r["B"]) + lam * float(r["E"])
        if arm == "S3":
            return float(r["S3"]) if r["S3"] is not None else -1e9
        if arm == "S2":
            return float(r["B"]) + lam * (float(r["S3"]) if r["S3"] is not None else 0.0)
        raise KeyError(arm)

    summary: Dict[str, Any] = {"split": a.split, "task": a.task, "shortlist": a.shortlist,
                               "lam": lam, "recipe": dict(RECIPE), "arms": {}, "plans": []}
    for arm in ("S0", "S1", "S2", "S3"):
        ranks, rrs, pos, n, sec = [], [], 0, 0, 0.0
        per_query = []
        for (q, ps) in plan:
            here = [(key_of[p]["key"], p, rows[(q["path"], p)]) for p in ps
                    if (q["path"], p) in rows]
            if not here:
                continue
            n += 1
            sec += sum(r["sec_c"] for (_k, _p, r) in here)
            ranked = [k for (k, _p, r) in sorted(here, key=lambda t: (-value(arm, t[2]), t[0]))]
            rk, rr = norm(ranked, correct_keys(q))
            ranks.append(rk)
            rrs.append(rr)
            if rk:
                pos += 1
            per_query.append({"q": q["key"], "cls": q["cls"], "rank": rk, "rr": rr,
                              "n_scored": len(here),
                              "mapped": [r["mapped"] for (_k, _p, r) in here],
                              "judged": [r["judged"] for (_k, _p, r) in here],
                              "s3_den": [r["s3_den"] for (_k, _p, r) in here]})
        if n:
            summary["arms"][arm] = {
                "n": n, "rank1": sum(1 for r in ranks if r == 1) / n,
                "mrr": sum(rrs) / n,
                # recall@S is the share of queries whose positive was IN the shortlist;
                # rank1 above can never exceed it, and both are printed together
                "recall_at_S": pos / n,
                "per_class": {c: sum(1 for r in per_query if r["cls"] == c and r["rank"] == 1)
                              / max(1, sum(1 for r in per_query if r["cls"] == c))
                              for c in sorted({r["cls"] for r in per_query})},
                "sec_per_query": sec / n}
            summary.setdefault("per_query", {})[arm] = per_query
    q_sec = [r.get("sec") for r in s1.values() if r.get("sec") is not None]
    summary["stage1"] = {"n_pairs": len(s1),
                         "sec_per_pair": float(np.mean(q_sec)) if q_sec else None}
    out = os.path.join(OUT, tag + "_read.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print("\n%-4s %-6s %-6s %-6s %-6s %-8s" % ("arm", "n", "rank1", "MRR", "recall", "s/query"))
    for arm in sorted(summary["arms"]):
        t = summary["arms"][arm]
        print("%-4s %-6d %-6.3f %-6.3f %-6.3f %-8.2f"
              % (arm, t["n"], t["rank1"], t["mrr"], t["recall_at_S"], t["sec_per_query"]))
    print("stage1 %.1f ms/pair" % (1000 * (summary["stage1"]["sec_per_pair"] or 0)))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
