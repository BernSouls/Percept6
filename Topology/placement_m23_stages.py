"""MISSION 23 -- the three stages the harness runs, kept apart so each can be read on its own."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Topology import placement_m23_eth80 as H

OUT = H.OUT


# --------------------------------------------------------------------------- item 1: T2
def stage_t2(workers: int = 2, s: int = H.S_DEFAULT, limit: int = 0) -> Dict[str, Any]:
    """The class score on a gallery with the query's own object REMOVED.

    Every query is one FIT object's `-022-000` view; the gallery is the other 39 FIT objects'
    references.  The door draws a shortlist from those 39, the shipped matcher scores it, and the
    three registered aggregates turn the instance scores into a class ranking.  The chance is
    exact and per query: k / (N - 1) with k = 4 (the query's class has 5 instances in FIT and the
    query's own object is gone) and N - 1 = 39.
    """
    from matching import placement_frontdoor as FD
    from matching import placement_classscore as CS
    from matching import placement_prefilter as PF
    from Topology.placement_m22_eth80 import _mcnemar_exact  # noqa: F401  (re-used below)
    objs = H.split_objects("FIT")
    if limit:
        objs = objs[:int(limit)]
    keys = [o["key"] for o in objs]
    refs = [o["ref"] for o in objs]
    queries = [o["query"] for o in objs]
    store = H.ensure_images(refs + queries, "fit", workers)
    params = PF.load_params()
    index = FD.PlaceIndex([store[r] for r in refs], keys)
    t0 = time.time()
    app = FD.appearance_scores(queries, refs)
    cnt, _diag = FD.count_scores(index, [store[p] for p in queries],
                                 n_hits=int(H._n("PLACEMENT_COUNT_HITS")))
    t_door = (time.time() - t0) / max(1, len(objs))
    k_class = (sum(1 for o in objs if o["cls"] == objs[0]["cls"]) - 1) if objs else 0
    res: Dict[str, Any] = {"split": "FIT", "n_objects": len(objs), "gallery_full": len(refs),
                           "gallery_reduced": len(refs) - 1,
                           "chance_top1": k_class / float(len(refs) - 1), "k_per_class": k_class,
                           "S": s,
                           "sec_per_query_door": t_door,
                           "note": "the query's own object is removed BEFORE the door runs; "
                                   "class_scores_t2 raises if it is present",
                           "agg": {}}
    shortlists: Dict[int, List[int]] = {}
    for i, k in enumerate(keys):
        cols = [j for j in range(len(refs)) if keys[j] != k]          # the 39
        sub_app = app[i, cols][None, :]
        sub_cnt = cnt[i, cols][None, :]
        fused = FD.rank_fuse(sub_app, sub_cnt)[0]
        order = sorted(range(len(cols)), key=lambda t: (-fused[t], cols[t]))
        shortlists[i] = [cols[t] for t in order[:s]]
    jobs = [(queries[i], refs[j]) for i in range(len(objs)) for j in shortlists[i]]
    scores = H.ensure_pairs(jobs, "t2", workers)
    per_agg = {agg: {"hits": 0, "per_class": {}, "rows": []} for agg in ("top2", "max", "mean")}
    class_of = {j: objs[j]["cls"] for j in range(len(refs))}
    for i, o in enumerate(objs):
        kept = [j for j in range(len(refs)) if j != i]
        inst = {j: scores[(o["query"], refs[j])] for j in shortlists[i]
                if (o["query"], refs[j]) in scores}
        present = {keys[j] for j in kept}
        if o["key"] in present:                                      # the guard, at the boundary
            raise AssertionError("T2 gallery still contains %r" % o["key"])
        for agg in per_agg:
            cs = CS.class_scores_t2(inst, class_of, o["key"], [keys[j] for j in kept], agg=agg)
            hit = CS.class_top1(cs, o["cls"])
            per_agg[agg]["hits"] += hit
            d = per_agg[agg]["per_class"].setdefault(o["cls"], [0, 0])
            d[0] += hit
            d[1] += 1
            per_agg[agg]["rows"].append({"q": o["key"], "cls": o["cls"], "hit": hit,
                                         "in_shortlist": sum(1 for j in shortlists[i]
                                                             if class_of[j] == o["cls"])})
    n = len(objs)
    for agg, d in per_agg.items():
        res["agg"][agg] = {"top1": d["hits"] / n, "hits": d["hits"], "n": n,
                           "chance": k_class / float(len(refs) - 1),
                           "per_class": {c: {"n": v[1], "top1": v[0] / v[1]}
                                         for c, v in sorted(d["per_class"].items())},
                           "class_coverage": float(np.mean(
                               [sum(1 for j in shortlists[i] if class_of[j] == objs[i]["cls"])
                                / float(max(1, k_class)) for i in range(n)]))}
        print("T2 %-5s class top-1 %.3f (%d/%d, chance %.3f), class coverage %.3f"
              % (agg, res["agg"][agg]["top1"], d["hits"], n, res["agg"][agg]["chance"],
                 res["agg"][agg]["class_coverage"]))
    return res


# --------------------------------------------------------------------------- item 3: the rerank
def _orders_for_split(split: str, workers: int, s: int,
                      seed_pairs: Optional[str] = None, limit: int = 0) -> Dict[str, Any]:
    from matching import placement_frontdoor as FD
    from matching import placement_prefilter as PF
    from Topology.placement_m22_eth80 import _mcnemar_exact
    objs = H.split_objects(split)
    if limit:
        objs = objs[:int(limit)]
    keys = [o["key"] for o in objs]
    refs = [o["ref"] for o in objs]
    queries = [o["query"] for o in objs]
    store = H.ensure_images(refs + queries, split.lower(), workers)
    index = FD.PlaceIndex([store[r] for r in refs], keys)
    app = FD.appearance_scores(queries, refs)
    cnt, _d = FD.count_scores(index, [store[p] for p in queries],
                              n_hits=int(H._n("PLACEMENT_COUNT_HITS")))
    fused = FD.rank_fuse(app, cnt)
    cands = {i: FD.shortlist(fused, i, s) for i in range(len(objs))}
    jobs = [(queries[i], refs[j]) for i in range(len(objs)) for j in cands[i]]
    if seed_pairs and os.path.exists(seed_pairs):
        dst = os.path.join(OUT, "m23_%s_pairs.jsonl" % split.lower())
        if not os.path.exists(dst):
            with open(dst, "w") as fh:
                for line in open(seed_pairs):
                    if line.strip():
                        r = json.loads(line)
                        if "error" not in r and "B" in r:
                            fh.write(json.dumps({"q": r["q"], "r": r["r"],
                                                 "B": float(r["B"])}, sort_keys=True) + "\n")
            print("seeded %s from %s" % (dst, seed_pairs))
    scores = H.ensure_pairs(jobs, split.lower(), workers)
    key_idx = {k: j for j, k in enumerate(keys)}
    out = {"split": split, "n": len(objs), "gallery": len(refs), "S": s,
           "chance_top1": 1.0 / len(refs), "chance_recall_s": s / len(refs), "rows": [],
           "arms": {}}
    hits = {"door": 0, "plain": 0, "fused": 0}
    rec = 0
    for i, o in enumerate(objs):
        truth = key_idx[o["key"]]
        c = cands[i]
        sc = {j: scores[(o["query"], refs[j])] for j in c if (o["query"], refs[j]) in scores}
        p = H.order_plain(c, sc)
        f = H.order_fused(c, sc)
        hd = 1 if c[0] == truth else 0
        hp = 1 if (p and p[0] == truth) else 0
        hf = 1 if (f and f[0] == truth) else 0
        hits["door"] += hd
        hits["plain"] += hp
        hits["fused"] += hf
        rec += 1 if truth in c else 0
        out["rows"].append({"q": o["key"], "cls": o["cls"], "door": hd, "plain": hp,
                            "fused": hf, "in_shortlist": int(truth in c),
                            "door_rank": (c.index(truth) + 1) if truth in c else None,
                            "plain_rank": (p.index(truth) + 1) if truth in p else None,
                            "fused_rank": (f.index(truth) + 1) if truth in f else None})
    n = len(objs)
    paired = {
        "plain_vs_door": {"plain_only": sum(1 for r in out["rows"] if r["plain"] and not r["door"]),
                          "door_only": sum(1 for r in out["rows"] if r["door"] and not r["plain"])},
        "fused_vs_door": {"fused_only": sum(1 for r in out["rows"] if r["fused"] and not r["door"]),
                          "door_only": sum(1 for r in out["rows"] if r["door"] and not r["fused"])},
        "fused_vs_plain": {"fused_only": sum(1 for r in out["rows"] if r["fused"] and not r["plain"]),
                           "plain_only": sum(1 for r in out["rows"] if r["plain"] and not r["fused"])},
    }
    for k, d in paired.items():
        d["mcnemar_p"] = _mcnemar_exact(d.get("fused_only", d.get("plain_only", 0)),
                                        d.get("door_only", d.get("plain_only", 0)))
    out["arms"] = {"door": hits["door"] / n, "plain": hits["plain"] / n, "fused": hits["fused"] / n}
    out["recall_s"] = rec / n
    out["paired"] = paired
    print("%s: door %.3f | plain re-order %.3f | RRF fused %.3f  (n=%d, chance %.3f, recall@%d %.3f)"
          % (split, out["arms"]["door"], out["arms"]["plain"], out["arms"]["fused"], n,
             out["chance_top1"], s, out["recall_s"]))
    return out


def stage_fused(workers: int = 2, s: int = H.S_DEFAULT, limit: int = 0) -> Dict[str, Any]:
    """Item 3 on FIT and DEV.  The choice is made on FIT only."""
    fit = _orders_for_split("FIT", workers, s,
                            seed_pairs=os.path.join(OUT, "m22_assemble_fit_pairs.jsonl"),
                            limit=limit)
    dev = _orders_for_split("DEV", workers, s,
                            seed_pairs=os.path.join(OUT, "m22_assemble_dev_pairs.jsonl"),
                            limit=limit)
    choice = ("fused" if fit["arms"]["fused"] > fit["arms"]["plain"] else "plain")
    print("chosen on FIT: %s re-order (%s)" % (choice, fit["arms"]))
    return {"fit": fit, "dev": dev, "chosen_on_fit": choice,
            "rule": "the higher FIT top-1 wins; a tie keeps Mission 22's plain re-order"}


# --------------------------------------------------------------------------- item 5: the read
def _read_rows_path() -> str:
    return os.path.join(OUT, "m23_read_rows.jsonl")


def stage_read(workers: int = 2, s: int = H.S_DEFAULT, rerank: str = "fused",
               views: Sequence[str] = H.VIEWS, limit: int = 0,
               max_minutes: float = 0.0) -> Dict[str, Any]:
    """Item 5, written so that a stopped run is still a usable read.

    Every query's pair scores are cached in `m23_read_pairs.jsonl` and its row is appended to
    `m23_read_rows.jsonl` as soon as the query is finished, so the read can be stopped at any
    point and summarised over the queries that actually ran (the registration fixes the order in
    which views are dropped, and `--max-minutes` only ever cuts the end of that order).
    """
    from matching import placement_frontdoor as FD
    from matching import placement_prefilter as PF
    t_start = time.time()
    objs = H.all_objects()
    if limit:
        objs = objs[:int(limit)]
    keys = [o["key"] for o in objs]
    refs = [o["ref"] for o in objs]
    key_idx = {k: j for j, k in enumerate(keys)}
    qlist: List[Tuple[Dict[str, Any], str, str]] = []
    for v in views:                                  # view-major: a partial run is whole views
        for o in objs:
            qlist.append((o, v, o["views"][v]))
    store = H.ensure_images(refs + [q[2] for q in qlist], "all80", workers)
    params = PF.load_params()
    index = FD.PlaceIndex([store[r] for r in refs], keys)
    t0 = time.time()
    app = FD.appearance_scores([q[2] for q in qlist], refs)
    cnt, _d = FD.count_scores(index, [store[q[2]] for q in qlist],
                              n_hits=int(H._n("PLACEMENT_COUNT_HITS")))
    fused = FD.rank_fuse(app, cnt)
    print("door: %.0fs for %d queries" % (time.time() - t0, len(qlist)), flush=True)
    door = [[j for j in sorted(range(len(refs)), key=lambda j: (-fused[i, j], j))]
            for i in range(len(qlist))]
    t1 = time.time()
    dep = [H.deployed_ranking(store, q[2], refs, params) for q in qlist]
    print("deployed prefilter: %.0fs for %d queries" % (time.time() - t1, len(qlist)), flush=True)
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rp = _read_rows_path()
    if os.path.exists(rp):
        for line in open(rp):
            if line.strip():
                r = json.loads(line)
                done[(r["object"], r["view"])] = r
    print("rows: %d of %d cached" % (len(done), len(qlist)), flush=True)
    for i, (o, v, qp) in enumerate(qlist):
        if (o["key"], v) in done:
            continue
        truth = key_idx[o["key"]]
        cd, cp = door[i][:s], dep[i][:s]
        union = sorted(set(cd) | set(cp))
        row: Dict[str, Any] = {"object": o["key"], "cls": o["cls"], "view": v,
                               "truth": truth, "door_s": cd, "deployed_s": cp,
                               "needed_pairs": int(truth in union)}
        if truth in union:
            scores = H.ensure_pairs([(qp, refs[j]) for j in union], "read", workers)
            sc = {j: scores[(qp, refs[j])] for j in union if (qp, refs[j]) in scores}
            asm = (H.order_fused if rerank == "fused" else H.order_plain)(cd, sc)
            dep_o = H.order_plain(cp, sc)
            row.update({"assembled": int(bool(asm) and asm[0] == truth),
                        "deployed": int(bool(dep_o) and dep_o[0] == truth),
                        "assembled_rank": (asm.index(truth) + 1) if truth in asm else None,
                        "deployed_rank": (dep_o.index(truth) + 1) if truth in dep_o else None})
        else:
            # top-1 can only be a shortlisted candidate: both arms are provably wrong here
            row.update({"assembled": 0, "deployed": 0, "assembled_rank": None,
                        "deployed_rank": None})
        # the appearance arm is the appearance CHANNEL's argmax, not the fused door's rank-1
        row["appearance"] = int(int(np.argmax(app[i])) == truth)
        row["door_in_s"] = int(truth in cd)
        row["deployed_in_s"] = int(truth in cp)
        with open(rp, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        done[(o["key"], v)] = row
        if max_minutes and (time.time() - t_start) / 60.0 > float(max_minutes):
            print("stopping after %d queries (max-minutes)" % len(done), flush=True)
            break
    return summarise_read(sorted(done.values(), key=lambda r: (r["view"], r["object"])),
                          views, s, rerank, len(objs))


def summarise_read(rows: Sequence[Dict[str, Any]], views: Sequence[str], s: int,
                   rerank: str, n_objects: int) -> Dict[str, Any]:
    """The registered statistics over whatever rows exist, with n and chance beside each."""
    from Topology.placement_m22_eth80 import _mcnemar_exact
    n = len(rows)
    per_object: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        per_object.setdefault(r["object"], []).append((float(r["assembled"]), float(r["deployed"])))
    agg = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
           for k, v in per_object.items()}
    boot = H.bootstrap_paired(agg) if len(agg) >= 2 else {}
    hits = {a: sum(int(r[a]) for r in rows) for a in ("assembled", "deployed", "appearance")}
    per_class: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = per_class.setdefault(r["cls"], {"n": 0, "assembled": 0, "deployed": 0})
        d["n"] += 1
        d["assembled"] += int(r["assembled"])
        d["deployed"] += int(r["deployed"])
    per_view = {}
    for v in views:
        here = [r for r in rows if r["view"] == v]
        if not here:
            continue
        per_view[v] = {"n": len(here),
                       "assembled": sum(int(r["assembled"]) for r in here) / len(here),
                       "deployed": sum(int(r["deployed"]) for r in here) / len(here),
                       "door_recall_s": sum(int(r["door_in_s"]) for r in here) / len(here)}
    res = {"design": "all 80 objects x %d views; gallery %d references" % (len(views), n_objects),
           "views": list(views), "n_queries": n, "n_objects_covered": len(agg),
           "S": s, "rerank": rerank, "chance_top1": 1.0 / n_objects,
           "chance_recall_s": s / float(n_objects),
           "primary": boot,
           "rates": {"assembled": hits["assembled"] / n if n else None,
                     "deployed": hits["deployed"] / n if n else None,
                     "appearance": hits["appearance"] / n if n else None,
                     "door_recall_s": sum(int(r["door_in_s"]) for r in rows) / n if n else None,
                     "deployed_recall_s": sum(int(r["deployed_in_s"]) for r in rows) / n if n else None},
           "counts": {a: hits[a] for a in hits},
           "queries_without_pairs": int(sum(1 for r in rows if not r["needed_pairs"])),
           "per_view": per_view,
           "per_class": {c: {"n": d["n"], "assembled": d["assembled"] / d["n"],
                             "deployed": d["deployed"] / d["n"]}
                         for c, d in sorted(per_class.items())},
           "rows": list(rows)}
    res["secondary_appearance_p"] = _mcnemar_exact(
        sum(1 for r in rows if r["assembled"] and not r["appearance"]),
        sum(1 for r in rows if r["appearance"] and not r["assembled"])) if n else None
    try:
        pw = json.load(open(os.path.join(OUT, "m23_power.json")))
        per_obj = pw.get("per_object_curve") or {}
        res["mde_reference"] = {"n_curve": pw.get("n_curve"),
                                "per_object_curve": per_obj,
                                "per_object_at_this_read": per_obj.get(
                                    str(max(1, int(round(n / max(1, n_objects)))))),
                                "planning_effect": pw.get("planning_effect")}
    except Exception:                                          # noqa: BLE001
        pass
    if boot:
        print("SUMMARY (%d queries, %d objects): assembled %.3f (%d) | deployed %.3f (%d) "
              "| appearance %.3f | chance %.4f"
              % (n, boot["n_objects"], res["rates"]["assembled"], hits["assembled"],
                 res["rates"]["deployed"], hits["deployed"], res["rates"]["appearance"],
                 res["chance_top1"]))
        print("primary: diff %.3f, 95%% CI [%.3f, %.3f], bootstrap p %.4f"
              % (boot["diff"], boot["ci95"][0], boot["ci95"][1], boot["p_bootstrap"]))
    return res
