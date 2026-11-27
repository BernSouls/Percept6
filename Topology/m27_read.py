"""M27 item 2: settle S with the full read, and item 3: where the matcher loses.

Cell: all 80 objects, query view -035-045, 80-object gallery at the reference view -000-000.
Everything the door needs comes from the cached records; only pair scores cost anything, and the
committed pair files are already in the cache (item 1a), so S = 8 is nearly free.

    python Topology/m27_read.py --plan                 # the pair count, the cache, the estimate
    python Topology/m27_read.py --run --budget-minutes 60   # resumable, progress file
    python Topology/m27_read.py --report --json out.json

Rule, fixed before the numbers (item 2):

    S is chosen on the FIT objects (40); the DEV + FRESH objects (40) report.
    flat S = 16 ships only if the FIT top-1 gain over S = 8 is >= +0.05 AND the confirmation
    gain is >= 0; otherwise S = 8.
    the adaptive rule (M26, margin < 0.007168 -> 16) ships only if it beats BOTH on FIT;
    otherwise it leaves the bundle default.

A query is REPORTED only when every one of its 16 candidate pairs has a score (a partial shortlist
would let the matcher hide the candidates it never saw).  Nothing is ever compared against a
constant, and a query with no score is listed, not zeroed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC  # noqa: E402

REF = "000-000"
VIEW = "035-045"
WIDE = 16
MARGIN_TAU = 0.007168          # the FIT median of (s8 - s9) / s1, fixed in M26
SEED = 20261118
RESAMPLES = 10000
PROGRESS = os.path.join(ROOT, "Topology", "results", "placement", "m27_read_progress.json")


# ------------------------------------------------------------------ the door over cached records
def gallery():
    from matching import placement_frontdoor as FD
    from matching.placement_gallery import Gallery
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    recs = [[REC.load(k, REF)] for k in keys]
    paths = [[REC.path_of(o, REF)] for o in objs]
    idx = FD.PlaceIndex([r[0] for r in recs], keys)
    app = np.asarray([FD.appearance_feature(p[0]) for p in paths])
    return Gallery(keys=keys, views=[REF], paths=paths, records=recs, index=idx,
                   appearance=app, digest="m26-records")


def door_rows(gal) -> List[Dict[str, Any]]:
    """The shipped door's order (top 16) for every query, and the adaptive decision.

    Batched: one appearance pass over both sides, one counted pass over all 80 queries.
    """
    from matching import placement_bundle as PB
    from matching import placement_frontdoor as FD
    b = PB.PlacementBundle.load()
    objs = REC.objects()
    qs = [REC.load(o["key"], VIEW) for o in objs]
    qpaths = [REC.path_of(o, VIEW) for o in objs]
    cnt, _ = FD.count_scores(gal.index, qs, n_hits=int(b.door_n_hits))
    app = FD.appearance_scores(qpaths, gal.flat_paths())
    fused = FD.rank_fuse(app, cnt, k=int(b.rrf_k))
    rows: List[Dict[str, Any]] = []
    for i, o in enumerate(objs):
        order = sorted(range(gal.n), key=lambda j: (-fused[i, j], j))
        # the M26 adaptive rule, evaluated HERE so the arm does not move when the bundle is
        # re-frozen with a different shortlist rule: margin = (s8 - s9) / s1, widen under tau.
        sv = [float(fused[i, j]) for j in order]
        margin = ((sv[7] - sv[8]) / sv[0]) if sv[0] else 0.0
        s_adaptive = WIDE if margin < MARGIN_TAU else 8
        rows.append({"object": o["key"], "cls": o["cls"], "split": o["split"],
                     "truth": gal.keys.index(o["key"]), "query": qpaths[i],
                     "order16": order[:WIDE], "adaptive_s": int(s_adaptive),
                     "door_scores": [float(fused[i, j]) for j in order[:WIDE]]})
    return rows



def pair_keys(gal, rows: Sequence[Dict[str, Any]], same_env: bool = True) -> List[Dict[str, Any]]:
    from matching import placement_recognise as RC
    from matching.placement_bundle import shipped_weight
    w = float(shipped_weight())
    cache = RC.load_pair_cache(same_env=same_env)
    out: List[Dict[str, Any]] = []
    for r in rows:
        for j in r["order16"]:
            rp = gal.paths[int(j)][0]
            key = RC._pair_key(r["query"], rp, w)
            out.append({"object": r["object"], "candidate": int(j), "ref": rp,
                        "query": r["query"], "key": key, "cached": key in cache,
                        "score": cache.get(key)})
    return out


def plan(rows: Optional[Sequence[Dict[str, Any]]] = None,
         same_env: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    gal = gallery()
    rows = list(rows) if rows is not None else door_rows(gal)
    pairs = pair_keys(gal, rows, same_env=same_env)
    todo = [p for p in pairs if not p["cached"]]
    print("gallery %d objects; door %d queries in %.1fs" % (gal.n, len(rows), time.time() - t0),
          flush=True)
    print("pairs: %d asked (16/query), %d cached IN THIS ENVIRONMENT, %d to score"
          % (len(pairs), len(pairs) - len(todo), len(todo)), flush=True)
    return {"gallery": gal, "rows": rows, "pairs": pairs, "todo": todo}


# ------------------------------------------------------------------ scoring
def _warm(images: Sequence[str]) -> Dict[str, Any]:
    """Build/load every unique graph AND side ONCE per process, before any pair is scored."""
    import matching.placement_frontend as FE
    t0 = time.time()
    for p in images:
        FE.cell(p, "m27read")
    return {"images": len(images), "seconds": round(time.time() - t0, 1)}


def query_units(rows: Sequence[Dict[str, Any]], todo: Sequence[Dict[str, Any]]
                ) -> List[Tuple[Dict[str, Any], List[Tuple[str, str, str]]]]:
    """Per-query jobs, in an order that keeps FIT and DEV+FRESH complete at the same rate."""
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for t in todo:
        by_q.setdefault(t["object"], []).append(t)
    fit = [r for r in rows if r["split"] == "FIT" and r["object"] in by_q]
    conf = [r for r in rows if r["split"] != "FIT" and r["object"] in by_q]
    order: List[Dict[str, Any]] = []
    for i in range(max(len(fit), len(conf))):
        if i < len(fit):
            order.append(fit[i])
        if i < len(conf):
            order.append(conf[i])
    out = []
    for r in order:
        jobs = [(t["query"], t["ref"], t["key"]) for t in sorted(by_q[r["object"]],
                                                                 key=lambda t: t["candidate"])]
        out.append((r["object"], jobs))
    return out


def _score_unit(arg: Tuple[str, List[Tuple[str, str, str]]]) -> List[Dict[str, Any]]:
    import matching.placement_frontend as FE
    from matching.placement_bundle import shipped_weight
    from matching.placement_recognise import pair_env
    obj, jobs = arg
    env = pair_env()
    w = float(shipped_weight())
    out: List[Dict[str, Any]] = []
    for qp, rp, key in jobs:
        try:
            sc = float(FE.pair_score(qp, rp))
        except Exception:
            continue
        out.append({"object": obj, "q": qp, "r": rp, "key": key, "B": sc, "w": w, "env": env})
    return out


def run(budget_minutes: float = 0.0, workers: int = 0, limit: int = 0,
        only_split: str = "") -> Dict[str, Any]:
    import multiprocessing as mp
    from matching.placement_records import placed_workers
    from matching import placement_recognise as RC
    workers = int(workers) or placed_workers()
    p = plan()
    gal, rows, todo = p["gallery"], p["rows"], p["todo"]
    if only_split:
        want = {"FIT": ("FIT",), "CONFIRMATION": ("DEV", "FRESH")}[only_split]
        rows = [r for r in rows if r["split"] in want]
        keep = {r["object"] for r in rows}
        todo = [t for t in todo if t["object"] in keep]
    units = query_units(rows, todo)
    if limit:
        units = units[:int(limit)]
    images = sorted({t[0] for _o, js in units for t in js} | {t[1] for _o, js in units for t in js})
    warm = _warm(images)
    print("warm: %d images in %.0fs; %d query units (%d pairs)"
          % (warm["images"], warm["seconds"], len(units),
             sum(len(j) for _o, j in units)), flush=True)
    t0 = time.time()
    budget = float(budget_minutes) * 60.0 if budget_minutes else 1e9
    done_units, done_pairs, failed = 0, 0, 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        it = pool.imap_unordered(_score_unit, units, chunksize=1)
        for res in it:
            done_units += 1
            if not res:
                failed += 1
            else:
                RC.append_pair_cache(res)
                done_pairs += len(res)
            el = time.time() - t0
            prog = {"queries_done": done_units, "of": len(units), "pairs": done_pairs,
                    "seconds": round(el, 1), "failed_units": failed}
            with open(PROGRESS, "w") as fh:
                json.dump(prog, fh)
            print("  query %d/%d (%d pairs, %.0fs, %.0f s/query, failed %d)"
                  % (done_units, len(units), done_pairs, el, el / max(1, done_units), failed),
                  flush=True)
            if el > budget:
                print("budget reached after %d queries -- rerun to continue" % done_units,
                      flush=True)
                pool.terminate()
                break
    out = {"queries_done": done_units, "of": len(units), "pairs": done_pairs,
           "failed_units": failed, "seconds": round(time.time() - t0, 1), "warm": warm,
           "workers": workers}
    with open(PROGRESS, "w") as fh:
        json.dump(out, fh)
    print(json.dumps(out), flush=True)
    return out


def _top1(pairs: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], s: int,
          adaptive: bool = False) -> Dict[str, Any]:
    """Assembled top-1 with S = s (or the adaptive rule), on FULLY scored queries only."""
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_q.setdefault(p["object"], []).append(p)
    per: Dict[str, Tuple[float, float]] = {}
    lacking: List[Dict[str, Any]] = []
    hits = 0
    used = 0
    for r in rows:
        s_eff = r["adaptive_s"] if adaptive else s
        ps = [p for p in by_q.get(r["object"], []) if p["candidate"] in r["order16"][:s_eff]]
        missing = [p for p in ps if p["score"] is None]
        if missing:
            lacking.append({"object": r["object"], "shortlist": s_eff,
                            "unscored": len(missing)})
            continue
        used += 1
        best = min(ps, key=lambda p: (-p["score"], p["candidate"]))
        hit = float(int(best["candidate"] == r["truth"]))
        ceiling = float(int(r["truth"] in r["order16"][:s_eff]))
        hits += int(hit)
        per[r["object"]] = (hit, ceiling)
    return {"n": used, "hits": hits, "top1": (hits / used) if used else None,
            "lacking": lacking, "per_object": per}


def _paired(a: Dict[str, Tuple[float, float]], b: Dict[str, Tuple[float, float]]
            ) -> Dict[str, Any]:
    """Paired, object-clustered difference of two top-1 indicator sets."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return {"n": len(keys), "diff": None}
    x = np.asarray([a[k][0] for k in keys])
    y = np.asarray([b[k][0] for k in keys])
    d = x - y
    rng = np.random.default_rng(SEED)
    boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(RESAMPLES)])
    p = float((np.abs(boot) >= abs(d.mean())).mean())
    return {"n": len(keys), "a": float(x.mean()), "b": float(y.mean()),
            "diff": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_bootstrap": p, "resamples": RESAMPLES, "seed": SEED}


def _one_ci(per_object: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """The arm's top-1 over its scored objects, with an object-clustered bootstrap interval."""
    vals = np.asarray([v[0] for v in per_object.values()])
    if vals.size == 0:
        return {"n": 0, "top1": None}
    rng = np.random.default_rng(SEED)
    boot = np.asarray([vals[rng.integers(0, vals.size, vals.size)].mean()
                       for _ in range(RESAMPLES)])
    return {"n": int(vals.size), "hits": int(vals.sum()), "top1": float(vals.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "resamples": RESAMPLES, "seed": SEED}


def report(json_out: str = "") -> Dict[str, Any]:
    p = plan()
    gal, rows, pairs = p["gallery"], p["rows"], p["pairs"]
    out: Dict[str, Any] = {"cell": {"gallery": gal.n, "view": VIEW, "queries": len(rows),
                                    "chance_top1": 1.0 / gal.n,
                                    "chance_recall_s8": 8.0 / gal.n,
                                    "chance_recall_s16": 16.0 / gal.n},
                           "pairs": {"asked": len(pairs),
                                     "cached": len(pairs) - len(p["todo"]),
                                     "missing": len(p["todo"])},
                           "arms": {}}
    arm = {"s8": _top1(pairs, rows, 8), "s16": _top1(pairs, rows, WIDE),
           "adaptive": _top1(pairs, rows, 8, adaptive=True)}
    for name, res in arm.items():
        per_class: Dict[str, Any] = {}
        for c in sorted({r["cls"] for r in rows}):
            objs = [r["object"] for r in rows if r["cls"] == c and r["object"] in res["per_object"]]
            if objs:
                per_class[c] = {"n": len(objs),
                                "top1": float(np.mean([res["per_object"][o][0] for o in objs])),
                                "ceiling": float(np.mean([res["per_object"][o][1] for o in objs]))}
        out["arms"][name] = dict(_one_ci(res["per_object"]),
                                 ceiling=(float(np.mean([v[1] for v in res["per_object"].values()]))
                                          if res["per_object"] else None),
                                 by_class=per_class, lacking=res["lacking"],
                                 scored_objects=sorted(res["per_object"]))
    fit_objs = {r["object"] for r in rows if r["split"] == "FIT"}
    conf_objs = {r["object"] for r in rows if r["split"] in ("DEV", "FRESH")}

    def sub(name: str, objs: set) -> Dict[str, Tuple[float, float]]:
        return {k: v for k, v in arm[name]["per_object"].items() if k in objs}

    splits: Dict[str, Any] = {}
    for label, objs in (("FIT", fit_objs), ("CONFIRMATION", conf_objs), ("ALL", fit_objs | conf_objs)):
        splits[label] = {name: _one_ci(sub(name, objs)) for name in arm}
    out["splits"] = splits
    comparisons: Dict[str, Any] = {}
    for split, objs in (("FIT", fit_objs), ("CONFIRMATION", conf_objs)):
        for other in ("s16", "adaptive"):
            comparisons["%s|%s_vs_s8" % (split, other)] = _paired(sub(other, objs), sub("s8", objs))
    out["comparisons"] = comparisons

    # ---- the S rule, applied mechanically
    def gain(arm_name: str, split: str) -> Optional[float]:
        a = splits[split].get(arm_name, {})
        return (None if a.get("top1") is None or splits[split]["s8"].get("top1") is None
                else float(a["top1"] - splits[split]["s8"]["top1"]))

    rule: Dict[str, Any] = {"pairs_adaptive_sum": int(sum(r["adaptive_s"] for r in rows)),
                            "queries_widened": int(sum(1 for r in rows if r["adaptive_s"] > 8)),
                            "fit_gain_s16": gain("s16", "FIT"),
                            "conf_gain_s16": gain("s16", "CONFIRMATION"),
                            "fit_adaptive_minus_s8": gain("adaptive", "FIT"),
                            "fit_adaptive_minus_s16": (
                                None if splits["FIT"]["adaptive"].get("top1") is None
                                else float(splits["FIT"]["adaptive"]["top1"]
                                           - splits["FIT"]["s16"]["top1"]))}
    rule["s16_ships"] = bool(rule["fit_gain_s16"] is not None
                             and rule["fit_gain_s16"] >= 0.05
                             and (rule["conf_gain_s16"] or 0.0) >= 0.0)
    rule["adaptive_ships"] = bool(rule["fit_adaptive_minus_s8"] is not None
                                  and rule["fit_adaptive_minus_s8"] > 0.0
                                  and (rule["fit_adaptive_minus_s16"] or -1.0) > 0.0)
    rule["shipped"] = ("s16" if rule["s16_ships"] else "s8")
    rule["adaptive_in_bundle"] = bool(rule["adaptive_ships"])
    out["rule"] = rule

    # ---- item 3: where the matcher loses, on the queries the door already found
    losses: List[Dict[str, Any]] = []
    per_class_loss: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if r["object"] not in arm["s16"]["per_object"]:
            continue
        ps = [x for x in pairs if x["object"] == r["object"]
              and x["candidate"] in r["order16"][:WIDE] and x["score"] is not None]
        if not ps:
            continue
        best = min(ps, key=lambda x: (-x["score"], x["candidate"]))
        truth_score = next((x["score"] for x in ps if x["candidate"] == r["truth"]), None)
        if truth_score is None or int(best["candidate"]) == r["truth"]:
            continue
        win_cls = gal.keys[int(best["candidate"])].split("/")[0]
        same = win_cls == r["cls"]
        d = per_class_loss.setdefault(r["cls"], {"same": 0, "different": 0})
        d["same" if same else "different"] += 1
        losses.append({"object": r["object"], "cls": r["cls"],
                       "winner": gal.keys[int(best["candidate"])], "winner_cls": win_cls,
                       "same_class": bool(same), "truth_score": truth_score,
                       "winner_score": float(best["score"]),
                       "gap": float(truth_score - best["score"]),
                       "same_class_in_shortlist": int(sum(
                           1 for j in r["order16"][:WIDE]
                           if gal.keys[j].split("/")[0] == r["cls"]))})
    found = arm["s16"]["n"]
    out["losses"] = {"scored_queries": found,
                     "truth_reachable": int(sum(1 for r in rows
                                                if r["object"] in arm["s16"]["per_object"]
                                                and r["truth"] in r["order16"][:WIDE])),
                     "losses": len(losses),
                     "same_class": sum(1 for x in losses if x["same_class"]),
                     "different_class": sum(1 for x in losses if not x["same_class"]),
                     "median_gap": (float(np.median([x["gap"] for x in losses]))
                                    if losses else None),
                     "median_same_class_in_shortlist": (
                         float(np.median([x["same_class_in_shortlist"] for x in losses]))
                         if losses else None),
                     "by_class": per_class_loss, "rows": losses}
    print(json.dumps({k: v for k, v in out.items() if k != "losses"}, indent=1))
    print(json.dumps({k: v for k, v in out["losses"].items() if k != "rows"}, indent=1))
    if json_out:
        with open(json_out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", json_out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget-minutes", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-split", default="", choices=["", "FIT", "CONFIRMATION"])
    ap.add_argument("--json", default=os.path.join(ROOT, "Topology", "results", "placement",
                                                   "m27_read.json"))
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    if a.plan:
        plan()
    if a.run:
        run(a.budget_minutes, a.workers, a.limit, a.only_split)
    if a.report:
        report(a.json)
    if not (a.plan or a.run or a.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
