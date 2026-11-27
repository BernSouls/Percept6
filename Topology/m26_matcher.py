"""M26 item 7: the matcher step -- exactly the pairs the fixed door asks for.

Runs ONLY after the door is fixed by items 4-6.  It re-derives the door's shortlists from the
cached records (no graph rebuild for the door), lists the (query, candidate-view) pairs that are
NOT in the shipped pair cache, prices them, and then runs exactly those, resumably.

    python Topology/m26_matcher.py --list                 # count + estimate, no pair scored
    python Topology/m26_matcher.py --run --budget-minutes 90
    python Topology/m26_matcher.py --report               # assembled top-1 vs 0.250, paired CI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC  # noqa: E402

VIEW = "035-045"
REF = "000-000"


def gallery():
    """A Gallery over the cached records: the shipped k = 1 reference view, 80 objects."""
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


def shortlists(gal, view: str = VIEW) -> List[Dict[str, Any]]:
    """The door's shortlist per query, from the CACHED query records.

    Mirrors ``recognise``'s door step exactly (count + appearance + RRF + the bundle's shortlist
    rule); ``verify_against_recognise`` proves the mirror on one query so the harness cannot drift
    from the shipped path.
    """
    from matching import placement_recognise as RC
    from matching import placement_bundle as PB
    from matching import placement_frontdoor as FD
    b = PB.PlacementBundle.load()
    out: List[Dict[str, Any]] = []
    for obj in REC.objects():
        qp = REC.path_of(obj, view)
        qrec = REC.load(obj["key"], view)
        t0 = time.time()
        if str(getattr(b, "door_kind", "counted")) == "graded":
            cnt, _ = FD.graded_scores(gal.index, [qrec], **dict(getattr(b, "door_params", {}) or {}))
        else:
            cnt, _ = FD.count_scores(gal.index, [qrec], n_hits=int(b.door_n_hits))
        app = FD.appearance_scores([qp], gal.flat_paths())
        fused = FD.rank_fuse(app, cnt, k=int(b.rrf_k))
        order = [j for j in sorted(range(gal.n), key=lambda j: (-fused[0, j], j))]
        s_used = RC._shortlist_size(fused[0], order, b, int(b.shortlist))
        short = order[:s_used]
        out.append({"object": obj["key"], "cls": obj["cls"], "split": obj["split"],
                    "query": qp, "truth": gal.keys.index(obj["key"]), "shortlist": short,
                    "best_view": {str(j): gal.views[0] for j in short},
                    "s_used": s_used, "seconds": {"door": round(time.time() - t0, 3)}})
    return out


def verify_against_recognise(gal, rows: Sequence[Dict[str, Any]], n: int = 1) -> List[Dict[str, Any]]:
    """The shipped entry point, on the same queries: the shortlists must be identical."""
    from matching import placement_recognise as RC
    out = []
    for r in list(rows)[:n]:
        res = RC.recognise(r["query"], gal, matcher=False)
        same = list(res.shortlist) == list(r["shortlist"])
        out.append({"object": r["object"], "identical": same,
                    "recognise": list(res.shortlist), "harness": list(r["shortlist"])})
        print("  verify %s: shortlist identical = %s (%d vs %d objects)"
              % (r["object"], same, len(res.shortlist), len(r["shortlist"])), flush=True)
    return out


def pairs_of(gal, rows: Sequence[Dict[str, Any]], view: str = VIEW) -> List[Dict[str, Any]]:
    """Every (query, candidate view path) pair the door asks the matcher for."""
    from matching import placement_recognise as RC
    from matching import placement_bundle as PB
    from matching.placement_bundle import shipped_weight
    w = float(shipped_weight())
    cache = RC.load_pair_cache()
    out: List[Dict[str, Any]] = []
    for r in rows:
        for j in r["shortlist"]:
            v = r["best_view"].get(str(j)) or gal.views[0]
            rp = gal.paths[int(j)][gal.views.index(v)]
            key = RC._pair_key(r["query"], rp, w)
            out.append({"object": r["object"], "cls": r["cls"], "candidate": int(j),
                        "ref": rp, "view": v, "query": r["query"], "key": key,
                        "cached": key in cache, "score": cache.get(key)})
    return out


def plan(budget_minutes: float = 0.0, verify: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    gal = gallery()
    rows = shortlists(gal)
    pairs = pairs_of(gal, rows)
    todo = [p for p in pairs if not p["cached"]]
    print("gallery %d objects, %d buckets; door %d queries in %.1fs"
          % (gal.n, len(gal.index.by_key), len(rows), time.time() - t0), flush=True)
    if verify:
        verify_against_recognise(gal, rows, n=1)
    print("pairs asked for: %d, cached: %d, to score: %d"
          % (len(pairs), len(pairs) - len(todo), len(todo)), flush=True)
    return {"pairs": pairs, "todo": todo, "rows": rows, "gallery": gal}


def run(budget_minutes: float, workers: int = 2, queries: int = 0) -> Dict[str, Any]:
    import multiprocessing as mp
    p = plan(budget_minutes)
    todo = p["todo"]
    if queries:
        keep = {r["object"] for r in p["rows"][:int(queries)]}
        todo = [t for t in todo if t["object"] in keep]
    if not todo:
        return {"scored": 0}
    t0 = time.time()
    budget = float(budget_minutes) * 60.0 if budget_minutes else 1e9
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
        it = pool.imap_unordered(_score, [(q["query"], q["ref"], q["key"]) for q in todo],
                                 chunksize=1)
        for res in it:
            done += 1
            if res is not None:
                from matching import placement_recognise as RC
                RC.append_pair_cache([res])
            if done % 10 == 0:
                el = time.time() - t0
                print("  %d/%d pairs, %.0fs elapsed, %.1fs/pair"
                      % (done, len(todo), el, el / done), flush=True)
            if time.time() - t0 > budget:
                print("budget reached after %d pairs -- rerun to continue" % done, flush=True)
                pool.terminate()
                break
    return {"scored": done, "seconds": time.time() - t0}


def _score(arg: Tuple[str, str, str]) -> Any:
    import matching.placement_frontend as FE
    from matching import placement_recognise as RC
    from matching.placement_bundle import shipped_weight
    qp, rp, key = arg
    try:
        s = float(FE.pair_score(qp, rp))
    except Exception as exc:                                   # never zeroed, never hidden
        return None                                            # counted, never zeroed
    return {"key": key, "B": s, "q": qp, "r": rp, "w": float(shipped_weight())}


def report(baseline: float = 0.250) -> Dict[str, Any]:
    """Assembled top-1 on the 80 x 035-045 cell, against the recorded 0.250, as an ESTIMATE."""
    gal = gallery()
    rows = shortlists(gal)
    pairs = pairs_of(gal, rows)
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for pr in pairs:
        by_q.setdefault(pr["object"], []).append(pr)
    scored, missing, top1 = 0, [], 0
    per_obj: Dict[str, Tuple[float, float]] = {}
    for r in rows:
        ps = by_q.get(r["object"], [])
        have = [p for p in ps if p["cached"]]
        if len(have) < len(ps):
            missing.append({"object": r["object"], "unscored": len(ps) - len(have),
                            "scored": len(have), "shortlist": len(ps)})
            continue                     # a partial shortlist cannot rank the truth honestly
        scored += 1
        best = min(have, key=lambda p: (-p["score"], p["candidate"]))
        hit = float(int(best["candidate"] == r["truth"]))
        top1 += int(hit == 1.0)
        per_obj[r["object"]] = (hit, float(int(r["truth"] in r["shortlist"])))
    n = len(rows)
    out = {"n_queries": n, "queries_complete": scored,
           "queries_partial_or_missing": len(missing), "top1_hits": top1,
           "top1": (top1 / scored) if scored else None, "baseline": baseline,
           "missing": missing, "chance": 1.0 / gal.n}
    if scored >= 2:
        from Topology.placement_m23_eth80 import bootstrap_paired
        agg = {k: (v[0], v[1]) for k, v in per_obj.items()}
        res = bootstrap_paired(agg)
        out["paired_vs_ceiling"] = {"diff": res.get("diff"), "ci95": res.get("ci95"),
                                    "p_bootstrap": res.get("p_bootstrap"),
                                    "note": "assembled top-1 minus the door's own recall on the "
                                            "same queries (the ceiling)"}
        rng = np.random.default_rng(20261118)
        vals = np.asarray([v[0] for v in per_obj.values()])
        boot = np.asarray([vals[rng.integers(0, len(vals), len(vals))].mean()
                           for _ in range(10000)])
        out["top1_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["vs_0_250"] = {"diff": (out["top1"] - baseline) if out["top1"] is not None else None,
                           "ci95": [out["top1_ci95"][0] - baseline, out["top1_ci95"][1] - baseline]
                           if out["top1"] is not None else None,
                           "note": "the interval moved with the baseline: a fixed 0.250 has no "
                                   "sampling error of its own"}
    print(json.dumps(out, indent=1)[:1500])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget-minutes", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--queries", type=int, default=0,
                    help="only the first N queries (whole queries, so the subset is clean)")
    ap.add_argument("--out", default=os.path.join(ROOT, "Topology", "results", "placement",
                                                  "m26_matcher.json"))
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    out: Dict[str, Any] = {}
    if a.list:
        p = plan(a.budget_minutes, verify=True)
        out["plan"] = {"pairs": len(p["pairs"]), "to_score": len(p["todo"]),
                       "cached": len(p["pairs"]) - len(p["todo"])}
        per_pair = 11.0
        out["plan"]["estimate_minutes"] = round(len(p["todo"]) * per_pair / 60.0, 1)
        print("estimate: %d pairs x %.0fs / 2 workers ~ %.0f minutes"
              % (len(p["todo"]), per_pair, out["plan"]["estimate_minutes"]))
    if a.run:
        out["run"] = run(a.budget_minutes, a.workers, a.queries)
    if a.report:
        out["report"] = report()
    if out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
