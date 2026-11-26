"""MISSION 25 -- the one read: does storing four views per object raise the door's ceiling?

Stages:

  seed    import Mission 23's cached pair scores into the core pair cache (keyed by the matcher's
          own digest), so the k = 1 arm on the `035-045` view costs nothing.  Reported, not hidden.
  read    for each arm (k = 1, k = 4) and each query view, run `placement_recognise.recognise` and
          write one row per query.  Resumable: a row already in the file is not recomputed, and a
          stopped read is a smaller read.
  summary the registered statistics over whatever rows exist: door recall@8 and assembled top-1
          per arm and view, the ceiling (truth in the shortlist), per class, per object; two
          primaries with a paired object-clustered bootstrap (10,000 resamples, seed 20261118) and
          Holm on the pair; seconds per query for the door and the matcher.

The design, the MDE and the decision rule were frozen in `MISSION25_PREREGISTRATION.md` before this
ran.  Nothing here re-fits anything: the arms differ only in the stored view set.

    python Topology/placement_m25_read.py --stage seed
    python Topology/placement_m25_read.py --stage read --arms k1,k4 --views 035-045,045-180
    python Topology/placement_m25_read.py --stage summary
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]

OUT = os.path.join(ROOT, "Topology", "results", "placement")
ROWS = os.path.join(OUT, "m25_read_rows.jsonl")
S_DEFAULT = 8
VIEWS = ("035-045", "045-180")          # the read's query views; never stored (see item 1)
BASE = "000-000"
PLANNING_EFFECT = 0.20                  # 0.55 -> 0.75 recall; the MDE must beat this to be a read


# --------------------------------------------------------------------------- objects and paths
def objects() -> List[Dict[str, Any]]:
    from Topology.placement_m23_eth80 import all_objects
    return all_objects()


def path_of(obj: Dict[str, Any], view: str) -> str:
    """The image for one object on one view label (`035-045` -> `<name>-035-045.png`)."""
    ref = obj["ref"]
    folder = os.path.dirname(ref)
    label = os.path.basename(ref).split("-")[0]
    p = os.path.join(folder, "%s-%s.png" % (label, view))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def view_set(k: int) -> List[str]:
    from matching import placement_views as V
    return V.reference_views(int(k), VIEWS)


def items_for(objs: Sequence[Dict[str, Any]], views: Sequence[str]) -> List[Tuple[str, Dict[str, str]]]:
    return [(o["key"], {v: path_of(o, v) for v in views}) for o in objs]


# --------------------------------------------------------------------------- the seed
def stage_seed() -> Dict[str, Any]:
    """Import Mission 23's pair scores into the core cache (matcher-digest keyed)."""
    from matching import placement_recognise as RC
    src = os.path.join(OUT, "m23_read_pairs.jsonl")
    rows: List[Dict[str, Any]] = []
    seen = set()
    if os.path.exists(src):
        for line in open(src):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            key = (os.path.basename(r["q"]), os.path.basename(r["r"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"q": r["q"], "r": r["r"], "w": 0.375, "B": float(r["B"])})
    RC.append_pair_cache(rows)
    n = len(RC.load_pair_cache())
    print("seed: imported %d Mission-23 pairs; the cache now holds %d entries for matcher %s"
          % (len(rows), n, RC.matcher_digest()), flush=True)
    return {"imported": len(rows), "cache_entries": n, "matcher_digest": RC.matcher_digest()}


# --------------------------------------------------------------------------- the read
def _rows() -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    done: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if os.path.exists(ROWS):
        for line in open(ROWS):
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["object"], r["view"])] = r
    return done


def stage_read(arms: Sequence[str] = ("k1", "k4"), views: Sequence[str] = VIEWS,
               workers: int = 2, limit: int = 0, max_minutes: float = 0.0,
               s: int = S_DEFAULT) -> Dict[str, Any]:
    from matching import placement_recognise as RC
    t_start = time.time()
    objs = objects()
    if limit:
        objs = objs[:int(limit)]
    keys = [o["key"] for o in objs]
    key_idx = {k: j for j, k in enumerate(keys)}
    done = _rows()
    print("rows: %d already on disk" % len(done), flush=True)
    for arm in arms:
        k = int(arm[-1])
        vs = view_set(k)
        t0 = time.time()
        gallery = RC.build_gallery(items_for(objs, vs), workers=workers, views=vs)
        print("[%s] gallery %.0fs views %s (%d objects x %d views)"
              % (arm, time.time() - t0, gallery.views, gallery.n, gallery.n_views), flush=True)
        for view in views:
            for o in objs:
                if (arm, o["key"], view) in done:
                    continue
                if max_minutes and (time.time() - t_start) / 60.0 > float(max_minutes):
                    print("stopping after %d rows (max-minutes)" % len(done), flush=True)
                    return {"stopped": True, "rows": len(done)}
                qp = path_of(o, view)
                truth = key_idx[o["key"]]
                try:
                    res = RC.recognise(qp, gallery, workers=workers, s=s)
                except Exception as exc:                    # a worker died (the box OOMed)
                    if workers <= 1:
                        raise
                    print("  pool died (%s): falling back to one worker for the rest of the read"
                          % type(exc).__name__, flush=True)
                    workers = 1
                    res = RC.recognise(qp, gallery, workers=workers, s=s)
                short = list(res.shortlist)
                rank = (res.ranking.index(truth) + 1) if truth in res.ranking else None
                door_rank = (short.index(truth) + 1) if truth in short else None
                row = {"arm": arm, "k": k, "views_stored": list(gallery.views),
                       "view": view, "object": o["key"], "cls": o["cls"], "truth": truth,
                       "shortlist": short, "door_order_head": list(res.door_order[:s]),
                       "door_s": short,
                       "in_shortlist": int(truth in short),
                       "assembled": int(bool(res.ranking) and res.ranking[0] == truth),
                       "assembled_rank": rank, "door_rank": door_rank,
                       "best_view": {str(j): v for j, v in res.best_view.items()},
                       "best_view_truth": res.best_view.get(truth),
                       "scores": {str(j): round(v, 6) for j, v in res.scores.items()},
                       "unscored": res.unscored,
                       "sec": {kk: round(vv, 2) for kk, vv in res.seconds.items()}}
                with open(ROWS, "a") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                done[(arm, o["key"], view)] = row
                if len(done) % 10 == 0:
                    print("  %d rows, %s/%s %.0fs" % (len(done), arm, view,
                                                      time.time() - t_start), flush=True)
        del gallery                     # one gallery in memory at a time (the box has 4 GB)
        gc.collect()
    print("read: %d rows total in %.0fs, workers ended at %d"
          % (len(done), time.time() - t_start, workers), flush=True)
    return {"rows": len(done), "workers": workers}


# --------------------------------------------------------------------------- the summary
def _holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a small family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * float(p))
        running = max(running, adj)
        out[name] = running
    return out


def summarise(rows: Sequence[Dict[str, Any]], s: int = S_DEFAULT, n_objects: int = 80,
              views: Sequence[str] = VIEWS) -> Dict[str, Any]:
    from Topology.placement_m23_eth80 import bootstrap_paired as boot
    chance_door = s / float(n_objects)
    chance_top1 = 1.0 / float(n_objects)
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        per_view: Dict[str, Any] = {}
        for v in views:
            rs = [r for r in sub if r["view"] == v]
            if not rs:
                continue
            per_view[v] = {"n": len(rs),
                           "door_recall_s": sum(r["in_shortlist"] for r in rs) / len(rs),
                           "assembled_top1": sum(r["assembled"] for r in rs) / len(rs),
                           "door_hits": sum(r["in_shortlist"] for r in rs),
                           "top1_hits": sum(r["assembled"] for r in rs),
                           "chance_door": chance_door, "chance_top1": chance_top1}
        per_class: Dict[str, Dict[str, Any]] = {}
        for c in sorted({r["cls"] for r in sub}):
            rs = [r for r in sub if r["cls"] == c]
            per_class[c] = {"n": len(rs),
                            "door_recall_s": sum(r["in_shortlist"] for r in rs) / len(rs),
                            "assembled_top1": sum(r["assembled"] for r in rs) / len(rs)}
        sec = {"query_record": float(np.mean([r["sec"].get("query_record", 0.0) for r in sub])),
               "door": float(np.mean([r["sec"]["door"] for r in sub])),
               "matcher": float(np.mean([r["sec"]["matcher"] for r in sub])),
               "total": float(np.mean([r["sec"]["total"] for r in sub]))}
        winners: Dict[str, Any] = {}
        for v in views:
            rs = [r for r in sub if r["view"] == v and r.get("best_view_truth")]
            if not rs:
                continue
            hit = [r for r in rs if r["in_shortlist"]]
            miss = [r for r in rs if not r["in_shortlist"]]
            winners[v] = {
                "hits": {x: sum(1 for r in hit if r["best_view_truth"] == x)
                         for x in sorted({r["best_view_truth"] for r in hit})},
                "misses": {x: sum(1 for r in miss if r["best_view_truth"] == x)
                           for x in sorted({r["best_view_truth"] for r in miss})},
                "hit_by_rank": {str(k): sum(1 for r in hit
                                            if (r["door_rank"] or 99) == k) for k in range(1, s + 1)},
            }
        nv = [r["best_view_truth"] for r in sub if r.get("best_view_truth")]
        per_arm[arm] = {"n": len(sub), "per_view": per_view, "per_class": per_class,
                        "seconds_per_query": sec,
                        "best_view_truth_counts": {str(x): nv.count(x) for x in sorted(set(nv))},
                        "view_winners": winners,
                        "truth_best_view_is_base": (sum(1 for x in nv if x == BASE) / len(nv))
                        if nv else None}
    # the two primaries: k4 - k1, paired, object-clustered, over COMPLETE cells only
    # (prereg amendments 6-7: a cell is (arm, view) and counts only when every object has a row)
    cells: Dict[str, Dict[str, int]] = {}
    for r in rows:
        cells.setdefault(r["view"], {}).setdefault(r["arm"], 0)
        cells[r["view"]][r["arm"]] += 1
    complete = [v for v in views
                if cells.get(v, {}).get("k1") == n_objects and cells.get(v, {}).get("k4") == n_objects]
    primaries: Dict[str, Any] = {}
    per_view_primaries: Dict[str, Any] = {}
    if complete:
        for metric, key in (("door_recall_s", "in_shortlist"), ("assembled_top1", "assembled")):
            def _agg(vs):
                per_obj: Dict[str, Dict[str, float]] = {}
                for r in rows:
                    if r["view"] in vs:
                        per_obj.setdefault(r["object"], {}).setdefault(r["arm"], []).append(float(r[key]))
                out: Dict[str, Tuple[float, float]] = {}
                for obj, d in per_obj.items():
                    a, b = d.get("k4", []), d.get("k1", [])
                    if a and b and len(a) == len(b):
                        out[obj] = (float(np.mean(a)), float(np.mean(b)))
                return out
            agg = _agg(complete)
            if len(agg) >= 2:
                res = boot(agg)
                primaries[metric] = {"n_objects": res.get("n_objects"), "per_obj": len(complete),
                                     "cells": list(complete),
                                     "k4": res.get("assembled"), "k1": res.get("deployed"),
                                     "diff": res.get("diff"), "ci95": res.get("ci95"),
                                     "p_bootstrap": res.get("p_bootstrap")}
            for v in complete:
                a_v = _agg([v])
                if len(a_v) >= 2:
                    res = boot(a_v)
                    per_view_primaries.setdefault(v, {})[metric] = {
                        "n_objects": res.get("n_objects"), "k4": res.get("assembled"),
                        "k1": res.get("deployed"), "diff": res.get("diff"),
                        "ci95": res.get("ci95"), "p_bootstrap": res.get("p_bootstrap")}
    if len(primaries) == 2:
        adj = _holm({k: float(v["p_bootstrap"]) for k, v in primaries.items()})
        for k, v in primaries.items():
            v["p_holm"] = adj[k]
    # the MDE the realised design actually buys (mission-23 power simulation, w = 0.5)
    mde_info: Dict[str, Any] = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "m23power", str(Path(__file__).resolve().parent.parent / "archive" / "Topology"
                             / "placement_m23_power.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n_obj, per_obj in sorted({(int(v["n_objects"]), int(v["per_obj"]))
                                      for v in primaries.values()}):
            mde_info["n=%d/per_obj=%d" % (n_obj, per_obj)] = float(
                mod.mde(0.5, n_obj=n_obj, per_obj=per_obj))
    except Exception as exc:                                   # pragma: no cover - reporting aid
        mde_info["error"] = repr(exc)
    return {"design": "80 objects x %d query views; S = %d objects" % (len(views), s),
            "views": list(views), "arms": per_arm, "primaries": primaries,
            "complete_cells": complete, "per_view_primaries": per_view_primaries,
            "mde": mde_info, "planning_effect": PLANNING_EFFECT,
            "chance": {"door_recall_s": chance_door, "top1": chance_top1},
            "resamples": 10000, "seed": 20261118, "n_objects": n_objects,
            "rows": list(rows)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="summary", choices=("seed", "read", "summary", "all"))
    ap.add_argument("--arms", default="k1,k4")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--s", type=int, default=S_DEFAULT)
    a = ap.parse_args(list(argv) if argv is not None else None)
    views = tuple(x for x in a.views.split(",") if x)
    arms = tuple(x for x in a.arms.split(",") if x)
    if a.stage in ("seed", "all"):
        stage_seed()
    if a.stage in ("read", "all"):
        stage_read(arms=arms, views=views, workers=a.workers, limit=a.limit,
                   max_minutes=a.max_minutes, s=a.s)
    if a.stage in ("summary", "all"):
        from matching import m15_manifest as MF
        rows = sorted(_rows().values(), key=lambda r: (r["arm"], r["view"], r["object"]))
        out = summarise(rows, s=a.s, views=views)
        MF.write_result(os.path.join(OUT, "m25_read.json"), out)
        for arm, d in out["arms"].items():
            print("%-3s n=%-4d door recall@8 %.3f  top-1 %.3f  (chance %.3f / %.4f)"
                  % (arm, d["n"], np.mean([v["door_recall_s"] for v in d["per_view"].values()]),
                     np.mean([v["assembled_top1"] for v in d["per_view"].values()]),
                     out["chance"]["door_recall_s"], out["chance"]["top1"]))
            for v, pv in d["per_view"].items():
                print("    %-8s n=%-3d door %.3f (%d)  top-1 %.3f (%d)  sec door %.1f matcher %.1f"
                      % (v, pv["n"], pv["door_recall_s"], pv["door_hits"],
                         pv["assembled_top1"], pv["top1_hits"],
                         d["seconds_per_query"]["door"], d["seconds_per_query"]["matcher"]))
        for metric, p in out["primaries"].items():
            print("PRIMARY %-14s k4 %.3f vs k1 %.3f  diff %+.3f  CI [%.3f, %.3f]  p %.4f  holm %.4f"
                  % (metric, p["k4"], p["k1"], p["diff"], p["ci95"][0], p["ci95"][1],
                     p["p_bootstrap"], p.get("p_holm", float("nan"))))
        print("wrote m25_read.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
