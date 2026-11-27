"""M30 A3: ONE read at S = 8 -- the A1 fused door, the A2 colour blend in the matcher.

    python Topology/m30_read.py --run --budget-minutes 120    # resumable, progress file
    python Topology/m30_read.py --report

Cell: 80 objects, query view -035-045, gallery -000-000 (one stored view), S = 8, chance 0.0125.
Door: RRF(counted, appearance, colour at weight 0.5) -- A1's fused door.
Matcher: the shipped pair score with `PLACEMENT_NODE_COLOUR` on at `PLACEMENT_NODE_COLOUR_W = 0.10`
(A2's three-term blend), unless `--no-colour` is given, in which case the shipped matcher runs.

MDE, stated BEFORE the read: 80 paired queries, an expected discordance of ~0.15 between two
top-1 arms gives SE(diff) = sqrt(0.15/80) = 0.043 and MDE(80 % power, alpha .05 two-sided) =
2.80 x 0.043 = **0.121**.  The planning effect is +0.075 (the door rule's margin, and the size of
the gain M27 measured between arms).  **MDE > effect**: this read therefore reports an ESTIMATE
with its object-clustered CI and makes no claim of improvement.

Query is REPORTED only when all 8 of its pairs are scored (never against a constant).
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

from Topology import m26_records as REC                    # noqa: E402
from Topology.m26_oracle import by_class, metrics           # noqa: E402
from Topology.m27_read import RESAMPLES, SEED, _one_ci      # noqa: E402

REF = "000-000"
VIEW = "035-045"
S = 8
COLOUR_W = 0.10
DOOR_COLOUR_W = 0.5
RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
PROGRESS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_progress.json")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m30_read.json")
M27_S8_TOP1 = 0.2375          # M27's shipped read, same cell, S = 8, n = 80
MDE = 0.121


def door(arm_colour: bool = True) -> List[Dict[str, Any]]:
    """The A1 fused door's top-8 for every query, from the committed records."""
    from matching import placement_frontdoor as FD
    from matching import placement_fuse as FU
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cells = PR.load_cells(RECORDS, [REF, VIEW])
    gal = [cells[(k, REF)] for k in keys]
    qs = [cells[(k, VIEW)] for k in keys]
    index = FD.PlaceIndex(gal, keys)
    counted, _ = FD.count_scores(index, qs, n_hits=8)
    app = FD.appearance_scores([REC.path_of(o, VIEW) for o in objs],
                               [REC.path_of(o, REF) for o in objs])
    colour = FU.colour_channel(gal, qs)
    fused = FU.fuse(counted, app, colour if arm_colour else None,
                    weights={"colour": DOOR_COLOUR_W if arm_colour else 0.0})
    app_only = FU.fuse(np.zeros_like(counted), app)          # the shipped door's appearance channel
    shipped = FU.fuse(counted, app)
    rows: List[Dict[str, Any]] = []
    for i, o in enumerate(objs):
        order = sorted(range(len(keys)), key=lambda j: (-fused[i, j], j))[:S]
        ship_order = sorted(range(len(keys)), key=lambda j: (-shipped[i, j], j))[:S]
        rows.append({"object": o["key"], "cls": o["cls"], "split": o["split"],
                     "truth": keys.index(o["key"]), "query": REC.path_of(o, VIEW),
                     "order": order, "shipped_order": ship_order,
                     "door_scores": [float(fused[i, j]) for j in order],
                     "shipped_scores": [float(shipped[i, j]) for j in ship_order]})
    return rows


def _score_unit(arg: Tuple[str, List[Tuple[str, str]], bool, float]
                ) -> List[Dict[str, Any]]:
    """One query: its S pairs, scored with the A2 blend (or the shipped matcher)."""
    import parameters as P
    import matching.placement_frontend as FE
    obj, jobs, use_colour, w = arg
    FE.apply_recipe()
    # The graph cache key is the switch digest, i.e. the VALUE of every declared front-end switch
    # -- including PLACEMENT_NODE_COLOUR.  Warm this query's graphs with the switch OFF (the digest
    # the warm stage built under), then turn the blend on: `cell()` memoises by path, so the blend
    # runs on the shipped graphs instead of rebuilding them under a second digest.
    P.PLACEMENT_NODE_COLOUR = False
    for p in sorted({q for q, _r in jobs} | {r for _q, r in jobs}):
        FE.cell(p, "m30read")
    if use_colour:
        P.PLACEMENT_NODE_COLOUR = True
        P.PLACEMENT_NODE_COLOUR_W = float(w)
    out: List[Dict[str, Any]] = []
    for qp, rp in jobs:
        try:
            sc = float(FE.pair_score(qp, rp))
        except Exception as exc:                     # counted, never silent
            out.append({"object": obj, "q": qp, "r": rp, "score": None, "error": repr(exc)})
            continue
        out.append({"object": obj, "q": qp, "r": rp, "score": sc, "colour": bool(use_colour),
                    "colour_w": float(w) if use_colour else 0.0})
    return out


def _load_pairs() -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if os.path.exists(PAIRS):
        for line in open(PAIRS, encoding="utf-8"):
            r = json.loads(line)
            if r.get("score") is not None:
                out[(r["object"], r["r"])] = float(r["score"])
    return out


def run(budget_minutes: float = 120.0, workers: int = 2, use_colour: bool = True,
        limit: int = 0, split: str = "") -> Dict[str, Any]:
    import multiprocessing as mp
    from matching.placement_records import placed_workers
    workers = int(workers) or placed_workers()
    rows = door(use_colour)
    if split:
        rows = sorted(rows, key=lambda r: (r["split"].upper() != split.upper(), r["object"]))
    done = _load_pairs()
    units: List[Tuple[str, List[Tuple[str, str]], bool, float]] = []
    for r in rows:
        jobs = [(r["query"], _ref_path(j)) for j in r["order"]
                if (r["object"], _ref_path(j)) not in done]
        if jobs:
            units.append((r["object"], jobs, use_colour, COLOUR_W))
    if limit:
        units = units[:int(limit)]
    print("door: %d queries, S = %d; %d pairs already scored, %d queries to go (%d pairs)"
          % (len(rows), S, len(done), len(units), sum(len(u[1]) for u in units)), flush=True)
    images = sorted({p for _o, js, _c, _w in units for _q, p in js}
                    | {u[0] for u in units})
    t0 = time.time()
    budget = float(budget_minutes) * 60.0
    done_units = done_pairs = failed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        it = pool.imap_unordered(_score_unit, units, chunksize=1)
        for res in it:
            done_units += 1
            with open(PAIRS, "a", encoding="utf-8") as fh:
                for r in res:
                    fh.write(json.dumps(r) + "\n")
            done_pairs += sum(1 for r in res if r.get("score") is not None)
            failed += sum(1 for r in res if r.get("score") is None)
            el = time.time() - t0
            _progress({"stage": "score", "queries_done": done_units, "of": len(units),
                       "pairs": done_pairs, "failed": failed, "seconds": round(el, 1),
                       "seconds_per_query": round(el / max(1, done_units), 1)})
            print("  query %d/%d  %d pairs  %.0fs (%.0f s/query, failed %d)"
                  % (done_units, len(units), done_pairs, el, el / max(1, done_units), failed),
                  flush=True)
            if el > budget:
                print("budget reached -- rerun to continue", flush=True)
                pool.terminate()
                break
    out = {"queries_done": done_units, "of": len(units), "pairs": done_pairs,
           "failed": failed, "seconds": round(time.time() - t0, 1), "workers": workers,
           "colour": use_colour, "colour_w": COLOUR_W if use_colour else 0.0}
    _progress({"stage": "done", **out})
    print(json.dumps(out), flush=True)
    return out


def _ref_path(j: int) -> str:
    """The gallery image path of object index ``j`` (the door's own entry, not a bare key)."""
    return REC.path_of(REC.objects()[j], REF)


def warm(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build (or load) every graph ONCE on disk, so the pool reads a warm shared cache."""
    import matching.placement_frontend as FE
    paths = sorted({r["query"] for r in rows}
                   | {_ref_path(j) for j in range(len(REC.objects()))})
    t0 = time.time()
    for i, p in enumerate(paths):
        FE.cell(p, "m30read")
        if (i + 1) % 20 == 0:
            _progress({"stage": "warm", "done": i + 1, "of": len(paths),
                       "seconds": round(time.time() - t0, 1)})
            print("  warm %d/%d  %.0fs" % (i + 1, len(paths), time.time() - t0), flush=True)
    return {"images": len(paths), "seconds": round(time.time() - t0, 1)}


def _key_of(row: Dict[str, Any], j: int) -> str:
    return REC.objects()[j]["key"]


def _progress(info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)


def report(json_out: str = OUT) -> Dict[str, Any]:
    rows = door(True)
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    got = _load_pairs()
    hits: Dict[str, Tuple[float, float]] = {}
    per_class: Dict[str, List[Tuple[int, int]]] = {}
    per_split: Dict[str, Dict[str, Tuple[float, float]]] = {}
    losses: List[Dict[str, Any]] = []
    lacking = []
    for r in rows:
        ps = [(j, got.get((r["object"], _ref_path(j)))) for j in r["order"]]
        if any(s is None for _j, s in ps):
            lacking.append({"object": r["object"],
                            "unscored": sum(1 for _j, s in ps if s is None)})
            continue
        best = min(ps, key=lambda t: (-t[1], t[0]))
        hit = int(best[0] == r["truth"])
        ceiling = int(r["truth"] in r["order"])
        hits[r["object"]] = (float(hit), float(ceiling))
        per_class.setdefault(r["cls"], []).append((hit, ceiling))
        per_split.setdefault(r["split"], {})[r["object"]] = (float(hit), float(ceiling))
        if not hit:
            same = best[0] != r["truth"] and keys[best[0]].split("/")[0] == r["cls"]
            truth_s = dict(ps).get(r["truth"])
            losses.append({"object": r["object"], "winner": keys[best[0]],
                           "same_class": bool(same),
                           "in_shortlist": truth_s is not None,
                           "gap": (float(best[1] - truth_s) if truth_s is not None else None)})
    ci = _one_ci(hits)
    out: Dict[str, Any] = {
        "cell": {"queries": len(rows), "scored": len(hits), "S": S, "view": VIEW,
                 "gallery": len(keys), "chance_top1": 1.0 / len(keys),
                 "chance_recall_s8": S / len(keys)},
        "arm": {"door": "RRF(counted, appearance, colour@0.5)",
                "matcher": "shipped + colour blend w_col=%.2f (w_cfg held)" % COLOUR_W,
                "top1": ci,
                "by_split": {sp: _one_ci(v) for sp, v in sorted(per_split.items())},
                "by_class": {c: {"n": len(v), "hits": sum(h for h, _ in v),
                                             "top1": sum(h for h, _ in v) / len(v),
                                             "ceiling_hits": sum(c for _, c in v)}
                                         for c, v in sorted(per_class.items())}},
        "m27_s8": {"top1": M27_S8_TOP1, "ci95": [0.150, 0.338], "n": 80,
                   "source": "MISSION27_REPORT.md section 2 (shipped configuration)"},
        "mde": {"value": MDE, "planning_effect": 0.075,
                "note": "MDE > effect: this read reports an estimate and claims nothing"},
        "losses": {"n": len(losses), "same_class": sum(1 for x in losses if x["same_class"]),
                   "other_class": sum(1 for x in losses if not x["same_class"]),
                   "out_of_shortlist": sum(1 for x in losses if not x["in_shortlist"]),
                   "median_gap": float(np.median([x["gap"] for x in losses if x["gap"] is not None]))
                   if any(x["gap"] is not None for x in losses) else None,
                   "by_class": {c: {"same": sum(1 for x in losses if x["same_class"]
                                               and x["object"].startswith(c)),
                                    "other": sum(1 for x in losses if not x["same_class"]
                                                 and x["object"].startswith(c))}
                                for c in sorted({x["object"].split("/")[0] for x in losses})}},
        "lacking": lacking,
        "manifest": {"environment": _env(), "records": os.path.basename(RECORDS),
                     "records_bytes": os.path.getsize(RECORDS)},
    }
    if ci.get("top1") is not None:
        out["estimate_vs_m27"] = {"diff": ci["top1"] - M27_S8_TOP1,
                                  "unpaired": True,
                                  "note": "M27's per-object vector is not in its json, so this "
                                          "difference is unpaired and is not a test"}
    with open(json_out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=1)[:2200])
    return out


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--budget-minutes", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-colour", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--split", default="", help="score this split first (FIT / CONFIRMATION)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    import matching.placement_frontend as FE
    FE.apply_recipe()
    if a.warm:
        print(json.dumps(warm(door(not a.no_colour))), flush=True)
    if a.run:
        run(a.budget_minutes, a.workers, not a.no_colour, a.limit, a.split)
    if a.report:
        report()
    if not (a.run or a.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
