"""M26 harness: build and cache the place records every door experiment reads.

The mission's speed rule is that door experiments touch CACHED PLACE RECORDS only -- no graph
rebuild, no matcher.  This file is the one thing that does rebuild: it turns each
``(object, view)`` image into a place record with the shipped front end
(``matching.placement_frontend.place_record``) and pickles it under ``.cache/m26/records``.

    python Topology/m26_records.py --build --workers 2      # resumable, safe to interrupt

Every later stage (``Topology/m26_census.py``, ``Topology/m26_oracle.py``) loads these pickles.
The records are the shipped ones: same recipe, same drops, same fields -- nothing here changes
what a record contains.

    python Topology/m26_records.py --report                 # what is on disk, and its build cost
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

CACHE = os.path.join(ROOT, ".cache", "m26", "records")

#: The reference view (what the gallery stores) plus the three registered query views.
REF_VIEW = "000-000"
QUERY_VIEWS = ("022-000", "035-045", "045-180")
VIEWS = (REF_VIEW,) + QUERY_VIEWS


def objects() -> List[Dict[str, Any]]:
    from Topology.placement_m23_eth80 import all_objects
    return all_objects()


def path_of(obj: Dict[str, Any], view: str) -> str:
    """``<class>/<inst>/<name>-<view>.png`` -- the ref's own folder and label."""
    ref = obj["ref"]
    folder = os.path.dirname(ref)
    label = os.path.basename(ref).split("-")[0]
    p = os.path.join(folder, "%s-%s.png" % (label, view))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def slug(obj_key: str, view: str) -> str:
    return "%s__%s.pkl" % (obj_key.replace("/", "-"), view)


def cache_path(obj_key: str, view: str) -> str:
    return os.path.join(CACHE, slug(obj_key, view))


def _job(arg: Tuple[str, str, str]) -> Tuple[str, str, float]:
    """Build one record and write it.  Runs in a worker; returns (key, view, seconds)."""
    obj_key, view, path = arg
    import matching.placement_frontend as FE
    out = cache_path(obj_key, view)
    if os.path.exists(out):
        return obj_key, view, 0.0
    rec = FE.place_record(path, tag="m26")
    rec["obj"] = obj_key
    rec["view"] = view
    os.makedirs(CACHE, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh, protocol=4)
    os.replace(tmp, out)
    return obj_key, view, float(rec.get("sec", 0.0))


def load(obj_key: str, view: str) -> Dict[str, Any]:
    with open(cache_path(obj_key, view), "rb") as fh:
        return pickle.load(fh)


def load_all(views: Sequence[str] = VIEWS) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for o in objects():
        for v in views:
            p = cache_path(o["key"], v)
            if os.path.exists(p):
                out[(o["key"], v)] = load(o["key"], v)
    return out


def build(views: Sequence[str] = VIEWS, workers: int = 2) -> Dict[str, Any]:
    import multiprocessing as mp
    t0 = time.time()
    jobs: List[Tuple[str, str, str]] = []
    for o in objects():
        for v in views:
            if not os.path.exists(cache_path(o["key"], v)):
                jobs.append((o["key"], v, path_of(o, v)))
    print("records: %d to build, %d on disk" % (len(jobs), len(objects()) * len(views) - len(jobs)),
          flush=True)
    done = 0
    spent = 0.0
    if jobs:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
            for key, view, sec in pool.imap_unordered(_job, jobs, chunksize=1):
                done += 1
                spent += sec
                if done % 10 == 0:
                    print("  %d/%d  %.0fs  (record time %.0fs)"
                          % (done, len(jobs), time.time() - t0, spent), flush=True)
    out = {"built": done, "seconds_wall": round(time.time() - t0, 1),
           "seconds_record_sum": round(spent, 1), "views": list(views)}
    print(json.dumps(out), flush=True)
    return out


def report(views: Sequence[str] = VIEWS) -> Dict[str, Any]:
    """What is on disk: cells, their place counts, drop counts, and the build cost."""
    cells: Dict[str, Any] = {}
    total_sec = 0.0
    for o in objects():
        for v in views:
            p = cache_path(o["key"], v)
            if not os.path.exists(p):
                continue
            r = load(o["key"], v)
            total_sec += float(r.get("sec", 0.0))
            cells["%s|%s" % (o["key"], v)] = {"split": o["split"], "places": len(r["places"]),
                                              "payload": r["n_payload"], "dropped": r["dropped"],
                                              "sec": round(float(r.get("sec", 0.0)), 2)}
    print(json.dumps({"cells": len(cells), "record_seconds": round(total_sec, 1)}, indent=1))
    return {"cells": cells, "record_seconds": total_sec}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(list(argv) if argv is not None else None)
    views = [v.strip().lstrip("-") for v in a.views.split(",") if v.strip()]
    if a.build:
        build(views, a.workers)
    if a.report:
        out = report(views)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(out, fh, indent=1, sort_keys=True)
            print("wrote", a.out)
    if not (a.build or a.report):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
