"""Task 4 PART 0: build dev graphs for arms B0 (L3/L4 OFF) and B1 (L3/L4 ON).

Dev run = lowest-rank dev instance per class (rank 3), view -090-000.
8 instances. Each arm is a PERCEPT_SET switch spec; the graph cache key
folds the switch digest in, so B0 and B1 never share a cache entry.

Usage:
    python t4_build_dev.py --arm B0 --workers 2
    python t4_build_dev.py --arm B1 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

#: arm -> PERCEPT_SET switch spec.  B0 = L3/L4 OFF (the historical default);
#: B1 = L3/L4 ON (current sampling).  Every other flag is the default.
ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
}


def _worker_init(spec):
    import guard
    guard.activate()
    import _bootstrap as b
    b.apply_switch_overrides(spec)


def _build_one(entry):
    from frontend import build_cached
    return build_cached(entry)


def _dev_entries():
    import guard
    guard.activate()
    import corpus
    split = corpus.split_tuning()
    # lowest-rank dev instance per class, view -090-000
    by_class = {}
    for e in split["dev"]:
        if e["view"] != "-090-000":
            continue
        cls = e["class"]
        if cls not in by_class or e["object"] < by_class[cls]["object"]:
            by_class[cls] = e
    return [by_class[c] for c in corpus.CLASSES]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    spec = ARMS[args.arm]
    os.environ["PERCEPT_SET"] = spec
    import _bootstrap
    applied = _bootstrap.apply_switch_overrides(spec)
    print("arm", args.arm, "applied:", applied, flush=True)

    import guard
    guard.activate()
    entries = _dev_entries()
    print("dev entries:", [corpus_name(e) for e in entries], flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    metas = []
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(spec,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_build_one, entries,
                                                    chunksize=1)):
            if isinstance(rec, dict) and "error" in rec:
                raise RuntimeError(rec["error"])
            metas.append(rec)
            if (i + 1) % 4 == 0 or i + 1 == len(entries):
                print("  front end %3d/%d (%.0fs)"
                      % (i + 1, len(entries), time.perf_counter() - t0),
                      flush=True)
    metas.sort(key=lambda m: m["name"])
    out = {
        "arm": args.arm,
        "switch_spec": spec,
        "applied": applied,
        "graphs": metas,
        "build_seconds": time.perf_counter() - t0,
    }
    path = os.path.join(ROOT, "Topology", "results", "t4",
                        "dev_graphs_%s.json" % args.arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


def corpus_name(e):
    return "%s%d%s" % (e["class"], int(e["object"]), e["view"])


if __name__ == "__main__":
    sys.exit(main())
