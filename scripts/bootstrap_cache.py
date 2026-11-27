"""Restore everything a fresh sandbox needs, and say what it cost (Mission 27, item 1d).

A fresh checkout used to spend ~40 minutes rebuilding 320 place records and another ~50 minutes
re-scoring pairs the programme had already paid for.  This script is the short path:

    python scripts/bootstrap_cache.py              # everything, idempotent
    python scripts/bootstrap_cache.py --json out.json

Steps, each skipped when already present:

    1. the ETH-80 images      (322 MB, Topology/fetch_corpus.py --fetch)
    2. the place records      (committed npz, data/placement_records_m27.npz -- no rebuild)
    3. the pair scores        (committed jsonl, adopted by basename under the current digest)
    4. the graph cache        (warmed lazily by whatever scores a pair; never up front)

Exit code 0 means a door experiment can run.  Nothing here needs the network except step 1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

CORPUS = os.path.join(ROOT, "data", "ETH-80-master")
RECORDS = os.path.join(ROOT, "data", "placement_records_m27.npz")


def _step(name: str, fn) -> Dict[str, Any]:
    t0 = time.time()
    try:
        info = fn() or {}
        return {"step": name, "ok": True, "seconds": round(time.time() - t0, 1), **info}
    except Exception as exc:                                    # reported, never swallowed
        return {"step": name, "ok": False, "seconds": round(time.time() - t0, 1),
                "error": repr(exc)}


def corpus() -> Dict[str, Any]:
    n = 0
    if os.path.isdir(CORPUS):
        for cls in os.listdir(CORPUS):
            d = os.path.join(CORPUS, cls)
            if not os.path.isdir(d):
                continue
            for inst in os.listdir(d):
                dd = os.path.join(d, inst)
                if os.path.isdir(dd):
                    n += len(os.listdir(dd))
    if n >= 8 * 10 * 41:
        return {"present": True, "files": n, "skipped": True}
    subprocess.run([sys.executable, os.path.join(ROOT, "Topology", "fetch_corpus.py"), "--fetch"],
                   check=True)
    return {"present": True, "fetched": True}


def records() -> Dict[str, Any]:
    from matching import placement_records as PR
    if not os.path.exists(RECORDS):
        return {"present": False,
                "how": "run Topology/m27_export_records.py --build on a machine that has the "
                       "pickles (.cache/m26/records); the export is what is committed"}
    info = PR.peek(RECORDS)
    return {"present": True, "mb": round(info["bytes"] / 1e6, 1), "views": info["views"],
            "objects": len(info["keys"])}


def pairs() -> Dict[str, Any]:
    from matching import placement_recognise as RC
    cache = RC.load_pair_cache()
    return {"entries": len(cache), "adopted": dict(RC.ADOPTED),
            "matcher_digest": RC.matcher_digest()}


def graphs() -> Dict[str, Any]:
    from matching import placement_cache as GC
    d = getattr(GC, "CACHE_DIR", None) or getattr(GC, "DIR", None)
    n = len(os.listdir(d)) if d and os.path.isdir(d) else 0
    return {"warm": False, "cached_graphs": n,
            "note": "graphs are warmed lazily by the first pair that needs them"}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    t0 = time.time()
    steps = [_step("corpus", corpus), _step("records", records), _step("pairs", pairs),
             _step("graphs", graphs)]
    out = {"steps": steps, "seconds": round(time.time() - t0, 1),
           "ready": all(s["ok"] and s.get("present", True) for s in steps)}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", a.json)
    return 0 if out["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
