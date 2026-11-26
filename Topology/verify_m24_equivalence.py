"""Mission 24, item 4 -- prove the core entry point IS the system that was measured.

Two checks, both against the Mission-23 read's own records, and neither of them runs a matcher
pair that is not already in the cache:

  DOOR     for every read query: ``recognise(..., matcher=False).shortlist`` must equal the
           ``door_s`` stored in ``m23_read_rows.jsonl`` (the shortlist the read actually used).
  MATCHER  for every read query whose whole shortlist is in ``m23_read_pairs.jsonl``: the shipped
           order over the CACHED scores -- ``placement_recognise.order_plain``, the function
           ``recognise`` itself uses -- must put the truth at the rank the row records
           (``assembled_rank``) and hit top-1 exactly when the row says so.

Run:
    python Topology/verify_m24_equivalence.py --workers 2            # both checks, all 80
    python Topology/verify_m24_equivalence.py --limit 4 --workers 2  # a smoke run
Writes ``Topology/results/placement/m24_equivalence.json`` through the manifest writer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]

from matching import placement_recognise as RC                                # noqa: E402
from matching import placement_bundle as PB                                   # noqa: E402

OUT = os.path.join(ROOT, "Topology", "results", "placement")

_G: Any = None                                                                # forked gallery


def _door_job(qpath: str) -> Tuple[List[int], List[int]]:
    r = RC.recognise(qpath, _G, workers=0, matcher=False)
    return list(r.shortlist), list(r.door_order)


def _read_rows() -> List[Dict[str, Any]]:
    p = os.path.join(OUT, "m23_read_rows.jsonl")
    return [json.loads(x) for x in open(p) if x.strip()]


def _pair_cache() -> Dict[Tuple[str, str], float]:
    p = os.path.join(OUT, "m23_read_pairs.jsonl")
    out: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" not in r:
            out[(r["q"], r["r"])] = float(r["B"])
    return out


def run(workers: int = 2, limit: int = 0) -> Dict[str, Any]:
    global _G
    from Topology.placement_m23_eth80 import all_objects
    from matching import m15_manifest as MF
    t0 = time.time()
    objs = all_objects()
    keys = [o["key"] for o in objs]
    refs = [o["ref"] for o in objs]
    rows = [r for r in _read_rows() if r["view"] == "-035-045"]
    rows.sort(key=lambda r: r["object"])
    if limit:
        rows = rows[:int(limit)]
    print("gallery: %d items" % len(objs), flush=True)
    t_g = time.time()
    _G = RC.build_gallery(list(zip(keys, refs)), workers=workers)
    print("gallery built %.0fs (%d items, digest %s)"
          % (time.time() - t_g, _G.n, _G.digest[:12]), flush=True)
    if _G.keys != keys:
        raise SystemExit("gallery key order does not match the read's object order: %r vs %r"
                         % (_G.keys[:3], keys[:3]))
    by_key = {o["key"]: o for o in objs}
    jobs = [by_key[r["object"]]["views"][r["view"]] for r in rows]
    t_d = time.time()
    if workers and int(workers) > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(int(workers)) as pool:
            got = pool.map(_door_job, jobs, chunksize=1)
    else:
        got = [_door_job(p) for p in jobs]
    print("door: %d queries in %.0fs" % (len(got), time.time() - t_d), flush=True)
    door_ok = 0
    door_bad: List[Dict[str, Any]] = []
    for r, (short, order) in zip(rows, got):
        if list(short) == list(r["door_s"]):
            door_ok += 1
        else:
            door_bad.append({"object": r["object"], "row": r["door_s"], "core": list(short)})
    pairs = _pair_cache()
    mat_ok = mat_bad = 0
    mat_rows = 0
    bad_mat: List[Dict[str, Any]] = []
    for i, (r, (short, _order)) in enumerate(zip(rows, got)):
        cd = list(r["door_s"])
        have = {j: pairs[(jobs[i], refs[j])] for j in cd if (jobs[i], refs[j]) in pairs}
        if len(have) < len(cd):
            continue
        mat_rows += 1
        order = RC.order_plain(cd, have)
        rank = (order.index(int(r["truth"])) + 1) if int(r["truth"]) in order else None
        top1 = int(bool(order) and order[0] == int(r["truth"]))
        if rank == r["assembled_rank"] and top1 == int(r["assembled"]):
            mat_ok += 1
        else:
            mat_bad += 1
            bad_mat.append({"object": r["object"], "row_rank": r["assembled_rank"],
                            "core_rank": rank, "row_top1": r["assembled"], "core_top1": top1})
    out = {
        "door": {"checked": len(rows), "equal": door_ok, "mismatched": len(door_bad),
                 "examples": door_bad[:3]},
        "matcher": {"queries_with_full_pair_cache": mat_rows, "equal": mat_ok,
                    "mismatched": mat_bad, "examples": bad_mat[:3],
                    "note": "the order is checked at the rank of the truth and at top-1, which "
                            "is what the read's rows store; no matcher pair was scored that was "
                            "not already cached"},
        "gallery": {"items": _G.n, "digest": _G.digest, "keys_match_read_order": True},
        "bundle": PB.PlacementBundle.load().as_dict(),
        "seconds": {"total": time.time() - t0},
        "how": "python Topology/verify_m24_equivalence.py --workers %d%s"
               % (workers, (" --limit %d" % limit) if limit else ""),
    }
    MF.write_result(os.path.join(OUT, "m24_equivalence.json"), out)
    print("DOOR    %d/%d equal to the read's door_s" % (door_ok, len(rows)), flush=True)
    print("MATCHER %d/%d equal (queries with a full pair cache: %d)"
          % (mat_ok, mat_rows, mat_rows), flush=True)
    print("wrote m24_equivalence.json", flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(list(argv) if argv is not None else None)
    out = run(workers=a.workers, limit=a.limit)
    ok = out["door"]["mismatched"] == 0 and out["matcher"]["mismatched"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
