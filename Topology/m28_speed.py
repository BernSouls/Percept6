"""M28 item 2: the matcher's profile, and the parity gate for any change to it.

    python Topology/m28_speed.py --profile      # cProfile one pair, top 10 by cumulative time
    python Topology/m28_speed.py --scores out.json --pairs 30   # the gate's own numbers
    python Topology/m28_speed.py --compare a.json b.json        # max |difference|, 1e-12 gate

The pairs are `m27_profile.committed_pairs` -- the same 30 the mission names, from this
environment's own cache.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def pairs(n: int) -> List[Any]:
    from Topology.m27_profile import committed_pairs
    return committed_pairs(n)


def profile_one() -> Dict[str, Any]:
    import matching.placement_frontend as FE
    FE.apply_recipe()
    q, r, _ref = pairs(1)[0]
    FE.cell(q, "prof")
    FE.cell(r, "prof")
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    s = FE.pair_score(q, r)
    pr.disable()
    wall = time.time() - t0
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(40)
    text = buf.getvalue()
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[0].isdigit() and parts[1].replace(".", "").isdigit():
            rows.append({"ncalls": parts[0], "tottime": float(parts[1]),
                         "cumtime": float(parts[3]), "what": parts[5]})
        if len(rows) >= 10:
            break
    out = {"score": float(s), "wall_s": round(wall, 2), "top10_cumulative": rows,
           "total_calls": text.splitlines()[1].strip() if len(text.splitlines()) > 1 else ""}
    print(json.dumps(out, indent=1))
    return out


def scores(path: str, n: int) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    FE.apply_recipe()
    ps = pairs(n)
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (q, r, ref) in enumerate(ps):
        t = time.time()
        s = FE.pair_score(q, r)
        out.append({"q": q, "r": r, "score": float(s), "seconds": round(time.time() - t, 2),
                    "committed_M23": ref})
        print("  %2d/%d %.6f  %.2f s" % (i + 1, len(ps), s, time.time() - t), flush=True)
    res = {"pairs": out, "n": len(out), "seconds": round(time.time() - t0, 1),
           "s_per_pair": round((time.time() - t0) / max(1, len(out)), 2),
           "manifest": {"environment": _env()}}
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    return res


def compare(a: str, b: str) -> Dict[str, Any]:
    A = json.load(open(a))
    B = json.load(open(b))
    pa = {(x["q"], x["r"]): x["score"] for x in A["pairs"]}
    pb = {(x["q"], x["r"]): x["score"] for x in B["pairs"]}
    shared = sorted(set(pa) & set(pb))
    d = max((abs(pa[k] - pb[k]) for k in shared), default=float("nan"))
    out = {"n": len(shared), "max_abs_diff": d, "identical_1e12": bool(d <= 1e-12),
           "s_per_pair_before": A.get("s_per_pair"), "s_per_pair_after": B.get("s_per_pair")}
    print(json.dumps(out, indent=1))
    return out


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--scores")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.profile:
        profile_one()
    if a.scores:
        scores(a.scores, a.pairs)
    if a.compare:
        compare(a.compare[0], a.compare[1])
    if not (a.profile or a.scores or a.compare):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
