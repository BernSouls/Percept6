"""MISSION 12 STEP 1 -- the speed harness that also proves nothing moved.

Thirty FIXED pairs (deterministic selection, no randomness), each matched twice --
placement matcher off (arm A) and on (arm C) -- and for every pair it records the
score, the mapping (as a sorted list of (u,v)) and the placement diagnostics, at full
precision.  Run with --save to write the baseline, then again after an optimisation to
diff.  Bit-identical means identical here: same floats, same mapping, byte for byte.

    python Topology/placement_m12_perf.py --save     # before
    python Topology/placement_m12_perf.py --check    # after (nonzero exit on any diff)

The pairs are synthetic M11 scenes (in-plane and out-of-plane, including out-of-plane
candidates so the alignment is not the easy case) plus, when the ETH-80 corpus is
present, real pairs, which is where the cost actually matters.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
BASE = os.path.join(OUT, "m12_perf_baseline.json")
N_PAIRS = 30


def pairs() -> List[Tuple[str, str, str]]:
    """(tag, query image key, reference image key) -- fixed, deterministic."""
    from Topology import placement_m11_families as F
    out: List[Tuple[str, str, str]] = []
    gal = F.gallery()
    ir = [s for s in gal if s.family == "IRREGULAR"]
    for i, s in enumerate(ir[:6]):
        lab, _v = F.VIEW_STEPS[i % 2]
        out.append(("syn", "%s|%s" % (s.name, lab), "%s|ref" % ir[(i + 1) % len(ir)].name))
    for i, s in enumerate(ir[6:14]):
        lab, _v = F.VIEW_STEPS[(i + 1) % 2]
        out.append(("syn", "%s|%s" % (s.name, lab), "%s|ref" % ir[i % 6].name))
    # real pairs, if the corpus is there
    try:
        from Topology.placement_m11_eth80 import catalog, _qkey, _rkey
        refs, queries = catalog()
        rmap = {_rkey(r): r for r in refs}
        for e in queries[:14]:
            out.append(("real", _qkey(e), _rkey(rmap and
                                                [r for r in refs if r["key"] == e["key"]][0])))
    except Exception as ex:      # corpus absent: synthetic only, and say so
        print("  (no real pairs: %r)" % (ex,), flush=True)
    return out[:N_PAIRS] if len(out) >= N_PAIRS else out


def _images() -> Dict[str, np.ndarray]:
    from Topology import placement_m11_families as F
    cache: Dict[str, np.ndarray] = {}
    for s in F.gallery():
        cache["%s|ref" % s.name] = s.scene.render().image_u8
        for lab, view in F.query_steps(s):
            sc, _pv = F.pose_fn(s, view)
            cache["%s|%s" % (s.name, lab)] = sc.render().image_u8
    try:
        from Topology.placement_m11_eth80 import _load_u8, catalog as eth_catalog
        refs, queries = eth_catalog()
        for e in refs:
            cache.setdefault(e["key"] + "|" + e["view"], _load_u8(e["path"]))
        for e in queries:
            cache.setdefault(e["key"] + "|" + e["view"], _load_u8(e["path"]))
    except Exception:
        pass
    return cache


def _job(args: Tuple[str, str, str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    tag, qk, rk, imgs = args
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    q_img, r_img = imgs[qk], imgs[rk]
    q = get_or_build(q_img, "perf|" + qk)
    r = get_or_build(r_img, "perf|" + rk)
    rec: Dict[str, Any] = {"tag": tag, "q": qk, "r": rk}
    for arm, on in (("A", False), ("C", True)):
        P.PLACEMENT_MATCHER_ENABLED = on
        P.PLACEMENT_SCORE_EVIDENCE = on
        P.PLACEMENT_SCORE_WEIGHT = 0.375 if on else 0.0
        t0 = time.time()
        res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                        placement_images=(q_img, r_img) if on else None)
        dt = time.time() - t0
        rec[arm] = {"score": repr(float(res.score)),
                    "shipped": repr(float(res.score_shipped)),
                    "ev": repr(float(res.placement_evidence)),
                    "judged": int(res.placement_judged),
                    "mapping": sorted((int(k), int(v)) for k, v in res.mapping.items()),
                    "sec": dt}
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    return rec


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    ps = pairs()
    imgs = _images()
    jobs = [(t, q, r, imgs) for (t, q, r) in ps]
    print("%d pairs (%d synthetic, %d real)" % (
        len(ps), sum(1 for p in ps if p[0] == "syn"),
        sum(1 for p in ps if p[0] == "real")), flush=True)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            rows.append(r)
    wall = time.time() - t0
    per = {arm: float(np.mean([r[arm]["sec"] for r in rows])) for arm in ("A", "C")}
    print("wall %.1fs  per-pair A %.2fs  C %.2fs" % (wall, per["A"], per["C"]),
          flush=True)
    if args.save:
        with open(BASE, "w") as fh:
            json.dump({"pairs": len(rows), "wall": wall, "per_pair": per,
                       "rows": rows}, fh)
        print("saved baseline %s" % BASE)
        return 0
    if args.check:
        with open(BASE) as fh:
            base = json.load(fh)
        def _norm(v: Any) -> Any:
            # `mapping` is stored as a sorted list of (int, int) pairs; JSON gives
            # back lists, the fresh run has tuples, so a raw != compares list vs
            # tuple and EVERY cell "differs" (28/28) even when nothing changed.
            if isinstance(v, list):
                return [list(x) if isinstance(x, (list, tuple)) else x for x in v]
            return v
        diffs = []
        for a, b in zip(base["rows"], rows):
            for arm in ("A", "C"):
                for k in ("score", "shipped", "ev", "judged", "mapping"):
                    if _norm(a[arm][k]) != _norm(b[arm][k]):
                        diffs.append((a["q"], a["r"], arm, k))
        print("identical: %s  (%d diffs)" % (not diffs, len(diffs)))
        if diffs:
            print(diffs[:5])
            return 1
        b_per = base["per_pair"]
        print("speedup A %.2fx  C %.2fx" % (b_per["A"] / per["A"],
                                            b_per["C"] / per["C"]))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
