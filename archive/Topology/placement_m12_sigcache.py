"""Mission 12 step 3 -- measure the redundancy, then prove the cache changes nothing.

(a) MEASURE.  For a graph pair, the matcher asks `node_sim` about every
same-lens node pair: `sum_l |A_l| * |B_l|` questions.  The similarity is a
function of the two signatures' CONTENT, so the questions a cache can answer
are the DISTINCT (lens, content(a), content(b)) triples.  `--measure` reports
both, per graph pair and cumulatively, then repeats the counting over a
retrieval-shaped run (many queries against one gallery) where the cumulative
distinct count is what a shared cache pays for.

(b) IDENTITY AND SPEED.  `--identity` runs the thirty FIXED pairs of
`placement_m12_perf.py` twice -- cache off, cache on -- and compares, at full
precision, the score, `score_shipped`, the placement evidence, the judged
count and the whole mapping.  Any difference is a failure.  It also records
per-pair seconds for both arms, which is the speed claim.

    python Topology/placement_m12_sigcache.py --measure   --workers 2
    python Topology/placement_m12_sigcache.py --identity  --workers 2

Writes `Topology/results/placement/m12_sigcache.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m12_sigcache.json")


# ---------------------------------------------------------------------------
# (a) how much of the work is a repeat
# ---------------------------------------------------------------------------
#: The pair key `(lens, content(a), content(b))` is 64-bit hashed per signature
#: and FOLDED in uint64 numpy: `h = (lens << 1) ^ (ha * M) ^ hb`, M odd.  Two
#: different triples colliding is a birthday event in 2**64 over ~1.7e7 pairs
#: (p ~ 0.008 expected collisions, each costing one distinct count); the folding
#: is checked against an EXACT Python-set count on one pair, and both numbers are
#: reported.  Sets of Python tuple keys are what ran the sandbox out of memory on
#: the first attempt, so the counting is done in fixed-width arrays here.
_M = 1000003


def _node_hashes(g: Any) -> Dict[Any, Any]:
    import numpy as np
    from matching import sig_sim_cache as SC
    out: Dict[Any, Any] = {}
    for lens, nodes in _by_lens(g).items():
        out[lens] = np.asarray([hash(SC.content_key(s)) & 0xFFFFFFFFFFFFFFFF
                                for s in nodes], dtype=np.uint64)
    return out


def _by_lens(g: Any) -> Dict[Any, List[Any]]:
    """lens -> the signature objects of its nodes (skipping signatures absent)."""
    out: Dict[Any, List[Any]] = {}
    for n, s in g.sigs.items():
        if s is None or s[1] is None:
            continue
        out.setdefault(s[0], []).append(s[1])
    return out


def _exact_distinct(ga: Any, gb: Any) -> int:
    from matching import sig_sim_cache as SC
    seen = set()
    for n, sa in ga.sigs.items():
        if sa is None or sa[1] is None:
            continue
        ka = SC.content_key(sa[1])
        for _m, sb in gb.sigs.items():
            if sb is None or sb[1] is None or sb[0] != sa[0]:
                continue
            seen.add((sa[0], ka, SC.content_key(sb[1])))
    return len(seen)


def _pair_counts(ga: Any, gb: Any) -> Dict[str, Any]:
    import numpy as np
    la, lb = _node_hashes(ga), _node_hashes(gb)
    pairs = 0
    parts = []
    for lens in set(la) & set(lb):
        ha, hb = la[lens], lb[lens]
        pairs += int(ha.size * hb.size)
        if not ha.size or not hb.size:
            continue
        folded = (np.uint64((hash(lens) & 0xFFFFFFFF) << 1)
                  ^ (ha[:, None] * np.uint64(_M)) ^ hb[None, :]).ravel()
        parts.append(folded)
    distinct = int(np.unique(np.concatenate(parts)).size) if parts else 0
    return {"node_pairs": pairs, "distinct": distinct, "hashes": parts,
            "all_pairs": sum(la[k].size for k in la) * sum(lb[k].size for k in lb)}


def measure(workers: int) -> Dict[str, Any]:
    import numpy as np
    from Topology import placement_m11_families as F
    from Topology.placement_cache import get_or_build
    refs = [s for s in F.gallery()]
    gal: Dict[str, Any] = {}
    for s in refs:
        gal[s.name] = get_or_build(s.scene.render().image_u8, "m12sig|%s|ref" % s.name)
    queries: List[Tuple[str, Any]] = []
    for s in refs:
        for lab, view in F.query_steps(s)[:1]:
            sc, _pv = F.pose_fn(s, view)
            queries.append(("%s|%s" % (s.name, lab),
                            get_or_build(sc.render().image_u8,
                                         "m12sig|%s|%s" % (s.name, lab))))
    per_pair = []
    acc: List[Any] = []
    tot_nodes = tot_distinct = 0
    t0 = time.time()
    for qk, gq in queries:
        for rk, gr in gal.items():
            c = _pair_counts(gq, gr)
            tot_nodes += c["node_pairs"]
            tot_distinct += c["distinct"]
            acc.extend(c["hashes"])
            per_pair.append({"q": qk, "r": rk, "node_pairs": c["node_pairs"],
                             "distinct": c["distinct"],
                             "ratio": (c["node_pairs"] / c["distinct"]) if c["distinct"] else 0.0})
    union = int(np.unique(np.concatenate(acc)).size) if acc else 0
    exact = _exact_distinct(queries[0][1], gal[refs[0].name]) if queries else 0
    fold_check = per_pair[0]["distinct"] if per_pair else 0
    wall = time.time() - t0
    ratios = [p["ratio"] for p in per_pair if p["distinct"]]
    return {
        "graph_pairs": len(per_pair),
        "queries": len(queries), "refs": len(gal),
        "node_pairs_total": tot_nodes,
        "distinct_summed_over_pairs": tot_distinct,
        "distinct_shared_over_run": union,
        "reuse_per_pair_median": float(sorted(ratios)[len(ratios) // 2]) if ratios else 0.0,
        "reuse_per_pair_min": float(min(ratios)) if ratios else 0.0,
        "reuse_per_pair_max": float(max(ratios)) if ratios else 0.0,
        "reuse_across_run": (tot_nodes / union) if union else 0.0,
        "fold_vs_exact_first_pair": {"folded": fold_check, "exact": exact,
                                     "agree": fold_check == exact},
        "wall_measure_s": wall,
        "rows": per_pair,
    }


# ---------------------------------------------------------------------------
# (b) thirty fixed pairs, cache off vs cache on
# ---------------------------------------------------------------------------
def _identity_job(args: Tuple[str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    tag, qk, rk, imgs = args
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from matching import sig_sim_cache as SC
    from Topology.placement_cache import get_or_build
    q_img, r_img = imgs[qk], imgs[rk]
    q = get_or_build(q_img, "perf|" + qk)
    r = get_or_build(r_img, "perf|" + rk)
    rec: Dict[str, Any] = {"tag": tag, "q": qk, "r": rk}
    for arm, cache_on in (("off", False), ("on", True)):
        P.PLACEMENT_MATCHER_ENABLED = True
        P.PLACEMENT_SCORE_EVIDENCE = True
        P.PLACEMENT_SCORE_WEIGHT = 0.375
        P.MATCH_SIG_SIM_CACHE = cache_on
        if cache_on:
            SC.reset()
        t0 = time.time()
        res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                        placement_images=(q_img, r_img))
        dt = time.time() - t0
        rec[arm] = {"score": repr(float(res.score)),
                    "shipped": repr(float(res.score_shipped)),
                    "ev": repr(float(res.placement_evidence)),
                    "judged": int(res.placement_judged),
                    "mapping": sorted((int(k), int(v)) for k, v in res.mapping.items()),
                    "sec": dt,
                    "cache": SC.stats() if cache_on else None}
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    P.MATCH_SIG_SIM_CACHE = False
    return rec


def pairs30() -> List[Tuple[str, str, str]]:
    """THIRTY fixed synthetic pairs: every irregular scene's second view against
    the other five scenes' references (6 x 5), deterministic, no randomness.

    The corpus pairs the perf harness would add are absent (ETH-80 is blocked),
    so the thirty are all synthetic and the read says so.
    """
    from Topology import placement_m11_families as F
    ir = [s for s in F.gallery() if s.family == "IRREGULAR"][:6]
    out: List[Tuple[str, str, str]] = []
    for i, s in enumerate(ir):
        lab, _v = F.VIEW_STEPS[(i + 1) % 2]
        for j, r in enumerate(ir):
            if j == i:
                continue
            out.append(("syn", "%s|%s" % (s.name, lab), "%s|ref" % r.name))
    return out


def identity(workers: int) -> Dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    from Topology import placement_m12_perf as PERF
    ps = pairs30()
    imgs = PERF._images()
    jobs = [(t, q, r, imgs) for (t, q, r) in ps]
    print("%d pairs" % len(ps), flush=True)
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_identity_job, jobs):
            rows.append(r)
            print("  %-28s off %6.2fs  on %6.2fs" % (r["q"][:28], r["off"]["sec"],
                                                      r["on"]["sec"]), flush=True)
    diffs = []
    for r in rows:
        for k in ("score", "shipped", "ev", "judged", "mapping"):
            if r["off"][k] != r["on"][k]:
                diffs.append({"q": r["q"], "r": r["r"], "field": k})
    t_off = sum(r["off"]["sec"] for r in rows)
    t_on = sum(r["on"]["sec"] for r in rows)
    calls = sum(r["on"]["cache"]["calls"] for r in rows if r["on"]["cache"])
    hits = sum(r["on"]["cache"]["hits"] for r in rows if r["on"]["cache"])
    return {"pairs": len(rows), "identical": not diffs, "diffs": diffs,
            "sum_sec_off": t_off, "sum_sec_on": t_on,
            "speedup": (t_off / t_on) if t_on else 0.0,
            "sum_calls": calls, "sum_hits": hits,
            "hit_rate": (hits / calls) if calls else 0.0,
            "rows": rows}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    res: Dict[str, Any] = {}
    if os.path.exists(RESULT):
        with open(RESULT) as fh:
            res = json.load(fh)
    if args.measure:
        res["measure"] = measure(args.workers)
        m = res["measure"]
        print("measure: %d graph pairs, node pairs %d, distinct summed %d, "
              "shared over the run %d, reuse %.1fx per pair (median), %.1fx across the run"
              % (m["graph_pairs"], m["node_pairs_total"],
                 m["distinct_summed_over_pairs"], m["distinct_shared_over_run"],
                 m["reuse_per_pair_median"], m["reuse_across_run"]), flush=True)
    if args.identity:
        res["identity"] = identity(args.workers)
        i = res["identity"]
        print("identity: %d pairs, identical %s, %.2fs -> %.2fs (%.2fx), hits %d/%d"
              % (i["pairs"], i["identical"], i["sum_sec_off"], i["sum_sec_on"],
                 i["speedup"], i["sum_hits"], i["sum_calls"]), flush=True)
        if not i["identical"]:
            print(i["diffs"][:5])
    with open(RESULT, "w") as fh:
        json.dump(res, fh)
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
