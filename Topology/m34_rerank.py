"""M34 items 3 (fallback) and 5: a channel as a RE-RANKER INSIDE THE EIGHT, cached pairs only.

The door returns a shortlist of eight.  This file does not touch the door, does not score a pair it
has not already been handed, and adds no new similarity rule: it takes the matcher's own order inside
the eight (from `m30_read_pairs.jsonl`, the FIT read's cached pairs) and merges ONE channel's order
into it with the tree's registered fusion -- reciprocal rank fusion, k = 60, matcher weight 1.0 and
the channel at the stated discount 0.5 (`matching.placement_fuse.SHAPE_WEIGHT`).  Nothing else is
tuned: no exponent, no per-class weight, no threshold.

    python Topology/m34_rerank.py --channel shape       # M34 item 5
    python Topology/m34_rerank.py --channel fragment    # M34 item 3, if 3(a) fails and 3(b) passes

WHAT IS REPORTED.  Top-1 on the FIT cells before and after, recall@8 (it CANNOT move inside the
eight: the truth is in the eight or it is not), and -- because a re-ranker that trades one error for
another is not progress -- the split of what it FIXES and what it BREAKS by whether the object that
blocked the truth was of the same class or another class.  Each gain is paired with an object-level
bootstrap CI, and the MDE of the paired difference is stated before the read.

BAR (M34 item 5, fixed in the mission, not here): FIT top-1 gain >= +0.075, the confirmation view not
below the incumbent by more than 0.025.  The confirmation view is NOT in the cached read (the cache
holds `-035-045` only); if the FIT number clears the bar this file says so and names the confirmation
as unmeasured rather than borrowing a number from somewhere else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402

PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
FRAG_CACHE = os.path.join(ROOT, ".cache", "m34_fragments")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m34_rerank.json")
REF = "000-000"
VIEW = "035-045"
RRF_K = 60
W_MATCHER = 1.0
W_CHANNEL = 0.5
BAR_TOP1 = 0.075
BAR_CONFIRM = -0.025
RESAMPLES = 10000
SEED = 20261118


def _pairs() -> Dict[str, List[Tuple[str, float]]]:
    """object key -> [(candidate object key, matcher score)] in the door's own order."""
    from matching import placement_records as PR
    out: Dict[str, List[Tuple[str, float]]] = {}
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if r.get("score") is None:
            continue
        out.setdefault(r["object"], []).append((PR.object_id(r["r"]), float(r["score"])))
    return {k: sorted(v, key=lambda t: (-t[1], t[0])) for k, v in out.items()}


def _shape_matrix(keys: Sequence[str]) -> np.ndarray:
    """The registered curvature channel, from the records alone, for the FIT query view."""
    from matching import placement_records as PR
    from matching import placement_shape as PS
    cells = PR.load_cells(RECORDS, [REF, VIEW])
    gal = [{"places": cells[(k, REF)]["places"]} for k in keys]
    qs = [{"places": cells[(k, VIEW)]["places"]} for k in keys]
    return PS.curvature_matrix(gal, qs, how="object_mean")


def _fragment_matrix(keys: Sequence[str]) -> np.ndarray:
    """The M34 fragment channel, from its own cache: the same coverage rule the FIT screen uses."""
    from matching import placement_fragment as PF
    import Topology.m34_fragments as M
    cells = M._load_cells((REF, VIEW))
    pool = np.vstack([cells[(k, REF)]["contour_fragment|desc"] for k in keys])
    sig = PF.scales(pool)
    Mx = np.zeros((len(keys), len(keys)), dtype=np.float64)
    for i, k in enumerate(keys):
        for j, gk in enumerate(keys):
            S, _g = M._fragment_matrix(cells[(k, VIEW)], cells[(gk, REF)], "contour_fragment", sig)
            Mx[i, j] = M._coverage(S)
    return Mx


def _rerank(order: Sequence[int], chan: np.ndarray, idx: int) -> List[int]:
    """RRF inside the eight: matcher rank (1..8) and channel rank (1..8), k = 60, weights 1.0/0.5."""
    cand = list(order)
    chan_order = sorted(cand, key=lambda j: (-float(chan[idx, j]), j))
    mr = {j: r + 1 for r, j in enumerate(cand)}
    cr = {j: r + 1 for r, j in enumerate(chan_order)}
    fused = sorted(cand, key=lambda j: (-(W_MATCHER / (RRF_K + mr[j])
                                        + W_CHANNEL / (RRF_K + cr[j])), mr[j]))
    return fused


def _bootstrap(diff: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> List[float]:
    rng = np.random.default_rng(seed)
    n = diff.size
    if n == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run(channel: str) -> Dict[str, Any]:
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls_of = {o["key"]: o["cls"] for o in objs}
    idx = {k: i for i, k in enumerate(keys)}
    split = {}
    for i, o in enumerate(objs):
        split.setdefault(str(o["split"]), []).append(i)
    pairs = _pairs()
    chan = _shape_matrix(keys) if channel == "shape" else _fragment_matrix(keys)

    base = np.zeros(len(keys), dtype=bool)          # truth at rank 1 before
    new = np.zeros(len(keys), dtype=bool)
    fixed_same = fixed_other = broke_same = broke_other = 0
    ranked_out = 0
    per_object: Dict[str, Any] = {}
    for k in keys:
        rows = pairs.get(k, [])
        order = [idx[c] for c, _s in rows if c in idx]
        if idx[k] not in order:
            ranked_out += 1
            continue
        i = idx[k]
        pos = order.index(i)
        merged = _rerank(order, chan, i)
        base[i] = pos == 0
        new[i] = merged.index(i) == 0
        if not base[i] and new[i]:
            blocker = rows[pos][0]
            if cls_of.get(blocker) == cls_of[k]:
                fixed_same += 1
            else:
                fixed_other += 1
        if base[i] and not new[i]:
            hijack = keys[merged[0]]
            if cls_of.get(hijack) == cls_of[k]:
                broke_same += 1
            else:
                broke_other += 1
        per_object[k] = {"matcher_top1": bool(base[i]), "reranked_top1": bool(new[i]),
                         "truth_rank_before": int(pos), "truth_rank_after": int(merged.index(i))}

    out: Dict[str, Any] = {
        "channel": channel, "view": VIEW, "objects": len(keys),
        "truth_in_shortlist": int(len(keys) - ranked_out), "truth_outside_shortlist": int(ranked_out),
        "fusion": {"rule": "rrf", "k": RRF_K, "w_matcher": W_MATCHER, "w_channel": W_CHANNEL},
        "bar": {"fit_top1_gain": BAR_TOP1, "confirmation_tolerance": BAR_CONFIRM},
    }
    fit = split.get("FIT", [])
    for name, sel in (("all", list(range(len(keys)))), ("FIT", fit),
                      ("DEV", split.get("DEV", [])), ("FRESH", split.get("FRESH", []))):
        if not sel:
            continue
        b = np.asarray([base[i] for i in sel], dtype=np.float64)
        n = np.asarray([new[i] for i in sel], dtype=np.float64)
        diff = n - b
        out[name] = {"n": len(sel), "top1_before": float(b.mean()), "top1_after": float(n.mean()),
                     "gain": float(diff.mean()), "ci95": _bootstrap(diff),
                     "mde": float(2.8 * float(np.std(diff, ddof=1) if len(sel) > 1 else 0.0)
                                  / max(np.sqrt(len(sel)), 1.0))}
    out["fixes_breaks"] = {"fixed_same_class": fixed_same, "fixed_other_class": fixed_other,
                          "broke_same_class": broke_same, "broke_other_class": broke_other}
    # WHY: does the channel order the eight like the matcher does?  A channel that agrees only by
    # chance cannot re-rank without costing accuracy; this number says whether the failure is the
    # fusion rule or the channel's information.  Spearman over the eight, averaged per object.
    rhos: List[float] = []
    for k in keys:
        rows = pairs.get(k, [])
        order = [idx[c] for c, _s in rows if c in idx]
        if len(order) < 4 or idx[k] not in order:
            continue
        i = idx[k]
        chan_order = sorted(order, key=lambda j: (-float(chan[i, j]), j))
        mr = np.asarray([order.index(j) for j in order], dtype=np.float64)
        cr = np.asarray([chan_order.index(j) for j in order], dtype=np.float64)
        rhos.append(float(np.corrcoef(mr, cr)[0, 1]))
    out["order_agreement_spearman"] = float(np.nanmean(rhos)) if rhos else float("nan")
    # the channel's own top-1 inside the eight against the matcher's, before any fusion
    top_same = 0
    for k in keys:
        rows = pairs.get(k, [])
        order = [idx[c] for c, _s in rows if c in idx]
        if not order or idx[k] not in order:
            continue
        chan_order = sorted(order, key=lambda j: (-float(chan[idx[k], j]), j))
        top_same += int(chan_order[0] == order[0])
    out["channel_top1_equals_matcher_top1"] = top_same
    out["per_object"] = per_object
    print(json.dumps({k: v for k, v in out.items() if k != "per_object"}, indent=1))
    prev = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
    prev[channel] = out
    with open(OUT, "w") as fh:
        json.dump(prev, fh, indent=1, sort_keys=True, default=float)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=("shape", "fragment"), default="shape")
    a = ap.parse_args(list(argv) if argv is not None else None)
    run(a.channel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
