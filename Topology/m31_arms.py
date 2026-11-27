"""M31 item 3: use colour where the matcher loses -- four arms over the ALREADY SCORED shortlists.

    python Topology/m31_arms.py --oracle      # no fusion: which channel ranks the truth best?
    python Topology/m31_arms.py --arms        # A/B/C/D, chosen on FIT, confirmed on the rest

No new pairs are scored: every number here is a re-ranking of the pairs `m30_read.py` already
scored (`Topology/results/placement/m30_read_pairs.jsonl`), joined to the door's own channels
computed from the committed records.

The arms, weights fixed at equal (no fitting, no sweep):

    A  matcher order                     -- the shipped re-order, the incumbent
    B  door order                        -- the fused door's order, the matcher ignored
    C  RRF(matcher rank, door rank)      -- k = 60, equal weights
    D  RRF(matcher rank, colour rank)    -- k = 60, equal weights, colour = the M28 rings

Chosen on FIT; the rest (DEV + FRESH) confirms.  Ship only if the FIT top-1 gain over A is
>= +0.075 AND the confirmation split is not below A by more than 0.025.
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

from Topology import m26_records as REC                       # noqa: E402
from Topology.m27_read import RESAMPLES, SEED                  # noqa: E402

PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m31_arms.json")
REF = "000-000"
VIEW = "035-045"
S = 8
K = 60
GAIN_REQUIRED = 0.075
CONFIRM_TOLERANCE = -0.025


def channels() -> Dict[str, Any]:
    """The door's three channels for every query object against the gallery, from the records."""
    from matching import placement_frontdoor as FD
    from matching import placement_fuse as FU
    from matching import placement_gallery as GL
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    cells = PR.load_cells(RECORDS, [REF, VIEW])
    gal_cells = [cells[(k, REF)] for k in keys]
    qs = [cells[(k, VIEW)] for k in keys]
    counted, _info = FD.count_scores(gal.index, qs, n_hits=8)
    app = FD.appearance_scores([REC.path_of(o, VIEW) for o in objs],
                               [REC.path_of(o, REF) for o in objs])
    colour = np.asarray(FU.colour_channel(gal_cells, qs), dtype=np.float64)
    fused = np.asarray(FU.fuse(counted, app, colour, weights={"colour": 0.5}, k=K),
                       dtype=np.float64)
    shipped = np.asarray(FD.rank_fuse(app, counted, k=K), dtype=np.float64)
    return {"keys": keys, "objs": objs, "counted": counted, "appearance": app, "colour": colour,
            "fused": fused, "shipped": shipped}


def _load_pairs() -> Dict[str, Dict[str, float]]:
    """{query object: {gallery object key: score}} for the objects with all S pairs scored."""
    by_q: Dict[str, Dict[str, float]] = {}
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if r.get("score") is None:
            continue
        k = os.path.dirname(os.path.basename(r["r"]))
        by_q.setdefault(r["object"], {})[str(r["r"])] = float(r["score"])
    return by_q


def _rank(scores: Dict[str, float], keys: Sequence[str]) -> Dict[str, int]:
    """1-based rank by descending score, ties by key (deterministic)."""
    order = sorted(keys, key=lambda k: (-scores[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def _top1(ranks: Dict[str, int], truth: str, shortlist: Sequence[str]) -> int:
    return int(min(shortlist, key=lambda k: (ranks[k], k)) == truth)


def _ci(hits: Sequence[int]) -> Dict[str, Any]:
    v = np.asarray(list(hits), dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "top1": None}
    rng = np.random.default_rng(SEED)
    boot = np.asarray([v[rng.integers(0, v.size, v.size)].mean() for _ in range(RESAMPLES)])
    return {"n": int(v.size), "hits": int(v.sum()), "top1": float(v.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "resamples": RESAMPLES, "seed": SEED}


def oracle(ch: Dict[str, Any], pairs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Of the matcher's losses inside its shortlist, how often does each channel rank the truth
    above the matcher's winner?  No fusion -- this is the mechanism, counted."""
    keys, objs = ch["keys"], ch["objs"]
    path_of = {k: os.path.join(os.path.dirname(keys and ""), "") for k in keys}   # unused
    qpath = {o["key"]: REC.path_of(o, VIEW) for o in objs}
    rpath_key = {}
    for k in keys:
        o = next(x for x in objs if x["key"] == k)
        rpath_key[REC.path_of(o, REF)] = k
    out = {"losses": 0, "same_class": 0, "other_class": 0,
           "colour_above": 0, "appearance_above": 0, "counted_above": 0, "any_above": 0}
    for qi, qk in enumerate(keys):
        got = pairs.get(qk) or {}
        if len(got) < S:
            continue
        short = [rpath_key[p] for p in got if p in rpath_key]
        if len(short) != len(got):
            continue
        truth = qk
        if truth not in short:
            continue                                     # the door refused it; no matcher loss
        winner = max(short, key=lambda k: (got[[p for p in got if rpath_key[p] == k][0]], k))
        if winner == truth:
            continue
        out["losses"] += 1
        same = winner.split("/")[0] == truth.split("/")[0]
        out["same_class"] += int(same)
        out["other_class"] += int(not same)
        above = []
        for name, mat in (("colour", ch["colour"]), ("appearance", ch["appearance"]),
                          ("counted", ch["counted"])):
            gi = {k: j for j, k in enumerate(keys)}
            a = float(mat[qi, gi[truth]])
            b = float(mat[qi, gi[winner]])
            above.append(a > b)
            if name == "colour" and a > b:
                out["colour_above"] += 1
            if name == "appearance" and a > b:
                out["appearance_above"] += 1
            if name == "counted" and a > b:
                out["counted_above"] += 1
        out["any_above"] += int(any(above))
    out["colour_share"] = out["colour_above"] / max(1, out["losses"])
    print("  oracle: %d losses (%d same-class, %d other-class); colour ranks the truth above the "
          "winner in %d (%.1f%%), appearance %d, counted %d, any %d"
          % (out["losses"], out["same_class"], out["other_class"], out["colour_above"],
             100.0 * out["colour_share"], out["appearance_above"], out["counted_above"],
             out["any_above"]), flush=True)
    return out


def arms(ch: Dict[str, Any], pairs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    keys, objs = ch["keys"], ch["objs"]
    gi = {k: j for j, k in enumerate(keys)}
    split_of = {o["key"]: o["split"] for o in objs}
    rpath_key = {REC.path_of(o, REF): o["key"] for o in objs}
    rows: Dict[str, Dict[str, List[int]]] = {}
    hits_by_obj: Dict[str, Dict[str, int]] = {}
    used = 0
    for qi, qk in enumerate(keys):
        got = pairs.get(qk) or {}
        if len(got) < S:
            continue
        short = [rpath_key[p] for p in got if p in rpath_key]
        if len(short) != len(got):
            continue
        used += 1
        matcher_scores = {k: got[[p for p in got if rpath_key[p] == k][0]] for k in short}
        door_scores = {k: float(ch["fused"][qi, gi[k]]) for k in short}
        colour_scores = {k: float(ch["colour"][qi, gi[k]]) for k in short}
        rm = _rank(matcher_scores, short)
        rd = _rank(door_scores, short)
        rc = _rank(colour_scores, short)
        sp = split_of[qk]
        hits_by_obj.setdefault(qk, {})
        for arm, ranks in (
                ("A", rm),
                ("B", rd),
                ("C", {k: (1.0 / (K + rm[k]) + 1.0 / (K + rd[k])) for k in short}),
                ("D", {k: (1.0 / (K + rm[k]) + 1.0 / (K + rc[k])) for k in short})):
            if arm in ("A", "B"):
                hit = _top1(ranks, qk, short)
            else:
                order = sorted(short, key=lambda k: (-ranks[k], k))
                hit = int(order[0] == qk)
            rows.setdefault(arm, {}).setdefault(sp, []).append(hit)
            hits_by_obj[qk][arm] = hit
    out: Dict[str, Any] = {"objects": used, "S": S, "rrf_k": K, "arms": {}}
    for arm in ("A", "B", "C", "D"):
        per = {sp: _ci(v) for sp, v in sorted(rows.get(arm, {}).items())}
        allv = [h for v in rows.get(arm, {}).values() for h in v]
        out["arms"][arm] = {"by_split": per, "all": _ci(allv)}
        print("  arm %s: ALL %d/%d = %.3f  FIT %s  conf %s"
              % (arm, out["arms"][arm]["all"]["hits"], out["arms"][arm]["all"]["n"],
                 out["arms"][arm]["all"]["top1"],
                 per.get("FIT", {}).get("top1"), per.get("DEV", {}).get("top1")), flush=True)
    for arm in ("B", "C", "D"):
        gained = [k for k, v in hits_by_obj.items() if v.get(arm) and not v.get("A")]
        lost = [k for k, v in hits_by_obj.items() if v.get("A") and not v.get(arm)]
        out["arms"][arm]["gained_vs_A"] = {"n": len(gained), "objects": sorted(gained)}
        out["arms"][arm]["lost_vs_A"] = {"n": len(lost), "objects": sorted(lost)}
        print("    %s vs A: +%d hits, -%d hits" % (arm, len(gained), len(lost)), flush=True)
    fit_a = out["arms"]["A"]["by_split"].get("FIT", {}).get("top1")
    conf_a = [v["top1"] for sp, v in out["arms"]["A"]["by_split"].items() if sp != "FIT"]
    best, best_gain = "A", 0.0
    for arm in ("B", "C", "D"):
        fit = out["arms"][arm]["by_split"].get("FIT", {}).get("top1")
        if fit is None or fit_a is None:
            continue
        gain = float(fit) - float(fit_a)
        conf = [v["top1"] for sp, v in out["arms"][arm]["by_split"].items() if sp != "FIT"]
        worst = (min(conf) - min(conf_a)) if conf and conf_a else None
        ok = gain >= GAIN_REQUIRED and (worst is None or worst >= CONFIRM_TOLERANCE)
        out["arms"][arm]["fit_gain_over_A"] = gain
        out["arms"][arm]["confirm_delta_vs_A"] = worst
        out["arms"][arm]["passes"] = bool(ok)
        print("    %s: FIT gain %+.3f, confirm delta %s -> %s"
              % (arm, gain, ("%+.3f" % worst) if worst is not None else "n/a",
                 "PASS" if ok else "fail"), flush=True)
        if ok and gain > best_gain:
            best, best_gain = arm, gain
    out["chosen"] = best
    out["rule"] = {"gain_required": GAIN_REQUIRED, "confirm_tolerance": CONFIRM_TOLERANCE,
                   "weights": "equal (RRF k = %d), no fitting, no sweep" % K}
    print("  chosen: %s (FIT gain %+.3f)" % (best, best_gain), flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--arms", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    data: Dict[str, Any] = {}
    if os.path.exists(OUT):
        try:
            data = json.load(open(OUT))
        except Exception:
            data = {}
    if a.oracle or a.arms:
        ch = channels()
        pairs = _load_pairs()
        data["scored_objects"] = len(pairs)
        if a.oracle:
            data["oracle"] = oracle(ch, pairs)
        if a.arms:
            data["arms_result"] = arms(ch, pairs)
        with open(OUT, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True, default=float)
        print("wrote", OUT)
    if not (a.oracle or a.arms):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
