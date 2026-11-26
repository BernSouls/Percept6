"""Mission 13 step 4 -- refit w by PAIRWISE MARGIN, not softmax.

Mission 12 fitted (T, w) to a softmax over the shortlist and landed on a corner
(`w = 1.55`, `T = 0.015`, both outside the declared bounds) because `S/T` is a pure
sharpening knob: the softmax likelihood has no interior maximum in T at all.  This
step removes the knob (T is fixed at the deployed 1.0) and replaces the objective
with a BRADLEY-TERRY PAIRWISE MARGIN likelihood over the same bank:

    for every query q, every ordered pair (true candidate t, rival c):
        m(t, c) = ( B(t) - B(c) ) + w * ( E(t) - E(c) )
        ll(w)   = sum over pairs of log sigmoid( m(t, c) )

`B` is the deployed shipped score, `E` the placement evidence on the STEP-2 scale.
A softmax over a 4-way shortlist can be driven by the *level* of the scores; a
pairwise margin only ever sees DIFFERENCES, and w is then identified exactly by the
trade-off between the two feature differences across pairs -- if the bank contains
that trade-off.  When it does not, the profile runs to a bound and the answer is
"unidentifiable, keep the shipped weight".  Both outcomes are reported as such.

STRATA ARE CUT BEFORE FITTING (declared here, not chosen by looking at w):

    S1 in-plane easy    5 / 10 / 15 deg
    S2 in-plane hard   25 / 30 / 35 deg
    S3 out-of-plane    y10 / y20 / y30 / x10 / x20
    S4 scale           s09 / s115 and the remaining views

Every stratum is fitted on its own and the pooled fit is reported separately, so a
weight that only works in one regime cannot hide behind the pool.

    python Topology/placement_m13_margin.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
BANK = os.path.join(OUT, "m12_fit_w_progress.jsonl")
RESULT = os.path.join(OUT, "m13_margin.json")

SHIPPED_W = 0.375          # deployed
WS = np.round(np.arange(0.0, 3.0001, 0.01), 3)
STRATA: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("S1_inplane_easy", ("i05", "i10", "i15")),
    ("S2_inplane_hard", ("i25", "i30", "i35")),
    ("S3_out_of_plane", ("y10", "y20", "y30", "x10", "x20")),
    ("S4_scale", ("s09", "s115")),
)


def _load() -> List[Dict[str, Any]]:
    from Topology import placement_m12_fit_w as F
    shorts = F.shortlists()
    truth_of = {q: shorts[q][0] for q in shorts}
    qs, _s = F.queries(), None
    held = {q for i, (q, *_r) in enumerate(qs) if i % 4 == 3}
    rows = []
    with open(BANK) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            d["_truth"] = truth_of.get(d["q"])
            d["_held"] = d["q"] in held
            d["_view"] = d["q"].split("|")[-1]
            rows.append(d)
    return rows


def _pairs(rows: List[Dict[str, Any]]) -> List[Tuple[float, float, float, str, bool]]:
    """(dB, dE, y, view, held) for every ordered (truth, rival) pair."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    out = []
    for q, cands in by.items():
        truth = cands[0]["_truth"]
        t = [r for r in cands if r["cand"] == truth]
        if not t:
            continue
        t = t[0]
        for r in cands:
            if r["cand"] == truth:
                continue
            out.append((t["B"] - r["B"], t["E"] - r["E"], 1.0,
                        t["_view"], bool(t["_held"])))
    return out


def _ll(pairs, w: float) -> float:
    d = np.asarray([p[0] + w * p[1] for p in pairs])
    return float(np.sum(-np.logaddexp(0.0, -d)))


def _profile(pairs, ws=WS) -> np.ndarray:
    return np.asarray([_ll(pairs, w) for w in ws])


def _fit(pairs, label: str) -> Dict[str, Any]:
    prof = _profile(pairs)
    i = int(np.argmax(prof))
    w = float(WS[i])
    # interiority: strictly inside the grid AND the profile must bend down
    interior = 0 < i < len(WS) - 1
    left, right = prof[max(i - 1, 0)], prof[min(i + 1, len(WS) - 1)]
    curved = bool(prof[i] > left and prof[i] >= right)
    # profile-likelihood interval: {w : 2*(ll_max - ll(w)) <= 3.841}
    drop = 2.0 * (prof[i] - prof)
    inside = WS[drop <= 3.841]
    dE = np.asarray([p[1] for p in pairs])
    dB = np.asarray([p[0] for p in pairs])
    # does E carry ANY information on its own?  (sign test on dE)
    return {"label": label, "n_pairs": len(pairs),
            "w_hat": w, "interior": bool(interior and curved),
            "at_bound": bool((not interior) or (i == len(WS) - 1)),
            "ll_at_hat": float(prof[i]), "ll_at_shipped": _ll(pairs, SHIPPED_W),
            "ll_at_zero": _ll(pairs, 0.0),
            "profile_w_min": float(inside.min()) if inside.size else float("nan"),
            "profile_w_max": float(inside.max()) if inside.size else float("nan"),
            "share_dE_positive": float(np.mean(dE > 0)),
            "corr_dB_dE": float(np.corrcoef(dB, dE)[0, 1]) if len(pairs) > 2 else float("nan"),
            "profile": [{"w": float(w_), "ll": float(v)} for w_, v in
                        zip(WS[::10], prof[::10])]}


def _margin(rows, w: float) -> Dict[str, float]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    margins, top1 = [], 0
    for q, cands in by.items():
        truth = cands[0]["_truth"]
        s = {r["cand"]: r["B"] + w * r["E"] for r in cands}
        if truth not in s:
            continue
        rival = max((v for k, v in s.items() if k != truth), default=float("-inf"))
        margins.append(s[truth] - rival)
        top1 += 1 if s[truth] >= rival else 0
    return {"mean_margin": float(np.mean(margins)) if margins else float("nan"),
            "top1": float(top1 / len(margins)) if margins else float("nan"),
            "n": len(margins)}


def main() -> int:
    rows = _load()
    pairs = _pairs(rows)
    held = [p for p in pairs if p[4]]
    train = [p for p in pairs if not p[4]]
    out: Dict[str, Any] = {"bank": BANK, "n_rows": len(rows), "n_pairs": len(pairs),
                           "shipped_w": SHIPPED_W, "pooled": {}, "strata": {}}
    out["pooled"]["train"] = _fit(train, "pooled_train")
    out["pooled"]["all"] = _fit(pairs, "pooled_all")
    for name, views in STRATA:
        sub = [p for p in pairs if p[3] in views]
        out["strata"][name] = (_fit(sub, name) if len(sub) >= 8 else
                               {"label": name, "n_pairs": len(sub), "skipped": True})
    # THE DECISIVE SUBSET.  E and B agree in sign on 84 % of pairs, and on those the
    # margin is monotone in w by construction -- they carry no information about w.
    # Only the pairs where the two disagree can identify it, so they are fitted on
    # their own; a weight that the informative subset rejects is not adoptable even
    # if the pooled profile runs away.
    dec = [p for p in pairs if p[1] != 0 and np.sign(p[1]) != np.sign(p[0])]
    out["decisive_subset"] = _fit(dec, "decisive") if len(dec) >= 8 else {"skipped": True}
    out["decisive_curve"] = [
        {"w": w, "mean_margin": float(np.mean([p[0] + w * p[1] for p in dec])),
         "frac_correct": float(np.mean([(p[0] + w * p[1]) > 0 for p in dec]))}
        for w in (0.0, 0.25, SHIPPED_W, 0.5, 1.0, 2.0, 3.0)] if dec else []
    # held-out margin: the only number that decides adoption
    hr = [r for r in rows if r["_held"]]
    out["held_out"] = {("w=%.3f" % w): _margin(hr, w)
                       for w in (0.0, SHIPPED_W, 1.0, 2.0,
                                 float(out["pooled"]["train"]["w_hat"]))}
    out["train_margin"] = {("w=%.3f" % w): _margin([r for r in rows if not r["_held"]], w)
                           for w in (0.0, SHIPPED_W, float(out["pooled"]["train"]["w_hat"]))}
    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k in ("pooled",):
        for kk, v in out[k].items():
            if "skipped" in v:
                continue
            print("%-14s %-12s pairs %5d  w_hat %.2f  interior %s  ll %.1f  (ll0 %.1f, ll.375 %.1f)"
                  "  profile [%.2f, %.2f]  P(dE>0) %.3f  corr %.3f"
                  % (k, kk, v["n_pairs"], v["w_hat"], v["interior"], v["ll_at_hat"],
                     v["ll_at_zero"], v["ll_at_shipped"], v["profile_w_min"],
                     v["profile_w_max"], v["share_dE_positive"], v["corr_dB_dE"]))
    for name, v in out["strata"].items():
        if "skipped" in v:
            print("%-18s skipped (n=%d)" % (name, v["n_pairs"]))
            continue
        print("%-18s pairs %5d  w_hat %.2f  interior %s  profile [%.2f, %.2f]  corr %.3f"
              % (name, v["n_pairs"], v["w_hat"], v["interior"], v["profile_w_min"],
                 v["profile_w_max"], v["corr_dB_dE"]))
    print("held-out:", json.dumps({k: round(v["mean_margin"], 4) for k, v in out["held_out"].items()}))
    print("held-out top1:", json.dumps({k: round(v["top1"], 4) for k, v in out["held_out"].items()}))
    print("train margin:", json.dumps({k: round(v["mean_margin"], 4) for k, v in out["train_margin"].items()}))
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
