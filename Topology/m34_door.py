"""M34 item 4: the fragment channel FUSED INTO THE DOOR, at a stated discount, measured on records.

The screens (item 3) put the fragment channel at 0.725 / 0.637 channel-alone recall@8 on the two FIT
views.  That is a channel, not a door.  This file does the one read the mission allows:

    door = rrf(counted 1.0, appearance 1.0, colour 0.5, fragment 0.5),  k = 60

The discount is the tree's registered one (`placement_fuse.COLOUR_WEIGHT` / `SHAPE_WEIGHT`): a
channel that has never been read in the shipped path may reorder the head, it may not outvote the
configuration.  No sweep: 0.5 was fixed by M30/M33, not chosen here.  The fragment channel's own
score for a (query, gallery) pair is the symmetric mean-of-best of item 3, unchanged.

STATES BEFORE THE READ.  n = 40 FIT objects per view, 80-object gallery, chance 0.100.  The bar is
**+0.050 mean FIT recall@8** with an object-clustered CI that excludes zero.  The paired difference
is a mean of per-object 0/±1 indicators, so its standard error at n = 80 (both views pooled) is
sqrt(p(1-p)/n) ~ 0.055 for p ~ 0.7; **MDE (2.8 SE) ~ 0.153**.  A gain smaller than that cannot be
distinguished from zero on this read, and saying so before the number is the point.

    python Topology/m34_door.py --run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
CHANNELS = os.path.join(ROOT, "Topology", "results", "placement", "m32_colour_channels.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m34_door.json")
REF = "000-000"
VIEWS = {"fit_a": "035-045", "fit_b": "045-180"}
S = 8
K = 60
W_FRAGMENT = 0.5
BAR = 0.050
RESAMPLES = 10000
SEED = 20261118


def _split() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(REC.objects()):
        out.setdefault(str(o["split"]), []).append(i)
    return out


def fragment_matrix(view: str, grouping: str = "contour_fragment") -> np.ndarray:
    """80 x 80 coverage scores from the M34 cache -- the item-3 rule, unchanged."""
    import Topology.m34_fragments as M
    from matching import placement_fragment as PF
    cells = M._load_cells((REF, view))
    keys = sorted({k for k, _v in cells})
    pool = np.vstack([cells[(k, REF)]["%s|desc" % grouping] for k in keys])
    sig = PF.scales(pool)
    out = np.zeros((len(keys), len(keys)), dtype=np.float64)
    for i, k in enumerate(keys):
        for j, gk in enumerate(keys):
            block, _g = M._fragment_matrix(cells[(k, view)], cells[(gk, REF)], grouping, sig)
            out[i, j] = M._coverage(block)
    return out


def _ranks_hit(M: np.ndarray, i: int) -> int:
    return int(np.nonzero(np.argsort(-M[i], kind="stable") == i)[0][0])


def _boot(diff: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    m = diff[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run() -> Dict[str, Any]:
    from matching import placement_fuse as F
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls = [o["cls"] for o in objs]
    split = _split()
    z = np.load(CHANNELS)
    out: Dict[str, Any] = {"view_weights": {"counted": 1.0, "appearance": 1.0, "colour": 0.5,
                                            "fragment": W_FRAGMENT},
                           "rrf_k": K, "bar": BAR, "mde_stated": 0.153, "views": {}}
    pooled_base: List[float] = []
    pooled_new: List[float] = []
    for name, view in VIEWS.items():
        cells = PR.load_cells(RECORDS, [REF, view])
        gal = [{"places": cells[(k, REF)]["places"]} for k in keys]
        qs = [{"places": cells[(k, view)]["places"]} for k in keys]
        colour = F.colour_channel(gal, qs)
        frag = fragment_matrix(view)
        counted = z["counted|%s" % view]
        app = z["appearance|%s" % view]
        base = F.rrf_weighted([counted, app, colour], [1.0, 1.0, 0.5], k=K)
        new = F.rrf_weighted([counted, app, colour, frag], [1.0, 1.0, 0.5, W_FRAGMENT], k=K)
        rb = np.asarray([_ranks_hit(base, i) for i in range(len(keys))])
        rn = np.asarray([_ranks_hit(new, i) for i in range(len(keys))])
        fit = split["FIT"]
        db = (rb[fit] < S).astype(np.float64)
        dn = (rn[fit] < S).astype(np.float64)
        out["views"][name] = {
            "n_fit": len(fit),
            "baseline_recall8": float(db.mean()), "fused_recall8": float(dn.mean()),
            "gain": float((dn - db).mean()),
            "fixed": int(((db == 0) & (dn == 1)).sum()), "broke": int(((db == 1) & (dn == 0)).sum()),
            "baseline_all80": float((rb < S).mean()), "fused_all80": float((rn < S).mean()),
            "per_class_gain": {c: float((dn - db)[np.asarray(cls)[fit] == c].mean())
                               for c in sorted(set(cls))},
        }
        pooled_base.append(db)
        pooled_new.append(dn)
        print("  %-6s %-8s base %.3f fused %.3f gain %+.4f (fixed %d, broke %d)"
              % (name, view, db.mean(), dn.mean(), (dn - db).mean(),
                 out["views"][name]["fixed"], out["views"][name]["broke"]), flush=True)
    b = np.concatenate(pooled_base)
    n = np.concatenate(pooled_new)
    diff = n - b
    out["pooled"] = {"n": int(diff.size), "baseline": float(b.mean()), "fused": float(n.mean()),
                     "mean_gain": float(diff.mean()), "ci95": _boot(diff),
                     "mde": float(2.8 * float(np.std(diff, ddof=1)) / np.sqrt(diff.size))}
    out["verdict"] = ("clears" if out["pooled"]["mean_gain"] >= BAR
                      and out["pooled"]["ci95"][0] > 0.0 else "does not clear")
    print(json.dumps({k: v for k, v in out.items() if k != "views"}, indent=1))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if viz := a.run:
        run()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
