"""M32 item 3: colour -- a real distance and a real aggregate, measured on the door.

    python Topology/m32_colour.py --channels   # cache the door's three channels per cell (records)
    python Topology/m32_colour.py --variants   # V1/V2/V3: colour alone, fused3 door, FIT/DEV/FRESH

Nothing here is fitted.  The distance and the sigma were fixed in writing before any score was
looked at:

    d^2 = sum_rings[(dL)^2 * 0.25 + (da)^2 + (db)^2]        scaled Lab units, two rings
    s   = exp(-d^2 / (2 sigma^2))
    sigma = median pairwise distance over a fixed label-free sample of pool places
            (np.random.default_rng(20261118), 4096 base-image places anyway), stored in the bundle

Aggregators, also fixed in writing: V1 max over places (the shipped one), V2 query-coverage (mean
over the query's places of the best match), V3 symmetric chamfer.

The cells are the door's: gallery = every object's base view (-000-000), queries = the cell's view.
FIT objects choose between V1/V2/V3; DEV + FRESH confirm.  BAR: mean FIT recall@8 gain over the
current fused3 >= +0.050 and the confirmation not below it by more than 0.025.
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

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m32_colour.json")
CH = os.path.join(ROOT, "Topology", "results", "placement", "m32_colour_channels.npz")
REF = "000-000"
S = 8
K = 60
GAIN_REQUIRED = 0.050
CONFIRM_TOLERANCE = -0.025
AGG = ("max", "coverage", "chamfer")


def _split() -> Dict[str, List[int]]:
    """The mission's FIT/DEV/FRESH split of the 80 objects, from the read's own manifest."""
    keys = [o["key"] for o in REC.objects()]
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(REC.objects()):
        out.setdefault(str(o["split"]), []).append(i)
    return out


# --------------------------------------------------------------------------- channels
def channels(views: Sequence[str]) -> Dict[str, np.ndarray]:
    """The door's counted + appearance channels for every (query view, gallery) cell."""
    from matching import placement_frontdoor as FD
    from matching import placement_gallery as GL
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    cells = PR.load_cells(RECORDS, [REF, *views])
    out: Dict[str, np.ndarray] = {}
    for v in views:
        qs = [cells[(k, v)] for k in keys]
        counted, _i = FD.count_scores(gal.index, qs, n_hits=8)
        app = FD.appearance_scores([REC.path_of(o, v) for o in objs],
                                   [REC.path_of(o, REF) for o in objs])
        out["counted|%s" % v] = np.asarray(counted, dtype=np.float64)
        out["appearance|%s" % v] = np.asarray(app, dtype=np.float64)
        print("  channels %s done" % v, flush=True)
    np.savez(CH, **out)
    return out


def _ch() -> Dict[str, np.ndarray]:
    if not os.path.exists(CH):
        raise SystemExit("run --channels first")
    z = np.load(CH)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- colour matrices
def pool_rows(keys: Sequence[str]) -> np.ndarray:
    """The label-free pool: every place of every base-view image (gallery, no labels)."""
    from matching import placement_records as PR
    cells = PR.load_cells(RECORDS, [REF])
    return np.asarray([p["content"] for k in keys for p in cells[(k, REF)]["places"]
                       if p.get("content") is not None], dtype=np.float64)


def colour_matrix(view: str, how: str, sig: float, keys: Sequence[str]) -> np.ndarray:
    """(80, 80) colour scores: query object under ``view`` against every base-view object."""
    from matching import placement_content as PC
    from matching import placement_records as PR
    cells = PR.load_cells(RECORDS, [REF, view])
    def packs(k: str, v: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for p in cells[(k, v)]["places"]:
            if p.get("content") is None:
                continue
            out.setdefault(str(p.get("lens")), []).append(p["content"])
        return {k2: np.asarray(v2, dtype=np.float64) for k2, v2 in out.items()}
    G = [packs(k, REF) for k in keys]
    Q = [packs(k, view) for k in keys]
    n = len(keys)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = PC.aggregate(Q[i], G[j], sig, how)
    return M


def metrics(M: np.ndarray, label: str) -> Dict[str, Any]:
    n = M.shape[0]
    order = np.argsort(-M, axis=1, kind="stable")
    ranks = np.asarray([int(np.nonzero(order[i] == i)[0][0]) for i in range(n)])
    tr = np.asarray([M[i, i] for i in range(n)])
    off = M[~np.eye(n, dtype=bool)]
    out = {"label": label, "top1": float((ranks == 0).mean()),
           "recall@8": float((ranks < S).mean()), "truth_rank_median": float(np.median(ranks)),
           "share_ge_0.99": float((off >= 0.99).mean()),
           "share_ge_0.999": float((off >= 0.999).mean()),
           "truth_median": float(np.median(tr)), "impostor_median": float(np.median(off)),
           "gap_medians": float(np.median(tr) - np.median(off)),
           "distinct_scores": int(np.unique(np.round(off, 12)).size)}
    return out


def _rank_matrix(M: np.ndarray) -> np.ndarray:
    return np.argsort(-M, axis=1, kind="stable")


def _recall8(M: np.ndarray, idx: Sequence[int]) -> float:
    n = M.shape[0]
    order = _rank_matrix(M)
    return float(np.mean([i in order[i][:S] for i in idx]))


def _paired(Ma: np.ndarray, Mb: np.ndarray, idx: Sequence[int]) -> Dict[str, int]:
    oa, ob = _rank_matrix(Ma), _rank_matrix(Mb)
    up = down = tie = 0
    for i in idx:
        a, b = i in oa[i][:S], i in ob[i][:S]
        if b and not a:
            up += 1
        elif a and not b:
            down += 1
        else:
            tie += 1
    return {"up": up, "down": down, "tied": tie}


def variants() -> Dict[str, Any]:
    from matching import placement_content as PC
    from matching import placement_fuse as FU
    keys = [o["key"] for o in REC.objects()]
    sig = PC.sigma(pool_rows(keys))
    ch = _ch()
    split = _split()
    views = {"fit_a": "035-045", "fit_b": "045-180"}
    res: Dict[str, Any] = {"sigma": sig, "lambda_L": PC.LAMBDA_L, "sigma_seed": PC.SIGMA_SEED,
                           "split": {k: len(v) for k, v in split.items()}, "cells": {}}
    print("sigma %.6f over the base-view pool" % sig, flush=True)
    for cell, view in views.items():
        counted = ch["counted|%s" % view]
        app = ch["appearance|%s" % view]
        from matching import placement_records as PR
        gal_cells = [PR.load_cell(RECORDS, k, REF) for k in keys]
        qs = [PR.load_cell(RECORDS, k, view) for k in keys]
        old = np.asarray(FU.colour_channel(gal_cells, qs), dtype=np.float64)
        base_fused = np.asarray(FU.fuse(counted, app, old, weights={"colour": 0.5}, k=K),
                                dtype=np.float64)
        cellout: Dict[str, Any] = {"view": view,
                                   "base_fused3": metrics(base_fused, "fused3 shipped colour"),
                                   "base_colour": metrics(old, "colour shipped cosine")}
        for how in AGG:
            M = colour_matrix(view, how, sig, keys)
            fused = np.asarray(FU.fuse(counted, app, M, weights={"colour": 0.5}, k=K),
                               dtype=np.float64)
            cellout[how] = {"colour": metrics(M, "colour %s" % how),
                            "fused3": metrics(fused, "fused3 colour %s" % how),
                            "paired_fit": _paired(base_fused, fused, split["FIT"]),
                            "paired_confirm": _paired(base_fused, fused,
                                                      split["DEV"] + split["FRESH"])}
            print("  %s %-8s colour r@8 %.3f (s>=0.99 %.3f) | fused3 r@8 %.3f vs base %.3f"
                  % (cell, how, cellout[how]["colour"]["recall@8"],
                     cellout[how]["colour"]["share_ge_0.99"],
                     cellout[how]["fused3"]["recall@8"], cellout["base_fused3"]["recall@8"]),
                  flush=True)
        res["cells"][cell] = cellout
    # the decision: mean FIT recall@8 gain, confirmation not below by more than the tolerance
    dec: Dict[str, Any] = {}
    for how in AGG:
        gains = [res["cells"][c][how]["fused3"]["recall@8"] - res["cells"][c]["base_fused3"]["recall@8"]
                 for c in views]
        conf = [res["cells"][c][how]["fused3"]["recall@8"] - res["cells"][c]["base_fused3"]["recall@8"]
                for c in views]
        dec[how] = {"fit_gain_mean": float(np.mean(gains)),
                    "confirm_gain_mean": float(np.mean(conf)),
                    "passes": bool(np.mean(gains) >= GAIN_REQUIRED
                                   and np.mean(conf) >= CONFIRM_TOLERANCE)}
    res["decision"] = dec
    res["winner"] = max([k for k in AGG if dec[k]["passes"]],
                        key=lambda k: dec[k]["confirm_gain_mean"], default=None)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps({"sigma": sig, "decision": dec, "winner": res["winner"]}, indent=1))
    print("wrote", OUT)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--variants", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.channels:
        channels(["035-045", "045-180"])
        return 0
    if a.variants:
        variants()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
