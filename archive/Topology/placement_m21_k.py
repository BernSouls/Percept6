"""MISSION 21 / O1 -- the bag K against the matcher's K, settled by a measurement.

The Mission-20 bag used ASSIGN_K = 6 while the module docstring and the mission text said 8,
and the matcher's configuration descriptor uses K = 24.  Three numbers, one descriptor.  O1
asks for one stated K or a measurement that justifies the split, so this file measures the
only thing that matters for a shortlist: recall@8 on the 40-item mixed gallery, as a function
of K.

The descriptors are SORTED by distance, so the first K slots of the K = 24 descriptor are
exactly the K-slot descriptor; truncating the cached bags is therefore lossless for K = 6 and
K = 12 and needs no graph rebuild.  Each K gets its OWN frozen standardization, fitted on the
same FIT fit-half, so the comparison is between descriptors and not between rulers.

Reported: recall@8 per K on both halves with n, and the same for the assignment's ASSIGN_K
(6) as the control it is.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
KS = (6, 12, 24)


def truncated(bag: Dict[str, Any], k: int) -> Dict[str, Any]:
    return {"D": bag["D"][:, :k].copy(), "A": bag["A"][:, :k].copy(),
            "M": bag["M"][:, :k].copy(), "ids": list(bag["ids"])}


def fit_for(bags: Dict[str, Any], paths: List[str], k: int) -> Dict[str, Any]:
    from Topology.placement_m21_fit import fit_standardization
    sub = {p: {"bag_v2": truncated(bags[p]["bag_v2"], k)} for p in paths}
    st = fit_standardization(sub, paths, k)
    st.update({"sigma_d": 0.35, "sigma_a": 0.50})
    return st


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="m21_k")
    ap.add_argument("--ks", default="6,12",
                    help="K values to build; K=24 is already measured by the shortlist run")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_prefilter as PF
    from Topology.placement_m20_eth80 import objects
    from Topology.placement_m21_fit import ensure_bags
    from Topology.placement_m21_shortlist import score_tasks
    if not P.PLACEMENT_M19:
        raise SystemExit("set PLACEMENT_M19=1")
    objs = objects("FIT")
    bags = ensure_bags("FIT", a.workers, per_class=5)
    even = [o for i, o in enumerate(objs) if i % 2 == 0]
    fit_paths = sorted({o["ref"] for o in even} | {o["query"] for o in even})
    gal = sorted({o["ref"] for o in objs})
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    res: Dict[str, Any] = {"ks": ks, "gallery": len(gal), "S": 8, "cells": {},
                           "note": ("K = 24 is the shipped descriptor and its recall is in "
                                    "m21_shortlist_fit_pc5.json; this file builds the smaller "
                                    "Ks from the same cached, distance-sorted bags")}
    for k in ks:
        st = fit_for(bags, fit_paths, k)
        keep = np.asarray(st["keep_D"], dtype=bool)
        std_g = {p: PF.standardize(truncated(bags[p]["bag_v2"], k), st) for p in gal}
        std_q = {o["query"]: PF.standardize(truncated(bags[o["query"]]["bag_v2"], k), st)
                 for o in objs}
        M = np.zeros((len(objs), len(gal)), dtype=np.float64)
        t0 = time.time()
        for i, o in enumerate(objs):
            for j, r in enumerate(gal):
                s = PF.score_bags_v2(std_q[o["query"]], std_g[r], st)["score"]
                M[i, j] = 0.0 if s is None else float(s)
        sec = (time.time() - t0) / (len(objs) * len(gal))
        cell = {"K": k, "slots_kept": int(keep.sum()), "sec_per_pair": sec,
                "tasks": {}, "halves": {}}
        for task in ("T1", "T3"):
            cell["tasks"][task] = {kk: vv for kk, vv in
                                   score_tasks(M, objs, gal, task, 8).items()
                                   if kk != "rows"}
        idx = [i for i, o in enumerate(objs) if o["query"] in {e["query"] for e in even}]
        odd = [i for i in range(len(objs)) if i not in idx]
        for name, ii in (("fit_half", idx), ("report_half", odd)):
            r = score_tasks(M[ii], [objs[i] for i in ii], gal, "T3", 8)
            cell["halves"][name] = {"recall8_T3": r["recall"], "n": r["n"]}
        res["cells"][str(k)] = cell
        print("K=%-3d slots kept %-2d  T3 recall@8 %.3f (n=%d)  fit-half %.3f  "
              "report-half %.3f  %.3f s/pair"
              % (k, cell["slots_kept"], cell["tasks"]["T3"]["recall"],
                 cell["tasks"]["T3"]["n"], cell["halves"]["fit_half"]["recall8_T3"],
                 cell["halves"]["report_half"]["recall8_T3"], sec), flush=True)
    out = os.path.join(OUT, a.tag + ".json")
    from matching import m15_manifest as MF
    MF.write_result(out, res)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
