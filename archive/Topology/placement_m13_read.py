"""Mission 13 step 5 -- the real-photo read, re-registered BY REGIME and taken once.

Mission 12 took one pooled read over four transforms and reported a single interval;
that pooled interval (P - N = -0.002 [-0.012, +0.007]) is uninterpretable, because it
mixes an exact shift (where correspondence works: 0.76-0.79) with rotations and scale
changes (where Step 2 measured detection at 0.09-0.17 -- the neighbourhood of a node
mostly does not exist after the transform).  Pooling a working regime with a broken one
produces a null that says nothing about either.

THIS REGISTRATION IS SIGNED BEFORE THE READ RUNS (`_registration()`, written to
`m13_prereg_read.json` at first use) and fixes, in order:

1. REGIMES, each its own stratum, never pooled with another:

   PRIMARY    trans8     exact 8-px shift: the regime the shipped placement claims
   SECONDARY  rot05     5 deg in-plane: declared BROKEN (Step 2/3 floor ~0.06)
   SECONDARY  rot15     15 deg in-plane: declared BROKEN
   SECONDARY  scale115  1.15x: declared BROKEN

2. ARMS.  A = placement OFF (shipped aligner).  B = placement ON, evidence OFF.
   C = placement ON, evidence ON, w = the shipped 0.375.  All three on the SAME
   graphs, built with the Step-3 sub-pixel anchors ON (`FE_SUBPIXEL_PEAK_POS`,
   `FE_SUBPIXEL_GABOR_ANCHOR`), which is what Step 3 adopted.

3. THE CONTRAST is C - A (the placement mechanism's own effect), on near-tie
   correspondence accuracy, photo-clustered bootstrap (4000 draws, 2.5/97.5).

4. MDE per stratum, computed from THAT stratum's realised bank
   (2.8 * 0.5 * sqrt(2) / sqrt(n_bank)); no borrowing across strata.

5. VERDICT RULE per stratum, applied mechanically:
   DETECTED          lo > 0 and mean >= that stratum's MDE
   POSSIBLY INFLATED lo > 0 and mean <  MDE
   NULL SUPPORTED    lo <= 0 and hi <  MDE   (the stratum had the power to see an
                                              effect of its own MDE and did not)
   UNDERPOWERED      lo <= 0 and hi >= MDE   (say so, do not call it a null)

   A SECONDARY stratum is declared BROKEN unless its own contrast is DETECTED; the
   verdict is therefore allowed to confirm the Step-3 floor rather than to be
   rescued by it.

6. ONE LOOK.  A completed result refuses a second run (`--start` refuses if
   `m13_read_real.json` exists); the progress checkpoint is append-only and a
   resumed run is reported as such.

    python Topology/placement_m13_read.py --start --workers 2
    python Topology/placement_m13_read.py --report
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
RESULT = os.path.join(OUT, "m13_read_real.json")
PROGRESS = os.path.join(OUT, "m13_read_real_progress.jsonl")
PREREG = os.path.join(OUT, "m13_prereg_read.json")
PARAMS = os.path.join(ROOT, "matching", "placement_score_params.json")

TRANSFORMS: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    ("PRIMARY", "trans8", {"shift": (8.0, 0.0)}),
    ("SECONDARY", "rot05", {"rot_deg": 5.0}),
    ("SECONDARY", "rot15", {"rot_deg": 15.0}),
    ("SECONDARY", "scale115", {"scale": 1.15}),
)
ARMS = ("A", "B", "C")
MD_ALPHA = 2.8


def _registration() -> Dict[str, Any]:
    from Topology import placement_m12_realphotos as RP
    names = sorted(RP.images("validation"))
    return {
        "mission": "M13 step 5",
        "signed": "before the read; this file is written at --start and refuses to be overwritten",
        "photos": {"set": "validation", "names": names, "n": len(names)},
        "front_end": {"FE_SUBPIXEL_PEAK_POS": True, "FE_SUBPIXEL_GABOR_ANCHOR": True,
                      "why": "Mission 13 step 3 (sub-pixel anchors; dark, defaults off)"},
        "strata": [{"name": lab, "role": role, "spec": spec} for role, lab, spec in TRANSFORMS],
        "arms": {"A": "placement OFF (shipped aligner)",
                 "B": "placement ON, evidence OFF",
                 "C": "placement ON, evidence ON, w = shipped 0.375"},
        "contrast": "C - A on near-tie correspondence accuracy",
        "cluster": "photo",
        "bootstrap": {"draws": 4000, "ci": [2.5, 97.5], "seed": 20261119},
        "mde": {"formula": "2.8 * 0.5 * sqrt(2) / sqrt(n_bank)", "per": "stratum"},
        "verdict_rule": {
            "DETECTED": "lo > 0 and mean >= stratum MDE",
            "POSSIBLY_INFLATED": "lo > 0 and mean < stratum MDE",
            "NULL_SUPPORTED": "lo <= 0 and hi < stratum MDE",
            "UNDERPOWERED": "lo <= 0 and hi >= stratum MDE"},
        "pooling": "FORBIDDEN.  Each stratum is read, verdicted and reported alone.",
        "broken_declaration": ("rot05, rot15, scale115 are declared BROKEN in advance: "
                               "Step 2 measured detection 0.09-0.17 there and Step 3 "
                               "only doubled a 0.04 floor.  A SECONDARY stratum leaves "
                               "BROKEN only if its own contrast is DETECTED."),
        "one_look": "a completed m13_read_real.json refuses a second --start",
    }


def _pair_job(job: Tuple[str, str, str, float]) -> Dict[str, Any]:
    name, label, spec_json, w = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology import placement_census as CEN
    from Topology import placement_m12_read_real as R
    from Topology.placement_gate import appearance_ambiguous

    spec = json.loads(spec_json)
    P.FE_SUBPIXEL_PEAK_POS = True
    P.FE_SUBPIXEL_GABOR_ANCHOR = True
    from Topology import placement_m12_realphotos as RP
    base = RP.images("validation")[name]
    kq, kr = "m13read|%s|%s" % (name, label), "m13read|%s|orig" % name
    q_img, r_img = R._transform(base, spec), base
    q = get_or_build(q_img, kq)
    r = get_or_build(r_img, kr)
    truth = R._truth(q, r, spec, (r_img.shape[0], r_img.shape[1]))
    o = sorted(truth)
    rows = CEN.similarity_rows(q, r, o, sorted(int(n) for n in r.subgraph.nodes))
    amb = appearance_ambiguous(q.sigs, r.sigs, rows)

    out: Dict[str, Any] = {"photo": name, "transform": label, "w": float(w),
                           "truth_n": len(truth),
                           "amb_n": int(sum(1 for u in o if amb.get(int(u), False)))}
    for arm in ARMS:
        P.PLACEMENT_NORMALISE = True
        P.PLACEMENT_E_NORMALISED = True
        P.PLACEMENT_MATCHER_ENABLED = (arm != "A")
        P.PLACEMENT_SCORE_EVIDENCE = (arm == "C")
        P.PLACEMENT_SCORE_WEIGHT = (float(w) if arm == "C" else 0.0)
        res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                        placement_images=((q_img, r_img) if arm != "A" else None))
        an = ah = aj = uh = un = 0
        for u in o:
            v = truth[u]
            hit = int(res.mapping.get(int(u), -1) == v)
            if amb.get(int(u), False):
                ah += hit
                an += 1
                aj += int(int(u) in res.mapping)
            else:
                uh += hit
                un += 1
        out[arm] = {"amb_hit": ah, "amb_n": an, "amb_judged": aj,
                    "un_hit": uh, "un_n": un,
                    "E": float(getattr(res, "placement_evidence", float("nan"))),
                    "judged": int(getattr(res, "placement_judged", 0))}
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    return out


def _boot(diffs: List[float], clusters: List[str], seed: int = 20261119) -> Dict[str, Any]:
    by: Dict[str, List[float]] = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(d)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(4000):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        draws.append(float(np.mean([v for k in pick for v in by[k]])))
    draws.sort()
    return {"mean": float(np.mean(diffs)), "lo": draws[int(0.025 * len(draws))],
            "hi": draws[int(0.975 * len(draws)) - 1], "n": len(diffs),
            "clusters": len(keys)}


def _verdict(paired: Dict[str, Any], mde: float) -> Dict[str, Any]:
    if paired["lo"] > 0:
        v = "DETECTED" if paired["mean"] >= mde else "POSSIBLY_INFLATED"
    else:
        v = "NULL_SUPPORTED" if paired["hi"] < mde else "UNDERPOWERED"
    return {"verdict": v, "lo": paired["lo"], "hi": paired["hi"],
            "mean": paired["mean"], "mde": mde}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()

    if not os.path.exists(PREREG):
        with open(PREREG, "w") as fh:
            json.dump(_registration(), fh, indent=1)
        print("signed %s" % PREREG)
    with open(PREREG) as fh:
        reg = json.load(fh)
    with open(PARAMS) as fh:
        w = float(json.load(fh)["w"])

    if args.start:
        if os.path.exists(RESULT):
            print("REFUSING: a completed read already exists at %s (one look)" % RESULT)
            return 3
        jobs = [(p, lab, json.dumps(spec), w)
                for p in reg["photos"]["names"] for _role, lab, spec in TRANSFORMS]
        done: List[Dict[str, Any]] = []
        if os.path.exists(PROGRESS):
            with open(PROGRESS) as fh:
                for line in fh:
                    try:
                        done.append(json.loads(line))
                    except Exception:
                        continue
        seen = {(d["photo"], d["transform"]) for d in done}
        todo = [j for j in jobs if (j[0], j[1]) not in seen]
        print("%d cells, %d to go (resumed=%s)" % (len(jobs), len(todo), bool(done)),
              flush=True)
        t0 = time.time()
        with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_pair_job, todo):
                done.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print("  %-12s %-9s amb %3d  A %3d  B %3d  C %3d"
                      % (r["photo"], r["transform"], r["amb_n"],
                         r["A"]["amb_hit"], r["B"]["amb_hit"], r["C"]["amb_hit"]),
                      flush=True)
        print("cells done in %.0fs" % (time.time() - t0), flush=True)

    rows: List[Dict[str, Any]] = []
    with open(PROGRESS) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    res: Dict[str, Any] = {"cells": len(rows), "w": w, "registered": reg["verdict_rule"],
                           "strata": {}, "arms": ARMS, "pooled": "NOT COMPUTED BY DESIGN"}
    for _role, lab, _spec in TRANSFORMS:
        sub = [r for r in rows if r["transform"] == lab]
        st: Dict[str, Any] = {"role": _role, "cells": len(sub),
                              "photos": len({r["photo"] for r in sub})}
        for bank, hit, tot in (("near_tie", "amb_hit", "amb_n"),
                               ("unambiguous", "un_hit", "un_n")):
            bank_n = sum(r["A"][tot] for r in sub)
            st[bank] = {"bank_nodes": bank_n}
            for arm in ARMS:
                h = sum(r[arm][hit] for r in sub)
                st[bank]["acc_" + arm] = (h / bank_n) if bank_n else float("nan")
            diffs = [r["C"][hit] / r["C"][tot] - r["A"][hit] / r["A"][tot]
                     for r in sub if r["A"][tot]]
            st[bank]["C_minus_A"] = _boot(diffs, [r["photo"] for r in sub if r["A"][tot]])
        md = MD_ALPHA * 0.5 * (2.0 ** 0.5) / np.sqrt(max(1, st["near_tie"]["bank_nodes"]))
        st["mde"] = float(md)
        st["primary_verdict"] = _verdict(st["near_tie"]["C_minus_A"], float(md))
        if _role == "SECONDARY" and st["primary_verdict"]["verdict"] != "DETECTED":
            st["primary_verdict"]["regime"] = "BROKEN (as declared, not repaired)"
        elif _role == "SECONDARY":
            st["primary_verdict"]["regime"] = "REPAIRED by this read"
        else:
            st["primary_verdict"]["regime"] = "PRIMARY"
        res["strata"][lab] = st
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: {"role": v["role"], "bank": v["near_tie"]["bank_nodes"],
                          "acc_A": v["near_tie"]["acc_A"], "acc_B": v["near_tie"]["acc_B"],
                          "acc_C": v["near_tie"]["acc_C"],
                          "C-A": v["primary_verdict"], "mde": v["mde"]}
                      for k, v in res["strata"].items()}, indent=1))
    print("wrote %s" % RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
