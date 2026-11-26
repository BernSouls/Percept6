"""MISSION 12 STEP 4 -- refit the shipped temperature and the placement weight, on a
hard synthetic FIT set that was NOT selected by the shipped score's errors.

The score under test is the mission's own:

    S'(A,B) = S_shipped(A,B) / T + w * E(A,B)

with `S_shipped` the deployed `JointMCS.score` from the placement-ON mapping
(arm B) and E the placement evidence on the STEP-2 scale
(``PLACEMENT_E_NORMALISED``: the mean over the slots a pair judged, refused below
the minimum).  Two parameters are fitted jointly by maximum likelihood on FIT:

    (T, w) = argmax  sum over queries q of log softmax over candidates c of
                     ( S_shipped(q,c)/T + w * E(q,c) )   at the TRUE c.

THE QUERY SET IS HARD BY CONSTRUCTION, and by construction only:

* the candidates of a query are the TRUE reference, TWO rivals from the SAME
  family (the twins the M9 families were built with: same structure, same
  material, different instance) and ONE rival from another family, so a query is
  never a stroll through the gallery;
* the views are stratified: easy in-plane steps (5-15 deg), hard in-plane steps
  (25-35 deg), out-of-plane y and x, and scale changes -- all listed below, none
  chosen by looking at which ones the shipped score got wrong;
* every FIT scene contributes the same views, so no scene is over-weighted.

A QUARTER OF THE QUERIES IS HELD OUT before the fit and is never optimised
against; the top-1 at w = 0, at the deployed w = 0.375 and at the fitted (T, w) are
reported on the held-out quarter.  The fitted pair goes to
``matching/placement_score_params_m12.json`` and is NOT installed as a default.

    python Topology/placement_m12_fit_w.py --start     # prewarm + run (resumable)
    python Topology/placement_m12_fit_w.py --report    # fit from the checkpoint
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
PROGRESS = os.path.join(OUT, "m12_fit_w_progress.jsonl")
RESULT = os.path.join(OUT, "m12_fit_w.json")
PARAMS = os.path.join(ROOT, "matching", "placement_score_params_m12.json")
IMAGES = os.path.join(ROOT, ".cache", "m12_fit_w_images.npz")

#: The stratified view grid.  NOTHING here was chosen by looking at an error.
VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("i05", {"rot_deg": 5.0}), ("i10", {"rot_deg": 10.0}),
    ("i15", {"rot_deg": 15.0}), ("i25", {"rot_deg": 25.0}),
    ("i30", {"rot_deg": 30.0}), ("i35", {"rot_deg": 35.0}),
    ("y10", {"rot_y_deg": 10.0}), ("y20", {"rot_y_deg": 20.0}),
    ("y30", {"rot_y_deg": 30.0}), ("x10", {"rot_x_deg": 10.0}),
    ("x20", {"rot_x_deg": 20.0}), ("s09", {"scale": 0.9}),
    ("s11", {"scale": 1.1}), ("sh8", {"shift": (8.0, 0.0)}),
    ("sh16", {"shift": (0.0, 16.0)}),
    ("i15y15", {"rot_deg": 15.0, "rot_y_deg": 15.0}),
    ("i25x15", {"rot_deg": 25.0, "rot_x_deg": 15.0}),
    ("y20s11", {"rot_y_deg": 20.0, "scale": 1.1}),
)
#: One candidate per role: true, two same-family twins, one other-family rival.
N_TWINS, N_FOREIGN = 2, 1


def gallery_specs() -> List[Any]:
    """The FIT gallery: the FIT halves of the M9 irregular and out-of-plane sets."""
    from Topology import placement_families_m9 as FM
    fit_ir, _ = FM.split_scenes(FM.all_specs_irregular(12))
    fit_oop, _ = FM.split_scenes(FM.all_specs_oop(4))
    return list(fit_ir) + list(fit_oop)


def queries() -> List[Tuple[str, Any, str, Dict[str, Any]]]:
    """(query key, spec, view label, view) for every FIT scene x the view grid."""
    out = []
    for spec in gallery_specs():
        for lab, view in VIEWS:
            out.append(("%s|%s" % (spec.name, lab), spec, lab, view))
    return out


def shortlists() -> Dict[str, List[str]]:
    """query key -> candidate reference names: true, twins, one foreign rival.

    Everything here is BY CONSTRUCTION: the twins are the other scenes of the same
    family (the family definitions pair them), the foreign rival is the next
    family's first scene.  Sorted by name so the choice never depends on order.
    """
    specs = gallery_specs()
    by_fam: Dict[str, List[str]] = {}
    for s in specs:
        by_fam.setdefault(s.family, []).append(s.name)
    fams = sorted(by_fam)
    out: Dict[str, List[str]] = {}
    for spec in specs:
        same = sorted(n for n in by_fam[spec.family] if n != spec.name)
        twins = same[:N_TWINS]
        foreign: List[str] = []
        for f in fams:
            if f == spec.family:
                continue
            foreign.extend(sorted(by_fam[f]))
        foreign = foreign[:N_FOREIGN]
        cands = [spec.name] + twins + foreign
        for qk, s2, lab, view in queries():
            if s2.name == spec.name:
                out[qk] = cands
    return out


def _image_keys() -> List[str]:
    keys = []
    for spec in gallery_specs():
        keys.append(spec.name + "|ref")
        for lab, _view in VIEWS:
            keys.append("%s|%s" % (spec.name, lab))
    return keys


def prewarm() -> None:
    if os.path.exists(IMAGES):
        return
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    specs = {s.name: s for s in gallery_specs()}
    out: Dict[str, Any] = {}
    for key in _image_keys():
        name, label = key.split("|")
        if label == "ref":
            sc = specs[name].scene
        else:
            view = dict(VIEWS)[label]
            sc, _pv = FM.pose_fn(specs[name], view)
        out[key] = sc.render().image_u8
    os.makedirs(os.path.dirname(IMAGES), exist_ok=True)
    np.savez_compressed(IMAGES, **out)
    print("prewarmed %d images -> %s" % (len(out), IMAGES), flush=True)


#: The prewarmed pixels, decompressed ONCE per worker process.  The first version
#: called `np.load` inside every job and spent most of the pair's wall time
#: decompressing 266 images it did not need.
_IMGS: Dict[str, Any] = {}


def _imgs() -> Any:
    if not _IMGS:
        with np.load(IMAGES) as z:
            for k in z.files:
                _IMGS[k] = z[k]
    return _IMGS


def _pair_job(job: Tuple[str, str, bool]) -> Dict[str, Any]:
    """One (query, candidate) pair: arm A (held-out only), arm B, and E.

    Arm A is the placement-OFF score.  It is needed only where the shipped score's
    own top-1 is reported, i.e. on the held-out quarter; running it for every
    training pair would double the cost of a fit that never reads it.
    """
    qkey, rname, held = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build

    imgs = _imgs()
    q_img = imgs[qkey]
    r_img = imgs[rname + "|ref"]
    q = get_or_build(q_img, "m12fit|" + qkey)
    r = get_or_build(r_img, "m12fit|" + rname)
    P.PLACEMENT_E_NORMALISED = True       # the STEP-2 reading of E
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    t0 = time.time()
    if held:
        a = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs)
        a_score = float(a.score)
    else:
        a_score = float("nan")
    ta = time.time() - t0
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0     # B: the mapping moves, the score does not
    t0 = time.time()
    b = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                  placement_images=(q_img, r_img))
    tb = time.time() - t0
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    return {"q": qkey, "cand": rname,
            "A": a_score, "B": float(b.score_shipped),
            "E": float(b.placement_evidence), "judged": int(b.placement_judged),
            "share": float(b.placement_share_judged),
            "mapped": int(len(b.mapping)), "sec_a": ta, "sec_b": tb}


def loglik(rows: List[Dict[str, Any]], T: float, w: float) -> float:
    """Sum over queries of log p(true) under a softmax over the shortlist."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    tot = 0.0
    for q, cands in by.items():
        logits = [r["B"] / T + w * r["E"] for r in cands]
        m = max(logits)
        z = m + float(np.log(np.sum(np.exp([x - m for x in logits]))))
        ti = [i for i, r in enumerate(cands) if r["cand"] == cands[0]["_truth"]]
        if not ti:
            continue
        tot += logits[ti[0]] - z
    return tot


def top1(rows: List[Dict[str, Any]], T: float, w: float) -> Dict[str, float]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    hit = tot = 0
    rr = 0.0
    for q, cands in by.items():
        truth = cands[0]["_truth"]
        order = sorted(cands, key=lambda r: (-(r["B"] / T + w * r["E"]), r["cand"]))
        rank = [r["cand"] for r in order].index(truth) + 1
        hit += 1 if rank == 1 else 0
        tot += 1
        rr += 1.0 / rank
    return {"top1": float(hit / tot) if tot else float("nan"),
            "mrr": float(rr / tot) if tot else float("nan"), "n": tot}


def _grid() -> List[Tuple[float, float]]:
    out = []
    for T in (0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6):
        for w in (0.0, 0.1, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0, 1.5):
            out.append((T, w))
    return out


def surface(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The (T, w) log-likelihood on a coarse lattice, recorded with the result.

    MEASURED, and the reason the two-parameter fit cannot be used: the surface
    increases monotonically as T falls and as w rises (see the report's table).
    T is a pure SHARPENING knob on `S_shipped/T`, so as T -> 0 the objective becomes
    "the shipped ranking, infinitely sharp", the softmax log-likelihood grows
    without bound, and the argmax runs to the corner of any grid it is given.  The
    fit reported `w = 1.55, T = 0.015` -- both OUTSIDE the declared bounds -- with a
    held-out ranking IDENTICAL to w = 0 and a train top-1 BELOW the shipped
    w = 0.375.  That is a degenerate optimum, not a tuning result.
    """
    Ts = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0)
    Ws = (0.0, 0.2, 0.375, 0.6, 1.0, 1.5, 2.0, 3.0)
    return {"T": list(Ts), "w": list(Ws),
            "loglik": [[float(loglik(rows, T, w)) for w in Ws] for T in Ts]}


def fit(rows: List[Dict[str, Any]], model: str = "2d") -> Dict[str, Any]:
    """Maximum likelihood on FIT.

    `model="2d"` is the originally registered fit and is kept so the degeneracy is
    reproducible; `model="1d"` fixes the temperature at the SHIPPED 1.0 (where the
    deployed score already lives) and fits `w` alone, which is the identifiable
    direction: `w` multiplies the evidence, whose scale is not tied to the shipped
    score's, and the likelihood is curved in it over a bounded range.
    """
    if model == "1d":
        best = (-1e18, 1.0, 0.0)
        for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375, 0.45, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5):
            v = loglik(rows, 1.0, w)
            if v > best[0]:
                best = (v, 1.0, w)
        _, T0, w0 = best
        for dw in (-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1):
            w = max(0.0, w0 + dw)
            v = loglik(rows, 1.0, w)
            if v > best[0] + 1e-12:
                best = (v, 1.0, w)
        return {"loglik": best[0], "T": best[1], "w": best[2], "model": "1d"}
    best = (-1e18, 1.0, 0.0)
    for (T, w) in _grid():
        v = loglik(rows, T, w)
        if v > best[0]:
            best = (v, T, w)
    # one coordinate refinement pass
    _, T0, w0 = best
    for dT in (-0.01, -0.005, 0.005, 0.01):
        for dw in (-0.05, -0.02, 0.02, 0.05):
            T, w = max(1e-3, T0 + dT), max(0.0, w0 + dw)
            v = loglik(rows, T, w)
            if v > best[0] + 1e-12:
                best = (v, T, w)
    return {"loglik": best[0], "T": best[1], "w": best[2], "model": "2d"}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", choices=("1d", "2d"), default="1d",
                    help="1d fixes T at the shipped 1.0 and fits w (the identifiable "
                         "direction); 2d reproduces the degenerate joint fit")
    args = ap.parse_args()
    import guard
    guard.activate()

    qs = queries()
    shorts = shortlists()
    truth_of = {qk: shorts[qk][0] for qk in shorts}
    # hold out every fourth query, by position in the deterministic list
    held = {qk for i, (qk, *_r) in enumerate(qs) if i % 4 == 3}
    jobs = [(qk, cand, qk in held) for (qk, *_r) in qs for cand in shorts[qk]]

    if args.start:
        done: List[Dict[str, Any]] = []
        if os.path.exists(PROGRESS):
            with open(PROGRESS) as fh:
                for line in fh:
                    try:
                        done.append(json.loads(line))
                    except Exception:
                        continue
        seen = {(d["q"], d["cand"]) for d in done}
        todo = [j for j in jobs if (j[0], j[1]) not in seen]
        print("%d queries (%d held out), %d pairs, %d to go"
              % (len(qs), len(held), len(jobs), len(todo)), flush=True)
        prewarm()
        t0 = time.time()
        # [M12 / step 4] The pool is RECYCLED and RESTARTED instead of dying with a
        # worker.  Two OOM kills were observed here: a long-lived worker accumulates
        # the module-level caches (the relational cache, the side cache, the graph
        # objects) and the kernel eventually kills it, and `ex.map` then raises
        # BrokenProcessPool and abandons the rest of the sweep.  `max_tasks_per_child`
        # bounds the growth, and the loop below re-reads the checkpoint and resumes,
        # so a kill costs one job instead of the run.
        from concurrent.futures.process import BrokenProcessPool
        while True:
            with open(PROGRESS, "a") as fh:
                try:
                    ex = ProcessPoolExecutor(max_workers=args.workers,
                                             max_tasks_per_child=25)
                except TypeError:                      # Python < 3.11
                    ex = ProcessPoolExecutor(max_workers=args.workers)
                try:
                    for r in ex.map(_pair_job, todo, chunksize=2):
                        done.append(r)
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        seen.add((r["q"], r["cand"]))
                        if len(done) % 40 == 0:
                            print("  %d/%d  %.0fs" % (len(done), len(jobs),
                                                      time.time() - t0), flush=True)
                except BrokenProcessPool as exc:
                    print("  pool died (%r): resuming from the checkpoint" % (exc,),
                          flush=True)
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)
            todo = [j for j in jobs if (j[0], j[1]) not in seen]
            if not todo:
                break
            print("  %d to go" % len(todo), flush=True)
        print("pairs done in %.0fs" % (time.time() - t0), flush=True)

    if not os.path.exists(PROGRESS):
        print("no checkpoint yet")
        return 1
    rows = []
    with open(PROGRESS) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            d["_truth"] = truth_of.get(d["q"])
            d["_held"] = d["q"] in held
            rows.append(d)
    train = [r for r in rows if not r["_held"]]
    test = [r for r in rows if r["_held"]]
    res = {"queries": len({r["q"] for r in rows}), "pairs": len(rows),
           "held_queries": len(held), "train_pairs": len(train),
           "test_pairs": len(test),
           "model": args.model,
           "surface": surface(train) if train else {},
           "fit": fit(train, args.model) if train else {}}
    f = res["fit"]
    res["held_out"] = {
        "w0_deployed_T1": top1(test, 1.0, 0.0),
        "w0375_T1": top1(test, 1.0, 0.375),
        "fit_T_w": top1(test, f.get("T", 1.0), f.get("w", 0.0)) if f else {},
    }
    res["train_top1"] = {
        "w0": top1(train, 1.0, 0.0), "w0375": top1(train, 1.0, 0.375),
        "fit": top1(train, f.get("T", 1.0), f.get("w", 0.0)) if f else {},
    }
    # [MISSION 12 / step 4] THE SANITY GATES, DECLARED BEFORE THE FIT IS READ.
    # A queued CONFIRMATION read must never be spent on a fit that is degenerate,
    # at the edge of its own search, or worse than the score it replaces.  The gates
    # are stated here, in the code that would have to pass them, and their verdict is
    # written into the result: the read refuses to run unless `gates.passed` is true.
    T_GRID = (0.02, 0.6)
    W_GRID = (0.0, 1.5)
    g = {}
    if not f:
        g["fit_present"] = False
    else:
        g["w_in_bounds"] = bool(W_GRID[0] < f["w"] < W_GRID[1])
        if args.model == "1d":
            # T is not fitted; the gate is the one the degeneracy left standing:
            # w must sit strictly inside its bounds, and the surface must show a
            # CURVE at the optimum (both grid ends worse by a declared margin),
            # so a corner cannot pass as an optimum.
            g["T_in_bounds"] = True
            g["not_at_grid_edge"] = bool(g["w_in_bounds"])
            ll_lo = loglik(train, 1.0, 0.05)
            ll_hi = loglik(train, 1.0, 1.25)
            g["curved"] = bool(f["loglik"] - max(ll_lo, ll_hi) > 0.5)
        else:
            g["T_in_bounds"] = bool(T_GRID[0] < f["T"] < T_GRID[1])
            g["not_at_grid_edge"] = bool(g["w_in_bounds"] and g["T_in_bounds"])
        ll0 = loglik(train, 1.0, 0.0)
        g["loglik_finite"] = bool(np.isfinite(f["loglik"]))
        g["loglik_beats_w0"] = bool(f["loglik"] > ll0)
        ho = res["held_out"]
        g["heldout_not_below_shipped"] = bool(
            ho["fit_T_w"].get("top1", 0.0) >= ho["w0_deployed_T1"].get("top1", 1.0) - 1e-12)
            # the shipped arm is w=0 at T=1; the fit must not lose to it
    g["passed"] = bool(all(bool(v) for k, v in g.items() if k != "passed"))
    g["declared"] = {
        "w_bounds": W_GRID, "T_bounds": T_GRID,
        "model": args.model,
        "rules": (["w strictly inside its bounds", "the profile likelihood is curved "
                   "(both ends worse by > 0.5 nats)"]
                  if args.model == "1d" else
                  ["w and T strictly inside their search bounds"]) + [
                  "the fit's log-likelihood is finite and beats w=0",
                  "held-out top-1 is not below the shipped score's own"],
        "on_failure": "the confirmation read is NOT run; this result says so",
    }
    res["gates"] = g
    res["coverage"] = {
        "judged_mean": float(np.mean([r["judged"] for r in rows])) if rows else 0.0,
        "share_mean": float(np.mean([r["share"] for r in rows])) if rows else 0.0,
        "mapped_mean": float(np.mean([r["mapped"] for r in rows])) if rows else 0.0,
        "sec_on": float(np.mean([r["sec_b"] for r in rows])) if rows else 0.0,
    }
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    if f and g.get("passed"):
        with open(PARAMS, "w") as fh:
            json.dump({"T": f["T"], "w": f["w"], "E": "PLACEMENT_E_NORMALISED reading",
                       "fitted_on": "M12 hard synthetic FIT (twins, OOP, stratified)",
                       "objective": "held-out-free softmax retrieval log-likelihood",
                       "held_out_queries": len(held), "loglik": f["loglik"]},
                      fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
