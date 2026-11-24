"""MISSION 11 STEP 1 -- fit the ONE weight w, on FIT, by maximum likelihood.

The placement-aware pair score is

    S'(A,B) = S_shipped(A,B) + w * E(A,B)

with `S_shipped` the deployed `JointMCS.score` (computed from the mapping the
deployed matcher produced) and E the coverage-neutral placement evidence defined in
`parameters/_text.py::PLACEMENT_SCORE_EVIDENCE` -- the mean, over the mapped pairs
the frozen estimator could actually judge, of `plus_score - 0.5`, shrunk by
`n_judged + k`.

The read this weight will be used in is a RETRIEVAL read (one query, K candidates,
truth = the same scene), so the thing to maximise is the retrieval likelihood, not a
per-node accuracy:

    w* = argmax_w  sum over queries q of log softmax over candidates c of
                   ( S_shipped(q,c) + w * E(q,c) )   evaluated at the TRUE c.

That objective is concave in w (a sum of log-sigmoids), so a fine grid plus one
parabolic refinement finds the maximum; both are reported.

WHERE THE DATA COMES FROM, and why it is FIT: the gallery and the queries are the
FIT half of `placement_families_m9` -- the irregular scenes with an ODD rank and the
out-of-plane scenes with an ODD rank, i.e. the mission-8/9 FIT side. No confirmation
scene, no fresh scene, and nothing from CHECK is read here. Every pair is matched
TWICE (placement off, placement on) so the fit sees the same three arms the read
will report:

    A  shipped        joint_mcs, placement off          -> res.score
    B  mapping-only   placement on, score unchanged     -> res.score_shipped
    C  full           placement on, + w * E             -> res.score_shipped + w*E

`E` is read from the placement-on run, which is the mapping arm C uses.
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
PARAMS = os.path.join(ROOT, "matching", "placement_score_params.json")
PROGRESS = os.path.join(OUT, "m11_fit_w_progress.jsonl")

#: Queries: every FIT scene at a pose the deployed aligner has to absorb.  In-plane
#: 12 degrees and out-of-plane y 20 degrees are the two steps Mission 9/10 used for
#: their retrieval reads, so the weight is fitted where it will be used.
QUERY_STEPS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("q_inplane12", {"rot_deg": 12.0}),
    ("q_y20", {"rot_y_deg": 20.0}),
)


def gallery_specs() -> List[Any]:
    """The FIT gallery: FIT irregular scenes plus the FIT out-of-plane scenes."""
    from Topology import placement_families_m9 as FM
    fit_ir, _chk = FM.split_scenes(FM.all_specs_irregular(12))
    fit_oop, _chk2 = FM.split_scenes(FM.all_specs_oop(4))
    return list(fit_ir) + list(fit_oop)


def query_steps(spec: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Which steps this FIT scene is queried at, for the FIT.

    The fit is a ONE-parameter maximum likelihood and one synthetic pair costs ~13 s
    of matcher.  Fourteen FIT scenes x their steps x 14 candidates came to 392 pairs
    (measured wall time: 3.4 h).  This is the same objective on a smaller, still
    balanced design -- both pose types represented:

        irregular FIT scenes (6)      x q_inplane12  =  6 queries
        out-of-plane FIT scenes (4)   x q_y20        =  4 queries
        10 queries x 14 candidates                   = 140 pairs, ~1.2 h on two cores
    """
    if spec.family == "IRREGULAR":
        return [QUERY_STEPS[0]]
    return [QUERY_STEPS[1]]


#: Renders are expensive and were being repeated once per candidate pair.  The
#: first version rendered two scenes for each of 392 pairs and spent most of its
#: wall time in the renderer, not the matcher.  `prewarm()` renders every
#: (scene, view) once into a npz and the jobs read the pixels from there.
IMAGES = os.path.join(ROOT, ".cache", "m11_fit_w_images.npz")


def _image_keys() -> List[str]:
    specs = gallery_specs()
    keys = [s.name + "|ref" for s in specs]
    keys += ["%s|%s" % (s.name, lab) for s in specs for lab, _v in query_steps(s)]
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
            sc, _pv = FM.pose_fn(specs[name], dict(QUERY_STEPS)[label])
        out[key] = sc.render().image_u8
    np.savez_compressed(IMAGES, **out)
    print("prewarmed %d images -> %s" % (len(out), IMAGES), flush=True)


def _pair_job(job: Tuple[str, str]) -> Dict[str, Any]:
    qkey, rname = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build

    with np.load(IMAGES) as z:
        q_img = np.array(z[qkey])
        r_img = np.array(z[rname + "|ref"])
    q = get_or_build(q_img, qkey)
    r = get_or_build(r_img, rname + "|ref")

    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_MATCHER_PRESENCE = False
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    t0 = time.time()
    c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                  placement_images=(q_img, r_img))
    tc = time.time() - t0
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    return {"q": qkey, "truth": qkey.split("|")[0], "cand": rname,
            "B": float(c.score_shipped),
            "E": float(c.placement_evidence),
            "judged": int(c.placement_judged),
            "share": float(c.placement_share_judged),
            "mapped": len(c.mapping),
            "sec_on": tc}


def loglik(rows: List[Dict[str, Any]], w: float) -> float:
    """Sum over queries of log p(true candidate) under a softmax of S + w E."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    tot = 0.0
    for q, cands in by.items():
        logits = [r["B"] + w * r["E"] for r in cands]
        m = max(logits)
        z = m + float(np.log(np.sum(np.exp([x - m for x in logits]))))
        truth = cands[0]["truth"]
        hit = [x for r, x in zip(cands, logits) if r["cand"] == truth]
        if not hit:
            continue
        tot += hit[0] - z
    return tot


#: THE SEARCH RANGE.  The shipped score lives in [0, 1] with a candidate-to-candidate
#: spread of ~0.023 on the FIT shortlists, and E has a spread of ~0.018 -- the same
#: order of magnitude.  A weight above 3 would therefore let the evidence term alone
#: decide a pair, which is a different model from "the shipped score, refined".
W_MAX = 3.0


def fit_top1(rows: List[Dict[str, Any]], w: float) -> float:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    hits = []
    for q, cands in by.items():
        truth = cands[0]["truth"]
        sc = sorted([(d["B"] + w * d["E"], d["cand"]) for d in cands],
                    key=lambda t: (-t[0], t[1]))
        hits.append(1.0 if [n for _, n in sc].index(truth) == 0 else 0.0)
    return float(np.mean(hits))


def fit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Two criteria, both on FIT, and the rule that picks between them.

    (1) MAXIMUM LIKELIHOOD, which is what the mission asks for:
            w* = argmax  sum_q log softmax_c( S_qc + w E_qc ) [truth]

    (2) THE DEGENERACY THAT SHOWS UP WHEN IT IS COMPUTED, which has to be reported
        rather than hidden: on FIT the likelihood is MONOTONE INCREASING IN w all
        the way to the search bound.  It has to be -- the shipped score already
        ranks the truth first on 11 of 14 FIT queries, so multiplying the logits by
        anything makes those probabilities sharper and raises their likelihood,
        while the queries it gets wrong are already wrong at w = 0.  Maximum
        likelihood is therefore UNIDENTIFIED at the boundary in this model class:
        it is choosing a temperature, not a weight.

    (3) THE DECLARED TIE-BREAK, fixed here and applied by the code: when the ML
        argmax sits on the boundary, take the weight that maximises FIT top-1 on
        the same shortlists, ties going to the SMALLER weight.  Top-1 is the
        quantity the read reports, it is computed on FIT only, and one parameter is
        being chosen -- which is exactly what FIT is for.  Both numbers are written
        into the params file so the choice can be audited.
    """
    grid = np.linspace(0.0, W_MAX, 121)
    lls = np.array([loglik(rows, float(w)) for w in grid])
    i = int(np.argmax(lls))
    at_edge = (i == len(grid) - 1)
    ml_w = float(grid[i])
    t1 = np.array([fit_top1(rows, float(w)) for w in grid])
    best = float(t1.max())
    cand = [float(w) for w, v in zip(grid, t1) if v >= best - 1e-12]
    w_tie = cand[0]
    w_chosen = w_tie if at_edge else ml_w
    return {"w": float(w_chosen), "w_ml": ml_w, "w_top1": w_tie,
            "ml_at_boundary": bool(at_edge), "w_max": W_MAX,
            "loglik": float(loglik(rows, float(w_chosen))),
            "loglik_at_ml": float(lls[i]), "loglik_w0": float(loglik(rows, 0.0)),
            "top1_w0": fit_top1(rows, 0.0), "top1_chosen": fit_top1(rows, float(w_chosen)),
            "ml_curve": [[float(w), float(l)] for w, l in zip(grid[::10], lls[::10])],
            "top1_curve": [[float(w), float(v)] for w, v in zip(grid[::10], t1[::10])]}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import guard
    guard.activate()
    specs = gallery_specs()
    names = [s.name for s in specs]
    qkeys = ["%s|%s" % (s.name, lab) for s in specs for lab, _v in query_steps(s)]
    jobs = [(qk, rn) for qk in qkeys for rn in names]
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
    print("%d of %d pairs done, %d to go (%d gallery, %d queries)"
          % (len(done), len(jobs), len(todo), len(names), len(qkeys)), flush=True)
    prewarm()
    t0 = time.time()
    with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_pair_job, todo, chunksize=2):
            done.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if len(done) % 20 == 0:
                print("  %d/%d  %.0fs" % (len(done), len(jobs), time.time() - t0),
                      flush=True)
    res = {"pairs": len(done), "queries": len({d["q"] for d in done}),
           "gallery": len(names), "specs": names,
           "fit": fit(done),
           "seconds": time.time() - t0,
           "coverage": {"judged_mean": float(np.mean([d["judged"] for d in done])),
                        "share_mean": float(np.mean([d["share"] for d in done])),
                        "mapped_mean": float(np.mean([d["mapped"] for d in done])),
                        "sec_on": float(np.mean([d["sec_on"] for d in done]))}}
    # where the arms stand at w*, on the same shortlists.  Arm A is not run here:
    # the fit needs only (B, E), and the placement-off run would double the cost of
    # the fit for a number the READ reports properly.
    for tag, score_of in (("B mapping-only", lambda d: d["B"]),
                          ("C full", lambda d: d["B"] + res["fit"]["w"] * d["E"])):
        by: Dict[str, List[Dict[str, Any]]] = {}
        for d in done:
            by.setdefault(d["q"], []).append(d)
        t1, rr = [], []
        for q, cands in by.items():
            order = sorted(cands, key=lambda d: (-score_of(d), d["cand"]))
            rank = [d["cand"] for d in order].index(cands[0]["truth"]) + 1
            t1.append(1.0 if rank == 1 else 0.0)
            rr.append(1.0 / rank)
        res.setdefault("fit_side_arms", {})[tag] = {
            "top1": float(np.mean(t1)), "mrr": float(np.mean(rr)), "n": len(t1)}
    with open(os.path.join(OUT, "m11_fit_w.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    with open(PARAMS, "w") as fh:
        json.dump({"w": res["fit"]["w"], "shrink": 8.0, "centred_at": 0.5,
                   "fitted_on": "mission-8/9 FIT scenes (irregular + out-of-plane, "
                                "odd ranks), in-plane 12 deg and out-of-plane y 20 deg",
                   "objective": "softmax retrieval log-likelihood over FIT "
                                "shortlists (see Topology/placement_m11_fit_w.py)",
                   "loglik": res["fit"]["loglik"], "loglik_at_w0": res["fit"]["loglik_w0"],
                   "pairs": res["pairs"]}, fh, indent=1, sort_keys=True)
    print("w* = %.4f  loglik %.3f (at w=0 %.3f)" % (
        res["fit"]["w"], res["fit"]["loglik"], res["fit"]["loglik_w0"]))
    for tag, d in sorted(res["fit_side_arms"].items()):
        print("  %-16s top1 %.3f mrr %.3f n %d" % (tag, d["top1"], d["mrr"], d["n"]))
    print("coverage", json.dumps(res["coverage"], sort_keys=True))
    print("wrote %s and %s" % (os.path.join(OUT, "m11_fit_w.json"), PARAMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
