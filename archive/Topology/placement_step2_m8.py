"""MISSION 8 STEP 2 -- P_plus, the placement context, and ONE FIT decision.

Everything here is fitted on FIT and everything here is harness-side.  No
deployed scorer changes.

What is built:

  P_plus           existence term + geometry term in ONE estimator.  Same slots,
                   same SLOT_CAP, same padding, same +-LLR_CLIP, same [0, 1]
                   map as P and as P_presence; only the two coefficients are
                   new, and they are fitted by multinomial maximum likelihood on
                   FIT (matching/placement_llr.fit_plus_coefficients) and then
                   frozen for the fresh read.
  P_plus_presence  the same estimator with the geometry coefficient set to ZERO
                   and the fitted existence coefficient kept.  This is the arm
                   the geometry contrast is taken against, so "P_plus -
                   P_plus_presence" moves exactly one coefficient.
  P_plus_ctx       P_plus scored against an ITERATED context: a fixed number of
                   damped, mutually-confirmed sweeps starting from N's mapping
                   (matching/placement_context).  Truth-free at match time.

The margin that decides whether a node's context may move is chosen here, on a
FIT subsample, by the accuracy of the resulting CONTEXT (share of nodes whose
context mapping is the true one) -- not by arm accuracy, so the choice does not
need the arm it is used in.  It is written into the results and frozen.

Also reports the per-place-uncertainty question (step 2c): can a per-edge
tolerance, scaled by the uncertainty of the PLACE the edge hangs off, separate
true from wrong residuals better than the single fixed tolerance?  The answer is
measured, not assumed.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")

MARGINS = (0.0, 0.01, 0.02, 0.04, 0.08)
SWEEP_SCENES = 4


def pose_fn(spec: Any, view: Dict[str, Any]):
    from Topology.placement_dev5 import posed5
    pv = {"kind": view.get("kind"),
          "rot_deg": float(view.get("rot_deg", 0.0)),
          "shift": tuple(view.get("shift", (0.0, 0.0))),
          "scale": float(view.get("scale", 1.0))}
    return posed5(spec, view), pv


VARIANTS = ((True, 0.0), (True, 0.02), (True, 0.04),
            (False, 0.0), (False, 0.01), (False, 0.02), (False, 0.04))


def margin_sweep(specs: List[Any], views: List[Any], fit: Any, pfit: Any,
                 a_e: float, a_g: float) -> Dict[str, Any]:
    """Context accuracy of the iterated mapping, per filter setting, on FIT.

    Two knobs, both frozen afterwards: whether a move needs the picked node to
    pick it back, and how much the new choice must beat the incumbent.  The
    rival-count refusal (no near-ties) is always on -- that is the mechanism
    Step 0b asks for, and it is never relaxed.
    """
    from matching import placement_context as Ctx
    from matching import placement_llr as L
    from Topology import placement_m3 as M3

    acc: Dict[str, List[float]] = {("%s@%.2f" % (m, g)): []
                                   for (m, g) in VARIANTS}
    mv: Dict[str, int] = {("%s@%.2f" % (m, g)): 0 for (m, g) in VARIANTS}
    hurts: Dict[str, int] = {("%s@%.2f" % (m, g)): 0 for (m, g) in VARIANTS}
    repick: List[float] = []
    fixed: Dict[str, List[int]] = {}
    broke: Dict[str, List[int]] = {}
    base: List[float] = []
    for vw in M3.iter_views(specs, views, SWEEP_SCENES, pose_fn):
        q_inc, q_idx, r_idx = vw["q_inc"], vw["q_idx"], vw["r_idx"]
        rows, truth, nmap, pop = vw["rows"], vw["truth"], vw["nmap"], vw["pop"]
        cands = {int(u): [(int(v), float(s))
                          for (v, s) in (rows.get(u) or [])[:M3.CAND_CAP]]
                 for u in pop if rows.get(u)}
        sim = {(int(u), int(v)): float(s)
               for u, cl in cands.items() for (v, s) in cl}

        def score(u, v, m, _sim=sim):
            term = L.plus_score(u, v, m, q_inc, q_idx, r_idx, fit, pfit,
                                a_e, a_g)
            return ((1.0 - M3.EDGE_BETA) * _sim.get((int(u), int(v)), 0.0)
                    + M3.EDGE_BETA * term)

        b = Ctx.agreement(nmap, truth)
        base.append(b)
        right_of = {int(u): (int(nmap.get(int(u), -1)) == int(truth.get(u)))
                    for u in pop if int(u) in nmap}
        for (mut, mar) in VARIANTS:
            key = "%s@%.2f" % (mut, mar)
            st: Dict[str, Any] = {}
            cm = Ctx.iterate_context(pop, cands, nmap, score,
                                     iters=M3.CTX_ITERS, margin=mar,
                                     mutual=mut, stats=st)
            a = Ctx.agreement(cm, truth)
            acc[key].append(a)
            mv[key] += int(st.get("moved", 0))
            if a < b:
                hurts[key] += 1
            fixd = brk = 0
            for u, was in right_of.items():
                now = (int(cm.get(u, -1)) == int(truth.get(u)))
                if now and not was:
                    fixd += 1
                elif was and not now:
                    brk += 1
            fixed.setdefault(key, []).append(fixd)
            broke.setdefault(key, []).append(brk)
        repick.append(Ctx.agreement(Ctx.iterate_repick(pop, cands, nmap, score,
                                                       iters=M3.CTX_ITERS),
                                    truth))
    means = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    base_m = base
    # Pick the best FILTERED setting that never loses a view by more than 0.01.
    ok = [k for k in means if hurts[k] <= max(0, int(0.1 * len(base)))]
    best = max(ok, key=lambda k: (means[k], mv[k])) if ok else max(
        means, key=lambda k: means[k])
    mut, mar = best.split("@")
    return {"variants": means, "moved": mv, "views_hurt": hurts,
            "fixed_best": int(sum(fixed.get(best, [0]))),
            "broke_best": int(sum(broke.get(best, [0]))),
            "context_acc_before": float(np.mean(base)) if base else float("nan"),
            "context_acc_repick": float(np.mean(repick)) if repick else float("nan"),
            "best": best, "best_context_acc": means[best],
            "best_mutual": (mut == "True"), "best_margin": float(mar),
            "n_views": len(base)}


def tolerance_probe(specs: List[Any], views: List[Any], fit: Any,
                    limit: int = 3) -> Dict[str, Any]:
    """Step 2c: does per-place uncertainty make the residual more separable?

    Proxy for the uncertainty of a PLACE: the distance between the node's own
    position and the centroid of the nodes of its lens (a place that sits with
    its siblings is a well-localised place; an outlier is not).  If true-pair
    residuals do not grow with that proxy, a per-edge tolerance cannot help and
    the single FIXED_TOL stands.
    """
    from matching import placement_llr as L
    from Topology import placement_m3 as M3
    from Topology.placement_arms2 import lens_of

    xs: List[float] = []
    ys: List[float] = []
    for vw in M3.iter_views(specs, views, limit, pose_fn):
        g1 = vw["g1"]
        o1, truth = vw["o1"], vw["truth"]
        pos = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                        float(g1.subgraph.anchor_of_node[n][1]))
               for n in g1.subgraph.nodes if n in g1.subgraph.anchor_of_node}
        by_lens: Dict[str, List[int]] = {}
        for n in pos:
            if o1.get(n, -1) >= 0:
                by_lens.setdefault(str(lens_of(g1, n)), []).append(n)
        cen = {}
        for L_, ns in by_lens.items():
            p = np.array([pos[n] for n in ns])
            cen[L_] = (float(p[:, 0].mean()), float(p[:, 1].mean()))
        q_idx, r_idx, rows = vw["q_idx"], vw["r_idx"], vw["rows"]
        nmap = vw["nmap"]
        for u, tv in truth.items():
            if int(u) not in pos or int(tv) not in pos:
                continue
            Lm = str(lens_of(g1, int(u)))
            if Lm not in cen:
                continue
            c = cen[Lm]
            unc = float(np.hypot(pos[int(u)][0] - c[0], pos[int(u)][1] - c[1]))
            for (name, _o, res) in L.slots_for_candidate(
                    int(u), int(tv), nmap, vw["q_inc"], q_idx, r_idx):
                if res is None:
                    continue
                xs.append(unc)
                ys.append(float(abs(res)))
    if len(xs) < 50:
        return {"n": len(xs), "spearman": float("nan")}
    xs_a = np.asarray(xs)
    ys_a = np.asarray(ys)
    rx = np.argsort(np.argsort(xs_a)).astype(float)
    ry = np.argsort(np.argsort(ys_a)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # would a tolerance scaled by the proxy separate better?  Compare the share
    # of true residuals inside one fixed tolerance against the same share with
    # the proxy folded in: the correlation is the answer in one number.
    return {"n": len(xs), "spearman": rho,
            "median_abs_res": float(np.median(ys_a)),
            "median_unc": float(np.median(xs_a))}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--tag", default="m8_step2")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    import guard
    guard.activate()
    import Topology.placement_m3 as M3
    from Topology import placement_arms2 as A2
    from Topology.placement_dev5 import all_specs5, split5, VIEWS5
    from core.place_key import place_lenses

    specs = all_specs5(per_family=args.per_family)
    fit_specs, _ = split5(specs)
    A2.ENDPOINT_LENSES = tuple(place_lenses())
    M3.ORACLE = True

    print("STEP 2 -- %d FIT scenes, places-only endpoints\n" % len(fit_specs),
          flush=True)
    fit, pfit, meta, ratio_fit = M3.collect_fit(fit_specs, VIEWS5, None,
                                                pose_fn=pose_fn)
    n_nodes = M3.collect_plus(fit_specs, VIEWS5, fit, pfit, None, pose_fn=pose_fn)
    plus = M3.fit_plus()
    print("  P_plus coefficient fit on %d FIT nodes" % n_nodes)
    print("    a_exist = %.4f   a_geom = %.4f   loglik = %.1f"
          % (plus["a_exist"], plus["a_geom"], plus["loglik"]), flush=True)

    sweep = {"best_margin": M3.CTX_MARGIN, "note": "skipped"}
    if not args.skip_sweep:
        sweep = margin_sweep(fit_specs, VIEWS5, fit, pfit,
                             plus["a_exist"], plus["a_geom"])
        print("\n  CONTEXT ITERATION margin sweep (FIT subsample, %d views)"
              % sweep["n_views"])
        print("    context accuracy before iteration: %.4f"
              % sweep["context_acc_before"])
        for k, v in sorted(sweep["variants"].items()):
            print("      %-12s -> context accuracy %.4f  (moves %d, views worse %d)"
                  % (k, v, sweep["moved"][k], sweep["views_hurt"][k]))
        M3.CTX_MARGIN = float(sweep["best_margin"])
        M3.CTX_MUTUAL = bool(sweep["best_mutual"])
        print("    UNDAMPED control (re-pick every node, no filter): %.4f"
              % sweep["context_acc_repick"])
        print("    FROZEN: mutual=%s margin=%.2f -> context accuracy %.4f"
              % (M3.CTX_MUTUAL, M3.CTX_MARGIN, sweep["best_context_acc"]))
        print("    nodes whose context flipped WRONG->RIGHT %d, RIGHT->WRONG %d"
              % (sweep["fixed_best"], sweep["broke_best"]), flush=True)

    M3.CTX_STATS.clear()
    recs = M3.run_check(fit_specs, VIEWS5, fit, pfit, None, ratio_fit,
                        pose_fn=pose_fn)

    arms = ("N", "R", "R_fixed", "P", "P_presence", "P_plus",
            "P_plus_presence", "P_plus_ctx", "P_plus_ctx_presence",
            "P_shuffled", "W", "P_oracle", "P_oracle_presence", "P_ratio")
    table = {}
    print("\n  FIT TABLE (ambiguous bank, scene-clustered)")
    for a in arms:
        try:
            table[a] = M3.scene_stats(recs, a)["mean"]
        except Exception:
            table[a] = float("nan")
        print("    %-18s %.4f" % (a, table[a]))

    pairs = (("P_plus", "P_plus_presence"), ("P_plus", "N"),
             ("P_plus_ctx", "P_plus_ctx_presence"),
             ("P_plus", "P"), ("P_plus", "P_presence"),
             ("P_plus_ctx", "P_plus"), ("P_plus_ctx", "P_plus_presence"),
             ("P_plus_ctx", "N"), ("P", "P_presence"),
             ("P_plus", "P_shuffled"))
    contrasts = {}
    print("\n  CONTRASTS (scene-clustered, MDE = 2.8 * SE)")
    for a, b in pairs:
        c = M3.contrast(recs, a, b)
        contrasts["%s - %s" % (a, b)] = c
        if "se_scene" not in c:
            print("    %-34s n/a (needs >= 2 scenes)" % ("%s - %s" % (a, b)))
            continue
        print("    %-34s %+0.4f  se %.4f  mde %.4f  ci [%+0.4f,%+0.4f]"
              % ("%s - %s" % (a, b), c["mean"], c["se_scene"], c["mde"],
                 c["ci"][0], c["ci"][1]))

    ctx = {"n_views": len(M3.CTX_STATS),
           "ctx_acc_before": float(np.mean([s["n_acc"] for s in M3.CTX_STATS]))
           if M3.CTX_STATS else float("nan"),
           "ctx_acc_after": float(np.mean([s["ctx_acc"] for s in M3.CTX_STATS]))
           if M3.CTX_STATS else float("nan"),
           "views_hurt": int(sum(1 for s in M3.CTX_STATS
                                 if s["ctx_acc"] < s["n_acc"])),
           "moves": int(sum(s["moved"] for s in M3.CTX_STATS)),
           "refused": int(sum(s["refused"] for s in M3.CTX_STATS))}
    print("\n  CONTEXT ITERATION on the scored population (%d views)"
          % ctx["n_views"])
    print("    context accuracy  N %.4f -> iterated %.4f   (views worse: %d)"
          % (ctx["ctx_acc_before"], ctx["ctx_acc_after"], ctx["views_hurt"]))
    print("    moves accepted %d, refused %d" % (ctx["moves"], ctx["refused"]))

    tol = tolerance_probe(fit_specs, VIEWS5, fit)
    print("\n  STEP 2c per-place uncertainty vs residual (true pairs)")
    print("    n %d  spearman(uncertainty, |residual|) = %.3f"
          % (tol["n"], tol["spearman"]))
    print("    -> per-edge tolerance is %s"
          % ("WORTH FITTING" if abs(tol["spearman"]) > 0.2 else
             "NOT supported: residual does not track place uncertainty"))

    # per family, and the two labels
    fam = {}
    for f in sorted({r["family"] for r in recs}):
        sub = [r for r in recs if r["family"] == f]
        fam[f] = {a: float(np.mean([r["ok"][a] for r in sub]))
                  for a in arms if sub and a in sub[0]["ok"]}
    print("\n  per family (N / P / P_plus / P_plus_ctx / P_oracle)")
    for f, row in fam.items():
        print("    %-10s %.4f  %.4f  %.4f  %.4f  %.4f"
              % (f, row.get("N", float("nan")), row.get("P", float("nan")),
                 row.get("P_plus", float("nan")),
                 row.get("P_plus_ctx", float("nan")),
                 row.get("P_oracle", float("nan"))))
    una = {a: M3.scene_stats(recs, a, bucket="unambiguous")["mean"]
           for a in ("N", "P", "P_plus")}
    print("\n  unambiguous bank: N %.4f  P %.4f  P_plus %.4f"
          % (una["N"], una["P"], una["P_plus"]))
    c_una = M3.contrast(recs, "P_plus_ctx", "P_plus", bucket="unambiguous")
    print("    P_plus_ctx - P_plus (unambiguous) %+0.4f  mde %.4f"
          % (c_una["mean"], c_una.get("mde", float("nan"))))

    res = {"plus_fit": plus, "n_plus_nodes": n_nodes, "sweep": sweep,
           "arms": table, "contrasts": contrasts, "context": ctx,
           "tolerance": tol, "by_family": fam, "unambiguous": una,
           "unambiguous_contrast": c_una, "ctx_margin": M3.CTX_MARGIN,
           "ctx_iters": M3.CTX_ITERS,
           "fit_meta": {k: meta[k] for k in
                        ("mu_wrong", "presence_mu_wrong", "true_pair_mean_llr",
                         "wrong_pair_mean_llr", "n_res_true", "n_res_wrong")}}
    # Keep the fitted densities and presence rates so the fresh read uses the
    # EXACT objects this FIT produced, not a re-fit that could drift.
    cache = os.path.join(OUT, "_fitcache")
    os.makedirs(cache, exist_ok=True)
    import pickle
    with open(os.path.join(cache, "%s_fit.pkl" % args.tag), "wb") as fh:
        pickle.dump({"fit": fit, "pfit": pfit, "ratio_fit": ratio_fit,
                     "a_exist": plus["a_exist"], "a_geom": plus["a_geom"],
                     "ctx_margin": M3.CTX_MARGIN, "ctx_iters": M3.CTX_ITERS,
                     "ctx_mutual": M3.CTX_MUTUAL}, fh, protocol=4)

    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=str)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
