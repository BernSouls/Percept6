"""Mission 3 runner: calibrated geometry arms on DEV-3, FIT then CHECK.

FIT   -- densities are fitted here (true vs wrong residuals, per-class edge
         presence), under the SAME deployed context mechanism the arms use.
         Truth-free at match time: the builder and the scorer see no object
         masks, no depth and no tracks; truth only labels the fitted pools.
CHECK -- every arm is scored here, once per declared look, clustered by SCENE.

Arms (all on the same graph, same candidate list):
  N            top similarity
  R            shipped edge term, presence-only, unnormalised sum  (FROZEN)
  R_fixed      normalised, geometry-verified, disagreement-penalised
  P            calibrated LLR: padded, clipped, summed slots        (NEW)
  P_presence   same caps/padding/normalisation, residual switched off
  P_shuffled   P with payload geometry permuted among same-class edges
  W            P with payloads rewired to wrong partners

Every arm that is not N or R blends similarity with its own term at the same
weight, `EDGE_BETA = 0.5`, so no arm wins by getting a different blend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = os.path.join(ROOT, "Topology", "results", "placement")
NBOOT = 2000
SEED = 20261118
EDGE_BETA = 0.5
#: Candidate list per node.  Capped so the rival set is the one the assignment
#: actually sees and so every pairwise loop is bounded.
CAND_CAP = 6
#: Run the DEV-only oracle-context ceiling diagnostic.
ORACLE = True

ARMS = ("N", "R", "R_fixed", "P", "P_presence", "P_shuffled", "W")


def _rows_for(g_q: Any, g_r: Any, pop: Sequence[int],
              ref_cands: Sequence[int]) -> Dict[int, List[Tuple[int, float]]]:
    from Topology import placement_census as C
    return C.similarity_rows(g_q, g_r, list(pop), list(ref_cands))


def _payload(mod: Any, g: Any, img: Any, obj: Dict[int, int],
             intra: bool = False) -> Dict[Tuple[int, int, Any], Any]:
    return mod.build_edges(g.subgraph, img, object_of=obj, intra_only=intra)


# ---------------------------------------------------------------------------
def collect_fit(specs: Sequence[Any], views: Sequence[Any], limit: Optional[int]
                ) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fit the residual densities and the presence rates on the FIT half."""
    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching import placement_llr as L
    from matching.placement_scorer import incident_index, payload_index
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, track_optimal

    res_t: List[float] = []
    res_w: List[float] = []
    rat_t: List[float] = []
    rat_w: List[float] = []
    pre_t: List[Tuple[str, bool]] = []
    pre_w: List[Tuple[str, bool]] = []
    n_cand_t = n_cand_w = 0
    used = 0
    for spec in (specs[:limit] if limit else specs):
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        refc = [v for v in sorted(int(n) for n in g0.subgraph.nodes)
                if o0.get(v, -1) >= 0]
        intra = (spec.family == "NEGATIVE-A")
        r_pay = _payload(PE, g0, r0.image_u8, o0, intra)
        r_idx = payload_index(r_pay)
        for label, view in views:
            pv = view_pose(spec, view)
            sc1 = posed(spec.scene, pv["rot_deg"], tuple(pv["shift"]),
                        pv["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            q_pay = _payload(PE, g1, r1.image_u8, o1, intra)
            q_idx, q_inc = payload_index(q_pay), incident_index(q_pay)
            truth = {int(q): int(r)
                     for r, q in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
            rows = _rows_for(g1, g0, pop, refc)
            nmap = {u: rows[u][0][0] for u in pop if rows.get(u)}
            pos_q = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                              float(g1.subgraph.anchor_of_node[n][1]))
                     for n in g1.subgraph.nodes if n in g1.subgraph.anchor_of_node}
            pos_r = {int(n): (float(g0.subgraph.anchor_of_node[n][0]),
                              float(g0.subgraph.anchor_of_node[n][1]))
                     for n in g0.subgraph.nodes if n in g0.subgraph.anchor_of_node}
            for u in pop:
                row = rows.get(u) or []
                for v, _s in row[:CAND_CAP]:
                    slots = L.slots_for_candidate(u, v, nmap, q_inc, q_idx, r_idx)
                    good = (truth.get(u) == v)
                    if good:
                        n_cand_t += 1
                    else:
                        n_cand_w += 1
                    for (name, _o, res) in slots:
                        exists = res is not None
                        if exists:
                            (res_t if good else res_w).append(res)
                        (pre_t if good else pre_w).append((name, exists))
                    for (_n2, rres) in L.ratio_slots(u, v, nmap, q_inc,
                                                    pos_q, pos_r):
                        if rres is not None:
                            (rat_t if good else rat_w).append(rres)
        used += 1
    fit = L.fit_llr(res_t, res_w)
    ratio_fit = L.fit_llr(rat_t, rat_w)
    fit.extra["ratio_true_mean"] = (float(np.mean([ratio_fit.of(r) for r in rat_t]))
                                    if rat_t else float("nan"))
    fit.extra["ratio_wrong_mean"] = (float(np.mean([ratio_fit.of(r) for r in rat_w]))
                                     if rat_w else float("nan"))
    fit.extra["n_ratio_true"] = len(rat_t)
    fit.extra["n_ratio_wrong"] = len(rat_w)
    # The mandatory sanity check on the fitted densities: a wrong pair must be
    # worth less than nothing and a true pair more than nothing.
    true_mean = float(np.mean([fit.of(r) for r in res_t])) if res_t else float("nan")
    wrong_mean = float(np.mean([fit.of(r) for r in res_w])) if res_w else float("nan")
    fit.extra["true_pair_mean_llr"] = true_mean
    fit.extra["wrong_pair_mean_llr"] = wrong_mean
    pfit = L.fit_presence(pre_t, pre_w)
    meta = {"scenes": used, "n_res_true": len(res_t), "n_res_wrong": len(res_w),
            "n_pres_true": len(pre_t), "n_pres_wrong": len(pre_w),
            "n_candidates_true": n_cand_t, "n_candidates_wrong": n_cand_w,
            "mu_wrong": fit.mu_wrong, "presence_mu_wrong": pfit.mu_wrong,
            "llr_by_bin": fit.llr.tolist(),
            "true_pair_mean_llr": fit.extra.get("true_pair_mean_llr"),
            "wrong_pair_mean_llr": fit.extra.get("wrong_pair_mean_llr"),
            "presence_llr": dict(pfit.llr)}
    meta["ratio_true_mean_llr"] = fit.extra["ratio_true_mean"]
    meta["ratio_wrong_mean_llr"] = fit.extra["ratio_wrong_mean"]
    meta["n_ratio_true"] = fit.extra["n_ratio_true"]
    meta["n_ratio_wrong"] = fit.extra["n_ratio_wrong"]
    return fit, pfit, meta, ratio_fit


# ---------------------------------------------------------------------------
def run_check(specs: Sequence[Any], views: Sequence[Any], fit: Any, pfit: Any,
              limit: Optional[int], ratio_fit: Any = None) -> List[Dict[str, Any]]:
    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching import placement_llr as L
    from matching.placement_scorer import incident_index, payload_index
    from Topology.placement_arms2 import (AMBIGUOUS_MARGIN, geometry_credit,
                                          is_ambiguous, is_ambiguous_part4,
                                          presence_credit, rewired_payloads,
                                          shipped_edges, shuffled_payloads)
    from Topology.placement_cache import get_or_build
    from Topology.placement_dev3 import view_pose
    from Topology.placement_scenes import posed
    from Topology.placement_truth import node_objects, track_optimal

    records: List[Dict[str, Any]] = []
    for spec in (specs[:limit] if limit else specs):
        r0 = spec.scene.render()
        g0 = get_or_build(r0.image_u8, spec.name + "|ref")
        o0 = node_objects(r0, g0)
        refc = [v for v in sorted(int(n) for n in g0.subgraph.nodes)
                if o0.get(v, -1) >= 0]
        intra = (spec.family == "NEGATIVE-A")
        r_pay = _payload(PE, g0, r0.image_u8, o0, intra)
        r_idx = payload_index(r_pay)
        r_ship = shipped_edges(g0.subgraph)
        ref_by_src: Dict[int, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list))
        ref_by: Dict[int, Dict[str, List[Tuple[int, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        for (s, d, et), params in sorted(r_ship.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
            nm = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
            ref_by_src[s][nm].append(d)
            ref_by[s][nm].append((d, params))
        for label, view in views:
            pv = view_pose(spec, view)
            sc1 = posed(spec.scene, pv["rot_deg"], tuple(pv["shift"]),
                        pv["scale"])
            r1 = sc1.render()
            g1 = get_or_build(r1.image_u8, spec.name + "|" + label)
            o1 = node_objects(r1, g1)
            q_pay = _payload(PE, g1, r1.image_u8, o1, intra)
            q_idx, q_inc = payload_index(q_pay), incident_index(q_pay)
            q_sh = shuffled_payloads(q_pay)
            q_w = rewired_payloads(q_pay)
            sh_idx, sh_inc = payload_index(q_sh), incident_index(q_sh)
            w_idx, w_inc = payload_index(q_w), incident_index(q_w)
            ship_inc: Dict[int, List[Tuple[int, bool, str]]] = defaultdict(list)
            q_by: Dict[int, Dict[str, List[Tuple[int, Any]]]] = defaultdict(
                lambda: defaultdict(list))
            q_ship = shipped_edges(g1.subgraph)
            for (s, d, et) in sorted(q_ship,
                                     key=lambda e: (int(e[0]), int(e[1]),
                                                    getattr(e[2], "name", ""))):
                nm = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
                ship_inc[int(s)].append((int(d), True, nm))
                q_by[int(s)][nm].append((int(d), q_ship[(s, d, et)]))
                q_by[int(d)][nm].append((int(s), q_ship[(s, d, et)]))
            truth = {int(q): int(r)
                     for r, q in track_optimal(spec.scene, r0, sc1, r1, g0, g1).items()}
            pop = [u for u in sorted(truth) if o1.get(u, -1) >= 0]
            rows = _rows_for(g1, g0, pop, refc)
            nmap = {u: rows[u][0][0] for u in pop if rows.get(u)}
            pos_q = {int(n): (float(g1.subgraph.anchor_of_node[n][0]),
                              float(g1.subgraph.anchor_of_node[n][1]))
                     for n in g1.subgraph.nodes if n in g1.subgraph.anchor_of_node}
            pos_r = {int(n): (float(g0.subgraph.anchor_of_node[n][0]),
                              float(g0.subgraph.anchor_of_node[n][1]))
                     for n in g0.subgraph.nodes if n in g0.subgraph.anchor_of_node}
            for u in pop:
                row = rows.get(u) or []
                if not row:
                    continue
                cand = row[:CAND_CAP]
                tv = truth.get(u)

                def blend(term: float, s: float) -> float:
                    return (1.0 - EDGE_BETA) * s + EDGE_BETA * term

                def pick(fn) -> int:
                    best, bs = -1, -float("inf")
                    for v, s in cand:
                        val = fn(v, s)
                        if val > bs:
                            bs, best = val, v
                    return best

                if ORACLE:
                    ARMS_ALL = ARMS + ("P_oracle", "P_oracle_presence", "P_ratio")
                else:
                    ARMS_ALL = ARMS
                scored: Dict[str, int] = {"N": cand[0][0]}
                scored["R"] = pick(lambda v, s: blend(
                    presence_credit(u, v, ship_inc, ref_by_src), s))
                scored["R_fixed"] = pick(lambda v, s: blend(
                    geometry_credit(u, v, ship_inc, ref_by, nmap, q_by), s))
                scored["P"] = pick(lambda v, s: blend(
                    L.score_candidate(u, v, nmap, q_inc, q_idx, r_idx, fit), s))
                scored["P_presence"] = pick(lambda v, s: blend(
                    L.presence_score(u, v, nmap, q_inc, q_idx, r_idx, pfit), s))
                scored["P_ratio"] = pick(lambda v, s: blend(
                    L.ratio_score(u, v, nmap, q_inc, pos_q, pos_r, ratio_fit), s))
                scored["P_shuffled"] = pick(lambda v, s: blend(
                    L.score_candidate(u, v, nmap, sh_inc, sh_idx, r_idx, fit), s))
                scored["W"] = pick(lambda v, s: blend(
                    L.score_candidate(u, v, nmap, w_inc, w_idx, r_idx, fit), s))
                # DEV-ONLY CEILING DIAGNOSTIC.  Oracle context = the neighbours'
                # images come from renderer truth instead of from N.  This
                # violates TRUTH-FREE AT MATCH TIME by construction, which is
                # exactly why it can never enter a deployed arm: it exists to
                # say whether the ceiling is set by the CONTEXT (N is right on
                # ~30% of nodes) or by the PAYLOAD/frames.  Reported as a
                # ceiling only.
                if ORACLE:
                    scored["P_oracle"] = pick(lambda v, s: blend(
                        L.score_candidate(u, v, truth, q_inc, q_idx, r_idx, fit), s))
                    scored["P_oracle_presence"] = pick(lambda v, s: blend(
                        L.presence_score(u, v, truth, q_inc, q_idx, r_idx, pfit), s))

                records.append({
                    "scene": spec.name, "view": label, "family": spec.family,
                    "node": int(u),
                    "bucket": "ambiguous" if is_ambiguous(row) else "unambiguous",
                    "part4": bool(is_ambiguous_part4(row)),
                    "margin": float(row[0][1] - row[1][1]) if len(row) > 1 else None,
                    "top_sim": float(row[0][1]),
                    "ok": {a: bool(scored[a] == tv) for a in ARMS_ALL},
                    "n_cand": len(cand),
                })
    return records


# ---------------------------------------------------------------------------
def scene_stats(records: Sequence[Dict[str, Any]], arm: str,
                bucket: str = "ambiguous", part4: bool = False
                ) -> Dict[str, Any]:
    """Scene-clustered accuracy: scenes are the top-level unit."""
    per: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r["bucket"] != bucket:
            continue
        if part4 and not r["part4"]:
            continue
        per[r["scene"]].append(float(r["ok"][arm]))
    keys = sorted(per)
    if not keys:
        return {"mean": float("nan"), "n": 0, "n_scenes": 0}
    means = np.array([float(np.mean(per[k])) for k in keys])
    return {"mean": float(np.mean([v for k in keys for v in per[k]])),
            "scene_mean": float(means.mean()),
            "scene_se": float(means.std(ddof=1) / np.sqrt(len(keys))) if len(keys) > 1 else float("nan"),
            "n": int(sum(len(per[k]) for k in keys)), "n_scenes": len(keys),
            "per_scene": {k: float(np.mean(per[k])) for k in keys}}


def contrast(records: Sequence[Dict[str, Any]], a: str, b: str,
             bucket: str = "ambiguous", part4: bool = False) -> Dict[str, Any]:
    per: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r["bucket"] != bucket:
            continue
        if part4 and not r["part4"]:
            continue
        per[r["scene"]].append(float(r["ok"][a]) - float(r["ok"][b]))
    keys = sorted(per)
    if len(keys) < 2:
        return {"mean": float("nan"), "n_scenes": len(keys)}
    d = np.array([float(np.mean(per[k])) for k in keys])
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(keys)))
    # Exact two-sided sign-flip over the scene differences.
    hits = total = 0
    for signs in product((1.0, -1.0), repeat=len(keys)):
        total += 1
        if abs(float(np.mean(signs * d))) >= abs(mean) - 1e-15:
            hits += 1
    return {"mean": mean, "se_scene": se, "mde": 2.8 * se,
            "ci": [mean - 1.96 * se, mean + 1.96 * se],
            "p_exact": hits / total, "n_scenes": len(keys),
            "per_scene": {k: float(np.mean(per[k])) for k in keys}}


def verdict(c: Dict[str, Any]) -> str:
    if not np.isfinite(c.get("mean", float("nan"))):
        return "NO DATA"
    if c["mean"] > c["mde"] and c["ci"][0] > 0:
        return "POSITIVE"
    if c["mean"] < -c["mde"] and c["ci"][1] < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-limit", type=int, default=None)
    ap.add_argument("--check-limit", type=int, default=None)
    ap.add_argument("--tag", default="m3")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args(argv)
    global ORACLE
    ORACLE = not args.no_oracle

    from Topology.placement_dev3 import VIEWS, all_specs, split

    specs = all_specs()
    fit_specs, check_specs = split(specs)
    print("DEV-3: %d scenes -> FIT %d / CHECK %d"
          % (len(specs), len(fit_specs), len(check_specs)))

    fit, pfit, meta, ratio_fit = collect_fit(fit_specs, VIEWS, args.fit_limit)
    print("FIT: %d scenes, %d true residuals, %d wrong residuals, mu_wrong %.4f"
          % (meta["scenes"], meta["n_res_true"], meta["n_res_wrong"],
             meta["mu_wrong"]))
    ok_fit = (meta["wrong_pair_mean_llr"] < 0.0 < meta["true_pair_mean_llr"])
    print("     fitted LLR: wrong-pair mean %+.4f < 0 < true-pair mean %+.4f  %s"
          % (meta["wrong_pair_mean_llr"], meta["true_pair_mean_llr"],
             "OK" if ok_fit else "*** FIT IS MISFIT ***"))

    records = run_check(check_specs, VIEWS, fit, pfit, args.check_limit,
                        ratio_fit)
    res: Dict[str, Any] = {"fit": meta, "n_records": len(records),
                           "scenes": len({r["scene"] for r in records}),
                           "arms": {}, "contrasts": {}}

    print("\n== CHECK, near-tie ambiguous nodes")
    arms_here = list(ARMS) + [a for a in ("P_oracle", "P_oracle_presence", "P_ratio")
                              if records and a in records[0]["ok"]]
    for arm in arms_here:
        s = scene_stats(records, arm)
        res["arms"][arm] = s
        print("  %-11s %.4f  scene_se %.4f  n=%d scenes=%d"
              % (arm, s["mean"], s["scene_se"], s["n"], s["n_scenes"]))

    print("\n  contrast            effect   scene_se     MDE        CI           p_exact  verdict")
    pairs = [("P", "N"), ("P", "P_presence"), ("P", "P_shuffled"),
             ("P", "W"), ("P", "R_fixed"), ("R", "N"), ("R_fixed", "R"),
             ("P_presence", "N")]
    if "P_oracle" in records[0]["ok"] if records else False:
        pairs += [("P_oracle", "P"), ("P_oracle", "P_presence"),
                  ("P_oracle_presence", "P_presence"), ("P_oracle", "P_oracle_presence"),
                  ("P_ratio", "N"), ("P_ratio", "P_presence"), ("P_ratio", "P")]
    for a, b in pairs:
        c = contrast(records, a, b)
        res["contrasts"]["%s-%s" % (a, b)] = c
        print("  %-18s %+.4f  %.4f    %.4f  [%+.4f,%+.4f]  %.4f  %s"
              % ("%s-%s" % (a, b), c["mean"], c["se_scene"], c["mde"],
                 c["ci"][0], c["ci"][1], c["p_exact"], verdict(c)))

    c = contrast(records, "P", "N", bucket="unambiguous")
    res["contrasts"]["unambiguous|P-N"] = c
    print("\n  unambiguous P-N   %+.4f  MDE %.4f  %s"
          % (c["mean"], c["mde"], verdict(c)))
    c2 = contrast(records, "P", "N", part4=True)
    res["contrasts"]["part4|P-N"] = c2
    print("  strict-label P-N  %+.4f  MDE %.4f  %s  (n_scenes %d)"
          % (c2["mean"], c2["mde"], verdict(c2), c2["n_scenes"]))

    path = os.path.join(OUT, "m3_%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    dump = os.path.join(OUT, "m3_%s_nodes.json" % args.tag)
    with open(dump, "w") as fh:
        json.dump(records, fh)
    print("\nwrote %s and %s" % (path, dump))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
