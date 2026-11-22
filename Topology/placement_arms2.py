"""The arms: N, R, R-fixed, P, P-shuffled, W -- on the placement edges.

IDENTICAL CODE RUNS ON DEV AND EVAL (`--set dev|eval`).  Nothing is chosen by
looking at EVAL: constants come from DEV and are frozen in the modules.

POPULATION
  Query nodes whose anchor sits ON AN OBJECT (object_id >= 0) and whose truth
  partner is known, with reference candidates restricted to ON-OBJECT nodes of
  the same lens.  Background lattice points are excluded because they have no
  orientation to read and no object to correspond (measured coherence 0.000);
  the restriction is stated, pre-registered, and reported alongside the
  all-node numbers in the census.

ARMS
  N          node similarity alone.
  R          the shipped matcher's edge term, FROZEN: a candidate is credited
             for having a same-class incident edge pair present, which is a
             PRESENCE term -- it does not look at the relation's geometry.
  R-fixed    the same term with presence replaced by GEOMETRY: a counterpart
             edge counts only if its parameters agree, and edges WITHOUT a
             counterpart are subtracted instead of ignored.
  P          node similarity x (1 + w*(support - 0.5)), ONE pre-stated w.
  P-shuffled same links, geometry permuted among edges of the same class.
  W          endpoints rewired, degree preserved, canonical sort before the
             seeded shuffle.

SCORE-LEVEL NEGATIVE CONTROLS (not accuracy)
  NEGATIVE-A  intra-object edges ONLY in the builder: total support for the
              identity mapping must equal the swap's, within the spread frozen
              on DEV.  A difference is a POSITIONAL LEAK.
  NEGATIVE-B  same scene, cross-object edges allowed: identity support must
              BEAT swap support -- a positive control for the mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")
NBOOT = 2000
SEED = 42
EDGE_BETA = 0.5          # the shipped matcher's appearance/structure balance


# ---------------------------------------------------------------------------
def lens_of(g: Any, n: int) -> str:
    s = g.sigs.get(n)
    return str(s[0]) if s is not None else "?"


def sim_rows(g_q: Any, g_r: Any, pop: List[int],
             cands: Dict[str, List[int]]) -> Dict[int, List[Tuple[int, float]]]:
    out: Dict[int, List[Tuple[int, float]]] = {}
    for u in pop:
        su = g_q.sigs.get(u)
        if su is None:
            continue
        row: List[Tuple[int, float]] = []
        for v in cands.get(lens_of(g_q, u), ()):
            sv = g_r.sigs.get(v)
            if sv is None:
                continue
            try:
                row.append((int(v), float(su[1].similarity(sv[1]))))
            except Exception:
                continue
        row.sort(key=lambda t: (-t[1], t[0]))
        out[int(u)] = row
    return out


#: A node is AMBIGUOUS when its top two candidates are a NEAR-TIE: the
#: similarity margin is at most this.  Truth-free and observable, and -- unlike
#: Part 4's label, which also demanded top similarity >= 0.99 -- it is the
#: quantity that actually decides whether a bounded modifier can overturn the
#: decision.  Measured on DEV, P benefits at EVERY margin band (P-N by band:
#: +0.079 at <=0.02, +0.032, +0.049, +0.029, +0.027, then 0.000 once the gap is
#: too wide for the modifier to matter), so this label describes the population
#: the arm is asked to win, not one chosen to flatter it.  Part 4's stricter
#: label is reported alongside as `ambiguous_part4`.
AMBIGUOUS_MARGIN = 0.05


def is_ambiguous(row: List[Tuple[int, float]],
                 margin_max: float = AMBIGUOUS_MARGIN) -> bool:
    """Near-tie?  (top two candidates within `margin_max`.)"""
    if len(row) < 2:
        return False
    return bool((row[0][1] - row[1][1]) <= margin_max)


def is_ambiguous_part4(row: List[Tuple[int, float]], top_sim: float = 0.99,
                       margin_max: float = 0.05) -> bool:
    """Part 4's stricter observable rule, kept for comparison."""
    if len(row) < 2:
        return False
    return bool(row[0][1] >= top_sim and (row[0][1] - row[1][1]) <= margin_max)


# ---------------------------------------------------------------------------
# edge terms
# ---------------------------------------------------------------------------
def shipped_edges(subgraph: Any) -> Dict[Tuple[int, int, Any], Any]:
    """The edge set R is allowed to see: everything EXCEPT the new class."""
    from core.types import EdgeType
    out = {}
    for (s, d, et) in subgraph.edges:
        if et == EdgeType.SOURCE_FRAME_PLACEMENT:
            continue
        out[(int(s), int(d), et)] = subgraph.edge_params.get((s, d, et))
    return out


def presence_credit(u: int, v: int,
                    inc: Dict[int, List[Tuple[int, bool, str]]],
                    ref_by_src: Dict[int, Dict[str, List[int]]]) -> float:
    """R's term, FROZEN: credit for HAVING a same-class edge, geometry ignored."""
    es = inc.get(u, ())
    if not es:
        return 0.0
    # SUM, not mean: this is the hub bias the mission suspects.  A candidate
    # that happens to have many same-class edges is credited for having them,
    # whatever their geometry says, and a node with a busy neighbourhood wins
    # the assignment without any relation being verified.  R-fixed replaces
    # this with a normalised, geometry-verified, disagreement-penalised term.
    n = 0
    for (_other, _is_src, name) in es:
        n += len(ref_by_src.get(v, {}).get(name, ()))
    return float(n)


#: R-fixed's agreement threshold, in tolerance units: a counterpart edge counts
#: as agreeing when its payload residual is at most this.  Fixed at ONE
#: tolerance unit before any CHECK was evaluated.
GEOM_AGREE_UNITS = 1.0


def params_agree(a: Any, b: Any, tol: float = 1.0) -> bool:
    """Do two SHIPPED edge parameter records describe the same relation?

    Public fields only: same registered class, same sign, and every quantized
    axis within ``tol`` bins.  Fixed before CHECK was evaluated.
    """
    if a is None or b is None:
        return False
    if getattr(a, "class_", None) != getattr(b, "class_", None):
        return False
    if int(getattr(a, "sign", 1)) != int(getattr(b, "sign", 1)):
        return False
    qa = getattr(a, "quantized", ()) or ()
    qb = getattr(b, "quantized", ()) or ()
    if len(qa) != len(qb):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(qa, qb))


def geometry_credit(u: int, v: int,
                    inc: Dict[int, List[Tuple[int, bool, str]]],
                    ref_by: Dict[int, Dict[str, List[Tuple[int, Any]]]],
                    mapping: Dict[int, int],
                    q_by: Optional[Dict[int, Dict[str, List[Tuple[int, Any]]]]] = None
                    ) -> float:
    """R-fixed's term: a counterpart must AGREE, and silence is subtracted.

    Two changes, both aimed at the measured harm of the presence term:
      * a counterpart edge counts only if its parameters agree
        (``edge_pair_credit`` >= 0.5), so being connected is not enough;
      * the denominator is every incident edge, and non-agreeing ones are
        SUBTRACTED, so a candidate is not rewarded merely for existing in a
        busy neighbourhood.
    """
    es = inc.get(u, ())
    if not es:
        return 0.0
    agree = 0
    dis = 0
    for (other, _is_src, name) in es:
        cand_edges = ref_by.get(v, {}).get(name, [])
        mo = mapping.get(int(other))
        qv = None
        if q_by is not None:
            src = int(u) if _is_src else int(other)
            for (d1, params) in q_by.get(src, {}).get(name, []):
                if _is_src and int(d1) == int(other):
                    qv = params
                    break
                if (not _is_src) and int(d1) == int(u):
                    qv = params
                    break
        hit = None
        for (d2, params) in cand_edges:
            if mo is not None and int(d2) == int(mo):
                hit = params
                break
        # A counterpart edge counts only if its PARAMETERS agree.  The first
        # implementation incremented `agree` as soon as a counterpart EDGE
        # existed and never looked at `params`, so R_fixed was presence with a
        # different denominator -- measured: it returned 1.0 for a matched and
        # a wildly mismatched relation alike (canary in test_placement_m3),
        # which is why mission 2 found R_fixed "only matched N".
        if hit is None or qv is None or not params_agree(qv, hit):
            dis += 1
            continue
        agree += 1
    return (agree - dis) / len(es)


# ---------------------------------------------------------------------------
def shuffled_payloads(p: Dict[Tuple[int, int, Any], Any], seed: int = 7):
    by_type: Dict[str, List[Tuple[int, int, Any]]] = defaultdict(list)
    for k in sorted(p, key=lambda kk: (kk[0], kk[1], getattr(kk[2], "name", ""))):
        by_type[getattr(k[2], "name", str(k[2]))].append(k)
    rng = random.Random(seed)
    out = dict(p)
    for name in sorted(by_type):
        keys = by_type[name]
        vals = [p[k] for k in keys]
        perm = list(vals)
        for _ in range(64):
            rng.shuffle(perm)
            if all(a is not b for a, b in zip(vals, perm)):
                break
        for k, v in zip(keys, perm):
            out[k] = v
    return out


def rewired_payloads(p: Dict[Tuple[int, int, Any], Any], seed: int = 0):
    keys = sorted(p, key=lambda kk: (kk[0], kk[1], getattr(kk[2], "name", "")))
    if len(keys) < 2:
        return dict(p)
    rng = random.Random(seed)
    dsts = [k[1] for k in keys]
    for _ in range(200):
        rng.shuffle(dsts)
        if all(a != b for a, b in zip([k[1] for k in keys], dsts)):
            break
    return {(k[0], nd, k[2]): p[k] for k, nd in zip(keys, dsts)}


# ---------------------------------------------------------------------------
def total_support(pop: Sequence[int], mapping: Dict[int, int],
                  q_pay: Dict[Tuple, Any], r_pay: Dict[Tuple, Any],
                  tol: float) -> Tuple[float, int]:
    """Sum of per-node support under a GIVEN mapping (identity vs swap).

    The score-level control compares the support the payload gives the two
    candidate correspondences, with no accuracy involved.
    """
    from matching.placement_scorer import (incident_index, payload_index,
                                           support_for_candidate)
    q_idx = payload_index(q_pay)
    q_inc = incident_index(q_pay)
    r_idx = payload_index(r_pay)
    total = 0.0
    n = 0
    for u in pop:
        if int(u) not in mapping:
            continue
        sv = support_for_candidate(int(u), int(mapping[int(u)]), mapping,
                                   q_inc, q_idx, r_idx, tol)
        if sv is None:
            continue
        total += float(sv)
        n += 1
    return total, n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=("dev", "eval"))
    ap.add_argument("--dump-nodes", default=None,
                    help="write per-node records (arm correctness) to this path")
    args = ap.parse_args(argv)

    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching import placement_scorer as node_support_mod
    from matching.placement_scorer import support_for_candidate as node_support_candidate
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import (node_objects, swap_truth,
                                          track_optimal)
    from Topology import placement_census as C
    from Topology.placement_scenes import posed

    if args.set == "eval":
        from Topology.placement_eval_spec import eval_specs, EVAL_VIEWS as VIEWS
        from Topology.placement_eval_spec import load_frozen
        specs = eval_specs()
        frozen = load_frozen()
        tag = "eval"
    else:
        from Topology.placement_dev import all_specs, DEV_VIEWS as VIEWS
        specs = all_specs()
        frozen = json.load(open(os.path.join(OUT, "devfit_tolerance.json")))
        tag = "dev"

    toltab = frozen["tolerance"]
    weight = float(getattr(__import__("parameters"), "PLACEMENT_WEIGHT", 0.25))
    if tag == "eval":
        # The run must be the PRE-REGISTERED one.  Assert every constant the
        # pre-registration fixed, so an unnoticed drift fails loudly instead of
        # quietly producing a different experiment.
        from Topology.placement_eval_spec import FROZEN
        assert abs(weight - FROZEN["placement_weight"]) < 1e-12, weight
        assert abs(EDGE_BETA - FROZEN["edge_beta"]) < 1e-12, EDGE_BETA
        assert abs(AMBIGUOUS_MARGIN - FROZEN["ambiguous_margin"]) < 1e-12
        assert abs(PE.FRAME_COHERENCE_MIN - FROZEN["frame_coherence_min"]) < 1e-12
        assert int(PE.TARGET_K) == int(FROZEN["target_k"])
        assert abs(PE.TARGET_RADIUS_SCALES - FROZEN["target_radius_scales"]) < 1e-12
        assert all(abs(float(v["tolerance"]) - FROZEN["tolerance_all_views"]) < 1e-9
                   for v in toltab.values()), toltab

    acc: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list))
    controls: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    flips: Dict[str, List[float]] = defaultdict(list)

    for spec in specs:
        r0 = spec.scene.render()
        g_ref = get_or_build(r0.image_u8, spec.name + "|ref")
        obj_r = node_objects(r0, g_ref)
        r_pop = [n for n in sorted(obj_r) if obj_r[n] >= 0]
        intra = (spec.family == "NEGATIVE-A")
        r_pay = PE.build_edges(g_ref.subgraph, r0.image_u8, tolerance=None,
                               object_of=obj_r, intra_only=intra)
        r_ship = shipped_edges(g_ref.subgraph)

        ref_by_src: Dict[int, Dict[Any, List[int]]] = defaultdict(
            lambda: defaultdict(list))
        ref_by: Dict[int, Dict[Any, List[Tuple[int, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        # Keyed by CLASS NAME, not by the EdgeType object: `presence_credit`
        # looks up by name, so an object-keyed dict silently misses every time
        # and R degenerates to N/2 (measured: R-N exactly 0.0000 twice).
        for (s, d, et), params in sorted(r_ship.items(),
                                         key=lambda kv: (kv[0][0], kv[0][1])):
            nm = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
            ref_by_src[s][nm].append(d)
            ref_by[s][nm].append((d, params))

        for label, view in VIEWS:
            if label == "v0_base":
                continue
            sc1 = posed(spec.scene, float(view["rot_deg"]),
                        tuple(view["shift"]), float(view["scale"]))
            r1 = sc1.render()
            g_q = get_or_build(r1.image_u8, spec.name + "|" + label)
            obj_q = node_objects(r1, g_q)
            q_pay = PE.build_edges(g_q.subgraph, r1.image_u8, tolerance=None,
                                   object_of=obj_q, intra_only=intra)
            tol = float(toltab.get(label, {}).get("tolerance", 1.5))

            fwd = track_optimal(spec.scene, r0, sc1, r1, g_ref, g_q)
            truth = {int(q): int(r) for r, q in fwd.items()}
            sw = swap_truth(spec.scene, r0, g_ref, g_q, truth)

            pop = [u for u in sorted(truth) if obj_q.get(u, -1) >= 0]
            if not pop:
                continue
            cands: Dict[str, List[int]] = defaultdict(list)
            for v in r_pop:
                cands[lens_of(g_ref, v)].append(v)
            rows = sim_rows(g_q, g_ref, pop, cands)

            # provisional mapping = N, restricted to the object population
            nmap = {u: rows[u][0][0] for u in pop if rows.get(u)}

            # the query node's incident SHIPPED edges, canonically ordered
            from core.types import EdgeType
            # Incident SHIPPED edges of the query node, by class name.  This is
            # `ship_inc`, NOT `q_inc`: the placement incident index built below
            # reuses that name and would overwrite the edge classes R reads,
            # making every credit zero and R/R_fixed collapse onto N (measured:
            # R-N and R_fixed-R both exactly 0.0000, twice).
            ship_inc: Dict[int, List[Tuple[int, bool, str]]] = defaultdict(list)
            for (s, d, et) in sorted(g_q.subgraph.edges,
                                     key=lambda e: (int(e[0]), int(e[1]),
                                                    getattr(e[2], "name", ""))):
                if et == EdgeType.SOURCE_FRAME_PLACEMENT:
                    continue
                nm = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
                ship_inc[int(s)].append((int(d), True, nm))

            # CANDIDATE-DEPENDENT support.  Indexes are built once per view;
            # the support for (u -> v) is read against the correspondence the
            # candidate induces, with the neighbours' images from N's mapping.
            q_idx = node_support_mod.payload_index(q_pay)
            q_inc = node_support_mod.incident_index(q_pay)
            r_idx = node_support_mod.payload_index(r_pay)
            q_sh = shuffled_payloads(q_pay)
            q_w = rewired_payloads(q_pay)
            sh_idx = node_support_mod.payload_index(q_sh)
            sh_inc = node_support_mod.incident_index(q_sh)
            w_idx = node_support_mod.payload_index(q_w)
            w_inc = node_support_mod.incident_index(q_w)

            for u in pop:
                row = rows.get(u) or []
                if not row:
                    continue
                amb = is_ambiguous(row)
                bucket = "ambiguous" if amb else "unambiguous"
                p4 = is_ambiguous_part4(row)
                truth_v = truth.get(u)
                cand_vals = [v for v, _s in row[:6]]
                if truth_v not in cand_vals:
                    cand_vals = row and [v for v, _s in row]
                # restrict candidate list to a manageable prefix for R
                scored: Dict[str, int] = {}
                scored["N"] = row[0][0]

                def pick(fn):
                    best, bs = -1, -float("inf")
                    for v, s in row:
                        val = fn(v, s)
                        if val > bs:
                            bs, best = val, v
                    return best

                scored["R"] = pick(lambda v, s: s * (1 - EDGE_BETA)
                                   + EDGE_BETA * presence_credit(u, v, ship_inc,
                                                                ref_by_src))
                scored["R_fixed"] = pick(lambda v, s: s * (1 - EDGE_BETA)
                                         + EDGE_BETA * geometry_credit(
                                             u, v, ship_inc, ref_by, nmap))
                def pick_placement(v, s, idx, inc):
                    sv = node_support_mod.support_for_candidate(
                        u, v, nmap, inc, idx, r_idx, tol,
                        max_credit=float(getattr(__import__("parameters"),
                                                 "PLACEMENT_MAX_NEIGHBOUR_CREDIT", 1.0)))
                    if sv is None:
                        return s        # uncheckable: no credit, no penalty
                    return s * (1.0 + weight * (float(sv) - 0.5))

                scored["P"] = pick(lambda v, s: pick_placement(v, s, q_idx, q_inc))
                scored["P_shuffled"] = pick(
                    lambda v, s: pick_placement(v, s, sh_idx, sh_inc))
                scored["W"] = pick(lambda v, s: pick_placement(v, s, w_idx, w_inc))
                skey = "%s|%s|%s" % (spec.family, spec.name, label)
                if args.dump_nodes:
                    records.append({
                        "scene": spec.name, "view": label,
                        "family": spec.family, "node": int(u),
                        "bucket": bucket, "part4": bool(p4),
                        "margin": (float(row[0][1] - row[1][1])
                                   if len(row) > 1 else float("nan")),
                        "top_sim": float(row[0][1]),
                        "ok": {a: bool(v == truth_v) for a, v in scored.items()},
                    })
                for arm, pred in scored.items():
                    acc[arm][bucket].append((skey, 1.0 if pred == truth_v else 0.0))
                    if arm in ("N", "P"):
                        acc[arm + "_part4"]["ambiguous" if p4 else
                                          "unambiguous"].append(
                            (skey, 1.0 if pred == truth_v else 0.0))
                if scored["P"] != scored["N"]:
                    flips["changed"].append(1.0)
                    n_ok = scored["N"] == truth_v
                    p_ok = scored["P"] == truth_v
                    # "broken" must mean N was RIGHT and P is not; a node where
                    # N was also wrong is not broken by P, it is neutral.
                    if p_ok and not n_ok:
                        flips["fixed"].append(1.0)
                    elif n_ok and not p_ok:
                        flips["broken"].append(1.0)
                    else:
                        flips["neutral"].append(1.0)

            # ---- score-level controls
            if sw:
                # SAME nodes under both correspondences.  Summing each mapping
                # over whatever subset happened to be resolvable compares
                # different populations (measured: n_identity 35 vs n_swap 28)
                # and produced a spurious "identity beats swap" in the first
                # NEGATIVE-A read.  Only nodes checkable under BOTH mappings
                # enter the sums.
                from matching.placement_scorer import (incident_index as _ii,
                                                       payload_index as _pi,
                                                       support_for_candidate as _spc)
                _qi, _qc, _ri = _pi(q_pay), _ii(q_pay), _pi(r_pay)
                s_id = s_sw = 0.0
                n_both = 0
                # TRUE identity vs TRUE swap.  Using N's pick here (nmap[u])
                # compares an appearance decision against a truth-anchored
                # swap -- apples to oranges -- and is not a control at all.
                for u in sorted(u for u in pop if u in sw):
                    if int(u) not in truth:
                        continue
                    if int(truth[int(u)]) == int(sw[u]):
                        continue          # identical objects: no swap to test
                    a = _spc(int(u), int(truth[u]), nmap, _qc, _qi, _ri, tol)
                    b = _spc(int(u), int(sw[u]), nmap, _qc, _qi, _ri, tol)
                    if a is None or b is None:
                        continue
                    s_id += float(a)
                    s_sw += float(b)
                    n_both += 1
                n_id = n_sw = n_both
                pop_s = [u for u in pop if u in sw]
                controls.append({
                    "family": spec.family, "scene": spec.name, "view": label,
                    "n": len(pop_s),
                    "support_identity": s_id, "support_swap": s_sw,
                    "identity_minus_swap": s_id - s_sw,
                    "share_identity": (s_id / (s_id + s_sw)) if (s_id + s_sw)
                    else float("nan"),
                    "per_node_identity": s_id / max(1, n_id),
                    "per_node_swap": s_sw / max(1, n_sw),
                    "n_identity": n_id, "n_swap": n_sw,
                })
            print("  %-15s %-12s pop=%3d inc=%3d (amb %d) flips so far %d"
                  % (spec.name, label, len(pop), len(ship_inc),
                     sum(1 for u in pop if is_ambiguous(rows.get(u) or [])),
                     len(flips["changed"])), flush=True)

    # ---------------- aggregate
    def cluster_boot(pairs, nboot=NBOOT, seed=SEED):
        d: Dict[str, List[float]] = defaultdict(list)
        for k, v in pairs:
            d[k].append(v)
        keys = sorted(d)
        if not keys:
            return {"mean": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "mde": float("nan"), "n": 0,
                    "n_eff": 0}
        rng = np.random.default_rng(seed)
        boots = np.empty(nboot)
        for b in range(nboot):
            pick = rng.integers(0, len(keys), len(keys))
            vals: List[float] = []
            for pi in pick:
                vals.extend(d[keys[pi]])
            boots[b] = float(np.mean(vals)) if vals else 0.0
        lo, hi = float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
        hw = (hi - lo) / 2.0
        se = hw / 1.96
        return {"mean": float(np.mean([v for _, v in pairs])), "lo": lo,
                "hi": hi, "hw": hw, "se": se, "mde": 2.8 * se,
                "n": len(pairs), "n_eff": len(keys)}

    def paired_boot(pa, pb, nboot=NBOOT, seed=SEED):
        da: Dict[str, List[float]] = defaultdict(list)
        db: Dict[str, List[float]] = defaultdict(list)
        for k, v in pa:
            da[k].append(v)
        for k, v in pb:
            db[k].append(v)
        keys = sorted(set(da) & set(db))
        if not keys:
            return {"mean": float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "mde": float("nan"), "n": 0,
                    "n_eff": 0}
        rng = np.random.default_rng(seed)
        boots = np.empty(nboot)
        for b in range(nboot):
            pick = rng.integers(0, len(keys), len(keys))
            vals: List[float] = []
            for pi in pick:
                k = keys[pi]
                m = min(len(da[k]), len(db[k]))
                idx = rng.integers(0, m, m)
                vals.extend(da[k][j] - db[k][j] for j in idx)
            boots[b] = float(np.mean(vals)) if vals else 0.0
        lo, hi = float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
        hw = (hi - lo) / 2.0
        se = hw / 1.96
        diffs = [a - b for (_, a), (_, b) in zip(pa, pb)]
        return {"mean": float(np.mean(diffs)), "lo": lo, "hi": hi, "hw": hw,
                "se": se, "mde": 2.8 * se, "n": len(diffs), "n_eff": len(keys)}

    def verdict(d):
        """Decisive only when the effect clears the DESIGN FLOOR and the CI
        excludes zero.

        The MDE (2.8 * clustered SE) is the smallest effect this design can
        resolve.  A contrast is called POSITIVE when the observed effect is
        larger than that floor *and* the two-level CI excludes zero; NEGATIVE
        by symmetry.  Anything else -- including a CI that excludes zero but
        an effect too small to be worth claiming -- is INCONCLUSIVE, and
        INCONCLUSIVE is never reported as "no effect".
        """
        if not d.get("n") or not math.isfinite(d.get("mean", float("nan"))):
            return "NO DATA"
        if d["mean"] > d["mde"] and d["lo"] > 0.0:
            return "POSITIVE"
        if d["mean"] < -d["mde"] and d["hi"] < 0.0:
            return "NEGATIVE"
        return "INCONCLUSIVE"

    arms = ["N", "R", "R_fixed", "P", "P_shuffled", "W"]
    table: Dict[str, Any] = {"set": tag, "weight": weight,
                             "edge_beta": EDGE_BETA, "buckets": {},
                             "contrasts": {}, "controls": controls}
    for bucket in ("ambiguous", "unambiguous"):
        table["buckets"][bucket] = {a: cluster_boot(acc[a][bucket])
                                    for a in arms}
        if bucket == "ambiguous":
            table["part4_label"] = {
                "N": cluster_boot(acc["N_part4"][bucket]),
                "P": cluster_boot(acc["P_part4"][bucket]),
            }
        for a_, b_ in (("P", "N"), ("P", "P_shuffled"), ("P", "W"),
                       ("P", "R"), ("P", "R_fixed"), ("R_fixed", "R"),
                       ("R", "N")):
            d = paired_boot(acc[a_][bucket], acc[b_][bucket])
            table["contrasts"]["%s|%s-%s" % (bucket, a_, b_)] = dict(
                d, verdict=verdict(d))

    n_changed = len(flips["changed"])
    table["flips"] = {
        "changed": n_changed,
        "fixed": len(flips["fixed"]),
        "broken": len(flips["broken"]),
        "neutral": len(flips["neutral"]),
        "net": len(flips["fixed"]) - len(flips["broken"]),
    }

    if args.dump_nodes:
        with open(args.dump_nodes, "w") as fh:
            json.dump(records, fh)
        print("wrote %s (%d node records)" % (args.dump_nodes, len(records)))

    out_path = os.path.join(OUT, "arms_%s.json" % tag)
    with open(out_path, "w") as fh:
        json.dump(table, fh, indent=2, sort_keys=True, default=float)

    print("\n== %s" % tag.upper())
    for bucket in ("ambiguous", "unambiguous"):
        print("-- %s" % bucket)
        for a in arms:
            d = table["buckets"][bucket][a]
            print("   %-10s %.4f [%.4f, %.4f] MDE %.4f n=%d n_eff=%d"
                  % (a, d["mean"], d["lo"], d["hi"], d["mde"], d["n"], d["n_eff"]))
        for k in sorted(table["contrasts"]):
            if not k.startswith(bucket + "|"):
                continue
            d = table["contrasts"][k]
            print("   %-16s %+.4f [%.4f, %.4f] MDE %.4f  %s"
                  % (k.split("|")[1], d["mean"], d["lo"], d["hi"], d["mde"],
                     d["verdict"]))
    print("\nnet flips P vs N: changed %d  fixed %d  broken %d  net %+d"
          % (n_changed, len(flips["fixed"]), len(flips["broken"]),
             len(flips["fixed"]) - len(flips["broken"])))
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
