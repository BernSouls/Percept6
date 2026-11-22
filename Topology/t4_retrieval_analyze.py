"""Task 4 RETRIEVAL — Part 1 offline analysis (1a-1f).

Reads the component JSONL from t4_retrieval_log.py and produces:
  1a  clustered re-statement of the headline (MRR primary, top-1, AUC).
  1b  edge-only ranking under N's FIXED mapping (R, W, W-P); chance 0.125.
  1c  density-bias correlations (fresh, not assumed).
  1d  score variants V0/V1/V2 + edge-only / node-only anchors.
  1e  edge-gated use (V1 on low-N-margin queries, cut frozen on rank-0).
  1f  same-object gap (R-N on same-object vs other refs, two-way clustered).

All intervals are TWO-LEVEL clustered by OBJECT (objects, then views).
MDE = 2.8 * SE, SE = half-width / 1.96.

TIE RULE (pre-registered here, before any result was inspected): all three
read-outs are tie-aware.  For a query whose same-object ref has score s,
let k = #refs scoring strictly above s and t = #refs scoring equal to s:
    top1 = 1/t if k == 0 else 0                       (P of being first)
    rr   = (1/t) * sum_{j=1..t} 1/(k+j)               (E[RR] under a fair
                                                       random tie-break)
    auc  = mean over other refs of (1, 0.5, 0 on >, =, <)
When no tie straddles the correct ref these equal the usual values.
"""
from __future__ import annotations
import json, os, sys, math, random
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

HERE = RESULTS_DIR / "t4"
NBOOT = 2000
SEED = 42


# --------------------------------------------------------------------------
# tie-aware ranking read-outs
# --------------------------------------------------------------------------
def rank_stats(scores, correct_idx):
    """Tie-aware (top1, rr, auc) for one query. See module docstring."""
    s = float(scores[correct_idx])
    others = [float(x) for i, x in enumerate(scores) if i != correct_idx]
    k = sum(1 for o in others if o > s)
    t = 1 + sum(1 for o in others if o == s)
    top1 = (1.0 / t) if k == 0 else 0.0
    rr = (1.0 / t) * sum(1.0 / (k + j) for j in range(1, t + 1))
    if others:
        auc = float(np.mean([1.0 if s > o else (0.5 if s == o else 0.0)
                             for o in others]))
    else:
        auc = 0.0
    return top1, rr, auc


# --------------------------------------------------------------------------
# two-level clustered bootstrap by object
# --------------------------------------------------------------------------
def _group(pairs):
    d = defaultdict(list)
    for obj, v in pairs:
        d[obj].append(v)
    return d


def clustered_boot(pairs, nboot=NBOOT, seed=SEED):
    """Two-level bootstrap: resample objects, then views within object."""
    d = _group(pairs)
    objs = sorted(d.keys())
    if not objs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "hw": float("nan"), "se": float("nan"), "mde": float("nan"),
                "n": 0, "n_eff": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, len(objs), len(objs))
        vals = []
        for pi in pick:
            vs = d[objs[pi]]
            m = len(vs)
            idx = rng.integers(0, m, m)
            vals.extend(vs[j] for j in idx)
        boots[b] = float(np.mean(vals)) if vals else 0.0
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    hw = (hi - lo) / 2.0
    se = hw / 1.96
    return {"mean": float(np.mean([v for _, v in pairs])), "lo": lo, "hi": hi,
            "hw": hw, "se": se, "mde": 2.8 * se, "n": len(pairs),
            "n_eff": len(objs)}


def paired_boot(pa, pb, nboot=NBOOT, seed=SEED):
    """Paired two-level bootstrap of mean(a-b).

    pa, pb: lists of (obj, value) in the SAME order (one entry per query).
    Objects are resampled; views resampled within object; the SAME resample
    indices are used for both arms (the pairing is preserved).
    """
    da, db = _group(pa), _group(pb)
    objs = sorted(set(da) & set(db))
    if not objs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "hw": float("nan"), "se": float("nan"), "mde": float("nan"),
                "n": 0, "n_eff": 0}
    # align within object by position
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, len(objs), len(objs))
        vals = []
        for pi in pick:
            o = objs[pi]
            va, vb = da[o], db[o]
            m = min(len(va), len(vb))
            idx = rng.integers(0, m, m)
            vals.extend(va[j] - vb[j] for j in idx)
        boots[b] = float(np.mean(vals)) if vals else 0.0
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    hw = (hi - lo) / 2.0
    se = hw / 1.96
    diffs = [a - b for a, b in zip([v for _, v in pa], [v for _, v in pb])]
    return {"mean": float(np.mean(diffs)), "lo": lo, "hi": hi, "hw": hw,
            "se": se, "mde": 2.8 * se, "n": len(diffs), "n_eff": len(objs)}


def fmt(d, name="", nd=4):
    return ("%s %.4f [%.4f, %.4f]  MDE %.4f  n=%d n_eff=%d"
            % (name, d["mean"], d["lo"], d["hi"], d["mde"], d["n"], d["n_eff"]))


def verdict(d):
    """INCONCLUSIVE unless the interval excludes effects above MDE."""
    if d["lo"] > d["mde"]:
        return "POSITIVE (CI lower bound > MDE)"
    if d["hi"] < -d["mde"]:
        return "NEGATIVE (CI upper bound < -MDE)"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------
def load_jsonl(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main() -> int:
    from t4_retrieval_log import _set_flags
    _set_flags()
    recs = load_jsonl(HERE / "retrieval_components.jsonl")
    print("loaded %d records" % len(recs), flush=True)

    by = {}
    for r in recs:
        qk = (r["query"][0], r["query"][1], r["query"][2])
        rk = (r["ref"][0], r["ref"][1])
        by[(qk, rk, r["arm"])] = r
    queries = sorted({(r["query"][0], r["query"][1], r["query"][2]) for r in recs})
    refs = sorted({(r["ref"][0], r["ref"][1]) for r in recs})
    print("queries=%d refs=%d" % (len(queries), len(refs)), flush=True)

    def same_idx(qk):
        for i, rk in enumerate(refs):
            if rk[0] == qk[0] and rk[1] == qk[1]:
                return i
        raise KeyError(qk)

    def sel(r):
        return r["fwd"] if r["chosen"] == "fwd" else r["rev"]

    def cscore(r):
        return sel(r)["score"]

    out = {"n_queries": len(queries), "n_refs": len(refs),
           "n_objects": len({(q[0], q[1]) for q in queries}),
           "tie_rule": "tie-aware: top1=1/t, rr=E[RR] over tie set, auc 0.5",
           "n_boot": NBOOT, "seed": SEED}

    # ================================================================== 1a
    print("\n=== 1a  clustered re-statement (MRR primary) ===", flush=True)
    stats1a, per_obj = {}, {}
    for arm in ("R", "N", "W"):
        t1, rr, au = [], [], []
        for qk in queries:
            sc = [cscore(by[(qk, rk, arm)]) for rk in refs]
            a, b, c = rank_stats(sc, same_idx(qk))
            t1.append((qk[0], a)); rr.append((qk[0], b)); au.append((qk[0], c))
        stats1a[arm] = {"MRR": clustered_boot(rr), "top1": clustered_boot(t1),
                        "AUC": clustered_boot(au)}
        per_obj[arm] = {o: float(np.mean([v for ob, v in rr if ob == o]))
                        for o in sorted({q[0] for q in queries})}
        print("  " + fmt(stats1a[arm]["MRR"], "%-2s MRR " % arm), flush=True)
        print("     top1 " + fmt(stats1a[arm]["top1"], "", 4)[1:], flush=True)
        print("     AUC  " + fmt(stats1a[arm]["AUC"], "", 4)[1:], flush=True)

    for da_, db_ in (("R", "N"), ("R", "W"), ("N", "W")):
        pa, pb = [], []
        for qk in queries:
            si = same_idx(qk)
            pa.append((qk[0], rank_stats([cscore(by[(qk, rk, da_)]) for rk in refs], si)[1]))
            pb.append((qk[0], rank_stats([cscore(by[(qk, rk, db_)]) for rk in refs], si)[1]))
        d = paired_boot(pa, pb)
        stats1a["%s-%s_MRR" % (da_, db_)] = d
        print("  " + fmt(d, "%s-%s MRR diff" % (da_, db_)), flush=True)
        print("     verdict: " + verdict(d), flush=True)
    stats1a["per_object_MRR"] = per_obj
    out["1a"] = stats1a
    print("\n  per-object MRR (R / N / W):", flush=True)
    for o in sorted(per_obj["R"]):
        print("    %-7s %.4f  %.4f  %.4f" % (o, per_obj["R"][o],
                                             per_obj["N"][o], per_obj["W"][o]),
              flush=True)

    # ================================================================== 1b
    print("\n=== 1b  edge-only ranking under N's FIXED mapping ===", flush=True)
    print("  (mapping comes from N, whose graphs have no edges: the mapping\n"
          "   never saw edge evidence.  chance top-1 = 1/8 = 0.125)", flush=True)
    from t4_retrieval_log import (_load, _arm_graph, _arm_graph_wp,
                                  edge_term_under_mapping)
    from t4_retrieval_measure import REF_VIEW

    ref_graphs = {}
    for cls, obj in refs:
        g = _load(cls, obj, REF_VIEW)
        ref_graphs[(cls, obj)] = {
            "R": g,
            "W": _arm_graph(g, "W"),
            "W-P": _arm_graph_wp(g),
        }
    print("  ref graphs loaded", flush=True)

    vals = {a: [] for a in ("R", "W", "W-P")}       # (obj, edge_term)
    diag = {a: [] for a in ("R", "W", "W-P")}
    for qi, qk in enumerate(queries):
        qg = _load(qk[0], qk[1], qk[2])
        qarms = {"R": qg, "W": _arm_graph(qg, "W"), "W-P": _arm_graph_wp(qg)}
        si = same_idx(qk)
        for arm in ("R", "W", "W-P"):
            sc = []
            for rk in refs:
                mrec = by[(qk, rk, "N")]
                mapping = {int(u): int(v) for u, v in mrec["mapping"]}
                ec, et, nm, na, nb, ca, cb = edge_term_under_mapping(
                    qarms[arm].subgraph, ref_graphs[rk][arm].subgraph, mapping)
                sc.append(et)
                diag[arm].append(ec)
            a, b, c = rank_stats(sc, si)
            vals[arm].append((qk[0], (a, b, c)))
        if (qi + 1) % 16 == 0:
            print("    query %d/%d" % (qi + 1, len(queries)), flush=True)

    stats1b = {}
    for arm in ("R", "W", "W-P"):
        t1 = [(o, v[0]) for o, v in vals[arm]]
        rr = [(o, v[1]) for o, v in vals[arm]]
        au = [(o, v[2]) for o, v in vals[arm]]
        stats1b[arm] = {"top1": clustered_boot(t1), "MRR": clustered_boot(rr),
                        "AUC": clustered_boot(au),
                        "mean_edge_credit": float(np.mean(diag[arm]))}
        print("  " + fmt(stats1b[arm]["top1"], "%-4s edge-only top1" % arm),
              flush=True)
        print("       (" + fmt(stats1b[arm]["AUC"], "AUC", 4)[1:] +
              "  mean edge_credit %.3f)" % stats1b[arm]["mean_edge_credit"],
              flush=True)
        print("       vs chance 0.125: " +
              ("ABOVE" if stats1b[arm]["top1"]["lo"] > 0.125 else "NOT above"),
              flush=True)

    # arm-to-arm differences on edge-only top-1 / MRR
    for da_, db_ in (("R", "W"), ("R", "W-P"), ("W-P", "W")):
        for metric, mi in (("MRR", 1), ("top1", 0)):
            pa = [(o, vals[da_][i][1][mi]) for i, (o, _) in enumerate(vals[da_])]
            pb = [(o, vals[db_][i][1][mi]) for i, (o, _) in enumerate(vals[db_])]
            d = paired_boot(pa, pb)
            stats1b["%s-%s_%s" % (da_, db_, metric)] = d
            print("  " + fmt(d, "%s-%s edge-only %s" % (da_, db_, metric)),
                  flush=True)
    out["1b"] = stats1b

    # ================================================================== 1c
    print("\n=== 1c  density bias (measured fresh) ===", flush=True)
    stats1c = {}
    for arm in ("R", "N"):
        ref_s, ref_e = defaultdict(list), {}
        q_s, q_e = defaultdict(list), {}
        for r in recs:
            if r["arm"] != arm:
                continue
            qk = (r["query"][0], r["query"][1], r["query"][2])
            rk = (r["ref"][0], r["ref"][1])
            s = cscore(r)
            ref_s[rk].append(s); ref_e[rk] = r["n_edges_r"]
            q_s[qk].append(s); q_e[qk] = r["n_edges_q"]
        rm = [float(np.mean(ref_s[rk])) for rk in refs]
        re_ = [ref_e[rk] for rk in refs]
        qm = [float(np.mean(q_s[qk])) for qk in queries]
        qe = [q_e[qk] for qk in queries]
        # [NaN] the N arm has NO edges at all, so ref/query edge counts are
        # constant 0 and a correlation is undefined, not zero.
        if len(set(re_)) < 2 or len(set(rm)) < 2:
            c1 = None
        else:
            c1 = float(np.corrcoef(rm, re_)[0, 1])
        if len(set(qe)) < 2 or len(set(qm)) < 2:
            c2 = None
        else:
            c2 = float(np.corrcoef(qm, qe)[0, 1])
        # ref-level bootstrap CI on the ref correlation (n = 8 refs: report
        # the imprecision instead of hiding it)
        c1_boot = None
        if c1 is not None:
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(NBOOT):
                idx = rng.integers(0, len(refs), len(refs))
                xx = [re_[i] for i in idx]
                yy = [rm[i] for i in idx]
                if len(set(xx)) < 2 or len(set(yy)) < 2:
                    continue
                bs.append(float(np.corrcoef(yy, xx)[0, 1]))
            if bs:
                c1_boot = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
        stats1c[arm] = {"ref_mean_score_vs_ref_edges": c1,
                        "ref_corr_boot_CI": c1_boot,
                        "query_mean_score_vs_query_edges": c2,
                        "ref_edge_count_spread": [int(min(re_)), int(max(re_))],
                        "query_edge_count_spread": [int(min(qe)), int(max(qe))]}
        if c1 is None:
            print("  %-2s ref-corr UNDEFINED (arm has no edges: counts all 0)"
                  % arm, flush=True)
        else:
            print("  %-2s ref-corr %+.4f [%+.4f, %+.4f] (n=%d refs, edge counts "
                  "%d..%d)   query-corr %+.4f (n=%d, edge counts %d..%d)"
                  % (arm, c1, c1_boot[0], c1_boot[1], len(refs), min(re_),
                     max(re_), c2, len(queries), min(qe), max(qe)), flush=True)
    out["1c"] = stats1c
    _c = stats1c["R"]["ref_mean_score_vs_ref_edges"]
    trig = abs(_c) if _c is not None else float("nan")
    print("  F-S trigger (|corr| >= 0.5, pooled R): |%.4f| -> %s"
          % (trig, "FIRES" if trig >= 0.5 else "does not fire"), flush=True)
    out["1c"]["F_S_trigger"] = {"value": trig, "fired": bool(trig >= 0.5)}

    # ================================================================== 1d
    print("\n=== 1d  score variants (recombining logged terms) ===", flush=True)
    print("  NOTE: V1/V2 recombine the LOGGED node_term/edge_term. The\n"
          "  matcher's MATCH_SEPARATE_TERM_NORMALIZATION flag is read only in\n"
          "  the mode=='all' branch, so it is inert under the committed\n"
          "  MATCH_EDGE_DENOMINATOR_MODE='checkable'.", flush=True)

    def variant_scores(r, var):
        """Return score for one direction-pair under variant `var`."""
        if var == "V0":
            return max(r["fwd"]["score"], r["rev"]["score"])
        w = 0.3 if var == "V1" else 0.1
        return max(2.0 * ((1.0 - w) * d["node_term"] + w * d["edge_term"])
                   for d in (r["fwd"], r["rev"]))
    def anchor_scores(r, key):
        return max(r["fwd"][key], r["rev"][key])

    variants = ["V0", "V1", "V2", "edge-only", "node-only"]
    stats1d = {a: {} for a in ("R", "N")}
    for arm in ("R", "N"):
        for var in variants:
            t1, rr, au = [], [], []
            for qk in queries:
                if var in ("edge-only", "node-only"):
                    key = "edge_term" if var == "edge-only" else "node_term"
                    sc = [anchor_scores(by[(qk, rk, arm)], key) for rk in refs]
                else:
                    sc = [variant_scores(by[(qk, rk, arm)], var) for rk in refs]
                a, b, c = rank_stats(sc, same_idx(qk))
                t1.append((qk[0], a)); rr.append((qk[0], b)); au.append((qk[0], c))
            stats1d[arm][var] = {"MRR": clustered_boot(rr),
                                 "top1": clustered_boot(t1),
                                 "AUC": clustered_boot(au)}
        print("  -- arm %s" % arm, flush=True)
        for var in variants:
            print("     " + fmt(stats1d[arm][var]["MRR"], "%-9s MRR" % var),
                  flush=True)

    # R-N per variant, object-clustered, plus per-object sign counts
    print("  -- R-N per variant (MRR, object-clustered)", flush=True)
    signs = {}
    for var in variants:
        pa, pb = [], []
        so = defaultdict(int)
        for qk in queries:
            si = same_idx(qk)
            if var in ("edge-only", "node-only"):
                key = "edge_term" if var == "edge-only" else "node_term"
                sa = rank_stats([anchor_scores(by[(qk, rk, "R")], key) for rk in refs], si)[1]
                sb = rank_stats([anchor_scores(by[(qk, rk, "N")], key) for rk in refs], si)[1]
            else:
                sa = rank_stats([variant_scores(by[(qk, rk, "R")], var) for rk in refs], si)[1]
                sb = rank_stats([variant_scores(by[(qk, rk, "N")], var) for rk in refs], si)[1]
            pa.append((qk[0], sa)); pb.append((qk[0], sb))
            so[qk[0]] += 1 if sa > sb else (-1 if sa < sb else 0)
        d = paired_boot(pa, pb)
        npos = sum(1 for v in so.values() if v > 0)
        nneg = sum(1 for v in so.values() if v < 0)
        stats1d["R-N_" + var] = {"diff": d, "objects_positive": npos,
                                 "objects_negative": nneg,
                                 "per_object_sign": dict(so)}
        print("     " + fmt(d, "%-9s R-N" % var), flush=True)
        print("        objects R>N: %d/8   R<N: %d/8   verdict: %s"
              % (npos, nneg, verdict(d)), flush=True)
    out["1d"] = stats1d

    # ================================================================== 1e
    print("\n=== 1e  edge-gated use (fixed rule, no fitting) ===", flush=True)
    nsc = {}
    for qk in queries:
        nsc[qk] = sorted([cscore(by[(qk, rk, "N")]) for rk in refs], reverse=True)
    margins = {qk: nsc[qk][0] - nsc[qk][1] for qk in queries}
    allm = np.array([margins[qk] for qk in queries])
    cut = float(np.quantile(allm, 1.0 / 3.0))
    mlo, mhi = float(np.quantile(allm, 1/3)), float(np.quantile(allm, 2/3))
    print("  N margin: min %.4f  p33 %.4f  p67 %.4f  max %.4f"
          % (allm.min(), mlo, mhi, allm.max()), flush=True)
    print("  frozen cut (lowest tercile, rank-0 objects only, n=%d) = %.6f"
          % (len(allm), cut), flush=True)

    gt1, grr = [], []
    pnw = {0: [], 1: [], 2: []}
    n_below = 0
    for qk in queries:
        si = same_idx(qk)
        mg = margins[qk]
        terc = 0 if mg < mlo else (1 if mg < mhi else 2)
        n_wrong = 1.0 - rank_stats([cscore(by[(qk, rk, "N")]) for rk in refs], si)[0]
        pnw[terc].append(n_wrong)
        if mg < cut:
            n_below += 1
            gsc = [variant_scores(by[(qk, rk, "R")], "V1") for rk in refs]
        else:
            gsc = [cscore(by[(qk, rk, "N")]) for rk in refs]
        a, b, _ = rank_stats(gsc, si)
        gt1.append((qk[0], a)); grr.append((qk[0], b))

    # baselines for comparison
    nt1 = [(qk[0], rank_stats([cscore(by[(qk, rk, "N")]) for rk in refs],
                              same_idx(qk))[0]) for qk in queries]
    nrr = [(qk[0], rank_stats([cscore(by[(qk, rk, "N")]) for rk in refs],
                              same_idx(qk))[1]) for qk in queries]

    v1t1 = [(qk[0], rank_stats([variant_scores(by[(qk, rk, "R")], "V1")
                                for rk in refs], same_idx(qk))[0])
            for qk in queries]
    v1rr = [(qk[0], rank_stats([variant_scores(by[(qk, rk, "R")], "V1")
                                for rk in refs], same_idx(qk))[1])
            for qk in queries]
    stats1e = {
        "cut": cut, "margin_p33": mlo, "margin_p67": mhi,
        "n_below_cut": n_below,
        "alwaysV1R_top1": clustered_boot(v1t1),
        "alwaysV1R_MRR": clustered_boot(v1rr),
        "gated_minus_alwaysV1R_MRR": paired_boot(grr, v1rr),
        "gated_top1": clustered_boot(gt1), "gated_MRR": clustered_boot(grr),
        "N_top1": clustered_boot(nt1), "N_MRR": clustered_boot(nrr),
        "gated_minus_N_top1": paired_boot(gt1, nt1),
        "gated_minus_N_MRR": paired_boot(grr, nrr),
        "P_N_wrong_by_margin_tercile": {
            "0_shrunk": float(np.mean(pnw[0])),
            "1_middle": float(np.mean(pnw[1])),
            "2_wide": float(np.mean(pnw[2]))},
        "n_per_tercile": {str(k): len(v) for k, v in pnw.items()},
    }
    print("  " + fmt(stats1e["gated_MRR"], "gated MRR"), flush=True)
    print("  " + fmt(stats1e["N_MRR"], "N     MRR"), flush=True)
    print("  " + fmt(stats1e["gated_minus_N_MRR"], "gated-N MRR"), flush=True)
    print("  " + fmt(stats1e["gated_top1"], "gated top1"), flush=True)
    print("  " + fmt(stats1e["N_top1"], "N     top1"), flush=True)
    print("  " + fmt(stats1e["alwaysV1R_MRR"], "alwaysV1R MRR"), flush=True)
    print("  " + fmt(stats1e["gated_minus_alwaysV1R_MRR"],
                      "gated-V1R MRR "), flush=True)
    print("  P(N wrong) by margin tercile: shrunk %.3f  middle %.3f  wide %.3f"
          % (stats1e["P_N_wrong_by_margin_tercile"]["0_shrunk"],
             stats1e["P_N_wrong_by_margin_tercile"]["1_middle"],
             stats1e["P_N_wrong_by_margin_tercile"]["2_wide"]), flush=True)
    p = stats1e["P_N_wrong_by_margin_tercile"]
    rising = p["0_shrunk"] > p["1_middle"] > p["2_wide"]
    print("  P(N wrong) falls as margin widens (i.e. rises as margin shrinks): %s"
          % rising, flush=True)
    out["1e"] = stats1e

    # ================================================================== 1f
    print("\n=== 1f  same-object gap (two-way clustered by query-object and ref) ===",
          flush=True)
    rows = []
    for qk in queries:
        for rk in refs:
            delta = cscore(by[(qk, rk, "R")]) - cscore(by[(qk, rk, "N")])
            same = 1.0 if (rk[0] == qk[0] and rk[1] == qk[1]) else 0.0
            rows.append(((qk[0], qk[1]), rk, delta, same))
    nobs = len(rows)
    qkeys = sorted({r[0] for r in rows})
    qidx = {k: i for i, k in enumerate(qkeys)}
    ridx = {k: i for i, k in enumerate(sorted({r[1] for r in rows}))}

    def design(rws):
        X = np.zeros((len(rws), 1 + 1 + 1 + 6))   # const, same, delta*... no
        X = np.zeros((len(rws), 2 + 6))           # const, same, 7 ref dummies -> 6
        y = np.zeros(len(rws))
        nref = len(ridx)
        X = np.zeros((len(rws), 2 + (nref - 1)))
        for i, (qk, rk, dlt, same) in enumerate(rws):
            X[i, 0] = 1.0
            X[i, 1] = same
            j = ridx[rk]
            if j > 0:
                X[i, 1 + j] = 1.0
            y[i] = dlt
        return X, y

    X, y = design(rows)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_same = float(beta[1])
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)

    # ---- two-way cluster-robust VCV (query-object x ref), CGM correction
    qg, rg = defaultdict(list), defaultdict(list)
    for i, (qk, rk, _d, _s) in enumerate(rows):
        qg[qk].append(i); rg[rk].append(i)
    def meat(groups):
        M = np.zeros((X.shape[1], X.shape[1]))
        for g in groups.values():
            gi = np.array(g)
            sc = X[gi].T @ resid[gi]
            M += np.outer(sc, sc)
        return M
    Gq, Gr = len(qg), len(rg)
    n, k = X.shape
    cq = Gq / (Gq - 1) * (n - 1) / (n - k)
    cr = Gr / (Gr - 1) * (n - 1) / (n - k)
    # intersection clusters
    inter = defaultdict(list)
    for i, (qk, rk, _d, _s) in enumerate(rows):
        inter[(qk, rk)].append(i)
    Gi = len(inter)
    ci = Gi / (Gi - 1) * (n - 1) / (n - k)
    V = XtX_inv @ (cq * meat(qg) + cr * meat(rg) - ci * meat(inter)) @ XtX_inv
    se_crve = float(math.sqrt(max(V[1, 1], 0.0)))

    # ---- two-way cluster BOOTSTRAP (primary): resample objects and refs
    rng = np.random.default_rng(SEED)
    qlist = sorted(qg.keys()); rlist = sorted(rg.keys())
    boots = []
    rowmap = {(i_q, i_r): i for i, (qk, rk, _d, _s) in enumerate(rows)
              for i_q in [qk] for i_r in [rk]}
    for _ in range(NBOOT):
        qs = [qlist[j] for j in rng.integers(0, len(qlist), len(qlist))]
        rs = [rlist[j] for j in rng.integers(0, len(rlist), len(rlist))]
        sub = [rows[rowmap[(a, b)]] for a in qs for b in rs]
        Xb, yb = design(sub)
        try:
            bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
            boots.append(float(bb[1]))
        except Exception:
            pass
    boots = np.array(boots)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    hw = (hi - lo) / 2.0
    stats1f = {
        "beta_same_object": b_same,
        "ci_lo": lo, "ci_hi": hi, "hw": hw, "se_boot": hw / 1.96,
        "mde": 2.8 * (hw / 1.96),
        "se_crve_twoway": se_crve,
        "n_obs": nobs, "n_query_objects": Gq, "n_refs": Gr,
        "n_boot": len(boots), "ref_fixed_effects": True,
        "clustering": "two-way (query-object, ref)",
    }
    print("  beta(same-object) = %.4f  boot CI [%.4f, %.4f]  MDE %.4f"
          % (b_same, lo, hi, stats1f["mde"]), flush=True)
    print("  two-way CRVE se %.4f  (n_obs=%d, %d query-objects x %d refs)"
          % (se_crve, nobs, Gq, Gr), flush=True)
    print("  verdict: " + verdict({"lo": lo, "hi": hi, "mde": stats1f["mde"]}),
          flush=True)
    # descriptive split
    same_d = [r[2] for r in rows if r[3] == 1.0]
    oth_d = [r[2] for r in rows if r[3] == 0.0]
    stats1f["descriptive"] = {
        "mean_delta_same_object": float(np.mean(same_d)),
        "mean_delta_other_ref": float(np.mean(oth_d)),
        "raw_gap": float(np.mean(same_d) - np.mean(oth_d)),
        "n_same": len(same_d), "n_other": len(oth_d)}
    print("  descriptive: same-object delta %.4f (n=%d)  other %.4f (n=%d)  gap %.4f"
          % (stats1f["descriptive"]["mean_delta_same_object"], len(same_d),
             stats1f["descriptive"]["mean_delta_other_ref"], len(oth_d),
             stats1f["descriptive"]["raw_gap"]), flush=True)
    out["1f"] = stats1f

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, (np.floating,)):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    write_json(HERE / "retrieval_part1_analysis.json", _clean(out))
    print("\n=== Part 1 analysis complete -> retrieval_part1_analysis.json ===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
