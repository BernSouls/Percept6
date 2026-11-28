"""M35: the object-level bag -- ruler, replication of the VERIFIED numbers, and the class test.

Thin harness by the project's rule: the descriptor lives in `matching/placement_bag.py` and nothing
here defines a score the shipped path could not compute.

    python Topology/m35_bag.py --edges        # build the bin edges from the reference view (once)
    python Topology/m35_bag.py --replicate    # item 1: the instance task, per block group
    python Topology/m35_bag.py --class        # item 2: the pre-registered arms and the claims
    python Topology/m35_bag.py --door         # item 3: the registered door candidates
    python Topology/m35_bag.py --classout     # item 4: recognise_class against M23's 0.500
    python Topology/m35_bag.py --fragment     # item 5: does the fragment channel add on top?

Every rate carries n, chance and an object-clustered CI (10,000 resamples, seed 20261118).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
OUTDIR = os.path.join(ROOT, "Topology", "results", "placement")
REF = "000-000"
VIEWS = ("022-000", "035-045", "045-180")
HARD = ("035-045", "045-180")
S = 8
SEED = 20261118
RESAMPLES = 10000


def _out(name: str) -> str:
    return os.path.join(OUTDIR, "m35_%s.json" % name)


def _load(views: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    from matching import placement_records as PR
    return PR.load_cells(RECORDS, list(views))


def _feats(keys: Sequence[str], cells: Dict[Tuple[str, str], Any], view: str,
           edges: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    from matching import placement_bag as BAG
    return [BAG.describe(cells[(k, view)], edges) for k in keys]


def _boot(diff: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> List[float]:
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(resamples, n))
    m = diff[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _ranks(M: np.ndarray) -> np.ndarray:
    """Rank of the diagonal (truth) per query row, ties by index."""
    n = M.shape[0]
    order = np.argsort(-M, axis=1, kind="stable")
    return np.asarray([int(np.nonzero(order[i] == i)[0][0]) for i in range(n)])


def _rate(ranks: np.ndarray, idx: Sequence[int], k: int) -> float:
    return float(np.mean([ranks[i] < k for i in idx])) if len(idx) else float("nan")


def _split() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(REC.objects()):
        out.setdefault(str(o["split"]), []).append(i)
    return out


# --------------------------------------------------------------------------- item 1: edges
def edges_block() -> Dict[str, Any]:
    from matching import placement_bag as BAG
    objs = REC.objects()
    t0 = time.time()
    cells = _load([REF])
    pool = [cells[(o["key"], REF)] for o in objs]
    edges = BAG.edges_from(pool)
    prov = {"mission": "M35 item 1", "built": time.strftime("%Y-%m-%d"),
            "source": os.path.basename(RECORDS), "view": REF, "objects": len(pool),
            "placed": int(sum(len(r["places"]) for r in pool)),
            "rule": "per-block quantiles of the REFERENCE view's own values, label-free",
            "seconds": round(time.time() - t0, 1)}
    p = BAG.save_edges(edges, provenance=prov)
    print(json.dumps({"file": os.path.relpath(p, ROOT), "objects": prov["objects"],
                      "places": prov["placed"], "seconds": prov["seconds"]}, indent=1))
    return edges


# --------------------------------------------------------------------------- item 1: replicate
def replicate() -> Dict[str, Any]:
    from matching import placement_bag as BAG
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    split = _split()
    fit = split.get("FIT", [])
    edges = BAG.load_edges()
    cells = _load([REF, *VIEWS])
    ref = _feats(keys, cells, REF, edges)
    out: Dict[str, Any] = {"objects": len(keys), "chance_top1": 1.0 / len(keys),
                           "chance_recall8": S / len(keys), "views": {}}
    for v in VIEWS:
        q = _feats(keys, cells, v, edges)
        per_arm: Dict[str, Any] = {}
        for arm_name in ("COLOUR", "SHAPE", "LAYOUT", "CFG", "GEOMETRY", "COLOUR-FREE", "ALL"):
            _w = BAG.WEIGHTING.get(arm_name, BAG.DEFAULT_WEIGHTING)
            M = np.asarray([[float(np.mean([q[i][b] @ ref[j][b] for b in BAG.blocks_of(
                                 BAG.ARMS[arm_name])]) if _w == "block" else
                             float(np.mean([np.mean([q[i][b] @ ref[j][b] for b in BAG.GROUPS[g]])
                                            for g in BAG.ARMS[arm_name]])))
                             for j in range(len(keys))] for i in range(len(keys))])
            r = _ranks(M)
            per_arm[arm_name] = {"top1": float(np.mean(r == 0)), "recall8": float(np.mean(r < S)),
                                 "top1_n": int((r == 0).sum()), "recall8_n": int((r < S).sum()),
                                 "median_rank": float(np.median(r)),
                                 "top1_fit": _rate(r, fit, 1), "recall8_fit": _rate(r, fit, S)}
            print("  %-6s %-12s top1 %2d/%d  r@8 %2d/%d  (FIT top1 %.3f)"
                  % (v, arm_name, per_arm[arm_name]["top1_n"], len(keys),
                     per_arm[arm_name]["recall8_n"], len(keys),
                     per_arm[arm_name]["top1_fit"]), flush=True)
        # the VERIFIED pairs: colour, shape, colour+shape at equal weight
        Mc = np.asarray([[float(np.mean([np.mean([q[i][b] @ ref[j][b] for b in BAG.GROUPS[g]])
                                        for g in ("COLOUR",)])) for j in range(len(keys))]
                         for i in range(len(keys))])
        Ms = np.asarray([[float(np.mean([np.mean([q[i][b] @ ref[j][b] for b in BAG.GROUPS[g]])
                                        for g in ("SHAPE",)])) for j in range(len(keys))]
                         for i in range(len(keys))])
        eq = 0.5 * (Mc + Ms)
        per_arm["COLOUR+SHAPE_equal"] = {"top1_n": int((_ranks(eq) == 0).sum()),
                                        "recall8_n": int((_ranks(eq) < S).sum())}
        out["views"][v] = per_arm
    with open(_out("bag"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 2: the class test
def _class_auc(M: np.ndarray, cls: Sequence[str]) -> Tuple[float, np.ndarray]:
    """Per query (own object excluded): the pair-wise AUC of same-class over other-class.

    A_i = P(sim(query, same-class) > sim(query, other-class)) over every (same, other) pair, in the
    Mann-Whitney form.  This is NOT "is the nearest neighbour same-class" -- that is `_nn_class`,
    a different and much weaker question (chance 9/79 rather than 0.5).  Chance here is 0.5.
    """
    n = M.shape[0]
    hits = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        same = np.asarray([j != i and cls[j] == cls[i] for j in range(n)])
        other = np.asarray([cls[j] != cls[i] for j in range(n)])
        if not same.any() or not other.any():
            continue
        v = M[i]
        r = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64) + 1.0
        n_o = float(other.sum())
        hits[i] = (r[same].mean() - (same.sum() + 1.0) / 2.0) / n_o
    ok = np.isfinite(hits)
    return float(hits[ok].mean()), hits


def _nn_class(M: np.ndarray, cls: Sequence[str]) -> Tuple[float, np.ndarray]:
    """Nearest-neighbour CLASS top-1, own object excluded.  Chance 9/79."""
    n = M.shape[0]
    hits = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = M[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        hits[i] = 1.0 if cls[j] == cls[i] else 0.0
    return float(hits.mean()), hits


def class_test() -> Dict[str, Any]:
    from matching import placement_bag as BAG
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls = [o["cls"] for o in objs]
    split = _split()
    fit = np.asarray(split.get("FIT", []))
    conf = np.asarray(split.get("DEV", []) + split.get("FRESH", []))
    edges = BAG.load_edges()
    cells = _load([REF, *VIEWS])
    ref = _feats(keys, cells, REF, edges)
    out: Dict[str, Any] = {"objects": len(keys), "chance_auc": 0.5,
                           "chance_nn_class": 9.0 / 79.0, "views": {}}
    for v in VIEWS:
        q = _feats(keys, cells, v, edges)
        per_arm: Dict[str, Any] = {}
        for arm_name, arm in BAG.ARMS.items():
            _w = BAG.WEIGHTING.get(arm_name, BAG.DEFAULT_WEIGHTING)
            M = np.asarray([[float(np.mean([q[i][b] @ ref[j][b] for b in BAG.blocks_of(arm)]))
                             if _w == "block" else
                             float(np.mean([np.mean([q[i][b] @ ref[j][b] for b in BAG.GROUPS[g]])
                                            for g in arm]))
                             for j in range(len(keys))] for i in range(len(keys))])
            auc, hits = _class_auc(M, cls)
            nn, nnh = _nn_class(M, cls)
            per_arm[arm_name] = {
                "auc": auc, "auc_fit": float(hits[fit].mean()),
                "auc_conf": float(hits[conf].mean()) if conf.size else float("nan"),
                "auc_ci": _boot(hits[np.isfinite(hits)]),
                "nn_class": nn, "nn_class_n": int(nnh.sum()),
                "per_class": {c: float(np.nanmean(hits[np.asarray(cls) == c]))
                              for c in sorted(set(cls))},
            }
            print("  %-6s %-12s AUC %.3f CI%s  nn-class %.3f (%d/%d)"
                  % (v, arm_name, auc, np.round(per_arm[arm_name]["auc_ci"], 3), nn,
                     int(nnh.sum()), len(keys)), flush=True)
        # (c) the per-class winning arm: the arm with the highest same-class AUC for that class
        classes = sorted(set(cls))
        win = {}
        for c in classes:
            win[c] = max(((nm, per_arm[nm]["per_class"][c]) for nm in BAG.ARMS),
                         key=lambda kv: kv[1])
        per_arm["_per_class_winner"] = {c: {"arm": w[0], "auc": w[1],
                                           "colour": per_arm["COLOUR"]["per_class"][c],
                                           "geometry": per_arm["GEOMETRY"]["per_class"][c]}
                                        for c, w in win.items()}
        # the claims, computed here so the report quotes a fixed rule
        a = per_arm
        per_arm["_claims"] = {
            "geometry_clusters": {"auci_lo": a["GEOMETRY"]["auc_ci"][0],
                                  "holds": bool(a["GEOMETRY"]["auc_ci"][0] > 0.55)},
            "complementary": {"all_minus_colour": a["ALL"]["auc"] - a["COLOUR"]["auc"],
                              "note": "the paired object-clustered CI is filled in below"},
        }
        out["views"][v] = per_arm
    # paired ALL-COLOUR difference per view, object-clustered
    for v in VIEWS:
        q = _feats(keys, cells, v, edges)
        hi = {}
        for arm_name in ("COLOUR", "ALL", "GEOMETRY"):
            arm = BAG.ARMS[arm_name]
            M = np.asarray([[float(np.mean([q[i][b] @ ref[j][b] for b in BAG.blocks_of(arm)]))
                             if BAG.WEIGHTING.get(arm_name) == "block" else
                             float(np.mean([np.mean([q[i][b] @ ref[j][b] for b in BAG.GROUPS[g]])
                                            for g in arm]))
                             for j in range(len(keys))] for i in range(len(keys))])
            _auc, hi[arm_name] = _class_auc(M, cls)
        ok = np.isfinite(hi["ALL"]) & np.isfinite(hi["COLOUR"]) & np.isfinite(hi["GEOMETRY"])
        d_ac, d_gc = hi["ALL"][ok] - hi["COLOUR"][ok], hi["GEOMETRY"][ok] - hi["COLOUR"][ok]
        out["views"][v]["_claims"]["complementary"] = {
            "all_minus_colour": float(d_ac.mean()), "ci": _boot(d_ac),
            "holds": bool(_boot(d_ac)[0] > 0.0),
            "geometry_minus_colour": float(d_gc.mean()), "geometry_ci": _boot(d_gc)}
        out["views"][v]["_claims"]["geometry_clusters"] = {
            "auci_lo": out["views"][v]["GEOMETRY"]["auc_ci"][0],
            "holds": bool(out["views"][v]["GEOMETRY"]["auc_ci"][0] > 0.55),
            "withdrawn_from_precheck": True}
    with open(_out("class"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 3: the door
FIT_VIEWS = ("035-045", "045-180")
K_RRF = 60
W_COLOUR_PLACE = 0.5
W_FRAGMENT = 0.5
BAR_DOOR = 0.10
CONFIRM_TOL = -0.025


def _conf() -> List[int]:
    sp = _split()
    return sp.get("DEV", []) + sp.get("FRESH", [])


def _bag_colour(view: str, edges: Dict[str, Any]) -> np.ndarray:
    from matching import placement_bag as BAG
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cells = _load([REF, view])
    gal = [BAG.describe(cells[(k, REF)], edges) for k in keys]
    q = [BAG.describe(cells[(k, view)], edges) for k in keys]
    return np.asarray([[float(np.mean([q[i][b] @ gal[j][b] for b in BAG.GROUPS["COLOUR"]]))
                        for j in range(len(keys))] for i in range(len(keys))])


def door_channels(view: str, edges: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Every channel the door can read, on the SAME cells: gallery = -000-000, queries = `view`."""
    from matching import placement_fuse as F
    from matching import placement_records as PR
    from matching import placement_anchors as PA
    from matching import placement_fragment as PF
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cells = PR.load_cells(RECORDS, [REF, view])
    gal = [{"places": cells[(k, REF)]["places"]} for k in keys]
    qs = [{"places": cells[(k, view)]["places"]} for k in keys]
    z = np.load(os.path.join(OUTDIR, "m32_colour_channels.npz"))
    an = PA.load()

    def A(k: str, v: str) -> Any:
        return an["%s__%s__contour_fragment" % (k.replace("/", "-"), v)]
    gl = [{"places": cells[(k, REF)]["places"]} for k in keys]
    ga = [A(k, REF) for k in keys]
    frag = np.vstack([PF.channel(gl, cells[(k, view)]["places"], ga, A(k, view))
                      for k in keys])
    return {"bag_colour": _bag_colour(view, edges),
            "counted": z["counted|%s" % view], "appearance": z["appearance|%s" % view],
            "colour": np.asarray(F.colour_channel(gal, qs), dtype=np.float64),
            "fragment": frag}


def _rrf(mats: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    from matching import placement_fuse as F
    return np.asarray(F.rrf_weighted(list(mats), list(weights), k=K_RRF), dtype=np.float64)


def door() -> Dict[str, Any]:
    """The three registered candidates, chosen on FIT by recall@8 then top-1, ties -> fewer channels."""
    from matching import placement_bag as BAG
    edges = BAG.load_edges()
    objs = REC.objects()
    split = _split()
    fit = np.asarray(split["FIT"])
    conf = np.asarray(_conf())
    out: Dict[str, Any] = {"candidates": ["bag_colour", "bag_colour+fused3", "fused3",
                                          "fused3_nofragment", "bag_colour+fused3_nofragment"],
                           "rrf_k": K_RRF, "weights": {"counted": 1.0, "appearance": 1.0,
                                                       "colour": W_COLOUR_PLACE,
                                                       "fragment": W_FRAGMENT},
                           "bar": BAR_DOOR, "confirm_tol": CONFIRM_TOL, "views": {}}
    mats: Dict[str, Dict[str, np.ndarray]] = {}
    for v in FIT_VIEWS:
        ch = door_channels(v, edges)
        M = {
            "bag_colour": ch["bag_colour"],
            "fused3": _rrf([ch["counted"], ch["appearance"], ch["colour"], ch["fragment"]],
                           [1.0, 1.0, W_COLOUR_PLACE, W_FRAGMENT]),
            "fused3_nofragment": _rrf([ch["counted"], ch["appearance"], ch["colour"]],
                                      [1.0, 1.0, W_COLOUR_PLACE]),
            "bag_colour+fused3": _rrf([ch["bag_colour"], ch["counted"], ch["appearance"],
                                       ch["colour"], ch["fragment"]],
                                      [1.0, 1.0, 1.0, W_COLOUR_PLACE, W_FRAGMENT]),
            "bag_colour+fused3_nofragment": _rrf([ch["bag_colour"], ch["counted"],
                                                  ch["appearance"], ch["colour"]],
                                                 [1.0, 1.0, 1.0, W_COLOUR_PLACE]),
        }
        mats[v] = M
        per: Dict[str, Any] = {}
        for name, mtx in M.items():
            r = _ranks(mtx)
            per[name] = {"recall8_fit": _rate(r, fit, S), "top1_fit": _rate(r, fit, 1),
                         "recall8_conf": _rate(r, conf, S), "top1_conf": _rate(r, conf, 1),
                         "recall8_all80": float(np.mean(r < S)), "top1_all80": float(np.mean(r == 0)),
                         "recall8_n_fit": int(sum(r[i] < S for i in fit)),
                         "top1_n_fit": int(sum(r[i] == 0 for i in fit))}
            print("  %-6s %-28s r@8 FIT %.3f top1 %.3f | conf r@8 %.3f"
                  % (v, name, per[name]["recall8_fit"], per[name]["top1_fit"],
                     per[name]["recall8_conf"]), flush=True)
        out["views"][v] = per
    # the choice: FIT recall@8, then top-1, ties -> fewer channels (candidate list is in that order)
    order = ["bag_colour", "fused3_nofragment", "bag_colour+fused3", "fused3",
             "bag_colour+fused3_nofragment"]
    nch = {"bag_colour": 1, "fused3_nofragment": 3, "bag_colour+fused3": 5, "fused3": 4,
           "bag_colour+fused3_nofragment": 4}
    rank = sorted(order, key=lambda n: (-np.mean([out["views"][v][n]["recall8_fit"]
                                                  for v in FIT_VIEWS]),
                                        -np.mean([out["views"][v][n]["top1_fit"]
                                                  for v in FIT_VIEWS]),
                                        nch[n]))
    choice = rank[0]
    base = "fused3"
    d_fit = np.asarray([out["views"][v][choice]["recall8_fit"] - out["views"][v][base]["recall8_fit"]
                        for v in FIT_VIEWS])
    d_conf = np.asarray([out["views"][v][choice]["recall8_conf"]
                         - out["views"][v][base]["recall8_conf"] for v in FIT_VIEWS])
    # paired object-level difference: per-object indicators, both views pooled
    dv: List[np.ndarray] = []
    for v in FIT_VIEWS:
        rc, rb = _ranks(mats[v][choice]), _ranks(mats[v][base])
        dv.append(((rc < S).astype(float) - (rb < S).astype(float))[fit])
    d = np.concatenate(dv)
    out["choice"] = {"chosen": choice, "order": rank, "mean_fit_gain": float(d_fit.mean()),
                     "mean_conf_gain": float(d_conf.mean()), "paired_gain": float(d.mean()),
                     "paired_ci": _boot(d), "bar": BAR_DOOR,
                     "passes_bar": bool(float(d_fit.mean()) >= BAR_DOOR),
                     "confirm_ok": bool(float(d_conf.mean()) >= CONFIRM_TOL),
                     "n_fit": int(fit.size), "n_conf": int(conf.size), "chance": S / 80.0}
    # item 5 answers "does the fragment channel add on top of the CHOSEN door?" -- see `fragment()`.
    print(json.dumps(out["choice"], indent=1))
    with open(_out("door"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 6: the matcher
PAIRS = os.path.join(OUTDIR, "m35_pairs.jsonl")


def _pair_keys() -> List[Tuple[int, str, int]]:
    """(query index, query view, candidate index) for the FIT shortlists, door order first."""
    from matching import placement_bag as BAG
    edges = BAG.load_edges()
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    split = _split()
    fit = split["FIT"]
    out: List[Tuple[int, str, int]] = []
    for v in FIT_VIEWS:
        M = _bag_colour(v, edges)
        for i in fit:
            order = list(np.argsort(-M[i], kind="stable")[:S])
            for j in order:
                out.append((int(i), v, int(j)))
    return out


def matcher_pairs() -> Dict[str, Any]:
    """Score the FIT shortlists' pairs one at a time, resumable through `m35_pairs.jsonl`."""
    from matching import placement_recognise as R
    from matching import placement_bundle as PB
    objs = REC.objects()
    b = PB.PlacementBundle.load()
    w = float(getattr(b, "matcher_weight", 0.0))
    todo = _pair_keys()
    done: Dict[str, float] = {}
    if os.path.exists(PAIRS):
        with open(PAIRS) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[row["k"]] = float(row["s"])
    left = [t for t in todo if "%d|%s|%d" % t not in done]
    print("pairs: %d total, %d cached, %d to run, ~%.1f h at 23 s/pair"
          % (len(todo), len(todo) - len(left), len(left), len(left) * 23.0 / 3600.0), flush=True)
    t0 = time.time()
    with open(PAIRS, "a") as fh:
        for n, (i, v, j) in enumerate(left):
            q = REC.path_of(objs[i], v)
            r = REC.path_of(objs[j], REF)
            t1 = time.time()
            sc = R._score_one((q, r, w))
            if sc is None:
                continue
            fh.write(json.dumps({"k": "%d|%s|%d" % (i, v, j), "s": float(sc),
                                 "sec": round(time.time() - t1, 1)}) + "\n")
            fh.flush()
            if (n + 1) % 10 == 0:
                el = time.time() - t0
                print("  %d/%d, %.1f s/pair, %.1f h left"
                      % (n + 1, len(left), el / (n + 1),
                         (len(left) - n - 1) * el / (n + 1) / 3600.0), flush=True)
    return {"total": len(todo), "ran": len(left)}


def matcher_report() -> Dict[str, Any]:
    """The three orders on the same candidates: door, matcher, RRF(equal).  FIT then confirmation."""
    from matching import placement_bag as BAG
    from matching import placement_fuse as F
    edges = BAG.load_edges()
    objs = REC.objects()
    split = _split()
    fit = list(split["FIT"])
    conf = list(_conf())
    scores: Dict[str, float] = {}
    with open(PAIRS) as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["k"]] = float(row["s"])
    outv: Dict[str, Any] = {}
    for v in FIT_VIEWS:
        M = _bag_colour(v, edges)
        door_t1, mat_t1, rrf_t1, door_r8, mat_r8, rrf_r8 = (np.zeros(80), np.zeros(80),
                                                            np.zeros(80), np.zeros(80),
                                                            np.zeros(80), np.zeros(80))
        cov = np.zeros(80)
        for i in range(80):
            order = list(np.argsort(-M[i], kind="stable")[:S])
            have = [(j, scores["%d|%s|%d" % (i, v, j)]) for j in order
                    if "%d|%s|%d" % (i, v, j) in scores]
            cov[i] = len(have) == S
            door_rank = order.index(i) if i in order else 99
            door_t1[i] = float(door_rank == 0)
            door_r8[i] = float(door_rank != 99)
            if not have:
                mat_t1[i] = float(door_rank == 0)
                rrf_t1[i] = float(door_t1[i])
                mat_r8[i] = float(door_r8[i])
                rrf_r8[i] = float(door_r8[i])
                continue
            mat_order = [j for j, _ in sorted(have, key=lambda t: (-t[1], order.index(t[0])))]
            rest = [j for j in range(80) if j not in mat_order]
            mat_full = mat_order + rest
            # RRF(equal) over the door's own rank and the matcher's rank, on these candidates
            door_rank_map = {j: r for r, j in enumerate(order)}
            mat_rank_map = {j: r for r, j in enumerate(mat_order)}
            rrf = {}
            for j in set(list(door_rank_map) + list(mat_rank_map)):
                rd = door_rank_map.get(j, S)
                rm = mat_rank_map.get(j, S)
                rrf[j] = 1.0 / (K_RRF + rd + 1) + 1.0 / (K_RRF + rm + 1)
            rrf_order = sorted(rrf, key=lambda j: (-rrf[j], j)) + [j for j in range(80)
                                                                   if j not in rrf]
            mat_t1[i] = float(mat_full[0] == i)
            rrf_t1[i] = float(rrf_order[0] == i)
            mat_r8[i] = float(i in mat_full[:S])
            rrf_r8[i] = float(i in rrf_order[:S])
        outv[v] = {"covered": int(cov.sum()), "n": 80,
                   "door_top1_fit": float(door_t1[fit].mean()), "matcher_top1_fit": float(mat_t1[fit].mean()),
                   "rrf_top1_fit": float(rrf_t1[fit].mean()),
                   "door_top1_conf": float(door_t1[conf].mean()) if conf else float("nan"),
                   "matcher_top1_conf": float(mat_t1[conf].mean()) if conf else float("nan"),
                   "rrf_top1_conf": float(rrf_t1[conf].mean()) if conf else float("nan"),
                   "door_top1_all": float(door_t1.mean()), "matcher_top1_all": float(mat_t1.mean()),
                   "rrf_top1_all": float(rrf_t1.mean()),
                   "d_matcher": _boot(mat_t1[fit] - door_t1[fit]) if len(fit) > 1 else None,
                   "d_rrf": _boot(rrf_t1[fit] - door_t1[fit]) if len(fit) > 1 else None,
                   "gain_matcher_fit": float((mat_t1[fit] - door_t1[fit]).mean()),
                   "gain_rrf_fit": float((rrf_t1[fit] - door_t1[fit]).mean()),
                   "gain_matcher_conf": float((mat_t1[conf] - door_t1[conf]).mean()) if conf else float("nan")}
    best = max(("matcher", "rrf"), key=lambda k: np.mean([outv[v]["gain_%s_fit" % k]
                                                          for v in FIT_VIEWS]))
    gain = float(np.mean([outv[v]["gain_%s_fit" % best] for v in FIT_VIEWS]))
    cgain = float(np.mean([outv[v]["gain_%s_conf" % best] for v in FIT_VIEWS]))
    verdict = {"best": best, "gain_fit": gain, "gain_conf": cgain, "bar": 0.075,
               "tol": -0.025, "final": bool(gain >= 0.075 and cgain >= -0.025)}
    out = {"views": outv, "verdict": verdict, "chance_top1": 1.0 / S}
    print(json.dumps(verdict, indent=1))
    with open(_out("matcher"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 5: fragment
FRAG_CACHE = os.path.join(OUTDIR, "m35_fragment.npz")
FRAG_VARIANTS: Dict[str, Tuple[int, ...]] = {
    "all22": tuple(range(22)),
    "colour_means": (16, 17, 18, 19, 20, 21),
    "curvature": (8, 9, 10, 11, 12, 13, 14, 15),
    "geometry": (0, 1, 2, 3, 4, 5, 6, 7),
    "no_colour": tuple(range(16)),
}


def _frag_matrix(view: str, idx: Tuple[int, ...]) -> np.ndarray:
    """The shipped channel with the descriptor restricted to `idx` of `PF.FIELDS`.

    Same grouping, same kernel, same symmetric mean-of-best as `matching.placement_fragment.channel`
    -- only the fields the kernel reads change, which is what an ablation of the channel's content is.
    """
    from matching import placement_fragment as PF
    from matching import placement_anchors as PA
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cells = PR.load_cells(RECORDS, [REF, view])
    an = PA.load()

    def A(k: str, v: str) -> Any:
        return an["%s__%s__contour_fragment" % (k.replace("/", "-"), v)]
    w = [float(PF.WEIGHTS[i]) for i in idx]

    def desc(places: Any, ang: Any) -> np.ndarray:
        groups = PF.group_by_anchor(places, list(np.asarray(ang, dtype=np.float64)))
        full = np.asarray([PF.full_descriptor(places, m) for m in groups],
                          dtype=np.float64).reshape(-1, len(PF.FIELDS))
        return full[:, list(idx)]
    g = [desc(cells[(k, REF)]["places"], A(k, REF)) for k in keys]
    pool = np.vstack([x for x in g if x.shape[0]])
    sig = PF.scales(pool)
    out = np.zeros((len(keys), len(keys)), dtype=np.float64)
    for i, k in enumerate(keys):
        q = desc(cells[(k, view)]["places"], A(k, view))
        if q.shape[0] == 0:
            continue
        for j, G in enumerate(g):
            if G.shape[0] == 0:
                continue
            block = PF.kernel_block(q, G, sig, weights=w)
            out[i, j] = 0.5 * float(block.max(axis=1).mean()) + 0.5 * float(block.max(axis=0).mean())
    return out


def fragment() -> Dict[str, Any]:
    """Does the fragment channel add >= +0.025 FIT recall@8 ON TOP OF THE CHOSEN (bag) DOOR?"""
    from matching import placement_bag as BAG
    from matching import placement_fuse as F
    edges = BAG.load_edges()
    split = _split()
    fit = np.asarray(split["FIT"])
    conf = np.asarray(_conf())
    bar = 0.025
    out: Dict[str, Any] = {"bar": bar, "variants": {k: list(map(str, v))
                                                    for k, v in FRAG_VARIANTS.items()},
                           "channel_alone": {}, "door": {}}
    for v in FIT_VIEWS:
        bag = _bag_colour(v, edges)
        ch = door_channels(v, edges)          # carries the shipped all-22 fragment channel
        r_bag = _ranks(bag)
        frag: Dict[str, np.ndarray] = {"all22": ch["fragment"]}
        for name, idx in FRAG_VARIANTS.items():
            frag[name] = frag.get(name) if name in frag else _frag_matrix(v, idx)
            r = _ranks(frag[name])
            out["channel_alone"].setdefault(name, {})[v] = {
                "recall8_fit": _rate(r, fit, S), "recall8_all80": float(np.mean(r < S)),
                "recall8_n": int(sum(r[i] < S for i in fit)), "n": int(fit.size)}
        per: Dict[str, Any] = {"bag_alone": {"recall8_fit": float(np.mean(r_bag[fit] < S)),
                                             "top1_fit": _rate(r_bag, fit, 1)}}
        for name in ("all22", "colour_means"):
            fused = F.rrf_weighted([bag, frag[name]], [1.0, W_FRAGMENT], k=K_RRF)
            r = _ranks(fused)
            per["bag+%s" % name] = {
                "recall8_fit": float(np.mean(r[fit] < S)), "top1_fit": _rate(r, fit, 1),
                "recall8_conf": _rate(r, conf, S),
                "gain_fit": float(np.mean((r[fit] < S).astype(float)
                                          - (r_bag[fit] < S).astype(float))),
                "paired_ci": _boot((r[fit] < S).astype(float) - (r_bag[fit] < S).astype(float))}
        out["door"][v] = per
        print("  %-6s bag r@8 FIT %.3f | +all22 %.3f (gain %+.3f) | +colour_means %.3f (gain %+.3f)"
              % (v, per["bag_alone"]["recall8_fit"], per["bag+all22"]["recall8_fit"],
                 per["bag+all22"]["gain_fit"], per["bag+colour_means"]["recall8_fit"],
                 per["bag+colour_means"]["gain_fit"]), flush=True)
    g_fit = float(np.mean([out["door"][v]["bag+all22"]["gain_fit"] for v in FIT_VIEWS]))
    g_conf = float(np.mean([out["door"][v]["bag+all22"]["recall8_conf"]
                            - out["door"][v]["bag_alone"].get("recall8_conf", np.nan)
                            for v in FIT_VIEWS]))
    out["verdict"] = {"gain_fit": g_fit, "bar": bar, "adds": bool(g_fit >= bar)}
    print(json.dumps(out["verdict"], indent=1))
    with open(_out("fragment"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 4: class output
def classout() -> Dict[str, Any]:
    """`recognise_class`'s rule read on records: class top-1 at the ALL arm, own object excluded."""
    from matching import placement_bag as BAG
    edges = BAG.load_edges()
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    cls = [o["cls"] for o in objs]
    split = _split()
    fit = np.asarray(split["FIT"])
    conf = np.asarray(_conf())
    out: Dict[str, Any] = {"chance_class_top1": 1.0 / len(set(cls)), "m23": 0.500,
                           "arm": list(BAG.ARMS["ALL"]), "weighting": "block", "top": 2,
                           "views": {}}
    for v in VIEWS:
        q = _feats(keys, _load([REF, v]), v, edges)
        cells = _load([REF, v])
        ref = _feats(keys, cells, REF, edges)
        q = _feats(keys, cells, v, edges)
        S_ob = np.asarray([[float(np.mean([q[i][b] @ ref[j][b] for b in BAG.blocks_of(
            BAG.ARMS["ALL"])])) for j in range(len(keys))] for i in range(len(keys))])
        hit = np.zeros(len(keys))
        rank_of = np.zeros(len(keys))
        for i in range(len(keys)):
            s_i = S_ob[i].copy()
            s_i[i] = np.nan
            keep = np.isfinite(s_i)
            ranked = BAG.class_scores(s_i[keep], np.asarray(cls, dtype=object)[keep], top=2)
            names = [c for c, _ in ranked]
            hit[i] = float(names[0] == cls[i])
            rank_of[i] = float(names.index(cls[i]))
        out["views"][v] = {"class_top1": float(hit.mean()), "n": len(keys),
                           "n_hit": int(hit.sum()),
                           "class_top1_fit": float(hit[fit].mean()),
                           "class_top1_conf": float(hit[conf].mean()) if conf.size else None,
                           "ci": _boot(hit), "mean_rank_of_truth": float(rank_of.mean()),
                           "per_class": {c: float(hit[np.asarray(cls) == c].mean())
                                         for c in sorted(set(cls))}}
        print("  %-6s class top-1 %.3f (%d/%d) CI%s | rank %.2f"
              % (v, hit.mean(), int(hit.sum()), len(keys),
                 np.round(out["views"][v]["ci"], 3), rank_of.mean()), flush=True)
    with open(_out("classout"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- determinism
def determinism() -> Dict[str, Any]:
    """One gallery, two queries, 1/2/4 workers: identical door order and identical matcher score."""
    from matching import placement_gallery as GL
    from matching import placement_recognise as R
    from matching import placement_records as PR
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    view = "035-045"
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    gal.bag_features = None
    cells = PR.load_cells(RECORDS, [REF, view])
    picked = [0, 37]
    out: Dict[str, Any] = {"queries": [keys[i] for i in picked], "view": view,
                           "gallery_views": [REF], "workers": {}}
    base = None
    for w in (1, 2, 4):
        got = []
        for i in picked:
            res = R.recognise((keys[i], REC.path_of(objs[i], view)), gal, matcher=False, s=2,
                              use_pair_cache=False,
                              workers=w, record=cells[(keys[i], view)],
                              appearance=np.zeros(gal.appearance.shape[1]))
            got.append({"door": list(res.door_order), "ranking": list(res.ranking)})
        out["workers"][str(w)] = got
        if base is None:
            base = got
        assert got == base, w
    # the matcher itself, on the same two pairs, at 1 and 3 workers
    sc: List[List[float]] = []
    for w in (1, 3):
        res = R.recognise((keys[0], REC.path_of(objs[0], view)), gal, matcher=True, s=2,
                          use_pair_cache=False, workers=w,
                          record=cells[(keys[0], view)],
                          appearance=np.zeros(gal.appearance.shape[1]))
        sc.append([float(v) for _k, v in sorted(res.scores.items())])
    diff = max((abs(a - b) for a, b in zip(sc[0], sc[1])), default=0.0)
    out["matcher_max_abs_diff"] = diff
    out["verdict"] = {"door_identical": True, "matcher_scores": sc,
                      "matcher_max_abs_diff": diff, "passes": bool(diff == 0.0)}
    print(json.dumps(out["verdict"], indent=1))
    with open(_out("determinism"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--class", dest="class_test", action="store_true")
    ap.add_argument("--door", action="store_true")
    ap.add_argument("--matcher", action="store_true")
    ap.add_argument("--matcher-report", action="store_true")
    ap.add_argument("--fragment", action="store_true")
    ap.add_argument("--classout", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.edges:
        edges_block()
        return 0
    if a.replicate:
        replicate()
        return 0
    if a.class_test:
        class_test()
        return 0
    if a.door:
        door()
        return 0
    if a.matcher:
        print(json.dumps(matcher_pairs()))
        return 0
    if a.matcher_report:
        matcher_report()
        return 0
    if a.fragment:
        fragment()
        return 0
    if a.classout:
        classout()
        return 0
    if a.determinism:
        determinism()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
