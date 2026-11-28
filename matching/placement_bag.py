"""M35 item 1: the OBJECT-LEVEL descriptor -- an object is a DISTRIBUTION over its places, not a max.

WHY THIS FILE EXISTS.  Every channel this tree shipped compares one place with one place and takes a
max (`placement_content`, `placement_shape`, `placement_fragment`).  M32 measured what that does: the
place-level colour cosine puts **97.8 %** of pairs at >= 0.999 and shape **97.4 %** -- a max over
almost-equal numbers ranks on noise.  An object-level HISTOGRAM of the same measurements does not
saturate (the M34 ablation in the mission brief: colour means alone recall@8 64/80 at `-035-045`).

THE OBJECT IS A BAG OF FOUR BLOCKS.  Everything below is read from the committed place records
(`data/placement_records_m32.npz`) and nothing else -- no image, no graph, no label:

  COLOUR   per place the record carries six CIELAB ring means in scaled units (L/100, a/128, b/128
           for ring 1 and ring 2).  Three blocks: the AB PLANE of ring 1 as a 5x5 histogram, the AB
           plane of ring 2 as 5x5, and the LIGHTNESS of ring 1 as a 6-bin histogram.
  SHAPE    the record's quantised key is (`lens`; theta, coherence, scale, kind, dim).  Blocks:
           a KEY BAG over lens x theta(12) x coherence(4) x scale-quantile(4) = 576 bins, plus
           EIGENVALUE RATIO and RESPONSE as 6-bin histograms PER LENS (18 + 18).
  LAYOUT   where the places are and how they point, in the object's own frame: log2 place scale (9),
           radial distance from the object's centroid in units of its own RMS radius (8), the D2
           distribution of pairwise place distances in the same units (12), and each place's
           orientation RELATIVE TO THE OBJECT'S DOMINANT AXIS (12, mod pi).
  CFG      the configuration row `Dz`/`Mz` the record already carries: 24 slots = the standardised
           log distances to the nearest neighbours at ranks 1..24, with the mask that says which
           ranks exist.  Read `matching/placement_context.config_descriptor` (which builds
           (sorted log distances, sorted source-frame directions) at K = 24) and
           `placement_prefilter.bag_v2` (which stores D at K = 24 with a mask): **each slot j is the
           standardised log distance to the (j+1)-th nearest neighbour**.  The 24 slots are read as
           SIX GROUPS OF FOUR RANKS (1-4, 5-8, ..., 21-24); for each group and place the mean of the
           LIVE standardised values is taken, and that per-group "shell distance" is histogrammed
           over the object's places into 6 bins.  6 groups x 6 bins = 36.

EACH BLOCK IS L1-NORMALISED AND THEN SQUARE-ROOTED (the Hellinger map, so a dot product between two
blocks is the Bhattacharyya coefficient -- a distance, not a count).  A GROUP's score is the MEAN of
its blocks' dot products, so a group with three blocks cannot outvote a group with one.

BIN EDGES COME FROM THE GALLERY'S REFERENCE VIEW (`-000-000`) AND ARE STORED IN A COMMITTED FILE WITH
PROVENANCE (`matching/placement_bag_edges.json`).  They are quantiles of the reference view's own
values -- label-free, no class touched -- and they are computed ONCE per block, so a query view is
described in the reference view's ruler.  A missing edges file is a REFUSAL, never a silent refit.

EXCLUDED, WITH THE REASON ON THE RECORD.  `var_x`, `var_y` and `rel_angle` are never read by any block
here: `var_x` is pinned at the `_parabola` clip (one distinct value over 46,059 places, M32) and
`rel_angle` is ~0 by construction, so a histogram of them is a delta.  A test asserts the exclusion.

    from matching import placement_bag as BAG
    edges = BAG.load_edges()                       # refuses if absent
    f = BAG.describe(record, edges)                # {block: vector}
    s = BAG.group_scores(f_q, f_g, ("LAYOUT", "CFG"))
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES_PATH = os.path.join(ROOT, "matching", "placement_bag_edges.json")

REF_VIEW = "000-000"
EPS = 1e-12

#: The blocks, their sizes and the group membership every arm is defined by.  Registered HERE, in
#: writing, before any number is read.
BLOCKS: Dict[str, int] = {
    "ab1": 25,      # COLOUR: ring 1 ab plane, 5 x 5
    "ab2": 25,      # COLOUR: ring 2 ab plane, 5 x 5
    "L1": 6,        # COLOUR: ring 1 lightness, 6 bins
    "key": 576,     # SHAPE: lens(3) x theta(12) x coherence(4) x scale quantile(4)
    "eig": 18,      # SHAPE: eigenvalue ratio, 6 bins per lens
    "resp": 18,     # SHAPE: response, 6 bins per lens
    "scale": 9,     # LAYOUT: log2 place scale
    "radial": 8,    # LAYOUT: distance from the object's centroid / RMS radius
    "d2": 12,       # LAYOUT: pairwise place distances / RMS radius
    "orient": 12,   # LAYOUT: place theta relative to the object's dominant axis, mod pi
    "cfg": 36,      # CFG: six groups of four neighbour ranks, 6 bins each
}
GROUPS: Dict[str, Tuple[str, ...]] = {
    "COLOUR": ("ab1", "ab2", "L1"),
    "SHAPE": ("key", "eig", "resp"),
    "LAYOUT": ("scale", "radial", "d2", "orient"),
    "CFG": ("cfg",),
}
#: the pre-registered arms (M35 item 2).  GEOMETRY has no local appearance statistic at all.
ARMS: Dict[str, Tuple[str, ...]] = {
    "COLOUR": ("COLOUR",),
    "SHAPE": ("SHAPE",),
    "LAYOUT": ("LAYOUT",),
    "CFG": ("CFG",),
    "GEOMETRY": ("LAYOUT", "CFG"),
    "COLOUR-FREE": ("SHAPE", "LAYOUT", "CFG"),
    "ALL": ("COLOUR", "SHAPE", "LAYOUT", "CFG"),
    "G4": ("COLOUR", "SHAPE", "LAYOUT", "CFG"),      # equal GROUP weights: see group_scores(g4=True)
}
#: how an arm's blocks are weighted.  `group` = mean over the arm's groups of the group's block
#: mean (so a group with three blocks cannot outvote a group with one); `block` = mean over ALL of
#: the arm's blocks, which is what makes ALL (11 blocks) differ from G4 (4 equal groups).
WEIGHTING: Dict[str, str] = {"ALL": "block"}
DEFAULT_WEIGHTING = "group"

#: columns the record carries and NO block may read (constant / zero by construction)
EXCLUDED: Tuple[str, ...] = ("var_x", "var_y", "rel_angle")

LENSES: Tuple[str, ...] = ("Blob", "BoundaryArc", "CornerPoint")
THETA_BINS, COH_BINS, SCALE_Q = 12, 4, 4
N_AB, N_L1 = 5, 6
N_SCALE, N_RADIAL, N_D2, N_ORIENT = 9, 8, 12, 12
N_CFG_GROUPS, N_CFG_BINS = 6, 6
N_CFG_SLOTS = 24


class BagRefusal(RuntimeError):
    """The descriptor cannot be built from this record -- refused, never zeroed."""


# --------------------------------------------------------------------------- basic field access
def _col(rec: Dict[str, Any], name: str) -> np.ndarray:
    """One record column as a float array, in place order; a missing column REFUSES."""
    ps = rec["places"]
    vals = [p.get(name) for p in ps]
    if any(v is None for v in vals):
        raise BagRefusal("the record has no %r for %d of %d places: it was exported without the "
                         "M32 enrichment this descriptor reads" % (name, sum(v is None for v in vals),
                                                                   len(ps)))
    return np.asarray([float(v) for v in vals], dtype=np.float64)


def _xy_s_theta(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = rec["places"]
    x = np.asarray([float(p["x"]) for p in P], dtype=np.float64)
    y = np.asarray([float(p["y"]) for p in P], dtype=np.float64)
    s = np.asarray([float(p.get("s") or 0.0) for p in P], dtype=np.float64)
    th = np.asarray([np.nan if p.get("theta") is None else float(p["theta"]) for p in P],
                    dtype=np.float64)
    return x, y, s, th


def _content(rec: Dict[str, Any]) -> np.ndarray:
    P = rec["places"]
    rows = [p.get("content") for p in P]
    if any(r is None for r in rows):
        raise BagRefusal("the record carries no colour content for %d places: build the record with "
                         "content.kind = colour_rings (M28), the descriptor cannot invent colour"
                         % sum(r is None for r in rows))
    return np.asarray([[float(v) for v in r[:6]] for r in rows], dtype=np.float64).reshape(-1, 6)


def _keys(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(lens index, 5 quantised key slots) per place, straight off the record's own layout."""
    P = rec["places"]
    lens = np.asarray([LENSES.index(p["lens"]) if p.get("lens") in LENSES else -1 for p in P],
                      dtype=np.int64)
    q = np.asarray([[float(x) for x in (p["key"][1] if isinstance(p.get("key"), (tuple, list))
                                        else ())][:5] for p in P], dtype=np.float64)
    return lens, q


def _frame(rec: Dict[str, Any]) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Object frame: centroid, RMS radius, dominant axis (principal axis, mod pi), and the axis."""
    x, y, s, th = _xy_s_theta(rec)
    c = np.asarray([x.mean(), y.mean()], dtype=np.float64)
    rel = np.stack([x - c[0], y - c[1]], axis=1)
    rms = float(np.sqrt(np.mean(np.sum(rel ** 2, axis=1)))) if len(x) else 1.0
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0
    if len(x) > 1:
        w, V = np.linalg.eigh(np.cov(rel.T))
        axis = V[:, -1]
        ang = float(math.atan2(axis[1], axis[0]))
    else:
        ang = 0.0
    return c, rms, ang, rel


# --------------------------------------------------------------------------- edges
def _q_edges(v: np.ndarray, n: int) -> np.ndarray:
    """n+1 quantile edges over a label-free pool, made strictly increasing."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(-1.0, 1.0, n + 1)
    qs = np.quantile(v, np.linspace(0.0, 1.0, n + 1))
    qs = np.maximum.accumulate(qs)
    for i in range(1, qs.size):                        # de-duplicate: a delta makes an empty bin
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-6
    return qs


def edges_from(pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Bin edges for every block, from the REFERENCE view's own values (label-free quantiles)."""
    ab1: List[np.ndarray] = []
    ab2: List[np.ndarray] = []
    L1: List[np.ndarray] = []
    th: List[np.ndarray] = []
    coh: List[np.ndarray] = []
    sc: List[np.ndarray] = []
    eig = {j: [] for j in range(len(LENSES))}
    resp = {j: [] for j in range(len(LENSES))}
    sca: List[np.ndarray] = []
    rad: List[np.ndarray] = []
    d2: List[np.ndarray] = []
    ori: List[np.ndarray] = []
    shells = {g: [] for g in range(N_CFG_GROUPS)}
    for rec in pool:
        C = _content(rec)
        ab1.append(C[:, 1]); ab2.append(C[:, 4]); L1.append(C[:, 0])
        lens, q = _keys(rec)
        th.append(q[:, 0]); coh.append(q[:, 1]); sc.append(q[:, 2])
        for j in range(len(LENSES)):
            m = lens == j
            if m.any():
                eig[j].append(_col(rec, "eigratio")[m])
                resp[j].append(_col(rec, "resp")[m])
        x, y, s, the = _xy_s_theta(rec)
        c, rms, ang, rel = _frame(rec)
        sca.append(np.log2(np.maximum(s, 1e-6)))
        rad.append(np.hypot(rel[:, 0], rel[:, 1]) / rms)
        if len(x) > 1:
            dd = np.hypot(rel[:, None, 0] - rel[None, :, 0], rel[:, None, 1] - rel[None, :, 1])
            d2.append(dd[np.triu_indices(len(x), k=1)] / rms)
        ok = np.isfinite(the)
        if ok.any():
            ori.append(((the[ok] - ang) % math.pi))
        for g, sh in _cfg_shells(rec).items():
            shells[g].append(sh)
    out: Dict[str, Any] = {
        "version": 1, "ref_view": REF_VIEW, "objects": len(pool), "places": int(sum(
            len(r["places"]) for r in pool)),
        "q": {
            "ab": [_q_edges(np.concatenate(ab1), N_AB).tolist(),
                   _q_edges(np.concatenate(ab2), N_AB).tolist()],
            "L1": _q_edges(np.concatenate(L1), N_L1).tolist(),
            "theta": [0.0, float(THETA_BINS)], "coh": [0.0, float(COH_BINS)],
            "scale_q": [float(x) for x in np.quantile(np.concatenate(sc), np.linspace(0, 1, SCALE_Q + 1))],
            "eig": [_q_edges(np.concatenate(eig[j]) if eig[j] else np.zeros(1), 6).tolist()
                    for j in range(len(LENSES))],
            "resp": [_q_edges(np.concatenate(resp[j]) if resp[j] else np.zeros(1), 6).tolist()
                     for j in range(len(LENSES))],
            "scale": _q_edges(np.concatenate(sca), N_SCALE).tolist(),
            "radial": _q_edges(np.concatenate(rad), N_RADIAL).tolist(),
            "d2": _q_edges(np.concatenate(d2), N_D2).tolist(),
            "orient": [float(x) for x in np.linspace(0.0, math.pi, N_ORIENT + 1)],
            "cfg": [_q_edges(np.concatenate(shells[g]) if shells[g] else np.zeros(1),
                             N_CFG_BINS).tolist() for g in range(N_CFG_GROUPS)],
        },
    }
    return out


def _cfg_shells(rec: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """Per place, the mean LIVE standardised log distance in each group of four neighbour ranks."""
    if not rec["places"] or "Dz" not in rec["places"][0]:
        raise BagRefusal("the record carries no configuration row (Dz/Mz)")
    D = np.asarray([p["Dz"] for p in rec["places"]], dtype=np.float64).reshape(-1, N_CFG_SLOTS)
    M = np.asarray([p["Mz"] for p in rec["places"]], dtype=np.float64).reshape(-1, N_CFG_SLOTS)
    out: Dict[int, np.ndarray] = {}
    for g in range(N_CFG_GROUPS):
        sl = slice(4 * g, 4 * g + 4)
        live = M[:, sl] > 0.0
        vals = np.where(live, D[:, sl], np.nan)
        n = live.sum(axis=1)
        out[g] = np.where(n > 0, np.nansum(vals, axis=1) / np.maximum(n, 1), np.nan)
    return out


def save_edges(edges: Dict[str, Any], path: Optional[str] = None,
               provenance: Optional[Dict[str, Any]] = None) -> str:
    p = path or EDGES_PATH
    d = dict(edges)
    if provenance:
        d["provenance"] = provenance
    with open(p, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return p


def load_edges(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or EDGES_PATH
    if not os.path.exists(p):
        raise BagRefusal(
            "no bin edges at %s -- the object descriptor is defined in the REFERENCE view's ruler, "
            "and a missing ruler is a refusal, not a silent refit.  Build it with "
            "`python Topology/m35_bag.py --edges`." % p)
    with open(p) as fh:
        d = json.load(fh)
    if "q" not in d:
        raise BagRefusal("the edges file at %s has no 'q' block" % p)
    return d


# --------------------------------------------------------------------------- the descriptor
def _hist(v: np.ndarray, edges: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    h, _ = np.histogram(v[ok], bins=np.asarray(edges, dtype=np.float64),
                        weights=None if weights is None else np.asarray(weights)[ok])
    return h.astype(np.float64)


def _h2d(a: np.ndarray, b: np.ndarray, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b)
    h, _, _ = np.histogram2d(a[ok], b[ok], bins=[np.asarray(ea), np.asarray(eb)])
    return h.ravel().astype(np.float64)


def _hellinger(h: np.ndarray) -> np.ndarray:
    """L1-normalise then square-root: a dot product of two of these is a Bhattacharyya coefficient."""
    v = np.asarray(h, dtype=np.float64).ravel()
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.zeros_like(v)
    return np.sqrt(v / s)


def describe(rec: Dict[str, Any], edges: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Every block of one object, as a sqrt-normalised vector.  Refuses what it cannot read."""
    if not rec.get("places"):
        raise BagRefusal("the record has no places")
    q = edges["q"]
    C = _content(rec)
    lens, kq = _keys(rec)
    x, y, s, th = _xy_s_theta(rec)
    c, rms, ang, rel = _frame(rec)
    out: Dict[str, np.ndarray] = {}
    out["ab1"] = _hellinger(_h2d(C[:, 1], C[:, 2], q["ab"][0], q["ab"][0]))
    out["ab2"] = _hellinger(_h2d(C[:, 4], C[:, 5], q["ab"][1], q["ab"][1]))
    out["L1"] = _hellinger(_hist(C[:, 0], q["L1"]))
    # SHAPE: the key bag over lens x theta x coherence x scale quantile, then eig/resp per lens
    key = np.zeros((len(LENSES), THETA_BINS, COH_BINS, SCALE_Q), dtype=np.float64)
    sq = np.asarray(q["scale_q"], dtype=np.float64)
    sc_bin = np.clip(np.searchsorted(sq, kq[:, 2], side="right") - 1, 0, SCALE_Q - 1)
    ok = (lens >= 0) & np.isfinite(kq[:, 0]) & np.isfinite(kq[:, 1])
    ti = np.clip(kq[:, 0].astype(np.int64), 0, THETA_BINS - 1)
    ci = np.clip(kq[:, 1].astype(np.int64), 0, COH_BINS - 1)
    np.add.at(key, (lens[ok], ti[ok], ci[ok], sc_bin[ok]), 1.0)
    out["key"] = _hellinger(key.ravel())
    eig = np.zeros((len(LENSES), 6), dtype=np.float64)
    rsp = np.zeros((len(LENSES), 6), dtype=np.float64)
    E = _col(rec, "eigratio")
    R = _col(rec, "resp")
    for j in range(len(LENSES)):
        m = lens == j
        if m.any():
            eig[j] = _hist(E[m], q["eig"][j])
            rsp[j] = _hist(R[m], q["resp"][j])
    out["eig"] = _hellinger(eig.ravel())
    out["resp"] = _hellinger(rsp.ravel())
    # LAYOUT
    out["scale"] = _hellinger(_hist(np.log2(np.maximum(s, 1e-6)), q["scale"]))
    rad = np.hypot(rel[:, 0], rel[:, 1]) / rms
    out["radial"] = _hellinger(_hist(rad, q["radial"]))
    if len(x) > 1:
        dd = np.hypot(rel[:, None, 0] - rel[None, :, 0], rel[:, None, 1] - rel[None, :, 1])
        out["d2"] = _hellinger(_hist(dd[np.triu_indices(len(x), k=1)] / rms, q["d2"]))
    else:
        out["d2"] = np.zeros(BLOCKS["d2"])
    a = np.where(np.isfinite(th), (th - ang) % math.pi, np.nan)
    out["orient"] = _hellinger(_hist(a, q["orient"]))
    # CFG
    shells = _cfg_shells(rec)
    cfg = np.zeros((N_CFG_GROUPS, N_CFG_BINS), dtype=np.float64)
    for g in range(N_CFG_GROUPS):
        cfg[g] = _hist(shells[g], q["cfg"][g])
    out["cfg"] = _hellinger(cfg.ravel())
    for k, n in BLOCKS.items():
        if out[k].size != n:
            raise BagRefusal("block %r has %d entries, the registry says %d" % (k, out[k].size, n))
    return out


def blocks_of(arm: Sequence[str]) -> Tuple[str, ...]:
    return tuple(b for g in arm for b in GROUPS[g])


def group_scores(fq: Dict[str, np.ndarray], fg: Dict[str, np.ndarray],
                 arm: Sequence[str], weighting: Optional[str] = None) -> np.ndarray:
    """(n_query, n_gallery) score of one ARM: mean over the arm's groups of their block means."""
    if weighting == "block":
        return np.mean([fq[b] @ fg[b].T for b in blocks_of(arm)], axis=0)
    return np.mean([np.mean([fq[b] @ fg[b].T for b in GROUPS[g]], axis=0) for g in arm], axis=0)


def arm_matrix(feats: Sequence[Dict[str, np.ndarray]], arm: Sequence[str]) -> np.ndarray:
    """All-pairs object matrix for one arm (query rows, gallery columns), same rule as
    `group_scores`: the mean over the arm's groups of the mean of the group's block dot products."""
    n = len(feats)
    bs = blocks_of(arm) if WEIGHTING.get(_name(arm)) == "block" else None
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if bs is not None:
                M[i, j] = float(np.mean([feats[i][b] @ feats[j][b] for b in bs]))
            else:
                M[i, j] = float(np.mean([np.mean([feats[i][b] @ feats[j][b] for b in GROUPS[g]])
                                         for g in arm]))
    return M


def _name(arm: Sequence[str]) -> str:
    for k, v in ARMS.items():
        if tuple(v) == tuple(arm):
            return k
    return ""


# --------------------------------------------------------------------------- the door (item 3)
#: The arm the door ships (chosen on FIT by recall@8, then top-1, ties -> fewer channels; see
#: `Topology/m35_bag.py --door`).  ONE group of three blocks, no fusion, no weights.
DOOR_ARM: Tuple[str, ...] = ("COLOUR",)


def door_scores(q_feat: Dict[str, np.ndarray], gal_feats: Sequence[Dict[str, np.ndarray]],
                n_obj: int, n_views: int, arm: Sequence[str] = DOOR_ARM,
                weighting: Optional[str] = None) -> np.ndarray:
    """(n_obj,) door score: the arm's score of the query cell against each gallery cell, MAX over
    the object's stored views -- the shipped door's own aggregation (`_max_over_views`)."""
    if len(gal_feats) != n_obj * n_views:
        raise BagRefusal("the gallery has %d cells for %d objects x %d views"
                         % (len(gal_feats), n_obj, n_views))
    flat = np.asarray([float(np.asarray(group_scores(q_feat, f, arm, weighting)).ravel()[0])
                       for f in gal_feats])
    return flat.reshape(n_obj, n_views).max(axis=1)


def class_scores(obj_scores: np.ndarray, classes: Sequence[str], top: int = 2) -> List[Tuple[str, float]]:
    """Class ranking: the MEAN OF THE TOP-`top` instance scores of the class, best first."""
    per: Dict[str, List[float]] = {}
    for j, c in enumerate(classes):
        per.setdefault(str(c), []).append(float(obj_scores[j]))
    out = [(c, float(np.mean(sorted(v, reverse=True)[:max(1, int(top))])))
           for c, v in per.items()]
    return sorted(out, key=lambda t: (-t[1], t[0]))


def feature_table(pool: Sequence[Dict[str, Any]], edges: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    return [describe(r, edges) for r in pool]
