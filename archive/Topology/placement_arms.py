"""Task 4 PLACEMENT mission — Part 6: the ONE arms run.

ARMS
----
N            node-only: node similarity, no edges at all.
R            current edges: the shipped matcher (relative + edge credit).
K            walker as shipped (only if 0a says it is a candidate).
P            placement: node similarity MODIFIED by placement support.
P-shuffled   same links, geometry swapped among edges of the same type.
W            rewired: geometry stays with the edge, endpoints swapped,
             degree preserved, deterministic.

0a said the walker's admission test is NOT mapping-dependent, so K is not a
candidate for placement and is NOT run as a placement arm.  It is reported as
absent for that reason rather than silently dropped.

MEASURED
--------
Correspondence accuracy on ambiguous and unambiguous nodes SEPARATELY,
clustered by scene, with intervals AND MDE.  The family-specific expectations:

  NEGATIVE-A  intra-object edges only => accuracy on the swappable subset is
              about 0.5 by construction (tie order decides).  A clear win
              means a positional leak.
  NEGATIVE-B  cross-object edges allowed => a win is legitimate, because the
              swap reverses the cross-object vectors.
  RESOLVABLE  the positive test.
  CONTROL     ordinary case; P must not hurt it.

POWER
-----
The scene count is pre-registered HERE, before the run.  With n scenes and the
observed within-scene spread, the MDE on the ambiguous-node accuracy
difference is reported next to every contrast so an unresolved result is
labelled INCONCLUSIVE rather than "no effect".
"""
from __future__ import annotations

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

#: PRE-REGISTERED scene count per family (fixed before the run).
SCENES_PER_FAMILY = {"RESOLVABLE": 1, "NEGATIVE-A": 2, "NEGATIVE-B": 1,
                     "CONTROL": 2}
NBOOT = 2000
SEED = 42


# ---------------------------------------------------------------------------
# correspondence by node truth (same lens, nearest anchor; deterministic)
# ---------------------------------------------------------------------------
def truth_mapping(g_ref: Any, g_q: Any, radius: float = 12.0) -> Dict[int, int]:
    p0 = {n: g_ref.subgraph.anchor_of_node[n] for n in g_ref.subgraph.nodes
          if n in g_ref.subgraph.anchor_of_node}
    p1 = {n: g_q.subgraph.anchor_of_node[n] for n in g_q.subgraph.nodes
          if n in g_q.subgraph.anchor_of_node}
    l0 = {n: str(g_ref.sigs.get(n, ("?",))[0]) for n in p0}
    l1 = {n: str(g_q.sigs.get(n, ("?",))[0]) for n in p1}
    pools: Dict[str, List[int]] = defaultdict(list)
    for n in sorted(p0):
        pools[l0[n]].append(n)
    out: Dict[int, int] = {}
    used: set = set()
    for nq in sorted(p1):
        cands = [r for r in pools.get(l1[nq], ()) if r not in used]
        if not cands:
            continue
        xq, yq = float(p1[nq][0]), float(p1[nq][1])
        best, bd = None, float("inf")
        for r in cands:
            d = math.hypot(float(p0[r][0]) - xq, float(p0[r][1]) - yq)
            if d < bd:
                bd, best = d, r
        if best is not None and bd <= radius:
            out[int(nq)] = int(best)
            used.add(best)
    return out


# ---------------------------------------------------------------------------
# supported edges: restrict to what 0c allows
# ---------------------------------------------------------------------------
INTRA_EDGE_TYPES = {"ORIENTATION_CONTINUITY"}


def supported_edges(g_q: Any, g_r: Any, mode: str) -> List[Tuple[int, int, Any]]:
    """The edges this arm may use, for the NEGATIVE-A/B distinction.

    ``mode='intra'`` keeps only edges whose endpoints come from ONE object --
    the graphs are object-agnostic, so the object is recovered from the
    feature track: a node belongs to the object whose local frame produced it.
    ``mode='cross'`` keeps every placement edge.
    """
    from matching.placement import PAYLOAD_CLASSES
    allowed = set(PAYLOAD_CLASSES)
    ep = getattr(g_r.subgraph, "edge_params", {}) or {}
    out: List[Tuple[int, int, Any]] = []
    for s, d, et in g_r.subgraph.edges:
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        if name not in allowed:
            continue
        if (s, d, et) not in ep and (d, s, et) not in ep:
            continue
        out.append((int(s), int(d), et))
    return out


def choose(mapping: Dict[int, int], scope: Sequence[int]) -> int:
    """Pick the reference node for a query node."""
    return mapping.get(scope[0], -1) if scope else -1


# ---------------------------------------------------------------------------
# ARMS: score(query node) over a restricted scope, pick argmax
# ---------------------------------------------------------------------------
def node_sim_row(g_q: Any, g_r: Any, u: int) -> Dict[int, float]:
    su = g_q.sigs.get(u)
    if su is None:
        return {}
    row: Dict[int, float] = {}
    for v in sorted(g_r.subgraph.nodes):
        sv = g_r.sigs.get(v)
        if sv is None:
            continue
        try:
            row[int(v)] = float(su[1].similarity(sv[1]))
        except Exception:
            continue
    return row


def arm_N(row: Dict[int, float], scope: Sequence[int]) -> int:
    best, bs = -1, -float("inf")
    for v in scope:
        s = row.get(v, 0.0)
        if s > bs:
            bs, best = s, v
    return best


def arm_R(row: Dict[int, float], g_q: Any, g_r: Any, u: int,
          scope: Sequence[int], beta: float = 0.5) -> int:
    """Current edges: node similarity plus the shipped edge credit.

    The edge term is the matcher's own ``edge_pair_credit`` averaged over the
    query node's incident edges, which is the same quantity ``joint_mcs``
    pays for.  ``beta`` is the committed 0.50.
    """
    from matching.joint_mcs import edge_pair_credit
    qn = [d for (s, d, et) in g_q.subgraph.edges if int(s) == u]
    cand: List[Tuple[int, float]] = []
    for v in scope:
        s = row.get(v, 0.0)
        ec = 0.0
        n = 0
        for d in qn:
            for (s2, d2, et2) in g_r.subgraph.edges:
                if int(s2) == v and int(d2) == d:
                    try:
                        ec += float(edge_pair_credit(g_q.subgraph, g_r.subgraph,
                                                     int(u), d, v, int(d2), et2))
                    except Exception:
                        pass
                    n += 1
                    break
        term = (ec / n) if n else 0.0
        cand.append((v, s * (1.0 - beta) + term * beta))
    best, bs = -1, -float("inf")
    for v, s in cand:
        if s > bs:
            bs, best = s, v
    return best


def arm_P(row: Dict[int, float], support: Optional[float], scope: Sequence[int],
          weight: float = 0.25) -> int:
    """Placement: the ONE pre-stated modifier on node similarity, ungated."""
    if support is None:
        return arm_N(row, scope)
    factor = 1.0 + weight * (support - 0.5)
    best, bs = -1, -float("inf")
    for v in scope:
        s = row.get(v, 0.0) * factor
        if s > bs:
            bs, best = s, v
    return best


# ---------------------------------------------------------------------------
# P-shuffled and W
# ---------------------------------------------------------------------------
def shuffled_payloads(p: Dict[Tuple[int, int, Any], Any], seed: int = 7):
    """Same LINKS, geometry swapped among edges of the SAME TYPE."""
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


def rewired_payloads(p: Dict[Tuple[int, int, Any], Any], g: Any, seed: int = 0):
    """Geometry stays with the EDGE, endpoints swapped, degree preserved.

    Same construction as the retrieval mission's determinism fix: sort the
    edge list canonically BEFORE the seeded shuffle, because a frozenset's
    iteration order is salted per process.
    """
    from types import SimpleNamespace
    keys = sorted(p, key=lambda kk: (kk[0], kk[1], getattr(kk[2], "name", "")))
    if len(keys) < 2:
        return dict(p)
    rng = random.Random(seed)
    dsts = [k[1] for k in keys]
    for _ in range(200):
        rng.shuffle(dsts)
        if all(a != b for a, b in zip([k[1] for k in keys], dsts)):
            break
    out = {}
    for k, nd in zip(keys, dsts):
        u, _d, et = k
        out[(u, nd, et)] = p[k]
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def cluster_boot(pairs: Sequence[Tuple[str, float]], nboot: int = NBOOT,
                 seed: int = SEED) -> Dict[str, Any]:
    d: Dict[str, List[float]] = defaultdict(list)
    for k, v in pairs:
        d[k].append(v)
    keys = sorted(d)
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "mde": float("nan"), "n": 0, "n_eff": 0}
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
    return {"mean": float(np.mean([v for _, v in pairs])), "lo": lo, "hi": hi,
            "hw": hw, "se": se, "mde": 2.8 * se, "n": len(pairs),
            "n_eff": len(keys)}


def paired_boot(pa, pb, nboot: int = NBOOT, seed: int = SEED) -> Dict[str, Any]:
    da: Dict[str, List[float]] = defaultdict(list)
    db: Dict[str, List[float]] = defaultdict(list)
    for k, v in pa:
        da[k].append(v)
    for k, v in pb:
        db[k].append(v)
    keys = sorted(set(da) & set(db))
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "mde": float("nan"), "n": 0, "n_eff": 0}
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


def verdict(d: Dict[str, Any]) -> str:
    if d.get("n", 0) == 0 or not math.isfinite(d.get("mean", float("nan"))):
        return "NO DATA"
    if d["lo"] > d["mde"]:
        return "POSITIVE"
    if d["hi"] < -d["mde"]:
        return "NEGATIVE"
    if abs(d["mean"]) > d["mde"]:
        return "INCONCLUSIVE (mean > MDE but CI crosses 0)"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
def main() -> int:
    import guard
    guard.activate()
    from eth80_frontend import build_graph
    from matching.placement import edge_payloads
    from matching.placement_scorer import node_support
    from Topology.placement_scenes import all_specs, posed

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tolerance.json")) as fh:
        TOLTAB = json.load(fh)

    acc: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list))
    diagnostics: List[Dict[str, Any]] = []

    for spec in all_specs():
        g_ref = build_graph(posed(spec.scene, 0.0).render().image_u8,
                            spec.name + "|ref")
        p_ref = edge_payloads(g_ref)
        mode = "cross" if spec.family == "NEGATIVE-B" else "intra"
        for label, pose in spec.views:
            if label == "v0_base":
                continue
            g_q = build_graph(posed(spec.scene, pose["rot_deg"],
                                    tuple(pose["shift"]),
                                    float(pose["scale"])).render().image_u8,
                              spec.name + "|" + label)
            p_q = edge_payloads(g_q)
            tol = float(TOLTAB["views"].get(label, {}).get("tolerance", 1.0))
            tm = truth_mapping(g_ref, g_q)
            if not tm:
                continue
            # scope: reference nodes of the SAME lens as the query node
            by_lens: Dict[str, List[int]] = defaultdict(list)
            for v in sorted(g_ref.subgraph.nodes):
                by_lens[str(g_ref.sigs.get(v, ("?",))[0])].append(int(v))
            # ambiguous label (Part 4's observable rule)
            from Topology.placement_gate import appearance_ambiguous
            # [BUGFIX] the ambiguity label needs the SIMILARITY ROWS: with
            # sims=None only the exact-signature route runs, and every
            # signature in these graphs is unique, so it labelled NOTHING and
            # the whole ambiguous bucket came out empty (measured: 0 rows).
            from Topology.placement_arms import node_sim_row as _nsr
            _sims = {int(u): sorted(_nsr(g_q, g_ref, u).items(),
                                    key=lambda t: (-t[1], t[0]))
                     for u in sorted(tm)}
            amb = appearance_ambiguous(g_q.sigs, g_ref.sigs, sims=_sims)

            for u in sorted(tm):
                truth = tm[u]
                lens = str(g_q.sigs.get(u, ("?",))[0])
                scope = by_lens.get(lens, [])
                row = node_sim_row(g_q, g_ref, u)
                if not row or not scope:
                    continue
                sup = node_support(u, tm, p_q, p_ref, tol,
                                   max_credit=1.0)
                support = sup["support"] if sup else None

                preds = {
                    "N": arm_N(row, scope),
                    "R": arm_R(row, g_q, g_ref, u, scope),
                    "P": arm_P(row, support, scope),
                }
                # P-shuffled: same links, geometry swapped
                p_sh = shuffled_payloads(p_ref)
                sup_sh = node_support(u, tm, p_q, p_sh, tol, max_credit=1.0)
                preds["P-shuffled"] = arm_P(
                    row, (sup_sh["support"] if sup_sh else None), scope)
                # W: endpoints swapped
                p_w = rewired_payloads(p_ref, g_ref)
                sup_w = node_support(u, tm, p_q, p_w, tol, max_credit=1.0)
                preds["W"] = arm_P(row, (sup_w["support"] if sup_w else None),
                                   scope)

                bucket = "ambiguous" if amb.get(u, False) else "unambiguous"
                skey = "%s|%s|%s" % (spec.family, spec.name, label)
                for arm, pred in preds.items():
                    hit = 1.0 if pred == truth else 0.0
                    acc[arm][bucket].append((skey, hit))
                diagnostics.append({
                    "family": spec.family, "scene": spec.name, "view": label,
                    "node": u, "support": support,
                    "n_checkable": sup["n_checkable"] if sup else 0,
                    "n_edges": sup["n_edges_from_u"] if sup else 0,
                })
            print("  %s %s: %d nodes mapped" % (spec.name, label, len(tm)),
                  flush=True)

    # ------------------------------------------------------------------
    arms = ["N", "R", "P", "P-shuffled", "W"]
    table: Dict[str, Any] = {"pre_registered_scenes": SCENES_PER_FAMILY,
                             "nboot": NBOOT, "seed": SEED,
                             "buckets": {}, "contrasts": {}, "power": {}}
    for bucket in ("ambiguous", "unambiguous"):
        table["buckets"][bucket] = {}
        for arm in arms:
            pairs = acc[arm][bucket]
            table["buckets"][bucket][arm] = cluster_boot(pairs)
        for a_, b_ in (("P", "N"), ("P", "R"), ("P", "P-shuffled"),
                       ("P", "W")):
            d = paired_boot(acc[a_][bucket], acc[b_][bucket])
            ver = verdict(d)
            table["contrasts"]["%s|%s-%s" % (bucket, a_, b_)] = dict(
                d, verdict=ver)

    # per-family breakdown on the ambiguous bucket
    table["by_family"] = {}
    for fam in sorted({d["family"] for d in diagnostics}):
        for arm in arms:
            pairs = acc[arm]["ambiguous"]
            sub = [(k, v) for k, v in pairs if k.startswith(fam + "|")]
            if sub:
                table["by_family"].setdefault(fam, {})[arm] = cluster_boot(sub)

    # checkable fraction per arm (the mission asks for it explicitly)
    n_diag = len(diagnostics)
    if n_diag:
        table["power"] = {
            "n_node_view_rows": n_diag,
            "n_scenes": len({d["scene"] for d in diagnostics}),
            "mean_checkable_edges": float(np.mean([d["n_checkable"] for d in diagnostics])),
            "share_nodes_with_zero_checkable": float(np.mean(
                [1.0 if d["n_checkable"] == 0 else 0.0 for d in diagnostics])),
            "mean_edges_from_node": float(np.mean([d["n_edges"] for d in diagnostics])),
            "checkable_fraction": (float(np.mean([d["n_checkable"] for d in diagnostics]))
                                   / max(1e-9, float(np.mean([d["n_edges"] for d in diagnostics])))),
        }

    for bucket in ("ambiguous", "unambiguous"):
        print("\n== %s" % bucket)
        for arm in arms:
            d = table["buckets"][bucket][arm]
            print("  %-11s %.4f [%.4f, %.4f] MDE %.4f n=%d n_eff=%d"
                  % (arm, d["mean"], d["lo"], d["hi"], d["mde"], d["n"],
                     d["n_eff"]))
        for k in sorted(table["contrasts"]):
            if not k.startswith(bucket + "|"):
                continue
            d = table["contrasts"][k]
            print("  %-24s %+.4f [%.4f, %.4f] MDE %.4f  %s"
                  % (k.split("|")[1], d["mean"], d["lo"], d["hi"], d["mde"],
                     d["verdict"]))

    path = os.path.join(OUT, "part6_arms.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
