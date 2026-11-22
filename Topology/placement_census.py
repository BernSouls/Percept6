"""Census on DEV: does the new edge class actually reach CHECKABLE nodes?

Three quantities the mission requires BEFORE any final run:

1. PAIR SURVIVAL -- of the base edges, what fraction has a counterpart edge in
   the reference with BOTH endpoints tracked?  Emitting edges is worthless if
   they do not survive to the other view.

2. COVERAGE on AMBIGUOUS nodes -- the share of ambiguous nodes that has at
   least one checkable placement edge, measured UNDER N'S FROZEN MAPPING, so
   coverage cannot be raised by mapping differently.  Target 50 %+, floor 25 %.

3. LANDMARK SHARE for TEXTURE nodes -- periodic texture is only disambiguated
   by an edge to an APERIODIC landmark (boundary, corner, junction, event), so
   the share of texture nodes that has one is reported.

"Checkable" is the scorer's own definition and is never relaxed here: an edge
is checkable only if the neighbour is mapped and the counterpart edge exists,
because those are the conditions under which `node_support` returns anything.

Run:  .venv/bin/python Topology/placement_census.py [--eval-spec]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement")

#: Lenses that are APERIODIC landmarks: an edge to one of these is what can
#: break the period of a texture, so they are counted separately.
LANDMARK_LENSES = frozenset({
    "Corner", "Boundary", "Onset", "IllusoryContour", "EndStopped",
    "CurveComplex", "Contour", "ContourFragment", "RegionShape",
})
#: Lenses that ARE a periodic texture family.
TEXTURE_LENSES = frozenset({"Texture", "TextureBoundary"})
#: Content lenses (used by Part 0c); reported for completeness.
CONTENT_LENSES = frozenset({"Texture", "TextureBoundary", "Color",
                            "Surface", "RegionShape"})


# ---------------------------------------------------------------------------
def lens_of(g: Any, n: int) -> str:
    s = g.sigs.get(n)
    return str(s[0]) if s is not None else "?"


def similarity_rows(g_q: Any, g_r: Any, nodes_q: List[int],
                    nodes_r: List[int]) -> Dict[int, List[Tuple[int, float]]]:
    """q node -> [(ref node, similarity)] sorted best first, deterministic."""
    out: Dict[int, List[Tuple[int, float]]] = {}
    by_lens: Dict[str, List[int]] = defaultdict(list)
    for v in nodes_r:
        by_lens[lens_of(g_r, v)].append(int(v))
    for u in nodes_q:
        su = g_q.sigs.get(u)
        if su is None:
            continue
        row: List[Tuple[int, float]] = []
        for v in by_lens.get(lens_of(g_q, u), ()):
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


def n_mapping(rows: Dict[int, List[Tuple[int, float]]]) -> Dict[int, int]:
    """N's frozen mapping: the best-similarity same-lens ref node.

    Deliberately the same rule the N arm uses, and deliberately computed here
    from the SAME rows, so coverage is measured under the mapping the arm will
    actually have.
    """
    out: Dict[int, int] = {}
    for u in sorted(rows):
        if rows[u]:
            out[int(u)] = int(rows[u][0][0])
    return out


def ambiguous_nodes(g_q: Any, g_r: Any,
                    rows: Dict[int, List[Tuple[int, float]]],
                    top_sim: float = 0.99,
                    margin_max: float = 0.05) -> Dict[int, bool]:
    """Part 4's observable near-tie rule, with the similarity rows supplied.

    (Supplying the rows matters: without them only the exact-signature route
    runs, every signature in these graphs is unique, and the label is empty.)
    """
    from Topology.placement_gate import appearance_ambiguous
    sims = {u: rows[u] for u in rows}
    return appearance_ambiguous(g_q.sigs, g_r.sigs, sims=sims)


def checkable_edges(u: int, nmap: Dict[int, int],
                    q_pay: Dict[Tuple, Any],
                    r_pay: Dict[Tuple, Any],
                    by_src: Dict[int, List[Tuple[int, Any]]],
                    by_dst: Dict[int, List[Tuple[int, Any]]]) -> List[Tuple]:
    """The edges incident to u that can actually be checked under `nmap`.

    Both directions count (a payload is framed in its source, so an incoming
    edge is framed in the neighbour's own frame -- still legitimate evidence),
    and a neighbour with no counterpart edge contributes nothing.
    """
    out: List[Tuple] = []
    mu = nmap.get(int(u))
    if mu is None:
        return out
    for (other, key) in by_src.get(int(u), ()):
        mo = nmap.get(other)
        if mo is None:
            continue
        for k in (key,):
            s, d, et = k
            if (int(mu), int(mo), et) in r_pay:
                out.append(k)
    for (other, key) in by_dst.get(int(u), ()):
        mo = nmap.get(other)
        if mo is None:
            continue
        s, d, et = key
        if (int(mo), int(mu), et) in r_pay:
            out.append(key)
    return out


def index_payloads(pay: Dict[Tuple, Any]):
    by_src: Dict[int, List[Tuple[int, Any]]] = defaultdict(list)
    by_dst: Dict[int, List[Tuple[int, Any]]] = defaultdict(list)
    for key in sorted(pay, key=lambda k: (k[0], k[1], getattr(k[2], "name", ""))):
        s, d, et = key
        by_src[int(s)].append((int(d), key))
        by_dst[int(d)].append((int(s), key))
    return by_src, by_dst


# ---------------------------------------------------------------------------
def object_mask(g: Any, render: Any) -> Dict[int, bool]:
    """node -> does its anchor sit ON an object (not the background)?

    This is the evaluation population, and it is a PHYSICAL statement, not a
    convenience: a background lattice point has no orientation to read (median
    coherence measured 0.000) and is not a feature of any object, so there is
    nothing to correspond and no frame to carry.  Measured on one DEV view:
    171 of 225 "ambiguous" nodes were background points with coherence 0.000,
    while on-object ambiguous nodes had coherence 0.975.  Reporting both
    populations keeps the choice visible.
    """
    from Topology.placement_truth import object_id_at
    out: Dict[int, bool] = {}
    for n in sorted(int(x) for x in g.subgraph.nodes):
        out[n] = object_id_at(render, g.subgraph.anchor_of_node.get(n)) >= 0
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-spec", action="store_true",
                    help="run on the PRE-REGISTERED EVAL set instead of DEV")
    args = ap.parse_args(argv)

    import guard
    guard.activate()
    from matching import placement_edges as PE
    from matching.placement import edge_payloads
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import track_optimal
    from Topology.placement_scenes import posed

    if args.eval_spec:
        from Topology.placement_eval_spec import eval_specs, EVAL_VIEWS as VIEWS
        specs = eval_specs()
        out_path = os.path.join(OUT, "census_eval.json")
    else:
        from Topology.placement_dev import all_specs, DEV_VIEWS as VIEWS
        specs = all_specs()
        out_path = os.path.join(OUT, "census_dev.json")

    per_view: List[Dict[str, Any]] = []
    agg: Dict[str, List[float]] = defaultdict(list)

    for spec in specs:
        sc0, r0 = spec.scene, spec.scene.render()
        g_ref = get_or_build(r0.image_u8, spec.name + "|ref")
        ref_nodes = sorted(int(n) for n in g_ref.subgraph.nodes)
        r_pay = PE.build_edges(g_ref.subgraph, r0.image_u8)
        ref_obj = object_mask(g_ref, r0)

        for label, view in VIEWS:
            if label == "v0_base":
                continue
            sc1 = posed(spec.scene, float(view["rot_deg"]),
                        tuple(view["shift"]), float(view["scale"]))
            r1 = sc1.render()
            g_q = get_or_build(r1.image_u8, spec.name + "|" + label)
            q_nodes = sorted(int(n) for n in g_q.subgraph.nodes)
            q_pay = PE.build_edges(g_q.subgraph, r1.image_u8)
            q_obj = object_mask(g_q, r1)

            fwd = track_optimal(sc0, r0, sc1, r1, g_ref, g_q)
            truth = {int(q): int(r) for r, q in fwd.items()}
            rows = similarity_rows(g_q, g_ref, q_nodes, ref_nodes)
            nmap = n_mapping(rows)
            amb = ambiguous_nodes(g_q, g_ref, rows)
            q_by_src, q_by_dst = index_payloads(q_pay)

            def cov(pop: List[int], mp: Dict[int, int]) -> float:
                if not pop:
                    return float("nan")
                n = 0
                for u in pop:
                    if checkable_edges(u, mp, q_pay, r_pay, q_by_src, q_by_dst):
                        n += 1
                return n / len(pop)

            amb_all = [u for u in q_nodes if amb.get(u, False)]
            amb_obj = [u for u in amb_all if q_obj.get(u)]
            una_obj = [u for u in q_nodes
                       if not amb.get(u, False) and q_obj.get(u)]

            num = den = 0
            for (sv, dv, et) in sorted(r_pay, key=lambda k: (k[0], k[1])):
                if int(sv) not in fwd or int(dv) not in fwd:
                    continue
                den += 1
                if (int(fwd[int(sv)]), int(fwd[int(dv)]), et) in q_pay:
                    num += 1

            tex = [u for u in q_nodes
                   if lens_of(g_q, u) in TEXTURE_LENSES and q_obj.get(u)]
            # `q_by_src[u]` holds (NEIGHBOUR, key) pairs; the neighbour is
            # element 0.  Taking element 1 (the key) here made every lookup
            # resolve to "?" and reported a landmark share of 0.000 that was
            # purely an indexing bug.
            tex_lm = [u for u in tex
                      if any(lens_of(g_q, nb) in LANDMARK_LENSES
                             for nb in [k[0] for k in q_by_src.get(u, ())]
                             + [k[0] for k in q_by_dst.get(u, ())])]

            row = {
                "scene": spec.name, "family": spec.family, "view": label,
                "nodes_q": len(q_nodes),
                "nodes_q_on_object": sum(1 for u in q_nodes if q_obj.get(u)),
                "ambiguous_all": len(amb_all),
                "ambiguous_on_object": len(amb_obj),
                "unambiguous_on_object": len(una_obj),
                "coverage_ambiguous_object_N": cov(amb_obj, nmap),
                "coverage_ambiguous_object_truth": cov(amb_obj, truth),
                "coverage_ambiguous_all_N": cov(amb_all, nmap),
                "coverage_unambiguous_object_N": cov(una_obj, nmap),
                "survival": (num / den if den else float("nan")),
                "survival_den": den,
                "n_payload_ref": len(r_pay), "n_payload_q": len(q_pay),
                "texture_nodes_on_object": len(tex),
                "texture_with_landmark": len(tex_lm),
                "landmark_share_texture": (len(tex_lm) / len(tex)
                                           if tex else float("nan")),
                "truth_cover": (len(truth) / max(1, len(q_nodes))),
                "shipped_payload_edges": len(edge_payloads(g_ref)),
            }
            per_view.append(row)
            print("  %-15s %-12s ambObj=%3d covN=%5.1f%% covT=%5.1f%% "
                  "surv=%5.1f%% tex-lm=%5.1f%%"
                  % (spec.name, label, row["ambiguous_on_object"],
                     100 * (row["coverage_ambiguous_object_N"] or 0),
                     100 * (row["coverage_ambiguous_object_truth"] or 0),
                     100 * (row["survival"] or 0),
                     100 * (row["landmark_share_texture"] or 0)), flush=True)
            for k in ("coverage_ambiguous_object_N",
                      "coverage_ambiguous_object_truth",
                      "coverage_ambiguous_all_N", "survival",
                      "landmark_share_texture"):
                v = row[k]
                if isinstance(v, float) and math.isfinite(v):
                    agg[k].append(v)

    summary = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                   "max": float(np.max(v)), "n": len(v)}
               for k, v in agg.items()}
    print("\nSUMMARY over %d scene-views" % len(per_view))
    for k in sorted(summary):
        s_ = summary[k]
        print("  %-32s mean %.3f  [%.3f, %.3f]" % (k, s_["mean"], s_["min"],
                                                   s_["max"]))
    with open(out_path, "w") as fh:
        json.dump({"summary": summary, "per_view": per_view}, fh, indent=2,
                  sort_keys=True)
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
