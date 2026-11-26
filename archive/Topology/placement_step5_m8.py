"""MISSION 8 STEP 5 -- real images (ETH-80).  Non-gating, but required.

Synthetic DEV-5 scenes are one click away from the renderer that made them.
This step asks the questions only a real photograph can answer:

  * does the front end still build at all, on real images, with the flags on,
  * do the node counts and the place layer look like the synthetic ones (or did
    something collapse),
  * what does a graph cost in wall time,
  * does the placement population still ATTACH -- how many object nodes have a
    place within ATTACH_RADIUS px,
  * and as a sanity only: does the best variant retrieve the right object
    instance at least as often as N.

No tuning happens here.  Nothing here may change a frozen constant.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
ATTACH_RADIUS = 4.0
IMG_SIZE = 256


def load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def retrieval(imgs: List[dict], by_cls: Dict[str, List[dict]],
              q_by_cls: Dict[str, List[dict]], args: Any) -> Dict[str, Any]:
    """Sanity only: does the chosen arm rank the right OBJECT at least as often?

    One query image per class (the first TEST-rank instance, front view) against
    six candidates: three instances of its own class and three of the class that
    follows it.  N scores a pair by its mean best similarity; the arm scores the
    same pair by the blended score of the context-iterated mapping.  Nothing is
    tuned here.
    """
    import pickle
    from Topology import corpus
    from Topology import placement_census as CEN
    from Topology import placement_m3 as M3
    from Topology.placement_cache import get_or_build
    from matching import placement_edges as PE
    from matching import placement_context as Ctx
    from matching import placement_llr as L
    from matching.placement_scorer import incident_index, payload_index

    with open(os.path.join(OUT, "_fitcache", "m8_step2_fit.pkl"), "rb") as fh:
        F = pickle.load(fh)
    fit, pfit = F["fit"], F["pfit"]
    ae, ag = float(F["a_exist"]), float(F["a_geom"])
    M3.A_E, M3.A_G = ae, ag
    M3.CTX_MARGIN = float(F["ctx_margin"])
    M3.CTX_ITERS = int(F["ctx_iters"])
    M3.CTX_MUTUAL = bool(F["ctx_mutual"])

    def graph(e: dict) -> Any:
        u8 = load_u8(e["path"])
        return get_or_build(u8, "real|" + corpus.name_of(e)), u8

    classes = list(corpus.CLASSES)
    hits = {"N": 0, "P_plus_ctx": 0}
    n = 0
    print("\nRETRIEVAL SANITY (one query per class, 6 candidate refs)")
    for ci, cls in enumerate(classes):
        q = q_by_cls[cls][0]
        gq, q_u8 = graph(q)
        q_pay = PE.build_edges(gq.subgraph, q_u8,
                               lens_map=PE.lens_map_of(gq))
        q_idx, q_inc = payload_index(q_pay), incident_index(q_pay)
        q_nodes = sorted(int(x) for x in gq.subgraph.nodes)
        others = classes[(ci + 1) % len(classes)]
        cands_e = by_cls[cls][:3] + by_cls[others][:3]
        scores: Dict[str, Dict[str, float]] = {"N": {}, "P_plus_ctx": {}}
        for e in cands_e:
            gr, r_u8 = graph(e)
            r_pay = PE.build_edges(gr.subgraph, r_u8,
                                   lens_map=PE.lens_map_of(gr))
            r_idx = payload_index(r_pay)
            r_nodes = sorted(int(x) for x in gr.subgraph.nodes)
            rows = CEN.similarity_rows(gq, gr, q_nodes, r_nodes)
            pop = [u for u in q_nodes if rows.get(u)]
            if not pop:
                continue
            nmap = {u: int(rows[u][0][0]) for u in pop}
            scores["N"][corpus.name_of(e)] = float(
                np.mean([float(rows[u][0][1]) for u in pop]))
            cands = {u: [(int(v), float(s)) for (v, s) in rows[u][:M3.CAND_CAP]]
                     for u in pop}
            sim = {(u, v): s for u, cl in cands.items() for (v, s) in cl}

            def score(u, v, m, _sim=sim):
                t = L.plus_score(u, v, m, q_inc, q_idx, r_idx, fit, pfit, ae, ag)
                return ((1.0 - M3.EDGE_BETA) * _sim.get((int(u), int(v)), 0.0)
                        + M3.EDGE_BETA * t)

            cm = Ctx.iterate_context(pop, cands, nmap, score,
                                     iters=M3.CTX_ITERS, margin=M3.CTX_MARGIN,
                                     mutual=M3.CTX_MUTUAL)
            scores["P_plus_ctx"][corpus.name_of(e)] = float(np.mean(
                [score(u, int(cm[u]), cm) for u in pop if u in cm]))
        row = {}
        for arm in ("N", "P_plus_ctx"):
            if not scores[arm]:
                continue
            best = max(scores[arm], key=lambda k: scores[arm][k])
            ok = best.startswith(cls)
            hits[arm] += 1 if ok else 0
            row[arm] = best
        n += 1
        print("  %-8s N -> %-14s  arm -> %-14s"
              % (cls, row.get("N", "-"), row.get("P_plus_ctx", "-")))
    out = {"queries": n,
           "top1_same_object": {k: (hits[k] / n) if n else float("nan")
                                for k in hits}}
    print("  top-1 retrieval  N %.3f   P_plus_ctx %.3f"
          % (out["top1_same_object"]["N"], out["top1_same_object"]["P_plus_ctx"]))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--tag", default="m8_real")
    ap.add_argument("--retrieval", action="store_true",
                    help="also run the retrieval sanity (N vs the chosen arm)")
    args = ap.parse_args()

    import guard
    guard.activate()
    from Topology import corpus
    from Topology.placement_cache import get_or_build
    from matching import placement_edges as PE
    from matching.placement_scorer import incident_index, payload_index
    from matching import placement_llr as L
    from matching import placement_context as Ctx
    from Topology.placement_census import lens_of
    from Topology import placement_arms2 as A2
    from core.place_key import place_lenses

    spl = corpus.split()
    refs = [e for e in spl["train"] if e["view"] == "-090-000"]
    queries = [e for e in spl["test"] if e["view"] == "-090-000"]
    by_cls: Dict[str, List[dict]] = defaultdict(list)
    for e in refs:
        by_cls[e["class"]].append(e)
    q_by_cls: Dict[str, List[dict]] = defaultdict(list)
    for e in queries:
        q_by_cls[e["class"]].append(e)

    A2.ENDPOINT_LENSES = tuple(place_lenses())
    place_set = set(place_lenses())

    counts = {"off": [], "on": []}
    times = {"off": [], "on": []}
    attach: List[float] = []
    dists: List[float] = []
    per_lens_occ: Dict[str, List[int]] = defaultdict(list)
    n_place_nodes: List[int] = []
    build_cache: Dict[str, Any] = {}

    imgs: List[dict] = []
    for cls in corpus.CLASSES:
        imgs.extend(q_by_cls[cls][:args.per_class])
        imgs.extend(by_cls[cls][:args.per_class])

    print("ETH-80 real images: %d\n" % len(imgs))
    import eth80_frontend as F
    for e in imgs:
        u8 = load_u8(e["path"])
        key = corpus.name_of(e)
        # flags ON (the frozen configuration)
        t0 = time.time()
        g = get_or_build(u8, "real|on|" + key)
        dt = time.time() - t0
        build_cache[key] = g
        times["on"].append(dt)
        counts["on"].append((len(g.subgraph.nodes), len(g.subgraph.edges)))
        kp = 0
        for n in g.subgraph.nodes:
            Lm = str(lens_of(g, n))
            per_lens_occ[Lm].append(1)
            if Lm in place_set:
                kp += 1
        n_place_nodes.append(kp)
        # attach: object nodes without a place must find one within R px
        kps = np.array([(float(g.subgraph.anchor_of_node[n][0]),
                         float(g.subgraph.anchor_of_node[n][1]))
                        for n in g.subgraph.nodes
                        if n in g.subgraph.anchor_of_node
                        and str(lens_of(g, n)) in place_set], dtype=float)
        if len(kps):
            for n in g.subgraph.nodes:
                a = g.subgraph.anchor_of_node.get(n)
                if a is None or str(lens_of(g, n)) in place_set:
                    continue
                d = float(np.min(np.hypot(kps[:, 0] - float(a[0]),
                                          kps[:, 1] - float(a[1]))))
                attach.append(1.0 if d <= ATTACH_RADIUS else 0.0)
                dists.append(d)

    print("%-22s %8s %8s %10s %8s" % ("image", "nodes", "edges",
                                      "place nodes", "build s"))
    for e, c, t, k in zip(imgs, counts["on"], times["on"], n_place_nodes):
        print("  %-20s %8d %8d %10d %8.1f"
              % (corpus.name_of(e), c[0], c[1], k, t))

    res: Dict[str, Any] = {
        "images": len(imgs),
        "nodes": {"mean": float(np.mean([c[0] for c in counts["on"]])),
                  "min": int(min(c[0] for c in counts["on"])),
                  "max": int(max(c[0] for c in counts["on"]))},
        "edges": {"mean": float(np.mean([c[1] for c in counts["on"]]))},
        "place_nodes": {"mean": float(np.mean(n_place_nodes))},
        "build_seconds": {"mean": float(np.mean(times["on"])),
                          "max": float(np.max(times["on"]))},
        "attach_coverage": {
            "radius_px": ATTACH_RADIUS,
            "n": len(attach),
            "coverage": float(np.mean(attach)) if attach else float("nan"),
            "median_dist": float(np.median(dists)) if dists else float("nan")},
        "per_lens": {k: len(v) for k, v in sorted(per_lens_occ.items())},
        "collapsed": [k for k, v in per_lens_occ.items()
                      if len(v) < max(5, 0.02 * sum(len(x) for x in
                                                    per_lens_occ.values()))],
    }
    print("\nnodes %.1f  edges %.1f  place nodes %.1f  build %.1f s"
          % (res["nodes"]["mean"], res["edges"]["mean"],
             res["place_nodes"]["mean"], res["build_seconds"]["mean"]))
    print("attach coverage %.3f (n %d)  median %.2f px"
          % (res["attach_coverage"]["coverage"], res["attach_coverage"]["n"],
             res["attach_coverage"]["median_dist"]))
    print("lenses present: %d   collapsed: %s"
          % (len(res["per_lens"]), res["collapsed"] or "none"))

    if args.retrieval:
        res["retrieval"] = retrieval(imgs, by_cls, q_by_cls, args)

    path = os.path.join(OUT, "%s.json" % args.tag)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
