"""M29: what the representation is, and how much it encodes.

Three censuses on one page, all read-only:

  --registry   the declared vocabulary: node classes and their slots/bins/circular cycles and
               population axes; edge classes and their parameter slots.  Nominal bits per class.
  --graphs     what the front end actually emits, on a fixed sample: nodes per image by class,
               edges per image by type, node degree, the fraction of edge slots actually measured
               (a slot the writer could not measure is ABSENT and carries no information).
  --info       what a node's content actually says: from the committed records, H(key), H(key |
               object) and the mutual information I(key; object) in bits, per lens and pooled,
               against H(object) = log2(80) = 6.32 bits and against a pixel-side ceiling (the
               image's own 4x4x4 colour + 12-bin gradient histogram, which is what the door's
               appearance channel uses).
  --edges      the matcher's own split of its score into the node term and the edge term, on a
               fixed pair (score with the shipped edge weight against score with it set to 0).

    python Topology/m29_representation.py --registry --info
    python Topology/m29_representation.py --graphs --edges
    python Topology/m29_representation.py --report      # everything -> m29_representation.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement", "m29_representation.json")
REF = "000-000"
VIEW = "035-045"

#: the graph sample: one view of six objects, one per class and one repeat (the repeat is what
#: makes the "does the same class emit the same vocabulary" question askable).
SAMPLE = ("apple/1", "car/1", "cow/1", "cup/1", "dog/1", "apple/2")


# --------------------------------------------------------------------------- entropy helpers
def _entropy(counts: Sequence[int]) -> float:
    n = float(sum(counts))
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def _cond_entropy(joint: Dict[Tuple[Any, Any], int], by_given: Dict[Any, int],
                  given_totals: Dict[Any, int]) -> float:
    """H(X | Y) from a joint table: sum_y p(y) * H(X | y)."""
    n = float(sum(joint.values()))
    if n <= 0:
        return 0.0
    h = 0.0
    for y, ny in given_totals.items():
        xs = [c for (x, yy), c in joint.items() if yy == y]
        h += (ny / n) * _entropy(xs)
    return h


# --------------------------------------------------------------------------- 1. the vocabulary
def registry() -> Dict[str, Any]:
    import core.registries as R
    import core.schema as S

    circ = getattr(R, "CIRCULAR_CYCLE_LENGTHS", {})
    node_rows: List[Dict[str, Any]] = []
    for cls, ranges in sorted(R.CONSTITUTIONAL_RANGES.items()):
        bins = [int(round(2.0 * float(r))) if float(r) > 0 else 1 for r in ranges]
        axes = R.POPULATION_AXES.get(cls)
        node_rows.append({
            "class": cls, "arity": len(ranges), "bins": bins,
            "circular": {str(k): v for k, v in (circ.get(cls) or {}).items()},
            "nominal_bits": float(sum(math.log2(max(2, b)) for b in bins)),
            "population": ([{"axes": [a.name for a in axes],
                             "bins": [int(a.n_bins) for a in axes],
                             "weights": [float(a.weight) for a in axes],
                             "pop_bits": float(sum(math.log2(max(2, int(a.n_bins))) * a.weight
                                                   for a in axes))}] if axes else []),
        })
    edge_rows: List[Dict[str, Any]] = []
    for cls, ranges in sorted(S.CONSTITUTIONAL_EDGE_RANGES.items()):
        bins = [int(round(2.0 * float(r))) if float(r) > 0 else 1 for r in ranges]
        edge_rows.append({"class": cls, "arity": len(ranges), "bins": bins,
                          "nominal_bits": float(sum(math.log2(max(2, b)) for b in bins)),
                          "circular": {str(k): v for k, v in
                                       (S.CIRCULAR_EDGE_CYCLE_LENGTHS.get(cls) or {}).items()}})
    lens = [d.name for d in __import__("lenses.derived_common", fromlist=["x"]).LENS_INPUT_DECLARATIONS]
    out = {
        "node_classes": len(node_rows), "edge_classes": len(edge_rows),
        "node_arity": {"min": min(r["arity"] for r in node_rows),
                       "max": max(r["arity"] for r in node_rows),
                       "mean": round(float(np.mean([r["arity"] for r in node_rows])), 2)},
        "edge_arity": {"min": min(r["arity"] for r in edge_rows),
                       "max": max(r["arity"] for r in edge_rows),
                       "mean": round(float(np.mean([r["arity"] for r in edge_rows])), 2)},
        "node_bits": {"mean": round(float(np.mean([r["nominal_bits"] for r in node_rows])), 2),
                      "max": round(max(r["nominal_bits"] for r in node_rows), 2),
                      "total_over_classes": round(sum(r["nominal_bits"] for r in node_rows), 1)},
        "edge_bits": {"mean": round(float(np.mean([r["nominal_bits"] for r in edge_rows])), 2),
                      "max": round(max(r["nominal_bits"] for r in edge_rows), 2),
                      "total_over_classes": round(sum(r["nominal_bits"] for r in edge_rows), 1)},
        "with_population": sorted(R.POPULATION_AXES),
        "lenses_declared": lens,
        "node_rows": node_rows, "edge_rows": edge_rows,
    }
    print("registry: %d node classes (mean %.1f slots, mean %.1f nominal bits), %d edge classes "
          "(mean %.1f slots, %.1f bits), %d lens tiers, %d classes with population axes"
          % (out["node_classes"], out["node_arity"]["mean"], out["node_bits"]["mean"],
             out["edge_classes"], out["edge_arity"]["mean"], out["edge_bits"]["mean"],
             len(lens), len(out["with_population"])))
    return out


# --------------------------------------------------------------------------- 2. what is emitted
def graphs(sample: Sequence[str] = SAMPLE, view: str = REF) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    from Topology import m26_records as REC
    objs = {o["key"]: o for o in REC.objects()}
    node_classes: Counter = Counter()
    node_sigs: Dict[str, Counter] = defaultdict(Counter)
    node_measured: Dict[str, List[int]] = defaultdict(list)
    node_pop: Dict[str, List[int]] = defaultdict(list)
    edge_types: Counter = Counter()
    edge_tuples: Dict[str, Counter] = defaultdict(Counter)
    edge_conf: Dict[str, List[float]] = defaultdict(list)
    edge_absent: Counter = Counter()
    edge_slots: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # measured, declared
    degrees: List[int] = []
    per_image: List[Dict[str, Any]] = []
    t0 = time.time()
    from core.registries import ABSENT
    import core.schema as S
    for k in sample:
        o = objs[k]
        _img, g, _side = FE.cell(REC.path_of(o, view), "m29census")
        sub = g.subgraph
        nodes = list(sub.nodes)
        n_by: Counter = Counter()
        for n in nodes:
            pair = g.sigs.get(n)
            if not pair:
                n_by["none"] += 1
                node_classes["none"] += 1
                continue
            cls, sig = pair
            n_by[cls] += 1
            node_classes[cls] += 1
            q = tuple(getattr(sig, "quantized", ()) or ())
            node_sigs[cls][q] += 1
            node_measured[cls].append(sum(1 for x in q if x != ABSENT))
            node_pop[cls].append(1 if getattr(sig, "population", ()) else 0)
        deg: Counter = Counter()
        e_by: Counter = Counter()
        for (u, v, et) in sub.edges:
            name = getattr(et, "name", str(et))
            e_by[name] += 1
            edge_types[name] += 1
            deg[u] += 1
            deg[v] += 1
        for key, params in sub.edge_params.items():
            name = getattr(key[2], "name", str(key[2]))
            q = tuple(getattr(params, "quantized", ()) or ())
            edge_tuples[name][q] += 1
            edge_conf[name].append(float(getattr(params, "confidence", 0.0) or 0.0))
            declared = len(S.CONSTITUTIONAL_EDGE_RANGES.get(name, ())) or len(q)
            edge_slots[name][0] += sum(1 for x in q if x != ABSENT)
            edge_slots[name][1] += declared
            if all(x == ABSENT for x in q) or not q:
                edge_absent[name] += 1
        degrees.extend(list(deg.values()) + [0] * max(0, len(nodes) - len(deg)))
        per_image.append({"object": k, "nodes": len(nodes), "edges": int(sum(e_by.values())),
                          "edges_with_params": len(sub.edge_params),
                          "nodes_by_class": dict(n_by.most_common()),
                          "edges_by_type": dict(e_by.most_common())})
        print("  %-10s nodes %4d  edges %5d  params %5d  %s"
              % (k, len(nodes), sum(e_by.values()), len(sub.edge_params),
                 dict(n_by.most_common(4))), flush=True)

    tot_nodes = sum(p["nodes"] for p in per_image)
    tot_edges = sum(p["edges"] for p in per_image)
    per_class = {}
    for c, n in edge_types.most_common():
        tk = edge_tuples.get(c) or Counter()
        per_class[c] = {
            "edges_per_image": round(n / max(1, len(sample)), 1),
            "param_tuples_distinct": len(tk),
            "H_params_bits": round(_entropy(list(tk.values())), 2) if tk else None,
            "slots_measured_frac": (round(edge_slots[c][0] / edge_slots[c][1], 3)
                                    if edge_slots[c][1] else None),
            "all_absent_frac": round(edge_absent[c] / max(1, n), 3),
            "mean_confidence": (round(float(np.mean(edge_conf[c])), 3)
                                if edge_conf.get(c) else None)}
    out = {"sample": list(sample), "view": view, "images": len(sample),
           "nodes_per_image": round(tot_nodes / max(1, len(sample)), 1),
           "edges_per_image": round(tot_edges / max(1, len(sample)), 1),
           "edges_per_node": round(tot_edges / max(1, tot_nodes), 2),
           "node_classes_emitted": len(node_classes),
           "edge_types_emitted": len(edge_types),
           "node_class_share": {c: round(n / max(1, tot_nodes), 3)
                                for c, n in node_classes.most_common()},
           "edge_type_share": {c: round(n / max(1, tot_edges), 3)
                               for c, n in edge_types.most_common()},
           "edge_type_mean_per_image": per_class,
           "degree": {"mean": round(float(np.mean(degrees)), 2) if degrees else None,
                      "median": float(np.median(degrees)) if degrees else None,
                      "max": int(max(degrees)) if degrees else None,
                      "frac_isolated": round(float(np.mean([d == 0 for d in degrees])), 3)
                      if degrees else None},
           "node_classes_table": {
               c: {"nodes_per_image": round(n / max(1, len(sample)), 1),
                   "distinct_signatures": len(node_sigs[c]),
                   "H_signature_bits": round(_entropy(list(node_sigs[c].values())), 2),
                   "slots_measured": (round(float(np.mean(node_measured[c])), 1)
                                      if node_measured.get(c) else None),
                   "payload_frac": (round(float(np.mean(node_pop[c])), 3)
                                    if node_pop.get(c) else None)}
               for c, n in node_classes.most_common()},
           "per_image": per_image, "seconds": round(time.time() - t0, 1)}
    print("graphs: %.1f nodes, %.1f edges per image (%.2f edges/node), %d node classes, "
          "%d edge types, median degree %s"
          % (out["nodes_per_image"], out["edges_per_image"], out["edges_per_node"],
             out["node_classes_emitted"], out["edge_types_emitted"], out["degree"]["median"]))
    return out


# --------------------------------------------------------------------------- 3. what it says
def _hist_key(img: np.ndarray) -> Tuple[int, ...]:
    """The door's appearance channel: 4x4x4 colour x 12-bin gradient, as a coarse key."""
    import numpy as np
    small = img.reshape(4, img.shape[0] // 4, 4, img.shape[1] // 4, 3).mean(axis=(1, 3))
    rgb = np.clip((small / 255.0 * 4).astype(int), 0, 3).reshape(-1)
    g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % math.pi
    sel = mag > np.percentile(mag, 80)
    hist = np.histogram(ang[sel], bins=12, range=(0, math.pi))[0]
    hist = tuple(int(round(4 * h / max(1, hist.sum()))) for h in hist)
    return tuple(int(v) for v in rgb) + hist


def info() -> Dict[str, Any]:
    """I(key; object) in bits, from the committed records; plus the pixel-side comparison."""
    from matching import placement_records as PR
    from Topology import m26_records as REC
    from matching.placement_frontend import load_image
    objs = REC.objects()
    t0 = time.time()
    cells = PR.load_cells(os.path.join(ROOT, "data", "placement_records_m28.npz"), [REF, VIEW])
    keys = [o["key"] for o in objs]
    out: Dict[str, Any] = {"n_objects": len(keys), "H_object_bits": round(math.log2(len(keys)), 2),
                           "views": {}}
    for view in (REF, VIEW):
        joint: Dict[Tuple[Any, Any], int] = defaultdict(int)
        by_obj: Dict[str, int] = defaultdict(int)
        per_lens_joint: Dict[str, Dict[Tuple[Any, Any], int]] = defaultdict(lambda: defaultdict(int))
        per_lens_by: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        key_counts: Counter = Counter()
        per_object_keys: Dict[str, set] = defaultdict(set)
        per_object_key_entropy: List[float] = []
        for k in keys:
            rec = cells[(k, view)]
            counts = Counter()
            for p in rec["places"]:
                key = (p["lens"], p["key"][1])
                joint[(key, k)] += 1
                by_obj[k] += 1
                per_lens_joint[p["lens"]][(key, k)] += 1
                per_lens_by[p["lens"]][k] += 1
                key_counts[key] += 1
                per_object_keys[k].add(key)
                counts[key] += 1
            per_object_key_entropy.append(_entropy(list(counts.values())))
        H = _entropy(list(key_counts.values()))
        Hcond = _cond_entropy(joint, by_obj, by_obj)
        lenses: Dict[str, Any] = {}
        for lens in ("Blob", "BoundaryArc", "CornerPoint"):
            lk = Counter()
            for (key, k), c in per_lens_joint[lens].items():
                lk[key] += c
            if not lk:
                continue
            lenses[lens] = {"places": int(sum(lk.values())), "distinct_keys": len(lk),
                            "H_key_bits": round(_entropy(list(lk.values())), 3),
                            "H_key_given_object_bits": round(
                                _cond_entropy(dict(per_lens_joint[lens]), dict(per_lens_by[lens]),
                                              dict(per_lens_by[lens])), 3),
                            "I_key_object_bits": round(
                                _entropy(list(lk.values()))
                                - _cond_entropy(dict(per_lens_joint[lens]),
                                                dict(per_lens_by[lens]),
                                                dict(per_lens_by[lens])), 3)}
        # the pixel-side comparison on the same objects, same view
        hist_joint: Dict[Tuple[Any, Any], int] = defaultdict(int)
        hist_by: Dict[str, int] = defaultdict(int)
        for o in objs:
            hk = _hist_key(load_image(REC.path_of(o, view)))
            hist_joint[(hk, o["key"])] += 1
            hist_by[o["key"]] += 1
        H_hist = _entropy([c for (_h, _k), c in hist_joint.items()])
        H_hist_cond = _cond_entropy(dict(hist_joint), dict(hist_by), dict(hist_by))
        out["views"][view] = {
            "places": int(sum(key_counts.values())),
            "distinct_keys": len(key_counts),
            "H_key_bits": round(H, 3),
            "H_key_given_object_bits": round(Hcond, 3),
            "I_key_object_bits": round(H - Hcond, 3),
            "distinct_keys_per_object": {"mean": round(float(np.mean(
                [len(v) for v in per_object_keys.values()])), 1),
                "median": float(np.median([len(v) for v in per_object_keys.values()]))},
            "H_key_within_object_bits": {"mean": round(float(np.mean(per_object_key_entropy)), 3),
                                         "median": round(float(np.median(per_object_key_entropy)), 3)},
            "by_lens": lenses,
            "pixel_ceiling": {"appearance_key_distinct": len({h for (h, _k) in hist_joint}),
                              "H_bits": round(H_hist, 3),
                              "H_given_object_bits": round(H_hist_cond, 3),
                              "I_key_object_bits": round(H_hist - H_hist_cond, 3)},
        }
        print("  %s: %d places, %d keys, H %.2f bits, I(key;object) %.2f of %.2f bits "
              "(pixel side %.2f)" % (view, out["views"][view]["places"],
                                     out["views"][view]["distinct_keys"],
                                     out["views"][view]["H_key_bits"],
                                     out["views"][view]["I_key_object_bits"],
                                     out["views"][view]["H_object_bits"]
                                     if "H_object_bits" in out["views"][view]
                                     else out["H_object_bits"],
                                     out["views"][view]["pixel_ceiling"]["I_key_object_bits"]),
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- 4. the edge term
def edges() -> Dict[str, Any]:
    """The matcher's own node/edge split: score with the shipped edge weight vs weight 0."""
    import parameters as P
    import matching.placement_frontend as FE
    from Topology import m26_records as REC
    from Topology.m27_profile import committed_pairs
    FE.apply_recipe()
    pairs = committed_pairs(8)
    rows: List[Dict[str, Any]] = []
    for q, r, _ref in pairs:
        base = FE.pair_score(q, r)
        w = float(getattr(P, "MATCH_EDGE_WEIGHT"))
        old = P.MATCH_EDGE_WEIGHT
        try:
            P.MATCH_EDGE_WEIGHT = 0.0
            node_only = FE.pair_score(q, r)
        finally:
            P.MATCH_EDGE_WEIGHT = old
        rows.append({"q": os.path.basename(q), "r": os.path.basename(r),
                     "score": base, "score_edge_weight_0": node_only,
                     "edge_share": (1.0 - node_only / base) if base else None,
                     "edge_weight": w})
        print("  %s vs %s: %.6f (edges 0 -> %.6f, share %.1f %%)"
              % (rows[-1]["q"], rows[-1]["r"], base, node_only,
                 100.0 * (rows[-1]["edge_share"] or 0.0)), flush=True)
    return {"pairs": rows,
            "edge_share_mean": round(float(np.mean([r["edge_share"] for r in rows
                                                    if r["edge_share"] is not None])), 3)}


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for f in ("registry", "graphs", "info", "edges"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    data: Dict[str, Any] = {}
    if os.path.exists(OUT):
        data = json.load(open(OUT))
    if a.report:
        a.registry = a.graphs = a.info = a.edges = True
    if a.registry:
        data["registry"] = registry()
    if a.graphs:
        data["graphs"] = graphs()
    if a.info:
        data["info"] = info()
    if a.edges:
        data["edge_term"] = edges()
    if any((a.registry, a.graphs, a.info, a.edges)):
        data["manifest"] = {"environment": _env(), "view": REF, "query_view": VIEW,
                            "sample": list(SAMPLE)}
        with open(OUT, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True, default=float)
        print("wrote", OUT)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
