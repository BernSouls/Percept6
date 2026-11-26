"""Task 4 PLACEMENT mission — Part 1: reach and landmark coverage.

Runs BEFORE the scorer.  Answers:

1a  reach: on the placement-edge graph, what fraction of frame nodes is
    reachable from a seed, the component sizes, degree, the share with zero
    placement edges, and the SPAN distribution (edge length, not
    nearest-neighbour spacing).  The historic stage-5 census said 16 % reach
    against a 25 % floor; it is superseded, so this re-measures.

1b  ambiguity period and landmark share: the sine period is 128/texture_freq
    px and the checker parity period is 2x that, at the scene's ACTUAL
    pixel_scale and depth.  A periodic region is broken by an edge to an
    APERIODIC landmark, not by a longer texture-to-texture edge.  Report the
    share of texture nodes with a landmark edge in reach.

Outputs JSON to Topology/results/placement/part1_reach.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

#: Bare ``EdgeType`` member names (see ``_etype_name``).
PLACEMENT_CLASSES = (
    "ORIENTATION_CONTINUITY",
    "BOUNDARY_ATTACHMENT",
    "CURVE_CONTAINMENT",
    "PART_CONTAINMENT",
)

OUT = os.path.join(ROOT, "Topology", "results", "placement")


def _etype_name(et: Any) -> str:
    """The EdgeType member name, e.g. "ORIENTATION_CONTINUITY".

    NOTE: ``EdgeType.name`` is the BARE member name; a ``str(et)`` renders
    "EdgeType.ORIENTATION_CONTINUITY" and the schema's class strings are
    "EDGE_ORIENTATION_CONTINUITY".  Every comparison in this file uses the
    bare name, and PLACEMENT_CLASSES is written in the same form.  (An earlier
    draft mixed the two and reported 0 placement edges.)
    """
    return getattr(et, "name", None) or str(et).replace("EdgeType.", "")


def placement_graph(g: Any) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, str]]:
    """The placement-edge adjacency plus each endpoint's lens type.

    Only PLACEMENT_CLASSES edges are walked, and the point of the class is
    that a placement edge has a MEASURED geometric quantity behind it.  Edges
    whose params are absent are dropped: an edge with no measurement carries
    no placement (and no penalty either, per the mission).
    """
    lens: Dict[int, str] = {n: str(g.sigs.get(n, ("?",))[0]) for n in g.subgraph.nodes}
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    ep = getattr(g.subgraph, "edge_params", {}) or {}
    for s, d, et in g.subgraph.edges:
        if _etype_name(et) not in PLACEMENT_CLASSES:
            continue
        if (s, d, et) not in ep and (d, s, et) not in ep:
            continue
        adj[s].append((d, et))
        if d != s:
            adj[d].append((s, et))
    for k in adj:
        adj[k] = sorted(set(adj[k]), key=lambda x: (x[0], _etype_name(x[1])))
    return dict(adj), lens


def anchors(g: Any) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        if a is not None and len(a) >= 2:
            out[int(n)] = (float(a[0]), float(a[1]))
    return out


def span_stats(adj: Dict[int, List[Tuple[int, float]]],
               pos: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
    """The SPAN (edge length in px) distribution -- not NN spacing.

    An edge's span is the distance between its endpoints' anchors.  Report the
    quantiles plus the median nearest-neighbour spacing for contrast, because
    the whole question is whether placement edges reach FARTHER than one
    nearest-neighbour step.
    """
    spans: List[float] = []
    seen = set()
    for u, nbrs in adj.items():
        for v, _et in nbrs:
            k = (min(u, v), max(u, v))
            if k in seen:
                continue
            seen.add(k)
            a, b = pos.get(u), pos.get(v)
            if a is None or b is None:
                continue
            spans.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    if not spans:
        return {"n": 0}
    arr = np.asarray(spans, dtype=np.float64)

    # Nearest-neighbour spacing over DISTINCT anchor positions, for contrast.
    # Anchors are SHARED between nodes (measured: 535 nodes on 284 distinct
    # positions, one position carrying 20 nodes), so a per-node NN distance is
    # 0 by construction and would make the ratio meaningless.
    uniq = sorted(set(pos.values()))
    nn: List[float] = []
    if len(uniq) > 1:
        P = np.asarray(uniq, dtype=np.float64)
        for i in range(len(P)):
            d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
            d[i] = np.inf
            nn.append(float(d.min()))
    q = lambda p: float(np.quantile(arr, p))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": q(0.5),
        "p05": q(0.05), "p25": q(0.25), "p75": q(0.75), "p95": q(0.95),
        "max": float(arr.max()),
        "nn_spacing_median": float(np.median(nn)) if nn else None,
        "span_over_nn": (q(0.5) / float(np.median(nn))) if nn and np.median(nn) > 0 else None,
    }


def reach(adj: Dict[int, List[Tuple[int, float]]], lens: Dict[int, str],
          pos: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
    """Reach from a seed over the placement graph.

    A "frame node" is a node whose lens is a frame lens; the mission asks what
    fraction of FRAME nodes is reachable.  Components are computed on the
    placement graph restricted to nodes that carry an anchor (nothing can be
    placed without one).
    """
    from matching.relation_frame import is_frame

    nodes = sorted(n for n in adj if n in pos)
    frame_nodes = sorted(n for n in nodes if is_frame(lens.get(n, "")))
    seen: Dict[int, int] = {}
    comps: List[List[int]] = []
    for n in nodes:
        if n in seen:
            continue
        comp: List[int] = []
        dq = deque([n])
        seen[n] = len(comps)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v, _et in adj.get(u, ()):
                if v in pos and v not in seen:
                    seen[v] = len(comps)
                    dq.append(v)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    # reach from the largest component's seed == the largest component itself
    largest = set(comps[0]) if comps else set()
    reachable_frame = [n for n in frame_nodes if n in largest]
    degs = {n: len(adj.get(n, ())) for n in nodes}
    isolated = [n for n in nodes if degs[n] == 0]
    zero_place = [n for n in sorted(lens) if n not in adj]

    return {
        "n_nodes_with_anchor": len(nodes),
        "n_frame_nodes": len(frame_nodes),
        "n_components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "component_sizes": [len(c) for c in comps[:12]],
        "frame_reachable_share": (len(reachable_frame) / len(frame_nodes)
                                  if frame_nodes else None),
        "n_frame_reachable": len(reachable_frame),
        "mean_degree": (sum(degs.values()) / len(degs)) if degs else 0.0,
        "median_degree": float(np.median(list(degs.values()))) if degs else 0.0,
        "n_zero_degree": len(isolated),
        "share_zero_degree": (len(isolated) / len(nodes)) if nodes else None,
        "n_nodes_without_any_placement_edge": len(zero_place),
        "share_without_any_placement_edge": (len(zero_place) / max(1, len(lens))),
    }


# --------------------------------------------------------------------------
# 1b ambiguity period and landmark coverage
# --------------------------------------------------------------------------
def texture_period_px(scene: Any, texture: str = "checker") -> Dict[str, float]:
    """The ambiguity period of a procedural texture, in PIXELS.

    ``texture_freq`` is cells per WORLD unit (Material docstring: the surface
    coordinate is metric, arc length in world units).  A world unit is
    ``1/pixel_scale`` pixels at the reference depth, so:

        sine period      = 128 / texture_freq  px     (the mission's own form,
                                                       which assumes
                                                       pixel_scale = 1/128)
        checker parity   = 2 x that                   (a checker repeats its
                                                       PARITY after two cells)

    Generalised to the scene's actual ``pixel_scale`` and depth, the cell size
    in pixels is ``(1/texture_freq) / pixel_scale`` under orthographic
    projection, and it is magnified by ``focal_px / depth`` under a pinhole.
    """
    freq = 8.0
    for o in getattr(scene, "objects", ()):
        m = getattr(o, "material", None)
        if m is not None and getattr(m, "texture", "none") != "none":
            freq = float(m.texture_freq)
            break
    ps = float(scene.pixel_scale)
    cell_px = (1.0 / freq) / ps if ps > 0 else 0.0
    if getattr(scene, "camera", None) is not None:
        # depth of the front object's centre
        zs = [float(np.asarray(o.centre, dtype=np.float64)[2])
              for o in scene.objects]
        depth = float(scene.camera_z) - (min(zs) if zs else 0.0)
        cell_px = cell_px * (float(scene.camera.focal_px) * ps) / max(depth, 1e-9)
    parity = 2.0 * cell_px
    return {"texture_freq": freq, "pixel_scale": ps,
            "cell_px": cell_px, "sine_period_px": cell_px,
            "checker_parity_px": parity,
            "mission_form_128_over_freq_px": 128.0 / freq}


def landmark_share(g: Any) -> Dict[str, Any]:
    """Share of texture-lens nodes with a LANDMARK edge in reach.

    A landmark is an APERIODIC node: boundary, corner, junction, event.  A
    texture node is any node whose lens is Texture / TextureBoundary.  The
    reach is one placement hop (a texture-to-texture edge does not break the
    period, so it is not counted).
    """
    from matching.relation_frame import is_frame

    LANDMARK_LENSES = frozenset({
        "Boundary", "Corner", "Junction", "T", "Y", "EndStopped",
        "Contour", "ContourFragment", "CurveComplex", "CONTOUR_PEAK",
        "PEAK_CHAIN", "IllusoryContour", "RegionShape",
    })
    TEXTURE_LENSES = frozenset({"Texture", "TextureBoundary"})
    lens = {n: str(g.sigs.get(n, ("?",))[0]) for n in g.subgraph.nodes}
    adj, _ = placement_graph(g)
    tex = sorted(n for n in lens if lens[n] in TEXTURE_LENSES)
    lm = sorted(n for n in lens if lens[n] in LANDMARK_LENSES)
    lm_set = set(lm)
    hit = [n for n in tex if any(v in lm_set for v, _e in adj.get(n, ()))]
    # also report two-hop reach, since a texture element may sit one hop from
    # a texture neighbour that is itself one hop from the landmark
    two = set()
    for n in tex:
        for v, _e in adj.get(n, ()):
            if v in lm_set:
                two.add(n)
                continue
            for w, _e2 in adj.get(v, ()):
                if w in lm_set:
                    two.add(n)
                    break
    carrier: Dict[str, int] = {}
    for n in sorted(adj):
        carrier[lens.get(n, "?")] = carrier.get(lens.get(n, "?"), 0) + 1
    return {
        "placement_carrier_lenses": dict(sorted(carrier.items())),
        "n_texture_nodes": len(tex),
        "n_landmark_nodes": len(lm),
        "landmark_lens_counts": {k: sum(1 for n in lm if lens[n] == k)
                                 for k in sorted({lens[n] for n in lm})},
        "n_texture_with_landmark_1hop": len(hit),
        "share_texture_with_landmark_1hop": (len(hit) / len(tex)) if tex else None,
        "n_texture_with_landmark_2hop": len(two),
        "share_texture_with_landmark_2hop": (len(two) / len(tex)) if tex else None,
    }


def main() -> int:
    import guard
    guard.activate()
    from eth80_frontend import build_graph
    from Topology.placement_scenes import all_specs, posed

    os.makedirs(OUT, exist_ok=True)
    results: Dict[str, Any] = {}

    for spec in all_specs():
        sc = posed(spec.scene, 0.0)
        truth = sc.render()
        g = build_graph(truth.image_u8, spec.name)
        adj, lens = placement_graph(g)
        pos = anchors(g)
        key = "%s|%s" % (spec.family, spec.name)
        results[key] = {
            "family": spec.family,
            "scene": spec.name,
            "n_nodes": len(g.subgraph.nodes),
            "n_edges": len(g.subgraph.edges),
            "n_placement_edges": sum(len(v) for v in adj.values()) // 2,
            "reach": reach(adj, lens, pos),
            "span": span_stats(adj, pos),
            "landmark": landmark_share(g),
            "texture_period": texture_period_px(spec.scene),
        }
        print("[%s] nodes=%d placement=%d frame_reach=%s landmark_share=%s"
              % (key, results[key]["n_nodes"], results[key]["n_placement_edges"],
                 ("%.3f" % results[key]["reach"]["frame_reachable_share"])
                 if results[key]["reach"]["frame_reachable_share"] is not None else "n/a",
                 ("%.3f" % results[key]["landmark"]["share_texture_with_landmark_1hop"])
                 if results[key]["landmark"]["share_texture_with_landmark_1hop"] is not None
                 else "n/a"), flush=True)

    path = os.path.join(OUT, "part1_reach.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True, default=float)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
