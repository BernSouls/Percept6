"""Front-end mechanisms (L3 spine + L4 junctions) emitted from Binder.emit_pending_edges('v4')."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, Tuple

import parameters as P

KEPT_L6_FAMILIES = frozenset({
    "CONTOUR_CONTINUITY",
    "ORIENTATION_JUNCTION",
    "CURVE_CONTAINMENT",
})


def compute_wl_addresses(
    nodes: Iterable[int],
    edges: Iterable[Tuple[int, int, Any]],
    sigs: Dict[int, Tuple[Any, ...]],
    kept_families: frozenset[str] = KEPT_L6_FAMILIES,
    max_k: int = 3,
) -> Dict[int, Dict[int, str]]:
    """Weisfeiler-Lehman structural address (k=1..max_k) on categorical types only."""
    node_list = sorted(int(n) for n in nodes)
    adj: Dict[int, list[Tuple[str, int]]] = {n: [] for n in node_list}
    for u, v, et in edges:
        etn = getattr(et, "name", str(et))
        if etn not in kept_families:
            continue
        ui, vi = int(u), int(v)
        if ui in adj and vi in adj:
            adj[ui].append((etn, vi))
            adj[vi].append((etn, ui))

    addr_by_k: Dict[int, Dict[int, str]] = {}
    addr_prev: Dict[int, str] = {
        n: str(sigs.get(n, ("?",))[0]) for n in node_list
    }
    addr_by_k[0] = dict(addr_prev)
    for k in range(1, max_k + 1):
        addr_k: Dict[int, str] = {}
        for n in node_list:
            nbrs = sorted((rel, addr_prev[nbr]) for rel, nbr in adj[n])
            payload = json.dumps([addr_prev[n], nbrs], separators=(",", ":"))
            addr_k[n] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        addr_by_k[k] = addr_k
        addr_prev = addr_k
    return addr_by_k


def _l6_address(binder) -> Dict[str, Any]:
    """L6: Compute WL structural addresses (k=1,2,3) on kept relation families."""
    sigs: Dict[int, Tuple[Any, ...]] = {}
    nodes = set()
    for d, nid in sorted(binder._stage6_detections(), key=lambda dn: int(dn[1])):
        ni = int(nid)
        nodes.add(ni)
        if ni not in sigs:
            sigs[ni] = (getattr(d, "k", "?"),)
    edges = []
    if hasattr(binder, "cg") and isinstance(getattr(binder.cg, "edges", None), dict):
        for k, v in binder.cg.edges.items():
            if isinstance(k, tuple) and len(k) >= 2:
                et = getattr(v, "edge_type", getattr(v, "type", k[2] if len(k) > 2 else "?"))
                edges.append((k[0], k[1], et))
    addrs = compute_wl_addresses(nodes, edges, sigs, KEPT_L6_FAMILIES, max_k=3)
    return {
        "flag": "FE_L6_ADDRESS",
        "status": "PRESENT",
        "node_count": len(nodes),
        "k_levels": [1, 2, 3],
        "unique_k1": len(set(addrs[1].values())),
        "unique_k2": len(set(addrs[2].values())),
        "unique_k3": len(set(addrs[3].values())),
        "canonical_order": True,
    }

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK",
    "ContourFragment",
    "PEAK_CHAIN",
    "EndStopped",
    "Corner",
    "Junction",
    "T",
    "Y",
    "RegionShape",
    "IllusoryContour",
})


def _l4_junctions(binder) -> Dict[str, Any]:
    """L4: Content-token junctions (vertices and occlusion T-points)."""
    from core.graph_types import EdgeType

    max_gap = float(getattr(P, "T_JUNCTION_MAX_GAP", 6.0))
    content_nodes: Dict[int, Tuple[float, float]] = {}
    for d, nid in sorted(binder._stage6_detections(), key=lambda dn: int(dn[1])):
        nid_int = int(nid)
        if getattr(d, "k", "") not in CONTENT_TOKEN_LENSES:
            continue
        if nid_int in content_nodes:
            continue
        pos = binder.cg.pattern_node_position(nid_int)
        if pos is not None and len(pos) >= 2:
            content_nodes[nid_int] = (float(pos[0]), float(pos[1]))

    nids = sorted(content_nodes)
    # P1: Derive junction max_gap from the frame's own content-token
    # nearest-neighbor gap distribution (pre-registered q=0.75 quantile, px).
    nn_gaps = []
    for i, u in enumerate(nids):
        ux, uy = content_nodes[u]
        best = float("inf")
        for j, v in enumerate(nids):
            if i == j:
                continue
            vx, vy = content_nodes[v]
            d = math.hypot(ux - vx, uy - vy)
            if d > 1e-6 and d < best:
                best = d
        if best < float("inf"):
            nn_gaps.append(best)
    if nn_gaps:
        nn_sorted = sorted(nn_gaps)
        q_idx = min(len(nn_sorted) - 1, int(0.75 * (len(nn_sorted) - 1)))
        max_gap = float(nn_sorted[q_idx])
    else:
        max_gap = float(getattr(P, "T_JUNCTION_MAX_GAP", 6.0))

    seen_edges: set[Tuple[int, int, str]] = set()
    candidates = []
    decision_log = []
    evaluated = 0
    rejected_gap = 0
    for i, u in enumerate(nids):
        ux, uy = content_nodes[u]
        best_v = None
        best_d = max_gap
        for v in nids[i + 1:]:
            evaluated += 1
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6:
                continue
            if dist <= best_d:
                best_d = dist
                best_v = v
            else:
                rejected_gap += 1
        if best_v is not None:
            a, b = (u, best_v) if u < best_v else (best_v, u)
            key = (a, b, "ORIENTATION_JUNCTION")
            if key not in seen_edges:
                seen_edges.add(key)
                strength = max(0.05, 1.0 - best_d / max(max_gap, 1e-6))
                candidates.append((a, b, float(strength), (a, b)))
                decision_log.append({
                    "gate": "T_JUNCTION_MAX_GAP",
                    "value": round(best_d, 4),
                    "threshold": max_gap,
                    "verdict": "ACCEPT",
                })

    if candidates:
        binder._process_channel(EdgeType.ORIENTATION_JUNCTION, candidates)

    return {
        "flag": "FE_L4_JUNCTIONS",
        "status": "PRESENT",
        "content_tokens_seen": len(content_nodes),
        "edges_emitted": len(candidates),
        "max_gap_px": max_gap,
        "decision_log_count": len(decision_log),
        "canonical_order": True,
    }


def _l3_spine(binder) -> Dict[str, Any]:
    """L3: Content-token spine (mutual-best successor/predecessor)."""
    from core.graph_types import EdgeType

    obs = getattr(binder, "_front_end_obs", None)
    img_shape = getattr(obs, "image_shape", None)
    if img_shape is not None and len(img_shape) >= 2:
        h, w = int(img_shape[0]), int(img_shape[1])
    else:
        h, w = 256, 256
    base_stride = min(h, w) / float(getattr(P, "RETINOTOPIC_GRID_DENSITY", 16))
    reach = base_stride * 1.5

    content_nodes: Dict[int, Tuple[float, float]] = {}
    for d, nid in sorted(binder._stage6_detections(), key=lambda dn: int(dn[1])):
        nid_int = int(nid)
        if getattr(d, "k", "") not in CONTENT_TOKEN_LENSES:
            continue
        if nid_int in content_nodes:
            continue
        pos = binder.cg.pattern_node_position(nid_int)
        if pos is not None and len(pos) >= 2:
            content_nodes[nid_int] = (float(pos[0]), float(pos[1]))

    nids = sorted(content_nodes)
    nearest: Dict[int, int] = {}
    nearest_d: Dict[int, float] = {}
    evaluated = 0
    rejected_reach = 0
    for u in nids:
        ux, uy = content_nodes[u]
        best_v = None
        best_d = reach
        for v in nids:
            if u == v:
                continue
            evaluated += 1
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6:
                continue
            if dist <= best_d:
                best_d = dist
                best_v = v
            else:
                rejected_reach += 1
        if best_v is not None:
            nearest[u] = best_v
            nearest_d[u] = best_d

    seen_edges: set[Tuple[int, int, str]] = set()
    spine_candidates = []
    rejected_mutual_best = 0
    for u in nids:
        v = nearest.get(u)
        if v is not None and nearest.get(v) == u:
            a, b = (u, v) if u < v else (v, u)
            key = (a, b, "CONTOUR_CONTINUITY")
            if key not in seen_edges:
                seen_edges.add(key)
                dist = nearest_d[u]
                strength = max(0.05, 1.0 - dist / max(reach, 1e-6))
                spine_candidates.append((a, b, float(strength), (a, b)))
        elif v is not None:
            rejected_mutual_best += 1

    if spine_candidates:
        binder._process_channel(EdgeType.CONTOUR_CONTINUITY, spine_candidates)

    n_nodes = max(1, len(content_nodes))
    return {
        "flag": "FE_L3_SPINE",
        "status": "PRESENT",
        "content_tokens_seen": len(content_nodes),
        "candidate_pairs_evaluated": evaluated,
        "rejected_by_reach": rejected_reach,
        "rejected_by_mutual_best": rejected_mutual_best,
        "edges_emitted": len(spine_candidates),
        "mean_degree": (2.0 * len(spine_candidates)) / n_nodes,
        "reach_px": reach,
        "canonical_order": True,
    }


def apply_once(binder) -> None:
    """Apply each enabled mechanism exactly once for this step."""
    applied = getattr(binder, "_front_end_mechanisms_applied", None)
    if applied is None:
        applied = set()
        binder._front_end_mechanisms_applied = applied
    diagnostics = binder.diagnostics.setdefault("front_end_mechanisms", {})
    if (bool(getattr(P, "FE_L4_JUNCTIONS", False))
            and "l4_junctions" not in applied):
        assert "l4_junctions" not in applied, "L4 junctions applied twice"
        diagnostics["l4_junctions"] = _l4_junctions(binder)
        applied.add("l4_junctions")
    if (bool(getattr(P, "FE_L3_SPINE", False))
            and "l3_spine" not in applied):
        assert "l3_spine" not in applied, "L3 spine applied twice"
        diagnostics["l3_spine"] = _l3_spine(binder)
        applied.add("l3_spine")
    if (bool(getattr(P, "FE_L6_ADDRESS", False))
            and "l6_address" not in applied):
        assert "l6_address" not in applied, "L6 address applied twice"
        diagnostics["l6_address"] = _l6_address(binder)
        applied.add("l6_address")
    diagnostics["applied_once"] = sorted(applied)
