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


def _seed_int(label: str) -> int:
    """Deterministic 64-bit seed from SHA-256 of a label (never hash/random)."""
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8],
                          "little")


def _det_lcg(seed: int) -> "iter[int]":
    """Deterministic 32-bit LCG keystream seeded from a SHA-256 seed.

    A fixed, reproducible integer generator (xorshift32).  Used to derive
    the fixed random symbol vectors and the orthogonal bindings; it is NOT
    Python's random() and is seeded only from SHA-256, so results are
    identical under any PYTHONHASHSEED.
    """
    state = seed & 0xFFFFFFFF or 1
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17) & 0xFFFFFFFF
        state ^= (state << 5) & 0xFFFFFFFF
        yield state & 0xFFFFFFFF


# cache of fixed random symbol vectors: type_name -> np.ndarray (unit norm)
_SYMBOL_VEC_CACHE: Dict[str, Any] = {}
# cache of orthogonal bindings: (relation, direction, dim) -> (perm, sign)
_BINDING_CACHE: Dict[Tuple[str, str, int], Tuple[Any, Any]] = {}


def _symbol_vector(type_name: str, dim: int) -> Any:
    """Fixed random unit vector for a node type, seeded from SHA-256."""
    import numpy as np
    key = (type_name, dim)
    cached = _SYMBOL_VEC_CACHE.get(key)
    if cached is not None:
        return cached
    rng = _det_lcg(_seed_int("symbol:" + type_name))
    vec = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        vec[i] = (next(rng) % 2001 - 1000) / 1000.0  # [-1, 1]
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    _SYMBOL_VEC_CACHE[key] = vec
    return vec


def _make_binding(relation: str, direction: str, dim: int) -> Tuple[Any, Any]:
    """Fixed orthogonal binding (permutation + sign flips) for a relation and
    direction, seeded from SHA-256(relation, direction).

    A signed permutation is orthogonal, so it preserves cosine similarity:
    cos(bind(v1), bind(v2)) == cos(v1, v2).  Different (relation, direction)
    pairs get different bindings, so the aggregated vector encodes the
    relation structure.  In/out differ; undirected uses a canonical dir.
    """
    import numpy as np
    key = (relation, direction, dim)
    if key in _BINDING_CACHE:
        return _BINDING_CACHE[key]
    rng = _det_lcg(_seed_int(f"binding:{relation}:{direction}"))
    perm = np.empty(dim, dtype=np.int64)
    for i in range(dim):
        perm[i] = i
    # deterministic Fisher-Yates shuffle from the keystream
    for i in range(dim - 1, 0, -1):
        j = int(next(rng)) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    sign = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        sign[i] = 1.0 if (next(rng) & 1) else -1.0
    _BINDING_CACHE[key] = (perm, sign)
    return (perm, sign)


def _bind_vector(vec, binding):
    import numpy as np
    perm, sign = binding
    return vec[perm] * sign


def _cosine(a, b) -> float:
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# cache of fixed random content bumps: (class, kind, i, j, dim) -> unit vector
_CONTENT_BUMP_CACHE: Dict[Tuple[str, str, int, int, int], Any] = {}


def _content_bump(class_name: str, kind: str, i: int, j: int, dim: int) -> Any:
    """Fixed random unit vector for one graded content axis.

    [A5] A "soft bump": each feature axis the matcher's node similarity
    compares (a quantized bin, a population axis/bin) gets a fixed random
    direction seeded from SHA-256.  A node's Level-0 content code is the
    weighted sum of these bumps, so the cosine of two Level-0 vectors
    reflects the SAME graded content the matcher compares -- without the
    address ever entering the matcher's objective.
    """
    import numpy as np
    key = (class_name, kind, i, j, dim)
    cached = _CONTENT_BUMP_CACHE.get(key)
    if cached is not None:
        return cached
    rng = _det_lcg(_seed_int("content:%s:%s:%d:%d" % (class_name, kind, i, j)))
    vec = np.empty(dim, dtype=np.float64)
    for k in range(dim):
        vec[k] = (next(rng) % 2001 - 1000) / 1000.0  # [-1, 1]
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    _CONTENT_BUMP_CACHE[key] = vec
    return vec


def _content_vector(sig: Any, dim: int) -> Any:
    """Graded content code of one node signature (soft bumps over the
    features node similarity compares: quantized bins + population axes).

    Returns a vector in the same space as the type vector.  ``sig`` is the
    ``TypedSignature`` (the second element of the ``(lens_type, sig)`` tuple
    the matcher compares).  ABSENT slots contribute nothing.
    """
    import numpy as np
    from core.types import ABSENT
    acc = np.zeros(dim, dtype=np.float64)
    if sig is None:
        return acc
    class_name = str(getattr(sig, "class_", "") or "?")
    # quantized bins: one soft bump per bin, weighted by the bin value
    for i, v in enumerate(getattr(sig, "quantized", ()) or ()):
        if v == ABSENT:
            continue
        try:
            vv = float(v)
        except (TypeError, ValueError):
            continue
        if vv == 0.0:
            continue
        acc += vv * _content_bump(class_name, "q", i, 0, dim)
    # population axes: one soft bump per (axis, bin), weighted by the count
    for ai, axis_vec in enumerate(getattr(sig, "population", ()) or ()):
        for bi, cnt in enumerate(axis_vec or ()):
            try:
                cc = float(cnt)
            except (TypeError, ValueError):
                continue
            if cc == 0.0:
                continue
            acc += cc * _content_bump(class_name, "p", ai, bi, dim)
    return acc


def compute_vector_addresses(
    nodes: Iterable[int],
    edges: Iterable[Tuple[int, int, Any]],
    sigs: Dict[int, Tuple[Any, ...]],
    kept_families: Optional[frozenset] = None,
    max_k: int = 3,
    dim: Optional[int] = None,
    weights: Optional[Dict[Tuple[int, int, Any], float]] = None,
):
    """Vector structural address (k=1..max_k) -- the graded counterpart of
    ``compute_wl_addresses``.

    L0 = normalize(type vector + graded content code).
    Lk = normalize(self_weight * addr_{k-1}(n)
                   + sum_edges edge_weight * bind(relation, dir, addr_{k-1}(nb)))

    [A5] The Level-0 address is the TYPE vector combined with a GRADED
    CONTENT code built from the SAME features node similarity compares
    (quantized bins + population axes), via fixed random soft bumps.  Each
    edge contributes its graded weight the same way.  ``kept_families=None``
    (the default) uses EVERY relation family the matcher scores; pass a set
    only for the historical restricted diagnostic.

    Bindings are fixed orthogonal signed permutations (cosine-preserving),
    seeded from SHA-256(relation, direction).  Neighbours are sorted by
    (relation type, neighbour canonical ID) for a canonical order.  Vectors
    are NEVER dict keys / identity -- only the old hash is the canonical ID.

    Returns {k: {node: np.ndarray}}.
    """
    import numpy as np
    from core.types import UNDIRECTED_EDGE_TYPES

    if dim is None:
        dim = int(getattr(P, "VECTOR_ADDRESS_DIM", 1024))
    self_weight = float(getattr(P, "VECTOR_ADDRESS_SELF_WEIGHT", 1.0))

    node_list = sorted(int(n) for n in nodes)
    directed = set(UNDIRECTED_EDGE_TYPES)

    # build per-node neighbour lists: (relation_name, is_undirected, nbr)
    adj: Dict[int, list] = {n: [] for n in node_list}
    for u, v, et in edges:
        etn = getattr(et, "name", str(et))
        if kept_families is not None and etn not in kept_families:
            continue
        ui, vi = int(u), int(v)
        if ui not in adj or vi not in adj:
            continue
        undirected = et in directed
        w = 1.0
        if weights is not None:
            w = float(weights.get((ui, vi, et), weights.get((vi, ui, et), 1.0)))
        # for directed: u->v contributes "out" at u, "in" at v
        # for undirected: canonical dir "out" at both
        if undirected:
            adj[ui].append((etn, "out", vi, w))
            adj[vi].append((etn, "out", ui, w))
        else:
            adj[ui].append((etn, "out", vi, w))
            adj[vi].append((etn, "in", ui, w))

    # L0 = type vector + graded content code (A5)
    addr_prev: Dict[int, Any] = {}
    for n in node_list:
        lens = str(sigs.get(n, ("?",))[0])
        type_vec = _symbol_vector(lens, dim)
        sig = sigs.get(n, (None, None))[1] if len(sigs.get(n, ())) > 1 else None
        content = _content_vector(sig, dim)
        acc = type_vec + content
        norm = float(np.linalg.norm(acc))
        addr_prev[n] = (acc / norm) if norm > 1e-12 else type_vec
    addr_by_k: Dict[int, Dict[int, Any]] = {0: dict(addr_prev)}
    for k in range(1, max_k + 1):
        addr_k: Dict[int, Any] = {}
        for n in node_list:
            # canonical order: relation type, then neighbour canonical ID
            nbrs = sorted(adj[n], key=lambda t: (t[0], t[2]))
            acc = self_weight * addr_prev[n]
            for rel, direction, nbr, w in nbrs:
                binding = _make_binding(rel, direction, dim)
                acc = acc + w * _bind_vector(addr_prev[nbr], binding)
            norm = float(np.linalg.norm(acc))
            if norm > 1e-12:
                acc = acc / norm
            addr_k[n] = acc
        addr_by_k[k] = addr_k
        addr_prev = addr_k
    return addr_by_k


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


def _l6_vector_address(binder) -> Dict[str, Any]:
    """L6 vector structural address: compute_vector_addresses beside the WL
    hash.  The vector is the graded counterpart; the hash stays the canonical
    ID.  Caches the result on the binder keyed by topology_revision."""
    rev = int(getattr(getattr(binder, "cg", None), "topology_revision", 0))
    cache = getattr(binder, "_vector_address_cache", None)
    if cache is None:
        cache = {}
        binder._vector_address_cache = cache
    if cache.get("rev") == rev and "addrs" in cache:
        return {
            "flag": "FE_VECTOR_ADDRESS",
            "status": "PRESENT",
            "cached": True,
            "topology_revision": rev,
            "canonical_order": True,
        }
    sigs: Dict[int, Tuple[Any, ...]] = {}
    nodes = set()
    for d, nid in sorted(binder._stage6_detections(), key=lambda dn: int(dn[1])):
        ni = int(nid)
        nodes.add(ni)
        if ni not in sigs:
            sigs[ni] = (getattr(d, "k", "?"),)
    edges = []
    weights: Dict[Tuple[int, int, Any], float] = {}
    if hasattr(binder, "cg") and isinstance(getattr(binder.cg, "edges", None), dict):
        for k, v in binder.cg.edges.items():
            if isinstance(k, tuple) and len(k) >= 2:
                et = getattr(v, "edge_type", getattr(v, "type",
                                                     k[2] if len(k) > 2 else "?"))
                edges.append((k[0], k[1], et))
                weights[(k[0], k[1], et)] = float(
                    getattr(v, "pending_signal", 1.0))
    max_k = int(getattr(P, "VECTOR_ADDRESS_MAX_K", 3))
    addrs = compute_vector_addresses(nodes, edges, sigs, None,
                                     max_k=max_k, weights=weights)
    cache["rev"] = rev
    cache["addrs"] = addrs
    return {
        "flag": "FE_VECTOR_ADDRESS",
        "status": "PRESENT",
        "cached": False,
        "node_count": len(nodes),
        "topology_revision": rev,
        "k_levels": list(range(1, max_k + 1)),
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


def _content_node_positions(binder) -> Dict[int, Tuple[float, float]]:
    """Positioned content-token nodes: node id -> (x, y).

    Resolves each content-lens detection's position from the step-6 position
    registry (`_stage6_positions`), keyed by the detection's own anchor --
    the SAME source `_run_boundary_successor` and `attachment.host_of` use.
    The ConceptGraph's node-id position registry (`pattern_node_position`)
    only covers nodes minted with a position at creation and MISSES the
    stage-6 content nodes (CONTOUR_PEAK / PEAK_CHAIN / ... minted at step 6
    with nid >= the registry's range), which silently starved L3/L4 to a
    fraction of the real content tokens.  `_stage6_positions` is the
    authoritative map for stage-6 anchors; the node-id registry is kept as a
    fallback for any content node that only carries a node-id position.
    """
    from perception import attachment as _att

    step_positions = binder._stage6_positions()
    content_nodes: Dict[int, Tuple[float, float]] = {}
    for d, nid in sorted(binder._stage6_detections(), key=lambda dn: int(dn[1])):
        nid_int = int(nid)
        if getattr(d, "k", "") not in CONTENT_TOKEN_LENSES:
            continue
        if nid_int in content_nodes:
            continue
        pos = _att._xy(step_positions, d.alpha)
        if pos is None:
            pos = binder.cg.pattern_node_position(nid_int)
        if pos is not None and len(pos) >= 2:
            content_nodes[nid_int] = (float(pos[0]), float(pos[1]))
    return content_nodes


def _l4_junctions_soft(binder) -> Dict[str, Any]:
    """L4 soft gate: every hard cut becomes a graded weight; dedup merges
    by max weight (keep the larger-weight site, never delete).

    Hard cuts replaced:
      * corner turn (turn > corner_rad)  -> graded over a band below it.
      * occlusion gap (dist <= occ_gap)  -> graded, small weight beyond.
      * dedup (drop close corners)       -> merge: keep max-weight site.
    """
    from core.graph_types import EdgeType
    from perception import attachment as _att

    corner_deg = float(getattr(P, "L4_CORNER_TURN_DEG", 45.0))
    corner_rad = math.radians(corner_deg)
    occ_gap = float(getattr(P, "JUNCTION_MAX_GAP", 3.0))
    dedup_px = float(getattr(P, "L4_CORNER_DEDUP_PX", 8.0))
    fallback_max_gap = float(getattr(P, "T_JUNCTION_MAX_GAP", 6.0))
    band = float(getattr(P, "FE_SOFT_GATE_BAND", 0.10))
    floor = float(getattr(P, "FE_SOFT_GATE_NOISE_FLOOR", 0.05))

    content_nodes = _content_node_positions(binder)
    nids = sorted(content_nodes)

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
        max_gap = fallback_max_gap

    peak_of: Dict[Any, Tuple[Any, Tuple[float, float], int]] = {}
    for d, nid in binder._stage6_detections():
        if getattr(d, "k", "") != "CONTOUR_PEAK":
            continue
        nid_int = int(nid)
        if nid_int not in content_nodes:
            continue
        if d.alpha in peak_of:
            continue
        peak_of[d.alpha] = (d, content_nodes[nid_int], nid_int)

    chain_of: Dict[Any, Tuple[Any, int, Tuple[Any, ...], bool]] = {}
    for d, nid in binder._stage6_detections():
        if getattr(d, "k", "") != "PEAK_CHAIN":
            continue
        members = _att._members(d)
        if len(members) < 2:
            continue
        q = d.sigma.quantized
        closed = bool(len(q) > 3 and float(q[3]) >= 0.5 and len(members) >= 3)
        for idx, a in enumerate(members):
            if a not in chain_of:
                chain_of[a] = (d.alpha, idx, tuple(members), closed)

    def _nearest(u, pos, exclude=None):
        ux, uy = pos
        best_v, best_d = None, max_gap * (1.0 + band)
        for v in nids:
            if v == u or (exclude is not None and v == exclude):
                continue
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6:
                continue
            if dist <= best_d:
                best_d = dist
                best_v = v
        return best_v, best_d

    seen_edges: set[Tuple[int, int, str]] = set()
    candidates = []
    corners = 0
    tpoints = 0

    # 1. corners: graded turn.  turn_strength rises across a band below the
    #    corner threshold (small weight for just-below-threshold turns).
    corner_sites: list = []
    for alpha, (det, pos, u) in peak_of.items():
        info = chain_of.get(alpha)
        if info is None:
            continue
        _ca, idx, members, closed = info
        n_m = len(members)
        if n_m < 2:
            continue
        prev_a = members[(idx - 1) % n_m]
        next_a = members[(idx + 1) % n_m]
        if prev_a not in peak_of or next_a not in peak_of:
            continue
        px, py = peak_of[prev_a][1]
        nx, ny = peak_of[next_a][1]
        ux, uy = pos
        s1 = math.atan2(uy - py, ux - px) % math.pi
        s2 = math.atan2(ny - uy, nx - ux) % math.pi
        turn = _att.axial_difference(s1, s2)
        # [PART 1b] localise the corner at the arm intersection.
        local_pos, _loc = _corner_localise(
            peak_of[prev_a][1], pos, peak_of[next_a][1])
        # graded: full at/above corner_rad, floor well below.
        turn_strength = _graded_ratio(turn, corner_rad, floor)
        corner_sites.append((alpha, local_pos, u, turn, turn_strength))

    # dedup merges keep-larger-weight: cluster by position, keep the
    # highest-weight site (NOT delete the near-duplicate).
    corner_sites.sort(key=lambda s: -s[4])
    kept_corners: list = []
    for site in corner_sites:
        _a, pos, _u, _turn, _ws = site
        if any(math.hypot(pos[0] - kp[0], pos[1] - kp[1]) <= dedup_px
               for _ka, kp, _ku, _kt, _kw in kept_corners):
            continue
        kept_corners.append(site)
    corners = len(kept_corners)
    for alpha, pos, u, turn, ws in kept_corners:
        v, dist = _nearest(u, pos)
        if v is not None:
            a, b = (u, v) if u < v else (v, u)
            key = (a, b, "ORIENTATION_JUNCTION")
            if key not in seen_edges:
                seen_edges.add(key)
                gap_strength = max(floor, 1.0 - dist / max(max_gap, 1e-6))
                weight = ws * gap_strength
                if weight < floor:
                    weight = floor
                candidates.append((a, b, float(weight), (a, b)))

    # 2. occlusion T-points: graded gap.
    nid_to_alpha: Dict[int, Any] = {}
    for d, nid in binder._stage6_detections():
        nid_to_alpha.setdefault(int(nid), d.alpha)
    for alpha, (det, pos, u) in peak_of.items():
        u_chain = chain_of.get(alpha, (None,))[0]
        best_v, best_d = None, occ_gap * (1.0 + band)
        ux, uy = pos
        for v in nids:
            if v == u:
                continue
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6 or dist > best_d:
                continue
            v_alpha = nid_to_alpha.get(v)
            v_chain = chain_of.get(v_alpha, (None,))[0] if v_alpha is not None else None
            if v_chain is not None and v_chain == u_chain:
                continue
            best_d = dist
            best_v = v
        if best_v is not None:
            a, b = (u, best_v) if u < best_v else (best_v, u)
            key = (a, b, "ORIENTATION_JUNCTION")
            if key not in seen_edges:
                seen_edges.add(key)
                strength = _graded_ratio(best_d, occ_gap, floor)
                strength = max(strength, floor)
                candidates.append((a, b, float(strength), (a, b)))
                tpoints += 1

    if candidates:
        binder._process_channel(EdgeType.ORIENTATION_JUNCTION, candidates)

    return {
        "flag": "FE_L4_JUNCTIONS_SOFT",
        "status": "PRESENT",
        "content_tokens_seen": len(content_nodes),
        "edges_emitted": len(candidates),
        "corners_detected": corners,
        "occlusion_tpoints": tpoints,
        "max_gap_px": max_gap,
        "corner_turn_deg": corner_deg,
        "occlusion_gap_px": occ_gap,
        "canonical_order": True,
    }


def _l4_junctions(binder) -> Dict[str, Any]:
    """L4: Content-token junctions (vertices and occlusion T-points).

    A junction edge is emitted ONLY at a detected junction site, so the
    family targets renderer vertices rather than arbitrary nearby tokens:

      * CORNER  -- a CONTOUR_PEAK whose chain tangent turns sharply against
        its chain-neighbour(s).  The peak IS the vertex; the edge anchors
        the vertex to its nearest content token.
      * OCCLUSION T-POINT -- a CONTOUR_PEAK that has a nearby content token
        belonging to a DIFFERENT chain (two contours meeting).  The edge
        joins the two peaks across the junction.

    The corner turn threshold and the occlusion gap are read from the
    parameters module (defaults pre-registered from the tracer calibration:
    corner turn q0.90 = 60 deg, occlusion gap = JUNCTION_MAX_GAP).
    """
    from core.graph_types import EdgeType
    from perception import attachment as _att

    if bool(getattr(P, "FE_SOFT_L3_L4_GATE", False)):
        return _l4_junctions_soft(binder)

    corner_deg = float(getattr(P, "L4_CORNER_TURN_DEG", 45.0))
    corner_rad = math.radians(corner_deg)
    occ_gap = float(getattr(P, "JUNCTION_MAX_GAP", 3.0))
    dedup_px = float(getattr(P, "L4_CORNER_DEDUP_PX", 8.0))
    fallback_max_gap = float(getattr(P, "T_JUNCTION_MAX_GAP", 6.0))

    content_nodes = _content_node_positions(binder)
    nids = sorted(content_nodes)

    # max_gap for the corner-edge partner lookup: the frame's own content
    # nearest-neighbor gap q0.75 (px), as before.
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
        max_gap = fallback_max_gap

    # peak_of: CONTOUR_PEAK alpha -> (det, pos, nid)
    peak_of: Dict[Any, Tuple[Any, Tuple[float, float], int]] = {}
    for d, nid in binder._stage6_detections():
        if getattr(d, "k", "") != "CONTOUR_PEAK":
            continue
        nid_int = int(nid)
        if nid_int not in content_nodes:
            continue
        if d.alpha in peak_of:
            continue
        peak_of[d.alpha] = (d, content_nodes[nid_int], nid_int)

    # chain structure: member alpha -> (chain_alpha, index, members, closed)
    chain_of: Dict[Any, Tuple[Any, int, Tuple[Any, ...], bool]] = {}
    for d, nid in binder._stage6_detections():
        if getattr(d, "k", "") != "PEAK_CHAIN":
            continue
        members = _att._members(d)
        if len(members) < 2:
            continue
        q = d.sigma.quantized
        closed = bool(len(q) > 3 and float(q[3]) >= 0.5 and len(members) >= 3)
        for idx, a in enumerate(members):
            if a not in chain_of:
                chain_of[a] = (d.alpha, idx, tuple(members), closed)

    def _tangent(det):
        return _att.raw_orientation(det, 12)

    # nearest content token to a position (excluding self), within max_gap
    def _nearest(u, pos, exclude=None):
        ux, uy = pos
        best_v, best_d = None, max_gap
        for v in nids:
            if v == u or (exclude is not None and v == exclude):
                continue
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6:
                continue
            if dist <= best_d:
                best_d = dist
                best_v = v
        return best_v, best_d

    seen_edges: set[Tuple[int, int, str]] = set()
    candidates = []
    decision_log = []
    corners = 0
    tpoints = 0

    # 1. corners -- GEOMETRIC turn of the chain polyline at the peak.  The
    # quantized tangent slots alternate along a straight edge (measured:
    # every turn reads 0 or 90 deg), so a tangent-difference test fires on
    # straight runs.  The polyline turn between consecutive peak segments
    # uses positions only and separates a real vertex (sharp turn) from a
    # straight run (no turn) cleanly.  Detected corners are then de-duplicated
    # by position (a box corner often carries two nearby peaks, each turning).
    corner_sites: list = []
    for alpha, (det, pos, u) in peak_of.items():
        info = chain_of.get(alpha)
        if info is None:
            continue
        _ca, idx, members, closed = info
        n_m = len(members)
        if n_m < 2:
            continue
        prev_a = members[(idx - 1) % n_m]
        next_a = members[(idx + 1) % n_m]
        if prev_a not in peak_of or next_a not in peak_of:
            continue
        px, py = peak_of[prev_a][1]
        nx, ny = peak_of[next_a][1]
        ux, uy = pos
        # incoming and outgoing segment directions (axial)
        s1 = math.atan2(uy - py, ux - px) % math.pi
        s2 = math.atan2(ny - uy, nx - ux) % math.pi
        turn = _att.axial_difference(s1, s2)
        if turn > corner_rad:
            # [PART 1b] localise the corner at the line-intersection of the
            # two straight arms through the neighbouring peaks; fall back to
            # the peak when the arms are degenerate.
            local_pos, _loc = _corner_localise(
                peak_of[prev_a][1], pos, peak_of[next_a][1])
            corner_sites.append((alpha, local_pos, u, turn))

    # de-duplicate: greedy cluster by position, keep the highest-turn site
    corner_sites.sort(key=lambda s: -s[3])
    kept_corners: list = []
    for site in corner_sites:
        _a, pos, _u, _turn = site
        if any(math.hypot(pos[0] - kp[0], pos[1] - kp[1]) <= dedup_px
               for _ka, kp, _ku, _kt in kept_corners):
            continue
        kept_corners.append(site)
    corners = len(kept_corners)
    for alpha, pos, u, turn in kept_corners:
        v, dist = _nearest(u, pos)
        if v is not None:
            a, b = (u, v) if u < v else (v, u)
            key = (a, b, "ORIENTATION_JUNCTION")
            if key not in seen_edges:
                seen_edges.add(key)
                strength = max(0.05, 1.0 - dist / max(max_gap, 1e-6))
                candidates.append((a, b, float(strength), (a, b)))
                decision_log.append({
                        "gate": "L4_CORNER_TURN",
                        "value": round(math.degrees(turn), 3),
                        "threshold_deg": corner_deg,
                        "verdict": "ACCEPT",
                    })

    # 2. occlusion T-points: a peak with a nearby content token of a
    #    different chain
    nid_to_alpha: Dict[int, Any] = {}
    for d, nid in binder._stage6_detections():
        nid_to_alpha.setdefault(int(nid), d.alpha)
    for alpha, (det, pos, u) in peak_of.items():
        u_chain = chain_of.get(alpha, (None,))[0]
        best_v, best_d = None, occ_gap
        ux, uy = pos
        for v in nids:
            if v == u:
                continue
            vx, vy = content_nodes[v]
            dist = math.hypot(ux - vx, uy - vy)
            if dist <= 1e-6 or dist > best_d:
                continue
            # v must be a peak of a different chain (or unchained)
            v_alpha = nid_to_alpha.get(v)
            v_chain = chain_of.get(v_alpha, (None,))[0] if v_alpha is not None else None
            if v_chain is not None and v_chain == u_chain:
                continue
            best_d = dist
            best_v = v
        if best_v is not None:
            a, b = (u, best_v) if u < best_v else (best_v, u)
            key = (a, b, "ORIENTATION_JUNCTION")
            if key not in seen_edges:
                seen_edges.add(key)
                strength = max(0.05, 1.0 - best_d / max(occ_gap, 1e-6))
                candidates.append((a, b, float(strength), (a, b)))
                tpoints += 1
                decision_log.append({
                    "gate": "L4_OCCLUSION_GAP",
                    "value": round(best_d, 4),
                    "threshold": occ_gap,
                    "verdict": "ACCEPT",
                })

    if candidates:
        binder._process_channel(EdgeType.ORIENTATION_JUNCTION, candidates)

    return {
        "flag": "FE_L4_JUNCTIONS",
        "status": "PRESENT",
        "content_tokens_seen": len(content_nodes),
        "edges_emitted": len(candidates),
        "corners_detected": corners,
        "occlusion_tpoints": tpoints,
        "max_gap_px": max_gap,
        "corner_turn_deg": corner_deg,
        "occlusion_gap_px": occ_gap,
        "decision_log_count": len(decision_log),
        "canonical_order": True,
    }


def _graded_ratio(value: float, threshold: float, floor: float) -> float:
    """Graded weight in [floor,1]: 1 at/above `threshold`, floor at/below 0,
    linear between.  Replaces a hard value >= threshold cut-off."""
    if threshold <= 0.0:
        return 1.0 if value > 0.0 else floor
    r = value / threshold
    if r >= 1.0:
        return 1.0
    if r <= 0.0:
        return floor
    return floor + (1.0 - floor) * r


def _line_intersection(p1, p2, p3, p4):
    """Intersection of line(p1,p2) and line(p3,p4), else None.

    Used for L4 corner localisation: the true vertex is where the two
    straight chain arms through the neighbouring peaks cross, which is more
    accurate than the (grid-offset) peak position.  Returns None for
    near-parallel or degenerate arms, so the caller falls back to the peak.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-6:
        return None
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
    x = x1 + t * dx1
    y = y1 + t * dy1
    return (x, y)


def _corner_localise(prev_pos, peak_pos, next_pos):
    """Localise a corner at the line-intersection of the two straight arms
    through the neighbouring peaks; fall back to the peak position when the
    arms are degenerate (near-parallel) or the intersection falls far from
    the peak (a broken arm)."""
    pt = _line_intersection(prev_pos, peak_pos, peak_pos, next_pos)
    if pt is None:
        return peak_pos, False
    # reject if the intersection is far from the peak (broken arms)
    if math.hypot(pt[0] - peak_pos[0], pt[1] - peak_pos[1]) > 24.0:
        return peak_pos, False
    return pt, True


def _l3_spine_soft(binder) -> Dict[str, Any]:
    """L3 soft gate: every hard cut becomes a graded weight.

    Hard cuts replaced:
      * reach (dist <= reach)      -> reach graded over [0, reach], small
                                      weight up to reach * band.
      * mutual-best (nearest[v]==u) -> graded mutual = d(v,nearest[v])/d(v,u).
    Just-below-threshold pairs keep a small weight (the noise floor), and the
    total weight is calibrated to the old emitted count downstream.
    """
    from core.graph_types import EdgeType

    obs = getattr(binder, "_front_end_obs", None)
    img_shape = getattr(obs, "image_shape", None)
    if img_shape is not None and len(img_shape) >= 2:
        h, w = int(img_shape[0]), int(img_shape[1])
    else:
        h, w = 256, 256
    base_stride = min(h, w) / float(getattr(P, "RETINOTOPIC_GRID_DENSITY", 16))
    reach = base_stride * 1.5
    band = float(getattr(P, "FE_SOFT_GATE_BAND", 0.10))
    floor = float(getattr(P, "FE_SOFT_GATE_NOISE_FLOOR", 0.05))
    reach_ext = reach * (1.0 + band)

    content_nodes = _content_node_positions(binder)
    nids = sorted(content_nodes)

    nearest: Dict[int, int] = {}
    nearest_d: Dict[int, float] = {}
    evaluated = 0
    for u in nids:
        ux, uy = content_nodes[u]
        best_v = None
        best_d = reach_ext
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
        if best_v is not None:
            nearest[u] = best_v
            nearest_d[u] = best_d

    seen_edges: set[Tuple[int, int, str]] = set()
    spine_candidates = []
    graded_mutual = 0
    for u in nids:
        v = nearest.get(u)
        if v is None:
            continue
        dist = nearest_d[u]
        # reach grading: full at dist<=reach, down to floor at reach*band.
        if dist <= reach:
            reach_strength = _graded_ratio(dist, reach, floor)
        else:
            reach_strength = floor * (1.0 - (dist - reach) /
                                      (reach * band + 1e-9))
            reach_strength = max(0.0, min(floor, reach_strength))
        # mutual grading: how strongly v's nearest agrees it is u.
        v_nearest = nearest.get(v)
        if v_nearest == u:
            mutual = 1.0
        elif v_nearest is not None:
            d_v_nb = nearest_d.get(v, 1e-9)
            d_v_u = math.hypot(content_nodes[v][0] - content_nodes[u][0],
                               content_nodes[v][1] - content_nodes[u][1])
            if d_v_u > 1e-6:
                mutual = min(1.0, d_v_nb / d_v_u)
                graded_mutual += 1
            else:
                mutual = 1.0
        else:
            mutual = floor
        weight = reach_strength * mutual
        if weight < floor:
            weight = floor
        a, b = (u, v) if u < v else (v, u)
        key = (a, b, "CONTOUR_CONTINUITY")
        if key not in seen_edges:
            seen_edges.add(key)
            spine_candidates.append((a, b, float(weight), (a, b)))

    if spine_candidates:
        binder._process_channel(EdgeType.CONTOUR_CONTINUITY, spine_candidates)

    n_nodes = max(1, len(content_nodes))
    return {
        "flag": "FE_L3_SPINE_SOFT",
        "status": "PRESENT",
        "content_tokens_seen": len(content_nodes),
        "candidate_pairs_evaluated": evaluated,
        "graded_mutual_pairs": graded_mutual,
        "edges_emitted": len(spine_candidates),
        "mean_degree": (2.0 * len(spine_candidates)) / n_nodes,
        "reach_px": reach,
        "canonical_order": True,
    }


def _l3_spine(binder) -> Dict[str, Any]:
    """L3: Content-token spine (mutual-best successor/predecessor)."""
    from core.graph_types import EdgeType

    if bool(getattr(P, "FE_SOFT_L3_L4_GATE", False)):
        return _l3_spine_soft(binder)

    obs = getattr(binder, "_front_end_obs", None)
    img_shape = getattr(obs, "image_shape", None)
    if img_shape is not None and len(img_shape) >= 2:
        h, w = int(img_shape[0]), int(img_shape[1])
    else:
        h, w = 256, 256
    base_stride = min(h, w) / float(getattr(P, "RETINOTOPIC_GRID_DENSITY", 16))
    reach = base_stride * 1.5

    content_nodes = _content_node_positions(binder)

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
    if (bool(getattr(P, "FE_VECTOR_ADDRESS", False))
            and "l6_vector_address" not in applied):
        assert "l6_vector_address" not in applied, "L6 vector address applied twice"
        diagnostics["l6_vector_address"] = _l6_vector_address(binder)
        applied.add("l6_vector_address")
    diagnostics["applied_once"] = sorted(applied)
