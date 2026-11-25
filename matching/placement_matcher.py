"""The CREDITED placement scorer, as the deployed matcher can actually use it.

Mission 8 credited one arm on a fresh pre-registered set: `P_plus_ctx`, existence
plus geometry with two fitted coefficients, scored against an ITERATED context,
blended with node similarity at a fixed weight.  What the matcher shipped before
was a different thing entirely -- a constant factor per ROW, which cannot reorder
the candidates within a row, so a node whose true partner sat one place down the
list could never be rescued.

This module turns the credited arm into a candidate-level re-scoring of the
matcher's seed matrix:

    for each row i (node u of side A):
        take the top CAND_CAP candidates by similarity  (the same population the
        harness scores -- the candidates a node would actually choose between)
        compute a context mapping N by row-wise argmax
        iterate it ITERS times, damped, refusing near-ties (truth-free)
        replace matrix[i][j] = (1 - BETA) * sim + BETA * term(u -> v, context)
        every other cell keeps its raw similarity

Everything is frozen: the coefficients, the densities, the caps, the iteration
constants and the weight all come from placement_frozen_params.json, which is
written from the Mission-8 FIT and cannot drift without this module refusing to
load.  Truth is never read here -- there is no truth argument to pass.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.join(_HERE, "placement_frozen_params.json")
#: [M17 / O1] the refitted table; read only when `PLACEMENT_LLR_TABLE_M17` is on, so the
#: deployed default stays byte-identical.
FROZEN_PATH_M17 = os.path.join(_HERE, "placement_frozen_params_m17.json")
#: [M18 / O1] the pose-balanced, fine-fronted, exp-tailed table.  Read only when
#: `PLACEMENT_LLR_TABLE_M18` is on; the deployed default stays byte-identical.
FROZEN_PATH_M18 = os.path.join(_HERE, "placement_frozen_params_m18.json")

#: Counters for the loud-but-survivable production path.  A test or a harness can
#: read them; production logs them.
FAILURES: Dict[str, int] = {"rescored": 0, "raised": 0, "fell_back": 0, "skipped": 0,
                            "desc_capped": 0}

_BUNDLE: Optional[Dict[str, Any]] = None
#: set while a FIT-only candidate bundle is installed; production leaves it None
OVERRIDE: Optional[Dict[str, Any]] = None
_FIT = None
_PFIT = None


def active_path() -> str:
    """Which table this process reads.  Flag OFF (the default) is the deployed one."""
    import parameters as P
    p = str(getattr(P, "PLACEMENT_LLR_TABLE_PATH", "") or "")
    if p:
        return p
    if bool(getattr(P, "PLACEMENT_LLR_TABLE_M18", False)):
        return FROZEN_PATH_M18
    if bool(getattr(P, "PLACEMENT_LLR_TABLE_M17", False)):
        return FROZEN_PATH_M17
    return FROZEN_PATH


def apply_node_descriptor(matrix: Sequence[Sequence[float]],
                          nodes_a: Sequence[int], nodes_b: Sequence[int],
                          side_a: Any, side_b: Any,
                          k: Optional[int] = None, w: Optional[float] = None,
                          max_cells: Optional[int] = None) -> List[List[float]]:
    """[M19 / O1] Re-rank the head of every row with the CONFIGURATION descriptor.

    The signature similarity is gated by five quantised slots, so places that share a key
    are separated only by the graded population term -- measured rank-1 share at 15 deg:
    0.059.  The configuration descriptor (sorted log neighbour distances, sorted
    source-frame directions) is a per-node geometry: it needs no mapping, it is
    rotation-invariant by construction, and it is already what the bijection start scores.

    `rank_score = (1-w)*sim + w*config`.

    WHICH CELLS.  Blending only the top ``k`` would be wrong: a blended head value can fall
    below an UNBLENDED cell further down and the row's argmax would then be a cell that was
    never re-ranked.  So every cell whose raw similarity is at least ``(1-w) * s_k`` (the
    k-th best, attenuated) is blended, which is a superset of the head and exactly the band
    that can be affected.  The band is truncated at ``max_cells`` in descending similarity
    and every truncation is counted in ``FAILURES["desc_capped"]``.

    Flag OFF (the default) returns the matrix object it was given, untouched.

    WHERE IT RUNS.  `rescore_matrix` calls it first, so the deployed path
    (`alignment.global_consistency_alignment` -> `rescore_matrix`, i.e. any caller with
    `PLACEMENT_MATCHER_ENABLED` and both sides) applies the same blend the evaluation
    harness applies.  Mission 19 shipped it harness-side only; that is what Mission 20
    item 1 ported.
    """
    import parameters as P
    if not bool(getattr(P, "PLACEMENT_NODE_DESC", False)):
        return matrix
    from matching import placement_context as Ctx
    kk = int(getattr(P, "PLACEMENT_NODE_DESC_K", 24) if k is None else k)
    ww = float(getattr(P, "PLACEMENT_NODE_DESC_W", 0.5) if w is None else w)
    mc = int(getattr(P, "PLACEMENT_NODE_DESC_MAXCELLS", 96) if max_cells is None else max_cells)
    ww = min(max(ww, 0.0), 1.0)
    desc_q = {int(u): Ctx.config_descriptor(side_a, int(u)) for u in nodes_a}
    desc_r = {int(v): Ctx.config_descriptor(side_b, int(v)) for v in nodes_b}
    out: List[List[float]] = [list(r) for r in matrix]
    for i, u in enumerate(nodes_a):
        row = out[i]
        live = [(j, float(row[j])) for j in range(len(row)) if float(row[j]) > 0.0]
        if not live:
            continue
        live.sort(key=lambda t: (-t[1], int(nodes_b[t[0]])))
        s_k = live[min(kk, len(live)) - 1][1]
        thr = (1.0 - ww) * s_k
        # mc <= 0 means THE WHOLE ROW.  A band of only the head is NOT scale-consistent:
        # a head cell whose configuration is poor is attenuated to (1-w)*s, which a
        # low-similarity TAIL cell -- never blended, so never attenuated -- can outscore.
        # Blending every positive cell keeps one score scale; truncation is opt-in, bounded
        # and counted.
        band = [(j, s) for (j, s) in live if s >= thr] if mc > 0 else list(live)
        if mc > 0 and len(band) > mc:
            FAILURES["desc_capped"] = FAILURES.get("desc_capped", 0) + 1
            band = band[:mc]
        dq = desc_q.get(int(u))
        for (j, s) in band:
            c = Ctx.config_similarity(dq, desc_r.get(int(nodes_b[j])))
            row[j] = (1.0 - ww) * s + ww * c
    return out


def adaptive_cap(base: int, row_size: int, k: Optional[float] = None,
                 cap_max: Optional[int] = None) -> int:
    """[M19 / O2] `clamp(round(k * row_size), base, cap_max)` -- monotone in the row size."""
    import parameters as P
    kk = float(getattr(P, "PLACEMENT_CAP_ADAPTIVE_K", 0.125) if k is None else k)
    cm = int(getattr(P, "PLACEMENT_CAP_ADAPTIVE_MAX", 64) if cap_max is None else cap_max)
    want = int(round(kk * max(int(row_size), 1)))
    return int(min(max(want, int(base)), max(int(base), cm)))


def top_candidates_dedup(matrix: Sequence[Sequence[float]],
                         nodes_b: Sequence[int], base: int,
                         keys_b: Optional[Sequence[Any]] = None
                         ) -> List[List[Tuple[int, int, float]]]:
    """`top_candidates`, with each row's cap spent on DISTINCT signature keys first.

    A row of 315 candidates whose 586-place population shares 136 distinct quantised keys
    is mostly copies: the same five slots five times over.  Spending the cap on the best
    candidate of each distinct key first, then filling the remainder in similarity order,
    buys diversity for free in compute -- and it can LOSE the truth, so the recall is
    measured, not assumed (`m19_dedup.json` reports the truth denominator before and after).

    `keys_b` is one key per node id in `nodes_b` (any hashable); a node with no key is
    treated as its own distinct key.  Flag OFF (the default) is exactly `top_candidates`.
    """
    import parameters as P
    if not bool(getattr(P, "PLACEMENT_CAP_DEDUP", False)) or keys_b is None:
        return top_candidates(matrix, nodes_b, base)
    key_of = {int(v): k for v, k in zip(nodes_b, keys_b)}
    out: List[List[Tuple[int, int, float]]] = []
    for i in range(len(matrix)):
        row = [(j, int(nodes_b[j]), float(matrix[i][j]))
               for j in range(len(nodes_b)) if float(matrix[i][j]) > 0.0]
        row.sort(key=lambda t: (-t[2], t[1]))
        seen: set = set()
        head: List[Tuple[int, int, float]] = []
        for t in row:
            k = key_of.get(t[1], ("__node__", t[1]))
            if k in seen:
                continue
            seen.add(k)
            head.append(t)
            if len(head) >= int(base):
                break
        chosen = {t[1] for t in head}
        rest = [t for t in row if t[1] not in chosen]
        out.append(head + rest[: max(0, int(base) - len(head))])
    return out


def top_candidates_adaptive(matrix: Sequence[Sequence[float]],
                            nodes_b: Sequence[int], base: int) -> List[List[Tuple[int, int, float]]]:
    """`top_candidates`, with each row's cap from `adaptive_cap` when the flag is on."""
    import parameters as P
    if not bool(getattr(P, "PLACEMENT_CAP_ADAPTIVE", False)):
        return top_candidates(matrix, nodes_b, base)
    out: List[List[Tuple[int, int, float]]] = []
    for i in range(len(matrix)):
        row = [(j, int(nodes_b[j]), float(matrix[i][j]))
               for j in range(len(nodes_b)) if float(matrix[i][j]) > 0.0]
        row.sort(key=lambda t: (-t[2], t[1]))
        cap = adaptive_cap(int(base), len(row))
        out.append(row[:cap])
    return out


def cand_cap() -> int:
    """The candidate cap: the frozen one unless `PLACEMENT_CAND_CAP` overrides it."""
    import parameters as P
    cap = int(getattr(P, "PLACEMENT_CAND_CAP", 0) or 0)
    return cap if cap > 0 else int(frozen()["cand_cap"])


def reset_cache() -> None:
    """Drop the loaded bundle.  A harness that switches table must call this."""
    global _BUNDLE, _FIT, _PFIT
    _BUNDLE = None
    _FIT = None
    _PFIT = None


def frozen() -> Dict[str, Any]:
    """The frozen bundle, loaded once and checked against the live constants."""
    global _BUNDLE, _FIT, _PFIT
    if _BUNDLE is None:
        from matching import placement_llr as L
        with open(active_path()) as fh:
            b = json.load(fh)
        _FIT, _PFIT, ae, ag = L.from_jsonbundle(b)   # raises on a cap mismatch
        b["_a_exist"], b["_a_geom"] = ae, ag
        _BUNDLE = b
    return _BUNDLE


def fit_objects():
    frozen()
    return _FIT, _PFIT


def override(bundle: Optional[Dict[str, Any]]) -> None:
    """FIT-ONLY: install a candidate bundle, or None to fall back to the file.

    The deployed path reads whatever `frozen()` returns, so a search over
    coefficients has to be able to present a candidate without touching the file
    on disk.  This is the seam: it validates the candidate exactly like the file
    (``from_jsonbundle`` refuses a bundle whose caps disagree with the live
    constants), it records that it is active in ``OVERRIDE`` so a caller can see
    it, and production never calls it -- nothing in the matcher, the alignment or
    the front end refers to it.
    """
    global _BUNDLE, _FIT, _PFIT, OVERRIDE
    if bundle is None:
        _BUNDLE = None
        _FIT = None
        _PFIT = None
        OVERRIDE = None
        return
    from matching import placement_llr as L
    b = dict(bundle)
    fit, pfit, ae, ag = L.from_jsonbundle(b)
    b["_a_exist"], b["_a_geom"] = float(ae), float(ag)
    _FIT, _PFIT = fit, pfit
    _BUNDLE = b
    OVERRIDE = {k: b[k] for k in ("beta", "cand_cap", "plus", "iteration") if k in b}


# ---------------------------------------------------------------------------
@dataclass
class Side:
    """One graph's placement payload, with the indices the scorer needs."""

    payload: Dict[Any, Any] = field(default_factory=dict)
    index: Dict[Tuple[int, int, str], Any] = field(default_factory=dict)
    incident: Dict[int, List[Tuple[int, bool, str]]] = field(default_factory=dict)
    lens: Dict[int, str] = field(default_factory=dict)
    nodes: Tuple[int, ...] = ()
    #: per-place data: the node positions the places were read at, and the local
    #: place spacing (nearest-neighbour distance among places).  Mission 9 step 5
    #: asks whether the tolerance should scale with this; it is carried here so
    #: the question can be asked without rebuilding anything.
    places: Tuple[Tuple[float, float], ...] = ()
    spacing_median: float = float("nan")
    spacing_by_node: Dict[int, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(self.payload) and bool(self.incident)


#: [MISSION 12 / step 1b] ONE SIDE PER IMAGE, NOT ONE PER PAIR-DIRECTION.
#: ``joint_mcs`` runs the alignment FORWARD and REVERSE, and each directional call
#: builds both sides -- so a gallery/query pair built the same image's payload up
#: to four times, and every build re-reads the edge payloads and the place
#: geometry off the same pixels.  ``build_side`` is a pure function of
#: ``(graph, image_u8, object_of, intra_only)`` plus the frozen parameter bundle,
#: so the result can be kept.
#:
#: The key is the IDENTITY of the graph object (held in the entry, so the id
#: cannot be recycled), a content hash of the pixels, and the switches that can
#: change the answer.  Anything else is a miss: a different graph object, a
#: different image, different switches or a different frozen bundle all rebuild.
#: A Side is a mutable dataclass, so the CACHE hands out the same object to every
#: caller -- callers must treat it as read-only, which every caller in the tree
#: does (the payload, its index and the incidence are built once and read).
_SIDE_CACHE: Dict[Tuple[Any, ...], Tuple[Any, "Side"]] = {}
_SIDE_CACHE_MAX = 4096


def _image_digest(image_u8: Any) -> Any:
    """A content hash of the pixels (the graph cache's own key, when possible)."""
    try:
        from Topology.placement_cache import key_for
        return key_for(image_u8)
    except Exception:
        try:
            import hashlib
            return hashlib.sha1(np.ascontiguousarray(image_u8).tobytes()).hexdigest()
        except Exception:
            return None


def _switch_digest() -> Any:
    """The frozen bundle (the only switch set that changes a payload)."""
    try:
        b = frozen()
        return tuple(sorted((str(k), repr(v)) for k, v in b.items()))
    except Exception:
        return None


def build_side_cached(graph: Any, image_u8: Any, *,
                      object_of: Optional[Dict[int, int]] = None,
                      intra_only: bool = False) -> Side:
    """``build_side`` memoised per (graph object, pixels, switches)."""
    key = (id(graph), _image_digest(image_u8), id(object_of), bool(intra_only),
           _switch_digest())
    ent = _SIDE_CACHE.get(key)
    if ent is not None and ent[0] is graph:
        return ent[1]
    side = build_side(graph, image_u8, object_of=object_of, intra_only=intra_only)
    if len(_SIDE_CACHE) >= _SIDE_CACHE_MAX:
        _SIDE_CACHE.clear()
    _SIDE_CACHE[key] = (graph, side)
    return side


def build_side(graph: Any, image_u8: Any, *,
               object_of: Optional[Dict[int, int]] = None,
               intra_only: bool = False) -> Side:
    """Build one side's payload from an ImageGraph and the pixels it came from."""
    from matching import placement_edges as PE
    from matching.placement_scorer import incident_index, payload_index

    lens = PE.lens_map_of(graph)
    # [M15 / item 3] THE ENDPOINT LIST IS A SWITCH.  With PLACEMENT_M15_PLACES on the
    # list is the atlas place-lens list and `build_edges` selects it; passing an
    # explicit list here would silently override the flag (an explicit argument wins
    # by design), so the frozen list is passed only when the flag is off.
    import parameters as _P
    _ends = (None if bool(getattr(_P, "PLACEMENT_M15_PLACES", False))
             else tuple(frozen()["endpoint_lenses"]))
    pay = PE.build_edges(graph.subgraph, image_u8, object_of=object_of,
                         intra_only=intra_only, lens_map=lens,
                         endpoint_lenses=_ends)
    sub = graph.subgraph
    pos = {int(n): (float(sub.anchor_of_node[n][0]),
                    float(sub.anchor_of_node[n][1]))
           for n in sorted(sub.nodes) if n in sub.anchor_of_node}
    places = tuple(pos[n] for n in sorted(pos))
    spacing_med = float("nan")
    spacing_by_node: Dict[int, float] = {}
    if len(places) >= 2:
        arr = np.asarray(places, dtype=float)
        d = np.hypot(arr[:, None, 0] - arr[None, :, 0],
                     arr[:, None, 1] - arr[None, :, 1])
        np.fill_diagonal(d, np.inf)
        spacing_med = float(np.median(np.min(d, axis=1)))
        for k, n in enumerate(sorted(pos)):
            spacing_by_node[int(n)] = float(np.min(d[k]))
    return Side(payload=pay, index=payload_index(pay),
                incident=incident_index(pay), lens=lens,
                nodes=tuple(sorted(int(n) for n in sub.nodes)),
                places=places, spacing_median=spacing_med,
                spacing_by_node=spacing_by_node)


# ---------------------------------------------------------------------------
def top_candidates(matrix: Sequence[Sequence[float]],
                   nodes_b: Sequence[int], cap: int
                   ) -> List[List[Tuple[int, int, float]]]:
    """Per row: the ``cap`` best candidates as ``(column, node id, similarity)``.

    The COLUMN indexes the matrix; the NODE ID is what the payload and the
    context speak.  Keeping both is what stops a position being read as an id --
    the first version of this module did exactly that and scored 0.002 agreement
    with the harness while looking plausible.  Deterministic: ties break on the
    node id, as the harness sorts.
    """
    out: List[List[Tuple[int, int, float]]] = []
    for i in range(len(matrix)):
        row = [(j, int(nodes_b[j]), float(matrix[i][j]))
               for j in range(len(nodes_b)) if float(matrix[i][j]) > 0.0]
        row.sort(key=lambda t: (-t[2], t[1]))
        out.append(row[:int(cap)])
    return out


def enun_rows(matrix, nodes_b, rows):
    """(row index, cells to score): every positive cell of the row.

    ``rows`` (the capped candidate list) fixes the CONTEXT; the cells returned
    here are what gets re-scored.  See rescore_matrix for why the two differ.
    """
    for i in range(len(matrix)):
        cells = [(j, int(nodes_b[j]), float(matrix[i][j]))
                 for j in range(len(nodes_b)) if float(matrix[i][j]) > 0.0]
        yield i, cells


def _pair_normalised() -> bool:
    """Is the placement term the mean over judged pairs?

    Two dark flags ask for the same shape: ``PLACEMENT_PAIR_NORMALISED`` (M10 step
    4b) and ``PLACEMENT_E_NORMALISED`` (M12 step 2, the FIT-chosen reading of E,
    which also carries the minimum-judged refusal).  With both off this returns
    False and every caller uses the sum-over-the-cap arithmetic it had before
    either flag existed.
    """
    try:
        import parameters as P
        return bool(getattr(P, "PLACEMENT_PAIR_NORMALISED", False)
                    or getattr(P, "PLACEMENT_E_NORMALISED", False))
    except Exception:
        return False


def _min_judged() -> int:
    """The minimum judged-slot count a pair needs to be read (DARK, M12 step 2)."""
    try:
        import parameters as P
        return int(getattr(P, "PLACEMENT_E_MIN_JUDGED", 0) or 0)
    except Exception:
        return 0


def _spacing_tol_fn(side_a: "Side", side_b: "Side"):
    """A per-pair tolerance function, or None when the spacing rule is dark.

    Mission 10 step 4a: the frozen tolerance was fitted where places sit ~6 px
    apart; on a denser image an absolute tolerance covers a different fraction of
    a place spacing.  When PLACEMENT_SPACING_TOL is on, the tolerance for a pair is
    scaled from the local spacing of the two places (maximum of the two, so the
    coarser place decides), using numbers fitted on synthetic FIT only.
    """
    try:
        from matching import placement_spacing as SP
        if not SP.enabled("tol"):
            return None
    except Exception:
        return None
    from matching import placement_llr as L

    def tol_of(u: int, v: int) -> float:
        return SP.tol_for(side_a.spacing_by_node.get(int(u)),
                          side_b.spacing_by_node.get(int(v)),
                          L.FIXED_TOL, fallback=side_a.spacing_median)
    return tol_of


def context_mapping(rows: Sequence[Sequence[Tuple[int, int, float]]],
                    nodes_a: Sequence[int],
                    side_a: Side, side_b: Side, *,
                    use_geometry: bool = True,
                    iters: Optional[int] = None,
                    margin: Optional[float] = None,
                    mutual: Optional[bool] = None,
                    max_rivals: Optional[int] = None,
                    rival_tol: Optional[float] = None,
                    stats: Optional[Dict[str, Any]] = None
                    ) -> Dict[int, int]:
    """N's mapping, then the damped context iteration over the same rows."""
    from matching import placement_context as Ctx
    from matching import placement_llr as L

    b = frozen()
    fit, pfit = fit_objects()
    a_e = float(b["_a_exist"])
    a_g = float(b["_a_geom"]) if use_geometry else 0.0
    it = int(b["iteration"]["iters"] if iters is None else iters)
    mar = float(b["iteration"]["margin"] if margin is None else margin)
    mut = bool(b["iteration"]["mutual"] if mutual is None else mutual)
    # [M17 / O2] the place-regime constants, when the re-fit flag is on.  They are read
    # here rather than baked into `placement_context` so the frozen tiling constants stay
    # exactly what they were for every existing caller.
    import parameters as _P
    if bool(getattr(_P, "PLACEMENT_CONTEXT_M17", False)):
        it = int(getattr(_P, "PLACEMENT_CONTEXT_M17_ITERS", it))
        mar = float(getattr(_P, "PLACEMENT_CONTEXT_M17_MARGIN", mar))
        mut = bool(getattr(_P, "PLACEMENT_CONTEXT_M17_MUTUAL", mut))
        if max_rivals is None:
            max_rivals = int(getattr(_P, "PLACEMENT_CONTEXT_M17_MAX_RIVALS", 1))
        if rival_tol is None:
            rival_tol = float(getattr(_P, "PLACEMENT_CONTEXT_M17_RIVAL_TOL", 0.02))
    beta = float(b["beta"])

    cands: Dict[int, List[Tuple[int, float]]] = {}
    sim: Dict[Tuple[int, int], float] = {}
    base: Dict[int, int] = {}
    for i, row in enumerate(rows):
        u = int(nodes_a[i])
        if not row:
            continue
        cands[u] = [(int(vid), float(sim_v)) for (_j, vid, sim_v) in row]
        for (_j, vid, sim_v) in row:
            sim[(u, int(vid))] = float(sim_v)
        base[u] = int(row[0][1])

    # [M17 / O2] THE ASSIGNMENT START.  It replaces the per-node argmax the damped
    # iteration starts from -- nothing else changes, and with the flag off this block
    # does not run at all.  Two mechanisms: a bijection (two places cannot claim the
    # same reference place) and a rotation-invariant configuration term, so the start
    # is not computed from the mapping it is supposed to fix.
    if bool(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS", False)) and cands:
        # [M18 / O2] the rotation-consensus start.  It SUPERSEDES the plain assignment
        # (it restricts that bijection to the shift-consistent pairs), so it is checked
        # first and the assignment flag becomes a fallback for when it has no votes.
        from matching import placement_context as _Ctx3
        _cs: Dict[str, Any] = {}
        _con = _Ctx3.consensus_mapping(
            sorted(cands), cands, side_a, side_b,
            lam=float(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS_LAMBDA", 0.5)),
            tol=float(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS_TOL", 0.175)),
            n_modes=int(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS_MODES", 1)),
            soft=bool(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS_SOFT", False)),
            bonus=float(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS_BONUS", 0.15)),
            stats=_cs)
        if _con:
            for _u, _v in _con.items():
                base[_u] = _v
        if stats is not None:
            stats.update({"consensus": _cs.get("consensus"),
                          "consensus_fallback": _cs.get("consensus_fallback")})
    if (bool(getattr(_P, "PLACEMENT_CONTEXT_ASSIGN", False)) and cands
            and not bool(getattr(_P, "PLACEMENT_CONTEXT_CONSENSUS", False))):
        from matching import placement_context as _Ctx2
        _asg = _Ctx2.assignment_mapping(
            sorted(cands), cands, side_a, side_b,
            lam=float(getattr(_P, "PLACEMENT_CONTEXT_ASSIGN_LAMBDA", 0.5)))
        for _u, _v in _asg.items():
            base[_u] = _v
        if stats is not None:
            stats.update({"assign": True, "assign_n": len(_asg)})

    tol_of = _spacing_tol_fn(side_a, side_b)
    norm = _pair_normalised()

    def score(u: int, v: int, m: Mapping[int, int]) -> float:
        term = L.plus_score(u, int(v), m, side_a.incident, side_a.index,
                            side_b.index, fit, pfit, float(a_e), float(a_g),
                            tol=(None if tol_of is None else tol_of(u, v)),
                            normalise=norm, min_judged=_min_judged())
        return (1.0 - beta) * sim.get((int(u), int(v)), 0.0) + beta * term

    if max_rivals is None:
        max_rivals = int(b["iteration"].get("max_rivals", 1))
    if rival_tol is None:
        rival_tol = float(b["iteration"].get("rival_tol", 0.02))
    return Ctx.iterate_context(sorted(cands), cands, base, score, iters=it,
                               margin=mar, mutual=mut, max_rivals=int(max_rivals),
                               rival_tol=float(rival_tol), stats=stats)


def rescore_matrix(matrix: Sequence[Sequence[float]],
                   nodes_a: Sequence[int], nodes_b: Sequence[int],
                   side_a: Side, side_b: Side, *,
                   use_geometry: bool = True,
                   stats: Optional[Dict[str, Any]] = None
                   ) -> List[List[float]]:
    """The candidate-level re-scoring the mission asks the matcher to apply."""
    from matching import placement_llr as L

    b = frozen()
    fit, pfit = fit_objects()
    a_e = float(b["_a_exist"])
    a_g = float(b["_a_geom"]) if use_geometry else 0.0
    beta = float(b["beta"])
    # [M20 / item 1] `cand_cap()` honours PLACEMENT_CAND_CAP, whose value is the frozen cap
    # at defaults.  Reading `b["cand_cap"]` directly made the deployed scorer ignore the
    # flag, and the M19 record was measured at cap 24 -- see parameters/_text.py.
    cap = cand_cap()

    # [M20 / item 1] THE M19 DESIGN, IN THE DEPLOYED PATH.  The descriptor re-ranks the
    # row BEFORE the candidate cap and the context see it, which is where the harness
    # applies it too; with the bundle off (or `PLACEMENT_NODE_DESC` off) this is the same
    # line it always was, and `apply_node_descriptor` returns the matrix untouched.
    matrix = apply_node_descriptor(matrix, nodes_a, nodes_b, side_a, side_b)
    if stats is not None:
        # [M20 / item 1] WHAT THE MATCHER ACTUALLY USED, at call time.  The values come from
        # `m15_manifest.effective_flags()` -- the module-level switches AS DERIVED -- so this
        # cannot report the process default while the bundle has moved the switch, which is
        # what the M19 result files did (a row could carry `desc: (24, 0.25)` next to a
        # manifest saying "descriptor off").  The table and the cap are resolver outputs, so
        # they are added beside the flags; neither is a switch this function reads.
        from matching.m15_manifest import effective_flags
        stats["effective"] = effective_flags()
        stats["effective"]["llr_table"] = os.path.basename(active_path())
        stats["effective"]["cand_cap_resolved"] = int(cap)
    rows = top_candidates(matrix, nodes_b, cap)
    ctx = context_mapping(rows, nodes_a, side_a, side_b,
                          use_geometry=use_geometry, stats=stats)
    out = [[float(x) for x in r] for r in matrix]
    changed = 0
    # [MISSION 9 / measured] The cap defines the CONTEXT population -- the six
    # candidates a node would actually choose between, exactly as the harness
    # defined it.  The re-scoring itself is applied to EVERY positive cell of the
    # row, because leaving the rest at raw similarity puts two scales in one row:
    # a blended candidate is (1-b)*sim + b*term <= sim, so an unscored cell with
    # a slightly worse raw similarity suddenly outranks it.  Measured with the
    # literal "top six only" reading: the deployed matcher's ambiguous accuracy
    # fell from 0.4226 to 0.2234 and it re-parted 708 nodes per match.
    tol_of = _spacing_tol_fn(side_a, side_b)
    norm = _pair_normalised()
    scored = 0
    for i, row_all in enun_rows(matrix, nodes_b, rows):
        u = int(nodes_a[i])
        for (j, vid, s) in row_all:
            scored += 1
            term = L.plus_score(u, int(vid), ctx, side_a.incident, side_a.index,
                                side_b.index, fit, pfit, a_e, a_g,
                                tol=(None if tol_of is None else tol_of(u, vid)),
                                normalise=norm, min_judged=_min_judged())
            new = (1.0 - beta) * float(s) + beta * float(term)
            if abs(new - out[i][j]) > 1e-12:
                changed += 1
            out[i][j] = new
    if stats is not None:
        stats["cells_changed"] = changed
        # [MISSION 12 / step 1d] THE COUNTER COUNTS WHAT WAS SCORED.  It used to
        # sum the CAPPED context rows, i.e. the candidates the context chose
        # between, while the loop above re-scores every positive cell of every
        # row -- two different populations, so the reported number was neither
        # the context population nor the scored one.  It is now the number of
        # (row, cell) pairs `plus_score` was actually called on.
        stats["cells_scored"] = int(scored)
        stats["rows"] = len(rows)
        stats["context_agreement_with_n"] = _agree(
            ctx, {int(nodes_a[i]): int(r[0][1]) for i, r in enumerate(rows) if r})
    return out


def argmax_mapping(rows: Sequence[Sequence[Tuple[int, int, float]]],
                   nodes_a: Sequence[int], nodes_b: Sequence[int],
                   side_a: Side, side_b: Side, *,
                   use_geometry: bool = True) -> Dict[int, int]:
    """EQUIVALENCE MODE: the harness's own decision rule, per node.

    Take the scored row, argmax, done.  Returns NODE IDS.  This exists so the
    deployed path can be shown to make the SAME choice as the harness on the same
    candidates; any difference is a wiring bug, not a modelling choice.
    """
    from matching import placement_llr as L

    b = frozen()
    fit, pfit = fit_objects()
    a_e = float(b["_a_exist"])
    a_g = float(b["_a_geom"]) if use_geometry else 0.0
    beta = float(b["beta"])
    ctx = context_mapping(rows, nodes_a, side_a, side_b,
                          use_geometry=use_geometry)
    norm = _pair_normalised()
    out: Dict[int, int] = {}
    for i, row in enumerate(rows):
        if not row:
            continue
        u = int(nodes_a[i])
        best_vid, best_val = -1, -float("inf")
        for (_j, vid, s) in row:
            term = L.plus_score(u, int(vid), ctx, side_a.incident, side_a.index,
                                side_b.index, fit, pfit, a_e, a_g,
                                normalise=norm, min_judged=_min_judged())
            val = (1.0 - beta) * float(s) + beta * float(term)
            if val > best_val:
                best_val, best_vid = val, int(vid)
        out[u] = best_vid
    return out


def _agree(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    keys = [k for k in a if k in b]
    if not keys:
        return float("nan")
    return float(sum(1 for k in keys if a[k] == b[k]) / len(keys))
