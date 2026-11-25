"""P_plus + CONTEXT: a fixed, damped, deterministic placement iteration.

Mission 7 showed the ceiling is set by the CONTEXT: every geometry arm is handed
its neighbours' images from the top-similarity mapping ``N``, and ``N`` is right
on ~30 % of the nodes it is asked about.  A wrong neighbour image produces a
wrong residual, and a wrong residual is worse than no residual at all.

This module lets an arm improve the context it is scored against, without any
truth, with three properties that keep it honest:

  * FIXED work: ``ITERS`` synchronous sweeps, no convergence test, no early exit
    on a data-dependent condition -- the cost of a match does not depend on how
    easy the instance is.
  * DAMPED: a node's mapping only moves when the new choice beats the current
    one by at least ``MARGIN`` and is not in a near-tie (at most one rival
    within ``RIVAL_TOL``).  Everything else KEEPS the mapping it came in with.
  * CONFIDENT-CONTEXT FILTER: a move also needs mutual agreement -- the node it
    picks must pick it back.  Step 0b measures how much of the incoming context
    is wrong; when that share exceeds a half the filter is MANDATORY, because an
    iteration started from a mostly-wrong context can amplify the error it was
    meant to remove (``amplified`` below reports whether it did).

Nothing here reads truth.  Every threshold is a module constant, fitted on FIT
and then frozen.
"""
from __future__ import annotations

import math

import numpy as np

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

#: Synchronous sweeps.  Small and fixed on purpose.  FIT context accuracy
#: saturates at four to five sweeps (k=1 .4304, k=2 .4452, k=3 .4524, k=5 .4550,
#: k=8 .4553 on the FIT probe), so five is the last sweep that buys anything.
ITERS = 5
#: A move must beat the incumbent by this much (score units, [0, 1] scale).
#: Fitted on FIT: 0.00 (.4398 at three sweeps) beats 0.01 (.4370) and 0.02
#: (.4301); the binding damping is therefore the rival-count refusal, which is
#: never relaxed.  Kept as a knob so the sweep can re-measure it.
MARGIN = 0.0
#: Two candidates closer than this are a "near-tie" and the move is refused.
RIVAL_TOL = 0.02
#: More rivals than this inside RIVAL_TOL of the best -> refuse the move.
MAX_RIVALS = 1
#: Require the picked node to pick this one back.  OFF: with six candidates a
#: mutual claim is rare, so the requirement froze the mapping in place --
#: measured context accuracy .3998 (mutual) against .4550 (rival count only) at
#: five sweeps, on the same views.  Reported rather than quietly dropped.
MUTUAL = False


def pick_once(u: int, cand: Sequence[Tuple[int, float]],
              cur: Mapping[int, int],
              score_fn: Callable[[int, int, Mapping[int, int]], float]
              ) -> Tuple[int, float, float, int]:
    """Best candidate for ``u`` under the mapping ``cur``.

    Returns ``(best, best_val, runner_up_val, n_rivals)`` where ``n_rivals``
    counts OTHER candidates within ``RIVAL_TOL`` of the best -- the near-tie
    count that the filter refuses to act on.
    """
    best_v, best_val, second = -1, float("-inf"), float("-inf")
    vals: List[float] = []
    for (v, _s) in cand:
        val = float(score_fn(u, int(v), cur))
        vals.append(val)
        if val > best_val:
            second = best_val
            best_v, best_val = int(v), val
        elif val > second:
            second = val
    n_rivals = sum(1 for x in vals if x >= best_val - RIVAL_TOL) - 1
    return best_v, best_val, second, max(0, n_rivals)


def iterate_context(pop: Sequence[int],
                    cands: Mapping[int, Sequence[Tuple[int, float]]],
                    base_map: Mapping[int, int],
                    score_fn: Callable[[int, int, Mapping[int, int]], float],
                    iters: int = ITERS, margin: float = MARGIN,
                    rival_tol: float = RIVAL_TOL, max_rivals: int = MAX_RIVALS,
                    mutual: bool = MUTUAL,
                    stats: Dict[str, Any] = None) -> Dict[int, int]:
    """Run ``iters`` damped sweeps from ``base_map``; return the final mapping."""
    cur: Dict[int, int] = {int(k): int(v) for k, v in base_map.items()}
    moved = kept = refused = 0
    for _ in range(int(iters)):
        picks: Dict[int, Tuple[int, float, float, int]] = {}
        for u in pop:
            cand = cands.get(int(u))
            if not cand:
                continue
            picks[int(u)] = pick_once(int(u), cand, cur, score_fn)
        nxt = dict(cur)
        for u, (best_v, best_val, _second, n_rivals) in picks.items():
            if best_v < 0 or best_v == cur.get(u):
                kept += 1
                continue
            if n_rivals > max_rivals:
                refused += 1
                continue
            incumb = cur.get(u, -1)
            inc_val = float("-inf")
            for (v, _s) in cands[u]:
                if int(v) == incumb:
                    inc_val = float(score_fn(u, int(v), cur))
                    break
            if best_val - inc_val < margin:
                refused += 1
                continue
            if mutual:
                back = picks.get(best_v)
                if back is None or back[0] != u:
                    refused += 1
                    continue
            nxt[u] = best_v
            moved += 1
        cur = nxt
    if stats is not None:
        stats.update({"moved": moved, "kept": kept, "refused": refused})
    return cur


def iterate_repick(pop: Sequence[int],
                   cands: Mapping[int, Sequence[Tuple[int, float]]],
                   base_map: Mapping[int, int],
                   score_fn: Callable[[int, int, Mapping[int, int]], float],
                   iters: int = ITERS,
                   stats: Dict[str, Any] = None) -> Dict[int, int]:
    """The UNDAMPED control: re-pick every node each sweep, no filter at all.

    This is what "iteration" means without the confident-context filter, and it
    is here to be measured rather than assumed: if the incoming context is
    mostly wrong, an undamped sweep re-derives a mostly wrong context and can
    make it worse.  Reported as the amplification control.
    """
    cur: Dict[int, int] = {int(k): int(v) for k, v in base_map.items()}
    moved = 0
    for _ in range(int(iters)):
        nxt = dict(cur)
        for u in pop:
            cand = cands.get(int(u))
            if not cand:
                continue
            best_v = max(cand, key=lambda t: (float(score_fn(int(u), int(t[0]),
                                                           cur)), -int(t[0])))[0]
            if int(best_v) != cur.get(int(u)):
                moved += 1
            nxt[int(u)] = int(best_v)
        cur = nxt
    if stats is not None:
        stats.update({"moved": moved, "kept": 0, "refused": 0})
    return cur


def agreement(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    """Share of shared keys on which two mappings agree."""
    keys = [int(k) for k in a if int(k) in b]
    if not keys:
        return float("nan")
    return float(sum(1 for k in keys if int(a[k]) == int(b[k])) / len(keys))


# ---------------------------------------------------------------------------
# [M17 / O2] THE CONFIDENT-CONTEXT FILTER THAT ACTUALLY HAS SIGNAL
#
# The O2 sweep moved the geometry AUC at 15 deg by < 0.005 across iters 0/2/5/8,
# margin 0/0.02, rival count 0/1/3, rival tolerance 0.02/0.05 and mutual on/off.  The
# measurement that explains it: the context ACCURACY at 15 deg -- the share of places
# whose mapping is the truth partner -- is 0.173 for the argmax-of-similarity the
# iteration starts from.  Five sweeps of a damped filter cannot rescue a start that is
# wrong 83 % of the time, and the score the filter optimises is computed FROM that
# mapping, so the loop is circular.
#
# Two mechanisms that are not circular, both truth-free:
#
#   CONFIGURATION DESCRIPTOR -- a place's neighbourhood as a multiset of
#   (log distance, direction in the source's own frame).  `PlacementVector.direction`
#   is already measured in the source node's frame (matching/placement.py:85-104), so
#   this descriptor is invariant to a rotation of the image and to a change of scale,
#   and -- unlike the residual -- it needs NO correspondence, which is what makes it
#   usable before the context is known.
#
#   ONE-TO-ONE ASSIGNMENT -- the mapping is a bijection, not a per-node argmax, so two
#   places cannot claim the same reference place.  Solved exactly with
#   `scipy.optimize.linear_sum_assignment`.
# ---------------------------------------------------------------------------

#: Weight of the configuration term when the assignment mode is used.  Fitted on FIT
#: (see MISSION17_REPORT.md); 0.0 is the pure-appearance assignment.
ASSIGN_LAMBDA = 0.5
#: Neighbours kept, nearest first.  Six is the number of slots the scorer can judge
#: (`SLOT_CAP = 8`) and more than the median place has.
ASSIGN_K = 6


def _axial(th: float) -> float:
    return (float(th) + math.pi / 2.0) % math.pi - math.pi / 2.0


def config_descriptor(side: Any, u: int, k: int = ASSIGN_K):
    """(sorted log distances, sorted source-frame directions) to ``u``'s neighbours."""
    rows: List[Tuple[float, float]] = []
    for (other, is_source, name) in side.incident.get(int(u), ()):
        key = ((int(u), int(other), name) if is_source
               else (int(other), int(u), name))
        pv = side.index.get(key)
        if pv is None:
            continue
        d = float(getattr(pv, "distance", float("nan")))
        if not math.isfinite(d) or d <= 0.0:
            continue
        th = float(getattr(pv, "direction", 0.0))
        rows.append((math.log(d), th if not getattr(pv, "axial", False) else _axial(th)))
    if len(rows) < 2:
        return None
    rows.sort()
    return ([r[0] for r in rows][:k], [r[1] for r in rows][:k])


def config_similarity(a, b, sigma_d: float = 0.35, sigma_a: float = 0.5) -> float:
    """Rank-aligned agreement of two configuration descriptors, in [0, 1]."""
    if a is None or b is None:
        return 0.0
    da, aa = a
    db, bb = b
    n = min(len(da), len(db))
    if n < 2:
        return 0.0
    sd = sum((da[i] - db[i]) ** 2 for i in range(n))
    sa = 0.0
    for i in range(n):
        d = (aa[i] - bb[i] + math.pi) % (2.0 * math.pi) - math.pi
        d = min(abs(d), abs(abs(d) - math.pi))
        sa += d * d
    return float(math.exp(-0.5 * (sd / (n * sigma_d ** 2)
                                  + sa / (n * sigma_a ** 2))))


def assignment_mapping(pop: Sequence[int],
                       cands: Mapping[int, Sequence[Tuple[int, float]]],
                       side_a: Any, side_b: Any,
                       lam: float = ASSIGN_LAMBDA) -> Dict[int, int]:
    """One-to-one assignment maximising ``(1-lam)*sim + lam*config``.

    Truth is not read: the input is the candidate rows the matcher already has and the
    two sides' own payloads.
    """
    from scipy.optimize import linear_sum_assignment
    desc_q = {int(u): config_descriptor(side_a, int(u)) for u in pop}
    cols = sorted({int(v) for u in pop for (v, _s) in cands.get(int(u), ())})
    if not cols:
        return {}
    cmap = {v: j for j, v in enumerate(cols)}
    rows = [int(u) for u in pop if cands.get(int(u))]
    desc_r = {v: config_descriptor(side_b, v) for v in cols}
    big = np.full((len(rows), len(cols)), -1e9, dtype=float)
    for i, u in enumerate(rows):
        dq = desc_q[u]
        for (v, s) in cands[u]:
            big[i, cmap[int(v)]] = ((1.0 - lam) * float(s)
                                    + lam * config_similarity(dq, desc_r[int(v)]))
    ri, ci = linear_sum_assignment(-big)
    return {rows[i]: cols[j] for i, j in zip(ri, ci) if big[i, j] > -1e8}


def assignment_context(pop: Sequence[int],
                       cands: Mapping[int, Sequence[Tuple[int, float]]],
                       side_a: Any, side_b: Any,
                       score_fn: Callable[[int, int, Mapping[int, int]], float],
                       iters: int = ITERS, lam: float = ASSIGN_LAMBDA,
                       stats: Dict[str, Any] = None) -> Dict[int, int]:
    """Assignment start, then ``iters`` damped sweeps of `score_fn` -- else unchanged.

    ``iters = 0`` is the assignment alone: the honest baseline for "what does the
    bijection buy without the iteration on top of it".
    """
    base = assignment_mapping(pop, cands, side_a, side_b, lam=lam)
    if int(iters) <= 0:
        if stats is not None:
            stats.update({"moved": 0, "kept": len(base), "refused": 0})
        return base
    return iterate_context(pop, cands, base, score_fn, iters=iters, stats=stats)


# ---------------------------------------------------------------------------
# [M18 / O2] THE GLOBAL TRANSFORM HYPOTHESIS, in the one component that carries the
# mission: ROTATION.
#
# `Side.index` holds a PlacementVector per edge, and its `source_theta` is the SOURCE
# node's continuous orientation (matching/placement.py:85-104).  So every node has a
# measured angle without touching a signature, and every candidate pair (u, v) implies a
# rotation shift  wrap(theta_v - theta_u).  Under a rotated image, the TRUE pairs all
# share one shift and the impostors spread over the circle -- that is a Hough vote, it is
# truth-free, and it uses absolute position for nothing.
#
# The M17 assignment is still solved on a PER-PAIR score; this restricts it to the pairs
# that agree with the consensus first, which is the difference between "each pair looks
# plausible" and "the whole configuration is consistent".
# ---------------------------------------------------------------------------

#: Half-width of the consensus window, in radians (10 degrees).  A 5 degree rotation
#: splits its true pairs between two bins, so the window cannot be tighter than the
#: transform uncertainty and cannot be so wide that impostors qualify.
CONSENSUS_TOL = 0.175
#: How many shift modes may be admitted (1 = strictest).  Two admits the second cluster,
#: which is what a 5 degree rotation needs.
CONSENSUS_MODES = 1
#: Share of the winning weight a mode must reach to be admitted as a second mode.
CONSENSUS_MODE_SHARE = 0.5


def node_theta(side: Any, u: int):
    """``(theta, axial)`` for node ``u``, or ``(None, False)``.

    The orientation comes from any OUTGOING payload vector: its `source_theta` is the
    source's own orientation, so it does not depend on which neighbour it points at.
    ``axial`` says whether that frame is a half-turn ambiguous axis, which changes how a
    shift may be compared.
    """
    for (other, is_source, name) in side.incident.get(int(u), ()):
        if not is_source:
            continue
        pv = side.index.get((int(u), int(other), name))
        if pv is None:
            continue
        th = float(getattr(pv, "source_theta", float("nan")))
        if math.isfinite(th):
            return th, bool(getattr(pv, "axial", False))
    return None, False


def _shift(a: float, b: float, axial: bool) -> float:
    d = (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi
    if axial:
        # an axis is a half-turn ambiguous frame: fold the shift into [0, pi)
        d = d % math.pi
        if d > math.pi / 2.0:
            d -= math.pi
    return d


def consensus_shifts(pop: Sequence[int],
                     cands: Mapping[int, Sequence[Tuple[int, float]]],
                     side_a: Any, side_b: Any,
                     n_modes: int = CONSENSUS_MODES,
                     tol: float = CONSENSUS_TOL,
                     stats: Optional[Dict[str, Any]] = None) -> List[float]:
    """The modal rotation shift(s) implied by the candidate rows.

    Votes are weighted by the appearance similarity, so a pair the descriptor likes
    counts for more than a marginal one.  Returns up to ``n_modes`` shifts, strongest
    first, and nothing at all when no node has a measured orientation.
    """
    th_a = {int(u): node_theta(side_a, int(u)) for u in pop}
    votes: List[Tuple[float, float]] = []
    seen_b: Dict[int, Any] = {}
    for u in pop:
        ta, axial_a = th_a.get(int(u), (None, False))
        if ta is None:
            continue
        for (v, s) in cands.get(int(u), ()):
            v = int(v)
            if v not in seen_b:
                seen_b[v] = node_theta(side_b, v)
            tb, axial_b = seen_b[v]
            if tb is None:
                continue
            votes.append((_shift(ta, tb, bool(axial_a or axial_b)), float(s)))
    if not votes:
        if stats is not None:
            stats["consensus"] = {"votes": 0, "modes": []}
        return []
    # weighted circular density on a fixed grid, then local maxima
    grid = np.linspace(-math.pi / 2.0, math.pi / 2.0, 73)
    dens = np.zeros_like(grid)
    for (d, w) in votes:
        dens += w * np.exp(-0.5 * ((grid - d) / (tol / 2.0)) ** 2)
    order = np.argsort(-dens)
    modes: List[float] = []
    total = float(dens.sum()) or 1.0
    best = float(dens[order[0]])
    for k in order:
        cand = float(grid[k])
        if any(abs(cand - m) < 2.0 * tol for m in modes):
            continue
        if modes and float(dens[k]) < CONSENSUS_MODE_SHARE * best:
            break
        if len(modes) >= int(n_modes):
            break
        modes.append(cand)
    if stats is not None:
        stats["consensus"] = {"votes": len(votes), "modes": modes,
                              "best_weight_share": best / total}
    return modes


def consensus_mapping(pop: Sequence[int],
                      cands: Mapping[int, Sequence[Tuple[int, float]]],
                      side_a: Any, side_b: Any,
                      lam: float = ASSIGN_LAMBDA,
                      tol: float = CONSENSUS_TOL,
                      n_modes: int = CONSENSUS_MODES,
                      soft: bool = False,
                      bonus: float = 0.0,
                      stats: Optional[Dict[str, Any]] = None
                      ) -> Dict[int, int]:
    """Bijection over the pairs that agree with the rotation consensus.

    Nodes with no consensus-consistent candidate fall back to their full row (they are
    the ones the vote could not speak for) -- and that fallback is COUNTED, never silent.

    ``soft`` replaces the hard restriction with a bounded NUDGE: every consistent pair
    gets ``+bonus`` added to its similarity and every row keeps its full candidate list.
    That matters because the hard version is only as good as the mode, and on a real
    photograph at 15 deg the mode is wrong (`m18_shift_cap6.json`: mode 0.04 against a
    true shift of 0.26, and 90 % of the true rows lose their truth partner).  A wrong
    mode can still hurt a soft version, but it cannot starve a row of its truth.
    """
    modes = consensus_shifts(pop, cands, side_a, side_b, n_modes=n_modes, tol=tol,
                             stats=stats)
    if not modes:
        if stats is not None:
            stats["consensus_fallback"] = "no measured orientation"
        return {}
    th_a = {int(u): node_theta(side_a, int(u)) for u in pop}
    th_b: Dict[int, Any] = {}

    def _tb(v: int):
        if v not in th_b:
            th_b[v] = node_theta(side_b, v)
        return th_b[v]

    kept: Dict[int, List[Tuple[int, float]]] = {}
    n_fell_back = 0
    n_no_orient = 0
    for u in pop:
        u = int(u)
        ta, axial_a = th_a.get(u, (None, False))
        row = list(cands.get(u, ()))
        if not row:
            kept[u] = row
            continue
        if ta is None:
            # this row cannot vote and cannot be checked against the vote: it falls back
            # to the full row, and it is COUNTED -- a silent fallback is the M17 defect
            kept[u] = row
            n_no_orient += 1
            continue
        good: List[Tuple[int, float]] = []
        for (v, s) in row:
            tb, axial_b = _tb(int(v))
            if tb is None:
                continue
            d = _shift(ta, tb, bool(axial_a or axial_b))
            if any(abs(d - m) <= tol for m in modes):
                good.append((int(v), float(s)))
        if soft:
            agree = {int(v) for (v, _s) in good}
            kept[u] = [(int(v), float(s) + (float(bonus) if int(v) in agree else 0.0))
                       for (v, s) in row]
            if not agree:
                n_fell_back += 1
            continue
        if good:
            kept[u] = good
        else:
            kept[u] = row
            n_fell_back += 1
    out = assignment_mapping(sorted(kept), kept, side_a, side_b, lam=lam)
    if stats is not None:
        stats["consensus"] = dict(stats.get("consensus") or {}, mapped=len(out),
                                  rows_without_consensus=n_fell_back,
                                  rows_without_orientation=n_no_orient)
    return out
