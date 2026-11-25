"""A CALIBRATED placement scorer: per-edge log-likelihood ratio.

The mission-2 score was a bounded ad hoc function -- `support` in [0, 1] from a
residual, averaged over however many edges a node happened to have.  That
average is why a node with three checkable edges was decided by three numbers
and why the wrong-candidate score tracked the *count* of checkable edges rather
than the geometry.

This module replaces it with one estimator, fixed in advance:

    score(u -> v) = sum over SLOT_CAP slots of  clip(LLR(slot), +-LLR_CLIP)
                    / (2 * LLR_CLIP * SLOT_CAP) + 0.5          ->  in [0, 1]

  * **Cap**: at most ``SLOT_CAP`` neighbour slots per candidate, chosen by a
    canonical sort (class name, then neighbour id).
  * **Pad**: every slot that cannot be checked -- no neighbour image, no
    matching edge, no frame -- is filled with ``mu_wrong``, the mean LLR of the
    wrong-pair slots.  This is what stops a wrong candidate from being rewarded
    for having *more* checkable edges: under padding, the expected score of a
    wrong candidate does not depend on its checkable-edge count.
  * **Clip**: each slot's LLR is clipped to +-``LLR_CLIP`` so one wild residual
    cannot decide a node.
  * **Sum**: slots add, so a noisy edge can beat no evidence only if its LLR
    clears ``mu_wrong``.

The LLR comes from residual DENSITIES fitted on FIT, true pairs against wrong
pairs, where "wrong" means the rivals the assignment actually sees -- the other
near-tie candidates of the same node -- not random nodes.

``presence_score`` is the SAME machinery with the residual dependence removed:
each slot is credited by the fitted per-class log-ratio of the corresponding
image edge EXISTING under a true versus a wrong candidate.  Comparing P against
``P_presence`` isolates exactly what the residual adds, by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Slots per candidate.  Capped so one node with a dense neighbourhood cannot
#: outvote a node with a sparse one.
SLOT_CAP = 8
#: LLR clip, in nats.
LLR_CLIP = 3.0
#: The residual normaliser.  FIXED BEFORE FITTING at the value mission 2
#: derived and pre-registered on DEV (largest DEV q90 of the true residual), so
#: that the density fit inherits a constant that was chosen without reference to
#: any CHECK number.  Under an LLR the scale is arbitrary as long as it is frozen
#: first -- the fitted densities absorb it -- and freezing it here is what keeps
#: "one tolerance" from becoming a tuning knob.
FIXED_TOL = 1.7009

#: Residual support, in scale units, and bins for the density fit.
RES_MAX = 6.0
N_BINS = 24
#: Laplace smoothing so an empty bin cannot produce an infinite LLR.
_PSEUDO = 0.5

Slot = Tuple[str, int, Optional[float]]      # (class name, neighbour id, residual)


def _admit_checkable_first() -> bool:
    """[M15 / item 2] the dark switch that changes WHICH slots are admitted."""
    try:
        import parameters as P
    except Exception:            # pragma: no cover
        return False
    return bool(getattr(P, "PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST", False))


def _count_judged_slots() -> bool:
    """[M15 / item 2] count slots whose residual EXISTS (dark)."""
    try:
        import parameters as P
    except Exception:            # pragma: no cover
        return False
    return bool(getattr(P, "PLACEMENT_SLOT_COUNT_JUDGED", False))


def judged_slots(slots: Sequence[Slot]) -> int:
    """How many of `slots` carry a residual -- i.e. how many can be checked."""
    return int(sum(1 for (_n, _o, r) in slots if r is not None))


def slots_for_candidate(u: int, v: int,
                        mapping: Mapping[int, int],
                        q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                        q_idx: Mapping[Tuple[int, int, str], Any],
                        r_idx: Mapping[Tuple[int, int, str], Any],
                        tol: Optional[float] = None
                        ) -> List[Slot]:
    """The (class, neighbour, residual) slots of u under the candidate u -> v.

    Canonically sorted and truncated to ``SLOT_CAP``.  A slot whose residual
    cannot be formed carries ``None`` -- it will be PADDED, never zero-filled.
    """
    from matching.placement_scorer import payload_residual
    mu = int(v)
    out: List[Slot] = []
    for (other, is_source, name) in sorted(q_inc.get(int(u), ()),
                                           key=lambda t: (t[2], int(t[0]))):
        mo = mapping.get(int(other))
        if mo is None:
            out.append((name, int(other), None))
            continue
        if is_source:
            qv = q_idx.get((int(u), int(other), name))
            rv = r_idx.get((mu, int(mo), name))
        else:
            qv = q_idx.get((int(other), int(u), name))
            rv = r_idx.get((int(mo), mu, name))
        if qv is None or rv is None:
            out.append((name, int(other), None))
            continue
        out.append((name, int(other),
                    float(payload_residual(qv, rv,
                                           FIXED_TOL if tol is None else tol))))
    if _admit_checkable_first():
        # [M15 / item 2] ADMIT THE EVIDENCE THAT EXISTS.
        #
        # The shipped order is (class name, neighbour id) over ALL slots, and the
        # truncation to SLOT_CAP = 8 then happens afterwards -- so with
        # TARGET_K = 16 the eight lowest-id neighbours win, whether or not their
        # residual can be formed, and a checkable edge at rank 9 is discarded
        # while uncheckable slots keep the credit of the padding value.
        #
        # The stated reliability rule is: a slot that can be checked is more
        # reliable than one that cannot, and among checkable slots the SMALLEST
        # RESIDUAL is the most reliable evidence (it is the slot whose geometry
        # agrees best, which is also the one the estimator can speak about most
        # precisely).  Ties break on (class, neighbour id) so the order stays
        # deterministic and node ids never decide alone.
        checkable = sorted((s for s in out if s[2] is not None),
                           key=lambda s: (float(s[2]), str(s[0]), int(s[1])))
        rest = sorted((s for s in out if s[2] is None),
                      key=lambda s: (str(s[0]), int(s[1])))
        out = checkable + rest
    return out[:SLOT_CAP]


# ---------------------------------------------------------------------------
@dataclass
class LLRFit:
    """Fitted residual densities and the padded wrong-pair expectation."""

    edges: np.ndarray
    llr: np.ndarray
    mu_wrong: float
    n_true: int
    n_wrong: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def of(self, res: Optional[float]) -> float:
        """LLR of one residual, or ``mu_wrong`` when it cannot be formed."""
        if res is None:
            return float(self.mu_wrong)
        i = int(np.searchsorted(self.edges, abs(float(res)), side="right") - 1)
        i = max(0, min(len(self.llr) - 1, i))
        return float(self.llr[i])


def _neutralise_empty_bins(llr: "np.ndarray", ht: "np.ndarray",
                           hw: "np.ndarray", edges: "np.ndarray",
                           min_count: Optional[int] = None) -> "np.ndarray":
    """[M15 / item 5c] Empty (or too thin) bins stop carrying the prior ratio.

    A bin is EMPTY when it has no TRUE and no WRONG sample (or fewer than
    ``PLACEMENT_LLR_MIN_BIN_COUNT`` in total): it takes 0.0.  A bin with one
    population empty but the other present takes the value of the nearest bin
    that has both -- the wrong-mean reading -- because one population is a
    measurement and the other is an absence, and the absence must not invent a
    ratio.  Bins that have both are left exactly as fitted.
    """
    import parameters as _P
    mc = int(getattr(_P, "PLACEMENT_LLR_MIN_BIN_COUNT", 0)) if min_count is None \
        else int(min_count)
    counts = np.asarray(ht, dtype=float) + np.asarray(hw, dtype=float)
    empty = counts <= max(0, mc - 1)
    out = np.array(llr, dtype=float, copy=True)
    good = (~empty) & (np.asarray(ht) > 0) & (np.asarray(hw) > 0)
    if not good.any():
        out[:] = 0.0
        return out
    idx_good = np.flatnonzero(good)
    for i in np.flatnonzero(empty | ~good)[:]:
        if counts[i] <= 0 or empty[i]:
            out[i] = 0.0 if counts[i] <= 0 else float(np.interp(
                i, idx_good, out[idx_good]))
        else:
            out[i] = float(np.interp(i, idx_good, out[idx_good]))
    return out


def llr_monotonicity(llr: "np.ndarray", edges: "np.ndarray") -> Dict[str, Any]:
    """[M15 / item 5c] Where does the fitted LLR RISE with the residual?

    The LLR must be non-increasing in |residual|: a closer placement is never
    worse evidence than a farther one.  This reports every bin where it rises,
    so the claim is checkable instead of asserted.
    """
    rises = [{"bin": int(i), "res_lo": float(edges[i]), "res_hi": float(edges[i + 1]),
              "llr": float(llr[i]), "next_llr": float(llr[i + 1]),
              "rise": float(llr[i + 1] - llr[i])}
             for i in range(len(llr) - 1) if float(llr[i + 1]) > float(llr[i]) + 1e-12]
    return {"rises": rises, "n_rises": len(rises), "n_bins": int(len(llr))}


def fit_llr(res_true: Sequence[float], res_wrong: Sequence[float],
            res_max: float = RES_MAX, n_bins: int = N_BINS) -> LLRFit:
    """Bin the two residual populations and take the log ratio.

    Both histograms use the same edges, are density-normalised and Laplace
    smoothed, so the LLR is comparable across bins and finite everywhere.
    """
    edges = np.linspace(0.0, float(res_max), int(n_bins) + 1)
    t = np.asarray(list(res_true), dtype=float)
    w = np.asarray(list(res_wrong), dtype=float)
    t = t[np.isfinite(t)]
    w = w[np.isfinite(w)]
    ht, _ = np.histogram(np.clip(t, 0.0, res_max - 1e-9), bins=edges)
    hw, _ = np.histogram(np.clip(w, 0.0, res_max - 1e-9), bins=edges)
    pt = (ht + _PSEUDO) / (ht.sum() + _PSEUDO * n_bins)
    pw = (hw + _PSEUDO) / (hw.sum() + _PSEUDO * n_bins)
    llr = np.log(pt / pw)
    # [M15 / item 5c] AN EMPTY BIN IS NOT EVIDENCE FOR A MATCH.
    #
    # With no samples the two smoothed densities are the two PRIORS, so the LLR
    # is log((N_wrong+12)/(N_true+12)) -- positive whenever there are more wrong
    # samples than true ones, which is always.  The shipped table therefore gives
    # +0.5966 to 13 of its 24 bins, i.e. "never measured here" is read as
    # "this residual supports the match".  On, a bin with NEITHER population
    # takes 0.0 (neutral) and a bin with one population empty takes the nearest
    # non-empty bin's value (the wrong-mean reading, never the prior).
    import parameters as _P
    if bool(getattr(_P, "PLACEMENT_LLR_EMPTY_BIN_NEUTRAL", False)):
        llr = _neutralise_empty_bins(llr, ht, hw, edges)
    llr = np.clip(llr, -LLR_CLIP, LLR_CLIP)
    # The padding value is the mean LLR the WRONG population would receive.
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(np.clip(w, 0.0, res_max - 1e-9), edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mu_wrong = float(np.mean(llr[idx])) if len(idx) else 0.0
    return LLRFit(edges=edges, llr=llr, mu_wrong=mu_wrong,
                  n_true=int(t.size), n_wrong=int(w.size),
                  extra={"centres": centres.tolist()})


def score_candidate(u: int, v: int, mapping: Mapping[int, int],
                    q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                    q_idx: Mapping[Tuple[int, int, str], Any],
                    r_idx: Mapping[Tuple[int, int, str], Any],
                    fit: LLRFit, slot_cap: int = SLOT_CAP,
                    pad: Optional[float] = None,
                    tol: Optional[float] = None) -> float:
    """The padded, clipped, summed LLR score for u -> v, mapped to [0, 1].

    ``pad`` overrides the padding value.  The default is ``mu_wrong`` -- an
    uncheckable slot is worth what a wrong pair is worth -- but STAGE 0b asks
    what happens if an absent edge is treated as NEUTRAL instead, because 33 %
    of edges vanish between views and the absence penalty may swamp weak
    geometry.
    """
    padv = float(fit.mu_wrong) if pad is None else float(pad)
    slots = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx,
                                tol=tol)[:slot_cap]
    total = 0.0
    for (_name, _other, res) in slots:
        total += fit.of(res)
    total += padv * (slot_cap - len(slots))
    return total / (2.0 * LLR_CLIP * slot_cap) + 0.5


# ---------------------------------------------------------------------------
# LEVER 3: FRAME-FREE relations -- an angle at u between two neighbours and the
# ratio of their distances.  Neither needs an orientation frame at all, so this
# is the one geometric read that does not depend on the field provider, and it
# is invariant to translation (only differences of positions enter) and to
# scale (a ratio of distances).  It is reflection-symmetric as an ANGLE only if
# the order of the two neighbours is dropped, so the angle is taken as the
# unsigned angle in [0, pi] and the two orders are the same test; the residual
# is therefore blind to a mirror image, which is stated rather than hidden.
#: Pairs of neighbours tried per candidate: C(8, 2) = 28 would be 28 checks for
#: every candidate, so the first RATIO_PAIRS pairs in canonical order are used.
RATIO_PAIRS = 6


def ratio_slots(u: int, v: int, mapping: Mapping[int, int],
                q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                pos_q: Mapping[int, Tuple[float, float]],
                pos_r: Mapping[int, Tuple[float, float]]
                ) -> List[Tuple[str, Optional[float]]]:
    """Frame-free (angle, distance-ratio) residuals for the candidate u -> v."""
    import math
    nb = sorted({int(o) for (o, _s, _n) in q_inc.get(int(u), ())
                 if int(o) in mapping and int(o) in pos_q}, )
    pu = pos_q.get(int(u))
    pv = pos_r.get(int(v))
    if pu is None or pv is None:
        return []
    out: List[Tuple[str, Optional[float]]] = []
    pairs = [(nb[i], nb[j]) for i in range(len(nb)) for j in range(i + 1, len(nb))]
    for (x, y) in pairs[:RATIO_PAIRS]:
        mx, my = mapping.get(int(x)), mapping.get(int(y))
        if mx is None or my is None:
            out.append(("RATIO", None))
            continue
        px, py = pos_q.get(int(x)), pos_q.get(int(y))
        rx, ry = pos_r.get(int(mx)), pos_r.get(int(my))
        if px is None or py is None or rx is None or ry is None:
            out.append(("RATIO", None))
            continue
        ax, ay = px[0] - pu[0], px[1] - pu[1]
        bx, by = py[0] - pu[0], py[1] - pu[1]
        cx, cy = rx[0] - pv[0], rx[1] - pv[1]
        dx, dy = ry[0] - pv[0], ry[1] - pv[1]
        na = math.hypot(ax, ay)
        nb_ = math.hypot(bx, by)
        nc = math.hypot(cx, cy)
        nd = math.hypot(dx, dy)
        if min(na, nb_, nc, nd) < 1e-6:
            out.append(("RATIO", None))
            continue
        ang_q = abs(math.atan2(ax * by - ay * bx, ax * bx + ay * by))
        ang_r = abs(math.atan2(cx * dy - cy * dx, cx * dx + cy * dy))
        dang = abs(ang_q - ang_r)                      # in [0, pi], axial-safe
        lrat = abs(math.log((na / nb_) / (nc / nd)))
        out.append(("RATIO", math.sqrt((dang / FIXED_TOL) ** 2 +
                                       (lrat / FIXED_TOL) ** 2)))
    return out


def ratio_score(u: int, v: int, mapping: Mapping[int, int],
                q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                pos_q: Mapping[int, Tuple[float, float]],
                pos_r: Mapping[int, Tuple[float, float]],
                fit: LLRFit, pair_cap: int = RATIO_PAIRS) -> float:
    """The same padded/clipped/normalised estimator, on frame-free relations."""
    slots = ratio_slots(u, v, mapping, q_inc, pos_q, pos_r)[:pair_cap]
    total = 0.0
    for (_n, res) in slots:
        total += fit.of(res)
    total += float(fit.mu_wrong) * (pair_cap - len(slots))
    return total / (2.0 * LLR_CLIP * pair_cap) + 0.5


# ---------------------------------------------------------------------------
# P_presence: the same estimator with the residual switched off
@dataclass
class PresenceFit:
    """Per-class log ratio of an image edge EXISTING true vs wrong."""

    llr: Dict[str, float]
    mu_wrong: float
    n_true: int
    n_wrong: int


def fit_presence(pairs_true: Sequence[Tuple[str, bool]],
                 pairs_wrong: Sequence[Tuple[str, bool]]) -> PresenceFit:
    """Per-class log ratio of the image edge EXISTING, true vs wrong candidate.

    ``pairs_*`` are ``(class name, does the candidate carry that class of edge)``
    for every slot of every candidate.  The first version of this estimator
    counted each class's SHARE of the existing edges, which is nearly constant
    across candidates and made the presence arm numerically identical to N
    (measured: P_presence - N exactly 0.0000).  The rate of existence *given*
    the class is what varies between candidates, and that is what is fitted now.
    """
    allc = sorted({c for c, _ in pairs_true} | {c for c, _ in pairs_wrong})

    def rate(pairs: Sequence[Tuple[str, bool]], c: str) -> float:
        n = sum(1 for k, _ in pairs if k == c)
        e = sum(1 for k, e in pairs if k == c and e)
        return (e + _PSEUDO) / (n + 2.0 * _PSEUDO)

    llr = {}
    for c in allc:
        pt, pw = rate(pairs_true, c), rate(pairs_wrong, c)
        llr[c] = float(np.clip(np.log(pt / pw), -LLR_CLIP, LLR_CLIP))
    # Padding: the mean slot value a WRONG candidate would receive.
    vals = [llr.get(c, 0.0) if e else 0.0 for c, e in pairs_wrong]
    mu_wrong = float(np.mean(vals)) if vals else 0.0
    return PresenceFit(llr=llr, mu_wrong=mu_wrong, n_true=len(pairs_true),
                       n_wrong=len(pairs_wrong))


def presence_score(u: int, v: int, mapping: Mapping[int, int],
                   q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                   q_idx: Mapping[Tuple[int, int, str], Any],
                   r_idx: Mapping[Tuple[int, int, str], Any],
                   pfit: PresenceFit, slot_cap: int = SLOT_CAP,
                   pad: Optional[float] = None,
                   tol: Optional[float] = None) -> float:
    """Presence credit under P's normalisation, caps and padding."""
    padv = float(pfit.mu_wrong) if pad is None else float(pad)
    slots = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx,
                                tol=tol)[:slot_cap]
    total = 0.0
    for (name, _other, res) in slots:
        # Credit ONLY when the candidate actually carries that class of edge;
        # `res is not None` is exactly "the matching ref edge exists".  The
        # residual itself is never used, which is what makes this the presence
        # arm rather than a second geometry arm.
        if res is not None:
            total += float(pfit.llr.get(name, 0.0))
    total += padv * (slot_cap - len(slots))
    return total / (2.0 * LLR_CLIP * slot_cap) + 0.5


# ---------------------------------------------------------------------------
# P_plus: existence and geometry in ONE estimator.
#
# Mission 7 shipped the two terms as separate arms, so a node was decided either
# by "does the candidate carry the edge" or by "is the residual small" -- never
# by a combination whose weights were themselves fitted.  P_plus keeps both
# terms exactly as they are (same slots, same caps, same padding, same clip,
# same [0, 1] map) and lets ONE pair of coefficients decide how much each is
# worth.  The pair is fitted by maximum likelihood on FIT and then frozen; the
# geometry coefficient is the only thing P_plus_presence removes, which makes
# "P_plus - P_plus_presence" the clean geometry contrast.
PLUS_SLOTS = SLOT_CAP


def plus_terms(u: int, v: int, mapping: Mapping[int, int],
               q_inc: Mapping[int, List[Tuple[int, bool, str]]],
               q_idx: Mapping[Tuple[int, int, str], Any],
               r_idx: Mapping[Tuple[int, int, str], Any],
               fit: "LLRFit", pfit: "PresenceFit",
               slot_cap: int = SLOT_CAP,
               tol: Optional[float] = None,
               normalise: bool = False) -> Tuple[float, float]:
    """Raw (existence, geometry) sums for u -> v, under P's caps and padding.

    ``normalise`` (DARK, PLACEMENT_PAIR_NORMALISED) turns the sum into the mean
    over the slots this pair actually judged: the slots that do not exist are NOT
    padded with ``mu_wrong``.  A sum over a cap of 24 is a coverage statistic -- a
    pair that keeps more edges scores higher even when every one of them is worse
    -- and both retrieval measurements in Mission 10 hit exactly that.

    The two sums are exactly the numerators of ``presence_score`` and
    ``score_candidate``: same slot list, same order, same cap, and the same
    padding value per term for slots that do not exist at all.
    """
    slots = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx,
                                tol=tol)[:slot_cap]
    e_sum = 0.0
    g_sum = 0.0
    for (name, _other, res) in slots:
        if res is not None:
            e_sum += float(pfit.llr.get(name, 0.0))
        g_sum += fit.of(res)
    miss = 0 if normalise else slot_cap - len(slots)
    if miss:
        e_sum += float(pfit.mu_wrong) * miss
        g_sum += float(fit.mu_wrong) * miss
    return e_sum, g_sum


def plus_map(e_sum: float, g_sum: float, a_e: float, a_g: float,
             slot_cap: int = SLOT_CAP) -> float:
    """The same [0, 1] map P uses, applied to the weighted sum."""
    raw = 0.5 + (float(a_e) * e_sum + float(a_g) * g_sum) / (
        2.0 * LLR_CLIP * slot_cap)
    return float(min(1.0, max(0.0, raw)))


def plus_score(u: int, v: int, mapping: Mapping[int, int],
               q_inc: Mapping[int, List[Tuple[int, bool, str]]],
               q_idx: Mapping[Tuple[int, int, str], Any],
               r_idx: Mapping[Tuple[int, int, str], Any],
               fit: "LLRFit", pfit: "PresenceFit",
               a_e: float, a_g: float, slot_cap: int = SLOT_CAP,
               tol: Optional[float] = None,
               normalise: bool = False,
               min_judged: int = 0) -> float:
    """``plus_map`` of ``plus_terms`` -- the deployable arm.

    ``tol`` is the per-pair tolerance of the spacing-aware rule (Mission 10 step
    4a).  It is None unless that dark flag is on, in which case the frozen
    ``FIXED_TOL`` is used exactly as before.

    ``normalise`` (DARK) divides by the slots this pair judged instead of by the
    cap, and stops padding the missing ones -- the mean-quality reading rather
    than the coverage reading.  Off, this function is byte-for-byte the old one.
    """
    e_sum, g_sum = plus_terms(u, v, mapping, q_inc, q_idx, r_idx,
                              fit, pfit, slot_cap=slot_cap, tol=tol,
                              normalise=normalise)
    if not normalise:
        return plus_map(e_sum, g_sum, a_e, a_g, slot_cap=slot_cap)
    _sl = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx,
                              tol=tol)[:slot_cap]
    # [M15 / item 2] "judged" must mean "has a residual".  len(slots) counts the
    # unformable ones too, so both the mean's denominator and the min_judged
    # refusal were computed over evidence that does not exist.
    n = judged_slots(_sl) if _count_judged_slots() else len(_sl)
    # [MISSION 12 / step 2] A MINIMUM JUDGED COUNT.  One or two measurements are
    # not evidence, and reading them on the mean scale gives a pair with a single
    # slot the same weight as a pair with eight.  Below `min_judged` the pair is
    # REFUSED: it contributes exactly 0.5, i.e. 0 to E.  Default 0 keeps every
    # existing caller bit-identical.
    if int(min_judged) > 0 and n < int(min_judged):
        return 0.5
    raw = 0.5 + (float(a_e) * e_sum + float(a_g) * g_sum) / (
        2.0 * LLR_CLIP * max(1, n))
    return float(min(1.0, max(0.0, raw)))


#: Coarse grid, then one refinement.  Chosen before looking at any accuracy: the
#: quantity maximised is the multinomial likelihood of the TRUE candidate, so the
#: grid only has to be dense enough not to quantise the optimum.
PLUS_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
PLUS_REFINE = 6


def _plus_ll(nodes: Sequence[Tuple[Sequence[Tuple[float, float]], int]],
             a_e: float, a_g: float) -> float:
    """Sum of log p(true candidate) under a softmax over the candidates."""
    tot = 0.0
    for terms, ti in nodes:
        if ti < 0 or ti >= len(terms):
            continue
        logits = [a_e * e + a_g * g for (e, g) in terms]
        m = max(logits)
        z = m + float(np.log(np.sum(np.exp([x - m for x in logits]))))
        tot += logits[ti] - z
    return tot


def fit_plus_coefficients(nodes: Sequence[Tuple[Sequence[Tuple[float, float]],
                                                int]]
                          ) -> Tuple[float, float, Dict[str, Any]]:
    """Maximum-likelihood (a_e, a_g) on FIT, by deterministic grid + refine.

    ``nodes`` holds, per node, the (existence, geometry) sums of every candidate
    the assignment sees, plus the index of the TRUE one.  Nothing here touches
    CHECK; the returned pair is meant to be written down and frozen.
    """
    used = [(t, i) for (t, i) in nodes if 0 <= i < len(t)]
    best = (float("-inf"), 0.0, 0.0)
    for a_e in PLUS_GRID:
        for a_g in PLUS_GRID:
            v = _plus_ll(used, a_e, a_g)
            if v > best[0]:
                best = (v, float(a_e), float(a_g))
    step = 0.125
    for _ in range(PLUS_REFINE):
        improved = False
        for d_e in (-step, 0.0, step):
            for d_g in (-step, 0.0, step):
                a_e = max(0.0, best[1] + d_e)
                a_g = max(0.0, best[2] + d_g)
                v = _plus_ll(used, a_e, a_g)
                if v > best[0] + 1e-12:
                    best = (v, a_e, a_g)
                    improved = True
        if not improved:
            step *= 0.5
    meta = {"a_exist": float(best[1]), "a_geom": float(best[2]),
            "loglik": float(best[0]), "n_nodes": len(used),
            "n_candidates_mean": float(np.mean([len(t) for (t, _i) in used]))
            if used else 0.0}
    return best[1], best[2], meta


# ---------------------------------------------------------------------------
# FREEZING.  A deployed matcher cannot fit anything, so the fitted densities
# travel as plain numbers: these two methods are the whole contract.
def _fit_to_dict(fit: "LLRFit") -> Dict[str, Any]:
    return {"edges": [float(x) for x in fit.edges],
            "llr": [float(x) for x in fit.llr],
            "mu_wrong": float(fit.mu_wrong),
            "n_true": int(fit.n_true), "n_wrong": int(fit.n_wrong)}


def _fit_from_dict(d: Mapping[str, Any]) -> "LLRFit":
    return LLRFit(edges=np.asarray(d["edges"], dtype=float),
                  llr=np.asarray(d["llr"], dtype=float),
                  mu_wrong=float(d["mu_wrong"]),
                  n_true=int(d.get("n_true", 0)),
                  n_wrong=int(d.get("n_wrong", 0)))


def _pres_to_dict(p: "PresenceFit") -> Dict[str, Any]:
    return {"llr": {k: float(v) for k, v in p.llr.items()},
            "mu_wrong": float(p.mu_wrong),
            "n_true": int(p.n_true), "n_wrong": int(p.n_wrong)}


def _pres_from_dict(d: Mapping[str, Any]) -> "PresenceFit":
    return PresenceFit(llr={k: float(v) for k, v in d["llr"].items()},
                       mu_wrong=float(d["mu_wrong"]),
                       n_true=int(d.get("n_true", 0)),
                       n_wrong=int(d.get("n_wrong", 0)))


def to_jsonbundle(fit: "LLRFit", pfit: "PresenceFit", a_e: float,
                  a_g: float) -> Dict[str, Any]:
    """Everything the deployed scorer needs, as JSON-able numbers."""
    return {"version": 1,
            "plus": {"a_exist": float(a_e), "a_geom": float(a_g)},
            "fit": _fit_to_dict(fit), "presence": _pres_to_dict(pfit),
            "constants": {"slot_cap": int(SLOT_CAP),
                          "llr_clip": float(LLR_CLIP),
                          "fixed_tol": float(FIXED_TOL),
                          "res_max": float(RES_MAX),
                          "n_bins": int(N_BINS)}}


def from_jsonbundle(b: Mapping[str, Any]) -> Tuple["LLRFit", "PresenceFit", float, float]:
    c = b.get("constants") or {}
    for name, mine in (("slot_cap", SLOT_CAP), ("llr_clip", LLR_CLIP),
                       ("fixed_tol", FIXED_TOL)):
        if name in c and abs(float(c[name]) - float(mine)) > 1e-12:
            raise ValueError(
                "frozen %s %r does not match the live constant %r: a deployed "
                "score must not be read with different caps" %
                (name, c[name], mine))
    return (_fit_from_dict(b["fit"]), _pres_from_dict(b["presence"]),
            float(b["plus"]["a_exist"]), float(b["plus"]["a_geom"]))
