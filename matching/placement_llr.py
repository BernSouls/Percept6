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


def slots_for_candidate(u: int, v: int,
                        mapping: Mapping[int, int],
                        q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                        q_idx: Mapping[Tuple[int, int, str], Any],
                        r_idx: Mapping[Tuple[int, int, str], Any]
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
                    float(payload_residual(qv, rv, FIXED_TOL))))
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
                    fit: LLRFit, slot_cap: int = SLOT_CAP) -> float:
    """The padded, clipped, summed LLR score for u -> v, mapped to [0, 1]."""
    slots = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx)[:slot_cap]
    total = 0.0
    for (_name, _other, res) in slots:
        total += fit.of(res)
    # PAD the unused slots with the wrong-pair expectation.
    total += float(fit.mu_wrong) * (slot_cap - len(slots))
    return total / (2.0 * LLR_CLIP * slot_cap) + 0.5


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
                   pfit: PresenceFit, slot_cap: int = SLOT_CAP) -> float:
    """Presence credit under P's normalisation, caps and padding."""
    slots = slots_for_candidate(u, v, mapping, q_inc, q_idx, r_idx)[:slot_cap]
    total = 0.0
    for (name, _other, res) in slots:
        # Credit ONLY when the candidate actually carries that class of edge;
        # `res is not None` is exactly "the matching ref edge exists".  The
        # residual itself is never used, which is what makes this the presence
        # arm rather than a second geometry arm.
        if res is not None:
            total += float(pfit.llr.get(name, 0.0))
    total += float(pfit.mu_wrong) * (slot_cap - len(slots))
    return total / (2.0 * LLR_CLIP * slot_cap) + 0.5
