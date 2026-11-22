"""Task 4 PLACEMENT mission — Part 3: the scorer.

For a CHECKABLE edge ``u -> v`` mapped to ``u' -> v'``:

    compare the QUERY payload of (u, v) with the REFERENCE payload of (u', v'),
    EACH IN ITS OWN SOURCE FRAME,
    support = a bounded function of  residual / tolerance.

Both payloads are already frame-relative (``matching/placement.py``), so the
comparison is invariant to where the two graphs sit in the image: rotating or
translating every node together changes neither payload.  **Nothing here reads
image coordinates.**

    residual = sqrt( (dpsi / tol)^2 + (dlogd / tol)^2 + (dlogscale / tol)^2 )
    support  = exp(-0.5 * residual^2)   for residual <= 3 sigma, else 0

so support is bounded in [0, 1], exactly 1 at residual 0, and **exactly 0**
beyond 3 sigma (the "bounded function", not a long tail).

NO CHECKABLE EDGES = NO CREDIT AND NO PENALTY
---------------------------------------------
An edge whose far end is unmapped, or either endpoint lacks a frame, leaves
BOTH the numerator and the denominator.  ``checkable_fraction`` is always
reported, so "support by mapping less" is visible as a number rather than
hidden inside an average.

PRESENCE IS NEVER REWARDED
--------------------------
There is no default contribution.  An edge that cannot be checked contributes
nothing; a node with no checkable edge has support ``None`` (unmeasured), not
0 and not 1.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: support is EXACTLY zero past this many sigmas
SUPPORT_CUTOFF_SIGMA = 3.0


# ---------------------------------------------------------------------------
# comparing two payloads
# ---------------------------------------------------------------------------
def payload_residual(q: Any, r: Any, tol: float) -> float:
    """The scale-free residual between two placement vectors.

    ``q`` and ``r`` are ``PlacementVector`` (either may be None => caller must
    treat the edge as uncheckable).  Direction is a circular difference, so a
    residual of 2*pi - eps is not "almost a whole turn".
    """
    if q is None or r is None:
        return float("inf")
    if tol is None or not math.isfinite(tol) or tol <= 0.0:
        return float("inf")
    from matching.placement import wrap_pi, wrap_ax
    # AXIAL frames are compared MODULO pi.  A structure-tensor frame is an
    # axis, so a flip of the source's axis is not a different placement; using
    # the signed difference there would penalise a vector for a sign the
    # provider cannot supply.  Both payloads must agree on the convention --
    # they are compared as axes if EITHER says axial, because a signed vector
    # against an axial one is only defined up to the flip.
    if bool(getattr(q, "axial", False)) or bool(getattr(r, "axial", False)):
        dpsi = abs(wrap_ax(float(q.direction) - float(r.direction)))
    else:
        dpsi = abs(wrap_pi(float(q.direction) - float(r.direction)))
    dlogd = abs(math.log(max(float(q.distance), 1e-9))
                - math.log(max(float(r.distance), 1e-9)))
    dlogs = abs(float(q.log_scale) - float(r.log_scale))
    return math.sqrt((dpsi / tol) ** 2 + (dlogd / tol) ** 2 + (dlogs / tol) ** 2)


def support_from_residual(residual: float, sigma: float = 1.0) -> float:
    """Bounded support in [0, 1]; EXACTLY 0 beyond ``SUPPORT_CUTOFF_SIGMA``."""
    if not math.isfinite(residual):
        return 0.0
    z = residual / max(float(sigma), 1e-9)
    if z > SUPPORT_CUTOFF_SIGMA:
        return 0.0
    return float(math.exp(-0.5 * z * z))


# ---------------------------------------------------------------------------
# node-level support
# ---------------------------------------------------------------------------
def node_support(u: int,
                 mapping: Mapping[int, int],
                 q_pay: Mapping[Tuple[int, int, Any], Any],
                 r_pay: Mapping[Tuple[int, int, Any], Any],
                 tol: float,
                 sigma: float = 1.0,
                 max_credit: float = 1.0) -> Optional[Dict[str, Any]]:
    """The placement support for node ``u`` under ``mapping``.

    Returns ``None`` when the node has NO checkable edge -- unmeasured, which
    is not the same as unsupported and must not be scored as 0.

    Each neighbour's contribution is CAPPED at ``max_credit`` (what "cap each
    neighbour's contribution" means), and the node's support is the mean over
    its CHECKABLE neighbours, so an unmapped neighbour neither helps nor hurts.
    """
    from matching.placement import PAYLOAD_CLASSES

    allowed = set(PAYLOAD_CLASSES)
    credits: List[float] = []
    n_total = 0
    n_unmapped = 0
    n_nopayload = 0
    # BOTH DIRECTIONS.  A payload edge is framed in its SOURCE, so when u is
    # the source the frame is u's own; when u is the TARGET the frame is the
    # neighbour's.  Both are legitimate evidence about where u sits, and
    # restricting to the outgoing side left most nodes unmeasured (measured:
    # support None on the first node tried, which would make P collapse onto
    # N and the whole arm vacuous).  The frame is never the target's, so this
    # is not a frame error -- it is two framings of the same node.
    for (s, d, et), qv in sorted(q_pay.items(),
                                 key=lambda kv: (kv[0][0], kv[0][1],
                                                 getattr(kv[0][2], "name", ""))):
        if int(s) == int(u):
            other, is_source = int(d), True
        elif int(d) == int(u):
            other, is_source = int(s), False
        else:
            continue
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        if name not in allowed:
            continue
        n_total += 1
        # THE REFERENCE-SIDE KEY IS THE EDGE'S OWN SOURCE IMAGE.
        # A payload edge (s, d) is framed in ITS source.  On the query side
        # that edge may be stored either way, but on the REFERENCE side the
        # counterpart edge is (mapping[s], mapping[d]) -- NOT (mapping[u], ...)
        # with u forced to the front.  Using the query-side orientation for
        # both sides was the bug: it asked for a reference edge that does not
        # exist and scored every node as having no payload.
        ms, md = mapping.get(int(s)), mapping.get(int(d))
        if ms is None or md is None:
            n_unmapped += 1
            continue
        w = md if is_source else ms
        key_s, key_d = int(ms), int(md)
        rv = None
        for (s2, d2, et2), cand in r_pay.items():
            if int(s2) == key_s and int(d2) == key_d and \
                    (getattr(et2, "name", None) == name):
                rv = cand
                break
        if rv is None:
            n_nopayload += 1
            continue
        res = payload_residual(qv, rv, tol)
        credits.append(min(max_credit, support_from_residual(res, sigma)))
    if not credits:
        return None
    return {
        "support": float(sum(credits) / len(credits)),
        "n_checkable": len(credits),
        "n_edges_from_u": n_total,
        "n_unmapped": n_unmapped,
        "n_no_payload": n_nopayload,
    }


# ---------------------------------------------------------------------------
# mapping-level report: the checkable fraction and the separation
# ---------------------------------------------------------------------------
def mapping_report(mapping: Mapping[int, int],
                   q_pay: Mapping[Tuple[int, int, Any], Any],
                   r_pay: Mapping[Tuple[int, int, Any], Any],
                   tol: float,
                   sigma: float = 1.0) -> Dict[str, Any]:
    """The whole mapping's placement statistics, including the per-arm
    ``checkable_fraction`` the mission asks to be visible."""
    from matching.placement import PAYLOAD_CLASSES

    allowed = set(PAYLOAD_CLASSES)
    n_edges = 0
    n_checkable = 0
    n_unmapped = 0
    n_no_payload = 0
    sups: List[float] = []
    for (s, d, et), qv in sorted(q_pay.items(),
                                 key=lambda kv: (kv[0][0], kv[0][1],
                                                 getattr(kv[0][2], "name", ""))):
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        if name not in allowed:
            continue
        n_edges += 1
        w = mapping.get(d)
        if w is None:
            n_unmapped += 1
            continue
        rv = None
        for (s2, d2, et2), cand in r_pay.items():
            if s2 == mapping.get(s) and d2 == w and \
                    (getattr(et2, "name", None) == name):
                rv = cand
                break
        if rv is None:
            n_no_payload += 1
            continue
        n_checkable += 1
        sups.append(support_from_residual(payload_residual(qv, rv, tol), sigma))
    return {
        "n_payload_edges": n_edges,
        "n_checkable": n_checkable,
        "n_unmapped": n_unmapped,
        "n_no_payload": n_no_payload,
        "checkable_fraction": (n_checkable / n_edges) if n_edges else None,
        "mean_support": (sum(sups) / len(sups)) if sups else None,
    }


# ---------------------------------------------------------------------------
# separation of the TRUE and WRONG residual distributions (Part 3's finding)
# ---------------------------------------------------------------------------
def separation_auc(true_res: Sequence[float],
                   wrong_res: Sequence[float]) -> Optional[float]:
    """AUC of the true-vs-wrong residual separation.

    0.5 means the payload carries NO placement at that view step.  Computed
    exactly (Mann-Whitney U with ties counted 0.5), not approximated.
    """
    t = [x for x in true_res if math.isfinite(x)]
    w = [x for x in wrong_res if math.isfinite(x)]
    if not t or not w:
        return None
    wins = 0.0
    for a in t:
        for b in w:
            if a < b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(t) * len(w))


def d_prime(true_res: Sequence[float], wrong_res: Sequence[float]) -> Optional[float]:
    """d' = (mean_wrong - mean_true) / sqrt(0.5*(var_true + var_wrong)).

    Higher residual = less support, so TRUE should have the LOWER mean.
    """
    import numpy as np
    t = np.asarray([x for x in true_res if math.isfinite(x)], dtype=np.float64)
    w = np.asarray([x for x in wrong_res if math.isfinite(x)], dtype=np.float64)
    if t.size < 2 or w.size < 2:
        return None
    pooled = math.sqrt(0.5 * (float(t.var(ddof=1)) + float(w.var(ddof=1))))
    if pooled <= 0.0:
        return None
    return float((float(w.mean()) - float(t.mean())) / pooled)


def main() -> int:
    import guard
    guard.activate()
    out = os.path.join(ROOT, "Topology", "results", "placement")
    path = os.path.join(out, "tolerance.json")
    if not os.path.exists(path):
        print("run Topology/placement_tolerance.py first", file=sys.stderr)
        return 2
    with open(path) as fh:
        table = json.load(fh)
    print("frozen tolerance table:")
    for k in sorted(table["views"]):
        print("  %-14s tol=%.4f" % (k, table["views"][k]["tolerance"]))
    print("\nScorer module is importable; the frozen table above is what it reads.")
    print("Separation and per-arm numbers are produced by Part 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# CANDIDATE-DEPENDENT support
# ---------------------------------------------------------------------------
# The Part 4 modifier was applied as `sim * (1 + w*(support(u) - 0.5))`, where
# support depended on the SOURCE only.  That factor multiplies EVERY candidate
# of u by the same number, so it cannot change which candidate wins: the arm is
# arithmetically incapable of differing from N.  Measured on the denser graphs:
# net flips 0 of 2373 nodes, twice.  The support has to be read against the
# correspondence the candidate INDUCES -- u -> v -- which is what this does.
def payload_index(pay: Mapping[Tuple[int, int, Any], Any]
                  ) -> Dict[Tuple[int, int, str], Any]:
    """(source, target, class name) -> vector."""
    out: Dict[Tuple[int, int, str], Any] = {}
    for key in pay:
        s, d, et = key
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        out[(int(s), int(d), name)] = pay[key]
    return out


def incident_index(pay: Mapping[Tuple[int, int, Any], Any]
                   ) -> Dict[int, List[Tuple[int, bool, str]]]:
    """node -> [(neighbour, is_source, class name)] for BOTH directions."""
    out: Dict[int, List[Tuple[int, bool, str]]] = {}
    for key in sorted(pay, key=lambda k: (int(k[0]), int(k[1]),
                                          getattr(k[2], "name", ""))):
        s, d, et = key
        name = getattr(et, "name", None) or str(et).replace("EdgeType.", "")
        out.setdefault(int(s), []).append((int(d), True, name))
        out.setdefault(int(d), []).append((int(s), False, name))
    return out


def support_for_candidate(u: int, v: int,
                          mapping: Mapping[int, int],
                          q_inc: Mapping[int, List[Tuple[int, bool, str]]],
                          q_idx: Mapping[Tuple[int, int, str], Any],
                          r_idx: Mapping[Tuple[int, int, str], Any],
                          tol: float,
                          max_credit: float = 1.0,
                          sigma: float = 1.0) -> Optional[float]:
    """Support for the SPECIFIC correspondence u -> v.

    Only u's own incident edges are scored, each framed in ITS source: when u is
    the source the frame is u's; when u is the target the frame is the
    neighbour's.  `mapping` supplies the neighbours' images; u's image is
    overridden by `v`.  Returns None when nothing is checkable -- and None means
    NO CREDIT AND NO PENALTY, never zero support.
    """
    mu = int(v)
    credits: List[float] = []
    for (other, is_source, name) in q_inc.get(int(u), ()):
        mo = mapping.get(int(other))
        if mo is None:
            continue
        if is_source:
            qv = q_idx.get((int(u), int(other), name))
            rv = r_idx.get((mu, int(mo), name))
        else:
            qv = q_idx.get((int(other), int(u), name))
            rv = r_idx.get((int(mo), mu, name))
        if qv is None or rv is None:
            continue
        res = payload_residual(qv, rv, tol)
        credits.append(min(float(max_credit), support_from_residual(res, sigma)))
    if not credits:
        return None
    return float(sum(credits) / len(credits))
