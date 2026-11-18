"""Why the LV('winner') cell collapsed 229 -> 56 matched nodes.

At beta=0.75, unmatched_cost=0.15, isolated weight 0.5 (the tuned cell):
   q(i,j) for a pair with an EDGELESS side = sim*(1-beta) + beta*0
        = (0.5*sig) * 0.25 = 0.125 * sig <= 0.125 < 0.15.
   Such a pair is UNMATCHABLE at any relational agreement -- the price of
   going unmatched is never paid back.  Connected pairs keep rel as an upside
   (q <= 0.25*sig + 0.75*rel), so they can still pay the price.

Prints, for the transfer pair (apple1-090-000 apple4-035-135 by default):
  1. degree-0 counts on both sides (share of nodes),
  2. the q distribution of ISOLATED pairs vs CONNECTED candidates
     (appearance-only bound for the latter; rel sampled for a sample),
  3. how many candidate pairs clear the price,
  4. the denominator decomposition for the winner cell.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import guard

guard.activate()

import parameters as P
import solver
import tune
from frontend import load
from memory.cores import _sig_similarity

HERE = Path(__file__).resolve().parent
WINNER = json.loads((HERE / "results" / "tune" / "winner.json").read_text())

BETA = float(WINNER["beta"])
W = float(WINNER["w"])
GAMMA = float(WINNER["gamma"])
UMC = float(WINNER["unmatched"])
ISO = 0.5
BETA_1 = 1.0 - BETA


def deg(sg):
    d = {}
    for s, e, _t in sg.edges:
        d[s] = d.get(s, 0) + 1
        d[e] = d.get(e, 0) + 1
    return d


def q(sig, rel, isolated):
    s = (ISO if isolated else 1.0) * sig
    return s * BETA_1 + BETA * rel


def main():
    data = json.loads((HERE / "results" / "tune" / "graphs.json").read_text())
    query = next(m for m in data["dev"] if m["name"] == "apple4-035-135")
    ref = sorted(data["references"], key=lambda m: m["name"])[0]
    qw, rw = load(query["cache"]), load(ref["cache"])
    r = rw.subgraph
    qsub = qw.subgraph

    node_credit_cache = {}

    def sig(u, v):
        key = (u, v)
        got = node_credit_cache.get(key)
        if got is None:
            su, sv = rw.sigs.get(u), qw.sigs.get(v)
            if su is None or sv is None or su[0] != sv[0]:
                got = 0.0
            else:
                got = float(_sig_similarity(su[1], sv[1]))
            node_credit_cache[key] = got
        return got

    na, nb = sorted(r.nodes), sorted(qsub.nodes)
    deg_r, deg_q = deg(r), deg(qsub)
    iso_r = [u for u in na if deg_r.get(u, 0) == 0]
    iso_q = [v for v in nb if deg_q.get(v, 0) == 0]
    print("transfer pair: ref %s (nodes %d, degree-0 %d, %.1f%%)" % (
        ref["name"], len(na), len(iso_r), 100.0 * len(iso_r) / max(1, len(na))))
    print("                 query %s (nodes %d, degree-0 %d, %.1f%%)" % (
        query["name"], len(nb), len(iso_q), 100.0 * len(iso_q) / max(1, len(nb))))
    print("winner: beta %.2f  w %.1f  gamma %.1f  unmatched %.2f  isolated weight %s" % (
        BETA, W, GAMMA, UMC, WINNER["isolated"]))
    print("edgeless side: q = %.2f * downweighted_sig = 0.125 * sig <= %.4f" % (BETA_1, 0.125))

    # the distribution of q for ISOLATED pairs (rel == 0, exact)
    iso_pairs = 0
    iso_clear = 0
    iso_max = 0.0
    hist_bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sig_pos_iso = []
    for u in iso_r:
        su = rw.sigs.get(u)
        for v in nb:
            sv = qw.sigs.get(v)
            if su is None or sv is None or su[0] != sv[0]:
                continue
            s = sig(u, v)
            if s <= 0.0:
                continue
            qq = BETA_1 * ISO * s
            iso_pairs += 1
            if qq > UMC:
                iso_clear += 1
            iso_max = max(iso_max, qq)
            hist_bins[min(10, int(qq * 10))] += 1
            sig_pos_iso.append(qq)

    # and a sample for CONNECTED candidates on the appearance-only bound
    # (a connected pair clears the price in the BEST case with rel = 1:
    #  q_max = 0.25*sig + 0.75)
    import random

    rng = random.Random(1729)
    conn_samples = 0
    conn_clear_best = 0
    conn_clear_app_only = 0
    tries = 0
    max_tries = 40000
    us = [u for u in na if deg_r.get(u, 0) > 0]
    while conn_samples < 2000 and tries < max_tries:
        tries += 1
        u = us[rng.randrange(len(us))]
        v = nb[rng.randrange(len(nb))]
        sv = qw.sigs.get(v)
        su = rw.sigs.get(u)
        if su is None or sv is None or su[0] != sv[0]:
            continue
        s = sig(u, v)
        if s <= 0.0:
            continue
        conn_samples += 1
        conn_clear_best += 1            # best case rel = 1: 0.25*s + 0.75 > 0.15 always
        conn_clear_app_only += 1 if 0.25 * s > UMC else 0

    sys.stdout.write("\nISOLATED candidates (same lens, sig > 0): %d\n" % iso_pairs)
    print("  q distribution (11 bins of width 0.1, = 0.125*sig):")
    for i, c in enumerate(hist_bins):
        if c:
            print("    [%.1f, %.1f)  %d" % (i / 10, (i + 1) / 10, c))
    print("  max q = %.4f   pairs clearing unmatched_cost = %d (%.2f%%)" % (
        iso_max, iso_clear, 100.0 * iso_clear / max(1, iso_pairs)))
    bound = BETA_1 * ISO
    print("  THE PRICE IS %.2f; the best possible isolated q is %.3f (=%.2f*1.0) -> unmatchable"
          % (UMC, bound, bound))

    sys.stdout.write("\nCONNECTED candidates (sample %d, same lens, sig > 0):\n" % conn_samples)
    print("  clear by appearance alone (0.25*sig > %.2f): %d (%.1f%%)" % (
        UMC, conn_clear_app_only, 100.0 * conn_clear_app_only / max(1, conn_samples)))
    print("  clear in the best case (rel = 1):        %d (%.1f%%)" % (
        conn_clear_best, 100.0 * conn_clear_best / max(1, conn_samples)))

    # ---------------- the denominator decomposition ----------------
    sys.stdout.write("\nDENOMINATOR DECOMPOSITION (apple1-090-000 -> apple4-035-135, winner cell)\n")
    base = {
        "MATCH_SYSTEMATICITY_WEIGHT": BETA,
        "MATCH_UNMATCHED_COST": WINNER["unmatched"],
        "MATCH_ALIGN_UNMAPPED_CREDIT": WINNER["occlusion"],
        "MATCH_EDGE_WEIGHT": W,
        "MATCH_COVERAGE_EXPONENT_GAMMA": GAMMA,
        "MATCH_EDGE_DENOMINATOR_MODE": WINNER["denom"],
        "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
        "MATCH_UNMATCHED_COST_DERIVED": False,
        "MATCH_SEPARATE_TERM_NORMALIZATION": False,
        "MATCH_ISOLATED_NODE_WEIGHT": 0.5,
    }
    for k, v in base.items():
        setattr(P, k, v)
    sim = solver.node_sim_matrix(rw, qw, rw.sigs, qw.sigs, isolated_weight=0.5)
    sim_t = tune._transpose(sim)
    fwd = solver.solve_direction(rw, qw, rw.sigs, qw.sigs, sim, beta=BETA,
                                 unmatched_cost=UMC, unmapped_credit=float(WINNER["occlusion"]))
    rev = solver.solve_direction(qw, rw, qw.sigs, rw.sigs, sim_t, beta=BETA,
                                 unmatched_cost=UMC, unmapped_credit=float(WINNER["occlusion"]))
    ff = solver.facts(rw, qw, rw.sigs, qw.sigs, sim, fwd)
    rf = solver.facts(qw, rw, qw.sigs, rw.sigs, sim_t, rev)
    # production picked fwd (downweight arm); decompose both, report the selected
    for tag, f in (("fwd", ff), ("rev", rf)):
        checkable_a, checkable_b = f.checkable()
        pac, pbc = f.presence_checkable()
        mass_min = 2 * min(checkable_a - pac, checkable_b - pbc)
        mass = mass_min ** GAMMA
        node_mass = f.na + f.nb
        print("  [%s] na=%d nb=%d  checkable_a=%d checkable_b=%d  presence_only_checkable=%d/%d" %
              (tag, f.na, f.nb, checkable_a, checkable_b, pac, pbc))
        print("        checkable mass = 2*min(%d-%d, %d-%d) = %d ;  gamma=%.1f -> edge_mass=%.6f" %
              (checkable_a, pac, checkable_b, pbc, mass_min, GAMMA, mass))
        print("        denominator = na+nb+edge_mass = %d + %d + %.6f = %.6f" % (
            f.na, f.nb, mass, node_mass + mass))
        w01, comp = solver.score(f, w=W, gamma=GAMMA, denom_mode="checkable",
                                 isolated="keep", presence_counts=WINNER["presence"])
        print("        score = 2*(node_credit + w*edge_credit)/denom")
        print("              = 2*(%.6f + %.1f*%.6f)/%.6f = %.6f" % (
            comp["node_credit"], W, comp["edge_credit"], comp["denom"], w01))
    print("\n  Production reads this exact form (joint_mcs.py, mode != 'all' branch)."
          " gamma only exponents the edge mass; node mass is linear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
