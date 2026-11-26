"""Diagnostic: does the joint OBJECTIVE prefer the matcher's mapping or the
truth-corrected one?

err_shape.py showed that 122 of the 137 transposition errors prefer the WRONG
partner on node-signature evidence alone.  This asks the follow-up: given the
alignment's own objective (sum over pairs of sim*(1-beta) + beta*rel), which
mapping scores higher -- the matcher's, or the one with the cycles undone?

    objective(corrected) > objective(matcher)  ->  the SEARCH failed to find it
    objective(corrected) < objective(matcher)  ->  the OBJECTIVE prefers it

    python obj_check.py --shift 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import guard

guard.activate()

TOL = 8.0
BETA = 0.5


def _build(image_path, shift):
    import eth80_frontend as FE
    from PIL import Image
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if shift != 0:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="obj_%d" % shift, record_co_located=False,
                       trace=trace)
    cg = trace["cg"]
    positions = cg.pattern_node_position_map()
    anchors = cg.pattern_node_anchor_map()
    anchor_pos = trace.get("anchor_positions") or {}
    for n in [n for n in g.subgraph.nodes if n not in positions]:
        a = anchors.get(n)
        p = anchor_pos.get(a) if a is not None else None
        if p is not None and len(p) >= 2:
            positions[n] = (float(p[0]), float(p[1]))
    lens = {n: g.sigs[n][0] for n in g.subgraph.nodes if n in g.sigs}
    return g, positions, lens


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(Path(guard.TREE) / "data" /
                                           "ETH-80-master" / "1" / "2" /
                                           "apple2-022-000.png"))
    ap.add_argument("--shift", type=int, default=2)
    args = ap.parse_args(argv)

    base, bpos, blens = _build(args.image, 0)
    other, opos, olens = _build(args.image, args.shift)

    expected = {u: (p[0] + args.shift, p[1]) for u, p in bpos.items()}
    cands = []
    for u, (ex, ey) in expected.items():
        lu = blens.get(u)
        for v, (px, py) in opos.items():
            if olens.get(v) != lu:
                continue
            d = math.hypot(px - ex, py - ey)
            if d <= TOL:
                cands.append((d, u, v))
    cands.sort()
    uu, vv, truth = set(), set(), {}
    for d, u, v in cands:
        if u in uu or v in vv:
            continue
        uu.add(u); vv.add(v); truth[u] = v

    from matching.joint_mcs import joint_mcs
    from memory.cores import _sig_similarity
    import matching.alignment as AL
    from matching.joint_mcs import edge_pair_credit

    res = joint_mcs(base.subgraph, other.subgraph, base.sigs, other.sigs)
    m = dict(res.mapping)

    na, nb = sorted(base.subgraph.nodes), sorted(other.subgraph.nodes)
    sig = {}
    for u in na:
        su = base.sigs.get(u)
        for v in nb:
            sv = other.sigs.get(v)
            if su is None or sv is None or su[0] != sv[0]:
                sig[(u, v)] = 0.0
            else:
                sig[(u, v)] = float(_sig_similarity(su[1], sv[1]))

    inc_a = AL.build_incidence(tuple(base.subgraph.edges),
                               anchors=getattr(base.subgraph, "anchor_of_node", None),
                               junctions=getattr(base.subgraph, "t_junctions", None) or ())
    inc_b = AL.build_incidence(tuple(other.subgraph.edges),
                               anchors=getattr(other.subgraph, "anchor_of_node", None),
                               junctions=getattr(other.subgraph, "t_junctions", None) or ())
    a_dir, a_und = AL._edge_indexes(base.subgraph.edges)
    b_dir, b_und = AL._edge_indexes(other.subgraph.edges)

    def edge_sim(x, y, u, v, et):
        return edge_pair_credit(base.subgraph, other.subgraph, x, y, u, v, et)

    def objective(mp):
        rev = {v: u for u, v in mp.items()}
        total = 0.0
        per = {}
        for u, v in sorted(mp.items()):
            s = sig.get((u, v), 0.0)
            if s <= 0.0:
                continue
            rel = AL._relational_agreement(
                u, v, mp, inc_a, inc_b, b_dir, b_und, a_dir, a_und, rev,
                edge_sim, checkable=True)
            q = s * (1.0 - BETA) + BETA * rel
            per[(u, v)] = q
            total += q
        return total, per

    # the truth-corrected mapping: start from the matcher's, then undo every
    # transposition the truth exposes (only where the truth partner is free or
    # held by a partner we also move -- i.e. genuine cycles).
    inv_m = {v: u for u, v in m.items()}
    corrected = dict(m)
    moved = set()
    for u, tv in truth.items():
        mv = corrected.get(u)
        if mv == tv or u in moved:
            continue
        holder = inv_m.get(tv)
        if holder is None:
            continue                      # the truth partner is unmatched
        # swap u and holder
        hu = corrected.pop(holder, None)
        corrected[u] = tv
        corrected[holder] = mv
        moved.add(u); moved.add(holder)
        inv_m = {v: k for k, v in corrected.items()}

    o_all, _ = objective(m)
    o_fix, _ = objective(corrected)

    def parts(mp):
        s_sum = r_sum = 0.0
        n = 0
        for u, v in sorted(mp.items()):
            s = sig.get((u, v), 0.0)
            if s <= 0.0:
                continue
            rel = AL._relational_agreement(
                u, v, mp, inc_a, inc_b, b_dir, b_und, a_dir, a_und,
                {vv: kk for kk, vv in mp.items()}, edge_sim, checkable=True)
            s_sum += s
            r_sum += rel
            n += 1
        return s_sum, r_sum, n

    sa, ra, na_ = parts(m)
    sf, rf, nf_ = parts(corrected)
    print("  decomposition:")
    print("    matcher  : sim %.3f  rel %.3f  pairs %d" % (sa, ra, na_))
    print("    corrected: sim %.3f  rel %.3f  pairs %d" % (sf, rf, nf_))
    print("    at beta %.2f: matcher %.3f  corrected %.3f"
          % (BETA, (1 - BETA) * sa + BETA * ra, (1 - BETA) * sf + BETA * rf))
    hits_before = sum(1 for u, v in m.items()
                      if truth.get(u) == v)
    hits_after = sum(1 for u, v in corrected.items()
                     if truth.get(u) == v)

    print("shift %+d   beta %.2f" % (args.shift, BETA))
    print("  matcher  : %d pairs, %d correct" % (len(m), hits_before))
    print("  corrected: %d pairs, %d correct (+%d)" % (len(corrected), hits_after,
                                                     hits_after - hits_before))
    print("  objective(matchel) = %.6f" % o_all)
    print("  objective(corrected) = %.6f   delta %+.6f" % (o_fix, o_fix - o_all))
    print("  VERDICT:", "SEARCH failed (corrected scores HIGHER)"
          if o_fix > o_all + 1e-9 else
          "OBJECTIVE prefers the matcher's mapping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
