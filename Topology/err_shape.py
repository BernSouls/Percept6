"""Diagnostic: what is the SHAPE of the matcher's +2px error?

Answers the question the go/no-go instrument cannot: the coupled 2-swap exists
to undo PAIRWISE TRANSPOSITIONS.  If the matcher's errors are transpositions,
the move should find them; if they are matches to an unrelated node, no
one-to-one local move can.

    python err_shape.py --shift 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR  # noqa: E402

TOL = 8.0


def _build(image_path, shift):
    import eth80_frontend as FE
    from PIL import Image
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if shift != 0:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="diag_%d" % shift, record_co_located=False,
                       trace=trace)
    cg = trace["cg"]
    positions = cg.pattern_node_position_map()
    anchors = cg.pattern_node_anchor_map()
    anchor_pos = trace.get("anchor_positions") or {}
    missing = [n for n in g.subgraph.nodes if n not in positions]
    for n in missing:
        a = anchors.get(n)
        p = anchor_pos.get(a) if a is not None else None
        if p is not None and len(p) >= 2:
            positions[n] = (float(p[0]), float(p[1]))
    lens = {n: g.sigs[n][0] for n in g.subgraph.nodes if n in g.sigs}
    return g, positions, lens


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(Path(guard.TREE).parent / "data" /
                                           "ETH-80-master" / "1" / "2" /
                                           "apple2-022-000.png"))
    ap.add_argument("--shift", type=int, default=2)
    args = ap.parse_args(argv)

    base, bpos, blens = _build(args.image, 0)
    other, opos, olens = _build(args.image, args.shift)

    # the continuous truth, exactly as known_transform builds it
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
    used_u, used_v, truth = set(), set(), {}
    for d, u, v in cands:
        if u in used_u or v in used_v:
            continue
        used_u.add(u)
        used_v.add(v)
        truth[u] = v

    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base.subgraph, other.subgraph, base.sigs, other.sigs)
    m = dict(res.mapping)

    inv_truth = {v: u for u, v in truth.items()}
    inv_m = {v: u for u, v in m.items()}

    edge_set = {(min(s, d), max(s, d)) for (s, d, _et) in base.subgraph.edges}
    image_edge_set = {(min(m.get(s, s), m.get(d, d)), max(m.get(s, s), m.get(d, d)))
                      for (s, d, _et) in base.subgraph.edges
                      if m.get(s) is not None and m.get(d) is not None}

    # classify every pair the TRUTH expects but the matcher got wrong
    wrong = []          # (u, truth_v, matcher_v)
    for u, tv in truth.items():
        mv = m.get(u)
        if mv != tv:
            wrong.append((u, tv, mv))

    transposition = 0     # the truth partner is matched to my partner
    unrelated = 0         # neither
    dropped = 0           # u is not matched at all
    adj = 0               # ...and the two are ADJACENT (a swap candidate)
    adj_edges = 0
    for u, tv, mv in wrong:
        if mv is None:
            dropped += 1
            continue
        if inv_m.get(tv) == mv or inv_m.get(tv) is not None:
            transposition += 1
            partner = inv_m.get(tv)
            if partner is not None and (
                    (min(u, partner), max(u, partner)) in edge_set
                    or (min(mv, tv), max(mv, tv)) in image_edge_set):
                adj += 1
        else:
            unrelated += 1

    print("shift %+d" % args.shift)
    print("  nodes: base %d other %d" % (len(base.subgraph.nodes),
                                         len(other.subgraph.nodes)))
    print("  truth pairs %d   matcher pairs %d" % (len(truth), len(m)))
    print("  matcher score %.4f" % res.score)
    print("  mismatched truth pairs: %d" % len(wrong))
    print("    transposition/cycle : %d" % transposition)
    print("    unrelated node      : %d" % unrelated)
    print("    query node unmatched: %d" % dropped)
    print("    of the transpositions, ADJACENT (a swap candidate): %d" % adj)

    # Is the NODE evidence enough to tell the two partners apart?
    from memory.cores import _sig_similarity
    s_uv = s_uw = s_up_w = s_up_v = 0.0
    ties = 0
    node_prefers_truth = 0
    node_prefers_wrong = 0
    for u, tv, mv in wrong:
        if mv is None or inv_m.get(tv) is None:
            continue
        up = inv_m[tv]
        su = base.sigs.get(u)
        sv = other.sigs.get(tv)
        sw = other.sigs.get(mv)
        sup = base.sigs.get(up)
        swp = other.sigs.get(m.get(up))
        if None in (su, sv, sw, sup, swp):
            continue
        d_uv = _sig_similarity(su[1], sv[1])
        d_uw = _sig_similarity(su[1], sw[1])
        d_pv = _sig_similarity(sup[1], swp[1])
        d_pw = _sig_similarity(sup[1], sw[1])
        if abs((d_uv + d_pw) - (d_uw + d_pv)) < 1e-9:
            ties += 1
        elif (d_uv + d_pw) > (d_uw + d_pv):
            node_prefers_truth += 1
        else:
            node_prefers_wrong += 1
    total = ties + node_prefers_truth + node_prefers_wrong
    print("  node-signature evidence on the transpositions (%d pairs):" % total)
    print("    identical (unresolvable by signature) : %d" % ties)
    print("    prefers the TRUTH                     : %d" % node_prefers_truth)
    print("    prefers the WRONG partner             : %d" % node_prefers_wrong)
    return 0


if __name__ == "__main__":
    sys.exit(main())
