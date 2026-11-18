"""STEP H — why does the walker admit ZERO seeds on renderer-truth scenes?

External instrumentation only (matching/relation_walker.py is frozen).
For 3 scenes at 0/+2px: build the pair, construct the walker's own _Index,
and count, per seeding gate, how many candidates are rejected and why.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import guard

guard.activate()

import matching.relation_walker as RW  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
SHIFTS = (0, 2)


def _build(scene, shift):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(SCENES_DIR / (scene + ".png")).convert("RGB"),
                       dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="h_%s_%d" % (scene, shift),
                       record_co_located=False, trace=trace)
    return g


def _diag(scene, shift):
    g = _build(scene, shift)
    sg = g.subgraph
    sigs = g.sigs
    gq, gr = RW._Index(sg, sigs), RW._Index(sg, sigs)
    # same graph here (identity) to expose the gate behaviour
    print("== %s shift %+d  frame %d / %d" % (scene, shift, len(gq.frame), len(gr.frame)))
    print("   SEED_TOPOLOGICAL =", RW.SEED_TOPOLOGICAL,
          " SEED_MIN_SHARED_TOKENS =", RW.SEED_MIN_SHARED_TOKENS,
          " RARE_TOKEN_MAX_NODES =", RW.RARE_TOKEN_MAX_NODES,
          " NODE_FLOOR =", RW.NODE_FLOOR,
          " BUDGET_FRACTION =", RW.BUDGET_FRACTION)
    # count carriers and token sharing directly
    def carriers(ix):
        out = {}
        for n in ix.frame:
            for t in ix.tok.get(n, ()):
                out.setdefault(t, []).append(n)
        return out
    cq, cr = carriers(gq), carriers(gr)
    shared = Counter()
    rare = Counter()
    for t in sorted(cq):
        if t not in cr:
            continue
        is_rare = (len(cq[t]) <= RW.RARE_TOKEN_MAX_NODES
                   and len(cr[t]) <= RW.RARE_TOKEN_MAX_NODES)
        for u in cq[t]:
            for v in cr[t]:
                shared[(u, v)] += 1
                if is_rare:
                    rare[(u, v)] += 1
    print("   tokens shared by >=1 pair: %d pairs carry >=1 shared token" % len(shared))
    dist = Counter(shared.values())
    print("   shared-token-count distribution:", dict(sorted(dist.items())))
    ge2 = sum(1 for v in shared.values() if v >= RW.SEED_MIN_SHARED_TOKENS)
    print("   pairs with >= %d shared tokens: %d" % (RW.SEED_MIN_SHARED_TOKENS, ge2))
    # now the measurement-bin path (the default): count relations that AGREE
    # in one bin between same nodes (no metrics needed)
    # NOTE: walk() itself produces 0 seeds here; this counts the ingredient.
    # frames' token vocab
    tok_vocab = set(cq) | set(cr)
    print("   distinct tokens per index: q %d r %d union %d" % (len(cq), len(cr), len(tok_vocab)))
    # how many frame nodes carry tokens at all
    with_tokens = sum(1 for n in gq.frame if gq.tok.get(n))
    print("   frame nodes carrying >=1 token: %d / %d" % (with_tokens, len(gq.frame)))
    # what do the tokens look like? sample the most common
    tok_cnt = Counter({t: len(us) for t, us in cq.items()})
    for t, n in tok_cnt.most_common(8):
        print("     token %s -> %d carriers" % (t, n))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["single_cylinder", "box_occlusion",
                                                   "disk_occlusion_haze"])
    args = ap.parse_args(argv)
    for scene in args.scenes:
        for shift in SHIFTS:
            _diag(scene, shift)
    return 0


if __name__ == "__main__":
    sys.exit(main())
