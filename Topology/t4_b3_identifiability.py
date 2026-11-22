"""Task 4 PART B3 — IDENTIFIABILITY TEST (dev only).

For each base node with an unambiguous counterpart, measure how often the true
counterpart is RANKED FIRST among same-lens shifted candidates under:
  (1) sig-only:        raw _sig_similarity (node_sim), no structure.
  (2) sig+1WL real:    sig similarity + 1-WL structural color (real edges).
  (3) sig+1WL rewired: sig similarity + 1-WL structural color (rewired edges).
  (4) rounds:          1-WL rounds to color stability (per scene, mean).

Reading:
  sig-only already ranks true counterpart first for most nodes → signatures
  alone identify almost everything; 1WL adds little.
  sig+1WL real > sig-only → structure adds identifiability.
  sig+1WL real ≈ sig+1WL rewired → the SPECIFIC edge topology does not matter;
    only the coarse degree/coloring does.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

D_CLASS = frozenset({"CONTOUR_PEAK", "Corner", "PEAK_CHAIN",
                     "RegionShape", "Orientation"})
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
TIE_TOL = 1e-9


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_dev_graph(arm, name):
    with open(PKL_DIR / ("%s_%s.pkl" % (arm, name)), "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _build_shifted(entry, spec, dx):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_b3" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], dx))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        shifted = roll_image(arr, dx, 0)
        Image.fromarray(shifted).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _node_sim(sig_a, sig_b):
    from memory.cores import _sig_similarity
    return _sig_similarity(sig_a, sig_b)


def _rewire_edges(edges, seed):
    from core.types import UNDIRECTED_EDGE_TYPES
    rng = random.Random(seed)
    und = [(min(a, b), max(a, b), t) for a, b, t in edges
           if t in UNDIRECTED_EDGE_TYPES]
    direct = [(a, b, t) for a, b, t in edges if t not in UNDIRECTED_EDGE_TYPES]
    out = set(und)
    for _ in range(20 * max(1, len(und))):
        if len(und) < 2:
            break
        i, j = rng.sample(range(len(und)), 2)
        a, b, t1 = und[i]
        c, d, t2 = und[j]
        if t1 != t2 or a == b or c == d:
            continue
        cand = [(a, d, t1), (c, b, t2)]
        if any(x == y for x, y, _ in cand):
            continue
        before = set(out)
        out.discard((a, b, t1))
        out.discard((c, d, t2))
        out.update(cand)
        if len(out) != len(before):
            out = before
            continue
        und[i] = cand[0]
        und[j] = cand[1]
    return out | set(direct)


def _rewired_base(g, seed):
    new_edges = _rewire_edges(g.subgraph.edges, seed)
    sg = replace(g.subgraph, edges=frozenset(new_edges), edge_params={})
    return replace(g, subgraph=sg)


def _wl_colors(nodes, edges, sigs, max_rounds=10):
    """1-WL coloring: initial color = (lens, sig-hash); recolor by neighbor
    color multiset. Returns (color dict, rounds to stability)."""
    from collections import Counter as C
    # initial color: lens + a coarse sig feature (class/relation)
    cmap = {}
    for n in nodes:
        s = sigs.get(n)
        if s is None:
            cmap[n] = ("?",)
        else:
            cmap[n] = (str(s[0]), str(s[1].class_), str(s[1].relation))
    # neighbor multiset per node
    nbrs = defaultdict(set)
    for a, b, t in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    for r in range(max_rounds):
        new_cmap = {}
        for n in nodes:
            new_cmap[n] = (cmap[n], tuple(sorted(C(nbrs[n]).items())))
        # re-hash to compact colors
        if len(set(new_cmap.values())) == len(set(cmap.values())):
            # check if coloring is stable (same partition)
            stable = True
            # compare partitions
            old_groups = defaultdict(set)
            new_groups = defaultdict(set)
            for n in nodes:
                old_groups[cmap[n]].add(n)
                new_groups[new_cmap[n]].add(n)
            if set(map(frozenset, old_groups.values())) == \
               set(map(frozenset, new_groups.values())):
                return cmap, r
        cmap = new_cmap
    return cmap, max_rounds


def _rank_first_rate(base_sig, candidates, score_fn):
    """Fraction where the best-scored candidate is the true one.
    candidates: list of (sh_node, sh_sig, is_true)."""
    if not candidates:
        return None
    scores = [(i, score_fn(base_sig, sh_sig))
              for i, (sh_node, sh_sig, is_true) in enumerate(candidates)]
    max_s = max(s for _, s in scores)
    top = [i for i, s in scores if abs(s - max_s) < TIE_TOL]
    if len(top) == 1:
        true_idx = next(i for i, (sh_node, sh_sig, is_true)
                        in enumerate(candidates) if is_true)
        return 1.0 if top[0] == true_idx else 0.0
    # tie: true is among top → fractional credit 1/len(top)
    true_idx = next(i for i, (sh_node, sh_sig, is_true)
                    in enumerate(candidates) if is_true)
    if true_idx in top:
        return 1.0 / len(top)
    return 0.0


def _paired_interval(a_vals, b_vals):
    n = len(a_vals)
    if n == 0:
        return None
    diffs = np.array(a_vals) - np.array(b_vals)
    mean = float(diffs.mean())
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    return {"mean": mean,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
            "hw": (float(np.quantile(boots, 0.975))
                   - float(np.quantile(boots, 0.025))) / 2}


def main() -> int:
    print("=== B3: IDENTIFIABILITY TEST ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    # Accumulate per-case rank-first rates
    rates = {"sig_only": defaultdict(list),
             "sig_1wl_real": defaultdict(list),
             "sig_1wl_rewired": defaultdict(list)}
    rounds_list = []
    case_counts = Counter()

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)

        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, ARM_B1, dx)
            sh_pos = _pos_lens(sh_g)
            cp, cp_info, _cp_stats = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)

            # 1WL colors
            bl_colors, bl_rounds = _wl_colors(
                base.subgraph.nodes, base.subgraph.edges, base.sigs)
            sh_colors, sh_rounds = _wl_colors(
                sh_g.subgraph.nodes, sh_g.subgraph.edges, sh_g.sigs)
            rounds_list.append((bl_rounds + sh_rounds) / 2)

            # rewired colors (seed 0)
            rb = _rewired_base(base, 0)
            rb_colors, _ = _wl_colors(
                rb.subgraph.nodes, rb.subgraph.edges, rb.sigs)

            # group shifted nodes by lens
            sh_by_lens = defaultdict(list)
            for nsh in sh_g.subgraph.nodes:
                sh_sig = sh_g.sigs.get(nsh)
                if sh_sig is None:
                    continue
                sh_by_lens[str(sh_sig[0])].append((nsh, sh_sig[1]))

            for bn, (bx, by, blens) in base_pos.items():
                if blens not in D_CLASS:
                    continue
                bsh = cp.get(bn)
                if bsh is None or bsh not in sh_pos:
                    continue
                b_sig = base.sigs.get(bn)
                if b_sig is None:
                    continue
                sh_sig = sh_g.sigs.get(bsh)
                if sh_sig is None:
                    continue
                true_lens = str(sh_sig[0])
                cands = [(nsh, sh_sig_v, nsh == bsh)
                         for nsh, sh_sig_v in sh_by_lens.get(true_lens, [])]

                # case from node_sim (same as B1)
                s_true = _node_sim(b_sig[1], sh_sig[1])
                comp = [(nsh, s) for nsh, sv, is_t in cands if not is_t
                        for s in [_node_sim(b_sig[1], sv)]]
                if not comp:
                    case = "STRICT-LEAD"
                else:
                    mx = max(s for _, s in comp)
                    T = sum(1 for _, s in comp if abs(s - mx) < TIE_TOL) + 1
                    if s_true > mx + TIE_TOL:
                        case = "STRICT-LEAD"
                    elif abs(s_true - mx) < TIE_TOL and T >= 2:
                        case = "EXACT-TIE"
                    else:
                        case = "OUTRANKED"
                case_counts[case] += 1

                # sig-only score
                def s_sig(bs, ss):
                    return _node_sim(bs, ss)

                # sig+1WL score: combine node_sim and color match
                def s_1wl_real(bs, ss, sh_node):
                    ns = _node_sim(bs, ss)
                    # color match: 1.0 if WL color equal else 0.0
                    # find color of sh_node
                    col = sh_colors.get(sh_node)
                    bcol = bl_colors.get(bn)
                    cm = 1.0 if col == bcol else 0.0
                    return 0.5 * ns + 0.5 * cm

                def s_1wl_rewired(bs, ss, sh_node):
                    ns = _node_sim(bs, ss)
                    col = sh_colors.get(sh_node)
                    bcol = rb_colors.get(bn)
                    cm = 1.0 if col == bcol else 0.0
                    return 0.5 * ns + 0.5 * cm

                r1 = _rank_first_rate(
                    b_sig[1], cands,
                    lambda bs, sv: s_sig(bs, sv))
                # sig+1WL real: combine node_sim and WL color match
                scores_real = [s_1wl_real(b_sig[1], sv, nsh)
                               for nsh, sv, is_t in cands]
                max_s = max(scores_real)
                top = [i for i, s in enumerate(scores_real)
                       if abs(s - max_s) < TIE_TOL]
                true_idx = next(i for i, (nsh, sv, is_t)
                                in enumerate(cands) if is_t)
                r2 = 1.0 / len(top) if true_idx in top else 0.0

                scores_rw = [s_1wl_rewired(b_sig[1], sv, nsh)
                             for nsh, sv, is_t in cands]
                max_s = max(scores_rw)
                top = [i for i, s in enumerate(scores_rw)
                       if abs(s - max_s) < TIE_TOL]
                r3 = 1.0 / len(top) if true_idx in top else 0.0

                if r1 is not None:
                    rates["sig_only"][case].append(r1)
                rates["sig_1wl_real"][case].append(r2)
                rates["sig_1wl_rewired"][case].append(r3)

    total = sum(case_counts.values())
    print("\n--- Case shares ---", flush=True)
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d (%.3f)"
              % (case[0], case_counts[case],
                 case_counts[case] / total if total else 0))

    print("\n--- Rank-first rates (true counterpart ranked best) ---",
          flush=True)
    results = {}
    for scheme, rdict in rates.items():
        results[scheme] = {}
        for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
            vals = rdict.get(case, [])
            if vals:
                results[scheme][case] = {
                    "n": len(vals), "rate": float(np.mean(vals))}
                print("  %-16s %s: %.4f (n=%d)"
                      % (scheme, case, np.mean(vals), len(vals)))

    # contrasts sig+1WL real vs sig-only
    print("\n--- Contrasts ---", flush=True)
    contrasts = {}
    for case in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        a = list(rates["sig_1wl_real"].get(case, []))
        b = list(rates["sig_only"].get(case, []))
        c = list(rates["sig_1wl_rewired"].get(case, []))
        contrasts[case] = {
            "sig1wl_real_minus_sig_only": _paired_interval(a, b),
            "sig1wl_real_minus_rewired": _paired_interval(a, c),
            "n": len(a),
        }
        for k, v in contrasts[case].items():
            if k == "n":
                print("  %s n=%d" % (case, v))
            elif v:
                print("  %s %-32s %.4f [%.4f, %.4f] hw=%.4f"
                      % (case, k, v["mean"], v["lo"], v["hi"], v["hw"]))

    out["rank_first_rates"] = results
    out["contrasts"] = contrasts
    out["wl_rounds_mean"] = float(np.mean(rounds_list)) if rounds_list else None
    print("\n  WL rounds mean: %.1f" % out["wl_rounds_mean"], flush=True)

    write_json(RESULTS_DIR / "t4" / "b3_identifiability.json", out)
    print("\n=== B3 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
