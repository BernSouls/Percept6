"""Task 4 PART 4e: junction recall, precision, survival with n.

Uses the held-out pickles. For the best arm (F) and shifts 0, ±2, ±3:
  junction recall   = fraction of base junction nodes with a counterpart
                      in the shifted graph (within 2 px)
  junction precision = fraction of matcher-mapped junction nodes that are
                      correct counterparts
  junction survival = fraction of base junction nodes whose matcher mapping
                      equals the counterpart
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

SHIFTS = (0, 2, -2, 3, -3)
TOL = 2.0
HOLDOUT_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _counterpart(base_pos, sh_pos, shift_px, tol):
    by_lens = defaultdict(list)
    for n, (x, y, l) in sh_pos.items():
        by_lens[l].append((n, x, y))
    out = {}
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = x0 + shift_px, y0
        best_n, best_d = None, tol
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            d = math.hypot(ex - xsh, ey - ysh)
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is not None:
            out[n] = best_n
    return out


def _run_matcher(base_g, sh_g):
    import parameters as P
    WINNER = {
        "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
        "MATCH_UNMATCHED_COST": 0.15,
        "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
        "MATCH_EDGE_WEIGHT": 0.1,
        "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
        "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
        "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
        "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
    }
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, sh_g.subgraph,
                    base_g.sigs, sh_g.sigs)
    return res.mapping


def _load_all():
    graphs = {}
    for p in HOLDOUT_DIR.glob("*.pkl"):
        parts = p.stem.rsplit("_", 1)
        arm_name = parts[0]
        shift = int(parts[1])
        arm, name = arm_name.split("_", 1)
        with open(p, "rb") as fh:
            graphs[(arm, name, shift)] = pickle.load(fh)
    return graphs


def main() -> int:
    best = "F"
    graphs = _load_all()
    img_names = sorted({n for (_a, n, _s) in graphs})
    print("loaded %d graphs, %d images" % (len(graphs), len(img_names)),
          flush=True)

    out = {"best_arm": best, "shifts": list(SHIFTS), "tol_px": TOL,
           "per_shift": {}}
    for shift in SHIFTS:
        recalls, precisions, survivals, ns = [], [], [], []
        for name in img_names:
            base = graphs[(best, name, 0)]
            sh_g = graphs[(best, name, shift)]
            base_pos = _pos_lens(base)
            sh_pos = _pos_lens(sh_g)
            cp = _counterpart(base_pos, sh_pos, shift, TOL)
            mapping = _run_matcher(base, sh_g)
            # junction nodes
            junc = [n for n, (x, y, l) in base_pos.items() if l == "Junction"]
            if not junc:
                continue
            n = len(junc)
            have_cp = sum(1 for x in junc if x in cp)
            recall = have_cp / n
            # precision: of shifted junction nodes mapped by matcher,
            # how many are correct counterparts
            sh_junc = [n2 for n2, (x, y, l) in sh_pos.items()
                       if l == "Junction"]
            inv = {v2: k2 for k2, v2 in cp.items()}
            mapped_junc = [n2 for n2 in sh_junc if n2 in mapping.values()]
            correct = 0
            for n2 in mapped_junc:
                bn = inv.get(n2)
                if bn is not None and mapping.get(bn) == n2:
                    correct += 1
            precision = correct / len(mapped_junc) if mapped_junc else 0.0
            surv = sum(1 for x in junc
                       if x in cp and mapping.get(x) == cp.get(x)) / n
            recalls.append(recall)
            precisions.append(precision)
            survivals.append(surv)
            ns.append(n)
        out["per_shift"][str(shift)] = {
            "recall_mean": float(np.mean(recalls)) if recalls else None,
            "precision_mean": float(np.mean(precisions)) if precisions else None,
            "survival_mean": float(np.mean(survivals)) if survivals else None,
            "n_junctions": int(np.sum(ns)) if ns else 0,
            "n_scenes": len(ns),
        }
        r = out["per_shift"][str(shift)]
        print("shift=%+d recall=%.4f precision=%.4f survival=%.4f "
              "(n=%d, scenes=%d)"
              % (shift, r["recall_mean"] or 0, r["precision_mean"] or 0,
                 r["survival_mean"] or 0, r["n_junctions"], r["n_scenes"]),
              flush=True)
    write_json(RESULTS_DIR / "t4" / "measure_4e.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
