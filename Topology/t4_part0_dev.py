"""Task 4 PART 0 (0e): dev run for B0 and B1 + freeze the branching tolerance.

Dev run = lowest-rank dev instance per class (rank 3), view -090-000.
Shifts 0, +2, -2 are cyclic np.roll of the whole image (translation).
For each arm (B0 = L3/L4 OFF, B1 = L3/L4 ON):
  * mean degree, per-family edge count (edge type value)
  * node correspondence at 0/+2/-2 (counterpart = nearest same-kind node
    within 2 px of the expected shifted position; no-counterpart = incorrect)
  * stability: base-edge survival at +2 vs the shifted-null (random permutation
    of the match mapping)
  * branching: candidates in the shifted graph satisfying the relation's type
    and parameters within a tolerance
  * the counterpart DRIFT at +2 on B1 -> q90 -> FROZEN tolerance

Matcher cell (Task 3 winner, identical for every arm):
  beta=0.75 unmatched=0.15 occlusion=0.5 pooled w=0.1 gamma=0.5
  denom=checkable isolated=downweight MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE=False

Usage:
    python t4_part0_dev.py --arm B0
    python t4_part0_dev.py --arm B1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

# Task 3 winner cell -- identical for every arm.
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
ARMS = {"B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
        "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
        # F: L3/L4 ON + sub-pixel peak position + position-wiring fix
        # (BINDER_COMPLETE_POSITION_MAP is now True in _text.py).
        "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true"}
SHIFTS = (0, 2, -2)
TOL_PRIMARY = 2.0  # px, the 4c primary radius


def _worker_init(spec):
    import guard
    guard.activate()
    import _bootstrap as b
    b.apply_switch_overrides(spec)


def _do(args2):
    """Module-level worker: build one (entry, shift) graph.

    The worker init has already applied the switch spec, so this just builds.
    """
    entry, shift = args2
    e = _roll_entry(entry, shift)
    from frontend import build
    g = build(e)
    return {"name": g.name, "shift": shift,
            "nodes": len(g.subgraph.nodes), "edges": len(g.subgraph.edges),
            "cache": None, "_graph": g}


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    by_class = {}
    for e in split["dev"]:
        if e["view"] != "-090-000":
            continue
        cls = e["class"]
        if cls not in by_class or e["object"] < by_class[cls]["object"]:
            by_class[cls] = e
    return [by_class[c] for c in corpus.CLASSES]


def _pos_lens(g):
    """node -> (x, y, lens) from anchor_of_node + sigs."""
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _roll_entry(entry, shift):
    """Return an entry dict pointing at a rolled PNG for `shift` px."""
    if shift == 0:
        return entry
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "t4_roll" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], shift))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        rolled = np.roll(arr, (shift, 0), axis=(0, 1))
        Image.fromarray(rolled).save(tmp)
    d = dict(entry)
    d["path"] = str(tmp)
    d["roll"] = shift
    return d


def _build_graph(entry, spec):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    return build(entry)


def _counterparts(base_pos, shift_pos, shift_px, tol):
    """Map each base node to its shifted counterpart (same lens, within tol).
    Returns (mapping, drift_list, no_counterpart_count)."""
    # group shifted nodes by lens
    by_lens = defaultdict(list)
    for n, (x, y, l) in shift_pos.items():
        by_lens[l].append((n, x, y))
    mapping = {}
    drifts = []
    no_c = 0
    for n, (x0, y0, l0) in base_pos.items():
        ex, ey = x0 + shift_px, y0
        best_n, best_d = None, tol
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            d = float(math.hypot(ex - xsh, ey - ysh))
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is None:
            no_c += 1
        else:
            mapping[n] = best_n
            drifts.append(best_d)
    return mapping, drifts, no_c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    spec = ARMS[args.arm]
    os.environ["PERCEPT_SET"] = spec

    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)

    entries = _dev_entries()
    print("arm", args.arm, "dev entries:",
          ["%s%d%s" % (e["class"], int(e["object"]), e["view"]) for e in entries],
          flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    tasks = [(e, s) for e in entries for s in SHIFTS]
    t0 = time.perf_counter()
    graphs = {}
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(spec,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_do, tasks, chunksize=1)):
            graphs[(rec["name"], rec["shift"])] = rec
            if (i + 1) % 6 == 0 or i + 1 == len(tasks):
                print("  built %3d/%d (%.0fs)"
                      % (i + 1, len(tasks), time.perf_counter() - t0),
                      flush=True)

    # ---- per-scene stats + correspondence + stability + branching ----
    per_scene = {}
    all_drifts_b1 = []
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        base = graphs[(name, 0)]["_graph"]
        bp = _pos_lens(base)
        deg = len(base.subgraph.edges) / max(1, len(base.subgraph.nodes))
        fam = Counter(int(getattr(et, "value", et))
                      for (_u, _v, et) in base.subgraph.edges)
        row = {"name": name, "nodes": len(base.subgraph.nodes),
               "edges": len(base.subgraph.edges), "degree": deg,
               "families": {str(k): int(v) for k, v in fam.items()}}
        for shift in (2, -2):
            sh = graphs[(name, shift)]["_graph"]
            sp = _pos_lens(sh)
            mapping, drifts, no_c = _counterparts(bp, sp, shift, TOL_PRIMARY)
            # node correspondence accuracy (D1: no-counterpart = incorrect)
            acc = len(mapping) / max(1, len(bp))
            # stability: base-edge survival under the mapping
            surv, null_surv = 0, 0
            b_edges = list(base.subgraph.edges)
            # shifted-null: random permutation of the match mapping
            rng = np.random.default_rng(1234 + shift)
            perm = list(mapping.values())
            rng.shuffle(perm)
            null_map = dict(zip(mapping.keys(), perm))
            for (u, v, et) in b_edges:
                eu, ev = mapping.get(u), mapping.get(v)
                if eu is not None and ev is not None:
                    key = (min(eu, ev), max(eu, ev), et)
                    if key in sh.subgraph.edges or (
                            (min(eu, ev), max(eu, ev), et) in
                            {(min(a, b), max(a, b), c)
                             for (a, b, c) in sh.subgraph.edges}):
                        surv += 1
                eu, ev = null_map.get(u), null_map.get(v)
                if eu is not None and ev is not None:
                    key = (min(eu, ev), max(eu, ev), et)
                    if key in sh.subgraph.edges or (
                            (min(eu, ev), max(eu, ev), et) in
                            {(min(a, b), max(a, b), c)
                             for (a, b, c) in sh.subgraph.edges}):
                        null_surv += 1
            row["shift_%d" % shift] = {
                "corr_acc": acc, "n_base": len(bp),
                "n_mapped": len(mapping), "n_no_counterpart": no_c,
                "drift_q90": float(np.quantile(drifts, 0.9)) if drifts else None,
                "edge_survival": surv / max(1, len(b_edges)),
                "null_survival": null_surv / max(1, len(b_edges)),
            }
            if args.arm == "B1" and shift == 2:
                all_drifts_b1.extend(drifts)
        per_scene[name] = row

    # ---- aggregate ----
    degs = [r["degree"] for r in per_scene.values()]
    fam_total = Counter()
    for r in per_scene.values():
        fam_total.update({int(k): v for k, v in r["families"].items()})
    accs = {s: [r["shift_%d" % s]["corr_acc"] for r in per_scene.values()]
            for s in (2, -2)}
    survs = {s: [r["shift_%d" % s]["edge_survival"] for r in per_scene.values()]
             for s in (2, -2)}
    nulls = {s: [r["shift_%d" % s]["null_survival"] for r in per_scene.values()]
             for s in (2, -2)}

    frozen_tol = None
    if all_drifts_b1:
        frozen_tol = float(np.quantile(all_drifts_b1, 0.9))

    out = {
        "arm": args.arm, "switch_spec": spec, "winner": WINNER,
        "shifts": list(SHIFTS), "tol_primary_px": TOL_PRIMARY,
        "mean_degree": float(np.mean(degs)),
        "per_family_total": {str(k): int(v) for k, v in fam_total.items()},
        "per_scene": per_scene,
        "node_corr_acc_mean": {s: float(np.mean(v)) for s, v in accs.items()},
        "edge_survival_mean": {s: float(np.mean(v)) for s, v in survs.items()},
        "null_survival_mean": {s: float(np.mean(v)) for s, v in nulls.items()},
        "frozen_tolerance_px": frozen_tol,
        "frozen_tolerance_n": len(all_drifts_b1),
        "build_seconds": time.perf_counter() - t0,
    }
    write_json(RESULTS_DIR / "t4" / ("part0_dev_%s.json" % args.arm), out)
    print("arm", args.arm, "mean_degree %.4f" % out["mean_degree"],
          "corr_acc +2 %.4f -2 %.4f"
          % (out["node_corr_acc_mean"][2], out["node_corr_acc_mean"][-2]),
          "surv +2 %.4f (null %.4f)"
          % (out["edge_survival_mean"][2], out["null_survival_mean"][2]),
          flush=True)
    if frozen_tol is not None:
        print("FROZEN tolerance (B1 +2 q90) = %.4f px (n=%d)"
              % (frozen_tol, len(all_drifts_b1)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
