"""STAGE 0 -- re-analyse the SPENT EVAL v1 at the scene level.

This is an analysis correction on data that has already been read, not a new
read: the arms run is re-executed only to emit per-node records, and it
reproduces the published table bit-for-bit (``arms_eval.json`` hashes to
``9d421df10728d0bc`` before and after).  Nothing here is allowed to choose
anything.

Why it matters
--------------
EVAL v1 clustered on ``family|scene|view`` -- 24 clusters, 4 scenes x 6 views.
Views of one scene are not independent: they share the scene's geometry,
materials and node layout, and they differ only by a pose applied to the same
graph.  Treating them as 24 independent clusters reports an interval that is
far too narrow.  The top-level unit is the SCENE, and EVAL v1 has FOUR of them.

With four clusters a bootstrap cannot say anything useful, so the scene-level
test here is an EXACT sign-flip permutation over the 2^4 = 16 sign assignments
of the four per-scene paired mean differences.  The two-sided p is therefore
floored at 2/16 = 0.125: **no contrast on EVAL v1 can reach 0.05 at the scene
level, whatever the effect size.**

Reported for each contrast:
  * the published view-clustered interval (for comparison),
  * the four per-scene paired differences,
  * the equal-weight mean over scenes,
  * the exact two-sided sign-flip p,
  * a scene-level bootstrap interval (descriptive only, n = 4).
"""
from __future__ import annotations

import json
import os
from itertools import product
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Topology", "results", "placement")
NODES = os.path.join(OUT, "eval_nodes.json")
ARMS = os.path.join(OUT, "arms_eval.json")

CONTRASTS = (("P", "N"), ("P", "P_shuffled"), ("P", "W"),
             ("R", "N"), ("R_fixed", "R"))


def load() -> List[Dict[str, Any]]:
    with open(NODES) as fh:
        return json.load(fh)


def scene_diffs(recs: Sequence[Dict[str, Any]], a: str, b: str,
                bucket: str = "ambiguous") -> Dict[str, float]:
    """Per-scene paired mean difference in accuracy, equal weight per scene."""
    per: Dict[str, List[float]] = {}
    for r in recs:
        if r["bucket"] != bucket:
            continue
        per.setdefault(r["scene"], []).append(
            float(r["ok"][a]) - float(r["ok"][b]))
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}


def exact_sign_flip(diffs: Dict[str, float]) -> Dict[str, Any]:
    """Exact two-sided sign-flip test over the scene-level differences."""
    keys = sorted(diffs)
    obs = float(np.mean([diffs[k] for k in keys]))
    n = len(keys)
    hits = 0
    total = 0
    for signs in product((1.0, -1.0), repeat=n):
        val = float(np.mean([s * diffs[k] for s, k in zip(signs, keys)]))
        total += 1
        if abs(val) >= abs(obs) - 1e-15:
            hits += 1
    return {"mean_equal_weight": obs, "p_exact_two_sided": hits / total,
            "n_sign_assignments": total, "n_scenes": n,
            "p_floor": 2.0 / total}


def scene_boot(diffs: Dict[str, float], nboot: int = 2000,
               seed: int = 20261118) -> Dict[str, float]:
    """Scene-level bootstrap.  Descriptive only: with n = 4 this is coarse."""
    keys = sorted(diffs)
    vals = np.array([diffs[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = np.array([float(np.mean(vals[rng.integers(0, len(vals), len(vals))]))
                      for _ in range(nboot)])
    lo, hi = float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
    return {"lo": lo, "hi": hi, "mde": 2.8 * ((hi - lo) / 2.0) / 1.96}


def family_table(recs: Sequence[Dict[str, Any]], bucket: str = "ambiguous"
                 ) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for fam in sorted({r["family"] for r in recs}):
        rows = [r for r in recs if r["family"] == fam and r["bucket"] == bucket]
        if not rows:
            continue
        out[fam] = {"n": len(rows),
                    "N": float(np.mean([r["ok"]["N"] for r in rows])),
                    "P": float(np.mean([r["ok"]["P"] for r in rows])),
                    "P_shuffled": float(np.mean([r["ok"]["P_shuffled"] for r in rows]))}
    return out


def main() -> int:
    recs = load()
    with open(ARMS) as fh:
        published = json.load(fh)

    res: Dict[str, Any] = {
        "source": "spent EVAL v1 (one read, arms_eval.json sha256 9d421df10728d0bc)",
        "clustering_v1": "family|scene|view -> 24 clusters (4 scenes x 6 views)",
        "n_records": len(recs),
        "n_scenes": len({r["scene"] for r in recs}),
        "contrasts": {},
    }

    print("Stage 0 -- EVAL v1 re-analysed with SCENE as the top-level cluster\n")
    print("scenes: %s" % ", ".join(sorted({r["scene"] for r in recs})))
    print("nodes : %d total, %d near-tie ambiguous, %d unambiguous\n"
          % (len(recs), sum(1 for r in recs if r["bucket"] == "ambiguous"),
             sum(1 for r in recs if r["bucket"] == "unambiguous")))

    hdr = ("%-14s %8s %8s %8s %8s %9s %8s" %
           ("contrast", "v1_mean", "v1_lo", "v1_hi", "scene_m", "p_exact", "sc_boot_mde"))
    print(hdr)
    print("-" * len(hdr))
    for a, b in CONTRASTS:
        diffs = scene_diffs(recs, a, b)
        ex = exact_sign_flip(diffs)
        boot = scene_boot(diffs)
        pub = published["contrasts"].get("ambiguous|%s-%s" % (a, b), {})
        res["contrasts"]["%s-%s" % (a, b)] = {
            "per_scene": diffs, **ex, "scene_boot": boot,
            "published_v1": {k: pub.get(k) for k in ("mean", "lo", "hi", "mde")},
        }
        print("%-14s %8.4f %8.4f %8.4f %8.4f %9.3f %8.4f"
              % ("%s-%s" % (a, b), pub.get("mean", float("nan")),
                 pub.get("lo", float("nan")), pub.get("hi", float("nan")),
                 ex["mean_equal_weight"], ex["p_exact_two_sided"], boot["mde"]))

    print("\nper-scene paired differences (near-tie ambiguous):")
    for a, b in CONTRASTS:
        d = res["contrasts"]["%s-%s" % (a, b)]["per_scene"]
        print("  %-16s " % ("%s-%s" % (a, b))
              + "  ".join("%s %+.3f" % (k.replace("eval_", ""), v)
                          for k, v in d.items()))

    res["by_family_ambiguous"] = family_table(recs, "ambiguous")
    print("\nby family (near-tie ambiguous), n / N / P / P_shuffled:")
    for fam, v in res["by_family_ambiguous"].items():
        print("  %-12s %5d  %.3f  %.3f  %.3f"
              % (fam, v["n"], v["N"], v["P"], v["P_shuffled"]))

    path = os.path.join(OUT, "stage0_scene_level.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
