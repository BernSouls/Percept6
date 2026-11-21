"""Supplement: run the solvers on the identity (0 px) pair for each arm."""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import guard

guard.activate()

import known_transform_arms as ARMS
import known_transform as KT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(Path(guard.TREE) / "data" /
                                           "ETH-80-master" / "1" / "2" / "apple2-022-000.png"))
    args = ap.parse_args(argv)
    from PIL import Image
    base = KT._worker(((0, 0), args.image, 256))
    base["seconds"] = 0.0
    other = base
    truth, expected = KT._match_continuous(base, other, (0, 0))
    tset = {(u, v) for u, v in truth.items()}
    arms = {
        "A_baseline_joint_mcs": lambda: ARMS._arm_joint_mcs(base, other),
        "B_graded_mcs_anchor_prior": lambda: ARMS._arm_graded_mcs(base, other, True),
        "C_graded_mcs_anchor_prior_no_req_edge": lambda: ARMS._arm_graded_mcs(base, other, False),
    }
    for name, fn in arms.items():
        t0 = time.perf_counter()
        matcher, score, diag = fn()
        secs = time.perf_counter() - t0
        hits = sum(1 for (u, v) in matcher.items()
                   if (u in expected and v in other["positions"]
                       and math.hypot(other["positions"][v][0] - expected[u][0],
                                      other["positions"][v][1] - expected[u][1]) <= 8.0))
        inter = len(tset & set(matcher.items()))
        print("%-40s score %.6f pairs %4d hits %4d frac %.4f jaccard %.4f  (%.1fs)"
              % (name, score, len(matcher), hits, hits / max(1, len(matcher)),
                 inter / max(1, len(tset | set(matcher.items()))), secs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
