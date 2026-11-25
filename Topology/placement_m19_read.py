"""MISSION 19 / O5 -- the registered read, reduced to its two primaries and its contrast.

The registration (`MISSION19_PREREGISTRATION.md` §5) fixed: arms A0 (M18 frozen), A1 (this
mission), A2 (A1 + soft consensus), A3 (appearance-only of the new descriptor, which the
harness reports as `app_ok` on A1); the primaries P1 = A1 - A0 and P2 = A1 - A3, paired per
image over 20 images, Holm at alpha 0.05; and the soft contrast A2 - A1 with its own
pre-stated margin (no image losing more than 0.05).

Everything here is per image first: a pooled number is printed only after the vector, and
`synthetic` and `photograph` are never pooled together.

    python Topology/placement_m19_read.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = "/home/user/Percept6"
OUT = os.path.join(ROOT, "Topology", "results", "placement")
T_DF19 = 2.093          # two-sided 95 %, 19 df
Z = 1.959964 + 0.841621  # alpha .05 two-sided, power .80


def load(name: str) -> Dict[str, Dict[str, float]]:
    d = json.load(open(os.path.join(OUT, name)))
    out: Dict[str, Dict[str, float]] = {}
    for r in d["rows"]:
        q = int(r["counts"]["queries"])
        if q <= 0:
            continue
        n = len(r["queries"])
        ctx = sum(1 for x in r["queries"] if x.get("ctx_ok"))
        out.setdefault(r["arm"], {})[r["tag"]] = {
            "q": q, "top1": r["counts"]["top1"] / q,
            "app": r["counts"].get("app_ok", 0) / q,
            "ctx": ctx / n if n else None,
            "geo": r.get("auc_geo_matcher"), "n": n}
    return out


def paired(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]],
           key: str = "top1") -> Dict[str, Any]:
    tags = sorted(set(a) & set(b))
    d = [a[t][key] - b[t][key] for t in tags]
    n = len(d)
    if n < 2:
        return {"n": n, "insufficient": True}
    mean = st.mean(d)
    sd = st.stdev(d)
    se = sd / math.sqrt(n)
    t = T_DF19 if n == 20 else 2.1
    return {"n": n, "mean": mean, "sd": sd, "se": se, "median": st.median(d),
            "ci": [mean - t * se, mean + t * se], "min": min(d), "max": max(d),
            "wins": sum(1 for x in d if x > 1e-12),
            "ties": sum(1 for x in d if abs(x) <= 1e-12),
            "losses": sum(1 for x in d if x < -1e-12),
            "mde": Z * sd / math.sqrt(n), "per_image": dict(zip(tags, d))}


def main() -> int:
    a1a2 = load("m19_o5_a1a2.json")
    a0 = load("m19_o5_a0.json")
    res: Dict[str, Any] = {}
    for lab, arms in (("A1", "m18-l0.5-it0"), ("A2", "soft-l0.5-m1-b0.15")):
        if arms in a1a2:
            tags = sorted(a1a2[arms])
            q = sum(a1a2[arms][t]["q"] for t in tags)
            res[lab] = {"images": len(tags), "queries": q,
                        "top1": sum(a1a2[arms][t]["top1"] * a1a2[arms][t]["q"]
                                    for t in tags) / max(q, 1),
                        "app": sum(a1a2[arms][t]["app"] * a1a2[arms][t]["q"]
                                   for t in tags) / max(q, 1)}
    a0arm = next(iter(a0), None)
    if a0arm:
        tags = sorted(a0[a0arm])
        q = sum(a0[a0arm][t]["q"] for t in tags)
        res["A0"] = {"images": len(tags), "queries": q,
                     "top1": sum(a0[a0arm][t]["top1"] * a0[a0arm][t]["q"]
                                 for t in tags) / max(q, 1),
                     "app": sum(a0[a0arm][t]["app"] * a0[a0arm][t]["q"]
                                for t in tags) / max(q, 1)}
    # A3 = the appearance-only arm OF A1 (the harness reports it per cell as `app`)
    a1 = a1a2.get("m18-l0.5-it0", {})
    a3 = {t: {"top1": v["app"], "q": v["q"]} for t, v in a1.items()}
    res["A3 (appearance-only, new descriptor)"] = {
        "images": len(a3),
        "queries": sum(v["q"] for v in a3.values()),
        "top1": (sum(v["top1"] * v["q"] for v in a3.values())
                 / max(sum(v["q"] for v in a3.values()), 1))}
    print("== arms (20 images, rot15)")
    for k, v in res.items():
        print("  %-38s images %2d q %5d top1 %s"
              % (k, v["images"], v["queries"],
                 "%.3f" % v["top1"] if v.get("top1") is not None else "None"))
    out: Dict[str, Any] = {"arms": res}
    if a0arm and "m18-l0.5-it0" in a1a2:
        p1 = paired(a1a2["m18-l0.5-it0"], a0[a0arm])
        out["P1_A1_minus_A0"] = p1
        print("\n== P1  A1 - A0 (top-1)   n=%d mean %+.4f sd %.4f CI [%+.4f,%+.4f] "
              "wins/ties/losses %d/%d/%d MDE %.4f"
              % (p1["n"], p1["mean"], p1["sd"], p1["ci"][0], p1["ci"][1],
                 p1["wins"], p1["ties"], p1["losses"], p1["mde"]))
    if a1 and a3:
        p2 = paired(a1, a3)
        out["P2_A1_minus_A3"] = p2
        print("== P2  A1 - A3 (top-1)   n=%d mean %+.4f sd %.4f CI [%+.4f,%+.4f] "
              "wins/ties/losses %d/%d/%d MDE %.4f"
              % (p2["n"], p2["mean"], p2["sd"], p2["ci"][0], p2["ci"][1],
                 p2["wins"], p2["ties"], p2["losses"], p2["mde"]))
    if "soft-l0.5-m1-b0.15" in a1a2 and a1:
        s = paired(a1a2["soft-l0.5-m1-b0.15"], a1)
        out["O3_soft_minus_A1"] = s
        print("== O3  A2 - A1 (top-1)   n=%d mean %+.4f sd %.4f CI [%+.4f,%+.4f] "
              "wins/ties/losses %d/%d/%d worst image %+.3f (margin -0.05)"
              % (s["n"], s["mean"], s["sd"], s["ci"][0], s["ci"][1],
                 s["wins"], s["ties"], s["losses"], s["min"]))
    # splits, never pooled across regimes
    for lab, sel in (("synthetic", lambda t: t.startswith("syn")),
                     ("photographs", lambda t: t.startswith("real"))):
        for nm, src in (("A0", a0.get(a0arm, {})), ("A1", a1),
                        ("A2", a1a2.get("soft-l0.5-m1-b0.15", {}))):
            tl = [t for t in src if sel(t)]
            if not tl:
                continue
            q = sum(src[t]["q"] for t in tl)
            print("  %-11s %-3s images %2d q %4d top1 %.3f app %.3f"
                  % (lab, nm, len(tl), q,
                     sum(src[t]["top1"] * src[t]["q"] for t in tl) / q,
                     sum(src[t]["app"] * src[t]["q"] for t in tl) / q))
    with open(os.path.join(OUT, "m19_read.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote m19_read.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
