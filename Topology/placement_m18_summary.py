"""Per-image, per-cell readout for the M18 result files.  No pooled headline.

Every rate here is printed with its n and its per-image spread, and the grass photograph
is shown both inside and outside the pooled figure (it is 656 of 1,166 rot15 queries and
its context accuracy is 0.061 against 0.269 for the photographs and the hatch, so a pooled
number is mostly a grass number).

    python Topology/placement_m18_summary.py results/placement/m18_full_cap6.json [more...]
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Topology", "results", "placement")


def load(name: str) -> dict:
    p = name if os.path.isabs(name) else os.path.join(OUT, name)
    return json.load(open(p))


def cell(r: dict) -> dict:
    n = len(r.get("queries") or [])
    c = r["counts"]
    q = int(c.get("queries") or n)
    ctx = sum(1 for x in (r.get("queries") or []) if x.get("ctx_ok"))
    return {"q": q, "top1": c.get("top1", 0) / max(q, 1),
            "app": c.get("app_ok", 0) / max(q, 1),
            "ctx": (ctx / n) if n else float("nan"),
            "geo": r.get("auc_geo_matcher"), "ceiling": r.get("auc_geo_truth")}


def main(argv: list) -> int:
    for name in argv or ["m18_full_cap6.json"]:
        d = load(name)
        rows = d["rows"]
        cap = None
        for r in rows:
            cap = r.get("config", {}).get("cap", cap)
        print("\n=== %s" % os.path.basename(name))
        by_arm = collections.OrderedDict()
        for r in rows:
            by_arm.setdefault(r["arm"], []).append(r)
        for arm, rs in by_arm.items():
            print("  arm %s" % arm)
            poses = sorted({r["pose"] for r in rs})
            for pose in poses:
                cells = [r for r in rs if r["pose"] == pose]
                print("    %-8s %-34s %-26s %-26s" % (pose, "per image", "pooled", "pooled w/o grass"))
                line_img = []
                for r in cells:
                    c = cell(r)
                    line_img.append("%s %.3f/%.3f/%.3f" % (r["tag"].split("|")[1][:7],
                                                           c["top1"], c["app"], c["ctx"]))
                print("      %s" % "  ".join(line_img))
                for label, sel in (("pooled", cells),
                                   ("no-grass", [r for r in cells if "grass" not in r["tag"]])):
                    q = sum(cell(r)["q"] for r in sel)
                    t1 = sum(r["counts"].get("top1", 0) for r in sel)
                    ap = sum(r["counts"].get("app_ok", 0) for r in sel)
                    ctx = sum(sum(1 for x in (r.get("queries") or []) if x.get("ctx_ok"))
                              for r in sel)
                    n = sum(len(r.get("queries") or []) for r in sel)
                    geo = [r["auc_geo_matcher"] for r in sel if r.get("auc_geo_matcher")]
                    sp = [cell(r)["top1"] for r in sel]
                    print("      %-8s q %5d  top1 %.3f  app %.3f  ctx %.3f  "
                          "geo %.3f (gap %.3f)  per-image %s sd %.3f"
                          % (label, q, t1 / max(q, 1), ap / max(q, 1), ctx / max(n, 1),
                             st.mean(geo) if geo else float("nan"),
                             0.902 - (st.mean(geo) if geo else float("nan")),
                             "/".join("%.3f" % x for x in sp),
                             st.pstdev(sp) if len(sp) > 1 else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
