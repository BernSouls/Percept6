"""Calibrate the tight truth rule on the FIRST read's graphs (instrument check)."""
from __future__ import annotations
import json, os, sys
import numpy as np
ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]

def main() -> int:
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology import placement_m12_read_real as R
    reg = json.load(open(os.path.join(ROOT, "Topology", "results", "placement",
                                      "m12_prereg_real.json")))
    imgs = R._images()
    rows = [json.loads(l) for l in open(R.PROGRESS) if l.strip()]
    pick = [r for r in rows if r["photo"] in ("chelsea", "coins", "ihc", "hubble")]
    P.PLACEMENT_MATCHER_ENABLED = False
    out = {}
    for r in pick:
        name, lab = r["photo"], r["transform"]
        spec = {t["label"]: t["spec"] for t in reg["transforms"]}[lab]
        kq, kr = "m12real|%s|%s" % (name, lab), "m12real|%s|orig" % name
        q, g = get_or_build(imgs[kq], kq), get_or_build(imgs[kr], kr)
        truth = R._truth(q, g, spec, imgs[kr].shape[:2])
        res = joint_mcs(q.subgraph, g.subgraph, q.sigs, g.sigs)
        tight = len(truth)
        hit = sum(1 for u, v in truth.items() if res.mapping.get(u, -1) == v)
        mapped = sum(1 for u in truth if u in res.mapping)
        print("%-10s %-9s tight_truth %3d  mapped %3d  acc %.3f  acc|mapped %.3f"
              % (name, lab, tight, mapped, hit / tight if tight else float("nan"),
                 hit / mapped if mapped else float("nan")), flush=True)
        out["%s|%s" % (name, lab)] = {"tight": tight, "mapped": mapped, "hit": hit}
    with open(os.path.join(ROOT, "Topology", "results", "placement",
                           "m12_truth_rule_calib.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
