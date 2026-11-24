"""MISSION 11 STEP 5b -- why the pear class inverts.

Mission 10's ETH-80 read found the placement score LOWER than node similarity on
exactly one class: pear, .50 -> .00 on eight queries.  Eight queries is small, but
"small" is not an explanation.  This walks the same chain as the coverage probe on
the pear class and on one contrast class, and prints, per candidate, everything the
pair score is built from: how many nodes the aligner mapped, how many of those pairs
it could judge, the evidence E, and the two scores.  If the inversion is in the code
it will be visible as a broken link somewhere in that chain; if it is in the data it
will be visible as a class that has nothing to judge.

Deterministic: the same fresh-query catalog as the ETH read, first four queries of
each class, same-class K = 8 gallery.  Report-only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
CLASSES = ("pear", "apple")


def _load_u8(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def catalog() -> List[Tuple[str, dict, List[dict]]]:
    """(class, query, its eight same-class references) for the first four queries."""
    from Topology.placement_m11_eth80 import catalog as eth_catalog
    refs, queries = eth_catalog()
    out = []
    for cls in CLASSES:
        same = [r for r in refs if r["cls"] == cls]
        qs = [q for q in queries if q["cls"] == cls][:4]
        for q in qs:
            out.append((cls, q, same))
    return out


def _pair(job: Tuple[str, dict, dict]) -> Dict[str, Any]:
    cls, q, r = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        w = float(json.load(fh)["w"])
    q_img, r_img = _load_u8(q["path"]), _load_u8(r["path"])
    qg = get_or_build(q_img, "eth|" + q["path"])
    rg = get_or_build(r_img, "eth|" + r["path"])
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    a = joint_mcs(qg.subgraph, rg.subgraph, qg.sigs, rg.sigs)
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = w
    c = joint_mcs(qg.subgraph, rg.subgraph, qg.sigs, rg.sigs,
                  placement_images=(q_img, r_img))
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    return {"cls": cls, "q": q["key"] + "|" + q["view"], "r": r["key"],
            "truth": r["key"] == q["key"], "A": float(a.score),
            "B": float(c.score_shipped), "C": float(c.score),
            "E": float(c.placement_evidence), "judged": int(c.placement_judged),
            "mapped": len(c.mapping), "share": float(c.placement_share_judged),
            "nodes_q": len(list(qg.subgraph.nodes)),
            "nodes_r": len(list(rg.subgraph.nodes))}


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor
    import guard
    guard.activate()
    jobs = [(cls, q, r) for cls, q, refs in catalog() for r in refs]
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(_pair, jobs):
            rows.append(r)
            print("  %-5s %-22s %-12s A %.4f B %.4f C %.4f E %+.4f j %d/%d  %.0fs"
                  % (r["cls"], r["q"].split("/")[-1], r["r"], r["A"], r["B"], r["C"],
                     r["E"], r["judged"], r["mapped"], time.time() - t0), flush=True)
    res: Dict[str, Any] = {"per_class": {}, "pairs": len(rows),
                           "seconds": time.time() - t0}
    for cls in CLASSES:
        sel = [r for r in rows if r["cls"] == cls]
        qs = sorted({r["q"] for r in sel})
        ranks = {}
        for arm in ("A", "B", "C"):
            t1, rr = [], []
            for q in qs:
                cands = sorted([r for r in sel if r["q"] == q],
                               key=lambda r: (-r[arm], r["r"]))
                rank = [r["r"] for r in cands].index(
                    [c["r"] for c in cands if c["truth"]][0]) + 1
                t1.append(1.0 if rank == 1 else 0.0)
                rr.append(1.0 / rank)
            ranks[arm] = {"top1": float(np.mean(t1)), "mrr": float(np.mean(rr)),
                          "n": len(t1)}
        res["per_class"][cls] = {
            "queries": len(qs), "arms": ranks,
            "judged_mean": float(np.mean([r["judged"] for r in sel])),
            "mapped_mean": float(np.mean([r["mapped"] for r in sel])),
            "share_mean": float(np.mean([r["share"] for r in sel])),
            "E_mean": float(np.mean([r["E"] for r in sel])),
            "E_truth_mean": float(np.mean([r["E"] for r in sel if r["truth"]])),
            "E_wrong_mean": float(np.mean([r["E"] for r in sel if not r["truth"]]))
            if any(not r["truth"] for r in sel) else float("nan"),
            "nodes_q_mean": float(np.mean([r["nodes_q"] for r in sel]))}
    with open(os.path.join(OUT, "m11_pear.json"), "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    for cls, d in sorted(res["per_class"].items()):
        print("%-6s n=%d  A %.3f B %.3f C %.3f  judged %.0f mapped %.0f share %.3f "
              "E truth %+.4f wrong %+.4f nodes %.0f"
              % (cls, d["queries"], d["arms"]["A"]["top1"], d["arms"]["B"]["top1"],
                 d["arms"]["C"]["top1"], d["judged_mean"], d["mapped_mean"],
                 d["share_mean"], d["E_truth_mean"], d["E_wrong_mean"],
                 d["nodes_q_mean"]))
    print("wrote %s" % os.path.join(OUT, "m11_pear.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
