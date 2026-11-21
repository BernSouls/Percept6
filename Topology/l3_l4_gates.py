"""L3+L4 family gates (+2px preservation, same/diff AUC) and STEP 0b loss classification."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402

OUT = ROOT / "results" / "l3_l4_gates.json"
CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}


def _match_nodes(s0, tr0, s2, tr2, tol=3.0):
    p0 = {
        n: tr0["cg"].pattern_node_position(n)
        for n in s0.subgraph.nodes
        if str(s0.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
        and tr0["cg"].pattern_node_position(n) is not None
    }
    p2 = {
        n: tr2["cg"].pattern_node_position(n)
        for n in s2.subgraph.nodes
        if str(s2.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
        and tr2["cg"].pattern_node_position(n) is not None
    }
    mapping = {}
    for n0, (x0, y0) in p0.items():
        l0 = str(s0.sigs.get(n0, ("?",))[0])
        best_n2, best_d = None, tol
        for n2, (x2, y2) in p2.items():
            if str(s2.sigs.get(n2, ("?",))[0]) != l0:
                continue
            d = float(np.hypot((x0 + 2.0) - x2, y0 - y2))
            if d <= best_d:
                best_d = d
                best_n2 = n2
        if best_n2 is not None:
            mapping[n0] = best_n2
    return mapping


def _roc_auc(pos_dists, neg_dists):
    if not pos_dists or not neg_dists:
        return 0.5
    wins = 0.0
    total = len(pos_dists) * len(neg_dists)
    for p in pos_dists:
        for n in neg_dists:
            if p < n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def main() -> None:
    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    try:
        # 2 object classes (square vs ring) x 2 instances for same-vs-diff AUC & +2px preservation
        scenes = []
        for cls_name in ("square", "ring"):
            for idx in range(2):
                img = np.full((64, 64, 3), 30, dtype=np.uint8)
                off = idx * 3
                if cls_name == "square":
                    img[14 + off:50 - off, 14 + off:50 - off] = (220, 190, 40)
                    img[20 + off:44 - off, 20 + off:44 - off] = (20, 90, 200)
                else:
                    yy, xx = np.mgrid[0:64, 0:64]
                    r = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
                    img[(r >= 12 + off) & (r <= 20 + off)] = (220, 190, 40)
                scenes.append((cls_name, img))

        fam_pres = {f: [0, 0] for f in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION")}
        step0b = Counter()
        hists = {f: [] for f in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION")}

        for cls_name, img in scenes:
            img2 = np.roll(img, 2, axis=1)
            tr0, tr2 = {}, {}
            s0 = FE.build_graph(img, name=f"gate_{cls_name}_0", record_co_located=False, trace=tr0)
            s2 = FE.build_graph(img2, name=f"gate_{cls_name}_2", record_co_located=False, trace=tr2)
            nmap = _match_nodes(s0, tr0, s2, tr2)

            edges2_by_pair = {}
            for u, v, et in s2.subgraph.edges:
                etn = getattr(et, "name", str(et))
                a, b = (u, v) if u < v else (v, u)
                edges2_by_pair.setdefault((a, b), set()).add(etn)

            for u, v, et in s0.subgraph.edges:
                etn = getattr(et, "name", str(et))
                lu = str(s0.sigs.get(u, ("?",))[0])
                lv = str(s0.sigs.get(v, ("?",))[0])
                if lu not in CONTENT_LENSES or lv not in CONTENT_LENSES:
                    continue
                if u in nmap and v in nmap:
                    u2, v2 = nmap[u], nmap[v]
                    a2, b2 = (u2, v2) if u2 < v2 else (v2, u2)
                    ets2 = edges2_by_pair.get((a2, b2), set())
                    if etn in ets2:
                        step0b["preserved"] += 1
                    elif ets2:
                        step0b["emitted_different_type"] += 1
                    else:
                        step0b["not_emitted"] += 1
                    if etn in fam_pres:
                        fam_pres[etn][1] += 1
                        if etn in ets2:
                            fam_pres[etn][0] += 1

            for f in hists:
                c = sum(1 for _, _, et in s0.subgraph.edges if getattr(et, "name", str(et)) == f)
                hists[f].append((cls_name, float(c)))

        fam_results = {}
        for f in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION"):
            num, den = fam_pres[f]
            pres = (num / den) if den else 0.0
            pos_d, neg_d = [], []
            pts = hists[f]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = abs(pts[i][1] - pts[j][1])
                    if pts[i][0] == pts[j][0]:
                        pos_d.append(d)
                    else:
                        neg_d.append(d)
            auc = _roc_auc(pos_d, neg_d)
            kept = bool(pres >= 0.70 and auc >= 0.75)
            fam_results[f] = {
                "preserved": num,
                "total_surviving_endpoints": den,
                "preservation_rate": pres,
                "same_vs_diff_auc": auc,
                "gate_pass": kept,
                "status": "KEPT" if kept else "DROPPED",
            }

        payload = {
            "protocol": "L3+L4 family gates + STEP 0b loss classification",
            "corpus": "local 64x64 multi-class smoke (ETH-80-master symlink target absent in sandbox)",
            "family_gates": fam_results,
            "step0b_loss_classification": dict(step0b),
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


if __name__ == "__main__":
    main()
