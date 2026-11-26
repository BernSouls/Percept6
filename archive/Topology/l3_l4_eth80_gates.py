"""ETH-80 PRIMARY family gates, STEP 0b loss classification, and degree-gap investigation."""
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

OUT = ROOT / "results" / "l3_l4_eth80_gates.json"
CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}
HIGHER_FAMILIES = (
    "CONTOUR_CONTINUITY",
    "ORIENTATION_JUNCTION",
    "COCIRCULAR",
    "PARALLEL",
    "JUNCTION_CONTOUR",
    "CURVE_CONTAINMENT",
)


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
    # 8 ETH-80 PRIMARY scenes across 2 classes (1=apple, 2=car)
    eth_root = ROOT / "data" / "ETH-80-master"
    scenes = []
    for cls_id in ("1", "2"):
        for inst_id in ("1", "2"):
            inst_dir = eth_root / cls_id / inst_id
            pngs = sorted(p for p in inst_dir.glob("*.png") if "maps" not in p.parts)[:2]
            for p in pngs:
                scenes.append((cls_id, p))

    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    try:
        fam_pres = {f: [0, 0] for f in HIGHER_FAMILIES}
        fam_counts = Counter()
        total_content_nodes = 0
        step0b = Counter()
        hists = {f: [] for f in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION")}

        for cls_id, path in scenes:
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            img2 = np.roll(img, 2, axis=1)
            tr0, tr2 = {}, {}
            s0 = FE.build_graph(img, name=f"eth80_{path.stem}_0", record_co_located=False, trace=tr0)
            s2 = FE.build_graph(img2, name=f"eth80_{path.stem}_2", record_co_located=False, trace=tr2)
            nmap = _match_nodes(s0, tr0, s2, tr2)

            c_nodes = [n for n in s0.subgraph.nodes if str(s0.sigs.get(n, ("?",))[0]) in CONTENT_LENSES]
            total_content_nodes += len(c_nodes)

            edges2_by_pair = {}
            for u, v, et in s2.subgraph.edges:
                etn = getattr(et, "name", str(et))
                a, b = (u, v) if u < v else (v, u)
                edges2_by_pair.setdefault((a, b), set()).add(etn)

            for u, v, et in s0.subgraph.edges:
                etn = getattr(et, "name", str(et))
                if etn in HIGHER_FAMILIES:
                    fam_counts[etn] += 1
                lu = str(s0.sigs.get(u, ("?",))[0])
                lv = str(s0.sigs.get(v, ("?",))[0])
                if lu not in CONTENT_LENSES or lv not in CONTENT_LENSES:
                    continue
                if u in nmap and v in nmap:
                    u2, v2 = nmap[u], nmap[v]
                    a2, b2 = (u2, v2) if u2 < v2 else (v2, u2)
                    ets2 = edges2_by_pair.get((a2, b2), set())
                    if etn in ets2:
                        step0b["(d) preserved"] += 1
                    elif ets2:
                        step0b["(b) emitted_different_type"] += 1
                    else:
                        step0b["(a) not_emitted"] += 1
                    if etn in fam_pres:
                        fam_pres[etn][1] += 1
                        if etn in ets2:
                            fam_pres[etn][0] += 1

            for f in hists:
                c = sum(1 for _, _, et in s0.subgraph.edges if getattr(et, "name", str(et)) == f)
                hists[f].append((cls_id, float(c)))

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

        degree_gap = {}
        total_deg = 0.0
        for f in HIGHER_FAMILIES:
            num, den = fam_pres[f]
            deg = (2.0 * fam_counts[f]) / max(1, total_content_nodes)
            total_deg += deg
            degree_gap[f] = {
                "edge_count": fam_counts[f],
                "mean_degree_contribution": deg,
                "preservation_2px": (num / den) if den else 0.0,
            }

        payload = {
            "protocol": "ETH-80 PRIMARY family gates + STEP 0b + degree-gap investigation",
            "corpus": "ETH-80 (PRIMARY; 8 scenes across classes 1 and 2)",
            "scenes_count": len(scenes),
            "total_content_nodes": total_content_nodes,
            "family_gates": fam_results,
            "step0b_loss_classification": dict(step0b),
            "degree_gap_investigation": degree_gap,
            "combined_higher_family_degree": total_deg,
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


if __name__ == "__main__":
    main()
