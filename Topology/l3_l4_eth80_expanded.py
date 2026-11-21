"""Expanded ETH-80 PRIMARY evaluation for L3+L4 (40 scenes, 8 classes).

Includes:
- Endpoint eligibility declaration for every family
- +2px and +3px preservation with n for CONTOUR_CONTINUITY & ORIENTATION_JUNCTION
- Shifted-null preservation baseline
- Selectivity (branching factor: mean, median, p95)
- Separate sheet-including truth map measurement for CURVE_CONTAINMENT
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402

OUT = ROOT / "results" / "l3_l4_eth80_expanded.json"
CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}


def _match_nodes(s0, tr0, s_sh, tr_sh, shift_px: int, include_sheet: bool = False, tol: float = 3.0):
    def _pos_map(s, tr):
        out = {}
        anchors = tr["cg"].pattern_node_anchor_map()
        apos = tr.get("anchor_positions") or {}
        for n in s.subgraph.nodes:
            lens = str(s.sigs.get(n, ("?",))[0])
            is_c = lens in CONTENT_LENSES
            if is_c:
                p = tr["cg"].pattern_node_position(n)
                if p is not None and len(p) >= 2:
                    out[n] = (float(p[0]), float(p[1]), lens)
            elif include_sheet:
                a = anchors.get(n)
                p = apos.get(a) if a is not None else None
                if p is not None and len(p) >= 2:
                    out[n] = (float(p[0]), float(p[1]), lens)
        return out

    p0 = _pos_map(s0, tr0)
    psh = _pos_map(s_sh, tr_sh)
    mapping = {}
    for n0, (x0, y0, l0) in p0.items():
        best_n, best_d = None, tol
        for nsh, (xsh, ysh, lsh) in psh.items():
            if lsh != l0:
                continue
            d = float(np.hypot((x0 + shift_px) - xsh, y0 - ysh))
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is not None:
            mapping[n0] = best_n
    return mapping


def main() -> None:
    eth_root = ROOT / "data" / "ETH-80-master"
    scenes = []
    for cls_id in [str(i) for i in range(1, 9)]:
        cls_dir = eth_root / cls_id
        for inst_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir())[:5]:
            pngs = sorted(p for p in inst_dir.glob("*.png") if "maps" not in p.parts)
            if pngs:
                scenes.append((cls_id, pngs[0]))

    eligibility = {
        "CONTOUR_CONTINUITY": {
            "endpoints": "CONTENT_TOKENS <-> CONTENT_TOKENS",
            "truth_map": "refined pattern_node_position_map",
            "denominator": "surviving content-content edges of type CONTOUR_CONTINUITY",
        },
        "ORIENTATION_JUNCTION": {
            "endpoints": "CONTENT_TOKENS <-> CONTENT_TOKENS",
            "truth_map": "refined pattern_node_position_map",
            "denominator": "surviving content-content edges of type ORIENTATION_JUNCTION",
        },
        "COCIRCULAR": {
            "endpoints": "SHEET_DERIVED_FRAME <-> SHEET_DERIVED_FRAME",
            "truth_map": "content-only truth map (ineligible: 0/0)",
            "status": "ABSENT under content-only truth map",
        },
        "PARALLEL": {
            "endpoints": "SHEET_DERIVED_FRAME <-> SHEET_DERIVED_FRAME",
            "truth_map": "content-only truth map (ineligible: 0/0)",
            "status": "ABSENT under content-only truth map",
        },
        "JUNCTION_CONTOUR": {
            "endpoints": "SHEET_DERIVED_FRAME <-> CONTENT_TOKENS",
            "truth_map": "content-only truth map (ineligible: 0/0)",
            "status": "ABSENT under content-only truth map",
        },
        "CURVE_CONTAINMENT": {
            "endpoints": "SHEET_DERIVED_FRAME <-> CONTENT_TOKENS",
            "truth_map": "mixed truth map (content refined + sheet anchor_positions)",
            "denominator": "surviving mixed/sheet-content edges of type CURVE_CONTAINMENT",
        },
    }

    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    try:
        pres = {
            "CONTOUR_CONTINUITY": {"+2px": [0, 0], "+3px": [0, 0], "null": [0, 0]},
            "ORIENTATION_JUNCTION": {"+2px": [0, 0], "+3px": [0, 0], "null": [0, 0]},
        }
        curve_cont = {"+2px": [0, 0], "+3px": [0, 0]}
        branching = {"CONTOUR_CONTINUITY": [], "ORIENTATION_JUNCTION": []}
        rng = np.random.default_rng(0)

        for cls_id, path in scenes:
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            tr0 = {}
            s0 = FE.build_graph(img, name=f"exp_{path.stem}_0", record_co_located=False, trace=tr0)

            for shift_px, key in ((2, "+2px"), (3, "+3px")):
                img_sh = np.roll(img, shift_px, axis=1)
                tr_sh = {}
                s_sh = FE.build_graph(img_sh, name=f"exp_{path.stem}_{shift_px}", record_co_located=False, trace=tr_sh)

                nmap = _match_nodes(s0, tr0, s_sh, tr_sh, shift_px, include_sheet=False)
                nmap_mix = _match_nodes(s0, tr0, s_sh, tr_sh, shift_px, include_sheet=True)

                edges_sh = {}
                deg_sh = defaultdict(lambda: defaultdict(int))
                for u, v, et in s_sh.subgraph.edges:
                    etn = getattr(et, "name", str(et))
                    a, b = (u, v) if u < v else (v, u)
                    edges_sh.setdefault((a, b), set()).add(etn)
                    deg_sh[etn][u] += 1
                    deg_sh[etn][v] += 1

                # Shifted null permutation of nmap values
                null_vals = list(nmap.values())
                if len(null_vals) > 1:
                    rng.shuffle(null_vals)
                    nmap_null = dict(zip(nmap.keys(), null_vals))
                else:
                    nmap_null = {}

                for u, v, et in s0.subgraph.edges:
                    etn = getattr(et, "name", str(et))
                    if etn in pres:
                        if u in nmap and v in nmap:
                            u2, v2 = nmap[u], nmap[v]
                            a2, b2 = (u2, v2) if u2 < v2 else (v2, u2)
                            pres[etn][key][1] += 1
                            if etn in edges_sh.get((a2, b2), set()):
                                pres[etn][key][0] += 1
                            if key == "+2px":
                                branching[etn].append(deg_sh[etn][u2])
                        if key == "+2px" and u in nmap_null and v in nmap_null:
                            un, vn = nmap_null[u], nmap_null[v]
                            an, bn = (un, vn) if un < vn else (vn, un)
                            pres[etn]["null"][1] += 1
                            if etn in edges_sh.get((an, bn), set()):
                                pres[etn]["null"][0] += 1
                    elif etn == "CURVE_CONTAINMENT":
                        if u in nmap_mix and v in nmap_mix:
                            u2, v2 = nmap_mix[u], nmap_mix[v]
                            a2, b2 = (u2, v2) if u2 < v2 else (v2, u2)
                            curve_cont[key][1] += 1
                            if etn in edges_sh.get((a2, b2), set()):
                                curve_cont[key][0] += 1

        summary = {}
        for fam in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION"):
            r2 = pres[fam]["+2px"][0] / max(1, pres[fam]["+2px"][1])
            r3 = pres[fam]["+3px"][0] / max(1, pres[fam]["+3px"][1])
            rn = pres[fam]["null"][0] / max(1, pres[fam]["null"][1])
            br = np.asarray(branching[fam], dtype=np.float64) if branching[fam] else np.asarray([0.0])
            summary[fam] = {
                "preservation_2px": r2,
                "n_2px": pres[fam]["+2px"][1],
                "preservation_3px": r3,
                "n_3px": pres[fam]["+3px"][1],
                "shifted_null_preservation": rn,
                "n_null": pres[fam]["null"][1],
                "selectivity_branching": {
                    "mean": float(np.mean(br)),
                    "median": float(np.median(br)),
                    "p95": float(np.percentile(br, 95)),
                },
                "status": "KEPT",
            }

        payload = {
            "protocol": "Expanded ETH-80 PRIMARY L3+L4 evaluation (40 scenes, 8 classes)",
            "corpus": "ETH-80 (PRIMARY; 40 scenes = 8 classes x 5 instances)",
            "eligibility": eligibility,
            "kept_families": summary,
            "curve_containment_mixed_truth_map": {
                "preservation_2px": curve_cont["+2px"][0] / max(1, curve_cont["+2px"][1]),
                "n_2px": curve_cont["+2px"][1],
                "preservation_3px": curve_cont["+3px"][0] / max(1, curve_cont["+3px"][1]),
                "n_3px": curve_cont["+3px"][1],
            },
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


if __name__ == "__main__":
    main()
