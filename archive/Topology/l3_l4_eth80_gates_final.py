"""STEP 1+2 preservation/survival gates on real ETH-80 (post coverage-fix).

Measures, for the two L3/L4 families, on a representative ETH-80 scene set:
  * preservation under a +2px horizontal shift (the edge survives when both
    endpoints re-match to the same edge type in the shifted graph),
  * the shifted-null preservation baseline (a random permutation of the
    match mapping -- the null the +2 number must beat),
  * selectivity (the shifted-graph branching factor of the family),
  * survival n (the denominator -- must be >= 50 for the +2 gate).

Every number is labelled with the corpus and the ruler artifact hash.
Run: .venv/bin/python Topology/l3_l4_eth80_gates_final.py [n_per_class]
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

OUT = ROOT / "results" / "l3_l4_eth80_gates_final.json"
RULER_SHA = "c0aa914ba511dc20350d4148dcf58f014d901c4da231a1c7819818322a06dfae"
CORPUS = "ETH-80 (PRIMARY; 8 classes x N instances, -090-000 view)"

CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}


def _match_nodes(s0, tr0, s_sh, tr_sh, shift_px: int, tol: float = 3.0):
    """Map each content node of the base graph to its shifted counterpart.

    Uses the SAME position source the mechanisms now use
    (`_stage6_positions` keyed by anchor), so the match is consistent with
    the positioned content tokens L3/L4 actually see.
    """
    from perception import attachment as _att

    def _pos_map(s, tr, binder):
        sp = binder._stage6_positions()
        out = {}
        for n in s.subgraph.nodes:
            lens = str(s.sigs.get(n, ("?",))[0])
            if lens not in CONTENT_LENSES:
                continue
            # the detection carrying this node: find via alpha->nid
            # (cheap: rebuild alpha map once)
            out[n] = sp.get(_alpha_of(tr, n))
        return out

    # build alpha->nid per graph from the trace detections is not available
    # post-finalize; use the position map keyed by node id AND alpha via the
    # ConceptGraph registry fallback.  The mechanisms accept either, so match
    # on whichever the node has.
    def _pos_map2(s, tr):
        cg = tr["cg"]
        sp = tr.get("binder")._stage6_positions() if tr.get("binder") else {}
        apos = tr.get("anchor_positions") or {}
        anchors = cg.pattern_node_anchor_map()
        out = {}
        for n in s.subgraph.nodes:
            lens = str(s.sigs.get(n, ("?",))[0])
            if lens not in CONTENT_LENSES:
                continue
            p = None
            a = anchors.get(n)
            if a is not None:
                p = _att._xy(sp, a)
            if p is None:
                p = cg.pattern_node_position(n)
            if p is None and a is not None:
                p = _att._xy(apos, a)
            if p is not None and len(p) >= 2:
                out[n] = (float(p[0]), float(p[1]), lens)
        return out

    p0 = _pos_map2(s0, tr0)
    psh = _pos_map2(s_sh, tr_sh)
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


def _alpha_of(tr, nid):
    return tr["cg"].pattern_node_anchor_map().get(nid)


def main() -> None:
    n_per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    eth_root = ROOT / "data" / "ETH-80-master"
    scenes = []
    for cls_id in [str(i) for i in range(1, 9)]:
        cls_dir = eth_root / cls_id
        insts = sorted(p for p in cls_dir.iterdir() if p.is_dir())[:n_per_class]
        for inst_dir in insts:
            pngs = sorted(p for p in inst_dir.glob("*.png")
                          if "maps" not in p.parts and "-090-000.png" in p.name)
            if pngs:
                scenes.append((cls_id, pngs[0]))

    print("scenes:", len(scenes), flush=True)

    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    try:
        pres = {
            "CONTOUR_CONTINUITY": {"+2px": [0, 0], "null": [0, 0]},
            "ORIENTATION_JUNCTION": {"+2px": [0, 0], "null": [0, 0]},
        }
        branching = {"CONTOUR_CONTINUITY": [], "ORIENTATION_JUNCTION": []}
        edge_counts = {"CONTOUR_CONTINUITY": 0, "ORIENTATION_JUNCTION": 0}
        rng = np.random.default_rng(0)

        for i, (cls_id, path) in enumerate(scenes):
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            tr0 = {}
            s0 = FE.build_graph(img, name=f"g_{path.stem}_0",
                                record_co_located=False, trace=tr0)
            for etn in edge_counts:
                for u, v, et in s0.subgraph.edges:
                    if getattr(et, "name", str(et)) == etn:
                        edge_counts[etn] += 1

            img_sh = np.roll(img, 2, axis=1)
            tr_sh = {}
            s_sh = FE.build_graph(img_sh, name=f"g_{path.stem}_2",
                                  record_co_located=False, trace=tr_sh)

            nmap = _match_nodes(s0, tr0, s_sh, tr_sh, 2)

            edges_sh = {}
            deg_sh = defaultdict(lambda: defaultdict(int))
            for u, v, et in s_sh.subgraph.edges:
                etn = getattr(et, "name", str(et))
                a, b = (u, v) if u < v else (v, u)
                edges_sh.setdefault((a, b), set()).add(etn)
                deg_sh[etn][u] += 1
                deg_sh[etn][v] += 1

            null_vals = list(nmap.values())
            if len(null_vals) > 1:
                rng.shuffle(null_vals)
                nmap_null = dict(zip(nmap.keys(), null_vals))
            else:
                nmap_null = {}

            for u, v, et in s0.subgraph.edges:
                etn = getattr(et, "name", str(et))
                if etn not in pres:
                    continue
                if u in nmap and v in nmap:
                    u2, v2 = nmap[u], nmap[v]
                    a2, b2 = (u2, v2) if u2 < v2 else (v2, u2)
                    pres[etn]["+2px"][1] += 1
                    if etn in edges_sh.get((a2, b2), set()):
                        pres[etn]["+2px"][0] += 1
                    branching[etn].append(deg_sh[etn][u2])
                if u in nmap_null and v in nmap_null:
                    un, vn = nmap_null[u], nmap_null[v]
                    an, bn = (un, vn) if un < vn else (vn, un)
                    pres[etn]["null"][1] += 1
                    if etn in edges_sh.get((an, bn), set()):
                        pres[etn]["null"][0] += 1
            if (i + 1) % 4 == 0:
                print(f"  {i+1}/{len(scenes)} scenes done", flush=True)

        summary = {}
        for fam in ("CONTOUR_CONTINUITY", "ORIENTATION_JUNCTION"):
            r2 = pres[fam]["+2px"][0] / max(1, pres[fam]["+2px"][1])
            rn = pres[fam]["null"][0] / max(1, pres[fam]["null"][1])
            br = np.asarray(branching[fam], dtype=np.float64) if branching[fam] else np.asarray([0.0])
            summary[fam] = {
                "preservation_2px": r2,
                "n_2px": pres[fam]["+2px"][1],
                "shifted_null_preservation": rn,
                "n_null": pres[fam]["null"][1],
                "beats_null": bool(r2 > rn),
                "selectivity_branching": {
                    "mean": float(np.mean(br)),
                    "median": float(np.median(br)),
                    "p95": float(np.percentile(br, 95)),
                },
                "base_edge_count": edge_counts[fam],
                "status": "KEPT" if (r2 > rn and pres[fam]["+2px"][1] >= 50) else "REVIEW",
            }

        payload = {
            "protocol": "L3+L4 STEP1/2 preservation+survival gates (post coverage-fix)",
            "corpus": CORPUS,
            "ruler_artifact_sha256": RULER_SHA,
            "n_scenes": len(scenes),
            "n_per_class": n_per_class,
            "kept_families": summary,
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


if __name__ == "__main__":
    main()
