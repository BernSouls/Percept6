"""L6 (Structural Address) evaluation: placement agreement (+2px/+3px) and class classification."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import corpus as C  # noqa: E402
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402
from perception.front_end_mechanisms import compute_wl_addresses, KEPT_L6_FAMILIES  # noqa: E402

OUT = ROOT / "results" / "l6_eval.json"
CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}


def _match_nodes(s0, tr0, s_sh, tr_sh, shift_px: int, tol: float = 3.0):
    p0 = {
        n: tr0["cg"].pattern_node_position(n)
        for n in s0.subgraph.nodes
        if str(s0.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
        and tr0["cg"].pattern_node_position(n) is not None
    }
    psh = {
        n: tr_sh["cg"].pattern_node_position(n)
        for n in s_sh.subgraph.nodes
        if str(s_sh.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
        and tr_sh["cg"].pattern_node_position(n) is not None
    }
    mapping = {}
    for n0, (x0, y0) in p0.items():
        l0 = str(s0.sigs.get(n0, ("?",))[0])
        best_n, best_d = None, tol
        for nsh, (xsh, ysh) in psh.items():
            if str(s_sh.sigs.get(nsh, ("?",))[0]) != l0:
                continue
            d = float(np.hypot((x0 + shift_px) - xsh, y0 - ysh))
            if d <= best_d:
                best_d = d
                best_n = nsh
        if best_n is not None:
            mapping[n0] = best_n
    return mapping


def _macro_auc(scores_by_scene, labels, classes):
    aucs = []
    for c in classes:
        pos = [s[c] for s, y in zip(scores_by_scene, labels) if y == c]
        neg = [s[c] for s, y in zip(scores_by_scene, labels) if y != c]
        if not pos or not neg:
            continue
        wins = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        aucs.append(wins / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else 0.5


def main() -> None:
    split = C.split_tuning()
    train_scenes = split["references"] + split["dev"]
    heldout_scenes = split["heldout"]

    old = (
        getattr(P, "FE_L3_SPINE", False),
        getattr(P, "FE_L4_JUNCTIONS", False),
        getattr(P, "FE_L6_ADDRESS", False),
    )
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    P.FE_L6_ADDRESS = True
    try:
        # 1) Placement agreement on first 24 heldout scenes (+2px and +3px)
        agree = {k: {"+2px": [0, 0], "+3px": [0, 0]} for k in (1, 2, 3)}
        for sc in heldout_scenes[:24]:
            sid = sc.get("scene_id", Path(sc["path"]).stem)
            img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
            tr0 = {}
            s0 = FE.build_graph(img, name=f"l6_pa_{sid}_0", record_co_located=False, trace=tr0)
            a0 = compute_wl_addresses(s0.subgraph.nodes, s0.subgraph.edges, s0.sigs, KEPT_L6_FAMILIES, 3)
            for shift_px, key in ((2, "+2px"), (3, "+3px")):
                img_sh = np.roll(img, shift_px, axis=1)
                tr_sh = {}
                s_sh = FE.build_graph(img_sh, name=f"l6_pa_{sid}_{shift_px}", record_co_located=False, trace=tr_sh)
                ash = compute_wl_addresses(s_sh.subgraph.nodes, s_sh.subgraph.edges, s_sh.sigs, KEPT_L6_FAMILIES, 3)
                nmap = _match_nodes(s0, tr0, s_sh, tr_sh, shift_px)
                for n0, nsh in nmap.items():
                    for k in (1, 2, 3):
                        agree[k][key][1] += 1
                        if a0[k].get(n0) == ash[k].get(nsh):
                            agree[k][key][0] += 1

        # 2) Address-histogram class classification (72 train -> 48 heldout)
        train_hists = {k: [] for k in (1, 2, 3)}
        train_labels = []
        for sc in train_scenes:
            sid = sc.get("scene_id", Path(sc["path"]).stem)
            img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
            tr = {}
            s = FE.build_graph(img, name=f"l6_tr_{sid}", record_co_located=False, trace=tr)
            addrs = compute_wl_addresses(s.subgraph.nodes, s.subgraph.edges, s.sigs, KEPT_L6_FAMILIES, 3)
            train_labels.append(sc.get("class", sc.get("cls")))
            for k in (1, 2, 3):
                train_hists[k].append(Counter(addrs[k].values()))

        test_hists = {k: [] for k in (1, 2, 3)}
        test_labels = []
        for sc in heldout_scenes:
            sid = sc.get("scene_id", Path(sc["path"]).stem)
            img = np.asarray(Image.open(sc["path"]).convert("RGB"), dtype=np.uint8)
            tr = {}
            s = FE.build_graph(img, name=f"l6_te_{sid}", record_co_located=False, trace=tr)
            addrs = compute_wl_addresses(s.subgraph.nodes, s.subgraph.edges, s.sigs, KEPT_L6_FAMILIES, 3)
            test_labels.append(sc.get("class", sc.get("cls")))
            for k in (1, 2, 3):
                test_hists[k].append(Counter(addrs[k].values()))

        classes = sorted(set(train_labels))
        cls_metrics = {}
        for k in (1, 2, 3):
            vocab = sorted({addr for h in train_hists[k] for addr in h})
            idx = {a: i for i, a in enumerate(vocab)}
            centroids = {}
            for c in classes:
                vecs = []
                for h, y in zip(train_hists[k], train_labels):
                    if y == c:
                        v = np.zeros(len(vocab), dtype=np.float64)
                        for a, cnt in h.items():
                            if a in idx:
                                v[idx[a]] = cnt
                        norm = np.linalg.norm(v)
                        if norm > 0:
                            v /= norm
                        vecs.append(v)
                centroids[c] = np.mean(vecs, axis=0)

            preds = []
            scores = []
            for h in test_hists[k]:
                v = np.zeros(len(vocab), dtype=np.float64)
                for a, cnt in h.items():
                    if a in idx:
                        v[idx[a]] = cnt
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
                sc_dict = {c: float(np.dot(v, centroids[c])) for c in classes}
                scores.append(sc_dict)
                preds.append(max(classes, key=lambda c: sc_dict[c]))

            top1 = float(np.mean([p == y for p, y in zip(preds, test_labels)]))
            auc = _macro_auc(scores, test_labels, classes)
            cls_metrics[f"k{k}"] = {
                "top1": top1,
                "macro_auc": auc,
                "n_train": len(train_labels),
                "n_heldout": len(test_labels),
            }

        placement = {
            f"k{k}": {
                "agreement_2px": agree[k]["+2px"][0] / max(1, agree[k]["+2px"][1]),
                "n_2px": agree[k]["+2px"][1],
                "agreement_3px": agree[k]["+3px"][0] / max(1, agree[k]["+3px"][1]),
                "n_3px": agree[k]["+3px"][1],
            }
            for k in (1, 2, 3)
        }
        payload = {
            "protocol": "L6 structural address evaluation (ETH-80 PRIMARY)",
            "corpus": "ETH-80 (PRIMARY; 24 refs + 48 dev -> 48 heldout)",
            "placement_agreement": placement,
            "classification": cls_metrics,
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS, P.FE_L6_ADDRESS = old


if __name__ == "__main__":
    main()
