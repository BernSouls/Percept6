"""FINAL CHECK (A + E): vector structural address quality and class signal.

A. Address quality (held-out +2px/+3px): top-1/top-5/MRR of the true
   counterpart among same-lens nodes by cosine; no-counterpart count;
   floor = node-type-only.
E. Class signal: scene code = sum of node address vectors; classify
   24 refs + 48 dev -> 48 held-out; top-1 + macro AUC per k vs old histogram.

Each graph is built ONCE and addresses computed for all k at once.
Writes results/vector_address_final_check.json.
"""
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
from perception.front_end_mechanisms import (  # noqa: E402
    compute_vector_addresses, compute_wl_addresses, KEPT_L6_FAMILIES, _cosine)

OUT = ROOT / "results" / "vector_address_final_check.json"
CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}


def _load_img(path):
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # truncated / corrupt image
        print(f"[skip] cannot load {path}: {exc}", flush=True)
        return None


def _build_addrs(img, name, max_k=3):
    tr = {}
    s = FE.build_graph(img, name=name, record_co_located=False, trace=tr)
    # _stage6_detections() is cleared by finalize_step(); read sigs from the
    # graph itself (s.sigs: node -> (lens_type, TypedSignature)).
    sigs = {int(n): (str(sigs_t[0]),) for n, sigs_t in s.sigs.items()}
    nodes = set(sigs)
    edges, weights = [], {}
    for k, v in tr["binder"].cg.edges.items():
        if isinstance(k, tuple) and len(k) >= 2:
            et = getattr(v, "edge_type", k[2] if len(k) > 2 else "?")
            edges.append((k[0], k[1], et))
            weights[(k[0], k[1], et)] = float(getattr(v, "pending_signal", 1.0))
    vec = compute_vector_addresses(nodes, edges, sigs, KEPT_L6_FAMILIES,
                                   max_k=max_k, weights=weights)
    wl = compute_wl_addresses(nodes, edges, sigs, KEPT_L6_FAMILIES, max_k=max_k)
    return s, tr, sigs, vec, wl


def _match_nodes(s0, tr0, s_sh, tr_sh, shift_px, tol=3.0):
    p0 = {n: tr0["cg"].pattern_node_position(n)
          for n in s0.subgraph.nodes
          if str(s0.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
          and tr0["cg"].pattern_node_position(n) is not None}
    psh = {n: tr_sh["cg"].pattern_node_position(n)
           for n in s_sh.subgraph.nodes
           if str(s_sh.sigs.get(n, ("?",))[0]) in CONTENT_LENSES
           and tr_sh["cg"].pattern_node_position(n) is not None}
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


def _rank_metrics(scores, true_idx):
    order = np.argsort(-np.asarray(scores))
    pos = int(np.where(order == true_idx)[0][0]) if true_idx in order else len(scores)
    return (1.0 if pos == 0 else 0.0, 1.0 if pos < 5 else 0.0, 1.0 / (pos + 1))


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
                wins += 1.0 if p > n else (0.5 if p == n else 0.0)
        aucs.append(wins / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else 0.5


def main() -> None:
    split = C.split_tuning()
    train_scenes = split["references"] + split["dev"]
    heldout = split["heldout"]

    old = (P.FE_L3_SPINE, P.FE_L4_JUNCTIONS, P.FE_L6_ADDRESS, P.FE_VECTOR_ADDRESS)
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    P.FE_L6_ADDRESS = True
    P.FE_VECTOR_ADDRESS = True
    try:
        report = {"protocol": "FINAL CHECK A+E (vector address quality + class signal)",
                  "corpus": "ETH-80 (24 refs + 48 dev -> 48 heldout)"}

        # ---------- A. Address quality ----------
        quality = {}
        for shift_px in (2, 3):
            t1 = t5 = mrr = 0.0
            n_pairs = n_no_c = 0
            for sc in heldout[:24]:
                sid = sc.get("scene_id", Path(sc["path"]).stem)
                img = _load_img(sc["path"])
                if img is None:
                    continue
                s0, tr0, sigs0, vec0, _ = _build_addrs(img, f"va_{sid}_0")
                img_sh = np.roll(img, shift_px, axis=1)
                s_sh, tr_sh, sigs_sh, vec_sh, _ = _build_addrs(
                    img_sh, f"va_{sid}_{shift_px}")
                nmap = _match_nodes(s0, tr0, s_sh, tr_sh, shift_px)
                lens_nodes = [n for n in vec_sh[1]
                              if str(sigs_sh.get(n, ("?",))[0]) in CONTENT_LENSES]
                for k in (1, 2, 3):
                    for n0, nsh in nmap.items():
                        if nsh not in vec_sh[k]:
                            continue
                        scores = [_cosine(vec_sh[k][nsh], vec_sh[k][c])
                                  for c in lens_nodes if c in vec_sh[k]]
                        true_idx = lens_nodes.index(nsh) if nsh in lens_nodes else -1
                        if true_idx < 0:
                            n_no_c += 1
                            continue
                        a, b, r = _rank_metrics(scores, true_idx)
                        t1 += a; t5 += b; mrr += r; n_pairs += 1
            quality[f"+{shift_px}px"] = {
                "top1": t1 / max(1, n_pairs), "top5": t5 / max(1, n_pairs),
                "mrr": mrr / max(1, n_pairs), "n_pairs": n_pairs,
                "n_no_counterpart": n_no_c}
        report["address_quality"] = quality

        # floor: node-type-only (L0 type vectors) top-1 on the same pairs
        floor_t1 = n_floor = 0
        for sc in heldout[:8]:
            sid = sc.get("scene_id", Path(sc["path"]).stem)
            img = _load_img(sc["path"])
            if img is None:
                continue
            s0, tr0, sigs0, vec0, _ = _build_addrs(img, f"fl_{sid}")
            img_sh = np.roll(img, 2, axis=1)
            s_sh, tr_sh, sigs_sh, vec_sh, _ = _build_addrs(img_sh, f"fl_{sid}_2")
            nmap = _match_nodes(s0, tr0, s_sh, tr_sh, 2)
            lens_nodes = [n for n in vec_sh[1]
                          if str(sigs_sh.get(n, ("?",))[0]) in CONTENT_LENSES]
            for n0, nsh in nmap.items():
                if nsh not in vec_sh[1] or nsh not in lens_nodes:
                    continue
                # type-only: rank nsh among same-lens nodes by L0 type vector
                scores = [_cosine(vec_sh[1][nsh], vec_sh[1][c])
                          for c in lens_nodes if c in vec_sh[1]]
                a, _, _ = _rank_metrics(scores, lens_nodes.index(nsh))
                floor_t1 += a; n_floor += 1
        report["floor_node_type_only_top1"] = floor_t1 / max(1, n_floor)
        report["floor_n"] = n_floor

        # ---------- E. Class signal (build once per scene, reuse all k) ----------
        dim = int(getattr(P, "VECTOR_ADDRESS_DIM", 1024))

        def _scene_code(vec, k):
            code = np.zeros(dim)
            for n in vec[k]:
                code += vec[k][n]
            nrm = np.linalg.norm(code)
            return code / nrm if nrm > 0 else code

        train_vec, train_wl, train_labels = [], [], []
        for sc in train_scenes:
            img = _load_img(sc["path"])
            if img is None:
                continue
            _, _, _, vec, wl = _build_addrs(img, f"e_tr_{sc['object']}")
            train_vec.append(vec); train_wl.append(wl)
            train_labels.append(sc.get("class", sc.get("cls")))
        test_vec, test_wl, test_labels = [], [], []
        for sc in heldout:
            img = _load_img(sc["path"])
            if img is None:
                continue
            _, _, _, vec, wl = _build_addrs(img, f"e_te_{sc['object']}")
            test_vec.append(vec); test_wl.append(wl)
            test_labels.append(sc.get("class", sc.get("cls")))

        cls, cls_old = {}, {}
        for k in (1, 2, 3):
            tcodes = [_scene_code(v, k) for v in train_vec]
            ecodes = [_scene_code(v, k) for v in test_vec]
            classes = sorted(set(train_labels))
            centroids = {c: np.mean([tcodes[i] for i, y in enumerate(train_labels)
                                     if y == c], axis=0) for c in classes}
            preds, scores = [], []
            for code in ecodes:
                sd = {c: float(np.dot(code, centroids[c])) for c in classes}
                scores.append(sd); preds.append(max(classes, key=lambda c: sd[c]))
            cls[f"k{k}"] = {"top1": float(np.mean([p == y for p, y in zip(preds, test_labels)])),
                            "macro_auc": _macro_auc(scores, test_labels, classes),
                            "n_train": len(train_labels), "n_heldout": len(test_labels)}
            # old histogram
            th = [Counter(v[k].values()) for v in train_wl]
            eh = [Counter(v[k].values()) for v in test_wl]
            vocab = sorted({a for h in th for a in h})
            idx = {a: i for i, a in enumerate(vocab)}
            cent = {}
            for c in classes:
                vecs = []
                for h, y in zip(th, train_labels):
                    if y == c:
                        vv = np.zeros(len(vocab))
                        for a, cnt in h.items():
                            if a in idx: vv[idx[a]] = cnt
                        nrm = np.linalg.norm(vv); vv = vv/nrm if nrm>0 else vv
                        vecs.append(vv)
                cent[c] = np.mean(vecs, axis=0) if vecs else np.zeros(len(vocab))
            preds2, scores2 = [], []
            for h in eh:
                vv = np.zeros(len(vocab))
                for a, cnt in h.items():
                    if a in idx: vv[idx[a]] = cnt
                nrm = np.linalg.norm(vv); vv = vv/nrm if nrm>0 else vv
                sd = {c: float(np.dot(vv, cent[c])) for c in classes}
                scores2.append(sd); preds2.append(max(classes, key=lambda c: sd[c]))
            cls_old[f"k{k}"] = {"top1": float(np.mean([p == y for p, y in zip(preds2, test_labels)])),
                                "macro_auc": _macro_auc(scores2, test_labels, classes)}
        report["class_signal_vector"] = cls
        report["class_signal_old_histogram"] = cls_old

        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS, P.FE_L6_ADDRESS, P.FE_VECTOR_ADDRESS = old


if __name__ == "__main__":
    main()
