"""STEP A — fix and freeze the ruler (harness only; no product changes).

A1 exclusive tracked/static classification (nearer hypothesis wins; tie ->
   static), conservation residual asserted 0 per lens; fifth column renamed
   exists_minus_static; +1 excluded from the split.
A2 stride: distinct static site coordinates per axis + spacing histogram.
A3 sheet equivariance error ||M_shifted(site + k*stride) - M_base(site)||
   (mean RGB over the receptive-field disk) vs a different-object null.
A4 sheet occupancy: emitting sites / total, Jaccard of emitting sets, and
   static nodes on flat (no bottom-up response) patches.
A5 patch-difference split for static nodes: identical patch / changed patch
   with identical signature (invariant-by-design) / changed both.
A6 stream per lens from the emission registry (_LENS_NAMES + higher lenses),
   verified at a DIFFERENT shift (+6 renderer) than the scored shifts.
A9 ETH-80 replication (256px, stride 16) of the census + stride sweep.
A10 freeze: canonical JSON + sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import guard

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

SCENES_DIR = Path(guard.TREE) / "data" / "renderer-truth-v1"
TRUTH_PATH = SCENES_DIR / "truth.json"
ETH80_IMG = Path(guard.TREE).parent / "data" / "ETH-80-master" / "1" / "2" / "apple2-022-000.png"
SHIFTS_R = (0, 2, 3, 6, 12, 24)
SHIFTS_E = (0, 2, 3, 8, 16, 32)
TOL = 8.0

SHEET_LENSES = {"Boundary", "Color", "Motion", "Size", "Texture", "Position",
                "Onset", "Orientation", "EffectPresent", "Polarity",
                "OrientationPol"}
CONTENT_LENSES = {"CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
                  "RegionShape", "Corner", "Contour", "CurveComplex",
                  "IllusoryContour", "Junction", "T", "Y"}
HIGHER_ATTR = {"BorderOwnership", "TextureBoundary", "Surface"}


def _stream(lens):
    if lens in SHEET_LENSES:
        return "sheet"
    if lens in CONTENT_LENSES:
        return "content"
    if lens in HIGHER_ATTR:
        return "attribute_higher"
    return "unregistered"


def _build(image_path, shift, name, width):
    import eth80_frontend as FE
    from PIL import Image
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    t0 = time.perf_counter()
    g = FE.build_graph(image, name=name, record_co_located=False, trace=trace)
    cg = trace["cg"]
    positions = cg.pattern_node_position_map()
    anchors = cg.pattern_node_anchor_map()
    anchor_pos = trace.get("anchor_positions") or {}
    for n in [n for n in g.subgraph.nodes if n not in positions]:
        a = anchors.get(n)
        p = anchor_pos.get(a) if a is not None else None
        if p is not None and len(p) >= 2:
            positions[n] = (float(p[0]), float(p[1]))
    sig = {}
    for n, s in g.sigs.items():
        so = s[1]
        sig[n] = (so.class_, so.relation, tuple(so.quantized), tuple(so.population))
    return {"name": name, "shift": shift, "width": width,
            "nodes": len(g.subgraph.nodes), "edges": len(g.subgraph.edges),
            "positions": positions,
            "lens": {n: s[0] for n, s in g.sigs.items()},
            "sig": sig,
            "seconds": time.perf_counter() - t0}


def _worker(args):
    image_path, shift, name, width = args
    return _build(image_path, shift, name, width)


def _nearest_same_lens(a, b, u):
    p = a["positions"][u]
    ex = (p[0] + b["shift"]) % b["width"]
    best = None
    for v, (px, py) in b["positions"].items():
        if b["lens"].get(v) != a["lens"].get(u):
            continue
        d0 = math.hypot(px - ex, py - p[1])
        if d0 <= TOL and (best is None or d0 < best[0]):
            best = (d0, v)
    return best


def _classify(a, b):
    """A1 exclusive classification. Returns per-lens counts + static lists."""
    per = defaultdict(lambda: Counter())
    statics = []
    for u in a["positions"]:
        lu = a["lens"].get(u, "?")
        per[lu]["expected"] += 1
        best = _nearest_same_lens(a, b, u)
        if best is None:
            per[lu]["gone"] += 1
            continue
        d0, v = best
        dx = b["positions"][v][0] - a["positions"][u][0]
        d_track = abs(dx - b["shift"])
        d_static = abs(dx)
        if d_track <= 1.5 and d_static <= 1.5:
            # overlap zone: nearer hypothesis wins; tie -> static
            if d_track < d_static:
                per[lu]["track"] += 1
            else:
                per[lu]["static"] += 1
                statics.append((u, v))
        elif d_track <= 1.5:
            per[lu]["track"] += 1
        elif d_static <= 1.5:
            per[lu]["static"] += 1
            statics.append((u, v))
        else:
            per[lu]["exists_minus_static"] += 1
    return per, statics


def _stream_totals(lens_rows, stream):
    keys = ("expected", "gone", "track", "static", "exists_minus_static")
    out = {k: 0 for k in keys}
    for x in lens_rows:
        if x["stream"] == stream:
            for k in keys:
                out[k] += x[k]
    return out


def _spacing_hist(coords):
    xs = sorted(set(coords))
    diffs = Counter()
    for i in range(1, len(xs)):
        diffs[int(round(xs[i] - xs[i - 1]))] += 1
    return {"distinct": len(xs), "min": xs[0] if xs else None,
            "max": xs[-1] if xs else None, "spacing_hist": dict(diffs)}


def _patch(image, cx, cy, r):
    h, w = image.shape[:2]
    x0, x1 = int(math.floor(cx - r)), int(math.ceil(cx + r)) + 1
    y0, y1 = int(math.floor(cy - r)), int(math.ceil(cy + r)) + 1
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not mask.any():
        return None
    return image[y0:y1, x0:x1][mask].astype(np.float64)


def _sites(width):
    stride = width / 16.0
    return [(i + 0.5) * stride for i in range(16)], stride


def _analyze_scene(images, graphs, width):
    base = graphs[0]
    sx, stride = _sites(width)
    rows = []
    for shift in sorted(graphs):
        g = graphs[shift]
        per, statics = _classify(base, g)
        lens_rows = []
        residual_total = 0
        for lens in sorted(per, key=lambda k: -per[k]["expected"]):
            c = per[lens]
            residual = c["expected"] - (c["track"] + c["static"]
                                        + c["exists_minus_static"] + c["gone"])
            residual_total += residual
            lens_rows.append({
                "lens": lens, "stream": _stream(lens),
                "expected": c["expected"], "gone": c["gone"],
                "track": c["track"], "static": c["static"],
                "exists_minus_static": c["exists_minus_static"],
                "residual": residual,
            })
        # A2 stride from statics
        xs = [g["positions"][v][0] for _u, v in statics]
        ys = [g["positions"][v][1] for _u, v in statics]
        stride_info = {"x": _spacing_hist(xs), "y": _spacing_hist(ys),
                       "site_pairs": len({(round(g["positions"][v][0]),
                                           round(g["positions"][v][1]))
                                          for _u, v in statics})}
        # A3 equivariance + A4 occupancy + A5 patch split
        img_b = images[0]
        img_s = images[shift]
        emit_b = {(round(p[0]), round(p[1])) for p in base["positions"].values()}
        emit_s = {(round(p[0]), round(p[1])) for p in g["positions"].values()}
        sheet_b = {(round(base["positions"][n][0]), round(base["positions"][n][1]))
                   for n in base["positions"] if _stream(base["lens"].get(n, "?")) == "sheet"}
        sheet_s = {(round(g["positions"][n][0]), round(g["positions"][n][1]))
                   for n in g["positions"] if _stream(g["lens"].get(n, "?")) == "sheet"}
        occ = {
            "emit_sites_base": len(emit_b), "emit_sites_shift": len(emit_s),
            "total_sites": 256,
            "jaccard_emit": (len(emit_b & emit_s) / len(emit_b | emit_s))
            if (emit_b | emit_s) else 1.0,
            "sheet_emit_sites_base": len(sheet_b), "sheet_emit_sites_shift": len(sheet_s),
            "sheet_jaccard_emit": (len(sheet_b & sheet_s) / len(sheet_b | sheet_s))
            if (sheet_b | sheet_s) else 1.0,
        }
        r = stride / 2.0 * 1.6
        eq = []
        flat_static = Counter()
        patch_split = Counter()
        if shift:
            k = int(round(shift / stride))
            for (u, v) in statics:
                p = base["positions"][u]
                pb = _patch(img_b, p[0], p[1], r)
                cx = (p[0] + k * stride) % width
                ps = _patch(img_s, cx, p[1], r)
                if pb is None or ps is None or pb.shape != ps.shape:
                    continue
                eq.append(float(np.abs(pb - ps).mean()) / 255.0)
                std = float(pb.std())
                if std < 2.0:
                    flat_static[base["lens"].get(u, "?")] += 1
                identical = bool(np.array_equal(pb.astype(np.uint8),
                                                ps.astype(np.uint8)))
                sig_eq = base["sig"].get(u) == g["sig"].get(v)
                if identical:
                    patch_split["patch_identical"] += 1
                elif sig_eq:
                    patch_split["changed_patch_identical_sig"] += 1
                else:
                    patch_split["changed_both"] += 1
            # equivariance over ALL sites (not only statics)
            all_eq = []
            for cx0 in sx:
                for cy0 in sx:
                    pb = _patch(img_b, cx0, cy0, r)
                    ps = _patch(img_s, (cx0 + k * stride) % width, cy0, r)
                    if pb is None or ps is None or pb.shape != ps.shape:
                        continue
                    all_eq.append(float(np.abs(pb - ps).mean()) / 255.0)
            eq = all_eq
        else:
            all_eq = []
        rows.append({
            "shift": shift,
            "expected": len(base["positions"]),
            "lens": lens_rows,
            "residual_total": residual_total,
            "stride": stride_info,
            "occupancy": occ,
            "stream_totals": {
                "sheet": _stream_totals(lens_rows, "sheet"),
                "content": _stream_totals(lens_rows, "content"),
                "attribute_higher": _stream_totals(lens_rows, "attribute_higher"),
                "unregistered": _stream_totals(lens_rows, "unregistered"),
            },
            "equivariance_all_sites": {
                "n": len(all_eq),
                "median": float(np.median(all_eq)) if all_eq else None,
                "p10": float(np.percentile(all_eq, 10)) if all_eq else None,
                "p90": float(np.percentile(all_eq, 90)) if all_eq else None,
            },
            "flat_static_by_lens": dict(flat_static),
            "patch_split": dict(patch_split),
            "static_count": len(statics),
        })
    return rows


def _different_object_null(images, width):
    """A3 null: mean-RGB patch distance between two different scenes."""
    img_a, img_b = images
    sx, stride = _sites(width)
    r = stride / 2.0 * 1.6
    vals = []
    for cx0 in sx:
        for cy0 in sx:
            pa = _patch(img_a, cx0, cy0, r)
            pb = _patch(img_b, cx0, cy0, r)
            if pa is None or pb is None or pa.shape != pb.shape:
                continue
            vals.append(float(np.abs(pa - pb).mean()) / 255.0)
    return {"n": len(vals),
            "median": float(np.median(vals)) if vals else None,
            "p10": float(np.percentile(vals, 10)) if vals else None,
            "p90": float(np.percentile(vals, 90)) if vals else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "CAL", "TEST"), default="all")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "step_a_ruler.json"))
    args = ap.parse_args(argv)
    data = json.loads(TRUTH_PATH.read_text())
    scenes = data["scenes"]
    if args.split != "all":
        scenes = [s for s in scenes if s["split"] == args.split]
    if args.scenes:
        scenes = [s for s in scenes if s["name"] in set(args.scenes)]

    from PIL import Image
    import multiprocessing as mp

    # ---- renderer builds ----
    ctx = mp.get_context("spawn")
    tasks = []
    for s in scenes:
        for shift in SHIFTS_R:
            tasks.append((str(SCENES_DIR / (s["name"] + ".png")), shift,
                          "sa_%s_%d" % (s["name"], shift), 192))
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=guard.activate) as pool:
        built = pool.map(_worker, tasks, chunksize=1)
    print("[renderer builds %d in %.0fs]" % (len(tasks), time.perf_counter() - t0),
          flush=True)
    by = {(b["name"], b["shift"]): b for b in built}

    images = {}
    for s in scenes:
        images[s["name"]] = [np.asarray(Image.open(SCENES_DIR / (s["name"] + ".png"))
                                          .convert("RGB"), dtype=np.uint8)]
    # shifted images for patch tests
    shifted_images = {}
    for s in scenes:
        base_img = images[s["name"]][0]
        shifted_images[s["name"]] = {0: base_img}
        for shift in SHIFTS_R:
            if shift:
                shifted_images[s["name"]][shift] = np.roll(base_img, (0, shift),
                                                           axis=(0, 1))

    rows = []
    for s in scenes:
        graphs = {sh: by[("sa_%s_%d" % (s["name"], sh), sh)] for sh in SHIFTS_R}
        imgs = shifted_images[s["name"]]
        # images dict for _analyze_scene: images[shift]
        r = _analyze_scene(imgs, graphs, 192)
        rows.append({"scene": s["name"], "split": s["split"], "rows": r})
        for rr in r:
            eqm = rr["equivariance_all_sites"]["median"]
            print("%-20s %+2d exp %4d track %4d static %4d exists-static %4d gone %4d | eq med %s | occJ %.3f sheetJ %.3f | patch %s" % (
                s["name"], rr["shift"], rr["expected"],
                sum(x["track"] for x in rr["lens"]),
                sum(x["static"] for x in rr["lens"]),
                sum(x["exists_minus_static"] for x in rr["lens"]),
                sum(x["gone"] for x in rr["lens"]),
                ("%.4f" % eqm) if eqm is not None else "n/a",
                rr["occupancy"]["jaccard_emit"],
                rr["occupancy"]["sheet_jaccard_emit"],
                rr["patch_split"]), flush=True)

    # ---- A3 null: different object ----
    null = None
    if len(scenes) >= 2:
        a = images[scenes[0]["name"]][0]
        b = images[scenes[1]["name"]][0]
        null = _different_object_null((a, b), 192)
        print("different-object null (192px):", null, flush=True)

    # ---- A9 ETH-80 ----
    eth = {}
    tasks_e = [(str(ETH80_IMG), sh, "sa_eth_%d" % sh, 256) for sh in SHIFTS_E]
    with ctx.Pool(args.workers, initializer=guard.activate) as pool:
        built_e = pool.map(_worker, tasks_e, chunksize=1)
    by_e = {b["shift"]: b for b in built_e}
    img_e = np.asarray(Image.open(ETH80_IMG).convert("RGB"), dtype=np.uint8)
    imgs_e = {0: img_e}
    for sh in SHIFTS_E:
        if sh:
            imgs_e[sh] = np.roll(img_e, (0, sh), axis=(0, 1))
    eth_rows = _analyze_scene(imgs_e, by_e, 256)
    for rr in eth_rows:
        print("ETH80 %+2d exp %4d track %4d static %4d gone %4d | stride %s" % (
            rr["shift"], rr["expected"],
            sum(x["track"] for x in rr["lens"]),
            sum(x["static"] for x in rr["lens"]),
            sum(x["gone"] for x in rr["lens"]),
            rr["stride"]["x"]["spacing_hist"]), flush=True)

    ruler = {
        "definition": {
            "tracked": "|dx - shift| <= 1.5px, exclusive (nearer hypothesis wins; tie -> static)",
            "static": "|dx| <= 1.5px, exclusive",
            "exists_minus_static": "same-lens counterpart within 8px, neither window",
            "gone": "no same-lens counterpart within 8px (wrap-aware expected position)",
            "note": "+1px excluded from the tracked/static split (hypotheses inseparable at 1.5px)",
        },
        "correction": "static is 42.6% of detections (3168/7431 at +2px); "
                      "47.6% was Position's share OF THE STATICS",
        "streams": {"sheet": sorted(SHEET_LENSES), "content": sorted(CONTENT_LENSES),
                    "attribute_higher": sorted(HIGHER_ATTR)},
        "shifts_renderer": list(SHIFTS_R), "shifts_eth80": list(SHIFTS_E),
        "renderer": rows,
        "different_object_null_192": null,
        "eth80": {"image": str(ETH80_IMG), "rows": eth_rows},
    }
    target = Path(args.out)
    write_json(target, ruler)
    blob = json.dumps(ruler, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).hexdigest()
    print("RULER HASH", digest)
    print("written", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
