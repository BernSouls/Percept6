"""STEP 1 junction recall/precision on renderer vertices + STEP 2 same/diff AUC.

On the rebuilt renderer-truth-v1 corpus (15 scenes, 192x192), for the L4
ORIENTATION_JUNCTION family:
  * TRUE VERTICES are derived from the renderer's own per-pixel truth:
      - corners: boundary pixels where the traced boundary turns > 45 deg,
      - occlusion junctions: boundary pixels adjacent (4-neighbour) to a
        pixel of a DIFFERENT object id (both non-background).
  * RECALL = share of true vertices that carry at least one L4 junction
    edge endpoint within `tol` px.
  * PRECISION = share of L4 junction edges whose nearer endpoint is within
    `tol` px of a true vertex.

For the L3 CONTOUR_CONTINUITY family and both families' same/diff-object
AUC, the measurement follows the established harness convention: over scene
pairs, same-class pairs should be closer in edge count than diff-class
pairs; AUC = P(same closer).

Every number is labelled with the corpus and the ruler artifact hash.
Run: .venv/bin/python Topology/l3_l4_renderer_gates.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402
from perception.synthetic3d import (  # noqa: E402
    Box, Cylinder, Material, Scene, SceneObject, Sphere,
    rotation_xyz, single_object_scene, occlusion_scene,
)

OUT = ROOT / "results" / "l3_l4_renderer_gates.json"
RULER_SHA = "c0aa914ba511dc20350d4148dcf58f014d901c4da231a1c7819818322a06dfae"
CORPUS = "renderer-truth-v1 (RECONSTRUCTED, 15 scenes, 192x192)"

CONTENT_LENSES = {
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
}
TOL = 12.0         # px tolerance for "edge at a true vertex"
CORNER_DEG = 45.0  # boundary turn threshold for a corner


def _scene_objects(name):
    """Rebuild a renderer scene by name (mirrors build_renderer_corpus.SCENES)."""
    PLAIN = Material(albedo=0.85)
    NOISE = Material(albedo=0.85, texture="noise", texture_freq=14.0,
                     texture_contrast=0.75)
    STRIPES = Material(albedo=0.85, texture="stripes", texture_freq=7.0,
                       texture_contrast=0.7)
    CHECKER = Material(albedo=0.85, texture="checker", texture_freq=9.0,
                       texture_contrast=0.65)
    DARK = Material(albedo=0.55)

    def _obj(n, shape, centre, rx=0.0, ry=0.0, rz=0.0, material=None, layer=0):
        return SceneObject(name=n, shape=shape, rotation=rotation_xyz(rx, ry, rz),
                           centre=np.asarray(centre, dtype=np.float64),
                           material=material or PLAIN, omega=np.zeros(3),
                           velocity=np.zeros(3), layer=int(layer))
    def _box(centre=(0, 0, 0), pose=(0, 0, 0), mat=PLAIN, layer=0,
             half=(0.42, 0.34, 0.30)):
        return _obj("box", Box(half_extents=half), centre, *pose, material=mat, layer=layer)
    def _sphere(centre=(0, 0, 0), radius=0.35, mat=PLAIN, layer=0):
        return _obj("sphere", Sphere(radius=radius), centre, material=mat, layer=layer)
    def _cyl(centre=(0, 0, 0), pose=(0, 0, 0), mat=PLAIN, layer=0):
        return _obj("cylinder", Cylinder(radius=0.30, half_height=0.55), centre, *pose,
                    material=mat, layer=layer)
    def _scene(objects, **kw):
        return Scene(objects=objects,
                     light=np.array([-0.4, -0.5, 0.75]) / np.linalg.norm([-0.4, -0.5, 0.75]),
                     background=0.02, camera_z=10.0, width=192, height=192,
                     extra_lights=(), blur_sigma=kw.get("blur_sigma", 0.0),
                     noise_sigma=kw.get("noise_sigma", 0.0), noise_seed=0,
                     haze_beta=kw.get("haze_beta", 0.0),
                     haze_airlight=kw.get("haze_airlight", 0.0))

    if name == "single_box":
        return _scene([_box()])
    if name == "single_cylinder":
        return _scene([_cyl()])
    if name == "single_sphere":
        return _scene([_sphere()])
    if name == "texture_box":
        return _scene([_box(mat=NOISE)])
    if name == "texture_sphere":
        return _scene([_sphere(mat=STRIPES)])
    if name == "box_occlusion":
        return _scene([_sphere((-0.10, -0.05, 0.55), radius=0.32, layer=0),
                       _box((0.10, 0.05, 0.0), (0.0, 0.2, 0.1), layer=1)])
    if name == "sphere_over_box":
        return _scene([_sphere((0.02, 0.08, 0.50), radius=0.30, layer=0),
                       _box((0.0, -0.06, 0.0), layer=1)])
    if name == "box_over_box":
        return _scene([_box((-0.16, 0.02, 0.45), (0.0, 0.0, 0.25), layer=0),
                       _box((0.16, -0.02, 0.0), (0.0, 0.1, 0.0), layer=1)])
    if name == "disk_occlusion_haze":
        return _scene([_sphere((-0.10, -0.05, 0.55), radius=0.32, layer=0),
                       _box((0.10, 0.05, 0.0), (0.0, 0.2, 0.1), layer=1)],
                      haze_beta=0.35, haze_airlight=0.18, blur_sigma=0.35)
    if name == "f_box_b":
        return _scene([_box((0.0, 0.0, 0.0), (0.15, 0.30, 0.0))])
    if name == "f_haze_a":
        return _scene([_cyl((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))],
                      haze_beta=0.5, haze_airlight=0.22)
    if name == "tv_same_scene":
        return _scene([_box((-0.25, 0.0, 0.0), mat=CHECKER, layer=0),
                       _box((0.25, 0.0, 0.0), mat=CHECKER, layer=1)])
    if name == "no_texture2":
        return _scene([_sphere((-0.25, 0.0, 0.0), radius=0.28, layer=0),
                       _box((0.25, 0.0, 0.0), mat=DARK, layer=1)])
    if name == "texture2_only":
        return _scene([_sphere((-0.25, 0.0, 0.0), radius=0.28, mat=NOISE, layer=0),
                       _box((0.25, 0.0, 0.0), mat=STRIPES, layer=1)])
    if name == "thin_bar":
        return _scene([_box((0.0, 0.0, 0.0), half=(0.46, 0.055, 0.18))])
    raise KeyError(name)


def _boundary_contour(mask):
    """Ordered 8-connected outer contour of a binary mask as [(x, y), ...].

    Moore-neighbour tracing (background kept on the right, clockwise) from
    the topmost-leftmost foreground pixel.  Returns the contour in image
    order (x = column, y = row), closed (first == last point excluded).
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    y0 = ys.min()
    x0 = xs[ys == y0].min()
    # clockwise directions E, SE, S, SW, W, NW, N, NE (dx, dy), y down
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def fg(x, y):
        return 0 <= y < h and 0 <= x < w and mask[y, x]

    # initial search direction: first background neighbour scanning from N
    d0 = None
    for k in range(8):
        dx, dy = dirs[(6 + k) % 8]
        if not fg(x0 + dx, y0 + dy):
            d0 = (6 + k) % 8
            break
    contour = [(x0, y0)]
    cx, cy = x0, y0
    c = d0
    while True:
        nxt = None
        for k in range(1, 9):
            dd = (c + k) % 8
            dx, dy = dirs[dd]
            nx, ny = cx + dx, cy + dy
            if fg(nx, ny):
                nxt = (nx, ny, dd)
                break
        if nxt is None:
            break
        nx, ny, nd = nxt
        if (nx, ny) == (x0, y0):
            break
        cx, cy = nx, ny
        contour.append((cx, cy))
        c = (nd + 5) % 8   # keep background on the right
    return contour


def _true_vertices(truth):
    """(corners, occlusion_junctions) as boolean masks over the frame."""
    oid = truth.object_id
    mask = truth.mask
    h, w = mask.shape

    # occlusion junctions: boundary pixels with a 4-neighbour of a different
    # non-background object id
    er4 = ndimage.binary_erosion(mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    boundary = mask & ~er4
    occ = np.zeros_like(mask)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = np.roll(np.roll(oid, dy, axis=0), dx, axis=1)
        both = mask & np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        occ |= (both & (oid != shifted))
    occ &= boundary

    # corners: trace the union contour, mark pixels where the boundary turns
    # more than CORNER_DEG.  The turn at contour vertex i is the angle between
    # the incoming and outgoing segment directions (axial, [0, pi/2]).
    corner = np.zeros_like(mask)
    contour = _boundary_contour(mask)
    n = len(contour)
    if n >= 4:
        thr = math.radians(CORNER_DEG)
        # segment directions between consecutive vertices
        segs = []
        for i in range(n):
            x0, y0 = contour[i]
            x1, y1 = contour[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            if dx == 0 and dy == 0:
                segs.append(None)
            else:
                segs.append(math.atan2(dy, dx) % math.pi)
        for i in range(n):
            a = segs[i]
            b = segs[(i - 1) % n]
            if a is None or b is None:
                continue
            d = abs(a - b) % math.pi
            turn = min(d, math.pi - d)
            if turn > thr:
                x, y = contour[i]
                corner[y, x] = True
    return corner, occ


def _roc_auc(pos, neg):
    if not pos or not neg:
        return 0.5
    wins = sum(1.0 for p in pos for n in neg if p < n) + \
           0.5 * sum(1.0 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def main() -> None:
    names = ["single_box", "single_cylinder", "single_sphere", "texture_box",
             "texture_sphere", "box_occlusion", "sphere_over_box", "box_over_box",
             "disk_occlusion_haze", "f_box_b", "f_haze_a", "tv_same_scene",
             "no_texture2", "texture2_only", "thin_bar"]

    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    try:
        rec_total = hit_total = 0
        prec_hit = prec_total = 0
        vertex_counts = {}
        edge_counts = {}
        class_of = {}   # scene name -> primary object class label
        feat = {}       # scene name -> (cc_count, oj_count) for AUC

        for name in names:
            truth = _scene_objects(name).render(192, 192, supersample=2)
            img = truth.image_u8
            corner, occ = _true_vertices(truth)
            n_corner = int(corner.sum())
            n_occ = int(occ.sum())
            vertex_counts[name] = {"corners": n_corner, "occlusion": n_occ,
                                   "total": n_corner + n_occ}
            # primary class label for same/diff AUC
            objs = list(truth.object_names)
            class_of[name] = objs[0] if objs else name

            tr = {}
            g = FE.build_graph(img, name=f"rg_{name}", record_co_located=False, trace=tr)
            # L4 edge endpoints (content-content ORIENTATION_JUNCTION only;
            # the sheet-level ORIENTATION_JUNCTION family is a different
            # mechanism and must not count against L4 precision)
            oj_edges = []
            cc_edges = []
            for u, v, et in g.subgraph.edges:
                etn = getattr(et, "name", str(et))
                lu = str(g.sigs.get(u, ("?",))[0])
                lv = str(g.sigs.get(v, ("?",))[0])
                if etn == "ORIENTATION_JUNCTION" and lu in CONTENT_LENSES and lv in CONTENT_LENSES:
                    oj_edges.append((u, v))
                elif etn == "CONTOUR_CONTINUITY" and lu in CONTENT_LENSES and lv in CONTENT_LENSES:
                    cc_edges.append((u, v))
            edge_counts[name] = {"ORIENTATION_JUNCTION": len(oj_edges),
                                 "CONTOUR_CONTINUITY": len(cc_edges)}
            feat[name] = (len(cc_edges), len(oj_edges))

            # position of each endpoint via the trace's anchor_positions
            # (the stage-6 position registry, keyed by anchor -- the SAME
            # source the mechanisms use; _stage6_positions() is cleared by
            # finalize_step() so the trace copy is the authoritative map)
            from perception import attachment as _att
            amap = tr.get("anchor_positions") or {}
            cg = tr["cg"]
            anchors = cg.pattern_node_anchor_map()

            def _pos(nid):
                a = anchors.get(nid)
                p = _att._xy(amap, a) if a is not None else None
                if p is None:
                    p = cg.pattern_node_position(nid)
                return p

            vert_pts = []
            yy, xx = np.mgrid[0:192, 0:192]
            for m in (corner, occ):
                vert_pts.extend(zip(xx[m].tolist(), yy[m].tolist()))

            # recall: true vertices with an L4 endpoint within TOL
            hit = 0
            for (vx, vy) in vert_pts:
                found = False
                for (u, v) in oj_edges:
                    for nid in (u, v):
                        p = _pos(nid)
                        if p is None:
                            continue
                        if abs(p[0] - vx) <= TOL and abs(p[1] - vy) <= TOL:
                            found = True
                            break
                    if found:
                        break
                if found:
                    hit += 1
            rec_total += len(vert_pts)
            hit_total += hit

            # precision: L4 edges with an endpoint within TOL of a true vertex
            for (u, v) in oj_edges:
                near = False
                for nid in (u, v):
                    p = _pos(nid)
                    if p is None:
                        continue
                    for (vx, vy) in vert_pts:
                        if abs(p[0] - vx) <= TOL and abs(p[1] - vy) <= TOL:
                            near = True
                            break
                    if near:
                        break
                if near:
                    prec_hit += 1
                prec_total += 1

        recall = hit_total / max(1, rec_total)
        precision = prec_hit / max(1, prec_total)

        # same/diff-object AUC for both families (edge-count distance)
        auc = {}
        for fam_idx, fam in ((0, "CONTOUR_CONTINUITY"), (1, "ORIENTATION_JUNCTION")):
            pos, neg = [], []
            names_l = list(feat.keys())
            for i in range(len(names_l)):
                for j in range(i + 1, len(names_l)):
                    d = abs(feat[names_l[i]][fam_idx] - feat[names_l[j]][fam_idx])
                    if class_of[names_l[i]] == class_of[names_l[j]]:
                        pos.append(d)
                    else:
                        neg.append(d)
            auc[fam] = _roc_auc(pos, neg)

        payload = {
            "protocol": "L3+L4 STEP1 renderer-vertex recall/precision + STEP2 same/diff AUC",
            "corpus": CORPUS,
            "ruler_artifact_sha256": RULER_SHA,
            "tolerance_px": TOL,
            "corner_threshold_deg": CORNER_DEG,
            "junction_recall": recall,
            "junction_precision": precision,
            "junction_recall_hits": hit_total,
            "junction_recall_total_vertices": rec_total,
            "junction_precision_hits": prec_hit,
            "junction_precision_total_edges": prec_total,
            "junction_gate_pass": bool(recall >= 0.80 and precision >= 0.70),
            "same_diff_auc": auc,
            "auc_gate_pass": bool(auc["CONTOUR_CONTINUITY"] >= 0.75 and auc["ORIENTATION_JUNCTION"] >= 0.75),
            "per_scene_vertices": vertex_counts,
            "per_scene_edges": edge_counts,
        }
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(json.dumps({k: payload[k] for k in
                          ("junction_recall", "junction_precision",
                           "junction_gate_pass", "same_diff_auc",
                           "auc_gate_pass")}, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


if __name__ == "__main__":
    main()
