"""Frame test: which direction providers are REAL?

The mission's requirement, verbatim in substance:

  Rotate an object on a plain background about the viewing axis by delta (and a
  textured object too).  Track each node with renderer truth.  Report the
  PER-NODE PAIRED difference of its provider direction: it must concentrate at
  delta.  No scene-level circular means.  Also run a 3D orbit step.

Why the earlier rejection was unproven: it rotated a FULL-IMAGE GRATING, which
has no object boundary, so Boundary / ContourFragment read the image frame,
which does not rotate.  Here the object rotates about the viewing axis, its
anchors are tracked by the KNOWN pose transform (renderer truth, not
appearance), and every node gets its own paired difference.

PROVIDERS
  lens:<Lens>     the node's stored theta (de-binned), folded to its axis
  field           the local orientation field at the node's ANCHOR, from the
                  image structure tensor -- independent of the lens's theta,
                  available to ANY node that has an anchor

AXIAL vs SIGNED
  A structure tensor gives an AXIS, not an arrow: it is 180-degree ambiguous.
  Each provider is therefore tested for its AXIAL response (pair difference
  folded by 180 deg) and, where the provider can carry a sign, for its SIGNED
  response.  A provider is admitted only if its axial response concentrates at
  the rotation; whether the signed response also concentrates decides whether
  the score must be made flip-symmetric.

Run:  .venv/bin/python Topology/placement_frame_test.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

from perception.synthetic3d import (Scene, SceneObject, Sphere, Box,  # noqa: E402
                                    Material)
from Topology.placement_scenes import posed  # noqa: E402
from matching.placement import node_theta  # noqa: E402

OUT = os.path.join(ROOT, "Topology", "results", "placement")
FIELD_WINDOW = 7          # odd; structure-tensor window in pixels
TRACK_RADIUS = 8.0        # px; a base node is tracked if a same-lens node of
                          # the posed graph lands within this of its predicted
                          # position
AXIAL_TOL = math.radians(15.0)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def sphere_inplane(textured: bool = True) -> Scene:
    """One sphere, centred, plain background.

    The texture runs along object-X so the pattern rotates IN THE IMAGE PLANE
    when the object turns about the viewing axis.  (With the default
    texture_dir, stripes are planes of constant z and a z-rotation barely
    changes the image -- measured while building this test.)
    """
    if textured:
        mat = Material(albedo=0.85, texture="stripes", texture_freq=10.0,
                       texture_contrast=0.8, texture_space="object",
                       texture_dir=(1.0, 0.0, 0.0))
    else:
        mat = Material(albedo=0.85, texture="none")
    return Scene(objects=[SceneObject(name="s", shape=Sphere(0.34),
                                      centre=np.array([0.0, 0.0, 0.0]),
                                      material=mat, layer=0)])


def sphere_and_box(textured: bool = True) -> Scene:
    """The 3D orbit case: two objects, so a rotation moves them in depth too."""
    if textured:
        m1 = Material(albedo=0.85, texture="stripes", texture_freq=10.0,
                      texture_contrast=0.8, texture_space="object",
                      texture_dir=(1.0, 0.0, 0.0))
        m2 = Material(albedo=0.6, texture="checker", texture_freq=8.0,
                      texture_contrast=0.7, texture_space="object",
                      texture_dir=(1.0, 0.0, 0.0))
    else:
        m1 = Material(albedo=0.85, texture="none")
        m2 = Material(albedo=0.6, texture="none")
    return Scene(objects=[
        SceneObject(name="s", shape=Sphere(0.30),
                    centre=np.array([-0.22, 0.0, 0.0]), material=m1, layer=0),
        SceneObject(name="b", shape=Box((0.20, 0.20, 0.20)),
                    centre=np.array([0.30, 0.04, 0.06]), material=m2, layer=0),
    ])


# ---------------------------------------------------------------------------
# the field provider
# ---------------------------------------------------------------------------
def field_angles(image_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel axial orientation of the local orientation field.

    Returns (angle, coherence).  The structure tensor's dominant eigenvector is
    the direction of MAXIMUM GRADIENT (the stripe normal); it is an axis, so
    the returned angle is on [0, pi).
    """
    img = np.asarray(image_u8, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    gy, gx = np.gradient(img)
    h, w = img.shape
    r = FIELD_WINDOW // 2
    pad = np.pad(gx, r, mode="edge")
    pady = np.pad(gy, r, mode="edge")

    def boxsum(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.cumsum(a, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        k = FIELD_WINDOW
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k])

    Jxx = boxsum(pad * pad)
    Jyy = boxsum(pady * pady)
    Jxy = boxsum(pad * pady)
    tr = Jxx + Jyy
    # dominant orientation of the gradient
    ang = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    # coherence: eigenvalue separation over trace
    disc = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    coh = np.where(tr > 1e-9, disc / np.maximum(tr, 1e-9), 0.0)
    out_ang = np.full((h, w), np.nan)
    out_coh = np.zeros((h, w))
    ys, xs = np.mgrid[0:h, 0:w]
    valid = (ys >= 0) & (xs >= 0)
    out_ang[valid] = ang[valid]
    out_coh[valid] = coh[valid]
    return out_ang, out_coh


def anchor_field(g: Any, ang: np.ndarray, coh: np.ndarray) -> Dict[int, float]:
    """node -> field angle at its anchor (nearest pixel)."""
    h, w = ang.shape
    out: Dict[int, float] = {}
    for n in sorted(g.subgraph.nodes):
        a = g.subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        x, y = int(round(float(a[0]))), int(round(float(a[1])))
        if not (0 <= x < w and 0 <= y < h):
            continue
        v = ang[y, x]
        if np.isfinite(v):
            out[int(n)] = float(v)
    return out


# ---------------------------------------------------------------------------
# renderer-truth tracking
# ---------------------------------------------------------------------------
def project(scene: Scene, p_world: np.ndarray) -> Tuple[float, float]:
    """World point -> pixel, matching the renderer's own camera.

    Orthographic is ``x = 0.5 W + X / pixel_scale`` (``Scene.pixel_to_world``
    inverted); perspective is the pinhole the renderer documents,
    ``0.5 W + f X / e`` with ``e = camera_z - Z``.
    """
    X, Y, Z = (float(p_world[0]), float(p_world[1]), float(p_world[2]))
    W, H = float(scene.width), float(scene.height)
    if scene.camera is None:
        return (0.5 * W + X / scene.pixel_scale,
                0.5 * H + Y / scene.pixel_scale)
    f = float(scene.camera.focal_px)
    e = max(float(scene.camera_z) - Z, 1e-9)
    return (0.5 * W + f * X / e, 0.5 * H + f * Y / e)


def object_centre_px(scene: Scene, name: str) -> Optional[Tuple[float, float]]:
    for o in scene.objects:
        if o.name == name:
            return project(scene, np.asarray(o.centre, dtype=np.float64))
    return None


def world_point_at(scene: Scene, render: Any, x: float, y: float
                   ) -> Optional[np.ndarray]:
    """The 3D WORLD point a node's anchor sits on, from the depth buffer.

    Under the orthographic camera a pixel fixes X and Y exactly
    (``pixel_to_world``) and the z-buffer fixes Z (``depth = camera_z - Z``).
    That is renderer truth, not an appearance guess, and it is what lets a node
    be tracked through ANY pose -- in-plane rotation, 3D orbit, multiple
    objects -- instead of only through a rotation about one assumed centre.
    """
    h, w = render.depth.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    d = float(render.depth[yi, xi])
    if not np.isfinite(d):
        return None
    if scene.camera is None:
        X = (float(x) - 0.5 * w) * scene.pixel_scale
        Y = (float(y) - 0.5 * h) * scene.pixel_scale
        Z = float(scene.camera_z) - d
        return np.array([X, Y, Z], dtype=np.float64)
    f = float(scene.camera.focal_px)
    e = d
    a = (float(x) - 0.5 * w) / f
    b = (float(y) - 0.5 * h) / f
    return np.array([a * e, b * e, float(scene.camera_z) - e], dtype=np.float64)


def transforms_by_name(scene0: Scene, scene1: Scene):
    """object name -> (centre_old, rotation_old, centre_new, rotation_new)."""
    old = {o.name: o for o in scene0.objects}
    new = {o.name: o for o in scene1.objects}
    out = {}
    for name in sorted(set(old) & set(new)):
        a, b = old[name], new[name]
        out[name] = (np.asarray(a.centre, dtype=np.float64),
                     np.asarray(a.rotation, dtype=np.float64),
                     np.asarray(b.centre, dtype=np.float64),
                     np.asarray(b.rotation, dtype=np.float64))
    return out


def predict(scene0: Scene, render0: Any, scene1: Scene, g0: Any
            ) -> Dict[int, Tuple[float, float]]:
    """base node -> its PIXEL in the posed image, from renderer truth.

    The anchor is lifted to its 3D world point from the depth buffer, carried
    through its own object's known rigid transform and re-projected.  No
    matching, no appearance: this is the node's true image position.

    A field provider is evaluated HERE, at the predicted pixel.  Matching to a
    nearby node instead would land up to `TRACK_RADIUS` px off, which on a
    periodic texture is a different phase of the pattern and shows up as frame
    noise that has nothing to do with the frame.
    """
    names = [o.name for o in scene0.objects]
    tf = transforms_by_name(scene0, scene1)
    out: Dict[int, Tuple[float, float]] = {}
    for n in sorted(g0.subgraph.nodes):
        a = g0.subgraph.anchor_of_node.get(n)
        if a is None:
            continue
        p = world_point_at(scene0, render0, float(a[0]), float(a[1]))
        if p is None:
            continue
        xi, yi = int(round(float(a[0]))), int(round(float(a[1])))
        oid = int(render0.object_id[yi, xi]) if render0.object_id is not None \
            else -1
        if oid < 0 or oid >= len(names):
            continue
        t = tf.get(names[oid])
        if t is None:
            continue
        c0, R0, c1, R1 = t
        q = c1 + (R1 @ (R0.T @ (p - c0)))
        out[int(n)] = project(scene1, q)
    return out


def track(scene0: Scene, render0: Any, scene1: Scene, g0: Any, g1: Any,
          radius: float = TRACK_RADIUS) -> Dict[int, int]:
    """base node -> posed node, using RENDERER TRUTH (depth + known pose).

    Each base anchor is lifted to its 3D world point from the depth buffer,
    carried through its own object's known rigid transform, and re-projected.
    The correspondence is then the nearest SAME-LENS node within `radius`.
    Appearance never decides it.
    """
    names = [o.name for o in scene0.objects]
    tf = transforms_by_name(scene0, scene1)
    by_lens: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for n in sorted(g1.subgraph.nodes):
        a = g1.subgraph.anchor_of_node.get(n)
        sg = g1.sigs.get(n)
        if a is None or sg is None:
            continue
        by_lens[str(sg[0])].append((int(n), float(a[0]), float(a[1])))

    cand: List[Tuple[float, int, int]] = []
    for n in sorted(g0.subgraph.nodes):
        a = g0.subgraph.anchor_of_node.get(n)
        sg = g0.sigs.get(n)
        if a is None or sg is None:
            continue
        p = world_point_at(scene0, render0, float(a[0]), float(a[1]))
        if p is None:
            continue
        xi, yi = int(round(float(a[0]))), int(round(float(a[1])))
        oid = int(render0.object_id[yi, xi]) if render0.object_id is not None \
            else -1
        if oid < 0 or oid >= len(names):
            continue
        t = tf.get(names[oid])
        if t is None:
            continue
        c0, R0, c1, R1 = t
        q = c1 + (R1 @ (R0.T @ (p - c0)))
        px, py = project(scene1, q)
        for (m, x, y) in by_lens.get(str(sg[0]), ()):  # noqa: E741
            d = math.hypot(x - px, y - py)
            if d <= radius:
                cand.append((d, int(n), m))
    cand.sort()
    out: Dict[int, int] = {}
    used: set = set()
    for d, n, m in cand:
        if n in out or m in used:
            continue
        out[n] = m
        used.add(m)
    return out


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def wrap_ax(x: float) -> float:
    """Fold to (-pi/2, pi/2]."""
    while x > math.pi / 2:
        x -= math.pi
    while x <= -math.pi / 2:
        x += math.pi
    return x


def wrap_two(x: float) -> float:
    """Fold to (-pi, pi]."""
    while x > math.pi:
        x -= 2 * math.pi
    while x <= -math.pi:
        x += 2 * math.pi
    return x


def paired_stats(diffs: List[float]) -> Dict[str, Any]:
    """Report the PER-NODE paired distribution, never a scene-level mean.

    ``median_signed`` is the quantity that decides a provider.  A provider that
    TRACKS the rotation leaves a residual concentrated at 0; a provider that
    does not move leaves a residual concentrated at +/-delta, which has the
    same |residual| and the same circular concentration.  Only the SIGN
    separates the two, so the sign is reported rather than folded away.
    """
    if not diffs:
        return {"n": 0}
    d = np.asarray(diffs, dtype=np.float64)
    ax = np.abs(d)
    conc = float(math.hypot(float(np.mean(np.cos(2 * d))),
                            float(np.mean(np.sin(2 * d)))))
    # axial median, via the doubling trick
    med = 0.5 * math.atan2(float(np.median(np.sin(2 * d))),
                           float(np.median(np.cos(2 * d))))
    return {
        "n": int(d.size),
        "median_signed": med,
        "median_signed_deg": math.degrees(med),
        "median_abs": float(np.median(ax)),
        "q75_abs": float(np.quantile(ax, 0.75)),
        "q90_abs": float(np.quantile(ax, 0.90)),
        "share_within_15deg": float(np.mean(ax <= AXIAL_TOL)),
        "share_within_30deg": float(np.mean(ax <= 2 * AXIAL_TOL)),
        "concentration": conc,
        "histogram_deg": [int(x) for x in
                          np.histogram(np.degrees(ax), bins=[0, 5, 10, 15, 30,
                                                             45, 60, 90])[0]],
    }


def paired_best_sign(t0: List[float], t1: List[float], delta: float
                     ) -> Tuple[str, Dict[str, Any]]:
    """The paired residual, with the renderer's rotation sense DETECTED.

    The sense is read off the PER-NODE RAW difference (``t1 - t0``), which
    concentrates unambiguously at +delta or -delta, and only then is the
    residual formed.  It must NOT be chosen by maximising the concentration of
    the RESIDUAL: a provider that does not move at all gives a residual that is
    perfectly CONSTANT (+/-delta), so it scores a concentration of 1.0 and wins
    the comparison -- which is exactly how a static provider came to be scored
    as tracking in the first version of this test (measured: field median
    residual 20.59 deg for a 10 deg turn, from a provider whose raw difference
    was +10.59).
    """
    raw = [wrap_ax(b - a) for a, b in zip(t0, t1)]
    if not raw:
        return "+", {"n": 0}
    med_raw = 0.5 * math.atan2(float(np.median(np.sin(2 * np.asarray(raw)))),
                               float(np.median(np.cos(2 * np.asarray(raw)))))
    sgn = "+" if abs(wrap_ax(med_raw - delta)) <= abs(wrap_ax(med_raw + delta)) \
        else "-"
    s = +1.0 if sgn == "+" else -1.0
    st = paired_stats([wrap_ax(r - s * delta) for r in raw])
    st["raw_median_deg"] = math.degrees(med_raw)
    return sgn, st


#: A provider passes if its per-node signed residual sits near zero for MOST
#: nodes.  These thresholds are fixed here, before the table is read.
PASS_MEDIAN_DEG = 12.0
PASS_SHARE = 0.60


def provider_verdict(r: Dict[str, Any]) -> str:
    """TRACKS / STATIC / NOISE, with the two hypotheses scaled by delta.

    A fixed absolute tolerance cannot separate them: at delta = 10 deg a
    provider that does not move at all leaves a residual of exactly 10 deg,
    which a 12 deg tolerance calls "tracks".  Both hypotheses are therefore
    judged relative to delta, which is the only scale the test has.
    """
    if r.get("n", 0) < 5:
        return "too few"
    d = float(r["delta_deg"])
    med = abs(float(r["median_signed_deg"]))
    tracks_tol = min(PASS_MEDIAN_DEG, 0.35 * d)
    if med <= tracks_tol and r["share_within_15deg"] >= PASS_SHARE:
        return "TRACKS"
    if abs(med - d) <= 0.35 * d:
        return "STATIC"
    return "NOISE"


def main() -> int:
    import guard
    guard.activate()
    from eth80_frontend import build_graph

    os.makedirs(OUT, exist_ok=True)
    deltas = [10.0, 20.0, 30.0]
    rows: List[Dict[str, Any]] = []

    for stim_name, make in (("sphere_inplane", sphere_inplane),
                            ("sphere_plain", lambda: sphere_inplane(False)),
                            ("sphere_box_orbit", sphere_and_box)):
        scene = make()
        base = posed(scene, 0.0).render()
        g0 = build_graph(base.image_u8, "%s|0" % stim_name)
        ang0, coh0 = field_angles(base.image_u8)
        f0 = anchor_field(g0, ang0, coh0)
        centre = object_centre_px(scene, scene.objects[0].name)
        for deg in deltas:
            delta = math.radians(deg)
            posed_scene = posed(scene, deg)
            p = posed_scene.render()
            g1 = build_graph(p.image_u8, "%s|%g" % (stim_name, deg))
            ang1, coh1 = field_angles(p.image_u8)
            f1 = anchor_field(g1, ang1, coh1)
            pred = predict(scene, base, posed_scene, g0)
            tm = track(scene, base, posed_scene, g0, g1)
            n0 = len(g0.subgraph.nodes)
            print("  %s %+g: %d/%d predicted, %d matched"
                  % (stim_name, deg, len(pred), n0, len(tm)), flush=True)

            # ---- provider: lens theta, per lens
            by_lens: Dict[str, List[float]] = defaultdict(list)
            for u in sorted(tm):
                m = tm[u]
                su, sm = g0.sigs.get(u), g1.sigs.get(m)
                if su is None or sm is None:
                    continue
                t0, t1 = node_theta(su[1]), node_theta(sm[1])
                if t0 is None or t1 is None:
                    continue
                by_lens[str(su[0])].append((t0, t1))
            for lens in sorted(by_lens):
                if len(by_lens[lens]) < 5:
                    continue
                t0s = [a for a, _b in by_lens[lens]]
                t1s = [b for _a, b in by_lens[lens]]
                sgn, st = paired_best_sign(t0s, t1s, delta)
                rows.append(dict(stimulus=stim_name, delta_deg=deg,
                                 provider="lens:" + lens, sign=sgn, **st))

            # ---- provider: local field at the anchor
            f0s: List[float] = []
            f1s: List[float] = []
            h1, w1 = ang1.shape
            for u in sorted(pred):
                if u not in f0:
                    continue
                px, py = pred[u]
                xi, yi = int(round(px)), int(round(py))
                if not (0 <= xi < w1 and 0 <= yi < h1):
                    continue
                v = ang1[yi, xi]
                if not np.isfinite(v):
                    continue
                f0s.append(f0[u])
                f1s.append(float(v))
            if len(f0s) >= 5:
                sgn, st = paired_best_sign(f0s, f1s, delta)
                rows.append(dict(stimulus=stim_name, delta_deg=deg,
                                 provider="field", sign=sgn, **st))

    for r in rows:
        r["verdict"] = provider_verdict(r)
    print("\n%-18s %6s %-24s %5s %11s %10s %8s %-12s"
          % ("stimulus", "delta", "provider", "n", "med signed", "share<15",
             "conc", "verdict"))
    for r in rows:
        print("%-18s %6g %-24s %5d %11.2f %10.2f %8.3f %-12s"
              % (r["stimulus"], r["delta_deg"], r["provider"], r["n"],
                 r.get("median_signed_deg", float("nan")),
                 r.get("share_within_15deg", float("nan")),
                 r.get("concentration", float("nan")), r["verdict"]))
    with open(os.path.join(OUT, "frame_test.json"), "w") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    print("\nwrote", os.path.join(OUT, "frame_test.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
