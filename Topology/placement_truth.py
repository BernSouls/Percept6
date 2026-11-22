"""Renderer-truth tracking: where does a node GO when the object moves?

Every pose in these scenes rotates each object about its OWN centre, so there is
no single image-plane rotation and no global centre to rotate about.  The only
honest correspondence is the renderer's own:

  * the node's anchor is lifted to its 3D world point from the DEPTH buffer
    (orthographic: a pixel fixes X and Y, the z-buffer fixes Z);
  * the point is carried through its own object's KNOWN rigid transform;
  * it is re-projected into the posed image.

The result is the node's true image position.  A node-level correspondence is
then the nearest SAME-LENS node within a radius -- the lens is used only to keep
the match sane, never appearance similarity.

This is used by the frame test (per-node paired residuals) and by the census and
arms (pair survival, coverage, accuracy).  Nothing here consults the graph's own
similarity, so it cannot leak into an arm's decision.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception.synthetic3d import Scene

#: px; a base node is matched to a same-lens pose node within this of its
#: predicted position.
TRACK_RADIUS = 8.0


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
