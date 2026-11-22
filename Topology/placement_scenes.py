"""Task 4 PLACEMENT mission — Part 5: scenes and derived tracks.

Built ONCE, complete, before any scorer work.

THE FOUR FAMILIES
-----------------
RESOLVABLE  occluded fragments of identical primitives; texture elements at
            varying distance from a face boundary, ``texture_space="object"``.
            Appearance ties, relational context differs.  The positive test.
NEGATIVE-A  two identical objects, same pose, no occlusion, INTRA-object edges
            only.  Expected accuracy on the swappable subset is about 0.5
            (the assignment is decided by tie order), NOT 0 or 1.
NEGATIVE-B  same scene, CROSS-object edges allowed.  A win is legitimate: the
            object swap reverses the cross-object vectors.
CONTROL     single ordinary objects.

TRACKS ARE DERIVED, NOT STORED
------------------------------
``synthetic3d`` has no track field.  A track is produced here by carrying each
feature's ``point_local`` on its own object across poses:

    p_world = SceneObject.local_to_world(p_local)
    p_image = camera_z / pixel_scale / focal_px  ->  projection

``derive_track`` does exactly that, and ``assert_identity`` checks the
mission's requirement that at zero pose change the derivation returns the
identity.

DETERMINISM
-----------
Every collection is sorted by a canonical key before it is returned, and no
dict ordering or frozenset iteration reaches a result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.synthetic3d import (Box, Camera, Cylinder, Material, Scene,
                                    SceneObject, Sphere)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
RESOLVABLE = "RESOLVABLE"
NEGATIVE_A = "NEGATIVE-A"
NEGATIVE_B = "NEGATIVE-B"
CONTROL = "CONTROL"
FAMILIES = (RESOLVABLE, NEGATIVE_A, NEGATIVE_B, CONTROL)


# ---------------------------------------------------------------------------
# a feature: one local point on one object, with the identity the truth keeps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Feature:
    """One tracked feature: a point in ONE object's local frame."""

    object_name: str          # which object carries it
    point_local: Tuple[float, float, float]
    kind: str                 # "texture" | "landmark" (truth label)
    #: the object-local texture coordinate, when the feature is a texture element
    uv_local: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SceneSpec:
    """A scene plus the features that are worth tracking in it."""

    name: str
    family: str
    scene: Scene
    features: Tuple[Feature, ...]
    #: view steps the mission asks for: (label, pose dict)
    views: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# view steps: translation, rotation 5-15 deg, scale, partial occlusion
# ---------------------------------------------------------------------------
def view_steps() -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    """The pre-registered view steps.  Fixed here, before any scoring."""
    return (
        ("v0_base",        {"rot_deg": 0.0,  "shift": (0.0, 0.0), "scale": 1.0}),
        ("v1_rot5",        {"rot_deg": 5.0,  "shift": (0.0, 0.0), "scale": 1.0}),
        ("v2_rot10",       {"rot_deg": 10.0, "shift": (0.0, 0.0), "scale": 1.0}),
        ("v3_rot15",       {"rot_deg": 15.0, "shift": (0.0, 0.0), "scale": 1.0}),
        ("v4_translate",   {"rot_deg": 0.0,  "shift": (0.04, 0.0), "scale": 1.0}),
        ("v5_scale",       {"rot_deg": 0.0,  "shift": (0.0, 0.0), "scale": 1.15}),
    )


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def posed(scene: Scene, rot_deg: float = 0.0, shift: Tuple[float, float] = (0.0, 0.0),
          scale: float = 1.0, occluder: bool = False) -> Scene:
    """A copy of ``scene`` under one pose step.

    Rotation is about each object's own centre (so a rigid rotation, which is
    what "every node moves together" means); ``shift`` translates every object
    by the same world vector; ``scale`` scales the primitive radii.  ``occluder``
    adds a near occluding bar in front of the scene's front object.
    """
    objs: List[SceneObject] = []
    for o in scene.objects:
        r = _rot_z(rot_deg) @ np.asarray(o.rotation, dtype=np.float64)
        sh = np.zeros(3, dtype=np.float64)
        sh[:len(shift)] = np.asarray(shift, dtype=np.float64)[:3]
        c = np.asarray(o.centre, dtype=np.float64) + sh
        shape = o.shape
        if scale != 1.0:
            if isinstance(shape, Sphere):
                shape = Sphere(radius=shape.radius * scale)
            elif isinstance(shape, Box):
                shape = Box(tuple(float(x) * scale for x in shape.half_extents))
            elif isinstance(shape, Cylinder):
                # The primitive stores `half_height`, not `height`; scaling the
                # old attribute name raised AttributeError the first time an
                # EVAL scene actually used a cylinder.
                shape = Cylinder(radius=shape.radius * scale,
                                 half_height=shape.half_height * scale)
        objs.append(replace(o, rotation=r, centre=c, shape=shape))
    if occluder:
        # A thin slab in front of the frontmost object.  It occludes part of
        # it WITHOUT changing any object's own pose, which is what "partial
        # occlusion" means for the track derivation (point_local is untouched).
        zs = [float(np.asarray(o.centre, dtype=np.float64)[2]) for o in objs]
        zf = (min(zs) if zs else 0.0) - 0.35
        objs.append(SceneObject(
            name="_occluder",
            shape=Box((0.16, 0.55, 0.02)),
            centre=np.array([0.0, 0.0, zf], dtype=np.float64),
            material=Material(albedo=0.5, texture="none"),
            layer=0,
        ))
    return Scene(objects=objs, light=scene.light, background=scene.background,
                 ambient=scene.ambient, camera_z=scene.camera_z,
                 pixel_scale=scene.pixel_scale, width=scene.width,
                 height=scene.height, camera=scene.camera,
                 extra_lights=scene.extra_lights, blur_sigma=scene.blur_sigma,
                 noise_sigma=scene.noise_sigma, noise_seed=scene.noise_seed)


# ---------------------------------------------------------------------------
# projection / track derivation
# ---------------------------------------------------------------------------
def project(scene: Scene, p_world: np.ndarray) -> Optional[Tuple[float, float]]:
    """World -> image pixel, using the scene's own optics.

    Orthographic (``camera is None``) is the ``e -> inf`` limit with
    ``f = e / pixel_scale``: ``(0.5W + X/pixel_scale, 0.5H + Y/pixel_scale)``.
    Pinhole uses ``focal_px`` and the axial depth ``camera_z - Z``.
    """
    x, y, z = float(p_world[0]), float(p_world[1]), float(p_world[2])
    W, H = float(scene.width), float(scene.height)
    if scene.camera is None:
        return (0.5 * W + x / float(scene.pixel_scale),
                0.5 * H + y / float(scene.pixel_scale))
    e = float(scene.camera_z) - z
    if e <= 1e-9:
        return None
    f = float(scene.camera.focal_px)
    return (0.5 * W + f * x / e, 0.5 * H + f * y / e)


def derive_track(spec: SceneSpec, view_label: str) -> Dict[str, Tuple[float, float]]:
    """The image position of every feature under ONE view step.

    This is the mission's derivation: carry ``point_local`` across the pose
    with ``local_to_world``, then project.  Returns ``{feature_key: (x, y)}``
    for features that project inside the frame; occluded features are still
    returned (occlusion is a RENDER fact, not a geometry fact) and the caller
    decides.
    """
    poses = dict(spec.views)
    pose = poses.get(view_label)
    if pose is None:
        raise KeyError("unknown view step %r (known: %s)"
                       % (view_label, [v[0] for v in spec.views]))
    sc = posed(spec.scene, pose["rot_deg"], tuple(pose["shift"]),
               float(pose["scale"]))
    by_name = {o.name: o for o in sc.objects}
    out: Dict[str, Tuple[float, float]] = {}
    for f in spec.features:
        o = by_name.get(f.object_name)
        if o is None:
            continue
        p = o.local_to_world(np.asarray(f.point_local, dtype=np.float64))
        xy = project(sc, p)
        if xy is None:
            continue
        out[_fkey(f)] = xy
    return out


def _fkey(f: Feature) -> str:
    return "%s|%.6f|%.6f|%.6f|%s" % (f.object_name, f.point_local[0],
                                     f.point_local[1], f.point_local[2], f.kind)


def assert_identity(spec: SceneSpec, tol: float = 1e-9) -> Dict[str, Any]:
    """At zero pose change the derivation must return the identity.

    Checks that the ``v0_base`` track equals a direct ``local_to_world`` +
    ``project`` on the UNPOSED scene, to ``tol``.
    """
    base = derive_track(spec, "v0_base")
    direct: Dict[str, Tuple[float, float]] = {}
    by_name = {o.name: o for o in spec.scene.objects}
    for f in spec.features:
        o = by_name.get(f.object_name)
        if o is None:
            continue
        p = o.local_to_world(np.asarray(f.point_local, dtype=np.float64))
        xy = project(spec.scene, p)
        if xy is not None:
            direct[_fkey(f)] = xy
    worst = 0.0
    for k in sorted(set(base) | set(direct)):
        a, b = base.get(k), direct.get(k)
        if a is None or b is None:
            return {"ok": False, "reason": "missing feature %s" % k}
        d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        worst = max(worst, d)
    return {"ok": worst <= tol, "worst_px": worst, "n": len(direct)}


# ---------------------------------------------------------------------------
# the ambiguity rule (truth-free), fixed BEFORE any scoring
# ---------------------------------------------------------------------------
def ambiguous_by_appearance(features: Sequence[Feature],
                            sig_of: Dict[str, Any]) -> Dict[str, bool]:
    """Truth-free ambiguity label: a feature is AMBIGUOUS when another
    feature in the same scene shares its appearance signature.

    The signature is the node's own ``(lens type, signature bytes)`` from the
    perceptual graph, so this uses only what the matcher can see.  Fixed here,
    before any arm is run.
    """
    buckets: Dict[Any, List[str]] = {}
    for f in features:
        k = _fkey(f)
        s = sig_of.get(k)
        if s is None:
            continue
        buckets.setdefault(s, []).append(k)
    out: Dict[str, bool] = {}
    for f in features:
        k = _fkey(f)
        s = sig_of.get(k)
        out[k] = bool(s is not None and len(buckets.get(s, ())) > 1)
    return out


# ---------------------------------------------------------------------------
# the scene families
# ---------------------------------------------------------------------------
def _texture_features(obj: str, radius: float, n: int, seed: int = 0,
                      kinds: Optional[Sequence[str]] = None) -> Tuple[Feature, ...]:
    """Deterministic texture-element features on a sphere's surface.

    Points are a Fibonacci lattice in the object's local frame, so they are
    fixed with the OBJECT (that is what makes a track meaningful) and they are
    spread over the surface rather than clustered.
    """
    rng = np.random.default_rng(seed)
    out: List[Feature] = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        zz = 1.0 - 2.0 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1.0 - zz * zz))
        th = ga * i
        p = (radius * rr * math.cos(th), radius * rr * math.sin(th),
             radius * zz)
        kind = "texture"
        if kinds is not None and i < len(kinds):
            kind = kinds[i]
        out.append(Feature(object_name=obj, point_local=p, kind=kind,
                           uv_local=(rng.random(), rng.random())))
    return tuple(out)


def _boundary_distance(p: Tuple[float, float, float], radius: float) -> float:
    """Unsigned distance from a surface point to the sphere's silhouette
    boundary in the object's own frame, in radians of azimuth.

    Truth only — used to LABEL the experiment (features at varying distance
    from a face boundary), never read by the scorer.
    """
    x, y, z = p
    return abs(math.atan2(y, x))


def scene_resolvable() -> SceneSpec:
    """RESOLVABLE: two identical primitives, one behind the other, so part of
    the far one is occluded; texture elements at varying distance from a
    boundary, addressed in the OBJECT frame."""
    sc = Scene(
        objects=[
            SceneObject(name="front", shape=Sphere(0.30),
                        centre=np.array([-0.16, 0.0, 0.0]),
                        material=Material(albedo=0.85, texture="checker",
                                          texture_freq=6.0,
                                          texture_contrast=0.55,
                                          texture_space="object"),
                        layer=0),
            SceneObject(name="back", shape=Sphere(0.30),
                        centre=np.array([0.16, 0.0, -0.30]),
                        material=Material(albedo=0.85, texture="checker",
                                          texture_freq=6.0,
                                          texture_contrast=0.55,
                                          texture_space="object"),
                        layer=1),
        ],
        width=256, height=256)
    feats = (_texture_features("front", 0.30, 24, seed=1)
             + _texture_features("back", 0.30, 24, seed=2))
    return SceneSpec(name="resolvable_two_spheres", family=RESOLVABLE,
                     scene=sc, features=feats, views=view_steps(),
                     notes="identical primitives; back partially occluded by "
                           "front; texture_space=object on both")


def scene_negative(num_objects: int = 2) -> SceneSpec:
    """NEGATIVE-A / NEGATIVE-B: identical objects at the SAME pose, fully
    separated, no occlusion.

    Intra-object edges only (NEGATIVE-A) cannot tell them apart, so accuracy on
    the swappable subset should be about 0.5.  Allowing cross-object edges
    (NEGATIVE-B) makes the swap reversible, so a win there is legitimate.
    """
    objs = []
    feats: List[Feature] = []
    for i in range(num_objects):
        name = "obj%d" % i
        cx = -0.28 + 0.28 * i
        objs.append(SceneObject(
            name=name, shape=Sphere(0.22),
            centre=np.array([cx, 0.0, 0.0]),
            material=Material(albedo=0.85, texture="checker",
                              texture_freq=6.0, texture_contrast=0.55,
                              texture_space="object"),
            layer=0))
        feats.extend(_texture_features(name, 0.22, 16, seed=10 + i))
    sc = Scene(objects=objs, width=256, height=256)
    return SceneSpec(name="neg_%d_identical" % num_objects, family=NEGATIVE_A,
                     scene=sc, features=tuple(feats), views=view_steps(),
                     notes="identical objects, same pose, no occlusion; "
                           "NEGATIVE-A uses intra-object edges only, "
                           "NEGATIVE-B allows cross-object")


def scene_control() -> SceneSpec:
    """CONTROL: single ordinary objects."""
    sc = Scene(objects=[
        SceneObject(name="solo", shape=Sphere(0.34),
                    centre=np.zeros(3),
                    material=Material(albedo=0.85, texture="checker",
                                      texture_freq=6.0, texture_contrast=0.55,
                                      texture_space="object"),
                    layer=0)], width=256, height=256)
    feats = _texture_features("solo", 0.34, 20, seed=21)
    return SceneSpec(name="control_single_sphere", family=CONTROL, scene=sc,
                     features=feats, views=view_steps(),
                     notes="single object; the ordinary case")


def all_specs() -> Tuple[SceneSpec, ...]:
    a = scene_negative(2)
    b = SceneSpec(name="neg_cross", family=NEGATIVE_B, scene=a.scene,
                  features=a.features, views=a.views,
                  notes="same scene as NEGATIVE-A, evaluated with cross-object "
                        "edges allowed")
    return (scene_resolvable(), a, b, scene_control())
