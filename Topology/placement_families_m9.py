"""MISSION 9 STEP 2 -- two families where placement CAN win, and why.

The four DEV-5 families are a periodic checker on two primitives under an
in-plane similarity.  On them the appearance of a node is nearly enough, so the
ceiling is low (N 0.31, oracle 0.40) and there is little left for a relational
term to buy.  Two families are added here, and both are built so that the
question has headroom BY CONSTRUCTION:

IRREGULAR -- aperiodic, non-repeating objects on an irregular layout, with
    REPEATED-MATERIAL twins: two elements carry the same shape and the same
    `noise` material (same frequency, same contrast, same albedo), so their
    nodes genuinely tie on appearance, while the rest of the scene differs.  The
    only thing that can separate the twins is the relation between them.

OUT-OF-PLANE -- the same objects seen under rotations about y and x (5, 10, 20
    degrees).  In-plane steps keep a 2-D similarity of the source frame an exact
    model; a y/x rotation does not, so this family tests the geometry model
    itself.  Tracks are renderer truth: `point_local` is carried through
    `local_to_world` and projected, which is identity at zero pose change by
    construction (asserted in the premise check).

Both are split BY SCENE into FIT and CHECK before anything is fitted.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from perception.synthetic3d import (Box, Cylinder, Material, Scene, SceneObject,
                                    Sphere)
from Topology.placement_scenes import Feature, SceneSpec

SEED_BASE_IRREGULAR = 24000
SEED_BASE_OOP = 26000

#: Views for IRREGULAR: the credited in-plane steps, so the family is comparable
#: to the four DEV-5 families and the in-plane model is not the question.
VIEWS_IRREGULAR: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("i1_rot5", {"rot_deg": 5.0}),
    ("i2_trans", {"rot_deg": 0.0, "shift": (0.04, 0.0)}),
    ("i3_rot15", {"rot_deg": 15.0}),
    ("i4_scale", {"rot_deg": 0.0, "scale": 1.15}),
)

#: Views for OUT-OF-PLANE: y and x rotations at 5, 10, 20 degrees.
VIEWS_OOP: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("o1_y5", {"rot_y_deg": 5.0}),
    ("o2_y10", {"rot_y_deg": 10.0}),
    ("o3_y20", {"rot_y_deg": 20.0}),
    ("o4_x5", {"rot_x_deg": 5.0}),
    ("o5_x10", {"rot_x_deg": 10.0}),
    ("o6_x20", {"rot_x_deg": 20.0}),
)

OOP_SHAPES = ("SPHERE", "BOX", "CYLINDER", "IRREGULAR")


def _noise_mat(rng: np.random.Generator, freq: float, albedo: float,
               contrast: float, direction: Tuple[float, float, float]
               ) -> Material:
    return Material(albedo=albedo, texture="noise", texture_freq=freq,
                    texture_contrast=contrast, texture_space="object",
                    texture_dir=direction)


def _twins(rng: np.random.Generator, kind: str) -> Tuple[Material, Material]:
    """Two materials that are the SAME up to the sampling grid.

    The frequency and contrast are identical and the albedo differs only inside
    the range the renderer can distinguish -- this is the appearance tie the
    family needs, not an accident of the seed.
    """
    freq = float(rng.uniform(7.0, 10.0))
    contrast = float(rng.uniform(0.6, 0.8))
    alb = float(rng.uniform(0.5, 0.65))
    a = _noise_mat(rng, freq, alb, contrast, (1.0, 0.0, 0.0))
    b = _noise_mat(rng, freq, alb + float(rng.uniform(-0.02, 0.02)), contrast,
                   (1.0, 0.0, 0.0))
    return a, b


def _irregular_objects(rng: np.random.Generator) -> List[SceneObject]:
    """Seven aperiodic elements on an irregular layout, TWO twin pairs.

    The first version had one twin pair out of five elements and measured an
    overall oracle-minus-N headroom of +0.0415 against the mission's 0.05 bar:
    the family was too easy, because most nodes were unambiguous.  Four of the
    seven elements now come in two appearance-identical pairs (same shape, same
    noise frequency, same contrast, albedo inside the renderer's own step), so a
    node on one of them can only be told from its twin by the relation between
    them.
    """
    pieces: List[SceneObject] = []
    for pair in range(2):
        twin_a, twin_b = _twins(rng, "twin")
        he = (float(rng.uniform(0.12, 0.17)), float(rng.uniform(0.12, 0.17)),
              float(rng.uniform(0.12, 0.17)))
        cx0 = -0.34 + 0.22 * pair
        for name, mat, cx in (("t%da" % pair, twin_a, cx0),
                              ("t%db" % pair, twin_b, cx0 + 0.46)):
            pieces.append(SceneObject(
                name=name, shape=Box(he), rotation=np.eye(3),
                centre=np.array([cx, float(rng.uniform(-0.03, 0.03)),
                                 float(rng.uniform(-0.04, 0.04))]),
                material=mat))
    # three further elements, each its own frequency and shape, deliberately
    # NOT on a lattice
    for name, kind in (("c", "sphere"), ("d", "cyl"), ("e", "box")):
        freq = float(rng.uniform(5.0, 12.0))
        mat = _noise_mat(rng, freq, float(rng.uniform(0.45, 0.75)),
                         float(rng.uniform(0.5, 0.85)), (1.0, 0.0, 0.0))
        cx = float(rng.uniform(-0.42, 0.42))
        cy = float(rng.uniform(-0.30, 0.30))
        cz = float(rng.uniform(0.0, 0.10))
        if kind == "sphere":
            shape = Sphere(radius=float(rng.uniform(0.10, 0.16)))
        elif kind == "cyl":
            shape = Cylinder(radius=float(rng.uniform(0.08, 0.13)),
                             half_height=float(rng.uniform(0.12, 0.20)))
        else:
            shape = Box(tuple(float(x) for x in rng.uniform(0.08, 0.15, size=3)))
        a = math.radians(float(rng.uniform(0.0, 60.0)))
        rotz = np.array([[math.cos(a), -math.sin(a), 0.0],
                         [math.sin(a), math.cos(a), 0.0],
                         [0.0, 0.0, 1.0]])
        pieces.append(SceneObject(name=name, shape=shape, rotation=rotz,
                                  centre=np.array([cx, cy, cz]), material=mat))
    return pieces


def _base_kwargs(rng: np.random.Generator) -> Dict[str, Any]:
    return dict(light=np.array([float(rng.uniform(-0.6, -0.2)),
                                float(rng.uniform(-0.7, -0.3)),
                                float(rng.uniform(0.6, 0.9))]),
                background=0.02, ambient=0.0, camera_z=10.0,
                pixel_scale=0.0078125, width=256, height=256)


def _oop_scene(shape: str, rng: np.random.Generator) -> Scene:
    if shape == "IRREGULAR":
        return Scene(objects=_irregular_objects(rng), **_base_kwargs(rng))
    # [MISSION 9 / repair] Both objects now carry the SAME noise material.  With
    # different frequencies the pair was merely a hard view, and the measured
    # oracle-minus-N headroom was +0.0296 against the 0.05 bar; with one material
    # the two bodies tie on appearance and only the relation between them can
    # separate their nodes -- which is the premise the family is supposed to test.
    _oop_mat = _noise_mat(rng, float(rng.uniform(7.0, 10.0)),
                          float(rng.uniform(0.5, 0.65)),
                          float(rng.uniform(0.6, 0.8)), (1.0, 0.0, 0.0))
    if shape == "SPHERE":
        objs = [SceneObject(name="a", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3), centre=np.array([-0.20, 0.0, 0.0]),
                            material=_oop_mat),
                SceneObject(name="b", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3), centre=np.array([0.20, 0.0, 0.06]),
                            material=_oop_mat)]
    elif shape == "BOX":
        he = tuple(float(x) for x in rng.uniform(0.15, 0.20, size=3))
        objs = [SceneObject(name="a", shape=Box(he), rotation=np.eye(3),
                            centre=np.array([-0.20, 0.0, 0.0]),
                            material=_oop_mat),
                SceneObject(name="b", shape=Box(he), rotation=np.eye(3),
                            centre=np.array([0.20, 0.0, 0.06]),
                            material=_oop_mat)]
    else:  # CYLINDER
        objs = [SceneObject(name="a", shape=Cylinder(radius=0.18, half_height=0.28),
                            rotation=np.eye(3), centre=np.array([-0.20, 0.0, 0.0]),
                            material=_oop_mat),
                SceneObject(name="b", shape=Cylinder(radius=0.18, half_height=0.28),
                            rotation=np.eye(3), centre=np.array([0.20, 0.0, 0.06]),
                            material=_oop_mat)]
    return Scene(objects=objs, **_base_kwargs(rng))


def _features_for(scene: Scene, rng: np.random.Generator
                  ) -> Tuple[Feature, ...]:
    """A few well-spread object-local points per object, as truth tracks.

    The points are drawn on each object's local unit directions scaled to its own
    size, so they are legitimate surface points under ANY rotation -- which is
    what makes the out-of-plane track derivation (`local_to_world` then project)
    correct by construction rather than by luck.
    """
    feats: List[Feature] = []
    for o in scene.objects:
        if o.name.startswith("_"):
            continue
        for k in range(4):
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            if isinstance(o.shape, Sphere):
                p = d * o.shape.radius
            elif isinstance(o.shape, Box):
                he = np.asarray(o.shape.half_extents, dtype=float)
                p = d * he * 0.85
            else:
                p = np.array([d[0] * o.shape.radius, d[1] * o.shape.radius,
                              d[2] * o.shape.half_height])
            feats.append(Feature(object_name=o.name,
                                 point_local=(float(p[0]), float(p[1]),
                                              float(p[2])),
                                 kind="texture"))
    return tuple(feats)


def all_specs_irregular(per_family: int = 8) -> List[SceneSpec]:
    out: List[SceneSpec] = []
    for i in range(int(per_family)):
        seed = SEED_BASE_IRREGULAR + i
        rng = np.random.default_rng(seed)
        sc = Scene(objects=_irregular_objects(rng), **_base_kwargs(rng))
        out.append(SceneSpec(name="ir_%02d" % i, family="IRREGULAR", scene=sc,
                             features=_features_for(sc, rng),
                             views=VIEWS_IRREGULAR, notes="seed %d" % seed))
    return out


def all_specs_oop(per_family: int = 4) -> List[SceneSpec]:
    """per_family scenes of EACH out-of-plane shape, interleaved."""
    out: List[SceneSpec] = []
    for i in range(int(per_family)):
        for fi, shape in enumerate(OOP_SHAPES):
            seed = SEED_BASE_OOP + 173 * fi + i
            rng = np.random.default_rng(seed)
            sc = _oop_scene(shape, rng)
            out.append(SceneSpec(name="oop_%s_%02d" % (shape.lower(), i),
                                 family="OUT_OF_PLANE",
                                 scene=sc, features=_features_for(sc, rng),
                                 views=VIEWS_OOP,
                                 notes="seed %d shape %s" % (seed, shape)))
    return out


def split_scenes(specs: List[SceneSpec]) -> Tuple[List[SceneSpec], List[SceneSpec]]:
    """FIT / CHECK by scene, stratified: odd rank -> FIT, even -> CHECK."""
    fit: List[SceneSpec] = []
    check: List[SceneSpec] = []
    for s in specs:
        idx = int(s.name.rsplit("_", 1)[-1])
        (check if idx % 2 == 0 else fit).append(s)
    return fit, check


def pose_fn(spec: SceneSpec, view: Dict[str, Any]):
    """The mission-9 posing function: in-plane AND out-of-plane, one path."""
    from Topology.placement_scenes import posed
    pv = {"kind": view.get("kind"),
          "rot_deg": float(view.get("rot_deg", 0.0)),
          "rot_y_deg": float(view.get("rot_y_deg", 0.0)),
          "rot_x_deg": float(view.get("rot_x_deg", 0.0)),
          "shift": tuple(view.get("shift", (0.0, 0.0))),
          "scale": float(view.get("scale", 1.0))}
    return posed(spec.scene, pv["rot_deg"], pv["shift"], pv["scale"],
                 rot_y_deg=pv["rot_y_deg"], rot_x_deg=pv["rot_x_deg"]), pv


# ---------------------------------------------------------------------------
# CONFIRMATION SETS -- scenes that did not exist when the family was developed.
#
# The 12 irregular and 16 out-of-plane scenes above were used to REPAIR the
# families (two twin pairs instead of one; one shared material instead of two
# frequencies) and to plan the read, so their graphs exist and they can never
# be confirmation data.  These are drawn from fresh seed ranges, are named
# differently, and were registered in the pre-registration BEFORE any of their
# graphs was built -- which is what `placement_m9_prereg.py` checks.
# ---------------------------------------------------------------------------
SEED_BASE_IRREGULAR_CONF = 24200
SEED_BASE_OOP_CONF = 26200


def confirmation_specs_irregular(n: int = 6) -> List[SceneSpec]:
    out: List[SceneSpec] = []
    for i in range(int(n)):
        seed = SEED_BASE_IRREGULAR_CONF + i
        rng = np.random.default_rng(seed)
        sc = Scene(objects=_irregular_objects(rng), **_base_kwargs(rng))
        out.append(SceneSpec(name="irc_%02d" % i, family="IRREGULAR",
                             scene=sc, features=_features_for(sc, rng),
                             views=VIEWS_IRREGULAR,
                             notes="confirmation seed %d" % seed))
    return out


def confirmation_specs_oop(n: int = 6) -> List[SceneSpec]:
    out: List[SceneSpec] = []
    for i in range(int(n)):
        for fi, shape in enumerate(OOP_SHAPES):
            if len(out) >= int(n):
                return out
            seed = SEED_BASE_OOP_CONF + 173 * fi + i
            rng = np.random.default_rng(seed)
            sc = _oop_scene(shape, rng)
            out.append(SceneSpec(name="oopc_%s_%02d" % (shape.lower(), i),
                                 family="OUT_OF_PLANE", scene=sc,
                                 features=_features_for(sc, rng),
                                 views=VIEWS_OOP,
                                 notes="confirmation seed %d" % seed))
    return out
