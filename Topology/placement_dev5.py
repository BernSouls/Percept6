"""MISSION 6 STEP 4 -- the DEV-5 scene families.

The DEV scenes (Topology/placement_dev3.py) are all Spheres with checker
texture: no vertices, no straight edges, no partial occlusion, and no cylinder.
A keypoint layer cannot be judged on scenes that contain none of the features it
detects, so DEV-5 adds three families and keeps the spheres as a control:

    SPHERE     the old family, unchanged in construction, new seeds
    BOX        a cuboid: real vertices, three visible faces, straight edges
    CYLINDER   a cylinder: two silhouette lines, curved caps, no corners
    OCCLUSION  a sphere pair taken from SPHERE with `posed(occluder=True)`,
               so part of the front object is hidden and the scene's keypoints
               are partly undetectable BY CONSTRUCTION

Every family is built from a seed, so the split is reproducible.  The split is
by SCENE and stratified by family and pose step, before anything is fitted.

Nothing here changes the front end.  This module only proposes scenes; a caller
builds them with the same `render()` -> `get_or_build` path as DEV-3.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Topology.placement_dev3 import SceneSpec  # type: ignore
from Topology.placement_scenes import (Box, Cylinder, Material, Scene,
                                       SceneObject, Sphere, posed)

FAMILIES = ("SPHERE", "BOX", "CYLINDER", "OCCLUSION")

#: Pose steps for DEV-5.  The first two are the gate's range (<= 5 deg and a
#: small translation); the third is a larger step, reported separately because
#: the gate's localisation bar only applies inside 5 deg.
VIEWS5: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("v1_rot5", {"kind": "rot", "rot_deg": 5.0}),
    ("v2_trans", {"kind": "translate", "shift": (0.04, 0.0)}),
    ("v3_rot15", {"kind": "rot", "rot_deg": 15.0}),
    ("v4_scale", {"kind": "scale", "scale": 1.15}),
)

_SEED_BASE = 9100


def _mat(rng: np.random.Generator, texture: str = "checker") -> Material:
    return Material(albedo=float(rng.uniform(0.45, 0.7)),
                    texture=texture,
                    texture_freq=float(rng.uniform(6.0, 11.0)),
                    texture_contrast=float(rng.uniform(0.6, 0.8)),
                    texture_space="object",
                    texture_dir=(1.0, 0.0, 0.0))


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _scene_for(family: str, rng: np.random.Generator) -> Scene:
    light = np.array([float(rng.uniform(-0.6, -0.2)),
                      float(rng.uniform(-0.7, -0.3)),
                      float(rng.uniform(0.6, 0.9))])
    base = dict(light=light, background=0.02, ambient=0.0, camera_z=10.0,
                pixel_scale=0.0078125, width=256, height=256)
    if family == "SPHERE":
        objs = [SceneObject(name="a", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3),
                            centre=np.array([-0.27, 0.0, 0.0]),
                            material=_mat(rng)),
                SceneObject(name="b", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3),
                            centre=np.array([0.27, 0.0, 0.09]),
                            material=_mat(rng))]
    elif family == "BOX":
        he = tuple(float(x) for x in rng.uniform(0.15, 0.20, size=3))
        objs = [SceneObject(name="a", shape=Box(he),
                            rotation=_rot_z(float(rng.uniform(0.0, 30.0))),
                            centre=np.array([-0.27, 0.0, 0.0]),
                            material=_mat(rng)),
                SceneObject(name="b", shape=Box(he),
                            rotation=np.eye(3),
                            centre=np.array([0.27, 0.0, 0.09]),
                            material=_mat(rng))]
    elif family == "CYLINDER":
        objs = [SceneObject(name="a", shape=Cylinder(radius=float(rng.uniform(0.16, 0.22)),
                                                     half_height=float(rng.uniform(0.24, 0.34))),
                            rotation=_rot_z(float(rng.uniform(0.0, 30.0))),
                            centre=np.array([-0.27, 0.0, 0.0]),
                            material=_mat(rng)),
                SceneObject(name="b", shape=Cylinder(radius=float(rng.uniform(0.16, 0.22)),
                                                     half_height=float(rng.uniform(0.24, 0.34))),
                            rotation=np.eye(3),
                            centre=np.array([0.27, 0.0, 0.09]),
                            material=_mat(rng))]
    else:  # OCCLUSION -- a sphere pair; the occluder is added by posed()
        objs = [SceneObject(name="a", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3),
                            centre=np.array([-0.24, 0.0, 0.0]),
                            material=_mat(rng)),
                SceneObject(name="b", shape=Sphere(radius=float(rng.uniform(0.20, 0.26))),
                            rotation=np.eye(3),
                            centre=np.array([0.24, 0.0, 0.05]),
                            material=_mat(rng))]
    return Scene(objects=objs, **base)


def all_specs5(per_family: int = 6) -> List[SceneSpec]:
    """per_family scenes of each family, seeds fixed, families interleaved."""
    out: List[SceneSpec] = []
    for i in range(per_family):
        for fi, fam in enumerate(FAMILIES):
            seed = _SEED_BASE + 173 * fi + i
            rng = np.random.default_rng(seed)
            sc = _scene_for(fam, rng)
            out.append(SceneSpec(
                name="d5_%s_%02d" % (fam.lower(), i),
                family=fam,
                scene=sc,
                notes="seed %d" % seed,
            ))
    return out


def split5(specs: List[SceneSpec]) -> Tuple[List[SceneSpec], List[SceneSpec]]:
    """FIT / CHECK split BY SCENE, stratified: odd index -> FIT, even -> CHECK.

    The index is the scene's position WITHIN its family, so every family lands
    on both sides at every pose step, which is what "stratified by family and
    pose step" means for a split that happens before any fitting.
    """
    fit: List[SceneSpec] = []
    check: List[SceneSpec] = []
    for s in specs:
        idx = int(s.name.rsplit("_", 1)[-1])
        (check if idx % 2 == 0 else fit).append(s)
    return fit, check


def posed5(spec: SceneSpec, view: Dict[str, Any]) -> Scene:
    """The posed scene for one DEV-5 view step."""
    kind = view.get("kind")
    occl = spec.family == "OCCLUSION"
    if kind == "rot":
        return posed(spec.scene, float(view.get("rot_deg", 5.0)),
                     occluder=occl)
    if kind == "translate":
        return posed(spec.scene, 0.0, shift=tuple(view.get("shift", (0.04, 0.0))),
                     occluder=occl)
    if kind == "scale":
        return posed(spec.scene, 0.0, scale=float(view.get("scale", 1.15)),
                     occluder=occl)
    return posed(spec.scene, occluder=occl)
