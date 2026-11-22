"""DEV scene set for the placement mission.

DEV is where constants are derived and frozen.  The EVAL set is a SEPARATE,
pre-registered set that is never built during development (see
``placement_eval_spec.py``).  Nothing in this file is read by the EVAL run.

Each scene is a small 3D scene rendered by ``perception.synthetic3d``.  The
families and what they are for:

  RESOLVABLE  different objects, so a correct correspondence exists and is
              identifiable -- the positive test
  NEGATIVE-A  two IDENTICAL objects.  The builder will be asked to keep
              INTRA-object edges only, so a correct score must not be able to
              tell the identity mapping from the swap (a difference is a
              positional leak)
  NEGATIVE-B  the same two identical objects, CROSS-object edges allowed, so the
              swap reverses the cross-object vectors and identity must WIN -- a
              positive control for the mechanism
  CONTROL     one object: ordinary case, placement must not hurt
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from perception.synthetic3d import (Box, Cylinder, Material, Scene, SceneObject,
                                    Sphere)
from Topology.placement_scenes import posed


def _stripes(albedo: float = 0.85, freq: float = 10.0,
             contrast: float = 0.80) -> Material:
    return Material(albedo=albedo, texture="stripes", texture_freq=freq,
                    texture_contrast=contrast, texture_space="object",
                    texture_dir=(1.0, 0.0, 0.0))


def _checker(albedo: float = 0.60, freq: float = 8.0,
             contrast: float = 0.70) -> Material:
    return Material(albedo=albedo, texture="checker", texture_freq=freq,
                    texture_contrast=contrast, texture_space="object",
                    texture_dir=(1.0, 0.0, 0.0))


def _noise(albedo: float = 0.70, freq: float = 12.0,
           contrast: float = 0.60) -> Material:
    return Material(albedo=albedo, texture="noise", texture_freq=freq,
                    texture_contrast=contrast, texture_space="object",
                    texture_dir=(1.0, 0.0, 0.0))


@dataclass(frozen=True)
class SceneSpec:
    name: str
    family: str             # RESOLVABLE / NEGATIVE-A / NEGATIVE-B / CONTROL
    scene: Scene
    notes: str = ""


#: The DEV views.  Every scene gets all of them.  ``posed`` rotates each object
#: about its OWN centre, so these are in-plane rotations, translations and a
#: scale change -- exactly the pose families the mission lists.
DEV_VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("v0_base", {"rot_deg": 0.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("v1_rot10", {"rot_deg": 10.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("v2_rot20", {"rot_deg": 20.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("v3_rot30", {"rot_deg": 30.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("v4_translate", {"rot_deg": 0.0, "shift": (0.06, -0.03), "scale": 1.0}),
    ("v5_scale", {"rot_deg": 0.0, "shift": (0.0, 0.0), "scale": 1.12}),
)


def scene_resolvable() -> SceneSpec:
    """Sphere + box: different shapes, materials and depths."""
    sc = Scene(objects=[
        SceneObject(name="front", shape=Sphere(0.28),
                    centre=np.array([-0.22, 0.0, 0.00]),
                    material=_stripes(), layer=0),
        SceneObject(name="back", shape=Box((0.20, 0.20, 0.20)),
                    centre=np.array([0.30, 0.04, 0.10]),
                    material=_checker(), layer=0),
    ])
    return SceneSpec("dev_resolvable", "RESOLVABLE", sc,
                     "sphere (stripes) + box (checker)")


def _identical_pair() -> Scene:
    """Two IDENTICAL spheres: same shape, same material, same contrast.

    Identical down to the pixel is what makes the swap a real ambiguity, and it
    is what the NEGATIVE-A control needs in order to mean anything.
    """
    return Scene(objects=[
        SceneObject(name="left", shape=Sphere(0.22),
                    centre=np.array([-0.26, 0.0, 0.0]),
                    material=_stripes(), layer=0),
        SceneObject(name="right", shape=Sphere(0.22),
                    centre=np.array([0.26, 0.0, 0.02]),
                    material=_stripes(), layer=0),
    ])


def scene_negative_a() -> SceneSpec:
    return SceneSpec("dev_neg_a", "NEGATIVE-A", _identical_pair(),
                     "two identical spheres; intra-object edges only")


def scene_negative_b() -> SceneSpec:
    return SceneSpec("dev_neg_b", "NEGATIVE-B", _identical_pair(),
                     "the SAME two identical spheres; cross-object edges allowed")


def scene_control() -> SceneSpec:
    return SceneSpec("dev_control", "CONTROL", Scene(objects=[
        SceneObject(name="solo", shape=Sphere(0.32),
                    centre=np.array([0.0, 0.0, 0.0]),
                    material=_noise(), layer=0),
    ]), "single sphere (noise texture)")


def all_specs() -> List[SceneSpec]:
    return [scene_resolvable(), scene_negative_a(), scene_negative_b(),
            scene_control()]


def render_pose(spec: SceneSpec, view: Dict[str, Any]) -> Any:
    sc = posed(spec.scene, float(view["rot_deg"]),
               tuple(view["shift"]), float(view["scale"]))
    return sc, sc.render()
