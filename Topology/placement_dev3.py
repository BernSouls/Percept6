"""DEV-3: a DEV set sized in SCENES, split FIT/CHECK by scene.

Why a new DEV set
-----------------
EVAL v1 had four scenes.  Stage 0 showed that with four scenes the exact
scene-level p cannot go below 0.125, so nothing could be resolved however large
the effect.  The unit of evidence is the SCENE, and scenes must therefore be
the thing that is counted.

Sizing
------
Stage 0 scene-level differences for P-N:  +0.050, +0.090, +0.090, +0.010
(sd 0.038).  For a scene-clustered MDE of 0.03 at MDE = 2.8*SE:
SE = 0.03/2.8 = 0.0107 -> n = (0.038/0.0107)^2 = 12.6 scenes per arm.
This file builds 20 CHECK scenes (and 20 FIT), i.e. MDE ~= 0.024 once the
envelope is a little wider than four scenes.  Planned effect 0.03 is therefore
inside the design, with margin.

Design
------
* 40 scenes: 20 FIT + 20 CHECK, split before any fitting.
* Families in every half: RESOLVABLE 10, NEGATIVE-A 5, CONTROL 5.
* 3 views per scene (few views per scene -- views of a scene are correlated).
* Every scene is a fresh SEED: random shapes, materials, texture frequencies,
  object sizes, separations and depths.  Poses are drawn from a fixed envelope
  (rotation 8-30 deg, translation up to 0.07, scale 0.90-1.12) so that EVAL-2
  can be drawn from the same envelope.
* Nothing here is read by an EVAL set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from perception.synthetic3d import (Box, Cylinder, Material, Scene, SceneObject,
                                    Sphere)
from Topology.placement_dev import SceneSpec

#: Pose envelope.  EVAL-2 must stay inside this.
ROT_RANGE = (8.0, 30.0)
SHIFT_MAX = 0.07
SCALE_RANGE = (0.90, 1.12)

VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("w1_rot", {"kind": "rot"}),
    ("w2_translate", {"kind": "translate"}),
    ("w3_scale", {"kind": "scale"}),
)

FAMILIES = ("RESOLVABLE", "NEGATIVE-A", "CONTROL")
PER_FAMILY = {"RESOLVABLE": 20, "NEGATIVE-A": 10, "CONTROL": 10}


def _material(rng: np.random.Generator, periodic: bool = True) -> Material:
    kind = "stripes" if periodic else "noise"
    return Material(
        albedo=float(rng.uniform(0.55, 0.88)),
        texture=kind,
        texture_freq=float(rng.uniform(7.0, 13.0)),
        texture_contrast=float(rng.uniform(0.55, 0.85)),
        texture_space="object",
        texture_dir=(1.0, 0.0, 0.0),
    )


def _checker(rng: np.random.Generator) -> Material:
    return Material(
        albedo=float(rng.uniform(0.55, 0.80)),
        texture="checker",
        texture_freq=float(rng.uniform(5.0, 9.0)),
        texture_contrast=float(rng.uniform(0.55, 0.80)),
        texture_space="object",
    )


def _shape(rng: np.random.Generator) -> Any:
    kind = rng.choice(["sphere", "box", "cyl"])
    if kind == "sphere":
        return Sphere(radius=float(rng.uniform(0.17, 0.30)))
    if kind == "box":
        h = float(rng.uniform(0.13, 0.22))
        return Box((h, h * float(rng.uniform(0.8, 1.2)), h))
    return Cylinder(radius=float(rng.uniform(0.15, 0.24)),
                    half_height=float(rng.uniform(0.20, 0.32)))


def _resolvable(rng: np.random.Generator) -> Scene:
    a = _shape(rng)
    b = _shape(rng)
    sep = float(rng.uniform(0.45, 0.60))
    return Scene(objects=[
        SceneObject(name="a", shape=a,
                    centre=np.array([-sep / 2, float(rng.uniform(-0.03, 0.03)),
                                     float(rng.uniform(-0.02, 0.04))]),
                    material=_material(rng), layer=0),
        SceneObject(name="b", shape=b,
                    centre=np.array([sep / 2, float(rng.uniform(-0.03, 0.03)),
                                     float(rng.uniform(0.0, 0.12))]),
                    material=_checker(rng), layer=0),
    ])


def _identical_pair(rng: np.random.Generator) -> Scene:
    """Two copies of ONE shape+material, so the swap is a genuine ambiguity."""
    r = float(rng.uniform(0.17, 0.24))
    shape = Sphere(radius=r)
    mat = _material(rng)
    sep = float(rng.uniform(0.42, 0.54))
    return Scene(objects=[
        SceneObject(name="left", shape=shape,
                    centre=np.array([-sep / 2, 0.0, 0.0]),
                    material=mat, layer=0),
        SceneObject(name="right", shape=shape,
                    centre=np.array([sep / 2, 0.0, float(rng.uniform(0.0, 0.04))]),
                    material=mat, layer=0),
    ])


def _control(rng: np.random.Generator) -> Scene:
    return Scene(objects=[
        SceneObject(name="solo", shape=_shape(rng),
                    centre=np.array([float(rng.uniform(-0.03, 0.03)),
                                     float(rng.uniform(-0.03, 0.03)), 0.0]),
                    material=_material(rng, periodic=False), layer=0),
    ])


_MAKERS = {"RESOLVABLE": _resolvable, "NEGATIVE-A": _identical_pair,
           "CONTROL": _control}


def all_specs() -> List[SceneSpec]:
    """40 scenes, seeds fixed, families interleaved so the split is stratified."""
    specs: List[SceneSpec] = []
    for fam in FAMILIES:
        n = PER_FAMILY[fam]
        for i in range(n):
            seed = 7000 + 97 * FAMILIES.index(fam) + i
            rng = np.random.default_rng(seed)
            specs.append(SceneSpec(
                "d3_%s_%02d" % (fam.lower().replace("-", ""), i),
                fam, _MAKERS[fam](rng),
                "seed %d" % seed))
    return specs


def view_pose(spec: SceneSpec, view: Dict[str, Any]) -> Dict[str, Any]:
    """A deterministic pose per (scene, view) inside the pre-registered envelope.

    Derived from a hash of the scene name so it is reproducible and independent
    of iteration order.
    """
    import hashlib
    h = int(hashlib.sha256((spec.name + "|" + view["kind"]).encode()).hexdigest()
            [:8], 16)
    rng = np.random.default_rng(h)
    kind = view["kind"]
    if kind == "rot":
        rot = float(rng.uniform(*ROT_RANGE))
        return {"rot_deg": rot, "shift": (0.0, 0.0), "scale": 1.0}
    if kind == "translate":
        sh = rng.uniform(-SHIFT_MAX, SHIFT_MAX, 2)
        return {"rot_deg": float(rng.uniform(0.0, 8.0)),
                "shift": (float(sh[0]), float(sh[1])), "scale": 1.0}
    return {"rot_deg": float(rng.uniform(0.0, 8.0)), "shift": (0.0, 0.0),
            "scale": float(rng.uniform(*SCALE_RANGE))}


def split(specs: List[SceneSpec]) -> Tuple[List[SceneSpec], List[SceneSpec]]:
    """FIT / CHECK split BY SCENE, stratified by family.

    Odd index -> CHECK, even -> FIT, so every family lands on both sides.
    """
    fit: List[SceneSpec] = []
    check: List[SceneSpec] = []
    for fam in FAMILIES:
        fam_specs = [s for s in specs if s.family == fam]
        for i, s in enumerate(fam_specs):
            (check if i % 2 else fit).append(s)
    return fit, check
