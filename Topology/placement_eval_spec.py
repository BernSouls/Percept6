"""PRE-REGISTERED EVAL set for the placement mission.

This file is committed BEFORE any EVAL graph is built or read.  Everything the
final run depends on is fixed here: the scenes, the poses, the materials, the
frozen constants, the ambiguity rule, the arms, the success criteria and the
design floor.  Nothing in it was chosen after looking at an EVAL number,
because no EVAL number existed when it was written.

The DEV set (``placement_dev.py``) is disjoint from this one: different shapes,
different materials, different texture frequencies, different object sizes,
different separations, different poses and different view magnitudes.  DEV is
where constants were derived; this is where they are spent, exactly once.

WHY ONE TOLERANCE FOR EVERY EVAL VIEW
-------------------------------------
Tolerance was derived on DEV as the q90 of the TRUE residual per view step
(``devfit_tolerance.json``): 1.4946 / 1.3915 / 1.7009 / 1.0645 / 1.5406.  The
EVAL views do not coincide with the DEV steps, so instead of interpolating
(five chances to flatter one arm) the pre-registered value is the LARGEST DEV
q90, **1.7009 scale units**, applied to every EVAL view.  A larger tolerance is
strictly the conservative side for the placement arms: it admits more candidate
matches as "supported", so it cannot manufacture precision for P.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from perception.synthetic3d import (Box, Cylinder, Material, Scene, SceneObject,
                                    Sphere)
from Topology.placement_dev import SceneSpec

# --------------------------------------------------------------------------
#: The EVAL poses.  Same KINDS of transformation as DEV (in-plane rotation,
#: translation, uniform scale, and one mixed case) at DIFFERENT magnitudes, so
#: the run tests the frozen rule rather than the DEV grid.
EVAL_VIEWS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("e1_rot12", {"rot_deg": 12.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("e2_rot24", {"rot_deg": 24.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("e3_rot33", {"rot_deg": 33.0, "shift": (0.0, 0.0), "scale": 1.0}),
    ("e4_translate", {"rot_deg": 0.0, "shift": (-0.07, 0.05), "scale": 1.0}),
    ("e5_scale", {"rot_deg": 0.0, "shift": (0.0, 0.0), "scale": 0.90}),
    ("e6_rot15_scale", {"rot_deg": 15.0, "shift": (0.03, 0.0), "scale": 1.06}),
)

#: Pre-registered tolerance: the largest DEV q90, for every EVAL view.
FROZEN_TOLERANCE = 1.7009

#: Everything else that must not move between the DEV fit and the EVAL read.
FROZEN: Dict[str, Any] = {
    "tolerance_all_views": FROZEN_TOLERANCE,
    "frame_coherence_min": 0.50,
    "target_k": 16,
    "target_radius_scales": 4.0,
    "placement_weight": 0.25,       # modifier strength, one pre-stated value
    "edge_beta": 0.5,               # R / R-fixed weight on the edge term
    "ambiguous_margin": 0.05,       # near-tie rule for the ambiguity label
    "track_radius": 8.0,
    "swap_radius_world": 0.06,
    "bootstrap_reps": 2000,
    "cluster_key": "scene|view",
    "mde_multiplier": 2.8,
    "seed": 20261118,
}


# --------------------------------------------------------------------------
# materials: frequencies and contrasts deliberately unlike DEV (10 / 8, 0.80)
def _stripes_eval() -> Material:
    return Material(albedo=0.78, texture="stripes", texture_freq=12.0,
                    texture_contrast=0.72, texture_space="object",
                    texture_dir=(1.0, 0.0, 0.0))


def _checker_eval() -> Material:
    return Material(albedo=0.66, texture="checker", texture_freq=6.0,
                    texture_contrast=0.68, texture_space="object")


def _noise_eval() -> Material:
    return Material(albedo=0.70, texture="noise", texture_freq=9.0,
                    texture_contrast=0.55, texture_space="object")


def eval_resolvable() -> SceneSpec:
    """Cylinder (stripes) + box (checker): different shapes and depths.

    Different from the DEV resolvable scene (sphere + box, other sizes).
    """
    sc = Scene(objects=[
        SceneObject(name="cyl", shape=Cylinder(radius=0.20, half_height=0.30),
                    centre=np.array([-0.25, 0.03, 0.02]),
                    material=_stripes_eval(), layer=0),
        SceneObject(name="cube", shape=Box((0.17, 0.17, 0.17)),
                    centre=np.array([0.28, -0.05, 0.12]),
                    material=_checker_eval(), layer=0),
    ])
    return SceneSpec("eval_resolvable", "RESOLVABLE", sc,
                     "cylinder (stripes) + box (checker)")


def _eval_pair() -> Scene:
    """Two IDENTICAL spheres, sizes and separation unlike the DEV pair."""
    return Scene(objects=[
        SceneObject(name="left", shape=Sphere(0.19),
                    centre=np.array([-0.24, 0.0, 0.0]),
                    material=_stripes_eval(), layer=0),
        SceneObject(name="right", shape=Sphere(0.19),
                    centre=np.array([0.24, 0.0, 0.03]),
                    material=_stripes_eval(), layer=0),
    ])


def eval_negative_a() -> SceneSpec:
    return SceneSpec("eval_neg_a", "NEGATIVE-A", _eval_pair(),
                     "two identical spheres; intra-object edges only")


def eval_negative_b() -> SceneSpec:
    return SceneSpec("eval_neg_b", "NEGATIVE-B", _eval_pair(),
                     "the SAME two identical spheres; cross-object edges allowed")


def eval_control() -> SceneSpec:
    """One box with a striped material: placement must not hurt a solo object."""
    return SceneSpec("eval_control", "CONTROL", Scene(objects=[
        SceneObject(name="solo", shape=Box((0.21, 0.21, 0.21)),
                    centre=np.array([0.02, 0.01, 0.0]),
                    material=_noise_eval(), layer=0),
    ]), "single box (noise texture)")


def eval_specs() -> List[SceneSpec]:
    return [eval_resolvable(), eval_negative_a(), eval_negative_b(),
            eval_control()]


def load_frozen() -> Dict[str, Any]:
    """The frozen table in the shape ``placement_arms2`` expects."""
    return {
        "tolerance": {label: {"tolerance": FROZEN_TOLERANCE,
                              "source": "max DEV q90 (pre-registered)"}
                      for label, _v in EVAL_VIEWS},
        "frozen": FROZEN,
    }
