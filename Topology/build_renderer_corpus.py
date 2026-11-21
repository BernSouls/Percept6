"""Reconstruct the 15-scene ``renderer-truth-v1`` probe corpus (harness utility).

The frozen ruler's *code* was shipped (``Topology/step_a_ruler.py`` and the
other Topology/ scripts) but its input corpus ``data/renderer-truth-v1`` was
not.  This script regenerates it FROM THE REPOSITORY'S OWN RENDERER
(``perception/synthetic3d.py``), deterministically, at the 192x192 size the
ruler hard-codes, with a fixed seed and no random input.  Scene names include
the three gate-log probes named in the fix brief (single_box, texture_box,
disk_occlusion_haze) and the three STEP H defaults.

This is a RECONSTRUCTION, not the user's frozen corpus: the frozen artifact
``results/step_a_ruler.json`` (sha256 15c2b7c9...) is ABSENT from this
checkout, so its hash cannot be reproduced.  The ruler DEFINITIONS are used
unchanged; every measurement in this session is labelled with the rebuilt
artifact's own hash.  Before/after numbers are always on this same fixed
corpus, so the comparison is valid even though the absolute values differ
from the user's historical ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TREE = Path(__file__).resolve().parents[1]
OUT = TREE / "data" / "renderer-truth-v1"
W = H = 192

# The harness is a directory of standalone scripts; make the product library
# importable when this file is run as ``python Topology/build_renderer_corpus.py``.
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from perception.synthetic3d import (  # noqa: E402
    Box, Cylinder, Material, Scene, SceneObject, Sphere,
    rotation_xyz, single_object_scene, occlusion_scene,
)


def _obj(name, shape, centre, rx=0.0, ry=0.0, rz=0.0, material=None,
         layer=0):
    return SceneObject(
        name=name, shape=shape, rotation=rotation_xyz(rx, ry, rz),
        centre=np.asarray(centre, dtype=np.float64),
        material=material or Material(), omega=np.zeros(3),
        velocity=np.zeros(3), layer=int(layer))


def _scene(objects, *, haze_beta=0.0, haze_airlight=0.0,
           blur_sigma=0.0, noise_sigma=0.0, noise_seed=0,
           extra_lights=()):
    return Scene(objects=objects, light=np.array([-0.4, -0.5, 0.75]) /
                 np.linalg.norm([-0.4, -0.5, 0.75]),
                 background=0.02, camera_z=10.0,
                 width=W, height=H, extra_lights=tuple(extra_lights),
                 blur_sigma=blur_sigma, noise_sigma=noise_sigma,
                 noise_seed=noise_seed, haze_beta=haze_beta,
                 haze_airlight=haze_airlight)


PLAIN = Material(albedo=0.85)
NOISE = Material(albedo=0.85, texture="noise", texture_freq=14.0,
                 texture_contrast=0.75)
STRIPES = Material(albedo=0.85, texture="stripes", texture_freq=7.0,
                   texture_contrast=0.7)
CHECKER = Material(albedo=0.85, texture="checker", texture_freq=9.0,
                   texture_contrast=0.65)
DARK = Material(albedo=0.55)


def _box(centre=(0.0, 0.0, 0.0), pose=(0.0, 0.0, 0.0), mat=PLAIN, layer=0,
         half=(0.42, 0.34, 0.30)):
    return _obj("box", Box(half_extents=half), centre, *pose, material=mat,
                layer=layer)


def _sphere(centre=(0.0, 0.0, 0.0), radius=0.35, mat=PLAIN, layer=0):
    return _obj("sphere", Sphere(radius=radius), centre, material=mat,
                layer=layer)


def _cyl(centre=(0.0, 0.0, 0.0), pose=(0.0, 0.0, 0.0), mat=PLAIN, layer=0):
    return _obj("cylinder", Cylinder(radius=0.30, half_height=0.55), centre,
                *pose, material=mat, layer=layer)


#: name -> (split, scene factory, description).  10 CAL + 5 TEST = 15 scenes.
SCENES = {
    "single_box":            ("CAL", lambda: _scene([_box()]),
                              "one box, flat background, axis aligned"),
    "single_cylinder":       ("CAL", lambda: _scene([_cyl()]),
                              "one cylinder, flat background"),
    "single_sphere":         ("CAL", lambda: _scene([_sphere()]),
                              "one sphere, flat background"),
    "texture_box":           ("CAL", lambda: _scene([_box(mat=NOISE)]),
                              "noise-textured box (heavy texture)"),
    "texture_sphere":        ("CAL", lambda: _scene([_sphere(mat=STRIPES)]),
                              "striped sphere"),
    "box_occlusion":         ("CAL", lambda: _scene([
                                  _sphere((-0.10, -0.05, 0.55), radius=0.32,
                                          layer=0),
                                  _box((0.10, 0.05, 0.0), (0.0, 0.2, 0.1),
                                       layer=1)]),
                              "sphere occludes a rotated box"),
    "sphere_over_box":       ("CAL", lambda: _scene([
                                  _sphere((0.02, 0.08, 0.50), radius=0.30,
                                          layer=0),
                                  _box((0.0, -0.06, 0.0), layer=1)]),
                              "sphere in front of box"),
    "box_over_box":          ("CAL", lambda: _scene([
                                  _box((-0.16, 0.02, 0.45), (0.0, 0.0, 0.25),
                                       layer=0),
                                  _box((0.16, -0.02, 0.0), (0.0, 0.1, 0.0),
                                       layer=1)]),
                              "two axis-aligned boxes, one in front"),
    "disk_occlusion_haze":   ("CAL", lambda: _scene([
                                  _sphere((-0.10, -0.05, 0.55), radius=0.32,
                                          layer=0),
                                  _box((0.10, 0.05, 0.0), (0.0, 0.2, 0.1),
                                       layer=1)],
                                  haze_beta=0.35, haze_airlight=0.18,
                                  blur_sigma=0.35),
                              "occlusion under distance haze + slight blur"),
    "f_box_b":               ("CAL", lambda: _scene([
                                  _box((0.0, 0.0, 0.0), (0.15, 0.30, 0.0))]),
                              "rotated box (tangent truth check)"),
    "f_haze_a":              ("TEST", lambda: _scene([
                                  _cyl((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))],
                                  haze_beta=0.5, haze_airlight=0.22),
                              "hazed cylinder"),
    "tv_same_scene":         ("TEST", lambda: _scene([
                                  _box((-0.25, 0.0, 0.0), mat=CHECKER, layer=0),
                                  _box((0.25, 0.0, 0.0), mat=CHECKER, layer=1)]),
                              "two textured boxes, same material"),
    "no_texture2":           ("TEST", lambda: _scene([
                                  _sphere((-0.25, 0.0, 0.0), radius=0.28,
                                          layer=0),
                                  _box((0.25, 0.0, 0.0), mat=DARK, layer=1)]),
                              "two untextured objects"),
    "texture2_only":         ("TEST", lambda: _scene([
                                  _sphere((-0.25, 0.0, 0.0), radius=0.28,
                                          mat=NOISE, layer=0),
                                  _box((0.25, 0.0, 0.0), mat=STRIPES, layer=1)]),
                              "two differently textured objects"),
    "thin_bar":              ("TEST", lambda: _scene([
                                  _box((0.0, 0.0, 0.0),
                                       half=(0.46, 0.055, 0.18))]),
                              "thin elongated box (bar)"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (split, factory, desc) in SCENES.items():
        truth = factory().render(W, H, supersample=2)
        image = truth.image_u8
        if image.shape != (H, W, 3) or image.dtype != np.uint8:
            raise RuntimeError("%s: bad render %r" % (name, image.shape))
        Image.fromarray(image, "RGB").save(out / (name + ".png"))
        entries.append({
            "name": name, "split": split, "description": desc,
            "width": W, "height": H,
            "objects": list(truth.object_names),
            "mask_pixels": int(truth.mask.sum()),
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        })
        print("%-22s %-4s mask %6d  mean %6.2f" %
              (name, split, truth.mask.sum(), image.mean()), flush=True)
    doc = {
        "corpus": "renderer-truth-v1 (RECONSTRUCTED with perception/synthetic3d.py)",
        "protocol": "192x192 uint8 RGB, fixed deterministic scenes, no random input; "
                    "probe shifts applied by np.roll(image,(0,shift),axis=(0,1))",
        "reason": "the frozen corpus was not shipped with this export; the ruler CODE "
                  "and all measurement definitions are used unchanged",
        "frozen_ruler_json": "ABSENT (results/step_a_ruler.json not in this checkout)",
        "frozen_ruler_sha256": "15c2b7c9... (reported by the brief; not verifiable here)",
        "scenes": entries,
        "counts": {"scenes": len(entries),
                   "CAL": sum(1 for e in entries if e["split"] == "CAL"),
                   "TEST": sum(1 for e in entries if e["split"] == "TEST")},
    }
    (out / "truth.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    print("written", out / "truth.json", "scenes", len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
