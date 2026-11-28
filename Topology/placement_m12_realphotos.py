"""MISSION 12 STEP 1 -- the real photographs this mission actually has.

ETH-80 never arrived (the download stalled at 322 MB of a ~4.5 GB tarball and was
killed), so the real-image work runs on REAL PHOTOGRAPHS THAT ARE ALREADY AVAILABLE:
the sample images bundled with scikit-image and scikit-learn, loaded from the
installed packages, converted to the working resolution (256 x 256, uint8 RGB) and
hashed.

The pixels live in `.cache/m12_real_photos/` (a cache, not a redistribution); the
MANIFEST (`Topology/results/placement/m12_real_photos.json`) records, for every
photo, the package and loader it came from, the original array shape, the sha256 of
the resized array, and the licence note. No licence is asserted that the package's
own data documentation does not state: every entry carries the note that it is a
package-bundled sample and points at that documentation, and the images are never
copied into the repository.

    python Topology/placement_m12_realphotos.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
MANIFEST = os.path.join(OUT, "m12_real_photos.json")
CACHE = os.path.join(ROOT, ".cache", "m12_real_photos")
IMG = 256

#: (name, package, loader, kind).  `kind` is the structure class the frame test and
#: the read stratify by -- it is a property of the photograph, not of a result.
PHOTOS: Tuple[Tuple[str, str, str, str], ...] = (
    ("astronaut", "skimage", "data.astronaut", "portrait"),
    ("chelsea", "skimage", "data.chelsea", "animal"),
    ("coffee", "skimage", "data.coffee", "object"),
    ("camera", "skimage", "data.camera", "object"),
    ("clock_motion", "skimage", "data.clock", "object"),
    ("rocket", "skimage", "data.rocket", "object"),
    ("coins", "skimage", "data.coins", "object"),
    ("motorcycle_left", "skimage", "data.stereo_motorcycle", "object"),
    ("motorcycle_right", "skimage", "data.stereo_motorcycle", "object"),
    ("page", "skimage", "data.page", "text"),
    ("brick", "skimage", "data.brick", "texture"),
    ("grass", "skimage", "data.grass", "texture"),
    ("gravel", "skimage", "data.gravel", "texture"),
    ("moon", "skimage", "data.moon", "scene"),
    ("hubble", "skimage", "data.hubble_deep_field", "scene"),
    ("ihc", "skimage", "data.immunohistochemistry", "microscopy"),
    ("retina", "skimage", "data.retina", "microscopy"),
    ("cell", "skimage", "data.cell", "microscopy"),
    ("china", "sklearn", "datasets.load_sample_images", "scene"),
    ("flower", "sklearn", "datasets.load_sample_images", "plant"),
)

#: The six reserved for the FRAME VALIDATION, so the read photos stay untouched by
#: any earlier graph: the validation builds graphs, the read must not have them.
VALIDATION = ("gravel", "grass", "page", "moon", "cell", "retina")


def _to_u8(img: Any) -> np.ndarray:
    """Any sample array -> 256x256 uint8 RGB."""
    from PIL import Image
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
        a = a.astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    if a.shape[2] == 4:
        a = a[..., :3]
    im = Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_all() -> Dict[str, np.ndarray]:
    """name -> 256x256 uint8 RGB, loaded from the installed packages."""
    from skimage import data as skdata
    out: Dict[str, np.ndarray] = {}
    for name, pkg, loader, _kind in PHOTOS:
        if pkg == "skimage":
            fn = getattr(skdata, loader.split(".", 1)[1])
            got = fn()
            if isinstance(got, (list, tuple)):      # the stereo pair
                got = got[0] if name.endswith("left") else got[1]
            arr = got
        else:
            from sklearn.datasets import load_sample_images
            imgs = load_sample_images().images
            arr = imgs[0] if name == "china" else imgs[1]
        out[name] = _to_u8(arr)
    return out


def build() -> Dict[str, Any]:
    os.makedirs(CACHE, exist_ok=True)
    from PIL import Image
    arrays = load_all()
    entries: List[Dict[str, Any]] = []
    for name, pkg, loader, kind in PHOTOS:
        a = arrays[name]
        path = os.path.join(CACHE, name + ".png")
        Image.fromarray(a).save(path)
        entries.append({
            "name": name, "package": pkg, "loader": loader, "kind": kind,
            "shape": [int(x) for x in a.shape],
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "path": os.path.relpath(path, ROOT),
            "licence": ("package-bundled sample image; the licence is stated by that "
                        "package's own data documentation (%s) and is not re-asserted "
                        "here. Used locally for measurement; the pixels are NOT "
                        "committed to the repository." % loader),
            "reserved_for": ("frame_validation" if name in VALIDATION else "read"),
        })
    manifest = {
        "count": len(entries),
        "resolution": [IMG, IMG],
        "note": ("ETH-80 never arrived (download stalled, killed); these are the real "
                 "photographs the mission's real-image steps run on."),
        "validation_set": list(VALIDATION),
        "read_set": [e["name"] for e in entries if e["name"] not in VALIDATION],
        "photos": entries,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


def images(which: str = "all") -> Dict[str, np.ndarray]:
    """Read the cached pixels: 'all', 'validation' or 'read'."""
    if not os.path.exists(MANIFEST):
        build()
    man = json.load(open(MANIFEST))
    keep = {"all": None, "validation": set(man["validation_set"]),
            "read": set(man["read_set"])}[which]
    out = {}
    for e in man["photos"]:
        if keep is not None and e["name"] not in keep:
            continue
        from PIL import Image
        out[e["name"]] = np.asarray(Image.open(os.path.join(ROOT, e["path"])),
                                    dtype=np.uint8)
    return out


def main() -> int:
    man = build()
    print(json.dumps({"count": man["count"], "validation": man["validation_set"],
                      "read": man["read_set"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
