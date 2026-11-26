"""MISSION 12 STEP 5 -- the pre-registration of the real-photo correspondence read.

Written BEFORE any graph exists for the read photographs: the script refuses to run
twice and it checks the graph cache for every (photo, transform) it will use.  If
any of them is already there, the read would not be fresh and the script refuses.

The question: under an EXACT KNOWN TRANSFORM, does the deployed placement matcher
correspond REAL PHOTOGRAPH nodes better than the shipped matcher -- on the nodes
where appearance alone is a near-tie (the frozen rule), and without harming the ones
where it is not?

    python Topology/placement_m12_prereg_real.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
PATH = os.path.join(OUT, "m12_prereg_real.json")

#: The transforms.  Every one of them is EXACT and known, applied about the image
#: centre with a fixed convention (see the manifest for the sign of the rotation).
TRANSFORMS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("rot05", {"rot_deg": 5.0}),
    ("rot15", {"rot_deg": 15.0}),
    ("scale115", {"scale": 1.15}),
    ("trans8", {"shift": (8.0, 0.0)}),
)

#: The frozen near-tie rule (M10's, unchanged):
#: top similarity >= 0.99 AND margin to the runner-up <= 0.05.
NEAR_TIE = {"top_sim": 0.99, "margin_max": 0.05}

#: Planning numbers for the MDE, stated before the bank exists:
#:   - a paired per-node difference is 0/ 1 with SD <= 0.5,
#:   - the bank is clustered by PHOTO (the transform does not create a new
#:     photograph), so the design effect is taken as 2.0 (a conservative allowance
#:     for within-photo correlation; it is stated, not estimated),
#:   - MDE = 2.8 * SD * sqrt(design_effect) / sqrt(n_bank).
PLANNING_SD = 0.5
DESIGN_EFFECT = 2.0
MDE_FACTOR = 2.8


def main() -> int:
    if os.path.exists(PATH):
        print("REFUSING: %s exists -- one registration, one read" % PATH)
        return 2
    import guard
    guard.activate()
    import numpy as np
    from Topology import placement_m12_realphotos as RP
    from Topology.placement_cache import CACHE_DIR, key_for

    man = json.load(open(RP.MANIFEST))
    read_set = list(man["read_set"])
    if len(read_set) < 10:
        print("REFUSING: only %d read photos" % len(read_set))
        return 3

    def _transform(img: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
        from scipy.ndimage import rotate, zoom, shift as nd_shift
        a = img.astype(np.float64)
        if "rot_deg" in spec:
            a = rotate(a, float(spec["rot_deg"]), reshape=False, order=1,
                       mode="nearest")
        if "scale" in spec:
            s = float(spec["scale"])
            a = zoom(a, (s, s, 1.0), order=1)
            h, w = a.shape[:2]
            cy, cx = h / 2.0, w / 2.0
            y0, x0 = int(round(cy - img.shape[0] / 2.0)), int(round(cx - img.shape[1] / 2.0))
            a = a[max(0, y0):y0 + img.shape[0], max(0, x0):x0 + img.shape[1]]
            a = np.pad(a, ((0, img.shape[0] - a.shape[0]),
                           (0, img.shape[1] - a.shape[1]), (0, 0)), mode="edge")
        if "shift" in spec:
            a = nd_shift(a, (float(spec["shift"][1]), float(spec["shift"][0]), 0.0),
                         order=1, mode="nearest")
        return np.clip(a, 0, 255).astype(np.uint8)

    imgs = RP.images("read")
    cached = 0
    for name in read_set:
        for lab, spec in TRANSFORMS:
            q = _transform(imgs[name], spec)
            k = key_for(q)
            if os.path.exists(os.path.join(CACHE_DIR, k + ".pkl")):
                cached += 1
            k0 = key_for(imgs[name])
            if os.path.exists(os.path.join(CACHE_DIR, k0 + ".pkl")):
                cached += 1

    with open(os.path.join(ROOT, "matching", "placement_score_params.json")) as fh:
        params = json.load(fh)
    w = float(params["w"])
    n_queries = len(read_set) * len(TRANSFORMS)
    reg: Dict[str, Any] = {
        "mission": "12 step 5 -- real-photo correspondence under exact known transforms",
        "photos": {"source": "package-bundled sample images (scikit-image, scikit-learn)",
                   "manifest": os.path.relpath(RP.MANIFEST, ROOT),
                   "n": len(read_set), "names": read_set,
                   "validation_photos_reserved": man["validation_set"],
                   "note": ("the six validation photos are NEVER used in this read; "
                            "they are what the frame validation is allowed to build")},
        "transforms": [{"label": lab, "spec": spec} for lab, spec in TRANSFORMS],
        "queries": {"n": n_queries, "rule": "one query per (photo, transform)"},
        "gallery": {"rule": "the photo's own original image; one candidate per query",
                    "correspondence": "the exact transform, applied to the node anchors"},
        "arms": {
            "N": "shipped: the aligner on node similarity alone (placement off)",
            "P": "deployed placement matcher (placement on), E on the STEP-2 scale",
        },
        "frozen": {
            "w": w, "w_source": "matching/placement_score_params.json (fitted before M12)",
            "near_tie_rule": NEAR_TIE,
            "e_reading": "PLACEMENT_E_NORMALISED (mean over judged slots, refused below "
                         "the minimum)",
            "frame_provider": ("the shipped field provider. If the Step-2 FIT decision "
                               "adopts the keypoint provider, this field is AMENDED, "
                               "with a dated note, before the first read graph exists "
                               "-- the read photos are untouched by the validation"),
            "target_k": 16, "edge_beta": 0.5,
        },
        "primary": {
            "bank": "near-tie nodes (frozen rule) that the transform maps into the image",
            "contrast": "P - N on correspondence accuracy (the mapped node is the "
                        "transformed true counterpart)",
            "interval": "paired bootstrap over PHOTOS, 4000 draws, 2.5/97.5",
        },
        "no_harm": {
            "bank": "unambiguous nodes",
            "rule": "the lower end of the photo-clustered interval for P - N must not "
                    "fall below -0.02",
        },
        "secondary": ["judged-pair share per arm", "the frame chain per photo",
                      "per-structure-class table", "correspondence accuracy overall"],
        "mde": {
            "planning_sd": PLANNING_SD, "design_effect": DESIGN_EFFECT,
            "formula": "MDE = 2.8 * 0.5 * sqrt(2.0) / sqrt(n_bank)",
            "declared_before_the_bank_exists": True,
            "note": ("the realized bank size is reported with the read and the "
                     "realized MDE replaces this planning number, never the verdict"),
        },
        "verdict_rule": ("DETECTED iff the photo-clustered interval excludes zero in "
                         "the pre-registered direction (P > N); a detected effect "
                         "below the realized MDE is labelled possibly inflated"),
        "freshness": {"graphs_in_cache_at_registration": cached,
                      "rule": "any nonzero count refuses the read"},
        "one_read": True,
    }
    if cached:
        print("REFUSING: %d graphs already cached for the read photos" % cached)
        return 4
    os.makedirs(OUT, exist_ok=True)
    with open(PATH, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    print("registered %d queries over %d photos, %d transforms, w=%.3f, 0 graphs cached"
          % (n_queries, len(read_set), len(TRANSFORMS), w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
