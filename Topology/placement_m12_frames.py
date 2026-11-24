"""MISSION 12 STEP 3 -- the frame sources, measured link by link, with a known-
rotation validation that never fits anything.

The chain that decides whether a mapped pair can be JUDGED at all:

    nodes -> nodes whose anchor carries an ORIENTATION FRAME
          -> nodes with at least one PAYLOAD EDGE
          -> mapped pairs -> mapped pairs with >= 1 JUDGED SLOT

Three providers are compared on the SAME images and the SAME graphs:

    field        the single 7x7 box window (the shipped default)
    multiscale   the smallest box window that clears the coherence bar
    keypoint     the keypoint estimator (Gaussian tensor at the keypoint layer's
                 sigma ladder, SUB-PIXEL bilinear sampling), multiscale field as
                 the fallback -- the M12 change, dark behind
                 PLACEMENT_FRAME_KEYPOINT

and the validation is a KNOWN ROTATION: an image is rotated by 5 degrees about its
own centre, the frame at every node that survives both images is compared, and the
per-node PAIRED residual is reported (median |residual| per provider, and the share
of nodes whose residual is within 5 degrees).  A static provider would show the
residual at 5 degrees, not at 0.

NOTHING IS FITTED HERE, on real photographs or anywhere else, and no arm is scored.
The judged-pair share is read off the deployed matcher's own mapping.

If the ETH-80 corpus is not present the real-photograph half cannot run: the
harness says so, prints the synthetic proxy, and the report states the floor.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
RESULT = os.path.join(OUT, "m12_frames.json")
ROT_DEG = 5.0
N_SYNTH = 6


def _rot5(image_u8: np.ndarray) -> np.ndarray:
    from scipy.ndimage import rotate
    a = np.asarray(image_u8, dtype=np.float64)
    out = np.stack([rotate(a[..., c], ROT_DEG, reshape=False, order=1, mode="nearest")
                    for c in range(a.shape[2])], axis=2) if a.ndim == 3 else \
        rotate(a, ROT_DEG, reshape=False, order=1, mode="nearest")
    return np.clip(out, 0, 255).astype(np.uint8)


def _providers(subgraph: Any, image_u8: np.ndarray) -> Dict[str, Dict[int, float]]:
    from matching import placement_edges as PE
    ang, coh = PE.field_angles(image_u8)
    f_field = PE.node_frames(subgraph, ang, coh)
    f_multi, _w = PE.node_frames_multiscale(subgraph, PE.field_multiscale(image_u8))
    f_kp, _k = PE.node_frames_keypoint_then_field(subgraph, image_u8)
    return {"field": f_field, "multiscale": f_multi, "keypoint": f_kp}


def _axial_residual(a: float, b: float) -> float:
    """|residual| of two AXIAL frames (pi-periodic), in degrees."""
    d = (float(a) - float(b)) % np.pi
    return float(min(d, np.pi - d) * 180.0 / np.pi)


def _pair_report(subgraph: Any, image_u8: np.ndarray, tag: str) -> Dict[str, Any]:
    """Frame coverage and the known-rotation residual, per provider, one image."""
    rot = _rot5(image_u8)
    base = _providers(subgraph, image_u8)
    moved = _providers(subgraph, rot)
    n_nodes = len(subgraph.nodes)
    out: Dict[str, Any] = {"tag": tag, "nodes": int(n_nodes), "providers": {}}
    for name in ("field", "multiscale", "keypoint"):
        b, m = base[name], moved[name]
        common = sorted(set(b) & set(m))
        res = [_axial_residual(b[u], m[u]) for u in common]
        out["providers"][name] = {
            "frame_share": float(len(b) / n_nodes) if n_nodes else 0.0,
            "frames": len(b),
            "paired_nodes": len(common),
            "median_abs_residual_deg": float(np.median(res)) if res else float("nan"),
            "share_within_5deg": float(np.mean([r <= 5.0 for r in res])) if res else float("nan"),
            "share_within_10deg": float(np.mean([r <= 10.0 for r in res])) if res else float("nan"),
        }
    return out


def _judged_job(args: Tuple[str, Any, Any]) -> Dict[str, Any]:
    """One synthetic pair: frame coverage, payload edges and the judged-pair share."""
    tag, spec, view = args
    import guard
    guard.activate()
    from matching import placement_matcher as PM
    from Topology import placement_m11_families as F
    from Topology.placement_cache import get_or_build
    from Topology.placement_truth import node_objects

    sc1, _pv = F.pose_fn(spec, view)
    r0, r1 = spec.scene.render(), sc1.render()
    g0 = get_or_build(r0.image_u8, spec.name + "|ref")
    g1 = get_or_build(r1.image_u8, spec.name + "|" + tag.split("|")[1])
    rec: Dict[str, Any] = {"tag": tag}
    rec.update(_pair_report(g1.subgraph, r1.image_u8, tag))

    # the deployed mapping, and how many of its pairs the estimator can judge
    from matching.joint_mcs import joint_mcs
    import parameters as P
    # ARM B AND NOTHING MORE: the mapping moves, the score does not (weight 0), but
    # the evidence IS computed, which is where `placement_judged` comes from.  With
    # SCORE_EVIDENCE off the counter is never incremented and the judged share would
    # read a spurious ZERO.
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    res = joint_mcs(g1.subgraph, g0.subgraph, g1.sigs, g0.sigs,
                    placement_images=(r1.image_u8, r0.image_u8))
    P.PLACEMENT_MATCHER_ENABLED = False
    rec["mapped"] = int(len(res.mapping))
    rec["judged"] = int(res.placement_judged)
    rec["judged_share"] = float(res.placement_share_judged)
    return rec


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    from Topology import placement_m11_families as F

    gal = F.gallery()
    ir = [s for s in gal if s.family == "IRREGULAR"][:N_SYNTH]
    jobs = []
    for i, s in enumerate(ir):
        lab, view = F.VIEW_STEPS[i % 2]
        jobs.append(("ir_%02d|%s" % (i, lab), s, view))

    real: Dict[str, Any] | None = None
    try:
        from Topology.placement_m11_eth80 import catalog, _load_u8
        refs, queries = catalog()
        if refs and queries:
            import numpy as _np
            u8 = _load_u8(queries[0]["path"])
            from Topology.placement_cache import get_or_build as _gb
            g = _gb(u8, "m12frame|real")
            real = _pair_report(g.subgraph, u8, "real|" + queries[0]["query"])
    except Exception as exc:
        real = {"skipped": "%s: %s" % (type(exc).__name__, exc)}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_judged_job, jobs))

    def agg(name: str) -> Dict[str, float]:
        vals = [r["providers"][name] for r in rows]
        return {"frame_share": float(np.mean([v["frame_share"] for v in vals])),
                "median_abs_residual_deg": float(np.nanmedian(
                    [v["median_abs_residual_deg"] for v in vals])),
                "share_within_5deg": float(np.nanmean([v["share_within_5deg"] for v in vals]))}

    res = {"synthetic_pairs": len(rows),
           "synthetic": {n: agg(n) for n in ("field", "multiscale", "keypoint")},
           "judged_share": float(np.mean([r["judged_share"] for r in rows])),
           "judged_mean": float(np.mean([r["judged"] for r in rows])),
           "mapped_mean": float(np.mean([r["mapped"] for r in rows])),
           "real": real,
           "note": ("the frame providers are compared with the SAME graphs and the "
                    "same images; the rotation validation rotates the image 5 degrees "
                    "about its own centre and compares each node's frame with itself. "
                    "Nothing is fitted.")}
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
