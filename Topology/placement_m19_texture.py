"""MISSION 19 / O2 -- the three statistics M18 left open, measured per image.

* TEXTURE PERIOD (px): the first local maximum of the horizontal autocorrelation of the
  luminance beyond lag 3 -- the lag at which the image correlates with itself again.  (The
  first version read lag 1 for every image, which is the autocorrelation's central peak and
  not a period; that bug is why the lag floor exists.)
* COHERENCE DISTRIBUTION: the per-place coherence SLOT (the signature's coarse slot,
  coherence/4).  It is a coarse reading, and it is reported as one -- the graded population
  axes of these classes are spectral, not coherence.
* ABSOLUTE PLACE SPACING (px): `distance` is dimensionless source-scale units, so the pixel
  spacing is `distance * source_scale`; the median over the source-frame vectors of one side.

    python Topology/placement_m19_texture.py --workers 2
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m19_texture.json")

DEV = ("syn|m11ir_00", "syn|m11ir_01", "real|cell", "real|grass")


def _period(img: np.ndarray, min_lag: int = 4) -> Dict[str, Any]:
    lum = img.astype(float).mean(axis=2)
    lum = lum - lum.mean()
    F = np.fft.rfft2(lum)
    ac = np.fft.irfft2(F * np.conj(F), s=lum.shape)
    ac = ac[: ac.shape[0] // 2, : ac.shape[1] // 2]
    ac = ac / max(float(ac[0, 0]), 1e-9)
    prof = ac[:, 0].copy()
    peaks = [int(k) for k in range(max(min_lag, 2), len(prof) - 1)
             if prof[k] >= prof[k - 1] and prof[k] >= prof[k + 1]]
    best = max(peaks, key=lambda k: prof[k]) if peaks else None
    # the 2-D structure: the radius of the peak of the radial power spectrum
    P = np.abs(F) ** 2
    P[0, 0] = 0.0
    yy, xx = np.mgrid[0: P.shape[0], 0: P.shape[1]]
    yy = np.minimum(yy, P.shape[0] - yy)
    radius = np.sqrt(yy * yy + xx * xx)
    hist = np.zeros(int(radius.max()) + 2)
    np.add.at(hist, radius.astype(int), P)
    k = int(np.argmax(hist[1:]) + 1)
    return {"period_px": best, "period_strength": float(prof[best]) if best else None,
            "spectral_radius": k,
            "spectral_period_px": (float(lum.shape[0]) / k) if k else None}


def _job(tag: str) -> Dict[str, Any]:
    import guard
    guard.activate()
    import parameters as P
    from matching import placement_matcher as PM
    from Topology.placement_cache import get_or_build
    from Topology.placement_m17_eval import M16, _image

    for k, v in M16.items():
        setattr(P, k, v)
    P.KP_GRADED_POPULATION = True
    img = _image(tag)
    out: Dict[str, Any] = {"tag": tag, "shape": list(img.shape)}
    out.update(_period(img))
    g = get_or_build(img, "m16|%s|orig|on" % tag)
    side = PM.build_side_cached(g, img)
    coh, scale, dist_px = [], [], []
    for v in g.subgraph.nodes:
        s = g.sigs.get(int(v))
        if s is None:
            continue
        q = getattr(s[1], "quantized", ()) or ()
        if len(q) >= 2:
            coh.append(float(q[1]))
        if len(q) >= 3:
            scale.append(float(q[2]))
    for u, lst in side.incident.items():
        for (other, is_source, name) in lst:
            if not is_source:
                continue
            pv = side.index.get((int(u), int(other), name))
            if pv is None:
                continue
            d = float(getattr(pv, "distance", float("nan")))
            sc = float(getattr(pv, "source_scale", float("nan")))
            if math.isfinite(d) and math.isfinite(sc) and sc > 0:
                dist_px.append(d * sc)
    out.update({"places": len(g.subgraph.nodes),
                "coherence_median": float(st.median(coh)) if coh else None,
                "coherence_q1": float(np.percentile(coh, 25)) if coh else None,
                "coherence_q3": float(np.percentile(coh, 75)) if coh else None,
                "coherence_hist": {str(int(k)): int(sum(1 for c in coh if c == k))
                                   for k in sorted(set(coh))},
                "spacing_px_median": float(st.median(dist_px)) if dist_px else None,
                "spacing_px_n": len(dist_px)})
    return out


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    from matching import m15_manifest as MF
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, DEV):
            rows.append(r)
            print("  %-14s period %4s px  spectral %6s px  coh %.1f [%.1f-%.1f]  "
                  "spacing %5.2f px  places %4d"
                  % (r["tag"], r["period_px"],
                     "None" if r["spectral_period_px"] is None else "%.1f" % r["spectral_period_px"],
                     r["coherence_median"] or float("nan"),
                     r["coherence_q1"] or float("nan"), r["coherence_q3"] or float("nan"),
                     r["spacing_px_median"] or float("nan"), r["places"]), flush=True)
    MF.write_result(OUT, {"rows": rows,
                          "units": "period and spacing in px; coherence in coarse slot units",
                          "note": "coherence is the signature's COARSE slot (coherence/4)"})
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
