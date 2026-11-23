"""Spacing-aware geometry: one fitted rule, dark, reading one frozen file.

The frozen tolerance (1.7009) and the neighbour radius (a multiple of the source's
own scale) were fitted on scenes whose places sit about six pixels apart.  Real
photographs put them 5.83 px apart and the synthetic tenth percentile is 2.03 px,
so an absolute number covers a different fraction of a place spacing on every
image.  Mission 10 step 4a asks whether the rule should scale, and this module is
the answer's carrier: it holds the fitted numbers, it computes nothing at import
time, and every caller asks it only while the corresponding flag is on.

    tolerance   tol(u, v) = base * clip((max(sp_u, sp_v) / spacing_ref) ** gamma,
                                       lo, hi)
    radius      radius_cap(u) = factor * sp_u            (only ever a CAP: the
                                                          selector keeps its own
                                                          scale-relative radius
                                                          when that is smaller)

`spacing_ref`, `gamma`, `lo`, `hi` and `factor` are FITTED ON SYNTHETIC FIT ONLY
(`Topology/placement_m10_spacing.py`) and written to
`matching/placement_spacing_params.json`.  Nothing here is fitted on real images:
they are the report, not the training set.

When the file is missing the module refuses to operate unless asked for the
identity rule, so a dark flag can never silently become a half-applied one.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(_HERE, "placement_spacing_params.json")

_PARAMS: Optional[Dict[str, Any]] = None


def installed() -> bool:
    return os.path.exists(PARAMS_PATH)


def params() -> Dict[str, Any]:
    """The frozen spacing rule.  Raises if the file is absent or malformed."""
    global _PARAMS
    if _PARAMS is None:
        if not installed():
            raise RuntimeError(
                "placement_spacing_params.json is missing: the spacing flags are "
                "dark, so nothing should be asking for this rule")
        with open(PARAMS_PATH) as fh:
            _PARAMS = json.load(fh)
        for key in ("spacing_ref", "tol", "radius", "version"):
            if key not in _PARAMS:
                raise ValueError("spacing params missing %r" % key)
    return _PARAMS


def enabled(kind: str) -> bool:
    """True when the given rule ('tol' or 'radius') is switched on."""
    import parameters as P
    if kind == "tol":
        return bool(getattr(P, "PLACEMENT_SPACING_TOL", False))
    if kind == "radius":
        return bool(getattr(P, "PLACEMENT_SPACING_RADIUS", False))
    raise KeyError(kind)


def tol_for(sp_u: Optional[float], sp_v: Optional[float], base: float,
            fallback: Optional[float] = None) -> float:
    """The per-pair tolerance, or ``base`` when the rule cannot be applied."""
    if not enabled("tol"):
        return float(base)
    p = params()
    ref = float(p["spacing_ref"])
    sp = max([float(x) for x in (sp_u, sp_v, fallback)
              if x is not None and math.isfinite(float(x)) and float(x) > 0.0],
             default=0.0)
    if sp <= 0.0 or ref <= 0.0:
        return float(base)
    gamma = float(p["tol"].get("gamma", 1.0))
    lo = float(p["tol"].get("lo", 0.5))
    hi = float(p["tol"].get("hi", 2.0))
    scale = (sp / ref) ** gamma
    return float(base) * float(min(hi, max(lo, scale)))


def radius_cap(sp_u: Optional[float]) -> Optional[float]:
    """An upper bound on the neighbour radius, or None when the rule is off."""
    if not enabled("radius"):
        return None
    p = params()
    if sp_u is None or not math.isfinite(float(sp_u)) or float(sp_u) <= 0.0:
        return None
    factor = float(p["radius"].get("factor", 1.0))
    lo = float(p["radius"].get("min_px", 1.0))
    return max(lo, factor * float(sp_u))
