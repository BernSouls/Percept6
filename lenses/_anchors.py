"""[M36/AUDIT-C2] ONE ANCHOR NAMESPACE PER LOOKUP.

An anchor is only meaningful INSIDE the band of the layer that minted it, and the
bands are disjoint by construction:

===========================  =========================================
producer                     alpha
===========================  =========================================
field-tiling (Color,         ``u``, the region id (0 .. a few hundred)
Texture, Orientation)
CONTOUR_PEAK                 ``PEAK_ANCHOR_BASE + i``                  (1e9 + i)
BorderOwnership              the fragment's CONTOUR_PEAK anchor        (1e9 + i)
PEAK_CHAIN                   ``PEAK_ANCHOR_BASE + CHAIN_ANCHOR_OFFSET + k``
Corner                       ``PEAK_ANCHOR_BASE + CORNER_ANCHOR_OFFSET + k``
===========================  =========================================

`PEAK_ANCHOR_BASE` is 1e9, `CHAIN_ANCHOR_OFFSET` is 1e5 and a frame has < 100
peaks, so a chain anchor (1e9 + 100000 + k) can NEVER equal a peak anchor
(1e9 + i < 1e9 + 100) or a region id.  Two consumers looked up across the bands:

* `SurfaceLens` used the PEAK_CHAIN anchor for `bo_by`, `color_by` and `tex_by`
  -- three different namespaces, and all three lookups miss on every row
  (measured: 0 hits in 21 chain rows; `bo_field` empty, `c_det`/`t_det` None, so
  every Surface fell through to the fallback material and the shape bin).
* `ContourLens` used an ORIENTATION anchor (a region id) for `bo_by_anchor`,
  which is keyed by PEAK_CHAIN/fragment anchors -- every row `continue`d before
  the ownership test.

This module resolves each lookup through its PRODUCER's namespace: a chain's own
member peaks (its `rho` member roles) for BorderOwnership, and the nearest
region-keyed detection by position for the field-tiling layers.  Every resolution
is COUNTED in `ANCHOR_RESOLUTION` so a zero-hit lookup is visible rather than
silent.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import parameters as P

#: hits/misses per consumer, for a test or a report to read.
ANCHOR_RESOLUTION: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"hit": 0, "miss": 0})

_PEAK_BASE = int(getattr(P, "PEAK_ANCHOR_BASE", 1000000000))
_CHAIN_OFF = int(getattr(P, "CHAIN_ANCHOR_OFFSET", 100000))
_CORNER_OFF = int(getattr(P, "CORNER_ANCHOR_OFFSET", 200000))


def _count(name: str, hit: bool) -> None:
    ANCHOR_RESOLUTION[name]["hit" if hit else "miss"] += 1


def peak_band(anchor: Any) -> bool:
    """True when `anchor` is a CONTOUR_PEAK / BorderOwnership anchor."""
    return (isinstance(anchor, int) and not isinstance(anchor, bool)
            and _PEAK_BASE <= anchor < _PEAK_BASE + _CHAIN_OFF)


def chain_band(anchor: Any) -> bool:
    return (isinstance(anchor, int) and not isinstance(anchor, bool)
            and _PEAK_BASE + _CHAIN_OFF <= anchor < _PEAK_BASE + _CORNER_OFF)


def region_band(anchor: Any) -> bool:
    """True when `anchor` is a field-tiling region id (a small non-negative int)."""
    return (isinstance(anchor, int) and not isinstance(anchor, bool)
            and 0 <= anchor < _PEAK_BASE)


def chain_member_peaks(det: Any) -> List[int]:
    """The peak anchors a PEAK_CHAIN detection names in its `rho` member roles."""
    out: List[int] = []
    for role in getattr(det, "rho", ()) or ():
        try:
            name, target = role[0], role[1]
        except (TypeError, IndexError):
            continue
        if name == "member" and peak_band(target):
            out.append(int(target))
    return out


def _position(anchor_positions: Optional[Dict[Any, Any]], anchor: Any
              ) -> Optional[Tuple[float, float]]:
    if not anchor_positions or anchor is None:
        return None
    pos = anchor_positions.get(anchor)
    if pos is None:
        return None
    try:
        return float(pos[0]), float(pos[1])
    except (TypeError, IndexError, ValueError):
        return None


def nearest(dets: Sequence[Any], pos: Optional[Tuple[float, float]],
            anchor_positions: Optional[Dict[Any, Any]],
            name: str = "nearest") -> Optional[Any]:
    """The detection whose own anchor position is nearest `pos`.

    A spatial join, which is the only well-defined translation between the
    region-keyed band and the peak-keyed band: the region a fragment's material
    is read from is the region its centroid falls in, i.e. the nearest one.
    """
    if pos is None or not dets:
        _count(name, False)
        return None
    best = None
    best_d = float("inf")
    for det in dets:
        p = _position(anchor_positions, getattr(det, "alpha", None))
        if p is None:
            continue
        d = math.hypot(p[0] - pos[0], p[1] - pos[1])
        if d < best_d:
            best_d, best = d, det
    _count(name, best is not None)
    return best


def nearest_all(dets: Sequence[Any], pos: Optional[Tuple[float, float]],
                anchor_positions: Optional[Dict[Any, Any]],
                name: str = "nearest_all") -> List[Any]:
    """Every detection at the nearest occupied position (ties kept together)."""
    if pos is None or not dets:
        _count(name, False)
        return []
    by_pos: Dict[Tuple[float, float], List[Any]] = defaultdict(list)
    best_d = float("inf")
    for det in dets:
        p = _position(anchor_positions, getattr(det, "alpha", None))
        if p is None:
            continue
        d = math.hypot(p[0] - pos[0], p[1] - pos[1])
        if d < best_d - 1e-9:
            best_d = d
        by_pos[p].append(det)
    if not by_pos:
        _count(name, False)
        return []
    out: List[Any] = []
    for p, group in by_pos.items():
        if abs(math.hypot(p[0] - pos[0], p[1] - pos[1]) - best_d) <= 1e-6:
            out.extend(group)
    _count(name, bool(out))
    return out


def bo_for_chain(det: Any,
                 bo_by: Dict[Any, List[Any]],
                 anchor_positions: Optional[Dict[Any, Any]] = None,
                 bo_dets: Optional[Iterable[Any]] = None) -> List[Any]:
    """The BorderOwnership rows that belong to a PEAK_CHAIN, by MEMBERSHIP.

    Primary: every member peak the chain names, looked up in the peak band.
    Fallback (a chain whose rho carries no member roles): the BorderOwnership
    rows at the position nearest the chain's own anchor.
    """
    out: List[Any] = []
    for peak in chain_member_peaks(det):
        out.extend(bo_by.get(peak, ()))
    if out:
        _count("surface.bo", True)
        return out
    if bo_dets is None:
        bo_dets = [d for group in bo_by.values() for d in group]
    pos = _position(anchor_positions, getattr(det, "alpha", None))
    out = nearest_all(list(bo_dets), pos, anchor_positions, name="surface.bo")
    return out


def bo_for_region(region_anchor: Any,
                  bo_dets: Sequence[Any],
                  anchor_positions: Optional[Dict[Any, Any]],
                  name: str = "contour.bo") -> Optional[Any]:
    """The BorderOwnership row nearest a region-anchored detection."""
    pos = _position(anchor_positions, region_anchor)
    return nearest(list(bo_dets), pos, anchor_positions, name=name)


def reset_counts() -> None:
    ANCHOR_RESOLUTION.clear()
