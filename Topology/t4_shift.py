"""Repaired shift instrument: ONE helper (dx, dy explicit) for both the image
roller and the expected-position function.

Strict ``<`` at a radius boundary: a counterpart at exactly the radius is NOT
counted (distance must be strictly less than the radius).

Convention: (dx, dy) = (horizontal shift in pixels, vertical shift in pixels).
  np.roll(arr, (dy, dx), axis=(0, 1))  shifts rows by dy, columns by dx.
  Expected position of a base node at (x0, y0): (x0 + dx, y0 + dy).

This is the ONLY shift helper used by all t4 scripts.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def roll_image(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Cyclic-roll an image by (dx, dy) pixels.

    dx shifts columns (horizontal), dy shifts rows (vertical).
    Non-integer shifts are not supported by np.roll; use Fourier shift for
    fractional shifts (see fourier_shift below).
    """
    dx_int = int(round(dx))
    dy_int = int(round(dy))
    if dx_int == 0 and dy_int == 0:
        return arr
    return np.roll(arr, (dy_int, dx_int), axis=(0, 1))


def fourier_shift(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Fourier shift by (dx, dy) pixels (supports fractional shifts).

    Standard phase-shift: multiply FFT by exp(-2j*pi*(kx*dx + ky*dy))
    where kx, ky are cycles-per-pixel frequencies from fftfreq.
    """
    from numpy.fft import fft2, ifft2, fftfreq
    ny, nx = arr.shape[:2]
    kx = fftfreq(nx)
    ky = fftfreq(ny)
    ramp = np.exp(-2j * np.pi * (kx[None, :] * dx + ky[:, None] * dy))

    if arr.ndim == 2:
        f = fft2(arr)
        out = np.real(ifft2(f * ramp))
        return out.astype(arr.dtype)
    # multi-channel
    out_channels = []
    for c in range(arr.shape[2]):
        f = fft2(arr[..., c])
        out_channels.append(np.real(ifft2(f * ramp)))
    return np.stack(out_channels, axis=-1).astype(arr.dtype)


def expected_position(x0: float, y0: float, dx: float, dy: float):
    """Expected shifted position of a node at (x0, y0) under shift (dx, dy)."""
    return (x0 + dx, y0 + dy)


def counterpart_map(base_pos, shift_pos, dx, dy, radius,
                    radius_by_lens=None):
    """One-to-one counterpart map (REPAIRED).

    base_pos: {node: (x, y, lens)}
    shift_pos: {node: (x, y, lens)}
    dx, dy: shift
    radius: PRIMARY POSITION RADIUS (strict <); default/fallback per lens
    radius_by_lens: optional {lens: radius}; overrides `radius` per lens

    ONE-TO-ONE RULE: forward nearest (base -> shift) and reverse nearest
    (shift -> base), both searched with INFINITY and the radius applied
    afterwards; keep only MUTUAL pairs; resolve competing claims greedily by
    smallest forward distance; drop the rest.

    Ambiguity: second_d <= 2 * best_d, NON-STRICT (<=), so two candidates
    both at distance 0 are flagged. A lone distance-0 candidate has
    second_d = inf, so inf <= 0 is False and it is NOT flagged.

    Returns:
      mapping: {base_node: shift_node} for mutual pairs only
      info: {base_node: {"dist", "ambiguous", "second_dist", "reverse_dist"}}
      stats: {"conflicts", "dropped", "mutual", "with_counterpart"}
    """
    by_lens = defaultdict(list)
    for n, (x, y, l) in shift_pos.items():
        by_lens[l].append((n, x, y))
    base_by_lens = defaultdict(list)
    for n, (x, y, l) in base_pos.items():
        base_by_lens[l].append((n, x, y))

    # Forward: base -> nearest shift (search inf, apply radius after)
    fwd = {}
    for n, (x0, y0, l0) in base_pos.items():
        r_eff = radius_by_lens.get(l0, radius) if radius_by_lens else radius
        ex, ey = expected_position(x0, y0, dx, dy)
        best_n, best_d = None, float("inf")
        second_d = float("inf")
        for nsh, xsh, ysh in by_lens.get(l0, ()):
            d = math.hypot(ex - xsh, ey - ysh)
            if d < best_d:
                second_d = best_d
                best_d = d
                best_n = nsh
            elif d < second_d:
                second_d = d
        if best_n is not None and best_d < r_eff:
            fwd[n] = (best_n, best_d, second_d)
        else:
            fwd[n] = None

    # Reverse: shift -> nearest base (search inf, apply radius after)
    rev = {}
    for nsh, (xsh, ysh, lsh) in shift_pos.items():
        r_eff = radius_by_lens.get(lsh, radius) if radius_by_lens else radius
        ex, ey = xsh - dx, ysh - dy
        best_n, best_d = None, float("inf")
        second_d = float("inf")
        for nb, xb, yb in base_by_lens.get(lsh, ()):
            d = math.hypot(ex - xb, ey - yb)
            if d < best_d:
                second_d = best_d
                best_d = d
                best_n = nb
            elif d < second_d:
                second_d = d
        if best_n is not None and best_d < r_eff:
            rev[nsh] = (best_n, best_d, second_d)
        else:
            rev[nsh] = None

    # Group forward claims by shift node; resolve competing claims greedily
    # by smallest forward distance.
    claims_by_shift = defaultdict(list)
    for n, claim in fwd.items():
        if claim is not None:
            claims_by_shift[claim[0]].append((n, claim[1], claim[2]))

    mapping = {}
    info = {}
    conflicts = 0
    for bn, claimants in claims_by_shift.items():
        if len(claimants) > 1:
            conflicts += len(claimants) - 1
        claimants.sort(key=lambda c: c[1])  # smallest forward distance first
        n_win, bd_win, sd_win = claimants[0]
        rn = rev.get(bn)
        if rn is not None and rn[0] == n_win:
            # Mutual: shift bn's reverse-nearest is n_win.
            mapping[n_win] = bn
            ambiguous = (sd_win <= 2 * bd_win)  # NON-STRICT
            info[n_win] = {"dist": bd_win, "ambiguous": ambiguous,
                           "second_dist": sd_win, "reverse_dist": rn[1]}

    with_claim = sum(1 for c in fwd.values() if c is not None)
    kept = set(mapping.keys())
    claimed = set()
    for cs in claims_by_shift.values():
        for c in cs:
            claimed.add(c[0])
    dropped = len(claimed) - len(kept & claimed)
    stats = {"conflicts": conflicts, "dropped": dropped,
             "mutual": len(mapping), "with_counterpart": with_claim}
    return mapping, info, stats


def primary_radius(positions_by_lens, min_px=1.0, max_px=8.0):
    """Compute PRIMARY POSITION RADIUS per token kind.

    half the median nearest-neighbour same-kind spacing on dev,
    clamped to [1 px, 8 px].

    positions_by_lens: {lens: [(x, y), ...]}
    Returns: {lens: radius or None if < 20 nodes}
    """
    result = {}
    for lens, pts in positions_by_lens.items():
        if len(pts) < 20:
            result[lens] = None  # ABSENT
            continue
        # For each point, find nearest same-kind neighbor
        nn_dists = []
        for i, (x, y) in enumerate(pts):
            best = float("inf")
            for j, (x2, y2) in enumerate(pts):
                if i == j:
                    continue
                d = math.hypot(x - x2, y - y2)
                if d < best:
                    best = d
            if best < float("inf"):
                nn_dists.append(best)
        if not nn_dists:
            result[lens] = None
            continue
        median_nn = float(np.median(nn_dists))
        radius = max(min_px, min(max_px, median_nn / 2.0))
        result[lens] = radius
    return result


def matcher_accuracy(base_pos, mapping, counterpart, token_kinds):
    """Matcher mapping accuracy on token kinds.

    mapping: matcher's mapping (base_node -> shift_node)
    counterpart: expected counterpart map (base_node -> shift_node)
    """
    total = correct = no_c = 0
    for n, (x, y, l) in base_pos.items():
        if l not in token_kinds:
            continue
        total += 1
        cp = counterpart.get(n)
        if cp is None:
            no_c += 1
            continue
        if mapping.get(n) == cp:
            correct += 1
    return {"total": total, "correct": correct, "no_counterpart": no_c,
            "acc": correct / total if total else 0.0}
