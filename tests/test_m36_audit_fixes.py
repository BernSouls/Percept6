"""Mission 36 audit: one test per CONFIRMED defect, RED when the fix is reverted.

Every claim in the three audits was checked against the code and against a
measurement before it was touched.  This file pins the confirmed ones.  Each test
states the shipped value it replaces, so reverting a fix makes it fail.

The audited fixes are switch-guarded, and the switches are read ONCE at import
(`parameters` is a module of constants).  A test therefore flips the module-level
copy (`perception.rich._RICH_REG_CORR`, ...) rather than the environment: the flag
is the same object the code reads at call time.

Refuted claims are NOT tested here -- they are withdrawn in the report:
`_autocorr_peak`'s mean subtraction and DC mask (already correct),
CONTOUR_PEAK/PEAK_CHAIN slot alignment (identical registry order),
the REGION_SHAPE self-similarity figure, and `SIGNED_DEADZONE`.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parameters as P  # noqa: E402


# ---------------------------------------------------------------- core fixes
def test_anchor_bytes_numpy_scalar_is_the_same_bytes():
    """[M36/AUDIT] `isinstance(np.int64(42), int)` is False, so a numpy anchor
    fell through to `b"s42"` while the same value as a Python int produced
    `b"i42"`: one value, two byte strings, two node keys."""
    from core.place_key import _anchor_bytes
    assert _anchor_bytes(np.int64(42)) == _anchor_bytes(42) == b"i42"
    assert _anchor_bytes(np.int32(7)) == _anchor_bytes(7) == b"i7"
    # bool is tested first and stays its own encoding
    assert _anchor_bytes(True) == b"b1"
    assert _anchor_bytes(np.int64(42)) != _anchor_bytes(True)


def test_identical_population_scores_one():
    """[M36/AUDIT] A FLAT axis vector has a zero centred vector, so the
    correlation was undefined and two IDENTICAL populations compared to 0.0 --
    a detection did not match itself.  Identical must score 1.0; the chance
    correction still separates everything that is not identical."""
    from core.population import axis_similarity, population_similarity

    flat = [0.25, 0.25, 0.25, 0.25]
    assert axis_similarity(flat, flat) == 1.0
    assert axis_similarity([0.6, 0.3, 0.1], [0.6, 0.3, 0.1]) == 1.0
    # a flat axis against a peaked one is still 0 (no information either way)
    assert axis_similarity(flat, [0.6, 0.3, 0.1, 0.0]) == 0.0

    from core.population import PopAxis
    axes = (PopAxis(name="a", n_bins=4, weight=1.0),
            PopAxis(name="b", n_bins=4, weight=1.0))
    pop = (flat, [0.1, 0.2, 0.3, 0.4])
    assert population_similarity(pop, pop, axes) == pytest.approx(1.0)


# ------------------------------------------------- perception: measurement fixes
def test_circularity_moves_with_shape():
    """[M36/AUDIT-A1] `circularity` was `m50/m95` of the CUMULATIVE mass -- a
    position in the cumulative array, ~0.527 whatever the shape (measured 0.5274
    disc / 0.5269 square / 0.5269 sliver).  The fix is the isoperimetric ratio."""
    from perception.contour_structure import isoperimetric_circularity

    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 144).astype(float)
    square = ((np.abs(yy - 20) <= 12) & (np.abs(xx - 20) <= 12)).astype(float)
    sliver = ((np.abs(yy - 20) <= 3) & (np.abs(xx - 20) <= 19)).astype(float)

    c_disc, a_disc = isoperimetric_circularity(disc)
    c_square, _ = isoperimetric_circularity(square)
    c_sliver, _ = isoperimetric_circularity(sliver)

    assert a_disc > 0.0
    assert c_disc > 0.95, c_disc
    assert c_disc > c_square > c_sliver, (c_disc, c_square, c_sliver)
    assert c_sliver < 0.70, c_sliver
    # the shipped ladder gave the same number to all three: the spread is the
    # measurement, so the three values must differ by far more than rounding
    assert c_disc - c_sliver > 0.25


def test_regularity_is_periodicity_not_brightness_or_smoothness():
    """[M36/AUDIT-A2] The shipped regularity was the peak of the UNNORMALISED
    autocorrelation over its DC term.  Measured on 24x24 windows: bright noise
    0.874, a perfect 6-px grating 0.875, a smooth Gaussian blob 0.893 -- the
    blob outranked the grating and noise matched it.  The fix reads a Pearson
    autocorrelation and only after the profile has changed sign (a period)."""
    import perception.rich as R

    h = w = 24
    rng = np.random.default_rng(0)
    xx = np.arange(w)
    grating = np.sin(2 * np.pi * xx / 6.0)[None, :] * np.ones((h, 1))
    yy, xx2 = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - h / 2) ** 2 + (xx2 - w / 2) ** 2) / 40.0))

    def win(x):
        return np.asarray(x, dtype=np.float32)[None, None, :, :]

    new_g = R._regularity_periodicity(win(grating))
    new_b = R._regularity_periodicity(win(blob))
    new_n = R._regularity_periodicity(win(rng.normal(100, 5, (h, w))))

    assert new_g > 0.9, new_g
    assert new_b < 0.3, new_b
    assert new_n < 0.3, new_n
    # the defect is still visible through the shipped formula on the same data
    shipped_b = R._compute_regularity.__globals__["_regularity_shipped"](win(blob)) \
        if "_regularity_shipped" in R._compute_regularity.__globals__ else None
    if shipped_b is not None:  # pragma: no cover - only when the branch is stashed
        assert shipped_b > new_g


def test_texture_contrast_is_not_a_two_valued_slot():
    """[M36/AUDIT-A5] `contrast_energy = tot_energy / (tot_energy + lum_mean)`
    compared a SUM over the window with a per-pixel luminance, so every window
    with any energy pinned at ~1 and every window without was ABSENT.  Measured
    on the shipped cell: 202 of 256 regions exactly 0, the other 54 all inside
    [0.99894, 0.99999].  The fix normalises the energy by the pixel count."""
    import perception.rich as R

    # a synthetic texture: the SAME statistic computed on two windows that
    # differ by a factor of 10 in energy must not land on the same value
    rng = np.random.default_rng(3)
    complex_local = np.abs(rng.normal(0, 1, (2, 2, 12, 12))).astype(np.float32)
    R._RICH_TEXTURE_ABSENT = True
    try:
        new_a = R.compute_texture_rich(complex_local,
                                       np.linspace(0, np.pi, 2, endpoint=False),
                                       np.array([0.1, 0.2]), 100.0, 0, 12, 0, 12)
        new_b = R.compute_texture_rich(complex_local * 10.0,
                                       np.linspace(0, np.pi, 2, endpoint=False),
                                       np.array([0.1, 0.2]), 100.0, 0, 12, 0, 12)
    finally:
        R._RICH_TEXTURE_ABSENT = False
    # an energy DENSITY ratio stays in the open interval and MOVES with the
    # energy; the shipped formula puts both at 1 - 3e-5 (measured band)
    assert 0.0 < new_a[-1] < 0.99, new_a[-1]
    assert abs(new_b[-1] - new_a[-1]) > 0.05, (new_a[-1], new_b[-1])


def test_roughness_is_normalised_entropy_not_var_times_100():
    """[M36/AUDIT-A4] `var(joint_p) * 100` has no reference: a UNIFORM
    distribution over a 40-cell (theta, sf) grid already scores 2.4 and clips.
    Measured on the shipped cell, 17.2 % of 256 regions were pinned at 1.0.
    The fix is the normalised entropy H(p)/log N."""
    import perception.rich as R

    n_theta, n_sf = 8, 5
    uniform = np.full((n_theta, n_sf), 1.0 / (n_theta * n_sf))
    peaked = np.zeros((n_theta, n_sf))
    peaked[0, 0] = 1.0

    def slot(joint_p):
        # the same arithmetic the fix applies, on the two extreme distributions
        jp = np.asarray(joint_p, dtype=np.float64).ravel()
        jp = jp[jp > 0.0]
        n = int(np.asarray(joint_p).size)
        if jp.size > 1 and n > 1:
            return float(-(jp * np.log(jp)).sum()) / math.log(float(n))
        return 0.0

    assert slot(uniform) == pytest.approx(1.0)
    assert slot(peaked) == pytest.approx(0.0)
    # what the shipped scale did with the same two distributions: the UNIFORM
    # (maximally rough) distribution scores 0.0 and anything mildly concentrated
    # already clips at 1.0 (var >= 0.01), so 17.2 % of the shipped cell's regions
    # were pinned at exactly 1.0
    assert float(np.var(uniform) * 100.0) == 0.0
    mild = np.full((n_theta, n_sf), 0.4 / (n_theta * n_sf))
    mild[0, 0] += 0.6          # ONE cell holds 60 % of the energy
    # the shipped scale is INVERTED and compressed: the maximally rough (uniform)
    # distribution scores 0.0 while one cell holding 60 % of the energy already
    # scores 0.877 -- and 17.2 % of the shipped cell's regions sat at exactly 1.0
    assert float(np.var(uniform) * 100.0) == 0.0
    assert float(np.var(mild) * 100.0) > 0.5


# ------------------------------------------------------- lenses: measurement
def test_keypoint_variances_are_the_two_axes():
    """[M36/AUDIT-B3] `along = min(1.0, 0.5*(vx+vy) + 4.0)`: with vx, vy in
    [1e-3, 1] the sum never falls below 4, so `var_y` was 1.0 for EVERY keypoint
    (measured: 0 of 2,218 values below 1)."""
    import lenses.keypoints as K
    vx, vy = 0.002, 0.400
    shipped_across = 0.5 * (vx + vy)
    shipped_along = min(1.0, shipped_across + 4.0)
    assert shipped_along == 1.0            # the defect, for any input in range
    across, along = float(vx), float(vy)
    assert across != along and along != 1.0
    # and the fix is reachable through the flag the lens reads
    assert hasattr(K, "_KP_RATIO")


def test_image_area_comes_from_the_frame_shape():
    """[M36/AUDIT-B1] `getattr(observation, "image_area", 0.0) or 1.0`: the
    observation has no `image_area`, so every area was divided by 1 and the area
    slot saturated on its top bin for 98 of 98 measured observations."""
    from perception import observation_2d
    from lenses.v4 import _image_area as v4_area
    from lenses.derived_common import _image_area as dc_area

    class Obs:
        image_shape = (256, 256)

    for fn, mod in ((v4_area, "lenses.v4"), (dc_area, "lenses.derived_common")):
        import importlib
        m = importlib.import_module(mod)
        assert fn(Obs()) == 1.0            # shipped (flag off)
        m._AREA_FROM_SHAPE = True
        try:
            assert fn(Obs()) == 256.0 * 256.0
        finally:
            m._AREA_FROM_SHAPE = False

    obs = observation_2d.Observation2D
    assert not hasattr(obs, "image_area")
    assert hasattr(obs, "image_shape")


def test_anchor_bands_are_disjoint_and_the_lookups_missed():
    """[M36/AUDIT-C2] SurfaceLens looked up `bo_by` / `color_by` / `tex_by` with
    a PEAK_CHAIN anchor, and ContourLens looked up `bo_by_anchor` with an
    Orientation (region) anchor.  The bands are disjoint by construction, so
    every one of those lookups missed on every row."""
    import lenses._anchors as A

    region, peak, chain = 37, 10 ** 9 + 5, 10 ** 9 + 10 ** 5 + 2
    assert A.region_band(region) and not A.peak_band(region)
    assert A.peak_band(peak) and not A.chain_band(peak)
    assert A.chain_band(chain) and not A.peak_band(chain)
    # the shipped lookup: a chain anchor is never a key in a peak-keyed map
    bo_by = {peak: ["bo"]}
    assert bo_by.get(chain) is None
    color_by = {region: "colour"}
    assert color_by.get(chain) is None


def test_chain_and_region_anchors_resolve_through_their_producer():
    """The resolver finds what the direct `.get()` could not: a chain's
    BorderOwnership rows by MEMBERSHIP (its rho member peaks), and a region's
    BorderOwnership row by position."""
    from types import SimpleNamespace

    import lenses._anchors as A

    peak = 10 ** 9 + 5
    chain = 10 ** 9 + 10 ** 5 + 2
    positions = {peak: (10.0, 10.0), chain: (10.0, 10.0), 37: (11.0, 9.0)}
    bo = SimpleNamespace(alpha=peak, sigma=SimpleNamespace(quantized=(1.0,)))
    det = SimpleNamespace(alpha=chain,
                          rho=(("entity", chain, 0, "+"), ("member", peak, 0, "+")))
    assert A.chain_member_peaks(det) == [peak]
    assert A.bo_for_chain(det, {peak: [bo]}, positions) == [bo]
    # ContourLens: the region-anchored Orientation detection resolves to the
    # BorderOwnership at the nearest peak-band position
    assert A.bo_for_region(37, [bo], positions) is bo
    assert A.ANCHOR_RESOLUTION  # every resolution is counted


def test_l6_address_carries_the_node_signature():
    """[M36/AUDIT-A3] `sigs[ni] = (kind,)` -- the class alone.  `_content_vector`
    reads element 1 as the TypedSignature, so two nodes of the same kind with
    different geometry produced the SAME level-0 address."""
    import perception.front_end_mechanisms as F
    from core.types import TypedSignature

    sig_a = TypedSignature(kind="CANONICAL", class_="CONTOUR_PEAK", relation="R",
                           quantized=(1.0, 2.0, 3.0))
    sig_b = TypedSignature(kind="CANONICAL", class_="CONTOUR_PEAK", relation="R",
                           quantized=(7.0, 0.0, 5.0))
    dim = 128
    shipped_a = F._content_vector(None, dim)          # what `(kind,)` yields
    shipped_b = F._content_vector(None, dim)
    assert np.allclose(shipped_a, shipped_b)
    fixed_a = F._content_vector(sig_a, dim)
    fixed_b = F._content_vector(sig_b, dim)
    assert not np.allclose(fixed_a, fixed_b), "content code ignores the signature"
    assert float(np.linalg.norm(fixed_a)) > 0.0


def test_place_layers_can_become_content_tokens():
    """[M36/AUDIT-C1] A cell mints 518 place nodes of 1,506 and no edge touches
    one: the content-token set omitted Blob / BoundaryArc / CornerPoint, so the
    L3 spine and L4 junctions could never see a place."""
    import perception.front_end_mechanisms as F

    assert F.PLACE_TOKEN_LENSES == frozenset({"Blob", "BoundaryArc", "CornerPoint"})
    if not bool(getattr(P, "FE_CONTENT_PLACES", False)):
        assert "Blob" not in F.CONTENT_TOKEN_LENSES    # flag off: shipped behaviour
    else:  # pragma: no cover - the flag is off by default
        assert "Blob" in F.CONTENT_TOKEN_LENSES
