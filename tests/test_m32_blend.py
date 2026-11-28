"""M32 item 5 and item 6: the blend's identity, the distance's own properties, the curvature's.

Every test here fails if the M32 behaviour is reverted:

  * ``test_zero_weights_is_the_deployed_matrix`` -- w_shape = w_colour = 0 returns the matrix
    itself; the blend cannot move a score it was not asked to touch.
  * ``test_refuses_what_the_ordering_forbids`` -- colour above the cap, shape below colour, a
    non-convex sum and a negative weight are all refused, not clamped.
  * ``test_ring_distance_separates_where_the_cosine_does_not`` -- the shipped cosine on the unit
    six-vector puts a true pair and an impostor pair within 1e-3 of each other; the real distance
    with lambda_L = 0.25 separates them by an order of magnitude.  This is item 2's measurement,
    reduced to two numbers.
  * ``test_kernel_and_embed_agree`` -- the affine-cosine path the gauge and the colour cell use is
    the same function as the explicit form, to 1e-12 (including the circular axis).
  * ``test_curvature_is_graded_and_keys_are_not`` -- |den| has hundreds of distinct values where
    var_x has one, and attaching it does not move a single key.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]


def test_zero_weights_is_the_deployed_matrix():
    from matching import placement_blend as BL
    M = np.arange(16, dtype=np.float64).reshape(4, 4) / 16.0
    out = BL.blend_channels(M)
    assert out is M                                    # the object itself, not a copy
    assert np.array_equal(out, M)
    assert BL.blend_channels(M, w_shape=0.0, w_colour=0.0) is M
    assert BL.shipped()["weights"] == {"w_shape": 0.0, "w_colour": 0.0}


def test_blend_is_the_stated_equation():
    from matching import placement_blend as BL
    M = np.full((3, 3), 0.4)
    S = np.full((3, 3), 0.9)
    C = np.full((3, 3), 0.1)
    got = BL.blend_channels(M, shape=S, colour=C, w_shape=0.10, w_colour=0.05)
    want = (1.0 - 0.15) * 0.4 + 0.10 * 0.9 + 0.05 * 0.1
    assert np.allclose(got, want)


def test_refuses_what_the_ordering_forbids():
    from matching import placement_blend as BL
    M = np.zeros((2, 2))
    with pytest.raises(ValueError):
        BL.blend_channels(M, colour=M, w_colour=0.20)                  # above the cap
    with pytest.raises(ValueError):
        BL.blend_channels(M, shape=M, colour=M, w_shape=0.05, w_colour=0.10)   # shape < colour
    with pytest.raises(ValueError):
        BL.blend_channels(M, shape=M, colour=M, w_shape=0.90, w_colour=0.15)   # not convex
    with pytest.raises(ValueError):
        BL.blend_channels(M, shape=M, w_shape=-0.1)
    assert set(BL.REFUSALS) >= {"colour_cap", "ordering", "convexity", "negative"}


def test_ring_distance_separates_where_the_cosine_does_not():
    """Item 2's finding as a property: the cosine is flat, the real distance is not."""
    from matching import placement_content as PC
    true_pair = np.asarray([[0.5, 0.10, 0.10, 0.5, 0.10, 0.10]])
    near_impostor = np.asarray([[0.52, 0.13, 0.09, 0.51, 0.11, 0.09]])
    far_impostor = np.asarray([[0.5, 0.30, 0.10, 0.5, 0.30, 0.10]])
    cos = PC.similarity(true_pair, np.vstack([true_pair, near_impostor]))[0]
    assert cos[1] > 0.99, cos                       # the cosine cannot tell them apart
    d = PC.d2(true_pair, np.vstack([true_pair, near_impostor]))[0]
    assert d[0] == 0.0 and d[1] > 0.0
    d_far = PC.d2(true_pair, far_impostor)[0, 0]
    assert d_far > 20.0 * d[1], (d_far, d[1])       # a colour difference is not a rounding error
    # the lightness weight is what the mission fixed, and it applies to L only
    light = np.asarray([[0.6, 0.10, 0.10, 0.6, 0.10, 0.10]])
    assert PC.d2(true_pair, light)[0, 0] == pytest.approx(2 * 0.25 * 0.1 ** 2)


def test_sigma_is_deterministic_and_labelfree():
    from matching import placement_content as PC
    pool = np.random.default_rng(0).uniform(0.0, 1.0, size=(5000, 6))
    a = PC.sigma(pool)
    b = PC.sigma(pool)
    assert a == b > 0.0
    assert PC.sigma(pool, seed=PC.SIGMA_SEED + 1) != a          # the seed is the identity


def test_aggregators_differ_and_max_prefers_the_extreme():
    from matching import placement_content as PC
    rng = np.random.default_rng(3)
    Q = {"Blob": rng.normal(0.3, 0.02, size=(40, 6))}
    G = {"Blob": np.vstack([Q["Blob"][:20], rng.normal(0.9, 0.02, size=(20, 6))])}
    sig = PC.sigma(np.vstack([Q["Blob"], G["Blob"]]))
    v1 = PC.aggregate(Q, G, sig, "max")
    v2 = PC.aggregate(Q, G, sig, "coverage")
    v3 = PC.aggregate(Q, G, sig, "chamfer")
    assert v1 >= v3 >= v3 - 1e-12
    assert v1 != v2 and v3 != v2
    with pytest.raises(ValueError):
        PC.aggregate(Q, G, sig, "median")


def test_kernel_and_embed_agree():
    from matching import placement_shape as PS
    rng = np.random.default_rng(11)
    A = rng.normal(size=(30, 9))
    A[:, 4] = rng.uniform(0.0, float(PS.CYCLE[4]), 30)          # the circular axis
    B = rng.normal(size=(17, 9))
    B[:, 4] = rng.uniform(0.0, float(PS.CYCLE[4]), 17)
    sig = PS.scales(A)
    K1 = PS.kernel_matrix(A, B, sig)
    K2 = PS.kernel_block(A, B, sig)
    assert float(np.abs(K1 - K2).max()) < 1e-12
    assert np.allclose(np.diag(PS.kernel_block(A, A, sig)), 1.0)
    # a period-length wrap: 0 and CYCLE are the same place
    x = np.zeros((1, 9))
    y = np.zeros((1, 9))
    y[0, 4] = float(PS.CYCLE[4])
    assert PS.kernel_matrix(x, y, np.full(9, 1.0), weights=np.eye(9)[4] * 1.0)[0, 0] == pytest.approx(1.0)


def test_curvature_is_graded_and_keys_are_not():
    from lenses import keypoints as KP
    from matching import placement_frontend as FE
    from Topology import m26_records as REC
    o = REC.objects()[0]
    img = FE.load_image(REC.path_of(o, "035-045"))
    ks = KP.detect(img)
    assert len(ks) > 50
    var = {round(float(k.var_x), 9) for k in ks}
    sharp = {round(float(k.sharp_x), 9) for k in ks}
    assert len(var) == 1, var                       # the clip: one value for every place
    assert len(sharp) > 0.8 * len(ks), (len(sharp), len(ks))
    # the key does not read the curvature: same signatures, before and after
    sigs = [KP.signature_of(k) for k in ks]
    import dataclasses
    moved = [dataclasses.replace(k, sharp_x=0.123, sharp_y=0.456, curv_x=7.0, curv_y=8.0)
             for k in ks]
    assert [KP.signature_of(k) for k in moved] == sigs
