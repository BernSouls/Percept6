"""M34 item 2: the fragment descriptor is registered, frame-relative, and refuses when unregistered.

Every test here fails if the mechanism it names is reverted:

  * `test_unregistered_class_returns_zero` -- delete the `register()` call and the block comes back
    non-zero, so the test fails.  This is the M19 rule one level up: a class the registries do not
    know may not answer with a small number.
  * `test_descriptor_is_frame_invariant` -- replace the fragment frame (dropping the principal-axis
    subtraction from `phi`, or the centroid from `rel`) and the rotated copy no longer matches.
  * `test_grouping_is_order_independent` / `..._is_deterministic` -- a set-based grouping that leaks
    input order fails.
  * `test_anchor_radius_is_absolute` -- a place outside the registered radius must join nothing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from matching import placement_fragment as PF


def _places(n: int = 6, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({"x": float(rng.uniform(20, 200)), "y": float(rng.uniform(20, 200)),
                    "s": float(rng.uniform(4, 12)),
                    "sharp_x": float(rng.normal()), "sharp_y": float(rng.normal()),
                    "curv_x": float(rng.normal()), "curv_y": float(rng.normal()),
                    "content": list(rng.uniform(0, 1, 6))})
    return out


def _transform(places: list, angle: float, dx: float, dy: float) -> list:
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for p in places:
        q = dict(p)
        q["x"] = ca * p["x"] - sa * p["y"] + dx
        q["y"] = sa * p["x"] + ca * p["y"] + dy
        out.append(q)
    return out


def test_class_is_registered_with_every_axis() -> None:
    assert PF.missing_fields() == [], PF.missing_fields()
    r = PF.registered()
    assert len(r["ranges"]) == len(PF.FIELDS) == 22
    assert len(r["weights"]) == 22
    assert r["axes"] == list(PF.FIELDS)


def test_unregistered_class_returns_zero() -> None:
    pool = np.asarray([PF.full_descriptor(_places(6, s), list(range(6))) for s in range(5)])
    sig = PF.scales(pool)
    try:
        PF.unregister()
        block = PF.similarity(pool, pool, sig)
        assert block.shape == (5, 5)
        assert float(np.abs(block).max()) == 0.0
        assert PF.missing_fields() != []
    finally:
        PF.register()
    assert float(np.abs(PF.similarity(pool, pool, sig)).max()) > 0.0
    assert PF.missing_fields() == []


def test_descriptor_is_frame_invariant() -> None:
    """Same fragment, moved and turned: every one of the 22 numbers is unchanged."""
    places = _places(8, seed=3)
    members = list(range(len(places)))
    a = PF.full_descriptor(places, members)
    moved = _transform(places, angle=0.7, dx=61.0, dy=-37.0)
    b = PF.full_descriptor(moved, members)
    turned = _descriptor_after_turn(places, members, 0.9)
    assert np.allclose(a, b, atol=1e-9), np.abs(a - b).max()
    assert np.allclose(a, turned, atol=1e-9), np.abs(a - turned).max()


def _descriptor_after_turn(places: list, members: list, angle: float) -> np.ndarray:
    """Turn the MEMBERS about their own centroid -- the frame the descriptor claims to use."""
    cx = float(np.mean([places[i]["x"] for i in members]))
    cy = float(np.mean([places[i]["y"] for i in members]))
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for p in places:
        q = dict(p)
        dx, dy = p["x"] - cx, p["y"] - cy
        q["x"], q["y"] = ca * dx - sa * dy + cx, sa * dx + ca * dy + cy
        out.append(q)
    return PF.full_descriptor(out, members)


def test_grouping_is_deterministic_and_order_independent() -> None:
    places = _places(12, seed=7)
    a = PF.group_by_proximity(places)
    b = PF.group_by_proximity(places)
    assert a == b
    order = np.random.default_rng(11).permutation(len(places))
    shuffled = [places[int(i)] for i in order]
    c = PF.group_by_proximity(shuffled)
    as_sets = sorted(tuple(sorted((round(places[i]["x"], 6), round(places[i]["y"], 6))
                                  for i in g)) for g in a)
    as_sets_shuffled = sorted(tuple(sorted((round(shuffled[i]["x"], 6),
                                            round(shuffled[i]["y"], 6)) for i in g)) for g in c)
    assert as_sets == as_sets_shuffled


def test_anchor_radius_is_absolute() -> None:
    places = [{"x": 10.0, "y": 10.0, "s": 5.0}, {"x": 200.0, "y": 200.0, "s": 5.0}]
    groups = PF.group_by_anchor(places, [(10.0, 10.0)], radius=PF.ANCHOR_RADIUS)
    assert groups == [[0]], groups           # the far place joins NOTHING, it is not pulled in


def test_descriptor_arity_matches_the_registry() -> None:
    d = PF.full_descriptor(_places(4), [0, 1, 2, 3])
    assert d.shape == (len(PF.FIELDS),)
    assert np.all(np.isfinite(d))
    with pytest.raises(ValueError):
        PF.describe(_places(4), [0, 1, 2, 3], fields=PF.FIELDS[:-1])
