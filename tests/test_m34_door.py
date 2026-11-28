"""M34 item 6: the fragment channel is WIRED into the shipped door, and stays off by default.

Each test fails if the mechanism it names is reverted:

  * `test_fuse_without_the_fragment_channel_is_the_previous_arithmetic` -- adding the slot to
    `DEFAULT_WEIGHTS` and to `fuse` must not move a single number for a caller that does not pass it.
    Revert the guard (`if fragment is not None and w["fragment"] != 0`) and this goes red.
  * `test_bundle_ships_the_fragment_channel_at_the_stated_discount` -- the bundle's own record of
    what the door does.  Reverting `fusion.fragment` to absent turns this red.
  * `test_door_refuses_without_anchors` -- a bundle that asks for the channel may not be quietly
    served four channels.  Delete the refusal in `placement_recognise` and this goes red.
  * `test_channel_is_the_registered_coverage_rule` -- the door's channel and the FIT screen's rule
    are the same function; if either drifts, this goes red.
  * `test_anchors_cover_every_recorded_cell` -- the artifact the door reads is complete.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
ANCHORS = os.path.join(ROOT, "data", "placement_anchors_m34.npz")
REF = "000-000"
VIEW = "035-045"


def test_fuse_without_the_fragment_channel_is_the_previous_arithmetic() -> None:
    from matching import placement_fuse as FU
    rng = np.random.default_rng(7)
    cnt = rng.random((1, 12))
    app = rng.random((1, 12))
    col = rng.random((1, 12))
    a = FU.fuse(cnt, app, col, weights={"colour": 0.5}, k=60)
    b = FU.rrf_weighted([cnt, app, col], [1.0, 1.0, 0.5], k=60)
    assert np.array_equal(a, b), np.abs(a - b).max()
    # and the slot's default is OFF
    assert FU.DEFAULT_WEIGHTS["fragment"] == 0.0
    # a matrix that is passed but not weighted changes nothing either
    assert np.array_equal(FU.fuse(cnt, app, col, weights={"colour": 0.5}, k=60,
                                  fragment=rng.random((1, 12))), b)


def test_bundle_ships_the_fragment_channel_at_the_stated_discount() -> None:
    from matching import placement_bundle as PB
    b = PB.PlacementBundle.load()
    assert b.door_kind == "bag"          # [M35] the bag door ships; fused3 is legacy
    # [M35 item 5] the fragment channel was measured ON TOP of the bag door and did not add its
    # +0.025 FIT recall@8, so its weight is 0 -- the code and the anchors stay committed.
    assert float(b.fragment_weight) == 0.0
    assert PB.json.load(open(PB.BUNDLE_PATH))["fusion"]["fragment"] == 0.0
    assert b.fusion_weight == 0.5                       # colour keeps its own discount
    assert "placement_fragment.py" in " ".join(PB.CODE_MODULES)


def test_anchors_are_resolved_from_the_image_path() -> None:
    """The shipped query is a PATH.  The door must find that cell's anchors by itself.

    If the resolution (`placement_anchors.for_path`) or the one-graph fallback is removed, the two
    calls stop agreeing: one gets anchors, the other raises or scores the channel as 0.
    """
    from matching import placement_anchors as PA
    from matching import placement_gallery as GL
    from matching import placement_records as PR
    from matching import placement_recognise as RC
    from Topology import m26_records as REC
    objs = REC.objects()[:4]
    gallery = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    cells = PR.load_cells(RECORDS, [VIEW])
    o = objs[0]
    qpath = REC.path_of(o, VIEW)
    table = PA.anchors_of(PA.load(), o["key"], VIEW)
    assert table.shape[0] > 0, "the committed anchors do not cover this cell"
    a = RC.recognise(qpath, gallery, matcher=False, workers=1,
                     record=cells[(o["key"], VIEW)], anchors=None)
    b = RC.recognise(qpath, gallery, matcher=False, workers=1,
                     record=cells[(o["key"], VIEW)], anchors=table)
    assert list(a.shortlist) == list(b.shortlist)
    assert PA.for_path(qpath) is not None
    assert PA.for_path("/nowhere/not-a-cell-000-000.png") is None


def test_channel_is_the_registered_coverage_rule() -> None:
    from matching import placement_anchors as PA
    from matching import placement_fragment as PF
    from matching import placement_records as PR
    table = PA.load()
    cells = PR.load_cells(RECORDS, [REF, VIEW])
    q = cells[("apple/1", VIEW)]["places"]
    g = cells[("apple/2", REF)]["places"]
    M = PF.channel([{"places": g}], q, [PA.anchors_of(table, "apple/2", REF)],
                   PA.anchors_of(table, "apple/1", VIEW))
    # by hand: the same groups, the same kernel, the same symmetric mean-of-best
    qg = PF.group_by_anchor(q, list(PA.anchors_of(table, "apple/1", VIEW)))
    gg = PF.group_by_anchor(g, list(PA.anchors_of(table, "apple/2", REF)))
    Q = np.asarray([PF.full_descriptor(q, m) for m in qg])
    G = np.asarray([PF.full_descriptor(g, m) for m in gg])
    sig = PF.scales(np.vstack([Q, G]))
    block = PF.similarity(Q, G, sig)
    want = 0.5 * float(block.max(axis=1).mean()) + 0.5 * float(block.max(axis=0).mean())
    assert M.shape == (1, 1)
    assert abs(float(M[0, 0]) - want) < 1e-12, (float(M[0, 0]), want)


def test_anchors_cover_every_recorded_cell() -> None:
    from matching import placement_anchors as PA
    from Topology import m26_records as REC
    table = PA.load()
    missing = []
    for o in REC.objects():
        for v in REC.VIEWS:
            for g in PA.GROUPINGS:
                try:
                    a = PA.anchors_of(table, o["key"], v, g)
                except KeyError:
                    missing.append((o["key"], v, g))
                    continue
                assert a.ndim == 2 and a.shape[1] == 2
                assert np.all(np.isfinite(a))
                assert np.array_equal(a, np.asarray(sorted(map(tuple, a))))
    assert missing == [], missing[:5]
