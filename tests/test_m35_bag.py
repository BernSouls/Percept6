"""M35: the object-level descriptor, the bag door, and the class output.

Each test fails if the mechanism it names is reverted:

  * `test_every_block_has_the_registered_arity` -- the block registry and `describe` must agree;
    add a block to one and not the other and this goes red.
  * `test_blocks_are_hellinger_normalised` -- L1 then sqrt.  Dropping the sqrt (or the L1) leaves a
    dot product that is a COUNT, not a Bhattacharyya coefficient, and this goes red.
  * `test_excluded_columns_are_not_read` -- corrupt `var_x`, `var_y`, `rel_angle` and the descriptor
    must not move a bit.  Reading any of them turns this red.
  * `test_missing_edges_refuse` / `test_missing_fields_refuse` -- no silent refit, no zero-fill.
  * `test_door_is_the_item1_colour_ranking` -- `recognise()` with the bag bundle reproduces the
    item-1 ranking on every one of the 160 cells.  Change the door's arm or its aggregation and this
    goes dark red.
  * `test_recognise_class_excludes_the_query_object` -- the class output may not be answered by the
    query's own descriptor (1.0) pinning its own class.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
REF = "000-000"
VIEWS = ("022-000", "035-045", "045-180")


@pytest.fixture(scope="module")
def cells():
    from matching import placement_records as PR
    return PR.load_cells(RECORDS, [REF, *VIEWS])


@pytest.fixture(scope="module")
def objs():
    from Topology import m26_records as REC
    return REC.objects()


@pytest.fixture(scope="module")
def edges():
    from matching import placement_bag as BAG
    return BAG.load_edges()


def test_every_block_has_the_registered_arity(objs, cells, edges) -> None:
    from matching import placement_bag as BAG
    f = BAG.describe(cells[(objs[0]["key"], REF)], edges)
    assert set(f) == set(BAG.BLOCKS)
    for k, n in BAG.BLOCKS.items():
        assert f[k].size == n, (k, f[k].size, n)
    # the arm registry names only registered groups, and every group only registered blocks
    for arm, groups in BAG.ARMS.items():
        assert groups, arm
        for g in groups:
            assert g in BAG.GROUPS, (arm, g)
            for b in BAG.GROUPS[g]:
                assert b in BAG.BLOCKS, (g, b)


def test_blocks_are_hellinger_normalised(objs, cells, edges) -> None:
    from matching import placement_bag as BAG
    f = BAG.describe(cells[(objs[1]["key"], "035-045")], edges)
    for k, v in f.items():
        s = float(np.sum(v ** 2))
        assert abs(s - 1.0) < 1e-9 or s == 0.0, (k, s)
        assert np.all(v >= 0.0)
    # a block dot product of a vector with itself is 1, i.e. a Bhattacharyya coefficient
    assert abs(float(f["ab1"] @ f["ab1"]) - 1.0) < 1e-12


def test_excluded_columns_are_not_read(objs, cells, edges) -> None:
    from matching import placement_bag as BAG
    assert set(BAG.EXCLUDED) == {"var_x", "var_y", "rel_angle"}
    k = objs[2]["key"]
    rec = cells[(k, REF)]
    base = BAG.describe(rec, edges)
    bad = {"places": [dict(p) for p in rec["places"]]}
    for i, p in enumerate(bad["places"]):
        p["var_x"] = 1e6 if i % 2 else -1e6
        p["var_y"] = 1e6 if i % 2 else -1e6
        p["rel_angle"] = 3.0 if i % 2 else -3.0
        for fld in ("var_x", "var_y", "rel_angle"):
            assert fld in p
    got = BAG.describe(bad, edges)
    for b in BAG.BLOCKS:
        assert np.array_equal(base[b], got[b]), b


def test_missing_edges_refuse(tmp_path) -> None:
    from matching import placement_bag as BAG
    with pytest.raises(BAG.BagRefusal) as exc:
        BAG.load_edges(str(tmp_path / "nope.json"))
    assert "refusal" in str(exc.value)
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"version": 1}))
    with pytest.raises(BAG.BagRefusal):
        BAG.load_edges(str(p))


def test_missing_fields_refuse(objs, cells, edges) -> None:
    from matching import placement_bag as BAG
    k = objs[3]["key"]
    rec = cells[(k, REF)]
    nocontent = {"places": [{kk: vv for kk, vv in p.items() if kk != "content"}
                            for p in rec["places"]]}
    with pytest.raises(BAG.BagRefusal):
        BAG.describe(nocontent, edges)
    nodz = {"places": [{kk: vv for kk, vv in p.items() if kk not in ("Dz", "Mz")}
                       for p in rec["places"]]}
    with pytest.raises(BAG.BagRefusal):
        BAG.describe(nodz, edges)
    with pytest.raises(BAG.BagRefusal):
        BAG.describe({"places": []}, edges)


def test_edges_carry_their_provenance(edges) -> None:
    assert edges["ref_view"] == REF
    assert int(edges["objects"]) == 80
    prov = edges["provenance"]
    assert prov["mission"].startswith("M35")
    assert prov["view"] == REF
    assert int(prov["objects"]) == 80
    assert "quantiles" in prov["rule"]


def test_bundle_ships_the_bag_door_and_no_fragment_vote() -> None:
    from matching import placement_bundle as PB
    b = PB.PlacementBundle.load()
    assert b.door_kind == "bag", b.door_kind
    assert b.colour_rings is True
    # item 5: the fragment channel did not add its +0.025 on top of the bag door, so its weight is 0
    assert float(b.fragment_weight) == 0.0
    assert str(b.door_params.get("arm")) == "COLOUR"


def test_door_is_the_item1_colour_ranking(objs, cells, edges) -> None:
    """`recognise()` on the bag bundle == the item-1 COLOUR ranking, on all 160 cells."""
    from matching import placement_bag as BAG
    from matching import placement_gallery as GL
    from matching import placement_recognise as R
    keys = [o["key"] for o in objs]
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    gal.bag_features = None
    b = _bundle()
    bag_feats = [BAG.describe(cells[(k, REF)], edges) for k in keys]
    gal.bag_features = bag_feats
    for v in ("035-045", "045-180"):
        qf = [BAG.describe(cells[(k, v)], edges) for k in keys]
        for i, k in enumerate(keys):
            res = R.recognise((k, ""), gal, bundle=b, record=cells[(k, v)], matcher=False,
                              use_pair_cache=False)
            want = BAG.door_scores(qf[i], bag_feats, len(keys), 1)
            order = sorted(range(len(keys)), key=lambda j: (-want[j], j))
            assert res.door_order == order, (v, k)
            assert len(res.door_order) == 80


def test_recognise_class_excludes_the_query_object(objs, cells, edges) -> None:
    from matching import placement_bag as BAG
    from matching import placement_gallery as GL
    from matching import placement_recognise as R
    keys = [o["key"] for o in objs]
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    gal.bag_features = [BAG.describe(cells[(k, REF)], edges) for k in keys]
    b = _bundle()
    k = objs[5]["key"]
    out = R.recognise_class((k, ""), gal, bundle=b, record=cells[(k, REF)])
    cls = [kk.split("/")[0] for kk in keys]
    assert out["excluded"] == k
    assert np.isnan(out["objects"][keys.index(k)])
    # the ranking is the ALL arm over the top-2 instance means
    keep = [j for j in range(len(keys)) if keys[j] != k]
    obj = np.asarray(out["objects"])[keep]
    want = BAG.class_scores(obj, np.asarray([cls[j] for j in keep], dtype=object), top=2)
    assert [c for c, _ in want] == [c for c, _ in out["classes"]]
    assert out["top"] == want[0][0]


def _bundle():
    from matching import placement_bundle as PB
    return PB.PlacementBundle.load()
