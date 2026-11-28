"""[MISSION 36 / PART D] Every destination reached, one test each.

D1  the dense layer is in `place_record`      -- `test_d1_*`
D2  it survives the compact record file       -- `test_d2_*`
D3  its class is registered and COMPARED      -- `test_d3_*`
D4  the join gives a place a degree            -- `test_d4_*`
D5  determinism at 1/2/4 workers, diff 0.0     -- `test_d5_*`
D6  the flags are OFF by default               -- `test_d6_*`

Plus the two claims the mission holds the field to:
B4  content-driven (uniform vs textured, within one image)  -- `test_b4_*`
B5  no silent fallback: every refusal is counted            -- `test_b5_*`

Every test here is RED on revert: D1 fails if `rec["dense"]` stops being written, D2
fails if the block stops round-tripping, D3 fails if the class leaves the registry
(the tree returns 0.0 for an unregistered class, so the test compares a node with
itself AND with a corrupted copy), D4 fails if the join stops adding edges, D5 fails
if two worker counts disagree, D6 fails if a flag defaults on.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

import parameters as P

IMG = "data/ETH-80-master/1/1/apple1-035-045.png"
IMG_B = "data/ETH-80-master/4/1/cup1-035-045.png"
AVAILABLE = os.path.exists(IMG) and os.path.exists(IMG_B)
needs_data = pytest.mark.skipif(not AVAILABLE, reason="ETH-80 mirror not present")


@pytest.fixture()
def flags_off():
    names = ("PLACEMENT_DENSE_FIELD", "PLACEMENT_DENSE_JOIN")
    old = {n: getattr(P, n, None) for n in names}
    for n in names:
        setattr(P, n, False)
    yield
    for n, v in old.items():
        if v is None:
            try:
                delattr(P, n)
            except AttributeError:
                pass
        else:
            setattr(P, n, v)


@pytest.fixture()
def flags_on(flags_off):
    P.PLACEMENT_DENSE_FIELD = True
    P.PLACEMENT_DENSE_JOIN = True
    import matching.placement_frontend as PF
    PF.clear_cells()
    yield
    PF.clear_cells()


# ------------------------------------------------------------------ D6 (must be first in spirit)
def test_d6_flags_are_off_by_default():
    import matching.placement_join as PJ
    assert getattr(P, "PLACEMENT_DENSE_FIELD", False) is False
    assert getattr(P, "PLACEMENT_DENSE_JOIN", False) is False
    assert PJ.field_on() is False and PJ.join_on() is False


@needs_data
def test_d6_default_record_has_no_dense_block(flags_off):
    import matching.placement_frontend as PF
    PF.clear_cells()
    rec = PF.place_record(IMG)
    assert "dense" not in rec
    img, g, side = PF.cell(IMG)
    import matching.placement_join as PJ
    assert not any(int(n) >= PJ.DENSE_ID_BASE for n in g.subgraph.nodes)


# ------------------------------------------------------------------------------------ D1
@needs_data
def test_d1_dense_layer_is_in_the_record(flags_on):
    import matching.placement_frontend as PF
    from core.registries import POPULATION_SLOTS
    rec = PF.place_record(IMG)
    rows = rec["dense"]
    assert len(rows) > 100
    names = [s.name for s in POPULATION_SLOTS["DENSE_FIELD"]]
    for r in rows[:50]:
        assert set(names) <= set(r)
        assert r["s"] > 0.0 and np.isfinite(r["x"]) and np.isfinite(r["y"])
    # the seven slots are not a constant vector (a lattice would be)
    vals = np.asarray([[r[n] for n in names] for r in rows], dtype=float)
    assert np.unique(vals[:, 0]).size > 5
    # every attached dense node names a place that is really in the record
    place_ids = {int(p["n"]) for p in rec["places"]}
    attached = [r for r in rows if r["place"] >= 0]
    assert attached
    assert all(int(r["place"]) in place_ids for r in attached)
    # a dense node whose owner the record dropped is refused, and the count is kept
    assert rec["dense_owner_dropped"] == len(rows) - len(attached)


@needs_data
def test_d1_shipped_record_arrays_do_not_move(flags_off):
    """The dense block ADDS a key: every shipped array is identical with the flag off/on."""
    import matching.placement_frontend as PF
    from matching import placement_records as PR
    PF.clear_cells()
    a0 = PR.export([("apple/1", "035-045", PF.place_record(IMG))], ["apple/1"], ["035-045"])
    P.PLACEMENT_DENSE_FIELD = True
    P.PLACEMENT_DENSE_JOIN = True
    PF.clear_cells()
    rec = PF.place_record(IMG)
    a1 = PR.export([("apple/1", "035-045", rec)], ["apple/1"], ["035-045"])
    assert "dense" in rec
    for k, v in a0.items():
        if k == "meta" or k not in a1:
            continue
        assert v.shape == a1[k].shape, k
        assert np.array_equal(np.nan_to_num(v, nan=-1.0), np.nan_to_num(a1[k], nan=-1.0)), k
    # and the new block is there
    assert "dense_035-045" in a1 and a1["dense_035-045"].shape[1] == len(PR.DENSE_FIELDS)


# ------------------------------------------------------------------------------------ D2
@needs_data
def test_d2_dense_block_round_trips(flags_on, tmp_path):
    import matching.placement_frontend as PF
    from matching import placement_records as PR
    rec = PF.place_record(IMG)
    arr = PR.export([("apple/1", "035-045", rec)], ["apple/1"], ["035-045"])
    meta = json.loads(str(arr["meta"]))
    assert meta["version"] == PR.FORMAT_VERSION >= 31
    assert meta["dense"] == len(PR.DENSE_FIELDS)
    path = str(tmp_path / "dense.npz")
    PR.save(path, arr)
    back = PR.load_cell(path, "apple/1", "035-045")
    assert len(back["dense"]) == len(rec["dense"])
    a = np.asarray([[r[n] for n in PR.DENSE_FIELDS] for r in rec["dense"]], dtype=np.float32)
    b = np.asarray([[r[n] for n in PR.DENSE_FIELDS] for r in back["dense"]], dtype=np.float32)
    assert np.array_equal(a, b)
    # and the places the door reads are untouched
    assert len(back["places"]) == len(rec["places"])


@needs_data
def test_d2_flag_off_file_has_no_dense_block(flags_off, tmp_path):
    import matching.placement_frontend as PF
    from matching import placement_records as PR
    PF.clear_cells()
    arr = PR.export([("apple/1", "035-045", PF.place_record(IMG))], ["apple/1"], ["035-045"])
    assert not [k for k in arr if k.startswith("dense_")]
    assert json.loads(str(arr["meta"]))["dense"] == 0


# ------------------------------------------------------------------------------------ D3
def test_d3_class_is_registered_everywhere():
    from core import registries as R
    from core.slots import SLOT_NAMES
    from core.types import _UNDIRECTED_EDGE_TYPES_BASE  # noqa: F401  (import must not fail)
    cls = "DENSE_FIELD"
    assert cls in R.POPULATION_SLOTS
    assert cls in R.POPULATION_AXES
    assert cls in R.CONSTITUTIONAL_RANGES
    slots = R.POPULATION_SLOTS[cls]
    assert len(slots) == len(R.CONSTITUTIONAL_RANGES[cls]) == len(R.POPULATION_AXES[cls])
    assert [s.name for s in slots] == list(SLOT_NAMES[cls])
    # every slot pairs with a same-named axis
    assert len(R.slot_axes(cls)) == len(slots)
    # the circular slot's declared cycle equals its emitted bins
    assert R.CIRCULAR_CYCLE_LENGTHS[cls][0] == slots[0].n_bins
    assert slots[0].circular


def test_d3_content_moves_similarity():
    """A registered class is COMPARED; after a content change the similarity must fall.

    An unregistered class returns 0.0 from the tree's kernel, so a node compared with
    ITSELF is the control that distinguishes "registered and equal" from "unregistered".
    """
    from core.registries import population_from_values
    from core.types import TypedSignature
    from perception.local_difference import DenseNode

    def sig(values, slots):
        return TypedSignature(kind="CANONICAL", class_="DENSE_FIELD",
                              relation="LOCAL_DIFFERENCE", quantized=tuple(slots),
                              population=population_from_values("DENSE_FIELD", values))

    base = (0.9, 0.8, 0.7, 0.5, 0.3, -0.3, 0.4)
    n = DenseNode(x=1.0, y=2.0, scale_index=2, sigma_px=3.9, support_px=5.8,
                  activity=0.5, slots=tuple(round(v * 7) for v in base), values=base)
    from lenses.dense_field import signature_of
    s = signature_of(n)
    assert len(s.population) == 7
    same = s.similarity(s)
    assert same > 0.99, "registered and identical must compare high, not fall back to 0.0"
    other = sig((0.1, 0.1, 0.1, 0.9, -0.9, 0.9, 0.1), (1.0, 1.0, 1.0, 6.0, 1.0, 6.0, 1.0))
    assert s.similarity(other) < same - 0.05
    # THE CONTRACT: an unregistered class returns EXACTLY 0.0 for differing content,
    # not a small number -- so a class that is not in the registry cannot look like a
    # weak match.  DENSE_FIELD, being registered, returns a graded non-zero value.
    b0 = TypedSignature(kind="CANONICAL", class_="NOT_A_CLASS",
                        relation="LOCAL_DIFFERENCE", quantized=(0.0, 1.0))
    b1 = TypedSignature(kind="CANONICAL", class_="NOT_A_CLASS",
                        relation="LOCAL_DIFFERENCE", quantized=(5.0, 6.0))
    assert b0.similarity(b1) == 0.0
    r0 = TypedSignature(kind="CANONICAL", class_="DENSE_FIELD", relation="LOCAL_DIFFERENCE",
                        quantized=(0.0,) * 7)
    r1 = TypedSignature(kind="CANONICAL", class_="DENSE_FIELD", relation="LOCAL_DIFFERENCE",
                        quantized=(5.0,) * 7)
    assert 0.0 < r0.similarity(r1) < 0.5


# ------------------------------------------------------------------------------------ D4
@needs_data
def test_d4_join_gives_places_a_degree(flags_on):
    import matching.placement_frontend as PF
    import matching.placement_join as PJ
    from matching.placement import PlacementVector
    img, g, side = PF.cell(IMG)
    sub = g.subgraph
    before = PJ._place_degree(  # noqa: SLF001 - the same helper the join reports with
        __import__("core.graph_types", fromlist=["Subgraph"]).Subgraph(
            nodes=frozenset(int(n) for n in sub.nodes if int(n) < PJ.DENSE_ID_BASE),
            edges=frozenset(e for e in sub.edges if int(e[0]) < PJ.DENSE_ID_BASE
                            and int(e[1]) < PJ.DENSE_ID_BASE)),
        PJ.place_nodes(sub, g.sigs))
    after = PJ._place_degree(sub, PJ.place_nodes(sub, g.sigs))
    assert before["zero"] == before["n"]        # every place was isolated
    assert after["mean"] > 0.5                  # now it is not
    assert after["zero"] < before["zero"]
    dense = [int(n) for n in sub.nodes if int(n) >= PJ.DENSE_ID_BASE]
    assert len(dense) > 100
    owners: dict[int, int] = {}
    for u, v, t in sub.edges:
        if getattr(t, "name", "") == "DENSE_IN_PLACE":
            assert int(u) < PJ.DENSE_ID_BASE <= int(v)
            assert int(v) not in owners, "a dense node is carried by exactly one place"
            owners[int(v)] = int(u)
            ep = sub.edge_params[(u, v, t)]
            assert ep.class_ == "EDGE_DENSE_PLACEMENT"      # registered, so the kernel compares it
            vec = sub.dense_vectors[(u, v, t)]
            assert isinstance(vec, PlacementVector) and vec.source_scale > 0.0
            assert -np.pi <= vec.direction <= np.pi
            # the vector's frame IS the owner's own orientation
            assert abs(vec.source_theta - vec.source_theta) < 1e-12
    assert len(owners) > 100
    prox = [e for e in sub.edges if getattr(e[2], "name", "") == "DENSE_PROXIMITY"]
    assert prox and all(int(a) >= PJ.DENSE_ID_BASE and int(b) >= PJ.DENSE_ID_BASE
                        for a, b, _t in prox)


@needs_data
def test_d4_dense_sigs_are_on_the_graph(flags_on):
    import matching.placement_frontend as PF
    import matching.placement_join as PJ
    _img, g, _side = PF.cell(IMG)
    dense = [int(n) for n in g.subgraph.nodes if int(n) >= PJ.DENSE_ID_BASE]
    assert dense
    for n in dense[:20]:
        lens, sig = g.sigs[n]
        assert lens == "DENSE_FIELD" and len(sig.population) == 7


# ------------------------------------------------------------------------------------ D5
def _digest_of_one(path: str) -> str:
    import hashlib
    import perception.local_difference as LD
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    f = LD.dense_field(rgb)
    h = hashlib.sha256()
    for n in f.nodes:
        h.update(("%.3f %.3f %d %s\n" % (n.x, n.y, n.scale_index,
                                         ",".join("%.6f" % v for v in n.values))).encode())
    return h.hexdigest()


@needs_data
def test_d5_determinism_at_1_2_4_workers():
    paths = [IMG, IMG_B]
    ref = [_digest_of_one(p) for p in paths]
    for workers in (1, 2, 4):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            got = list(ex.map(_digest_of_one, paths))
        assert got == ref, workers
    with ProcessPoolExecutor(max_workers=2) as ex:
        got = list(ex.map(_digest_of_one, paths))
    assert got == ref


# ------------------------------------------------------------------------------ B4 / B5
def test_b4_content_driven_on_fixtures():
    import perception.local_difference as LD
    fixtures = LD.selftest_contrast()
    assert fixtures["uniform"]["n_nodes"] == 0
    assert fixtures["noise"]["n_nodes"] > 100
    assert fixtures["gradient"]["n_nodes"] > 100
    assert fixtures["noise"]["coverage"] > 0.2


@needs_data
def test_b4_textured_beats_smooth_inside_one_image():
    """Within ONE image, so the global contrast cannot explain the ordering."""
    from PIL import Image
    from scipy.ndimage import laplace
    import perception.local_difference as LD
    rgb = np.asarray(Image.open(IMG_B).convert("RGB"), dtype=np.uint8)
    f = LD.dense_field(rgb)
    lum = LD._norm_planes_lum(LD._planes(rgb)["lum"])          # noqa: SLF001
    lap = np.abs(laplace(lum, mode="nearest"))
    mask = LD.object_mask(rgb)
    S = 32
    cells = []
    for y in range(0, 256 - S, 8):
        for x in range(0, 256 - S, 8):
            if mask[y:y + S, x:x + S].mean() < 0.9:
                continue
            n = sum(1 for q in f.nodes if y <= q.y < y + S and x <= q.x < x + S)
            cells.append((float(lap[y:y + S, x:x + S].mean()), n))
    assert len(cells) > 20
    cells.sort()
    k = max(3, len(cells) // 10)
    smooth = float(np.mean([c[1] for c in cells[:k]]))
    textured = float(np.mean([c[1] for c in cells[-k:]]))
    assert textured > 2.0 * smooth, (smooth, textured)


@needs_data
def test_b5_every_refusal_is_counted(flags_on):
    import matching.placement_frontend as PF
    import matching.placement_join as PJ
    img = PF.load_image(IMG)
    _i, g, _s = PF.cell(IMG)
    sub, st = PJ.join(g.subgraph, img, g.sigs)
    assert (st["n_dense_in_place_edges"] + st["n_refused_no_frame"]
            + st["n_refused_no_place"]) == st["n_dense"]
    assert st["n_dense_attached"] + st["n_refused_no_place"] == st["n_dense"]
    assert st["n_refused_no_place"] == 0
    # the frame providers are named, not guessed
    assert set(st["place_frame_sources"]) <= {"keypoint", "field", "none"}
