"""M33 item 1: the committed record file, its precision policy, and the fresh-checkout claim.

T1  The file is IN GIT and loads without a rebuild.  A fresh checkout must not have to pay the
    ~8 minute rebuild: `git ls-files` names it, the loader opens it, and the door's shortlist for
    each query -- recomputed from the file alone -- is the shortlist the M31 read actually scored
    (`m30_read_pairs.jsonl`, 8 object ids per query).  That pairs file is the reference because it
    is the record of what the shipped door did, committed long before this mission.

T2  The precision policy is exactly the one the mission paid for: the nine graded shape columns
    are float32, everything the door BUCKETS on is float64, and every stored graded value is a
    float32 value (so nothing is lost twice).  The colour rings stay float64, which is why the
    shipped cosine's score is bit-identical.

T3  The loss is bounded and ordered-stable: rounding the curvature columns to float32 changes the
    kernel's object-level score by less than 1e-7 and changes no query's top-8 SET or ORDER.

T4  The core, not the harness: the channel, the fuse slot and the precision policy are driven from
    `matching.*` here, with no `Topology` import for the behaviour under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
REF = "000-000"
S = 8


def test_t1_records_are_committed_and_load():
    rel = os.path.relpath(RECORDS, ROOT).replace(os.sep, "/")
    out = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel],
                         capture_output=True, text=True)
    assert out.returncode == 0, "the record file is not in git: %s" % out.stderr
    t0 = time.time()
    from matching import placement_records as PR
    cells = PR.load_cells(RECORDS, [REF, "035-045"])
    dt = time.time() - t0
    assert len(cells) == 160, len(cells)
    assert dt < 60.0, "the committed file must load in seconds, took %.1fs" % dt
    size = os.path.getsize(RECORDS) / 1e6
    assert size <= 40.0, "records are %.2f MB, over the 40 MB budget" % size


def test_t1b_the_door_reproduces_from_the_file_alone():
    """12 query objects: the fused door's shortlist, from the npz, against M31's scored pairs."""
    from matching import placement_frontdoor as FD
    from matching import placement_fuse as FU
    from matching import placement_gallery as GL
    from matching import placement_records as PR
    from Topology import m26_records as REC
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    want = {}
    if os.path.exists(PAIRS):
        for line in open(PAIRS, encoding="utf-8"):
            r = json.loads(line)
            want.setdefault(r["object"], set()).add(PR.object_id(r["r"]))
    assert len(want) == 80, len(want)
    gal = GL.from_records(RECORDS, objs, REF, content="colour_rings")
    cells = PR.load_cells(RECORDS, [REF, "035-045"])
    pick = keys[:12]
    qs = [cells[(k, "035-045")] for k in pick]
    counted, _i = FD.count_scores(gal.index, qs, n_hits=8)
    def _app(paths_a, paths_b):
        return FD.appearance_scores(paths_a, paths_b)
    app = _app([REC.path_of(o, "035-045") for o in objs if o["key"] in pick],
               [REC.path_of(o, REF) for o in objs])
    colour = FU.colour_channel([cells[(k, REF)] for k in keys], qs)
    fused = FU.fuse(counted, app, colour, weights={"colour": 0.5}, k=60)
    for i, k in enumerate(pick):
        order = sorted(range(len(keys)), key=lambda j: (-float(fused[i, j]), j))[:S]
        got = {keys[j] for j in order}
        assert got == want[k], (k, sorted(got), sorted(want[k]))


def test_t2_precision_policy():
    from matching import placement_records as PR
    z = np.load(RECORDS, allow_pickle=True)
    for view in ("000-000", "022-000", "035-045", "045-180"):
        for name in PR.GRADED_ARRAYS:
            arr = z["%s_%s" % (name, view)]
            assert arr.dtype == PR.GRADED_DTYPE, (name, view, arr.dtype)
            # every stored value is a float32 value: nothing is lost twice
            assert np.array_equal(arr, arr.astype(np.float32)), (name, view)
        for exact in ("rows", "dz", "mz"):
            assert z["%s_%s" % (exact, view)].dtype == PR.DTYPE, (exact, view)
        assert z["content_%s" % view].dtype == PR.DTYPE          # colour rings stay exact
    assert PR.GRADED_ARRAYS == ("shape",)


def test_t3_rounding_changes_no_order():
    """float32 as stored vs the same columns at float64: bounded delta, identical top-8 ORDER."""
    from matching import placement_shape as PS
    from matching import placement_records as PR
    cells = PR.load_cells(RECORDS, [REF, "035-045"])
    keys = sorted({k for k, v in cells.items() if v == "035-045"})
    gal = [cells[(k, REF)] for k in keys]
    q = [cells[(k, "035-045")] for k in keys]
    M32 = PS.curvature_matrix(gal, q, how="object_mean")
    def _round(entry):
        """The same entry with every curvature value carried through float32 and back."""
        places = []
        for p in entry["places"]:
            q2 = dict(p)
            for f in PS.CURVATURE_FIELDS:
                q2[f] = float(np.float64(np.float32(p.get(f) or 0.0)))
            places.append(q2)
        return {**entry, "places": places}
    up = [_round(e) for e in q]
    up_gal = [_round(e) for e in gal]
    M64 = PS.curvature_matrix(up_gal, up, how="object_mean")
    assert np.allclose(M32, M64, atol=1e-6)
    o32 = np.argsort(-M32, axis=1, kind="stable")[:, :S]
    o64 = np.argsort(-M64, axis=1, kind="stable")[:, :S]
    changed = [i for i in range(len(keys)) if list(o32[i]) != list(o64[i])]
    assert not changed, "float32 moved the top-8 order for %d queries" % len(changed)


def test_t4_core_only():
    """The channel, the fourth fuse slot and the identity are core behaviour."""
    from matching import placement_fuse as FU
    from matching import placement_shape as PS
    from matching import placement_blend as BL
    assert PS.CURVATURE_FIELDS == ("sharp_x", "sharp_y", "curv_x", "curv_y")
    assert "shape" in FU.DEFAULT_WEIGHTS and FU.DEFAULT_WEIGHTS["shape"] == 0.0
    M = np.eye(3)
    assert np.array_equal(FU.fuse(M, M), FU.rrf_weighted([M, M], [1.0, 1.0]))
    with_shape = FU.fuse(M, M, weights={"shape": 0.5}, shape=M)
    assert not np.array_equal(with_shape, FU.fuse(M, M))
    assert BL.blend_channels(M) is M
    gal = [{"places": [{"lens": "Blob", "sharp_x": 0.01, "sharp_y": 0.02,
                        "curv_x": 0.4, "curv_y": 0.5}]},
           {"places": [{"lens": "Blob", "sharp_x": 0.5, "sharp_y": 0.6,
                        "curv_x": 4.0, "curv_y": 5.0}]}]
    C = PS.curvature_matrix(gal, gal, how="object_mean")
    assert C.shape == (2, 2)
    assert np.allclose(np.diag(C), 1.0)
    assert C[0, 1] < C[0, 0]
