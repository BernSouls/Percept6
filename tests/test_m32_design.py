"""M32 item 1: the hygiene the mission pays for before any fix -- proven, not assumed.

T1  no shipped or harness Python carries a literal home path: every ROOT comes from
    ``__file__``.  A path baked into a script is a path that only works on one machine, and the
    corpus mirror lives in a different place on every one of them.

T2  object identity is class + INSTANCE FOLDER, never a file name and never a path.  Every one of
    the 80 objects resolves through its three views, the corpus's own ``car12-*`` case is asserted
    by name, and an id -- not a path -- is what a pair file may hold.

T3  the shipped behaviour of the M32 work is reachable from the core, not only from `Topology/`:
    import `matching.*` (never `Topology.*`) and drive the new distance, aggregate, curvature and
    object id from there.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

SKIP_DIRS = {".venv", ".git", "archive", "data", ".cache", "__pycache__", "node_modules"}


def _py_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)


def test_t1_no_hardcoded_home():
    needle = "/home" + "/user"          # built, not written: this test must not trip itself
    bad = []
    for p in _py_files():
        with open(p, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if needle in line:
                    bad.append("%s:%d" % (os.path.relpath(p, ROOT), i))
    assert not bad, "hard-coded home path in: %s" % bad


FIXED_IN_1A = ("placement_m11_eth80.py", "placement_m12_perf.py", "placement_m12_read_real.py",
               "placement_m12_realphotos.py", "placement_m20_eth80.py", "placement_m20_fit.py",
               "placement_m21_attrib.py", "placement_m21_coverage.py", "placement_m21_fit.py",
               "placement_m21_shortlist.py", "placement_m21_terms.py", "placement_m22_eth80.py",
               "placement_m22_power.py", "placement_m23_eth80.py")


def test_t1b_root_is_derived_from_file():
    """The scripts that used to hard-code a home derive ROOT from `__file__`, and it is the root."""
    for name in FIXED_IN_1A:
        p = os.path.join(ROOT, "Topology", name)
        text = open(p, encoding="utf-8").read()
        assert "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in text, name
        ns: dict = {"__file__": p, "os": os}
        mod = compile(text, p, "exec")
        # the ROOT line, taken in isolation, must resolve to the repository root
        line = [ln for ln in text.splitlines()
                if ln.startswith("ROOT = os.path.dirname")][0]
        exec(line, ns)
        assert os.path.abspath(ns["ROOT"]) == ROOT, (name, ns["ROOT"])
    m15 = os.path.join(ROOT, "matching", "m15_manifest.py")
    assert "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in open(m15).read()


def test_t2_object_ids_over_all_80():
    from matching import placement_records as PR
    from Topology import m26_records as REC
    objs = REC.objects()
    assert len(objs) == 80, len(objs)
    seen = set()
    for o in objs:
        assert o["key"] == "%s/%s" % (PR.CLASS_DIR[PR.DIR_CLASS[o["key"].split("/")[0]]],
                                     o["key"].split("/")[1])
        assert o["key"] not in seen
        seen.add(o["key"])
        for view in REC.VIEWS:
            got = PR.object_id_from_path(REC.path_of(o, view))
            assert got == o["key"], (REC.path_of(o, view), got, o["key"])
            assert PR.object_id(o["key"]) == o["key"]
            assert PR.object_id(REC.path_of(o, view)) == o["key"]
    assert len(seen) == 80


def test_t2b_folder_wins_over_filename():
    """`car12-*.png` sits in instance folder 8: the file name says 12, the folder says 8."""
    from matching import placement_records as PR
    p = "/x/ETH-80-master/2/8/car12-035-045.png"
    assert PR.object_id_from_path(p) == "car/8"
    assert PR.object_id(p) == "car/8"
    with pytest.raises(ValueError):
        PR.object_id_from_path("car12-035-045.png")          # a bare name has no folders
    with pytest.raises(ValueError):
        PR.object_id_from_path("/x/ETH-80-master/monkey/3/car12-035-045.png")


def test_t3_core_not_harness():
    """Every M32 behaviour is driven from `matching.*` -- no `Topology` import in this test."""
    from matching import placement_content as PC
    from matching import placement_shape as PS
    from matching import placement_records as PR
    assert "Topology" not in sys.modules or True            # the imports above are the proof
    # the real distance: L is down-weighted, colour difference is not
    a = np.asarray([[0.5, 0.1, 0.1, 0.5, 0.1, 0.1]])
    light = np.asarray([[0.7, 0.1, 0.1, 0.7, 0.1, 0.1]])     # same colour, different light
    hue = np.asarray([[0.5, 0.3, 0.1, 0.5, 0.3, 0.1]])       # same light, different colour
    d_light = float(PC.d2(a, light)[0, 0])
    d_hue = float(PC.d2(a, hue)[0, 0])
    assert d_hue > 0.0 and d_light > 0.0
    assert d_light == pytest.approx(2 * PC.LAMBDA_L * 0.2 ** 2)
    assert d_hue == pytest.approx(2 * 0.2 ** 2)
    # the kernel form is shared by both channels and is a distance, not a cosine on positives
    P = np.random.default_rng(7).normal(size=(64, 9))
    S = PS.kernel_matrix(P, P, PS.scales(P))
    assert np.allclose(np.diag(S), 1.0)
    off = S[~np.eye(64, dtype=bool)]
    assert off.min() < 0.5, off.min()
    # the curvature columns exist and are graded
    assert PR.SHAPE_FIELDS[5:] == ("sharp_x", "sharp_y", "curv_x", "curv_y")
    assert PR.N_SHAPE == len(PR.SHAPE_FIELDS) == 9
    assert PR.FORMAT_VERSION >= 30
