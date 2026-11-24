"""Mission 14 audit pin: every switch the payload builder reads MUST key the cache.

`Topology/placement_cache.key_for` folds in `eth80_frontend.switch_suffix()`, which
is built from `_FRONTEND_SWITCHES`.  A switch that changes the graph but is absent
from that tuple lets a cached graph built under one value be served under another --
the defect Mission 8 found for `FE_PLACE_LAYER` ("a dark run could read a lit
graph") and Mission 14 found again on the payload path, where six
`PLACEMENT_*` switches were read through `_param(name, default)` and declared
nowhere at all.

This test fails if a build-path file starts reading a `FE_*`/`PLACEMENT_*` switch
that is not in the digest, so the hole cannot reopen silently.
"""
from __future__ import annotations

import ast
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Every module that runs INSIDE a graph build (`Topology/placement_cache`).
BUILD_PATH = (
    "matching/placement_edges.py",
    "matching/placement_context.py",
    "matching/placement_matcher.py",
    "matching/placement_llr.py",
    "matching/placement_scorer.py",
    "matching/relation_frame.py",
    "matching/feature_attachment.py",
    "lenses/base.py",
    "lenses/keypoints.py",
    "core/place_key.py",
    "orchestration/binder.py",
)


def _digest_names():
    """The tuple's string constants, read with `ast` -- not with a regex.

    A regex on ")\n" stops at the tuple's own comments ("the payloads (M11
    step 5)"), and one on "\n)" misses the closing paren that shares a line
    with the last element. The syntax tree has neither problem.
    """
    src = open(os.path.join(ROOT, "eth80_frontend.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "_FRONTEND_SWITCHES"
                        for t in node.targets)):
            return {e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    raise AssertionError("could not find _FRONTEND_SWITCHES")


def _read_names(path):
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    # both the literal form ("PLACEMENT_X") and the _param("PLACEMENT_X", ...) form
    return set(re.findall(r'"(FE_[A-Z0-9_]+|PLACEMENT_[A-Z0-9_]+)"', src))


class SwitchDigestTests(unittest.TestCase):
    def test_every_build_path_switch_keys_the_cache(self):
        digest = _digest_names()
        missing = {}
        for rel in BUILD_PATH:
            for name in _read_names(rel):
                if name not in digest:
                    missing.setdefault(name, []).append(rel)
        self.assertEqual(
            missing, {},
            "switches read on the graph-build path but absent from "
            "eth80_frontend._FRONTEND_SWITCHES (a cached graph could be served "
            "across them): %r" % (missing,))

    def test_the_six_payload_switches_are_declared(self):
        import parameters as P
        for name in ("PLACEMENT_COLLAPSE_PLACES", "PLACEMENT_PLACE_GRID_PX",
                     "PLACEMENT_ENDPOINT_LENSES", "PLACEMENT_EXCLUDE_LENSES",
                     "PLACEMENT_RIDGE_ACROSS_ONLY", "PLACEMENT_RIDGE_ACROSS_MIN"):
            self.assertTrue(hasattr(P, name), "%s is not declared in parameters" % name)


if __name__ == "__main__":
    unittest.main()
