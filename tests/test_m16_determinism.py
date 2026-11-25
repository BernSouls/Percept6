"""M16 quality bar 7 -- fresh-process bit-identical readings.

The M16 change is a change to a SIMILARITY, i.e. to a pure function of two
signatures.  It must therefore be identical in a fresh process, independent of the
order the arms were measured in (the first version of the harness registered a
constitution and then "cleared its bookkeeping", so the arm that was supposed to be
the shipped one inherited the fix -- caught by exactly this comparison).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SNIPPET = r"""
import hashlib, math, sys
sys.path[:0] = [%r, %r]
import parameters as P
P.KP_GRADED_SIMILARITY = True
P.KP_GRADED_POPULATION = True
from lenses.keypoints import Keypoint, signature_of
def kp(d, c=0.8, s=1.6, kind="blob"):
    return Keypoint(kind=kind, x=1.0, y=1.0, scale=s, theta=math.radians(d),
                    coherence=c, resp=1.0, var_x=1.0, var_y=1.0, dim=2)
vals = []
base = signature_of(kp(11.0))
for d in (0, 5, 15, 30, 45, 90):
    vals.append(round(base.similarity(signature_of(kp(11.0 + d))), 12))
for c in (0.35, 0.8, 0.99):
    vals.append(round(base.similarity(signature_of(kp(11.0, c=c))), 12))
for s in (1.0, 1.6, 2.5, 4.0):
    vals.append(round(base.similarity(signature_of(kp(11.0, s=s))), 12))
print(hashlib.sha256(repr(vals).encode()).hexdigest())
""" % (ROOT, os.path.join(ROOT, "Topology"))


def _digest() -> str:
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    out = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True,
                         text=True, cwd=ROOT, env=env)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-2000:])
    return out.stdout.strip()


class DeterminismTests(unittest.TestCase):
    def test_two_fresh_processes_agree(self):
        self.assertEqual(_digest(), _digest())

    def test_hash_seed_does_not_matter(self):
        env = dict(os.environ, PYTHONHASHSEED="1")
        a = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True,
                           text=True, cwd=ROOT, env=env).stdout.strip()
        env["PYTHONHASHSEED"] = "12345"
        b = subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True,
                           text=True, cwd=ROOT, env=env).stdout.strip()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
