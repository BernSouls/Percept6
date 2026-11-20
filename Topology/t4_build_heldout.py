"""Task 4 PART 4c: build held-out graphs for arms B0/B1/F at shifts 0,±2,±3.

Held-out run = lowest-rank held-out instance per class, both views (16 images).
Shifts are cyclic np.roll of the whole image (translation).  Each (arm, image,
shift) graph is built once and pickled to disk under results/t4/heldout/.

Usage:
    python t4_build_heldout.py --arms B0,B1,F --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()

ARMS = {
    "B0": "FE_L3_SPINE=false;FE_L4_JUNCTIONS=false",
    "B1": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true",
    "F": "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_SUBPIXEL_PEAK_POS=true",
}
SHIFTS = (0, 2, -2, 3, -3)
OUT = Path(ROOT) / "Topology" / "results" / "t4" / "heldout"


def _worker_init(spec):
    import guard
    guard.activate()
    import _bootstrap as b
    b.apply_switch_overrides(spec)


def _heldout_entries():
    import corpus
    split = corpus.split_tuning()
    by_class = {}
    for e in split["heldout"]:
        cls = e["class"]
        if cls not in by_class or e["object"] < by_class[cls]["object"]:
            by_class[cls] = e
    out = []
    for c in corpus.CLASSES:
        e = by_class[c]
        for v in corpus.QUERY_VIEWS:
            # find the entry for this view
            for e2 in split["heldout"]:
                if e2["class"] == c and e2["object"] == e["object"] \
                        and e2["view"] == v:
                    out.append(e2)
                    break
    return out


def _do(args2):
    arm, entry, shift = args2
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARMS[arm])
    from frontend import build
    # roll the image
    if shift == 0:
        e = entry
    else:
        tmp = Path("/tmp/t4_heldout") / ("%s%d%s_%d.png"
                                         % (entry["class"], int(entry["object"]),
                                            entry["view"], shift))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if not tmp.is_file():
            with Image.open(entry["path"]) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            from t4_shift import roll_image
            rolled = roll_image(arr, dx=shift, dy=0)
            Image.fromarray(rolled).save(tmp)
        e = dict(entry)
        e["path"] = str(tmp)
    name = "%s%d%s" % (entry["class"], int(entry["object"]), entry["view"])
    g = build(e)
    return {"arm": arm, "name": name, "shift": shift, "graph": g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B0,B1,F")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    entries = _heldout_entries()
    print("held-out entries:", [
        "%s%d%s" % (e["class"], int(e["object"]), e["view"]) for e in entries],
        flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks = [(arm, e, s) for arm in arms for e in entries for s in SHIFTS]
    print("total builds: %d (%d arms x %d images x %d shifts)"
          % (len(tasks), len(arms), len(entries), len(SHIFTS)), flush=True)
    t0 = time.perf_counter()
    done = 0
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=("",)) as pool:
        for rec in pool.imap_unordered(_do, tasks, chunksize=1):
            arm, name, shift, g = rec["arm"], rec["name"], rec["shift"], rec["graph"]
            path = OUT / ("%s_%s_%d.pkl" % (arm, name, shift))
            with open(path, "wb") as fh:
                pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print("  built %3d/%d (%.0fs)"
                      % (done, len(tasks), time.perf_counter() - t0),
                      flush=True)
    print("done in %.0fs" % (time.perf_counter() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
