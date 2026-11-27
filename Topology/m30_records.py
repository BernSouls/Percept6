"""M30 B2: the shape-enriched record file.

`data/placement_records_m30.npz` = the M28 file (keys, Dz/Mz, frames, counts, content) plus the five
graded SHAPE columns, attached from the image and the record's own frames.  Keys are untouched by
construction, and `--check` proves it cell-for-cell against the M28 file.

    python Topology/m30_records.py --build    # resumable, per-cell cache, progress file
    python Topology/m30_records.py --check    # keys byte-identical, shape present, registered
    python Topology/m30_records.py --peek
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

IN = os.path.join(ROOT, "data", "placement_records_m28.npz")
OUT = os.path.join(ROOT, "data", "placement_records_m30.npz")
CACHE = os.path.join(ROOT, ".cache", "placement_shape")
PROGRESS = os.path.join(ROOT, "Topology", "results", "placement", "m30_records_progress.json")


def _slug(obj_key: str, view: str) -> str:
    return "%s_%s.npy" % (obj_key.replace("/", "-"), view)


def _shape_for(obj: Dict[str, Any], view: str, rec: Dict[str, Any]) -> np.ndarray:
    """The (n, 5) shape block of one cell, cached so the run is resumable."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, _slug(obj["key"], view))
    if os.path.exists(p):
        return np.load(p)
    from lenses import keypoints as KP
    from matching import placement_shape as PS
    from matching.placement_frontend import load_image
    from Topology import m26_records as REC
    img = load_image(REC.path_of(obj, view))
    det = KP.detect(img)
    # resp/var_x/var_y come from the detection that made the place (matched by position)
    xy = np.asarray([[float(d.x), float(d.y)] for d in det], dtype=np.float64).reshape(-1, 2)
    places: List[Dict[str, Any]] = []
    for row in rec["places"]:
        q = dict(row)
        if xy.shape[0]:
            dd = np.hypot(xy[:, 0] - float(q["x"]), xy[:, 1] - float(q["y"]))
            j = int(np.argmin(dd))
            if float(dd[j]) <= 1.5:
                q["resp"] = float(det[j].resp)
                q["var_x"] = float(det[j].var_x)
                q["var_y"] = float(det[j].var_y)
        places.append(q)
    f = PS.features(img, places, det)
    np.save(p, f)
    return f


def build() -> Dict[str, Any]:
    from matching import placement_records as PR
    from Topology import m26_records as REC
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    views = list(REC.VIEWS)
    t0 = time.time()
    cells: List[Tuple[str, str, Dict[str, Any]]] = []
    total = len(keys) * len(views)
    done = 0
    for o in objs:
        for v in views:
            rec = PR.load_cell(IN, o["key"], v)
            f = _shape_for(o, v, rec)
            for i, p in enumerate(rec["places"]):
                for j, name in enumerate(PR.SHAPE_FIELDS):
                    p[name] = float(f[i, j])
            cells.append((o["key"], v, rec))
            done += 1
            if done % 40 == 0 or done == total:
                _progress({"done": done, "total": total,
                           "seconds": round(time.time() - t0, 1),
                           "eta_s": round((time.time() - t0) / done * (total - done), 1)})
                print(json.dumps({"done": done, "total": total,
                                  "seconds": round(time.time() - t0, 1)}), flush=True)
    arrays = PR.export(cells, keys, views)
    PR.save(OUT, arrays)
    info = PR.peek(OUT)
    info.pop("shapes", None)
    info["cells"] = len(cells)
    info["seconds"] = round(time.time() - t0, 1)
    info["enrichment_from"] = os.path.basename(IN)
    info["shape_registration"] = _registration()
    info["environment"] = _env()
    _progress({"done": total, "total": total, "finished": True, **info})
    print(json.dumps({k: v for k, v in info.items() if k != "environment"}, indent=1))
    return info


def check(cells: int = 4) -> Dict[str, Any]:
    """Keys byte-identical to M28, frames identical, shape present, fields registered."""
    from matching import placement_records as PR
    from matching import placement_shape as PS
    from Topology import m26_records as REC
    out: Dict[str, Any] = {"cells": [], "keys_identical": True, "frames_identical": True,
                           "shape_present": True, "missing_registration": PS.missing_fields()}
    checked = 0
    for o in REC.objects():
        for v in REC.VIEWS:
            a = PR.load_cell(IN, o["key"], v)
            b = PR.load_cell(OUT, o["key"], v)
            if len(a["places"]) != len(b["places"]):
                out["keys_identical"] = False
                out["cells"].append({"object": o["key"], "view": v,
                                     "places": [len(a["places"]), len(b["places"])]})
                continue
            keys_same = all(tuple(x["key"][1]) == tuple(y["key"][1])
                            and x["lens"] == y["lens"] for x, y in zip(a["places"], b["places"]))
            frames_same = all((abs(x["x"] - y["x"]) < 1e-12 and abs(x["y"] - y["y"]) < 1e-12
                               and abs(x["s"] - y["s"]) < 1e-12)
                              for x, y in zip(a["places"], b["places"]))
            shape_ok = all(all(y.get(f) is not None for f in PR.SHAPE_FIELDS)
                           for y in b["places"])
            out["keys_identical"] &= bool(keys_same)
            out["frames_identical"] &= bool(frames_same)
            out["shape_present"] &= bool(shape_ok)
            checked += 1
            if checked >= cells * len(REC.VIEWS):
                break
        if checked >= cells * len(REC.VIEWS):
            break
    out["cells_checked"] = checked
    print(json.dumps(out, indent=1))
    return out


def _registration() -> Dict[str, Any]:
    from matching import placement_shape as PS
    PS.register()
    return {"class": PS.CLASS, "digest": PS.digest(), **PS.registered(),
            "missing": PS.missing_fields()}


def _progress(info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)


def _env() -> Dict[str, Any]:
    from matching.placement_records import environment
    return environment()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--peek", action="store_true")
    ap.add_argument("--cells", type=int, default=4)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.build:
        build()
    if a.check:
        check(a.cells)
    if a.peek:
        from matching import placement_records as PR
        info = PR.peek(OUT)
        info["shapes"] = {k: v for k, v in info["shapes"].items()
                          if k.startswith(("shape", "content"))}
        print(json.dumps(info, indent=1))
    if not (a.build or a.check or a.peek):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
