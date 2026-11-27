"""M28 item 6: the record file that carries the place's content.

TWO WAYS TO GET THE CONTENT INTO A RECORD, and why this script takes the second:

* a full re-derivation -- 320 cells through `placement_frontend.place_record(content=...)`, priced
  from M27's own measurement at 1,824.5 s; it re-derives the frozen keys as a side effect, and
  item 1b's own result says a re-derivation is the one thing that can move them between
  environments (cross-sandbox score drift +0.004, cause = environment, not the worker count).
* ENRICHMENT -- every frozen field (keys, Dz/Mz, frames, counts) comes from the committed M27 file
  byte-for-byte, and the six content columns are new, computed by `matching/placement_content.py`
  from the image and the record's OWN frames.  Content is a pure function of those two, so this is
  the same number a re-derivation would produce; `--check` proves it on a live cell.

This script writes `data/placement_records_m28.npz`: the M27 file plus `content_<view>`.

    python Topology/m28_records.py --build     # resumable, per-cell cache, progress file
    python Topology/m28_records.py --check     # enrichment == place_record(content=...)
    python Topology/m28_records.py --peek
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

IN = os.path.join(ROOT, "data", "placement_records_m27.npz")
OUT = os.path.join(ROOT, "data", "placement_records_m28.npz")
CACHE = os.path.join(ROOT, ".cache", "placement_content")
PROGRESS = os.path.join(ROOT, "Topology", "results", "placement", "m28_records_progress.json")
KIND = "colour_rings"


def _slug(obj_key: str, view: str) -> str:
    return "%s_%s.npy" % (obj_key.replace("/", "-"), view)


def _content_for(obj_key: str, view: str, rec: Dict[str, Any],
                 obj: Dict[str, Any]) -> np.ndarray:
    """The cell's (n, 6) content, cached per cell so the run is resumable."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, _slug(obj_key, view))
    if os.path.exists(p):
        return np.load(p)
    from matching import placement_content as PC
    from matching.placement_frontend import load_image
    from Topology import m26_records as REC
    img = load_image(REC.path_of(obj, view))
    f = PC.features(img, rec["places"])
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
    done = 0
    total = len(keys) * len(views)
    for o in objs:
        for v in views:
            rec = REC.load(o["key"], v)          # the committed export; frozen fields as they are
            f = _content_for(o["key"], v, rec, o)
            for i, p in enumerate(rec["places"]):
                p["content"] = [float(x) for x in f[i]]
            rec["content_kind"] = KIND
            cells.append((o["key"], v, rec))
            done += 1
            if done % 20 == 0 or done == total:
                info = {"done": done, "total": total, "seconds": round(time.time() - t0, 1),
                        "kind": KIND, "eta_s": round((time.time() - t0) / done * (total - done), 1)}
                _write_progress(info)
                print(json.dumps(info), flush=True)
    arrays = PR.export(cells, keys, views)
    PR.save(OUT, arrays)
    info = PR.peek(OUT)
    info.pop("shapes", None)
    info["cells"] = len(cells)
    info["content_kind"] = KIND
    info["seconds"] = round(time.time() - t0, 1)
    info["enrichment_from"] = os.path.basename(IN)
    info["environment"] = _env()
    _write_progress({"done": total, "total": total, "finished": True, **info})
    print(json.dumps({k: v for k, v in info.items() if k != "environment"}, indent=1))
    return info


def check(n: int = 2) -> Dict[str, Any]:
    """Enrichment must equal the pipeline's own `place_record(content=...)`, place for place."""
    from matching import placement_content as PC
    from matching import placement_records as PR
    from matching.placement_frontend import place_record
    from Topology import m26_records as REC
    objs = REC.objects()
    out: Dict[str, Any] = {"cells": [], "max_abs_diff": 0.0}
    for o in objs[:n]:
        rec_npz = PR.load_cell(OUT, o["key"], REC.REF_VIEW)
        live = place_record(REC.path_of(o, REC.REF_VIEW), tag="m28check", content=KIND)
        a = np.asarray([p["content"] for p in rec_npz["places"]], dtype=np.float64)
        b = np.asarray([p["content"] for p in live["places"]], dtype=np.float64)
        same_n = a.shape == b.shape
        d = float(np.max(np.abs(a - b))) if same_n else float("nan")
        out["cells"].append({"object": o["key"], "places_npz": int(a.shape[0]),
                             "places_live": int(b.shape[0]), "max_abs_diff": d,
                             "quantised_keys_equal": bool(same_n and all(
                                 PC.quantise(x) == PC.quantise(y) for x, y in zip(a, b)))})
        out["max_abs_diff"] = max(out["max_abs_diff"], d if same_n else 1.0)
    out["equal_to_1e12"] = out["max_abs_diff"] <= 1e-12
    print(json.dumps(out, indent=1))
    return out


def _write_progress(info: Dict[str, Any]) -> None:
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
    ap.add_argument("--cells", type=int, default=2)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.build:
        build()
    if a.check:
        check(a.cells)
    if a.peek:
        from matching import placement_records as PR
        info = PR.peek(OUT)
        info["shapes"] = {k: v for k, v in info["shapes"].items() if k.startswith("content")}
        print(json.dumps(info, indent=1))
    if not (a.build or a.check or a.peek):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
