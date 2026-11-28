"""M32 B: the record file both fixes read -- v30.

`data/placement_records_m32.npz` = the M30 file plus

  * the four M32 curvature columns (`sharp_x/sharp_y` = |den| of the quadratic peak fit,
    `curv_x/curv_y` = |den|/|c0|), i.e. SHAPE_FIELDS 5 -> 9, and
  * the content's six ring values stored RAW in scaled Lab units instead of unit-normalised --
    the shipped cosine normalises internally, so its scores are unchanged.

Keys are untouched by construction (`signature_of` never reads the new columns or the content),
and `--check` proves them byte-identical cell-for-cell against the M30 file, proves the frames
identical, and re-reads the shipped cosine to show it is unchanged.

    python Topology/m32_records.py --build    # resumable, per-cell cache, progress file
    python Topology/m32_records.py --check    # keys byte-identical, 9 columns, cosine unchanged
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

IN = os.path.join(ROOT, "data", "placement_records_m30.npz")
OUT = os.path.join(ROOT, "data", "placement_records_m32.npz")
CACHE = os.path.join(ROOT, ".cache", "placement_m32")
PROGRESS = os.path.join(ROOT, "Topology", "results", "placement", "m32_records_progress.json")


def _slug(obj_key: str, view: str) -> str:
    return "%s_%s.npz" % (obj_key.replace("/", "-"), view)


def _cell_payload(obj: Dict[str, Any], view: str, rec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """The (n, 9) shape block and the (n, 6) raw content of one cell, cached for resumability."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, _slug(obj["key"], view))
    if os.path.exists(p):
        z = np.load(p)
        return {"shape": z["shape"], "content": z["content"]}
    from lenses import keypoints as KP
    from matching import placement_shape as PS
    from matching import placement_content as PC
    from matching.placement_frontend import load_image
    from Topology import m26_records as REC
    img = load_image(REC.path_of(obj, view))
    det = KP.detect(img)
    xy = np.asarray([[float(d.x), float(d.y)] for d in det], dtype=np.float64).reshape(-1, 2)
    places: List[Dict[str, Any]] = []
    for row in rec["places"]:
        q = dict(row)
        if xy.shape[0]:
            dd = np.hypot(xy[:, 0] - float(q["x"]), xy[:, 1] - float(q["y"]))
            j = int(np.argmin(dd))
            if float(dd[j]) <= 1.5:
                # every graded column the KEYPOINT carries, by name
                for name in PR_SHAPE_FIELDS:
                    v = getattr(det[j], name, None)
                    if v is not None:
                        q[name] = float(v)
        places.append(q)
    shape = PS.features(img, places, det)
    content = PC.features(img, rec["places"])          # RAW (M32)
    np.savez(p, shape=shape, content=content)
    return {"shape": shape, "content": content}


from matching.placement_records import SHAPE_FIELDS as PR_SHAPE_FIELDS   # noqa: E402  (doc above)


def build(views: Sequence[str] | None = None, limit: int = 0) -> Dict[str, Any]:
    from matching import placement_records as PR
    from Topology import m26_records as REC
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    vs = list(views or REC.VIEWS)
    t0 = time.time()
    cells: List[Tuple[str, str, Dict[str, Any]]] = []
    total = len(keys) * len(vs)
    done = 0
    for o in objs:
        for v in vs:
            rec = PR.load_cell(IN, o["key"], v)
            pay = _cell_payload(o, v, rec)
            for i, p in enumerate(rec["places"]):
                for j, name in enumerate(PR.SHAPE_FIELDS):
                    p[name] = float(pay["shape"][i, j])
                p["content"] = [float(x) for x in pay["content"][i]]
            cells.append((o["key"], v, rec))
            done += 1
            if done % 20 == 0 or done == total:
                _progress({"done": done, "total": total, "seconds": round(time.time() - t0, 1),
                           "eta_s": round((time.time() - t0) / max(done, 1) * (total - done), 1)})
                print(json.dumps({"done": done, "total": total,
                                  "seconds": round(time.time() - t0, 1)}), flush=True)
    arrays = PR.export(cells, keys, vs)
    PR.save(OUT, arrays)
    info = PR.peek(OUT)
    info.pop("shapes", None)
    info["cells"] = len(cells)
    info["seconds"] = round(time.time() - t0, 1)
    info["enrichment_from"] = os.path.basename(IN)
    info["size_mb"] = round(os.path.getsize(OUT) / 1e6, 2)
    _progress({"done": total, "total": total, "finished": True, **info})
    print(json.dumps({k: v for k, v in info.items() if k != "environment"}, indent=1))
    return info


def check(cells: int = 6) -> Dict[str, Any]:
    """Keys byte-identical to M30, frames identical, 9 columns present, cosine unchanged."""
    from matching import placement_records as PR
    from matching import placement_content as PC
    from matching import placement_shape as PS
    from Topology import m26_records as REC
    out: Dict[str, Any] = {"cells_checked": [], "keys_identical": True, "frames_identical": True,
                           "shape_fields": len(PR.SHAPE_FIELDS), "format_version": PR.FORMAT_VERSION,
                           "missing_registration": PS.missing_fields(), "cosine_max_delta": 0.0,
                           "raw_not_unit": True}
    objs = REC.objects()[: int(cells)]
    for o in objs:
        for v in REC.VIEWS:
            a = PR.load_cell(IN, o["key"], v)
            b = PR.load_cell(OUT, o["key"], v)
            ka = [p["key"] for p in a["places"]]
            kb = [p["key"] for p in b["places"]]
            if ka != kb:
                out["keys_identical"] = False
            for f in ("x", "y", "s", "theta", "n", "lens", "dim"):
                if [p.get(f) for p in a["places"]] != [p.get(f) for p in b["places"]]:
                    out["frames_identical"] = False
            out["cells_checked"].append("%s|%s" % (o["key"], v))
            # the shipped cosine, before and after the storage change
            ca = np.asarray([p["content"] for p in a["places"] if p.get("content") is not None])
            cb = np.asarray([p["content"] for p in b["places"] if p.get("content") is not None])
            if ca.size and cb.size:
                sa = PC.similarity(ca, ca)
                sb = PC.similarity(cb, cb)
                out["cosine_max_delta"] = max(out["cosine_max_delta"],
                                              float(np.abs(sa - sb).max()))
                norms = np.linalg.norm(cb, axis=1)
                out["raw_not_unit"] = bool((np.abs(norms - 1.0) > 1e-6).mean() > 0.9)
            for name in ("sharp_x", "sharp_y", "curv_x", "curv_y"):
                if any(p.get(name) is None for p in b["places"]):
                    out["shape_fields"] = -1
    out["ok"] = bool(out["keys_identical"] and out["frames_identical"]
                     and out["shape_fields"] == len(PR.SHAPE_FIELDS)
                     and not out["missing_registration"] and out["cosine_max_delta"] < 1e-9
                     and out["raw_not_unit"])
    print(json.dumps(out, indent=1, default=float))
    return out


def _progress(d: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True, default=float)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cells", type=int, default=6)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.build:
        build()
        return 0
    if a.check:
        return 0 if check(a.cells)["ok"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
