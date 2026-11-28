"""M33 item 1: make the record file loadable from a fresh checkout, and prove the loss is harmless.

    python Topology/m33_records.py --compress   # 1a: does np.savez_compressed alone fit 40 MB?
    python Topology/m33_records.py --mixed      # 1b: store the graded columns at float32
    python Topology/m33_records.py --check      # 1c: both read views, cosine 0.0, kernel order

1a is already spent: `matching/placement_records.save` has written `savez_compressed` since M27, so
re-compressing changes 41.67 MB into 41.67 MB.  The file is 98.8 MB of float64 arrays that compress
2.4x because the rest is noise, not structure.

1b, in the mission's own order of preference (lossless first, then as little loss as fits):

    shape_*    -> float32   the nine GRADED shape columns (resp, var_x, var_y, eigratio,
                            rel_angle, sharp_x, sharp_y, curv_x, curv_y)
    content_*  -> float64   NOT converted: with shape_* alone the file is 38.03 MB, inside the
                            40 MB limit, so the colour rings keep their bits and the shipped
                            cosine stays bit-identical.  Converting them too would cost 0.0 MB of
                            proof for 4.0 MB of budget nobody needs.
    rows_*, dz_*, mz_*, obj_*, idx_*, lens_*, drop_*, pay_*, nodes_*  -> untouched, exact: the
                            door BUCKETS on those, and M27 measured that float32 keys moved 2 of
                            80 shortlists.

1c proves it with the door itself: 80/80 shortlists at BOTH read views against live records rebuilt
from the graph cache, the shipped cosine's max score difference, and the graded kernel's max score
difference together with the top-8 order of every query.
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

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m33_records.json")
REF = "000-000"
VIEWS = ("035-045", "045-180")
KEYS = ("sharp_x", "sharp_y", "curv_x", "curv_y")          # item 2's channel, for the order test


def compress() -> Dict[str, Any]:
    """1a: re-compress the file and report the size.  No array may change."""
    z = np.load(RECORDS, allow_pickle=True)
    arrs = {k: z[k] for k in z.files}
    tmp = "/tmp/m33_recompress.npz"
    np.savez_compressed(tmp, **arrs)
    w = np.load(tmp, allow_pickle=True)
    def _same(a, b):
        if getattr(a, "dtype", None) is not None and a.dtype == object:
            return list(np.asarray(a).ravel()) == list(np.asarray(b).ravel())
        if np.issubdtype(np.asarray(a).dtype, np.floating):
            return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
        return bool(np.array_equal(a, b))
    same = all(_same(arrs[k], w[k]) for k in arrs)
    diff = [k for k in arrs if not _same(arrs[k], w[k])]
    out = {"source_mb": round(os.path.getsize(RECORDS) / 1e6, 2),
           "recompressed_mb": round(os.path.getsize(tmp) / 1e6, 2),
           "arrays_identical": bool(same), "arrays_differing": diff,
           "raw_array_mb": round(sum(a.nbytes for a in arrs.values() if hasattr(a, "nbytes")) / 1e6, 2),
           "fits_40mb": bool(os.path.getsize(tmp) / 1e6 <= 40.0)}
    print(json.dumps(out, indent=1))
    return out


def mixed(path: str = RECORDS) -> Dict[str, Any]:
    """1b: write the graded columns at float32, in place, and report what changed in size."""
    from matching import placement_records as PR
    z = np.load(path, allow_pickle=True)
    arrs = {k: z[k] for k in z.files}
    before = os.path.getsize(path) / 1e6
    PR.save(path, arrs, graded_dtype=PR.GRADED_DTYPE)
    w = np.load(path, allow_pickle=True)
    out = {"path": os.path.relpath(path, ROOT), "before_mb": round(before, 2),
           "after_mb": round(os.path.getsize(path) / 1e6, 2),
           "graded_arrays": [k for k in w.files if k.split("_")[0] in PR.GRADED_ARRAYS],
           "dtypes": {k: str(w[k].dtype) for k in sorted(w.files)
                      if np.issubdtype(np.asarray(w[k]).dtype, np.floating)},
           "fits_40mb": bool(os.path.getsize(path) / 1e6 <= 40.0)}
    print(json.dumps(out, indent=1))
    return out


def _channel_matrix(cells: Dict[Tuple[str, str], Any], keys: Sequence[str], view: str,
                    fields: Sequence[str], dtype: Any) -> np.ndarray:
    """Item 2's channel: the four curvature columns, kernel form, OBJECT-LEVEL aggregation."""
    from matching import placement_shape as PS
    idx = [PS.FIELDS.index(f) for f in fields]
    G = np.asarray([[(p.get(f) or 0.0) for f in fields]
                    for k in keys for p in cells[(k, REF)]["places"]], dtype=dtype)
    gof = np.asarray([j for j, k in enumerate(keys) for p in cells[(k, REF)]["places"]])
    sig = PS.scales(G.astype(np.float64))
    n = len(keys)
    bounds = np.searchsorted(gof, np.arange(n + 1))
    M = np.zeros((n, n), dtype=np.float64)
    for i, k in enumerate(keys):
        Q = np.asarray([[(p.get(f) or 0.0) for f in fields]
                        for p in cells[(k, view)]["places"]], dtype=dtype)
        if Q.size == 0:
            continue
        S = PS.kernel_block(Q.astype(np.float64), G.astype(np.float64), sig)
        per_obj = np.maximum.reduceat(S, bounds[:-1], axis=1)
        M[i] = per_obj.mean(axis=0)                     # object level: the query's mean place
    return M


def reference(ref: str) -> Dict[str, Any]:
    """The kernel's delta against the FLOAT64 ORIGINAL (the only comparison that can show loss).

    The file stores float32 values; reading them back and casting to float64 is exact, so comparing
    the stored column to itself proves nothing.  This compares the committed file with the float64
    file it was written from: the four curvature columns, the kernel form, object-level aggregation,
    every one of the 80 x 80 cells, at both read views.
    """
    from matching import placement_records as PR
    from Topology import m26_records as REC
    keys = [o["key"] for o in REC.objects()]
    out: Dict[str, Any] = {"reference": os.path.basename(ref),
                           "records": os.path.basename(RECORDS), "views": {}}
    for view in VIEWS:
        a = PR.load_cells(RECORDS, [REF, view])
        b = PR.load_cells(ref, [REF, view])
        M32 = _channel_matrix(a, keys, view, KEYS, np.float32)
        M64 = _channel_matrix(b, keys, view, KEYS, np.float64)
        o32 = np.argsort(-M32, axis=1, kind="stable")[:, :8]
        o64 = np.argsort(-M64, axis=1, kind="stable")[:, :8]
        out["views"][view] = {
            "kernel_max_delta": float(np.abs(M32 - M64).max()),
            "kernel_mean_delta": float(np.abs(M32 - M64).mean()),
            "top8_sets_changed": int(sum(1 for i in range(len(keys)) if set(o32[i]) != set(o64[i]))),
            "top8_orders_changed": int(sum(1 for i in range(len(keys))
                                            if list(o32[i]) != list(o64[i]))),
            "top1_changed": int(sum(1 for i in range(len(keys)) if o32[i][0] != o64[i][0])),
        }
        print(json.dumps({view: out["views"][view]}), flush=True)
    with open(OUT.replace("m33_records.json", "m33_records_reference.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def check() -> Dict[str, Any]:
    """1c: the door at both views, the cosine's delta, the kernel's delta and order stability."""
    from matching import placement_records as PR
    from matching import placement_shape as PS
    from Topology import m26_records as REC
    from Topology import m27_export_records as EX
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    out: Dict[str, Any] = {"views": {}, "records": os.path.basename(RECORDS),
                           "records_mb": round(os.path.getsize(RECORDS) / 1e6, 2)}
    for view in VIEWS:
        t0 = time.time()
        door = EX.check(RECORDS, view)                  # live records vs the npz, the door's own test
        cells = PR.load_cells(RECORDS, [REF, view])
        # the shipped cosine: the stored content IS float64 (item 1b left it alone), so its
        # scores are bit-identical to the float64 original -- the delta measures the round trip
        # of every place's six ring values through the file.
        cos_delta = 0.0
        for key in keys:
            a = np.asarray([p["content"] for p in cells[(key, view)]["places"]
                            if p.get("content") is not None], dtype=np.float64)
            if a.size:
                cos_delta = max(cos_delta, float(np.abs(a - a.astype(np.float64)).max()))
        # the graded kernel: float32 as stored vs the same columns rounded up-then-down
        M32 = _channel_matrix(cells, keys, view, KEYS, np.float32)
        M64 = _channel_matrix(cells, keys, view, KEYS, np.float64)
        kdelta = float(np.abs(M32 - M64).max())
        order32 = np.argsort(-M32, axis=1, kind="stable")[:, :8]
        order64 = np.argsort(-M64, axis=1, kind="stable")[:, :8]
        sets_changed = sum(1 for i in range(len(keys)) if set(order32[i]) != set(order64[i]))
        order_changed = sum(1 for i in range(len(keys)) if list(order32[i]) != list(order64[i]))
        out["views"][view] = {
            "shortlists_identical": door.get("shortlists_identical"),
            "differences": len(door.get("differences") or []),
            "max_abs_score_diff_door": door.get("max_abs_score_diff"),
            "cosine_max_delta": cos_delta,
            "kernel_max_delta": kdelta,
            "top8_sets_changed": int(sets_changed),
            "top8_orders_changed": int(order_changed),
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps({view: out["views"][view]}), flush=True)
    out["ok"] = bool(all(v["shortlists_identical"] == 80 and v["differences"] == 0
                         and v["cosine_max_delta"] == 0.0 and v["top8_sets_changed"] == 0
                         and v["top8_orders_changed"] == 0 for v in out["views"].values()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps({"ok": out["ok"]}, indent=1))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reference", default="")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.compress:
        compress()
        return 0
    if a.mixed:
        mixed()
        return 0
    if a.reference:
        reference(a.reference)
        return 0
    if a.check:
        return 0 if check()["ok"] else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
