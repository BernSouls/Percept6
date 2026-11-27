"""A compact, committed form of the place records (Mission 27, item 1b).

A fresh sandbox used to rebuild 320 place records in 2,338 s before any door experiment could run,
because nothing durable carried them: the records are pickled per cell under ``.cache/`` and the
graph cache that produces them is not in the repository.  This module is the durable form.

WHAT IS STORED.  Everything the DOOR reads, and nothing else:

    per place      quantised tuple (theta, coherence, scale, kind, dim), the frozen-ruler
                   configuration row Dz and its mask Mz, and the frame (x, y, s, theta)
    per cell       the record's dropped counts, n_payload, n_all_nodes

``std_all`` -- the standardised descriptor channels the DEPLOYED PREFILTER differences -- is NOT
stored: no door and no experiment in this programme reads it from a record, and it is the bulk of
the pickle.  ``load_cell`` therefore returns a record the door can index and score, and a consumer
that needs ``std_all`` must rebuild from the image (``placement_frontend.place_record``).

FORMAT.  One ``.npz`` (float32 matrices + int16 keys), per view, plus a JSON sidecar of keys and
counts.  ``export`` writes it from the live cache; ``load`` reads it back into records that are
field-for-field what the door sees.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The lens classes a keypoint record can carry, in fixed order (stored as an int8 code).
LENSES: Tuple[str, ...] = ("Blob", "BoundaryArc", "CornerPoint")
DROP_REASONS: Tuple[str, ...] = ("no_sig", "no_config", "no_position", "no_scale")

#: The graded shape fields a record may carry (M30), in fixed order.
SHAPE_FIELDS: Tuple[str, ...] = ("resp", "var_x", "var_y", "eigratio", "rel_angle")
SLOTS = 5

#: The record format.  28 adds the optional CONTENT block (`content.kind` != "keypoint"):
#: `N_CONTENT` float columns per place, and the same number of bin-ints when counted.
FORMAT_VERSION = 29
N_CONTENT = 6

#: M30: the graded SHAPE columns (resp, var_x, var_y, eigratio, rel_angle).  Same discipline as
#: content: columns on the record, never new key slots, so every key stays byte-identical.
N_SHAPE = 5

#: The stored float width.  float64, because the door MATCHES ON EXACT KEY EQUALITY: a quantised
#: value stored as float32 (1.6 -> 1.6000000238) lands in a different bucket, and the loader's
#: shortlists stop matching the live ones.  Measured: 12.8 MB in float64, well inside the budget.
DTYPE = np.float64


def export(cells: Sequence[Tuple[str, str, Dict[str, Any]]], keys: Sequence[str],
           views: Sequence[str]) -> Dict[str, np.ndarray]:
    """``(obj_key, view, record)`` triples -> the arrays of the compact file.

    ``rows_<view>``: 9 float32 columns per place -- the 5 quantised values, x, y, s, theta.
    ``dz_<view>`` / ``mz_<view>``: the frozen ruler's configuration row and mask, (P, L) float32 --
    these are read by the door's ``cfg_distance`` and are NOT truncatable to the tuple length.
    """
    key_index = {k: i for i, k in enumerate(keys)}
    widths: Dict[str, int] = {}
    for _obj, view, rec in cells:
        for r in rec["places"]:
            widths[view] = max(widths.get(view, 0), len(r["Dz"]))
    out: Dict[str, np.ndarray] = {
        "keys": np.asarray(list(keys), dtype=object),
        "views": np.asarray(list(views), dtype=object),
        "meta": np.asarray(json.dumps({"version": FORMAT_VERSION, "slots": SLOTS,
                                       "shape": N_SHAPE if any(
                                           r.get("eigratio") is not None for _o, _v, rec in cells
                                           for r in rec["places"]) else 0,
                                       "content": N_CONTENT if any(
                                           r.get("content") for _o, _v, rec in cells
                                           for r in rec["places"]) else 0,
                                       "lenses": list(LENSES),
                                       "drop_reasons": list(DROP_REASONS),
                                       "n_cells": len(cells),
                                       "complete": len(cells) == len(keys) * len(views)})),
    }
    buckets: Dict[str, Dict[str, List[Any]]] = {
        v: {"rows": [], "dz": [], "mz": [], "obj": [], "idx": [], "drop": [], "pay": [],
            "nodes": [], "lens": []} for v in views}
    content: Dict[str, List[Any]] = {v: [] for v in views}
    shape: Dict[str, List[Any]] = {v: [] for v in views}
    for obj_key, view, rec in cells:
        b = buckets[view]
        width = widths.get(view, 0)
        for r in rec["places"]:
            q = list(r["key"][1])
            q = (q + [0.0] * SLOTS)[:SLOTS]
            th = r.get("theta")
            b["rows"].append(q + [r["x"], r["y"], r["s"],
                                  (np.nan if th is None else float(th))])
            dz = list(r["Dz"])
            mz = list(r["Mz"])
            b["dz"].append(dz + [0.0] * (width - len(dz)))
            b["mz"].append(mz + [0.0] * (width - len(mz)))
            b["obj"].append(key_index[obj_key])
            b["idx"].append(int(r["n"]))
            b["lens"].append(LENSES.index(r["lens"]) if r["lens"] in LENSES else -1)
            c = r.get("content")
            if c is not None:
                cc = list(c)
                content[view].append(cc + [0.0] * (N_CONTENT - len(cc)))
            if r.get("eigratio") is not None:
                shape[view].append([float(r.get(k, 0.0) or 0.0) for k in SHAPE_FIELDS])
        d = rec.get("dropped") or {}
        b["drop"].append([int(d.get(k, 0)) for k in DROP_REASONS])
        b["pay"].append(int(rec.get("n_payload") or 0))
        b["nodes"].append(int(rec.get("n_all_nodes") or 0))
    for v in views:
        b = buckets[v]
        n = len(b["rows"])
        width = widths.get(v, 0)
        out["rows_%s" % v] = (np.asarray(b["rows"], dtype=DTYPE) if n
                              else np.zeros((0, SLOTS + 4), dtype=DTYPE))
        out["dz_%s" % v] = (np.asarray(b["dz"], dtype=DTYPE) if n
                            else np.zeros((0, width), dtype=DTYPE))
        out["mz_%s" % v] = (np.asarray(b["mz"], dtype=DTYPE) if n
                            else np.zeros((0, width), dtype=DTYPE))
        out["obj_%s" % v] = np.asarray(b["obj"], dtype=np.int16) if n else np.zeros(0, np.int16)
        out["idx_%s" % v] = np.asarray(b["idx"], dtype=np.int32) if n else np.zeros(0, np.int32)
        out["lens_%s" % v] = np.asarray(b["lens"], dtype=np.int8) if n else np.zeros(0, np.int8)
        out["drop_%s" % v] = np.asarray(b["drop"], dtype=np.int32).reshape(len(keys),
                                                                          len(DROP_REASONS))
        out["pay_%s" % v] = np.asarray(b["pay"], dtype=np.int32)
        out["nodes_%s" % v] = np.asarray(b["nodes"], dtype=np.int32)
        if content[v]:
            out["content_%s" % v] = np.asarray(content[v], dtype=DTYPE)
        if shape[v]:
            out["shape_%s" % v] = np.asarray(shape[v], dtype=DTYPE)
    return out


def save(path: str, arrays: Dict[str, np.ndarray]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def peek(path: str) -> Dict[str, Any]:
    """The file's keys/views/meta without materialising the matrices."""
    with np.load(path, allow_pickle=True) as z:
        keys = [str(x) for x in z["keys"]]
        views = [str(x) for x in z["views"]]
        meta = json.loads(str(z["meta"]))
        shapes = {k: tuple(z[k].shape) for k in z.files}
    return {"keys": keys, "views": views, "meta": meta, "shapes": shapes,
            "bytes": os.path.getsize(path)}


def load_cell(path: str, obj_key: str, view: str) -> Dict[str, Any]:
    """One record, exactly the fields the door reads."""
    with np.load(path, allow_pickle=True) as z:
        return _cell_from(z, obj_key, view)


def load_cells(path: str, views: Optional[Sequence[str]] = None
               ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as z:
        keys = [str(x) for x in z["keys"]]
        have = [str(x) for x in z["views"]]                                  # noqa: F841
        vs = list(views) if views else [str(x) for x in z["views"]]
        for k in keys:
            for v in vs:
                out[(k, v)] = _cell_from(z, k, v)
    return out


def _cell_from(z: Any, obj_key: str, view: str) -> Dict[str, Any]:
    keys = [str(x) for x in z["keys"]]
    i = keys.index(obj_key)
    rows = z["rows_%s" % view]
    dz_all = z["dz_%s" % view] if ("dz_%s" % view) in z.files else None
    mz_all = z["mz_%s" % view] if ("mz_%s" % view) in z.files else None
    obj = z["obj_%s" % view]
    idx = z["idx_%s" % view]
    lens = z["lens_%s" % view]
    content_all = z["content_%s" % view] if ("content_%s" % view) in z.files else None
    shape_all = z["shape_%s" % view] if ("shape_%s" % view) in z.files else None
    sel = np.flatnonzero(obj == i)
    places: List[Dict[str, Any]] = []
    for j in sel:
        r = rows[j]
        q = tuple(float(x) for x in r[:SLOTS])
        dz = ([float(x) for x in dz_all[j]] if dz_all is not None
              else [float(x) for x in r[SLOTS:2 * SLOTS]])
        mz = ([float(x) for x in mz_all[j]] if mz_all is not None
              else [float(x) for x in r[2 * SLOTS:3 * SLOTS]])
        x, y, s, th = (float(r[SLOTS]), float(r[SLOTS + 1]),
                       float(r[SLOTS + 2]), float(r[SLOTS + 3]))
        code = int(lens[j])
        lens_name = LENSES[code] if 0 <= code < len(LENSES) else "Unknown"
        places.append({"n": int(idx[j]), "lens": lens_name,
                       "key": (lens_name, q), "Dz": dz, "Mz": mz,
                       "x": x, "y": y, "s": s,
                       "theta": (None if not np.isfinite(th) else th)})
        if content_all is not None:
            places[-1]["content"] = [float(v) for v in content_all[j]]
        if shape_all is not None:
            for _f, _v in zip(SHAPE_FIELDS, shape_all[j]):
                places[-1][_f] = float(_v)
    drop = {name: int(z["drop_%s" % view][i][d]) for d, name in enumerate(DROP_REASONS)}
    return {"path": None, "obj": obj_key, "view": view, "places": places,
            "dropped": drop, "n_payload": int(z["pay_%s" % view][i]),
            "n_all_nodes": int(z["nodes_%s" % view][i]), "sec": 0.0,
            "from_npz": True}


# --------------------------------------------------------------------------- the environment
def environment() -> Dict[str, Any]:
    """What produced a number: CPU, cores, interpreter, and the numeric stack, to the version.

    Mission 27 measured that a pair score is NOT invariant across environments (+0.004 mean,
    max +0.010 on scores near 0.35, while the place records were identical).  A result file that
    does not say which environment produced it is therefore not readable, so every manifest carries
    this block (see ``matching/m15_manifest.manifest``).
    """
    import platform
    import sys
    model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = platform.processor()
    out: Dict[str, Any] = {
        "cpu": model,
        "physical_cores": physical_cores(limit=1024),
        "logical_cores": int(os.cpu_count() or 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "scipy", "PIL"):
        try:
            m = __import__(mod)
            out[mod.lower().rstrip(".")] = str(getattr(m, "__version__", "?"))
        except Exception:
            out[mod.lower()] = "missing"
    try:
        from matching.placement_recognise import pair_env
        out["pair_env"] = pair_env()
    except Exception:
        out["pair_env"] = "unavailable"
    return out


# --------------------------------------------------------------------------- how many workers
def physical_cores(limit: int = 4) -> int:
    """Physical cores on this box, capped at ``limit`` -- the default worker count (M27 item 1c).

    ``os.cpu_count()`` counts LOGICAL processors; on a hyperthreaded box that oversubscribes the
    matcher by 2x, which is why the count is read from ``/proc/cpuinfo``'s (physical id, core id)
    pairs when it exists.  The cap keeps a big host from starting more processes than the pair job
    can keep busy; 4 is where the throughput curve flattened in `m27_profile.py`.
    """
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if pairs:
            return max(1, min(int(limit), len(pairs)))
    except OSError:
        pass
    return max(1, min(int(limit), os.cpu_count() or 1))


def placed_workers(limit: int = 4) -> int:
    """The default worker count: the physical cores, with ONE hyperthread when that is all there is.

    This box reports 1 physical core and 2 logical processors.  Running two matcher processes on
    one physical core is not oversubscription of the kind that costs throughput -- the workload is
    memory- and branch-heavy enough that the sibling thread earns its keep (``m27_profile.py
    --profile`` measures 1, 2 and 4 workers and the report carries the numbers) -- but past the
    sibling there is nothing to win, so the count never exceeds ``limit`` and never exceeds two per
    physical core.
    """
    phys = physical_cores(limit)
    logical = os.cpu_count() or 1
    per_core = max(1, logical // max(1, phys))
    return max(1, min(int(limit), phys * min(2, per_core)))
