"""Mission 15 item 9 -- every result file carries the vector that produced it.

A number in this repository is only readable if the reader can tell WHICH configuration
produced it.  Mission 14's audit found that no read had ever recorded its full flag
vector, so two results that disagreed could not be compared.  This module makes the
record mandatory:

    from matching.m15_manifest import write_result
    write_result(path, {"my_number": 0.5})     # -> refuses without a manifest

`manifest()` records
  * `flags`      -- the EFFECTIVE value of every front-end switch and every
                    `PLACEMENT_*` parameter (read off the modules, not from a list of
                    names someone remembered to type),
  * `frame`      -- the frame provider actually in force,
  * `weight`     -- the weight actually in force and WHERE IT CAME FROM,
  * `endpoints`  -- the endpoint lens list actually in use,
  * `code`       -- a digest of the Python sources, so a result can be tied to a tree.

Item 9 also covers the refinement BUDGET: `budget_report()` counts, per graph size, how
often the alignment refinement budget was exhausted, so "the cap bound" is a measured
fact rather than an assumption.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".cache", "node_modules", "build",
              "dist", "results", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _walk_py() -> List[str]:
    out: List[str] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for f in sorted(files):
            if f.endswith(".py"):
                out.append(os.path.relpath(os.path.join(base, f), ROOT))
    return out


_CODE_CACHE: Dict[str, Any] = {}


def code_digest(short: bool = True) -> str:
    """sha256 over every Python source in the tree (path + bytes), cached per run."""
    h = hashlib.sha256()
    for rel in _walk_py():
        h.update(rel.encode())
        with open(os.path.join(ROOT, rel), "rb") as fh:
            h.update(fh.read())
    d = h.hexdigest()
    return d[:12] if short else d


def effective_flags() -> Dict[str, Any]:
    """Every front-end switch and every PLACEMENT_* parameter, with its live value."""
    import parameters as P
    from eth80_frontend import _FRONTEND_SWITCHES

    names: List[str] = list(_FRONTEND_SWITCHES)
    for k in dir(P):
        if k.startswith(("PLACEMENT_", "FE_")):
            names.append(k)
    out: Dict[str, Any] = {}
    for k in sorted(set(names)):
        v = getattr(P, k, None)
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        else:
            out[k] = list(v) if isinstance(v, (tuple, list)) else str(v)
    return out


def frame_provider() -> Dict[str, Any]:
    import parameters as P
    return {"keypoint": bool(getattr(P, "PLACEMENT_FRAME_KEYPOINT", False)),
            "keypoint_sigma": str(getattr(P, "PLACEMENT_FRAME_KEYPOINT_SIGMA", "smallest")),
            "fallback": str(getattr(P, "PLACEMENT_FRAME_FALLBACK", "field"))}


def weight_source() -> Dict[str, Any]:
    """The weight in force and the file/field it was read from."""
    import parameters as P
    path = os.path.join(ROOT, "matching", "placement_score_params.json")
    src: Dict[str, Any] = {"file": os.path.relpath(path, ROOT), "field": None,
                           "value": None}
    try:
        with open(path) as fh:
            d = json.load(fh)
        for field in ("w_top1", "w", "weight"):
            if field in d:
                src["field"], src["value"] = field, d[field]
                break
        src["ml_at_boundary"] = d.get("ml_at_boundary")
        src["w_ml"] = d.get("w_ml")
    except Exception as exc:                                  # pragma: no cover
        src["error"] = "%s: %s" % (type(exc).__name__, exc)
    if hasattr(P, "PLACEMENT_M15_WEIGHT"):
        src["override_PLACEMENT_M15_WEIGHT"] = getattr(P, "PLACEMENT_M15_WEIGHT")
    return src


def endpoints() -> Dict[str, Any]:
    import parameters as P
    from matching.placement_matcher import frozen
    used = tuple(getattr(P, "PLACEMENT_M15_PLACE_LENSES", ()) or ())
    return {"flag_on": bool(getattr(P, "PLACEMENT_M15_PLACES", False)),
            "lenses": list(used) if used else list(frozen()["endpoint_lenses"]),
            "source": ("PLACEMENT_M15_PLACE_LENSES" if used
                       else "matching/placement_frozen_params.json:endpoint_lenses")}


def environment() -> Dict[str, Any]:
    """M28 item 1b: the machine that produced a number -- CPU model, cores, interpreter and the
    pinned libraries.  One source of truth: ``matching.placement_records.environment()``."""
    from matching.placement_records import environment as _env
    return _env()


def manifest() -> Dict[str, Any]:
    return {"flags": effective_flags(), "frame": frame_provider(),
            "weight": weight_source(), "endpoints": endpoints(),
            "code": code_digest(), "environment": environment()}


def check(payload: Dict[str, Any], where: str = "<memory>") -> None:
    """Raise if a result payload would record a number without its vector.

    A result is readable only with `manifest`: flags, frame provider, weight source,
    endpoint list and the code digest.  `write_result` attaches one; this is the gate
    that fails when something writes around it (e.g. a checkpoint builder that calls
    `json.dump` directly).
    """
    if not isinstance(payload, dict) or "manifest" not in payload:
        raise ValueError(
            "item 9: %r would record a number without a manifest "
            "(flags / frame / weight / endpoints / code digest).  Call "
            "m15_manifest.write_result() instead of json.dump()." % where)
    for key in ("flags", "frame", "weight", "endpoints", "code"):
        if key not in payload["manifest"]:
            raise ValueError("item 9: %r manifest is missing %r" % (where, key))


def write_result(path: str, payload: Dict[str, Any], *, force: bool = False
                 ) -> Dict[str, Any]:
    """Write `payload` with a manifest attached (always present in the file)."""
    payload = dict(payload)
    if "manifest" not in payload:
        payload["manifest"] = manifest()
    check(payload, path)
    if force:                                                 # pragma: no cover
        payload["manifest"]["forced"] = True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=float)
    return payload


# --------------------------------------------------------------------------- budget
def budget_report(rows: Iterable[Any]) -> Dict[str, Any]:
    """Item 9: how often the alignment refinement budget BINDS, by graph size.

    `rows` are `(n_cells, Alignment)` -- the alignment objects the matcher actually
    returned, so this counts the shipped code's exhaustion flag rather than modelling
    it.  `n_cells` is the number of admissible cells in the pair (the quantity the
    scaled budget is proportional to).
    """
    out: List[Dict[str, Any]] = []
    for cells, al in rows:
        out.append({"cells": int(cells),
                    "matched": len(getattr(al, "mapping", {}) or {}),
                    "exhausted": bool(getattr(al, "budget_exhausted", False))})
    n = len(out)
    n_ex = sum(1 for r in out if r["exhausted"])
    by_size: Dict[str, List[int]] = {}
    for r in out:
        bucket = ("0" if r["cells"] == 0 else
                  "1-64" if r["cells"] <= 64 else
                  "65-256" if r["cells"] <= 256 else ">256")
        by_size.setdefault(bucket, []).append(1 if r["exhausted"] else 0)
    return {"pairs": n, "exhausted": n_ex,
            "share": (n_ex / n) if n else None,
            "by_cell_bucket": {k: {"pairs": len(v), "exhausted": sum(v)}
                               for k, v in sorted(by_size.items())},
            "rows": out}
