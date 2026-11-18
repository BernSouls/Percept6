"""Minimal reconstruction of the export's missing ``_bootstrap`` module.

``eth80_frontend.py`` (the repository's own image -> match-graph front end)
imports ROOT / DATA_DIR / CACHE_DIR / switch_overrides_key from here.  The
original bootstrap also owned the diagnostics package path and the
``PERCEPT_SET`` switch overrides; only what the shipped front end reads is
reconstructed, with the same semantics described in its docstrings:

  * DATA_DIR  -- root of the ETH-80 corpus (data/ETH-80-master).
  * CACHE_DIR -- on-disk graph cache (keyed by image bytes + source digest).
  * switch_overrides_key() -- canonical key for run-time parameter overrides
    (``PERCEPT_SET``), folded into the cache key so two arms of an experiment
    never share graphs.

Reconstructed by the front-end fix session; see work/topology/README.md.
"""
from __future__ import annotations

import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "PERCEPT_DATA_DIR", os.path.join(ROOT, "data", "ETH-80-master"))
CACHE_DIR = os.environ.get(
    "PERCEPT_CACHE_DIR", os.path.join(ROOT, ".cache", "eth80_graphs"))

_SWITCH_PREFIX = "PERCEPT_"


def switch_overrides(spec: str = "") -> dict:
    """Parse ``NAME=VALUE;NAME=VALUE`` into a dict of canonical assignments.

    VALUES are interpreted as int / float / bool / str in that order; an
    unparseable value stays a string.  Unknown names are kept (the caller
    applies them with ``setattr``); this is a diagnostic hook, not a config
    file.
    """
    out = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name, value = name.strip(), value.strip()
        if not name:
            continue
        if value.lower() in ("true", "false"):
            v = value.lower() == "true"
        else:
            try:
                v = int(value)
            except ValueError:
                try:
                    v = float(value)
                except ValueError:
                    v = value
        out[name] = v
    return out


def apply_switch_overrides(spec: str = "") -> dict:
    """Apply ``PERCEPT_SET`` overrides to ``parameters`` and return them."""
    import parameters as P

    spec = spec if spec else os.environ.get("PERCEPT_SET", "")
    applied = {}
    for name, value in switch_overrides(spec).items():
        if hasattr(P, name):
            setattr(P, name, value)
            applied[name] = value
    return applied


def switch_overrides_key(spec: str = "") -> str:
    """Canonical, order-independent digest of the run's switch overrides.

    Empty string means "no overrides": the front end uses that as the
    sentinel for the default configuration.
    """
    spec = spec if spec else os.environ.get("PERCEPT_SET", "")
    ov = switch_overrides(spec)
    if not ov:
        return ""
    canonical = ";".join("%s=%r" % (k, ov[k]) for k in sorted(ov))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# The export's original bootstrap installed the diagnostics package alias.
# Keep the same effect for any module that still refers to it.
if not os.path.isdir(os.path.join(ROOT, "diagnostics")):
    os.makedirs(os.path.join(ROOT, "diagnostics"), exist_ok=True)
