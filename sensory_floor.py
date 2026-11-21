"""Compatibility shim for the pre-Move-3a ``sensory_floor`` module.

The repository is an *export*: ``sensory_floor.py`` itself was not shipped,
but its top-level statements were moved verbatim into ``core/types.py``
(TypedSignature / Detection / serialize_detection),
``perception/observation.py`` (Action lens symbols / resolve_anchor_position)
and ``perception/polar.py`` (polar_boundary_fragments).  The extracted
modules' own docstrings say:

    "The re-export shim in sensory_floor.py keeps `from sensory_floor
     import X` working until the importer repoint retires it."

Ten shipped modules still use ``from sensory_floor import ...``.  This file
reconstructs exactly that shim: no perception or matching logic lives here,
it only re-exports the canonical definitions so the shipped import graph
resolves.  Reconstructed by the front-end fix session; see
work/topology/README.md.
"""
from __future__ import annotations

from core.types import (  # noqa: F401
    TypedSignature,
    Detection,
    serialize_detection,
)
from perception.observation import (  # noqa: F401
    ACTION_LENS_NAME,
    ACTION_ROLE_NAME,
    action_signature,
    resolve_anchor_position,
)
from perception.polar import polar_boundary_fragments  # noqa: F401

__all__ = (
    "TypedSignature",
    "Detection",
    "serialize_detection",
    "ACTION_LENS_NAME",
    "ACTION_ROLE_NAME",
    "action_signature",
    "resolve_anchor_position",
    "polar_boundary_fragments",
)
