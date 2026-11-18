"""Compatibility shim for the pre-Move-3.5 ``derived_lenses`` module.

``lenses/derived.py`` is the aggregator that now owns the legacy surface:
its docstring states "Their public names are re-exported here, so the legacy
`from derived_lenses import X` surface is unchanged."  The export shipped the
aggregator but not the shim, so this file reconstructs it as a pure
re-export.  Reconstructed by the front-end fix session; see
work/topology/README.md.
"""
from lenses.derived import *  # noqa: F401,F403

# Explicit names used by orchestration/pipeline.py's `from derived_lenses
# import (...)`, so they are visible to static readers as well.
from lenses.derived import (  # noqa: F401
    BorderOwnershipLens,
    ContourGainState,
    ContourLens,
    CurveComplexLens,
    JunctionLens,
    EndStoppedLens,
    TextureBoundaryLens,
    SurfaceLens,
    RegionShapeLens,
    IllusoryContourLens,
    CurveComplexLens_B13,
    DerivedLenses,
    HigherLens,
)
