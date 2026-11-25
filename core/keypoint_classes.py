"""core/keypoint_classes.py -- the keypoint lens classes' CONSTITUTION (M16).

WHY THIS FILE EXISTS (the verified defect, M16 report section 1)

`lenses/keypoints.py::signature_of` writes a `TypedSignature(kind="CANONICAL",
class_=<lens name>, quantized=(theta_bin, coherence_bin, scale, kind_index, dim))`
with **no `population`** and with a `class_` that is the DISPLAY lens name
(`CornerPoint`, `Blob`, `BoundaryArc`).  Every other lens in the tree writes the
UPPER_SNAKE registry key (`CORNER`, `BOUNDARY`, `COLOR`, ...).  The registries are
keyed by those UPPER_SNAKE names, so for a keypoint:

  * `get_similarity_ranges("Blob")` is **None**  -> `core/types.py:494-496` returns 0.0,
  * `get_population_axes("Blob")` is **None**     -> the graded branch is skipped,
  * `self.quantized == other.quantized` -> 1.0    (`core/types.py:492-493`).

So a keypoint pair scores exactly 1.0 when its quantised tuple is equal and exactly
0.0 otherwise: `THETA_BINS = 12` over pi means a 15 deg rotation is ONE bin, and one
bin is the difference between 1.0 and 0.0.  That is the mechanism behind Mission
15's (ii)@2 falling 1.00 -> 0.40 (5 deg) -> 0.05 (15 deg); the anchor stays put
((i)@2 0.82-0.93) while identity is destroyed by a bin edge.

WHAT THIS FILE ADDS

The three keypoint classes get a CONSTITUTION -- ranges, slot weights, circular
cycle lengths and (optionally) population axes -- so the SAME quantised tuple is
compared by the graded kernel that every other class uses.  Nothing here runs
unless `parameters.KP_GRADED_SIMILARITY` is on: registration is a data change to
the registries, and a graded reading changes every similarity the tree computes,
which is exactly what a dark flag is for.

UNITS (required by the mission, stated once here and repeated in the report)

Slot 0 theta      : BIN units, 12 bins over pi (15 deg per bin), CIRCULAR.
Slot 1 coherence  : BIN units, 4 bins over [0, 1].
Slot 2 scale      : PIXELS of Gaussian sigma ({1.0, 1.6, 2.5, 4.0} in the tree).
Slot 3 kind       : CATEGORICAL index (corner 0, blob 1, edge 2).
Slot 4 dim        : CATEGORICAL (1 = line-like, 2 = blob-like).

`CONSTITUTIONAL_RANGES[c]` is the axis span in the slot's OWN unit; the graded
kernel uses sigma_i = ranges[i] / 3 (the house convention, `core/types.py:521-524`).
So `ranges[0] = 3.0` means sigma_theta = ONE BIN = 15 deg, and `ranges[0] = 12.0`
means sigma_theta = four bins = 60 deg.  Those are the two variants measured in
`Topology/placement_m16_similarity.py`; the winner is selected on FIT, not by
argument.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

#: The classes this file governs: the three keypoint lenses plus the fallback name
#: `lenses/keypoints.py` uses for an unknown kind.
KEYPOINT_CLASSES: Tuple[str, ...] = ("CornerPoint", "Blob", "BoundaryArc", "Keypoint")

#: (theta_span_bins, weights) per variant.  theta span is in BINS; the kernel's
#: sigma_theta = span / 3, so `narrow` = 1 bin = 15 deg and `wide` = 4 bins = 60 deg.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "narrow": {
        "ranges": (3.0, 4.0, 3.0, 1.0, 1.0),
        "weights": (4.0, 1.0, 2.0, 1.0, 1.0),
        "cycle": {0: 12},
        "note": "sigma_theta = 1 bin = 15 deg; a 15 deg rotation is 1 sigma",
    },
    "wide": {
        "ranges": (12.0, 4.0, 3.0, 1.0, 1.0),
        "weights": (4.0, 1.0, 2.0, 1.0, 1.0),
        "cycle": {0: 12},
        "note": "sigma_theta = 4 bins = 60 deg; edge wobble is nearly free",
    },
}

#: EVERY registration this module performed, and every REFUSAL, so a dark feature
#: that silently does nothing cannot be mistaken for one that works (quality bar 3).
REGISTRATIONS: Dict[str, Any] = {}
REFUSALS: Dict[str, str] = {}
#: The registries as they were BEFORE this module touched them, keyed by explicit
#: TABLE NAME (never by `id()`: an id-keyed map silently restored one table's value
#: into another and made `core.slots` read a string as a PopAxis).
_PRIOR: Dict[str, Dict[str, Any]] = {}
_INSTALLED: Dict[str, set] = {}
_TABLES = ("CONSTITUTIONAL_RANGES", "SLOT_WEIGHT_PRIORS", "CIRCULAR_CYCLE_LENGTHS",
           "POPULATION_AXES")


def _registries() -> Any:
    from core import registries as R
    return R


def _install(table_name: str, cls: str, value: Any) -> None:
    R = _registries()
    table = getattr(R, table_name)
    _PRIOR.setdefault(table_name, {})
    if cls not in _PRIOR[table_name]:
        _PRIOR[table_name][cls] = table.get(cls, None)
    table[cls] = value
    _INSTALLED.setdefault(table_name, set()).add(cls)


def unregister() -> None:
    """Put back exactly what this module installed (idempotent, table by table)."""
    R = _registries()
    for table_name, classes in list(_INSTALLED.items()):
        table = getattr(R, table_name)
        prior = _PRIOR.get(table_name, {})
        for cls in sorted(classes):
            was = prior.get(cls, None)
            if was is None:
                table.pop(cls, None)
            else:
                table[cls] = was
        _INSTALLED[table_name] = set()
    REGISTRATIONS.clear()


def register(variant: str = "narrow", *, strict: bool = True) -> Dict[str, Any]:
    """Install the keypoint constitution into the live registries.  Idempotent.

    `strict` raises on an unknown variant instead of silently registering nothing
    (the programme has been bitten five times by a dark path that no-ops).
    """
    if variant not in VARIANTS:
        REFUSALS["register:%s" % variant] = "unknown variant"
        if strict:
            raise ValueError("keypoint_class variant %r unknown; have %s"
                             % (variant, sorted(VARIANTS)))
        return {}
    if variant in REGISTRATIONS:
        return REGISTRATIONS[variant]
    spec = VARIANTS[variant]
    for cls in KEYPOINT_CLASSES:
        _install("CONSTITUTIONAL_RANGES", cls, tuple(spec["ranges"]))
        _install("SLOT_WEIGHT_PRIORS", cls, tuple(spec["weights"]))
        _install("CIRCULAR_CYCLE_LENGTHS", cls, dict(spec["cycle"]))
    REGISTRATIONS[variant] = {"classes": list(KEYPOINT_CLASSES),
                              "ranges": list(spec["ranges"]),
                              "weights": list(spec["weights"]),
                              "cycle": dict(spec["cycle"]), "note": spec["note"]}
    return REGISTRATIONS[variant]


# --------------------------------------------------------------------- population
#: [M16 / the thesis] Bins are the INDEX; the graded vector is the reading.  The
#: population arm gives the keypoint classes tuning curves on the same axes their
#: bins quantise, so a rotation that stays inside a bin still moves the reading
#: (sub-bin resolution) instead of being invisible until the edge.
#: (name, n_bins, circular, sigma_bins, weight) x 3 axes, per measured variant.
#: The DEFAULT is the FIT winner ("theta2-balanced": mean AUC 0.959 over
#: 0/5/15 deg/1.15x on the synthetic FIT images, against 0.942 for theta-sigma 1
#: and 0.752 for the shipped 0/1 reading).
POPULATION_SPECS: Dict[str, Tuple[Tuple[str, int, bool, float, float], ...]] = {
    "theta1-heavy": (("theta", 12, True, 1.0, 8.0),
                     ("coherence", 4, False, 1.0, 1.0),
                     ("scale", 4, False, 1.0, 1.0)),
    "theta1-balanced": (("theta", 12, True, 1.0, 4.0),
                        ("coherence", 4, False, 1.0, 1.0),
                        ("scale", 4, False, 1.0, 2.0)),
    "theta2-heavy": (("theta", 12, True, 2.0, 8.0),
                     ("coherence", 4, False, 1.0, 1.0),
                     ("scale", 4, False, 1.0, 1.0)),
    "theta2-balanced": (("theta", 12, True, 2.0, 4.0),
                        ("coherence", 4, False, 1.0, 1.0),
                        ("scale", 4, False, 1.0, 2.0)),
}
POPULATION_AXIS_SPEC = POPULATION_SPECS["theta2-balanced"]


def population_spec() -> Tuple[Tuple[str, int, bool, float, float], ...]:
    """The spec in force: the parameter's variant, refused loudly if unknown."""
    import parameters as P
    name = str(getattr(P, "KP_GRADED_POP_VARIANT", "theta2-balanced"))
    spec = POPULATION_SPECS.get(name)
    if spec is None:
        REFUSALS["population_variant:%s" % name] = "unknown population variant"
        raise ValueError("KP_GRADED_POP_VARIANT %r unknown; have %s"
                         % (name, sorted(POPULATION_SPECS)))
    return spec


def population_axes() -> Tuple[Any, ...]:
    from core.population import PopAxis
    return tuple(PopAxis(name=n, n_bins=int(b), circular=bool(c),
                         sigma_bins=float(s), weight=float(w))
                 for (n, b, c, s, w) in population_spec())


def register_population() -> Dict[str, Any]:
    """Install the population axes for the keypoint classes.  Idempotent."""
    axes = population_axes()
    for cls in KEYPOINT_CLASSES:
        _install("POPULATION_AXES", cls, axes)
    REGISTRATIONS["population"] = {"axes": [a.name for a in axes],
                                   "classes": list(KEYPOINT_CLASSES)}
    return REGISTRATIONS["population"]


def population_of(p: Any) -> Tuple[Tuple[float, ...], ...]:
    """The graded activity vector of one keypoint, or () if the arm is off.

    theta      : the SUB-BIN angle over [0, pi), circular.
    coherence  : the measured coherence over [0, 1].
    scale      : log2 of the Gaussian sigma, over the tree's four sigmas
                 (1.0 .. 4.0 -> 0 .. 2 octaves), so a scale step is one octave.
    """
    import math
    import parameters as P
    if not bool(getattr(P, "KP_GRADED_POPULATION", False)):
        return ()
    from core.population import soft_bump_from_value
    # NOTE (measured, not assumed): the axes come from the SPEC, not the registry, so
    # this always returns a vector when the flag is on.  What decides whether the
    # reader USES it is `get_population_axes(class_)` plus `POPULATION_CODING_ENABLED`
    # at core/types.py:457-469 -- i.e. the arm needs the constitution REGISTERED on
    # the reading side, and the payload PRESENT on the writing side.  A graph built
    # with `KP_GRADED_POPULATION` off carries no payload even after re-registration;
    # the graph harness's equivalence build is what caught that.
    axes = population_axes()
    th = float(p.theta) % math.pi
    lvl = math.log2(max(float(p.scale), 1e-6) / 1.0)
    vals = ((th, 0.0, math.pi), (float(p.coherence), 0.0, 1.0), (lvl, 0.0, 2.0))
    out = []
    for (v, lo, hi), ax in zip(vals, axes):
        out.append(soft_bump_from_value(v, lo, hi, ax))
    return tuple(out)


def active() -> bool:
    import parameters as P
    return bool(getattr(P, "KP_GRADED_SIMILARITY", False))


def maybe_register(*, strict: bool = True) -> Dict[str, Any]:
    """Register when the flag is on; report (never raise) when it is off.

    Bar 3 -- NO SILENT FALLBACK: the population arm REQUIRES the graded arm (its
    axes must be registered), so asking for it alone raises rather than quietly
    producing bins-only signatures.
    """
    import parameters as P
    if not active():
        if bool(getattr(P, "KP_GRADED_POPULATION", False)):
            REFUSALS["population_without_graded"] = (
                "KP_GRADED_POPULATION needs KP_GRADED_SIMILARITY: the axes are not "
                "registered, so the activity vector would never be read")
            if strict:
                raise RuntimeError(REFUSALS["population_without_graded"])
        return {}
    got = register(str(getattr(P, "KP_GRADED_VARIANT", "narrow")), strict=strict)
    if bool(getattr(P, "KP_GRADED_POPULATION", False)):
        got = dict(got)
        got["population"] = register_population()
    return got


def report() -> Dict[str, Any]:
    """What the registries hold for the keypoint classes RIGHT NOW."""
    R = _registries()
    return {cls: {"ranges": R.CONSTITUTIONAL_RANGES.get(cls),
                  "weights": tuple(R.SLOT_WEIGHT_PRIORS.get(cls, ())),
                  "cycle": R.CIRCULAR_CYCLE_LENGTHS.get(cls),
                  "axes": R.get_population_axes(cls)}
            for cls in KEYPOINT_CLASSES}
