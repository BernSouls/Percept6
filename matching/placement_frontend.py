"""The front end the shipped system is measured under -- one place, in core (Mission 24).

Everything the shipped recogniser needs to turn an image into a *place record* lives here:

    apply_recipe()   set the front-end switches the record was built with
    load_image()     uint8 RGB
    cell()           (image, graph, side), memoised per process
    place_record()   the per-image record the door and the matcher read: one row per payload
                     place (content key, frozen configuration row + mask, frame) plus the
                     STANDARDISED per-place descriptors the deployed prefilter scores with

Why this file exists.  Until Mission 24 these three hundred lines were spread over
``Topology/placement_m20_eth80.py`` (the recipe and ``_cell``) and ``Topology/placement_m23_eth80.py``
(``image_job``), so the working system could only be run by experiment scripts.  A core package
may not import from ``Topology/``, so the code moved; the harnesses keep their own copies and a
test asserts the two recipes are IDENTICAL, which is what makes the moved copy trustworthy rather
than merely similar.

The recipe is not a set of knobs to tune here: it is the graph cache's identity.  ``RECIPE`` below
is the Mission-16/M20 recipe (the one the M19 matcher record was measured under), and it is folded
into the cache key through the front-end switch digest, so a record can never be served from
graphs built under a different recipe.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: The front end every shipped number is measured under (the M16 recipe, which is what the M19
#: record used).  Identical, key for key, to ``Topology.placement_m20_eth80.RECIPE``.
RECIPE: Dict[str, Any] = dict(FE_KEYPOINTS=True, FE_PLACE_LAYER=True, FE_SUBPIXEL_PEAK_POS=True,
                              FE_SUBPIXEL_GABOR_ANCHOR=True, PLACEMENT_FRAME_KEYPOINT=True,
                              PLACEMENT_FRAME_KEYPOINT_SIGMA="largest",
                              PLACEMENT_FRAME_FALLBACK="none", PLACEMENT_M15_PLACES=True,
                              KP_GRADED_SIMILARITY=True)

#: Switches the recipe implies but which are not front-end switches proper.
RECIPE_EXTRA: Dict[str, Any] = dict(KP_GRADED_POPULATION=True)

#: The keypoint class registration the payload lives under.
RECIPE_KEYPOINT_CLASS = "wide"

#: The deployed matcher's flags, exactly as ``placement_m22_eth80.pair_score_job`` sets them.
#: ``weight`` is read from the M19 bundle at call time (``matching/placement_score_params.json``).
MATCHER_FLAGS: Dict[str, Any] = dict(PLACEMENT_MATCHER_ENABLED=True,
                                     PLACEMENT_SCORE_EVIDENCE=True,
                                     PLACEMENT_MATCHER_STRICT=True)


def apply_recipe() -> None:
    """Set the front-end switches.  Idempotent; cheap; must precede any graph build."""
    import parameters as P
    for k, v in RECIPE.items():
        setattr(P, k, v)
    for k, v in RECIPE_EXTRA.items():
        setattr(P, k, v)
    from core import keypoint_classes as KPC
    KPC.unregister()
    KPC.register(RECIPE_KEYPOINT_CLASS)
    KPC.register_population()


def load_image(path: str) -> np.ndarray:
    """uint8 RGB, exactly as the harnesses load it."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


#: per-process memo of (image, graph, side) -- see ``Topology.placement_m20_eth80._CELLS`` for the
#: measurement that made it necessary (without it the side cache never hits and a worker grows by
#: hundreds of MB per image).
_CELLS: Dict[str, Tuple[Any, Any, Any]] = {}
_CELLS_MAX = 4


def clear_cells() -> None:
    """Drop the memo AND the matcher's side cache (they are keyed on ``id(graph)``)."""
    from matching import placement_matcher as PM
    _CELLS.clear()
    PM._SIDE_CACHE.clear()


def cell(path: str, tag: str = "core") -> Tuple[Any, Any, Any]:
    """(image, graph, side) for one image, memoised per process."""
    ent = _CELLS.get(path)
    if ent is not None:
        return ent
    from matching import placement_matcher as PM
    from matching import placement_cache as GC
    apply_recipe()
    img = load_image(path)
    g = GC.get_or_build(img, tag + "|" + path)
    ent = (img, g, PM.build_side_cached(g, img))
    if len(_CELLS) >= _CELLS_MAX:
        clear_cells()
    _CELLS[path] = ent
    return ent


def place_record(path: str, tag: str = "core",
                 content: Optional[str] = None) -> Dict[str, Any]:
    """One image -> the record every later stage reads.

    The body is ``Topology.placement_m23_eth80.image_job`` moved verbatim: same drops, same
    order of checks, same fields.  ``places`` is what the door indexes (content key + the frozen
    configuration row and its mask + the frame), ``std_all`` is every standardised descriptor
    channel the deployed prefilter differences (``D``, ``C``, ``S``, their mask and the node ids).
    Both come from ONE graph build, which is why this is one function and not two.
    """
    import time
    from matching import placement_prefilter as PF
    from matching import placement_edges as PE
    t0 = time.time()
    img, g, side = cell(path, tag)
    params = PF.load_params()
    all_nodes = sorted(int(n) for n in g.subgraph.nodes)
    pay = sorted({int(k[0]) for k in side.payload} | {int(k[1]) for k in side.payload})
    bag_all = PF.bag_v2(side, all_nodes, params["K"])
    std_all = PF.standardize(bag_all, params)
    raw = PF.bag_v2(side, pay, params["K"])
    std = PF.standardize(raw, params)
    frames, _sig = PE.node_frames_keypoint(g.subgraph, img)
    sub = g.subgraph
    rows: List[Dict[str, Any]] = []
    dropped = {"no_sig": 0, "no_config": 0, "no_position": 0, "no_scale": 0}
    for r, n in enumerate(raw["ids"]):
        s = g.sigs.get(n)
        if s is None:
            dropped["no_sig"] += 1
            continue
        if float(std["M"][r].sum()) < 2:
            dropped["no_config"] += 1
            continue
        anchor = sub.anchor_of_node.get(n) if hasattr(sub, "anchor_of_node") else None
        if anchor is None:
            dropped["no_position"] += 1
            continue
        sp = side.spacing_by_node.get(n)
        if sp is None or not np.isfinite(sp) or sp <= 0:
            dropped["no_scale"] += 1
            continue
        lens, sig = s
        rows.append({"n": int(n), "lens": str(lens),
                     "key": (str(lens), tuple(float(x) for x in
                                              (getattr(sig, "quantized", ()) or ()))),
                     "Dz": [float(x) for x in std["D"][r]],
                     "Mz": [float(x) for x in std["M"][r]],
                     "x": float(anchor[0]), "y": float(anchor[1]), "s": float(sp),
                     "theta": (None if n not in frames else float(frames[n]))})
    rec: Dict[str, Any] = {"path": path, "places": rows, "dropped": dropped,
            "n_payload": len(pay), "n_all_nodes": len(all_nodes),
            # every channel `standardize` produces: `score_bags_v2` differences D, C and S
            "std_all": {k: np.asarray(std_all[k], dtype=np.float32)
                        for k in ("D", "C", "S", "M")}
            | {"ids": np.asarray(std_all["ids"], dtype=np.int64)},
            "sec": time.time() - t0}
    if content and content != "keypoint":
        # M28 item 6: the place's content beyond the keypoint signature, at the place's own
        # scale, behind the bundle value `content.kind`.  Default builds are untouched.
        from matching import placement_content as PC
        PC.attach(rec, img, content)
    return rec


def matcher_flags(weight: float) -> Dict[str, Any]:
    """The exact flag set the shipped pair score runs under (M22 ``pair_score_job``)."""
    out = dict(MATCHER_FLAGS)
    out["PLACEMENT_SCORE_WEIGHT"] = float(weight)
    return out


def pair_score(qpath: str, rpath: str, weight: Optional[float] = None) -> float:
    """The shipped matcher's score for one (query, gallery) pair.

    Identical arithmetic to ``Topology.placement_m22_eth80.pair_score_job`` -- same entry point,
    same flags, same ``score_shipped`` -- so the shipped ordering IS this ordering.  The caller is
    responsible for restoring the flags (``recognise`` does it in a ``finally``); this function
    restores the ones it sets even when the pair raises.
    """
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from matching.placement_bundle import shipped_weight
    w = shipped_weight() if weight is None else float(weight)
    # the ORDER matters: images and graphs first (the recipe is applied by `cell`), THEN the
    # matcher flags, exactly as `pair_score_job` does it.  Building a cell under a differently
    # flagged `parameters` is not the configuration the record was measured in.
    q_img, q, _sq = cell(qpath, "corepair")
    r_img, r, _sr = cell(rpath, "corepair")
    flags = matcher_flags(w)
    old = {k: getattr(P, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(P, k, v)
        c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs, placement_images=(q_img, r_img))
        return float(c.score_shipped)
    finally:
        for k, v in old.items():
            setattr(P, k, v)
