"""The one frozen configuration of the shipped recogniser (Mission 24).

A ``PlacementBundle`` is a JSON file (``matching/placement_bundle.json``) that pins EVERYTHING
the shipped path does, and the fingerprint of the code that reads it:

* the M19 matcher bundle it calls (``matching/placement_score_params.json``) and the weight
  ``w = 0.375`` that record chose;
* the front-end recipe (the graph identity), key for key;
* the door settings: the counted door's ``n_hits``, the appearance descriptor's bin convention
  (named, not copied -- the code is the definition) and the rank-fusion constant ``k = 60``;
* the shortlist size ``S = 8`` and the final order: the matcher's plain re-order;
* provenance (mission, date, what measured it) and ``code_digest``.

**A missing or mismatched file is a refusal.**  Nothing in the shipped path has a default for any
of these numbers, and a refusal names what is wrong and how to re-freeze.  That is the whole point
of the file: the configuration that produced a number travels with the number, and an edit to the
code cannot silently inherit an old configuration's claims.

Freezing is an operator action, not a runtime one:

    python -m matching.placement_recognise --freeze
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The bundle file itself.
BUNDLE_PATH = os.path.join(ROOT, "matching", "placement_bundle.json")

#: The M19 record the matcher's weight comes from.
SCORE_PARAMS = os.path.join(ROOT, "matching", "placement_score_params.json")

#: The frozen Mission-21 ruler (per-slot standardisation + widths) the door's configuration
#: distance and the deployed prefilter read.
PREFILTER_PARAMS = os.path.join(ROOT, "matching", "placement_prefilter_params.json")

#: Core modules whose bytes define the shipped behaviour.  ``placement_bundle.py`` is deliberately
#: NOT in this list: it holds the digest, and a file cannot contain its own hash.
CODE_MODULES = ("matching/placement_frontend.py", "matching/placement_cache.py",
                "matching/placement_frontdoor.py", "matching/placement_gallery.py",
                "matching/placement_recognise.py", "matching/placement_views.py",
                "matching/placement_content.py",
                # [M32] the digest is supposed to cover every module whose bytes can move a
                # shipped score.  The fuse, the matcher, the graded kernel, the shape similarity,
                # the record loader and the channel blend all could, and none of them was in the
                # tuple: a change to any of them froze a bundle that still said "this tree".
                "matching/placement_fuse.py", "matching/placement_matcher.py",
                "matching/placement_kernel.py", "matching/placement_shape.py",
                "matching/placement_records.py", "matching/placement_blend.py",
                # [M34] the fragment channel and the anchors it reads
                "matching/placement_fragment.py", "matching/placement_anchors.py")

#: Data files whose bytes define the shipped behaviour (frozen parameters, not code).
CODE_DATA = ("matching/placement_score_params.json", "matching/placement_prefilter_params.json")


class BundleRefusal(RuntimeError):
    """The shipped configuration is missing, unreadable, or not the configuration of this code."""


def compute_code_digest() -> str:
    """sha256 over the shipped path's own bytes: the code modules and the frozen parameters."""
    h = hashlib.sha256()
    for rel in CODE_MODULES + CODE_DATA:
        path = os.path.join(ROOT, rel)
        h.update(rel.encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _content_digest(kind: str) -> str:
    """The content class's own digest, or "" when the kind has no registered class."""
    try:
        from matching import placement_content as PC
        d = getattr(PC, "digest", None)
        return str(d() if callable(d) else "")[:16]
    except Exception:                                   # pragma: no cover - defensive
        return ""


def shipped_weight() -> float:
    """The M19 weight (``w = 0.375``), read from that record and never hard-coded."""
    with open(SCORE_PARAMS) as fh:
        return float(json.load(fh)["w"])


@dataclass(frozen=True)
class PlacementBundle:
    """The frozen configuration.  Immutable: a runtime cannot re-tune the shipped path."""

    name: str
    mission: str
    date: str
    shortlist: int
    rrf_k: int
    rerank: str
    door_n_hits: int
    appearance: str
    matcher_weight: float
    recipe: Dict[str, Any]
    recipe_extra: Dict[str, Any]
    keypoint_class: str
    reference_views: List[str] = field(default_factory=list)
    reserved: List[str] = field(default_factory=list)
    references: Dict[str, Any] = field(default_factory=dict)
    door_kind: str = "counted"                 # "counted" (M22-M25) or "graded" (M26)
    door_params: Dict[str, Any] = field(default_factory=dict)
    shortlist_rule: str = "fixed"              # "fixed" (M22-M25) or "adaptive" (M26 item 6D)
    shortlist_wide: int = 16
    margin_tau: float = 0.0
    content: Dict[str, Any] = field(default_factory=dict)   # M28: {"kind": "keypoint"|"colour_rings"}
    fusion: Dict[str, float] = field(default_factory=dict)  # M31: {"colour": 0.5} (RRF weights)
    matcher_blend: Dict[str, float] = field(default_factory=dict)  # M31: {"w_cfg":0.25,"w_col":0.10}
    provenance: Dict[str, Any] = field(default_factory=dict)
    code_digest: str = ""
    file_digest: str = ""

    # ---- content (M28 item 6)
    @property
    def content_kind(self) -> str:
        """`content.kind`: what a place's content is beyond the keypoint signature.

        Absent -> "keypoint", which is what every record before M28 carries (no fallback: an
        unknown value is refused by ``matching.placement_content.kind``)."""
        from matching import placement_content as PC
        raw = self.content or {}
        k = raw.get("kind", PC.DEFAULT_KIND) if isinstance(raw, dict) else raw
        return PC.kind(str(k))

    # ---- fusion (M31)
    @property
    def fusion_weight(self) -> float:
        """The colour channel's RRF weight: `door.params.colour_weight`, else 0 (no colour)."""
        raw = dict(self.door_params or {})
        return float(raw.get("colour_weight", self.fusion.get("colour", 0.0)))

    @property
    def fragment_weight(self) -> float:
        """[M34] the fragment channel's RRF weight: `fusion.fragment`, else 0 (no fragments).

        A bundle that names a non-zero weight is a bundle that asks the door for the channel; the
        recogniser refuses to run it without anchors rather than quietly computing the four-channel
        door its bundle does not describe.
        """
        raw = dict(self.fusion or {})
        return float(raw.get("fragment", 0.0))

    @property
    def colour_rings(self) -> bool:
        return self.content_kind == "colour_rings"

    # ---- io
    @classmethod
    def load(cls, path: Optional[str] = None) -> "PlacementBundle":
        """Load and VALIDATE the bundle.  Refuses rather than defaulting."""
        p = path or BUNDLE_PATH
        if not os.path.exists(p):
            raise BundleRefusal(
                "no frozen bundle at %s -- the shipped recogniser refuses to run on defaults. "
                "Create it with `python -m matching.placement_recognise --freeze` and review the "
                "diff before it is committed." % p)
        try:
            with open(p, "rb") as fh:
                raw_bytes = fh.read()
            d = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise BundleRefusal("bundle at %s is unreadable: %r" % (p, exc)) from exc
        missing = [k for k in ("name", "shortlist", "rrf_k", "rerank", "door", "appearance",
                               "matcher", "frontend", "references", "code_digest") if k not in d]
        if missing:
            raise BundleRefusal("bundle at %s is missing %s -- refusing rather than guessing"
                                % (p, ", ".join(missing)))
        if str(d.get("shortlist_rule", "fixed")) not in ("fixed", "adaptive"):
            raise BundleRefusal("bundle at %s names a shortlist rule %r this tree does not "
                                "implement (have: fixed, adaptive)"
                                % (p, d.get("shortlist_rule")))
        if str(d["door"].get("kind", "counted")) not in ("counted", "graded", "fused3", "bag"):
            raise BundleRefusal("bundle at %s names a door kind %r this tree does not implement "
                                "(have: counted, graded, fused3, bag)" % (p, d["door"].get("kind")))
        have = str(d["code_digest"])
        want = compute_code_digest()
        if have != want:
            raise BundleRefusal(
                "bundle at %s was frozen for different code (code_digest %s, this tree %s).  The "
                "shipped configuration no longer matches the shipped code: re-freeze with "
                "`python -m matching.placement_recognise --freeze`, or restore the code."
                % (p, have[:12], want[:12]))
        b = cls(name=str(d["name"]),
                mission=str(d.get("mission", "")),
                date=str(d.get("date", "")),
                shortlist=int(d["shortlist"]),
                rrf_k=int(d["rrf_k"]),
                rerank=str(d["rerank"]),
                door_n_hits=int(d["door"]["n_hits"]),
                door_kind=str(d["door"].get("kind", "counted")),
                door_params=dict(d["door"].get("params") or {}),
                shortlist_rule=str(d.get("shortlist_rule", "fixed")),
                shortlist_wide=int(d.get("shortlist_wide", 16)),
                margin_tau=float(d.get("margin_tau", 0.0)),
                appearance=str(d["appearance"]),
                matcher_weight=float(d["matcher"]["weight"]),
                recipe=dict(d["frontend"]["recipe"]),
                recipe_extra=dict(d["frontend"].get("extra_pre") or {}),
                keypoint_class=str(d["frontend"].get("keypoint_class", "wide")),
                reference_views=[str(v) for v in (d["references"].get("views") or [])],
                reserved=[str(v) for v in (d["references"].get("reserved") or [])],
                references=dict(d["references"]),
                content=dict(d.get("content") or {}),
                fusion=dict(d.get("fusion") or {}),
                matcher_blend=dict(d["matcher"].get("blend") or {}),
                provenance=dict(d.get("provenance") or {}),
                code_digest=have,
                file_digest=hashlib.sha256(raw_bytes).hexdigest())
        return b

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mission": self.mission, "date": self.date,
                "shortlist": self.shortlist, "rrf_k": self.rrf_k, "rerank": self.rerank,
                "code_digest": self.code_digest}


def freeze(path: Optional[str] = None, *, date: str, provenance: Dict[str, Any],
           k: Optional[int] = None, reserved: Sequence[str] = ("035-045", "045-180"),
           door_kind: str = "counted", door_params: Optional[Dict[str, Any]] = None,
           shortlist: int = 8, shortlist_rule: str = "fixed", shortlist_wide: int = 16,
           margin_tau: float = 0.0,
           content_kind: str = "keypoint", fusion: Optional[Dict[str, float]] = None,
           matcher_blend: Optional[Dict[str, float]] = None,
           mission: str = "M24"
           ) -> Dict[str, Any]:
    """Write the bundle for THIS tree.  Operator action; prints the digest it froze.

    ``k`` is the number of stored views per object the bundle ships as its default; the views
    themselves come from ``matching/placement_views.reference_views`` (the score-free rule), and the
    read's query views are recorded as RESERVED so they can never be stored.
    """
    from matching import placement_frontend as FE
    from matching import placement_views as V
    import json as _json
    k = int(k if k is not None else 4)
    views = V.reference_views(k, reserved)
    p = path or BUNDLE_PATH
    with open(SCORE_PARAMS) as fh:
        w = float(_json.load(fh)["w"])
    d = {
        "name": "placement-shipped-v1",
        "mission": str(mission),
        "date": date,
        "shortlist": int(shortlist),
        "shortlist_rule": str(shortlist_rule),
        "shortlist_wide": int(shortlist_wide),
        "margin_tau": float(margin_tau),
        "rrf_k": 60,
        "rerank": "plain",
        "door": {"n_hits": 8, "kind": str(door_kind), "params": dict(door_params or {}),
                 "note": "the counted door: each query place counts its n_hits nearest pool "
                         "places by the frozen configuration ruler, one vote per object; the "
                         "object score is count / sqrt(n_query_places * n_object_places)"},
        "appearance": "colour 4x4x4 + gradient-orientation 12 bins, L2, cosine "
                      "(matching/placement_frontdoor.appearance_feature)",
        "matcher": {"bundle": "matching/placement_score_params.json", "weight": w,
                    "entry": "matching.joint_mcs.joint_mcs -> score_shipped",
                    "flags": FE.MATCHER_FLAGS,
                    "blend": dict(matcher_blend or {})},
        "content": {"kind": str(content_kind), "digest": _content_digest(str(content_kind)),
                    "kinds": ["keypoint", "colour_rings"],
                    "note": "what a place carries beyond its keypoint signature.  'colour_rings' "
                            "is six CIELAB ring means at the place's own scale "
                            "(matching/placement_content.py); every record and read that uses a "
                            "non-default kind names it in its manifest."},
        "fusion": dict(fusion or {}),
        "frontend": {"recipe": dict(FE.RECIPE), "extra_pre": dict(FE.RECIPE_EXTRA),
                     "keypoint_class": FE.RECIPE_KEYPOINT_CLASS,
                     "params": "matching/placement_prefilter_params.json",
                     "note": "the recipe is the graph cache's identity; it is the M16/M20 recipe "
                             "the M19 matcher record was measured under"},
        "references": {"k": k, "views": views, "reserved": list(reserved),
                       "rule": "kcenter-4-pinned-000-000-v1: {000-000} union the 3-subset of "
                               "the non-reserved grid views that minimises the covering radius "
                               "over all 41 views (exhaustive, ties by label) -- see "
                               "matching/placement_views.py",
                       "grid_views": len(V.GRID),
                       "min_spacing_deg": round(V.min_spacing(), 2),
                       "covering_deg": round(V.covering_radius(views), 2),
                       "nearest_stored": V.distances_to_nearest(views, reserved)},
        "provenance": provenance,
        "code_digest": compute_code_digest(),
    }
    # Documentation keys that are NOT part of the shipped configuration (`measured`) are carried
    # over rather than silently dropped: freeze() writes the configuration, and the record of what
    # was measured under it is the report's, not the freezer's, to rewrite.
    try:
        with open(p) as fh:
            prior = _json.load(fh)
        for key, val in prior.items():
            if key not in d:
                d[key] = val
    except (OSError, ValueError):
        pass
    with open(p, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return d


def refreeze_from_current(path: Optional[str] = None) -> Dict[str, Any]:
    """Re-freeze THIS tree, keeping the configuration the bundle already ships.

    ``freeze()`` writes whatever configuration it is given and its defaults are the M24 ones
    (counted door, keypoint content).  Re-freezing after a code edit must NOT silently revert the
    shipped configuration to those defaults -- M32 hit exactly that, and the door kind fell back
    from "fused3" to "counted" while the digest said the tree had changed.  This reads the
    current file's own configuration (the digest is not consulted, that is the point) and hands it
    back to ``freeze()``, so the only thing that moves is ``code_digest``.
    """
    import json as _json
    p = path or BUNDLE_PATH
    with open(p) as fh:
        prior = _json.load(fh)
    door = dict(prior.get("door") or {})
    content = dict(prior.get("content") or {})
    matcher = dict(prior.get("matcher") or {})
    refs = dict(prior.get("references") or {})
    return freeze(
        p,
        date=str(prior.get("date") or ""),
        provenance=dict(prior.get("provenance") or {}),
        k=int(refs.get("k") or 4),
        reserved=tuple(refs.get("reserved") or ("035-045", "045-180")),
        door_kind=str(door.get("kind") or "counted"),
        door_params=dict(door.get("params") or {}),
        shortlist=int(prior.get("shortlist") or 8),
        shortlist_rule=str(prior.get("shortlist_rule") or "fixed"),
        shortlist_wide=int(prior.get("shortlist_wide") or 16),
        margin_tau=float(prior.get("margin_tau") or 0.0),
        content_kind=str(content.get("kind") or "keypoint"),
        fusion=dict(prior.get("fusion") or {}),
        matcher_blend=dict(matcher.get("blend") or {}),
        mission=str(prior.get("mission") or "M24"),
    )


def load(path: Optional[str] = None) -> PlacementBundle:
    """Convenience alias for ``PlacementBundle.load``."""
    return PlacementBundle.load(path)


if __name__ == "__main__":                       # pragma: no cover - operator entry point
    import argparse as _ap
    _p = _ap.ArgumentParser(description="re-freeze the shipped bundle for THIS tree")
    #: the shipped configuration is kept; only the code digest moves
    _p.add_argument("--refreeze", action="store_true")
    _a = _p.parse_args()
    if _a.refreeze:
        _d = refreeze_from_current()
        print("re-frozen: door %s, content %s, shortlist %s, code_digest %s"
              % (_d["door"]["kind"], _d["content"]["kind"], _d["shortlist"],
                 _d["code_digest"][:16]))
    else:
        _p.print_help()
