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
from typing import Any, Dict, Optional

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
                "matching/placement_recognise.py")

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
    provenance: Dict[str, Any] = field(default_factory=dict)
    code_digest: str = ""
    file_digest: str = ""

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
                               "matcher", "frontend", "code_digest") if k not in d]
        if missing:
            raise BundleRefusal("bundle at %s is missing %s -- refusing rather than guessing"
                                % (p, ", ".join(missing)))
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
                appearance=str(d["appearance"]),
                matcher_weight=float(d["matcher"]["weight"]),
                recipe=dict(d["frontend"]["recipe"]),
                recipe_extra=dict(d["frontend"].get("extra_pre") or {}),
                keypoint_class=str(d["frontend"].get("keypoint_class", "wide")),
                provenance=dict(d.get("provenance") or {}),
                code_digest=have,
                file_digest=hashlib.sha256(raw_bytes).hexdigest())
        return b

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mission": self.mission, "date": self.date,
                "shortlist": self.shortlist, "rrf_k": self.rrf_k, "rerank": self.rerank,
                "code_digest": self.code_digest}


def freeze(path: Optional[str] = None, *, date: str, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Write the bundle for THIS tree.  Operator action; prints the digest it froze."""
    from matching import placement_frontend as FE
    import json as _json
    p = path or BUNDLE_PATH
    with open(SCORE_PARAMS) as fh:
        w = float(_json.load(fh)["w"])
    d = {
        "name": "placement-shipped-v1",
        "mission": "M24",
        "date": date,
        "shortlist": 8,
        "rrf_k": 60,
        "rerank": "plain",
        "door": {"n_hits": 8,
                 "note": "the counted door: each query place counts its n_hits nearest pool "
                         "places by the frozen configuration ruler, one vote per object; the "
                         "object score is count / sqrt(n_query_places * n_object_places)"},
        "appearance": "colour 4x4x4 + gradient-orientation 12 bins, L2, cosine "
                      "(matching/placement_frontdoor.appearance_feature)",
        "matcher": {"bundle": "matching/placement_score_params.json", "weight": w,
                    "entry": "matching.joint_mcs.joint_mcs -> score_shipped",
                    "flags": FE.MATCHER_FLAGS},
        "frontend": {"recipe": dict(FE.RECIPE), "extra_pre": dict(FE.RECIPE_EXTRA),
                     "keypoint_class": FE.RECIPE_KEYPOINT_CLASS,
                     "params": "matching/placement_prefilter_params.json",
                     "note": "the recipe is the graph cache's identity; it is the M16/M20 recipe "
                             "the M19 matcher record was measured under"},
        "provenance": provenance,
        "code_digest": compute_code_digest(),
    }
    with open(p, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return d


def load(path: Optional[str] = None) -> PlacementBundle:
    """Convenience alias for ``PlacementBundle.load``."""
    return PlacementBundle.load(path)
