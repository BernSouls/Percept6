"""MISSION 22 / (B) -- THE ASSEMBLED SYSTEM: front door, then the matcher, and nothing removed.

    appearance shortlist (S = 8)  ->  the shipped matcher scores those 8  ->  re-order

The rerank REORDERS the shortlist and never removes anything from it: the assembled system can
only be as good as its front door at S = 8 and no better, which is exactly why both numbers are
reported in the same run.  If the matcher ran first it would be the M19 system and there would
be no front door at all; if it removed candidates it would be a filter and its recall would be
the front door's recall, not its own.

The score the matcher contributes is the SHIPPED pair score (``score_shipped`` from
``joint_mcs`` with the deployed flags: the identity score plus the placement evidence at the
frozen weight).  This module does not re-implement it -- it calls the harness's own pair job, so
"assembled" means the same scorer the record is about, not a lookalike.

Flags: ``PLACEMENT_ASSEMBLED`` (default off).  With it off nothing in the tree calls
`assemble`; the front door and the matcher stay as they are.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def rank_by(score: Dict[int, float], cands: Sequence[int]) -> List[int]:
    """Order ``cands`` by score, deterministic, never dropping one."""
    return sorted(cands, key=lambda j: (-float(score.get(j, 0.0)), j))


def assemble(shortlist: Sequence[int], score_of: Callable[[int], float]
             ) -> Tuple[List[int], Dict[int, float]]:
    """Re-order a shortlist with the matcher's score.  Every candidate is kept.

    ``shortlist``    object indices from the front door, in the front door's order
    ``score_of``     object index -> the shipped pair score for (query, that object)

    Returns (re-ordered list, the score table).  A candidate the scorer could not score keeps
    its front-door position at the END of the scored ones -- it is never dropped and never
    silently promoted, and the count of unscoreable candidates is reported by the caller.
    """
    scores: Dict[int, float] = {}
    missing: List[int] = []
    for j in shortlist:
        try:
            scores[j] = float(score_of(j))
        except Exception:                                     # noqa: BLE001 - reported
            missing.append(j)
    scored = [j for j in shortlist if j in scores]
    ordered = rank_by(scores, scored) + [j for j in shortlist if j in missing]
    return ordered, scores


def top1(ordered: Optional[Sequence[int]], truth: int) -> int:
    """1 when the truth is first, 0 otherwise.  Absent counts as 0, never as a pass."""
    return 1 if (ordered and ordered[0] == truth) else 0
