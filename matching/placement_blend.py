"""M32 item 5: the ONE arithmetic a channel blend is allowed to use, shipped at zero.

    sim' = (1 - w_shape - w_colour) * matrix + w_shape * shape + w_colour * colour

`matrix` is what the matcher's row already carries, `shape` and `colour` are per-cell channel
similarities in [0, 1].  The configuration term keeps its shipped 0.25 -- it is part of `matrix`
and this module never touches it.

THE RULE THIS MODULE ENFORCES (fixed in writing before the read):

  * ``w_colour <= 0.15`` -- colour enters below shape and may not take over the row;
  * ``w_shape >= w_colour`` -- shape is the backbone; colour must earn parity by measurement;
  * ``0 <= w_shape + w_colour <= 1`` -- the blend is convex, so a score stays in [0, 1];
  * ``w_shape = w_colour = 0`` returns the matrix ITSELF, not a copy and not a rescale -- the
    deployed score is reproduced exactly, which is what M32 ships because no channel cleared
    its bar (see `MISSION32_REPORT.md`, items 3 and 4).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

#: [M32] the weights the bundle ships: nothing entered.  Both are ZERO by measurement -- the
#: colour distance fixed the flattening but lost rank to lightness, and the unclipped shape
#: columns gained +0.091 of a +0.10 bar.  Named ceilings, not a quiet weight.
SHIPPED: Dict[str, float] = {"w_shape": 0.0, "w_colour": 0.0}
MAX_W_COLOUR: float = 0.15

REFUSALS: Dict[str, str] = {}


def check_weights(w_shape: float, w_colour: float) -> None:
    """Refuse a blend the mission's ordering or the convexity rule forbids."""
    ws, wc = float(w_shape), float(w_colour)
    if ws < 0.0 or wc < 0.0:
        REFUSALS["negative"] = "w_shape %r / w_colour %r" % (ws, wc)
        raise ValueError("channel blend weights must be >= 0, got %r / %r" % (ws, wc))
    if wc > MAX_W_COLOUR + 1e-12:
        REFUSALS["colour_cap"] = "w_colour %r > %r" % (wc, MAX_W_COLOUR)
        raise ValueError("w_colour %r exceeds the mission's cap %r" % (wc, MAX_W_COLOUR))
    if ws < wc - 1e-12:
        REFUSALS["ordering"] = "w_shape %r < w_colour %r" % (ws, wc)
        raise ValueError("shape is the backbone: w_shape %r must be >= w_colour %r" % (ws, wc))
    if ws + wc > 1.0 + 1e-12:
        REFUSALS["convexity"] = "sum %r" % (ws + wc,)
        raise ValueError("the blend is convex: w_shape + w_colour = %r exceeds 1.0" % (ws + wc,))


def blend_channels(matrix: Any, shape: Optional[np.ndarray] = None,
                   colour: Optional[np.ndarray] = None,
                   w_shape: float = 0.0, w_colour: float = 0.0) -> Any:
    """The blend.  With both weights zero it returns ``matrix`` unchanged and untouched."""
    check_weights(w_shape, w_colour)
    ws, wc = float(w_shape), float(w_colour)
    if ws == 0.0 and wc == 0.0:
        return matrix                                   # identity: the deployed score itself
    M = np.array(matrix, dtype=np.float64, copy=True)
    if ws > 0.0:
        if shape is None:
            raise ValueError("w_shape %r needs a shape channel" % (ws,))
        M = (1.0 - ws - wc) * M + ws * np.asarray(shape, dtype=np.float64)
    if wc > 0.0:
        if colour is None:
            raise ValueError("w_colour %r needs a colour channel" % (wc,))
        M = (1.0 - ws - wc) * np.asarray(matrix, dtype=np.float64) if ws == 0.0 else M
        M = M + wc * np.asarray(colour, dtype=np.float64)
    return M


def shipped() -> Dict[str, Any]:
    """What this tree ships, for the bundle record and the report."""
    return {"weights": dict(SHIPPED), "max_w_colour": MAX_W_COLOUR,
            "shipped_because": "no channel cleared its pre-registered bar in M32 (colour: "
                               "aggregator; shape: +0.091 of +0.10); the blend is present, "
                               "tested and at zero, so the deployed score is reproduced exactly"}
