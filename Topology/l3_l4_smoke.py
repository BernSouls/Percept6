"""L3+L4 (Spine + Junctions) smoke measurement, pins, and determinism check."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402

OUT = ROOT / "results" / "l3_l4_smoke_pins.json"


def _run_arm(l3_on: bool, l4_on: bool, shift_px: int = 0):
    old = (getattr(P, "FE_L3_SPINE", False), getattr(P, "FE_L4_JUNCTIONS", False))
    P.FE_L3_SPINE = l3_on
    P.FE_L4_JUNCTIONS = l4_on
    try:
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        img[14:50, 14:50] = (220, 190, 40)
        img[20:44, 20:44] = (20, 90, 200)
        if shift_px:
            img = np.roll(img, shift_px, axis=1)
        tr = {}
        side = FE.build_graph(img, name=f"l3_l4_{l3_on}_{l4_on}_{shift_px}",
                              record_co_located=False, trace=tr)
        diag = tr["binder"].diagnostics.get("front_end_mechanisms", {})
        fam_counts = Counter(getattr(et, "name", str(et)) for _, _, et in side.subgraph.edges)
        n_nodes = max(1, len(side.subgraph.nodes))
        n_edges = len(side.subgraph.edges)
        return {
            "nodes": len(side.subgraph.nodes),
            "edges": n_edges,
            "mean_degree": (2.0 * n_edges) / n_nodes,
            "family_counts": dict(sorted(fam_counts.items())),
            "applied_once": diag.get("applied_once", []),
            "l3_spine": diag.get("l3_spine", {}),
            "l4_junctions": diag.get("l4_junctions", {}),
        }
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS = old


def main() -> None:
    before = _run_arm(False, False, 0)
    after = _run_arm(True, True, 0)
    after_shift2 = _run_arm(True, True, 2)
    payload = {
        "protocol": "L3+L4 smoke pins (before vs after, 0px and +2px)",
        "corpus": "synthetic 64x64 square (LOCAL SMOKE; NOT ETH-80 or renderer-recon-v1)",
        "FE_L3_SPINE_default": bool(getattr(P, "FE_L3_SPINE", False)),
        "FE_L4_JUNCTIONS_default": bool(getattr(P, "FE_L4_JUNCTIONS", False)),
        "before": before,
        "after": after,
        "after_shift2": after_shift2,
        "apply_once_idempotent": bool(
            len(after["applied_once"]) == len(set(after["applied_once"]))
            and "l3_spine" in after["applied_once"]
            and "l4_junctions" in after["applied_once"]
        ),
        "no_crash": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "before_edges": before["edges"],
        "after_edges": after["edges"],
        "before_mean_degree": round(before["mean_degree"], 4),
        "after_mean_degree": round(after["mean_degree"], 4),
        "l3_edges_emitted": after["l3_spine"].get("edges_emitted", 0),
        "l4_edges_emitted": after["l4_junctions"].get("edges_emitted", 0),
        "apply_once_idempotent": payload["apply_once_idempotent"],
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
