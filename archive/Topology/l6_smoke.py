"""L6 (Structural Address) smoke pins and determinism check."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import parameters as P  # noqa: E402
import eth80_frontend as FE  # noqa: E402
from perception.front_end_mechanisms import compute_wl_addresses, KEPT_L6_FAMILIES  # noqa: E402

OUT = ROOT / "results" / "l6_smoke_pins.json"


def main() -> None:
    old = (
        getattr(P, "FE_L3_SPINE", False),
        getattr(P, "FE_L4_JUNCTIONS", False),
        getattr(P, "FE_L6_ADDRESS", False),
    )
    P.FE_L3_SPINE = True
    P.FE_L4_JUNCTIONS = True
    P.FE_L6_ADDRESS = True
    try:
        img = np.full((64, 64, 3), 30, dtype=np.uint8)
        img[14:50, 14:50] = (220, 190, 40)
        img[20:44, 20:44] = (20, 90, 200)
        tr = {}
        side = FE.build_graph(img, name="l6_smoke", record_co_located=False, trace=tr)
        diag = tr["binder"].diagnostics.get("front_end_mechanisms", {})
        l6_diag = diag.get("l6_address", {})

        addrs = compute_wl_addresses(
            side.subgraph.nodes,
            side.subgraph.edges,
            side.sigs,
            KEPT_L6_FAMILIES,
            max_k=3,
        )
        n_nodes = max(1, len(side.subgraph.nodes))
        n_edges = len(side.subgraph.edges)
        mean_degree = (2.0 * n_edges) / n_nodes

        payload = {
            "protocol": "L6 structural address smoke pins",
            "corpus": "synthetic 64x64 square (LOCAL SMOKE)",
            "FE_L6_ADDRESS_default": bool(old[2]),
            "node_count": len(side.subgraph.nodes),
            "node_count_gt_zero": len(side.subgraph.nodes) > 0,
            "selected_graph_edges": n_edges,
            "selected_graph_mean_degree": mean_degree,
            "unique_addresses_selected_graph": {
                "k1": len(set(addrs[1].values())),
                "k2": len(set(addrs[2].values())),
                "k3": len(set(addrs[3].values())),
            },
            "canonical_order": bool(l6_diag.get("canonical_order", False)),
            "no_crash": bool(l6_diag.get("status") == "PRESENT"),
            "apply_once_idempotent": bool(
                len(diag.get("applied_once", [])) == len(set(diag.get("applied_once", [])))
                and "l6_address" in diag.get("applied_once", [])
            ),
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "node_count": payload["node_count"],
            "mean_degree": round(mean_degree, 4),
            "unique_k1_k2_k3": [
                payload["unique_addresses_selected_graph"]["k1"],
                payload["unique_addresses_selected_graph"]["k2"],
                payload["unique_addresses_selected_graph"]["k3"],
            ],
            "apply_once_idempotent": payload["apply_once_idempotent"],
            "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        }, indent=2, sort_keys=True))
    finally:
        P.FE_L3_SPINE, P.FE_L4_JUNCTIONS, P.FE_L6_ADDRESS = old


if __name__ == "__main__":
    main()
