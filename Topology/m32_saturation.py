"""M32 item 2: measure the FLATTENING in both channels, before any fix.

    python Topology/m32_saturation.py

Same 80 x 80 cell for both channels (query -035-045 against gallery -000-000), records only:

  COLOUR -- `matching.placement_content`'s unit six-vector of CIELAB ring means, compared with the
            shipped cosine, aggregated the shipped way (max over the query's places of the best
            same-lens match in the object).
  SHAPE  -- the five graded columns, compared with the shipped graded similarity, aggregated the
            same way.

For each channel: the distribution of the 6,400 pair scores, the share >= 0.99 and >= 0.999, the
truth median against the impostor median, and the number of distinct values each COLUMN actually
takes.  This is the baseline every later number in the mission is compared against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                       # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m30.npz")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m32_saturation.json")
REF, VIEW = "000-000", "035-045"


def _cells(views: Sequence[str]) -> Dict[Tuple[str, str], Any]:
    from matching import placement_records as PR
    return PR.load_cells(RECORDS, list(views))


def _channel_matrix(kind: str, chunk: int = 8) -> np.ndarray:
    """(80, 80) pair scores for one channel, aggregated over places by max (the shipped way)."""
    from matching import placement_content as PC
    from matching import placement_shape as PS
    keys = [o["key"] for o in REC.objects()]
    C = _cells([REF, VIEW])
    n = len(keys)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n, chunk):
        for j in range(0, n, chunk):
            for a in range(i, min(n, i + chunk)):
                for b in range(j, min(n, j + chunk)):
                    qa = [p for p in C[(keys[a], VIEW)]["places"]]
                    gb = [p for p in C[(keys[b], REF)]["places"]]
                    best = -np.inf
                    for lens in ("Blob", "BoundaryArc", "CornerPoint"):
                        Q = _rows(qa, lens, kind)
                        G = _rows(gb, lens, kind)
                        if Q is None or G is None:
                            continue
                        S = _sim(Q, G, kind)
                        if S.size:
                            best = max(best, float(S.max()))
                    out[a, b] = best if np.isfinite(best) else 0.0
    return out


def _rows(places: Sequence[Dict[str, Any]], lens: str, kind: str) -> Any:
    from matching import placement_shape as PS
    if kind == "colour":
        rows = [p["content"] for p in places
                if p.get("lens") == lens and p.get("content") is not None]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 6) if rows else None
    rows = [([p.get(f, 0.0) or 0.0 for f in PS.FIELDS]) for p in places if p.get("lens") == lens]
    return np.asarray(rows, dtype=np.float64) if rows else None


def _sim(Q: np.ndarray, G: np.ndarray, kind: str) -> np.ndarray:
    from matching import placement_shape as PS
    if kind == "colour":
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        return qn @ gn.T                                    # the shipped cosine
    return PS.graded_matrix(Q, G)                            # the shipped graded form


def _stats(M: np.ndarray, keys: Sequence[str], label: str) -> Dict[str, Any]:
    n = M.shape[0]
    truth = np.asarray([M[i, i] for i in range(n)])
    off = M[~np.eye(n, dtype=bool)]
    out = {"label": label, "pairs": int(off.size),
           "share_ge_0.99": float((off >= 0.99).mean()),
           "share_ge_0.999": float((off >= 0.999).mean()),
           "share_ge_0.9999": float((off >= 0.9999).mean()),
           "truth_median": float(np.median(truth)), "impostor_median": float(np.median(off)),
           "truth_mean": float(truth.mean()), "impostor_mean": float(off.mean()),
           "distinct_scores": int(np.unique(np.round(off, 9)).size),
           "min": float(off.min()), "max": float(off.max()),
           "gap_medians": float(np.median(truth) - np.median(off))}
    print("  %-6s share>=0.99 %.3f   truth median %.6f   impostor median %.6f   gap %+.6f   "
          "distinct %d" % (label, out["share_ge_0.99"], out["truth_median"],
                           out["impostor_median"], out["gap_medians"], out["distinct_scores"]),
          flush=True)
    return out


def columns() -> Dict[str, Any]:
    """Distinct values per COLUMN: the six colour dims and the five shape fields, over all places."""
    from matching import placement_shape as PS
    keys = [o["key"] for o in REC.objects()]
    C = _cells([REF, VIEW])
    out: Dict[str, Any] = {"views": {}}
    for view in (REF, VIEW):
        per: Dict[str, Any] = {}
        lab = np.asarray([p["content"] for k in keys for p in C[(k, view)]["places"]
                          if p.get("content") is not None], dtype=np.float64)
        per["colour"] = {"places": int(lab.shape[0]),
                         "dims": [int(np.unique(np.round(lab[:, d], 9)).size)
                                  for d in range(lab.shape[1])] if lab.size else [],
                         "dim_min": [float(lab[:, d].min()) for d in range(lab.shape[1])] if lab.size else [],
                         "dim_max": [float(lab[:, d].max()) for d in range(lab.shape[1])] if lab.size else []}
        for f in PS.FIELDS:
            v = np.asarray([p.get(f) for k in keys for p in C[(k, view)]["places"]
                            if p.get(f) is not None], dtype=np.float64)
            per.setdefault("shape", {})[f] = {"places": int(v.size),
                                              "distinct": int(np.unique(np.round(v, 9)).size),
                                              "min": float(v.min()) if v.size else None,
                                              "max": float(v.max()) if v.size else None}
        out["views"][view] = per
        print("  %s columns: colour dims distinct %s ; shape distinct %s"
              % (view, per["colour"]["dims"],
                 {f: per["shape"][f]["distinct"] for f in PS.FIELDS}), flush=True)
    return out


def run() -> Dict[str, Any]:
    keys = [o["key"] for o in REC.objects()]
    out: Dict[str, Any] = {"cell": {"queries": len(keys), "view": VIEW, "gallery": REF},
                           "columns": columns()}
    for kind in ("colour", "shape"):
        M = _channel_matrix(kind)
        out[kind] = _stats(M, keys, kind)
        out[kind]["matrix_shape"] = list(M.shape)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print("wrote", OUT)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.parse_args(list(argv) if argv is not None else None)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
