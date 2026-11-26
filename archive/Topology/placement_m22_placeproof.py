"""MISSION 22 / (A) -- WHERE the place-voting recall is lost, measured, not guessed.

The voting door scores 1.000 top-1 against itself and at chance across views, for EVERY content
key tried (8 keys, `m22_keysweep.json`): no two votes ever agree on a transform, so no
correspondence is being found.  Two things can be wrong, and they have different fixes:

  (1) THE KEY       the bucket does not contain the true partner  -> a better key fixes it
  (2) THE PLACES    the partners the shipped matcher itself chooses do not agree on a
                    transform      -> no key can fix it; place voting cannot work here

This script settles it.  For a handful of same-object view pairs it runs the SHIPPED matcher,
takes its OWN chosen correspondence, and measures on exactly those pairs

    bucket_hit      share of the matcher's pairs whose partner sits in the same content bucket
    cfg_rank        where the partner ranks inside that bucket under the frozen ruler
    transform       the pair's (dx, dy) in query spacings, log2 scale ratio and axial dtheta --
                    a working voting door needs these to agree across pairs; the spread IS the
                    answer, and the same numbers for a SHUFFLED pairing are the null.

Read-only: builds graphs, reads the shipped matcher, writes one JSON.
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path[:0] = ["/home/user/Percept6", "/home/user/Percept6/Topology"]
OUT = "/home/user/Percept6/Topology/results/placement"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--tag", default="m22_placeproof")
    a = ap.parse_args(argv)
    import guard
    guard.activate()
    import parameters as P
    P.PLACEMENT_M19 = True
    from matching import placement_prefilter as PF
    from matching.joint_mcs import joint_mcs
    from matching import placement_frontdoor as FD
    from Topology import placement_m22_eth80 as H
    from Topology.placement_m20_eth80 import _cell, _recipe
    store = H.objects("FIT")
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for o in store[:int(a.pairs)]:
        _recipe()
        q_img, q, _sq = _cell(o["query"], "proof")
        r_img, r, _sr = _cell(o["ref"], "proof")
        P.PLACEMENT_MATCHER_ENABLED = True
        P.PLACEMENT_SCORE_EVIDENCE = False
        P.PLACEMENT_SCORE_WEIGHT = 0.0
        P.PLACEMENT_MATCHER_STRICT = True
        c = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                      placement_images=(q_img, r_img))
        P.PLACEMENT_MATCHER_STRICT = False
        qe = H.place_job(o["query"])
        re_ = H.place_job(o["ref"])
        qrow = {int(x["n"]): x for x in qe["places"]}
        rrow = {int(x["n"]): x for x in re_["places"]}
        by_key: Dict[Any, List[Dict[str, Any]]] = {}
        for x in re_["places"]:
            by_key.setdefault(x["key"], []).append(x)
        # THE LENGTH UNIT.  `Side.spacing_median` is 0.0 on these images (co-located places), and
        # dividing by it turned every dx into 1e10 -- the first run of this script reported
        # transform_iqr.dx ~ 1e10 for that reason alone, which measured the divisor, not the
        # data.  The unit is the door's own: the median nearest-distinct-neighbour distance.
        _q_scales, sp_q = FD.local_scales(qe)
        assert sp_q > 0.0
        hits = 0
        ranks: List[int] = []
        tr: List[List[float]] = []
        for u, v in sorted(c.mapping.items()):
            qx = qrow.get(int(u))
            rx = rrow.get(int(v))
            if qx is None or rx is None:
                continue
            pool = by_key.get(qx["key"], [])
            hits += 1 if rx in pool else 0
            if rx in pool:
                ranks.append(sorted(pool, key=lambda t: (FD.cfg_distance(qx, t), t["n"])).index(rx) + 1)
            dx = (rx["x"] - qx["x"]) / max(1e-9, sp_q)
            dy = (rx["y"] - qx["y"]) / max(1e-9, sp_q)
            ls = math.log2(max(1e-9, rx["s"]) / max(1e-9, qx["s"]))
            th = (None if (qx["theta"] is None or rx["theta"] is None)
                  else float(((rx["theta"] - qx["theta"]) % math.pi)))
            tr.append([dx, dy, ls, -1.0 if th is None else th])
        T = np.asarray(tr) if tr else np.zeros((0, 4))
        # null: the same query places against a random partner from the same object
        rnd = np.random.default_rng(7)
        ids = sorted(rrow)
        Tn = []
        for u in sorted(qrow):
            qx = qrow[u]
            rx = rrow[ids[int(rnd.integers(len(ids)))]]
            Tn.append([(rx["x"] - qx["x"]) / max(1e-9, sp_q),
                       (rx["y"] - qx["y"]) / max(1e-9, sp_q),
                       math.log2(max(1e-9, rx["s"]) / max(1e-9, qx["s"]))])
        Tn = np.asarray(Tn)
        # agreement: share of pairs inside +-1 bin of the modal transform (bins from BINS)
        def agree(M: np.ndarray, bins=(1.0, 1.0, 0.5, math.pi / 6.0)) -> float:
            if M.shape[0] < 2:
                return float("nan")
            Q = M.copy()
            for j, b in enumerate(bins[:M.shape[1]]):
                Q[:, j] = np.round(Q[:, j] / b)
            cells: Dict[tuple, int] = {}
            for row in Q:
                k = tuple(int(x) for x in row)
                cells[k] = cells.get(k, 0) + 1
            return max(cells.values()) / M.shape[0]
        row = {"object": o["key"], "mapping": len(c.mapping), "pairs_used": len(tr),
               "bucket_hit": hits / max(1, len(tr)),
               "cfg_rank_median": float(np.median(ranks)) if ranks else None,
               "transform_iqr": {n: float(np.subtract(*np.percentile(T[:, j], [75, 25])))
                                 for j, n in enumerate(("dx", "dy", "log2s", "dtheta"))}
               if T.shape[0] else {},
               "length_unit_px": float(sp_q),
               "theta_known_share": float(np.mean(T[:, 3] >= 0)) if T.shape[0] else None,
               "agree_best_cell": agree(T), "agree_null": agree(Tn),
               "agree3": agree(T[:, :3]) if T.shape[0] else None,
               "agree3_null": agree(Tn),
               "theta_iqr": float(np.subtract(*np.percentile(T[T[:, 3] >= 0, 3], [75, 25])))
               if (T.shape[0] and np.any(T[:, 3] >= 0)) else None}
        rows.append(row)
        print("%-28s mapped %-4d bucket_hit %.3f cfg_rank %-5s agree %.3f (null %.3f) "
              "IQR dx %.1f dy %.1f ls %.2f th %s"
              % (o["key"], row["mapping"], row["bucket_hit"], row["cfg_rank_median"],
                 row["agree_best_cell"], row["agree_null"],
                 row["transform_iqr"].get("dx", float("nan")),
                 row["transform_iqr"].get("dy", float("nan")),
                 row["transform_iqr"].get("log2s", float("nan")),
                 ("%.2f" % row["theta_iqr"]) if row["theta_iqr"] is not None else "-"),
              flush=True)
    out = {"pairs": rows, "sec": time.time() - t0}
    print("mean bucket_hit %.3f, mean agree %.3f (null %.3f)"
          % (np.mean([r["bucket_hit"] for r in rows]),
             np.mean([r["agree_best_cell"] for r in rows]),
             np.mean([r["agree_null"] for r in rows])))
    with open("/home/user/Percept6/Topology/results/placement/%s.json" % a.tag, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
