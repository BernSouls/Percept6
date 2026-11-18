"""One-pair timing probe for solver.py (no labels, no accuracy)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import guard
guard.activate()

import solver
from frontend import load

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results" / "arms-seed0.json").read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in data["graph_records"]}
    a = load(by_name["apple1-090-000"]["cache"])
    b = load(by_name["apple5-090-000"]["cache"])
    t0 = time.perf_counter()
    sim = solver.node_sim_matrix(a, b, a.sigs, b.sigs)
    t_sim = time.perf_counter() - t0
    print("sim matrix", t_sim, flush=True)
    rows = []
    for beta in (0.0, 0.15, 0.3, 0.5, 0.75):
        for uc in (0.0, 0.15, 0.3):
            for occ in ((0.0,) if beta == 0.0 else (0.0, 0.25, 0.5)):
                t1 = time.perf_counter()
                fwd = solver.solve_direction(a, b, a.sigs, b.sigs, sim,
                                            beta=beta, unmatched_cost=uc,
                                            unmapped_credit=occ)
                rev = solver.solve_direction(b, a, b.sigs, a.sigs,
                                            [[sim[i][j] for i in range(len(sim))]
                                             for j in range(len(sim[0]))],
                                            beta=beta, unmatched_cost=uc,
                                            unmapped_credit=occ)
                fa = solver.facts(a, b, a.sigs, b.sigs, sim, fwd)
                fb = solver.facts(b, a, b.sigs, a.sigs,
                                  [[sim[i][j] for i in range(len(sim))]
                                   for j in range(len(sim[0]))], rev)
                t2 = time.perf_counter() - t1
                rows.append((beta, uc, occ, t2, len(fwd.mapping), len(rev.mapping)))
                print("beta %.2f uc %.2f occ %.2f  %.3fs  maps %d/%d"
                      % (beta, uc, occ, t2, len(fwd.mapping), len(rev.mapping)),
                      flush=True)
    total = sum(r[3] for r in rows)
    print("configs", len(rows), "total %.1fs" % total,
          "per full grid pair %.1fs" % total)
    print("projected 1152 pairs / 12 workers: %.1f min"
          % (total * 1152 / 12 / 60))


if __name__ == "__main__":
    sys.exit(main())
