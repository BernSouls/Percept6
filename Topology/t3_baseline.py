"""Reproduce the TUNING_REPORT baseline with solver.py (score-time isolated).

Uses the baseline winner cell on the soft-OFF graphs (identical census to the
TUNING_REPORT: 321.3 nodes / 144.3 edges).  score-time isolated (the baseline
instrument) vs joint_mcs's pre-alignment isolated.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Topology"), str(ROOT)]
import numpy as np
import guard; guard.activate()
import corpus, solver
from guard import RESULTS_DIR, write_json
TUNE = RESULTS_DIR / "tune"

_REFS = []
def _init(refs):
    global _REFS
    from frontend import load
    _REFS = [(m["class_"], load(m["cache"])) for m in refs]

def _worker(qm):
    from frontend import load
    q = load(qm["cache"])
    row = {c: -1e9 for c in corpus.CLASSES}
    for cls, r in _REFS:
        sim = solver.node_sim_matrix(r, q, r.sigs, q.sigs)
        sim_t = [list(rr) for rr in zip(*sim)]
        fwd = solver.solve_direction(r, q, r.sigs, q.sigs, sim, beta=0.75,
                                     unmatched_cost=0.15, unmapped_credit=0.5)
        rev = solver.solve_direction(q, r, q.sigs, r.sigs, sim_t, beta=0.75,
                                     unmatched_cost=0.15, unmapped_credit=0.5)
        fa = solver.facts(r, q, r.sigs, q.sigs, sim, fwd)
        fb = solver.facts(q, r, q.sigs, r.sigs, sim_t, rev)
        a = solver.score(fa, w=0.1, gamma=0.5, denom_mode="checkable",
                         isolated="downweight", presence_counts=True,
                         formula="pooled", delta=0.0)[0]
        b = solver.score(fb, w=0.1, gamma=0.5, denom_mode="checkable",
                         isolated="downweight", presence_counts=True,
                         formula="pooled", delta=0.0)[0]
        row[cls] = max(row[cls], max(a, b))
    return qm["name"], qm["class_"], row

def main():
    import multiprocessing as mp
    data = json.load(open(TUNE / "graphs_softoff.json"))
    refs = data["references"]
    held = data["heldout"]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(held)
    held = held[:n]
    ctx = mp.get_context("spawn")
    with ctx.Pool(4, initializer=_init, initargs=(refs,)) as pool:
        rows = list(pool.imap_unordered(_worker, held, chunksize=1))
    rows.sort(key=lambda r: r[0])
    mat = np.array([[r[2][c] for c in corpus.CLASSES] for r in rows])
    truth = np.array([corpus.CLASSES.index(r[1]) for r in rows])
    pred = mat.argmax(axis=1)
    t1 = float((pred == truth).mean())
    tot = cnt = 0
    for i, t in enumerate(truth):
        for d in range(8):
            if d == t: continue
            tot += 1.0 if mat[i, t] > mat[i, d] else (0.5 if mat[i, t] == mat[i, d] else 0.0)
            cnt += 1
    auc = tot / cnt
    print("baseline solver.py (score-time iso) top1 %.4f AUC %.4f n=%d"
          % (t1, auc, len(rows)))
    write_json(TUNE / "t3_baseline.json",
               {"top1": t1, "auc": auc, "n": len(rows),
                "instrument": "solver.py score-time isolated"})

if __name__ == "__main__":
    main()
