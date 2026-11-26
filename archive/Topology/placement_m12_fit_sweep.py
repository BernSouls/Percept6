"""Mission 12 step 4 -- the w sweep that explains why the refit's gates refused.

Reads the fit's own checkpoint (`m12_fit_w_progress.jsonl`) and prints, for w over
the whole credible range at the shipped temperature T = 1:

* the FIT log-likelihood (what the fit maximises),
* FIT top-1,
* HELD-OUT top-1 and MRR (never optimised against).

The two criteria disagree in sign: the log-likelihood rises with w, monotonic and
without a plateau inside the bounds, while the held-out ranking is flat up to the
deployed 0.375 and falls after it.  That is the mechanism behind the failed gates,
and it is a property of the objective, not of the search.

    python Topology/placement_m12_fit_sweep.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]

WS = (0.0, 0.1, 0.2, 0.3, 0.375, 0.45, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 1.6, 2.0)


def main() -> int:
    import guard
    guard.activate()
    from Topology import placement_m12_fit_w as F
    qs, shorts = F.queries(), F.shortlists()
    truth_of = {q: shorts[q][0] for q in shorts}
    held = {q for i, (q, *_r) in enumerate(qs) if i % 4 == 3}
    rows = []
    with open(F.PROGRESS) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            d["_truth"] = truth_of.get(d["q"])
            d["_held"] = d["q"] in held
            rows.append(d)
    train = [r for r in rows if not r["_held"]]
    test = [r for r in rows if r["_held"]]
    print(" w      FIT_ll   FIT_top1   HELD_top1   HELD_mrr")
    out = []
    for w in WS:
        t, h = F.top1(train, 1.0, w), F.top1(test, 1.0, w)
        ll = float(F.loglik(train, 1.0, w))
        out.append({"w": w, "fit_loglik": ll, "fit_top1": t["top1"],
                    "held_top1": h["top1"], "held_mrr": h["mrr"]})
        print("%.3f  %9.2f   %.4f     %.4f      %.4f"
              % (w, ll, t["top1"], h["top1"], h["mrr"]))
    path = os.path.join(ROOT, "Topology", "results", "placement", "m12_fit_w_sweep.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
