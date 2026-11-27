"""M28 item 1b: is one score the same score everywhere inside one environment?

Mission 27 found the committed pair scores do not reproduce (+0.004 mean, max +0.010 on scores near
0.35) and could not say whether that is the environment or nondeterminism.  This script answers the
second half: the SAME ten pairs, scored serially and in pools of 1, 2 and 4 workers, and with the
BLAS/OpenMP thread pools pinned to one thread, must agree to 1e-12.  If they do, the drift is the
environment's, and ``pair_env`` (which pins the pipeline) is the right key.

    python Topology/m28_determinism.py --run        # all four configurations, as subprocesses
    python Topology/m28_determinism.py --one 2      # a single configuration (used by --run)
    python Topology/m28_determinism.py --records    # do the RECORDS drift between sandboxes?
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement", "m28_determinism.json")
N_PAIRS = 10
CONFIGS = ("serial", "w1", "w2", "w4")


def pairs(n: int = N_PAIRS) -> List[Tuple[str, str]]:
    """Ten committed pairs, resolved inside THIS checkout, in file order (deterministic)."""
    from Topology.m27_profile import committed_pairs
    return [(q, r) for q, r, _b in committed_pairs(limit=n)]


def _score(job: Tuple[str, str]) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    return {"q": job[0], "r": job[1], "B": float(FE.pair_score(job[0], job[1]))}


def one(config: str) -> Dict[str, Any]:
    import matching.placement_frontend as FE
    from matching.placement_records import environment
    FE.apply_recipe()
    jobs = pairs()
    images = sorted({p for j in jobs for p in j})
    t0 = time.time()
    for p in images:
        FE.cell(p, "m28det")
    warm = time.time() - t0
    t0 = time.time()
    if config == "serial":
        out = [_score(j) for j in jobs]
    else:
        w = int(config[1:])
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=w, maxtasksperchild=4) as pool:
            out = list(pool.imap_unordered(_score, jobs, chunksize=1))
    el = time.time() - t0
    out.sort(key=lambda r: (r["q"], r["r"]))
    return {"config": config, "scores": out, "seconds": round(el, 1),
            "seconds_per_pair": round(el / len(jobs), 2), "warm_seconds": round(warm, 1),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "environment": environment()}


def run() -> Dict[str, Any]:
    res: Dict[str, Any] = {"pairs": N_PAIRS, "configs": {}}
    for cfg in CONFIGS:
        env = dict(os.environ)
        if cfg == "serial1":                          # kept for the record, not in CONFIGS
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        cmd = [sys.executable, os.path.abspath(__file__), "--one", cfg]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
        if out.returncode != 0:
            raise RuntimeError("config %s failed: %s" % (cfg, out.stderr[-2000:]))
        res["configs"][cfg] = json.loads(out.stdout.strip().splitlines()[-1])
        print("  %-7s %5.1fs  %.2f s/pair" % (cfg, res["configs"][cfg]["seconds"],
                                              res["configs"][cfg]["seconds_per_pair"]), flush=True)
    # the thread-pinned serial run, with BLAS/OpenMP at one thread
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", "serial"],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError("thread-pinned serial failed: %s" % out.stderr[-2000:])
    res["configs"]["serial_threads1"] = json.loads(out.stdout.strip().splitlines()[-1])
    print("  serial_threads1 %5.1fs" % res["configs"]["serial_threads1"]["seconds"], flush=True)

    base = res["configs"]["serial"]["scores"]
    worst, table = 0.0, {}
    for cfg, blob in res["configs"].items():
        d = max(abs(a["B"] - b["B"]) for a, b in zip(base, blob["scores"]))
        table[cfg] = {"max_abs_diff_vs_serial": d, "identical_1e12": d <= 1e-12}
        worst = max(worst, d)
    res["comparison"] = table
    res["max_abs_diff"] = worst
    res["deterministic_within_environment"] = worst <= 1e-12
    res["manifest"] = {"environment": res["configs"]["serial"]["environment"]}
    print("\nmax |difference| vs serial: %.3e -> %s"
          % (worst, "IDENTICAL to 1e-12" if worst <= 1e-12 else "DIFFERENT"), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    return res


def records() -> Dict[str, Any]:
    """Do the place RECORDS drift across sandboxes?  Rebuild one cell and compare with the npz.

    The committed export was written by Mission 27's sandbox.  If a record rebuilt here has the
    same place count but different signature values, the drift is in the graph builder's numerics
    (a library version), which no thread pinning can fix and which ``pair_env`` now pins.
    """
    import numpy as np
    from Topology import m26_records as REC
    from matching import placement_records as PR
    from matching import placement_frontend as FE
    FE.apply_recipe()
    key = REC.objects()[0]["key"]
    view = "000-000"
    fresh = FE.place_record(REC.path_of(REC.objects()[0], view))
    with np.load(REC.NPZ, allow_pickle=True) as z:
        old = PR._cell_from(z, key, view)
    same_n = len(fresh["places"]) == len(old["places"])
    max_d = None
    keys_equal = None
    if same_n:
        a = np.asarray([p["key"][1] for p in fresh["places"]], dtype=np.float64)
        b = np.asarray([p["key"][1] for p in old["places"]], dtype=np.float64)
        max_d = float(np.max(np.abs(a - b)))
        keys_equal = bool(np.array_equal(a, b))
    res = {"cell": "%s|%s" % (key, view), "places_fresh": len(fresh["places"]),
           "places_committed": len(old["places"]), "same_place_count": same_n,
           "max_abs_signature_diff": max_d, "signatures_identical": keys_equal,
           "manifest": {"environment": __import__("matching.placement_records", fromlist=["x"])
                        .environment()}}
    print(json.dumps(res, indent=1))
    path = os.path.join(ROOT, "Topology", "results", "placement", "m28_records_drift.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", path)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--one", default="")
    ap.add_argument("--records", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.one:
        print(json.dumps(one(a.one)))
    elif a.run:
        run()
    elif a.records:
        records()
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
