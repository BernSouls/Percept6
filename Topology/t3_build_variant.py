"""Task 3: build a front-end graph variant (switch overrides applied).

Sets PERCEPT_SET, applies the overrides to ``parameters``, then builds every
graph in the tuning split ONCE and caches it under the switch-specific key
(``frontend._key`` folds ``switch_overrides_key`` in).  Writes a variant
``graphs.json`` index.

Usage:
    python t3_build_variant.py --variant softoff
    python t3_build_variant.py --variant softon_vecon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

VARIANTS = {
    # M0: everything OFF (soft gates off, vector address off).
    "softoff": "FE_SOFT_CHANNEL_GATE=false;FE_SOFT_L3_L4_GATE=false",
    # M3: soft gates ON (default) + vector address ON.
    "softon_vecon": "FE_VECTOR_ADDRESS=true",
}


def _worker_init(spec):
    import guard
    guard.activate()
    import _bootstrap as b
    b.apply_switch_overrides(spec)


def _build_one(entry):
    from frontend import build_cached
    return build_cached(entry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    os.environ["PERCEPT_SET"] = spec
    import _bootstrap
    applied = _bootstrap.apply_switch_overrides(spec)
    print("applied:", applied, flush=True)

    import guard
    guard.activate()
    import corpus

    split = corpus.split_tuning()
    entries = split["references"] + split["dev"] + split["heldout"]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    metas = []
    t0 = time.perf_counter()
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(spec,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_build_one, entries,
                                                    chunksize=1)):
            if isinstance(rec, dict) and "error" in rec:
                raise RuntimeError(rec["error"])
            metas.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(entries):
                print("  front end %3d/%d (%.0fs)"
                      % (i + 1, len(entries), time.perf_counter() - t0),
                      flush=True)
    metas.sort(key=lambda m: m["name"])
    by_name = {m["name"]: m for m in metas}
    out = {
        "variant": args.variant,
        "switch_spec": spec,
        "split": split,
        "graphs": metas,
        "references": [by_name[corpus.name_of(e)] for e in split["references"]],
        "dev": [by_name[corpus.name_of(e)] for e in split["dev"]],
        "heldout": [by_name[corpus.name_of(e)] for e in split["heldout"]],
        "counts": split["counts"],
        "build_seconds": time.perf_counter() - t0,
    }
    path = os.path.join(ROOT, "Topology", "results", "tune",
                        "graphs_%s.json" % args.variant)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
