"""Task 4 RETRIEVAL — build ETH-80 graphs for the retrieval split (arm B1).

Split (pre-registered):
  - 8 primary objects: rank-0 instance per class (apple1..tomato1).
  - reference view: -090-000  -> 8 primary references.
  - 16 cross-view queries per object (views listed below, excluding -090-000).
  - full 80-ref pool: all 10 instances x 8 classes at -090-000.

Builds graphs with ARM_B1 (FE_L3_SPINE=true;FE_L4_JUNCTIONS=true) so edges
are present, cached to disk via build_cached.  2 workers, foreground.
"""
from __future__ import annotations
import argparse, json, os, sys
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()

ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
REF_VIEW = "-090-000"
QUERY_VIEWS = ("-000-000", "-022-000", "-022-090", "-022-180", "-022-270",
               "-035-135", "-035-315", "-045-000", "-045-090", "-045-180",
               "-045-270", "-066-027", "-066-117", "-066-207", "-066-297",
               "-090-135")
CLASSES = ["apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato"]


def _worker_init():
    import _bootstrap
    _bootstrap.apply_switch_overrides(ARM_B1)


def _build_one(entry):
    from frontend import build_cached
    meta = build_cached(entry)
    return entry["class"], entry["object"], entry["view"], meta["cache"]


def _entries():
    import corpus
    out = []
    seen = set()
    # primary refs: rank-0 (lowest instance) per class, -090-000
    for cls in CLASSES:
        inst = corpus._instances(cls)
        ids = sorted(inst)
        obj = ids[0]  # rank 0
        e = corpus._entry(cls, obj, REF_VIEW, inst[obj][REF_VIEW])
        out.append(e); seen.add((cls, obj, REF_VIEW))
    # queries: rank-0 per class, 16 views
    for cls in CLASSES:
        inst = corpus._instances(cls)
        ids = sorted(inst)
        obj = ids[0]
        for v in QUERY_VIEWS:
            e = corpus._entry(cls, obj, v, inst[obj][v])
            out.append(e); seen.add((cls, obj, v))
    # full 80-ref pool: all 10 instances x 8 classes at -090-000
    for cls in CLASSES:
        inst = corpus._instances(cls)
        for obj in sorted(inst):
            if (cls, obj, REF_VIEW) in seen:
                continue
            e = corpus._entry(cls, obj, REF_VIEW, inst[obj][REF_VIEW])
            out.append(e); seen.add((cls, obj, REF_VIEW))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    entries = _entries()
    print("building %d graphs (arm B1: %s)..." % (len(entries), ARM_B1), flush=True)
    done = 0
    with Pool(args.workers, initializer=_worker_init) as pool:
        for cls, obj, view, cache in pool.imap_unordered(_build_one, entries):
            done += 1
            if done % 10 == 0 or done == len(entries):
                print("  built %d/%d (%s%d%s)" % (done, len(entries), cls, obj, view),
                      flush=True)
    print("done: %d graphs cached" % done, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
