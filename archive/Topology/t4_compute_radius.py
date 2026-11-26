"""Task 4 PART 0c — Compute and FREEZE the per-kind primary radius on dev."""
from __future__ import annotations
import os, pickle, sys
from collections import defaultdict
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402
from t4_shift import primary_radius  # noqa: E402
PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"


def _pos_lens(g):
    out = defaultdict(list)
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[lens].append((float(a[0]), float(a[1])))
    return out


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def main() -> int:
    print("=== PART 0c: PER-KIND PRIMARY RADIUS (dev, frozen) ===", flush=True)
    all_by_lens = defaultdict(list)
    for e in _dev_entries():
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        with open(PKL_DIR / ("B1_%s.pkl" % name), "rb") as fh:
            g = pickle.load(fh)
        for lens, pts in _pos_lens(g).items():
            all_by_lens[lens].extend(pts)
    radius = {}
    for lens, pts in sorted(all_by_lens.items()):
        r = primary_radius({lens: pts})
        radius[lens] = r[lens]
        print("  %-20s n=%-6d radius=%s" % (lens, len(pts), r[lens]))
    write_json(RESULTS_DIR / "t4" / "primary_radius_dev.json", radius)
    print("  frozen to primary_radius_dev.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
