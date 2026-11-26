"""MISSION 20 / item 2 -- the ETH-80 split, written before any graph exists.

Why this file exists before anything is measured: the M20 tasks are chosen on FIT, read on
DEV, and finally registered on FRESH.  If the split were drawn after the graphs were built
the draw could be steered by what looked easy, so the assignment is a pure function of a
SALT and the object's name -- no image, no graph, no score, nothing measured is read.

THE RULE (fixed in this docstring, not after looking):

  * split unit is the OBJECT (one class-instance), never the view.  FIT / DEV / FRESH are
    disjoint by object, so no instance can contribute a reference to one split and a query
    to another.
  * per class, the ten instances are ordered by ``sha256("<salt>|<class>|<instance>")`` and
    dealt in that order: the first 5 objects -> FIT, the next 3 -> DEV, the last 2 -> FRESH.
  * every class is present in every split (the class set is a parameter, so a confusable
    subset like apple/pear/tomato re-deals inside itself and still covers all three splits).
  * views are not split: an object keeps all 41 views, because T1 needs "same object, other
    view" and dropping views would shrink the very thing under test.

Counts are written per class, per object and per view to
``Topology/results/placement/m20_splits.json``; ``--check`` re-derives them and diffs, which
is how the file's determinism is proven without storing a hash inside itself.

Usage:
    python Topology/placement_m20_splits.py                 # write the file
    python Topology/placement_m20_splits.py --check         # re-derive, diff, exit 1 on drift
    python Topology/placement_m20_splits.py --classes apple,pear,tomato --out .../x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

TREE = ROOT / "data" / "ETH-80-master"
OUT = ROOT / "Topology" / "results" / "placement" / "m20_splits.json"
SALT = "M20-split-v1-2026-11-18"
CLASS_DIR = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
             "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}
#: per class, dealt in hash order: FIT first, then DEV, then FRESH
DEAL = (("FIT", 5), ("DEV", 3), ("FRESH", 2))
NAME = re.compile(r"^([a-z]+)(\d+)-(\d{3})-(\d{3})\.png$")


def order(cls: str, instances: List[str]) -> List[str]:
    return sorted(instances, key=lambda i: hashlib.sha256(
        ("%s|%s|%s" % (SALT, cls, i)).encode()).hexdigest())


def build(classes: List[str], tree: Path = TREE, salt: str = SALT) -> Dict[str, Any]:
    out: Dict[str, Any] = {"salt": salt, "rule": "sha256(salt|class|instance), 5/3/2 per class",
                           "unit": "object", "tree": str(tree), "classes": list(classes),
                           "deal": dict(DEAL), "splits": {}, "counts": {}}
    per_split: Dict[str, Dict[str, Any]] = {s: {"objects": [], "views": 0} for s, _ in DEAL}
    counts: Dict[str, Any] = {}
    views_of: Dict[str, List[str]] = {}
    for cls in classes:
        d = tree / CLASS_DIR[cls]
        if not d.is_dir():
            raise SystemExit("missing class directory %s (run Topology/fetch_corpus.py)" % d)
        inst = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "maps")
        if len(inst) != 10:
            raise SystemExit("%s: %d instances, expected 10" % (cls, len(inst)))
        ordered = order(cls, inst)
        counts[cls] = {"objects": {s: [] for s, _ in DEAL}, "views": {}}
        pos = 0
        for split, n in DEAL:
            for i in ordered[pos:pos + n]:
                od = d / i
                views = sorted(f.name for f in od.iterdir()
                               if f.is_file() and NAME.match(f.name))
                if len(views) != 41:
                    raise SystemExit("%s/%s: %d views, expected 41" % (cls, i, len(views)))
                views_of["%s/%s" % (cls, i)] = views
                # the object's identity is the (class, directory) pair; the FILENAME
                # instance number is recorded beside it because the two disagree in this
                # corpus (directory 4 of `car` holds the images of car11), so a later
                # reader must be able to see which object was actually held out
                lab = sorted({NAME.match(v).group(2) for v in views})
                per_split[split]["objects"].append({"class": cls, "instance": i,
                                                    "label": "%s%s" % (cls, lab[0]),
                                                    "views": len(views)})
                per_split[split]["views"] += len(views)
                counts[cls]["objects"][split].append(i)
                counts[cls]["views"][split] = counts[cls]["views"].get(split, 0) + len(views)
            pos += n
        if pos != len(inst):
            raise SystemExit("%s: dealt %d of %d objects" % (cls, pos, len(inst)))
    for split, _n in DEAL:
        objs = per_split[split]["objects"]
        out["splits"][split] = {"n_objects": len(objs), "n_views": per_split[split]["views"],
                                "objects": objs}
    out["counts"] = {
        "per_class": counts,
        "totals": {s: {"objects": out["splits"][s]["n_objects"],
                       "views": out["splits"][s]["n_views"]} for s, _ in DEAL},
        "classes": {s: sorted({o["class"] for o in out["splits"][s]["objects"]})
                    for s, _ in DEAL},
        "views_per_object": sorted({len(v) for v in views_of.values()}),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default=",".join(CLASS_DIR))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    bad = [c for c in classes if c not in CLASS_DIR]
    if bad:
        raise SystemExit("unknown classes %r" % (bad,))
    fresh = build(classes)
    if a.check:
        have = json.load(open(a.out))
        if have != fresh:
            diff = [k for k in set(have) | set(fresh) if have.get(k) != fresh.get(k)]
            print("SPLIT DRIFT in %r" % (diff,))
            return 1
        print("split identical: %s" % a.out)
        return 0
    with open(a.out, "w") as fh:
        json.dump(fresh, fh, indent=1, sort_keys=True)
    t = fresh["counts"]["totals"]
    print("%-8s %-6s %-6s %-6s" % ("split", "obj", "views", "classes"))
    for s in ("FIT", "DEV", "FRESH"):
        print("%-8s %-6d %-6d %d" % (s, t[s]["objects"], t[s]["views"],
                                    len(fresh["counts"]["classes"][s])))
    for cls in classes:
        print("  %-7s FIT %-22s DEV %-12s FRESH %s"
              % (cls, ",".join(fresh["counts"]["per_class"][cls]["objects"]["FIT"]),
                 ",".join(fresh["counts"]["per_class"][cls]["objects"]["DEV"]),
                 ",".join(fresh["counts"]["per_class"][cls]["objects"]["FRESH"])))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
