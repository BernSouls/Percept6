"""M26 item 1: the census of the place population (read-only, from the cached records).

Answers, per cell and per lens class:

  * which lens classes the record holds and their share of the place rows;
  * whether each class has a CONSTITUTIONAL_RANGES entry (a graded kernel exists) and a
    POPULATION_AXES entry (a population could be compared, if the record carried one);
  * the record's DROPPED counts by reason (no_sig / no_config / no_position / no_scale);
  * the share of query places that can vote (a live configuration of >= 2 slots);
  * and the audit list: every call site in the tree that compares places by a quantised key.

    python Topology/m26_census.py --out Topology/results/placement/m26_census.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC  # noqa: E402


#: The files whose quantised-key comparisons ARE the object of this audit.  Other hits in the tree
#: are a different ``by_key`` (a pattern-graph node id map, a feature map) and are not place keys.
KEEP_FILES = {"matching/placement_frontdoor.py", "matching/placement_frontend.py",
              "matching/placement_recognise.py", "matching/placement_gallery.py",
              "matching/placement_bundle.py", "core/types.py", "core/schema.py",
              "core/keypoint_classes.py"}


def call_sites() -> List[Dict[str, Any]]:
    """Every place in matching/ + core/ that looks a place up by its quantised key."""
    pats = ("by_key", "quantized ==", "_keyfn", 'key = (str(lens)')
    out: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(ROOT):
        if any(s in root for s in ("/.git", "/.venv", "/.cache", "/archive", "/__pycache__",
                                   "/tests", "/Topology")):
            continue
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            except OSError:
                continue
            if os.path.relpath(p, ROOT) not in KEEP_FILES:
                continue
            for i, line in enumerate(lines, 1):
                if any(pat in line for pat in pats):
                    out.append({"file": os.path.relpath(p, ROOT), "line": i,
                                "text": line.strip()[:120]})
    return out


def live(row: Dict[str, Any]) -> bool:
    return int(np.count_nonzero(row["Mz"])) >= 2


def census(views: Sequence[str] = REC.VIEWS) -> Dict[str, Any]:
    from core import registries as R
    import matching.placement_frontend as FE
    FE.apply_recipe()                      # the shipped recipe is what registers the constitution
    cells: Dict[str, Any] = {}
    lens_rows: Counter = Counter()
    lens_places: Counter = Counter()
    lens_cells_with_class: Counter = Counter()
    dropped: Counter = Counter()
    per_view: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"places": 0, "live": 0, "cells": 0, "dropped": Counter(), "lens_rows": Counter()})
    n_cells = 0
    for o in REC.objects():
        for v in views:
            if not REC.have(o["key"], v):
                continue
            rec = REC.load(o["key"], v)
            rows = rec["places"]
            n_cells += 1
            c = Counter(r["lens"] for r in rows)
            n_live = sum(1 for r in rows if live(r))
            for lens, n in c.items():
                lens_rows[lens] += n
                lens_places[lens] += 1
                lens_cells_with_class[lens] += 1
            for k, n in (rec.get("dropped") or {}).items():
                dropped[k] += int(n)
                per_view[v]["dropped"][k] += int(n)
            per_view[v]["places"] += len(rows)
            per_view[v]["live"] += n_live
            per_view[v]["cells"] += 1
            for lens, n in c.items():
                per_view[v]["lens_rows"][lens] += n
            cells["%s|%s" % (o["key"], v)] = {
                "split": o["split"], "places": len(rows), "live": n_live,
                "payload": rec.get("n_payload"), "nodes": rec.get("n_all_nodes"),
                "lens": dict(sorted(c.items())), "dropped": dict(rec.get("dropped") or {})}
    total_rows = sum(lens_rows.values()) or 1
    classes: Dict[str, Any] = {}
    for lens, n in sorted(lens_rows.items(), key=lambda kv: -kv[1]):
        ranges = R.get_similarity_ranges(lens)
        axes = R.get_population_axes(lens)
        classes[lens] = {
            "place_rows": n, "share_of_rows": round(n / total_rows, 4),
            "cells_with_class": lens_cells_with_class[lens],
            "ranges": (list(ranges) if ranges else None),
            "graded_kernel_available": bool(ranges),
            "population_axes_available": bool(axes),
            "population_axes": ([a.name for a in axes] if axes else None),
            "slot_weights": (list(R.get_slot_weights(lens, len(ranges))) if ranges else None),
            "cycle": (dict(R.CIRCULAR_CYCLE_LENGTHS.get(lens) or {}) if ranges else None),
        }
    out = {
        "cells": len(cells), "views": list(views),
        "total_place_rows": total_rows,
        "classes": classes,
        "dropped_total": dict(dropped),
        "per_view": {v: {"cells": d["cells"], "places": d["places"], "live": d["live"],
                         "live_share": round(d["live"] / d["places"], 4) if d["places"] else None,
                         "dropped": dict(d["dropped"]),
                         "lens_rows": dict(sorted(d["lens_rows"].items(), key=lambda kv: -kv[1]))}
                     for v, d in sorted(per_view.items())},
        "call_sites": call_sites(),
        "cells_detail": cells,
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default=",".join(REC.VIEWS))
    ap.add_argument("--out")
    a = ap.parse_args(list(argv) if argv is not None else None)
    views = [v.strip().lstrip("-") for v in a.views.split(",") if v.strip()]
    out = census(views)
    print("cells %d, place rows %d" % (out["cells"], out["total_place_rows"]))
    for lens, c in out["classes"].items():
        print("  %-12s rows %6d (%5.1f%%)  kernel %-5s  population %-5s"
              % (lens, c["place_rows"], 100 * c["share_of_rows"],
                 c["graded_kernel_available"], c["population_axes_available"]))
    print("dropped:", out["dropped_total"])
    for v, d in out["per_view"].items():
        print("  view %s: %d cells, %d places, %.1f%% can vote, dropped %s"
              % (v, d["cells"], d["places"], 100 * (d["live_share"] or 0), d["dropped"]))
    print("call sites on a quantised key:")
    for s in out["call_sites"]:
        print("  %s:%d  %s" % (s["file"], s["line"], s["text"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
