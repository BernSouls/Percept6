"""M34 audit: which machinery exists, and which machinery the shipped path actually reads.

The user's charge, in their words: "tons of the machinery is somehow unused and unwired and you
don't bother to wire them -- you add a necessary mechanism to a single lens/pattern and then
completely forget the rest".  This file tests that charge with a number per claim instead of an
opinion.

    python Topology/m34_audit.py --run

Four censuses, all static (AST) plus one dynamic (the graph vocabulary of a real image):

  MODULES    the transitive import closure of the shipped entry point (`matching/placement_recognise`
             plus the bundle's CODE_MODULES).  Every `.py` under `matching/`, `core/`, `lenses/`,
             `perception/` is then SHIPPED (reachable), TEST-ONLY (a kept test imports it) or ORPHAN.
  CLASSES    every class name in `core/registries.CONSTITUTIONAL_RANGES`: is the NAME a string
             literal anywhere in a shipped module?  A class the registries know and no shipped code
             ever asks for is machinery that cannot fire.
  EDGES      every edge class in `core/schema.py`, the same test, plus what the front end emits on a
             real image and how much of it the shipped path reads.
  FLAGS      every `PLACEMENT_*` / `KP_*` switch: declared, and read by whom.

Output: Topology/results/placement/m34_audit.json (the tables) and a printed summary.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

OUT = os.path.join(ROOT, "Topology", "results", "placement", "m34_audit.json")

PACKAGES = ("matching", "core", "lenses", "perception", "orchestration")
#: Root-level scripts are the real entry points (``eth80_frontend`` builds the graph, the pipeline
#: lives in ``orchestration``).  Leaving them out of the scan made the first run call two thirds of
#: the repo orphan -- the seeds were wrong, not the code.  A module reached only through a string
#: literal (a registry / ``__import__``) is classified SHIPPED-STR, never ORPHAN.
ROOT_SEEDS = ("eth80_frontend", "orchestration.pipeline", "orchestration.binder",
              "derived_lenses", "consolidator", "_bootstrap", "concept_graph")
SKIP_DIRS = {"__pycache__", "archive", ".venv"}


def _files(root: str) -> List[str]:
    out: List[str] = []
    for base, dirs, files in os.walk(os.path.join(ROOT, root)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out.extend(os.path.join(base, f) for f in files if f.endswith(".py"))
    return sorted(out)


def _module_name(path: str) -> str:
    rel = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
    return rel[:-9] if rel.endswith(".__init__") else rel


def _imports(path: str) -> Set[str]:
    """Repo-internal module names this file imports (absolute, package-qualified)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            base = n.module
            out.add(base)
            for a in n.names:
                out.add("%s.%s" % (base, a.name))
    return out


def _strings(path: str) -> Set[str]:
    """Every string literal in a file (what a registry lookup would have to name)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    return out


def _closure(seeds: Sequence[str], imports_of: Dict[str, Set[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(seeds)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        for dep in imports_of.get(m, set()):
            if dep in imports_of:
                stack.append(dep)
    return seen


def modules() -> Dict[str, Any]:
    sys.path[:0] = [ROOT]
    from matching import placement_bundle as PB
    entries = ["matching.placement_recognise"]
    for rel in PB.CODE_MODULES:
        entries.append(rel[:-3].replace("/", "."))
    entries.extend(ROOT_SEEDS)
    imports_of: Dict[str, Set[str]] = {}
    files: Dict[str, str] = {}
    for pkg in PACKAGES:
        for p in _files(pkg):
            name = _module_name(p)
            files[name] = p
            imports_of[name] = {m for m in _imports(p) if m.split(".")[0] in PACKAGES}
    for f in sorted(os.listdir(ROOT)):
        if f.endswith(".py"):
            name = f[:-3]
            files[name] = os.path.join(ROOT, f)
            imports_of[name] = {m for m in _imports(files[name])
                                if m.split(".")[0] in PACKAGES}
    shipped = _closure(entries, imports_of)
    # A registry or ``__import__("x.y")`` reaches a module without an import statement: a module
    # named as a literal somewhere in the shipped closure is SHIPPED-STR, which is still shipped.
    named = set()
    for m in shipped:
        p = files.get(m)
        if p:
            named |= {s for s in _strings(p) if s in files}
    dynamic = sorted(named - shipped, key=lambda k: k)
    tests = set()
    test_files = sorted(f for f in os.listdir(os.path.join(ROOT, "tests")) if f.endswith(".py"))
    for f in test_files:
        p = os.path.join(ROOT, "tests", f)
        for m in _imports(p):
            if m.split(".")[0] in PACKAGES:
                tests.add(m)
    tests_closure = _closure(tests, imports_of)
    # HARNESS: reachable from a Topology/ mission script but from no shipped entry point.  This is
    # the bucket the user's charge is really about -- machinery that exists to be MEASURED and is
    # never wired into the thing that answers a door.
    harness_seeds = set()
    topo = os.path.join(ROOT, "Topology")
    for f in sorted(os.listdir(topo)):
        if f.endswith(".py"):
            for m in _imports(os.path.join(topo, f)):
                if m.split(".")[0] in PACKAGES:
                    harness_seeds.add(m)
    harness_closure = _closure(harness_seeds, imports_of)
    rows: Dict[str, Dict[str, Any]] = {}
    for name, path in files.items():
        if name in shipped:
            status = "SHIPPED"
        elif name in dynamic:
            status = "SHIPPED-STR"
        elif name in harness_closure:
            status = "HARNESS-ONLY"
        elif name in tests_closure:
            status = "TEST-ONLY"
        else:
            status = "ORPHAN"
        rows[name] = {"status": status, "lines": sum(1 for _ in open(path, encoding="utf-8",
                                                                     errors="replace")),
                      "rel": os.path.relpath(path, ROOT)}
    by = collections.Counter(r["status"] for r in rows.values())
    lines = collections.Counter()
    for r in rows.values():
        lines[r["status"]] += r["lines"]
    orphans = sorted([k for k, v in rows.items() if v["status"] == "ORPHAN"],
                     key=lambda k: -rows[k]["lines"])[:25]
    harness_only = sorted([k for k, v in rows.items() if v["status"] == "HARNESS-ONLY"],
                          key=lambda k: -rows[k]["lines"])[:25]
    test_only = sorted([k for k, v in rows.items() if v["status"] == "TEST-ONLY"],
                       key=lambda k: -rows[k]["lines"])[:15]
    return {"entries": entries, "counts": dict(by), "lines": dict(lines),
            "dynamic": dynamic[:25],
            "modules": len(rows), "total_lines": sum(lines.values()),
            "top_orphans": [(k, rows[k]["lines"]) for k in orphans],
            "top_test_only": [(k, rows[k]["lines"]) for k in test_only],
            "top_harness_only": [(k, rows[k]["lines"]) for k in harness_only],
            "rows": rows}


def classes() -> Dict[str, Any]:
    """Registered classes: does any shipped module name them?"""
    import core.registries as R
    shipped_files = [os.path.join(ROOT, m.replace(".", os.sep) + ".py")
                     for m in _shipped_modules()]
    shipped_files = [p for p in shipped_files if os.path.exists(p)]
    mentioned: Set[str] = set()
    for p in shipped_files:
        mentioned |= _strings(p)
    rows = {}
    for cls in sorted(R.CONSTITUTIONAL_RANGES):
        rows[cls] = {"read_by_shipped_code": cls in mentioned}
    never = [c for c, v in rows.items() if not v["read_by_shipped_code"]]
    return {"registered_classes": len(rows), "never_named_in_shipped": len(never),
            "examples": never[:20], "rows": rows}


def _shipped_modules() -> Set[str]:
    saved = modules()
    return {k for k, v in saved["rows"].items() if v["status"].startswith("SHIPPED")}


def edges() -> Dict[str, Any]:
    from core.schema import CONSTITUTIONAL_EDGE_RANGES as ER
    shipped_files = [os.path.join(ROOT, m.replace(".", os.sep) + ".py")
                     for m in _shipped_modules()]
    shipped_files = [p for p in shipped_files if os.path.exists(p)]
    mentioned: Set[str] = set()
    for p in shipped_files:
        mentioned |= _strings(p)
    rows = {name: {"read_by_shipped_code": name in mentioned} for name in sorted(ER)}
    never = [c for c, v in rows.items() if not v["read_by_shipped_code"]]
    return {"edge_classes": len(rows), "never_named_in_shipped": len(never),
            "examples": never[:20], "rows": rows}


def emitted() -> Dict[str, Any]:
    """What the front end emits on one image, and how many of those node/edge types ship code reads."""
    import matching.placement_frontend as FE
    from Topology import m26_records as REC
    from matching import placement_records as PR
    o = REC.objects()[0]
    cells = PR.load_cells(os.path.join(ROOT, "data", "placement_records_m32.npz"), ["035-045"])
    places = {int(p["n"]) for p in cells[(o["key"], "035-045")]["places"]}
    _img, g, _side = FE.cell(REC.path_of(o, "035-045"), "m34audit")
    sub = g.subgraph
    node_cls = collections.Counter((g.sigs.get(n)[0] if g.sigs.get(n) else "none")
                                   for n in sub.nodes)
    edge_cls = collections.Counter(getattr(et, "name", str(et)) for (_u, _v, et) in sub.edges)
    shipped_files = [os.path.join(ROOT, m.replace(".", os.sep) + ".py")
                     for m in _shipped_modules()]
    mentioned: Set[str] = set()
    for p in shipped_files:
        if os.path.exists(p):
            mentioned |= _strings(p)
    place_edges = sum(1 for (u, v, _e) in sub.edges if int(u) in places or int(v) in places)
    return {"nodes": len(sub.nodes), "edges": len(sub.edges),
            "node_classes_emitted": len(node_cls), "edge_types_emitted": len(edge_cls),
            "node_classes_read_by_shipped": sum(1 for c in node_cls if c in mentioned),
            "edge_types_read_by_shipped": sum(1 for c in edge_cls if c in mentioned),
            "place_nodes": len(places),
            "edges_touching_a_place_node": place_edges,
            "node_class_counts": dict(node_cls.most_common()),
            "edge_class_counts": dict(edge_cls.most_common())}


def flags() -> Dict[str, Any]:
    import parameters as P
    names = sorted(k for k in dir(P) if k.startswith(("PLACEMENT_", "KP_")) and
                   not callable(getattr(P, k)))
    from eth80_frontend import _FRONTEND_SWITCHES
    shipped_files = [os.path.join(ROOT, m.replace(".", os.sep) + ".py")
                     for m in _shipped_modules()]
    mentioned: Set[str] = set()
    for p in shipped_files:
        if os.path.exists(p):
            mentioned |= _strings(p)
    rows = {n: {"in_frontend_digest": n in _FRONTEND_SWITCHES,
                "read_by_shipped_code": n in mentioned,
                "value": getattr(P, n)} for n in names}
    never = [n for n, v in rows.items() if not v["read_by_shipped_code"]]
    return {"flags": len(rows), "read_by_shipped": len(rows) - len(never),
            "never_read_by_shipped": len(never), "examples": never[:20], "rows": rows}


def all_flags() -> Dict[str, Any]:
    """Every flag in ``parameters/`` (612 of them), not just the two prefixes.

    A flag is READ when its own name is a string literal in a reachable module (``getattr(P, "X")``,
    a RECIPE tuple entry) or occurs as an attribute (``P.X``).  A flag whose name no reachable module
    contains is a switch wired to nothing.  Grouped by prefix so the census names subsystems.
    """
    import re
    import parameters as P
    names: Set[str] = set()
    for f in sorted(os.listdir(os.path.join(ROOT, "parameters"))):
        if f.endswith(".py"):
            src = open(os.path.join(ROOT, "parameters", f), encoding="utf-8", errors="replace").read()
            names |= set(re.findall(r"^([A-Z][A-Z0-9_]{2,})\s*=", src, re.M))
    shipped_files = [os.path.join(ROOT, m.replace(".", os.sep) + ".py")
                     for m in _shipped_modules()]
    mentioned: Set[str] = set()
    text = []
    for pth in shipped_files:
        if os.path.exists(pth):
            mentioned |= _strings(pth)
            text.append(open(pth, encoding="utf-8", errors="replace").read())
    blob = "\n".join(text)
    rows = {}
    for n in sorted(names):
        if not hasattr(P, n):
            continue
        rows[n] = {"read_by_shipped": (n in mentioned) or (("P." + n) in blob),
                   "value": getattr(P, n)}
    never = [n for n, v in rows.items() if not v["read_by_shipped"]]
    # Split the unread: named somewhere in the tree (so owned by unreachable machinery) or named
    # nowhere at all (a constant nothing has ever consulted).
    # Plain text, not string literals: a flag named in a comment or as `P.NAME` still counts.
    other_text: List[str] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in
                   {"Topology", "tests", "data", ".git", "archive", "docs", "parameters"}]
        for f in files:
            if f.endswith(".py"):
                other_text.append(open(os.path.join(base, f), encoding="utf-8",
                                       errors="replace").read())
    blob_other = "\n".join(other_text)
    owned_by_unreachable = [n for n in never if n in blob_other]
    named_nowhere = [n for n in never if n not in blob_other]
    groups = collections.Counter(n.split("_")[0] for n in never)
    return {"flags": len(rows), "read_by_shipped": len(rows) - len(never),
            "never_read_by_shipped": len(never),
            "never_owned_by_unreachable": len(owned_by_unreachable),
            "never_named_nowhere": len(named_nowhere),
            "named_nowhere_examples": named_nowhere[:30],
            "never_by_prefix": dict(groups.most_common(20)),
            "never_examples": never[:40],
            "never_rows": never}


#: The modules that turn a graph into a NUMBER (a score, a shortlist, a door decision).  A type named
#: by the module that WRITES an edge, or by `category_bridge`'s type->category table, is not wired
#: into a decision -- it is only spelled.
DECISION_MODULES = ("matching.joint_mcs", "matching.placement_frontend", "matching.placement_matcher",
                    "matching.placement_recognise", "matching.placement_fuse",
                    "matching.placement_shape", "matching.placement_records", "matching.alignment",
                    "matching.feature_attachment", "matching.perceptual_selection", "matching.placement",
                    "matching.slots", "matching.structural_similarity", "matching.placement_content")


def edge_wiring() -> Dict[str, Any]:
    """For every edge type the front end EMITS: who spells it, and does a decision module see it?"""
    counts = emitted()["edge_class_counts"]
    text: Dict[str, str] = {}
    for m in sorted(_shipped_modules()):
        if m in ("core.types", "core.schema"):
            continue                                   # the definition is not a reader
        p = os.path.join(ROOT, m.replace(".", os.sep) + ".py")
        if os.path.exists(p):
            text[m] = open(p, encoding="utf-8", errors="replace").read()
    rows: Dict[str, Any] = {}
    for et, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        hits = [m for m, t in text.items() if et in t]
        rows[et] = {"emitted": int(cnt), "spelled_by": hits,
                    "decision": [m for m in hits if m in DECISION_MODULES]}
    wired = [et for et, r in rows.items() if r["decision"]]
    return {"edge_types_emitted": len(rows), "seen_by_a_decision_module": len(wired),
            "only_writer_and_table": [et for et, r in rows.items() if not r["decision"]],
            "rows": rows}


def run() -> Dict[str, Any]:
    out = {"modules": modules()}
    out["classes"] = classes()
    out["edges"] = edges()
    out["emitted"] = emitted()
    out["flags"] = flags()
    out["all_flags"] = all_flags()
    out["edge_wiring"] = edge_wiring()
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(json.dumps({
        "modules": {k: out["modules"][k] for k in ("counts", "lines", "modules",
                                                   "total_lines", "top_orphans", "top_test_only")},
        "classes": {k: out["classes"][k] for k in ("registered_classes", "never_named_in_shipped")},
        "edge_classes_never_read": out["edges"]["never_named_in_shipped"],
        "emitted": {k: out["emitted"][k] for k in ("nodes", "edges", "node_classes_emitted",
                                                   "edge_types_emitted",
                                                   "node_classes_read_by_shipped",
                                                   "edge_types_read_by_shipped",
                                                   "place_nodes", "edges_touching_a_place_node")},
        "flags": {k: out["flags"][k] for k in ("flags", "read_by_shipped",
                                               "never_read_by_shipped", "examples")},
    }, indent=1))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.run:
        run()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
