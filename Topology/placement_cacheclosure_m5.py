"""MISSION 5 / 0d -- prove the graph-cache digest covers the import closure.

`eth80_frontend.frontend_source_digest()` decides whether a cached graph may be
served.  If a module on the BUILD path is edited but is not hashed, the cache
returns a graph that the current code would not produce.  This walks imports
from `build_graph` over repository files and asserts every one of them is
covered by the digest's directory/file list.

Exit code is non-zero when anything on the closure is uncovered, so it can be
used both as a check and inside the test suite.
"""
from __future__ import annotations

import ast
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def closure(root: str = ROOT, entry: str = "eth80_frontend") -> set:
    for p in (root, os.path.join(root, "Topology")):
        if p not in sys.path:
            sys.path.insert(0, p)
    seen, stack = set(), [entry]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        f = getattr(m, "__file__", None)
        if not f or not f.startswith(root) or ".venv" in f:
            continue
        try:
            with open(f) as fh:
                src = fh.read()
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    stack.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    stack.append(node.module)
    out = set()
    for n in seen:
        m = sys.modules.get(n)
        f = getattr(m, "__file__", None)
        if not f or not f.startswith(root) or ".venv" in f or not f.endswith(".py"):
            continue
        out.add(os.path.relpath(f, root).replace(os.sep, "/"))
    return out


def covered(rel_path: str) -> bool:
    import eth80_frontend as F
    dirs = set(F._FRONTEND_SOURCE_DIRS)
    files = set(F._FRONTEND_SOURCE_FILES)
    return (rel_path.split("/")[0] in dirs and "/" in rel_path) or rel_path in files


def main() -> int:
    import guard  # noqa: F401  (activates the repo import path)
    files = closure()
    missing = sorted(r for r in files if not covered(r))
    print("closure: %d repository files on the build path" % len(files))
    if missing:
        print("UNCOVERED BY THE CACHE DIGEST (%d):" % len(missing))
        for r in missing:
            print("   ", r)
        return 1
    print("cache digest covers the whole closure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
