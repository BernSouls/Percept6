"""Strict runtime guard for the structural-topology classifier.

Two separate obligations, both enforced here rather than promised in prose.

TREE.  Every product module a measurement touches must resolve to a
``__file__`` under this tree (``...\\work``).  ``activate()`` rebuilds
``sys.path`` so that only this tree's library, this tree's front-end driver and
the running interpreter's own stdlib / site-packages can supply a module;
``load()`` and ``assert_product_paths()`` assert the resolved path of every
module the measurement names, and every product module that is loaded.

BANNED MACHINERY.  This experiment stores whole graphs and matches them whole.
It must not import or call the discriminative machinery:

    memory.cores.evidence_score / mcs_score / mcs_similarity
    memory.cores.CoreLibrary / CoreEntry / CategoryManager
    memory.cores._extract_category_core
    orchestration.pipeline.Pipeline._extract_category_core
    consolidator._content_discriminative_filter
    matching.structural_similarity.structural_intersection / materialize_template

``assert_banned_absent()`` refuses to run when ``consolidator`` or
``category_manager`` is loaded at all, or when one of those callables is bound
in a module other than the one that defines it.  ``tripwire()`` then replaces
every banned callable with a raise for the duration of the run, so a call
cannot be silent even through a path the scan does not see.

The one name this path does take from ``memory.cores`` is ``_sig_similarity``,
the per-pair node similarity that ``matching.joint_mcs`` calls.  It is not on
the banned list, and the scan below makes the distinction explicit rather than
leaving it implied.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]          # ...\work
SRC = TREE / "src"
FRONTEND_DIR = TREE / "diagnostics"
HARNESS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HARNESS_DIR / "results"
CACHE_DIR = HARNESS_DIR / "cache"

PRODUCT_PACKAGES = frozenset({
    "core", "dynamics", "input", "learning", "lenses", "matching", "memory",
    "orchestration", "parameters", "perception", "action", "phase3",
})
PRODUCT_MODULES = frozenset({
    "concept_graph", "consolidator", "activation_engine", "sensory_floor",
    "derived_lenses", "pipeline", "prediction", "mapper", "builder", "binder",
    "content", "primitives", "slot_learning", "utility_engine",
    "fertility_engine", "executor", "higher_lens_helpers", "innate_reflex",
    "category_manager", "eth80_frontend", "structural_similarity",
})

#: modules that must not be imported by this path at all
BANNED_MODULES = ("consolidator", "category_manager")

#: module-level names that must not be bound anywhere but their definition site
BANNED = {
    "memory.cores": ("evidence_score", "mcs_score", "mcs_similarity",
                     "CoreLibrary", "CoreEntry", "CategoryManager",
                     "_extract_category_core"),
    "matching.structural_similarity": ("structural_intersection",
                                       "materialize_template"),
    "consolidator": ("_content_discriminative_filter",),
}

#: methods of a loaded class that must not be callable during the run
BANNED_METHODS = (("orchestration.pipeline", "Pipeline",
                   "_extract_category_core"),)

_ACTIVE = False


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root)
    except ValueError:                                    # pragma: no cover
        return False


def activate() -> Path:
    """Rebuild ``sys.path`` so the tree is the only source of product modules."""
    global _ACTIVE
    if not (SRC / "core" / "types.py").is_file():
        raise RuntimeError("not the expected tree: %s" % SRC)
    roots = (Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve())
    keep = []
    for entry in sys.path:
        p = Path(entry or os.getcwd()).resolve()
        if p in (SRC, FRONTEND_DIR, HARNESS_DIR, TREE):
            continue
        if "site-packages" in p.parts or "dist-packages" in p.parts:
            keep.append(entry)
            continue
        if any(_is_under(p, r) for r in roots):
            keep.append(entry)
    sys.path[:] = [str(SRC), str(FRONTEND_DIR), str(HARNESS_DIR)] + keep
    _ACTIVE = True
    return SRC


def assert_under_tree(module) -> Path:
    """The provenance assertion.  No path-name guessing."""
    file = getattr(module, "__file__", None)
    if file is None:
        locations = getattr(module, "__path__", None)
        if locations is not None:
            paths = [Path(p).resolve() for p in locations]
            if paths and all(_is_under(p, TREE) for p in paths):
                return paths[0]
        raise RuntimeError("%s has no local __file__ or local namespace path"
                           % module.__name__)
    path = Path(file).resolve()
    if not _is_under(path, TREE):
        raise RuntimeError("WRONG TREE: %s loaded from %s; expected under %s"
                           % (module.__name__, path, TREE))
    return path


def load(name: str):
    import importlib
    if not _ACTIVE:
        activate()
    module = importlib.import_module(name)
    if getattr(module, "__file__", None) is None:
        raise RuntimeError("measured target %s is a namespace, not a module" % name)
    assert_under_tree(module)
    return module


def product_paths() -> dict:
    out = {}
    for name, module in sorted(sys.modules.items()):
        if module is not None and name.split(".")[0] in (PRODUCT_PACKAGES
                                                         | PRODUCT_MODULES):
            out[name] = str(assert_under_tree(module))
    return out


def assert_product_paths() -> dict:
    """Every loaded product module must sit under ``work/src``.

    ``eth80_frontend`` is the tree's own front-end DRIVER, kept in
    ``work/diagnostics``; it is allowed there and nowhere else.
    """
    paths = product_paths()
    bad = {n: p for n, p in paths.items()
           if not (_is_under(Path(p), SRC) or _is_under(Path(p), FRONTEND_DIR))}
    if bad:
        raise RuntimeError("product module outside %s: %s" % (SRC, bad))
    return paths


def source_digest() -> str:
    """sha256 over every product source file and this tree's front-end driver."""
    h = hashlib.sha256()
    paths = []
    for base, dirs, files in os.walk(SRC):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                paths.append(Path(base) / f)
    paths.append(FRONTEND_DIR / "eth80_frontend.py")
    paths.append(FRONTEND_DIR / "_bootstrap.py")
    for p in sorted(paths):
        h.update(str(p.relative_to(TREE)).replace("\\", "/").encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# --------------------------------------------------------------- banned path

def _handles():
    plain, methods = {}, {}
    for mod_name, names in BANNED.items():
        module = sys.modules.get(mod_name)
        if module is None:
            continue
        for name in names:
            obj = getattr(module, name, None)
            if obj is not None:
                plain[(mod_name, name)] = obj
    for mod_name, cls_name, attr in BANNED_METHODS:
        module = sys.modules.get(mod_name)
        cls = getattr(module, cls_name, None) if module is not None else None
        if cls is not None and hasattr(cls, attr):
            methods[(mod_name, cls_name, attr)] = getattr(cls, attr)
    return plain, methods


def assert_banned_absent() -> dict:
    """Refuse to run if the banned machinery is loaded or bound on this path.

    Call it AFTER the path has imported everything it is going to import (the
    first match lazily loads ``memory.cores`` for ``_sig_similarity``), and
    again after the run.
    """
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in BANNED_MODULES)
    if loaded:
        raise AssertionError("banned module(s) imported on this path: %s"
                             % (loaded,))
    plain, methods = _handles()
    hits = []
    for mod_name, module in sorted(sys.modules.items()):
        namespace = getattr(module, "__dict__", None) or {}
        for (owner, name), obj in plain.items():
            if mod_name != owner and namespace.get(name) is obj:
                hits.append("%s.%s is %s.%s" % (mod_name, name, owner, name))
    if hits:
        raise AssertionError("banned machinery bound on this path:\n  "
                             + "\n  ".join(hits))
    return {
        "banned_modules_loaded": loaded,
        "banned_symbols_defined": sorted("%s.%s" % key for key in plain),
        "banned_methods_defined": sorted(
            "%s.%s.%s" % key for key in methods),
        "foreign_bindings": hits,
        "note": ("a definition site may exist (matching.structural_similarity "
                 "defines structural_intersection); being DEFINED is not being "
                 "imported or called.  The scan fails on any other binding, and "
                 "the tripwire fails on any call."),
    }


class _Tripwire:
    """Replace every banned callable with one that raises, for the run."""

    def __init__(self):
        self.armed = []
        self._saved = []

    def __enter__(self):
        plain, methods = _handles()
        for (owner, name), obj in sorted(plain.items()):
            module = sys.modules[owner]
            self._saved.append((module, name, obj))
            setattr(module, name, _trap("%s.%s" % (owner, name)))
            self.armed.append("%s.%s" % (owner, name))
        for (owner, cls_name, attr), obj in sorted(methods.items()):
            cls = getattr(sys.modules[owner], cls_name)
            self._saved.append((cls, attr, obj))
            setattr(cls, attr, _trap("%s.%s.%s" % (owner, cls_name, attr)))
            self.armed.append("%s.%s.%s" % (owner, cls_name, attr))
        return self

    def __exit__(self, *exc):
        for target, name, obj in reversed(self._saved):
            setattr(target, name, obj)
        self._saved.clear()
        return False


def _trap(label):
    def _raise(*_a, **_k):
        raise AssertionError("banned machinery called on the classification "
                             "path: %s" % label)
    return _raise


def tripwire() -> _Tripwire:
    return _Tripwire()


def summarize_paths(paths: dict) -> dict:
    """Which directories the measurement actually loaded product code from."""
    roots = {}
    for name, path in paths.items():
        key = str(Path(path).parent.relative_to(TREE)).replace("\\", "/")
        roots.setdefault(key, []).append(name)
    return {k: sorted(v) for k, v in sorted(roots.items())}


def write_json(path, value) -> None:
    import json
    path = Path(path).resolve()
    if not _is_under(path, TREE):
        raise RuntimeError("refusing to write outside the tree: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
