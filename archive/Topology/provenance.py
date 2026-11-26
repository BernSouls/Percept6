"""Provenance diffstat for the tuning changes, with CRLF normalized.

``work/`` is not tracked by the repository (the repo's tracked ``src/`` is a
different copy), so the baseline is the pre-session export at
``work/bak/pre_percept5_20261114_161025``.  Line diffs are computed on
``\\r\\n``-normalized text; a file whose normalized content is identical but
whose bytes differ is reported as CHURN-ONLY (line-ending churn), not as a
code change.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import guard
from guard import TREE, write_json

BASELINE = TREE / "bak" / "pre_percept5_20261114_161025"
OUT = TREE / "topology" / "results" / "provenance.json"

#: files this tuning pass touched (new files have no baseline counterpart)
TOUCHED = [
    "src/matching/joint_mcs.py",
    "src/matching/alignment.py",
    "src/parameters/_text.py",
    "tests/matcher/conftest.py",
    "tests/matcher/test_phase3_f1_f4.py",
]


def _text(path):
    raw = path.read_bytes()
    return raw, raw.replace(b"\r\n", b"\n").decode("utf-8", "replace")


def diffstat(old: Path, new: Path):
    old_raw, old_text = _text(old)
    new_raw, new_text = _text(new)
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    added = removed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {
        "added": added, "removed": removed,
        "raw_equal": old_raw == new_raw,
        "normalized_equal": old_text == new_text,
        "churn_only": old_raw != new_raw and old_text == new_text,
    }


def main():
    rows = []
    for rel in TOUCHED:
        old = BASELINE / rel
        new = TREE / rel
        if not new.is_file():
            continue
        if not old.is_file():
            rows.append({"file": rel, "status": "new (no baseline)", "added": None,
                         "removed": None, "churn_only": False})
            continue
        stat = diffstat(old, new)
        rows.append({"file": rel, "status": "modified", **stat})
    out = {"baseline": str(BASELINE), "tracked_repo": "work/ is untracked",
           "normalization": "CRLF -> LF before line diff", "rows": rows}
    write_json(OUT, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
