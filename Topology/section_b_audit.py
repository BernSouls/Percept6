"""SECTION B verification on HEAD: PRESENT / FIXED, with the quoted line.

Reads the two matcher files as text (no import side effects), applies one
predicate per ledger item, and records the exact line that justifies the
verdict.  Written BEFORE any edit so the "before" verdicts are auditable;
re-run after the fixes to confirm every item reads FIXED.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
JM = TREE / "matching" / "joint_mcs.py"
GM = TREE / "matching" / "graded_mcs.py"


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def _find(src, pattern):
    rx = re.compile(pattern)
    for i, line in enumerate(src, 1):
        if rx.search(line):
            return i, line.rstrip()
    return None, ""


def audit():
    j, g = _lines(JM), _lines(GM)
    jt, gt = "\n".join(j), "\n".join(g)
    rows = []

    def add(item, file, present, quote, note):
        rows.append({"item": item, "file": file,
                     "verdict": "PRESENT" if present else "FIXED",
                     "quoted_line": quote, "note": note})

    # ---- matching/joint_mcs.py -----------------------------------------
    ln, q = _find(j, r"elif p is None or q is None:")
    nxt = j[ln] if ln else ""
    add("B1", "matching/joint_mcs.py",
        bool(re.search(r"if p is None or q is None:\s*\n\s*return 1\.0", jt)),
        "%d: %s | %d: %s" % (ln or 0, q, (ln or 0) + 1, nxt.strip()),
        "one-None must be 0.0; both-None may be a full unit")

    ln, q = _find(j, r"cat = edge_categorical_axes\(p, q\)")
    add("B2", "matching/joint_mcs.py",
        "edge_categorical_axes" not in jt,
        "%d: %s" % (ln or 0, q),
        "categorical topology_class agreement must earn credit")

    ln, q = _find(j, r"@lru_cache\(maxsize=None\)")
    ln2, q2 = _find(j, r"def node_sim\(u,v\):")
    add("B3", "matching/joint_mcs.py",
        not re.search(r"@lru_cache[^\n]*\n\s*def node_sim", jt),
        "%d: %s | %d: %s" % (ln or 0, q, ln2 or 0, q2),
        "node_sim must be memoized")

    present = "used_target_edges" in jt
    ln, q = _find(j, r"na, nb = sorted\(a\.nodes\), sorted\(b\.nodes\)")
    add("B4", "matching/joint_mcs.py", present,
        ("%d: %s" % (ln or 0, q)) if not present else
        _find(j, r"used_target_edges")[1],
        "dead code under a 1:1 node map; absent on HEAD = removed")

    ln, q = _find(j, r"return replace\(forward, budget_exhausted=True\)")
    add("B5", "matching/joint_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "forward branch must still report reverse exhaustion")

    ln, q = _find(j, r"score=2\.0\*\(nodes\+w\*edge_credit\)/denom")
    add("B6", "matching/joint_mcs.py",
        not bool(re.search(r"w=float\(P\.MATCH_EDGE_WEIGHT\)", jt)),
        "%d: %s" % (ln or 0, q),
        "pooled branch must read MATCH_EDGE_WEIGHT")

    ln, q = _find(j, r"under MATCH_SEPARATE_TERM_NORMALIZATION the edge term")
    add("B7", "matching/joint_mcs.py",
        not bool(re.search(r"under MATCH_SEPARATE_TERM_NORMALIZATION", jt)),
        "%d: %s" % (ln or 0, q),
        "docstring must not claim the pooled-only subtraction universally")

    ln, q = _find(j, r"na, nb = sorted\(_active\(a\)\), sorted\(_active\(b\)\)")
    add("B8", "matching/joint_mcs.py", ln is not None,
        "%d: %s" % (ln or 0, q or "(no removal line)"),
        "exclude_isolated must down-weight, not delete nodes")

    ln, q = _find(j, r"unmatched_cost = \(math\.fsum\(positive\) / len\(positive\)\)")
    ln2, q2 = _find(j, r"unmatched_cost = .*\* price_scale")
    add("B9", "matching/joint_mcs.py", ln is not None and ln2 is None,
        "%d: %s | %d: %s" % (ln or 0, q, ln2 or 0, q2),
        "derived price must be on q's scale (beta + isolated weight)")

    # ---- matching/graded_mcs.py ----------------------------------------
    present = "def _neighbourhood_coherence" in gt
    ln, q = _find(g, r"def _static_coherence\(")
    add("B10", "matching/graded_mcs.py", present,
        "%d: %s" % (ln or 0, q),
        "byte-identical duplicate removed; only static remains")

    present = bool(re.search(r"^\s*def .*\(.*edge_mass", gt, re.M))
    ln, q = _find(g, r"def _edge_terms\(")
    add("B11", "matching/graded_mcs.py", present,
        "%d: %s" % (ln or 0, q),
        "no unread edge_mass parameter in any signature")

    ln, q = _find(g, r"^def objective_value\(.*node_mass")
    if not ln:
        ln, q = _find(g, r"node_mass: Optional\[Dict\[int, float\]\] = None\)")
    add("B12", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "node_mass must reach objective_value and the reported node_term")

    ln, q = _find(g, r"if best is None or value > best:")
    ln2, q2 = _find(g, r"best_a = 0\.0")
    add("B13", "matching/graded_mcs.py", ln is None or ln2 is None,
        "%d: %s | %d: %s" % (ln or 0, q, ln2 or 0, q2),
        "one term per query edge with the BEST compatible reference edge in BOTH")

    ln, q = _find(g, r"const = -W_U \* \(1\.0 \+ 1\.0\)")
    ln2, q2 = _find(g, r"\+=? const|\) \+ const|\+ const\)")
    add("B14", "matching/graded_mcs.py", ln is not None and ln2 is None,
        "%d: %s | %d: %s" % (ln or 0, q, ln2 or 0, q2),
        "computed const must be used, not a hardcoded 2.0*W_U")

    ln, q = _find(g, r'_EDGE_AGREEMENT_FAILURES\["count"\] \+= 1')
    add("B15", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "the claimed failure counter exists")

    ln, q = _find(g, r"return max\(0\.0, min\(1\.0, res\.score / scale\)\)")
    add("B16", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "match_evidence clamps to [0,1]")

    ln, q = _find(g, r"js = \[j for j in support\.get\(i, \(\)\) if j in measured_terms\]")
    add("B17", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "NODE_REQUIRES_EDGE restricted to MEASURED support")

    ln, q = _find(g, r"q_types and c_types")
    add("B18", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "structurally supported pairs unioned before the joint solve")

    ln, q = _find(g, r"q_deg = \{u: sum\(1 for e in Q\.edges")
    add("B19", "matching/graded_mcs.py", ln is not None,
        "%d: %s" % (ln or 0, q or "(single-pass degree map)"),
        "degree map must be O(|E|), not O(n_q x |E|)")

    ln, q = _find(g, r"obj = objective_value\(correspondence")
    add("B20", "matching/graded_mcs.py", ln is None,
        "%d: %s" % (ln or 0, q),
        "reported score is objective_value(correspondence)")

    ln, q = _find(g, r"return 1\.0")
    ln2, q2 = _find(g, r"kappa = 1\.0")
    add("B21", "matching/graded_mcs.py",
        bool(re.search(r"if not rels_u or not rels_v:[\s\S]{0,300}?return 1\.0", gt))
        or "kappa = 1.0" in gt,
        "%d: %s | %d: %s" % (ln or 0, q, ln2 or 0, q2),
        "edgeless reads 0.0 unconditionally, like alignment.py")

    return {"files": {"matching/joint_mcs.py": str(JM),
                      "matching/graded_mcs.py": str(GM)},
            "items": rows,
            "counts": {"PRESENT": sum(r["verdict"] == "PRESENT" for r in rows),
                       "FIXED": sum(r["verdict"] == "FIXED" for r in rows)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TREE / "results" / "section_b_verdict.json"))
    a = ap.parse_args(argv)
    result = audit()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    for r in result["items"]:
        print("%-4s %-7s %s" % (r["item"], r["verdict"], r["quoted_line"][:110]))
    print(result["counts"], "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
