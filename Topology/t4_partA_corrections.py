"""Task 4 PART A — FIX THE REPORT'S ERRORS.

A1: Arithmetic: content-content preservation = 4302/5910 = 72.8%.
    Per-family stability table with present_frac.
A2: Class precedence + bridge table by kind.
A9: Dev vs held-out citation.
A11: Determinism hashes recomputed note.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

CONTENT_TOKEN_LENSES = frozenset({
    "CONTOUR_PEAK", "ContourFragment", "PEAK_CHAIN", "EndStopped",
    "Corner", "Junction", "T", "Y", "RegionShape", "IllusoryContour",
})


def main():
    print("=== PART A: CORRECTIONS ===", flush=True)
    out = {}

    # ---- A1: Arithmetic + per-family stability ----
    print("\n--- A1: Arithmetic and per-family stability ---", flush=True)
    d1 = json.load(open(RESULTS_DIR / "t4" / "part1_edge_diag.json"))
    pp = d1["1a_edge_stability"]["B1"]["partition_pairs"]["content-content"]
    total_cc = pp["preserved"] + pp["lost_ep"] + pp["lost_both"]
    a1_content_content = total_cc
    a1_content_content_preserved = pp["preserved"]
    a1_content_content_pct = 100.0 * pp["preserved"] / total_cc
    print("  content-content: %d/%d = %.1f%% (was 4302/6350 = 67.7%%)"
          % (pp["preserved"], total_cc, a1_content_content_pct))

    fam_table = {}
    for fam, v in sorted(d1["1a_edge_stability"]["B1"]["per_family"].items(),
                         key=lambda x: -x[1]["total_base"]):
        both_track = v["preserved"] + v["lost_both"]
        present_frac = v["preserved"] / both_track if both_track > 0 else 0.0
        fam_table[fam] = {
            "total": v["total_base"],
            "preserved": v["preserved"],
            "lost_endpoint": v["lost_ep"],
            "lost_both_survive": v["lost_both"],
            "added_new_node": v["added_new"],
            "added_surviving": v["added_surv"],
            "present_frac_among_both_track": present_frac,
        }
    out["A1"] = {
        "content_content_preservation": "%d/%d = %.1f%%" % (
            a1_content_content_preserved, a1_content_content, a1_content_content_pct),
        "per_family_stability": fam_table,
    }

    # ---- A2: Class precedence + bridge table ----
    print("\n--- A2: Class precedence + bridge table ---", flush=True)
    d0 = json.load(open(RESULTS_DIR / "t4" / "part0_corrections.json"))
    part_data = d0["0a_partition"]

    # Recompute classes with precedence: ABSENT-LIKE first, then EQUIVARIANT,
    # then STATIC-WHEN-PRESENT, else MIXED.
    bridge = {}
    for kind, info in part_data.items():
        n = info["n"]
        partition = info.get("partition", "?")
        if info.get("class") == "ABSENT" or n < 20:
            cls = "ABSENT"
        elif n < 20:
            cls = "ABSENT"
        else:
            persisters_frac = info.get("persisters_frac", 0)
            tracked_frac = info.get("tracked_frac", 0)
            static_frac = info.get("static_frac", 0)
            if persisters_frac < 0.50:
                cls = "ABSENT-LIKE"
            elif tracked_frac >= 0.70:
                cls = "EQUIVARIANT"
            elif persisters_frac > 0 and static_frac / persisters_frac >= 0.80:
                cls = "STATIC-WHEN-PRESENT"
            else:
                cls = "MIXED"
        in_dclass = cls in ("EQUIVARIANT", "STATIC-WHEN-PRESENT")
        bridge[kind] = {
            "partition": partition,
            "n": n,
            "class": cls,
            "in_D_class": in_dclass,
            "tracked_frac": info.get("tracked_frac", 0),
            "static_frac": info.get("static_frac", 0),
            "persisters_frac": info.get("persisters_frac", 0),
        }
        print("  %-20s [%s] n=%-6d class=%-20s in_D_class=%s"
              % (kind, partition, n, cls, in_dclass))
    out["A2_bridge"] = bridge

    # ---- A9: Dev vs held-out citation ----
    print("\n--- A9: Dev vs held-out ---", flush=True)
    out["A9"] = {
        "dev_pooled_real_minus_no_edge": "-0.0019 [-0.0117, +0.0099]",
        "heldout_pooled_real_minus_no_edge": "-0.0196 [-0.0281, -0.0123]",
        "note": "Dev is one view (-090-000, 24 scenes); held-out has two views "
                "(16 images). The by-view split happens in the final comparison only.",
    }
    print("  dev: -0.0019 [-0.0117, +0.0099]")
    print("  heldout: -0.0196 [-0.0281, -0.0123]")

    # ---- A11: Determinism hashes ----
    print("\n--- A11: Determinism ---", flush=True)
    det = json.load(open(RESULTS_DIR / "t4" / "determinism_check.json"))
    out["A11"] = {
        "note": "Determinism hashes were recomputed on the rebuilt graphs "
                "during the previous pass (commits c62719e/1543827/caedf1e). "
                "The hashes below are from that recomputation.",
        "arms": det["arms"],
        "all_match": det["all_match"],
        "seeds": det["seeds"],
    }
    print("  all_match: %s (seeds %s)" % (det["all_match"], det["seeds"]))

    write_json(RESULTS_DIR / "t4" / "partA_corrections.json", out)
    print("\n=== PART A corrections complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
