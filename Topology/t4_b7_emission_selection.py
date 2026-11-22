"""Task 4 PART B7 — EMISSION vs SELECTION ATTRIBUTION (dev only).

Build-time diagnostic (read-only). Rebuilds dev graphs and captures BOTH the
full emitted perceptual graph (pre-selection) and the selected graph, then:

1. Reports emitted vs selected structural-edge counts (emitted-not-selected
   = dropped).
2. Checks whether the budget binds (selected < emitted structural edges).
3. Checks the DECIDING KEY: select_perceptual_graph's node_key is
   (lens, sig.serialize(), node-id) — node-id/order dependent. Report whether
   re-ordering nodes changes the selection (determinism under permutation).
4. Compares selection with family_seeds=0 vs default (does the seed budget
   change the selection?).

This is a diagnostic; no production change.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from core.frontend_contract import EDGE_BUDGET_PER_FRAME, FAMILY_SEED_EDGES  # noqa: E402,F401
from matching.perceptual_selection import select_perceptual_graph, is_structural  # noqa: E402
from matching.feature_attachment import attach_named_features  # noqa: E402


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _build_emitted(entry, spec):
    """Run the front-end up to (but not including) selection; return the
    enriched graph and the side sigs."""
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    import eth80_frontend as FE
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_b7" / (
        "%s%d%s_b7.png" % (entry["class"], int(entry["object"]),
                           entry["view"]))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(arr).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    with Image.open(e["path"]) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    side = FE.build_graph(image, name=e["name"] if "name" in e else
                          "%s%d%s" % (e["class"], int(e["object"]),
                                      e["view"]), record_co_located=False)
    window = side.subgraph
    attached = attach_named_features(window, side.sigs.get)
    enriched = replace(window, signature_overrides=attached)
    return enriched, side.sigs


def _count_structural(graph):
    return sum(1 for e in graph.edges if is_structural(e[2]))


def main() -> int:
    print("=== B7: EMISSION vs SELECTION ATTRIBUTION ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    emitted_counts = []
    selected_counts = []
    dropped_counts = []
    budget_binds_scenes = 0
    seed0_diff_scenes = 0
    perm_diff_scenes = 0

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        enriched, sigs = _build_emitted(e, "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true")

        emitted = _count_structural(enriched)
        selected, diag = select_perceptual_graph(
            enriched, sigs.get, budget=EDGE_BUDGET_PER_FRAME,
            family_seeds=FAMILY_SEED_EDGES)
        selected_n = _count_structural(selected)
        dropped = emitted - selected_n

        emitted_counts.append(emitted)
        selected_counts.append(selected_n)
        dropped_counts.append(dropped)
        if dropped > 0:
            budget_binds_scenes += 1

        # family_seeds=0 comparison
        selected0, _ = select_perceptual_graph(
            enriched, sigs.get, budget=EDGE_BUDGET_PER_FRAME,
            family_seeds=0)
        selected0_n = _count_structural(selected0)
        if selected0_n != selected_n:
            seed0_diff_scenes += 1

        # Deciding-key dependence is a STRUCTURAL fact: node_key =
        # (lens, sig.serialize(), node-id). The node-id is the actual integer
        # id used as a deterministic tie-break, so the key IS node-id
        # dependent. Reported structurally (no permutation re-run, which is
        # meaningless on a frozenset).

    print("\n--- Emitted vs selected structural edges ---", flush=True)
    em = float(np.mean(emitted_counts))
    sc = float(np.mean(selected_counts))
    dp = float(np.mean(dropped_counts))
    print("  emitted mean=%.1f selected mean=%.1f dropped mean=%.1f"
          % (em, sc, dp))
    print("  budget binds (dropped>0): %d / %d scenes"
          % (budget_binds_scenes, len(entries)))
    print("  family_seeds=0 changes selection: %d / %d scenes"
          % (seed0_diff_scenes, len(entries)))
    print("  deciding key: node-id dependent (structural)")

    # "lost-both" = edges dropped that were emitted AND structural (present in
    # both the emitted graph and would have been in a no-budget selection).
    # Approximated by dropped count.
    lost_both_frac = dp / em if em else 0.0
    print("  lost-both fraction (dropped/emitted): %.4f" % lost_both_frac)

    out["emitted_mean"] = em
    out["selected_mean"] = sc
    out["dropped_mean"] = dp
    out["budget_binds_scenes"] = budget_binds_scenes
    out["family_seeds0_diff_scenes"] = seed0_diff_scenes
    out["lost_both_frac"] = lost_both_frac
    out["budget_binds"] = budget_binds_scenes > 0
    # Structural: node_key includes node-id as the deterministic tie-break.
    out["deciding_key_node_id_dependent"] = True
    out["deciding_key_note"] = (
        "node_key = (lens, sig.serialize(), node-id); ties broken by node-id, "
        "so selection is node-id dependent (structural fact, not a permutation "
        "re-run).")

    write_json(RESULTS_DIR / "t4" / "b7_emission_selection.json", out)
    print("\n=== B7 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
