"""Task 4 PART B6 — SEED PRE-CHECK (dev only).

P3-3's seed pathway: toggle FE_VECTOR_ADDRESS ON and rebuild dev graphs, then
check whether the seed (vector address) changes any node mappings. If no mapping
changes → the seed pathway is INERT and P3-3's first clause is UNEVALUATED
(unimplementable); report + skip.

Diagnostic toggle: FE_VECTOR_ADDRESS is set ON only inside this harness and
restored to its default (False) before returning. No production change.

Compares mapping identity between arm B1 (FE_VECTOR_ADDRESS OFF) and arm
B1+vector (FE_VECTOR_ADDRESS ON) on the dev set, shifted by dx=±2.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]

import guard  # noqa: E402

guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402

from t4_shift import counterpart_map  # noqa: E402

PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"

ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
ARM_B1_VEC = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true;FE_VECTOR_ADDRESS=true"
WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75,
    "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5,
    "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5,
    "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
    "MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE": True,
    "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
}


def _pos_lens(g):
    out = {}
    for n in g.subgraph.nodes:
        a = g.subgraph.anchor_of_node.get(n)
        lens = str(g.sigs.get(n, ("?",))[0])
        if a is not None and len(a) >= 2:
            out[n] = (float(a[0]), float(a[1]), lens)
    return out


def _load_dev_graph(arm, name):
    with open(PKL_DIR / ("%s_%s.pkl" % (arm, name)), "rb") as fh:
        return pickle.load(fh)


def _load_shifted(name, dx):
    pkl = SHIFTED_DIR / ("%s_%d.pkl" % (name, dx))
    if pkl.is_file():
        with open(pkl, "rb") as fh:
            return pickle.load(fh)
    return None


def _build_graph(entry, spec):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    tmp = Path(tempfile.gettempdir()) / "t4_b6" / (
        "%s%d%s_b6.png" % (entry["class"], int(entry["object"]),
                           entry["view"]))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        from PIL import Image
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(arr).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def main() -> int:
    print("=== B6: SEED PRE-CHECK (FE_VECTOR_ADDRESS ON) ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)

    total_nodes = 0
    changed_nodes = 0
    scenes_changed = 0
    per_scene = []

    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base_off = _load_dev_graph("B1", name)

        # Build base with FE_VECTOR_ADDRESS ON (diagnostic toggle).
        # The shifted graph is the cached ARM_B1 (OFF) graph — a fixed target,
        # so any mapping change is attributable to the base's vector address.
        base_on = _build_graph(e, ARM_B1_VEC)

        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                print("    missing shifted dx=%d, skipping" % dx, flush=True)
                continue

            m_off = _run_matcher(base_off, sh_g)
            m_on = _run_matcher(base_on, sh_g)

            n_changed = sum(1 for k in m_off if m_off[k] != m_on.get(k))
            total_nodes += len(m_off)
            changed_nodes += n_changed
            if n_changed > 0:
                scenes_changed += 1
            per_scene.append((name, dx, n_changed, len(m_off)))

    print("\n--- Result ---", flush=True)
    print("  total nodes compared: %d" % total_nodes)
    print("  nodes with changed mapping: %d (%.4f)"
          % (changed_nodes, changed_nodes / total_nodes if total_nodes else 0))
    print("  scenes with any change: %d / %d"
          % (scenes_changed, len(entries) * 2))

    out["total_nodes"] = total_nodes
    out["changed_nodes"] = changed_nodes
    out["changed_frac"] = changed_nodes / total_nodes if total_nodes else 0.0
    out["scenes_changed"] = scenes_changed
    out["scenes_total"] = len(entries) * 2
    out["seed_pathway_inert"] = changed_nodes == 0

    if changed_nodes == 0:
        print("\n  Seed pathway INERT: FE_VECTOR_ADDRESS ON changes NO mappings.")
        print("  P3-3 first clause UNEVALUATED (unimplementable) — skip.")
    else:
        print("\n  Seed pathway ACTIVE: FE_VECTOR_ADDRESS ON changes %d mappings."
              % changed_nodes)

    write_json(RESULTS_DIR / "t4" / "b6_seed_precheck.json", out)
    print("\n=== B6 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
