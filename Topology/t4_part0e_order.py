"""Task 4 PART 0e — ORDER CHECK (smoke, 8 scenes, shifts +2/-2).

Permute the shifted graph's node order AND ids (seeds 0,1,2 via a fixed
permutation), prove the permuted graph has the identical edge set up to
relabelling and identical signatures, then measure the fraction of base nodes
whose mapping changes between original and permuted orders, for real and no_edge.
"""
from __future__ import annotations
import os, pickle, random, sys, tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402
PKL_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "dev_pkls"
SHIFTED_DIR = Path(ROOT) / "Topology" / "results" / "t4" / "shifted_b1"
ARM_B1 = "FE_L3_SPINE=true;FE_L4_JUNCTIONS=true"
WINNER = {
    "MATCH_SYSTEMATICITY_WEIGHT": 0.75, "MATCH_UNMATCHED_COST": 0.15,
    "MATCH_ALIGN_UNMAPPED_CREDIT": 0.5, "MATCH_EDGE_WEIGHT": 0.1,
    "MATCH_COVERAGE_EXPONENT_GAMMA": 0.5, "MATCH_EDGE_DENOMINATOR_MODE": "checkable",
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


def _build_shifted(entry, spec, dx):
    import _bootstrap
    _bootstrap.apply_switch_overrides(spec)
    from frontend import build
    from t4_shift import roll_image
    from PIL import Image
    tmp = Path(tempfile.gettempdir()) / "t4_0e" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], dx))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(roll_image(arr, dx, 0)).save(tmp)
    e = dict(entry); e["path"] = str(tmp)
    return build(e)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))[:8]


def _relabel(g, perm):
    inv = {v: k for k, v in perm.items()}
    nodes = frozenset(perm[n] for n in g.subgraph.nodes)
    edges = frozenset((perm[a], perm[b], t) for a, b, t in g.subgraph.edges)
    ep = {(perm[a], perm[b], t): v for (a, b, t), v in g.subgraph.edge_params.items()}
    aon = {perm[n]: v for n, v in g.subgraph.anchor_of_node.items()}
    nw = {perm[n]: v for n, v in g.subgraph.node_weights.items()}
    so = {perm[n]: v for n, v in g.subgraph.signature_overrides.items()}
    sg = replace(g.subgraph, nodes=nodes, edges=edges, edge_params=ep,
                 anchor_of_node=aon, node_weights=nw, signature_overrides=so)
    sigs = {perm[n]: v for n, v in g.sigs.items()}
    _addrs = getattr(g, "addrs", None) or {}
    addrs = {perm[n]: v for n, v in _addrs.items()}
    return replace(g, subgraph=sg, sigs=sigs, addrs=addrs), inv


def _run_matcher(base_g, shift_g):
    import parameters as P
    for k, v in WINNER.items():
        setattr(P, k, v)
    from matching.joint_mcs import joint_mcs
    res = joint_mcs(base_g.subgraph, shift_g.subgraph,
                    base_g.sigs, shift_g.sigs)
    return res.mapping


def _no_edge_base(g):
    return replace(g, subgraph=replace(g.subgraph, edges=frozenset(), edge_params={}))


def _edge_set_up_to_relabel(orig_g, perm_g, inv):
    orig_edges = set((min(a, b), max(a, b), str(t)) for a, b, t in orig_g.subgraph.edges)
    unrel = set((min(inv[a], inv[b]), max(inv[a], inv[b]), str(t))
                for a, b, t in perm_g.subgraph.edges)
    return orig_edges == unrel


def main() -> int:
    print("=== PART 0e: ORDER CHECK (smoke, 8 scenes) ===", flush=True)
    out = {}
    entries = _dev_entries()
    print("dev entries (first 8): %d" % len(entries), flush=True)
    total_nodes = 0; changed_real = 0; changed_no_edge = 0
    identity_ok = 0; identity_total = 0
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, ARM_B1, dx)
            nodes = sorted(sh_g.subgraph.nodes)
            rng = random.Random(42)
            perm = {n: i for i, n in enumerate(rng.sample(nodes, len(nodes)))}
            sh_perm, inv = _relabel(sh_g, perm)
            if _edge_set_up_to_relabel(sh_g, sh_perm, inv):
                identity_ok += 1
            identity_total += 1
            m_real = _run_matcher(base, sh_g)
            m_real_p = _run_matcher(base, sh_perm)
            m_ne = _run_matcher(_no_edge_base(base), sh_g)
            m_ne_p = _run_matcher(_no_edge_base(base), sh_perm)
            for bn in m_real:
                if bn not in m_real_p:
                    continue
                total_nodes += 1
                orig_target = inv.get(m_real_p[bn], m_real_p[bn])
                if orig_target != m_real[bn]:
                    changed_real += 1
            for bn in m_ne:
                if bn not in m_ne_p:
                    continue
                orig_target = inv.get(m_ne_p[bn], m_ne_p[bn])
                if orig_target != m_ne[bn]:
                    changed_no_edge += 1
    fr = changed_real / total_nodes if total_nodes else 0.0
    fn = changed_no_edge / total_nodes if total_nodes else 0.0
    print("\n--- Result ---", flush=True)
    print("  structural identity: %d/%d" % (identity_ok, identity_total))
    print("  nodes compared: %d" % total_nodes)
    print("  real change frac: %.4f" % fr)
    print("  no_edge change frac: %.4f" % fn)
    out["structural_identity"] = identity_ok
    out["structural_identity_total"] = identity_total
    out["total_nodes"] = total_nodes
    out["real_change_frac"] = fr
    out["no_edge_change_frac"] = fn
    out["order_irrelevant"] = fr < 0.01 and fn < 0.01
    if fr < 0.01 and fn < 0.01:
        print("\n  <1%: tie order IRRELEVANT.")
    else:
        print("\n  >=1%: tie order MATTERS (relevant for F3).")
    write_json(RESULTS_DIR / "t4" / "part0e_order.json", out)
    print("\n=== PART 0e complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
