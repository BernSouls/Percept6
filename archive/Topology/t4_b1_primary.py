"""Task 4 PART 0d — B1 ON PRIMARY with the REPAIRED map + per-kind radius.

PRIMARY = EQUIVARIANT content kinds (CONTOUR_PEAK, PEAK_CHAIN, RegionShape,
Corner). Reports case shares (a)/(b)/(c) by clean node_sim, old vs new side by
side, Orientation/MIXED/ABSENT-LIKE separately, both radii.
"""
from __future__ import annotations
import json, os, pickle, sys, tempfile
from collections import Counter, defaultdict
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
PRIMARY_KINDS = frozenset({"CONTOUR_PEAK", "PEAK_CHAIN", "RegionShape", "Corner"})
ORIENTATION_KINDS = frozenset({"Orientation"})
MIXED_KINDS = frozenset({"ContourFragment", "EndStopped"})
ABSENT_KINDS = frozenset({"Contour", "CurveComplex"})
TIE_TOL = 1e-9


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
    tmp = Path(tempfile.gettempdir()) / "t4_b1p" / (
        "%s%d%s_%d.png" % (entry["class"], int(entry["object"]),
                           entry["view"], dx))
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.is_file():
        with Image.open(entry["path"]) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        Image.fromarray(roll_image(arr, dx, 0)).save(tmp)
    e = dict(entry)
    e["path"] = str(tmp)
    return build(e)


def _save_shifted(name, dx, g):
    SHIFTED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SHIFTED_DIR / ("%s_%d.pkl" % (name, dx)), "wb") as fh:
        pickle.dump(g, fh)


def _dev_entries():
    import corpus
    split = corpus.split_tuning()
    return sorted([e for e in split["dev"] if e["view"] == "-090-000"],
                  key=lambda e: (e["class"], e["object"], e["view"]))


def _node_sim(sig_a, sig_b):
    from memory.cores import _sig_similarity
    return _sig_similarity(sig_a, sig_b)


def _classify(base_sig, sh_sigs_by_lens, true_sh_node, true_lens):
    len_list = sh_sigs_by_lens.get(true_lens, [])
    true_sig = None
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            true_sig = sh_sig
            break
    if true_sig is None:
        return "ABSENT", 0.0, 0, 0.0
    s_true = _node_sim(base_sig, true_sig)
    competitors = []
    for sh_node, sh_sig in len_list:
        if sh_node == true_sh_node:
            continue
        s = _node_sim(base_sig, sh_sig)
        competitors.append((sh_node, s))
    if not competitors:
        return "STRICT-LEAD", s_true, 1, 0.0
    max_comp = max(s for _, s in competitors)
    at_max = [s for _, s in competitors if abs(s - max_comp) < TIE_TOL]
    T = len(at_max) + 1
    if s_true > max_comp + TIE_TOL:
        return "STRICT-LEAD", s_true, 1, s_true - max_comp
    elif abs(s_true - max_comp) < TIE_TOL and T >= 2:
        return "EXACT-TIE", s_true, T, 0.0
    else:
        return "OUTRANKED", s_true, 1, max_comp - s_true


def _classify_population(base, sh_g, base_pos, sh_pos, cp, cp_info,
                         radius_by_lens, radius_8, kinds):
    sh_sigs_by_lens = defaultdict(list)
    for nsh in sh_g.subgraph.nodes:
        sh_sig = sh_g.sigs.get(nsh)
        if sh_sig is None:
            continue
        sh_sigs_by_lens[str(sh_sig[0])].append((nsh, sh_sig[1]))
    counts = Counter()
    n_total = 0
    for bn, (bx, by, blens) in base_pos.items():
        if blens not in kinds:
            continue
        n_total += 1
        bsh = cp.get(bn)
        if bsh is None or bsh not in sh_pos:
            continue
        b_sig = base.sigs.get(bn)
        sh_sig = sh_g.sigs.get(bsh)
        if b_sig is None or sh_sig is None:
            continue
        true_lens = str(sh_sig[0])
        case, s_true, T, margin = _classify(
            b_sig[1], sh_sigs_by_lens, bsh, true_lens)
        if case == "ABSENT":
            continue
        counts[case] += 1
    return counts, n_total


def main() -> int:
    print("=== PART 0d: B1 ON PRIMARY (repaired map, per-kind radius) ===",
          flush=True)
    out = {}
    radius_by_lens = json.load(open(RESULTS_DIR / "t4" / "primary_radius_dev.json"))
    radius_by_lens = {k: (v if v is not None else 8.0)
                      for k, v in radius_by_lens.items()}
    entries = _dev_entries()
    print("dev entries: %d" % len(entries), flush=True)
    prim_pk = Counter(); prim_8 = Counter()
    orient = Counter(); mixed = Counter(); absent = Counter()
    prim_pk_n = prim_8_n = 0
    for e in entries:
        name = "%s%d%s" % (e["class"], int(e["object"]), e["view"])
        print("  %s..." % name, flush=True)
        base = _load_dev_graph("B1", name)
        base_pos = _pos_lens(base)
        for dx in [2, -2]:
            sh_g = _load_shifted(name, dx)
            if sh_g is None:
                sh_g = _build_shifted(e, ARM_B1, dx)
                _save_shifted(name, dx, sh_g)
            sh_pos = _pos_lens(sh_g)
            cp, cp_info, _ = counterpart_map(
                base_pos, sh_pos, dx, 0, 8.0, radius_by_lens=radius_by_lens)
            cp8, cp_info8, _ = counterpart_map(base_pos, sh_pos, dx, 0, 8.0)
            c_pk, n_pk = _classify_population(
                base, sh_g, base_pos, sh_pos, cp, cp_info,
                radius_by_lens, 8.0, PRIMARY_KINDS)
            c_8, n_8 = _classify_population(
                base, sh_g, base_pos, sh_pos, cp8, cp_info8,
                radius_by_lens, 8.0, PRIMARY_KINDS)
            c_o, _ = _classify_population(
                base, sh_g, base_pos, sh_pos, cp, cp_info,
                radius_by_lens, 8.0, ORIENTATION_KINDS)
            c_m, _ = _classify_population(
                base, sh_g, base_pos, sh_pos, cp, cp_info,
                radius_by_lens, 8.0, MIXED_KINDS)
            c_a, _ = _classify_population(
                base, sh_g, base_pos, sh_pos, cp, cp_info,
                radius_by_lens, 8.0, ABSENT_KINDS)
            for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
                prim_pk[c] += c_pk[c]; prim_8[c] += c_8[c]
                orient[c] += c_o[c]; mixed[c] += c_m[c]; absent[c] += c_a[c]
            prim_pk_n += n_pk; prim_8_n += n_8
    def _shares(counts, total):
        return {c: (counts[c], counts[c] / total if total else 0.0)
                for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]}
    print("\n--- PRIMARY (per-kind, repaired) ---", flush=True)
    pk = _shares(prim_pk, prim_pk_n)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d (%.3f)" % (c[0], pk[c][0], pk[c][1]))
    print("  total n=%d" % prim_pk_n)
    print("\n--- PRIMARY (8px, repaired) [sensitivity] ---", flush=True)
    p8 = _shares(prim_8, prim_8_n)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d (%.3f)" % (c[0], p8[c][0], p8[c][1]))
    print("  total n=%d" % prim_8_n)
    if prim_pk_n < 0.7 * prim_8_n:
        print("\n  PRIMARY (per-kind) shrank >30%% vs 8px: %d vs %d"
              % (prim_pk_n, prim_8_n))
    print("\n--- Orientation (separate) ---", flush=True)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d" % (c[0], orient[c]))
    print("\n--- MIXED (separate) ---", flush=True)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d" % (c[0], mixed[c]))
    print("\n--- ABSENT-LIKE (separate) ---", flush=True)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  (%s) n=%d" % (c[0], absent[c]))
    old = {"STRICT-LEAD": (7354, 0.941), "EXACT-TIE": (2, 0.000),
           "OUTRANKED": (463, 0.059)}
    print("\n--- OLD (prior B1) vs NEW (PRIMARY, per-kind, repaired) ---",
          flush=True)
    for c in ["STRICT-LEAD", "EXACT-TIE", "OUTRANKED"]:
        print("  %s: old n=%d (%.3f) -> new n=%d (%.3f) delta=%+d"
              % (c[0], old[c][0], old[c][1], pk[c][0], pk[c][1],
                 pk[c][0] - old[c][0]))
    out["primary_per_kind"] = pk; out["primary_8px"] = p8
    out["orientation"] = orient; out["mixed"] = mixed; out["absent_like"] = absent
    out["old_prior_b1"] = old; out["radius_by_lens"] = radius_by_lens
    write_json(RESULTS_DIR / "t4" / "b1_primary.json", out)
    print("\n=== PART 0d complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
