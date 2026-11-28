"""M34 items 1 and 3: build fragments from the unwired region machinery, then screen them.

    python Topology/m34_fragments.py --probe     # item 1's premise, measured on one graph
    python Topology/m34_fragments.py --build     # fragments + descriptors, per cell, resumable
    python Topology/m34_fragments.py --report    # item 1: counts, sizes, ungrouped share, stability
    python Topology/m34_fragments.py --screen    # item 3: channel-alone recall@8 and the loss oracle

THE PREMISE (`--probe`).  M34's item 1 names four edge classes as the grouping.  This measures
whether any of them touches a place node.  Measured on apple/1 at -035-045: 1,506 nodes, 2,686
edges, **0** with a place endpoint -- the front end emits places and region machinery as two
disjoint universes.  The grouping this file actually uses is registry-before-number: a place joins
the fragment of its nearest `ContourFragment` anchor (fine), its nearest `Contour` anchor (coarse),
or -- the geometric control -- a single-linkage cluster at 16 px.  All three are in
`matching.placement_fragment.GROUPINGS`; the screen decides between them, on FIT, and states it.

THE SCREENS (item 3), both required, both before any door read:

    (a) CHANNEL-ALONE recall@8 on the FIT cells (-035-045, -045-180) against the 80-object gallery,
        chance 0.100.  BAR: >= 0.675.
    (b) ORACLE ON THE LOSSES: of M31's 31 same-class losses, in how many does the fragment channel
        rank the truth above the matcher's winner?  Colour read 22/30.  BAR: >= 15/31.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

from Topology import m26_records as REC                        # noqa: E402

RECORDS = os.path.join(ROOT, "data", "placement_records_m32.npz")
CACHE = os.path.join(ROOT, ".cache", "m34_fragments")
OUT = os.path.join(ROOT, "Topology", "results", "placement", "m34_fragments.json")
PAIRS = os.path.join(ROOT, "Topology", "results", "placement", "m30_read_pairs.jsonl")
REF = "000-000"
VIEWS = ("000-000", "035-045", "045-180")
FIT_CELLS = {"fit_a": "035-045", "fit_b": "045-180"}
S = 8
BAR_A = 0.675
BAR_B = 15
COLOUR_LOSSES = 22


def _split() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(REC.objects()):
        out.setdefault(str(o["split"]), []).append(i)
    return out


# --------------------------------------------------------------------------- item 1: the premise
def probe() -> Dict[str, Any]:
    """Does any edge class the mission names touch a place node?  One graph, every edge counted."""
    import matching.placement_frontend as FE
    from matching import placement_records as PR
    o = REC.objects()[0]
    cells = PR.load_cells(RECORDS, ["035-045"])
    rec = cells[(o["key"], "035-045")]
    places = {int(p["n"]) for p in rec["places"]}
    _img, g, _side = FE.cell(REC.path_of(o, "035-045"), "m34probe")
    sub = g.subgraph
    per: Dict[str, Dict[str, int]] = {}
    for (u, v, et) in sub.edges:
        name = getattr(et, "name", str(et))
        row = per.setdefault(name, {"total": 0, "both_place": 0, "one_place": 0})
        row["total"] += 1
        pu, pv = int(u) in places, int(v) in places
        if pu and pv:
            row["both_place"] += 1
        elif pu or pv:
            row["one_place"] += 1
    named = ("EDGE_BOUNDARY_ATTACHMENT", "EDGE_COCIRCULAR", "EDGE_BORDER_OWNERSHIP",
             "EDGE_BOUNDARY_SUCCESSOR", "BOUNDARY_ATTACHMENT", "COCIRCULAR",
             "BORDER_OWNERSHIP_EDGE", "BOUNDARY_SUCCESSOR")
    named_rows = {k: v for k, v in per.items() if k in named}
    degree = collections.Counter()
    for (u, v, _et) in sub.edges:
        degree[int(u)] += 1
        degree[int(v)] += 1
    isolated = sum(1 for n in places if degree.get(int(n), 0) == 0)
    out = {"object": o["key"], "view": "035-045", "nodes": len(sub.nodes), "edges": len(sub.edges),
           "place_nodes": len(places), "place_nodes_isolated": isolated,
           "place_endpoint_edges_total": sum(v["both_place"] + v["one_place"] for v in per.values()),
           "named_edge_classes_present": named_rows,
           "all_edge_types": dict(sorted(per.items(), key=lambda kv: -kv[1]["total"]))}
    print(json.dumps({k: v for k, v in out.items() if k != "all_edge_types"}, indent=1))
    with open(OUT.replace("m34_fragments.json", "m34_premise.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


# --------------------------------------------------------------------------- item 1: the build
def _anchors(objects: Dict[str, Any], grouping: str) -> List[Tuple[float, float]]:
    import matching.placement_frontend as FE
    o = objects["_o"]
    _img, g, _side = FE.cell(REC.path_of(o, objects["_view"]), "m34frag")
    sub = g.subgraph
    if grouping == "contour_fragment":
        want = "ContourFragment"
    elif grouping == "contour":
        want = "Contour"
    else:
        return []
    out: List[Tuple[float, float]] = []
    for n in sub.nodes:
        pair = g.sigs.get(n)
        if pair and pair[0] == want:
            a = sub.anchor_of_node.get(n)
            if a is not None and len(a) >= 2:
                out.append((float(a[0]), float(a[1])))
    return out


def build(views: Sequence[str] = VIEWS, groupings: Sequence[str] = None) -> Dict[str, Any]:
    from matching import placement_fragment as PF
    from matching import placement_records as PR
    groupings = list(groupings or PF.GROUPINGS)
    os.makedirs(CACHE, exist_ok=True)
    objs = REC.objects()
    t0 = time.time()
    total = len(objs) * len(views)
    done = 0
    for o in objs:
        for v in views:
            slug = os.path.join(CACHE, "%s_%s.npz" % (o["key"].replace("/", "-"), v))
            done += 1
            if os.path.exists(slug):
                continue
            cells = PR.load_cells(RECORDS, [v])
            rec = cells[(o["key"], v)]
            places = rec["places"]
            payload: Dict[str, np.ndarray] = {"n_places": np.asarray([len(places)])}
            for gname in groupings:
                if gname == "proximity":
                    groups = PF.group_by_proximity(places)
                else:
                    anch = _anchors({"_o": o, "_view": v}, gname)
                    groups = PF.group_by_anchor(places, anch)
                payload["%s|count" % gname] = np.asarray([len(groups)])
                payload["%s|sizes" % gname] = np.asarray([len(x) for x in groups], dtype=np.int32)
                payload["%s|members" % gname] = np.asarray(
                    [len(x) for x in groups] and
                    [i for x in groups for i in (list(x) + [-1])], dtype=np.int32)
                desc = np.asarray([PF.full_descriptor(places, x) for x in groups],
                                  dtype=np.float64).reshape(-1, len(PF.FIELDS))
                payload["%s|desc" % gname] = desc
            np.savez(slug, **payload)
            if done % 40 == 0 or done == total:
                print("  %d/%d cells, %.0fs" % (done, total, time.time() - t0), flush=True)
    return {"cells": total, "seconds": round(time.time() - t0, 1)}


def _load_cells(views: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for o in REC.objects():
        cov = {v: os.path.exists(os.path.join(CACHE, "%s_%s.npz"
                                              % (o["key"].replace("/", "-"), v)))
               for v in views}
        if not all(cov.values()):
            continue
        for v in views:
            z = np.load(os.path.join(CACHE, "%s_%s.npz" % (o["key"].replace("/", "-"), v)))
            out[(o["key"], v)] = {k: z[k] for k in z.files}
    return out


# --------------------------------------------------------------------------- item 1: the report
def report() -> Dict[str, Any]:
    from matching import placement_fragment as PF
    cells = _load_cells(VIEWS)
    keys = sorted({k for k, _v in cells})
    out: Dict[str, Any] = {"cells": len(cells), "groupings": {}}
    for gname in PF.GROUPINGS:
        per_view: Dict[str, Any] = {}
        sizes: Dict[str, List[int]] = collections.defaultdict(list)
        counts: Dict[str, List[int]] = collections.defaultdict(list)
        member_sets: Dict[Tuple[str, str], List[List[int]]] = {}
        for v in VIEWS:
            for k in keys:
                z = cells.get((k, v))
                if z is None:
                    continue
                sz = z["%s|sizes" % gname].tolist()
                sizes[v].extend(sz)
                counts[v].append(int(z["%s|count" % gname][0]))
                flat = z["%s|members" % gname].tolist()
                groups, cur = [], []
                for x in flat:
                    if x < 0:
                        groups.append(cur)
                        cur = []
                    else:
                        cur.append(int(x))
                if cur:
                    groups.append(cur)
                member_sets[(k, v)] = groups
        for v in VIEWS:
            n = len(sizes[v])
            covered = sum(sum(len(m) for m in member_sets[(k, v)])
                          for k in keys if (k, v) in cells)
            tot_places = sum(int(cells[(k, v)]["n_places"][0]) for k in keys if (k, v) in cells)
            per_view[v] = {"fragments_per_object_mean": float(np.mean(counts[v])) if counts[v] else 0.0,
                           "places_per_fragment_mean": float(np.mean(sizes[v])) if n else 0.0,
                           "places_per_fragment_median": float(np.median(sizes[v])) if n else 0.0,
                           "fragment_size_p95": float(np.percentile(sizes[v], 95)) if n else 0.0,
                           "ungrouped_share": float(1.0 - covered / max(tot_places, 1)),
                           "objects": len(counts[v])}
        # stability: the size spectrum in one view against the other, per object and pooled
        if len(VIEWS) >= 2:
            a, b = VIEWS[1], VIEWS[2] if len(VIEWS) > 2 else VIEWS[1]
            corr = _size_corr(member_sets, keys, a, b)
            per_view["stability"] = {"views": [a, b],
                                     "size_hist_l1_units": float(sum(
                                         abs(np.histogram(sizes[a], bins=10, range=(1, 50))[0][i]
                                             - np.histogram(sizes[b], bins=10, range=(1, 50))[0][i])
                                         for i in range(10))),
                                     "per_object_size_spearman": corr,
                                     "note": "the same object's fragment-size spectrum in the two "
                                             "views: Spearman over its (sorted) sizes, matched by "
                                             "rank, plus the pooled 10-bin L1 count difference"}
            per_view.pop("size_hist_l1_units", None)
        out["groupings"][gname] = per_view
        print("  %-16s %s" % (gname, json.dumps({v: {k: (round(x, 3) if isinstance(x, float) else x)
                                                     for k, x in per_view[v].items()}
                                                 for v in VIEWS})))
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def _size_corr(member_sets: Dict[Tuple[str, str], List[List[int]]],
               keys: Sequence[str], a: str, b: str) -> float:
    cs: List[float] = []
    for k in keys:
        sa = sorted(len(m) for m in member_sets.get((k, a), []))
        sb = sorted(len(m) for m in member_sets.get((k, b), []))
        n = min(len(sa), len(sb))
        if n >= 3:
            cs.append(float(np.corrcoef(sa[:n], sb[:n])[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


# --------------------------------------------------------------------------- item 3: the screens
def _fragment_matrix(query: Dict[str, Any], gallery: Dict[str, Any], gname: str,
                     sig: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    from matching import placement_fragment as PF
    Q = query["%s|desc" % gname]
    G = gallery["%s|desc" % gname]
    if Q.size == 0 or G.size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return PF.similarity(Q, G, sig), G


def _coverage(S: np.ndarray) -> float:
    """One number per (query, gallery) pair from a fragment-by-fragment similarity block.

    REGISTERED BEFORE THE READ: symmetric mean-of-best.  Each of the query's parts is matched to
    the gallery part that explains it best and the means are taken in both directions and averaged,
    so a query with many parts the gallery cannot explain is penalised for them -- the whole point
    of a PART descriptor over a single best pair.  Nothing is tuned: no exponent, no top-k, no
    pooling weight beyond the 1/2 that makes it symmetric.
    """
    if S.size == 0:
        return 0.0
    return 0.5 * (float(S.max(axis=1).mean()) + float(S.max(axis=0).mean()))


def screen() -> Dict[str, Any]:
    from matching import placement_fragment as PF
    cells = _load_cells(VIEWS)
    objs = REC.objects()
    keys = [o["key"] for o in objs]
    have = [k for k in keys if (k, REF) in cells]
    split = _split()
    out: Dict[str, Any] = {"objects_with_fragments": len(have), "objects": len(keys),
                           "chance": {"recall@8": 0.100, "top1": 0.0125}, "groupings": {}}
    for gname in PF.GROUPINGS:
        pool = np.vstack([cells[(k, REF)]["%s|desc" % gname] for k in have
                          if cells[(k, REF)]["%s|desc" % gname].size]) \
            if any(cells[(k, REF)]["%s|desc" % gname].size for k in have) else np.zeros((0, len(PF.FIELDS)))
        if pool.shape[0] < 4:
            out["groupings"][gname] = {"error": "no fragments"}
            continue
        sig = PF.scales(pool)
        res: Dict[str, Any] = {"sigma_columns": int(sig.size)}
        for cell, view in FIT_CELLS.items():
            M = np.zeros((len(keys), len(keys)), dtype=np.float64)
            for i, k in enumerate(keys):
                if (k, view) not in cells:
                    continue
                Q = cells[(k, view)]["%s|desc" % gname]
                if Q.size == 0:
                    continue
                for j, gk in enumerate(keys):
                    if (gk, REF) not in cells:
                        continue
                    block, _G = _fragment_matrix(cells[(k, view)], cells[(gk, REF)], gname, sig)
                    M[i, j] = _coverage(block)
            order = np.argsort(-M, axis=1, kind="stable")
            ranks = np.asarray([int(np.nonzero(order[i] == i)[0][0]) for i in range(len(keys))])
            fit = split["FIT"]
            res[cell] = {"recall@8": float((ranks < S).mean()),
                         "top1": float((ranks == 0).mean()),
                         "truth_rank_median": float(np.median(ranks)),
                         "share_ge_0.99": float((M[~np.eye(len(keys), dtype=bool)] >= 0.99).mean()),
                         "per_class": {c: float(np.mean([ranks[i] < S for i, o in enumerate(objs)
                                                         if o["cls"] == c]))
                                       for c in sorted({o["cls"] for o in objs})},
                         "fit_recall8": float(np.mean([ranks[i] < S for i in fit]))}
            out.setdefault("_matrices", {})["%s|%s" % (gname, cell)] = M
            print("  %-16s %-6s r@8 %.3f top1 %.3f rank_med %.1f share99 %.3f"
                  % (gname, cell, res[cell]["recall@8"], res[cell]["top1"],
                     res[cell]["truth_rank_median"], res[cell]["share_ge_0.99"]), flush=True)
        out["groupings"][gname] = res
    # (b) the oracle on M31's same-class losses
    out["oracle"] = _oracle(out, cells.keys(), cells, keys, split)
    out.pop("_matrices", None)
    with open(OUT.replace("m34_fragments.json", "m34_screen.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    return out


def _oracle(out: Dict[str, Any], _keys: Any, cells: Dict[Tuple[str, str], Any],
            keys: Sequence[str], split: Dict[str, List[int]]) -> Dict[str, Any]:
    """Of M31's same-class losses, in how many does the fragment channel rank truth > winner?"""
    from matching import placement_records as PR
    if not os.path.exists(PAIRS):
        return {"error": "no cached pairs"}
    want = {}
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        if r.get("score") is None:
            continue
        want.setdefault(r["object"], []).append((PR.object_id(r["r"]), float(r["score"])))
    cls_of = {o["key"]: o["cls"] for o in REC.objects()}
    res: Dict[str, Any] = {}
    for gname in out["groupings"]:
        if "error" in out["groupings"][gname]:
            continue
        M = out.get("_matrices", {}).get("%s|fit_a" % gname)
        if M is None:
            continue
        idx = {k: i for i, k in enumerate(keys)}
        n_same = n_beat = 0
        for q, rows in want.items():
            if q not in idx:
                continue
            rows = sorted(rows, key=lambda t: -t[1])
            winner = rows[0][0]
            if winner == q or cls_of.get(winner) != cls_of.get(q):
                continue                                  # other-class losses are not this oracle
            n_same += 1
            if float(M[idx[q], idx[q]]) > float(M[idx[q], idx[winner]]):
                n_beat += 1
        res[gname] = {"same_class_losses": n_same, "truth_above_winner": n_beat,
                      "colour_reference": "%d/30" % COLOUR_LOSSES, "bar": ">= %d/31" % BAR_B}
        print("  oracle %-16s %d/%d" % (gname, n_beat, n_same), flush=True)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.probe:
        probe()
        return 0
    if a.build:
        print(json.dumps(build(), indent=1))
        return 0
    if a.report:
        report()
        return 0
    if a.screen:
        screen()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
