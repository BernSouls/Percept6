"""Task 4 RETRIEVAL — C1 jitter control on arm N.

Relabel query+ref graphs with a random node-id/order permutation (the 0e
mechanism) and re-measure N's top-1 / AUC on a subset of queries.  Reports the
drop vs the un-permuted N, as a tie-order robustness envelope.
"""
from __future__ import annotations
import json, os, random, sys
from dataclasses import replace
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
import guard  # noqa: E402
guard.activate()
from guard import RESULTS_DIR, write_json  # noqa: E402
from t4_retrieval_measure import (_arm_graph, _run_matcher, _auc, _boot, _paired,
                                  WINNER, BETA, QUERY_VIEWS, CLASSES, REF_VIEW)
from t4_part0e_order import _relabel
import corpus
from frontend import build_cached, load


def _load(cls, obj, view):
    inst = corpus._instances(cls)
    e = corpus._entry(cls, obj, view, inst[obj][view])
    return load(build_cached(e)["cache"])


def _score_n(q_arms, r_arms, nq_refs):
    sc = [ _run_matcher(q_arms, r_arms)[0] for _ in range(0) ]  # placeholder
    return sc


def main() -> int:
    print("=== C1: jitter control on N (16 queries) ===", flush=True)
    refs = []
    for cls in CLASSES:
        inst = corpus._instances(cls); ids = sorted(inst); o = ids[0]
        g = _load(cls, o, REF_VIEW)
        refs.append({"class": cls, "object": o, "graph": g,
                     "arms": {"N": _arm_graph(g, "N")}})
    # 16 queries: 2 views x 8 classes
    queries = []
    for cls in CLASSES:
        inst = corpus._instances(cls); ids = sorted(inst); o = ids[0]
        for v in ("-022-000", "-090-135"):
            g = _load(cls, o, v)
            queries.append({"class": cls, "object": o, "view": v, "graph": g,
                            "arms": {"N": _arm_graph(g, "N")}})
    print("refs=%d queries=%d" % (len(refs), len(queries)), flush=True)

    def measure(qs, rs, label):
        to1, auc = [], []
        for q in qs:
            sc = []
            for r in rs:
                s, _ = _run_matcher(q["arms"]["N"], r["arms"]["N"])
                sc.append(s)
            labels = [1 if (r["class"] == q["class"] and
                            r["object"] == q["object"]) else 0 for r in rs]
            ci = labels.index(1)
            rank = 1 + sum(1 for s in sc if s > sc[ci])
            to1.append(1.0 if rank == 1 else 0.0)
            auc.append(_auc(sc, labels))
        print("  %s: top1=%s auc=%s" % (label, _boot(to1), _boot(
            [x for x in auc if x is not None])), flush=True)
        return to1, auc

    t1_orig, auc_orig = measure(queries, refs, "N original")
    # permute
    seeds = [42, 7]
    for seed in seeds:
        qs_p, rs_p = [], []
        for q in queries:
            nodes = sorted(set(q["graph"].subgraph.nodes))
            rng = random.Random(seed)
            perm = {n: i for i, n in enumerate(rng.sample(nodes, len(nodes)))}
            gq, _ = _relabel(q["graph"], perm)
            qs_p.append({"class": q["class"], "object": q["object"], "graph": gq,
                         "arms": {"N": _arm_graph(gq, "N")}})
        for r in refs:
            nodes = sorted(set(r["graph"].subgraph.nodes)
                           | set(queries[0]["graph"].subgraph.nodes))
            rng = random.Random(seed)
            perm = {n: i for i, n in enumerate(rng.sample(nodes, len(nodes)))}
            gr, _ = _relabel(r["graph"], perm)
            rs_p.append({"class": r["class"], "object": r["object"], "graph": gr,
                         "arms": {"N": _arm_graph(gr, "N")}})
        t1_p, auc_p = measure(qs_p, rs_p, "N permuted seed=%d" % seed)
        write_json(RESULTS_DIR / "t4" / "retrieval_c1_jitter.json", {
            "seed": seed,
            "top1_orig": _boot(t1_orig), "top1_perm": _boot(t1_p),
            "top1_drop": _paired(t1_orig, t1_p),
            "auc_orig": _boot([x for x in auc_orig if x is not None]),
            "auc_perm": _boot([x for x in auc_p if x is not None]),
            "auc_drop": _paired([x for x in auc_orig if x is not None],
                                [x for x in auc_p if x is not None]),
        })
    print("=== C1 complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
