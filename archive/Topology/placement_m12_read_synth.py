"""MISSION 12 STEP 6a -- the fresh synthetic read, executed against its registration.

The registration is `Topology/results/placement/m12_prereg_synth.json` and it was
written before any graph existed for these scenes (it records
`graphs_in_cache_at_registration: 0`).  This script is the read and nothing else:
it refuses to run without the registration, uses exactly the registered scenes,
views, shortlist size and arms, and reports the registered primary contrast with the
registered interval.

    python Topology/placement_m12_read_synth.py --start    # resumable
    python Topology/placement_m12_read_synth.py --report
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
REG = os.path.join(OUT, "m12_prereg_synth.json")
PROGRESS = os.path.join(OUT, "m12_read_synth_progress.jsonl")
RESULT = os.path.join(OUT, "m12_read_synth.json")
IMAGES = os.path.join(ROOT, ".cache", "m12_read_images.npz")
NBOOT, SEED = 4000, 20261118


FIT_RESULT = os.path.join(OUT, "m12_fit_w.json")


def _gate_check() -> Dict[str, Any]:
    """[MISSION 12 / step 4] The read runs ONLY on a sane fit.

    The fit writes `gates.passed`; this refuses the read if the file is missing,
    if the gates failed, or if the gates are absent (no fit, no read).
    """
    if not os.path.exists(FIT_RESULT):
        raise SystemExit("REFUSING: no fit result at %s -- the read needs a gated fit"
                         % FIT_RESULT)
    fit = json.load(open(FIT_RESULT))
    g = fit.get("gates") or {}
    if not g.get("passed"):
        raise SystemExit("REFUSING: the fit's sanity gates did not pass: %s"
                         % json.dumps({k: v for k, v in g.items() if k != "declared"},
                                      sort_keys=True))
    return {"gates": {k: v for k, v in g.items() if k != "declared"},
            "T": (fit.get("fit") or {}).get("T"), "w": (fit.get("fit") or {}).get("w")}


def _setup() -> Tuple[Dict[str, Any], List[Any], List[str], Dict[str, List[str]]]:
    """(registration, fresh specs, query keys, shortlists) -- truth free."""
    from Topology import placement_m12_prereg_synth as PR
    from Topology import placement_families_m9 as FM
    reg = json.load(open(REG))
    specs = PR._fresh_specs(FM)
    if [s.name for s in specs] != reg["freshness"]["scenes"]:
        raise SystemExit("the fresh scenes do not match the registration")
    views = {k: v for k, v in PR.VIEWS.items() if k in set(reg["queries"]["views"])}
    qkeys = ["%s|%s" % (s.name, lab) for s in specs for lab in sorted(views)]
    if len(qkeys) != reg["queries"]["n"]:
        raise SystemExit("query count %d != registered %d" % (len(qkeys), reg["queries"]["n"]))
    return reg, specs, qkeys, views


def prewarm(specs: List[Any], qkeys: List[str], views: Dict[str, Any]) -> None:
    if os.path.exists(IMAGES):
        return
    import guard
    guard.activate()
    from Topology import placement_families_m9 as FM
    by = {s.name: s for s in specs}
    out: Dict[str, Any] = {}
    for s in specs:
        out[s.name + "|ref"] = s.scene.render().image_u8
    for qk in qkeys:
        name, lab = qk.split("|")
        sc, _pv = FM.pose_fn(by[name], views[lab])
        out[qk] = sc.render().image_u8
    os.makedirs(os.path.dirname(IMAGES), exist_ok=True)
    np.savez_compressed(IMAGES, **out)
    print("prewarmed %d images" % len(out), flush=True)


_IMGS: Dict[str, Any] = {}


def _imgs() -> Any:
    if not _IMGS:
        with np.load(IMAGES) as z:
            for k in z.files:
                _IMGS[k] = z[k]
    return _IMGS


_REF: Dict[str, Any] = {}


def _shortlist_job(qk: str) -> Dict[str, Any]:
    """The K=8 shortlist by node similarity -- computed by CONSTRUCTION, not truth."""
    import guard
    guard.activate()
    from Topology import placement_census as CEN
    from Topology.placement_cache import get_or_build
    imgs = _imgs()
    q = get_or_build(imgs[qk], "m12read|" + qk)
    rows_out: Dict[str, float] = {}
    qnodes = sorted(int(n) for n in q.subgraph.nodes)
    for name in sorted(_REFS()):
        g = _ref(name)
        rows = CEN.similarity_rows(q, g, qnodes, sorted(int(n) for n in g.subgraph.nodes))
        best = [max((s for (_v, s) in rows.get(u, [])), default=0.0) for u in qnodes]
        rows_out[name] = float(np.mean(best)) if best else 0.0
    order = sorted(rows_out, key=lambda c: (-rows_out[c], c))[:8]
    return {"q": qk, "top": order}


_REFS_CACHE: List[str] = []


def _REFS() -> List[str]:
    """The fresh reference names, filled from the registration's scene list."""
    return _REFS_CACHE


def _ref(name: str) -> Any:
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    if name not in _REF:
        _REF[name] = get_or_build(_imgs()[name + "|ref"], "m12read|" + name)
    return _REF[name]


def _pair_job(job: Tuple[str, str, float]) -> Dict[str, Any]:
    """One (query, candidate): arm A, arm B/C, and the evidence."""
    qk, rname, w = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    imgs = _imgs()
    q = _ref(qk.split("|")[0])
    r = _ref(rname)
    q_img, r_img = imgs[qk], imgs[rname + "|ref"]
    P.PLACEMENT_E_NORMALISED = True
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    a = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs)
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    b = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                  placement_images=(q_img, r_img))
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    return {"q": qk, "cand": rname,
            "A": float(a.score), "B": float(b.score_shipped),
            "C": float(b.score_shipped) + float(w) * float(b.placement_evidence),
            "E": float(b.placement_evidence), "judged": int(b.placement_judged),
            "share": float(b.placement_share_judged)}


def _ranks(rows: List[Dict[str, Any]], arm: str) -> Dict[str, int]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    out: Dict[str, int] = {}
    for q, cands in by.items():
        truth = q.split("|")[0]
        if truth not in [c["cand"] for c in cands]:
            continue
        order = sorted(cands, key=lambda c: (-c[arm], c["cand"]))
        out[q] = [c["cand"] for c in order].index(truth) + 1
    return out


def _paired_ci(diffs: List[float], scenes: List[str]) -> Dict[str, Any]:
    """Scene-clustered bootstrap of the mean paired difference."""
    if not diffs:
        return {"mean": float("nan"), "n": 0}
    by: Dict[str, List[float]] = {}
    for d, s in zip(diffs, scenes):
        by.setdefault(s, []).append(d)
    keys = sorted(by)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(NBOOT):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals = [v for k in pick for v in by[k]]
        draws.append(float(np.mean(vals)))
    draws.sort()
    return {"mean": float(np.mean(diffs)), "n": len(diffs),
            "lo": float(draws[int(0.025 * len(draws))]),
            "hi": float(draws[int(0.975 * len(draws)) - 1]),
            "clusters": len(keys)}


def main() -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    import guard
    guard.activate()
    reg, specs, qkeys, views = _setup()
    if args.start:
        _gate_check()
    _REFS_CACHE[:] = [s.name for s in specs]
    w = float(json.load(open(os.path.join(ROOT, "matching",
                                         "placement_score_params.json")))["w"])

    if args.start:
        prewarm(specs, qkeys, views)
        sl_path = os.path.join(OUT, "m12_read_synth_shortlists.json")
        if os.path.exists(sl_path):
            shorts = json.load(open(sl_path))
        else:
            print("computing %d shortlists" % len(qkeys), flush=True)
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
                shorts = {r["q"]: r["top"] for r in ex.map(_shortlist_job, qkeys)}
            with open(sl_path, "w") as fh:
                json.dump(shorts, fh, indent=1, sort_keys=True)
            print("shortlists done", flush=True)
        jobs = [(qk, cand, w) for qk in qkeys for cand in shorts[qk]]
        done: List[Dict[str, Any]] = []
        if os.path.exists(PROGRESS):
            with open(PROGRESS) as fh:
                for line in fh:
                    try:
                        done.append(json.loads(line))
                    except Exception:
                        continue
        seen = {(d["q"], d["cand"]) for d in done}
        todo = [j for j in jobs if (j[0], j[1]) not in seen]
        print("%d pairs, %d to go" % (len(jobs), len(todo)), flush=True)
        t0 = time.time()
        with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_pair_job, todo, chunksize=2):
                done.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
        print("pairs done in %.0fs" % (time.time() - t0), flush=True)

    if not os.path.exists(PROGRESS):
        print("no checkpoint")
        return 1
    rows = []
    with open(PROGRESS) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    ra, rb, rc = _ranks(rows, "A"), _ranks(rows, "B"), _ranks(rows, "C")
    qs = sorted(set(ra) & set(rb) & set(rc))
    scenes = [q.split("|")[0] for q in qs]
    fam = ["oop" if "oop" in s else "inplane" for s in scenes]
    d_ca = [float((rc[q] == 1) - (ra[q] == 1)) for q in qs]
    d_ba = [float((rb[q] == 1) - (ra[q] == 1)) for q in qs]
    res = {
        "queries": len(qs), "registered": reg["queries"]["n"],
        "top1": {"A": float(np.mean([ra[q] == 1 for q in qs])),
                 "B": float(np.mean([rb[q] == 1 for q in qs])),
                 "C": float(np.mean([rc[q] == 1 for q in qs]))},
        "mrr": {k: float(np.mean([1.0 / v[q] for q in qs]))
                for k, v in (("A", ra), ("B", rb), ("C", rc))},
        "primary_C_minus_A": _paired_ci(d_ca, scenes),
        "secondary_B_minus_A": _paired_ci(d_ba, scenes),
        "by_family": {},
        "chance": reg["gallery"]["chance"],
        "mde": reg["mde"],
        "judged_share": float(np.mean([r["share"] for r in rows])) if rows else 0.0,
        "judged_mean": float(np.mean([r["judged"] for r in rows])) if rows else 0.0,
        "w": w,
    }
    for f in ("inplane", "oop"):
        idx = [i for i, x in enumerate(fam) if x == f]
        if not idx:
            continue
        res["by_family"][f] = {
            "n": len(idx),
            "top1": {k: float(np.mean([v[qs[i]] == 1 for i in idx]))
                     for k, v in (("A", ra), ("B", rb), ("C", rc))},
            "C_minus_A": _paired_ci([d_ca[i] for i in idx], [scenes[i] for i in idx]),
        }
    p = res["primary_C_minus_A"]
    res["verdict"] = ("DETECTED" if (p.get("lo", 1) > 0.0) else "NOT DETECTED")
    res["type_M"] = bool(res["verdict"] == "DETECTED"
                         and p.get("mean", 0.0) < reg["mde"]["mde"])
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print(json.dumps(res, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
