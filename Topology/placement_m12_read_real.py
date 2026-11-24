"""Mission 12 step 5 -- the pre-registered REAL-PHOTO correspondence read.

The registration is `Topology/results/placement/m12_prereg_real.json`, written
before any graph for these photographs existed.  This script executes it and
nothing else: it reads the registration, refuses if it is missing, refuses if a
read graph is already cached (a second look at the same graphs is not a read),
and applies the frozen analysis.

WHAT WAS FIXED BEFORE THE FIRST GRAPH EXISTED (all of it is the registration's;
the two implementation details the registration leaves open are declared here,
also before the run):

* arms N (placement off) and P (deployed placement matcher, w = 0.375 read from
  `matching/placement_score_params.json` exactly as registered, E normalised,
  evidence on);
* transforms `rot05`, `rot15`, `scale115`, `trans8`, applied to the photograph
  exactly as the registration applied them when it hashed them;
* truth = the EXACT transform's inverse applied to the query node's anchor,
  snapped to the nearest reference node anchor within `TRUTH_TOL` px, made
  INJECTIVE (a reference node claimed by two query nodes releases both), and
  only for anchors at least `BORDER` px inside both images;
* the near-tie bank = `placement_gate.appearance_ambiguous` on the node
  similarity rows (top >= 0.99 and margin <= 0.05), restricted to nodes with
  truth; the unambiguous bank is the rest of the truth set;
* **declared scoring**: a bank node counts as CORRECT iff the arm's mapping maps
  it onto its true counterpart.  An UNMAPPED bank node counts as a MISS in the
  primary number (`acc_bank`), because the read asks how often the matcher
  answers correctly, and "no answer" is not a correct answer; the accuracy over
  the answered nodes (`acc_judged`) is reported beside it, not instead of it;
* primary contrast: P - N on `acc_bank` over the near-tie bank, paired per query,
  **photo-clustered** bootstrap (4000 draws, 2.5/97.5);
* no-harm: the lower end of the photo-clustered interval for P - N on the
  unambiguous bank must not fall below -0.02;
* verdict: DETECTED iff the primary interval excludes zero in P's favour; a
  detected effect below the realised MDE is labelled possibly inflated.  One
  read.  The realised bank size and realised MDE are reported with it.

    python Topology/placement_m12_read_real.py --start   --workers 2
    python Topology/placement_m12_read_real.py --report
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = "/home/user/Percept6"
sys.path[:0] = [ROOT, os.path.join(ROOT, "Topology")]
OUT = os.path.join(ROOT, "Topology", "results", "placement")
REG = os.path.join(OUT, "m12_prereg_real.json")
PARAMS = os.path.join(ROOT, "matching", "placement_score_params.json")
CACHE_DIR = os.path.join(ROOT, ".cache", "placement_graphs")
PROGRESS = os.path.join(OUT, "m12_read_real_progress.jsonl")
RESULT = os.path.join(OUT, "m12_read_real.json")

TRUTH_TOL = 2.0          # px, declared before the run
BORDER = 8               # px, same border the payload discards
NBOOT = 4000
SEED = 20261118


def _registration() -> Dict[str, Any]:
    if not os.path.exists(REG):
        raise SystemExit("REFUSING: no pre-registration at %s" % REG)
    with open(REG) as fh:
        return json.load(fh)


def _transform(img: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """The registration's own transform, verbatim."""
    from scipy.ndimage import rotate, zoom, shift as nd_shift
    a = img.astype(np.float64)
    if "rot_deg" in spec:
        a = rotate(a, float(spec["rot_deg"]), reshape=False, order=1, mode="nearest")
    if "scale" in spec:
        s = float(spec["scale"])
        a = zoom(a, (s, s, 1.0), order=1)
        h, w = a.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y0, x0 = int(round(cy - img.shape[0] / 2.0)), int(round(cx - img.shape[1] / 2.0))
        a = a[max(0, y0):y0 + img.shape[0], max(0, x0):x0 + img.shape[1]]
        a = np.pad(a, ((0, img.shape[0] - a.shape[0]),
                       (0, img.shape[1] - a.shape[1]), (0, 0)), mode="edge")
    if "shift" in spec:
        a = nd_shift(a, (float(k) for k in (spec["shift"][1], spec["shift"][0], 0.0)),
                     order=1, mode="nearest")
    return np.clip(a, 0, 255).astype(np.uint8)


def _inverse(spec: Dict[str, Any], pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Query-image pixels -> original-image pixels, for the registration's spec.

    The rotation convention is the one the Step-2 frame test CALIBRATED: with the
    image rotated by `+delta`, an anchor at p in the original sits at
    `c + R(-delta)(p - c)` in the rotated image, so the inverse is
    `p = c + R(+delta)(q - c)`.
    """
    h, w = shape
    c = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
    p = np.asarray(pts, dtype=float)
    if "rot_deg" in spec:
        th = np.deg2rad(float(spec["rot_deg"]))
        ct, st = float(np.cos(th)), float(np.sin(th))
        d = p - c
        p = np.stack([ct * d[:, 0] - st * d[:, 1], st * d[:, 0] + ct * d[:, 1]],
                     axis=1) + c
    if "scale" in spec:
        s = float(spec["scale"])
        y0 = int(round((s * h) / 2.0 - h / 2.0))
        x0 = int(round((s * w) / 2.0 - w / 2.0))
        p = np.stack([(p[:, 0] + x0) / s, (p[:, 1] + y0) / s], axis=1)
    if "shift" in spec:
        p = np.stack([p[:, 0] - float(spec["shift"][0]),
                      p[:, 1] - float(spec["shift"][1])], axis=1)
    return p


def _anchors(g: Any) -> np.ndarray:
    sub = g.subgraph
    return np.asarray([(float(sub.anchor_of_node[n][0]), float(sub.anchor_of_node[n][1]))
                       for n in sorted(sub.nodes) if n in sub.anchor_of_node], dtype=float)


def _truth(gq: Any, gr: Any, spec: Dict[str, Any],
           shape: Tuple[int, int]) -> Dict[int, int]:
    """query node -> the reference node the exact transform says it is."""
    subq, subr = gq.subgraph, gr.subgraph
    qn = [int(n) for n in sorted(subq.nodes) if n in subq.anchor_of_node]
    rn = [int(n) for n in sorted(subr.nodes) if n in subr.anchor_of_node]
    if not qn or not rn:
        return {}
    qa = np.asarray([subq.anchor_of_node[n] for n in qn], dtype=float)
    ra = np.asarray([subr.anchor_of_node[n] for n in rn], dtype=float)
    back = _inverse(spec, qa, shape)
    h, w = shape
    cand: List[Tuple[float, int, int]] = []
    for i, (x, y) in enumerate(back):
        if not (BORDER <= x < w - BORDER and BORDER <= y < h - BORDER):
            continue
        d = np.hypot(ra[:, 0] - x, ra[:, 1] - y)
        j = int(np.argmin(d))
        if float(d[j]) <= TRUTH_TOL:
            cand.append((float(d[j]), int(qn[i]), int(rn[j])))
    cand.sort()
    used_r: Dict[int, int] = {}
    for _d, u, v in cand:
        used_r.setdefault(v, 0)
        used_r[v] += 1
    return {u: v for _d, u, v in cand if used_r[v] == 1}


def _graph_job(args: Tuple[str, str]) -> Tuple[str, Any]:
    import guard
    guard.activate()
    from Topology.placement_cache import get_or_build
    key, img = args
    return key, get_or_build(img, key)


_IMG: Dict[str, np.ndarray] = {}


def _images() -> Dict[str, np.ndarray]:
    """every image the read needs, by the cache key it will be stored under."""
    if _IMG:
        return _IMG
    from Topology import placement_m12_realphotos as RP
    reg = _registration()
    photos = list(reg["photos"]["names"])
    imgs = RP.images("read")
    out: Dict[str, np.ndarray] = {}
    for name in photos:
        out["m12real|%s|orig" % name] = imgs[name]
        for t in reg["transforms"]:
            out["m12real|%s|%s" % (name, t["label"])] = _transform(imgs[name], t["spec"])
    _IMG.update(out)
    return _IMG


def _freshness() -> int:
    """How many of the read's graphs are already in the cache (must be 0)."""
    from Topology.placement_cache import key_for
    imgs = _images()
    n = 0
    for k, img in imgs.items():
        if os.path.exists(os.path.join(CACHE_DIR, key_for(img) + ".pkl")):
            n += 1
    return n


def _pair_job(job: Tuple[str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """One (photo, transform): truth, the two banks, both arms."""
    name, label, spec_json, w = job
    import guard
    guard.activate()
    import parameters as P
    from matching.joint_mcs import joint_mcs
    from Topology.placement_cache import get_or_build
    from Topology import placement_census as CEN
    from Topology.placement_gate import appearance_ambiguous

    spec = json.loads(spec_json)
    imgs = _images()
    kq = "m12real|%s|%s" % (name, label)
    kr = "m12real|%s|orig" % name
    q_img, r_img = imgs[kq], imgs[kr]
    q = get_or_build(q_img, kq)
    r = get_or_build(r_img, kr)
    truth = _truth(q, r, spec, (r_img.shape[0], r_img.shape[1]))
    o = sorted(truth)
    rows = CEN.similarity_rows(q, r, o, sorted(int(n) for n in r.subgraph.nodes))
    amb = appearance_ambiguous(q.sigs, r.sigs, rows)

    P.PLACEMENT_NORMALISE = True
    P.PLACEMENT_E_NORMALISED = True
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0
    n_res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs)
    P.PLACEMENT_MATCHER_ENABLED = True
    P.PLACEMENT_SCORE_EVIDENCE = True
    P.PLACEMENT_SCORE_WEIGHT = float(w)
    p_res = joint_mcs(q.subgraph, r.subgraph, q.sigs, r.sigs,
                      placement_images=(q_img, r_img))
    P.PLACEMENT_MATCHER_ENABLED = False
    P.PLACEMENT_SCORE_EVIDENCE = False
    P.PLACEMENT_SCORE_WEIGHT = 0.0

    def _score(res: Any) -> Dict[str, Any]:
        an = ah = aj = uh = un = 0
        for u in o:
            v = truth[u]
            hit = int(res.mapping.get(int(u), -1) == v)
            if amb.get(int(u), False):
                ah += hit
                an += 1
                aj += int(int(u) in res.mapping)
            else:
                uh += hit
                un += 1
        return {"amb_hit": ah, "amb_n": an, "amb_judged": aj,
                "un_hit": uh, "un_n": un}

    return {"photo": name, "transform": label,
            "w": float(w),
            "truth_n": len(truth),
            "amb_n": int(sum(1 for u in o if amb.get(int(u), False))),
            "N": _score(n_res), "P": _score(p_res),
            "E": float(p_res.placement_evidence),
            "judged": int(p_res.placement_judged),
            "share": float(p_res.placement_share_judged),
            "sec": None}


def _boot(diffs: List[float], clusters: List[str]) -> Dict[str, Any]:
    by: Dict[str, List[float]] = {}
    for d, c in zip(diffs, clusters):
        by.setdefault(c, []).append(d)
    keys = sorted(by)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(NBOOT):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals = [v for k in pick for v in by[k]]
        draws.append(float(np.mean(vals)))
    draws.sort()
    return {"mean": float(np.mean(diffs)) if diffs else float("nan"),
            "n": len(diffs), "clusters": len(keys),
            "lo": float(draws[int(0.025 * len(draws))]) if draws else float("nan"),
            "hi": float(draws[int(0.975 * len(draws)) - 1]) if draws else float("nan")}


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
    reg = _registration()
    with open(PARAMS) as fh:
        w = float(json.load(fh)["w"])
    if abs(w - float(reg["frozen"]["w"])) > 1e-12:
        print("REFUSING: params w=%.6f but the registration froze %.6f"
              % (w, float(reg["frozen"]["w"])))
        return 3

    if args.start:
        if os.path.exists(RESULT):
            print("REFUSING: a completed read already exists at %s" % RESULT)
            return 3
        n = _freshness()
        # The registration's freshness rule is about the graphs that existed WHEN
        # IT WAS SIGNED (0, checked then).  Here a nonzero count means this read
        # already built some of its own graphs -- a resumed run, which the
        # append-only progress file makes auditable -- so it is reported, not
        # treated as a second look; a COMPLETED result is what refuses.
        print("read graphs already cached (resumed run): %d" % n, flush=True)
        jobs = [(p, t["label"], json.dumps(t["spec"]), w)
                for p in reg["photos"]["names"] for t in reg["transforms"]]
        done: List[Dict[str, Any]] = []
        if os.path.exists(PROGRESS):
            with open(PROGRESS) as fh:
                for line in fh:
                    try:
                        done.append(json.loads(line))
                    except Exception:
                        continue
        seen = {(d["photo"], d["transform"]) for d in done}
        todo = [j for j in jobs if (j[0], j[1]) not in seen]
        print("%d queries, %d to go" % (len(jobs), len(todo)), flush=True)
        t0 = time.time()
        with open(PROGRESS, "a") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_pair_job, todo):
                done.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print("  %-12s %-9s truth %4d amb %3d  N %3d/%3d  P %3d/%3d"
                      % (r["photo"], r["transform"], r["truth_n"], r["amb_n"],
                         r["N"]["amb_hit"], r["N"]["amb_n"],
                         r["P"]["amb_hit"], r["P"]["amb_n"]), flush=True)
        print("queries done in %.0fs" % (time.time() - t0), flush=True)

    if not os.path.exists(PROGRESS):
        print("no checkpoint")
        return 1
    rows: List[Dict[str, Any]] = []
    with open(PROGRESS) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # ---- the frozen analysis ------------------------------------------------
    amb_n = sum(r["amb_n"] for r in rows)
    res: Dict[str, Any] = {"queries": len(rows), "w": w,
                           "registered": {"near_tie": reg["frozen"]["near_tie_rule"],
                                          "primary": reg["primary"],
                                          "no_harm": reg["no_harm"]}}
    for bank, hot, tot in (("near_tie", "amb_hit", "amb_n"),
                           ("unambiguous", "un_hit", "un_n")):
        diffs = []
        clusters = []
        acc_n = acc_p = 0.0
        bank_n = 0
        for r in rows:
            if not r["N"][tot]:
                continue
            a = r["N"][hot] / r["N"][tot]
            b = r["P"][hot] / r["P"][tot]
            diffs.append(b - a)
            clusters.append(r["photo"])
            acc_n += r["N"][hot]
            acc_p += r["P"][hot]
            bank_n += r["N"][tot]
        res[bank] = {"bank_nodes": bank_n,
                     "acc_N": (acc_n / bank_n) if bank_n else float("nan"),
                     "acc_P": (acc_p / bank_n) if bank_n else float("nan"),
                     "paired": _boot(diffs, clusters)}
    md = 2.8 * 0.5 * (2.0 ** 0.5) / (max(1, res["near_tie"]["bank_nodes"]) ** 0.5)
    res["mde"] = {"planning_mde": float(md),
                  "formula": "2.8 * 0.5 * sqrt(2.0) / sqrt(n_bank)",
                  "n_bank": res["near_tie"]["bank_nodes"]}
    pr = res["near_tie"]["paired"]
    res["verdict"] = {
        "detected": bool(pr["lo"] > 0.0),
        "possibly_inflated": bool(pr["lo"] > 0.0 and pr["mean"] < md),
        "primary": "P - N on near-tie correspondence accuracy",
        "interval": [pr["lo"], pr["hi"]],
        "mean": pr["mean"]}
    nh = res["unambiguous"]["paired"]
    res["no_harm"] = {"ok": bool(nh["lo"] >= -0.02), "lo": nh["lo"],
                      "rule": "photo-clustered lower end of P - N on the unambiguous bank >= -0.02"}
    res["frame_chain"] = {"mean_judged": float(np.mean([r["judged"] for r in rows])),
                          "mean_evidence": float(np.mean([r["E"] for r in rows])),
                          "mean_share": float(np.mean([r["share"] for r in rows]))}
    res["per_photo"] = _per_photo(rows)
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("queries", "near_tie", "unambiguous",
                                          "mde", "verdict", "no_harm", "frame_chain")},
                     indent=1)[:2500])
    print("wrote %s" % RESULT)
    return 0


def _per_photo(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in rows:
        e = out.setdefault(r["photo"], {"near_tie_nodes": 0, "N": 0, "P": 0})
        e["near_tie_nodes"] += r["amb_n"]
        e["N"] += r["N"]["amb_hit"]
        e["P"] += r["P"]["amb_hit"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
