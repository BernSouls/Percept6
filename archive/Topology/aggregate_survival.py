"""Aggregate the survival_table results into the per-scene tables."""
import json
import statistics
import sys
from pathlib import Path

SRC = Path(r"E:\Take25\work\topology\results\survival_table.json")


def m(v):
    return "%.3f" % v if isinstance(v, (int, float)) else str(v)


def row(r):
    jm = r["matchers"].get("joint_mcs", {})
    rw = r["matchers"].get("relation_walker", {})
    rwt = r["matchers"].get("relation_walker_tokens", {})
    ce = jm.get("rel_common_eligible", 0)
    return {
        "scene": r["scene"], "mode": r["mode"], "shift": r["shift"][0],
        "expected": r["expected_nodes"], "truth": r["truth_nodes"],
        "survival": r["survival"], "hun": r["truth_hungarian"],
        "dlt": r["truth_delta"],
        "rel_t": r["rel_true_frac"],
        "jm_rec": jm.get("recall"), "jm_precS": jm.get("prec_surv"), "jm_precA": jm.get("prec_all"),
        "jm_rel": jm.get("rel_frac"), "jm_common": jm.get("rel_common_matcher_preserved"),
        "rw_rec": rw.get("recall"), "rwt_rec": rwt.get("recall"),
        "rw_common": rw.get("rel_common_matcher_preserved"),
        "true_common": jm.get("rel_common_true_preserved"),
        "common_elig": ce,
        "tie0": jm.get("zero_px_same_signature_errors"),
        "grid_pitch": r["grid"]["best_grid_pitch"], "grid_chime": r["grid"]["best_grid_chime"],
        "int_x": r["grid"]["integral_x"], "int_y": r["grid"]["integral_y"], "pos": r["grid"]["positions"],
    }


def main():
    d = json.loads(SRC.read_text())
    rows = [row(r) for r in d["rows"]]
    order = [(r["scene"], r["mode"], r["shift"]) for r in rows]

    hdr = ("%-20s %-6s %3s | %3s %-5s %-6s | %-6s %-6s %-6s %-6s %-6s | %-6s %-6s | %-4s %-4s" % (
        "scene", "mode", "sh", "exp", "surv", "hun", "relTr", "recall", "precS", "precA", "relJM",
        "rwRec", "rwtRec", "cmnJM", "cmnTr"))
    print("Joint_mcs matches on 0-row; RW = default walker; RWT = D1 token arm (diagnostic):")
    print(hdr)
    for r in rows:
        if r["shift"] == 0:
            continue
        rw = r  # shorthand, read from raw rows below
        print("%-20s %-6s %+3d | %3d %-5.3f %-6d | %-6.3f %-6.3f %-6.3f %-6.3f %-6.3f | %-6s %-6s | %-4s %-4s" % (
            r["scene"], r["mode"], r["shift"], r["expected"], r["survival"], r["hun"],
            r["rel_t"], r["jm_rec"], r["jm_precS"], r["jm_precA"], r["jm_rel"],
            m(r["rw_rec"]), m(r["rwt_rec"]), m(r["jm_common"]), m(r["true_common"])))

    print("\nAGGREGATE by mode/shift (survival / JM recall / prec@surv / prec_all / relJM-frac — means over scenes):")
    for mode in ("full", "object"):
        for shift in (2, 3):
            rs = [r for r in rows if r["mode"] == mode and r["shift"] == shift]
            print("  %-6s %+d  surv %.3f  recall %.3f  precS %.3f  precA %.3f  relJM %.3f  true-rel %.3f" % (
                mode, shift,
                statistics.mean(r["survival"] for r in rs),
                statistics.mean(r["jm_rec"] for r in rs),
                statistics.mean(r["jm_precS"] for r in rs),
                statistics.mean(r["jm_precA"] for r in rs),
                statistics.mean(r["jm_rel"] for r in rs),
                statistics.mean(r["rel_t"] for r in rs)))

    print("\nGRID / ANCHOR-SNAPPING DIAGNOSTICS (per scene, full frame, 0px graph):")
    for r in rows:
        if r["mode"] == "full" and r["shift"] == 0:
            print("  %-20s positions %4d  integral-x %4d/%4d  integral-y %4d/%4d  best grid pitch %2d (chime %.3f)" % (
                r["scene"], r["pos"], r["int_x"], r["pos"], r["int_y"], r["pos"],
                r["grid_pitch"], r["grid_chime"]))

    print("\n0px INSTRUMENT CHECKS:")
    for r in rows:
        if r["shift"] != 0:
            continue
        # rel preservation under truth must be total
        ok = r["rel_t"] >= 1.0 and r["survival"] == 1.0
        print("  %-20s %-6s surv %.1f relT %.3f JM rec %.3f (tie-errors %d) %s" % (
            r["scene"], r["mode"], r["survival"], r["rel_t"], r["jm_rec"], r["tie0"] or 0,
            "OK" if ok else "BROKEN"))


if __name__ == "__main__":
    main()
