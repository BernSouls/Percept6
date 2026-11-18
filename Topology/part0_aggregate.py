"""Aggregate Part 0 results into the tables for the report."""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(r"E:\Take25\work\topology\results\part0_ruler.json")


def load():
    return json.loads(SRC.read_text())["rows"]


def fmt(d, nd=1):
    return (("%.*f" % (nd, v)) if isinstance(v := d, (int, float)) else str(v))


def main():
    rows = load()
    print("=== PART 0 AGGREGATE ===")
    print()
    for shift in (0, 2, 3):
        rs = [r for r in rows if r["shift"] == shift]
        if not rs:
            continue
        print("shift %+d  (15 scenes)" % shift)
        print("  tolerance survivals (1.5px/2.5px/4.0px): %.3f / %.3f / %.3f (micro)" % (
            sum(r["survival_1_5"] * r["expected"] for r in rs) / sum(r["expected"] for r in rs),
            sum(r["survival_2_5"] * r["expected"] for r in rs) / sum(r["expected"] for r in rs),
            sum(r["survival_4_0"] * r["expected"] for r in rs) / sum(r["expected"] for r in rs)))
        g = [r["graded_identity_all"] for r in rs]
        gf = [r["graded_identity_frame"] for r in rs]
        print("  graded identity (all pairs): mean sim true %.3f | null-med %.3f null-p95 %.3f | "
              "above per-pair-p95 %.3f | above pooled-p95 %.3f" % (
                  statistics.mean(x["mean_sim_true"] for x in g),
                  statistics.mean(x["null_max_median"] for x in g),
                  statistics.mean(x["null_max_p95"] for x in g),
                  statistics.mean(x["identity_above_null_p95"] for x in g),
                  statistics.mean(x["frac_above_pooled_null_p95"] for x in g)))
        print("  bin identity %.3f  full identity %.3f  population-only %.3f  bin-changed %.3f" % (
            statistics.mean(x["bin_identity_frac"] for x in g),
            statistics.mean(x["full_identity_frac"] for x in g),
            statistics.mean(x["population_only_frac"] for x in g),
            statistics.mean(x["bin_changed_frac"] for x in g)))
        print("  graded identity (FRAME pairs): mean sim %.3f frac>null-p95 %.3f bin-id %.3f" % (
            statistics.mean(x["mean_sim"] for x in gf),
            statistics.mean(x["frac_above_null_p95"] for x in gf),
            statistics.mean(x["bin_identity_frac"] for x in gf)))
        st = sum(r["static"]["tracked_static_count"] for r in rs)
        print("  statics total %d" % st)
        print()

    # per-lens static breakdown at +2 and +3
    for shift in (2, 3):
        agg = Counter()
        for r in rows:
            if r["shift"] != shift:
                continue
            for lens, n in r["static"]["static_lenses"].items():
                agg[lens] += n
        mod8, mod16, seam, identical = Counter(), Counter(), 0, 0
        for r in rows:
            if r["shift"] != shift:
                continue
            for k, v in r["static"]["mod8"].items():
                mod8[int(k)] += v
            for k, v in r["static"]["mod16"].items():
                mod16[int(k)] += v
            seam += r["static"]["seam_band_statics"]
            identical += r["static"]["static_identical_full_count"]
        tot = sum(agg.values())
        top8 = sorted(mod8.items(), key=lambda kv: -kv[1])[:8]
        top16 = sorted(mod16.items(), key=lambda kv: -kv[1])[:8]
        print("shift %+d  static detections by lens:" % shift)
        for k, v in agg.most_common():
            print("  %-20s %4d  %.1f%%" % (k, v, 100.0 * v / tot))
        print("  seam band statics %d | static sig identical at 0px %d | total %d" % (seam, identical, tot))
        print("  mod8 histogram  %s" % dict(top8))
        print("  mod16 histogram %s" % dict(top16))
        print()

    # per-lens everything (frame/attribute), at +2/+3: tracks/statics/neither/gone
    for shift in (2, 3):
        print("per-lens table at %+dpx" % shift)
        print("  %-20s %-8s %4s %5s %5s %5s %6s" % ("lens", "split", "exp", "gone", "trk", "stat", "neither"))
        agg = defaultdict(lambda: defaultdict(int))
        for r in rows:
            if r["shift"] != shift:
                continue
            for x in r["lens_report"]:
                a = agg[x["lens"]]
                for k, v in x.items():
                    if isinstance(v, int):
                        a[k] += v
        for lens, x in sorted(agg.items(), key=lambda kv: -kv[1]["expected"]):
            print("  %-20s %-8s %4d %5d %5d %5d %6d" % (
                lens, "frame" if x.get("track", 0) is not None and _is_frame(lens) else "attro",
                x["expected"], x.get("gone", 0), x.get("track", 0), x.get("static", 0), x.get("neither", 0)))
        print()

    # duplicates at 0px
    print("true duplicates at 0px (same lens, <1px, graded>=.98, bins equal):")
    tot_dups = Counter()
    tot_nbrs = Counter()
    tot = Counter()
    for r in rows:
        if r["shift"] != 0:
            continue
        for lens, n in r["duplicates"]["true_dups_per_lens"].items():
            tot_dups[lens] += n
        tot_nbrs["all"] += r["duplicates"]["density_neighbors_total"]
        print("  %-20s dups %3d  neigh %4d" % (r["scene"], r["duplicates"]["true_dups_total"], r["duplicates"]["density_neighbors_total"]))
    print("  per-lens true duplicates (sum over scenes):", dict(tot_dups))
    print("  density neighbors (sum over scenes):", tot_nbrs["all"])


def _is_frame(lens):
    import sys
    sys.path.insert(0, r"E:\Take25\work\src")
    import matching.relation_frame as RF
    return RF.is_frame(lens)


if __name__ == "__main__":
    main()
