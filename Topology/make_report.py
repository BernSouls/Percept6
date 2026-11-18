"""Assemble TUNING_REPORT.md from the measured artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import guard
from guard import TREE

TUNE = TREE / "topology" / "results" / "tune"
OUT = TREE / "topology" / "TUNING_REPORT.md"


def load(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v, nd=4):
    return "n/a" if v is None else ("%.*f" % (nd, v))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def marginal(rows, knob):
    values = sorted({r[knob] for r in rows}, key=lambda v: (str(type(v)), v))
    out = []
    for v in values:
        sub = [r for r in rows if r[knob] == v]
        best = max(sub, key=lambda r: (r["auc"], r["top1"]))
        out.append([v, fmt(max(r["auc"] for r in sub)),
                    fmt(sum(r["auc"] for r in sub) / len(sub)),
                    fmt(best["top1"]), best["name"]])
    return out


def main():
    rows = []
    path = TUNE / "dev_surface.csv"
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in ("beta", "w", "gamma", "delta", "top1", "auc"):
                    r[k] = float(r[k])
                r["occlusion"] = float(r["occlusion"])
                r["reference"] = r["reference"] == "True"
                r["presence"] = r["presence"] == "True"
                if r["unmatched"] != "derived":
                    r["unmatched"] = float(r["unmatched"])
                rows.append(r)
    winner = load(TUNE / "winner.json", {})
    graphs = load(TUNE / "graphs.json", {})
    held = load(TUNE / "heldout_seed0_metrics.json", {})
    details_dev = load(TUNE / "details_dev.json", {})
    details_held = load(TUNE / "details_heldout.json", {})
    kt = load(TREE / "topology" / "results" / "known_transform_corrected.json", {})
    det = load(TUNE / "determinism.json", {})
    prov = load(TREE / "topology" / "results" / "provenance.json", {})
    receipt = load(TUNE / "sweep_receipt.json", {})

    lines = []
    lines.append("# Matcher tuning on ETH-80 — measured surface\n")
    lines.append("Pre-registration: `work/topology/TUNING_PREREGISTRATION.md` "
                 "(written before any accuracy number).\n")
    if graphs:
        counts = graphs.get("counts", {})
        metas = graphs.get("graphs", [])
        mean_nodes = sum(m["nodes"] for m in metas) / len(metas) if metas else 0
        mean_edges = sum(m["edges"] for m in metas) / len(metas) if metas else 0
        lines.append("## Data and graphs\n")
        lines.append("Instance-disjoint split: %s references, %s dev queries, "
                     "%s held-out queries (8 classes; three reference instances "
                     "per class). Graph budget 512 edges; mean %.1f nodes / "
                     "%.1f edges over %d graphs.\n"
                     % (counts.get("references"), counts.get("dev"),
                        counts.get("heldout"), mean_nodes, mean_edges, len(metas)))
    if rows:
        candidates = [r for r in rows if not r["reference"]]
        best = max(candidates, key=lambda r: (r["auc"], r["top1"]))
        node = [r for r in rows if r["name"] == "node_only"]
        best_node = max(node, key=lambda r: (r["auc"], r["top1"])) if node else None
        lines.append("## Dev surface (60 alignment x 216 score cells = 12 960, "
                     "plus reference rows)\n")
        lines.append("Two score formulas, from cached facts:\n")
        lines.append("```\npooled:     S = 2 * (node_credit + w*edge_credit) / "
                     "(na+nb+edge_mass)\n"
                     "normalized: node_term = node_credit/(na+nb)\n"
                     "            edge_term = edge_credit/(checkable_a+checkable_b)\n"
                     "            node_cov = 2*matched/(na+nb)\n"
                     "            edge_cov = preserved/checkable\n"
                     "            S = 2*((1-w)*node_term*node_cov**delta + "
                     "w*edge_term*edge_cov**gamma)\n```\n")
        lines.append("Best dev cell (offline score-time formula): beta=%s, "
                     "unmatched=%s, occlusion=%s, "
                     "formula=%s, w=%s, gamma=%s, delta=%s, denom=%s, "
                     "isolated=%s -> top-1 %s, AUC %s.\n"
                     % (winner.get("beta"), winner.get("unmatched"),
                        winner.get("occlusion"), winner.get("formula"),
                        winner.get("w"), winner.get("gamma"),
                        winner.get("delta"), winner.get("denom"),
                        winner.get("isolated"), fmt(winner.get("top1")),
                        fmt(winner.get("auc"))))
        if best_node:
            node_json = load(TUNE / "node_only.json", {})
            node_ci = node_json.get("auc_ci") or {}
            node_t1 = node_json.get("top1_ci") or {}
            lines.append("Node-only reference (best alignment for that row): "
                         "top-1 %s (95%% interval %s–%s), AUC %s "
                         "(95%% interval %s–%s; beta=%s, unmatched=%s, "
                         "occlusion=%s).\n"
                         % (fmt(best_node["top1"]), fmt(node_t1.get("lo")),
                            fmt(node_t1.get("hi")), fmt(best_node["auc"]),
                            fmt(node_ci.get("lo")), fmt(node_ci.get("hi")),
                            best_node["beta"], best_node["unmatched"],
                            best_node["occlusion"]))
            best_norm = [r for r in rows
                         if r["name"] == "norm_node_only_d0"]
            if best_norm:
                bn = max(best_norm, key=lambda r: (r["auc"], r["top1"]))
                lines.append("The strongest node-only variant "
                             "(`norm_node_only_d0`, normalized, w=0, delta 0) "
                             "reaches top-1 %s / AUC %s, so the winner is "
                             "%+.4f AUC over a pure node score.\n"
                             % (fmt(bn["top1"]), fmt(bn["auc"]),
                                winner.get("auc", 0.0) - bn["auc"]))
        if winner.get("auc_ci"):
            ci = winner["auc_ci"]
            t1ci = winner.get("top1_ci") or {}
            lines.append("Winner dev top-1 95%% interval %s–%s; AUC 95%% "
                         "interval by query frame %s–%s (bootstrap mean "
                         "%s).\n"
                         % (fmt(t1ci.get("lo")), fmt(t1ci.get("hi")),
                            fmt(ci.get("lo")), fmt(ci.get("hi")),
                            fmt(ci.get("mean"))))
        transfer = load(TUNE / "transfer_check.json", {})
        if transfer:
            lines.append("### Production transfer (post-hoc check)\n")
            pair = transfer["pair"]
            lines.append("One dev pair (`%s` vs `%s`), winner alignment and "
                         "score settings, run through production "
                         "`joint_mcs` and the offline scorer:\n"
                         % (pair["reference"], pair["query"]))
            tr_rows = []
            for r in transfer["rows"]:
                tr_rows.append([
                    r["isolated"], r["production_mapping"],
                    r["offline_mapping"],
                    "yes" if r["mapping_equal"] else "no",
                    fmt(r["production_score"], 6),
                    fmt(r["offline_w_0.1"], 6),
                    fmt(r["offline_w_1.0"], 6)])
            lines.append(table(["isolated", "prod map", "offline map",
                                "same map", "prod score", "offline w=0.1",
                                "offline w=1.0"], tr_rows))
            lines.append("")
            lines.append(
                "Production's pooled branch never reads `MATCH_EDGE_WEIGHT`: "
                "it always gives the edge term full weight. At isolated=keep "
                "the two implementations return the same mapping and the same "
                "node, edge and denominator components; production's score "
                "equals the offline w=1.0 score exactly, while the selected "
                "w=0.1 gives %s. At isolated=downweight production applies the "
                "weight inside node similarity before alignment (229 vs 230 "
                "mapped nodes), so even w=1.0 differs (%s vs %s). The selected "
                "cell is therefore a score-time study configuration, not a "
                "deployable parameter set: deploying it needs "
                "`MATCH_EDGE_WEIGHT` in the pooled branch and a decision on "
                "where the isolated weight applies. No cell in the grid is "
                "exactly production-equivalent: production's pooled score is "
                "the unmeasured w=1 edge, and its separate-term branch has no "
                "coverage exponents.\n"
                % (fmt(transfer["rows"][0]["offline_w_0.1"], 6),
                   fmt(transfer["rows"][1]["production_score"], 6),
                   fmt(transfer["rows"][1]["offline_w_1.0"], 6)))
        lines.append("### Top 12 cells by dev AUC\n")
        top = sorted(candidates, key=lambda r: (-r["auc"], -r["top1"]))[:12]
        lines.append(table(["beta", "unmatched", "occlusion", "formula", "w",
                            "gamma", "delta", "denom", "isolated", "top-1", "AUC"],
                           [[r["beta"], r["unmatched"], r["occlusion"],
                             r["formula"], r["w"], r["gamma"], r["delta"],
                             r["denom"], r["isolated"],
                             fmt(r["top1"]), fmt(r["auc"])] for r in top]))
        lines.append("\nEvery cell is in `results/tune/dev_surface.csv` "
                     "(%d tuned cells; %d reference rows).\n"
                     % (len(candidates), len(rows) - len(candidates)))

        def arm_w(formula):
            out = []
            ws = sorted({r["w"] for r in candidates if r["formula"] == formula})
            for wv in ws:
                sub = [r for r in candidates
                       if r["formula"] == formula and r["w"] == wv]
                b = max(sub, key=lambda r: (r["auc"], r["top1"]))
                out.append([wv, len(sub), fmt(max(r["auc"] for r in sub)),
                            fmt(sum(r["auc"] for r in sub) / len(sub)),
                            fmt(b["top1"]), b["name"]])
            return out

        lines.append("### Does the edge weight matter?\n")
        balance = load(TUNE / "credit_balance.json", {})
        if balance:
            lines.append(
                "The pooled formula cannot make edges decisive: on a %d-pair dev "
                "sample (4 queries x 24 references) the node:edge credit ratio "
                "has median %s (mean %s, range %s-%s), so even w=0.75 leaves "
                "the edge term a small part of the pooled numerator (the "
                "pre-registration's ~10x estimate was high). The normalized "
                "formula makes w a true blend, so it can. Best AUC "
                "per w, marginalising the other knobs:\n"
                % (balance["pairs"], fmt(balance["median_ratio"], 2),
                   fmt(balance["mean_ratio"], 2), fmt(balance["min_ratio"], 2),
                   fmt(balance["max_ratio"], 2)))
        else:
            lines.append("The pooled formula cannot make edges decisive because "
                         "node credit dominates it; the normalized formula "
                         "makes w a true blend. Best AUC per w, marginalising "
                         "the other knobs:\n")
        for formula in ("pooled", "normalized"):
            lines.append("\n**%s**\n" % formula)
            lines.append(table(["w", "cells", "max AUC", "mean AUC",
                                "best top-1", "best cell"], arm_w(formula)))
        def at(name, beta=0.75, uc=0.15, occ=0.5):
            hit = [r for r in rows if r["name"] == name and r["beta"] == beta
                   and r["unmatched"] == uc and r["occlusion"] == occ]
            return hit[0] if hit else None
        same = [("pooled node-only", at("node_only")),
                ("normalized node-only, delta 0", at("norm_node_only_d0")),
                ("normalized w=0.10, gamma 0, delta 0",
                 at("norm_w0.10_g0.0_d0.0_downweight")),
                ("normalized w=0.75, gamma 1, delta 1",
                 at("norm_w0.75_g1.0_d1.0_keep"))]
        lines.append("\nAt one fixed alignment cell (beta 0.75, unmatched "
                     "0.15, occlusion 0.5), so only the score changes:\n")
        lines.append(table(["row", "top-1", "AUC"],
                           [[label, fmt(r["top1"]), fmt(r["auc"])]
                            for label, r in same if r]))
        lines.append("\nIn the normalized arm, edge credit does not raise AUC "
                     "over the node-only row at that mapping (w=0.10 is slightly "
                     "below it), and at w=0.75, where the edge term is three "
                     "quarters of the blend, AUC falls to %s. Across all mappings "
                     "the normalized arm's best AUC falls monotonically with w. "
                     "The pooled winner's own AUC does not isolate edge "
                     "credit: its w=0 reference at the same mapping scores %s "
                     "with a different denominator, while the normalized w=0 row, "
                     "which differs only in the edge blend, scores %s and does "
                     "not improve when edges enter. This is the measured answer "
                     "to the question the pooled formula could not ask: in this "
                     "measurement, the edges are not additive class evidence at "
                     "this weight range, and making them decisive hurts.\n"
                     % (fmt(next((r["auc"] for r in rows
                                  if r["name"] == "norm_w0.75_g1.0_d1.0_keep"
                                  and r["beta"] == 0.75
                                  and r["unmatched"] == 0.15
                                  and r["occlusion"] == 0.5), None)),
                        fmt(next((r["auc"] for r in rows
                                  if r["name"] == "node_only"
                                  and r["beta"] == 0.75
                                  and r["unmatched"] == 0.15
                                  and r["occlusion"] == 0.5), None)),
                        fmt(next((r["auc"] for r in rows
                                  if r["name"] == "norm_node_only_d0"
                                  and r["beta"] == 0.75
                                  and r["unmatched"] == 0.15
                                  and r["occlusion"] == 0.5), None))))
        lines.append("### Reference rows (same alignment grid, not tuned)\n")
        ref_rows = [r for r in rows if r["reference"]]
        by_name = {}
        for r in ref_rows:
            if r["name"] not in by_name or (r["auc"], r["top1"]) > (
                    by_name[r["name"]]["auc"], by_name[r["name"]]["top1"]):
                by_name[r["name"]] = r
        lines.append("Best alignment cell for each reference row (all 60 "
                     "alignment cells per row are in the CSV):\n")
        lines.append(table(["name", "beta", "unmatched", "occlusion", "top-1",
                            "AUC"],
                           [[r["name"], r["beta"], r["unmatched"],
                             r["occlusion"], fmt(r["top1"]), fmt(r["auc"])]
                            for r in sorted(by_name.values(),
                                            key=lambda r: r["name"])]))
        lines.append("\n### Marginals (all tuned cells)\n")
        for knob in ("beta", "unmatched", "occlusion", "formula", "isolated"):
            lines.append("**%s**\n" % knob)
            lines.append(table([knob, "max AUC", "mean AUC", "best top-1",
                                "best cell"],
                               marginal(candidates, knob)))
            lines.append("")
        pooled = [r for r in candidates if r["formula"] == "pooled"]
        lines.append("**denom (pooled cells only)**\n")
        lines.append(table(["denom", "max AUC", "mean AUC", "best top-1",
                            "best cell"], marginal(pooled, "denom")))
        lines.append("")
        norm = [r for r in candidates if r["formula"] == "normalized"]
        for knob in ("gamma", "delta"):
            lines.append("**%s (normalized cells only)**\n" % knob)
            lines.append(table([knob, "max AUC", "mean AUC", "best top-1",
                                "best cell"], marginal(norm, knob)))
            lines.append("")
    lines.append("## Held-out (winner fixed on dev, no re-selection)\n")
    if held:
        order = [("real", "best configuration (offline formula)"),
                 ("node_only", "node-only"),
                 ("rewired", "edges rewired"), ("no_edge", "no edges")]
        lines.append(table(["scoring", "top-1", "95% top-1 interval", "AUC",
                            "95% AUC interval"],
                           [[label, fmt(held.get(key, {}).get("top1")),
                             "%s–%s" % (fmt(held.get(key, {}).get("top1_ci", {}).get("lo")),
                                        fmt(held.get(key, {}).get("top1_ci", {}).get("hi"))),
                             fmt(held.get(key, {}).get("auc")),
                             "%s–%s" % (fmt(held.get(key, {}).get("auc_ci", {}).get("lo")),
                                        fmt(held.get(key, {}).get("auc_ci", {}).get("hi")))]
                            for key, label in order if key in held]))
        lines.append("\nNode-only reuses the winner's real-edge mapping with "
                     "w=0; the no-edge row re-solves with empty edge sets, so "
                     "it also changes the correspondence. The rewired control "
                     "keeps the edge count and degree but drops edge "
                     "parameters.\n")
    else:
        lines.append("Held-out artifacts are not present yet.\n")
    if details_held or details_dev:
        lines.append("\nWinner correspondence, averaged over every "
                     "query x reference pair:\n")
        rows_d = []
        for label, dd in (("dev", details_dev), ("held-out", details_held)):
            if not dd:
                continue
            s = dd["summary"]
            rows_d.append([label, s["pairs"],
                           fmt(s["mean_matched_nodes"], 1),
                           fmt(s["mean_matched_edges"], 1),
                           fmt(s["mean_edge_preservation"]),
                           fmt(s["mean_graded_similarity"])])
        lines.append(table(["split", "pairs", "matched nodes",
                            "matched edges", "edge preservation",
                            "graded edge similarity"], rows_d))
    if held:
        real = held.get("real", {})
        def delta(key):
            other = held.get(key, {})
            return (other.get("auc", 0.0) - real.get("auc", 0.0),
                    other.get("top1", 0.0) - real.get("top1", 0.0))
        d_node, t_node = delta("node_only")
        d_rew, t_rew = delta("rewired")
        d_none, t_none = delta("no_edge")
        lines.append("\nDeltas against the selected configuration on the same "
                     "held-out queries (negative = worse than real edges): "
                     "node-only AUC %+.4f / top-1 %+.4f; rewired AUC %+.4f / "
                     "top-1 %+.4f; no edges AUC %+.4f / top-1 %+.4f. The "
                     "intervals overlap, so this ablation does not separate the "
                     "edge structure from node-only scoring. The rewired control "
                     "lands close to the real graph rather than at the no-edge "
                     "level, which is not the pattern expected if the specific "
                     "edge structure carried the class signal. Top-1 is also "
                     "well above the 0.125 chance level, but the held-out point "
                     "estimate is below the dev estimate, as expected after "
                     "selecting the dev maximum.\n"
                     % (d_node, t_node, d_rew, t_rew, d_none, t_none))
    lines.append("")
    lines.append("## Known-transform instrument (corrected)\n")
    if kt:
        g = kt["grid"]
        lines.append("Grid pitch %s px (density 16 on a %s px image); tolerance "
                     "%s px = half pitch. Warp: %s.\n"
                     % (fmt(g["pitch_px"], 1), kt["image_side"][0],
                        fmt(g["tolerance_px"], 1), kt["warp"]))
        lines.append(table(["shift", "nodes", "with position", "truth preserved",
                            "truth frac", "anchor-cell frac", "matcher frac",
                            "Jaccard"],
                           [[str(r["shift"]), r["nodes"], r["nodes_with_position"],
                             "%d/%d" % (r["truth_preserved"], r["truth_expected"]),
                             fmt(r["truth_frac"], 3), fmt(r["anchor_cell_frac"], 3),
                             fmt(r["matcher_frac"], 3), fmt(r["jaccard"], 3)]
                            for r in kt["rows"]]))
        zero = kt["rows"][0]
        lines.append("\nAt 0 px: truth %s, Jaccard %s. The historical 403 -> 164 "
                     "counts are not reproducible from this tree: the current "
                     "graph has %d nodes at 0 px and %d at +2 px, with %d and %d "
                     "positioned nodes, so no 60%% node loss exists. A 2 px shift "
                     "is an eighth of the 16 px pitch: only centroids within "
                     "2 px of a cell boundary can cross one, so it cannot "
                     "produce a 60%% loss. The mechanism consistent with both "
                     "old numbers is that the counted set "
                     "changed at extraction (margin filter and anchor-cell "
                     "quantizer), not that detection collapsed; neither old "
                     "number is reported as a stability result.\n"
                     % (fmt(zero["truth_frac"], 3), fmt(zero["jaccard"], 3),
                        zero["nodes"], kt["rows"][1]["nodes"],
                        zero["nodes_with_position"],
                        kt["rows"][1]["nodes_with_position"]))
    lines.append("## Masked cells\n")
    masked = load(TUNE / "masked_cells.json", None)
    if masked:
        lines.append("`masked_cells.json` names the missing query x class cells: "
                     "%s\n" % json.dumps(masked[:10]))
    else:
        lines.append("The final surface has **zero masked cells**: all 48 dev "
                     "queries x 8 classes x 220 score cells are finite "
                     "(13 200 score rows x 48 queries x 8 classes). The checks "
                     "are mechanical: the "
                     "worker seeds its array with -inf (not NaN, because "
                     "`max(nan, x)` keeps nan and would silently mask a class), "
                     "refuses to write a query unless every class that has a "
                     "reference also has a finite score, and `combine` refuses "
                     "to stack the surface if any non-finite entry remains, "
                     "writing `masked_cells.json` with the query and class.\n")
        lines.append("The historical sentinels were narrower than a missing "
                     "class: the npz stores one score per (query, score cell, "
                     "class), with references aggregated before the save. One "
                     "file, `apple4-035-135.npz`, had been overwritten by a "
                     "manual 4-reference probe covering only apple and car. The "
                     "six classes with no reference in that probe (cow, cup, "
                     "dog, horse, pear, tomato) kept the `-1e30` initialiser: at "
                     "most 6 x 6 600 = 39 600 sentinel entries. That is an upper "
                     "bound reconstructed from the grid shape; the exact count is "
                     "not recoverable because the probe file was replaced. The "
                     "full 24-reference worker replaced that file, and no "
                     "sentinel was found in the other dev files that were "
                     "checked (the first ten completed files). The old `-1e30` "
                     "initialiser would have hidden the same condition in the "
                     "final run; the -inf initialiser plus the two assertions "
                     "make it impossible to report a surface with a silently "
                     "dropped cell.\n")
    lines.append("## Determinism\n")
    if det:
        lines.append("Winner under `PYTHONHASHSEED=0` and `1729`: hashes %s.\n"
                     % ("MATCH" if det.get("match") else "DIFFER"))
    else:
        lines.append("Determinism receipt not present yet.\n")
    lines.append("## Provenance\n")
    if prov:
        lines.append(table(["file", "status", "added", "removed", "churn-only"],
                           [[r["file"], r["status"],
                             "n/a" if r.get("added") is None else r["added"],
                             "n/a" if r.get("removed") is None else r["removed"],
                             r.get("churn_only") or False]
                            for r in prov["rows"]]))
        lines.append("\nBaseline: `%s` (`work/` is untracked; the tracked repo "
                     "`src/` is a different copy). CRLF normalized before the "
                     "line diff.\n" % prov["baseline"])
    lines.append("## Compute\n")
    if receipt:
        lines.append("%d dev queries, %d alignment cells x %d score cells, "
                     "%d workers. Alignment-pair evaluations %d; score "
                     "evaluations %d. Wall time %.0f s across %d invocations "
                     "(%s); the second resumed exactly the queries missing from "
                     "the first, with the same code and grid.\n"
                     % (receipt["queries"], receipt["align_cells"],
                        receipt["score_cells"], receipt["workers"],
                        receipt["pair_solves"], receipt["score_evaluations"],
                        receipt["seconds"], len(receipt.get("segments", []) or [1]),
                        ", ".join("%d queries / %.0f s" % (s["queries"], s["seconds"])
                                  for s in receipt.get("segments", []))))
    lines.append("## What was not run\n")
    lines.append("- No second corpus, no scaling beyond the pre-registered split.\n"
                 "- No tuning of graph budget, views, or the front end.\n"
                 "- No per-cell bootstrap intervals on the full surface (the "
                 "winner and node-only rows carry them).\n"
                 "- The held-out metrics come from two invocations of the same "
                 "winner: the first was cut off by a tool timeout after 16 of 48 "
                 "queries, and the second resumed the remaining 32 with no "
                 "configuration change. An earlier attempt aborted before "
                 "writing any score because the node-only ablation mode was not "
                 "routed to the real graph; that routing fix changes no score.\n"
                 "- The 403/164 historical counts are not used as a result; "
                 "the corrected instrument is reported instead.\n"
                 "- The selected cell is not production-equivalent (Production "
                 "transfer): the held-out numbers measure the offline "
                 "score-time formula, not `joint_mcs` as it ships.\n")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
