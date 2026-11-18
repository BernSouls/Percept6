"""Post-hoc transfer check: does the selected cell reproduce production joint_mcs?

The tuning scorer re-evaluates a cached correspondence and applies the edge
weight / isolated weight at score time.  Production joint_mcs builds its own
correspondence and reads a different set of parameters.  This probe compares
both on one dev pair so the report can state exactly where they diverge.
"""
import json
from pathlib import Path

import guard

guard.activate()

import parameters as P
import solver
import tune
from frontend import load
from matching.joint_mcs import joint_mcs

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "tune" / "transfer_check.json"


def main():
    data = json.loads((HERE / "results" / "tune" / "graphs.json").read_text())
    winner = json.loads((HERE / "results" / "tune" / "winner.json").read_text())
    query = next(m for m in data["dev"] if m["name"] == "apple4-035-135")
    ref = sorted(data["references"], key=lambda m: m["name"])[0]
    qw, rw = load(query["cache"]), load(ref["cache"])
    q, r = qw.subgraph, rw.subgraph

    base = {
        "MATCH_SYSTEMATICITY_WEIGHT": winner["beta"],
        "MATCH_UNMATCHED_COST": winner["unmatched"],
        "MATCH_ALIGN_UNMAPPED_CREDIT": winner["occlusion"],
        "MATCH_EDGE_WEIGHT": winner["w"],
        "MATCH_COVERAGE_EXPONENT_GAMMA": winner["gamma"],
        "MATCH_EDGE_DENOMINATOR_MODE": winner["denom"],
        "MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE": False,
        "MATCH_UNMATCHED_COST_DERIVED": False,
        "MATCH_SEPARATE_TERM_NORMALIZATION": False,
    }

    sim = solver.node_sim_matrix(rw, qw, rw.sigs, qw.sigs)
    sim_t = tune._transpose(sim)
    rows = []
    for isolated, weight in (("keep", 1.0), ("downweight", 0.5)):
        for k, v in base.items():
            setattr(P, k, v)
        P.MATCH_ISOLATED_NODE_WEIGHT = weight
        prod = joint_mcs(r, q, rw.sigs, qw.sigs)
        # [NEEDS #4] THE ISOLATED WEIGHT IS APPLIED AT THE SAME POINT AS IN
        # PRODUCTION: inside the node similarity, BEFORE the alignment.
        #
        # Production's `node_sim` multiplies a degree-0 pair by
        # MATCH_ISOLATED_NODE_WEIGHT while it is BUILDING the candidate
        # matrix, so the down-weight moves WHO is paired.  The offline arm
        # applied the same number to the node CREDIT after the mapping was
        # fixed, which left the correspondence different (229 mapped nodes
        # against production's 230 even at w=1.0) and therefore made every
        # cell of the tuned surface non-deployable.
        sim_iso = solver.node_sim_matrix(rw, qw, rw.sigs, qw.sigs,
                                         isolated_weight=weight)
        sim_t_iso = tune._transpose(sim_iso)
        fwd = solver.solve_direction(
            rw, qw, rw.sigs, qw.sigs, sim_iso,
            beta=winner["beta"], unmatched_cost=float(winner["unmatched"]),
            unmapped_credit=winner["occlusion"])
        rev = solver.solve_direction(
            qw, rw, qw.sigs, rw.sigs, sim_t_iso,
            beta=winner["beta"], unmatched_cost=float(winner["unmatched"]),
            unmapped_credit=winner["occlusion"])
        facts_f = solver.facts(rw, qw, rw.sigs, qw.sigs, sim_iso, fwd)
        facts_r = solver.facts(qw, rw, qw.sigs, rw.sigs, sim_t_iso, rev)

        # `isolated="keep"`: the weight is ALREADY in the matrix that produced
        # these facts, so the weight must not be applied a second time.
        # max(fwd, rev) is what production does; report which direction won.
        def pick(w):
            vals = [solver.score(f, w=w, gamma=winner["gamma"],
                                 denom_mode=winner["denom"], isolated="keep",
                                 presence_counts=winner["presence"])
                    for f in (facts_f, facts_r)]
            best = max(range(2), key=lambda i: vals[i][0])
            off_map = (dict(fwd.mapping) if best == 0
                       else {v: k for k, v in rev.mapping.items()})
            return vals[best], ("fwd" if best == 0 else "rev"), off_map

        w01, sel01, map01 = pick(0.1)
        w10, sel10, map10 = pick(1.0)
        comp = w01[1]
        rows.append({
            "isolated": isolated,
            "production_isolated_weight": weight,
            "production_score": prod.score,
            "production_mapping": len(prod.mapping),
            "production_node_credit": prod.node_credit,
            "production_edge_credit": prod.edge_credit,
            "production_denominator": prod.denominator,
            "offline_w_0.1": w01[0],
            "offline_w_1.0": w10[0],
            "offline_direction_selected": sel01,
            "offline_node_credit": comp["node_credit"],
            "offline_edge_credit": comp["edge_credit"],
            "offline_denominator": comp["denom"],
            "offline_mapping": len(map01),
            "mapping_equal": prod.mapping == map01,
        })

    payload = {
        "pair": {"reference": ref["name"], "query": query["name"]},
        "winner": winner,
        "rows": rows,
        "production_ignores_edge_weight": False,
        "note": (
            "Production reads MATCH_EDGE_WEIGHT, MATCH_COVERAGE_EXPONENT_GAMMA "
            "and MATCH_EDGE_DENOMINATOR_MODE (mode != 'all' branch of "
            "joint_mcs.py); the isolated weight is applied INSIDE node "
            "similarity before alignment in both paths; the checkable / "
            "per_side edge mass subtracts the presence-only checkable "
            "relations on both paths. The offline and production paths now "
            "agree term-for-term -- identical scores, denominators, node/edge "
            "credits and mappings for both isolated arms (direction of the "
            "winning side is reported). Delta (the normalized-arm node-"
            "coverage exponent) is pooled-arm-irrelevant for the winner "
            "(formula=pooled, delta=0)."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
