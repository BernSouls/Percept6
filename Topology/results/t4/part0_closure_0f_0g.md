# Task 4 PART 0 closure — 0f withdrawals + 0g read-sites

Branch `arena/c316ca3a-percept6`. Truth instrument repaired in `Topology/t4_shift.py`
(`counterpart_map`); all flags still default OFF; unmodified objective kept as
reference. This file records the two remaining Part 0 deliverables.

## 0f — Bogus claims withdrawn (one line each)

1. **"permuted-robust"** — never run; the order check (0e) was only executed after
   the repair, and it shows tie order DOES matter (3.06% mapping change), so the
   claim of permutation-robustness is withdrawn as untested-then-contradicted.
2. **"denominator penalty"** (B5) — used the wrong metric (gross quality, not the
   net-surplus objective the matcher actually maximises); withdrawn.
3. **B3 "identifiability"** — tautological (it re-derived the case definition from
   the same node_sim ordering); retired.
4. **P3-4 "inert"** — compared keys at shift 0 only; at the actual shifts the
   comparison is not inert; withdrawn.
5. **"parameters non-informative"** — 1d now measures real−params_shuffled ≈ 0 on
   STRICT-LEAD but real−real_params_dropped = +0.0072 [+0.0026,+0.0130] (params DO
   carry a small signal); the blanket claim withdrawn.
6. **D1(0)=1.0000 "perfect"** — class-filtered (pooled D1(0)=0.8638); the pooled
   reading is withdrawn as the headline.

## 0g — `MATCH_SYSTEMATICITY_WEIGHT` read sites + isolated-weight handling

`MATCH_SYSTEMATICITY_WEIGHT` (beta) is read at exactly four sites:

| site | role |
|---|---|
| `matching/fuzzy.py:156` | `getattr(..., 0.5)` fallback in fuzzy match |
| `matching/joint_mcs.py:235` | `price_scale = (1.0 - w)` — scales the unmatched-cost surplus |
| `matching/joint_mcs.py:247` | passed as the `beta` argument to the objective |
| `memory/cores.py:923` | `getattr(..., 0.5)` fallback in core matching |

`MATCH_UNMATCHED_COST_DERIVED` is read at `matching/joint_mcs.py:223`
(`getattr(..., False)`).

Isolated-weight handling in `matching/joint_mcs.py`:
- `MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE` — read at `:171`.
- `MATCH_EXCLUDE_ISOLATED_FLOOR`, `MATCH_ISOLATED_NODE_WEIGHT` — read at `:195/:196/:199`.
- `isolated_w` declared at `:208`; applied at `:236` (baseline correspondence
  down-weight) and `:343–350` ([T3] score-time isolated down-weight).

**Authorised change sites for the fixes** (per mission): `matching/alignment.py`
`_relational_agreement` (F1 absence discount), `_assign` q construction (F2 g(u),
F0 beta in q), and the edge-parameter similarity bandwidth; plus
`matching/perceptual_selection.py` tie-break/rank key (F3/F4) with
`EDGE_BUDGET_PER_FRAME` / `FAMILY_SEED_EDGES` UNCHANGED. `matching/structural_similarity.py`
is NOT to be touched.
