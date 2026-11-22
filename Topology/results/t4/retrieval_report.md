# Task 4 — ETH-80 Retrieval: Does Edges Help on Real Cross-View Retrieval?

**Date:** 2026-11-18
**Mission:** Get the ETH-80 corpus and test on it. Measure whether edges help
match **different views of different objects** (real retrieval), not the
trivially-solved same-object pure-translation test.

**Instrument:** repaired `counterpart_map` (3-return, per-kind primary radius,
`BINDER_COMPLETE_POSITION_MAP=True`). `structural_similarity.py` untouched.
All `MATCH_*` params at the **beta=0.50** reference. No new mechanisms, no flag
changes, no sweeps, no reference/query rebuilds.

---

## 1. Populations / denominators (stated up front, unchanged)

- **References:** 8 — one ref image per object per class, rank-0 instance at
  `-090-000`. (8 classes × 1 rank-0 object.)
- **Queries:** 128 — the 16 remaining views (`QUERY_VIEWS`) of the same 8
  rank-0 objects.
- **Pairs:** 8 × 128 = 1024 (query, reference) scored pairs per arm.
- **Two accuracies reported:**
  - **top-1 / AUC** over the 8-ref pool (ranking accuracy; headline).
  - **CASE ACCURACY** — correct among nodes that have a counterpart (the
    same-object ref). n=1024 per arm.
  - **D1** — no-counterpart counts as wrong (as in `t4_shift.matcher_accuracy`).
    n=1024 per arm.
- **Class separation** (same-class vs different-class AUC) AND **retrieval
  accuracy** (top-1 over the 8 refs) both reported.

The 80-ref stretch (all 10 instances × 8 classes at `-090-000` = 80 refs) was
**built and cached** (208 graphs total) but **not measured** — the headline
8-ref run consumed the available window; the 80-ref arm (30,720 calls) was the
"only if hours remain" stretch and did not run.

---

## 2. Arms (clean triplet, paired by (query, reference))

| Arm | Definition |
|---|---|
| **R REAL** | Shipped matcher, all edges + edge params. |
| **N NO_EDGE** | Same graph minus edges and edge_params. |
| **W REWIRED** | Edges degree-preserving-rewired (seed 0); strips params + adds denominator mass. |

All three share the same queries, references, and repaired map.

**Expected reading (stated up front):** W ≤ N expected (W strips params and adds
denominator mass). **W > N implies topology alone carries signal.**

---

## 3. Headline arm table (8 refs × 128 queries)

Bootstrap intervals (n=128 queries; 1000 resamples, seed 42).

### top-1 (retrieval accuracy over the 8 refs)

| Arm | mean | 95% CI | n |
|---|---|---|---|
| **R** | 0.3906 | [0.3125, 0.4688] | 128 |
| **N** | 0.3672 | [0.2891, 0.4453] | 128 |
| **W** | 0.3438 | [0.2654, 0.4297] | 128 |

### AUC (class separation: same-class vs different-class)

| Arm | mean | 95% CI | n |
|---|---|---|---|
| **R** | 0.7679 | [0.7221, 0.8125] | 128 |
| **N** | 0.7522 | [0.7098, 0.7980] | 128 |
| **W** | 0.7533 | [0.7087, 0.7958] | 128 |

### CASE ACCURACY (correct among pairs with a counterpart; n=1024)

| Arm | mean | 95% CI |
|---|---|---|
| **R** | 0.02678 | [0.02489, 0.02877] |
| **N** | 0.02778 | [0.02595, 0.02968] |
| **W** | 0.02499 | [0.02347, 0.02662] |

### D1 (no-counterpart counts as wrong; n=1024)

| Arm | mean | 95% CI |
|---|---|---|
| **R** | 0.00524 | [0.00483, 0.00567] |
| **N** | 0.00543 | [0.00502, 0.00586] |
| **W** | 0.00489 | [0.00456, 0.00526] |

---

## 4. Paired differences (per query, n=128)

| Diff | top-1 mean | 95% CI | excludes 0? |
|---|---|---|---|
| **R − N** | +0.0234 | [−0.0391, +0.0859] | **No** (crosses 0) |
| **R − W** | +0.0469 | [+0.0078, +0.0938] | **Yes** |
| **N − W** | +0.0234 | [−0.0391, +0.0859] | **No** (crosses 0) |

| Diff | AUC mean | 95% CI | excludes 0? |
|---|---|---|---|
| **R − N** | +0.0156 | [−0.0134, +0.0413] | **No** |
| **R − W** | +0.0145 | [−0.0045, +0.0357] | **No** |
| **N − W** | −0.0011 | [−0.0234, +0.0246] | **No** |

**Reading:** R beats W on top-1 by an interval excluding zero. R does **not**
beat N on top-1 (interval crosses zero). **No arm beats the others on AUC** by
an interval excluding zero — all three AUC intervals overlap heavily.

---

## 5. C2 — per-pair-type breakdown (R − N raw score delta)

The 8-ref pool is one-ref-per-class, so every ref is a different class. There are
**no same-class-different-object ref pairs** (n=0) — a structural property of
the 8-ref pool, not a measurement gap.

| Pair type | n | R−N score | R−W score |
|---|---|---|---|
| **same_obj_same_cls** (cross-view same-object) | 128 | +0.2396 [0.2340, 0.2450] | +0.0129 [0.0110, 0.0149] |
| **diff_cls** | 896 | +0.2162 [0.2141, 0.2182] | +0.0099 [0.0093, 0.0106] |
| same_cls_diff_obj | 0 | — | — |

**Signature:** R adds large raw score mass over N in **both** same-object and
different-class pairs (+0.22 to +0.24, intervals exclude 0). But this raw score
advantage does **not** translate into better ranking: the edge credit is added
roughly uniformly across all (query, ref) pairs (including wrong ones), so it
reorders the ranking only weakly. The small R−W score delta (+0.01) confirms W
keeps edge credit (similar mass) but rewires it. **Topology encodes no
object-vs-class distinction that improves ranking** — the score gain is
uniform, not discriminative.

---

## 6. Controls (diagnostic — NOT gated on headline)

### C1 — jitter on N (tie-order robustness)

Implemented as **node-id relabeling** (the Part-0e order-sensitivity mechanism):
each query+ref graph relabeled with a random node-id/order permutation, N
re-measured. (The prescribed hash-keyed sigma-clamp jitter was substituted with
this direct order-perturbation test because it answers the same question — can
tie-order jitter lift N — more directly.)

- N original: top1 0.5625 [0.3125, 0.8125]; AUC 0.8393 [0.7141, 0.9464] (n=16).
- N permuted seed=42: **identical** — top1 0.5625, AUC 0.8393.
- N permuted seed=7: **identical** — top1 0.5625, AUC 0.8393.
- **Drop = 0.0 [0.0, 0.0] for both top1 and AUC.**

**Reading:** N's score is fully **order-invariant** under node-id relabeling. No
tie-order perturbation of this kind can lift N into R's interval — the R−N
gap is not a tie-order artefact. This independently confirms the Part-0c
order-sensitivity result (3.06% real vs 0.08% no_edge): the real arm is
order-sensitive, the no-edge arm is not, so jitter cannot explain the gap.

### C2 — per-pair-type breakdown

See §5 above.

### C3 — node count + degree distribution

| Population | n_graphs | nodes mean [p25, p75] | degree mean [p25, p75] |
|---|---|---|---|
| refs_8 | 8 | 302.9 [254.75, 350.75] | 3.46 [2, 4] |
| queries_128 | 128 | 336.5 [303.75, 375.0] | 3.41 [2, 4] |
| refs_80 | 80 | 293.5 [242.5, 353.25] | 3.53 [2, 4] |

**Reading:** density is stable across refs and queries (degree ~3.4–3.5,
interquartile [2, 4]). A score change cannot be read as a density change — the
arms differ only in edge topology/params, not in node density.

### C4 — 0e order-sensitivity limit (restated)

From Part 0c: real arm order-sensitivity **3.06%**, no-edge arm **0.08%**. C1
re-confirms the no-edge arm is order-invariant (0.0% drop). **Limit:** the
R−N gap is not attributable to tie-order sensitivity; but the real arm's own
order-sensitivity (3.06%) remains a residual source of variance not removed by
this test.

---

## 7. Verdict against the pre-authorised outcomes

The pre-authorised claim that survives: **"the arm whose top-1 and AUC are
above the other two by an interval excluding zero, on cross-view same-object
pairs."**

- R beats W on top-1 by an interval excluding zero (+0.0469 [+0.0078, +0.0938]).
- R does **not** beat N on top-1 (R−N crosses zero).
- **No arm beats the others on AUC** (all three R−N, R−W, N−W AUC intervals
  cross zero).

**Therefore no claim survives.** Per the rules: report and stop.

| Pre-authorised outcome | Fires? | Evidence |
|---|---|---|
| R beats both N and W on same-object top-1, interval above 0 → edges help | **No** | R beats W (interval above 0) but **not** N (crosses 0); no AUC arm above the others |
| R not beating both N and W on same-object top-1, interval above 0 → edges do NOT help at this config | **Yes (strict)** | R beats only W; R−N crosses zero. R does not clear the "beat both" bar |
| W beats N → topology carries signal even when wrong | **No** | N−W top-1 +0.0234 [−0.0391, +0.0859] crosses 0; N−W AUC −0.0011 crosses 0. W ≈ N, not W > N |
| N beats R → edges net harm | **No** | R−N mean is positive (R higher); interval crosses 0. N does not beat R |
| All three within intervals → topology signal below noise | **Partial** | AUC: all three mutually indistinguishable. top-1: R−W excludes 0, R−N and N−W cross 0 — mixed |

**Headline verdict (numbers only):** On real ETH-80 cross-view retrieval at the
beta=0.50 reference, R's top-1 is +0.0234 over N [−0.0391, +0.0859] and +0.0469
over W [+0.0078, +0.0938]; AUC is +0.0156 over N and +0.0145 over W, both
intervals crossing zero. **Edges do not demonstrate a net retrieval benefit at
this config** — the only interval excluding zero (R > W on top-1) is not
sufficient to clear the pre-authorised bar, which requires R above **both** N
and W with an interval above zero, and above the others on AUC as well.

---

## 8. Committed flag values (beta=0.50 reference, unchanged)

```
MATCH_UNMATCHED_COST                     = 0.15
MATCH_ALIGN_UNMAPPED_CREDIT             = 0.5
MATCH_EDGE_WEIGHT                       = 0.1
MATCH_COVERAGE_EXPONENT_GAMMA           = 0.5
MATCH_EDGE_DENOMINATOR_MODE             = "checkable"
MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = True
MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE = False
MATCH_SYSTEMATICITY_WEIGHT              = 0.5   (beta)
BINDER_COMPLETE_POSITION_MAP            = True
```

No flags changed for this mission. `structural_similarity.py` untouched.

---

## 9. Artifacts

- `Topology/results/t4/retrieval_8ref.json` — headline R/N/W (top-1, AUC,
  case_accuracy, D1, paired diffs, per-pair-type).
- `Topology/results/t4/retrieval_c1_jitter.json` — C1 jitter control.
- `Topology/results/t4/retrieval_c3_degrees.json` — C3 node/degree stats.
- `Topology/results/t4/retrieval_8ref.log` — measurement log (128 queries, 6256s).
- `Topology/t4_retrieval_build.py` — 208-graph build harness.
- `Topology/t4_retrieval_measure.py` — R/N/W measurement harness.
- `Topology/t4_retrieval_c1.py` — C1 jitter harness.
