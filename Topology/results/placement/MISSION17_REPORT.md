# MISSION 17 — make placement rank, not just qualify

Branch `arena/c316ca3a-percept6`, from the Mission 16 tip `3b2fc61`.
`MISSION17_PREREGISTRATION.md` was committed alone (`80dd36c`) **before any Mission 17
number existed**; §O4 and the fresh-read design below are that document, unedited except
for the planning sd, which the registration said would be filled from the DEV cells
before a fresh graph was built.

Everything is behind a flag that is **OFF** by default; the defaults are byte-identical
(§8). Units: every residual, tolerance and bin width below is in **tolunits**
(`payload_residual` is tolerance-normalised and dimensionless; no px claim is made).

---

## THE FOUR NUMBERS

| | before | after | where |
|---|---|---|---|
| geometry AUC at 15°, matcher's context | 0.631 | **0.710** | O2, 4 DEV images, 1,166 queries |
| top-1 at 15° (ties fail) | 0.105 | **0.198** | O3, same cells |
| context accuracy at 15° | 0.173 | **0.288** | O2 diagnostic |
| placement's contribution to top-1 at 15° | — | **+0.109 [+0.088, +0.129]** | O6 fresh read, **20 images**, MDE 0.0064 |

The change that produced them: the context no longer starts from a per-node argmax of
appearance; it starts from a **one-to-one assignment** over a score that adds a
**rotation-invariant configuration descriptor**. The damped iteration's own constants were
re-fitted first, and **none of them moved anything** (§O2).

The table refit of O1 was built, validated and **not shipped** — it loses the registered
objective (§O1, §O4). The defect it repairs is therefore live, and that is the single next
change.

---

## O1 — THE DEPLOYED LLR IS DEFENSIBLE

### The defect, as a number

`m17_llr.json`, `matching/placement_frozen_params.json` vs the refit:

| | **deployed** | **M17 refit** |
|---|---|---|
| support × resolution | [0, 6.0] × 24 bins (0.25 tol) | [0, 1.8] × 36 bins (0.05 tol) |
| fitted on | 2,574 true / 4,684 wrong | **13,578 true / 16,871 wrong** (pooled places) |
| rises with \|residual\| | **5** | **0** |
| positive bins | **16 of 24** | **2 of 36** |
| LLR at 5.0 tol (a far candidate) | **+0.597** | **−2.96** |
| LLR at 1.5 tol (a near one) | −2.237 | −2.96 |
| per-bin true/wrong counts stored | **no** | **yes** |
| empty/thin bins paying positive credit | 13 of 24 (M15's comment block) | **0** |

**The single sentence:** the deployed table pays a candidate 5 tolunits away **+0.597
nats** while paying one 1.5 tolunits away **−2.237** — it pays 2.83 nats *more* for being
farther. That is the Laplace prior of an empty bin read as support, and it is live.

Exposure, measured on the pooled bank: **2 of 16,871 wrong residuals (0.0119 %) exceed
3.5 tol**, 23 (0.136 %) exceed 2.75, and **3 of 13,578 true residuals (0.022 %) exceed
the refit's 1.8 support**. So the tail is rare but not empty, and it is the *most*
distant evidence that gets the *best* credit.

### The guard, and what it now refuses

`placement_llr.audit_bundle` separates two kinds of finding:

* **incoherent** — the file contradicts its own edges (declared support/resolution vs the
  edge array, or counts that do not span the table). **Always** an error.
* **unshippable** — internally consistent but carrying a decided defect (no weight
  profile; a rise; an empty/thin bin with positive credit). **Strict** mode
  (`PLACEMENT_FREEZE_GUARD_SUPPORT`) refuses it; tolerant mode **records** it in
  `GUARD_RECORD` and `assert_guard_clean()` fails.

Consequence, pinned by tests: with the strict gate on, **the deployed bundle is refused**,
and the refit loads clean. That is the O1 finding, not a regression.

### The reading, re-run (as required, because `plus_terms` changes with the table)

15°, the four DEV images, 1,166 queries, top-1 strict (ties fail):

| arm | top-1 (pooled) | top-1 per image | geometry AUC (matcher ctx) |
|---|---|---|---|
| deployed table, argmax context (**incumbent**) | 0.105 | 0.282 / 0.206 / 0.141 / 0.026 | 0.640 |
| **refit table**, argmax context | 0.106 | 0.224 / 0.206 / 0.174 / 0.034 | **0.545** |
| deployed table, **assignment context** | **0.198** | 0.269 / 0.429 / 0.418 / 0.059 | **0.710** |
| refit table, assignment context | 0.210 | 0.244 / 0.335 / 0.516 / 0.084 | 0.659 |

**The refit is defensible and it is not free.** It is a wash on top-1 at 15° (0.106 vs
0.105 with the argmax context, +0.012 with the assignment context) and it *costs* 0.07
geometry AUC under the assignment context, because a table fitted on a mixture dominated
by 0° true pairs (median residual 0.021 tol) is a near-step function that is **flat beyond
0.3 tol** — 30 of its 36 bins collapse into one −2.96 plateau after the projection — and
at 15° almost everything is beyond 0.3. The deployed table's softer slope is exactly what
carries a gradient there.

**Decision, taken by the registered objective, not by preference:** the objective is
top-1 at 15° on the FIT images, and the assignment context with the **deployed** table
wins it (FIT 0.354 vs 0.290; 2.4 SE). So the shipped change of this mission is the
context, and the refit table ships **available and validated but not selected**
(`PLACEMENT_LLR_TABLE_M17`, off). **Deviation recorded:** O1's text implies the refit
becomes the operating table; it does not, and the reason is that it fails the registered
objective's guardrail on FIT. The defect it repairs therefore remains in the operating
path until a table is refit *and* verified not to lose top-1 — §7.

---

## O2 — THE CONTEXT LEVERS, RE-MEASURED ON THE REGIME THAT RUNS

### The levers do nothing (measured first — 2 FIT images, 15°, 326 queries)

| arm (levers) | top-1 | geometry AUC (matcher ctx) |
|---|---|---|
| iters 0 (no iteration) | 0.199 | 0.692 |
| iters 2 | 0.233 | 0.693 |
| **iters 5 (incumbent)** | 0.233 | 0.693 |
| iters 8 | 0.242 | 0.692 |
| margin 0.02 | 0.233 | 0.693 |
| max_rivals 0 | 0.233 | 0.694 |
| max_rivals 3 | 0.239 | 0.695 |
| rival_tol 0.05 | 0.242 | 0.690 |
| mutual **on** | 0.199 | 0.692 |

Every lever is inside **±0.005 geometry AUC** and ±0.01 top-1 (one standard error is
0.024 at this n). `mutual` reproduces iters-0 exactly: it freezes every move. The
damped iteration does buy +0.034 top-1 over no iteration at all, and nothing beyond.

### Why — the measurement that explains it (`m17_context.json`)

**Context accuracy**: the share of places whose mapping is the truth partner, truth used
only to *score* the mapping:

| pose | argmax of similarity (what the iteration starts from) | + configuration term | + one-to-one assignment |
|---|---|---|---|
| 0° | 0.962 | **1.000** | **1.000** |
| 5° | 0.355 | 0.265 | 0.277 |
| **15°** | **0.173** | 0.224 | **0.288** |
| 1.15× | 0.529 | 0.400 | 0.443 |

The iteration starts from a mapping that is **wrong 83 % of the time** at 15°, and the
score it optimises is computed *from that mapping*: `context_mapping` closes over `cur`
(`matching/placement_matcher.py:334-339`) and `iterate_context` only accepts a move that
beats the incumbent under the current mapping (`matching/placement_context.py`,
`pick_once`/`iterate_context`). That loop is circular, so no damping constant can help.

### The fix, written in the same step

`matching/placement_context.py::assignment_mapping` — the start is a **bijection**
(`scipy.optimize.linear_sum_assignment`) over a score that adds a **rotation-invariant
configuration descriptor**: a place's neighbourhood as a multiset of
`(log distance, direction in the source's own frame)`. `PlacementVector.direction` is
already measured in the source node's frame (`matching/placement.py:85-104`), so the
descriptor is invariant to a rotation of the image and to a change of scale, and it needs
**no correspondence** — which is what makes it usable before the context exists.

| 15°, four DEV images | incumbent | **assignment context** |
|---|---|---|
| geometry AUC, matcher's context | 0.631 | **0.710** |
| geometry AUC, truth context (ceiling) | 0.902 | 0.902 |
| **a − b gap** | **0.271** | **0.192** |
| top-1 | 0.105 | **0.198** |
| top-1 per image | 0.282 / 0.206 / 0.141 / 0.026 | 0.269 / 0.429 / 0.418 / 0.059 |
| context accuracy | 0.173 | 0.288 |

**Which lever moved it:** the assignment *start* (with its configuration term), and
nothing else. Iterating on top of it changes nothing (it0 = it2 = it5, measured), and the
damped filter's own constants are inert. Acceptance was "geometry AUC at 15° rises from
0.631 toward 0.902": it does, **0.631 → 0.710**, closing 29 % of the a–b gap.

---

## O3 — TOP-1 AT THE TARGET POSE RISES

### The diagnosis first (incumbent, 15°, 1,166 queries, ties fail)

Each failing query is attributed by which term put an impostor first:

| component | share of failures | what it means |
|---|---|---|
| **CONTEXT** | **604 (52 %)** | the geometry term is wrong under the matcher's mapping and would be right under the true one |
| **APPEARANCE** | 438 (38 %) | the appearance similarity ranks an impostor first *and* perfect geometry would not save it |
| GEOMETRY | 2 (0.2 %) | right under the matcher, wrong under truth (a noisy residual) |
| BLEND | 0 | both terms right, the blend wrong |

So the loss is **two-thirds geometry-through-context, one-third appearance**, and the
largest component is the context.

### The fix and its attribution

The assignment context (O2) applied to the largest component:

| 15° | top-1 | CONTEXT failures | APPEARANCE failures |
|---|---|---|---|
| incumbent | 0.105 | 604 (52 %) | 438 (38 %) |
| **+ assignment context** | **0.198** | 541 (46 %) | 378 (32 %) |

**+0.093 absolute (+89 % relative), 1,166 queries, 4 images:** 0.282→0.269, 0.206→0.429,
0.141→0.418, 0.026→0.059 per image. With the refit table instead of the deployed one it is
0.210. **The named change is the bijection + configuration start of the context**, not the
table and not the damped filter.

What remains is 46 % context and 32 % appearance: the appearance term still cannot
separate near-misses at 15° (appearance-only top-1 is 0.059), and that is the honest
ceiling of this mission's change.

---

## O4 — THE OBJECTIVE WAS PRE-REGISTERED AND FOLLOWED

Registered before any number: **top-1 at 15°, place endpoints, ties fail, chosen on the
two FIT images, tie-break by fewer changes, guardrail that geometry AUC must not fall.**

How it was applied:

* O1's table and O2's context were both evaluated; the table **lost** the FIT objective
  (0.290 vs 0.354) and was not shipped — despite being the thing O1 asked for. That
  deviation is reported in §O1 rather than absorbed.
* The context won, moved the guardrail in the right direction (0.631 → 0.710), and is
  shipped behind its flag.
* The tie-break never had to be used: the winner's margin is 2.4 SE at the FIT level and
  8.8 SE on the pooled four-image reading.

**The M16 mistake is not repeated:** no rule was reinterpreted, and the one place where
the objective and a stated acceptance disagree is written down as a deviation.

---

## O5 — CACHE COHERENCE FOR THE SIGNATURE PAYLOAD

The population vector rides in the signature payload and **not** in the graph's node
content, so a graph built with `KP_GRADED_POPULATION` off cannot serve the population
arm — M16 measured this by accident (identical keys and digest, different AUC).

Now (both directions tested, `tests/test_m17_payload_key.py`):

* **payload present → serves.** `population` attached, the population branch runs, no
  misses recorded.
* **payload absent → refuses.** `core/types.py` counted every such comparison in
  `POPULATION_PAYLOAD_MISSES` before, and now records it and — with
  `KP_POPULATION_PAYLOAD_STRICT` — raises `PopulationPayloadMissing` with the reason.
  Tolerant mode is **not silent**: the counter is the ledger, and the tests assert the
  pinned defect (the pair reads 1.0 and calls itself the population arm).
* The cache key half is already true: `KP_GRADED_POPULATION` is in `_FRONTEND_SWITCHES`,
  so the front-end digest moves with it (`test_m14_switch_digest`).

---

## O6 — ONE FRESH READ, SIZED IN IMAGES

**Design (registered in `MISSION17_PREREGISTRATION.md` before any fresh graph existed):**
20 images — 16 new-seed synthetic scenes (seeds 71000+`i`, a range no fit, read or report
has ever used) plus the four photographs M15/M16 never touched (`page`, `gravel`, `moon`,
`retina`); poses 15° and 0°; primary contrast = the deployed blend vs appearance-only on
top-1 at 15°, image-clustered, paired. Planning sd 0.0438 (filled from the four DEV
images, 3 df) → **MDE = 0.660 × 0.0438 = 0.0289**, well under the registered planning
effect of 0.42, so the read **is powered**. The registered correction of the arm
definition (the deployed blend has no separate `w`) is recorded in the pre-registration and
in the manifest.

### Non-vacuity (checked before the verdict)

| counter | value |
|---|---|
| images | **20** |
| images with a decided row (a query where full ≠ appearance) | **20 / 20** |
| images where appearance alone is already perfect | 0 |
| queries at 15° | **4,320** |
| fewest queries from one image | 139 |
| truth pairs whose true partner was admissible | 100 % (the M16 result, re-measured on these graphs) |

### The result

| 15°, n = 20 images | value |
|---|---|
| full placement, top-1 (pooled over queries) | **0.184** |
| appearance-only (w = 0), top-1 | **0.088** |
| **paired per-image difference, mean** | **+0.1088** |
| sd across images | 0.0436 |
| standard error | 0.0097 |
| t (19 df) | **11.17** |
| 95 % CI on the mean difference | **[+0.0884, +0.1292]** |
| MDE at this n | 0.0064 |
| **verdict** | **DETECTION** — the CI excludes 0 by 9 SE and clears the MDE |

**Every one of the 20 images is positive**: +0.035 (`gravel`) to +0.183 (`m17ir_09`).
The synthetic scenes run +0.050 to +0.183 on 16 images; the four photographs run +0.035 to
+0.140. This is the first number in the mission that is not a development number.

**Control at 0° (20 images, 8,899 queries):** full 0.989 vs appearance-only 0.966,
paired mean **+0.0275** (sd 0.0206) — placement helps where appearance is already good and
destroys nothing, which is what a term that is *supposed* to rank should do.

**The honest comparison to the pre-registration's planning effect:** the registered
planning effect was +0.42, inherited from M16's bank; the fresh read measures **+0.109**.
Both clear their MDE. The difference is not a disappointment but a unit change: M16's
+0.42 was measured against a fully tied appearance column with ties counted as successes,
and it was on 5,322 pooled queries from four images. On the strict definition registered
here — ties are failures — the effect is +0.109 with a 95 % CI of ±0.020.

---

## 5. QUALITY BARS

| bar | where it is pinned | goes red on revert because |
|---|---|---|
| 1 flags OFF, defaults byte-identical | `placement_m12_perf.py --check`, 14 pairs: **0 diffs** at the M16 tip (`3b2fc61`, 204.3 s) and **0 diffs** (196.1 s, 14 pairs) — and the *first* attempt at this check is part of the
result: it reported **63 diffs** on a tree whose flags were all off, because the block
that adds the assignment start had been inserted by a pattern that matched two functions
and landed in `rescore_matrix` too, where the name it reads does not exist. The matcher
caught the `NameError` and **fell back to the raw matrix** — silently, as far as any
result file was concerned. That is this programme's recurring bug in its purest form, and
the defaults check is what caught it. The duplicate is removed, and
`NoSilentFallbackTests` pins both halves: `rescore_matrix` must not increment
`FAILURES["fell_back"]`, and only `context_mapping` may read the switch. after the digest change | any default move shifts a float |
| 2 a test that goes red if the fix is reverted | `test_m17_llr_bundle.py` (9): the deployed table's `of(5.0) > of(1.5)` and its missing profile; `test_m17_assignment_context.py` (7): argmax collides, the assignment does not; `test_m17_payload_key.py` (4) | reverting the assignment restores the collision; reverting the table check restores the rising table |
| 3 no silent fallback | `audit_bundle` + `GUARD_RECORD` + `assert_guard_clean`; `POPULATION_PAYLOAD_MISSES` + `PopulationPayloadMissing`; `KP_GRADED_VARIANT` raising on an unknown name | a silent path leaves the counters empty |
| 4 units stated | every result file's `units`; §O1 states every residual in tolunits | — |
| 5 determinism | `test_m17_assignment_context.py::test_it_is_deterministic`; the linear-sum solver is deterministic for a fixed matrix | — |
| 6 manifest + per-cell rows | `write_result` on every file, `rows`/`cells` retained; the assignment sweep keeps every per-image row | dropping rows changes the file |
| 7 switches declared first | `_FRONTEND_SWITCHES` carries all nine new switches; `test_m14_switch_digest` green (it caught the six context switches before I declared them) | an undeclared switch fails the scan |

---

## 6. OPEN PROBLEMS, RANKED

1. **The operating table still rewards distance** (§O1): the refit repairs it and loses
   top-1 on FIT, so the defect is live. The mechanism is now known — a monotone table
   fitted on a 0°-dominated mixture has no gradient beyond 0.3 tol, which is where 15°
   lives.
2. **46 % of the remaining 15° failures are still context** (541 of 1,166): the
   assignment start halves the mapping error (0.173 → 0.288 accuracy) but the bijection is
   still solved on a *per-pair* score; a global transform hypothesis (rotation angle from
   the configuration descriptors, then inlier counting) is the next structural step.
3. **32 % is appearance** (378 failures): appearance-only top-1 at 15° is 0.059, and no
   context can fix a row where the true partner is 17th by similarity. The population arm
   and the graded kernel both peak here; a place-level layout descriptor is the untried
   direction.
4. **`grass` is a different regime** — top-1 0.084 against 0.418–0.516 on the other three
   images at 15°, in every arm. It carries 656 of the 1,166 queries, so it dominates every
   pooled number in this mission.
5. The damped iteration is **inert** on top of a bijection (measured: it0 = it2 = it5). It
   may still matter when the start is worse than a bijection; not measured here.

## 7. THE SINGLE NEXT CHANGE

**Fit the table on the deployment mixture, not on the 0°-dominated pool**: weight the
pooled residuals by pose (or fit per pose and carry three tables) so the tail keeps a
gradient, then re-check the O1 acceptance (0 rises, no thin-bin positive credit, counts
stored) *and* the registered objective (top-1 at 15° on FIT must not fall). The refit
apparatus is already there — `Topology/placement_m17_llr.py` needs a weighted histogram
and a `--pose` selector; the guard and the tests are reusable as they stand.

## 8. DEFAULTS, SUITES

* `python -m unittest discover -s tests`: **71 tests, OK** (was 47 at the M16 tip;
  +9 assignment, +9 bundle, +4 payload, +2 no-silent-fallback).
* `PYTHONPATH=.:Topology python -m unittest Topology.test_placement Topology.test_placement_m3`:
  **29 tests, OK**.
* Determinism: `test_it_is_deterministic` pins the assignment; the fresh read is a new
  tree of graphs, so its numbers cannot be a cache artefact of anything earlier.
* Flags, all **OFF** by default: `PLACEMENT_LLR_TABLE_M17`, `PLACEMENT_CONTEXT_M17` (and
  its five lever constants), `PLACEMENT_CONTEXT_ASSIGN` (+`_LAMBDA`, `_ITERS`),
  `KP_POPULATION_PAYLOAD_STRICT`; the M16 flags are unchanged.
* Every result file carries a manifest and keeps its per-cell rows. The largest is
  `m17_fresh.json` at 4.1 MB (40 cells × every query); nothing is dropped.
