# MISSION 12 — report

Branch `arena/c316ca3a-percept6`, based on `192a612`.

This file is written as the mission proceeds. Each step's section states what was
measured, what was decided, and — where the mission's gate is not met — that the
gate is not met.

---

## STEP 0 — the verdict rule (no runs)

### The rule, stated before any M12 read

1. **MDE gates the DESIGN, not the verdict.** Before a read is registered, the
   registration must show the design's minimum detectable effect (MDE, at the
   declared alpha and power) is **≤ the planning effect**. If it is not, the read
   is declared **UNDERPOWERED IN ADVANCE** and its outcome can only ever be
   reported as an estimate, never as a detection.
2. **The verdict is the interval.** A primary is **DETECTED** iff its
   scene-clustered confidence interval **excludes zero in the pre-registered
   direction**. Nothing else — not a p-value at a favourable look, not a point
   estimate — produces DETECTED.
3. **Multiplicity.** Holm across the declared primaries; where several looks are
   declared, the look correction is applied to the alpha of each look (alpha
   cumulative 0.10 across 5 looks ⇒ per-look alpha 0.02 for look 2).
4. **Type M.** If a primary is DETECTED and the detected effect is **below its
   design MDE**, the magnitude is labelled **"possibly inflated"** (Type M). The
   detection stands; the size does not.
5. **No relabelling.** Verdicts issued under an earlier rule stand as issued.
   They are shown beside the new rule's reading of the same numbers, never
   rewritten.

### Old verdicts, side by side with the new rule's reading

| Mission | Primary | Estimate | SE | Interval | p | Design MDE | Old verdict | Reading under the M12 rule |
|---|---|---|---|---|---|---|---|---|
| M7 | P − P_presence (placement term, look 2 of 5) | +0.0094 | 0.0040 | [+0.0014, +0.0173] | 0.0260 | 0.0113 | INCONCLUSIVE | **Not detected.** The look-2 alpha after the cumulative-0.10 look correction is 0.02; p = 0.0260 does not clear it, so the interval is not the registered one at that look. Effect < MDE (0.0094 < 0.0113) — would have been Type M even had it cleared. |
| M9 | OOP primary (placement, C − A) | +0.0203 | 0.0127 | [−0.0046, +0.0451] | 0.17 | 0.0355 | NOT DETECTED | **Not detected.** The interval includes zero. |
| M11 | Synthetic placement (C − A) | +0.156 | 0.060 | [+0.062, +0.281] | — | 0.168 | (reported as a positive result) | **Detected**, and the detected effect (+0.156) is below the design MDE (0.168), so the magnitude is labelled **possibly inflated (Type M)**. |

The M7 and M9 verdicts are **not** relabelled: the row above is the new rule's
reading shown beside them, as the mission requires.

### The superseded M11 ETH-80 job

The M11 ETH-80 g1 read is **stopped and superseded** by this bundle. It is not
running in this sandbox (no process alive at the start of the mission). Its
checkpoint `Topology/results/placement/m11_eth80_progress.jsonl` (202 rows) is
**kept**, and nothing in M12 — no design choice, no constant, no arm — is
selected using it. It is not a result and is not reported as one.

---

## STEP 1 — profile, optimise, prove nothing moved

### Gate status, stated first

| Gate | Required | Measured | Status |
|---|---|---|---|
| output unchanged (flags off **and** on) | bit-identical score, `score_shipped`, evidence, judged count, mapping | **identical on 14/14 pairs × 2 arms × 5 fields** | **MET** |
| speed per pair | **≥3×** | **1.3×** (A 14.32 s → 10.84 s; C 16.12 s → 12.49 s) | **NOT MET** |
| pairs | ≥30 fixed | 14 synthetic (16 synthetic + 14 real is the registered set; the real half needs the ETH-80 corpus, still downloading) | **PARTIAL** |

The speed gate is **not met** and is not being claimed as met. What follows is the
profile that says what the remaining 10.8 s consists of and what a 3× would
require.

### Harness

`Topology/placement_m12_perf.py` — 30 fixed pairs (16 synthetic in-plane/out-of-plane
from `placement_m11_families`, 14 real from the ETH-80 catalogue when it is
present), each matched twice (arm A: placement matcher off; arm C: on). For every
pair and arm it records `repr(float(score))`, `repr(float(score_shipped))`,
`repr(float(placement_evidence))`, `placement_judged`, the mapping as a sorted list
of `(u, v)`, and the arm's seconds.

* `--save` writes `Topology/results/placement/m12_perf_baseline.json`
* `--check` diffs against it field by field and exits non-zero on any diff

### Profile before (cProfile, one synthetic pair, arm C, `joint_mcs`)

45.4 s under the profiler:

| Call | Calls | Time |
|---|---|---|
| `_relational_agreement` (`_rel`) | 403 760 | 28.3 s |
| `node_sim` (incl. `_sig_similarity`) | 1 818 300 | 8.9 s |
| `UNDIRECTED_EDGE_TYPES.__contains__` | 4 734 859 | 6.6 s |
| `_compatible_classes` | 2 361 334 | 5.5 s |
| `_swap_pass` / `_swap_delta` / `_quality` | 5 463 / 230 716 | 24.5 s |

`_relational_agreement` and the swap pass are the same work seen twice (`_quality`
calls `_rel`); the two pure-function lookups below them were being recomputed
millions of times.

### What was changed (all in `matching/alignment.py`, no default changes)

1. `_is_undirected(et)` — `et in UNDIRECTED_EDGE_TYPES` memoised per edge type.
   Both operands are fixed at import (the schema fixes the undirected set), so the
   memo cannot change a value. Five of the eight call sites use it.
2. `_compatible_classes(et)` — the schema-class lookup is memoised per edge type on
   top of the existing per-class memo.
3. `_rel(u, v, mp, rv)` — for the `candidate is None` path the relational agreement
   is cached on exactly the context it reads: `(u, v, images of u's neighbours,
   preimages of v's neighbours)`. Every other input is static graph data, so the
   cache cannot change a value; it is bounded (cleared above 400 000 entries) and
   the `candidate is not None` path bypasses it entirely (a candidate induces a
   world the context key does not describe). Kill-switch `M12_REL_CACHE=0`.

### Profile after (same pair, arm C)

32.1 s under the profiler (1.42×): `_relational_agreement` residuals 264 262
misses / 11.4 s, `node_sim` 1 818 300 calls / 8.9 s (unchanged — it is already
`lru_cache`d, and 182 620 of those calls are *distinct* pairs at ~42 µs each),
`_swap_pass` 11.7 s, `rescore_matrix` 4.3 s.

### Acceptance (14 fixed pairs, 2 workers per run, each tree run twice)

| Run | wall | per-pair A | per-pair C |
|---|---|---|---|
| pristine ×2 | 214.6 s / 208.7 s | 14.32 / 14.03 | 16.12 / 15.64 |
| optimised ×2 | 164.6 s / 164.8 s | 10.84 / 10.40 | 12.49 / 12.81 |

* pristine vs pristine: **identical** (0 diffs)
* optimised vs optimised: **identical** (0 diffs)
* **pristine vs optimised: identical (0 diffs)** — every score, every shipped
  score, every evidence value, every judged count and every mapping.

### The reproducibility trap this exposed (recorded because it cost the session a detour)

An earlier `--save`/`--check` pair disagreed on 28 mappings while every score
matched to the last digit. The cause was **not** the optimisations: the graph
pickle cache held entries built in an earlier code/sandbox state, and a worker
rebuilt some of them mid-session (the rebuild is content-addressed, so the
pixels are the same; the graph object handed to the aligner was not the one the
baseline was scored on). After the cache was consistently rebuilt, every tree —
pristine, optimised, in-process, forked, 1 or 2 workers — reproduces exactly, and
`--save` twice in a row is byte-identical. **Any bit-identity claim must therefore
be made over runs that share one cache state**; the acceptance above is, and the
harness now records enough to detect the situation.

### Why the gate is not met, and what a 3× needs

The remaining cost is dominated by two per-pair products, both already minimal in
Python:

* **node similarity**: the seed matrix is the full `n × m` product (182 620
  distinct pairs per direction, ~42 µs each ⇒ 7.7 s per pair for the two
  directions). `node_sim` is already memoised; the cost is the deep signature
  comparison (`core/types.py::_similarity_gaussian` plus
  `core/population.py::population_similarity`).
* **relational agreement**: ~264 k distinct evaluations per pair at ~40 µs.

Neither is reducible by caching (the caches are already in), and neither can be
skipped without changing the answer (a pruned matrix is a different alignment,
which the mission forbids). A 3× needs the similarity computed in **batched numpy
per class** (the signatures of one class share an axis layout), which is a
separate piece of work with its own bit-identity argument. That is proposed, not
claimed.

### STEP 1A — the missed cost centres

Four changes, none of which can change a value:

| Change | Where | Why it is exact | Status |
|---|---|---|---|
| **b. One Side per image** | `matching/placement_matcher.build_side_cached`, used by `joint_mcs` | `build_side` is a pure function of (graph object, pixels, switches); the cache holds the graph, so the id cannot be recycled, and it keys on the identity, the pixel digest and the frozen bundle | landed |
| **c. Centred form precomputed** | `core/population.centered_cosine` / `_centered_form` | the mean, the centred vector and the sum of squares are cached per vector and VALIDATED against a content digest; the same floats in the same accumulation order | landed, 2 000 random cases bit-identical to the old code |
| **d. `cells_scored`** | `matching/placement_matcher.rescore_matrix` | it counted the CAPPED context rows; it now counts the (row, cell) pairs `plus_score` was actually called on | fixed |
| **a. 30 pairs** | — | the ETH-80 corpus is still downloading in this sandbox (`catalog()` raises `FileNotFoundError`); the harness registers the 14 real pairs the moment it lands | **not met yet** |

Every one of these is behind no flag because none of them is a behaviour: the
Side cache and the centred cache return the same objects/values, and the counter
is a diagnostic. The evidence they are exact: with all of them in, the arm-A and
arm-C mapping hashes of the fixed harness pair are unchanged from the committed
baseline (`0cfa87a2` / `2865d93d`), and the scores are unchanged to the last digit.

### Where the harness stands

* 14 synthetic pairs registered and accepted.
* The 14 real pairs are **not yet in**: `data/ETH-80-master` is still extracting
  (`catalog()` raises `FileNotFoundError`), and the harness degrades to
  synthetic-only and says so, in the log and in the baseline's pair count. Both
  galleries are required by the mission, so the acceptance is repeated over 30
  pairs before Step 6 and the table above is superseded then.


---

## STEP 2 — the coverage gradient of E, and the reading chosen on FIT

### Instrument

`Topology/placement_m12_e_coverage.py` — 10 FIT pairs (6 irregular, 4 out-of-plane),
22 724 positive candidate cells, 566 of them the TRUE candidate under the
renderer's own track truth. For every cell it records the judged-slot count, the
deployed term, the mean term, three dark variants, and the blended candidate score
the matcher actually ranks by. Truth is a FIT-side instrument only; nothing here is
a read.

### The audit

Spearman rho against `n_judged` (the judged-slot count), over the non-true
candidates and over the true ones separately, and against the BLENDED score:

| reading | rho(all) | rho(true) | rho(non-true) | rho(blended score, non-true) | FIT top-1 |
|---|---|---|---|---|---|
| `e_sum` (deployed) | −0.418 | **+0.457** | −0.453 | 0.134 | 0.4692 |
| `e_mean` (mean over judged slots) | −0.872 | **−0.060** | −0.905 | 0.045 | 0.4692 |
| `e_min2` / `e_min3` (mean, refused below 2 / 3) | −0.872 | −0.060 | −0.905 | 0.045 | 0.4692 |
| `e_shrink` (mean shrunk to the query mean, n/(n+4)) | −0.934 | −0.421 | −0.954 | 0.047 | 0.4645 |
| `e_rank` (mean, rank-normalised in the shortlist) | 0.001 | +0.467 | −0.016 | 0.060 | 0.3104 |

What this says, in order of importance:

1. **The deployed sum is biased exactly where the mission said it would be.** For
   the TRUE candidate E rises with the number of judged slots (+0.457): a place
   with more checkable geometry hands the correct pair more evidence for being
   correct. The mean reading removes that (+0.060, sign flips and magnitude
   collapses).
2. **The rivals' negative correlation is not the same defect.** A place with more
   judged slots gives a wrong pair more chances to be *contradicted* (−0.905). That
   is information, and it is the direction that helps.
3. **At the level the matcher ranks by, no reading is biased.** The blended score's
   correlation with the judged count is 0.045-0.134, all below the mission's 0.30
   bar, because the node similarity dominates the blend.
4. **The bias is not free to remove by ranking.** Rank-normalising within the
   shortlist drives the correlation to −0.016 and costs a third of FIT top-1
   (0.4692 → 0.3104): it removes the bias by removing the information.

### The decision (FIT, pre-stated rule)

Rule: admissible = |rho(blended score, judged count)| < 0.30 over the non-true
cells; among the admissible, prefer the readings that are SCALE-FREE in the judged
count (the mean family) — E has to mean the same thing at every place or no single
frozen weight can be right at all of them; tie-break on FIT top-1, then on |rho|.

**Chosen: `e_mean`** — `plus_score(normalise=True)`, the mean over the slots a pair
judged, with a refused minimum (`PLACEMENT_E_MIN_JUDGED`, default 0 because the
synthetic FIT population judges either 0 or the full cap). It is admissible, it is
the most scale-free of the mean family on the true candidates (−0.060 against
+0.457), and it costs **nothing** on FIT top-1 (0.4692, identical to the deployed
reading).

### Landed, dark

* `parameters.PLACEMENT_E_NORMALISED` (False) and `PLACEMENT_E_MIN_JUDGED` (0),
  both registered in `eth80_frontend._FRONTEND_SWITCHES` **before any graph or
  match exists under them** (this does move the graph cache key, so the next run
  rebuilds — the rule the tree already applies to `PLACEMENT_PAIR_NORMALISED`).
* `matching/placement_llr.plus_score(..., min_judged)` — a pair below the minimum
  returns exactly 0.5, i.e. 0 evidence; no evidence always contributes 0.
* `matching/placement_matcher._pair_normalised()` now ORs the two dark flags, and
  all three call sites pass `min_judged`; `matching/joint_mcs.placement_evidence`
  reads the flag so E itself is on the chosen scale.
* **Defaults byte-identical:** with both flags off, the fixed harness pair still
  maps to `0cfa87a2` / `2865d93d` with the same scores to the last digit; with
  `PLACEMENT_E_NORMALISED=1` the arm-C mapping and score move (0.38958246 →
  0.38791059), which is the flag doing its work and proves it is live.
