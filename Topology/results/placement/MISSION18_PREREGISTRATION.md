# MISSION 18 — PRE-REGISTRATION (written before any Mission 18 measurement)

Committed alone, before any Mission 18 graph is built, any table is refitted, or any arm
is scored. `git log` order is the evidence.

---

## THE GOVERNING RULE, RESTATED SO IT CANNOT BE QUIETLY BROKEN

A change that is correct and loses the objective is **not finished**. It is neither
reverted nor shipped. The loss is a new defect to diagnose and repair. M18 exists to make
the defensible table **and** the ranking number true at the same time; if that outcome
cannot be met, the report carries the measured floor, the mechanism and every attempt with
its numbers.

---

## O4 FIRST: THE MEASUREMENT DISCIPLINE (because every other number depends on it)

* **Every rate is reported per image with its spread and its image count.** A pooled
  figure may appear only beside the per-image table that produced it, never alone.
* `grass` carries 656 of 1,166 queries at 15° and reads 0.084 where the other three read
  0.418–0.516. **No headline number in this mission is allowed to be dominated by it
  silently**: every headline is given pooled, per-image, and **excluding `grass`**.
* Units: `payload_residual` is tolerance-normalised and dimensionless — every residual,
  tolerance and bin width below is in **tolunits**, and no px claim is made anywhere.

---

## THE REGISTERED OBJECTIVE AND GATES

**Objective (selector): highest FIT top-1 at 15°, ties fail, on the two synthetic DEV
images.**

**Hard gates — a configuration that fails any gate cannot be shipped, however well it
scores:**

| gate | threshold | from |
|---|---|---|
| G1 defensibility | 0 rises; no empty/thin bin with positive credit; per-bin counts stored; LLR(5.0 tol) < LLR(1.5 tol) strictly | O1 |
| G2 non-flat where 15° lives | the 15° true residuals must not fall mostly in a plateau; the table's slope across that region is reported and must be non-zero | O1 |
| G3 FIT top-1 at 15° | **≥ 0.354** (the deployed-table incumbent, M17) | O1 |
| G4 geometry AUC at 15° | **≥ 0.710** (M17 incumbent) | O1/O2 |
| G5 no regression at 0° | context accuracy at 0° stays **1.000**; top-1 at 0° ≥ 0.980 | O2 |
| G6 defaults | `placement_m12_perf.py --check` 0 diffs, and no fallback counter increments at defaults | bar 1 |

**Tie-break, registered in advance:** if two gate-passing configurations' FIT top-1 at 15°
differ by less than one binomial standard error, prefer the one that changes fewer things
(the smaller lever, then the shorter diff). No post-hoc tie-break may be invented; if the
registered ones do not separate them, the result is "indistinguishable" and the incumbent
is kept.

**Deviations are recorded, not reinterpreted.** If an outcome's acceptance and the
objective disagree, both are reported and the objective is followed.

---

## O1 — WHAT "COMPETITIVE" WILL BE MEASURED ON

Before any refit, and reported per pose:

1. the distribution of **15° TRUE residuals** (median, quartiles, q90), because "non-flat
   where it matters" is defined by where they actually lie;
2. the fraction of 15° true pairs falling in a **plateau** of the candidate table (a run of
   bins with equal LLR), and the fitted **slope** across the interquartile range of those
   residuals (Δ LLR per tolunit);
3. the same for the deployed table and for the M17 refit, so the comparison is exact.

Routes to be measured, each reported with its number whether it works or not: pose-weighted
pooling; per-pose tables chosen by a truth-free pose estimate; a wider support with a
count-scaled minimum; and a **parametric monotone decay** (`LLR(r) = a + b·exp(-r/τ)`,
`b ≥ 0`) fitted to pose-weighted bin log-ratios — which cannot rise by construction and
preserves slope everywhere, in place of the PAVA projection that collapsed 30 of 36 bins
into one value.

---

## O2 — THE CONTEXT OUTCOME

Acceptance: **context accuracy at 15° above 0.288**, **geometry AUC above 0.710** toward
the 0.902 ceiling, and the **a−b gap below 0.192**; no regression at 0° (1.000 today).

The named candidate is a **truth-free rotation consensus**: each candidate pair implies a
circular bin shift between the two signatures, true pairs share one shift, and impostors
do not — a Hough vote on the shift, then a bijection restricted to the modal shift. It is
measured against the M17 assignment as the incumbent, per image, and it is allowed to fail
so long as its failure is measured.

---

## O3 — THE APPEARANCE OUTCOME, AND ITS FLOOR

First, **where the true partner sits**: median rank in the similarity row, and the share
outside the candidate cap. If it is outside the cap, no context can reach it and the fix
belongs in the descriptor or in candidate generation; that is a legitimate floor, and it
must be reported with the number.

Acceptance: appearance-only top-1 at 15° **above 0.059**, or a measured floor with the
mechanism and every attempt.

---

## O5 — THE FRESH READ, SIZED IN IMAGES, DESIGN FIXED NOW

* **n = 20 images**, all synthetic: seeds **81000 + i** (`i = 0..15`, 16 scenes) plus four
  more from the same new range at `82000 + i` (`i = 0..3`). The photographic pool is
  **exhausted** — all six photographs in the repository have been used by M15/M16/M17 — so
  this read is synthetic-only; that limitation is stated here, before it can look like an
  excuse.
* Poses: **15° (target)** and **0° (control)**.
* Primary contrast: full placement vs appearance-only on **top-1 at 15°, ties fail,
  paired, image-clustered**.
* **Planning sd = 0.0436**, taken from the M17 fresh read of the same design (19 df) →
  **MDE = 0.660 × 0.0436 / √20 = 0.0064**. Planning effect: **+0.109** (the M17 fresh
  measurement). MDE 0.0064 ≪ 0.109, so the read **is powered**; if that ever changes, the
  result is an estimate and is labelled one.
* Non-vacuity counters reported before any verdict: images with a decided row, queries,
  smallest per-image query count, images where appearance alone is already perfect.
* Full manifest: code digest, flags, endpoints, frame, weight, image list with seeds,
  per-image rows.

---

## WHAT WOULD COUNT AS FAILURE

If G1–G5 cannot be met simultaneously, the mission reports the **floor** — the best
gate-passing number with its CI — the mechanism that blocks it, and every route measured
with its number. Silence, a revert, and a pooled-only headline are the three things this
mission may not do.

---

## ADDENDUM: O5, the fresh read -- REGISTERED BEFORE THE READ (2026-11-18, this session)

Everything below was written after the FIT measurements and **before** the fresh read was
run, so the read cannot be shaped by its own outcome.  It registers the frozen design, the
sample, the statistic and the MDE; the numbers come later, in the report.

### The frozen design (what is being read)

| switch | value | what it is |
|---|---|---|
| `PLACEMENT_CAND_CAP` | 24 | candidate generation, truth-free; **not** 6, the deployed cap |
| `PLACEMENT_LLR_TABLE_M18` | on | the pose-balanced table (`matching/placement_frozen_params_m18.json`) |
| `PLACEMENT_CONTEXT_ASSIGN` | on, lambda 0.5, iters 0 | the M17 shipped bijection start, no damped sweeps |
| `PLACEMENT_CONTEXT_CONSENSUS` | **off** | measured and rejected at FIT: on a real photograph its mode is wrong (0.04 against a true 0.26) |
| everything else | as M16/M17 shipped | `M16` flag set, `KP_GRADED_POPULATION` |

Defaults are untouched: with every switch at its default the pipeline still reads the
deployed table with cap 6, and `placement_m12_perf.py --check` must report 0 diffs.

### The sample

`fresh20` -- the 20 images `Topology/placement_m17_fresh.tags()` builds: 16 synthetic
scenes at seeds `81000 + i` and the four photographs `page`, `gravel`, `moon`, `retina`.
None of them entered the M18 table fit, whose bank is the four DEV images.

STATED LIMITATION: this set was read once before, under M17's design (report O6).  It is
held out from the M18 *design* but it is not virgin to the programme, and the reuse can
only help through image selection, which was fixed before either read.

### The statistic

Pose `rot15` only (the mission's primary pose).  Per image `i`:

    d_i = top1_full(i) - top1_appearance_only(i)

paired, image-clustered, n = 20.  Report the mean, the sd, the 95 % CI (t, 19 df), the
median, the full per-image vector, and the count of ties.

ACCEPTANCE: mean > 0 **and** the 95 % CI excludes 0 **and** every `d_i >= 0` (a tie is
not a win: ties fail, so the count of ties is reported and no claim is made on a tie).

### The MDE, stated before the read

Planning sd of the paired difference = **0.0436** (M17's O6 read, the same 20 images and
the same pairing).  Two-sided alpha 0.05, power 0.80, n = 20:

    MDE = (1.959964 + 0.841621) * 0.0436 / sqrt(20) = 0.0273

Planning effect, for comparison, is the FIT measurement (+0.403, 0.462 - 0.059).  The MDE
is an order of magnitude below it, so the read is not vacuous *if* the effect survives the
image change.  If the observed sd differs from the planning sd, the MDE is recomputed from
the observed sd and both are reported.

### Non-vacuity counters, checked before any verdict

* queries per image > 0 (report n per image; a cell with 0 queries is a failed read, not a
  zero);
* context rows counted > 0 (the context accuracy is not measured on nothing);
* the appearance-only baseline > 0 in at least half the images (if appearance alone never
  scores, the difference is a comparison against a constant zero and the read is vacuous);
* no `float("inf")` / NaN may stand in for a rate -- a rate with no denominator is `None`
  and is reported as such.

### Splitting the report (never pool regimes)

Pooled over all 20, then the 16 synthetic and the 4 photographs **separately**, then per
image.  The grass-type photograph (`gravel`) is also shown on its own wherever it differs.
