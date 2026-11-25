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
