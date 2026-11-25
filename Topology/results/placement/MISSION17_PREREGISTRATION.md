# MISSION 17 — PRE-REGISTRATION (written before any Mission 17 number exists)

Committed alone, before any graph is built, any table is refitted, or any arm is scored.
`git log` order is the evidence: this file precedes every measurement it governs.

---

## O4. THE OBJECTIVE

**Primary objective: TOP-1 AT THE TARGET POSE.**

* **Target pose: 15° in-plane** (`rot_deg = 15.0`). It is the pose where M16's
  acceptance moved and the pose whose top-1 (0.273) is the stated problem.
* **Endpoints: place endpoints** (M15/M16 configuration: `FE_KEYPOINTS`,
  `FE_PLACE_LAYER`, `PLACEMENT_M15_PLACES`, frame from the keypoint anchor). The
  tiling-lens readings are not part of this objective.
* **Bank: the M15/M16 cell bank** — the truth pairs of §M15 item 1, i.e. for each
  query node whose true partner is (i) an admissible candidate in the same row
  (`similarity > 0`, same lens, true partner within `TRUTH_TOL = 1.7009` positional
  tolerance) and (ii) has at least one impostor candidate in that row.
* **Top-1 definition: `max(score | true) > max(score | impostor)`. A tie is a
  FAILURE.** The M16 harness counted appearance ties as successes (`>=`), which
  inflated the "before" column to 1.000 on a fully tied row; that variant is reported
  as a secondary number and is never the objective.
* **Arms:** configurations that differ only in (a) the frozen LLR table
  (`m16` shipped vs `m17` refit) and (b) the context iteration constants
  (`iters`, `margin`, `max_rivals`, `rival_tol`, `mutual`), everything else held at the
  M16 shipped value (`KP_GRADED_SIMILARITY` on, `KP_GRADED_VARIANT="wide"`,
  `KP_GRADED_POPULATION` off, frozen `beta` and `w`).
* **Decision rule (registered now, not after the numbers):**
  1. Evaluate every arm on the **FIT images** (the two synthetic DEV images,
     `syn|m11ir_00`, `syn|m11ir_01`), pooled over their 15° cells.
  2. Report the **READ images** (`real|cell`, `real|grass`) for the winner and for the
     incumbent, but do not select on them.
  3. Choose the arm with the **highest FIT top-1 at 15°**.
  4. **Tie-break, registered in advance:** if two arms' FIT top-1 differ by less than
     one binomial standard error, prefer the one that changes **fewer** things
     (fewer levers, then the smaller lever), and if still tied, the one closer to the
     incumbent. No new tie-break may be invented after seeing the numbers; if none of
     the registered tie-breaks separates them the result is "indistinguishable" and the
     incumbent is kept.
  5. A secondary guardrail, not a selector: **geometry AUC at 15° under the matcher's
     context must not fall** relative to the incumbent (0.631 on the M16 cells). An arm
     that wins top-1 while losing more than 0.02 geometry AUC is reported as a trade-off
     and not shipped, and that decision is recorded as a deviation.

**If a proxy (mean AUC over poses, appearance AUC, similarity AUC) disagrees with the
objective, both are reported and the objective is followed.** That is the M16 mistake
this rule exists to prevent.

---

## O6. THE FRESH READ — DESIGN, SIZE AND MDE (before any fresh graph exists)

**Non-vacuity first.** The read is void if any of these is zero: images with at least
one truth pair at 15°; truth pairs whose true partner is admissible; pairs with at least
one impostor; and images contributing a non-tied top-1 decision.

* **Unit of analysis: the IMAGE.** n = 20 images. Four images cannot carry a useful
  interval, which is why M16's +0.42 is a development number and not a result.
* **Composition, fixed now:** **16 fresh synthetic scenes**, seeds `71000 + i`
  (`i = 0..15`), family IRREGULAR from the Mission-9 builder, named `m17ir_00..m17ir_15`
  — a seed range never used by any fit, read, or report, so the scenes did not exist when
  M15/M16 were fitted; plus **4 real photographs not used by M15 or M16**
  (`page`, `gravel`, `moon`, `retina` from the validation set; M16 used `cell` and
  `grass`). 20 images = 20 clusters.
* **Poses: 15° (target) and 0° (control).** Two poses per image, 40 cells.
* **Primary contrast: full placement (`w = 0.375`, the shipped blend) vs
  appearance-only (`w = 0`) on TOP-1 at 15°**, same candidate rows, same graphs, same
  appearance similarity — only the blend differs, so the contrast isolates exactly what
  placement buys.
* **Test: paired t-test over the 20 image-level differences** (top-1 pooled within an
  image), two-sided, α = 0.05, power 0.80. Per-image spread and the number of images with
  a non-vacuously decided row are reported.
* **MDE (registered formula):** `MDE = (t_{0.975,19} + t_{0.80,19}) · sd_diff / √20
  = 0.660 · sd_diff`, where `sd_diff` is the standard deviation of the per-image
  differences.
* **Planning effect: +0.42** (M16: top-1 0.116 at w=0 → 0.539 at w=0.375, on 5,322 DEV
  queries).
* **Planning sd (filled before a single fresh graph was built, from the four DEV images
  at 15°, arm `llr17`): per-image differences full-minus-appearance-only
  +0.0513, +0.0824, +0.1196, +0.0168 → mean **+0.0675**, sd **0.0438** (3 df).
  Therefore **MDE = 0.660 × 0.0438 = 0.0289**.
* **Verdict on power, taken now:** MDE 0.0289 is far below the registered planning
  effect (+0.42) and below the DEV-measured gain (+0.0675), so the read **is powered**
  for a ~+0.03 effect at its registered size. It is also recorded that the *effect this
  mission actually has to detect* is far smaller than the +0.42 the pre-registration
  inherited from M16's bank: the DEV gain on this definition is +0.0675, because M16's
  +0.42 was measured against a fully tied appearance column and with ties counted as
  successes. Both are stated; the read judges the smaller one.
* **Verdict rule, registered now:** if `MDE > 0.42` the read is **not powered**; the
  result is then reported as an **estimate with its interval**, never as a detection, and
  the number of images that would have been needed is stated. If `MDE ≤ 0.42` and the
  observed paired difference clears the MDE, the read is a **detection**.
* **Full manifest**: code digest, flags, endpoints, frame, weight, image list with its
  seed or photograph name, and the per-image rows.

---

## O1, O2, O3, O5 — WHAT WILL BE JUDGED (acceptance, restated as numbers)

| # | acceptance | how it is reported |
|---|---|---|
| O1 | refit table: 0 rises, 0 empty bins with positive credit, counts stored, guard refuses support/resolution/weight-profile mismatch | old and new tables side by side; the exposure that changed (M16: 0.15 % of wrong pairs beyond 3.5 tol units); geometry reading re-run and reported before/after |
| O2 | geometry AUC at 15° (matcher's context) rises from **0.631** toward the **0.902** ceiling | the a–b gap before/after, and **which lever** moved it, with the per-lever sweep table; if no lever moves it, the mechanism is named with a code citation and a measurement |
| O3 | top-1 at 15° rises, attributed | the loss on the incumbent decomposed into appearance / geometry / context / blend, as shares of the failing queries, then the largest component fixed |
| O5 | payload: the key covers it, or a graph without it refuses to serve a population arm | both directions tested: payload present → serves; payload absent → refuses (strict) / records (tolerant) |

All numbers carry the image count and the per-image spread. Units: any residual,
tolerance or bin width is in **tolunits** (`payload_residual` is tolerance-normalised and
dimensionless; no "px" claim is made anywhere in this mission).

---

## CORRECTION, recorded before the fresh read was scored

The primary contrast above is written as "full placement (`w = 0.375`, the shipped blend)
vs appearance-only (`w = 0`)". **The deployed blend has no separate `w`**: `context_mapping`
computes `(1 - beta) * sim + beta * plus_score` (`matching/placement_matcher.py:339`), and
`plus_score` already carries the fitted geometry weight through `a_exist`/`a_geom`. The
`w = 0.375` in Mission 16's bank was a *hypothetical knob that was never wired*; M16's
"+0.42 at w = 0.375" was therefore a profile value, not the deployed reading.

What is measured here is the deployed blend against appearance-only. **The contrast is
unaffected**, because the appearance-only arm is `score = sim` under both readings (at
`w = 0` the geometry term drops out; at `beta` the appearance term is the only term left
when `plus_score` is replaced by its neutral 0.5). The arm is renamed in the report, and no
number was chosen after this correction: it was written before the first fresh graph
existed.
