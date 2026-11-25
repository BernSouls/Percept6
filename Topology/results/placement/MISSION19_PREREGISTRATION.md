# MISSION 19 — PRE-REGISTRATION (written before any number, any graph, any match)

Branch `arena/c316ca3a-percept6`, from the M18 tip `e796ae4`.  This file is committed
**alone**, before a single M19 graph exists or a single M19 number is produced.

## The rule that governs this mission

A change that is CORRECT but loses a number is **not finished**.  It is not reverted and
not shipped; the loss is a new defect to diagnose and repair.  M18's pattern: the clean M17
table lost top-1 because it was flat where 15° lives, and the answer was a table that is
both defensible and sloped — not a return to the broken one.

**The new trap, registered here so it cannot be misread later:** if the descriptor improves,
`full − appearance-only` **shrinks**.  That is a success, not a regression.  The M18
baselines are therefore registered explicitly in §0 and every O5 comparison is against a
named baseline, never against a difference.

## §0 Baselines (M18, 4 DEV images, 15° unless stated; these are the only "before" values)

| quantity | value | n | source |
|---|---|---|---|
| full placement top-1 | 0.462 | 1,166 q | `m18_eval_tbhyb.json` |
| context accuracy | 0.459 | 1,166 q | same |
| geo AUC (ceiling 0.902) | 0.796, gap 0.106 | 1,166 q | same |
| appearance-only top-1 | 0.059 | 1,166 q | same |
| median true-partner rank | 11 | 1,166 q | `m18_diagnose.json` |
| share outside cap 6 | 0.696 | 1,166 q | same |
| per-image top-1 (15°) | .410 / .518 / .668 / .402 | m11ir_00 / m11ir_01 / cell / grass | same |
| 0° full top-1 | 0.995 | 2,028 q | same |
| grass 15° top-1 | 0.402 | 656 q | same |
| grass median similarity row | 586 places | — | `m18_diagnose.json` |
| O5 (M18 fresh read, 20 images) | mean +0.2564, CI [+0.2145, +0.2983], 20/20 wins | 4,320 q | `m18_o5_fresh.json` |

Everything M19 reports is measured against these, per image with its `n`, never pooled-only.

## §1 O1 — the descriptor's ordering

**Acceptance (all three, on DEV at 15°):** appearance-only top-1 **> 0.059**; median
true-partner rank **< 11**; the share of true partners outside the **operating cap** falls
below 0.696.

**Guardrails (all three must hold in the same run):** 0° full top-1 **≥ 0.995**; 15° full
top-1 **≥ 0.462**; geo AUC **≥ 0.796**.  A candidate that meets the acceptance but breaks a
guardrail is a defect to diagnose, not a result.

**Mechanism requirement.**  "The descriptor is weak" is not a mechanism.  The diagnostic to
run first, truth-free in the matcher and truth-read only for measurement, is the **occupancy
of the five coarse slots** (theta/12, coherence/4, scale, kind, dim): for each image, the
number of reference places sharing a query's quantised signature, per image and per lens,
plus the same for the true partner.  The hypothesis to test: the ordering cannot improve
while many places share a signature by construction, and a node descriptor that does not
collapse onto those slots breaks the ties.

**Routes, registered (the winner is chosen by measurement, the order is the tie-break):**

* **R1** the rotation-invariant **configuration descriptor** already computed for the
  bijection start (`config_descriptor` / `config_similarity`) used as a **node descriptor**
  to re-rank the top-**K** candidates of each row, `K ∈ {24, 72}`, weight `w ∈ {0.25, 0.5}`:
  `rank_score = (1−w)·sig + w·config`.  It reads each node's own star (sorted log
  distances and sorted source-frame directions) and needs no mapping, so it is truth-free
  and it is not a placement term.
* **R2** the same descriptor over the **whole row** (no top-K restriction), same weights.
* **R3** if R1/R2 fail: the measured floor, with the occupancy diagnostic as the mechanism
  and every attempt's number.

**Tie-break between candidates that meet the acceptance and the guardrails:** higher geo
AUC, then higher 15° full top-1, then smaller K, then smaller w.

**The hypothesis worth testing (reported, not required):** if a better start improves the
bijection, do context accuracy and geo AUC rise with it?

## §2 O2 — dense rows become reachable (grass)

**Acceptance:** grass top-1 at 15° **> 0.402**, produced by a **row-size mechanism**, with
**no other image regressing by more than 0.02** absolute.

**Registered mechanisms (one is chosen by measurement, order is the tie-break):**

* **M1** a cap that scales with the row: `cap_eff = clamp(round(k·row_size), 24, cap_max)`
  with `k ∈ {1/16, 1/8}` and `cap_max ∈ {48, 64}`.
* **M2** partial deduplication of near-identical places: collapse candidates whose
  **quantised signature is identical** to an already-taken candidate, keeping the lowest
  node id, so the row stops being 586 copies of the same five slots.
* **M3** a cheap prefilter that keeps recall (reported only if M1/M2 both fail).

**Denominator discipline (mandatory):** the truth denominator — queries whose true partner
is inside the reference population — is reported **before and after** any dedup, per image.
A shrinking denominator may never be read as progress.  Dedup that removes the true partner
from the population is a recall loss and must be counted as such.

**The three statistics M18 left open, per image:** texture **period** in pixels (autocorrelation
of the luminance, peak lag), **coherence distribution** (per-place `coherence` slot: median and
IQR), and **absolute place spacing in pixels** (median nearest-neighbour distance between
places).  Reported per image, characterising the regime rather than guessing it.

## §3 O3 — the soft consensus, settled by registration

`PLACEMENT_CONTEXT_CONSENSUS_SOFT` with `_BONUS` 0.15 is carried as a **registered arm**, not
as a design change.  Its contrast is **paired within the same design**, on the same fresh
images: soft-on minus soft-off, per image.

**Acceptance:** the paired mean > 0 with a 95 % CI excluding 0, **and no image losing more
than 0.05** absolute full top-1.  Otherwise it is **rejected with the number**.

It is not re-litigated on DEV.  M18's DEV measurement (+0.016 top-1, geo +0.003, spread
0.108 → 0.087, one cell −0.027) stands as the planning input.

## §4 O4 — the cap cost, measured where it is paid

**Acceptance:** cost per image **inside the placement harness** at the caps actually used
(6, 24, and O2's adaptive cap), with the judged-slot count per cell, medians over repeats.
Reported in seconds per cell and in judged slots per query; "the cap is free" is only a claim
about the pipeline that runs.

## §5 O5 — one fresh read, two registered primaries

**Sample (registered before any graph exists):** 16 synthetic irregular scenes at seeds
**91000 + i** (a range no fit, read or report has used) built by the unchanged Mission-9
builder, plus **four photographs no read has touched**: `brick` (texture), `coins` (object
with many similar parts), `camera` (hard-edged object), `hubble` (scene, low texture).
20 images total.  Pose `rot15` (primary) with `rot00` reported as a guardrail.

**Arms (all four in one run):**
* **A0** the M18 frozen design (cap 24, M18 table, assignment λ 0.5 iters 0, consensus off);
* **A1** this mission's design (O1's winning descriptor + O2's winning row mechanism);
* **A2** A1 + soft consensus (`_BONUS` 0.15);
* **A3** the new appearance-only arm: the descriptor of A1 without any context/geometry.

**Primaries (Holm over the two, α = 0.05 family-wise):**
* **P1** `A1 − A0`, paired per image: does this mission improve the deployed design?
* **P2** `A1 − A3`, paired per image: does placement still add over the *new* descriptor?

**Stated in advance:** P2 may legitimately **shrink** if O1 succeeds.  A smaller P2 with a
larger P1 is a success; P2 is a check that placement still adds, not a competition for size.

**Sizing, in IMAGES.**  The design that may enter the read is gated: it must show **≥ +0.05**
absolute top-1 at 15° on DEV over 0.462 (the planning effect), so the read is sized for a
0.05 effect, not for the 0.016 a marginal change would need.  Planning sd **0.06** (two
variants of the same pipeline on the same images differ per image by about this much; M18's
fresh read observed a paired sd of 0.0896 for a much larger design change).  Two-sided
α 0.05, power 0.80, n = 20:

    MDE = (1.959964 + 0.841621) * 0.06 / sqrt(20) = 0.0376   <   0.05  (planning effect)

If the DEV gate is not met, the design does not enter the read as a primary: O5 then reports
that gate failure as the outcome and reads only P2 and the soft-consensus contrast.

**Non-vacuity, checked before any verdict:** queries per image > 0; appearance-only > 0 in at
least half the images; context rows counted on every image; no rate with no denominator
(reported as `None`, never as 0 or `inf`).

**Ties fail:** a tie in a paired difference is not a win; the tie count is reported.

## §6 Quality bars (unchanged, and how each is met)

1. Every new switch **OFF by default**, declared in `parameters` and in
   `eth80_frontend._FRONTEND_SWITCHES` **before** any graph is built under it;
   `placement_m12_perf.py --check` **0 diffs** on the 14 fixed pairs before and after any
   cache-digest change; and a test that **no new path increments a fallback counter at
   defaults**.
2. Every change carries a test that **goes RED if it is reverted**.
3. No silent fallback: strict mode raises, every refusal is counted, tolerant mode records.
4. Units: residuals in **tolunits** (dimensionless), LLR in **nats**, shifts in **radians**,
   spacing and texture period in **pixels** (and only there).
5. Determinism: fresh-process bit-identical, two hash seeds, canonical ordering, ties broken
   on node id.
6. Manifest on every result file, per-cell rows kept.
7. Every new switch declared before use (see 1).
8. Deviations from this file are listed in the report, not silently absorbed.  If a criterion
   turns out to be unreachable by every registered candidate, the report says so and
   registers a refinement rather than widening the search.

## §7 What is NOT claimed

The DEV set is four images; a DEV gate is a *gate*, not evidence about the world.  All
measurement here is on **cached graphs** for the DEV images (a probe, not a deployable
capture).  Only O5's fresh read speaks to images no design step has seen.
