# MISSION 15 — placement on real places, and the estimator that reads it

Branch `arena/c316ca3a-percept6`, base `9482bd2`. Every change below is behind a flag
that is **OFF** by default, the defaults are proven byte-identical (14/14 pairs, 0
diffs, `Topology/placement_m12_perf.py --check`), matching stays truth-free, and every
number carries the image count and the spread it was measured over.

Files added: `matching/m15_places.py` (items 3, 4), `matching/m15_manifest.py`
(item 9), `Topology/placement_m15_atlas.py` (item 1), `Topology/placement_m15_geometry.py`
(items 2, 3, 4, 5, 7, 10), `Topology/placement_m15_writers.py` (item 8),
`Topology/placement_m15_budget.py` (item 9), `tests/test_m15_slot_admission.py`,
`tests/test_m15_degree0_neutral.py`, `tests/test_m15_manifest.py`.
Results: `m15_atlas.json`, `m15_geometry.json`, `m15_item5b.json`,
`m15_position_writers.json`, `m15_budget.json`, `m15_budget.json` — each written through
`m15_manifest.write_result`, so each carries the effective flag vector, the frame
provider, the weight source, the endpoint list and a 12-hex code digest.

---

## 1. REPEATABILITY ATLAS — the place lenses

**Re-detection is defined twice, and every number says which.**

* **(i)** the other view has a node of the **same lens** whose anchor is within `tol` px
  of the **exact transformed anchor** of this one;
* **(ii)** additionally the **same quantised signature** — i.e. the same node key the
  tree mints from `(lens, signature)`.

**The place-lens bar (stated before the measurement) uses (i) at 2 px AND (ii).**
Identity is required because the matcher pairs *(lens, signature)* classes: a lens that
re-detects in place but re-signs itself cannot be paired, so calling it a place would
make item 3's endpoint list a list of things the matcher cannot match.

Design: **6 synthetic IRREGULAR scenes + 3 real photographs** (`cell`, `grass`,
`gravel`), 5 exact transforms (stride-multiple 8 px shift, 5 px shift, 5°, 15°, 1.15×),
two configurations (**defaults** and the M15 configuration = `FE_KEYPOINTS` +
`FE_PLACE_LAYER` + `FE_SUBPIXEL_PEAK_POS` + `FE_SUBPIXEL_GABOR_ANCHOR` + keypoint frame
provider σ `largest` / fallback `none`). 90 cells, 45 image-cells per configuration,
9 images per pose. Full table: `m15_atlas.json`; per-image spread is `_min`/`_max` for
every entry.

### M15 configuration, all poses pooled (9 images)

| lens | n | (i)@1 | **(i)@2** | (ii)@1 | **(ii)@2** | i@2 over images | ridge | along / across |
|---|---|---|---|---|---|---|---|---|
| Blob | 11,977 | 0.857 | **0.938** | 0.573 | **0.592** | 0.798 – 1.000 | no | – |
| BoundaryArc | 7,998 | 0.839 | **0.933** | 0.531 | **0.548** | 0.716 – 1.000 | yes | 0.09 / 0.27 px |
| CornerPoint | 1,291 | 0.852 | **0.891** | 0.531 | **0.543** | 0.625 – 1.000 | no | – |
| ContourFragment | 4,379 | 0.570 | 0.667 | 0.077 | 0.081 | — | yes | 0.55 / 1.51 px |
| CONTOUR_PEAK | 5,365 | 0.525 | 0.613 | 0.386 | 0.400 | — | yes | 0.73 / 1.06 px |
| BorderOwnership | 3,620 | 0.478 | 0.561 | 0.208 | 0.213 | — | no | – |
| Orientation | 2,144 | 0.323 | 0.403 | 0.051 | 0.056 | — | no | – |
| Texture | 4,978 | 0.227 | 0.281 | 0.078 | 0.086 | — | no | – |
| Boundary | 5,162 | 0.214 | 0.266 | 0.077 | 0.089 | – | yes | 3.32 / 2.67 px |
| Color | 3,428 | 0.170 | 0.205 | 0.098 | 0.116 | — | no | – |
| Contour | 283 | 0.053 | 0.092 | 0.014 | 0.016 | — | yes | 10.06 / 10.12 px |

### Deliverable lists (item 1's second half)

* **place lenses** = `Blob`, `BoundaryArc`, `CornerPoint` —
  `PLACEMENT_M15_PLACE_LENSES` in `parameters/_text.py`.
* **ridge places (across-ridge only)** = `BoundaryArc`. Measured along-ridge
  displacement 0.09 px vs across-ridge 0.27 px: its anchor is repeatable in both
  (both < 1 px), and *less* repeatable across the ridge, i.e. the arc sample is an
  arc-length position, so the truth rule for it uses the across component only and
  this is stated rather than assumed.
* **attribute lenses** = everything else, including **all five tiling lenses of the
  frozen eight-lens whitelist**: `Orientation` (0.403 / 0.056), `Texture` (0.281 /
  0.086), `Color` (0.205 / 0.116), `Contour` (0.092 / 0.016), `Boundary` (0.266 /
  0.089).

### The two findings the atlas was built to make

1. **At the shipped defaults there is no place lens at all.** Best default
   (i)@2 = 0.088 (`Boundary`), best default (ii)@2 = 0.072 (`Color`); the keypoint
   lenses `Blob`/`BoundaryArc`/`CornerPoint` do not exist under the defaults (the
   keypoint layer creates them). Every result the earlier missions reported at
   defaults was therefore reported on attributes.
2. **Positions are repeatable; identities are not.** Per pose, M15 configuration:
   | pose | Blob (i)@2 / (ii)@2 | BoundaryArc | CornerPoint |
   |---|---|---|---|
   | exact 8 px shift | 1.000 / 1.000 | 0.99 / 0.99 | 0.99 / 0.98 |
   | exact 5 px shift | 1.000 / 1.000 | 1.00 / 0.99 | 1.00 / 1.00 |
   | 5° | 0.87 / **0.40** | 0.85 / **0.32** | 0.84 / **0.37** |
   | 15° | 0.92 / **0.05** | 0.93 / **0.02** | 0.82 / **0.06** |
   | 1.15× | 0.90 / **0.52** | 0.90 / **0.41** | 0.80 / **0.31** |

   The anchor survives rotation almost perfectly; the **signature dies**. That is the
   single most important number in this mission and it is what item 10's a–b gap then
   names.

   **The "first detection wins" hypothesis**: under the 8 px (stride-multiple) shift
   neither (i) nor (ii) moves at all — 1.00/1.00 — so a tiling node's position is *not*
   re-chosen when the same region fires; under the 5 px (non-multiple) shift it is also
   1.00/1.00 for the keypoint lenses, while the tiling lenses stay at 0.03–0.08. The
   hypothesis is therefore **not supported for the place lenses** (their anchor is not
   the lattice point of the first firing region) and **unmeasurable for the tiling
   lenses**, which never pass (i) at any tolerance — consistent with their position
   being a lattice centroid that no exact transform can land on twice.

---

## 2. SLOT ADMISSION (before any refit)

Code: `PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST` — checkable slots are admitted first
(sorted by residual, then class and neighbour id), the unformable ones fill the
remainder, truncation to `SLOT_CAP = 8` happens after; `PLACEMENT_SLOT_COUNT_JUDGED`
makes `normalise`/`min_judged` count slots that actually carry a residual.

**The defect, pinned by `tests/test_m15_slot_admission.py`:** a node with eight
uncheckable neighbours whose ids sort first and one checkable neighbour whose id sorts
last. Off: the checkable edge is **discarded** (0 judged). On: it is admitted
(1 judged), and a node with ≤ 8 slots keeps every slot. `TARGET_K = 16`, `SLOT_CAP = 8`,
so real nodes routinely have more slots than the cap.

**Measured (12 cells: 4 images × 3 poses, 578,828 candidate cells, 1.16M slot reads):**

| | frozen endpoint list | | place endpoint list | |
|---|---|---|---|---|
| | shipped | flag on | shipped | flag on |
| judged share, ALL candidates | 0.0052 | 0.0100 | 0.0041 | 0.0070 |
| judged share, **TRUE** candidate | **0.342** | **0.479** | **0.383** | **0.505** |
| judged share, wrong candidate | 0.0035 | 0.0079 | 0.0024 | 0.0051 |
| candidate cells that gain a checkable slot | — | **8,649** | — | **4,748** |

Spread across the 12 frozen cells: all-candidate share 0.0006–0.0130 (shipped) →
0.0009–0.0193 (on). The fix moves the **true** candidate's judged share by +0.12 to
+0.14 while moving the wrong candidates by +0.003: it does not "give every pair more
slots", it stops the estimator from throwing away the evidence that decides the case.
Defaults: the flag is off, and `--check` reports 0 diffs over 14 pairs.

---

## 3. ENDPOINTS AND CONTEXT

Code: `PLACEMENT_M15_PLACES` selects `PLACEMENT_M15_PLACE_LENSES`; a ridge place is
admitted across-ridge only; every other node keeps its signature, loses its vote and is
**attached to the nearest place** (`m15_places.attribute_attachment`);
`placement_edges._require_m15_configuration()` refuses the flag unless the keypoint
layer, the adopted frame provider (`KEYPOINT=1`, σ `largest`, fallback `none`) and the
sub-pixel anchors are all on; `PLACEMENT_M15_PLACES`/`PLACE_LENSES` are now declared in
`_FRONTEND_SWITCHES`, so a cached payload cannot be served across them (the M14 test
that caught this is green).

`build_side` used to pass the frozen endpoint list **explicitly**, which silently
overrode the flag (an explicit argument wins by design) — fixed in
`matching/placement_matcher.py`; the flag now reaches `build_edges`.

Endpoint counts (places mode, 4 images × 3 poses; the same cells as item 2):

| image | nodes | old list (share) | new list (share) | attributes attached |
|---|---|---|---|---|
| real `cell` | 889 | 629 (0.71) | 332 (0.37) | 557 |
| real `grass` | 2,592 | 1,724 (0.67) | 1,112 (0.43) | 1,480 |
| syn `m11ir_00` | 1,173 | 689 (0.59) | 338 (0.29) | 835 |
| syn `m11ir_01` | 1,089 | 632 (0.58) | 314 (0.29) | 775 |

Old list: `Blob, Boundary, BoundaryArc, Color, Contour, CornerPoint, Orientation,
Texture`. New list: `Blob, BoundaryArc, CornerPoint`. **The endpoint set shrinks by
40–50 %**, and what it drops is exactly the mass that fails (i).

---

## 4. TRUTH RULE

Code: `PLACEMENT_M15_TRUTH_RULE` / `m15_places.lens_truth` (same lens, same position;
across-ridge only for a ridge lens) against the shipped positional rule
(`positional_truth`, reused verbatim from the read code). Every rate below is printed
beside both counts.

| cells | OLD pairs | NEW pairs | new confirmed by old |
|---|---|---|---|
| 12 (frozen endpoint list) | 4,837 | 4,918 | 0.70–0.92 per cell (mean 0.81) |
| 12 (place endpoint list) | 4,837 | 4,424 | 0.78–1.00 per cell (mean 0.88) |

Per cell: `real|cell` shift8 393 new / 359 old (323 confirmed), `real|grass` shift8
1,229 / 1,159 (1,025), `syn|m11ir_00` shift8 372 / 421 (343 — the *new* rule is
stricter here), `syn|m11ir_00` rot15 173 / 184 (121). The new rule is **not** simply a
smaller denominator: it keeps most of the old pairs and drops exactly the cross-lens
ones (the M14 audit's "48 of 68 truth pairs killed by the lens-class gate" was the same
effect seen from the other side).

---

## 5. GEOMETRY TERM

### (a) True and wrong residual distributions (place endpoints, 4 images × 3 poses)

| population | n | median | q90 | IQR |
|---|---|---|---|---|
| TRUE pairs | 13,578 | **0.0206 px** | 0.0599 | 0.0126 – 0.0599 |
| WRONG pairs | 16,871 | **0.4497 px** | — | — |
| noise, exact shift, real photos (q90) | 8,207 true residuals | 0.0126 | **0.0235** | — |
| noise, exact shift, synthetic (q90) | 5,371 true residuals | 0.0056 | 0.0599 | — |

The true-vs-wrong separation is two orders of magnitude, and the measured real-photo
residual noise under an **exact** shift (which is a pure re-detection test, no geometry
change) is **0.0235 px at q90**. State both, as required: **noise 0.0235 px, bin width
0.05 px.**

### (b) Pooled refit (synthetic FIT + real known transforms, 30,060 residuals)

The shipped table is 24 bins over [0, 6] = **0.25 px per bin**, and it puts **13,252 of
13,578 true residuals (97.6 %) into bin 0** — the term is a step function with a 0.25 px
resolution on a quantity whose signal lives at 0.01–0.06 px. The refit therefore uses
the finest width the noise allows: **0.05 px, 36 bins over [0, 1.8], min count 20 per
bin, empty bins neutral** (`m15_item5b.json`):

```
bin width 0.05  llr  [3.00, 1.10, -0.89, -1.78, -2.15, -2.73, -3.00, -2.93, ...]
true  per bin        [9962, 2811, 289, 110, 80, 50, 15, 45, 30, 20, 20, ...]
wrong per bin        [ 563, 1164, 876, 812, 862, 963, 1037, 1056, 1108, ...]
mu_wrong = -2.265    first-bin share 0.734    rises after smoothing: 8
```

Same fit at the shipped 0.25 px: `[+1.35, -3.00 × 14, 0.0 × 9]`, **1 rise**, first-bin
share 0.976, `mu_wrong -1.898`.

**Reported, not smoothed away: the 0.05 px table rises 8 times, all in the sparse tail
(bins 7–25, residual > 0.35 px), where the per-bin counts are 1–45.** The term is
monotone through the whole region that carries the mass (bins 0–6 descend
3.00 → −3.00); the rises are sampling noise in bins the min-count rule of 20 does not
suppress. A monotonicity constraint (pool-adjacent-violators, or a min count that
scales with the population) is required before this table can be shipped — see the open
problems. **Bin width and noise, both stated: 0.05 px used, 0.0235 px measured** (real
photographs, exact shift, q90).

### (c) The shipped empty-bin rule

`fit_llr` gives every empty bin `log((N_wrong+12)/(N_true+12))` = **+0.5966** — "never
measured here" read as "supports the match". Under `PLACEMENT_LLR_EMPTY_BIN_NEUTRAL`
those bins take 0.0 or the nearest non-empty value; on the pooled fit that is exactly
the 9 trailing bins above, and the smoke test pins the whole table:
`fit_llr([0.1,0.2,0.3],[2.0,1.9,2.1])` off → `[1.609, 1.099, 0, 0, 0]` (1 rise, the
prior-driven sign flip), on → all zero, 0 rises, `mu_wrong 0.0`.

### (d) The geometry term ALONE, true vs impostor

Candidates are the matcher's own (same lens, positive signature similarity) — an
impostor must be a cell the matcher would actually consider, or the AUC measures
distance instead of discrimination. `ctx_truth` is the exact-transform context (the DEV
ceiling), `ctx_matcher` the matcher's own mapping (deployable).

| endpoints | context | AUC | images | cells |
|---|---|---|---|---|
| frozen list | matcher | 0.625 | 4 | 12 |
| frozen list | truth | 0.769 | 4 | 12 |
| **place list** | matcher | **0.670** | 4 | 12 |
| **place list** | truth | **0.927** | 4 | 12 |

**Geometry separates places.** On place endpoints with an honest context it reaches
0.92 (ceiling) and 0.67 (deployable); per-cell spread and the pose split are item 10.

---

## 6. NO CHECKABLE EDGE = NEUTRAL

Code: `MATCH_DEGREE0_PAIR_USES_APPEARANCE` — a pair whose two nodes are both degree-0
takes its own appearance `s` in the assignment (`_quality`) and in the final prune
(`q = s`); not 0, not 1.

**Pinned by `tests/test_m15_degree0_neutral.py`:** flag off → quality `s(1−β)` = 0.5 for
`s = 1, β = 0.5`; on → 1.0 at `s = 1` and 0.3 at `s = 0.3`; a connected pair that agrees
perfectly still scores ≥ `s` (`s(1−β)+β·1 ≥ s` for all `s ∈ [0,1]`).

**Correction to the mission's premise, measured:** the defect is invisible when the
*whole graph* has no edges — `rel` is then hard-coded to 1.0 (`alignment.py:1263`) — and
only appears when the graph has edges elsewhere and this pair has none. That is the real
case (most nodes are degree-0) but it means the fix cannot be shown on a toy empty graph;
the test builds a graph with an edge on nodes 1–2 and reads the isolated pair.

---

## 7. WEIGHT

The weight is fitted **offline on the recorded bank** of the item-5/10 cells (the same
cells, re-read rather than re-measured): per query, the appearance similarity of every
candidate and the item-2-fixed evidence `E = plus_score − 0.5`; score
`(1−β)·sim + β·(0.5 + w·E)` at the frozen `β`. Real photographs are inside the bank
(1,693 queries in the real-photo profile, 625 synthetic, 2,318 pooled).

| bank | w=0 | 0.125 | 0.25 | 0.375 | 0.5 | 1.0 | 2.0 | 3.0 | interior optimum |
|---|---|---|---|---|---|---|---|---|---|
| frozen list, AUC | 0.758 | 0.842 | 0.844 | 0.845 | 0.846 | 0.850 | 0.855 | **0.859** | no |
| place list, AUC | 0.500 | **0.814** | 0.814 | 0.814 | 0.814 | 0.814 | 0.814 | 0.814 | flat after 0.125 |
| place list, real only | 0.500 | **0.782** | 0.782 | 0.782 | 0.782 | 0.782 | 0.782 | 0.782 | flat after 0.125 |

**No interior optimum on either bank, so the weight is set to 0 and the file says so.**
Two honest reasons, both visible in the table: (i) on the frozen list the profile is
monotone increasing to the grid's boundary — `w` is *not identifiable* from this bank,
exactly as M12 found (`ml_at_boundary`, `w_ml 3.0`); (ii) on the place list the profile
is **flat from 0.125 onward**, i.e. any positive weight does the same thing, and the
`w = 0` point is degenerate: the appearance similarity **saturates at 1.0 within a lens
class**, so at `w = 0` every candidate of a query ties (AUC 0.5, top-1 1.0 by tie). The
weight cannot be fitted on a bank whose appearance term carries no ranking; the shipped
`w_top1 = 0.375` sits inside the flat region and is neither refuted nor supported here.
`m15_geometry.json:item7.*.weight_source` records this, and `m15_manifest.weight_source()`
records the shipped value and the file it came from.

---

## 8. POSITION WRITER

Code: the anchor is written **once**, at minting
(`ConceptGraph.note_pattern_node_position`, first-write-wins) from
`Binder._step_anchor_positions` — a tiling position. A finer writer exists only as a
front-end switch: `FE_PLACE_LAYER` → `gabor_place_positions` (sub-pixel peak) and
`FE_SUBPIXEL_PEAK_POS` → `node_frames_keypoint`.

Measured (`m15_position_writers.json`, one synthetic scene, four configurations, node
identity across configurations taken from the **node key** `(lens, quantised
signature)`, never the node id — ids are minted in mint order and are not comparable
across front ends):

| lens | nodes | moved by `FE_PLACE_LAYER` | by `FE_SUBPIXEL_PEAK_POS` | by both |
|---|---|---|---|---|
| Texture | 33 | 0.67 | 0.67 | 1.00 |
| Boundary | 69 | 0.51 | 0.51 | 1.00 |
| Color | 4 | 1.00 | 1.00 | 1.00 |
| Contour | 13 | 0.08 | 0.08 | 1.00 |
| Orientation | 36 | 0.08 | 0.08 | 1.00 |
| ContourFragment | 41 | 0.00 | 0.00 | 1.00 |
| Position / RegionShape / Surface / TextureBoundary / CONTOUR_PEAK / Corner | 5–128 | 0.00 | 0.00 | 0.00 |

**Verdict, per the measurement:** at the shipped defaults **every** lens's position is a
tiling centroid and no finer writer is active — so the atlas's default numbers are a
property of the writer, not of the lens. Under `FE_PLACE_LAYER` or
`FE_SUBPIXEL_PEAK_POS` a finer writer exists for most lenses, and with both, for nearly
all. **Tiling lenses do have a finer writer available but are attributes by item 1's
bar**, so "prefer precision" is admissible for them yet pointless: an attribute does not
vote. Precision is preferred exactly where item 1 admitted a place, which is where the
keypoint layer and the adopted frame provider already put it.

---

## 9. BUDGET AND MANIFEST

**Manifest.** `matching/m15_manifest.py` records the effective flag vector (every
`_FRONTEND_SWITCHES` entry plus every `PLACEMENT_*` and `FE_*` parameter, read off the
modules), the frame provider, the weight and its source file/field, the endpoint list
and a 12-hex digest of every Python source. `write_result` attaches it; `check` raises
when a number would be written without one.

**Proof the rule bites.** `tests/test_m15_manifest.py` requires the refusal, the five
required fields, the digest's stability, and the attachment. **The M14 switch-digest
test failed on the first run of this mission** and named six switches my new code read on
the build path but had declared nowhere (`PLACEMENT_M15_PLACES`,
`PLACEMENT_M15_PLACE_LENSES`, `PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST`,
`PLACEMENT_SLOT_COUNT_JUDGED`, `PLACEMENT_LLR_EMPTY_BIN_NEUTRAL`,
`PLACEMENT_LLR_MIN_BIN_COUNT`) — they are now in `_FRONTEND_SWITCHES`, so the cache
cannot serve a place-lens payload to a frozen-list run. That is item 9's test doing its
job on item 9's own author.

**Budget.** 8 pairs (2 images × {exact shift, 15°} × {shipped, scaled}),
place-node populations of 754–2,044 positive cells:

| image / pose | positive cells | matched | exhausted |
|---|---|---|---|
| syn `m11ir_00` shift8 | 2,044 | 338 | no |
| syn `m11ir_00` rot15 | 1,032 | 109 | no |
| real `cell` shift8 | 1,045 | 331 | no |
| real `cell` rot15 | 754 | 193 | no |

**0 of 8 pairs exhaust the refinement budget**, so the budget does not bind at these
graph sizes and `MATCH_ALIGN_BUDGET_SCALES_WITH_CELLS` stays OFF — measured, not
assumed. It remains available (and now measured) for the larger photographs, which is
why the flag exists.

---

## 10. DECISIVE PROBE (labelled: a probe on cached graphs)

Geometry only, place endpoints, cached graphs, 4 images × 3 poses. Arms: **(a)** the
matcher's own place-node mapping, **(b)** the exact transform (DEV ceiling); **(c)**
synthetic, **(d)** real photographs; plus the **OLD endpoint list on the same cells**.
Top-1 is over each query's candidate row.

| arm | AUC (a) matcher | AUC (b) ceiling | top-1 (a) | top-1 (b) | queries |
|---|---|---|---|---|---|
| place list · synthetic · exact 8 px shift | **0.894** (0.867–0.920) | 0.999 | 0.823 | 0.998 | 515 |
| place list · synthetic · 5° | 0.795 (0.788–0.802) | 0.930 | 0.737 | 0.901 | 100 |
| place list · synthetic · 15° | 0.238 (0.200–0.275) | 0.938 | 0.333 | 0.833 | **10** |
| place list · **real** · exact 8 px shift | **0.899** (0.806–0.992) | 0.949 | 0.836 | 0.923 | 1,198 |
| place list · **real** · 5° | 0.766 (0.674–0.857) | 0.878 | 0.680 | 0.848 | 440 |
| place list · **real** · 15° | 0.431 (0.382–0.479) | 0.867 | 0.176 | 0.844 | **55** |
| old list · real · exact 8 px shift | 0.750 (0.691–0.809) | 0.770 | 0.770 | 0.860 | 1,369 |
| old list · real · 5° | 0.586 (0.483–0.690) | 0.784 | 0.571 | 0.812 | 467 |
| old list · real · 15° | 0.521 (0.500–0.542) | 0.798 | 0.092 | 0.698 | 131 |

**One sentence each, as required.**

* *Does geometry separate places on real photographs?* **Yes under an exact shift**
  (AUC 0.899, 2 images, 1,198 queries, spread 0.806–0.992) and **partly at 5°** (0.766),
  and **no at 15°** (0.431 with the matcher's context) — the term itself is not the
  failure there.
* *How much of the synthetic gain survives place-only endpoints?* **Essentially all of
  it at an exact shift** (synthetic 0.894 vs real 0.899: the synthetic-to-real gap is
  −0.005, inside the spread) and 0.029 at 5° (0.795 vs 0.766) — once the endpoints are
  real places, the synthetic and the real readings agree; the old endpoint list is 0.15
  worse on the same cells (0.750 at an exact shift, 0.586 at 5°).
* *Which part fails?* **The context (a–b gap) and, behind it, signature repeatability.**
  The a–b gap is 0.05–0.11 on real photos at 0–5° but **0.44 at 15°** (0.431 vs 0.867),
  and the candidate row collapses with it (1,198 → 55 queries): at 15° the atlas says the
  place is still re-detected at 0.82–0.93 but its signature survives only 0.02–0.06 of
  the time, so the matcher has no candidate to score. Not bin width (item 5b resolves
  the term at 0.05 px), not slot admission (item 2 lifts the true candidate's judged
  share from 0.38 to 0.51 and is applied in this run), not the geometry term (its
  ceiling stays at 0.867).

---

## What else changed, and the defaults

* `matching/placement_llr.py`: checkable-first admission, judged counting, empty-bin
  neutralisation, `llr_monotonicity`.
* `matching/alignment.py`: degree-0 neutral in assignment and prune, budget scaling.
* `matching/placement_edges.py`: `_require_m15_configuration`, endpoint branch.
* `matching/placement_matcher.py`: the endpoint list is a switch (was overridden).
* `eth80_frontend.py`: the six M15 switches are in the cache digest.
* `parameters/_text.py`: the M15 block, **all OFF**, with the measured place list.

**Defaults byte-identical:** `Topology/placement_m12_perf.py --check` → 14 pairs,
`identical: True (0 diffs)`, run **twice**: once before and once after the six switches
entered the cache digest (a re-key rebuilds the graphs, and the numbers did not move).
Suites: `Topology/test_placement.py` OK (17), `Topology/test_placement_m3.py` OK (12),
`tests` OK (30, including the M14 switch-digest test that caught the six undeclared
switches).

---

## Open problems, ranked

1. **Signature stability under rotation is the binding constraint** (item 1: (ii)@2
   falls 1.00 → 0.40 at 5° → 0.05 at 15° while (i)@2 stays ≥ 0.82). Everything
   downstream — the candidate row, the a–b gap, top-1 at 15° — is a consequence. No
   estimator change can fix it; the signature has to become rotation-tolerant or the
   matcher needs a rotation-aware class gate.
2. **The true candidate's judged share is 0.51 even after the fix** (item 2): half the
   true pair's slots still carry no residual, so half the evidence is padding. The
   admission order is now right; the *residual formation* (which neighbour pairs produce
   one) is the remaining lever.
3. **The appearance term carries no ranking within a lens class** (item 7: similarity
   saturates, AUC 0.5 at `w = 0`), so `w` is not identifiable and the geometry term is
   doing all the work. Until the appearance term ranks, a weight cannot be fitted
   honestly.
4. **The shipped LLR resolution is 0.25 px on a signal at 0.01–0.06 px** (item 5b):
   97.6 % of true residuals fall in one bin. The 0.05 px refit is **not shippable as it
   stands**: it rises 8 times in the sparse tail, so it needs a monotonicity constraint
   (pool-adjacent-violators) or a population-scaled min count first.
5. **The budget does not bind at these sizes** (0/8), so item 9's scaling stays dark;
   untested on the 2,600-node photographs, where it may.
6. **Item 8's per-switch attribution is at 2-decimal resolution on one image**, and
   `FE_PLACE_LAYER` and `FE_SUBPIXEL_PEAK_POS` moved exactly the same set of nodes —
   plausible (both re-anchor on the same response) but not separated here.

## The single next change

**Make the signature rotation-tolerant — one canonical orientation per place, taken
from the adopted frame provider, folded into the quantised signature — and re-run the
atlas.** Item 1 says the position already survives rotation; only the identity does not,
and every remaining failure in items 2, 7 and 10 is downstream of a candidate row that
rotation empties. This is the one change that moves (ii)@2, the a–b gap and the item-7
bank together, and it is measurable in the harness that already exists.
