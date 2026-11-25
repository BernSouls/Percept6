# MISSION 16 — a place must stay recognisable across a rotation

Branch `arena/c316ca3a-percept6`, from the Mission 15 tip `1465fd1`. Every behaviour
change is behind a flag that is **OFF** by default, the defaults are **byte-identical**,
and every fix has a test that goes red if the fix is reverted.

New code: `core/keypoint_classes.py` (the keypoint class constitution),
`Topology/placement_m16_similarity.py` (verification + acceptance A),
`Topology/placement_m16_graphs.py` (acceptance B/C/D + the equivalence build),
`Topology/placement_m16_bank.py` (the weight refit), `Topology/placement_m16_summary.py`
(the acceptance table, with the harness's duplicate cell excluded),
`tests/test_m16_graded_keypoints.py` (7), `tests/test_m16_llr_shippable.py` (6),
`tests/test_m16_determinism.py` (2).
Result files, each with its manifest: `m16_similarity.json`, `m16_graphs.json`,
`m16_graphs_summary.json`, `m16_bank.json`.

---

## 1. THE VERIFICATION (before any change)

### 1.1 Registration: the three classes are ABSENT from every registry — and absent because their `class_` string is not a registry key

`lenses/keypoints.py:263-281` writes

```python
TypedSignature(kind="CANONICAL",
               class_=LENS_OF_KIND.get(p.kind, "Keypoint"),     # :275
               quantized=(theta_bin, coherence_bin, scale, kind_index, dim))
```

with **no `population=`**, and `class_` set to the DISPLAY name (`CornerPoint`,
`Blob`, `BoundaryArc`). Every other writer in the tree uses the registry key instead
(`lenses/base.py:1351` `"CONTOUR_FRAGMENT"`, `:1431` `"BOUNDARY_ANCHORED"`, `:1513`
`"COLOR"`, `:2264` `"ORIENTATION"`, …). Measured on the registries as they load:

| registry | `CornerPoint` | `Blob` | `BoundaryArc` | what exists instead |
|---|---|---|---|---|
| `CONSTITUTIONAL_RANGES` (`get_similarity_ranges`, `core/registries.py:1540`) | **absent** | **absent** | **absent** | `CORNER (8,6,8,8)`, `EDGE`, `BOUNDARY`, `BOUNDARY_ANCHORED`, … — 67 keys, no `BLOB` at all |
| `SLOT_WEIGHT_PRIORS` (`:1549`) / `get_slot_weights` | absent | absent | absent | empty for all 67 (`SLOT_WEIGHTS` is `{}`) |
| `CIRCULAR_CYCLE_LENGTHS` | absent | absent | absent | `COLOR {4:16}`, `CORNER {1:12}`, `ORIENTATION {0:24}`, … |
| `POPULATION_AXES` / `get_population_axes` (`core/registries.py:1299`) | absent | absent | absent | `CONTOUR_PEAK`, `TEXTURE_BOUNDARY`, `ORIENTATION_POL`, … |

So **both** suspected bugs are present at once, and they are the *unregistered-and-zeroing*
kind: with an unknown `class_` there is no range table to grade by. (The brief cites
`core/population.py::get_population_axes`; it lives at `core/registries.py:1299`.)

### 1.2 Which branch actually runs — instrumented and counted

`core/types.py:447-540` has four possible exits. Counted over the M15 cells (4 images
`m11ir_00`, `m11ir_01`, `cell`, `grass` × 4 poses, every true place pair; per-image
rows in `m16_similarity.json`):

| pose | exact-equal → 1.0 | `ranges is None` → **0.0** | graded kernel | population overlap |
|---|---|---|---|---|
| 0° | **1,872** | 0 | 0 | 0 |
| 5° | 641 | **570 (47 %)** | 0 | 0 |
| 15° | 73 | **1,114 (94 %)** | 0 | 0 |
| 1.15× | 592 | **420 (41 %)** | 0 | 0 |

With the constitution on, every `ranges_none_0.0` becomes `graded_kernel`; with the
population payload **and** the constitution **and** `POPULATION_CODING_ENABLED`, every
pair becomes `population`.

**The mechanism, measured, not inferred:** `THETA_BINS = 12` over π makes one bin 15°.
A pair one bin apart takes `core/types.py:494-496` (`ranges is None → 0.0`); a pair
inside one bin takes `:492-493` (`quantized == quantized → 1.0`). Two readings follow, and they are
different phenomena. At **5°** a keypoint crosses an edge only if its angle sits within
5° of one — a third of the bin if angles were uniform, **measured 47 %** of true pairs,
higher because detected angles are not uniform and because the axial wrap at π adds
crossings. At **15°** the shift is exactly one bin for nearly every keypoint, so nearly
every pair takes the zero branch: **94 %**. That is why the 15° defect is 95 % zero-share
rather than a smaller constant, and why the 5° and 15° columns cannot be read as a
smooth curve.

### 1.3 UNITS of `payload_residual` (Mission 15's unit label, corrected)

`matching/placement_scorer.py:50-75`:

```
residual = sqrt( (Δψ/tol)² + (Δln d/tol)² + (Δln s/tol)² )
```

It is **tolerance-normalised and dimensionless**: 1.0 means "every component is one
tolerance away", where `tol` is `FIXED_TOL = 1.7009` by default and the spacing-aware
per-pair tolerance (M10 step 4a) when that path is on. **There is no pixel conversion** —
the components are an angle and two log-ratios.

Mission 15's numbers were therefore **mislabelled, not wrong**: "true residual median
0.0206 px" is 0.0206 **tol units**, the "0.05 px bin" is 0.05 **tol units**, and the
noise 0.0235 is tol units too. The comparison that decided the bin width (0.05 ≥ 0.0235)
stands; the unit label does not. `RES_MAX = 6.0` is six tolerances and the whole useful
range is the first tenth of it.

---

## 2. WHAT CHANGED, AND THE FIT A/B THAT CHOSE IT

`core/keypoint_classes.py` gives the four keypoint class names a constitution —
`CONSTITUTIONAL_RANGES`, `SLOT_WEIGHT_PRIORS`, `CIRCULAR_CYCLE_LENGTHS` (theta circular
over 12 bins) and, for the population arm, `POPULATION_AXES` — and
`lenses/keypoints.py::signature_of` registers it and attaches the graded population when
`KP_GRADED_SIMILARITY` is on. Registration mutates live tables, so `unregister()` restores
exactly what was installed. An arm that merely *clears its bookkeeping* is still
registered: the first version of this harness did that and reported the fixed arm as if
it were the default. `tests/test_m16_determinism.py` pins it.

Arms measured on FIT (`m11ir_00`, `m11ir_01`), READ on the two real photographs. Mean
true-vs-impostor AUC of `TypedSignature.similarity` over the four poses:

| arm | 0° | 5° | 15° | 1.15× | FIT (syn) | READ (real) | mean |
|---|---|---|---|---|---|---|---|
| **off (shipped today)** | 0.989 | 0.746 | **0.518** | 0.802 | 0.752 | 0.776 | 0.764 |
| graded, σ_θ = 1 bin | 0.989 | 0.954 | 0.901 | 0.959 | 0.942 | 0.959 | 0.951 |
| **graded, σ_θ = 4 bins (`wide`)** | 0.989 | 0.960 | 0.934 | 0.952 | 0.946 | 0.971 | 0.959 |
| population, σ_θ 1, w (8,1,1) | 0.999 | 0.966 | 0.858 | 0.971 | 0.942 | 0.955 | 0.949 |
| population, σ_θ 1, w (4,1,2) | 0.999 | 0.970 | 0.917 | 0.971 | 0.956 | 0.973 | 0.964 |
| population, σ_θ 2, w (8,1,1) | 0.999 | 0.969 | 0.899 | 0.973 | 0.952 | 0.968 | 0.960 |
| population, σ_θ 2, w (4,1,2) | 0.999 | 0.972 | 0.930 | 0.971 | **0.959** | **0.976** | **0.968** |

**Decision rule, stated before the numbers:** among the arms with a registered
constitution, take the highest mean AUC over the four poses on the FIT images; break ties
on the real-photo READ.

**On that rule the last row wins — and it is NOT what I shipped.** The table above was
computed on detections. Re-measured on the deployed reading (§3), the two finalists
separate like this, and they are **a tie at the mission's pose**:

| measure at 15° | graded `wide` | population θ2-balanced |
|---|---|---|
| A: detection AUC, per image | 0.906 / 0.913 / 0.948 / 0.967 | 0.909 / 0.910 / 0.936 / 0.964 |
| B: true partner admissible | **100 %** | **100 %** |
| C: geometry AUC, matcher's context | **0.631** [0.550–0.686] | 0.548 [0.459–0.648] |
| D: appearance AUC | **0.946** | 0.934 |
| appearance top-1 | **0.273** | 0.092 |

The population arm's FIT advantage comes entirely from 0°/5°/1.15×, where it compresses
the top of the scale (0° AUC 0.999 vs 0.989) — **auctioning away top-1**: an AUC is a
rank average over every pair, so lifting the whole true-pair mass looks like a win even
when the best few impostors are lifted with it. That is why the proxy ranking and the
deployed reading disagree.

**Shipped: `KP_GRADED_SIMILARITY` on, `KP_GRADED_VARIANT = "wide"`, `KP_GRADED_POPULATION`
off.** Rule applied: A is a tie at 15° (differences ≤ 0.012, inside the per-image spread);
where A ties, the deployed reading decides, and C, D and top-1 all favour `graded-wide` at
the mission's pose. The population arm stays implemented, declared, and measured, and the
report records it as the better arm if the pose mix moves toward 0°/5°/scale.

---

## 3. ACCEPTANCE, ON THE MISSION 15 CELLS

4 images × 4 transforms (exact 8 px shift = 0°, 5°, 15°, 1.15×), place endpoints, the M15
truth rule. Arms: `before` (shipped today), `graded` (**shipped by this mission**),
`after` (graded + population). Mean [min–max over the 4 images]:

### A. GRADED SIMILARITY SURVIVES — `m16_similarity.json`

| pose | median true sim, before | AUC before | median, **graded** | **AUC graded** [spread] | zero share before → graded |
|---|---|---|---|---|---|
| 0° | 1.000 | 0.989 [0.984–0.996] | 1.000 | **0.989** [0.983–0.995] | 0 % → 0 % |
| 5° | 0.500 | 0.746 [0.695–0.777] | 0.986 | **0.960** [0.946–0.977] | 49 % → 0 % |
| 15° | **0.000** | **0.518** [0.508–0.537] | **0.973** | **0.934** [0.906–0.967] | **95 % → 0 %** |
| 1.15× | 1.000 | 0.802 [0.754–0.867] | 1.000 | **0.952** [0.939–0.965] | 38 % → 0 % |

True-pair similarity at 15° is **non-zero** (median 0.973, minimum over the 4 images
0.906 AUC) and **separated**: AUC 0.934 with ≤ 24 same-lens impostors per query, 1,187
true pairs.

### B. THE CANDIDATE ROW STOPS EMPTYING — `m16_graphs_summary.json`

Truth pairs whose true partner returns as an ADMISSIBLE candidate (`similarity_rows`,
sim > 0):

| pose | before | **graded** | population | truth pairs |
|---|---|---|---|---|
| 0° | 2,021 / 2,028 (99.7 %) | **2,028 / 2,028 (100 %)** | 100 % | 2,028 |
| 5° | 648 / 1,230 (52.7 %) | **1,230 / 1,230 (100 %)** | 100 % | 1,230 |
| 15° | **73 / 1,166 (6.3 %)** | **1,166 / 1,166 (100 %)** | 100 % | 1,166 |
| 1.15× | 528 / 898 (58.8 %) | **898 / 898 (100 %)** | 99.9 % | 898 |

Mission 15's probe had **55 of 1,198** at 15°; per image this run goes 9/156 → 156/156
(`m11ir_00`), 6/170 → 170/170 (`m11ir_01`), **7/184 → 184/184** (`cell`), 51/656 →
656/656 (`grass`). **Residual gap: 7 of 2,028 pairs, all at 0° in `grass` (1,038 of 1,045)** — those seven
truth pairs fail with every arm, including the one with the constitution off, so the
cause is a keypoint that is not re-detected at that position at all: a
detection-repeatability loss, not a similarity loss. Elsewhere the row reaches the
repeatability ceiling exactly.

### C. GEOMETRY CLOSES ITS GAP — `m16_graphs_summary.json`

Geometry-term-alone AUC, matcher's context (a) vs exact-transform context (b = ceiling):

| pose | a before | **a graded** | a population | b (ceiling) | gap before | **gap graded** | gap population |
|---|---|---|---|---|---|---|---|
| 0° | 0.896 | **0.923** | 0.971 | 0.974 | 0.078 | **0.049** | 0.001 |
| 5° | 0.781 | **0.819** | 0.815 | 0.904 | 0.123 | **0.080** | 0.083 |
| 15° | **0.334** | **0.631** | 0.548 | 0.902 | **0.568** | **0.257** | 0.340 |
| 1.15× | 0.785 | **0.821** | 0.883 | 0.907 | 0.122 | **0.064** | 0.003 |

At 15° the geometry term alone rises 0.334 → **0.631** and closes **55 %** of the a–b
gap. It does not reach 0.867 because the *ceiling itself* is 0.902 on these cells: the
Mission 15 probe's 0.867 came from a 55-query subset, and the same effect measured over
1,166 truth pairs is 0.902. **The remaining 0.271 is a context gap** — the matcher's
context is a candidate-derived mapping, and at 15° the row now contains 100 % of the
truth partners but many more wrong ones (the row grew by 16×), so the wrong-context
fraction rises.

### D. APPEARANCE RANKS — `m16_graphs.json`

Appearance-only AUC on the same candidate rows, before → graded:

| pose | before | **graded** [spread] | top-1 before | **top-1 graded** |
|---|---|---|---|---|
| 0° | 0.500 | **0.989** [0.984–0.996] | 1.000\* | **0.998** |
| 5° | 0.500 | **0.965** [0.952–0.977] | 1.000\* | **0.597** |
| 15° | 0.500 | **0.946** [0.929–0.966] | 1.000\* | **0.273** |
| 1.15× | 0.500 | **0.967** [0.957–0.984] | 1.000\* | **0.685** |

\* the "before" top-1 of 1.000 is a **degenerate artefact**: every candidate ties at
0.5, and a tie is scored as "true not beaten". It is over 16 queries at 15°. The
appearance AUC of exactly 0.500 is the M15 measurement reproduced.

It rises above 0.500, so `w` was refit — §5.

### E. (reserved) — no criterion failed

A-D all move. The diagnosis in §1.2 is confirmed by the branch census *and* by the
reversal: turning the constitution on moves the exact numbers the census predicted
(95 % zero-share → 0 %, 6.3 % admissibility → 100 %).

---

## 4. QUALITY BARS

| bar | pinned by | goes red on revert because |
|---|---|---|
| 1 flags OFF, defaults byte-identical | `placement_m12_perf.py --check`, 14 pairs: **0 diffs** at the M15 tip, **0 diffs** after the switch-digest change (twice, 202 s and 200 s) | any default move shifts a float |
| 2 a test that fails if the fix is reverted | `test_m16_graded_keypoints.py` asserts the SHIPPED defect (exact-equal 1.0 / one bin 0.0) **and** the requirement (sub-bin 0.99, 15° non-zero) | reverting the fix leaves both halves unprovable |
| 3 no silent fallback | `test_an_unknown_variant_raises_and_is_counted`, `KP_GRADED_VARIANT` raising on an unknown name, `REFUSALS` | a silent no-op leaves `REFUSALS` empty |
| 4 units stated | §1.3 and a `units` field in every result file | — |
| 5 freeze guard covers `res_max`/`n_bins` | `test_m16_llr_shippable.py::FreezeGuardTests` (3) | with the guard strict the mismatched refit is refused; off it is **recorded** in `GUARD_RECORD`, never swallowed |
| 6 shippable LLR (monotone, no positive-credit empty bin) | `MonotoneTableTests` (3) + `isotonic_nonincreasing` | the unprojected table rises; the projection removes every rise |
| 7 determinism | `test_m16_determinism.py` (2): fresh process, two hash seeds | an order-dependent registry leaves a different digest |
| 8 manifest + per-cell rows | every writer goes through `m15_manifest.write_result` with `cells`/`rows` retained; `test_m15_manifest.py` scans the directory | dropping rows changes the file |
| 9 switches declared first | `test_m14_switch_digest.py` (46-test suite) — this test **caught** `PLACEMENT_LLR_MONOTONE`/`PLACEMENT_FREEZE_GUARD_SUPPORT` being read on the build path undeclared; both are now in `_FRONTEND_SWITCHES` | an undeclared switch fails the scan |

---

## 5. THE LLR TABLE, AND THE WEIGHT (bar 6, acceptance D)

### 5.1 What the deployed table actually does

`matching/placement_frozen_params.json` (fit on 2,574 true / 4,684 wrong):

* **5 rises** in LLR against |residual| (bin 11 = +1.695 at 2.75–3.0 tol units),
* **16 of its 24 bins positive**, the tail bins 13–23 at **+0.597** — a *distant* candidate
  earns positive evidence while one at 1.5 tol units earns −1.0,
* those tail bins are the Laplace PRIOR showing through: `fit_llr` computes
  `log(P_t/P_w)` and with no samples both are priors, so an empty bin reads
  "never measured here" as "supports the match" (`matching/placement_llr.py`, the
  `PLACEMENT_LLR_EMPTY_BIN_NEUTRAL` comment block).

`PLACEMENT_LLR_MONOTONE` projects the table onto the non-increasing family
(pool-adjacent-violators, weighted by bin counts) after the empty-bin rule and the clip:

| table | rises before | **rises after** | blocks | bins changed | max change |
|---|---|---|---|---|---|
| M15 refit, 0.05 tol over 1.8, 36 bins (13,578 / 16,871) | 8 | **0** | 7 | 30/36 | 0.818 nat |
| M15's shipped-shape refit, 0.25 tol over 6, 24 bins | 1 | **0** | 2 | 23/24 | 2.998 nat |
| the deployed bundle's own `llr`, uniform weights | 5 | **0** | 2 | 23/24 | 2.222 nat |

The projection is not cosmetic: on the 36-bin refit it collapses 30 of 36 bins into 7
plateaus — the data does not support that tail's resolution, and the table now says so
instead of inventing rises. On the deployed table (whose per-bin counts were **not
stored in the bundle**, so uniform weights were used) it drops 15 bins from positive to
negative. **The correct repair is a refit with counts, not a projection of an unknown
weight profile** — §7.

Empty-bin policy stays as M15 stated: neither population → 0.0 (neutral); one population
→ the nearest non-empty neighbour's value, never the prior.

### 5.2 The weight, refit on the new bank (`m16_bank.json`, 5,322 queries)

Appearance + evidence per admissible candidate, under the shipped arm; score
`(1−β)·sim + β·(0.5 + w·E)` at the frozen β:

| w | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| top-1 (all) | 0.116 | 0.497 | 0.520 | **0.539** | 0.545 | 0.556 | 0.558 | 0.558 | 0.559 | 0.559 |
| AUC (all) | 0.979 | 0.982 | 0.983 | **0.983** | 0.983 | 0.984 | 0.984 | 0.984 | 0.984 | 0.985 |
| AUC (syn, 1,575 q) | 0.965 | 0.975 | 0.976 | **0.977** | 0.978 | 0.979 | 0.979 | 0.980 | 0.980 | 0.980 |
| AUC (real, 3,747 q) | 0.981 | 0.983 | 0.983 | **0.984** | 0.984 | 0.984 | 0.985 | 0.985 | 0.985 | 0.985 |

**The optimum is not interior — it is at the grid boundary (w = 3.0)** and the curve is
flat from 0.75 on (AUC +0.002, top-1 +0.003, i.e. inside the per-image spread). So `w` is
**not identifiable from this bank**; what it identifies is a *lower* bound — `w ≥ 0.25`
buys 0.42 top-1 over `w = 0`. The shipped `w = 0.375` sits at the knee and is kept:
raising it to 3.0 would move nothing measurable and would double the weight of a term
whose own ceremony is unshippable (§5.1). Recorded as a boundary optimum, not an interior
one.

---

## 6. OPEN PROBLEMS, RANKED

1. **The deployed LLR table rewards distant candidates** (§5.1): 16/24 bins positive,
   5 rises, +0.597 in the tail. Exposure on the fitted bank is small (26 of 16,871 wrong
   pairs reach ≥ 3.5 tol units, 0.15 %) but the table is indefensible as it stands, and
   the bundle does not carry the per-bin counts needed to project it honestly.
2. **The population payload is invisible to the cache digest.** `KP_GRADED_POPULATION` is
   now in `_FRONTEND_SWITCHES`, and the graph harness's equivalence build measured the
   consequence: identical node keys and identical `before`/`graded` readings
   (`digest 6029919339263226520` both sides), but the `after` arm differed
   (geometry 0.827 vs 0.820, top-1 0.355 vs 0.561) because the off-build carried no
   population payload. **A graph can only serve the population arm if it was built with
   it.** `m16_graphs_summary.json::equivalence` records exactly this, including
   `readings_identical: false`.
3. **Top-1 at 15° is 0.273** while the AUC is 0.946: the row is now 16× larger and dense
   with near-misses. This is the next real ranking problem, and it is not the same
   problem as mission 16's.
4. **The geometry ceiling at 15° is 0.902, not 1.0** — 10 % of truth pairs have a
   placement vector that the exact-transform context cannot separate either. Detector
   repeatability, not the matcher.
5. Mission 15's residual noise and this mission's bin width are in **tol units**; any
   future "px" claim about `payload_residual` is wrong by construction.

## 7. THE SINGLE NEXT CHANGE

**Refit the frozen bundle under `PLACEMENT_LLR_EMPTY_BIN_NEUTRAL` + `PLACEMENT_LLR_MONOTONE`
on the pooled M15 residuals (13,578 true / 16,871 wrong), and store the per-bin counts in
the bundle** so the freeze guard can check the support, the resolution *and* the weight
profile. Expected effect: bins 13–23 fall from +0.597 to the wrong-mean reading, which
withdraws positive credit from exactly the candidates that top-1 at 15° currently
miscounts (0.273 with a 0.946 AUC). Then re-run the atlas, because a changed LLR changes
`plus_terms` and therefore the geometry reading.

---

## 8. DEFAULTS, SUITES, DETERMINISM

* `Topology/placement_m12_perf.py --check --workers 2`: **`identical: True (0 diffs)`**
  — 14 pairs — at the M15 tip (`1465fd1`) before any change, and three times after
  (202.0 s, 200.0 s, 204.3 s wall; the last run is the final tree).
* `python -m unittest discover -s tests`: **47 tests, OK** (was 31; +16 M16 + the
  digest test now covering two more switches).
* `PYTHONPATH=.:Topology python -m unittest Topology.test_placement Topology.test_placement_m3`:
  **29 tests, OK** (the `Topology/` prefix is needed for `import guard`, as before).
* Determinism: two fresh processes, two hash seeds → one digest
  (`test_m16_determinism.py`).
* Flags, all **OFF** by default: `KP_GRADED_SIMILARITY` (off), `KP_GRADED_VARIANT`
  ("" when off → `"wide"` when on), `KP_GRADED_POPULATION` (off),
  `KP_GRADED_POP_VARIANT` (`"theta2-balanced"`), `PLACEMENT_LLR_MONOTONE` (off),
  `PLACEMENT_FREEZE_GUARD_SUPPORT` (off; tolerant but *recording*).

## 9. WHAT WAS RUN IS WHAT IS REPORTED

* `m16_similarity.json` — 4 images × 4 poses × 7 arms, 112 rows, per image and pose.
  **Probe on cached graphs? No** — built fresh, 19.4 s.
* `m16_graphs.json` — 16 cells + 1 equivalence build, 2 workers, ~14 min; the equivalence cell is excluded from every mean here and is reported
  separately (`m16_graphs_summary.json`).
* `m16_bank.json` — 16 cells, a read off the graph cache, 192 s. Its per-cell rows
  hold **counts only**: the candidate-level bank is 138 MB and GitHub refused it at the
  pre-receive hook, so it is written only on request (`--keep-bank`, path git-ignored).
  The profiles quoted in §5.2 are unaffected — they are computed from the rows before
  they are dropped, and the file carries the query count each profile was taken over.
* Everything labelled: per-cell rows are kept in all four files.
