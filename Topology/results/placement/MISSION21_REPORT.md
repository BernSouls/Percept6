# MISSION 21 — the shortlist on one ruler, a second channel chosen on recall, and the two open
# questions made answerable

BUILD and FIT only; no fresh FRESH read. Every number is on graphs that already existed (FIT,
DEV, and the cached renderer scenes). The registered FRESH null read is **WITHDRAWN** (O7) and
was not run.

Governing rule: *a measurement that cannot produce the answer it reports is not a result.*
Three instruments were broken when the mission started; each was repaired rather than quoted:

| # | the broken instrument | what it could not say | what replaced it |
|---|---|---|---|
| 1 | stage 1 standardized each pair by the **reference's** own mean/std/kept dims | anything about comparability across gallery items — two references scored the same query on two rulers | one ruler frozen on FIT (`matching/placement_prefilter_params.json`), applied identically to every image |
| 2 | the AGGREGATION tag was "the truth's shipped score is above the winner's" while the winner was the argmax of that same score | AGGREGATION was unreachable by construction; a zero was reported as a measurement | the composite is computed **in the tagger** and the winner is its argmax (O4) |
| 3 | the coverage control **filtered** candidates out of a ±5 % band | whether the shipped advantage is a coverage artifact; it collapsed every arm to 2/15 | subsample each candidate's mapped places to m = min over the shortlist and **recompute** every arm (O3) |

M20 numbers produced by instruments 1–3 are marked SUPERSEDED and are not re-quoted as
evidence.

---

## 0. The tree

| file | what it is |
|---|---|
| `matching/placement_prefilter.py` | `bag_v2` / `standardize` / `pair_terms` / `score_terms` / `score_bags_v2` / `image_score_v2` — stage 1 under a FROZEN ruler, K = 24, a mask instead of padding |
| `matching/placement_prefilter_params.json` | the ruler: per-slot mu/sd/keep, K, sigma, provenance (split, images, rule, code digest, the DEV leak note) |
| `matching/placement_context.py` | `SIGMA_D`/`SIGMA_A` globals; `PLACEMENT_NODE_DESC_VARIANT` (`sorted`/`ratio`/`rank`/`angles`) |
| `parameters/_text.py` | `PLACEMENT_PREFILTER_V2` (off), `PLACEMENT_NODE_DESC_VARIANT` (`sorted`), `PLACEMENT_M21_COVERAGE_SEED` (20261118), `PLACEMENT_M21_COVERAGE_DRAWS` (5) |
| `Topology/placement_m21_fit.py` | fits the ruler on the FIT fit-half; `--stage params|terms|choose|report` |
| `Topology/placement_m21_terms.py` | the sigma-free pair table: one pass, every width for free |
| `Topology/placement_m21_shortlist.py` | recall@8 per task with chance on its own units (O5) and the channel grid (O2) |
| `Topology/placement_m21_before.py` | the BEFORE number under the M20 ruler, same gallery, same tasks |
| `Topology/placement_m21_k.py` | the bag K against the matcher's K, by recall@8 |
| `Topology/placement_m21_attrib.py` | the tagger that computes the composite it tests (O4) |
| `Topology/placement_m21_coverage.py` | subsample-and-recompute at equal coverage (O3) |
| `Topology/placement_m21_yaw.py` | the descriptor's invariance under pitch/yaw/scale, then the variants (O6) |
| `Topology/placement_m21_analysis.py` | the paired before/after, the sigma grid, the padding and overlap measurement |
| `tests/test_m21_bundle.py` | 14 tests; each names the property and goes RED if it is reverted |

Flags OFF by default. `PLACEMENT_PREFILTER_V2` and `PLACEMENT_NODE_DESC_VARIANT` are declared in
`parameters/_text.py` **before** any shortlist or graph is built under them. The defaults check
is §8.

---

## 1. O1 — one ruler, one K, no padding

### 1.1 The change

* **One scale, precomputed, frozen.** `matching/placement_prefilter_params.json` holds, per
  slot, the mean and sd of the log nearest-neighbour distances and of the direction's cos and
  sin. They are fitted once on the FIT objects at **even index** (40 images, 61,362 places,
  all 24 slots kept) and applied identically to every image. No reference contributes to the
  scale; no dimension set is chosen per pair. A missing file is a **refusal** (`RuntimeError`),
  not a fallback.
* **Padding → mask.** A place now keeps exactly the slots it has. `bag_v2` returns `D`, `A`
  and a `M` mask; the kernel sums only over the overlap and divides by the overlap, exactly the
  convention `config_similarity` already used (`n = min(len(a), len(b))`). Nothing is repeated
  to fill a row.
* **One stated K.** The bag K in the params file is **24**, the same K the matcher's
  configuration descriptor uses (`ASSIGN_K` = 6 belongs to the assignment's local subset and is
  untouched). The M20 docstring/mission text said 8; the code was 6; the file now says 24 and
  §1.4 measures what that costs.

### 1.2 Before / after on the registered surface (FIT, 40-item mixed gallery, S = 8, n = 40)

| channel | T1 same-class (chance 0.200) | T2 class (chance 0.617) | T3 object (chance 0.025) | s/pair |
|---|---|---|---|---|
| **BEFORE** — M20 `score_bags`, per-reference ruler | 1.000 (lift 5.00) | 0.425 (lift 0.69) | **0.250** (lift 10.00) | 0.0148 |
| **AFTER** — frozen ruler, σ 0.35/0.50 | 1.000 (lift 5.00) | 0.475 (lift 0.77) | **0.175** (lift 7.00) | 0.33 |

Paired, same 40 queries (this is the number that decides whether the difference exists):

| task | both hit | v1 only | v2 only | neither | net |
|---|---|---|---|---|---|
| T1 | 40 | 0 | 0 | 0 | 0 |
| T2 | 12 | 5 | 7 | 16 | **+2** |
| T3 | 4 | 6 | 3 | 27 | **−3** |

**Acceptance: NOT met.** The registered bar was recall@8 > 0.25 for the shortlist score, and the
frozen ruler reaches **0.175** (7/40) at the registered widths. The paired table says why this
must not be over-read in either direction: the change moves 3 queries on T3 and 2 on T2, and at
n = 40 one query is 0.025, so **the two rulers are not distinguishable on this instrument**
(exact McNemar p ≈ 0.51 on T3). The frozen ruler is a correctness fix with no measurable
recall cost and no measurable gain. What it does buy is stated in §1.3.

The σ grid on the whole query set (diagnostic only — the widths were chosen on the fit half and
the registered defaults kept):

| σ_d \ σ_a | 0.20 | 0.35 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| 0.20 | 0.225 | 0.175 | 0.175 | 0.225 | 0.225 |
| **0.35** | 0.275 | 0.200 | **0.175** | 0.175 | 0.200 |
| 0.50 | 0.225 | 0.225 | 0.175 | 0.200 | 0.175 |
| 0.75 | 0.225 | 0.225 | 0.225 | 0.200 | 0.175 |
| 1.00 | 0.225 | 0.225 | 0.200 | 0.200 | 0.200 |

Every cell lies in [0.175, 0.275] — four queries wide. The fit half chose a cell at the grid's
σ_d boundary (0.20, 0.75) for 0.250 vs the default's 0.200; **a boundary optimum is not a fit**,
so the widths stayed at the registered 0.35/0.50 and the grid is reported as a flatness
measurement.

### 1.3 The asymmetry the mission asked for

|s(A,B) − s(B,A)| over the **same 120 pairs**:

| ruler | mean | p95 | score level |
|---|---|---|---|
| M20, per-reference | 0.0219 | 0.0540 | 0.876 |
| frozen | **0.0738** | 0.1590 | 0.328 |

The frozen ruler does not remove the directional asymmetry and **makes it larger**, because the
aggregation is still "the mean over the query's places of the best reference place" and the M20
scores were compressed near 1 (level 0.876) while the frozen ones are not (0.328). What the
change removes is the *reference-dependent ruler*: `s(A,B)` is now a function of A, B and the
frozen file only. Stated plainly: asymmetry is a property of the aggregation direction and it is
unchanged in kind; the mission's "before and after" is 0.0219 → 0.0738 with the levels above.

### 1.4 K, measured

Each K gets its own standardization fitted on the same FIT fit-half, built by truncating the
distance-sorted cached descriptors (lossless):

| K | slots kept | T3 recall@8 (n=40) | fit half | report half | s/pair |
|---|---|---|---|---|---|
| 6 (= `ASSIGN_K`) | 6 | **0.250** | 0.250 | 0.250 | 0.078 |
| 12 | 12 | 0.225 | 0.200 | 0.250 | 0.079 |
| 24 (shipped) | 24 | 0.175 | 0.200 | 0.200 | ~0.40 |

K is **not identified** at n = 40 (three queries separate the extremes), and 6 is five times
cheaper. The shipped file keeps 24 because that is the K the fit was frozen with and the mission's
rule forbids moving a value the instrument cannot resolve; the measurement that argues the other
way is recorded here and is the first item of §7.

### 1.5 What padding would have fabricated, and what the mask does

| quantity | value |
|---|---|
| place rows in the frozen bags (40 refs) | 62,095 |
| live places (≥ 2 slots) | 41,428 (33.3 % of all rows; 82,876 rows carry no descriptor at all) |
| support among live places | mean 16.35, median 18, p05 4, p95 24 |
| live places below K = 24 | **69.8 %** |
| live places below 6 | 9.4 % |
| overlap actually used (mean shared slots per live place pair, 100 seeded pairs) | **12.9** (min 5.9, max 17.4) |

Mission 20 would have repeated the last entry to fill a median 6 of 24 slots for 70 % of live
places. The mask compares them on the 12.9 slots they share on average.

**The test that pins O1:** `tests/test_m21_bundle.py::FrozenRulerTests` — an independent
reimplementation of the score from the frozen file alone (RED if any statistic of the reference
bag creeps back), a source-level assertion that no per-reference `.std(`/`keep = sd` exists in the
v2 path, the refusal test, and `PaddingAndKTests::test_padding_is_a_mask` (RED if the last entry
is repeated).

---

## 2. O2 — the second channel, chosen on recall

Fixed, unfitted appearance descriptor: 4×4×4 colour histogram + 12-bin gradient-orientation
histogram, L2, cosine. Weight chosen on the FIT fit half; the all-query column is the report.

| w (appearance weight) | T3 fit half (n=20) | T3 all (n=40) |
|---|---|---|
| 0.00 (geometry only) | 0.200 | 0.175 |
| 0.15 | 0.350 | 0.325 |
| 0.30 | 0.400 | 0.400 |
| 0.50 | 0.500 | 0.475 |
| 0.80 | 0.500 | 0.500 |
| 1.50 | 0.550 | 0.625 |
| 3.00 | 0.500 | 0.625 |
| 8.00 | **0.650** | **0.675** |
| appearance alone | — | **0.675** |

**The union degenerates to its appearance channel.** Recall is monotone in w and the maximum is
at the grid's boundary, where the union equals the appearance channel alone (0.675 T3, 0.800 T2
on FIT). The geometry channel contributes nothing to the union at S = 8, and O2's acceptance
("recall@8 above the geometry-only figure") is met only in that degenerate sense:
**0.675 vs 0.175 geometry-only**. A boundary optimum is not a fit, so no weight is shipped; the
finding is that the appearance channel dominates and the additive union is the wrong shape for
it, not that 8.0 is the right weight.

DEV, reporting only (widths and w fixed by FIT): gallery 16, n = 16, chance T1 0.500 / T2 0.533 /
T3 0.062.

| channel | T1 | T2 | T3 |
|---|---|---|---|
| geometry | 1.000 | 0.625 | 0.438 |
| appearance | 1.000 | 0.625 | 0.938 |
| union w = 0.5 | 1.000 | 0.562 | 0.750 |
| union w = 8.0 | 1.000 | 0.688 | 0.938 |

Same ordering, and the appearance channel is again the whole story. Cost: appearance is
**0.6 ms/image** (a 64+12-bin histogram on a 128×128 PNG); geometry stage 1 is 0.33 s/pair at
K = 24 and 0.078 s/pair at K = 6.

---

## 3. O3 — the equal-coverage control, as registered

Run: FIT, T3, stride-5 queries (one per class, **n = 8**), S = 8, 64 pairs, 0 errors, **0 silent
fallbacks** (`falls_back` sum = 0), 207.6 s/query, m ∈ [1010, 1368] mapped places, 1,371 places
per pair on average. Five draws, seed **20261118** (`PLACEMENT_M21_COVERAGE_SEED`), stated in
`parameters/_text.py` so a re-run reproduces the same subsamples.

| arm | top-1 unmatched | top-1 at equal coverage (mean of 5 draws) | spread over draws |
|---|---|---|---|
| S0 (B) | 1/8 = 0.125 | 0.125 | [1, 1] |
| S1 (B + w·E) | 1/8 | 0.125 | [1, 1] |
| S2 (B + λ·S3) | 1/8 | 0.125 | [1, 1] |
| S3 (S3) | 1/8 | 0.125 | [1, 1] |

**The shipped advantage survives the control on this run** — and, more usefully, nothing changes:
every arm keeps its top-1 at m = min over the shortlist, with n unchanged (8 → 8). S0 is the
built-in control (B does not read the mapping, so it *must* be invariant) and it is. The
score-vs-count Spearman has **n = 1 query**: only one query's S3 vector had non-zero variance
across its 8 candidates (−0.119 before equalisation, +0.314 after), so it is reported as a single
observation and not as a correlation. That half of the acceptance is **not** answered and is
listed as open (problem 5).

Honest limits: 8 queries and 1 hit make this a control with almost no power. It shows the
operation now *is* subsample-and-recompute and that it does not change the ranking on a sample
where coverage varies 1,010–1,368 places; it does not establish the absence of a coverage effect
across 40 queries.

---

## 4. O4 — AGGREGATION made reachable

The tagger computes each arm's composite itself and picks the winner as **its** argmax:

```
S0 = B            S1 = B + w·E (w = 0.375 from matching/placement_score_params.json)
S2 = B + λ·S3     λ = 0.0 (the M20 fit was at the boundary, and a boundary optimum is not a fit)
```

and then tags: SHORTLIST (positive absent) → AGGREGATION (truth's **shipped** score above the
winner's while the composite puts the winner first) → CORRESPONDENCE (< 80 % of the winner's
mapped/judged) → APPEARANCE.

Re-tagging the M20 runs on disk (16-item gallery, n = 16):

| run | arm | top-1 | SHORTLIST | AGGREGATION | CORRESPONDENCE | APPEARANCE |
|---|---|---|---|---|---|---|
| FIT T3 | S0 | 10/16 | 4 | 0 | 0 | 2 |
| FIT T3 | S1 | 10/16 | 4 | 0 | 0 | 2 |
| FIT T3 | S2 | 11/16 | 4 | 0 | 0 | 1 |
| DEV T3 | S0 | 10/16 | 5 | 0 | 0 | 1 |
| DEV T3 | S1 | 9/16 | 5 | **2** | 0 | 0 |
| DEV T3 | S2 | 10/16 | 5 | **1** | 0 | 0 |

The tag is reachable and non-degenerate: on DEV the evidence term **reverses two queries that the
shipped score orders correctly** (S1: 10/16 → 9/16) and the S3 term reverses one. On FIT the
distribution is non-degenerate through SHORTLIST/APPEARANCE (no composite reversal occurred).
`selftest()` pins the non-circularity: rows where the truth has the highest B and a lower
composite must be tagged AGGREGATION by S1/S2 and must **not** be tagged by S0 —
`TaggersThatRankByTheQuantityTheyTest` fail that, and the test asserts it.

T0 remains a sanity control only, not evidence.

---

## 5. O5 — chance, on its own units, and the M20 restatement

Chance is per task and per gallery, not one number:

| task | what is scored | chance formula | FIT 40-gallery | DEV 16-gallery |
|---|---|---|---|---|
| T1 | one item of a same-class gallery | 1/\|gallery\| | 0.200 (5 items) | 0.500 (2 items) |
| T2 | the CLASS (first same-class item) | hypergeometric: P(a random top-S holds one of the other instances) | **0.617** (4 of 39) | 0.533 (1 of 15) |
| T3 | one item of the mixed gallery | 1/\|gallery\| | 0.025 | 0.062 |

Restating M20, which compared raw rates across different chance:

| M20 statement | what it should have said |
|---|---|
| "T2 = 0.062 = chance (1/16)" | on a 16-item gallery with 3 other instances of the class, chance@1 is 3/15 = **0.200**, so 0.062 is **below** chance (lift 0.31). The 1/16 figure is the chance of naming one particular item, which is T3's question, not T2's |
| "T1 0.786 (n=14)" vs "T3 0.625" | T1's chance is 1/5 (0.200) and T3's is 1/16 (0.0625): lifts 3.9 and 10.0. The raw rates are not comparable and T3 is the stronger result once chance is accounted for |
| T3 0.625 at 16 → 0.25 at 40 | chance moved 0.0625 → 0.025 at the same time: lift 10.0 both times. The fall from 0.625 to 0.25 is the gallery growing, not the score degrading |

---

## 6. O6 — the yaw wall, measured, and the breaking term

### 6.1 The estimator's own invariance (diagnosis, renderer scenes, no matcher)

Per-place decomposition of the descriptor change at the TRUE partner; a constant shift is what a
uniform scale does, the residual is what the estimator actually sees. n is the number of places;
the impostor set is capped deterministically at 80 places per cell and query places at 250.

| view | n | shift (mean \|Δd\|) | shape (sd after removing the shift) | angle (mean circular \|Δθ\|, rad) |
|---|---|---|---|---|
| identity | 306 | 0.000 | 0.000 | 0.000 |
| in-plane 15° | 266 | 0.376 | 0.346 | 0.650 |
| **uniform scale 1.15×** | 232 | 0.342 | 0.313 | 0.655 |
| **yaw 20°** | 202 | 0.403 | 0.296 | 0.659 |
| pitch 20° | 266 | 0.240 | 0.238 | 0.537 |
| yaw 35° | 172 | 0.454 | 0.358 | 0.679 |
| pitch 35° | 226 | 0.303 | 0.249 | 0.596 |

**`config_similarity` is not invariant to a uniform scale, measured exactly:** a scale s adds
log s to every log distance, the estimator differences the vectors, so the similarity is
exp(−½·log(s)²/σ_d²). At 1.15× that is **0.9234**, not 1. The `ratio` variant (centre each
place's distances) removes the term exactly (`test_config_similarity_is_not_invariant_to_a_
uniform_scale` pins both halves).

### 6.2 Variants, through the deployed blend

Node accuracy, 10 report scenes, σ chosen on the other 10, K = 24, cap 24, assignment start:

| variant | identity | in-plane 15° | scale 1.15× | **yaw 20°** |
|---|---|---|---|---|
| `sorted` (shipped) | 0.934 | 0.203 | 0.215 | **0.234** |
| `ratio` (scale-free) | 0.934 | 0.186 | 0.201 | 0.205 |
| `rank` (order only) | 0.934 | 0.197 | 0.194 | 0.213 |
| `angles` (no distances) | 0.934 | 0.196 | 0.190 | 0.213 |
| `seed_only` (the 0.75 similarity channel alone) | 0.902 | 0.168 | 0.179 | 0.203 |

n at yaw 20°: **1,896 places over 10 scenes**, per-scene spread 0.08–0.37.

The same measurement averaged over ALL cells (both σ values, all 20 scenes, 800 cells —
`m21_yaw_summary.json`) is flatter still: `sorted` 0.222, `ratio` 0.208, `rank` 0.207,
`angles` 0.209, `seed_only` 0.198 at yaw 20°, i.e. a 0.024 band across variants that range from
"the shipped descriptor" to "no distances at all".

**Acceptance: 0.234 > 0.208, met numerically — and it does not mean what it was meant to mean.**
Three things the numbers say:

1. the shipped descriptor is the **best** of the five at yaw (0.234); every cheaper variant is
   within 0.03 below it, and the scale-free variant is *worse*. The descriptor is not the
   breaking term;
2. yaw 20° (0.234) scores **above** in-plane 15° (0.203) on this harness, in every channel —
   so this instrument shows **no yaw-specific loss at all**, which contradicts the M20
   probe's pitch-0.332-vs-yaw-0.208 pattern. The absolute yaw number reproduces M20 (0.208 →
   0.234, inside the per-scene spread), the *contrast* does not, because pitch was not run
   through this accuracy harness (only through the descriptor diagnosis, §6.1, where pitch is
   the milder view: shape 0.238 vs 0.296, angle 0.537 vs 0.659);
3. the 0.75 channel alone (`seed_only`) is 0.203 at yaw and 0.168 in-plane, i.e. **weaker than
   the blend everywhere** — so the wall, if it exists, is not isolated by removing the
   descriptor.

**The measured floor, named:** the configuration descriptor is *not* what limits yaw on this
harness — its distance shape changes by 0.296 sd at yaw 20° versus 0.346 at in-plane 15°, and
variants that discard spacing (`rank`) or distances entirely (`angles`) move yaw accuracy by
≤ 0.03. The breaking term cannot be named from these cells because the missing cell is the
**pitch control on the same accuracy harness**; until x20 is measured there, a yaw wall on this
instrument is not established.

**Superseded:** the first diagnosis pass recorded the **max** impostor similarity over a set that
included the truth, which made the identity row read self 1.000 / impostor 1.000 and every
rotated row read impostor *above* self. That column is an order statistic over 80 draws and is not
comparable with a single self value; it is replaced by the mean impostor and an AUC in the code and
was **not** re-measured. The shift/shape/angle columns and the `config_similarity` measurement do
not use it.

---

## 7. O7 — the withdrawal, and the re-registration

**Withdrawn:** `MISSION21_PREREGISTRATION.md`'s registered FRESH null read (≈6 h, guaranteed null
by the MDE arithmetic). It was not run and no FRESH graph was built.

**Re-registered, to be executed after O1–O6 are settled:** the fresh read is re-sized to the
effect the *new* design shows on FIT **and** DEV, and it is not a null read:

* **T3 on the 40-item gallery, appearance channel** — FIT 0.675 (n=40) and DEV 0.938 (n=16)
  against a geometry baseline of 0.175 / 0.438. The effect to detect is the *difference from the
  geometry channel*, which is 27 lifts of chance on FIT; an MDE sized for that difference needs
  far fewer scenes than the withdrawn null read, whose planning effect was a 0.06 AUC difference.
* **T2 at chance 0.617 on FIT** is the honest blocker: the class task is not separable from
  chance at this gallery size, so a FRESH read must fix T2's gallery (a 16-way class pool) before
  it can inherit the registered T2 endpoint.
* The read must report **chance with every rate** (O5) and **per-arm top-1 at equal coverage**
  (O3) in its own manifest, or it repeats M20's two circular instruments.

---

## 8. Quality bars

| bar | state |
|---|---|
| flags OFF by default | `PLACEMENT_PREFILTER_V2=0`, `PLACEMENT_NODE_DESC_VARIANT=sorted`; both declared in `parameters/_text.py` |
| defaults byte-identical | see §8.1; the switch-digest tuple is unchanged (deviation 10), so the digest is M20's |
| a test that goes RED on revert, per change | §1.5, plus `ScaleFindingTests`, `AttributionTests`, `CoverageTests`, `ChanceTests`, `PairTableTests` — **14 tests, all passing** |
| no silent fallback, refusals counted | `score_bags_v2` raises without the frozen file (pinned); the coverage run records `falls_back` = 0 over 64 pairs |
| units stated | log distances, radians, seconds/query, places, slots, chance per task |
| determinism at two hash seeds | `test_coverage_is_deterministic_across_processes` runs the same draws under `PYTHONHASHSEED=0` and `12345` |
| manifest with EFFECTIVE flags at call time | every M21 result JSON is written through `m15_manifest.write_result` |
| per-cell rows kept | `m21_yaw_diag.jsonl` (140 cells), `m21_yaw_variants.jsonl` (800 cells), `m21_cov_fit_t3_pairs.jsonl` (64 pairs with every mapped place) |

### 8.1 Defaults check

`placement_m12_perf.py --check --workers 1` (28 pairs: 14 synthetic, 14 real):

```
wall 827.2s  per-pair A 6.83s  C 7.77s
identical: True  (0 diffs)
```

**0 diffs.** Every flag added in this mission is off on the default path and the deployed
numbers are byte-identical to the M20 record's. (M20's own check took 806.4 s on the same box;
827.2 s here is the same run within contention noise — no regression, and the per-pair costs are
6.83 s / 7.77 s against 6.63 s / 7.42 s.)

---

## 9. Deviations from the registration

1. **σ was not fitted.** The fit-half grid was flat (0.175–0.250) and its argmax sat on the σ_d
   boundary (0.20, 0.75). The registered defaults 0.35/0.50 were kept and the grid is reported as
   a flatness measurement. The `chosen_by` block in the params file says so.
2. **O1's acceptance is not met** (0.175 vs the required > 0.25) and is reported as a failure of
   the expected direction, not smoothed over.
3. **O2's weight is at the boundary** (w = 8.0, where the union equals appearance alone). No weight
   is shipped; the finding is the degeneracy.
4. **The DEV pipeline leaked once.** The fit script's `--stage all` ran `choose` on DEV and
   rewrote the frozen ruler (σ_a = 1.00). It was caught, the widths were restored to the FIT
   decision, every DEV number was re-measured under them, and the incident is recorded in
   `Topology/results/placement/m21_dev_leak.json`. The one file written under the leak
   (`m21_shortlist_dev_pc2.json`) was deleted rather than quoted.
5. **O3 ran on 8 queries, not 40** (stride-5, one per class), and its score-vs-count Spearman is
   n = 1 — that half of the acceptance is unanswered.
6. **O6's third guardrail (pitch)** was not run through the accuracy harness, only through the
   descriptor diagnosis. The yaw acceptance is therefore met numerically without the contrast
   that would make it meaningful.
7. **O6's first diagnosis pass is superseded** (max-impostor column, §6.2) and was not re-run.
8. **K = 24 was kept** although the measurement favours 6 (0.250 vs 0.175, 5× cheaper), because
   the difference is inside the instrument's resolution and moving it would require re-freezing
   the terms table.
9. **O4's rows are M20's runs** (16-item gallery), so its top-1 rates are not comparable with
   §1's 40-item numbers; only the tag distribution and its non-circularity are evidence.
10. **The switch-digest guard caught a real hole.** `tests/test_m14_switch_digest.py` failed on
    `PLACEMENT_NODE_DESC_VARIANT` being read inside `matching/placement_context.py`, which is on
    its build-path list. The descriptor is computed from a payload SIDE at match time and no
    build-path caller exists, so the graph does not depend on the switch; the read was moved to
    `matching/placement_prefilter.py` (a scoring module), the reasoning is written in both
    files, and `eth80_frontend.switch_suffix()` is unchanged — the 1.3 GB graph cache is not
    re-keyed, and the default path is byte-identical. Had the switch actually been baked into
    the graph, the digest would have had to change and every cached graph re-built; that
    decision is recorded rather than taken silently.
11. The `_cls_dir` helper was replaced mid-mission by a split-table lookup after a path-parsing
    version mis-derived classes; two scripts were corrected for the rename.

---

## 10. Open problems, ranked

1. **The instrument cannot resolve what O1 was supposed to measure.** 3 queries on n = 40. Every
   conclusion about the ruler is "no measurable difference"; the next read must either grow n (40
   → 160 queries) or use a finer endpoint than recall@8 (MRR, rank distribution).
2. **Geometry contributes nothing to the union at S = 8** (0.175 alone, ≤ 0.675 with appearance,
   monotone in the appearance weight). The additive union is the wrong shape; the geometry channel
   has to earn its place as a *filter* or a *tie-breaker*, tested as such.
3. **The yaw wall is not reproduced by this accuracy harness** (yaw 0.234 > in-plane 0.203). The
   missing cell is pitch (x20) on the same harness; until it exists, "the yaw wall" is a claim
   about the M20 probe, not about the configuration.
4. **T2 is below chance on FIT at the 40-gallery** (0.475 vs chance 0.617) — the class task, not
   the machinery.
5. **The coverage control has 1 hit.** It needs all 40 queries (≈3.5 h at 207 s/query) before the
   Spearman half of O3 can be answered.
6. **K is unresolved** (6 → 0.250 vs 24 → 0.175, 5× cost) because the terms table is frozen at 24.
7. **The frozen ruler's σ is a flat plateau**, so the kernel width is unconstrained and every unit
   of "tuning" spent on it is spent on noise.

---

## 11. The single next change

**Measure pitch (x20) and yaw (y20) through the same accuracy harness and the same 20 scenes, and
report each *channel* separately** (`seed_only` vs the blend at w = 0.25): the mission's premise is
that yaw is twice as destructive as pitch, and on this mission's instrument the in-plane and yaw
cells are indistinguishable (0.203 vs 0.234) while the descriptor diagnosis says pitch is the
milder view (shape 0.238 vs 0.296). One run of `placement_m21_yaw.py --stage variants` with x20
restored to `VAR_VIEWS` settles whether a yaw wall exists on this instrument at all — and if it
does, which channel owns it. Everything else in §10 is blocked behind that question being asked
correctly.
