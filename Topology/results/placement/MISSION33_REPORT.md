# MISSION 33 — the curvature channel at object level, and a record anyone can load

Branch `arena/c316ca3a-percept6`, from `75255f2` (M32).  S = 8, one physical core, long jobs in the
background and resumable.  Core in `matching/`, `core/`, `lenses/`; `Topology/` holds thin scripts
that import core.

**The question.**  M32 found that the four curvature columns (`sharp_x`, `sharp_y`, `curv_x`,
`curv_y`) separate same-class instances better than anything this project has measured — 0.596 at
place level, **0.671 at object level** against a 0.505 baseline — but it applied its +0.10 bar to the
*place-level* gauge, so the object-level number was reported as a diagnostic and never registered as
an arm.  The gauge is not the door: **no recall@8 had ever been measured for object-level curvature
aggregation.**  This mission measured it, and made the record file loadable from a fresh checkout.

---

## 1. The record file, loadable from a fresh checkout

**(a) Compression alone is already spent.**  `matching/placement_records.save` has written
`savez_compressed` since M27.  Re-compressing the 41.67 MB file gives **41.67 MB** (`--compress`:
98.79 MB of raw float64 arrays that compress 2.4×, because the rest is noise, not structure).  No.

**(b) Mixed precision, and as little of it as fits.**  The width policy is now explicit in
`matching/placement_records` (`GRADED_DTYPE`, `GRADED_ARRAYS`, and a `save(graded_dtype=...)`
argument that defaults to **exact** — a later writer cannot inherit the loss by accident):

| array | stored as | why |
|---|---|---|
| `shape_*` (nine graded columns) | **float32** | continuous measurements nobody buckets on |
| `content_*` (the six colour rings) | **float64** | at this width the file is already inside the budget, so the shipped cosine stays BIT-IDENTICAL.  Converting them too would have saved 4.0 MB nobody needed at a cost in proof |
| `rows_*` (the quantised key values + frames) | float64 | the door BUCKETS on these: 1.6 → 1.60000002 moves a bucket, and M27 measured 2 of 80 shortlists moving under float32 keys |
| `dz_*`, `mz_*` (the frozen ruler) | float64 | distance inputs |

**Size: 41.67 MB → 38.03 MB**, i.e. inside the 40 MB budget, so the file is committed
(`.gitignore` now exempts it, with the reason).

**(c) The proof, at both read views** (`Topology/m33_records.py --check`, 212 s + 228 s):

| gate | `-035-045` | `-045-180` |
|---|---|---|
| shortlists identical (live records vs the committed file) | **80/80**, 0 differences | **80/80**, 0 differences |
| the door's max abs score difference | **0.0** | **0.0** |
| the shipped cosine's max abs score difference | **0.0** | **0.0** |
| the graded kernel's max abs score difference (vs the float64 original) | **6.13e-09** | **5.30e-09** |
| the graded kernel's mean abs difference | 3.3e-10 | 3.4e-10 |
| queries whose top-8 SET changed | **0** | **0** |
| queries whose top-8 ORDER changed | **0** | **0** |
| queries whose top-1 changed | **0** | **0** |

**No column was refused float32.**  The one thing float32 *could* have cost — a different order —
cost nothing at either view, on all 80 × 80 cells of the channel that consumes those columns.

**(d) The fresh-checkout claim is a test.**  `tests/test_m33_records.py::test_t1_records_are_committed_and_load`
fails if the file is not in git (`git ls-files --error-unmatch`) or does not load in seconds;
`test_t1b_the_door_reproduces_from_the_file_alone` rebuilds the fused door's shortlist for 12 query
objects **from the committed npz alone** and requires it to equal the eight object ids M31 actually
scored, per query, in `m30_read_pairs.jsonl` — a file committed long before this mission.  No
rebuild is involved anywhere in it.

**Bootstrap time.**  Loading all 320 cells from the committed file: **21.3 s** (two views: 10.9 s).
Rebuilding the same cells from images: **~490 s** warm (`m32_records.py --build`, per-cell cache) and
far longer cold.  The file therefore replaces about eight minutes of work with twenty seconds, at a
stored cost of 38.03 MB.

## 2. The registered arm — curvature at object level, as a door channel

**Registered in writing before any number** (`Topology/m33_curvature.py`, docstring and git
history): the channel is the four curvature columns only, kernel form with the pool's own column
spread, aggregated at object level (for a query object and a gallery object: the mean over the
query's places of the best same-lens match in that gallery object); the five detection-strength
columns are excluded *because M32 measured all nine at 0.558 against the four alone at 0.596*; the
channel enters the fused door as a **fourth channel at the same half-vote colour carries**
(`rrf(counted 1.0, appearance 1.0, colour 0.5, shape 0.5)`, k = 60).  Bar: mean FIT recall@8 gain
≥ +0.050, object-clustered paired bootstrap CI (10,000 resamples, seed 20261118) excluding zero, and
DEV+FRESH not below the shipped door by more than 0.025.

Gallery = all 80 objects at `-000-000`; queries = FIT objects at `-035-045` and `-045-180`.
Cached records only, no matcher.  (`m33_curvature.py --arm`, 120 s.)

| cell | channel r@8 | channel top-1 | channel truth-rank median | channel share ≥ 0.99 | shipped fused3 r@8 / top-1 | fused4 r@8 / top-1 | FIT up / down / tied |
|---|---|---|---|---|---|---|---|
| `-035-045` | 0.225 | 0.075 | 20.0 | **0.908** | 0.662 / 0.250 | **0.738** / 0.200 | **3 / 0 / 37** |
| `-045-180` | 0.263 | 0.037 | 20.5 | **0.908** | 0.625 / 0.250 | 0.625 / 0.263 | **0 / 0 / 40** |

Mean FIT gain **+0.0375** (bar +0.050); pooled object-clustered CI **[0.000, 0.0875]**, which
**includes zero**; DEV+FRESH +0.0375 (inside the −0.025 tolerance, so the confirmation half is not
the problem).  **The bar fails on two of its three clauses**, and the channel is flat by the
mission's own test (90.8 % of its scores ≥ 0.99).

**Also reported, as a measurement and not a gate:** the gauge gain's own paired object-clustered CI,
over the eight sample objects (10,000 resamples, seed 20261118) — baseline 0.505, unclipped 0.596,
**Δ +0.091, CI [+0.042, +0.140]**.  It excludes zero (the gain is real) and it *contains* +0.10
(+0.10 is inside the interval), so M32's near-miss is **statistically indistinguishable from the
bar** — neither proved short nor proved over.  That is the honest reading of "+0.099 against 0.10".

## 3. Item 3 — skipped, and the mechanism named

Item 3 was registered as "only if item 2's channel is not flat and its CI is informative".  It is
flat (0.908) and its CI includes zero, so **item 3 is skipped** as specified, and the failure is
diagnosed instead: is it the AGGREGATION or the COLUMNS?  (`--mechanism`, 180 s.)

**The aggregation is not the limit.**  The same cells, the same columns, the other aggregator:

| aggregation | `-035-045` r@8 / top-1 / share ≥ 0.99 | `-045-180` r@8 / top-1 / share ≥ 0.99 |
|---|---|---|
| object-level mean (the arm) | 0.225 / 0.075 / 0.908 | 0.263 / 0.037 / 0.908 |
| place-level max (the M32 form) | 0.250 / 0.037 / 1.000 | 0.212 / 0.050 / 1.000 |

Two aggregations, 0.212–0.263 either way: the mean is not losing a discriminative place that the
max would have kept — and the max is *perfectly* flat.

**The columns are the limit, and the number is the intraclass correlation** — between-object
variance over total variance, per column, per view:

| column | `-000-000` | `-035-045` | `-045-180` |
|---|---|---|---|
| `sharp_x` | 0.096 | 0.152 | 0.096 |
| `sharp_y` | 0.142 | 0.191 | 0.197 |
| `curv_x` | 0.013 | 0.016 | 0.014 |
| `curv_y` | 0.009 | 0.017 | 0.028 |

**80 %–99 % of every curvature column's variance is WITHIN an object**, across that object's own
places.  A column with an ICC of 0.01–0.20 can separate two objects only as a weak population code:
no aggregation of it recovers an identity the column does not carry.  That is why the object-level
gauge (0.671) and the door channel (r@8 0.23) can both be true about the same quantity — an AUC over
80 objects survives on a weak-but-consistent signal, while a top-8 of 80 needs a strong one.

**The cross-view transfer, for completeness** (decile correlation of the per-object profile,
`-000-000` vs each query view): `sharp_x` 0.591 / 0.558, `sharp_y` 0.775 / 0.792, `curv_x` 0.395 /
0.514, `curv_y` 0.581 / 0.607.  The *profile* transfers; the identity does not.  `sharp_*` (the raw
curvature) transfers better than `curv_*` (the response-normalised one), which is the opposite of
what scale-normalisation was for, and is worth knowing before any future attempt.

## 4. Item 4 — skipped

The read runs only if item 2 or item 3 passes.  Neither did, so **no read was run and no MDE was
spent**: the all-80 estimate stands unchanged at **23/80 = 0.2875 [0.188, 0.388]** (MDE 0.121 at
n = 80, from M31).  No channel weight changed, so the deployed score is untouched — which the
identity test and item 5's `--check` both prove rather than assert.

## 5. Ship

| gate | number |
|---|---|
| records committed | `data/placement_records_m32.npz` **38.03 MB** (≤ 40 MB budget), exempted from `.gitignore` with its reason |
| records reproduce the door | **80/80 shortlists identical at BOTH read views**, 0 differences, door max \|Δ\| **0.0**, cosine max \|Δ\| **0.0**, kernel max \|Δ\| **6.1e-09 / 5.3e-09**, top-8 order changes **0 / 0** |
| bootstrap | 320 cells load in **21.3 s** from the file against **~490 s** to rebuild |
| suite | **252 passed, 6 skipped** |
| tests RED on revert | reverting `placement_shape.py` + `placement_fuse.py` to `75255f2` fails 2 of 5 M33 tests; reverting `placement_records.py`'s precision policy fails T2; all pass with M33's code |
| core-not-harness | `test_m33_records.py::test_t4_core_only` drives `PS.curvature_matrix`, the fourth `FU.fuse` slot and the blend identity from `matching.*` with no `Topology` import |
| determinism, 1/2/4 workers (`m31_determinism.py --pairs 2`) | **max abs diff 0.000e+00**, gate 1e-12, passes |
| legacy `--check` | 80/80 shortlists identical, 0 differences, max abs score diff 0.0 (both views) |
| bundle | re-frozen, `code_digest 3cf30de6f1189bb3`, door `fused3`, shortlist 8, `channels` at **w_shape 0.0 / w_colour 0.0**, with the M33 arm and its ceiling recorded in `channels.measured_m33` |
| ARCHIVE_INDEX / KEEP_MAP | M33's three harnesses added to `KEEP_MAP` §3; nothing archived this mission (the M28 one-offs already left in M32) |

## One paragraph

**Curvature at object level is a real signal and not a door channel.**  As a channel it ranks the
true object at a median rank of 20 of 80 (r@8 0.225/0.263 against chance 0.100) and is itself 90.8 %
saturated, and fused at a half vote it moves the shipped door by **+0.0375 mean FIT recall@8** — 3
objects up, 0 down, at one view and nothing at all at the other — with an object-clustered CI of
**[0.000, 0.0875] that includes zero** against a +0.050 bar.  The mechanism is in the columns, not
the aggregation (max and mean both land at 0.21–0.26): **80–99 % of every curvature column's
variance is within an object** (ICC 0.009–0.197), so the profile transfers across views (decile
correlation 0.56–0.79) while the identity does not; and M32's +0.099 gauge near-miss, now with its
own interval **[+0.042, +0.140]**, is indistinguishable from the +0.10 bar rather than short of it.
**The record now costs a fresh checkout 38.03 MB and 21 seconds** instead of eight minutes of
rebuilding — nine graded columns at float32, everything the door buckets on still exact, 80/80
shortlists identical at both views, cosine delta 0.0, and not one of 80 queries changing its top-8
order — and the all-80 estimate is unchanged at **0.2875 [0.188, 0.388]** because nothing entered the
door.  The single next lever is no longer shape-as-a-channel but **shape-as-a-re-ranker inside the
eight**: a channel that is 2.3× chance cannot earn a fusion vote, but at ICC ≤ 0.20 it can still
decide between two *same-class* candidates that the door already shortlisted — which is where 31 of
M31's 57 losses were lost, and the only place a weak-but-consistent signal pays.
