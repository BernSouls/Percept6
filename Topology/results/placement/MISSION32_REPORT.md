# MISSION 32 — stop flattening the channels: colour and shape, one diagnosis, two fixes

Branch `arena/c316ca3a-percept6`, from `ed87d54` (M31).  S = 8, one physical core, every long job
background, resumable, with a progress file.  Core in `matching/`, `core/`, `lenses/`; `Topology/`
holds thin scripts that import core.

**The diagnosis this mission tested.**  Both channels carry signal; both are read through a
transform that removes their variation.  Colour: the shipped similarity is a cosine on a
unit-normalised six-vector of CIELAB ring means, and the ring vector's norm is almost entirely
lightness — the direction is nearly the same for every place, so the score is nearly 1 for every
pair.  Shape: `lenses/keypoints.py::_parabola` returns `var = 1/(2|den|)` clipped to `[1e-3, 1]`,
and on normalised responses `|den|` is far below the clip for every place, so the column is a
constant.  M31 proved a constant column inert but replaced it with *different* quantities (0.508,
no better).  The untried fix was to expose the SAME quantity, unclipped.

---

## 1. Hygiene

**(a) No hard-coded home.**  Fifteen scripts derived `ROOT = "/home/user/Percept6"`; each now
derives it from `__file__` (`Topology/placement_m11_eth80.py`, `m12_perf`, `m12_read_real`,
`m12_realphotos`, `m20_eth80`, `m20_fit`, `m21_attrib`, `m21_coverage`, `m21_fit`, `m21_shortlist`,
`m21_terms`, `m22_eth80`, `m22_power`, `m23_eth80`, `matching/m15_manifest.py`).  Zero literals
remain outside `.venv`/`archive`, and `tests/test_m32_design.py::test_t1_no_hardcoded_home` walks
every `.py` in the tree and fails on one; `test_t1b_root_is_derived_from_file` executes each ROOT
line and asserts it resolves to the repository root.

**(b) Pair files key on OBJECT IDS.**  `matching/placement_records.object_id_from_path` reads the
class FOLDER and the instance FOLDER, never the file name, and refuses a malformed path rather than
falling back — the corpus's `car12-000-000.png` lives in class folder `2` and instance folder `8`,
so the name is a different numbering.  `object_id()` accepts an id or a path so a pair file may hold
either.  `test_t2_object_ids_over_all_80` walks all 80 objects × 3 views (240 round trips) and
`test_t2b_folder_wins_over_filename` pins the `car/8` case and both refusals.  `Topology/m31_arms.py`
was keyed on `os.path.dirname(basename(path))` before this mission; it now calls `object_id`.

**(c) The archive.**  `MISSION30_REPORT.md` and `MISSION31_REPORT.md` moved next to M14–M29
(`Topology/results/placement/`).  Sixteen one-off M26–M31 harnesses moved to `archive/Topology/`
with `git mv`, chosen by AST closure: kept are exactly the scripts M32's gates or the records/read
chain need (`m26_records`, `m26_oracle`, `m27_read`, `m27_export_records`, `m30_read`,
`m30_records`, `m30_parity`, `m31_determinism`), everything else in that range left the live tree.
`ARCHIVE_INDEX.md` §"Mission 32" lists each file and its reason; `KEEP_MAP.md` §3 now names the M32
harnesses.

**Four defects the hygiene work uncovered, each fixed and tested:**

| what | how it was wrong | fix |
|---|---|---|
| `matching/placement_bundle.freeze()` | its defaults are the M24 ones, so re-freezing after a code edit silently reverted the shipped configuration — M32 watched the door kind fall from `fused3` to `counted` while the digest said "this tree" | `refreeze_from_current()` + `python -m matching.placement_bundle --refreeze`: the configuration is kept, only `code_digest` moves |
| `CODE_MODULES` | the digest covered seven modules; the fuse, the matcher, the graded kernel, the shape similarity, the record loader and the channel blend could all move a shipped score and none was covered | all six added — the digest now covers every module that can change a score |
| `placement_recognise.build_gallery` | never passed the bundle's content kind, so a gallery built through the entry point was `keypoint` under a `colour_rings` bundle and was refused downstream | passes `content=b.content_kind` |
| the test suite | `test_m15_manifest` leaked the keypoint registries into `test_m16_graded_keypoints` (pre-existing at M31's tip, proven in a worktree: 1 failed there too), and three M24/M26/M30 tests pinned the M24 shortlist 16, the M24 door kinds and the five-column arity | isolation in the M16 setup; the stale pins now read the shipped configuration |

Suite: **247 passed, 6 skipped** (was 195 at M27, and *not* green at M31's tip).

## 2. Measure the flattening — the baseline

`Topology/m32_saturation.py` (`--run`, 62 s, records only).  ONE cell per channel: the 80 query
objects at `-035-045` against the 80 base-view (`-000-000`) objects, 6,400 pairs, both channels
read on the same cell with the aggregator each channel ships (max over place pairs).

| channel | share ≥ 0.99 | share ≥ 0.999 | truth median | impostor median | truth − impostor | distinct |
|---|---|---|---|---|---|---|
| colour (cosine on the unit six-vector) | **1.000** | 0.978 | 0.999995 | 0.999969 | **+0.000026** | 6,084 |
| shape (graded form on the five columns) | **1.000** | 0.974 | 0.999703 | 0.999641 | **+0.000062** | 6,291 |

Both channels are flat, and they are flat in the SAME way and by the same amount — the mission's
colour number reproduces exactly (97.8 % of pairs ≥ 0.999) and the shape channel is 97.4 %.  The
columns are not flat: the six colour dimensions take ~43,000 distinct values each over the 46,059
places of a view, and so do `resp`, `eigratio` and `rel_angle` (39,419 / 35,774 / 40,047 at
`-000-000`).  What is flat is the SCORE.  `var_x` is the exception: 4 distinct values at
`-000-000`, **1** at `-035-045`, 2 at `-045-180` — the clip, exactly as M31 found.

So: the signal is in the columns and the transform throws it away.  Two channels, one mechanism.

## 3. Colour — a real distance and a real aggregate

Fixed in writing before any score was looked at (`matching/placement_content.py`):

```
d² = Σ_rings [ (Δa)² + (Δb)² + λ_L (ΔL)² ]      λ_L = 0.25, scaled Lab units
s  = exp( −d² / (2σ²) )                          σ  = median pairwise distance over a fixed,
                                                     label-free sample of 4,096 base-view places
                                                     (np.random.default_rng(20261118))
```

σ = **0.350541** (stored in the bundle under `channels`).  `features()` now returns the ring vector
RAW; the shipped cosine normalises internally, so its scores are unchanged (proved, §6).  Three
aggregators, fixed in writing: **V1 max** over place pairs (the shipped one), **V2 query-coverage**
(mean over the query's places of the best gallery match), **V3 symmetric chamfer**.

Cells: gallery = all 80 objects at `-000-000`; queries = the FIT objects at `-035-045` and at
`-045-180`.  `Topology/m32_colour.py --channels --variants` (155 s + 103 s).

| FIT cell | variant | colour r@8 | colour top-1 | colour share ≥ 0.99 | colour gap | fused3 r@8 | fused3 top-1 | FIT up/down/tied |
|---|---|---|---|---|---|---|---|---|
| `-035-045` | shipped cosine | 0.713 | 0.175 | 1.000 | 0.000026 | **0.662** | 0.250 | — |
| | V1 max | 0.675 | 0.150 | **0.416** | **0.01201** | 0.650 | 0.200 | 3 / 4 / 33 |
| | V2 coverage | 0.100 | 0.013 | 0.938 | 0.000000 | 0.475 | 0.113 | 0 / 7 / 33 |
| | V3 chamfer | 0.100 | 0.013 | 0.188 | 0.000000 | 0.475 | 0.113 | 0 / 7 / 33 |
| `-045-180` | shipped cosine | 0.600 | 0.163 | 1.000 | 0.000028 | **0.625** | 0.250 | — |
| | V1 max | 0.575 | 0.150 | **0.384** | **0.01308** | 0.588 | 0.237 | 0 / 2 / 38 |
| | V2 coverage | 0.100 | 0.013 | 0.900 | 0.000000 | 0.487 | 0.138 | 0 / 6 / 34 |
| | V3 chamfer | 0.100 | 0.013 | 0.175 | 0.000000 | 0.487 | 0.138 | 0 / 6 / 34 |

**Verdict: none passes.**  Mean FIT fused3 recall@8 gain over the current door: V1 **−0.0225**,
V2 **−0.1625**, V3 **−0.1625**; the bar was +0.050.  Blame, split exactly as the mission asked:

* **The DISTANCE is fixed.**  The flattening is gone: the share of pairs ≥ 0.99 falls from 1.000 to
  **0.416** (V1 at `-035-045`), and the truth−impostor gap grows from 0.000026 to **0.01201** — 460×.
  That is what an unflattened channel looks like, and it is the first time this project has had one.
* **The AGGREGATOR is to blame for V2/V3.**  A mean over ~500 query places is a bulk statistic: it
  scores exactly chance (r@8 0.100 = 1/10 at 8 of 80), and its gap is 0.000000 — the mean is the
  same number for every gallery object.  Coverage and chamfer are not usable aggregators for this
  content at this place count.
* **For V1 the rank was never colour.**  With L down-weighted to 0.25 the colour-alone r@8 *falls*
  (0.713 → 0.675 and 0.600 → 0.575).  The shipped "colour" channel ranks on LIGHTNESS: studio
  lighting is stable across an object's own photographs, so a lightness-dominated cosine is a good
  object cue that happens to be called colour.  De-flattening the channel removes exactly the cue
  that was carrying it.  That is the mechanism, and it says the door's colour term is misnamed, not
  mis-scaled.

Per the mission: no variant passes, the blame is named, and the channel stops here.

## 4. Shape — unclip

**(a) The quantity is carried.**  `lenses/keypoints.py` now exposes, per axis, `|den|` of the
quadratic peak fit (`sharp_x`, `sharp_y`) and `|den|/|c0|` (`curv_x`, `curv_y`), through
`_parabola_curv`; `matching/placement_shape.FIELDS` grows 5 → 9 and `FORMAT_VERSION` 29 → 30, with
ranges (0.12/0.13/1.4/1.5, measured 99th percentiles), weights and population axes all registered
(`PS.missing_fields() == []`, an unregistered class still returns 0.0).  They are GRADED COLUMNS on
the place record, not new key slots: `signature_of` reads `theta`, `coherence`, `scale`, `kind` and
never these, and `Topology/m32_records.py --check` proves every key byte-identical cell-for-cell
against the M30 file, the frames identical, and the shipped cosine unchanged (max Δ 0.0).

Distinct values, per view, over every place (the test of whether a column can separate at all):

| column | `-000-000` | `-035-045` | `-045-180` |
|---|---|---|---|
| `var_x` | 4 (sd 0.0005) | **1 (sd 0)** | 2 (sd 0.00003) |
| `var_y` | 8 | 2 | 16 |
| `sharp_x` | **38,716** (sd 0.044) | **41,887** | **42,803** |
| `sharp_y` | 38,739 | 41,950 | 42,871 |
| `curv_x` | 40,363 (sd 1.07) | 42,971 (sd 0.50) | 43,955 (sd 0.53) |
| `curv_y` | 40,341 | 43,016 | 43,972 |

**(b) The same distance form as colour.**  The shipped shape similarity (`graded_matrix`) divides
each axis by an arbitrary physical span on a one-sided range and averages `1 − |Δ|`: at `-035-045`
that gives a truth−impostor gap of 0.000062, i.e. the shape channel is flat for exactly the colour
channel's reason.  `matching/placement_shape.kernel_matrix` / `kernel_block` replace it with
`exp(-½ Σ_j w_j ((Δ_j)/σ_j)²)`, `σ_j` the pool's own column spread, the circular axis compared on
the circle — the same form, the same standardisation and the same unit-variance logic as item 3.

**(c) The M28 gauge** (`Topology/m32_shape.py --gauge`, 8 objects, one per class, both views,
**8,721** query places against all 80 base-view objects, chance 0.500):

| variant | G_any | **G_same** | G_any (object) | G_same (object) |
|---|---|---|---|---|
| shipped signature (baseline) | 0.524 | **0.505** | 0.624 | 0.572 |
| live M30 columns, shipped form | 0.659 | 0.564 | 0.713 | 0.611 |
| **unclipped columns alone, kernel form** | **0.742** | **0.596** | **0.834** | **0.671** |
| all nine, kernel form | 0.634 | 0.558 | 0.662 | 0.590 |

**Verdict: fails by 0.009.**  The unclipped columns gain **+0.091** same-class over the baseline
this harness measures (0.505 → 0.596); the bar is +0.10.  Read against the mission's pre-registered
0.494 the same number is +0.102, but the valid comparison is the paired one on one harness, so the
honest verdict is *short*, and it is recorded as short.  Three observations, all mechanism:

* **The COLUMN is not the limit.**  `sharp_*`/`curv_*` have 38,000–44,000 distinct values per view
  where `var_x` has one.  The thing M31 could not get is there; the metric is what fails to use all
  of it.
* **The SIMILARITY is the limit, and only just.**  Same-class separation is 0.596 against a chance
  of 0.512 on this sample — real, graded, and 0.009 short of the bar.
* **The five shipped columns DILUTE the four new ones.**  Combined (9) = 0.558 against unclipped
  (4) = 0.596: `resp`, `eigratio` and `rel_angle` are detection-strength and orientation, not
  instance identity, and equal weights let them vote.  Any future attempt should weight the
  curvature columns alone rather than add them to the current five.
* **Object-level aggregation is worth +0.075.**  Averaging each query object's places before the
  comparison lifts G_same from 0.596 to **0.671** (and G_any to 0.834) — the opposite of the colour
  channel, where the mean destroyed the signal.  That asymmetry is the finding to carry forward.

## 5. Wire and the one read

The blend is implemented, in core, with the mission's exact arithmetic and constraints
(`matching/placement_blend.py`):

```
sim' = (1 − w_shape − w_colour)·matrix + w_shape·shape + w_colour·colour
```

with the configuration term keeping its shipped 0.25 (it is inside `matrix`), `w_colour ≤ 0.15`,
`w_shape ≥ w_colour`, `0 ≤ w_shape + w_colour ≤ 1`, and `w_shape = w_colour = 0` returning the
matrix OBJECT itself — the deployed score, byte-identical, which is
`tests/test_m32_blend.py::test_zero_weights_is_the_deployed_matrix`.  Non-convexity, a colour weight
above the cap, shape below colour and a negative weight are refused, not clamped.

**No channel cleared its bar, so nothing entered: the shipped weights are `w_shape = 0.0`,
`w_colour = 0.0`, and the bundle records them with the measurements that produced them
(`channels` block, `matching/placement_bundle.json`).**  The read was therefore not run: with both
weights zero it *is* the deployed arithmetic, pair for pair, so a 4.4 h run would have re-measured
M31's number and could not have measured an effect.  The MDE that would have been needed — 0.121 at
n = 80 — against the planning effect is recorded here instead, and the all-80 estimate is unchanged
from M31 by construction: **23/80 = 0.2875 [0.188, 0.388]**, FIT 0.350, DEV 0.208, FRESH 0.250.

## 6. Ship

| gate | number |
|---|---|
| suite | **247 passed, 6 skipped** |
| tests RED on revert | reverting `matching/placement_content.py` and `lenses/keypoints.py` to `ed87d54` fails **5** of the 13 M32 tests; restoring passes all 13 |
| core-not-harness | `test_m32_design.py::test_t3_core_not_harness` drives the distance, the aggregators, the kernel and the record columns from `matching.*`/`lenses.*` only |
| determinism, 1/2/4 workers (`m31_determinism.py --pairs 2`) | 126 s / 50 s / 90 s, **max abs diff 0.000e+00** |
| legacy `--check` on the M32 records (`m27_export_records.check`) | **80/80 shortlists identical, 0 differences, max abs score diff 0.0** |
| records size | `data/placement_records_m32.npz` **41.67 MB** — above the 40 MB commit limit, so NOT committed; it is rebuilt by `Topology/m32_records.py --build` (per-cell cache in `.cache/placement_m32`, ~8 min warm) and `--check` re-proves it |
| bundle | re-frozen, `code_digest f15eabde22d313a4`, door `fused3`, content `colour_rings`, shortlist 8, `channels` at zero |

## One paragraph

Saturation, before and after, for both channels: colour went from **100 % of pairs ≥ 0.99 and a
truth−impostor gap of 0.000026** to **41.6 % and 0.01201** — a real channel for the first time — and
shape went from **97.4 % ≥ 0.999, gap 0.000062** with one distinct value in `var_x` to a kernel form
that puts **38,000+ distinct curvature values** and a same-class gauge of **0.596** (from 0.505) on
the table.  What shipped is the measurement machinery and the honesty: the records carry the
curvature as graded columns (keys byte-identical, cosine unchanged, `--check` 0/80 differences), the
colour channel has a real distance with a stated σ and λ_L, both channels have a common kernel form,
the blend exists in core at zero weight with an identity test, and the bundle says why.  The all-80
estimate is unchanged at **23/80 = 0.2875 [0.188, 0.388]** (MDE 0.121): neither channel could clear
its pre-registered bar on FIT, so nothing was allowed to enter the door.  The single next lever is
the one this mission exposed and did not spend: **the shape channel's aggregation** — `sharp_`/`curv_`
with equal weights over four columns, aggregated at OBJECT level rather than as a place-pair max,
which the gauge already measures at **G_same 0.671 / G_any 0.834** against the place-level 0.596 /
0.742, and which is the only channel in this project that separates same-class instances above
chance by a margin the door can use.
