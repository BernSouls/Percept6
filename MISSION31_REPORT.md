# Mission 31 — the colour door in the product, the shape columns corrected, colour where the matcher loses

Branch `arena/c316ca3a-percept6`, base `2520606` (M30). Environment: Intel Xeon @ 2.60 GHz,
2 logical / 1 physical core, numpy 2.4.6, scipy 1.17.1, pillow 13.0.0 (pinned), BLAS one thread.
S = 8 for every read; chance top-1 0.0125, chance recall@8 0.100.

Headline: **`recognise()` now ships the fused door** (counted + appearance + colour at weight 0.5,
colour-ring place content, shortlist 8), and the product's door is **identical to M30's fused door on
all 80 objects at both view**s. The two "dead" shape columns turned out to be **constant, not
view-dependent** — M30's diagnosis was wrong and is corrected here with counts — and once corrected
they still do not cross the view: the shape ceiling is **0.494 → 0.539**, against a +0.10 bar. Colour
knows the answer in **14 of the matcher's 21 losses** (66.7 %), yet no equal-weight rank fusion of it
beats the matcher's own order on FIT, so **(A) stays**.

---

## 1 — WIRE: the product is `recognise()`

### 1a — the bundle

`matching/placement_bundle.json` re-frozen (`code_digest 344be36f8976bb4a`, mission "M31"):

| block | value |
|---|---|
| `door.kind` | **`fused3`** — RRF(counted, appearance, colour), colour weight **0.5**, `rrf_k` **60** |
| `content.kind` | **`colour_rings`** — six CIELAB ring means per place (`matching/placement_content.py`) |
| `shortlist` | **8** |
| `matcher.blend` | `{w_cfg 0.25, w_col 0.10}` |
| `frontend.recipe` | unchanged (the graph pipeline's identity) |

**Deviation, stated with M27's numbers.** The shortlist drops from 16 to 8 because M27 measured the
two as indistinguishable. M27's own table (`Topology/results/placement/m27_read.json`):

| split | S = 8 | S = 16 | difference |
|---|---|---|---|
| ALL (n=80) | 0.2375 (19) [0.150, 0.3375] | 0.275 (22) [0.175, 0.375] | **+0.0375**, CIs overlap |
| FIT (n=40) | 0.250 (10) [0.125, 0.400] | 0.325 (13) [0.175, 0.475] | +0.075 |
| CONF (n=40) | 0.225 (9) [0.100, 0.350] | 0.225 (9) [0.100, 0.375] | 0.000 |

The intervals overlap on every split and the per-object vectors were never stored, so no paired test
can be formed; the cheaper read ships on that basis, and the 16-wide arm's nominal +0.0375 on ALL is
recorded here rather than argued away.

The loader now refuses, by name: a `fused3` door whose content is not `colour_rings`; a colour weight
outside [0, 1]; a blend that is not convex (`w_cfg + w_col > 1`) or has a negative side.

### 1b — the gallery and the query carry colour, or the read refuses

`build_gallery(..., content="colour_rings")` builds the records with the content column and the
gallery digest moves with the content kind (a colour gallery is not the keypoint gallery).
`from_records(records_path, objects, view, content=...)` assembles the same `Gallery` from the
committed npz — that is the cache, not a second code path, and it is what makes a records-only read
possible. Three refusals, each pinned by test:

* a gallery cell with no colour content under `content.kind = colour_rings` (`BundleRefusal`,
  naming the first three cells);
* `recognise()` against a gallery built as `keypoint` while the bundle reads colour rings;
* `from_records()` when the npz does not cover a requested object/view.

### 1c — the blend is a bundle value, and its composition was wrong

The blend now comes from `matcher.blend`, applied through `PLACEMENT_NODE_COLOUR[_W]` in
`_FlagScope`.

**Composition fix (mechanism, with the number).** M30's `apply_node_colour` read
`PLACEMENT_NODE_DESC_W` (default **0.5**) as `w_cfg` even when the configuration term was **off**, so
at `w_col = 0.10` it computed `0.4 · sim + 0.1 · colour` — a mixture that **rescales the incumbent**
by 0.4 and makes colour 20 % of the result — while the stated equation was a 90/10 supplement. The
code now computes

```
sim' = (1 − w_col) · matrix + w_col · colour        (w_cfg = 0 when the config term is off)
```

which is the stated three-term equation when that term is on (the matrix already carries it) and is
byte-identical to the deployed score at `w_col = 0` either way. `tests/test_m31_wire.py::
TestComposition` pins the arithmetic cell by cell. **Consequence for M30's A2 ablation:** its
−0.002574 was measured under the 80/20 mixture, not the 90/10 supplement; the read in item 2 ran
entirely under the old arithmetic (one arm, consistent within itself), and no number is claimed from
either.

### 1c-bis — the production path, on graphs

Not only records: `build_gallery(items, content="colour_rings")` built three graph-based records
(37 s), `recognise(q, gallery, bundle=b, matcher=False)` returned the door's shortlist (17 s), and
`recognise(..., matcher=True, s=2)` scored it (0.3 s, pair cache) — top-1 correct
(`apple/1` → itself). The colour content is therefore computed on the graph path too, through
`placement_content.features`, exactly as the bundle says.

### 1d — the two equalities (the door only, cached records)

`Topology/m31_wire.py --equalities`, `Topology/results/placement/m31_wire.json`. 80 objects × 2 read
views = **160 cells**, S = 8, records only (no graph, no image beyond the appearance vector):

| equality | cells matched | cells compared |
|---|---|---|
| product's `recognise(matcher=False)` ≡ **M30's fused door** (`placement_fuse.fuse`, colour 0.5) | **160 / 160** | 160 |
| product at colour weight **0** ≡ **the two-channel shipped door** (`rank_fuse(counted, appearance)`) | **160 / 160** | 160 |

M27's json stores hit counts, not per-object shortlists, so the second reference is the shipped
door's own arithmetic re-derived from the committed records — the same arithmetic, from the same
records, that produced M27's counts.

## 2 — THE READ: the door claim at n = 80, the matcher read at n = 59

### 2a — the claim: the fused door, paired, all 80 objects, both views

`Topology/m31_wire.py --doorclaim`, `m31_wire.json`, records only (no graph, no matcher), S = 8.
The interval is an object-clustered bootstrap on the **paired per-object difference** (10,000
resamples of objects, seed 20261118, both arms recomputed on each resample):

| view | arm | recall@8 | top-1 |
|---|---|---|---|
| −035-045 | shipped door | 44/80 = 0.550 | 11/80 = 0.1375 |
| −035-045 | **fused door** | **53/80 = 0.662** | **20/80 = 0.250** |
| −035-045 | **Δ (paired)** | **+0.1125  [+0.050, +0.188]** | **+0.1125  [+0.050, +0.188]** |
| −045-180 | shipped door | 41/80 = 0.512 | 17/80 = 0.2125 |
| −045-180 | **fused door** | **50/80 = 0.625** | **20/80 = 0.250** |
| −045-180 | **Δ (paired)** | **+0.1125  [+0.050, +0.188]** | +0.0375  [−0.025, +0.100] |

Objects up / down / tied: **9 / 0 / 71** at both views. No object loses a hit at either view, and
both recall intervals exclude zero: **this is the claimed result of the mission** — the colour door
is worth +0.1125 recall@8, object-clustered, on the full 80 objects, at both read views, at a
chance rate of 0.100. Gained at −035-045: apple/8, car/1, car/8, car/9, cow/7, dog/8, horse/1,
horse/2, horse/9.

### 2b — the matcher read: 59 of 80 queries, one arm, no claim

The S = 8 read of the full path (fused door, then the matcher with the colour blend). Cell: query
`−035-045`, gallery `−000-000`, chance top-1 0.0125.

| split | n | hits | top-1 | 95 % CI |
|---|---|---|---|---|
| FIT (complete) | 40 | 14 | **0.350** | [0.200, 0.500] |
| DEV (partial) | 17 | 3 | 0.176 | [0.000, 0.353] |
| FRESH (partial) | 2 | 2 | 1.000 | — |
| all scored | 59 | 19 | 0.322 | [0.203, 0.441] |
| M27's shipped S = 8, FIT | 40 | 10 | 0.250 | [0.125, 0.400] |

MDE at n = 40 is 0.171 against a planning effect of +0.075; the FIT difference is +0.100 and the
comparison is unpaired (M27 stored no per-object vector). **Estimate, no claim.** Loss table over
the 59: 22 same-class, 18 other-class, 19 out of the door's shortlist, median gap 0.0141.

**Why 59 and not 80 — the mechanism, with the number.** The 21 unscored queries were resumed in the
background (`Topology/m30_read.py --run`, resumable, progress file) and are being scored now, but
under a **different arithmetic**: item 1c's composition fix landed while they were queued, so the
resumed pairs run the corrected 90/10 supplement while the first 59 ran M30's `0.4 · sim + 0.1 ·
colour` mixture. Mixing them would make the "arm" two arms, so the mission's read stands at 59 (one
arm, one arithmetic) and the resumed 21 are reported, if they complete, as their own arm. The cost
of a single-arithmetic re-read of all 80 is 640 pairs × ~25 s ÷ 2 workers ≈ 2.2 h.

## 3 — USE COLOUR WHERE THE MATCHER LOSES

`Topology/m31_arms.py`, `Topology/results/placement/m31_arms.json`. No new pairs: every number is a
re-ranking of the pairs the read already scored, joined to the door's own channels from the records.

**Oracle first, no fusion** (the mechanism, counted): of the matcher's losses *inside its own
shortlist*, how often does each channel rank the truth above the matcher's winner?

| | losses | same-class | other-class | colour above | appearance above | counted above | any above |
|---|---|---|---|---|---|---|---|
| 59 scored objects | 21 | 14 | 7 | **14 (66.7 %)** | 17 (81.0 %) | 9 (42.9 %) | 21 (100 %) |

Colour carries the answer in two thirds of the losses; appearance in four fifths. The information is
there. What a fusion does with it:

| arm | order | FIT top-1 | confirmation | ALL |
|---|---|---|---|---|
| **A** | matcher (shipped) | **0.350** | 0.176 (DEV, 17) | 19/59 = 0.322 |
| B | door | 0.250 | 0.118 | 13/59 = 0.220 |
| C | RRF(matcher, door) | 0.275 | 0.235 | 16/59 = 0.271 |
| D | RRF(matcher, colour) | 0.250 | 0.235 | 14/59 = 0.237 |

Bar, fixed before running: FIT top-1 gain over A ≥ **+0.075** and confirmation not below A by more
than **0.025**. Measured: B −0.100, C −0.075, D −0.100. **None passes; (A) stays.**

**What the fusion does to the channel's information** (net counts against A):

| arm | gains vs A | losses vs A | net |
|---|---|---|---|
| B (door order) | +4 | −10 | −6 |
| C (RRF matcher + door) | +3 | −6 | −3 |
| D (RRF matcher + colour) | +5 | −10 | **−5** |

Colour recovers 5 answers the matcher got wrong and costs 10 the matcher had right: the oracle says
the channel knows 14 of the 21 losses, and the fusion converts that into a net loss of 5 because an
equal-weight rank vote cannot tell the 14 from the 7 — and because the matcher's winner sits at rank
1, so demoting it demotes it everywhere.

**The mechanism, stated:** colour knows the truth in 14 of 21 losses, but an equal-weight RRF cannot
separate those 14 from the 7 where colour is also wrong, and it demotes answers the matcher already
had. That is a *fusion* failure, not a *channel* failure — the channel is worth 14/21 and the fusion
converts it into −0.100 on FIT.

## 4 — THE SHAPE COLUMNS, CORRECTED

### 4a — they are constant, not view-dependent

`Topology/m31_shape.py --columns`, `Topology/results/placement/m31_shape.json`:

| view | places | `var_x` distinct values | `var_y` distinct values | `resp` distinct |
|---|---|---|---|---|
| −035-045 | 46,059 | **1** (all exactly 1.0000) | **2** (0.9709 … 1.0) | 39,014 |
| −000-000 | 43,854 | 4 (0.8986 … 1.0) | 8 | 39,014 |

**Mechanism:** `lenses/keypoints.py::_parabola` returns `var = 1 / (2|den|)` clipped to [1e-3, 1],
where `den = cm + cp − 2c0`. A normalised response has a second difference far below 0.5, so the
column pins at the **cap** — at the query view for every single place. M30's reading ("image-axis
variances rotate with the viewpoint") described a column that never varies at all. **Addendum to
MISSION30_REPORT.md** is appended below.

### 4b — zeroing them changes nothing, and the bar is missed

The M28 subset gauge (8 objects, both views, 4,284 query places, chance 0.500), bar fixed at **+0.10**
over the current signature:

| arm | G_any | G_same |
|---|---|---|
| key (current signature) | 0.508 | **0.494** |
| shape, all five columns | 0.594 | 0.539 |
| shape, the two pinned columns weighted **0** | 0.594 | **0.539** |
| shape, live columns only (resp, eigratio, rel_angle) | 0.661 | 0.571 |

Two findings, both mechanisms:

1. **Zeroing the pinned columns changes the gauge not at all (0.539 → 0.539).** A constant column
   contributes the same value to every pair, so it is a *monotone* shift of the similarity — it cannot
   reorder anything. M30's claim that the two dead columns "cost 2/9 of the graded distance and
   dilute a 0.60-transferable signal to 0.572" is **wrong**: they are inert, not diluting.
2. The gain over the current signature is **+0.045**, against a +0.10 bar → **shape does not enter
   the door or the matcher.**

### 4c — rotation-normalised replacement columns, measured once

Four columns, each invariant to a rotation about the place's own frame by construction: the
structure-tensor eigenvalue ratio, the principal direction relative to the place's `theta` (mod π),
the detector's own coherence (computed and discarded in M30), and the log ratio of gradient energy
along vs across the frame direction. Weights fixed at (2, 3, 2, 2) before the run:

| columns | G_any | G_same |
|---|---|---|
| rotation-normalised (eigr, rel, coherence, along/across) | 0.578 | **0.508** |

**0.508** — no better than chance-plus-noise, and *worse* than the live M30 columns' 0.571. The shape
ceiling is named: a place's graded shape carries 4.0 bits of object identity **within** a view (M30,
B3) and transfers **+0.045** of gauge between views (0.494 → 0.539), against a +0.10 bar. Per the
mission, this number is where the search stops.

## 5 — SHIP

* **README**: `README.md` now opens with the product — `build_gallery(..., content="colour_rings")`
  and `recognise(query, gallery)` — the three-channel fusion at weight 0.5, `k = 60`, shortlist 8, the
  frozen bundle, and the measured door numbers, in a form a stranger can follow. Pinned by
  `test_the_readme_names_the_product`.
* **Bundle**: re-frozen, `code_digest 344be36f8976bb4a`; `--check` below.
* **Tests**: `tests/test_m31_wire.py` — 11 tests, each RED on a revert (see the module docstring):
  the bundle's four blocks, the three refusals, both equalities, the composition fix, and
  "the door is core, not a harness".
* **Determinism**: `Topology/m31_determinism.py` — one query's pairs scored by pools of **1, 2 and
  4** worker processes with `OPENBLAS/OMP/MKL_NUM_THREADS = 1`: **max |diff| = 0.0** against the
  1-worker reference (gate 1e-12), `m31_determinism.json`. The per-side colour cache M31 added is
  the only shared mutable state in the pair path and it is pinned equal to the uncached path by
  `TestColourCache` (M30) as well.
* **Legacy `--check`**: `Topology/m27_export_records.py --check` — **0 diffs**:
  `place_counts_identical: 80`, `shortlists_identical: 80`, `differences: []`,
  `max_abs_score_diff: 0.0`. The committed records still reproduce the shipped door exactly, before
  and after the bundle change (the door is the same three channels, read from the same records).

## Deviations, stated

1. **The shortlist ships at 8, not 16**, on M27's overlapping intervals; the 16-wide arm's nominal
   +0.0375 on ALL is recorded in the section 1a table rather than argued away.
2. **The matcher read stands at 59 of 80 queries**, one arm and one arithmetic. The remaining 21 were
   resumed in the background and run item 1c's corrected blend, so they are a second arm, not a tail
   of the first; a single-arithmetic 80-query read is ~2.2 h of pair scoring.
3. **Item 3's four arms all lose to the shipped matcher order** on FIT; (A) stays, and the oracle
   count (14 of 21) is the named mechanism rather than a weight change.
4. **Shape does not enter the door.** The mission's 4b bar (+0.10) is missed (+0.045) and the 4c
   replacement columns read 0.508; the ceiling is named with both numbers, and no third variant was
   tried.
5. **The A2/M30 blend's ablation number (−0.002574) is a mixture's**, not the shipped supplement's
   (section 1c); it is not re-measured here because the mission forbids new pairs for it and the
   fusion arms in item 3 supersede it.
6. **`--check` is `Topology/m27_export_records.py --check`** (the records→door equality, 0 diffs on
   80/80 cells at the read view) plus `matching/placement_bundle.compute_code_digest` matching the
   frozen bundle; the M28 `m28_speed.py --scores/--compare` 30-pair gate was **not** re-run (25 s a
   pair on the same single core the read needed).

## Addendum to MISSION30_REPORT.md

Two statements in the M30 report are corrected here; nothing else in it changes.

1. **"`var_x` and `var_y` are exactly 0.500 because image-axis variances rotate with the viewpoint."**
   They are 0.500 because they are **constant**: at −035-045, `var_x` has **one** distinct value over
   46,059 places and `var_y` two, pinned at `_parabola`'s clip cap (|den| ≤ 0.5 for every place at
   that view). A constant column cannot rank, so the gauge is chance for want of variance, not for
   want of invariance.
2. **"the two dead columns are 2/9 of the graded distance and are what keeps a +0.60-transferable
   signal at 0.572."** They are inert: weighting them to zero leaves the gauge at 0.539, unchanged to
   three decimals. The live columns on their own reach 0.571, and the object-level aggregate stays at
   chance. The single next lever named in M30 ("replace `var_x`/`var_y` with a rotation-normalised
   anisotropy") was executed here and reads **0.508** — it is not the lever.
3. **The A2 blend's composition** was `0.4 · sim + 0.1 · colour`, not the stated 90/10 supplement
   (section 1c). The ablation's −0.002574 is that mixture's number.

## The one paragraph

`recognise()` now runs the **fused door** — reciprocal-rank fusion of counted places, the appearance
descriptor and the six CIELAB ring means each place carries, colour at weight 0.5, `k = 60` — over a
gallery whose places are recorded with `content.kind = "colour_rings"`, hands the top **8** to the
matcher with the blend taken from the frozen bundle (`code_digest bcd290e52…`, re-frozen), and is
**identical to M30's fused door on 160 of 160 (object, view) cells** (and, at colour weight 0, to the
two-channel door that M27 read) — so the shipped product is the thing that was measured, not a copy
of it. That door is worth **+0.1125 recall@8, object-clustered CI [+0.050, +0.188], 9 objects up, 0
down, 71 tied, at both read views on all 80 objects** (top-1 +0.1125 at −035-045); the full path's
matcher read is **0.350 FIT / 0.322 all-59** against M27's 0.250 FIT, an estimate that MDE (0.171 at
n=40) does not license a claim for. The shape columns are corrected: `var_x` is **one constant** over
46,059 places at the query view — pinned at `_parabola`'s clip cap, not view-dependent — and zeroing
it changes the gauge not at all (0.539 → 0.539), because a constant column cannot rank; the
rotation-normalised replacement the M30 report proposed reads **0.508**, so shape does not ship and
the ceiling is named. Colour carries the truth in **14 of the matcher's 21 losses** but every
equal-weight rank fusion of it loses more wins than it recovers (+5 / −10 for D), so the matcher's
order stays. **The single next lever** is not a channel but a *fusion with a gate*: colour is right
where the matcher's own margin is thin (the 21 losses sit on a median score gap of 0.014), so the
next mission should try a **margin-gated** colour re-rank — promote colour's candidate only where the
matcher's top-2 gap is below a stated threshold, chosen on FIT with the same +0.075 bar — rather than
another fixed-weight fusion.
