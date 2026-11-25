# MISSION 18 — a score that ranks, and the two failure sources behind it

Branch `arena/c316ca3a-percept6`, pre-registration `b2b8829`, the O5 addendum registered
before the fresh read (`03095d3`).  Every number below is on the **four DEV images**
(`syn|m11ir_00`, `syn|m11ir_01`, `real|cell`, `real|grass`) unless it says otherwise, and
every rate carries its `n` and its per-image spread.  Residuals are **tolunits**, LLR is in
**nats**, AUC is unitless, shifts are in **radians**.

Two things were changed and one was rejected.  The two that shipped behind switches:

1. **candidate generation** — the cap goes from 6 to 24 (`PLACEMENT_CAND_CAP`);
2. **the table** — a pose-balanced, fine-fronted, exp-tailed LLR table
   (`PLACEMENT_LLR_TABLE_M18` → `matching/placement_frozen_params_m18.json`).

The one rejected: **the rotation consensus** (`PLACEMENT_CONTEXT_CONSENSUS`), because on a
real photograph its modal shift is confidently wrong and it deletes the truth (§3.4).

Defaults are untouched: with every switch off the pipeline reads the deployed table with
cap 6, `placement_m12_perf.py --check` reports **0 diffs** (§8), and no new path counts a
fallback.

---

## 1. THE FIVE NUMBERS

15°, 1,166 queries, `asg-l0.5-it0` (the M17 bijection start, λ=0.5, 0 iterations), same
graphs for all three columns.  `a−b gap` = the geometry ceiling 0.902 minus the measured
AUC.  Bars from the pre-registration: top-1 ≥ 0.354, geo AUC ≥ 0.710, ctx > 0.288,
gap < 0.192, no 0° regression.

| | top-1 | ctx acc | geo AUC (gap) | app-only | per-image top-1 | sd |
|---|---|---|---|---|---|---|
| **deployed** (M16 table, cap 6) | 0.210 | 0.177 | 0.659 (0.243) | 0.059 | .244 / .335 / .516 / .084 | 0.156 |
| **+ cap 24** | 0.434 | 0.449 | 0.746 (0.156) | 0.059 | .353 / .329 / .679 / .412 | 0.140 |
| **+ cap 24 + M18 table** | **0.462** | **0.459** | **0.796 (0.106)** | 0.059 | .410 / .518 / .668 / .402 | 0.108 |

- Bars met at the third column: top-1 0.462 ≥ 0.354 ✓, geo 0.796 ≥ 0.710 ✓, ctx 0.459 >
  0.288 ✓, gap 0.106 < 0.192 ✓.
- **Without grass** (n = 510): 0.373 → 0.463 → **0.539**; geo 0.692 → 0.747 → **0.813**.
  Grass is 656 of the 1,166 rot15 queries (56 %), so the pooled figure is mostly a grass
  figure.
- **0°** (n = 2,028): top-1 0.991 → 0.993 → **0.995**; geo 0.971 → 0.973 → 0.973; ctx
  0.963 → 0.967 → 0.969. No regression, and the worst cell improved (m11ir_01 0.968 →
  0.987).
- Appearance-only is **0.059 in all three columns**: neither change touches the full-row
  argmax, which is the whole point of §4.

Sources: `m18_full_cap6.json`, `m18_full_cap24.json`, `m18_eval_tbhyb.json`
(`Topology/placement_m18_summary.py` prints the per-image table).

---

## 2. O1 — ONE TABLE, DEFENSIBLE AND COMPETITIVE

### 2.1 Why the M17 table lost, and why a wider first bin was never the answer

`m18_bank_cap24.json` (pose-balanced bank, 4 images × 4 poses, 0.05-tol histograms): at
15° the TRUE residual **median is 0.135 tol with IQR [0.062, 0.230]** and Q90 0.627; only
19.4 % exceed 0.3 tol and **0 %** exceed 1.8.  The deployed table's first bin is 0.25 wide
to 6.0 and pays **+1.386** exactly where that mass sits.  The M17 refit has 0.05 bins to
1.8, so the typical 15° true pair (0.135) lands in bin 2 and reads **−0.892**.  Pooling
regimes is what made the refit punish 15°: 0° true residuals have median 0.013, so the fit
put nearly all its credit below 0.05.

### 2.2 The candidates, and the criterion that killed most of them

The list was written before any candidate was scored; C7–C9 were added after C2–C6 were
measured and are recorded as **deviations** (§10) with the reason.

| cand | grid / support | rises | thin bins w/ +credit | LLR(5.0) vs LLR(1.5) | non-flat over [0.062, 0.230]: levels / plateau | bank AUC 15° |
|---|---|---|---|---|---|---|
| `dep` (shipped M16) | 0.25 × 24, 6.0 | **5** | **16** | +0.60 vs −2.24 → **wrong way** | 1 / **1.00** | 0.841 |
| `m17` (refit) | 0.05 × 36, 1.8 | 0 | 0 | −2.96 vs −2.96 → **not strict** | 4 / 0.25 | 0.861 |
| `tb24` / `tb24m` / `tbnu` | 0.25 / non-unif | 0 | 0 | equal | 1 / **1.00** ✗ | 0.843 |
| `tb30m` | 0.20 × 30, 6.0 | 0 | 0 | equal | 2 / 0.50 | 0.859 |
| `tbexp` | 0.20 × 30, parametric | 0 | 0 | −3.59 vs −3.44 ✓ | 2 / 0.50 | 0.859 |
| `tbfine` | 0.05→1.0, then 0.25→3.0, 1.0→6.0 | 0 | 0 | −6.72 vs −6.72 ✗ (PAVA pools the tail) | 4 / 0.25 | 0.880 |
| `tbexpf` | fine grid, parametric | 0 | 0 | −3.16 vs −3.12 ✓ | 4 / 0.25 | 0.879 |
| **`tbhyb` (shipped)** | fine front + exp tail | **0** | **0** | **−7.04 vs −4.38 ✓** | **4 / 0.25** | **0.880** |

Three findings worth naming:

1. **The pre-registered non-flat criterion cannot be met by any grid coarser than the
   region.**  The region where 15° truth lives is 0.168 tol wide; every registered grid
   had a 0.20–0.25 bin, so `plateau_share` was forced to 1.00 by arithmetic, not by the
   data.  That is why C7 refines *resolution* inside the region instead of only widening
   the support.
2. **A monotone non-increasing table cannot satisfy "LLR(5.0) strictly < LLR(1.5)" if its
   tail is pooled** — a PAVA-pooled tail reads equal at both.  `tbfine` measured best
   (top-1 0.467, geo 0.795) and still failed the bar.  The repair was not to lower the bar:
   the tail is where counts are thinnest, so it is the part that deserves a *form*.  `tbhyb`
   = measured PAVA front (0.05 tol to 1.0) joined to a strictly decreasing exponential tail,
   and it satisfies both.
3. **The bank cannot choose a table on its own**: every candidate separates at AUC
   0.84–0.88, because the decision is the *sum* over slots and the cap competition, not one
   residual's rank.  The bank is the fit instrument; the matcher is the choice instrument.

### 2.3 The shipped table

`matching/placement_frozen_params_m18.json` — 31 bins: 0.05 tol to 1.0 (20 bins), 0.25 to
3.0 (8), 1.0 to 6.0 (3); equal pose weights (0/5/15° and 1.15×); PAVA with count weights;
a count-scaled minimum (a bin with < 20 pooled counts pays no positive credit); an
exponential tail fitted by weighted least squares.  Over the region where 15° truth lives
the LLR reads **+1.747, +1.192, +0.659, −0.037** — a slope, not a plateau.
`mu_wrong` = **−1.554** (the wrong-pair expectation Σ p_wrong·LLR), which is what an
unformable slot pays.

`L.audit_bundle` → `{"incoherent": [], "unshippable": [], "n_rises": 0}`; per-bin true and
wrong counts are stored for every bin (and per pose in `m18_fit.json`), so the freeze guard
can refuse any mismatch of support, resolution or weight profile.

---

## 3. O2 — CONTEXT: THE 46 % SHARE FALLS

### 3.1 The acceptance, before and after

| | before | after | bar |
|---|---|---|---|
| ctx accuracy at 15° | 0.177 | **0.459** | > 0.288 ✓ |
| geo AUC | 0.659 | **0.796** | > 0.710 ✓ |
| a−b gap (ceiling 0.902) | 0.243 | **0.106** | < 0.192 ✓ |
| 0° top-1 | 0.991 | **0.995** | no regression ✓ |

Per image at 15° ctx: 0.250/0.306/0.408/0.061 → **0.429/0.512/0.717/0.312** (grass 0.312,
n = 656).  CONTEXT-failure share falls with it: the geometry now decides 0.459 of the
queries against 0.177 before.

### 3.2 The mechanism that did it: reach, not ranking

`m18_diagnose.json`: at 15° the true partner's **median rank inside the similarity row is
11** and **69.6 % of true partners are outside the deployed cap of 6** (grass 86.0 %, cell
46.2 %, synthetic 49.4–50.0 %).  A candidate that is never generated cannot be ranked, and
no term inside the blend can reach it.  With the cap at 24 the same context, the same table
and the same iteration reach 0.449 — that is the whole of the cap's effect, and it is the
largest single change in this mission.

### 3.3 The global-transform hypothesis, tested

`node_theta` reads the per-node `source_theta` the front end already computes; every
candidate pair implies a shift `wrap(θ_v − θ_u)`; under a rotation the TRUE pairs share one
shift and the impostors spread.  `consensus_mapping` solves the bijection over the
shift-consistent pairs first (`m18_shift_cap6.json`, `m18_shift_cap24.json`).

It works on what rotates coherently and it **fails on real texture**:

| cell (15°) | consensus mode | true shift | hit | share of TRUE rows that keep their partner |
|---|---|---|---|---|
| m11ir_00 | 0.26 | 0.26 | ✓ | 0.872 |
| m11ir_01 | 0.26 | 0.31 | ✓ | 0.982 |
| **real|cell** | **0.04** | **0.26** | **✗** | **0.098** |
| real|grass | 0.09 | 0.26 | ✗ (cap 6) | 0.623 / 0.745 (cap 24) |

At 15°, `source_theta` on the cell photograph is pinned near the identity while the image
rotated by 0.26 rad, so the consensus votes for a shift that is consistent, confident and
wrong, and the hard restriction then deletes the truth from 90 % of the true rows.

Matcher effect (4 DEV images):

| config | rot15 top-1 | geo AUC | per-image | sd |
|---|---|---|---|---|
| consensus, cap 6 | 0.156 | 0.691 | .276 / .435 / **.147** / .058 | 0.142 |
| consensus, cap 24 | 0.318 | 0.762 | .436 / .506 / **.185** / .279 | 0.127 |
| consensus, cap 36 | 0.370 | **0.785** | .436 / .518 / .402 / .306 | **0.076** |
| no consensus, cap 24, M18 table | **0.462** | **0.796** | .410 / .518 / .668 / .402 | 0.108 |

So the hypothesis carries real signal — at cap 36 it has the *smallest* spread of any
configuration measured, which is what a correct global constraint should do — but the hard
filter cannot be shipped while one photograph loses 0.5 of its top-1.  Rejected, off by
default, and the reason is measured rather than assumed.

### 3.4 The repair: the SOFT consensus, measured AFTER the freeze and not shipped

`PLACEMENT_CONTEXT_CONSENSUS_SOFT` + `_BONUS` (0.15): no row loses a candidate; consistent
pairs get a bounded similarity bonus.  At cap 24 on the same four DEV images at 15°
(`m18_eval_soft.json`, the M18 table):

| | top-1 | ctx | geo AUC (gap) | per-image | sd |
|---|---|---|---|---|---|
| hard-off (shipped design) | 0.462 | 0.459 | 0.796 (0.106) | .410 / .518 / .668 / .402 | 0.108 |
| **soft bonus 0.15** | **0.478** | **0.470** | **0.799 (0.103)** | .423 / .518 / **.641** / .434 | **0.087** |

The repair does what it was built to do: nothing collapses, the spread narrows from 0.108 to
0.087, and grass gains +0.032.  It is nevertheless **not shipped**, for two reasons that are
about evidence rather than about the number: the pooled gain is +0.016 top-1 and +0.003 geo
(both inside one cell's worth of queries), one cell *loses* 0.027 (5 queries of 184), and
the registered fresh read (§6) was run on the design without it.  Taking a marginal,
post-hoc, unregistered change into the frozen design is exactly what the mission's
registration rule forbids, so the honest status is: **measured, promising, unconfirmed**.

---

## 4. O3 — APPEARANCE: THE FLOOR IS NAMED

Acceptance: appearance-only top-1 at 15° > 0.059, or a measured floor with mechanism and
every attempt.

- **Measured: 0.059, unchanged by both changes** (n = 1,166; per image .173 / .124 / .054 /
  .017).  The cap cannot move it by construction — appearance-only is the argmax of the
  full row — so 0.059 is the descriptor's own rank-1 share at 15°.
- **Mechanism: the descriptor's ordering.**  Median true-partner rank 11; rank-1 share 5.9 %;
  69.6 % of true partners fall outside the top 6.  The cap change converts *reach* into
  0.459 context accuracy, but nothing in this mission improves the *order* the descriptor
  produces; the geometry has to outvote it, which is why ctx ≈ top-1 at 15°.
- With the cap at 24 the appearance top-1 is exactly the same 0.059, confirming the cap and
  the descriptor are independent instruments.

**The floor is 0.059 at 15° (n = 1,166), and it is a ranking floor, not a reach floor.**

---

## 5. O4 — EVERY NUMBER PER IMAGE, AND GRASS DIAGNOSED

Nothing in this report is pooled-only: §1 and §4 give the per-image vector next to every
pooled figure, §5 gives grass inside and outside the pool.

| cell (rot15) | queries | median similarity-row size | min row | true-rank median | rank-1 | outside cap 6 | ctx (before → after) | top-1 (before → after) |
|---|---|---|---|---|---|---|---|---|
| m11ir_00 | 156 | 199 | 9 | 6.5 | 0.173 | 0.500 | 0.250 → 0.429 | 0.244 → 0.410 |
| m11ir_01 | 170 | 158 | 16 | 6.0 | 0.124 | 0.494 | 0.306 → 0.512 | 0.335 → 0.518 |
| real\|cell | 184 | 114 | 65 | 6.0 | 0.054 | 0.462 | 0.408 → 0.717 | 0.516 → 0.668 |
| real\|grass | 656 | **586** | 34 | **15.0** | **0.017** | **0.860** | 0.061 → 0.312 | 0.084 → 0.402 |

**The grass diagnosis.** Grass is not a hard texture problem, it is a *candidate
generation* problem with a measurable signature: its median similarity row is **586 places
against 114–199 for the others** (5× the cell photograph, 3× the synthetic hatch), so almost
every place is a plausible partner for almost every place; the true partner's median rank is
**15** and **86 %** are outside cap 6; the descriptor's rank-1 share is 1.7 %.  Its query count
(places that HAVE a resolvable truth partner) is 1,045 at 0° against 314–338 for the other
three images, and its median candidate row is 586 against 114–199: grass places are dense
enough that almost every place is a plausible partner for almost every place.  At 15° it is
56 % of the pooled population, which is why a pooled headline at this pose is mostly a
statement about grass.

NOT measured in M18, and named rather than guessed: texture period, coherence and absolute
place spacing in pixels.  The row size and the query count are the observables this mission
measured; the three texture statistics remain open (§9).

---

## 6. O5 — THE FRESH READ

_Registered before the run in `MISSION18_PREREGISTRATION.md` (ADDENDUM, `03095d3`):
frozen design = cap 24 + M18 table + assignment context (consensus off); sample = the 20
`fresh20` images; pose rot15; primary = per-image paired `top1_full − top1_app`, n = 20,
ties fail; MDE stated before the read = 2.8016 × 0.0436 / √20 = **0.0273**._

**DETECTION.**  20 images, 4,320 queries, `m18_o5_fresh.json`.

| | value |
|---|---|
| n (images) | 20 |
| mean paired difference `top1_full − top1_app` | **+0.2564** |
| sd | 0.0896 |
| 95 % CI (t, 19 df) | **[+0.2145, +0.2983]** |
| median | +0.2503 |
| strict wins / ties / losses | **20 / 0 / 0** |
| min, max | +0.093, +0.399 |
| MDE from the observed sd (α .05, power .80) | **0.0561** |
| registered planning MDE (sd 0.0436) | 0.0273 |
| planning effect (FIT, +0.403) | well above both MDEs |

Acceptance as registered — mean > 0 ✓, CI excludes 0 ✓, every `d_i >= 0` ✓, and no tie to
hide behind (0 ties).  The observed sd is **twice** the planning sd, so the registered MDE
was optimistic by 2×; the honest MDE is 0.0561, which is still 4.6× below the observed
effect, so the read is not vacuous.

Per image (`d` = full − appearance-only):

```
real|gravel  q 687  full 0.290  app 0.017  d +0.272  ctx 0.281  geo 0.692
real|moon    q 264  full 0.451  app 0.076  d +0.375  ctx 0.428  geo 0.752
real|page    q 603  full 0.270  app 0.048  d +0.222  ctx 0.279  geo 0.697
real|retina  q 260  full 0.169  app 0.065  d +0.104  ctx 0.181  geo 0.562
syn|m17ir_00 q 158  full 0.367  app 0.133  d +0.234  ctx 0.361  geo 0.692
syn|m17ir_01 q 163  full 0.485  app 0.098  d +0.387  ctx 0.491  geo 0.806
syn|m17ir_02 q 147  full 0.293  app 0.136  d +0.156  ctx 0.279  geo 0.636
syn|m17ir_03 q 148  full 0.507  app 0.108  d +0.399  ctx 0.459  geo 0.806
syn|m17ir_04 q 148  full 0.324  app 0.115  d +0.209  ctx 0.297  geo 0.719
syn|m17ir_05 q 140  full 0.207  app 0.114  d +0.093  ctx 0.193  geo 0.631
syn|m17ir_06 q 173  full 0.289  app 0.127  d +0.162  ctx 0.266  geo 0.631
syn|m17ir_07 q 163  full 0.350  app 0.141  d +0.209  ctx 0.380  geo 0.741
syn|m17ir_08 q 183  full 0.437  app 0.120  d +0.317  ctx 0.426  geo 0.796
syn|m17ir_09 q 153  full 0.379  app 0.144  d +0.235  ctx 0.405  geo 0.751
syn|m17ir_10 q 148  full 0.480  app 0.095  d +0.385  ctx 0.493  geo 0.759
syn|m17ir_11 q 147  full 0.483  app 0.218  d +0.265  ctx 0.497  geo 0.777
syn|m17ir_12 q 159  full 0.396  app 0.126  d +0.270  ctx 0.371  geo 0.769
syn|m17ir_13 q 158  full 0.316  app 0.082  d +0.234  ctx 0.329  geo 0.718
syn|m17ir_14 q 139  full 0.396  app 0.115  d +0.281  ctx 0.388  geo 0.794
syn|m17ir_15 q 179  full 0.397  app 0.078  d +0.318  ctx 0.397  geo 0.719
```

Split, never pooled across regimes: **16 synthetic** (q 2,506) full **0.382** vs app 0.121,
geo 0.734; **4 photographs** (q 1,814) full **0.289** vs app 0.043, geo 0.676.  The
photographs gain least (retina +0.104) and it is the same cell that the rotation consensus
broke most (§3.3): real texture remains the weak regime, and the fresh read says so
independently of the DEV set the design was measured on.

Non-vacuity, checked before the verdict: 0 cells with zero queries; appearance-only > 0 in
**20 of 20** images; context rows counted on every image (the `ctx` column above).

---

## 7. EVERY FAILED ATTEMPT, WITH ITS NUMBER AND ITS MECHANISM

| # | attempt | number | mechanism |
|---|---|---|---|
| 1 | M17 table in the cap-24 pipeline | top-1 0.434 (deployed table identical) — no gain | its bins are 0.05 wide to 1.8, so 15° truth (median 0.135) reads −0.89; **and** LLR(1.5) = LLR(5.0) exactly, so it fails O1's strict order |
| 2 | pose-balanced mixture on the deployed 0.25 grid (`tb24`, `tb24m`, `tbnu`) | plateau 1.00 over the 15° IQR → non-flat fails | the region (0.168 wide) is narrower than the bin (0.25): no grid coarse enough to represent the data can have slope inside it |
| 3 | 0.20 grid (`tb30m`) | passes the letter (plateau 0.50) but steps from +1.71 to −0.63 at 0.20 | the M17 cliff in miniature: a bin boundary lands inside the true-residual IQR |
| 4 | fine front with a PAVA tail (`tbfine`) | best measurement of the mission (top-1 **0.467**, geo **0.795**) but LLR(5.0) = LLR(1.5) = −6.72 | PAVA pools the thin tail into a level; repaired as `tbhyb` (§2.2) |
| 5 | parametric everywhere (`tbexpf`) | strict order ✓, 4 levels ✓, but 0.455 / 0.775 | a single exponential cannot follow the measured front's curvature; it gives up 0.007 top-1 and 0.021 geo |
| 6 | `mu_wrong` as the mean of `log(llr_raw)` | **−11.8 nats** → top-1 0.175 on *both* tables | every unresolvable slot paid −11.8 and the score gridlocked; caught because two different tables gave the same number.  Fixed to Σ p_wrong·LLR = −1.554; pinned by `TableSanityTests` |
| 7 | hard rotation consensus | 0.156 (cap 6) / 0.318 (cap 24) against 0.434 / 0.462 | §3.3: the mode is confidently wrong on the cell photograph and the filter deletes the truth from 90 % of the true rows |
| 8 | consensus mode-count 2 | identical to 1 mode in every cell | the second mode never passes `CONSENSUS_MODE_SHARE` 0.5; reported so it is not mistaken for an untested lever |
| 9 | soft consensus as a *shippable* win | +0.016 top-1, +0.003 geo, sd 0.108 → 0.087, one cell −0.027 | real but marginal and post-hoc; not carried into the frozen design (§3.4), because a change measured after the registration cannot be the registered read |

---

## 8. QUALITY CHECKS

- **Defaults byte-identical**: `Topology/placement_m12_perf.py --check --workers 2` →
  `identical: True (0 diffs)`, 14 pairs, wall 188.7 s, run **after** all matcher, table and
  parameter changes (`/tmp/m18_check.log`; exit 0).
- **No silent fallback, and no new fallback counter**: at defaults `active_path()` is the
  deployed file, `cand_cap()` is the frozen cap, and the consensus block does not run.
  `test_no_new_path_counts_a_fallback_at_defaults` asserts `PM.FAILURES` is unchanged
  through a load and a score; `NoSilentFallbackTests` (M17) still pins `rescore_matrix`.
- **Switches declared before use**: `PLACEMENT_CONTEXT_CONSENSUS{,_TOL,_MODES,_LAMBDA,_SOFT,
  _BONUS}`, `PLACEMENT_CAND_CAP`, `PLACEMENT_LLR_TABLE_PATH`, `PLACEMENT_LLR_TABLE_M18` are
  in `parameters/_text.py` and in `eth80_frontend._FRONTEND_SWITCHES`, all four tests of
  `DefaultsTests` check both.
- **Tests**: 86 pass (`python -m unittest discover -s tests`), of which 15 are new
  (`tests/test_m18_consensus.py`), and each of them goes RED if the behaviour it pins is
  reverted (the consensus restriction, the counters, the soft path's non-deletion, the
  table's strict tail order, `mu_wrong`'s band and its identity as the wrong-pair
  expectation).
- **Determinism**: candidate caps and consensus break ties on node id; `top_candidates`
  sorts `(-similarity, node id)`; the fit's PAVA and the exp search are closed-form or
  fixed-grid, no randomness.  Repeated cells (§3.3, cap 6 vs 24) reproduce their numbers.
- **Units** stated at the head of this report and in every result file's `units` key.
- **Probe, not deployable**: all of this is measured on **cached graphs** the DEV images
  were built into (M16 cache keys); the strip is DEV.  Nothing here was measured on a live
  capture.

---

## 9. RANKED OPEN PROBLEMS, AND THE SINGLE NEXT CHANGE

1. **The descriptor's ordering (O3) never improved** — 0.059 at 15°, median rank 11.  This
   is now the binding constraint: context can only outvote a descriptor whose top-1 is
   0.059 by being right about almost everything else.
2. **The soft consensus is measured but unconfirmed** (§3.4): +0.016 top-1, geo +0.003,
   spread 0.108 → 0.087, one cell −0.027.  It is the only route so far that raises the pooled
   number *and* narrows the spread, and it deserves a registered read.
3. **Grass's 586-place rows** (§5): the row size, not the texture, is what defeats reach.
   A cap that scales with row size, or a partial-deduplication of near-identical places,
   is the next instrument.
4. ~~Cap cost~~ **measured, and it is free**: `m18_capcost.json`, the same pairs through
   `joint_mcs`, cap 6 vs cap 24, best of 2 repeats — median 9.069 s vs 8.823 s, **0.97×**,
   and the judged-slot count is identical (64 → 64, 61 → 61).  The cap bounds the context's
   candidate list, not the re-scored cells, so quadrupling it costs nothing measurable.
   Caveat: measured on the `perf|` pairs, not inside the placement harness (deviation 4).
5. **Texture statistics for grass** (period, coherence, spacing in pixels) remain unmeasured.

**The single next change**: register and run a fresh read of
`PLACEMENT_CONTEXT_CONSENSUS_SOFT` (`_BONUS` 0.15) on top of the shipped design — same
20-image set, same pose, same paired statistic and the same MDE discipline — against this
mission's O5 baseline (mean +0.2564, CI [+0.2145, +0.2983]).  The DEV measurement (§3.4)
says the effect is real but small; only a registered read can say whether it survives the
image change, and the precedent from §3.3 (a mechanism that looked strong at cap 36 and
collapsed on one photograph) is exactly why it must be read rather than assumed.

---

## 10. DEVIATIONS FROM THE REGISTRATION

1. **C7 `tbfine`, C8 `tbexpf`, C9 `tbhyb` were added after C2–C6 were measured.**  Reason
   recorded in §2.2: every registered grid was coarser than the region the criterion is
   about, so the criterion could not be met by *any* registered candidate.  The addition is
   a refinement of the registered route "wider support + count-scaled minimum" (the support
   is wider in all three) and it was scored on the same bank, against the same references.
2. **The strict-order checker had its inequality inverted in the first version** of
   `placement_m18_fit._checks` (`1.5 < 5.0` instead of `5.0 < 1.5`).  Corrected to the
   pre-registered wording; no candidate's numbers changed, and the correction is why
   `tbfine` was seen to fail rather than pass.
3. **The O1 bar was measured on the 4 DEV images**, the same set M17's headline came from,
   not on fresh images; O5 is the fresh read.
4. **Cap timing** is measured in `Topology/placement_m18_capcost.py` (0.97×, §9 item 4)
   rather than inside `placement_m12_perf.py`, which has no cap knob; the harness-level
   cost is therefore not in the defaults digest.
5. **The bank was recorded at cap 24**, i.e. under the design that would ship, not at cap 6
   — otherwise the table would be fitted to a candidate population the pipeline no longer
   produces (the same class of error as pooling regimes).
