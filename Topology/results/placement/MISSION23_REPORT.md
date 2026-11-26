# Mission 23 -- close it

Four fixes, one read, one README.  No new research direction.
Branch `arena/c316ca3a-percept6`, from `717d98d` (Mission 22).  Every rate below is quoted with
its n and its chance, and every result JSON carries a manifest.

---

## 1. The class score, measured properly (T2)

Mission 22's `(C)` = 0.625 was **instance identification**: the gallery contained the query's own
object, so the best-scoring instance was the query's own reference photograph and the number
answered "can you find this object again".  The honest task removes the query's own object from
the gallery **before** the door runs -- 39 candidates for a 40-object split -- so the chance of a
hit on the reduced gallery is not `1/8` but `k/(N-1)` with `k` the number of instances of the
query's class still present.

Shipped door, shortlist S = 8 taken from the 39, matcher re-order inside it, a class scored by
pooling its instances' matcher scores:

| aggregate (fixed before the run) | class top-1 | n | chance |
|---|---|---|---|
| **top-2 pooling (primary)** | **0.500** (20/40) | 40 | 0.103 |
| `max` | 0.525 (21/40) | 40 | 0.103 |
| `mean` | 0.475 (19/40) | 40 | 0.103 |

Per class (top-2 pooling): apple 1.0, cup 0.8, car 0.4, cow 0.2, dog 0.4, horse 0.4, pear 0.4,
tomato 0.4 -- all n = 5 except where the split holds fewer.  Class coverage -- the share of the
query's-class instances that reach the shortlist at all -- is **0.463**: on average the 8
candidates contain under two of the four same-class instances, so the class score is limited by
the shortlist before it is limited by the pooling rule.

**Verdict: T2 is not at or below chance.**  It is 4.9x chance (0.500 against 0.103), so the
branch that would have named the loss with the Mission-21 tagger does not apply, and nothing is
"fixed": this is what the system's class claim is worth, and the README states it as 0.500
against 0.103.

The test that keeps this honest is `tests/test_m23_bundle.py::TestT2Gallery`, which asserts that
the query's own object is absent from the gallery the door is given; it fails if that invariant
is broken, so the instance-identification measurement cannot come back by accident.

---

## 2. Geometric voting, settled with true correspondences

Mission 22 measured the transform-histogram door on the matcher's **own, mostly wrong,
correspondences** (`bucket_hit` 0.33-0.38).  A wrong pair cannot agree on a transform whatever the
transform model is, so that measurement was about the matcher.  Re-measured on the synthetic
renderer, where the true partner of every place is known by geometry (`placement_m23_geovote.py`,
20 scenes, ~2,500-4,200 true pairs per cell):

| view | true pairs n | similarity agreement | shuffled null | ratio | affine agreement | shuffled null | ratio |
|---|---|---|---|---|---|---|---|
| identity | 4,167 | 1.000 | 0.015 | **66.2x** | 1.000 | 0.015 | 67.8x |
| in-plane 15 deg | 3,302 | 0.037 | 0.007 | 5.0x | 0.019 | 0.019 | 1.0x |
| scale 1.15x | 2,955 | 0.041 | 0.010 | 4.2x | 0.057 | 0.021 | 2.7x |
| pitch 20 deg | 3,340 | 0.142 | 0.008 | **17.2x** | 0.048 | 0.019 | 2.5x |
| **yaw 20 deg** | 2,537 | 0.072 | 0.011 | **6.6x** | 0.056 | 0.024 | **2.3x** |

The decision rule was fixed and committed before the run: build the vote as an extra arm only if
similarity or affine agrees at **>= 10x** its shuffled null on **yaw 20**.  Ratios there are
**6.6x** (similarity) and **2.3x** (affine).  **Neither passes.**

**So: geometric voting does not work on out-of-plane views**, and the transform-histogram door is
not one of the shipped options.  The counted door is what ships, named for what it does -- a
**content-count door**.

The residual fit says why, and it is not noise.  Fitting one similarity to a true pair leaves an
RMS residual of 3.2 place-spacings (against 36.0 for the shuffled null) -- better than the null
but far wider than the one-spacing bin, which is the vote's resolution.  A global transform is
loosely present; a transform a *place* can vote on is not, and two places rarely land in the same
bin.  Agreeing on a similarity at 7 % of public pairs is exactly the number the door's Hough
option would have had to climb (Mission 22 measured it at FIT 0.575 recall@8, chance 0.200).

**Pitch against yaw, side by side.**  The same accuracy harness (shipped `sorted` descriptor,
σ_d chosen on the fit half exactly as M21 did it), the missing pitch cell run read-only through
it:

| view | node accuracy | n |
|---|---|---|
| yaw 20 deg | 0.226 | 3,753 |
| **pitch 20 deg** | **0.373** | 4,981 |

Yaw is the wall, and it is now measured against pitch on one harness: a 20-degree pitch costs the
descriptor nothing that matters (0.373), a 20-degree yaw costs it 40 % of its accuracy (0.226).
That also squares the vote's mirror image -- it nearly works at pitch (17.2x) and fails at yaw
(6.6x) -- and it closes Mission 21's open "no yaw-specific loss" question with the cell that was
missing when it was asked.

---

## 3. Rerank, fused

Final order = RRF (k = 60, fixed) of the door's rank and the matcher's rank, over the same 8
candidates; paired, clustered by object (one object, one query here):

| split | n | chance | door alone | matcher plain re-order | RRF fused |
|---|---|---|---|---|---|
| FIT | 40 | 0.025 | 0.425 (17/40) | **0.550 (22/40)** | 0.500 (20/40) |
| DEV | 24 | 0.042 | 0.458 (11/24) | **0.750 (18/24)** | 0.625 (15/24) |

Paired, exact (McNemar on the discordant queries): FIT fusion against plain **2 vs 4, p = 0.688**;
DEV fusion against plain **1 vs 4, p = 0.375**; DEV plain against the door alone **8 vs 1,
p = 0.039**; FIT plain against the door alone 9 vs 4, p = 0.267.  So the re-order's advantage over
the door is small at n = 40 and only becomes separable on DEV, which is not where the choice is
made.

**Chosen on FIT only: the matcher's plain re-order wins** (0.550 against the fusion's 0.500), so
Mission 22's re-order stays shipped and the fusion is measured, reported and not used.  DEV agrees
(0.750 against 0.625) but DEV did not decide it.

---

## 4. The door on DEV, stated plainly

Added to the README's numbers table and repeated here because Mission 22's door choice rests on
FIT and DEV disagrees with it:

* counted door, DEV recall@8: **0.750 (18/24)**, chance 0.333;
* appearance alone, DEV recall@8: **0.875 (21/24)**;
* fusion (shipped door), DEV recall@8: **0.875 (21/24)** -- equal to appearance alone.

So the counted door's FIT advantage reverses on DEV.  No new tuning was done in response, and
nothing was re-chosen: FIT is the rule the mission fixed, DEV is the report, and the README says
both.

---

## 5. The one confirmatory read

*(written from `m23_read_rows.jsonl`; see `MISSION23_PREREGISTRATION.md` §1-§6 for the design and
the amendment recorded before the run)*

**What ran.**  All 80 ETH-80 objects, one query each, view `-035-045`, an 80-item gallery of
`-000-000` references.  That is 80 queries, not the 240 the design asked for: the measured cost was
2.4 minutes per query, and §7 of the registration records the reduction before any row was
summarised -- the second view was dropped per the fixed drop order, and **the MDE at 80 queries x
1 query per object is 0.166-0.182** (`m23_power.json::per_object_curve`), against the planning
effect of 0.375.  Chance: top-1 **0.0125** (1/80), recall@8 **0.10** (8/80).

**Primary, and it is significant.**  Assembled against deployed, paired per query and clustered by
object, 10,000 bootstrap resamples, seed 20261118:

| arm | top-1 | n | chance |
|---|---|---|---|
| **assembled (shipped M23 configuration)** | **0.250** (20/80) | 80 | 0.0125 |
| deployed (M19 shipped) | 0.075 (6/80) | 80 | 0.0125 |

**difference +0.175, 95 % CI [0.0875, 0.2625], bootstrap p = 0.0001.**

Read it exactly for what it is.  The difference is real and the interval excludes zero, so the
assembled system is better than the deployed one on a view nothing had read.  The difference is
also **less than half the 0.375 the read was planned around** (FIT's assembled 0.550 against
deployed 0.175), and 0.175 sits at the top of what 80 single-query objects can resolve (MDE
0.166-0.182) -- so the direction is established and the *size* is not: an effect of this size is
at the edge of this read's resolution, and a second view would have narrowed it.  That is a
statement about the interval, not about the sign.

**Secondary, reported and NOT verdicted.**  Assembled 0.250 (20/80) against appearance-only
0.0875 (7/80) gives an exploratory exact p of **0.011**; the door's recall@8 is **0.550 (44/80)**
against a chance of 0.10, and the deployed pipeline's stage-1 recall@8 is **0.100 (8/80)** against
the same chance.  **36 of the 80 queries** had the truth in neither shortlist -- exact zeroes for
both arms, per §5.3 of the registration, so the assembled system's 0.250 is achieved on the 44
queries where it had something to re-order (0 / 36 where it did not).

Per-class top-1, assembled (n = 10 each): apple 0.5, tomato 0.5, dog 0.3, cow 0.2, cup 0.2,
car 0.1, horse 0.1, pear 0.1.  The spread is the same story as T2: two classes carry most of the
signal.

**What this does not say.**  It is **view-held-out, not object-held-out** -- every one of these 80
objects appeared in FIT or DEV, where the door, the fusion and the class aggregation were chosen.
It is one view, so it is one draw of "an unseen view", not an average over view space.  And
nothing was re-fitted for it: the configuration is exactly what item 3 ships (plain re-order,
counted door, S = 8, RRF k = 60 unused at this point of the pipeline because the re-order ships).

The closing claim, in one paragraph.  **Proven:** with the query's own object removed, the class
claim is 0.500 against 0.103 (n = 40); the transform vote does not work on out-of-plane views
(6.6x its null at yaw 20 against the 10x the rule required, on true correspondences), while the
counted content door plus the matcher reaches 0.550 top-1 on FIT and 0.750 on DEV (n = 40 / 24,
chance 0.025 / 0.042); the shipped configuration beats the deployed one on a never-read view,
+0.175 (20 against 6 of 80, CI [0.088, 0.263]).  **Not proven:** the size of that advantage (the
interval is wide and the read was one view), any object-held-out claim, class recognition beyond
half, and any pose recovery -- rotation between views is not a similarity (2.8 rad interquartile
range on true pairs), so the system cannot say how an object is turned.  **Next:** a second
never-read view to narrow the interval, and a descriptor that survives yaw -- 0.226 node accuracy
at yaw 20 against 0.373 at pitch 20 is a descriptor problem, not a place problem.

---

## 6. What this mission changed in the record

1. `(C)` = 0.625 in Mission 22 was **instance identification** and is withdrawn as a class-level
   claim; the class score is **0.500 (20/40) against a chance of 0.103** (n = 40, gallery 39).
2. The Mission-22 transform failure was measured on **the matcher's own mostly-wrong
   correspondences**; measured on true correspondences, geometric voting agrees at **6.6x** its
   null at yaw 20, under the 10x the rule required, so **geometric voting does not work on
   out-of-plane views** and the transform door is deleted from the shipped options.
3. The final order is the **matcher's plain re-order**, not the rank fusion (FIT 0.550 against
   0.500).
4. The door is named for what it does: **content-count door**.  On DEV it is **0.750 (18/24)**,
   below appearance's 0.875 (21/24).
5. The read in §5 is the last confirmatory measurement of the sequence, and it is labelled
   **view-held-out**, not object-held-out: all 80 objects, the two views no earlier mission had
   read.

Proof of work: `tests/test_m23_bundle.py` (14 tests) plus the existing suite green (**158 tests,
OK**), `placement_m12_perf.py --check --workers 1` **0 diffs** (980.7 s, 28 pairs), and a manifest
on every result JSON.

One thing learned while proving it, recorded because it would have been easy to call a
regression: the parity check is **not neutral in `PLACEMENT_M19`**.  Run as
`PLACEMENT_M19=1 ... --check` it reports **65 diffs**, all in arm C (the placement matcher on),
with arm A identical; one pair run both ways shows why -- arm C's score is 0.2898 under
`PLACEMENT_M19=1` and 0.3337 without it, and 0.3337 is the baseline byte for byte.  The deployed
flag changes what the placement matcher does; the parity check exists to prove that *the shipped
defaults* did not move, so it is run with the flags off (as Mission 12 and Mission 22 ran it).
The 0 above is that run.
