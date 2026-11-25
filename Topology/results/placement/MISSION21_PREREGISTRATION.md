# MISSION 21 — pre-registration (written before any FRESH graph exists)

Committed alone, before a single FRESH object has been rendered, matched or scored. No
FRESH graph exists in the cache at this commit; every M20 number is FIT or DEV.

## 1. What is frozen

| item | value | where it was fixed |
|---|---|---|
| scorer | the M19 bundle: descriptor w 0.25, K 24, whole row; M18 LLR table; assignment start; candidate cap 24 | `PLACEMENT_M19=1`, proven equal to the M19 record (item 1: 0.569 / 0.826, 0 disagreements in 1,166 queries) |
| front end | the M16 recipe (`FE_KEYPOINTS`, `FE_PLACE_LAYER`, sub-pixel peaks and Gabor anchor, keypoint frame `largest`, M15 places, graded similarity) | `Topology/placement_m20_eth80.py::RECIPE` |
| views | reference `-000-000`, query `-022-000` (T1, T2, T3); T0 = the reference image rotated 15° | `M20_TASKS_PREREGISTRATION.md` |
| gallery | **2 objects per class, 16 items**, all eight classes, same split as the queries | chosen on FIT (item 6) |
| shortlist | **S = 8**, stage-1 = the configuration bag score (no mapping) | FIT, with its recall stated below |
| aggregate | S2 = shipped + λ·S3, **λ = 1.0** | `matching/placement_m20_weights.json`, fitted on FIT |
| S1 weight | w = 0.375, `matching/placement_score_params.json` (Mission 11, untouched) | unchanged |

## 2. FIT numbers these choices were made from (n stated, nothing pooled)

Gallery 16, S = 8, queries 16 (2 per class), FIT:

| arm | T1 rank-1 / MRR | T2 rank-1 / MRR | T3 rank-1 / MRR | null AUC (true vs impostor pairs) |
|---|---|---|---|---|
| S0 shipped | 0.786 / 0.821 (n=14) | 0.062 / 0.276 (n=16) | 0.625 / 0.688 (n=16) | 0.955 (12 vs 116) |
| S1 shipped + E | 0.786 / 0.821 | 0.062 / 0.271 | 0.625 / 0.688 | 0.958 |
| **S2 shipped + λ·S3** | 0.786 / 0.821 | 0.062 / 0.286 | **0.688 / 0.719** | 0.952 |
| S3 aggregate alone | 0.643 / 0.750 | 0.062 / 0.226 | 0.375 / 0.487 | 0.719 |

Stage-1 recall@S on FIT, **reported with every shortlist number**: T1 1.00 (S=8, 16-item
gallery; 0.62 at S=1, 1.00 at S=2), T2 0.62, T3 0.75. On the full 40-item gallery: T3
recall@8 0.25, @16 0.55, @24 0.75 — which is why the gallery was cut to 16 items before
choosing S.

**T2 is at chance on FIT** (1/16 = 0.0625, and 1/16 is exactly what it scores). This is
registered as a finding, not hidden: the shipped score carries no same-class, other-instance
signal in this configuration. M21 must report T2 against its chance level, and the primary
contrast on T2 is still registered below because a CI around chance is the informative
number.

## 3. The READ (registered)

* **Split:** FRESH, 16 objects (2 per class), none of which any M20 run has seen.
* **Queries:** 3 registered view pairs per object — `-022-000`, `-035-045`, `-045-180`
  against the `-000-000` reference — so **48 query images in 16 object clusters**. The
  single-view-pair version (16 images) is reported as a secondary, because that is the
  configuration the FIT numbers above were measured on.
* **Tasks:** T1, T2, T3 as registered in `M20_TASKS_PREREGISTRATION.md`, plus T0 (the rotated
  self, the aggregation control).
* **Not pooled:** FRESH rows are separate from FIT and DEV; the three view pairs are
  separate rows; synthetic renderer scenes (item 5) are never pooled with ETH-80.

## 4. Primaries, statistics, and the power statement

**Primary quantities (4):** T2 top-1, T2 MRR, T1 top-1, T1 MRR — each as the paired
contrast **S2 − S0** over query images, clustered by object.

* Paired per-image differences, image-clustered bootstrap CI (10,000 resamples over the 16
  object clusters, seed 20261118), reported with the number of images up, down and tied.
* **Holm** across the four primaries, family-wise 0.05, in the registered order (T2 top-1,
  T2 MRR, T1 top-1, T1 MRR).
* **MDE, sized in images, stated before the read:** the FIT paired sd of (S2 − S0) on T3 is
  0.250, which gives MDE = 2.8·sd/√n ≈ **0.175 at n = 16** and ≈ **0.10 at n = 48** (equal
  clusters assumed; clustering can only raise it).
* **The planning effect is the FIT estimate, +0.0625 (= one image).** It is BELOW the MDE at
  both n. Therefore: **this read may not verdict the aggregate.** What it can do is put a CI
  on the contrast and report the per-class and per-image spread. A CI that excludes zero is
  informative; a CI that includes zero is reported as "not resolved at this n", and no
  conclusion about the aggregate is drawn from it either way.
* **Spread:** per-class rank-1 for every arm (8 classes × 3 tasks), the per-image rank of
  the positive, and the min/max per class — a headline with no spread is not a result.

## 5. Non-vacuity counters (a read with zero of these is not verdictable)

* every task reports its n of queries with ≥2 scored candidates, and the read is
  non-vacuous only if that is ≥ 90 % of the registered queries;
* `PM.FAILURES["fell_back"]` must be 0 with `PLACEMENT_MATCHER_STRICT=True` (a fallback is a
  wiring bug, not a result);
* every pair reports `mapped > 0` and `judged > 0`; the number of pairs with either at zero
  is printed beside the result;
* shortlist recall@8 is printed for every task, next to every top-1 that used the shortlist;
* T0's top-1 is the aggregation control: if T0 is below the T3 result, the aggregate is
  hurting images whose content is identical, and that outranks any T1–T3 difference.

## 6. Decision rules, written now

1. If **T0 ≤ T3** (the aggregation control does no better than the recognition task), the
   aggregate is rejected regardless of T1–T3, because identical content must not be ranked
   worse than different content.
2. If the paired CI on a primary excludes 0 **and** Holm rejects, the registered direction is
   claimed for that primary only.
3. If every primary CI includes 0: the read is reported as **flat at this n**, the MDE is
   restated, and no aggregate claim is made. The single next change is then the one thing
   FIT already points at — the **class-level aggregate for T2** (the score is at chance
   across instances), not a re-tune of λ, which FIT shows is flat within one query.

## 7. What may not change after this commit

Gallery size, S, λ, the arms, the view keys, the tasks, the tags and the statistics. Any
deviation is a new registration, reported as a deviation, with the numbers from this one
still reported.
