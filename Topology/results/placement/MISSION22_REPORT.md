# Mission 22 -- finish the mechanism: a front door, an assembler, a class score

Branch `arena/c316ca3a-percept6`, from Mission 21's tip `7a86284`.  Plain language.  Every rate
carries its **n** and its **chance**.  The registration this mission is judged against is
`MISSION22_PREREGISTRATION.md`, written before any FRESH image was touched.

## 0. Verdict, in one paragraph

The mechanism is finished and shipped: a cheap front door (colour+gradient appearance fused with a
graph-native place-voting door) puts the truth inside 8 candidates **0.900 of the time on FIT
(36/40, chance 0.200)** and 0.875 on DEV (21/24, chance 0.333), and the shipped matcher then
re-orders those 8 into a **top-1 of 0.550 (22/40, chance 0.025)** against the appearance door's
0.275 (11/40) and the shipped matcher alone at 0.175 (7/40) -- **3.1x the deployed system's
top-1**.  The graph-native door's *geometric* vote was built,
measured and **failed** for a measured reason (the correspondences the matcher chooses do not
agree on a similarity transform: rotation interquartile range 2.8 rad, the whole circle), so the
door as shipped **counts** agreeing places instead of clustering their transforms; that is the
mechanism, and it is 22x cheaper per candidate than the matcher's first stage.  The class score
(C) reaches **0.625 (25/40)** against a hypergeometric chance of 0.125 (n = 40) and every fixed arm is
further above it, so the class signal is not lost -- it is the strongest thing in this report.  **The one
evaluation on FRESH was not run**: at 16 held-out objects its minimum detectable effect is 0.706
while the planning effect is 0.50, and this mission's own rule forbids a read whose MDE exceeds
its planning effect.  There is therefore **no confirmatory number** in this report; every number
is FIT or DEV, the splits the mechanism was chosen on.

## 1. What was built

| piece | file | what it is |
|---|---|---|
| (A) audit | `Topology/placement_m22_audit.py` | read-only: do places carry a frame and a content key? |
| (A) front door | `matching/placement_frontdoor.py` | appearance channel, pooled place index, transform vote, counted vote, RRF, shortlist |
| (B) assembler | `matching/placement_assemble.py` | shortlist -> shipped matcher -> re-order, never dropping a candidate |
| (C) class score | `matching/placement_classscore.py` | per-class pooling of instance scores, `top2` primary, `max`/`mean` arms |
| harness | `Topology/placement_m22_eth80.py` | place records, the front doors, the assembled arms, the class read |
| diagnosis | `Topology/placement_m22_placeproof.py` | where the vote's recall is lost, measured on the matcher's own pairs |
| key sweep | `Topology/placement_m22_keysweep.py` | can any content key make place voting work? (no) |
| power | `Topology/placement_m22_power.py` | the arithmetic behind the gate in §6 |

All of it is off by default: `PLACEMENT_FRONTDOOR=off`, `PLACEMENT_ASSEMBLED=0`,
`PLACEMENT_CLASS_SCORE=0`, declared in `parameters/_text.py` before use.

## 2. (A.1) The audit: what a place carries

Read-only, FIT, three images, `m22_audit.json`:

| image | places | with position (payload + subgraph nodes) | with scale | with orientation | lenses | edges |
|---|---|---|---|---|---|---|
| apple1-022-000 | 473 | 1,625 | **473 (100 %)** | 436 (92 %) | 3 | 4,529 |
| apple3-022-000 | 473 | 1,588 | **473 (100 %)** | 422 (89 %) | 3 | 2,962 |
| apple6-022-000 | 503 | 1,835 | **503 (100 %)** | 411 (82 %) | 3 | 2,926 |

**Nothing is missing.**  Position is `subgraph.anchor_of_node`, on every payload place.  Scale is
already carried: `Side.spacing_by_node` is the place's distance to its nearest neighbour, i.e. the
local size of its neighbourhood.  Orientation is already carried: `PE.node_frames_keypoint` gives
an axial angle in radians.  The only gap is the 8-18 % of places with no frame; they are handled as
an **optional** extra histogram (a place without a frame votes on the 3-D transform only) and never
imputed.

## 3. (A) The graph-native front door

### 3.1 The index

One pooled index per gallery, built once, `PlaceIndex`: for every gallery place its content key,
its frozen-ruler configuration row (Mission 21's per-slot standardisation and overlap mask) and its
frame.  On the FIT gallery: **20,323 places over 40 objects in 533 content buckets**.  Bucket size
was measured before it was chosen (FIT, 6 objects, 3,163 places, `m22_audit.json`):

| key | buckets | median | p95 | singletons |
|---|---|---|---|---|
| lens only | 3 | 1,246 | -- | 0 |
| **lens + quantised bins (shipped)** | **370** | **3** | **37** | **112** |
| lens + bins + configuration | 1,676 | 1.0 | 6 | 1,141 |

The table is the six-object pool the key was chosen on; on the full 40-object gallery the shipped
key gives **533 buckets, median 10, p95 186** (`m22_keysweep.json`), and every statistic in this
mission is from that pool unless it says otherwise.  A key that is too fine cannot vote: a place
alone in its bucket has nothing to pair with.  The shipped key is the middle row.

### 3.2 The vote, and why it does not work here

Each query place probes its bucket, ranks the bucket by the frozen configuration distance, keeps
its `n_hits` nearest pool places, and each of them proposes the similarity that would map the
query place's frame onto its own and votes for that pool place's object in a coarse histogram
(translation 1.0 query place-spacings, log2 scale 0.5, axial rotation pi/6).  An object's score is
its largest agreeing bin, normalised by `sqrt(n_query_places x n_object_places)`.

It scored **0.575 recall@8 on FIT (23/40, chance 0.200)** -- above chance, far below the door we
need.  The reason is measured, not guessed (`m22_placeproof.json`, three FIT objects, the shipped
matcher's OWN chosen correspondences):

| object | pairs used | partner in the same bucket | configuration rank of the true partner | transform IQR dx / dy (place-spacings) | log2 scale IQR | rotation IQR |
|---|---|---|---|---|---|---|
| apple/1 | 415 | 0.381 | 2 | 12.9 / 31.1 | 1.09 | 2.91 rad |
| apple/3 | 417 | 0.329 | 2 | 57.0 / 52.2 | 1.14 | 2.81 rad |
| apple/6 | 428 | 0.327 | 2 | 51.4 / 52.3 | 1.33 | 2.77 rad |

Agreement (share of pairs in the modal transform bin) is **0.006 against a shuffled null of
0.003**.  The correspondences are real -- a third of them land in the same content bucket and the
frozen ruler ranks the true partner second on average -- but **they are not a similarity**: the two
views differ by an out-of-plane rotation, so the translation ranges over tens of place-spacings,
the apparent size over 2x, and the axial rotation over the whole [0, pi) circle.  No bin width that
keeps the histogram meaningful can make those votes agree.  **This is where the recall was lost:
in the assumption that a place correspondence is a similarity transform, not in the index, not in
the key and not in the bins.**  A sweep of seven content keys on the same FIT gallery and the same
vote (`m22_keysweep.py`, `m22_keysweep.json`) shows the same pattern for every one of them: top-1
**1.000 against its own image** (a key is perfect when nothing moves) and **0.200-0.600 recall@8
across views** with a vote peak of **0.001-0.003** of all votes -- one vote in 300 or worse ever
agrees with another, whatever the key.  An eighth key (lens only, one bucket per lens, ~20k places
each) is constructed in the script and not run: it cannot discriminate content by construction and
costs minutes per query.

### 3.3 The counted door -- the vote's own limit, and what ships

Taking the bin width to its limit turns "the largest agreeing cluster" into "every vote counted
once": an object scores the number of query places that find a place in it, over the object's own
place count.  Same index, same probe, same normalisation, one dict lookup and one sort per query
place, **no pairwise loop**.

| front door | FIT recall@8 (n = 40, chance 0.200) | FIT top-1 (chance 0.025) | DEV recall@8 (n = 24, chance 0.333) | DEV top-1 (chance 0.042) |
|---|---|---|---|---|
| appearance (M21 channel) | 0.675 (27/40) | 0.275 (11/40) | 0.875 (21/24) | 0.375 (9/24) |
| place voting, transform histogram | 0.575 (23/40) | 0.175 (7/40) | 0.708 (17/24) | 0.250 (6/24) |
| place voting, counted | 0.875 (35/40) | 0.350 (14/40) | 0.750 (18/24) | 0.375 (9/24) |
| RRF(appearance, transform vote) | 0.775 (31/40) | 0.225 (9/40) | 0.917 (22/24) | 0.292 (7/24) |
| **RRF(appearance, counted) -- SHIPPED** | **0.900 (36/40)** | **0.425 (17/40)** | 0.875 (21/24) | **0.458 (11/24)** |

The combined door wins on FIT (10 objects gained against 1 lost versus appearance alone) and its
top-1 is the best of every door on both splits.  **DEV's registered bar was 0.938; the best door
reaches 0.917 (22/24), one object short** -- and that is the appearance-plus-transform-vote door,
not the shipped one.  The mission's ship rule is the FIT recall, so the shipped door is the
combined one; DEV is reported, not used.

Cost: appearance 0.019 s/query, counted vote **0.606 s for the whole 40-object gallery**
(0.015 s per candidate), against the shipped matcher's first stage at 0.33 s *per pair*.  That is
**22x cheaper per candidate**, and the door never constructs a pair.

## 4. (B) The assembled system

`matching/placement_assemble.py::assemble` re-orders a shortlist by the shipped pair score and
**never removes a candidate** -- a candidate the scorer could not score keeps its front-door order
at the end and is counted.  The harness calls the same `joint_mcs` entry point with the same
deployed flags as Mission 20's `_pair_job`, so "assembled" is the shipped scorer's ordering, not a
lookalike -- and that is checked, not asserted: six cached pairs were re-scored with the full
Mission-20 job and the two scores agree to the last digit (**max |lite - full| = 0.0**,
`m22_assemble_fit_verify.json`; `--stage verify` re-runs it).

| arm | FIT top-1 (n = 40, chance 0.025) | notes |
|---|---|---|
| M19 shipped, matcher over all 40 (Mission 21 AFTER row) | 0.175 (7/40) | the deployed system |
| appearance-only (front door top-1, no matcher) | 0.275 (11/40) | the floor |
| **assembled (B): appearance 8 -> matcher re-order** | **0.450 (18/40)** | ceiling = the door's recall@8, 0.675 |
| **assembled (A): shipped door 8 -> matcher re-order** | **0.550 (22/40)** | ceiling = the door's recall@8, 0.900 |

The rerank does real work and it is not free of cost to the door: over the appearance shortlist it
turns 10 objects from a miss into a hit and 3 from a hit into a miss (exact McNemar **p = 0.092**,
n = 40); over the shipped door's shortlist, 9 and 4 (**p = 0.267**).  Neither paired difference is
significant at n = 40 on its own, and that is the honest reading: the assembled system's advantage
is against the *deployed matcher* (0.550 against 0.175) and against *appearance alone* (0.450
against 0.275), not something this sample can separate from the door's own top-1 of 0.425.

Spread, per class (assembled over the shipped door, top-1 of 5 each): apple 0.6, car 0.8, cow 0.2,
cup 0.6, dog 0.8, horse 0.4, pear 0.4, tomato 0.6.  Per-image assembled rank: median 1, with 4 of
40 queries where the door never offered the truth at all (2 cow, 2 pear); those 4 can never be a
hit, and the other 14 misses are rerank orderings, not exclusions -- which is the design: the
assembled system can only be as good as its front door's recall@8, and it is.

## 5. (C) The class score

Per class, its shortlisted instances' matcher scores are pooled: **sum of the top two** (primary),
`max` and `mean` (fixed arms, not tuned).  The class score cannot read the query's label:
`class_scores` receives instance scores and the gallery's class table only, and
`tests/test_m22_bundle.py::test_it_cannot_read_the_query_label` rebuilds every number with the
query's class deleted and requires them identical.

| arm | class top-1, appearance shortlist | class top-1, shipped door | n | hypergeometric chance |
|---|---|---|---|---|
| **top2 (primary)** | **0.600 (24/40)** | **0.625 (25/40)** | 40 | 0.125 (5 of 40) |
| max | 0.800 (32/40) | 0.775 (31/40) | 40 | 0.125 |
| mean | 0.675 (27/40) | 0.775 (31/40) | 40 | 0.125 |

Every cell is far above chance (exact binomial p <= 1.7e-12 for the top2 arm), so the mission's
"if T2 stays at or below chance, name where the class signal is lost" does not fire: on the
shortlisted instances the class signal is the strongest in this report.  Two observations, both
fixed in advance and neither tuned:

1. **The primary is not the best arm.**  Summing the top two instances (0.625) is *worse* than
   trusting the single best instance (`max`, 0.775) on the shipped door.  The primary stays the
   primary -- it was registered -- and the fact that it loses is reported rather than reversed.
2. **(C) is not comparable with Mission 21's T2.**  M21's T2 was *instance* top-1 on a 16-item
   gallery (0.475 against a hypergeometric chance of 0.617, i.e. below chance); (C) is *class*
   top-1 out of 8 classes over a 40-item gallery, and it is measured on the instances a door
   shortlisted (0.900 of queries have the truth in the 8 on FIT, 36/40).  It answers "same kind
   of thing", which is a much easier question than "this instance".

## 6. The one evaluation: not run, and why

The registration fixed n in objects from the front-door planning effect (appearance 0.675 against
geometry 0.175, a difference of 0.50) and required the MDE to be stated before the read.  The
arithmetic (`m22_power.py`, exact McNemar on discordant objects) gives, at alpha 0.05 and 80 %
power:

| n objects | MDE (pi_d = 1, most favourable) | MDE (pi_d = 0.6) |
|---|---|---|
| **16 (FRESH)** | **0.706** | **0.833** |
| 24 (DEV) | 0.594 | 0.777 |
| 40 (FIT) | 0.443 | 0.594 |

**0.706 > 0.50: the FRESH read is underpowered by construction, and the rule says it is not run.**
It would need at least 36 objects (pi_d = 1) or 55 (pi_d = 0.6) to resolve the planning effect, and
the effect the assembled comparison actually has to resolve is smaller than 0.50, so more.  ETH-80
has 80 instances and all 80 are already allocated 40 / 24 / 16; there is no spare object and no
re-split that is not post-hoc.  **No FRESH graph, key, pool or score was built** -- the held-out
split is untouched and a future mission with a larger held-out set inherits it intact.

Consequence, stated in the registration and repeated here: this mission has no confirmatory
read.  FIT and DEV are the splits the door was chosen on, so the numbers in §3-§5 are exploratory
and are labelled so.

## 7. Quality bars

| bar | state |
|---|---|
| flags OFF by default | `PLACEMENT_FRONTDOOR=off`, `PLACEMENT_ASSEMBLED=0`, `PLACEMENT_CLASS_SCORE=0`, `PLACEMENT_VOTE_HITS=3`, `PLACEMENT_VOTE_ROT=1`, all declared in `parameters/_text.py` |
| defaults byte-identical | `placement_m12_perf.py --check --workers 1`: **identical: True (0 diffs)**, 816.1 s, A 6.79 s/pair, C 7.64 s/pair (Mission 21: 827.2 s, 6.83 / 7.77) |
| tests | `Ran 144 tests ... OK`, including the length-unit defect, the count normalisation, gallery-order independence, RRF scale-freeness, "never drops", "cannot read the query label", and the chance column |
| determinism | every sort in the door, the fusion and the assembler breaks ties by index; the vote is a pure function of the two records |
| the assembled score IS the shipped score | 6 pairs re-scored with the full Mission-20 pair job: max abs diff **0.0** (`m22_assemble_fit_verify.json`) |
| no silent fallback | `assemble` counts unscoreable candidates and never promotes them; `class_scores` raises on an unknown aggregate; the harness refuses to run without `PLACEMENT_M19=1` |
| manifest with effective flags | every result JSON through `m15_manifest.write_result` |
| per-image rows kept | `ranks` per query in every front-door JSON; per-pair rows in `m22_assemble_fit_pairs.jsonl` |

## 8. Deviations from the registration

1. **The FRESH read was not run** (§6).  Registered in advance as a possibility; it happened.
2. **The door that ships is the counted vote, not the transform histogram** (§3.2-3.3).  The
   mission asked for a vote and a bin design; the vote was built, measured, and the bins were shown
   to be the wrong instrument on this data.  The counted vote is that instrument's limit, not a
   different mechanism, and it is reported beside the bin design rather than instead of it.
3. **The placeproof's first run was wrong and was re-run.**  It divided translations by
   `Side.spacing_median`, which is 0.0 on these images (co-located places), so it reported a
   transform IQR of 1.6e10 -- a measurement of the divisor.  The script now uses the door's own
   unit (median nearest-distinct-neighbour distance) and the corrected numbers are the ones quoted
   in §3.2.
4. **The class score is measured on the shortlist**, so its ceiling is the front door's; that is
   what the mission asked for ("its shortlisted instances"), and it is why the class number cannot
   be compared with a full-gallery class read.
5. **DEV's 0.938 bar is missed** (0.917 best, one object).  Reported, not smoothed: §3.3.
6. **The missing pitch cell was not added.**  It may be added read-only, and it is the one
   measurement that would settle the yaw question; it is named as the next step rather than done,
   because the mission's out-of-scope clause keeps the yaw harness out and this report would rather
   name the gap than half-fill it.

## 9. Proven / not proven / next

**Proven.**  A front door that reads only pooled place content and counts agreeing places puts the
right object inside 8 candidates 0.900 of the time on the FIT gallery (36/40, chance 0.200) at
0.015 s per candidate -- 22x cheaper than the matcher's first stage -- and the shipped matcher then
re-orders those 8 to a top-1 of 0.450 (18/40, assembled over the appearance shortlist) and 0.550
(22/40, over the shipped door's shortlist), against 0.175 (7/40) for the deployed matcher alone.  The class score
(C) turns instance evidence into a class ranking at 0.625 (25/40) against 0.125 chance.  All of it is
behind flags that are off by default, with the deployed numbers byte-identical.

**Not proven.**  Anything on held-out data: the FRESH split is 16 objects and its MDE (0.706)
exceeds the planning effect (0.50), so no confirmatory claim is made and the fitted-looking numbers
are FIT and DEV only.  Nor is the geometric vote: it is measured, it fails, and the reason is
recorded, but "place voting by transform agreement" remains unproven on this task rather than
disproven.  Nor is the operating range complete: in-plane and pitch are descriptor-level diagnoses,
and yaw is unresolved because the pitch cell was never run through the accuracy harness.

**Next.**  (1) Measure pitch (x20) and yaw (y20) through the same accuracy harness on the same 20
scenes -- one run of `placement_m21_yaw.py --stage variants` -- to settle whether a yaw wall exists
on that instrument at all.  (2) When a held-out set of at least 36 objects exists (a fourth split,
or 10 instances per class from a second dataset), run the registered read unchanged: this report
already carries the arms, the constants and the decision rule.  (3) If the geometric vote is ever
wanted, it needs a correspondence model that is not a similarity -- an affine or a
view-conditioned transform -- because the measured failure is that the transforms of true
correspondences do not agree, and no key or bin width can repair that.
