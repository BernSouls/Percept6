# The placement system, for someone who has not read the other thirty reports

This folder is the evidence shelf for the object-placement work (Missions 1-23).  This file is the
entry point: what ships, how to run it, what its numbers mean, and what it does not do.
Everything here is measured on **ETH-80** (8 classes x 10 instances, one photograph per instance
per pose bin), plus a synthetic renderer with exact ground truth for the view question.  Every
rate carries its **n** and its **chance**.

## 1. What the system does

Given a query photograph of one object and a gallery of one photograph per known object, return a
ranking of the gallery.  Two layers, the first cheap and approximate, the second expensive and
accurate:

```
image -> graph (perception pipeline) -> PLACES  (nameable parts, each with a configuration)
                                          |
     FRONT DOOR  (cheap, no matching)   ->  shortlist of S = 8 objects
       1. CONTENT-COUNT DOOR: every query place probes a pooled index of gallery places by its
          content key, keeps its nearest pool places under the frozen configuration ruler, and
          COUNTS how many of the query's places found a place in each object.  The score is
          count / sqrt(query places x object places).  No pairwise loop, one dict probe and one
          sort per query place.
       2. APPEARANCE: a fixed 4x4x4 colour + 12-bin gradient histogram, cosine.
       3. COMBINED (shipped): reciprocal rank fusion of the two lists, k = 60, no weight.
                                          |
     MATCHER (Mission 19, unchanged)     ->  re-orders the 8 by the shipped pair score.
                                          |
                                      ranking of the gallery
```

The re-order **never removes a candidate**, so the assembled system's ceiling is the door's
recall@8, and both numbers are always reported together.

**Why the door counts instead of voting on a transform.**  The obvious graph-native door is a
Hough vote: each place pairing proposes a similarity transform and votes for the object it came
from; the object with the largest agreeing cluster wins.  It was built (Mission 22) and measured:
FIT recall@8 **0.575 (23/40, chance 0.200)**.  With exact correspondences from the renderer
(Mission 23) the reason is measurable -- true pairs agree on a similarity at **6.6x** their
shuffled null at yaw 20, against the 10x the mission required before it would ship such a vote --
so **geometric voting does not work on out-of-plane views**, and the transform vote is not one of
the shipped options.  The counted door is what is left, and it is named for what it does.

## 2. How to run it

```bash
export PLACEMENT_M19=1                      # the deployed matcher

# the front doors on a split -> recall@8, top-1, chance, per class, per image
.venv/bin/python -u Topology/placement_m22_eth80.py --split FIT --stage frontdoor --workers 2

# the assembled system: shortlist -> matcher -> re-order (resumable, keeps a pairs cache)
.venv/bin/python -u Topology/placement_m22_eth80.py --split FIT --stage assemble --workers 2 \
    --front fused --tag m22_assemble_fit_fused

# T2 done properly: the query's own object REMOVED from the gallery before the door runs
.venv/bin/python -u Topology/placement_m23_eth80.py --stage t2 --workers 2

# the three final orders (door / matcher re-order / rank fusion), FIT and DEV
.venv/bin/python -u Topology/placement_m23_eth80.py --stage fused --workers 2

# the read over all 80 objects and the unseen views (stopped safely at any point)
.venv/bin/python -u Topology/placement_m23_eth80.py --stage read --workers 2 --rerank plain \
    --views=-035-045,-045-180

# the view question, with exact correspondences, and the pitch/yaw accuracy cells
.venv/bin/python -u Topology/placement_m23_geovote.py --scenes 20 --workers 2
.venv/bin/python -u Topology/placement_m21_yaw.py --stage variants --views y20,x20 \
    --variants sorted --tag m23_yaw_pitch --workers 2
```

Splits (`m20_splits.json`, salt `M20-split-v1-2026-11-18`, rule `sha256(salt|class|instance)`):
**FIT 40, DEV 24, FRESH 16 objects**.  FIT is where choices are made, DEV reports, FRESH is held
out -- and Mission 22 showed the FRESH split is too small to resolve the question (MDE 0.706
against a planning effect of 0.50), so Mission 23's confirmatory read is over **all 80 objects and a
never-read view** (one of the two the design named, per the registration's budget rule), labelled
for what that is: view-held-out, not object-held-out.

## 3. The shipped configuration, and its flags

| flag | default | what it turns on |
|---|---|---|
| `PLACEMENT_FRONTDOOR` | `off` | which door a harness may build a shortlist with |
| `PLACEMENT_ASSEMBLED` | off | the matcher re-orders a front door's shortlist |
| `PLACEMENT_CLASS_SCORE` | off | a class is scored by pooling its instances' matcher scores |
| `PLACEMENT_VOTE_HITS` | 3 | pool places one query place may vote with |
| `PLACEMENT_VOTE_ROT` | on | whether a place with an orientation frame also votes on rotation |

With all of them off, nothing in the tree calls the new code and the deployed numbers are
byte-identical to the Mission-20 record (`placement_m12_perf.py --check`: 0 diffs, 816 s).

Fixed constants, none of them tuned: shortlist **S = 8**; RRF **k = 60**; one query place probes
its content bucket and counts its **3** nearest pool places; object score
`count / sqrt(n_query_places x n_object_places)`; the configuration ruler is Mission 21's frozen
file (sigma 0.35 / 0.50).  **The final order is the matcher's plain re-order, not the rank
fusion**: on FIT the fusion scored 0.500 against the re-order's 0.550 (n = 40, chance 0.025) and
on DEV 0.625 against 0.750 (n = 24, chance 0.042), so the fusion is measured, reported and not
shipped.

## 4. The numbers this ships with

| what | rate | n | chance |
|---|---|---|---|
| door recall@8, FIT (40-item gallery) | **0.900** (36/40) | 40 | 0.200 |
| door recall@8, DEV (24-item gallery) | 0.875 (21/24) | 24 | 0.333 |
| assembled top-1, FIT | **0.550** (22/40) | 40 | 0.025 |
| assembled top-1, DEV | 0.750 (18/24) | 24 | 0.042 |
| deployed matcher top-1, FIT (M21 AFTER) | 0.175 (7/40) | 40 | 0.025 |
| class top-1, T2 with the query's own object removed (top-2 pooling) | **0.500** (20/40) | 40 | 0.103 |
| class top-1, T2, `max` arm | 0.525 (21/40) | 40 | 0.103 |
| assembled top-1, **view-held-out read** (view `-035-045`, 80-item gallery) | **0.250** (20/80) | 80 | 0.0125 |
| deployed matcher, same read | 0.075 (6/80) | 80 | 0.0125 |
| door recall@8, same read | 0.550 (44/80) | 80 | 0.100 |

**The confirmatory read** (`MISSION23_REPORT.md` §5): on view `-035-045`, which no earlier mission
had read, the assembled system beats the deployed one by **+0.175 (20 against 6 of 80, 95 % CI
[0.088, 0.263], bootstrap p = 0.0001)**.  It is **one view and view-held-out, not object-held-out**
-- all 80 objects were used to choose the door -- and the interval is wide; the effect is real, its
size is not pinned.

**The door on DEV, stated plainly.**  The counted door and the fusion were chosen on FIT.  On DEV
the counted door alone reaches **0.750 (18/24)** -- *below* appearance alone at 0.875 (21/24) --
and the fusion's DEV recall equals appearance's 0.875 (21/24).  The door's advantage is a FIT
result that reverses on DEV, and the shipped choice rests on FIT because that is the rule the
mission fixed in advance, not because DEV agrees.

## 5. The operating range, measured

| view (renderer, 20 scenes) | what the descriptor does | geometric vote |
|---|---|---|
| in-plane rotation, 15 deg | node accuracy 0.203 (M21, report half) | agrees at 5.0x null -- not enough |
| uniform scale, 1.15x | 0.923 self-similarity (M21: a known penalty of the shipped descriptor) | 4.2x null |
| **pitch, 20 deg** | **node accuracy 0.373** (n = 4,981) | 17.2x null -- *it would work here* |
| **yaw, 20 deg** | **node accuracy 0.226** (n = 3,753) | 6.6x null -- fails |

Reading: the system is usable up to in-plane 15 deg, uniform scale 1.15x and **pitch 20 deg**, and
**yaw 20 deg is where it degrades** -- the two out-of-plane cells, measured side by side on the
same harness for the first time in Mission 23, are 0.373 (pitch) against 0.226 (yaw).  The
transform vote is the mirror image: it nearly works at pitch (17x) and fails at yaw (6.6x), which
is exactly why the counted door ships.

## 6. What it does and does not do

**Does:** find a known object from another view inside a gallery of 24-80 known objects -- top-1
**0.250** on the hardest case measured (an 80-item gallery, a view never read, 20/80) and
0.550-0.750 on the FIT/DEV galleries, with the right object inside 8 candidates 0.550-0.900 of the
time, at 0.015 s per candidate for the door against 0.33 s per pair for the
matcher's own first stage.  It also says **which class** a new instance belongs to: with the
query's own object removed from the gallery, the class is the top-scoring one 0.500 of the time
(20/40) against a chance of 0.103.

**Does not:** recognise a *new* instance of a class reliably -- 0.500 against 0.103 chance is real
but half the time it is wrong, and the ceiling is not the class score: the door's 8 candidates
contain, on average, only 46 % of the query's class instances.  Its confirmatory evidence is **one unseen view**, not a set of unseen objects -- every object in
the read helped choose the door.  It does not recover pose: the transform between two views of an
object is not a similarity (measured on true correspondences:
rotation interquartile range 2.8 rad, translation tens of place-spacings), so no similarity vote
can cluster.  It does not handle yaw beyond ~20 deg.  And it has **no object-held-out confirmatory
number**: FIT and DEV chose the door, and every object is in Mission 23's view-held-out read.

## 7. Where the evidence is

| file | what it holds |
|---|---|
| `MISSION23_REPORT.md` | the current mission: T2 done properly, the view question closed, the read |
| `MISSION23_PREREGISTRATION.md` | the read's design, its MDE, and the amendment recorded before it ran |
| `MISSION22_REPORT.md` | the front door and the assembler; §10 is Mission 23's correction of two of its claims |
| `MISSION21_REPORT.md` | the frozen ruler, the coverage control, the yaw question |
| `MISSION20_REPORT.md` | the splits, the tasks, the deployed matcher's record |
| `m23_t2.json`, `m23_fused.json`, `m23_read.json` | items 1, 3 and 5 |
| `m23_geovote.json`, `m23_yaw_pitch.json` | item 2: agreement on true correspondences, and pitch vs yaw |
| `m22_fit.json`, `m22_dev.json` | the front doors per split |
| `m23_power.json` | the arithmetic that let the read run |

Every result JSON is written through `matching/m15_manifest.write_result`, so it carries the
manifest of the flags in force when it was produced.
