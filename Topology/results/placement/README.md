# The placement system, for someone who has not read the other thirty reports

This folder is the evidence shelf for the object-placement work (Missions 1-23).  This file is the
entry point: what ships, how to run it, what its numbers mean, and what it does not do.
Everything here is measured on **ETH-80** (8 classes x 10 instances, one photograph per instance
per pose bin), plus a synthetic renderer with exact ground truth for the view question.  Every
rate carries its **n** and its **chance**.

## 1. What the system does

Given a query photograph of one object and a gallery of one photograph per known object, return a
ranking of the gallery.  Two layers, the first cheap and approximate, the second expensive:

```
image -> graph (perception pipeline) -> PLACES  (nameable parts with a configuration and a frame)
                                          |
   CONTENT-COUNT DOOR (cheap, no matching) -> shortlist of S = 8 objects
       an object may be stored under several views (M25: k = 1 by default); its score is the MAX
       over its views, so a stored view that suits the query is what counts.
       every query place probes a pooled index of gallery places by content key, keeps its
       nearest 8 pool places under the frozen Mission-21 ruler, and COUNTS which object they
       belong to: score = count / sqrt(n_query_places x n_object_places).  One dict probe and
       one sort per query place, no pairwise loop.
   + APPEARANCE: a fixed 4x4x4 colour + 12-bin gradient histogram, cosine.
   -> RRF (k = 60, no weight) fuses the two lists.
                                          |
   MATCHER (Mission 19, unchanged)         ->  re-orders the 8 by the shipped pair score.
                                          |
                                     ranking of the whole gallery
```

The re-order never removes a candidate, so the assembled system's ceiling is the door's recall@8,
and both numbers are reported together.

**Why the door counts instead of voting on a transform.**  The graph-native alternative is a Hough
vote: each place pairing proposes a similarity transform and votes for the object it came from.
Measured with *exact* correspondences from the renderer (Mission 23), true pairs agree on a
similarity at **6.6x** their shuffled null at yaw 20, against the 10x the mission required before
it would ship such a vote.  **Geometric voting does not work on out-of-plane views**, so the
transform vote is not one of the shipped options; the counted door is, and it is named for what it
does.

## 2. How to run it -- the whole recipe

```python
from matching.placement_recognise import build_gallery, recognise

items   = [("apple/1", "apple1-000-000.png"), ("apple/2", "apple2-000-000.png"), ...]
gallery = build_gallery(items)                  # place graphs, pooled index, appearance; cached
result  = recognise("apple3-035-045.png", gallery)

# several stored views per object (M25): the object's score is the max over its views, and the
# matcher scores it on the one view the door chose (8 pairs per query, whatever k is)
items = [("apple/1", {"000-000": "apple1-000-000.png", "045-000": "apple1-045-000.png"}), ...]
gallery = build_gallery(items, views=["000-000", "045-000"])
result.best_view                                # {object index: the stored view that won}

result.top, gallery.keys[result.top]            # the best gallery item
result.ranking                                 # EVERY gallery index, best first (re-order drops none)
result.shortlist                               # the S = 8 the matcher actually scored
result.door_order                              # the door's own ranking of the whole gallery
result.scores, result.unscored                 # matcher scores per index; refused pairs are listed,
                                               # never zeroed and never defaulted
result.best_view                                # per shortlisted object, the stored view that won
result.seconds                                 # {"appearance", "query_record", "door", "matcher", "total"}
```

That is the shipped system: no environment variable to set, no flag to remember, no import from
`Topology/`.  `recognise(..., matcher=False)` stops after the door and costs no pair score.
`workers=2` scores the 8 pairs in parallel and the result is collected in shortlist order, so the
answer is identical to the serial run (`tests/test_m24_core.py` asserts it).

**The configuration is one frozen file**: `matching/placement_bundle.json` -- the M19 weight
`w = 0.375`, the front-end recipe, the door's `n_hits = 8`, `k = 60`, `S = 8`, `rerank = plain`,
provenance, and a digest of the code that reads it.  A missing bundle, or one frozen for other code
or another recipe, is a **refusal**: nothing in the shipped path has a default for these numbers.
Re-freeze (an operator action, review the diff) with:

```bash
python -m matching.placement_recognise --freeze --date YYYY-MM-DD
python -m matching.placement_recognise --describe
```

The old `PLACEMENT_*` environment flags are **legacy**: the entry point does not read them, applies
its own bundle internally, and restores every switch it touched before returning (asserted by the
same test).  They stay in `parameters/_text.py` because the graph cache key and the M14 switch-
digest test both name them -- see `archive/ARCHIVE_INDEX.md`.

## 3. Where everything lives now (Mission 24)

| what | where |
|---|---|
| **the entry point** | `matching/placement_recognise.py` -- `build_gallery`, `recognise`, `order_plain` |
| the frozen configuration | `matching/placement_bundle.json` (+ `placement_score_params.json`, `placement_prefilter_params.json`) |
| the front end and place records | `matching/placement_frontend.py` (+ `matching/placement_cache.py` for graphs) |
| the door | `matching/placement_frontdoor.py` |
| the gallery and its cache | `matching/placement_gallery.py` (k views per object, one pooled index) |
| **the stored-view rule** | `matching/placement_views.py` -- the grid, the angular metric, and `reference_views(k)`, which chooses the views with no score in sight |
| what the shipped path touches | **`KEEP_MAP.md`** |
| everything moved out of the way | **`archive/ARCHIVE_INDEX.md`** (375 files, nothing deleted) |
| the checks that prove it is the same system | `Topology/verify_m24_equivalence.py`, `Topology/placement_m12_perf.py --check`, `tests/test_m24_core.py` |

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
| door recall@8, same read, **four stored views** (M25) | 0.575 (46/80) | 80 | 0.100 |
| assembled top-1, same read, four stored views (M25) | **0.375** (30/80) | 80 | 0.0125 |
| door recall@8, view `-045-180` (never read before M25), one stored view | 0.512 (41/80) | 80 | 0.100 |
| assembled top-1, view `-045-180`, one stored view | 0.225 (18/80) | 80 | 0.0125 |

**The confirmatory read** (`MISSION23_REPORT.md` §5): on view `-035-045`, which no earlier mission
had read, the assembled system beats the deployed one by **+0.175 (20 against 6 of 80, 95 % CI
[0.088, 0.263], bootstrap p = 0.0001)**.  It is **one view and view-held-out, not object-held-out**
-- all 80 objects were used to choose the door -- and the interval is wide; the effect is real, its
size is not pinned.

**Does storing more views raise the ceiling?  (M25, `MISSION25_REPORT.md`).**  No.  Storing four
views per object instead of one -- chosen by a score-free rule, covering radius 65.5 deg instead of
90 deg, putting the query view 30.0 deg from its nearest stored view instead of 35.0 -- moves the
door's recall@8 by **+0.025 (44 -> 46 of 80, 95 % CI [-0.100, 0.150], p = 0.77)**.  The assembled
top-1 moves further, **+0.125 (20 -> 30 of 80, CI [0.013, 0.250])**, because the matcher gets a view
that suits the query, but that is not significant across the mission's two primaries (Holm 0.092),
the read finished with one complete cell (`per_obj = 1`, MDE 0.187 against a planning effect of
0.20), and the mission keys the shipping decision on the recall.  **The bundle stays at k = 1.**
The measured reason the door misses is not view distance -- it is flat from 30 to 45 deg (0.575 at
30, 0.550 at 35, 0.512 at 45) -- but what the door counts: 0.20 for cows and horses and 0.25 for
pears against 0.90 for cups (n = 20 each), i.e. a count of shared place-keys that thins out over
objects built from generic parts.

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
| `m24_equivalence.json` | Mission 24: the core entry point against the read's own records (80/80 door, 44/44 matcher order) |
| `../../KEEP_MAP.md` | what the shipped path touches and where each file lives |
| `../../archive/ARCHIVE_INDEX.md` | the 375 files Mission 24 moved out of the way, with the reason for each |

Every result JSON is written through `matching/m15_manifest.write_result`, so it carries the
manifest of the flags in force when it was produced.

**The one-paragraph version of Mission 24.**  The working system is now a normal part of the core:
`matching/placement_recognise.py` builds a gallery and recognises a query, configured by one frozen
file that refuses rather than defaults, importing nothing from `Topology/` and needing no
environment variable; `matching/placement_frontend.py`, `placement_cache.py`, `placement_truth.py`
and `placement_gallery.py` are the code that moved there to make that true.  Everything the shipped
path does not need -- 200 top-level scripts and 175 result files, 56.8 MB -- is in `archive/` with an
index that says which mission each served, and nothing was deleted.  The same system was proved to
be the same system, not a rewrite: the door's shortlist matches the Mission-23 read for **80 of 80**
queries, the shipped order matches the row's recorded order for **44 of 44** queries whose pairs
were cached, the legacy parity check is **identical: True (0 diffs)**, and the suite is green (168
tests).  **The single next lever is the door's recall**: 0.55 on the hard views is the ceiling the
whole system inherits, and no amount of matcher work can recover a truth the shortlist never
contained.
