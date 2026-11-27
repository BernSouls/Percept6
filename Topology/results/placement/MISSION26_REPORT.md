# Mission 26 -- the door's content key: hard equality -> graded

The shipped door compares two places by exact equality of `(lens, quantised tuple)` and looks the
bucket up in a dict.  The matcher already compares places by the registry's graded kernel.  This
mission measured whether replacing the door's equality with that kernel moves the shortlist
ceiling.  It does not, and the reason is in the numbers rather than in the register.

Everything below runs on the cached place records (`Topology/m26_records.py`, 320 cells, 2,338 s of
wall clock on 2 workers) and on the images for the appearance floor.  No graph is built for a door
experiment and the matcher runs only in item 7.  Registration:
`MISSION26_PREREGISTRATION.md`, written before any door variant was scored.

---

## 1. The census (item 1)

`Topology/m26_census.py` -> `m26_census.json`.  320 cells (80 objects x 4 views), **169,436 place
rows**, 322-856 places per cell (median 506).

| lens class | place rows | share | graded kernel (ranges) | population axes |
|---|---|---|---|---|
| Blob | 95,291 | 56.2 % | **yes** (12, 4, 3, 1, 1) | yes (payload NOT in the record) |
| BoundaryArc | 67,175 | 39.6 % | **yes** (same) | yes (payload NOT in the record) |
| CornerPoint | 6,970 | 4.1 % | **yes** (same) | yes (payload NOT in the record) |

* **Dropped places, by reason, over all 320 cells:** `no_config` **2,727**, and nothing else
  (`no_sig` 0, `no_position` 0, `no_scale` 0) -- **1.58 % of the 172,163 payload places**.
* **Places that can vote (live configuration >= 2 slots): 100.0 %** of the rows that survive the
  record's drops, on every view (000-000 40,399 places, 022-000 41,953, 035-045 43,020,
  045-180 44,064).
* So the ladder's migration branch (item 6C, ">= 25 % of query places dropped") is **not
  reachable on this corpus**: 1.6 % are dropped and every survivor can vote.  The population
  payload is absent from the record, but its absence is not costing places -- it is costing the
  *reading*, which is what item 2 exists to make cheap.

**The audit: every place in the tree that compares places by a quantised key** (file:line, from
the census):

| where | line | what it is |
|---|---|---|
| `matching/placement_frontdoor.py` | 137, 141, 149-151 | `PlaceIndex.by_key` -- the door's exact-key bucket, built from `row["key"]` |
| `matching/placement_frontdoor.py` | 177 | `index.by_key.get(self._keyfn(q))` -- the transform-vote door (not shipped) |
| `matching/placement_frontdoor.py` | 279 | `index.by_key.get(index._keyfn(q))` -- **the shipped counted door** |
| `matching/placement_frontend.py` | 176 | `key = (lens, tuple(sig.quantized))` -- where the record's key is written |
| `core/types.py` | 527 | `if self.quantized == other.quantized: return 1.0` -- the kernel's own equality shortcut |
| `core/types.py` | 390, 620 | equality used for compatibility/fast paths, not for a score |
| `core/schema.py` | 1652 | the schema's equality shortcut |

Nothing else in `matching/` or `core/` turns a quantised tuple into a retrieval decision.

## 2. The vectorised registry kernel (item 2)

`matching/placement_kernel.py`: the registry's constitution, slot weights, circular theta cycle,
absent-slot renormalisation and `core.slots.class_slot_distance` overrides, evaluated as an
`(nq, np)` block.  sigma_i = ranges_i / 3 -> theta 4 bins, coherence 1.33 bins, scale 1.0 px,
kind and dim 1/3 -- i.e. one categorical mismatch costs exp(-1) = 0.368.

**Parity, and it fails on revert:** `tests/test_m26_kernel.py` compares the block against
`TypedSignature.similarity` (quantised branch, population absent, shipped recipe flags) on

| pairs | classes | max deviation | verdict |
|---|---|---|---|
| 2,500 random per class x 4 classes (10,000 pairs) | CornerPoint, Blob, BoundaryArc, Keypoint | < 1e-9 | equal |
| 1,600 real pairs per class from the ETH-80 records (4,800 pairs) | the three registered classes | < 1e-9 | equal |

The test also pins that the kernel is **not** an equality function (one theta bin: 0.973, not 0.0
or 1.0), that theta wraps (bin 11 against bin 0 equals bin 0 against bin 1), that one categorical
slot costs exp(-1), that absent slots renormalise, and that a class with no ranges is equality
**and is counted** in `FALLBACKS`.

## 3. The graded door (item 3)

`matching/placement_frontdoor.py::graded_scores`, with the bundle's `door.kind = "graded"`:

```
s(q, p) = kernel(lens, q.quantised, p.quantised) x exp(-d2 / 2)
cover(q, o) = max over o's places of s(q, p) >= tau          tau = 0.5
w(q) = log((N + 1) / (df(q) + 1))                            df = objects q covers
score(o) = sum_{q covers o} w(q) / sum_q w(q)
```

Both constants (`tau = 0.5`, the `1/2`) were fixed in the registration before any number existed.
The counted door is untouched: `door.kind = "counted"` still calls `count_scores` and the recorded
M24/M25 check (door 80/80, matcher 44/44) is pinned by `tests/test_m25_views.py` on
`m25_k1_equivalence.json`; a new test pins that an unknown `door.kind` is refused.

## 4. The oracle table (item 4)

`Topology/m26_oracle.py` -> `m26_oracle.json` (+ the raw score matrices in `m26_oracle.npz`,
gitignored).  80-object gallery at the reference view `000-000`; chance is recall@8 = 8/80 =
**0.100**, top-1 = **0.0125**; n = 40 per cell.

**Selection cells -- FIT objects (n = 40 per view, chance 0.100):**

| variant | `035-045` | `045-180` | `022-000` (saturation) |
|---|---|---|---|
| a counted door (shipped, alone) | 0.350 (14/40) | 0.450 (18/40) | 0.725 (29/40) |
| b appearance (shipped floor) | **0.575 (23/40)** | 0.500 (20/40) | 0.500 (20/40) |
| **c RRF(a,b) -- the shipped door** | **0.550 (22/40)** | **0.550 (22/40)** | **0.725 (29/40)** |
| d graded signature only | 0.100 (4/40) | 0.050 (2/40) | 0.050 (2/40) |
| e configuration only | 0.125 (5/40) | 0.200 (8/40) | 0.125 (5/40) |
| f the graded door | 0.325 (13/40) | 0.325 (13/40) | 0.375 (15/40) |
| g f without distinctiveness | 0.300 (12/40) | 0.375 (15/40) | 0.500 (20/40) |
| h f with pooled top-8 | 0.250 (10/40) | 0.400 (16/40) | 0.500 (20/40) |
| i graph-native global layout | 0.300 (12/40) | 0.400 (16/40) | 0.550 (22/40) |
| j RRF(b,f) | 0.500 (20/40) | 0.525 (21/40) | 0.575 (23/40) |
| j RRF(b,f,i) | 0.525 (21/40) | 0.575 (23/40) | 0.750 (30/40) |
| j RRF(b,a,f) | 0.500 (20/40) | 0.575 (23/40) | 0.700 (28/40) |
| **UNION ORACLE a..i** | **0.825 (33/40)** | **0.725 (29/40)** | **0.875 (35/40)** |

**Confirmation cells -- DEV + FRESH objects (n = 40 per view, chance 0.100):**

| variant | `035-045` | `045-180` |
|---|---|---|
| c shipped door | 0.550 (22/40) | 0.475 (19/40) |
| b appearance | 0.600 (24/40) | 0.425 (17/40) |
| a counted alone | 0.300 (12/40) | 0.400 (16/40) |
| f graded door | 0.200 (8/40) | 0.200 (8/40) |
| j RRF(b,f) | 0.550 (22/40) | 0.425 (17/40) |
| j RRF(b,f,i) | 0.450 (18/40) | 0.450 (18/40) |
| j RRF(b,a,f) | 0.500 (20/40) | 0.475 (19/40) |
| UNION ORACLE a..i | 0.825 (33/40) | 0.675 (27/40) |

Per class on the selection view `035-045` (n = 5 objects per class, chance 0.100):

| class | shipped door (c) | graded door (f) | counted alone (a) | appearance (b) |
|---|---|---|---|---|
| apple | 1.0 (5/5) | 0.6 (3/5) | 0.8 (4/5) | 0.4 (2/5) |
| car | 0.4 (2/5) | 0.0 (0/5) | 0.0 (0/5) | 0.8 (4/5) |
| cow | 0.0 (0/5) | 0.0 (0/5) | 0.0 (0/5) | 0.2 (1/5) |
| cup | 1.0 (5/5) | 1.0 (5/5) | 1.0 (5/5) | 0.8 (4/5) |
| dog | 0.4 (2/5) | 0.2 (1/5) | 0.2 (1/5) | 0.6 (3/5) |
| horse | 0.2 (1/5) | 0.2 (1/5) | 0.0 (0/5) | 0.4 (2/5) |
| pear | 0.6 (3/5) | 0.0 (0/5) | 0.4 (2/5) | 0.6 (3/5) |
| tomato | 0.8 (4/5) | 0.6 (3/5) | 0.4 (2/5) | 0.8 (4/5) |

**Thin objects.** No failure of the counted door sits on an object with fewer than 50 % of the
median place count (share of failures = **0.00** in every cell; the records hold 322-856 places,
median 506), so thinness is not an explanation for anything measured here.

**Why the graded door fails, measured rather than argued.**  With ``tau = 0.5`` and a per-object
max over every place, a query place covers **77 of 80 objects on average** (`mean_df` 76.9-77.6 in
every cell): 500 places per object is enough that some place always looks similar, so the
distinctiveness weight `log((N+1)/(df+1))` is ~0.04 for everyone and the door degenerates into a
near-uniform vote.  A sensitivity sweep on the selection view `035-045` (post-hoc, exploratory --
the registered variant stays `tau = 0.5`) shows no rescue in the threshold:

| tau | 0.30 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| recall@8 | 0.225 (9/40) | 0.325 (13/40) | 0.300 (12/40) | 0.275 (11/40) | 0.275 (11/40) |
| mean df | 79.2 | 77.0 | 67.9 | 41.6 | 8.4 |

and none in the configuration scale (exponent 1, 1/2, 1/4 -> 0.350, 0.325, 0.350).  The single
channel that reaches chance-level is **(d) the graded signature compared place-to-place: 0.100
(4/40) -- exactly chance**.  The exact key works not because equality is a better *similarity* but
because it is a hard partition that leaves few candidates to count; the graded kernel spreads mass
over the whole pool.

## 5. The selection rule, applied (item 5)

The rule (registration section 5): ship only if a variant's FIT-object recall@8 beats the shipped
door (c) by **>= 0.075** and its confirmation-object difference is **>= -0.025**.

* the shipped door (c) scores **0.550** on `035-045` and 0.550 on `045-180` (FIT);
* the best graded variant is (f) at **0.325** and (j) RRF(b,f,i) at **0.525** -- both **below** c;
* nothing comes within 0.075 of c, on either selection view, in either split.

**No graded variant passes.**  The confirmation cells confirm the ordering (f 0.200/0.200, c
0.550/0.475), so this is not a FIT accident.

## 6. The outcome ladder (item 6)

**A -- a graded variant becomes the door: NO** (section 5).

**B -- the union oracle >= 0.80 on the selection cells: PARTLY, and it does not ship.**  The union
of the cheap channels reaches 0.825 (33/40) on `035-045` and 0.725 (29/40) on `045-180`; over the
80 selection cells it is **0.775 (62/80)**, below the 0.80 the ladder requires, so B does not
trigger.  Even where the union is high (0.825, 0.875 with the saturation view) **no fixed pair
reaches it**: the best fixed fusion on the selection cells is RRF(b,f,i) at 0.525/0.575, and the
shipped RRF(a,b) at 0.550/0.550.  The union is an upper bound that rank fusion does not collect --
the channels agree on *where* the truth can be found and disagree on *how to rank* it.

**C -- the record migration: NOT REACHABLE** (1.58 % of places dropped, section 1).

**D -- adaptive S: SHIPPED.**  The rule, fixed on the FIT-object cells before the confirmation
cells were looked at: for each query take the fused door's scores, sort them, and

```
margin = (score[8] - score[9]) / score[1]        (0-based: the 8th and 9th candidates)
S = 16 when margin < 0.007168, else S = 8        0.007168 = the FIT median margin (n = 80 cells)
```

| cells | ceiling at S = 8 | **adaptive S** | ceiling at S = 16 (all) | widened | pairs / query |
|---|---|---|---|---|---|
| FIT selection (n = 80) | 0.550 | **0.613** | 0.688 | 50 % of queries | 12.0 |
| DEV + FRESH confirmation (n = 80) | 0.512 | **0.575** | 0.625 | 48 % of queries | 11.8 |

The ceiling is the door's recall@S; the matcher can only re-order what the door hands it.  Adaptive
S buys **+0.063** of ceiling on both splits for half the queries and 4 extra pairs on those.  Two
things are stated plainly because the numbers say them: **a fixed S = 16 beats the adaptive rule**
(0.688 against 0.613 on FIT, 0.625 against 0.575 on confirmation) for 4 more pairs per query, and
**the top-8 margin is a poor predictor of where widening helps** -- on the selection cells the
truth sits in ranks 9-16 for 7 of the 60 widened queries (11.7 %) against 10 of the 60 "clear" ones
(16.7 %).  The rule ships because the mission fixed it; the report records that a flat S = 16 is
the stronger and simpler setting, at 16 pairs per query against 12.

## 7. The matcher step (item 7)

The door is fixed (counted, adaptive S), so the matcher's job is the one M23 measured: re-order
the shortlist.  `Topology/m26_matcher.py` re-derives the door's shortlists from the cached query
records (verified identical to `recognise` on a live query: same 16-object shortlist), and asks for
**one pair per shortlisted object, on the view the door chose** (k = 1, so the reference view):

| | |
|---|---|
| pairs asked for, 80 queries | **992** (12.4 per query: 8 pairs for the clear half, 16 for the widened half) |
| pairs already cached | **0** (this sandbox's pair cache was empty) |
| estimate at the measured rate | 992 x 11 s / 2 workers -- **the first batch ran at 15.9 s per pair** (cold graphs: both images of every pair had to be built), i.e. **~4.4 h** for the whole cell |
| run under the budget | **209 pairs** in 53 minutes (the first 40 queries) |
| fully scored queries | **17** of 80 |
| **assembled top-1 (ESTIMATE)** | **0.353 (6/17)**, 95 % CI [0.118, 0.588], chance 0.0125 |
| against the recorded 0.250 | **+0.103**, CI [-0.132, 0.338] -- indistinguishable from M23/M25's number, on n = 17 |
| paired against the door's own ceiling on the same queries | **-0.235**, CI [-0.471, -0.059], p = 0.02 |

The last row is the honest reading of what the matcher does: it is given 17 shortlists that
contain the truth 17 times (ceiling 1.000 on that subset) and it puts the truth first 6 times, so
the matcher *loses* objects the door had already found -- the same behaviour M23 measured at
0.250 against 0.550, not a new defect and not improved by the wider shortlist.

**Queries still lacking a score: 63**, with **783 pairs** unscored.  They are listed one by one in
`m26_matcher.json` (`report.missing`, each with its shortlist size and how many pairs are missing).
No pair is zeroed and no query is dropped silently: the estimate above uses only the 17 queries
whose whole shortlist was scored, because ranking a partial shortlist would flatter the matcher by
hiding the candidates it never saw.

To finish the cell: `python Topology/m26_matcher.py --run --budget-minutes 240` (resumable; the 209
scored pairs are already cached and cost nothing to re-read).

## 8. Ship (item 8)

* **the door is unchanged**: `door.kind = "counted"`, `count_scores`, n_hits 8 -- the recorded
  M24/M25 equivalence (door 80/80, matcher 44/44) still holds for it, and `tests/test_m25_views.py`
  keeps pinning the recorded check;
* **the graded door ships as a configuration, not a default**: `door.kind = "graded"` with
  `params = {"tau": 0.5}` selects it, and the measurement says it is worse (0.325 against 0.550) --
  kept so that the next mission can extend it rather than rediscover it;
* **the shortlist rule ships adaptive** (`shortlist_rule = "adaptive"`, `shortlist_wide = 16`,
  `margin_tau = 0.007168`), with `shortlist_rule = "fixed"` reproducing M22-M25's S = 8 exactly;
* the legacy path is untouched: `Topology/placement_m12_perf.py --check` is run and reported below;
* `KEEP_MAP.md`, `archive/ARCHIVE_INDEX.md` (nothing archived: no file left the tree) and the
  results `README.md` carry the measured numbers.

| check | result |
|---|---|
| `tests/` (whole suite) | **200 tests OK** (186 before + 14 new M26, which include the counted-door pins) |
| kernel parity vs `core.types` | 14,800 pairs (10,000 random over 4 classes, 4,800 real over 3), max deviation < 1e-9 |
| counted door -- the M22 formula, written out again in the test | exact to 1e-12 on real records |
| M24/M25 equivalence (recorded) | door 80/80, matcher 44/44 -- still pinned by `tests/test_m25_views.py` |
| graded door across hash seeds | identical matrices for `PYTHONHASHSEED` 0 / 1 / 12345 |
| per-object max order-independence, idf permutation equivariance | exact (1e-12) |
| `core/` imports of `Topology/` | none (AST test over the five placement modules) |
| legacy path `Topology/placement_m12_perf.py --check` | **identical: True (0 diffs)**, 503.9 s |
| archived in this mission | nothing -- no file left the tree, so `archive/ARCHIVE_INDEX.md` is unchanged |

---

## The paragraph the mission asks for

**Which door ships, and what it costs.**  The shipped door is still the **counted door** (exact
`(lens, quantised)` bucket, `n_hits = 8`) fused with appearance at RRF k = 60 -- unchanged, because
replacing its equality with the registry's graded kernel *lowered* the shortlist from 0.550 (22/40)
to 0.325 (13/40) on the FIT selection cells and from 0.550 to 0.200 on the confirmation objects.
Its recall at the hard views, per class on `-035-045` (n = 5 per class, chance 0.100): apple 1.0,
cup 1.0, tomato 0.8, pear 0.6, car 0.4, dog 0.4, horse 0.2, **cow 0.0** -- and at 0.550 (22/40) and
0.550 (22/40) over the two FIT views, 0.550/0.475 on the confirmation objects.  **The ceiling of
the cheap channels** is 0.825 (33/40) on `-035-045` and 0.725 (29/40) on `-045-180` (0.775 over the
80 selection cells; 0.875 on the saturation view), but no fixed RRF pair reaches it -- the best is
0.575 -- so what ships from the ladder is **adaptive S**: the shortlist widens to 16 when the door's
own top-8 margin falls under the FIT median (0.0072), which lifts the ceiling from 0.550 to
**0.613** on FIT and from 0.512 to 0.575 on DEV+FRESH, for **12 pairs per query instead of 8** and a
door that costs 1.3 s (counted) plus 0.02 s (appearance) per query -- the door itself never became
expensive.  **The single next lever is descriptor enrichment**: the graded signature compared place
to place scores **0.100 (4/40), exactly chance**, and a query place covers 77 of 80 objects, so the
place descriptor is not discriminative at this resolution -- what makes the exact key work is that
it is a *partition*, not that equality is a good similarity.  The number that justifies the lever
is that gap: 0.100 (graded, place-to-place) against 0.350 (the same places, hard-partitioned) --
the descriptor needs enough concentration that a graded comparison can substitute for a bucket.

---

## Addendum (Mission 27, item 5)

This is an addendum, not a rewrite: the sections above stand as written, and three things in them
are corrected here because they were wrong when measured, not because the measurements changed.

**1. The row "paired against the door's own ceiling, p = 0.02" is WITHDRAWN.**  The comparison was
the assembled top-1 against the door's recall@S *on the same 17 queries*, and on that subset the
door's recall is 1.000 -- every one of the 17 shortlists contains the truth.  A paired test against
a column that is constant by construction tests nothing about the matcher: the difference is just
`top1 - 1` and its p-value describes how often the matcher wins with 17 coin flips.  The measurement
behind the row stands, as a count and not as a test: on the 17 fully scored queries the door's
shortlist held the truth 17 times and the matcher returned it 6 times (0.353).  No p-value is
attached to that.  This is the same defect the mission's standing rule names -- never compare
against a constant -- and the shipped estimate in section 7 (0.353, CI [0.118, 0.588], against the
recorded 0.250 as an interval difference) is unaffected.

**2. "The graded door fails" holds for the WIDE constitution only.**  Every graded-door number in
section 6 was measured with the kernel `core/keypoint_classes.py` registers as `wide`: theta range
12.0 bins, i.e. sigma_theta = 4 bins = 60 deg.  The SAME file registers a `narrow` constitution
(theta range 3.0 bins, sigma_theta = 1 bin = 15 deg), and it was never measured.  So the correct
statement is that *this* kernel, with a 60-degree theta tolerance that makes a rotation almost free,
fails; the registered narrow kernel is tested in Mission 27 item 4a, and the relaxed partition
(item 4b) was never tried at all.  What does not change is the diagnosis of *why* the wide kernel
loses: a query place covers 77 of 80 objects (`df`), so idf is inert and the vote is nearly uniform.

**3. The 992 pairs were mostly already cached: 621 of them.**  Section 7 says "pairs already cached
0 (this sandbox's pair cache was empty)" and then bought 209 pairs at 15.9 s each.  The sandbox's
pair cache was empty, but the REPOSITORY was not: `Topology/results/placement/m23_read_pairs.jsonl`
already held 621 scored pairs of exactly these queries against exactly these references, from M23's
read of the same gallery.  They were invisible because the cache key was the absolute path of both
images, and only ~352 pairs (the ranks 9-16 that the widened shortlist reaches) were actually new.
Two defects followed from one root cause: ~53 minutes were spent re-buying 621 scores, and every new
sandbox would do it again.  Mission 27 item 1a fixes the key (basename, not path) and adopts the
committed files under the current matcher digest; `tests/test_m27_cache.py` fails if a path prefix
change turns a cached pair into a miss again.

**4. Adaptive S shipped without support, and this addendum does not defend it.**  Section 6's own
table shows the margin rule is a poor predictor: it widens on 50 % of queries where widening helps
11.7 % of the time, against 16.7 % of those it did not widen.  Flat S = 16 reaches 0.688 on FIT
against the adaptive rule's 0.613.  Item 2 of Mission 27 settles S on the full 80-query read, by the
rule fixed in `MISSION27_PREREGISTRATION.md`; whatever that rule says is what the bundle carries.
