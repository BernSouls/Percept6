# MISSION 25 -- pre-registration: more reference views against the door's ceiling

Written **before** any graph, door score or matcher score exists for this read.  The only numbers
that existed when this file was written are geometric (view angles) and the Mission-23 record it is
compared against.  One 3-object smoke run of the new code path had happened when this was written
(apple instances 1-3 as the gallery, apple 4 as the query, on the `035-045` view) -- it produced no
number that appears here and none of its objects is scored in the read.  It is disclosed because
"before any score exists" should be true, not nearly true.

## 1. The reference-view rule (item 1)

**The grid.**  ETH-80 stores 41 views per object, named `<polar>-<azimuth>` in degrees:

| polar | azimuths | views |
|---|---|---|
| 000 | 000 | 1 |
| 022 | 000, 090, 180, 270 | 4 |
| 035 | 045, 135, 225, 315 | 4 |
| 045 | 000, 090, 180, 270 | 4 |
| 066 | 027, 063, ... 333 (8, 45 deg apart) | 8 |
| 068 | 000, 090, 180, 270 | 4 |
| 090 | 000, 022, ... 338 (16, 22.5 deg apart) | 16 |

**The metric.**  Two views are two directions; the distance is the great-circle angle between
`v(a,b) = (sin a cos b, sin a sin b, cos a)`.  This was checked against the pictures rather than
assumed: a compound-rotation model (turntable + tilt arm, in which `045-180` would be the object
upside down) was tried first and rejected, because `apple1-045-180` is not upside down -- it is the
far side seen from a 45-degree tilt, which is what a direction on the sphere predicts.

**The distances.**  The grid's smallest non-zero separation is **22.0 deg** (adjacent views of the
16-view equator ring).  `000-000` alone covers the grid to a radius of **90.0 deg**.

**The rule.**  `S = {000-000} union T`, where `T` is the 3-subset of the 39 non-reserved views that
minimises the covering radius `R(S) = max over all 41 views of min over r in S of d(v,r)`; ties
broken by the sorted label tuple; the search is exhaustive over all C(39,3) = 9,139 subsets.  It
uses view labels and nothing else: no image, no score, no correspondence, no result.  The rule was
not allowed to look at the two query views, and does not: a rule that placed references near them
would score better here and mean less.

**The chosen views (k = 4):** `000-000`, `045-000`, `066-117`, `066-243`.
Covering radius **65.5 deg**.  `000-000` is pinned so that k = 1 is a subset of k = 4 and the k = 1
arm is exactly Mission 24.

**The reserved views (never references, never stored):** `035-045` and `045-180`.

**The held-out query views, and their distance to the nearest stored view:**

| query view | k = 1: nearest stored, distance | k = 4: nearest stored, distance |
|---|---|---|
| `035-045` | `000-000`, **35.0 deg** | `045-000`, **30.0 deg** |
| `045-180` | `000-000`, **45.0 deg** | `000-000`, **45.0 deg** (unchanged) |

Stated here, before the run, because it constrains what the read can show: **the coverage rule
improves one query view by 5 degrees and leaves the other exactly where it was.**  The mission
asked for a spread rule and this is the best spread available at k = 4; a rule that moved a
reference next to `045-180` would be fitting the reference set to the queries.  So this read tests
whether 5 degrees of extra proximity on one view, plus three extra chances to be the best match on
both, moves the door.  If it does not, that is a statement about how far the grid can be covered
with four views, not an accident of the implementation.

## 2. The design of the read (item 4)

| element | value |
|---|---|
| objects | all 80 ETH-80 objects |
| stored views | k = 1: `000-000`; k = 4: the rule's set (above) |
| query views | `035-045` (the Mission-23 view, so the k = 1 arm is paired with the 0.550 door / 0.250 assembled record) and `045-180` (never read) |
| queries | 2 per object, 160 in total; unit of resampling = the object (80) |
| shortlist | S = 8 **objects** in both arms -- the same S, so chance is unchanged: recall@8 = 8/80 = **0.10**, top-1 = 1/80 = **0.0125** |
| primary 1 | **door recall@8**, k = 4 minus k = 1, paired per query, object-clustered bootstrap, 10,000 resamples, seed 20261118, alpha 0.05 two-sided |
| primary 2 | **assembled top-1**, k = 4 minus k = 1, same test |
| multiplicity | **Holm on the two primaries** |
| matcher cost | 8 pairs per query per arm -- each shortlisted object is scored on the ONE stored view its door chose, so more views do not buy more matcher work |
| frozen | everything else: the bundle's weight (0.375), n_hits (8), k (60), S (8), rerank (plain). Nothing is re-fitted; the only thing that changes between arms is the stored view set |
| reported, not verdicted | per view, per class, the number of queries whose truth is in the shortlist (the ceiling), which view won for each object, seconds per query per stage |

Secondary, reported and not verdicted: how often a non-`000-000` view wins, since that is the
mechanism the design assumes.

## 3. Power and MDE, stated before the run

The test is a paired bootstrap on the per-object hit-rate difference, 80 objects, 2 queries each.
The simulation is `Topology`'s registered power model (mission 23, `placement_m23_power.py`, now in
`archive/`), run at `per_obj = 2` with seed 20261118:

| clustering | MDE at 80 % power | power at the 0.20 planning effect |
|---|---|---|
| w = 0 (queries independent) | **0.131** | 0.97 |
| w = 0.5 | **0.127** | 0.99 |
| w = 1 (object dominates) | **0.113** | 1.00 |

**Planning effect: 0.20** -- the gap from the door's measured 0.55 recall@8 to the ceiling the
mission names (at least 0.75).  For the assembled arm the same 0.20 is used: top-1 was 0.250, and
a rise in recall is what can lift it.  **MDE (0.113-0.131) < planning effect (0.20) at every
clustering: the read runs.**  It is judged only on effects at least as large as the MDE; a smaller
estimate is reported with its interval and gets no verdict.

## 4. The decision rule for shipping (item 6), fixed now

* k = 4 becomes the bundle's default **iff** both primaries' 95 % intervals exclude zero after
  Holm, and the door's recall@8 rises by at least the 0.20 planning effect.
* Otherwise k = 1 stays the default, the measured numbers are recorded in the bundle and the
  README, and the report says in one paragraph what the per-query rows show about why the door
  misses -- with the view distances above as the first candidate explanation and the data as the
  arbiter.

## 5. Cost (measured on this box, stated so the read's budget is auditable)

Graph build ~14 s per image on one core.  The k = 4 gallery is 320 images; 160 of them (the
`000-000` refs and the `035-045` queries) are already in the graph cache from Missions 23-24, so
~320 new builds at 2 workers ≈ 40 min.  Matcher pairs: 8 per query per arm, ~19 s wall at 2
workers -- the k = 1 arm on `035-045` is fully cached (Mission 23's pairs, imported by digest), so
the read scores at most 160 x 8 + 80 x 8 = ~1,900 pairs ≈ 5 h.  Every row is written as it is
produced and every pair score is cached, so a stopped read is a smaller read, never a lost one.

## 6. Amendment -- the budget rule, recorded before any k = 4 row exists

The read as launched costs ~90 s per query end to end, so the full 320-query design is of order
five hours of matcher time on this box.  Rows already written: the whole k = 1 arm.  Rows not yet
written: the k = 4 arm, which is the point of the mission.

Nothing about the design changes, and no query is chosen by its result.  What is fixed here is the
order in which the k = 4 arm is filled and what a partial read reports:

1. the arms run arm-major, view-major, object-major: k = 1 (`035-045`, then `045-180`), then k = 4
   (`035-045`, then `045-180`), objects in sort order inside each cell;
2. a cell is REPORTED only when every object in it has a row; a partially filled cell is reported
   as a partial cell with its n, never merged into a complete one;
3. the two primaries are computed on the objects that have BOTH arms in BOTH views, so a partial
   k = 4 arm shrinks n and is reported with the MDE that n implies (`per_obj = 2`, the mission-23
   simulation: n = 80 -> 0.113-0.131, n = 60 -> 0.146, n = 40 -> 0.17);
4. the PRIMARIES are computed over complete cells: a cell is (arm, view); the primary pairs the two
   arms inside each complete cell, and then pools the complete cells object by object (so a run that
   finishes `035-045` and not `045-180` gives one query per object, `per_obj = 1`, and says so);
   the MDE printed with the primary is the mission-23 simulation re-run at the realised (n, per_obj);
5. if the k = 4 arm has no complete cell at all, the report says the read was underpowered by
   budget, gives the k = 1 numbers and makes no verdict -- the same rule as Mission 23's MDE gate.

The 8 pairs per query per arm are cached by (matcher digest, query, view), so any cell that is run
once is free to re-read, and stopping costs only the cell that was interrupted.
