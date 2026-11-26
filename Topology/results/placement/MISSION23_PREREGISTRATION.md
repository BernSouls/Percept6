# Mission 23 -- pre-registration of the one confirmatory read (item 5)

Written 2026-11-18 on `arena/c316ca3a-percept6` at `717d98d`, before any graph, place record,
door score or matcher score for this read was built.  The only numbers in this file are the ones
already in the record (Mission 21 and 22) and the power arithmetic of
`Topology/placement_m23_power.py`, which reads no data.

## 1. The design, fixed here

| element | value |
|---|---|
| objects | **all 80 ETH-80 objects** (8 classes x 10 instances) |
| gallery | one reference view per object, `-000-000`, **80 items** |
| queries | three unseen view pairs per object: `-022-000`, `-035-045`, `-045-180` = **240 queries** |
| unit of testing | the **object** (80), each contributing 3 paired queries |
| arms | assembled (the shipped M23 configuration, item 3) vs deployed (M19 shipped: the M21 AFTER design, frozen ruler prefilter -> top-8 -> the shipped matcher -> re-order) |
| primary | **assembled top-1 vs deployed top-1**, paired per query, object-clustered bootstrap, **10,000 resamples, seed 20261118**, alpha 0.05 two-sided |
| planning effect | **0.375** (assembled 0.550 vs deployed 0.175 on the 40-item FIT gallery, n = 40) |

**Labelled honestly, before the fact:** object-disjointness is gone.  FIT and DEV objects chose the
front door, the fusion and the class aggregation, and they are in this read.  What is genuinely
unseen is the **view pairs**: no earlier mission ever read `-035-045` or `-045-180`.  So this is a
**view-held-out** read, not an object-held-out one, and it is reported under that name.

Nothing is re-fitted.  The frozen configuration is whatever item 3 ships, at the constants
Mission 22 fixed (shortlist S = 8, RRF k = 60, vote hits 3, counted-vote normalisation, bins as
declared in `matching/placement_frontdoor.py`).

## 2. Power: stated before the read, and it passes

`Topology/placement_m23_power.py` simulates the registered test under a stated model: each object
has a shared difficulty `u_o ~ N(0, 1)` mixed with per-query noise by weight `w` (w = 0 independent
queries, w = 1 the object dominates), the deployed arm's top-1 is 0.175 and the assembled arm's is
`0.175 + delta`; the test is the paired bootstrap over the 80 objects.

| clustering | MDE at 80 % power | power at the planning effect 0.375 |
|---|---|---|
| w = 0 (independent queries) | **0.108** | 1.00 |
| w = 0.5 | **0.103** | 1.00 |
| w = 1 (object dominates) | **0.091** | 1.00 |

**MDE (0.09-0.11) << planning effect (0.375): the read runs.**  This is a different arithmetic from
Mission 22's, and deliberately so: there the unit was one binary query per object and the test an
exact sign test (MDE 0.706 at n = 16); here each object contributes three queries and the test is a
bootstrap on the per-object hit-rate difference, so the same 80 objects resolve a difference six
times smaller.

## 3. Secondary readings -- reported, never verdicted

Against their exact chance, with n and chance beside each: assembled vs appearance-only, the
door's recall@8, the class top-1 of item 1 (measured on the reduced gallery, `k/(N-1)` chance),
per-class spread across the 8 classes, per-image rank.

**Explicitly not attempted: assembled vs appearance.**  Their difference is ~0.175, which at this
design needs roughly 250 objects; reporting a p-value for it here would be the error Mission 22
refused to make.  It is reported as two rates with their CIs and no verdict.

## 4. What is NOT allowed after this point

- No arm added, no test swapped, no constant changed after a score exists.
- No query dropped because of its result.  The only pre-registered reduction is in §5.
- No re-quoting of a retired instrument (M20's AGG tag, the +-5 % coverage filter).
- If the primary comes out flat, that is the answer; the mechanism does not get a second read on
  the same 240 queries.

## 5. Compute budget, and the only permitted reduction

Each query needs the matcher on the union of two 8-candidate shortlists (~11 pairs).  At the
per-pair cost measured in Mission 22 on this box (13.5 s wall for a `--workers 2` pair), 240
queries is of order 2,300 pairs ~ 9 hours.  The budget rule, fixed **now**, before any score
exists:

1. Run the full 240 queries if the measured cost allows.
2. If it does not, drop views in this fixed order -- `-045-180` first, then `-035-045` -- never an
   object and never a query chosen by its result.  The read then reports its actual n (80 or 160
   queries; 80 objects either way) and its MDE from `m23_power.json`'s `n_curve`.
3. A query whose truth is in neither shortlist is scored 0 for both arms **without** running the
   matcher: top-1 can only be a shortlisted candidate, so both arms are provably wrong there.
   This is exact arithmetic, not an approximation, and the count of such queries is reported.

## 6. Amendment, recorded before any read score existed

Two things were learned while items 1-3 were being built, and both are corrected here rather than
silently in the code.  No read score, no read place record and no read shortlist existed at the
time of writing.

1. **The measured pair cost.**  A shipped-matcher pair costs **30 s of CPU** on this box (two
   probes, one under numpy 1.26 and one under 2.4.6, with byte-identical scores), i.e. ~15 s of
   wall time at two workers.  The full 240-query read is therefore of order **12 hours**, not the
   nine estimated in §5, so the budget rule will be used.
2. **The drop order is corrected.**  §5.2 dropped `-045-180` first, then `-035-045`, which would
   have kept `-022-000` to the end -- the one view Mission 22 already read, contradicting the
   whole point of the read.  The views are now run in the order **`-035-045`, `-045-180`,
   `-022-000`**, so that a partial read is always a *never-read* view and any cut leaves whole
   views, never a query chosen by its result.  Items 1-3 are unaffected.

Nothing else in this registration changes: the statistic, the arms, the paired object-clustered
bootstrap, the 10,000 resamples, the seed 20261118 and the planning effect 0.375 are as written
in §1.

## 7. Amendment 2 -- the read runs at 80 queries, one never-read view.  Recorded before any row
##    was summarised (the rows exist on disk; no score had been read, aggregated or printed)

The measured cost of the read as actually run, once the graph cache was warm: **~14 pairs per
query at 19 s of wall time per pair on two workers, i.e. 2.4 minutes per query**.  That puts 240
queries at ~9.6 hours and 80 queries at ~3.2 hours on this box, inside the session the work is
being done in.

The budget rule of §5 binds: views are dropped in the fixed order `-045-180`, `-035-045`,
`-022-000`, never an object and never a query chosen by its result.  **The read therefore reports
the `-035-045` view for all 80 objects = 80 queries, 80 objects, one query each**, and
`-045-180`/`-022-000` are not read.  Because a partial read keeps whole views, the label is
unchanged and still honest: **view-held-out**, over a view no earlier mission ever read.

The MDE at that size was computed into `m23_power.json` (`per_object_curve`, added for exactly
this case, with the same model, seed and 80 objects, varying only the queries per object):

| queries per object | MDE at 80 % power, w = 0.5 | w = 1 (object dominates) |
|---|---|---|
| 1 (this read) | **0.182** | **0.166** |
| 2 | 0.124 | 0.114 |
| 3 (the registered design) | 0.103 | 0.089 |

**MDE 0.166-0.182 against a planning effect of 0.375: the read still runs**, and power at the
planning effect is 1.00 at every one of these sizes.  What the smaller read costs is precision of
the interval, not the ability to see a difference the size of the one the mission planned for.
If the box frees up, `-045-180` is appended to the same resumable file and the summary recomputed;
both reports state the n actually used.
