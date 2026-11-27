# MISSION 27 — pre-registration

Written before any Mission 27 number was produced. Everything here is fixed now: the S rule, the
door thresholds, the estimator, the seed and the reporting cells. Deviations, if any, are stated in
the report with the reason.

Branch `arena/c316ca3a-percept6` from `e9d55d4` (the M26 ship). Core code in `matching/`, thin
scripts in `Topology/`. No new descriptors, no new mechanisms, no yaw work, no class score, no
transform votes, no extra reference views.

---

## 1. Make the environment cheap

**1a. The pair cache must key on the file, not the path.** `load_pair_cache` adopts the committed
pair files (`m23_read_pairs.jsonl`, `m22_assemble_*.jsonl`) under the CURRENT matcher digest, keyed
by basename, and every row it writes is basename-keyed too. Proof: three committed pairs recomputed
from the shipped matcher must come back identical (`m27_profile.py --proof`). Test: a cached pair
must remain a hit when the path prefix changes, and a row written with a foreign prefix must be
adopted (`tests/test_m27_cache.py`). Reported: how many of the 992 M26 pairs are cached now.

**1b. The records must be committed in a form the door can read.** `matching/placement_records.py`
exports the 320 cached records (80 objects × 4 views) to a compressed float32 npz; `Topology/
m26_records.load` falls back to it, so a sandbox with no pickles can seed a gallery. `std_all` is
not stored (no door reads it; a consumer that needs it rebuilds). Committed only if the file is
under 40 MB. Proof: the door's shortlist from the npz must equal the door's shortlist from the live
records, 80/80 (`m27_export_records.py --check`).

**1c. The default worker count is the physical core count**, `min(os.cpu_count(), 4)`, and every
unique graph is warmed once before a pair job. `m27_profile.py --profile` separates graph build
from matching at 1, 2 and 4 workers; the numbers go in the report.

**1d. `scripts/bootstrap_cache.py` restores a fresh sandbox and reports its own seconds.**

## 2. Settle S with the full read

Cell: all 80 objects, query view `-035-045`, 80-object gallery at `-000-000`. Every query gets a
flat S = 16 shortlist, so S = 8, S = 16 and the adaptive rule are read off the SAME pair scores.
A query is reported only when all 16 of its candidate pairs are scored; unscored queries are listed,
never zeroed.

**Rule, fixed now.** S is chosen on the FIT objects (40); the DEV + FRESH objects (40) report.
Flat S = 16 ships **only if** the FIT top-1 gain over S = 8 is **≥ +0.05** AND the confirmation gain
is **≥ 0**. Otherwise S = 8. The adaptive rule (M26: margin `(s8 − s9)/s1` < 0.007168 → 16) ships
**only if** it beats **both** flat arms on FIT; otherwise it leaves the bundle default (code stays
behind the flag) and the report says its margin predictor was poor.

**Estimator.** Paired, object-clustered bootstrap: 10,000 resamples of the 80 objects (FIT and
DEV + FRESH reported separately), seed 20261118, 95 % percentile interval, two-sided bootstrap
p = share of resamples whose |mean| ≥ |observed|. Chance beside every rate: top-1 = 1/80 = 0.0125,
recall@8 = 0.100, recall@16 = 0.200. Nothing is compared against a constant.

## 3. Where the matcher loses

No new scoring; item 2's pair scores only. For every reported query whose truth is in its S = 16
shortlist: whether the winning candidate is the **same class** as the truth or a **different class**,
the score gap (truth − winner), the number of same-class candidates the shortlist held, and the
per-class split. Reported as counts with n, not as a rate on a subset it was selected from.

## 4. Two cheap door variants — the last door work

Door only, cached cells, same rule and cells as M26: **FIT** recall@8 at `-035-045` and `-045-180`
must beat the shipped door by **≥ 0.075**, and **DEV + FRESH** must not be below it by more than
**0.025**. The shipped door is c = RRF(counted, appearance), the bundle's door; its numbers come from
the M26 oracle (n = 40 per cell, chance 0.100) and are reproduced from scratch in this mission, which
is itself a check that the cached records and the cached pairs still give the shipped answer:

| cell | shipped door c |
|---|---|
| FIT `035-045` | 0.550 (22/40) |
| FIT `045-180` | 0.550 (22/40) |
| DEV+FRESH `035-045` | 0.550 (22/40) |
| DEV+FRESH `045-180` | 0.475 (19/40) |

so the pass line is **≥ 0.625** on both FIT cells and **≥ 0.525 / ≥ 0.450** on the confirmation
cells.

**(a) Graded kernel in its registered NARROW constitution** (`core/keypoint_classes.VARIANTS
["narrow"]`: theta range 3.0 bins ⇒ sigma = 1 bin; weights unchanged), at **tau = 0.5** and
**tau = 0.7**. M26 only ever measured the WIDE constitution (theta sigma = 4 bins).

**(b) Relaxed partition**: a pool place counts for a query place when its key is **exact** (weight
1.0) **or its theta bin is ±1, cyclic mod 12** (weight 0.5); counted per object (best weight per
object per query place, summed over query places, normalised exactly as the counted door); then
fused with the shipped counted door by **RRF, k = 60**, the same fusion the shipped shortlist uses.

If neither passes, the **door is FROZEN as the counted door** and the report says no further door
work belongs in this programme.

## 5. The M26 report gets an addendum, not a rewrite

Three corrections: the "paired against the door's own ceiling, p = 0.02" row is **withdrawn** (a
constant 1.000 is not a comparison); the graded-door finding holds for the **wide** constitution
only; the 992 M26 pairs were mostly already cached.

## 6. Ship and freeze

The bundle takes the item 2 result (and the item 4 verdict if a variant passes). The README states
the shipped configuration, the measured assembled top-1 at `-035-045` over all 80 objects with its
interval and chance, per-class rates, the operating range (in-plane 15°, pitch 20°, yaw unresolved),
what it does and does not do, and item 3's finding. `tests/` green, `Topology/placement_m12_perf.py
--check` 0 diffs.
