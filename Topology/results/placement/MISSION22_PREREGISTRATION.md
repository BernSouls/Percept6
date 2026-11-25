# Mission 22 -- pre-registration of the one evaluation (written before any FRESH graph exists)

Written 2026-11-18 on branch `arena/c316ca3a-percept6` at commit `7a86284` (Mission 21 tip),
before any FRESH image has been read, keyed, pooled or scored by any Mission-22 code.  The
only Mission-22 results that existed when this file was written are `m22_audit.json` (read-only
frame/content audit, FIT, 3 images), `m22_placeproof.json` (FIT, 3 objects), `m22_fit.json` and
`m22_dev.json` (front doors on FIT and DEV) and `m22_power.json` (the arithmetic below).

## 1. Splits, unchanged

`m20_splits.json`, salt `M20-split-v1-2026-11-18`, rule `sha256(salt|class|instance)`, 8 classes,
5 / 3 / 2 instances per class: **FIT 40, DEV 24, FRESH 16 objects**.  Eight classes, so FRESH is
2 instances per class.

Every read is one query view per object (`-022-000`) against the split's reference views
(`-000-000`).  This is the Mission-20/21 convention, so every number here is comparable with the
record.

## 2. The arms, fixed here

| arm | what it is |
|---|---|
| appearance-only | the M21 colour + gradient-histogram channel, cosine, fixed bins, no fitting |
| M19 shipped | the deployed matcher scored against **every** gallery item (no front door) |
| assembled (B) | appearance shortlist S = 8 -> the shipped matcher re-orders the 8, dropping nothing |
| assembled + door (A) | shortlist S = 8 taken from the graph-native place-voting door, then the same rerank |
| class score (C) | per class, the sum of the top two instance matcher scores (arms `max` and `mean`, fixed) |

Constants fixed in advance, not tuned: shortlist S = 8; RRF k = 60 for the combined door;
`n_hits` = 3 pool places per query place (measured plateau 2-32, reported); histogram bins
translation 1.0 x the query's median place spacing, scale log2 0.5, rotation pi/6; object score
`votes_max / sqrt(n_query_places x n_object_places)`; class aggregation `top2` primary, `max` and
`mean` as fixed arms.

## 3. The primaries and the decision rule

Primaries, paired on objects, clustered by object (one observation per object, so the cluster
*is* the object):

1. **T3 top-1, assembled vs appearance-only** on the FRESH gallery.
2. **T1 top-1, assembled vs appearance-only** on the FRESH same-class gallery.

Test: exact McNemar (sign) on the discordant objects, Holm over the two primaries.  Secondary,
against the exact chance and not tested: T2 class top-1 for each `agg` arm, against the
hypergeometric chance (a random top-1 gallery item is of the query's class with probability
k/N, k = gallery items of that class).  Ship rule: the front door with the best FIT recall@8;
DEV only reports.

## 4. n, and the gate -- the read is NOT run

The planning effect is the front door's, already measured on FIT and DEV in Mission 21: appearance
recall@8 **0.675** on FIT against geometry's **0.175**, a difference of **0.50** absolute.

The power arithmetic is in `Topology/placement_m22_power.py`, and its output in `m22_power.json`.
A paired binary read on objects is exact McNemar over the discordant objects, so for a difference
`delta` and a discordance rate `pi_d` the power is `P(X >= k | n_d, (1 + delta/pi_d)/2)` with `k`
the two-sided exact critical value.  Its most favourable case (every object discordant,
`pi_d = 1`) gives, at alpha = 0.05:

| objects | MDE at 80 % power (pi_d = 1) | MDE at pi_d = 0.6 |
|---|---|---|
| **16 (FRESH)** | **0.706** | **0.833** |
| 24 (DEV) | 0.594 | 0.777 |
| 40 (FIT) | 0.443 | 0.594 |
| 60 | 0.383 | 0.483 |
| 80 | 0.326 | 0.422 |

**MDE(FRESH, 16 objects) = 0.706 in the most favourable case, 0.833 in the pessimistic one.
The planning effect is 0.50.  0.706 > 0.50, so the gate fails: the FRESH read is not run.**
Holm over two primaries only widens that gap.

The effect would need **at least 36 objects** (pi_d = 1) or **55 objects** (pi_d = 0.6) before a
0.50 difference could be detected at 80 % power, and the assembled-vs-appearance effect measured
on FIT is smaller than 0.50, so the true requirement is larger still.  ETH-80 has 80 instances in
total and all 80 are already allocated 40 / 24 / 16; there is no spare object, and re-splitting
would be a post-hoc change to a split that Mission 20 fixed and Mission 21 read.  The fix is
therefore out of this mission's scope, and the mission's own rule applies: a read whose MDE
exceeds its planning effect is not run.

**Consequence, stated plainly now so it cannot be softened later:** Mission 22 has **no
confirmatory read**.  Every number in `MISSION22_REPORT.md` is measured on FIT and DEV, the two
splits the mechanism was built and chosen on, and is labelled exploratory.  The FRESH split stays
untouched and unread, so a future mission with a bigger held-out set (or a pruned question with a
larger expected effect) inherits it intact.

## 5. What will NOT be done under this registration

- No FRESH graph, key, pool or score is built, so no FRESH image is read at all.
- No re-tuning of sigma, K or lambda (Mission 21: the surface is flat), no new bin widths chosen
  by their score, no threshold picked to make a rate look better.
- No arm added after seeing a result, no test swapped after seeing a p-value.
- The class score may not read the query's label: `class_scores` receives instance scores and the
  gallery's class table only, and `tests/test_m22_bundle.py` rebuilds every class score with the
  query's class deleted from that table and requires identical numbers.

## 6. Non-circularity of (C), restated as a test

`matching/placement_classscore.py::class_scores(instance_scores, class_of, agg)` has no argument
that can carry the query's class, and the label is read once, in `class_top1`, at the reporting
boundary.  The test deletes the query's class from `class_of` and re-scores: the per-class numbers
must not move.  If that test ever fails, (C) is void and its number is withdrawn.
