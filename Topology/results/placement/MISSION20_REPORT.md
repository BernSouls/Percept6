# MISSION 20 — from correspondence to recognition

Branch `arena/c316ca3a-percept6`, from the M19 tip. Every new behaviour sits behind a flag
that is OFF by default; the defaults are proved byte-identical, not asserted. All builds and
all fits in this mission are on **FIT** only. The registered read is Mission 21.

---

## Item 1 — port the M19 design into the deployed matcher, and prove parity

**What changed.** The M19 design (configuration descriptor, w = 0.25, K = 24, whole row; the
M18 LLR table; the assignment start; candidate cap 24) existed only in
`Topology/placement_m17_eval.py`. It is now ONE switch bundle, `PLACEMENT_M19` (env, default
off), whose switches are DERIVED in `parameters/_text.py` rather than written a second time,
so `m15_manifest.effective_flags()` cannot disagree with what the matcher uses.

`matching/placement_matcher.rescore_matrix` now calls `apply_node_descriptor` **before** the
candidate cap and the context — the same order the harness uses — and records
`stats["effective"]` from `effective_flags()` **at call time**. M19 wrote the process
defaults instead, so a result row could carry `desc: (24, 0.25)` beside a manifest saying
"descriptor off". That is the stale-docstring/manifest defect, fixed at the source.

**The cap was the whole difference, and finding it took one wrong run.** The first parity
run reproduced **0.509**, not 0.569. That is exactly the M19 *cap-6* row: `rescore_matrix`
read the frozen `cand_cap` of 6 directly and ignored `PLACEMENT_CAND_CAP`, while the record
was measured at cap 24 ("plain cap 24 + descriptor"). The scorer now reads `cand_cap()`,
which returns the frozen value at defaults, and the bundle sets 24. Nothing about the
descriptor was wrong.

**Parity (the four DEV images @ 15°, n = 1166 queries, `m20_parity.json`).**

| image | q | harness top-1 | deployed top-1 | disagreements | cells compared | max abs score diff | geo AUC |
|---|---|---|---|---|---|---|---|
| syn\|m11ir_00 | 156 | 0.545 (85) | 0.545 (85) | 0 | 26,068 | 0.0 | 0.812 |
| syn\|m11ir_01 | 170 | 0.535 (91) | 0.535 (91) | 0 | 23,419 | 0.0 | 0.802 |
| real\|cell | 184 | 0.788 (145) | 0.788 (145) | 0 | 21,168 | 0.0 | 0.897 |
| real\|grass | 656 | 0.521 (342) | 0.521 (342) | 0 | 343,350 | 0.0 | 0.793 |
| **pooled** | **1166** | **0.569** | **0.569** | **0 / 1166** | **414,005** | **0.0** | — |

The maximum per-query difference is zero: not "within tolerance", the same decision and the
same score on every cell. The pooled geo AUC is **0.800 over pairs** and **0.826 as the mean
of the four cell AUCs** — the number M19 printed. Both are stated because they are different
pools; the per-image values are identical to the record either way.

**Defaults byte-identical.** `Topology/placement_m12_perf.py --check --workers 1`: 28 pairs
(14 synthetic, 14 real), `identical: True (0 diffs)`. Wall 820.9 s against the 633.8 s
baseline of the M19 close-out; the difference is contention with two other runs on the same
two-core box, not work.

**The test that would catch the old defect.** `tests/test_m20_bundle.py` runs a child
process with `PLACEMENT_M19=1` and fails if the manifest reports the descriptor off, or if
w / K / cap / band / table / assignment disagree with the bundle. It also pins that the
deployed call site's own `stats["effective"]` agrees with the manifest. With the bundle off
the same test pins the deployed table and no descriptor. Suite: **101 tests OK**.

---

## Item 2 — the corpus, the split, the tasks

**Corpus.** `Topology/fetch_corpus.py --fetch` fetched the `chenchkx/ETH-80` mirror and
verified **8 classes × 10 instances × 41 views = 3,280 images** (plus 3,280 masks), each
256×256, one image and one mask per view, 41 view keys present in all 80 objects, every
image readable. Kept in `.cache/` (outside git) behind `data/ETH-80-master`.

Two real defects were found in the fetcher and fixed, with the evidence in the file:

* `extractall` sat one indent too deep **inside** the member loop, so it ran once per
  member: 6,560 extractions of a 322 MB archive. Measured before it was stopped: more than
  an hour of CPU, one class of eight on disk. The download was never the problem.
* `verify()` asserted a layout this mirror does not have — a `maps/` subdirectory and two
  hard-coded view keys, one of which (`-035-135`) exists only by accident. It would have
  rejected a good corpus. It now checks the invariants the work needs.

**Split** (`Topology/placement_m20_splits.py` → `m20_splits.json`). Unit is the OBJECT; per
class the ten instances are ordered by `sha256("M20-split-v1-2026-11-18|class|instance")` and
dealt 5 / 3 / 2 to FIT / DEV / FRESH. Disjoint by object across splits, every class in every
split, class set a parameter, views not split. `--check` re-derives the file and diffs.

| split | objects | views | classes |
|---|---|---|---|
| FIT | 40 | 1,640 | 8 |
| DEV | 24 | 984 | 8 |
| FRESH | 16 | 656 | 8 |

The file also records each object's FILENAME instance number beside its directory, because
the two disagree in this corpus (`car` directory 4 holds `car11`).

**Tasks** are fixed in `M20_TASKS_PREREGISTRATION.md`, written before any ETH-80 graph
existed: T0 (same image, 15° — the aggregation control), T1 (same object, other view), T2
(other instance, same class — class-level truth; no place truth exists and none is
reported), T3 (same object, other view, mixed gallery). View keys `-000-000` (reference) and
`-022-000` (query).

---

## Item 3 — the staged matcher and its cost

**Stage 1** (`matching/placement_prefilter.py`) is a per-image bag of configuration
descriptors — sorted log neighbour distances and (cos, sin) of the source-frame directions of
each place — compared by "mean over query places of the best configuration similarity to any
reference place". No mapping, no context iteration, no LLR table, no pixels of the other
image, no truth.

* Cost: **13.2 ms per pair** (FIT, 256 pairs in 3.4 s after the bags exist). Building the
  bags is the one-off: 80 images at the M16 recipe, 505 s, cached.
* **Recall@S on FIT, reported with every shortlist number** (40-item gallery): T1 0.30 /
  0.95 / 1.00 at S = 1 / 4 / 8; T2 0.10 / 0.125 / 0.40 / 0.975 at S = 1 / 2 / 8 / 24; T3
  0.025 / 0.175 / 0.25 / 0.75 at S = 1 / 4 / 8 / 24.
* On the **16-item gallery chosen in item 6**: T1 1.00 at S = 8 (0.62 at S = 1, 1.00 at
  S = 2), T2 0.62, T3 0.75. **S = 8 chosen on that recall**, not on taste.

**Stage 2** is the shipped pair scorer on the shortlist: **14.5 s per pair** on these graphs
(1,625 nodes, two-worker mean; the M11 figure of 14 s/pair was the old front end at 877
nodes, so the cost has roughly doubled with the M16 recipe). At S = 8 that is
**201.9 s per query**, measured, not estimated.

**Does the shortlist work where M10 failed?** Partly, and for a stated reason: within a class
the geometry bag recovers **1.00 recall@8**, where M10's appearance shortlist reached 0.78
within class; but at instance level in a mixed gallery it is **0.25 at S = 8** — M10's
full-gallery appearance figure was 0.235. Geometry does the *within-class* job appearance
could not; it does not do the mixed-gallery job either.

---

## Item 4 — the four arms, the null, and the coverage control

FIT, gallery 16 (2 objects per class), S = 8, queries 16 (2 per class), view pair
`-000-000` → `-022-000`. T1 has n = 14: two queries had fewer than two same-class
candidates in the shortlist and are not counted in it.

| arm | T1 rank-1 / MRR | T2 rank-1 / MRR | T3 rank-1 / MRR | null AUC | Spearman(score, mapped) |
|---|---|---|---|---|---|
| S0 shipped | 0.786 / 0.821 | 0.062 / 0.276 | 0.625 / 0.688 | 0.955 | **0.518** |
| S1 shipped + E (w 0.375) | 0.786 / 0.821 | 0.062 / 0.271 | 0.625 / 0.688 | 0.958 | 0.483 |
| S2 shipped + λ·S3 (λ = 1.0) | 0.786 / 0.821 | 0.062 / 0.286 | **0.688 / 0.719** | 0.952 | 0.325 |
| S3 aggregate alone | 0.643 / 0.750 | 0.062 / 0.226 | 0.375 / 0.487 | 0.719 | **−0.007** |

* **The null** is the truth pair against all impostor pairs of the same query (12 positives
  against 116 negatives on FIT): every arm in the shipped family separates them at AUC ≈ 0.95,
  so the internal ranking is calibrated; the aggregate alone is not (0.72).
* **Coverage neutrality, measured two ways.** Rank correlation between an arm's score and
  the number of mapped places: **0.518 shipped → 0.325 with the new aggregate added → −0.007
  for the aggregate alone.** The new denominator does what it was designed to do — the count
  of mapped places stops driving the score. Whether it *replaces* it with quality is the
  question the FIT numbers answer: on T3, S2 gains one image (0.625 → 0.688) and on T1 and T2
  nothing, which is inside the MDE (item 7).
* **The coverage-matched control** (keep only candidate pairs whose mapped count is within
  ±5 % of the median, 1,281–1,416 places): every arm collapses to the **same 0.133 (2 of 15
  queries)**. Read carefully: inside the band the arms are indistinguishable, but the band
  also removes the true partner from most queries, so this control shows that the four arms
  **do not differ once coverage is equalised** — it does not show that the shipped score's
  FIT advantage is *caused* by mapped count.
* **The operationalisation is checked, not assumed.** The new denominator counts query places
  with a graded incident relation. On 6 of 128 pairs the evidence term judged *more* pairs
  than that denominator (395–490 judged against 356–453 places), so the denominator is not a
  strict upper bound; that is reported rather than smoothed over, and it is why S3 and E land
  numerically close (they differ only in which side of the ratio moves).

---

## Item 5 — unseen regimes: where it breaks

`Topology/placement_m20_oop.py`, 20 renderer scenes × 4 views, the deployed configuration
(bundle on), truth from `track_optimal`. **20° is one of Mission 10's registered views; 35°
is new and no registration covers it — those rows are probe-only.** Fit rows and the item-1
DEV record are never pooled with these.

| view | cells | queries | node accuracy | ambiguity bank | off bank | geo AUC |
|---|---|---|---|---|---|---|
| y20 (20° yaw, registered) | 20 | 3,765 | 0.208 | 0.190 | 0.278 | 0.574 |
| x20 (20° pitch, registered) | 20 | 4,989 | 0.332 | 0.334 | 0.365 | 0.668 |
| **y35** (new, probe) | 20 | 2,999 | **0.126** | 0.109 | 0.193 | 0.537 |
| **x35** (new, probe) | 20 | 4,017 | 0.242 | 0.239 | 0.279 | 0.619 |

Per-shape spread (mean over the 20 scenes, range across scenes in brackets from the JSON):
at y35 the accuracy runs 0.025–0.259 and the geometry term is at **0.537**, i.e. barely above
chance. **Operating range, in one sentence:** the shipped configuration holds in plane at 15°
(0.569, item 1) and survives a 20° pitch (0.33) but not 20° yaw (0.21) or 35° either way
(0.13–0.24), so its operating range is roughly **±20° of pitch and under 20° of yaw**, and
yaw degrades roughly twice as fast as pitch.

---

## Item 6 — headroom

Chosen on FIT: **gallery = 2 objects per class, 16 items**, all eight classes (no confusable
subset was needed), S = 8. Shipped (S0) FIT top-1 per task:

| task | S0 top-1 (n) | in [0.30, 0.85]? |
|---|---|---|
| T0 (same image, 15° — the aggregation control) | **1.000 (16)** | above the band, by design: identical content |
| T1 (same object, other view, same-class gallery) | 0.786 (14) | **yes** |
| T2 (other instance, same class) | 0.062 (16) | **no — exactly chance (1/16)** |
| T3 (same object, other view, mixed gallery) | 0.625 (16) | **yes** |

T2 is the benchmark's floor and the mission's most useful negative: the shipped score ranks
another instance of the same class **exactly as often as chance** (1 in 16), i.e. the
placement evidence carries no class-level appearance transfer between differently shaped
instances. No enlargement was applied to T1/T3 — both are inside the band; enlarging T2's
gallery would only move its chance level, not its information, so it is left as registered.

---

## Item 7 — the fit and the registration

`matching/placement_m20_weights.json`: S3 = num/(den+8), S2 = shipped + λ·S3, **λ = 1.0**,
fitted on FIT, provenance recorded (tag, gallery, S, objective, grid, the live w of the
existing evidence term, the fact that the grid is flat within one query). The optimum is
interior (0.491 at λ = 0 → 0.533 at 1.0 → 0.467 at 4.0), so the boundary rule does not fire
and no boundary value is shipped.

`MISSION21_PREREGISTRATION.md` is committed alone, before any FRESH graph exists. It fixes
the scorer, front end, views, gallery, S (with recall), λ, the four primaries (T2 top-1, T2
MRR, T1 top-1, T1 MRR as S2 − S0), image-clustered bootstrap CI over object clusters, Holm
across the four, the **MDE stated up front (0.175 at n = 16, ≈0.10 at n = 48) against a
planning effect of +0.0625 — below the MDE at both sizes**, so the read may not verdict the
aggregate, and the decision rules including "T0 ≤ T3 rejects the aggregate regardless".

---

## Item 6b — DEV, for the record (dev-only, not a registered read)

DEV, T3, gallery 16, S = 8, n = 16 queries (2 per class, none of them in FIT):

| arm | rank-1 | MRR | recall@8 |
|---|---|---|---|
| S0 shipped | 0.625 | 0.646 | 0.688 |
| S1 shipped + E | 0.562 | 0.615 | 0.688 |
| S2 shipped + λ·S3 | 0.625 | 0.646 | 0.688 |
| S3 aggregate alone | 0.188 | 0.310 | 0.688 |

The FIT ordering survives the split change (S2 ≥ S0 > S1 > S3) but the S2 − S0 gain does
**not** reproduce: it is +0.0625 on FIT and 0.000 on DEV. That is exactly the MDE problem
item 7 registered, met before the registered read rather than after it.

---

## Item 8 — failure attribution

`Topology/placement_m20_attrib.py` tags every failed query of the shipped arm (S0) with one
disjoint tag, in a fixed order: **SHORTLIST** (the positive was never scored), then
**CORRESPONDENCE** (the true pair mapped or judged fewer than 80 % of the winner's places),
then **APPEARANCE** (comparable places, lower shipped score), then **AGGREGATION** (the true
pair has the *higher* shipped score but lost the composite). Counts are per task and per
class, and never pooled across splits.

FIT, T3, gallery 16, S = 8, n = 16 queries (10 correct, 6 failures):

| tag | count | where |
|---|---|---|
| SHORTLIST | 4 | pear ×2, dog ×1, cow ×1 |
| APPEARANCE | 2 | cow ×1, cup ×1 |
| CORRESPONDENCE | 0 | — |
| AGGREGATION | 0 | — |

FIT, T0 (the aggregation control), n = 16: **zero failures of any kind** — with identical
content the shipped score ranks the truth first every time, so nothing in the failure
inventory above can be blamed on the score's arithmetic.

**What that says.** On FIT the benchmark is limited by **candidate generation, not by
scoring**: two thirds of the failures are queries whose positive the stage-1 bag never
surfaced, and the remaining third are pairs the placement evidence simply did not favour
(other instance-like appearance inside a class). There are no aggregation failures to fix,
which is also why the new aggregate can only move one image.

DEV, T3, same configuration (n = 16 queries, 2 per class, 10 correct, 6 failures):

| tag | count | where |
|---|---|---|
| SHORTLIST | 5 | cow, dog, horse, pear, tomato |
| APPEARANCE | 1 | dog |
| CORRESPONDENCE | 0 | — |
| AGGREGATION | 0 | — |

DEV agrees with FIT in kind and differs in mix: **the shortlist again dominates** (5 of 6
against 4 of 6), and there is still **not one aggregation failure** on either split. The
DEV failure list is spread across five classes instead of two, which is what a second split
should do to a FIT-tuned filter.


---

## What changed in the repository

| file | what it is |
|---|---|
| `parameters/_text.py` | `PLACEMENT_M19` bundle (+`_W`, `_K`, cap), derived switches, `PLACEMENT_PREFILTER` |
| `matching/placement_matcher.py` | descriptor applied in `rescore_matrix`; `cand_cap()`; `stats["effective"]` from `effective_flags()` |
| `eth80_frontend.py` | the three bundle switches in the manifest list |
| `matching/placement_prefilter.py` | stage-1 bag score (no mapping) and its shortlist/recall helpers |
| `matching/placement_m20_weights.json` | S3 definition and λ = 1.0, with provenance |
| `Topology/placement_m20_parity.py` | deployed vs harness, per query |
| `Topology/placement_m20_splits.py` | the object-disjoint FIT/DEV/FRESH split, `--check` |
| `Topology/placement_m20_eth80.py` | bags → shortlist → pairs → arms on ETH-80 |
| `Topology/placement_m20_fit.py` | arms, null, coverage control, λ sweep on FIT |
| `Topology/placement_m20_oop.py` | 20°/35° out-of-plane probe |
| `Topology/placement_m20_attrib.py` | failure tags |
| `Topology/fetch_corpus.py` | two fetch defects fixed (extraction loop, layout check) |
| `tests/test_m20_bundle.py` | the manifest-cannot-lie test |

## Shipped baseline per task (FIT / DEV, gallery 16, S = 8)

| task | FIT shipped top-1 | DEV shipped top-1 | shortlist recall@8 |
|---|---|---|---|
| T0 same image, 15° (control) | **1.000** (16) | — | 1.000 |
| T1 same object, other view | **0.786** (14) | — | 1.000 |
| T2 other instance, same class | **0.062** (16) | — | 0.62 |
| T3 same object, other view, mixed gallery | **0.625** (16) | 0.625 (16) | 0.75 / 0.688 |

Out of plane (renderer probe, deployed configuration): node accuracy 0.332 at 20° pitch,
0.208 at 20° yaw, 0.242 at 35° pitch, 0.126 at 35° yaw.

## Open problems, ranked

1. **Candidate generation is the whole benchmark.** Two thirds of every failure on both
   splits is a query whose positive the stage-1 bag never surfaced (recall@8 0.75 FIT /
   0.688 DEV, and 0.25 in a 40-item gallery). Nothing downstream — no aggregate, no weight
   — can recover a pair that was never scored.
2. **T2 is at chance.** The shipped score ranks another instance of the same class exactly
   as often as chance (1/16 on FIT). There is no class-level signal in the placement
   evidence across differently shaped instances; this is the recognition problem the
   mission title names, and it is untouched.
3. **No place truth for real image pairs.** ETH-80 has no keypoint annotations, so T1–T3 are
   image-level only; node-level claims stay on the renderer scenes, where truth is exact.
4. **The aggregate is a one-image effect.** λ is flat within one query, S2 == S0 on DEV, and
   the MDE at n = 16 is 0.175 against a planning effect of 0.0625.
5. **Yaw is twice as destructive as pitch** (0.208 vs 0.332 at 20°), and the geo term is at
   0.537 by 35° yaw — the failure mode is geometric, not appearance.

## The single next change

**Fix candidate generation, not the aggregate.** The measured facts point one way: 9 of 12
failures across FIT and DEV are shortlist misses, the aggregate has no aggregation failures
to repair, and λ is flat. The next change is a stage-1 filter that unions the configuration
bag with a cheap appearance channel and a class-level part descriptor, chosen on FIT by
recall@8 on the 40-item gallery (where geometry alone reaches 0.25), with the S = 8 budget
held fixed so the comparison is like-for-like.
