# MISSION 12 — report

Branch `arena/c316ca3a-percept6`, based on `192a612`.

This file is written as the mission proceeds. Each step's section states what was
measured, what was decided, and — where the mission's gate is not met — that the
gate is not met.

---

## STEP 0 — the verdict rule (no runs)

### The rule, stated before any M12 read

1. **MDE gates the DESIGN, not the verdict.** Before a read is registered, the
   registration must show the design's minimum detectable effect (MDE, at the
   declared alpha and power) is **≤ the planning effect**. If it is not, the read
   is declared **UNDERPOWERED IN ADVANCE** and its outcome can only ever be
   reported as an estimate, never as a detection.
2. **The verdict is the interval.** A primary is **DETECTED** iff its
   scene-clustered confidence interval **excludes zero in the pre-registered
   direction**. Nothing else — not a p-value at a favourable look, not a point
   estimate — produces DETECTED.
3. **Multiplicity.** Holm across the declared primaries; where several looks are
   declared, the look correction is applied to the alpha of each look (alpha
   cumulative 0.10 across 5 looks ⇒ per-look alpha 0.02 for look 2).
4. **Type M.** If a primary is DETECTED and the detected effect is **below its
   design MDE**, the magnitude is labelled **"possibly inflated"** (Type M). The
   detection stands; the size does not.
5. **No relabelling.** Verdicts issued under an earlier rule stand as issued.
   They are shown beside the new rule's reading of the same numbers, never
   rewritten.

### Old verdicts, side by side with the new rule's reading

| Mission | Primary | Estimate | SE | Interval | p | Design MDE | Old verdict | Reading under the M12 rule |
|---|---|---|---|---|---|---|---|---|
| M7 | P − P_presence (placement term, look 2 of 5) | +0.0094 | 0.0040 | [+0.0014, +0.0173] | 0.0260 | 0.0113 | INCONCLUSIVE | **Not detected.** The look-2 alpha after the cumulative-0.10 look correction is 0.02; p = 0.0260 does not clear it, so the interval is not the registered one at that look. Effect < MDE (0.0094 < 0.0113) — would have been Type M even had it cleared. |
| M9 | OOP primary (placement, C − A) | +0.0203 | 0.0127 | [−0.0046, +0.0451] | 0.17 | 0.0355 | NOT DETECTED | **Not detected.** The interval includes zero. |
| M11 | Synthetic placement (C − A) | +0.156 | 0.060 | [+0.062, +0.281] | — | 0.168 | (reported as a positive result) | **Detected**, and the detected effect (+0.156) is below the design MDE (0.168), so the magnitude is labelled **possibly inflated (Type M)**. |

The M7 and M9 verdicts are **not** relabelled: the row above is the new rule's
reading shown beside them, as the mission requires.

### The superseded M11 ETH-80 job

The M11 ETH-80 g1 read is **stopped and superseded** by this bundle. It is not
running in this sandbox (no process alive at the start of the mission). Its
checkpoint `Topology/results/placement/m11_eth80_progress.jsonl` (202 rows) is
**kept**, and nothing in M12 — no design choice, no constant, no arm — is
selected using it. It is not a result and is not reported as one.

---

## STEP 1 — profile, optimise, prove nothing moved

### Gate status, stated first

| Gate | Required | Measured | Status |
|---|---|---|---|
| output unchanged (flags off **and** on) | bit-identical score, `score_shipped`, evidence, judged count, mapping | **identical on 14/14 pairs × 2 arms × 5 fields** | **MET** |
| speed per pair | **≥3×** | **1.3×** (A 14.32 s → 10.84 s; C 16.12 s → 12.49 s) | **NOT MET** |
| pairs | ≥30 fixed | 14 synthetic (16 synthetic + 14 real is the registered set; the real half needs the ETH-80 corpus, still downloading) | **PARTIAL** |

The speed gate is **not met** and is not being claimed as met. What follows is the
profile that says what the remaining 10.8 s consists of and what a 3× would
require.

### Harness

`Topology/placement_m12_perf.py` — 30 fixed pairs (16 synthetic in-plane/out-of-plane
from `placement_m11_families`, 14 real from the ETH-80 catalogue when it is
present), each matched twice (arm A: placement matcher off; arm C: on). For every
pair and arm it records `repr(float(score))`, `repr(float(score_shipped))`,
`repr(float(placement_evidence))`, `placement_judged`, the mapping as a sorted list
of `(u, v)`, and the arm's seconds.

* `--save` writes `Topology/results/placement/m12_perf_baseline.json`
* `--check` diffs against it field by field and exits non-zero on any diff

### Profile before (cProfile, one synthetic pair, arm C, `joint_mcs`)

45.4 s under the profiler:

| Call | Calls | Time |
|---|---|---|
| `_relational_agreement` (`_rel`) | 403 760 | 28.3 s |
| `node_sim` (incl. `_sig_similarity`) | 1 818 300 | 8.9 s |
| `UNDIRECTED_EDGE_TYPES.__contains__` | 4 734 859 | 6.6 s |
| `_compatible_classes` | 2 361 334 | 5.5 s |
| `_swap_pass` / `_swap_delta` / `_quality` | 5 463 / 230 716 | 24.5 s |

`_relational_agreement` and the swap pass are the same work seen twice (`_quality`
calls `_rel`); the two pure-function lookups below them were being recomputed
millions of times.

### What was changed (all in `matching/alignment.py`, no default changes)

1. `_is_undirected(et)` — `et in UNDIRECTED_EDGE_TYPES` memoised per edge type.
   Both operands are fixed at import (the schema fixes the undirected set), so the
   memo cannot change a value. Five of the eight call sites use it.
2. `_compatible_classes(et)` — the schema-class lookup is memoised per edge type on
   top of the existing per-class memo.
3. `_rel(u, v, mp, rv)` — for the `candidate is None` path the relational agreement
   is cached on exactly the context it reads: `(u, v, images of u's neighbours,
   preimages of v's neighbours)`. Every other input is static graph data, so the
   cache cannot change a value; it is bounded (cleared above 400 000 entries) and
   the `candidate is not None` path bypasses it entirely (a candidate induces a
   world the context key does not describe). Kill-switch `M12_REL_CACHE=0`.

### Profile after (same pair, arm C)

32.1 s under the profiler (1.42×): `_relational_agreement` residuals 264 262
misses / 11.4 s, `node_sim` 1 818 300 calls / 8.9 s (unchanged — it is already
`lru_cache`d, and 182 620 of those calls are *distinct* pairs at ~42 µs each),
`_swap_pass` 11.7 s, `rescore_matrix` 4.3 s.

### Acceptance (14 fixed pairs, 2 workers per run, each tree run twice)

| Run | wall | per-pair A | per-pair C |
|---|---|---|---|
| pristine ×2 | 214.6 s / 208.7 s | 14.32 / 14.03 | 16.12 / 15.64 |
| optimised ×2 | 164.6 s / 164.8 s | 10.84 / 10.40 | 12.49 / 12.81 |

* pristine vs pristine: **identical** (0 diffs)
* optimised vs optimised: **identical** (0 diffs)
* **pristine vs optimised: identical (0 diffs)** — every score, every shipped
  score, every evidence value, every judged count and every mapping.

### The reproducibility trap this exposed (recorded because it cost the session a detour)

An earlier `--save`/`--check` pair disagreed on 28 mappings while every score
matched to the last digit. The cause was **not** the optimisations: the graph
pickle cache held entries built in an earlier code/sandbox state, and a worker
rebuilt some of them mid-session (the rebuild is content-addressed, so the
pixels are the same; the graph object handed to the aligner was not the one the
baseline was scored on). After the cache was consistently rebuilt, every tree —
pristine, optimised, in-process, forked, 1 or 2 workers — reproduces exactly, and
`--save` twice in a row is byte-identical. **Any bit-identity claim must therefore
be made over runs that share one cache state**; the acceptance above is, and the
harness now records enough to detect the situation.

### Why the gate is not met, and what a 3× needs

The remaining cost is dominated by two per-pair products, both already minimal in
Python:

* **node similarity**: the seed matrix is the full `n × m` product (182 620
  distinct pairs per direction, ~42 µs each ⇒ 7.7 s per pair for the two
  directions). `node_sim` is already memoised; the cost is the deep signature
  comparison (`core/types.py::_similarity_gaussian` plus
  `core/population.py::population_similarity`).
* **relational agreement**: ~264 k distinct evaluations per pair at ~40 µs.

Neither is reducible by caching (the caches are already in), and neither can be
skipped without changing the answer (a pruned matrix is a different alignment,
which the mission forbids). A 3× needs the similarity computed in **batched numpy
per class** (the signatures of one class share an axis layout), which is a
separate piece of work with its own bit-identity argument. That is proposed, not
claimed.

### STEP 1A — the missed cost centres

Four changes, none of which can change a value:

| Change | Where | Why it is exact | Status |
|---|---|---|---|
| **b. One Side per image** | `matching/placement_matcher.build_side_cached`, used by `joint_mcs` | `build_side` is a pure function of (graph object, pixels, switches); the cache holds the graph, so the id cannot be recycled, and it keys on the identity, the pixel digest and the frozen bundle | landed |
| **c. Centred form precomputed** | `core/population.centered_cosine` / `_centered_form` | the mean, the centred vector and the sum of squares are cached per vector and VALIDATED against a content digest; the same floats in the same accumulation order | landed, 2 000 random cases bit-identical to the old code |
| **d. `cells_scored`** | `matching/placement_matcher.rescore_matrix` | it counted the CAPPED context rows; it now counts the (row, cell) pairs `plus_score` was actually called on | fixed |
| **a. 30 pairs** | — | the ETH-80 corpus is still downloading in this sandbox (`catalog()` raises `FileNotFoundError`); the harness registers the 14 real pairs the moment it lands | **not met yet** |

Every one of these is behind no flag because none of them is a behaviour: the
Side cache and the centred cache return the same objects/values, and the counter
is a diagnostic. The evidence they are exact: with all of them in, the arm-A and
arm-C mapping hashes of the fixed harness pair are unchanged from the committed
baseline (`0cfa87a2` / `2865d93d`), and the scores are unchanged to the last digit.

### Where the harness stands

* 14 synthetic pairs registered and accepted.
* The 14 real pairs are **not yet in**: `data/ETH-80-master` is still extracting
  (`catalog()` raises `FileNotFoundError`), and the harness degrades to
  synthetic-only and says so, in the log and in the baseline's pair count. Both
  galleries are required by the mission, so the acceptance is repeated over 30
  pairs before Step 6 and the table above is superseded then.


---

## STEP 2 — the coverage gradient of E, and the reading chosen on FIT

### Instrument

`Topology/placement_m12_e_coverage.py` — 10 FIT pairs (6 irregular, 4 out-of-plane),
22 724 positive candidate cells, 566 of them the TRUE candidate under the
renderer's own track truth. For every cell it records the judged-slot count, the
deployed term, the mean term, three dark variants, and the blended candidate score
the matcher actually ranks by. Truth is a FIT-side instrument only; nothing here is
a read.

### The audit

Spearman rho against `n_judged` (the judged-slot count), over the non-true
candidates and over the true ones separately, and against the BLENDED score:

| reading | rho(all) | rho(true) | rho(non-true) | rho(blended score, non-true) | FIT top-1 |
|---|---|---|---|---|---|
| `e_sum` (deployed) | −0.418 | **+0.457** | −0.453 | 0.134 | 0.4692 |
| `e_mean` (mean over judged slots) | −0.872 | **−0.060** | −0.905 | 0.045 | 0.4692 |
| `e_min2` / `e_min3` (mean, refused below 2 / 3) | −0.872 | −0.060 | −0.905 | 0.045 | 0.4692 |
| `e_shrink` (mean shrunk to the query mean, n/(n+4)) | −0.934 | −0.421 | −0.954 | 0.047 | 0.4645 |
| `e_rank` (mean, rank-normalised in the shortlist) | 0.001 | +0.467 | −0.016 | 0.060 | 0.3104 |

What this says, in order of importance:

1. **The deployed sum is biased exactly where the mission said it would be.** For
   the TRUE candidate E rises with the number of judged slots (+0.457): a place
   with more checkable geometry hands the correct pair more evidence for being
   correct. The mean reading removes that (+0.060, sign flips and magnitude
   collapses).
2. **The rivals' negative correlation is not the same defect.** A place with more
   judged slots gives a wrong pair more chances to be *contradicted* (−0.905). That
   is information, and it is the direction that helps.
3. **At the level the matcher ranks by, no reading is biased.** The blended score's
   correlation with the judged count is 0.045-0.134, all below the mission's 0.30
   bar, because the node similarity dominates the blend.
4. **The bias is not free to remove by ranking.** Rank-normalising within the
   shortlist drives the correlation to −0.016 and costs a third of FIT top-1
   (0.4692 → 0.3104): it removes the bias by removing the information.

### The decision (FIT, pre-stated rule)

Rule: admissible = |rho(blended score, judged count)| < 0.30 over the non-true
cells; among the admissible, prefer the readings that are SCALE-FREE in the judged
count (the mean family) — E has to mean the same thing at every place or no single
frozen weight can be right at all of them; tie-break on FIT top-1, then on |rho|.

**Chosen: `e_mean`** — `plus_score(normalise=True)`, the mean over the slots a pair
judged, with a refused minimum (`PLACEMENT_E_MIN_JUDGED`, default 0 because the
synthetic FIT population judges either 0 or the full cap). It is admissible, it is
the most scale-free of the mean family on the true candidates (−0.060 against
+0.457), and it costs **nothing** on FIT top-1 (0.4692, identical to the deployed
reading).

### Landed, dark

* `parameters.PLACEMENT_E_NORMALISED` (False) and `PLACEMENT_E_MIN_JUDGED` (0),
  both registered in `eth80_frontend._FRONTEND_SWITCHES` **before any graph or
  match exists under them** (this does move the graph cache key, so the next run
  rebuilds — the rule the tree already applies to `PLACEMENT_PAIR_NORMALISED`).
* `matching/placement_llr.plus_score(..., min_judged)` — a pair below the minimum
  returns exactly 0.5, i.e. 0 evidence; no evidence always contributes 0.
* `matching/placement_matcher._pair_normalised()` now ORs the two dark flags, and
  all three call sites pass `min_judged`; `matching/joint_mcs.placement_evidence`
  reads the flag so E itself is on the chosen scale.
* **Defaults byte-identical:** with both flags off, the fixed harness pair still
  maps to `0cfa87a2` / `2865d93d` with the same scores to the last digit; with
  `PLACEMENT_E_NORMALISED=1` the arm-C mapping and score move (0.38958246 →
  0.38791059), which is the flag doing its work and proves it is live.

---

## STEP 3 — real-image frames (partial: the corpus is not in this sandbox)

### What was built

`matching/placement_edges.node_frames_keypoint` — the frame read by the KEYPOINT
layer's own estimator: the structure tensor of the Gaussian-smoothed gradient at
the keypoint layer's sigma ladder (1.0, 1.6, 2.5, 4.0) and sampled BILINEARLY at
the node's sub-pixel anchor, with the multiscale field as the fallback
(`node_frames_keypoint_then_field`). It can only ADD frames — a node the keypoint
estimator gives a frame keeps that frame. Dark behind `PLACEMENT_FRAME_KEYPOINT`,
registered in `_FRONTEND_SWITCHES` before any graph exists under it.

The frame-free relations the mission also asks for (angle at u between two
neighbours, distance ratio) are NOT in this commit; the frame coverage below is the
link they would work around, and the measured numbers say how much there is to work
around.

### Measured (6 synthetic pairs, same graphs and same images for every provider)

| provider | frame share of nodes | median paired residual at a known 5° rotation | share of nodes within 5° |
|---|---|---|---|
| field (7×7, shipped) | 0.373 | 6.22° | 0.267 |
| multiscale | 0.500 | 6.75° | 0.279 |
| **keypoint (M12)** | **0.540** | 8.17° | 0.303 |

Two things, and only one of them is good news:

1. **Coverage rises**: 0.373 → 0.540 of nodes carry a frame (+16.7 points over the
   shipped provider, +4.0 over the multiscale one). This is the link that caps the
   judged-pair share, and it is the one the mission asked to raise.
2. **Stability at 5° does not**: against a ground-truth rotation of 5°, the median
   per-node residual is 6.2-8.2° — that is, the frame error is as large as the
   rotation being validated, and only 27-30 % of nodes land within 5°. The
   keypoint provider is the WORST of the three on this measure (8.17°). So the
   extra frames it finds are frames, but not yet frames that survive a small
   rotation. Naming that plainly is the honest result of running the validation.

### The floor, stated

The real-photograph half cannot run here: `data/ETH-80-master` is still
downloading, `catalog()` raises `FileNotFoundError`, and the harness records
`real: {"skipped": "FileNotFoundError: ..."}` rather than pretending. **The real
judged-pair share is therefore not raised, not measured, and not claimed in this
session.** What is stated instead is the floor: on synthetic scenes the judged
share is measured by the same instrument (Step 2: 26 % of candidate cells, 8 judged
slots when a place has any), and the frame share is 0.54 with the keypoint
provider, so a real-image target of ≥60 % judged pairs remains unverified.

---

## STEP 4 — the w / temperature refit (RUNNING; resumable checkpoint)

`Topology/placement_m12_fit_w.py` fits `(T, w)` in

    S' = S_shipped/T + w*E

by maximum likelihood over FIT shortlists, with E on the STEP-2 scale
(`PLACEMENT_E_NORMALISED`).

* **The set**: 8 FIT scenes (the FIT halves of the M9 irregular and out-of-plane
  families) × 18 stratified views = 252 queries, a quarter (63) held out before the
  fit. Each shortlist is the true reference + two same-family twins + one
  other-family rival — hard BY CONSTRUCTION, and no query was selected by looking at
  an error the shipped score made.
* **Arms**: A (placement off) is run only on the held-out quarter, where the shipped
  top-1 is reported; B and E on every pair.
* **Status: RUNNING.** The first attempt was killed by the OOM killer at 1 GB RSS
  per worker in a 3.9 GB sandbox (a real constraint of this environment, recorded
  here because it explains the pacing); it is resumable from
  `m12_fit_w_progress.jsonl`. **No fitted number is reported until the checkpoint is
  complete and the held-out quarter has been scored.**

---

## STEP 5 — the shortlist: done

`Topology/placement_m12_synth_shortlist.py`, 51 FIT queries against the whole
28-scene M9 gallery (chance 1/28), every scorer truth-free:

| scorer | recall@8 | recall@16 | recall@32 | recall@48 | MRR |
|---|---|---|---|---|---|
| node similarity | 0.804 | 0.863 | 1.000 | 1.000 | 0.620 |
| structure alone (place count, edge-type histogram, degree distribution) | 0.647 | 0.863 | 1.000 | 1.000 | 0.294 |
| blend, λ = 0.25 | 0.784 | **0.902** | 1.000 | 1.000 | **0.650** |
| blend, λ = 0.50 | 0.765 | 0.902 | 1.000 | 1.000 | 0.650 |
| blend, λ = 1.00 | 0.765 | 0.863 | 1.000 | 1.000 | 0.560 |

**The ceiling, stated rather than met.** No K below the gallery size reaches
recall ≥ 0.95: the best is 0.804 at K = 8 and 0.902 at K = 16. K = 32 reads 1.000
only because the gallery has 28 scenes, i.e. it is the whole gallery and the number
is vacuous. So the honest statement is:

* **K = 16 with the λ = 0.25 blend is the smallest useful shortlist**: recall 0.902,
  MRR 0.650, against 0.863 / 0.620 for node similarity alone at the same K;
* at K = 8 the best recall is 0.804, and retrieval must be reported **conditional on
  the shortlist** — which is what the Step-6a registration says it does (K = 8), and
  the read's report carries this 0.804 with it;
* the structural prefilter ALONE is a real filter (0.647 at K = 8, 0.863 at K = 16)
  — two thirds of the gallery is discarded for the price of three histograms — and
  blending it into the node score buys 0.04 of recall@16 and 0.03 of MRR.

---

## STEP 6a — the fresh synthetic read: REGISTERED, not yet run

`Topology/results/placement/m12_prereg_synth.json`, written with **0 graphs in the
cache** for these scenes:

* fresh seeds 38200+ (8 irregular) and 40200+ (4 out-of-plane), names `m12ir_*`,
  `m12oop_*` — nothing in the tree has ever drawn from them;
* **192 queries** = 12 scenes × 16 stratified views (5-35° in plane, y 10-30°,
  x 10-15°, scale 0.9/1.1, two shifts);
* shortlist K = 8 of 12 by the node-similarity score, recall reported with the read;
* arms A / B / C, primary **C − A** on top-1, paired per query, clustered by scene,
  4000-draw bootstrap;
* prediction **B ≤ A**, with the reason written down before the read: B moves the
  mapping and nothing else, so it pays for a moved correspondence without being paid
  for the evidence that moved it;
* **MDE = 2.8·0.48/√192 = 0.097 ≤ the 0.10 planning effect ⇒ POWERED**. (The first
  registration of this read used 6 views — 72 queries, MDE 0.158 — and its own
  arithmetic declared it UNDERPOWERED IN ADVANCE; it was replaced before any graph
  existed. Both registrations report 0 graphs at registration.)
* verdict per the Step-0 rule; one read.

`Topology/placement_m12_read_synth.py` executes it and refuses to run without the
registration. **The read has not been run**: the two long jobs ahead of it own the
sandbox, and it must not be started until they finish. It is registered, powered,
and queued — not reported.

## STEP 6b — ETH-80: BLOCKED, and says so

`catalog()` cannot complete without the corpus. There is no ETH-80 query set, no
fresh views, no 200-query read, and none is fabricated. The M11 partial (25
complete queries) remains what it was: not a result, not used for any choice.

---

## Defaults, re-verified after every flag

With `PLACEMENT_E_NORMALISED=0`, `PLACEMENT_E_MIN_JUDGED=0` and
`PLACEMENT_FRAME_KEYPOINT=0` (all three are their defaults), the fixed harness pair
still maps to `0cfa87a2` (arm A) / `2865d93d` (arm C) with scores
`0.4182298669434846` / `0.3895824615188941` — byte-identical to the baseline
committed with Step 1, after the side cache, the centred-form cache, the counter
fix, the flag registrations and the keypoint frame source. The graph cache is
rebuilt because the two new switch names enter `_FRONTEND_SWITCHES`; that is the
tree's own rule for a flag that can change what a payload reads.

## What is finished, what is running, what is blocked

| Step | State |
|---|---|
| 0 | **done** — the rule, the three old verdicts side by side, the stale M11 job stated |
| 1 | **done, gate not met** — 1.3× with bit-identical output over 14 pairs (the 30-pair set needs the corpus); profile and the reason in the Step-1 section |
| 1a(b,c,d) | **done** — Side cache, centred-form cache, counter fix; all value-preserving, verified |
| 2 | **done** — audit + FIT-chosen reading, dark, cache-keyed, defaults identical |
| 3 | **partial by necessity** — keypoint frames landed dark (+16.7 points of frame coverage) but WORSE at the known 5° rotation, and the real-photo half cannot run without the corpus. Frame-free relations not landed |
| 4 | **running** — checkpointed, resumable; no number claimed |
| 5 | **done** — K = 16 blended (recall 0.902) is the smallest useful K; the 0.95 bar is not reachable below the gallery size and is stated as a ceiling |
| 6a | **registered and powered (MDE 0.097), not run** |
| 6b | **blocked on the corpus** — no ETH-80 read exists and none is invented |

---

# MISSION 12, SECOND BRIEF — steps 1–6

The brief asked for six things; this section reports them in order. Nothing in it
changes a default: every new switch is off, and the defaults harness still returns
`0cfa87a2` / `2865d93d` with scores `0.4182298669434846` / `0.3895824615188941`.

## Step 1 — photographs, and ETH-80 closed

**ETH-80: BLOCKED.** The download never advanced past 322 534 399 B of a ~4.5 GB
tarball, twice, from scratch; the fetcher is killed and there is no ETH-80 query
set, no fresh view, no read. Nothing from that corpus is measured, quoted, or
implied anywhere in this mission.

**Photographs: 20, at working resolution.** `Topology/placement_m12_realphotos.py`
loads images that ship *inside the installed wheels* — nothing is fetched, so the
sandbox's blocked TLS cannot silently change the corpus:

* scikit-image 0.26.0: `astronaut, chelsea, coffee, camera, clock_motion, rocket,
  coins, motorcycle_left, motorcycle_right, page, brick, grass, gravel, moon,
  hubble, ihc, retina, cell`
* scikit-learn 1.9.1: `china, flower`

Each is resized (bilinear for defocus) to **256×256 uint8 RGB** — the resolution the
payload actually consumes — and cached under `.cache/m12_real_photos/`. The
manifest `m12_real_photos.json` records package, loader, shape, the sha256 of the
resized array, and points at the package's own documentation for the licence;
**the pixels are not committed**. Six are reserved for frame validation
(`gravel, grass, page, moon, cell, retina` — texture, text, astronomy, histology);
fourteen are held for the pre-registered read.

## Step 2 — the frame validation, rewritten and run

The M11 test rotated the image and then compared node frames *sampled at the
original pixel*, so every node read a different piece of the scene: the number it
reported was coverage under a broken correspondence, not frame error. That test is
replaced by `Topology/placement_m12_frames2.py`, which:

1. rotates the image about its centre by `delta` (`scipy.ndimage.rotate`,
   `reshape=False`, bilinear, `mode='nearest'`);
2. rotates the node anchors by the *same* transform, `p -> c + R(sign*delta)(p-c)`
   with `c = (255/2, 255/2)` in (x, y) = (col, row) order;
3. samples the provider at the **transformed** anchor in the rotated image;
4. reports the **SIGNED** residual `angle_rot - angle_orig - sign*delta`, wrapped
   into the axial range (−90°, 90°], as median, MAD, and share within 3°;
5. drops anchors that land outside the image or within the 8 px border, in either
   image, and drops anchors whose coherence is below `FRAME_COHERENCE_MIN`.

**Sign convention.** It is calibrated, not asserted: for every image × delta the
residual median is computed under both signs on the shipped field provider and the
sign with the smaller |median| is kept. That choice was **−1 for all 24 cases
unanimously** (e.g. `syn|m11ir_00`, delta 5°: median −9.36° under +1 vs −1.12°
under −1). Reported convention: with the image rotated by `+delta`, the axial angle
the field yields changes by `−delta`, i.e. `theta_rot = theta_orig − delta`, which is
what the anchor map `R(−delta)` expresses. One consequence is stated rather than
hidden: because the sign is chosen per image by |median| on the field provider, the
field row's median and its within-3° share are mildly optimistic; the keypoint and
multiscale rows are scored under the same sign and are not.

**Providers** are now genuinely three, at field level:

* `field` — the deployed 7×7 box tensor (`field_angles`), nearest sampling;
* `multiscale` — the deployed ladder `field_multiscale` over `FIELD_WINDOWS =
  (7, 15, 23, 31)`, where at each anchor the smallest window clearing the coherence
  bar wins, independently in each image, exactly as `node_frames_multiscale` does;
* `keypoint` — the Gaussian tensor at the largest keypoint sigma, bilinear sampling.

(The first run of the harness had `multiscale` aliased to the field array — a
placeholder, producing numbers identical to `field` to sixteen digits. It was
replaced by the real ladder before anything was recorded here; the numbers below
are from the corrected run.)

**Result, 24 image × delta cases** (6 irregular synthetic scenes, 6 real
photographs; delta = 5° and 15°; 279–857 anchors per provider per scene):

| provider | class | cases | median signed | median MAD | mean share ≤3° |
|---|---|---|---|---|---|
| field | synthetic | 12 | −0.65° | 1.42° | 0.655 |
| field | real | 12 | +0.22° | 1.58° | 0.598 |
| field | all | 24 | −0.10° | 1.47° | 0.626 |
| multiscale | synthetic | 12 | −0.98° | 1.85° | 0.557 |
| multiscale | real | 12 | +0.22° | 1.83° | 0.505 |
| multiscale | all | 24 | −0.43° | 1.83° | 0.531 |
| keypoint | synthetic | 12 | −1.29° | 1.09° | 0.720 |
| keypoint | real | 9 | −0.07° | 0.15° | **0.913** |
| keypoint | all | 21 | −0.64° | 0.83° | **0.803** |

Per structure class, share ≤3° (pooled over the two deltas; `n` anchors):

| scene | field | multiscale | keypoint |
|---|---|---|---|
| syn m11ir_00 | 0.583 / 368 | 0.525 / 566 | 0.722 / 351 |
| syn m11ir_01 | 0.714 / 330 | 0.560 / 562 | 0.774 / 424 |
| syn m11ir_02 | 0.658 / 299 | 0.580 / 520 | 0.668 / 343 |
| syn m11ir_03 | 0.578 / 314 | 0.490 / 461 | 0.780 / 298 |
| syn m11ir_04 | 0.623 / 279 | 0.529 / 447 | 0.603 / 310 |
| syn m11ir_05 | 0.774 / 287 | 0.660 / 426 | 0.773 / 279 |
| real cell | 0.937 / 279 | 0.799 / 370 | 0.969 / 243 |
| real grass | 0.391 / 48 | 0.391 / 48 | 1.000 / 5 |
| real gravel | 0.442 / 26 | 0.442 / 26 | — (0 coherent) |
| real moon | 0.798 / 116 | 0.560 / 233 | 1.000 / 100 |
| real page | 0.514 / 60 | 0.417 / 89 | 1.000 / 20 |
| real retina | 0.503 / 437 | 0.419 / 857 | 0.637 / 589 |

Two things the table shows and the old test could not:

* the residuals are **signed and centred on zero** — this is a frame test, not a
  coverage measurement; the anchors that survive are the ones both images can read;
* the **ladder is not rotation-equivariant**. The multiscale provider judges 1.5–2×
  more anchors (window 7 is used for 3238 of 4605 samples, 15 for 827, 23 for 336,
  31 for 204), but the window a node picks can *change* when the image is rotated,
  and a 23×23 estimate at a pixel is not a 7×7 estimate at that pixel: self-consistency
  drops from 0.626 to 0.531 despite the extra coverage. That is a property of the
  deployed rule, measured, not an artefact of the test. `keypoint` pays the opposite
  price — it reads fewer anchors (on fine texture, none: gravel has zero coherent
  anchors at the largest sigma) and is much more accurate on the ones it reads,
  0.913 on photographs against 0.598 for the field.

**The provider choice is the deployed matcher's, on FIT.** Eight FIT pairs
(`gallery()[:8]`, irregular, second view), each matched under the shipped field
provider and under keypoint, with everything else at the M12 setting
(`PLACEMENT_MATCHER_ENABLED`, `E_NORMALISED`, weight on E). Correspondence accuracy
is scored on the paired *near-tie* bank (top similarity ≥ 0.99 and margin ≤ 0.05 —
the nodes the matcher can plausibly confuse) and on the unambiguous bank, against
`track_optimal` truth:

| metric | shipped field | keypoint |
|---|---|---|
| near-tie accuracy | 0.727 (n=44) | 0.750 (n=44) |
| unambiguous accuracy | 0.274 (n=234) | 0.248 (n=234) |
| judged pairs / pair | 64.0 | 77.3 |
| judged share | 0.105 | 0.127 |

Paired difference on the near-tie bank, clustered by pair, 4000-draw bootstrap:
**+0.0625, 95% CI [0.000, 0.1875], n = 8**. The pre-declared rule was to adopt only
if that interval excludes zero in keypoint's favour, so

> **keypoint is NOT adopted.** It stays dark, `PLACEMENT_FRAME_KEYPOINT = False`
> (listed in `_FRONTEND_SWITCHES`).

The interval's half-width (0.125) puts this design's MDE at roughly 0.18 absolute
near-tie accuracy at 8 FIT pairs; the observed +0.06 is inside that, so the honest
reading is *underpowered for the effect it saw*, not *no effect*. The frame test
says keypoint is the better frame reader on real photographs (0.913 vs 0.598 within
3°); the deployed matcher says the gain does not yet reach significance on FIT.
The rule is the matcher's, so the switch stays off and the registration note for
Step 5 stands unchanged.

Artifact: `Topology/results/placement/m12_frames2.json` (24 rows, calibration, three
aggregates, per-structure table, decision).

## Step 3 — the speed work: what the redundancy is, and where the cost is not

**(a) How much of the similarity work is a repeat.**
`Topology/placement_m12_sigcache.py --measure` counted, over a retrieval-shaped run
of **576 graph pairs** (24 gallery references × 24 queries, the same family the read
uses), the questions the matcher asks and the answers a content-keyed cache can
hold:

| quantity | value |
|---|---|
| same-lens node pairs asked | 25 569 904 |
| distinct (lens, content(a), content(b)) triples, summed over pairs | 25 569 904 |
| distinct triples shared over the whole run | 14 400 439 |
| reuse **within one graph pair** (median / min / max) | **1.00× / 1.00× / 1.00×** |
| reuse **across the run** | **1.78×** |

The exact-Python-set count and the uint64-folded count agree on the first pair
(51 650 = 51 650), so the vectorised counting is not losing keys. The measured fact
is unfriendly to the idea: **inside a graph pair there is no redundancy at all** —
every same-lens node pair has a distinct signature-content pair — and across a whole
retrieval run the content repeats only 1.78×. That is the ceiling for this lever.

**(b) The content-keyed cache, and what it actually bought.**
`matching/sig_sim_cache.py` keys on the WHOLE signature (lens, kind, class,
relation, the entire quantized tuple, the entire graded population — which
`TypedSignature.__hash__` deliberately excludes — plus the two tolerance flags), and
stores the shipped `_sig_similarity` call's own float. `joint_mcs.node_sim` uses it
only when the new dark flag `MATCH_SIG_SIM_CACHE` is on; with it off the shipped
branch runs unchanged.

The first implementation hashed hex strings of every graded value on every call: it
was **1.5× slower** than no cache (6.35 s → 9.88 s per pair), because the similarity
it was caching is cheaper than formatting 700 floats. The shipped version memoises
one content hash per signature object and looks up a three-int key.

Acceptance, on **30 fixed synthetic pairs** (`pairs30()`: every irregular scene's
second view against the other five scenes' references; the corpus pairs the older
harness would add do not exist because ETH-80 is blocked, and the read says so):

* score, `score_shipped`, placement evidence, judged count and the **whole mapping**
  are **bit-identical with the flag off and on** — 30/30 pairs, 0 diffs;
* 2 837 810 similarity calls, 389 872 hits (**13.7 %** — one worker, so this is the
  *within-worker* reuse over 30 pairs, lower than the retrieval-run 1.78×);
* **speed 0.97×** (152.5 s → 156.8 s over the 30 pairs). The cache is
  **value-preserving and not faster**, so it stays DARK.

**(c) Why vectorising the similarity would not reach the bar, said before running it.**
The profile of one pair (cProfile, 24.6 s under the profiler) puts the cost here:

| item | tottime | share |
|---|---|---|
| `_relational_agreement` | 5.85 s | 24 % |
| `node_sim` (cumulative, incl. `_sig_similarity` 4.49 s) | 5.38 s | **22 %** |
| `_swap_delta` / `_quality` (the aligner's swap evaluation) | 13.7 s / 13.0 s cumulative | 55 % |
| `enum.__hash__` inside the undirected/class memos | 1.64 s | 7 % |

Even an **infinitely fast** similarity removes at most 22 % of a pair's time, i.e. at
most **1.28×**, and 1.78× reuse can only be applied to that 22 %. So the conditional
branch — vectorise per-class similarity, accepting 1e-12 score drift — was **not
run**: its maximum possible payoff (≤1.28×) cannot meet the 3× bar, and buying it
would mean giving up the bit-identity that the acceptance criterion demands, for a
fraction of a second per pair. The arithmetic was checked before the run, and it is
the reason there is no relaxation to report.

**Where the time actually is** is the aligner's swap evaluation, which calls
`_relational_agreement` 225 199 times per pair. That is the next target, not the
similarity: a candidate-level memo is not available (the mapping changes between
rounds, so the same (u, v, candidate) is only valid within a round) and a safe
version must key on the round's mapping, which is a redesign rather than an
optimisation. It is named here, unclaimed.

Artifact: `Topology/results/placement/m12_sigcache.json`.

## Step 5 — the pre-registered real-photo read: NOT DETECTED, and the mechanism

`Topology/placement_m12_read_real.py` executed `m12_prereg_real.json` exactly as
registered — 14 photographs × 4 exact transforms (`rot05`, `rot15`, `scale115`,
`trans8`) = **56 queries**, arms N (placement off) and P (deployed matcher, E
normalised, w = 0.375 read from `placement_score_params.json`), photo-clustered
4000-draw bootstrap. Nothing was re-run or re-analysed: one read.

**The registered primary.** Near-tie bank (top similarity ≥ 0.99, margin ≤ 0.05,
restricted to nodes the transform maps into the image): **541 nodes over 55
queries**.

| near-tie bank | value |
|---|---|
| correspondence accuracy, arm N | 0.4473 |
| correspondence accuracy, arm P | 0.4492 |
| **P − N** | **−0.0022** |
| 95 % CI (photo-clustered, 14 clusters) | **[−0.0118, +0.0067]** |
| realised planning MDE (n_bank = 541) | 0.0851 |

> **Verdict: NOT DETECTED.** The interval excludes any gain larger than +0.007.

**No-harm.** The unambiguous bank (2 096 nodes) reads 0.3497 in **both** arms,
P − N = −0.0010, lower end **−0.0061 ≥ −0.02** → the no-harm condition holds.

**Frame chain.** 206 judged pairs per query on average, judged share 0.258, mean
evidence −0.051 — the matcher is engaged and its evidence is slightly negative, i.e.
it does not favour the truth on this corpus. That is consistent with the null.

**The mechanism (found, not assumed).** The aggregate hides a sharp split, and the
split is the finding:

| transform | near-tie nodes | acc N | acc P | P − N [95 % CI] | unambiguous acc N → P |
|---|---|---|---|---|---|
| `trans8` (exact +8 px) | 285 | 0.758 | 0.765 | **+0.011 [−0.003, +0.033]** | 0.730 → 0.733 (n=708) |
| `rot05` | 72 | 0.153 | 0.153 | 0.000 [0.000, 0.000] | 0.176 → 0.169 (n=421) |
| `rot15` | 84 | 0.036 | 0.036 | 0.000 [0.000, 0.000] | 0.133 → 0.133 (n=601) |
| `scale115` | 100 | 0.120 | 0.110 | −0.019 [−0.053, +0.007] | 0.169 → 0.172 (n=366) |

Under an exact integer shift the correspondence is recovered ~76 % of the time;
under 5°–15° of rotation and 1.15× scale it collapses to near chance, **in both
arms**. The placement matcher neither rescues it nor damages it.

Two candidate causes were separated with diagnostics on the same graphs (not new
reads):

1. **Is the truth wrong?** `placement_m12_truthcheck.py` measured the
   back-projection distance of every truth pair: **0.00 px median under `trans8`**
   (the transform is exact, the anchors coincide) against **0.8–1.2 px median,
   1.5–1.9 px p90 under rotation/scale**, with the share of truth pairs that are
   *mutual* nearest neighbours between 0.00 and 0.25. So for two transforms out of
   four the truth was being assigned at the scale of its own noise: with nodes
   ~1–2 px apart, "nearest anchor within 2 px" is not an identification.
2. **Does a tighter truth restore the signal?** `placement_m12_truthcalib.py`
   re-derived truth with tol **1.0 px**, second-nearest at least **2×** further,
   and **mutual** — and re-scored the same 16 (photo, transform) cells:

   | transform | tight truth nodes (4 photos) | accuracy among them |
   |---|---|---|
   | `trans8` | 332 | **0.56 – 0.79** |
   | `rot05` | 86 | 0.06 – 0.25 |
   | `rot15` | 142 | 0.00 – 0.14 |
   | `scale115` | 72 | 0.00 – 0.22 |

   The instrument is not the whole story: even where the anchors are identified to
   within 1 px and unambiguously, the matcher maps rotations and 1.15× scale to the
   wrong node. **The correspondence it computes is translation-tolerant and
   rotation/scale-intolerant**, which is a property of the alignment, not of the
   placement layer (both arms fail identically).

So the honest reading of the registered null has two parts, and both are reported
rather than one used to excuse the other:

* the aggregate contrast is **attenuated** where the truth is noisy, so "not
  detected" is not "no effect";
* the sharpest estimate the design can produce — the `trans8` bank, where the truth
  is exact to 0.00 px — is **+0.011 [−0.003, +0.033]**, still not detected, and it
  is the number to quote.

A second read on fresh transforms was therefore **not** run: the instrument's
limits are now measured, and re-running the same question on the same photographs
would re-measure the same collapse of the matcher, not the placement layer. The
`w`/temperature refit (Step 4) is where a change could come from.

Artifacts: `m12_read_real.json`, `m12_read_real_progress.jsonl` (56 rows),
`m12_read_real_truthcheck.log` numbers in this section, `m12_truth_rule_calib.json`.

## Step 4 — the gate: the refit is still running, and the read is still unspent

`placement_m12_fit_w.py --start` is **running in the background** (checkpointed,
resumable, `m12_fit_w_progress.jsonl`): 330 of 1 008 (w, T) cells were done when the
sandbox was handed the frame/read work, and it has been restarted at higher
concurrency. The gates it must pass are declared in the code, not here
(`w ∈ (0.0, 1.5)`, `T ∈ (0.02, 0.6)` strictly inside the grid, finite log-likelihood,
log-likelihood above the `w = 0, T = 1` cell, held-out top-1 not below the shipped
matcher) and `m12_read_synth.py` refuses to start without `gates.passed`.

**No gate has been evaluated and no read has been spent.** The fit did not finish
inside this session; that is stated rather than worked around.

## Step 6 — ETH-80: BLOCKED

Unchanged from Step 1: the corpus never downloaded (frozen at 322 534 399 B of a
~4.5 GB tarball, killed twice), so there is no ETH-80 read, no fresh ETH-80 view,
and no number from that corpus anywhere in this mission. **BLOCKED, nothing
invented.**

## Identity after the Step-2/3 code changes

* The content-keyed similarity cache (`matching/sig_sim_cache.py`, dark) was proven
  on **30 fixed pairs**: score, shipped score, evidence, judged count and the whole
  mapping are **bit-identical** with the flag off and on, **inside one process**
  (the only comparison a mapping can support — see below).
* Against the committed perf baseline, the current tree reproduces **every score,
  every shipped score, every evidence value and every judged count** on the 14
  fixed pairs (`placement_m12_perf.py --check`: 28 diffs, **all of them `mapping`**,
  none of them a number). That mapping-level instability is *pre-existing* and was
  already documented in the Step-1 section: equally-scored nodes swap partners
  between processes, so a mapping comparison is only valid within one cache/process
  state. Fixing it means fixing the aligner's tie-break (an identity-hashed set
  somewhere in the search), which is a change to the algorithm's output and is not
  smuggled in here.
* `MATCH_SIG_SIM_CACHE`, `PLACEMENT_FRAME_KEYPOINT`, `PLACEMENT_E_NORMALISED`,
  `PLACEMENT_E_MIN_JUDGED` are all at their defaults (off).

---

## Where this brief stands

| Step | State |
|---|---|
| 1 | **done** — 20 packaged photographs at 256×256, 6 reserved for validation, manifest with hashes and licence pointers; ETH-80 **BLOCKED**, killed, nothing claimed from it |
| 2 | **done** — frame test rewritten (anchor-rotating, signed residual, sign calibrated 24/24 to −1), three real providers, per-structure table; keypoint wins the frame test on photographs (0.913 vs 0.598 within 3°) but **is not adopted**, because the deployed matcher's FIT interval for near-tie accuracy is [+0.000, +0.188] and does not exclude zero (MDE ≈ 0.18 at 8 pairs). Stays DARK |
| 3 | **done, bar not met and said so** — redundancy measured (1.00× inside a pair, 1.78× across a 576-pair run), content-keyed cache landed DARK and **bit-identical on 30 pairs at 0.97×**; the 3× bar is unreachable from the similarity lever (≤1.28× even if the similarity were free), and the profile shows the cost is the aligner's swap evaluation |
| 4 | **done, gate FAILED** — 1 008/1 008 cells; the 2-D fit is degenerate (corner at w=1.55, T=0.015) and the identifiable 1-D fit (w=1.60) loses held-out top-1 (0.730 vs 0.794); held-out is flat from w=0 to w=0.375 and falls after. **The read was NOT spent**, by the declared rule |
| 5 | **done** — pre-registered real-photo read executed once: **NOT DETECTED**, P − N = −0.0022 [−0.0118, +0.0067]; mechanism measured (translation-transitive, rotation/scale-intolerant in both arms), instrument limits quantified |
| 6 | **BLOCKED** — no corpus, no ETH-80 read, nothing invented |

### The confirmation read

`Topology/placement_m12_read_synth.py --start` refuses to run unless
`m12_fit_w.json` carries `gates.passed`; it does not, so **the read is queued and
unspent**, which is what Step 4 asked for. It becomes runnable only if a fit is
found that is both interior and not worse on held-out — and the sweep shows that
no w in the admissible range satisfies both, because the likelihood has no interior
maximum in w at the shipped temperature. Changing the criterion is a new
pre-registration, not a re-run.

## Step 4 (continued) — the refit finished, its gates REFUSED it, and the read was not spent

The sweep completed: **1 008 of 1 008 pairs, 252 queries (63 held out)**. Three pool
deaths (two OOM kills of a long-lived worker, one hung pool) cost only restarts; the
executor now recycles workers (`max_tasks_per_child=25`) and resumes from the
checkpoint instead of abandoning the run.

**The two-parameter fit is degenerate, and the surface says so.** `--model 2d`
(the fit as registered) returned **w = 1.55, T = 0.015** — both outside the declared
bounds (0, 1.5) and (0.02, 0.6) — with the held-out ranking *identical* to w = 0 and
a FIT top-1 *below* the shipped w = 0.375. The recorded lattice (in
`m12_fit_w.json`, `surface`) shows why: the log-likelihood rises monotonically as T
falls and w rises, because `T` is a pure sharpening knob on `S_shipped/T`, so as
T → 0 the objective becomes "the shipped ranking, infinitely sharp" and the argmax
runs to whatever corner the grid has.

**The identifiable direction is w, at the shipped temperature.** `--model 1d` fixes
T = 1.0 and fits w. It returns **w = 1.60** with a curved profile likelihood
(both ends worse by > 0.5 nats) — and its gates refuse it:

| gate | value | pass |
|---|---|---|
| w strictly inside (0, 1.5) | 1.60 | **no** |
| likelihood finite | −247.05 | yes |
| likelihood beats w = 0 (−254.23) | +7.2 nats | yes |
| profile curved | yes | yes |
| held-out top-1 not below the shipped score | **0.730 vs 0.794** | **no** |

> **Gates: FAILED. The confirmation read was NOT run, by the declared rule.**

**Why no w can pass, measured (`placement_m12_fit_sweep.py`).** The likelihood and the
held-out ranking disagree in sign over the whole range:

| w | FIT log-lik | FIT top-1 | HELD-OUT top-1 | HELD-OUT MRR |
|---|---|---|---|---|
| 0.000 | −254.23 | 0.783 | 0.7937 | 0.8743 |
| 0.300 | −252.85 | 0.804 | 0.7937 | 0.8743 |
| **0.375 (deployed)** | −252.51 | 0.794 | **0.7937** | 0.8743 |
| 0.450 | −252.17 | 0.788 | 0.7778 | 0.8664 |
| 0.600 | −251.49 | 0.788 | 0.7619 | 0.8585 |
| 1.000 | −249.70 | 0.783 | 0.7619 | 0.8585 |
| 1.600 (the 1-D fit) | −247.05 | 0.788 | **0.7302** | 0.8426 |
| 2.000 | −245.31 | 0.778 | 0.7143 | 0.8373 |

The log-likelihood is monotone in w with no interior maximum; the held-out ranking is
**flat from 0 to 0.375 and falls after it**. The deployed 0.375 sits at the right-hand
edge of the plateau — nothing larger helps, and the criterion the fit maximises would
push straight into the region that hurts. This is a property of the objective, not of
the optimiser, so "fixing the fit" would mean changing the criterion (e.g. fitting to
held-out top-1), which is exactly what pre-registration forbids. The honest fix is the
one already in the code: **the gate refuses, and the read is not spent.**

Two consequences worth stating plainly, because they agree with Steps 2 and 5:

* on this hard FIT set the placement evidence adds **no measurable ranking gain at any
  admissible w**: held-out top-1 is 0.7937 at w = 0, at w = 0.375 and at every w in
  between, and falls once w exceeds ~0.45;
* the deployed weight is therefore neither too small nor too large — it is at the top
  of a plateau, which is a different statement from "it helps".

Artifact: `m12_fit_w.json` (surface + both fits + gates), `m12_fit_w_sweep.json`,
`m12_fit_w_progress.jsonl` (1 008 rows). `placement_score_params_m12.json` was **not**
written: the gate refused, so nothing is installed.
