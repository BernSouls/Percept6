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

### Where the harness stands

* 14 synthetic pairs registered and accepted.
* The 14 real pairs are **not yet in**: `data/ETH-80-master` is still extracting
  (`catalog()` raises `FileNotFoundError`), and the harness degrades to
  synthetic-only and says so, in the log and in the baseline's pair count. Both
  galleries are required by the mission, so the acceptance is repeated over 30
  pairs before Step 6 and the table above is superseded then.

