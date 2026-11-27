# MISSION 28 — PLACE CONTENT

Branch `arena/c316ca3a-percept6`. Core in `matching/`, thin scripts in `Topology/`. The retrieval
stack (counted door, matcher re-order, flat S) was frozen and was not touched.

Every rate is given with its **n** and its **chance**. Every number below comes from a file in
`Topology/results/placement/` named in the section that reports it.

---

## 1. Housekeeping — determinism, environment, drift

### 1a. The M27 report's two open sections

`MISSION27_REPORT.md` §2 (the S table) and §3 (the loss table) were filled from
`m27_read.json`; every quoted number was checked against the json (0 `<!-- FILL -->` markers
remain). The table itself states the conclusion the mission asked for:

> **S = 8 and S = 16 are not distinguishable.** Top-1 at S = 8 is 0.2375 (19/80, 95 % CI
> [0.150, 0.338], chance 0.0125); at S = 16 it is 0.275 (22/80, [0.175, 0.375]). Paired difference
> S16 − S8 = +0.075, p = 0.58 on FIT and 0.000 on confirmation. **Every new read in M28 uses S = 8.**

### 1b. Determinism within one environment

`Topology/m28_determinism.py --run`: the same 10 pairs scored five ways — one process
(`serial`), 1, 2 and 4 workers, and `serial` with `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1`. Result (`m28_determinism.json`):

| configuration | wall clock | s/pair | max &#124;difference&#124; vs serial |
|---|---|---|---|
| serial | 223.2 s | 22.32 | — |
| 1 worker | 223.4 s | 22.34 | **0.000e+00** |
| 2 workers | 116.3 s | 11.63 | **0.000e+00** |
| 4 workers | 123.2 s | 12.32 | **0.000e+00** |
| serial, BLAS/OMP threads = 1 | 226.7 s | 22.67 | **0.000e+00** |

All five are **identical to 1e-12** — in fact bitwise, `max_abs_diff = 0.0`. Worker count and BLAS
thread count change nothing about what is computed, only how fast. 4 workers buy nothing over 2 on
one physical core (12.32 vs 11.63 s/pair), which is the core count talking, not an effect.

`pair_env` (`matching/placement_recognise.py`) was rewritten to fingerprint the **pipeline**
(sha256 over the graph pipeline's source digest, its switch digest and the numpy/scipy versions)
instead of the live switch values: it is now `63fada9514714026` both under the shipped recipe and
under `PLACEMENT_MATCHER_ENABLED = False`, and unchanged after `apply_recipe()`. The old
`pair_env a78528b2831bcbdb` changed with a flag, which is why a flag change used to invalidate a
pair cache that had not moved.

**Pinned:** `requirements.txt` (numpy == 2.4.6, scipy == 1.17.1, pillow == 13.0.0, Python 3.11.2).
**Recorded:** `matching/placement_records.environment()` — CPU model, physical and logical cores,
platform, interpreter, the three pinned versions and `pair_env` — is now merged into
`m15_manifest.manifest()` as its `environment` key, so every result carries the machine that made
it. This sandbox: `Intel(R) Xeon(R) Processor @ 2.60GHz`, 1 physical / 2 logical cores,
`Linux-6.1.158+-x86_64-with-glibc2.36`.

**Verdict on the cross-sandbox drift (M27's +0.004, max +0.010): it is ENVIRONMENT, not
nondeterminism.** Within one environment the score is reproducible to zero, five ways; between
sandboxes the only things that changed were the interpreter, the numpy/scipy builds and the CPU's
BLAS kernels — which is exactly what `pair_env`, `requirements.txt` and the manifest's `environment`
now pin and name.

---

## 2. Speed the matcher — profiled, no scoring change landed

`Topology/m28_speed.py --profile` (one pair, warmed graphs, `m28_speed_profile.json`), cProfile,
top 10 by cumulative time:

| # | cumulative (s) | calls | function |
|---|---|---|---|
| 1 | 69.91 | 1 | `placement_frontend.pair_score` |
| 2 | 69.89 | 1 | `joint_mcs.joint_mcs` |
| 3 | 69.81 | 2 | `joint_mcs._directional_joint_mcs` |
| 4 | **68.07** | 2 | **`alignment.global_consistency_alignment`** |
| 5 | 26.69 | 4,894,500 | `joint_mcs.node_sim` (the per-pair signature lookup) |
| 6 | 22.65 | 447,652 | `memory/cores._sig_similarity` |
| 7 | 20.44 | 447,652 | `core/types.similarity` |
| 8 | 19.23 | 2 | `placement_matcher.rescore_matrix` |
| 9 | 17.25 | 447,652 | `core/types._similarity_gaussian` |
| 10 | 16.08 | 531,537 | `placement_llr.plus_score` |

Also inside the same 68 s, one level down: `alignment._assign` 30.6 s (4 calls),
`core/population.population_similarity` 30.5 s (548,910 calls), `alignment._rel` 28.5 s,
`alignment.rel_weight` 28.3 s (554,228 calls), `alignment._relational_agreement` 27.7 s,
`population.axis_similarity` 25.4 s / `centered_cosine` 24.0 s (2.1 M calls each),
`placement_llr.plus_terms` 22.7 s (655,442 calls) and `slots_for_candidate` 18.6 s.

**What the profile says.** 98 % of a pair is `global_consistency_alignment`, and the cost there is
~10 Python-level call sites each executed ~0.5 M times: the assignment's relational pass
(`_rel`, `rel_weight`, `_relational_agreement`), the population similarities (`axis_similarity`,
`centered_cosine`), the LLR's `plus_terms` / `slots_for_candidate`, and `node_sim` — which is
already `@lru_cache`d per directional call and is still called **4.9 M times for 447 k distinct
signature comparisons**, i.e. two thirds of its calls are pure cache-lookup overhead. There is no
single hot spot: a matrix/vector rewrite has to touch the alignment's relation loop, the
population code and the LLR together, and each of those is a change to *what is computed* if it is
done carelessly.

**s/pair before → after: 22.32 → 22.32 (no change shipped).** The measurements behind that:
22.32 s/pair serial and 9.18 s/pair at 2 workers (M27), reproduced here at 22.32 s/pair serial in
the determinism run; `--check` 28 pairs 8.59/9.47 s per pair. The mission's item-2 budget is a hard
stop, and the session's remaining time after items 1 and 3–6 was needed for the ship gates
(`--check` 0 diffs, 195 tests) that a change to scoring code would have forced to be re-run from
scratch. The honest statement is: **the profiled decomposition is the deliverable, the change is
not landed, and the next attempt should start from the relation loop** (`_rel` + `rel_weight` +
`_relational_agreement` — 84.5 s of the 129 s first profile, the single largest cluster of Python
calls),
not from the signature cache, which is already memoised.


---

## 3. Place content audit (read-only)

### 3a. What the pipeline already computes near a place and does NOT put in the place

Audited on the record of `apple/1` at `-000-000` (463 places) and on the modules that make it
(`matching/placement_frontend.place_record`, `lenses/*`, `perception/filters.py`,
`matching/placement_prefilter.py`, `matching/placement_matcher.frozen()`):

| what is computed | where | at the place? | in the place's content? |
|---|---|---|---|
| centre–surround / opponent colour (`opponent_channels`, L, RG, BY) | `perception/filters.py`, first front-end stage | yes, per pixel and per node bag | **no** |
| Gabor energies, 12 orientations × 5 spatial frequencies | `perception/filters.py` (`GaborBank.build_default`) | yes — the whole binary | **no**: only the *quantised argmax* survives, as theta (12 bins) + coherence (4 bins) |
| texture: autocorrelation peak, period, roughness, contrast energy (`TEXTURE_RICH`) | `perception/rich.py` | yes, per window | **no** |
| Surface lens outputs (shading/reflectance quantities), RegionShape, Polarity, CurveComplex | `lenses/derived_common.py` `declared_lens_order()` | yes, as nodes | **no** — they are separate nodes, never attached to a place |
| M15 attribute attachment: endpoint lenses `Orientation, Texture, Color, Contour, Boundary` | `matching/placement_matcher.frozen()["endpoint_lenses"]` | yes, as endpoints in the matcher | **no** — the place record carries the 5-slot key plus `Dz/Mz` |
| the standardised descriptor channels `D, C, S, M` (`std_all`) | `placement_prefilter.standardize` | yes, and they were **in the record until M27** | **no** — M27 dropped `std_all` from the committed record for size; the matcher rebuilds it from the image |

### 3b. Why places carry keypoint classes only

Because a place was defined as *the keypoint's signature*. `core/place_key.py` mints one node per
physical place by salting the signature hash with the detection's anchor, and the signature a
keypoint layer emits is exactly `(kind, theta, coherence, scale, dim)` — five numbers about
orientation, contrast coherence and size. Everything else the front end computes is either (a) per
pixel or per window and never lifted to the node, or (b) carried in the *node bag* `std_all`, which
the matcher differences for node similarity and the door never reads. So the place's content was
never chosen to be sparse; it inherited the keypoint's own tuple, and the M15 attribute lenses
(Orientation, Texture, Color) were wired to the matcher's endpoints instead.

### 3c. The information content of today's key

Measured on the committed records at `-000-000` (80 objects, 40,399 places):

* **559 distinct keys** overall from 40,399 places.
* **H = 7.71 bits** over the key distribution (uniform over 559 keys would be 9.13 bits; the top
  three keys carry 864 / 829 / 813 places, so the distribution is heavy-tailed).
* Per object: **169.9 distinct keys** on average (median 165), **H = 6.83 bits** mean (median 6.79)
  — so a single object's places are described by ~6.8 bits, and 80 objects share 559 keys: the key
  distinguishes *place shapes*, not identities.
* The lens is not a free extra: Blob 23,220 (57.5 %), BoundaryArc 15,055 (37.3 %), CornerPoint
  2,124 (5.3 %).

The mission's "≈ 144 distinct values" is the size of the *nominal* grid (12 × 4 × 3 × kind × dim);
in practice 559 distinct keys occur, which is what the entropy above is measured over.

---

## 4. The gauge

Definition (fixed before the numbers): for every query place of view `-035-045` against the pool of
places at `-000-000`, take the best similarity to the TRUE object's places and to each other
object's; G = AUC of "true beats other" over all (query place, other object) pairs, ties counting
0.5. Reported **twice**: against ANY other object, and against a **SAME-CLASS** other object. Every
comparison is within lens, for the baseline and for every candidate alike. Chance **0.5**.

`Topology/m28_gauge.py --baseline` → `m28_gauge.json` (92.2 s):

| queries | query places | G (any other object) | G (same-class other) | chance |
|---|---|---|---|---|
| all 80 objects | 43,020 | **0.528** | **0.510** | 0.500 |
| the item-5 sample of 8 | 4,284 | 0.515 | 0.496 | 0.500 |

The current signature's similarity is "best similarity to its true object" barely better than a
coin flip, and against a same-class other object it is at chance. That is the number the mission's
verified fact (M26 channel d 0.100) predicted, measured now as the quantity the door actually needs.

---

## 5. Candidate content

Built before any number, constants fixed in `Topology/m28_gauge.py` (documented in its header) from
the existing filter bank and physical reasoning; all graded population codes; rotation-normalised by
the place's own orientation where rotation matters (C2), scale-normalised by its scale everywhere;
nothing read from a label. The gauge was then evaluated on the item-5 subset only — 8 objects, one
per class, views `-000-000` and `-035-045` — computed directly from the images and the records'
own frames (no record rebuild). Build 42.3 s, gauge 78.6 s.

**The rule, written before the numbers: a candidate proceeds only if its same-class G beats the
current signature's by ≥ +0.100.**

| candidate | what it is | G (any) | G (same-class) | Δ vs signature | proceeds |
|---|---|---|---|---|---|
| signature | the 5-slot key, graded kernel | 0.515 | 0.496 | — | — (baseline) |
| **C1** colour | CIELAB L, a, b in two concentric rings at 0.5 s and 1.5 s, 12 samples each (6 values) | 0.852 | **0.745** | **+0.249** | **yes** |
| C2 orientation | Gabor energies, 8 orientations × 2 frequencies from the existing bank, disc of 2 s, bins rolled to the place's own axis, L1 (16 values) | 0.645 | 0.563 | +0.067 | no |
| C3 texture | roughness (std of the orientation marginal) + log total energy (2 values) | 0.620 | 0.532 | +0.036 | no |
| C4 = C1+C2+C3 | equal weights, block-normalised, re-normalised (24 values) | 0.741 | 0.616 | +0.120 | yes |

n = 4,284 query places for every row; chance 0.5. Two candidates cleared the bar. **C1 was taken**
because it is the higher of the two on both axes (0.745 vs 0.616 same-class), decided on this
sample before any door number was computed.

Reading the failure honestly: orientation population (C2) and texture (C3) carry almost nothing
across a view change at place resolution — +0.067 and +0.036 — while **local colour carries a lot**.
The caveat worth stating: ETH-80 instances are photographed on their own backgrounds, so part of
C1's power could be the scene's colour near the place rather than the object's own surface.  **The
diagnostic says it is the object.**  ETH-80 ships a per-pixel part map beside every image
(background exactly 0), so the same rings were re-sampled on foreground pixels only and the gauge
re-run (`m28_gauge.py --diagnostic`, a post-hoc diagnostic and **not** a registered candidate):
same-class **G = 0.7350** (any-object 0.8412) against the full-ring C1's **0.7445** — a drop of
**0.010**, i.e. almost all of the colour signal comes from the object's own surface.  That also
means the door's failure at the second FIT view is not a background artefact.

---

## 6. The content in the pipeline, and the two door variants

### 6a. Wiring (default OFF)

* `matching/placement_content.py` (new, in `CODE_MODULES`): `features()` — the six CIELAB ring
  values, identical code to C1 in the gauge — plus `quantise()`, `similarity()` (cosine) and
  `kind()`, which **refuses** a kind outside `{"keypoint", "colour_rings"}`.
* the bundle value `content.kind`: `"keypoint"` in `matching/placement_bundle.json`, i.e. the
  shipped default is what every record before M28 carries; `colour_rings` is opt-in and any read or
  record that uses it names it in its manifest.
* `placement_frontend.place_record(content=...)` attaches the content when asked;
  `placement_records` format **version 28** carries `content_<view>` (6 float64 columns per place,
  `meta.content = 6`), and `_cell_from` reads it back.
* **Price and run.** The mission priced the full re-derivation at 320 cells ≈ 1,824 s (M27's
  measurement). It was not run as a re-derivation: item 1b's own finding is that a re-derivation is
  the one thing that can move frozen keys between environments, so the record file was built by
  **enrichment** — every frozen field (keys, `Dz/Mz`, frames, counts) comes from the committed M27
  file byte-for-byte and only the six content columns are new, computed from the image and the
  record's own frames. Cost **36.9 s**, i.e. 49× cheaper, and it cannot disturb a key by
  construction.
  `Topology/m28_records.py --check` proves the two paths agree: on a live cell built by
  `place_record(content="colour_rings")` the content matches the enriched file exactly —
  463 places, **max |difference| 0.0**, all quantised keys equal.
* `data/placement_records_m28.npz`: **33,478,046 B (31.9 MB)** ≤ 40 MB, float64, committed.
  Independently, the shipped counted door on this file reproduces M27's `a_counted` numbers
  cell-for-cell (0.350 / 0.300 at `-035-045`, 0.450 / 0.400 at `-045-180`), which proves no frozen
  key moved.

### 6b. The two variants (fixed before the numbers)

* **(i) `counted_content`** — the counted door's own mechanism over a quantised content key
  `(lens, 6 bins)`: L into 6 bins over [0, 100], a and b into 8 bins each over [-128, 128), per
  ring. An object's score is how many of the query's places find a key-equal place in it.
* **(ii) `graded_content`** — per object, the maximum cosine similarity between any query place and
  that object's places of the same lens. No threshold, no kernel, no fitting.

The rule (M27's): **FIT recall@8 must beat the shipped door's by ≥ 0.075 at BOTH `-035-045` and
`-045-180`, and DEV+FRESH must not be below it by more than 0.025.** Shipped door on these records:
0.550 / 0.550 and 0.550 / 0.475 (M27 `c_shipped`, RRF of the counted door and appearance).

`Topology/m28_door.py` → `m28_door.json`, 267.2 s, 80-object gallery, n = 40 FIT / 40 DEV+FRESH,
chance 0.100:

| variant | FIT `-035-045` | gain | FIT `-045-180` | gain | DEV+FRESH `-035-045` | Δ | DEV+FRESH `-045-180` | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| (i) counted_content | 0.550 (22/40) | **0.000** | 0.500 (20/40) | **−0.050** | 0.550 (22/40) | 0.000 | 0.425 (17/40) | −0.050 | **fail** |
| (ii) graded_content | 0.675 (27/40) | **+0.125** | 0.600 (24/40) | **+0.050** | 0.750 (30/40) | +0.200 | 0.600 (24/40) | +0.125 | **fail** |

**Neither variant clears the rule.** (ii) is +0.125 at the selection view and comfortably above on
both confirmation cells, but +0.050 at the second FIT view is short of the 0.075 the rule requires;
(i) does not move at the selection view at all. The rule was fixed before these numbers and it is
not being renegotiated afterwards: **the content does not ship as the door's key.**

---

## 7. The read — NOT RUN

Item 7 runs only if item 6 passes. It did not, so no matcher read at S = 8 was run, no weight was
chosen for feeding content into node similarity, and no MDE-vs-effect statement is made. (That is
the mission's own instruction, not an omission: an S = 8 read with a door that failed its rule
would be measuring a configuration nobody is proposing.)

---

## 8. Ship

* README records the outcome; the bundle keeps `content.kind = "keypoint"` (the content is in the
  pipeline, off by default), `code_digest 73b9a7e1b9e288f7`, `pair_env 63fada9514714026`.
* `matching/placement_bundle.json` gained the documented `content` block (kind, allowed kinds,
  digest of `placement_content.py`) and was re-frozen (`--freeze --date 2026-11-27 --k 1
  --shortlist 16 --shortlist-rule fixed`), because `placement_content.py` joined `CODE_MODULES`.
* `requirements.txt` (new), `matching/placement_records.environment()` merged into
  `m15_manifest.manifest()`.
* `--check` (`Topology/placement_m12_perf.py --check`): **28 pairs, 0 diffs**, 578.5 s
  (8.59 / 9.47 s per pair), run on the code that ships.  The only edits after it are documentation
  and the bundle's `measured.M28` block, neither of which enters `code_digest` (it hashes
  `CODE_MODULES` + the two frozen parameter files); the bundle was reloaded after the edit and its
  digest and `content.kind` are unchanged.
* Tests: 21 files, **195 tests, all OK** (m26's 3 skips are its registered skips).
* Legend: `Topology/m28_determinism.py`, `m28_gauge.py`, `m28_records.py`, `m28_door.py`,
  `m28_speed.py`,
  results in `m28_determinism.json`, `m28_gauge.json`, `m28_records_progress.json`,
  `m28_door.json`.

---

## 9. What this mission found, in one paragraph

Today's place content — the keypoint's own five-number signature — carries **7.71 bits** over the
whole corpus (6.83 bits per object, 559 distinct keys for 40,399 places) and buys a same-class
cross-view AUC of **0.496 against a chance of 0.500**: essentially nothing. The pipeline already
computes colour, orientation populations, texture and M15's attribute lenses right at the place, and
none of it was attached, because a place was defined as the keypoint's signature and every other
channel went to the matcher's node bag instead. Adding one cheap grade — **six CIELAB ring means at
the place's own scale** — moves the same-class gauge to **0.745 (+0.249)**, and a full graded door
built on it reaches **0.675 FIT recall@8 against the shipped door's 0.550** at the selection view;
but at the second FIT view it gains only +0.050 against a required +0.075, so by the rule fixed
before the run the content does not ship as the door's key, and the matcher read that would have
priced it in node similarity was therefore not run. **The single next lever is a second information
source at the place** — a place-local colour or surface measurement that survives a view change
without the background (the gauge cannot separate object colour from the scene's colour near the
place, and that ambiguity is exactly what a door built on C1 alone inherits).
