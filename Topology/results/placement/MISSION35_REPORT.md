# MISSION 35 — the object-level descriptor, the class test, the door, and what the matcher adds

**Question.** Every channel this tree has shipped compares ONE PLACE with ONE PLACE and takes a max.
M32 measured what that does: the place-level colour cosine puts 97.8 % of pairs at ≥ 0.999 and shape
97.4 % — a max over almost-equal numbers ranks on noise.  Mission 35 asks a different question.  If an
object is a DISTRIBUTION over its places, does the distribution carry identity, does it carry CLASS,
does geometry cluster by class where colour fails, and what does the matcher add on top?

**Answers, in one screen:**

* **A colour histogram over an object's places is a better door than every channel the tree has
  shipped.**  Door recall@8 on the FIT half: **1.000 at `-035-045`, 0.925 at `-045-180`** (chance
  0.100), against **0.800 / 0.650** for the shipped fused three-channel + fragment door and **0.675 /
  0.650** for the fused door without the fragment.  Mean FIT gain **+0.2375**, object-clustered CI
  **[0.1375, 0.3375]**, confirmation **+0.300**.  The bar was +0.100.  The door is one channel, one
  group, three histogram blocks.
* **Geometry clusters by class, and it is beaten by colour at every view this mission reads.**
  GEOMETRY (LAYOUT+CFG, no appearance at all) class AUC **0.732 / 0.679 / 0.622** with CI lower bounds
  **0.695 / 0.641 / 0.583** at `-022-000` / `-035-045` / `-045-180` — both hard views clear the
  registered 0.55 floor.  COLOUR reads **0.728 / 0.738 / 0.728**.  Where colour fails, geometry wins
  loudly: **car** colour 0.48/0.51/0.48 vs geometry 0.82/0.90/0.82; **pear** the other way (colour
  0.90/0.90/0.89 vs geometry 0.77/0.54/0.44).
* **The blocks are complementary.**  ALL − COLOUR = **+0.053 [+0.038, +0.069]** and **+0.055
  [+0.037, +0.074]** at the two hard views: both exclude zero.  ALL reads **0.791 / 0.783** class AUC.
* **The matcher does not earn FINAL on this door.**  See §6: the door already recalls 1.000 / 0.925 of
  the FIT half, and the paired matcher comparison costs 640 pairs (~6 h on this box).  The rule fires
  on what it can fire on — the door's order is final and the matcher ships as flag `verify`.
* **The fragment channel's M34 win is withdrawn and re-attributed.**  On top of the bag door it adds
  **0.000 FIT recall@8** (bar +0.025), and its own ablation says why: the descriptor's 22 fields beat
  nothing — **colour means alone** are 33/40 at `-035-045` against 28/40 for all 22, and pure
  **geometry** (fields 0-7) is 11/40 ≈ chance.  "Composition" was not the carrier; **part-level
  colour means** were.
* **The tree is closed**: 12 orphan modules (**6,143 lines**) moved to `archive/` with reasons, 6 left
  in place with reasons, parameters untouched, lens registry identical, `--check` 0 diffs.

Chance, n and CI are stated at every number.  All CIs are object-clustered, 10,000 resamples, seed
20261118.  Nothing here is a weight sweep: every arm, bin count, group and arm weighting was written
down in `matching/placement_bag.py` before a number was taken.

---

## 1. The descriptor (`matching/placement_bag.py`), and the item-1 replication

**What an object is.**  A bag of four groups over its places, all read from the committed place
records (`data/placement_records_m32.npz`) — no image, no graph, no label:

| group | blocks | size | what it measures |
|---|---|---|---|
| COLOUR | `ab1` 5×5, `ab2` 5×5, `L1` 6 | 56 | the two CIELAB ring means each place carries (scaled Lab), as the ab plane of ring 1, the ab plane of ring 2 and the ring-1 lightness |
| SHAPE | `key` 3×12×4×4, `eig` 3×6, `resp` 3×6 | 612 | the record's own quantised key (lens × theta × coherence × scale quantile) plus eigenvalue ratio and response, six bins per lens |
| LAYOUT | `scale` 9, `radial` 8, `d2` 12, `orient` 12 | 41 | log2 place scale, radial distance from the object's centroid in units of its own RMS radius, the D2 distribution of pairwise place distances, and each place's orientation **relative to the object's dominant axis** (mod π) |
| CFG | `cfg` 6×6 | 36 | the 24-slot configuration row, read as six groups of four neighbour ranks |

**The 24 config slots, stated from the build path (not from a row).**
`matching/placement_context.config_descriptor(side, u, k)` returns *(sorted log distances, sorted
source-frame directions)*; `matching/placement_prefilter.bag_v2` stores `D` at `K = 24` with a mask
`M` = 1 where that rank exists, and `standardize` divides `(D − μ_D)/σ_D` slot by slot
(`placement_prefilter_params.json` carries 24-long `mu_D`/`sd_D`/`support_D`).  **So slot j is the
standardised log distance to the (j+1)-th nearest neighbour**, sorted ascending, and the mask says
which ranks a place actually has.  (The mission brief's fallback reading — 12 log distances then 12
angles — is wrong: `A` never enters `D`.)  The 24 slots are read as **six groups of four ranks**
(1-4, 5-8, …, 21-24); for each group and place we take the mean of the LIVE standardised values — a
"shell distance" — and histogram that over the object's places into 6 bins.

**Excluded, with a test.**  `var_x` (pinned at `_parabola`'s clip), `var_y` and `rel_angle` (≈ 0 by
construction) are never read; `tests/test_m35_bag.py::test_excluded_columns_are_not_read` corrupts all
three columns to ±1e6/±3.0 and asserts every block is bit-identical.

**Normalisation and aggregation.**  Each block is L1-normalised then square-rooted (Hellinger), so a
block dot product is a Bhattacharyya coefficient; a GROUP's score is the mean of its blocks' dot
products; an ARM's score is the mean over its groups (ALL is the mean over all 11 blocks, G4 the mean
over the 4 groups — that is the only difference between the two pre-registered arms).

**Bin edges.**  Quantiles of the REFERENCE view's own values (`-000-000`, 80 objects, 40,399 places),
label-free, stored once in `matching/placement_bag_edges.json` with provenance; a missing file is a
refusal (`load_edges`), never a silent refit.  The file's sha256 prefix is recorded in the bundle.

### The item-1 table (instance task: gallery = every object at `-000-000`, query = the cell's view)

80 objects, chance top-1 = 0.0125, chance recall@8 = 0.100.

| arm | `-022-000` top1 / r@8 | `-035-045` top1 / r@8 | `-045-180` top1 / r@8 |
|---|---|---|---|
| COLOUR | **65/80** / 80/80 | **58/80** / 80/80 | **43/80** / 76/80 |
| SHAPE | 47/80 / 74/80 | 17/80 / 48/80 | 21/80 / 48/80 |
| LAYOUT | 19/80 / 46/80 | 7/80 / 26/80 | 5/80 / 19/80 |
| CFG | 8/80 / 24/80 | 8/80 / 31/80 | 3/80 / 17/80 |
| GEOMETRY | 21/80 / 45/80 | 13/80 / 37/80 | 8/80 / 23/80 |
| COLOUR-FREE | 39/80 / 69/80 | 17/80 / 49/80 | 17/80 / 47/80 |
| ALL | 64/80 / 80/80 | 48/80 / 78/80 | 37/80 / 69/80 |
| COLOUR+SHAPE (equal) | 68/80 / 80/80 | 54/80 / 79/80 | 42/80 / 75/80 |

### Replication against the VERIFIED list — and the cause of every difference

| number | VERIFIED | this tree | difference |
|---|---|---|---|
| colour top-1 at `-035-045` | 48/80 | **58/80** | +10 |
| colour recall@8 at `-035-045` | 79/80 | 80/80 | +1 |
| colour top-1 at `-022-000` | 59/80 | 65/80 | +6 |
| colour top-1 at `-045-180` | 38/80 | 43/80 | +5 |
| shape top-1 / recall@8 | 10/80, 40/80 | 17/80, 48/80 | +7, +8 |
| class AUC colour | 0.733 / 0.743 / 0.732 | 0.728 / 0.738 / 0.728 | −0.005 everywhere |
| class AUC geometry | 0.723 / 0.659 / 0.627 | 0.732 / 0.679 / 0.622 | +0.009, +0.020, −0.005 |
| class AUC colour-free | 0.773 / 0.695 / 0.705 | 0.787 / 0.733 / 0.721 | +0.014, +0.038, +0.016 |
| class AUC all | 0.816 / 0.783 / 0.783 | 0.806 / 0.791 / 0.783 | −0.010, +0.008, 0.000 |
| colour+shape ≤ colour at the hard views | holds | **holds** (54 vs 58; 42 vs 43) | — |

**The cause is measured, not guessed: the bin edges are a degree of freedom worth 22 objects.**  The
mission's rule ("bin edges from the gallery's reference view") is a QUANTILE ruler, which equalises
bin mass; the VERIFIED run's edges were a different (range) choice.  Holding everything else fixed and
swapping only the colour edges at `-035-045`: **quantile edges 58/80 top-1, range edges 36/80**
(recall@8 80 vs 75).  A ±10-object difference from the VERIFIED table is therefore INSIDE the
descriptor's own edge sensitivity, and the choice is now pinned by a committed file rather than left
in the air.

What matters is that the ORDERING of channels replicates exactly, because every later item rests on
it: **colour ≫ shape ≫ layout ≥ CFG**, colour+shape never beats colour, and the class AUCs agree to
0.005-0.038.

---

## 2. Class-level representation test (pre-registered arms, own object excluded)

AUC = the pair-wise P(same-class object scores above other-class object) per query, own object
excluded, averaged over 80 queries — a different and harder question than "is the nearest neighbour
same-class" (chance 0.5 here, 9/79 = 0.114 there; both reported).  n = 80, 10,000 object-clustered
resamples.

| arm | `-022-000` AUC [CI] | nn-class | `-035-045` AUC [CI] | nn-class | `-045-180` AUC [CI] | nn-class |
|---|---|---|---|---|---|---|
| COLOUR | 0.728 [0.687, 0.766] | 0.575 | 0.738 [0.697, 0.775] | 0.600 | 0.728 [0.689, 0.764] | 0.537 |
| SHAPE | 0.778 [0.738, 0.815] | 0.600 | 0.713 [0.670, 0.754] | 0.412 | 0.734 [0.697, 0.771] | 0.550 |
| LAYOUT | 0.776 [0.745, 0.807] | 0.425 | 0.738 [0.697, 0.780] | 0.362 | 0.655 [0.608, 0.698] | 0.325 |
| CFG | 0.624 [0.590, 0.656] | 0.388 | 0.570 [0.537, 0.604] | 0.287 | 0.564 [0.528, 0.600] | 0.237 |
| **GEOMETRY** | **0.732 [0.695, 0.767]** | 0.388 | **0.679 [0.641, 0.716]** | 0.400 | **0.622 [0.583, 0.659]** | 0.287 |
| COLOUR-FREE | 0.787 [0.750, 0.822] | 0.600 | 0.733 [0.694, 0.772] | 0.463 | 0.721 [0.681, 0.761] | 0.487 |
| ALL | 0.806 [0.771, 0.837] | 0.600 | 0.791 [0.755, 0.824] | 0.588 | 0.783 [0.749, 0.814] | 0.500 |
| G4 | 0.802 [0.768, 0.833] | 0.637 | 0.788 [0.753, 0.820] | 0.575 | 0.782 [0.748, 0.812] | 0.537 |

**Claim 1 — "geometry clusters by class"** (registered as: GEOMETRY AUC CI lower bound > 0.55 at BOTH
hard views): **HOLDS.**  0.641 at `-035-045`, 0.583 at `-045-180`.  It holds with the least margin at
the view where the object has turned furthest, which is the shape of a real but viewpoint-fragile
signal: **0.732 → 0.679 → 0.622** as the query rotates 0° → 35°/45° → 45°/180°.

**Claim 2 — "complementary"** (ALL − COLOUR CI excludes 0 at both hard views): **HOLDS.**
+0.053 [+0.038, +0.069] and +0.055 [+0.037, +0.074] (and +0.078 at `-022-000`).  ALL is the best arm
at every view.

**Claim 3 — "geometry wins where colour fails", per class** (same-class AUC at the hard view):

| class | COLOUR | GEOMETRY | winner at `-035-045` | reading |
|---|---|---|---|---|
| car | 0.48 / 0.51 / 0.48 | **0.82 / 0.90 / 0.82** | LAYOUT 0.92 (COLOUR-FREE 0.92) | geometry wins by +0.39; colour is at chance |
| pear | **0.90 / 0.90 / 0.89** | 0.77 / 0.54 / 0.44 | COLOUR 0.90 | colour wins by +0.36 |
| cup | 0.69 / 0.69 / 0.69 | **0.85 / 0.80 / 0.64** | COLOUR-FREE 0.90 | geometry wins |
| cow | 0.66 / 0.67 / 0.67 | 0.60 / 0.57 / 0.55 | SHAPE 0.85 / 0.81 / 0.78 | the key bag wins |
| tomato | 0.93 / 0.94 / 0.90 | 0.84 / 0.81 / 0.78 | COLOUR 0.94 | colour wins |
| apple | 0.77 / 0.78 / 0.77 | 0.65 / 0.61 / 0.63 | ALL 0.83 / 0.83 / 0.80 | both, together |
| dog | 0.73 / 0.73 / 0.73 | 0.71 / 0.67 / 0.67 | G4 0.83 / 0.82 / 0.82 | both, together |
| horse | 0.66 / 0.68 / 0.69 | 0.62 / 0.52 / 0.44 | ALL / COLOUR 0.68-0.71 | weak for everyone |

**Mechanism.**  The two classes where geometry carries the answer are the two whose COLOUR is nearly
identical across instances — cars share a paint-and-glass palette and a low, wide place cloud; pears
share a shape and a hue.  GEOMETRY's own failure mode is the same coin: LAYOUT and CFG read the
object's dominant axis and its shell distances, which rotate with the viewpoint (LAYOUT 0.776 → 0.738
→ 0.655 across the three views).

---

## 3. The door

`door.kind = "bag"`.  Candidates, all on the same cells (gallery = `-000-000` records, queries = the
cell's view, 80 objects, FIT = 40 even-index objects, confirmation = the other 40):

| candidate | `-035-045` r@8 FIT / top-1 | `-045-180` r@8 FIT / top-1 | conf r@8 (mean) |
|---|---|---|---|
| **(i) COLOUR (bag)** | **1.000 / 0.625** | **0.925 / 0.500** | **0.988** |
| (ii) RRF(bag COLOUR 1.0, counted 1.0, appearance 1.0, colour 0.5, fragment 0.5) | 0.950 / 0.450 | 0.800 / 0.425 | 0.900 |
| (iii) shipped fused3 door (+fragment 0.5) | 0.800 / 0.275 | 0.650 / 0.300 | 0.688 |
| (iii′) fused3 without the fragment | 0.675 / 0.250 | 0.650 / 0.275 | 0.625 |

**Choice (pre-registered: FIT recall@8, then top-1, ties → fewer channels): COLOUR alone.**  Mean FIT
gain over the current door **+0.2375**, paired object-clustered CI **[0.1375, 0.3375]** (n = 80 object-
views), confirmation **+0.300** (tolerance −0.025).  The bar (+0.10) is cleared by 2.4×.

Two things worth recording.  First, **adding the old channels HURTS the new descriptor** (0.950 / 0.800
against 1.000 / 0.925): an RRF fusion cannot improve on a channel that is already right, and it can
only dilute it.  Second, the reproduced (iii) row is a cross-check on M34 — 0.800 / 0.650 is exactly
the number `m34_door.py` reported, computed here from the shipped `placement_fragment.channel` and the
regenerated M32 channel cache.

Wiring: `recognise()` computes the descriptor for the query and reads the gallery's cached bag
features; `build_gallery()` builds and caches them once.  A query given as a path is enriched the same
way a record is (content from the front end, graded shape columns from the image via
`placement_shape.attach`); a record missing any field REFUSES (`BagRefusal`), never zero-fills.  The
fused three-channel door stays implemented as the legacy path and `tests/test_m31_wire.py` now
exercises it explicitly, so its arithmetic is still pinned.

---

## 4. The class output

`recognise_class(query, gallery, top=2)` — class score = the **mean of the class's top-2 instance
scores**, descriptor ALL (equal block weight), own object excluded.

| view | class top-1 | CI | mean rank of the true class | item-2 ALL nn-class |
|---|---|---|---|---|
| `-022-000` | **0.650** (52/80) | [0.550, 0.750] | 0.61 | 0.600 |
| `-035-045` | **0.575** (46/80) | [0.462, 0.688] | 0.80 | 0.588 |
| `-045-180` | **0.500** (40/80) | [0.388, 0.612] | 0.91 | 0.500 |

Chance is 1/8 = 0.125 (nine classes, one class supplying two objects in a few folds: 9 distinct
classes).  Against **M23's 0.500**: the top-2 mean rule is at or above it at every view and above it
at `-035-045` (0.575), but its CI includes 0.500 — the difference is **not established**.
**Protocol differences** from M23: M23 scored classes through the place-level door's counts with the
whole tree's matcher state, on the then-shipped gallery, and reported a single pooled number with no
CI; this reads a different descriptor (object bag, ALL), excludes the query's own object explicitly,
and bootstraps over objects.  Shipping decision: the entry point ships (it is the product's class
answer and it is at least M23's), with the CI stated on the row rather than a claim of improvement.

---

## 5. Addendum to M34: the fragment channel, re-measured on top of the bag door

**5a. Does it still earn its vote?**  RRF(bag COLOUR 1.0, fragment 0.5) against bag COLOUR alone:

| view | bag r@8 FIT | + all-22 fragment | gain | paired CI |
|---|---|---|---|---|
| `-035-045` | 1.000 | 1.000 | **+0.000** | [0.000, 0.000] |
| `-045-180` | 0.925 | 0.925 | **+0.000** | [−0.075, +0.075] |

Mean +0.000 against a +0.025 bar (**top-1 also falls**: 0.625→0.600 and 0.500→0.350).
**Verdict: bundle weight 0.0.**  Code and anchors stay (nothing is deleted); the door no longer reads
the channel, and `fusion.fragment` records the measurement.  Note this is not a contradiction of M34:
M34 added the channel to a WEAKER door (0.675 / 0.650) where it had headroom and it bought +0.125 at
one view; the bag door has already claimed that headroom.

**5b. What the channel was actually carrying** (channel-alone recall@8 on the FIT half, n = 40, chance
0.100; each variant restricts the descriptor's fields and keeps the grouping, kernel and mean-of-best):

| variant | fields | `-035-045` | `-045-180` |
|---|---|---|---|
| all22 (shipped) | 0-21 | 28/40 = 0.700 | 25/40 = 0.625 |
| **colour means** | 16-21 (lab0…lab5) | **33/40 = 0.825** | 24/40 = 0.600 |
| curvature | 8-15 | 24/40 = 0.600 | 26/40 = 0.650 |
| geometry | 0-7 | 11/40 = 0.275 | 11/40 = 0.275 |
| no colour | 0-15 | 23/40 = 0.575 | 18/40 = 0.450 |

**"Composition" is withdrawn.**  M34's report attributed the channel's win to part-level arrangement
(size, aspect, member spread, alignment).  The ablation says otherwise: the eight arrangement fields
alone are at **chance** (0.275), removing colour costs 20 points at the hard view (0.575 vs 0.825), and
the six colour means ALONE beat the full 22-field descriptor at `-035-045`.  The channel's signal is
**part-level colour means**: a fragment that covers the same-coloured part matches, and it is the
colour, not the arrangement, that transfers across views.  (Cross-check against the VERIFIED list,
which was over all 80 queries: all-22 58/51, colour-means 64/52, curvature 48/47, geometry 17/20,
no-colour 39/36 — the same ordering, twice the counts.)

---

## 6. What the matcher adds

**Cost first, as the mission requires.**  The door's FIT shortlists are 40 queries × 8 candidates × 2
views = **640 uncached pairs**; a pair costs **19-48 s** on this box (`_score_one` timed at 48.3 s cold,
19.4 s warm), i.e. **≈ 4-6 hours of one physical core**.  There is no pair cache in a fresh checkout
(`.cache` is not committed), so the count is the real price.  The job runs in the background and is
resumable row by row (`Topology/results/placement/m35_pairs.jsonl`, keyed `object|view|candidate`); the
progress file is the artifact.

**The rule, and its verdict.**  The matcher stays FINAL only if its order (or an equal-weight RRF of
its order and the door's) beats the door by **≥ +0.075 FIT top-1** with confirmation not below by more
than 0.025.  **The comparison could not be completed inside the mission's compute window** (progress at
the end of the session is in `Topology/results/placement/m35_pairs.jsonl`; `--matcher-report` prints
the verdict for whatever is complete, and refuses to call a partially-covered view a measurement).
**Therefore the door's order is final and the matcher ships as flag `verify`.**

**What the matcher is proven for, and what it does not add here.**  It is proven for place
correspondence: M19 measured **+0.248** over appearance alone on pairwise correspondence, and M34
measured it fixing **5 of 8** shortlist errors with **0** broken (all five same-class impostors) on a
door that recalled 0.675/0.650.  On the bag door it has almost nothing left to do: the door's FIT
recall@8 is **1.000 / 0.925**, so at `-035-045` every truth is already inside the eight and the
matcher can only reorder; the bar (+0.075 of top-1) is 3 objects per view.  That is the honest
statement the README carries.

---

## 7. Closing the tree

**The census after the archive** (`Topology/m34_audit.py --run`, re-run): **113 → 101 modules**,
**78,587 → 72,980 lines** (the 6,143 moved lines are out of the tree, 536 in because M35 adds
`matching/placement_bag.py`).  Reachable: **SHIPPED 89 / 65,817 = 90.2 %** (M34: 88 / 65,281 = 83.1 %;
the percentage rises because the denominator lost more than the numerator).  Orphan: **7 / 4,148 =
5.7 %** (M34: 19 / 10,291 = 13.1 %), and the seven are exactly the six this mission deliberately left
plus the `lenses/` module it was told not to touch: `perception.surface_fragments` 2,638,
`lenses.surface_fragment_lens` 396, `perception.frames` 382, `matching.signature_matrix` 377,
`matching.calibration` 185, `eth80_split` 127, `sensory_floor` 43.  HARNESS-ONLY 4 / 1,831,
TEST-ONLY 1 / 1,174, SHIPPED-STR 1 / 10.

**12 modules moved to `archive/`, in three batches, suite green after each** (`git mv`, history
intact): `matching/relation_walker` (1,494), `matching/fuzzy` (385), `matching/consensus` (305),
`matching/search` (244), `matching/m15_places` (197), `matching/canonical_order` (88) →
`archive/matching/`; `perception/relatable_completion` (1,181), `perception/contour_relations` (790),
`perception/ring_hypothesis` (740), `perception/psychophysics` (486), `perception/surface_fragment_cache`
(151) → `archive/perception/`; `core/junctions` (82) → `archive/core/`.  Total **6,143 lines**.

**Each was checked before moving**: its dotted name AND its bare name were grepped in `.py`, `.json`
and `.md` across the tree (excluding `archive/`, `.cache`, `.venv`); its imports were resolved by AST
(relative imports resolved against the file's package), because a bare-name grep alone cannot tell
`import fuzzy` from the word "fuzzy".  The two `__import__` call sites the mission names
(`placement_recognise.py:252`, `placement_records.py:313`) import only `numpy`, `scipy` and `PIL` —
they name no tree module.  The lens registry listing is unchanged: `core.registries` carries the same
tables and the same lens/population names before and after (`CONSTITUTIONAL_RANGES` 67,
`POPULATION_AXES` 31, `POPULATION_SLOTS` 23, `CIRCULAR_CYCLE_LENGTHS` 31).

**6 left in place, with reasons** (all would leave a KEPT module with a broken import):

| module | lines | why it stayed |
|---|---|---|
| `perception/surface_fragments` | 2,638 | imported by `lenses/surface_fragment_lens.py`, and `lenses/` is excluded from this mission's candidates |
| `perception/frames` | 382 | imported by `lenses/surface_fragment_lens.py` (same reason) |
| `matching/signature_matrix` | 377 | imported by `memory/cores.py` |
| `matching/calibration` | 185 | imported by `memory/cores.py` |
| `eth80_split` | 127 | imported by `eth80_frontend.py` (the corpus's own split loader) |
| `sensory_floor` | 43 | imported by `concept_graph.py`, `consolidator.py` and `dynamics/activation.py` |

**Parameters untouched.**  No file under `parameters/` was edited; the bundle records the new door.
A generated list of the flags nothing reads (225 named nowhere, from the M34 census) is recorded in
`archive/ARCHIVE_INDEX.md` as **legacy**.

---

## 8. The gates

| gate | result |
|---|---|
| full suite | **272 passed, 6 skipped, 4 subtests** (3:30-5:40, one core, two background jobs running) |
| suite after each archive batch | green (272 / 272 / 272) |
| records `--check` (`m32_records.py --check --cells 6`) | 24 cells, keys + frames identical, cosine max Δ 1.0e-15, `ok: true` |
| anchors `--check` (`m34_anchors.py`) | 640 cells, 0 diffs (see `Topology/results/placement/m34_anchors.json`) |
| determinism, 1 / 2 / 4 workers | door order identical; matcher scores **max abs diff 0.0** |
| bundle | re-frozen, `code_digest` matches the tree, `door.kind = "bag"`, `fusion.fragment = 0.0` |
| new tests | `tests/test_m35_bag.py` (9 tests): arity, Hellinger, exclusions, both refusals, edges provenance, bundle, door == item-1 ranking on all 160 cells, class output |

---

## 9. Limits, stated

* The benchmark is **80 objects, 8 classes, 10 instances, plain backgrounds** (ETH-80).  Nothing here
  is measured on a real scene, on clutter, or on an object photographed at a distance where the places
  are the background's.
* **Operating range**: the descriptor is a distribution over `n` places.  Objects with fewer than
  ~40 places have histograms with few occupied bins (the record's own `drop_` and the `no_config`
  refusal keep those out of the door's gallery); the door has not been read below that.
* The colour blocks read CIELAB means around each place, so a **lighting change** moves every bin.  No
  white-balance or exposure invariance was measured.
* The class output is a **retrieval** answer over a fixed gallery, not a classifier: it needs the
  gallery to contain the class, and it inherits the gallery's class balance.
* The matcher's contribution on this door is **unmeasured** (§6) — a named ceiling, not a claim.

## 10. The single next lever

**A second dataset where colour does not solve identity** — the class table says why: colour already
answers 0.69-0.94 for six of nine classes, so a benchmark whose classes differ in pigmentation cannot
separate "geometry clusters by class" from "colour clusters by class".  The lever is a corpus of
same-coloured, differently-shaped instances (or real photographs of one object family under a fixed
light) where the colour arm is at chance by construction and GEOMETRY's 0.62-0.73 becomes the
difference between a working class answer and a coin.

---

## 11. Addenda, recorded so the README's claims have one home

**M28's gauge (saturation).**  M28 measured the shape channel with a same-class gauge and found it
**saturated**: 0.505 with the shipped clip against 0.596 with the columns unclipped, i.e. the
instrument, not the object, was setting the ceiling.  That is why M32 re-read shape through the
unclipped `sharp_*`/`curv_*` columns (keys byte-identical, `SHAPE_FIELDS` 5 → 9) and why M35's SHAPE
block reads the record's graded columns rather than the clipped ones.

**M31 / M32 / M33's shape verdict (instrument).**  Shape never cleared a registered bar by measurement:
M31 +0.0909 against +0.10; M33's object-level curvature +0.0375 with a CI **[0.000, 0.0875] including
zero** against +0.050, with the mechanism named as the COLUMNS (80-99 % of every curvature column's
variance is within an object, ICC 0.009-0.197); and M34's shape re-ranker at ρ = 0.016 against the
matcher's order, FIT top-1 −0.125 [−0.225, −0.025].  M35 does not re-open that verdict: SHAPE ships as
a BLOCK in the class descriptor (its class AUC is 0.713-0.778, the best single group at two of three
views) and never as a door channel, which is exactly what those three measurements support.

**M34's composition claim (withdrawn here).**  See §5b: the fragment channel's win is carried by
**part-level colour means**, not by part arrangement; the eight arrangement fields alone are at chance
(0.275) and the six colour means alone beat all 22 (0.825 vs 0.700 at `-035-045`).  The M34 report's
sentence about "composition" should be read as withdrawn; the channel's numeric results stand, and its
bundle weight is now 0.0 because it does not add on top of the bag door.

---

## Closing paragraph

**Does geometry cluster by class, and where does it beat colour?**  Yes, weakly and by measurement:
GEOMETRY — place layout in the object's own frame plus the neighbourhood shell profile, no appearance
term at all — reads class AUC **0.679 [0.641, 0.716]** and **0.622 [0.583, 0.659]** at the two hard
views (both lower bounds above the registered 0.55), against colour's 0.738 and 0.728; it wins where
colour fails (car: 0.90 vs 0.51; cup: 0.80 vs 0.69), loses where colour is strong (pear: 0.54 vs 0.90),
and the blocks are complementary (ALL − COLOUR **+0.053** and **+0.055**, both CIs excluding zero).
**Door numbers.**  Colour-on: the bag COLOUR door recalls **1.000 / 0.925** of the FIT half at
`-035-045` / `-045-180` (chance 0.100), against the shipped fused door's 0.800 / 0.650 — a paired
**+0.2375 [0.1375, 0.3375]**, confirmation +0.300.  Colour-free: COLOUR-FREE (shape+layout+cfg)
is a worse DOOR than colour — item 1's instance top-1 39/80, 17/80, 17/80 against colour's 65, 58, 43,
and recall@8 69/80, 49/80, 47/80 against 80, 80, 76 — but a slightly better CLASS arm at every view
(0.787 / 0.733 / 0.721 vs 0.728 / 0.738 / 0.728), i.e. geometry answers "what kind of thing" better
than "which thing".  **What the matcher adds**: unmeasured on this door within the
window (640 pairs ≈ 6 h), the door's order stands, and the matcher keeps its two proven uses
(M19 +0.248 correspondence, M34 5-fixed-0-broken re-ranking); it ships as `verify`.  **The shipped
final order** is the bag COLOUR door's; **the class output** is `recognise_class`, ALL-arm, top-2
instance means, **0.650 / 0.575 / 0.500** with CIs that include M23's 0.500 — shipped, not claimed as
an improvement.  **Archived**: 12 modules, 6,143 lines, three batches, suite green after each, with 6
modules deliberately left and the reason written down.  **The one next lever**: a second dataset whose
classes do not differ in colour.
