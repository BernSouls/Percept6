# MISSION 34 — compose places into fragments, in a registered frame

Branch `arena/c316ca3a-percept6`, from `1ed41c2` (M33).  S = 8, one physical core, long jobs in the
background and resumable.  Core in `matching/`, `core/`, `lenses/`; `Topology/` holds thin scripts
that import core.

**The question.**  Every descriptor this tree has is either one place (`placement_content`,
`placement_shape`) or one image (`placement_frontdoor`).  `config` (M19) pools one place's star — a
neighbourhood, not a part.  A fragment is a SET OF PLACES: is there identity in a part that no place
carries?

Two things were demanded besides the build.  Screens before any door read, at fixed bars
(channel-alone recall@8 ≥ **0.675** on FIT; oracle ≥ **15/31** on M31's same-class losses).  And a
verdict, with counts, on the suspicion that *"tons of the machinery is somehow unused and unwired"* —
that a mechanism is added to one lens or one pattern and the rest is never wired.  §7 answers that
with a census, not an opinion.

---

## 1. Item 1's premise does not exist on this graph

**(a) Measured, before anything was built** (`Topology/m34_fragments.py --probe`, apple/1 at
`-035-045`, `Topology/results/placement/m34_premise.json`):

| what | count |
|---|---|
| nodes / edges on the graph the matcher consumes | 1,506 / 2,686 |
| place nodes (Blob 226, BoundaryArc 220, CornerPoint 28) | **474** |
| place nodes with **degree 0** | **474** |
| edges with a place node as **either** endpoint | **0 of 2,686** |

The four edge classes the mission names as the grouping — `EDGE_BOUNDARY_ATTACHMENT`,
`EDGE_COCIRCULAR`, `EDGE_BORDER_OWNERSHIP`, `EDGE_BOUNDARY_SUCCESSOR` — exist in `core/types.py` and
`core/schema.py`, and the front end emits three of them (BORDER_OWNERSHIP_EDGE 292, COCIRCULAR 29,
BOUNDARY_SUCCESSOR 0).  Every one of those edges joins two *region-machinery* nodes
(Contour / ContourFragment / RegionShape / Surface / BorderOwnership).  The front end emits **two
disjoint node universes**: places, and region machinery.  Grouping places by those edges groups
**nothing, 474 times out of 474**.

So the mission's grouping rule is unavailable, and silently redefining "fragment" as a group of
*region* nodes would be exactly the defect §7 counts: a mechanism attached to a lens nothing else
reads.  The substitution is registered here, in writing, before any number:

| grouping | rule | radius |
|---|---|---|
| `contour_fragment` | a place joins the fragment of its nearest `ContourFragment` anchor | 48 px |
| `contour` | the same with the coarser `Contour` anchors | 48 px |
| `proximity` | single-linkage clustering of place positions — the geometric control | 16 px |

This does one thing the mission did not ask for but the corpus requires: it **wires the region
machinery to the places** — the contour part nodes become the fragments' anchors.  Whether that
wiring carries identity is item 3's question, not an assumption.

**(b) What the three groupings actually produce** (`--report`; 80 objects × 3 views, 3,469 s;
`Topology/results/placement/m34_fragments.json`).  Columns: fragments per object; places per
fragment (mean, median); size p95; share of places inside **no** fragment; per-object fragment-size
Spearman across the two query views.

| grouping | fragments / object | places / fragment | p95 | ungrouped | stability |
|---|---|---|---|---|---|
| `contour_fragment` | 13.2 / 14.7 / 16.5 | 38.0 / 36.1 / 32.9 (32 / 31 / 27) | 95 / 91 / 84 | 0.9 % / 1.6 % / 1.3 % | **0.933** |
| `contour` | 6.6 / 7.4 / 7.7 | 64.7 / 61.6 / 60.2 (52 / 52 / 51) | 180 / 155 / 145 | 15.5 % / 15.7 % / 15.4 % | 0.897 |
| `proximity` | 3.8 / 4.0 / 3.7 | 133 / 135 / 150 (10 / 24.5 / 20) | 564 / 586 / 598 | 0 % | n/a |

(views `000-000` / `035-045` / `045-180`.)

**The grouping instability, stated as the mechanism the mission asked for.**  The size *spectrum*
transfers across views (per-object Spearman **0.933**), but the fragment *count* does not: the same
object yields **14.7** fragments at one query view and **16.5** at the other, **+12 %**, and the
pooled 10-bin size histogram moves by an L1 of 187 fragments.  So a fragment is a repeatable *kind*
of part at a repeatable *size*, and not a repeatable *number* of parts.  The geometric control is
worse than unstable, it is degenerate: single linkage at 16 px **chains** — mean 133 places per
fragment against a median of 10, one cluster swallowing the image — so `proximity` is not a part
decomposition at all.  1.3 % of places (contour_fragment) lie beyond every anchor's 48 px and join
nothing; they are reported, not absorbed.

## 2. Item 2 — the descriptor, in the fragment's own frame, registered before any number

`matching/placement_fragment.py`.  Frame: **centre** = mean member position; **orientation** θ =
principal axis of the member positions (mod π); **scale** = median member place scale.  Twenty-two
numbers, all dimensionless and all relative to that frame:

| block | fields | why it is frame-relative |
|---|---|---|
| size | `log_size` | places, not pixels |
| shape of the set | `aspect`, `spread`, `d_mean`, `d_sd` | eigenvalues and radii ÷ the fragment's own scale |
| arrangement | `align1`, `align2`, `cross2` | member angles **measured from θ** |
| pooled curvature | `sharp_x_m/y_m`, `curv_x_m/y_m` (+ four sds) | means and sds over members |
| pooled colour | `lab0..lab5` | means over the members' six CIELAB rings |

Similarity is the tree's one form — `exp(-½ Σ_j w_j (d_j/σ_j)²)`, σ = the pool's own column spread —
so a fragment is compared the way a place and a colour are.  The class is registered in
`CONSTITUTIONAL_RANGES`, `SLOT_WEIGHT_PRIORS`, `CIRCULAR_CYCLE_LENGTHS` (empty: no circular axis) and
`POPULATION_AXES` (22 axes), and `similarity()` returns **0.0** when the class is absent from the
registries.

`tests/test_m34_fragment.py` pins that, pins frame invariance (a fragment turned by 0.9 rad about its
own centroid and translated by (61, −37) gives the same 22 numbers to 1e-9), pins order-independent
grouping, and pins that a place outside the registered radius joins **nothing** rather than being
pulled in.  Two reverts were run: deleting the import-time `register()` turns
`test_class_is_registered_with_every_axis` **red**; setting θ = 0 turns
`test_descriptor_is_frame_invariant` **red**.

## 3. Item 3 — the two screens

**(a) Channel-alone recall@8**, 80-object gallery, chance 0.100, bar 0.675.  The channel's score for a
(query, gallery) pair is *symmetric mean-of-best*: each of the query's parts is matched to the gallery
part that explains it best, the mean is taken, the same is done in the other direction, and the two
are averaged — so a query with parts the gallery cannot explain pays for them.  Registered before the
read; no exponent, no top-k, no weight.

| grouping | `-035-045` r@8 | top-1 | median truth rank | `-045-180` r@8 | top-1 | pairs ≥ 0.99 |
|---|---|---|---|---|---|---|
| **`contour_fragment`** | **0.725** | 0.287 | 2.0 | **0.637** | 0.250 | 0.000 |
| `contour` | 0.675 | 0.250 | 3.0 | 0.537 | 0.212 | 0.000 |
| `proximity` | 0.588 | 0.200 | 5.0 | 0.588 | 0.175 | 0.000 |

Per class, `contour_fragment`: apple 0.80 / 0.70, car 1.00 / 1.00, cow 0.60 / 0.40, cup 0.40 / 0.40,
dog 0.90 / 0.60, horse 0.60 / 0.70, pear 0.60 / 0.50, tomato 0.90 / 0.80.

**(b) The oracle on M31's 31 same-class losses** — in how many does the fragment channel rank the
truth above the matcher's winner: `contour_fragment` **18/31**, `contour` **23/31**, `proximity`
**17/31**, against a bar of **≥ 15/31** (colour read 22/30).  The screen picks the grouping on
(a) → `contour_fragment`, and reads 18/31 on (b): both bars clear.

**The bar, honestly.**  Item 3's bar is one number on two FIT cells.  Read per cell: **0.725 clears,
0.637 does not**.  Read as the mean across the two FIT cells — the precedent M30 set for "mean FIT"
— **0.681 clears by 0.006**, which at n = 80 is **one ninth of a standard error** (SE ≈ 0.053).  The
pass is nominal, and saying that here is the point of citing the SE rather than the mean.  The oracle
is not nominal: 18 is three above the bar.

## 4. Item 4 — the door read, and the bridge to the shipped path

**MDE stated before the read.**  The bar is a +0.050 mean FIT recall@8 with an object-clustered CI
excluding zero.  The paired difference is a mean of per-object 0/±1 indicators, so at n = 80 the
rough MDE is **≈ 0.15**; the realised paired MDE turned out to be **0.076** (only five objects move,
which shrinks the difference's variance).  Both are reported.

`Topology/m34_door.py --run` — records only, no matcher: `rrf(counted 1.0, appearance 1.0,
colour 0.5, fragment 0.5)`, k = 60, the tree's registered discount.

| view | baseline (fused3) | with fragments | gain | fixed | broke |
|---|---|---|---|---|---|
| `-035-045` (FIT) | 0.675 | **0.800** | **+0.125** | 5 | 0 |
| `-045-180` (FIT) | 0.650 | 0.650 | 0.000 | 0 | 0 |
| pooled FIT | 0.6625 | **0.725** | **+0.0625**, CI **[0.0125, 0.1125]** | 5 | 0 |

The baseline is not taken on trust: reconstructing the colour channel from the records reproduces the
M30 record's **0.675 / 0.650** exactly, so the five-channel number is the four-channel door plus one
channel and nothing else.  Five objects are fixed and **none is broken**, one view only; the
sign-test reading of 5-up-0-down is p = 0.031 before any CI is consulted.

**The matcher read on cached shortlists** (`Topology/m34_rerank.py --channel fragment`, the same RRF
at 1.0 / 0.5 inside the eight): top-1 **0.2875 → 0.350** on all 80 (+0.0625, CI [0.0125, 0.1125]),
**0.350 → 0.400** on FIT (+0.050, CI [0.000, 0.125]), DEV +0.042, FRESH +0.125 — and what it changes
is **5 fixed, 0 broken, all five blocked by a same-class impostor**.  Truth is inside the eight for 53
of 80, which is the ceiling this lever can move.

**Shipped, wired, proven.**  The channel is the door's **fifth slot** with default weight **0.0** — a
caller that does not pass it gets the old arithmetic bit-for-bit (tested) — and the bundle now fuses
it at 0.5 (`matching/placement_bundle.json`, `fusion.fragment = 0.5`).  One thing the mission did not
anticipate: the anchors a fragment needs are **not on the place record**, so a records-only door could
not compute the channel.  Rather than shipping a channel the door cannot compute, the anchors are
exported as data (`matching/placement_anchors.py`, `data/placement_anchors_m34.npz`, 616 KB, 320 cells
× 2 groupings) exactly as the records are; the door resolves a query's anchors **from its image path**
and computes them from the front end when the file does not cover the image — it never zeroes the
channel.  `Topology/m34_wire.py --verify` runs the shipped `recognise(matcher=False)` and compares its
order with the item-4 arithmetic, twice per object — once with the anchors handed in, once by path
only: **6/6 identical, 6/6 resolved from the path**, across five classes.

## 5. Item 5 — shape as a re-ranker inside the eight (the independent lever)

`Topology/m34_rerank.py --channel shape`.  Nothing is rescored: the matcher's own order inside the
cached eight is merged with the registered curvature channel's order by the same RRF, k = 60, matcher
1.0, channel at the stated discount **0.5**.

| split | n | top-1 before | top-1 after | **gain** | object-clustered CI95 | MDE |
|---|---|---|---|---|---|---|
| all | 80 | 0.2875 | 0.2375 | **−0.050** | [−0.125, +0.013] | 0.098 |
| **FIT** | 40 | 0.3500 | 0.2250 | **−0.125** | **[−0.225, −0.025]** | 0.148 |
| DEV | 24 | 0.2083 | 0.2500 | +0.042 | [−0.083, +0.167] | 0.205 |
| FRESH | 16 | 0.2500 | 0.2500 | 0.000 | [0.000, 0.000] | 0.000 |

Bar: FIT top-1 **≥ +0.075**.  Measured **−0.125**, CI excluding zero on the wrong side.  It fixes
**2** FIT objects (both blocked by a same-class impostor) and breaks **6** (3 same-class, 3 other).

**The mechanism, with a number.**  The curvature channel's order inside the eight is *uncorrelated*
with the matcher's: mean Spearman over the 53 eligible objects **0.016**, and its own top-1 coincides
with the matcher's on **3 of 53**.  Half a vote from a channel that cannot order a short list can only
cost accuracy — the same finding M33 reached from the other end (ICC 0.009–0.197), now measured at the
head of a short list.  Contrast with the fragment channel, Spearman **0.363** and top-1 agreement
17/53, which is why one lever fixes 5 and breaks 0 and the other fixes 2 and breaks 6.
**Named ceiling: curvature is not a re-ranking signal at half vote.**

## 6. Ship

| gate | result |
|---|---|
| `tests/` (whole suite) | **263 passed, 6 skipped** |
| fragment tests RED on revert | registration removed → 1 red; θ = 0 → 1 red (§2) |
| door identity without the channel | `fuse(...)` bit-identical to `rrf_weighted` (test) |
| the shipped door == the measured door | `m34_wire.py --verify`: **6/6 orders identical, 6/6 resolved from the path** |
| determinism, 1 / 2 / 4 workers | **max abs diff 0.0** (`m31_determinism.py --pairs 2`) |
| `m33_records.py --check`, both views | **80/80 shortlists identical, 0 differences, cosine delta 0.0, top-8 orders changed 0** |
| anchors re-derivable | `Topology/m34_anchors.py --check` (320 cells; the export is the same function) |
| bundle | re-frozen, `code_digest 27c6a3b756e4ef8f`, door `fused3`, colour 0.5 + **fragment 0.5** |
| records loadable from a fresh checkout | unchanged, 38.03 MB, `--check` above |
| `-check` cost | 219.7 s (`-035-045`) + 245.5 s (`-045-180`) |

`ARCHIVE_INDEX`/`KEEP_MAP` record the new files (§ below).  The bundle's `measured.M34` block carries
the door numbers, the oracle, the ceiling, and the anchors' reason for existing.

## 7. The charge: is the machinery wired?  A census, with counts

`Topology/m34_audit.py --run` → `Topology/results/placement/m34_audit.json`.  Static (AST import
closure, string census) plus one dynamic read (the graph vocabulary of a real image).  Entry points
are the shipped ones — `matching/placement_recognise`, the bundle's `CODE_MODULES`, `eth80_frontend`,
`orchestration.pipeline` — not "does some test import it".

### 7.1 Modules

| bucket | modules | lines | share of lines |
|---|---|---|---|
| SHIPPED (reachable from an entry point) | 88 | 65,281 | **83.1 %** |
| HARNESS-ONLY (reachable only from a `Topology/` script) | 4 | 1,831 | 2.3 % |
| TEST-ONLY | 1 | 1,174 | 1.5 % |
| ORPHAN (nothing reachable and nothing kept imports it) | 19 | 10,291 | **13.1 %** |
| total | 113 | 78,587 | |

(The census is run AFTER §4 wired the fragment channel: `matching/placement_fragment.py` sat in
HARNESS-ONLY when the audit was first run, and its 279 lines are in SHIPPED here.  Left un-wired it
would have been the twentieth orphan.)

The first run of this census said 58 orphans and 64 % of the tree dead — and it was **wrong**: its
entry points were the packages under study, so `eth80_frontend.py` (which imports `lenses.keypoints`
and `lenses.base` inside functions) was invisible.  Seeding the census from the real entry points
moved `lenses/base.py` (3,096 lines) and the keypoint detector back to SHIPPED.  The counts below are
post-correction; the correction is itself an instance of the charge.

The orphans are not scraps: `perception/surface_fragments.py` (**2,638**) +
`lenses/surface_fragment_lens.py` (396) + `perception/surface_fragment_cache.py` (151) is **3,185
lines of FRAGMENT machinery** that no shipped module imports — the very level this mission was asked
to build, already written once.  Then `matching/relation_walker.py` (**1,494**, a complete
relation-grown matcher with its own switch `MATCH_RELATION_WALKER = False`, whose only reader is
*inside the unreachable module*: flipping the switch changes nothing because nothing shipped calls
`match_evidence`), `perception/relatable_completion.py` 1,181, `perception/contour_relations.py` 790,
`perception/ring_hypothesis.py` 740, `perception/psychophysics.py` 486, `matching/fuzzy.py` 385,
`perception/frames.py` 382, `matching/signature_matrix.py` 377, `matching/consensus.py` 305,
`matching/search.py` 244, `matching/m15_places.py` 197, `matching/calibration.py` 185,
`eth80_split.py` 127, `matching/canonical_order.py` 88, `core/junctions.py` 82, `sensory_floor.py`
43.  `matching/graded_mcs.py` (1,174) is TEST-ONLY: an entire alternative matcher kept alive by tests.

### 7.2 Switches

612 flags in `parameters/`.  **234** (38 %) are named by a module the shipped path can reach.  **378**
are not; of those, **153** are named only by unreachable modules and **225 are named nowhere in the
tree outside `parameters/`** — a constant no code consults.  (`MATCH_RELATION_WALKER` and the whole
`ADAPTER_*` family sit in the first group; `ADAPTER_*` belongs to `perception/synthetic3d.py`, which
is HARNESS-ONLY.)  Registered classes: **67**, of which **65** are named by shipped code — that census
comes out clean.

### 7.3 The graph the matcher actually consumes

One image: 1,506 nodes, 2,686 edges, **21 node classes**, **24 edge types**.

* **8 of 24** edge types are named by any module that turns a graph into a number (CONTACT,
  CONTOUR_CONTINUITY, CO_LOCATED, CURVATURE, CURVE_CONTAINMENT, DEGREE, ORIENTATION_CONTINUITY,
  PARALLEL — 402 of 2,686 edges = **15 % of the edge mass**);
* **16 of 24** appear only in the lens/binder that writes them and in `category_bridge`'s
  type→category table: TEXTURE_BOUNDARY_EDGE (1,133 edges), REGION_CONTENT, BORDER_OWNERSHIP_EDGE,
  COLOR_OPPONENT_BORDER, CONTOUR_ORIENTATION, TOPOLOGY, CURVATURE_SIGN_CONTINUITY, COCIRCULAR,
  RADIALITY, LINE_TERMINATION, AMODAL_LINK, SHARED_BOUNDARY, ORIENTATION_BEND, SYMMETRY,
  ORIENTATION_JUNCTION, PART_CONTAINMENT — **85 % of the edge mass**;
* `joint_mcs` — the scorer — names **none** of them individually: it splits edges into
  directed/undirected and matches them by type equality.  No module reads a single type's semantics.
* **474 place nodes, 0 edges.**  The relational half of the score never touches a place.

The relational term is not decorative: on a truth pair (apple/1 vs apple/2 at `-035-045`)
`node_term` **0.3665**, `edge_term` **0.0959**, `score_shipped` 0.3248 — edges are **29.5 %** of the
shipped score, 662 edges matched.  The machinery is *used* and *undifferentiated* at the same time.

### 7.4 Verdict on the charge

**Right in kind, wrong in scale, and right in a place I did not expect.**

* **Right in kind.**  Machinery is routinely built and left unreachable: 19 modules, 10,291 lines,
  13.1 % of the tree, including **fragment machinery that already did the composition M34 was asked to
  build** and a complete alternative matcher reachable only from tests.  37 % of the parameter surface
  is read by nothing at all.
* **Right one level down — the sharper form.**  A mechanism is attached to the thing the author was
  looking at (a node class, an edge writer) and not to the thing that decides: 16 of 24 emitted edge
  types reach no decision module, the 474 places reach no edge, and a switch for the unreachable
  matcher is not even read when flipped.  The pattern the charge names is real; it lives at the seam
  between the graph's vocabulary and the scorer, not at the module level.
* **Wrong in scale.**  83.1 % of the lines are reachable, 65 of 67 registered classes are named by
  shipped code, and the graph's relational term carries 29.5 % of the shipped score.  The tree is
  mostly *undifferentiated*, not mostly dead — a fifth of the dead part is genuinely dead.
* **And it would have happened again.**  `matching/placement_fragment.py` was HARNESS-ONLY when this
  audit first ran.  The reason it is not the twentieth orphan is §4: the channel cleared its bars,
  entered the bundle, and has a wire test that fails if the shipped door stops computing it.

## DONE — the paragraph

A part-level descriptor carries identity a place-level one does not, and the difference is measured:
`placement_shape` at object level reads 0.225 / 0.263 channel-alone recall@8 and, re-ranking the
shipped shortlists, *costs* 0.125 FIT top-1; the fragment channel reads 0.725 / 0.637 and *gains*
0.0625 with five objects fixed and none broken.  What made the difference is not more numbers per
unit — the fragment's twenty-two are all frame-relative summaries of what the members already carry —
but the *composition*: a fragment is the first object in this tree that owns a shape of its own
(principal axis, aspect, arrangement) rather than inheriting one from a single detection, and the
four edge classes the mission wanted to group places by do not exist, so the wiring had to be built
(`contour part anchors`) before any of it could be read.  What shipped: the fragment channel as the
door's fifth slot at 0.5, `fusion.fragment = 0.5` in the bundle, the anchors as committed data, the
descriptor registered with a 0.0 refusal, and the audit that keeps score of what is wired.  What did
not ship, as a named ceiling: shape as a re-ranker (−0.125 FIT, CI [−0.225, −0.025], mechanism:
Spearman 0.016 between its order and the matcher's inside the eight).  The next lever is the one the
census points at: the **fragment count is unstable across views (14.7 → 16.5, +12 %) while the size
spectrum transfers (r = 0.933)** — making the grouping itself view-stable (anchors merged across the
two views, or a part count read from the record rather than the graph) is the cheapest thing left that
would move the channel, and the 3,185 orphaned lines of `surface_fragments` are where to look for it.
