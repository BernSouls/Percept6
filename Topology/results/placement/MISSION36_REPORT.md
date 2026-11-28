# MISSION 36 — A DENSE LOCAL-DIFFERENCE FIELD AND SPARSE PLACES, JOINED IN THE CELL BUILD PATH

Branch `arena/c316ca3a-percept6`, from `b1b5ac8`. Every number below was measured in this
tree; the commands and the files that produced them are named at the point of use.
`M36_AUDIT_REPORT.md` (the same commit range's predecessor) is assumed read.

**What was built, in one paragraph.** A dense local-difference field
(`perception/local_difference.py`) that mints a node wherever a centre-surround measurement of
pooled quadrature energy differs — on luminance and on both colour-opponent channels, at three
scales of the Gabor bank's own ladder — and carries seven registered slots at that node
(orientation, coherence, difference, three colours, scale). A lens that hands those nodes out as
detections (`lenses/dense_field.py`). A join (`matching/placement_join.py`) that puts them IN the
cell's graph, owned by the sparse places, with every vector in the SOURCE node's own frame and
both refusals counted. The record and the compact record file carry them
(`matching/placement_frontend.place_record`, `matching/placement_records` v31). Both switches are
**OFF by default**; with them off the 40-cell structural digest is bit-identical to `b1b5ac8`.

---

## §1 PART A — WHICH LANDED REPAIRS REACH A PLACEMENT CELL

Measured once, ON against OFF in the same process, on three frames (`FE.cell`, nodes / edges /
digest):

| landed fix | apple1-035-045 | apple1-022-000 | cup1-035-045 | verdict |
|---|---|---|---|---|
| `BOUNDARY_CIRCULARITY_RADIAL` | 1552→1555, 2985→3001 DIFF | E 2686→2683 DIFF | 1704→1693, 1710→1678 DIFF | **reaches the cell** |
| `LENS_ANCHOR_RESOLUTION` | 1552→1568, 2985→3306 | 1506→1514, 2686→2941 | 1704→1707, 1710→1762 | **reaches the cell**, largest effect |
| `RICH_TEXTURE_ABSENT` | unchanged | unchanged | E 1710→1706 DIFF | reaches it on one frame of three |
| `RICH_REGULARITY_CORRELATION`, `RICH_ROUGHNESS_FULL_SCALE`, `KP_ANISOTROPY_RATIO`, `LENS_AREA_FROM_SHAPE`, `FE_L6_SIGMA_CONTENT`, `FE_CONTENT_PLACES` | identical | identical | identical | **payload-only or binder-stage**: they do not move a placement cell |

That last row is the reason Part C is a stage in `place_record`/`build_edges` and not a flag flip:
the feature flags that look like joins are read by binder stages that never run in a placement
cell, so a "before/after" built on them would have been a comparison of nothing.

---

## §2 PART B — THE FIELD, AND THE TWO MECHANISMS THE MEASUREMENTS FOUND

**Construction** (`perception/local_difference.py`). Opponent planes as the chains build them;
the bank is the tree's own `GaborBank.build_default` (12 orientations × 5 spatial frequencies ×
2 phases), built at the image's own scale (`_front_end_scale`, so the same object at 128 px and
256 px is analysed at the same *relative* scale). Pooled quadrature energy per luminance and per
colour-opponent channel; a difference of Gaussians over the pooled map (centre 0.5 σ, surround
1.5 σ); the ratio of that difference to its denominator is the node's activity; a node is minted
where the activity clears `DENSE_TAU = 0.25` and the site survives a spacing rule of
`1.5 σ` at that scale. **No region, no growth, no boundary, no merge** — one threshold and one
spacing, which is the whole of the mission's permission. Measured cost at 256×256: opponent
planes 0.002 s, bank build 0.021 s, responses 0.50 s per channel; a full field **1.20 s per image**
(mean over 40 images).

**Two mechanisms the measurements exposed, and the repair for each.**

1. *Noise amplification.* With the Weber ratio taken literally, the SMOOTHEST decile of 32×32
   patches inside the object produced **25.6 nodes per patch** on `cup1-035-045` and the MOST
   TEXTURED decile produced **8.3** — the field read the absence of content as content, because a
   surround near zero makes any numerator large. The repair is a denominator floor at
   **0.5 × the frame's median surround energy at that scale** (pre-registered, not tuned):
   smooth 25.6 → **1.2**, textured 8.3 → **8.3**. Same test on `apple1-035-045`: 21.5/16.4 →
   **2.8/14.9**. The ordering is now the one the mission requires, at a factor of 5–7.
2. *Sparse sampling mistaken for sparse coding.* Local maxima alone gave **514 nodes against the
   cell's 518 places** — one dense node per place, a second level that adds nothing. The mission's
   own distinction says the sampling must be dense while the activity stays sparse, so the node
   set is now the greedy packing of the above-threshold sites (strongest first, one node per
   `1.5 σ`): **1344 / 1216 / 1809** nodes on the three census frames, **2.8 / 2.5 / 2.3 per place**,
   covering **88 %** of the frame (frame-mean 73 %, object-mean **88.6 %**).

**What the field now measures.**

| claim | measurement |
|---|---|
| content-driven, not a lattice | uniform frame **0** nodes, noise **501**, luminance gradient **669**, uniform grating **0** (a uniformly periodic field has no local difference — correct by the rule) |
| textured beats smooth *within one image* | cup1: smooth decile 1.2 nodes/patch vs textured **8.3**; apple1: 2.8 vs **14.9** |
| count varies with the image | 40 images: mean **1141**, min **767**, max **1471**, sd **179**; correlation with the image's own mean absolute Laplacian **+0.164** |
| dense coverage | object coverage mean **0.886** (min 0.638), frame coverage mean **0.728** — against the places' 518 points on 65 536 pixels |
| activity stays sparse | the above-threshold fraction is 0.3–1.5 % of pixels per scale; the field is *dense in coverage, sparse in units*, which is the stated target |

**The one honest negative.** Across images the count is only weakly related to a pixel-level
variation measure (+0.164): the field is band-passed and peak-based, so a frame whose variation is
spread over the pixel lattice yields a flatter ratio field and fewer isolated maxima even though
its activity is higher. The correlation is stated, not explained away, and it is why the decisive
content test above is the WITHIN-image one.

**Stability across views (B5), and its null.** 8 objects, 000-000 against 035-045, positional
matching within 6 px: content correlation **0.343** (min 0.265, max 0.408, all 8 positive); the
same measure between DIFFERENT objects is **0.265** (min 0.153); same beats cross in **6 of 8**
pairs; mean match distance **2.39 px**. The separation is real and small, and it is small for a
reason that is the mission's own claim: a dense node does not need view-repeatability, because the
place it hangs from carries that. Nothing in this report claims view-repeatability for the nodes.

---

## §3 PART C — THE JOIN, WITH THE CENSUS BEFORE AND AFTER

Three frames, `PLACEMENT_DENSE_FIELD`/`PLACEMENT_DENSE_JOIN` OFF then ON, same process
(`/tmp/m36_census.json`):

| frame | nodes OFF→ON | edges OFF→ON | degree-0 OFF→ON | place degree mean (zero/n) | `DENSE_IN_PLACE` | `DENSE_PROXIMITY` | refused: no place / no frame |
|---|---|---|---|---|---|---|---|
| apple1-035-045 | 1506 → **2850** | 2686 → **7892** | 763 → **457** | 0.00 → **2.683** (173/479) | 1285 | 4238 | 0 / 59 |
| apple2-022-000 | 1623 → **2839** | 2496 → **6849** | 799 → **480** | 0.00 → **2.471** (159/482) | 1191 | 3600 | 0 / 25 |
| cup1-035-045 | 1704 → **3513** | 1710 → **8824** | 1028 → **519** | 0.00 → **2.326** (278/773) | 1798 | 5400 | 0 / 11 |

- **C1 ownership.** Every dense node is attached to exactly one place — the nearest PAYLOAD place
  (the ones the record and the matcher carry), ties to the lower node id, assignment computed
  vectorised so it is order-independent. A dense node with no place is refused and counted
  (0 here); a place whose own frame was never measured refuses the edge and is counted (59/25/11).
  Dense ids start at `1<<20`, so no dense id can collide with a pattern id.
- **C2 vectors in the graph.** `DENSE_IN_PLACE` (place → dense) carries the dense node's position
  in the PLACE's axial frame; `DENSE_PROXIMITY` (dense → dense, k = 4 inside a place) carries the
  target's position in the SOURCE's own frame. The exact floats live on the graph
  (`subgraph.dense_vectors`, `PlacementVector`), and the door's own container
  (`subgraph.edge_params`) holds the same relation as `EdgeParams(class_="EDGE_DENSE_PLACEMENT")`
  — a class **registered in `core/schema.EDGE_SCHEMAS`** (bearing 12-cycle over π, distance 8,
  log-scale 8). That registration is not decoration: an unregistered edge class returns **0.0** for
  every comparison (`EdgeParams.similarity`), which is exactly the "silently inert" failure the M36
  audit was written about.
- **C3 place identity is untouched.** The place keeps its own signature, its own M15
  repeatability, its own record row; the join adds nodes and edges and changes no place's key.
  Verified as an array equality in D1's second test.
- **C4 the census is a lens membership count, not a node total** (`PJ.census`): the same function
  reports 21 lens kinds before and the same 21 plus `DENSE_FIELD` after, with the counts per kind,
  per edge type, and the degree distribution. Nothing is summed into a single "n".
- **C5 every pair names its scale and its frame source.** `place_frame_sources` reports which
  provider gave each owned place its frame (keypoint estimator vs the field fallback of
  `node_frames_keypoint_then_field`); the vector carries `source_theta` and `source_scale`; the
  proximity vector carries `log_scale = log2(support_target/support_source)`.

**Where the layer sits in the build.** `FE.cell` builds the graph and the matcher side, then — only
when the flag is on — measures the field once and joins a SHALLOW COPY of the graph. The cached
graph object is never mutated (`matching/placement_cache` hands one object per content hash), the
side and the payload are the ones the shipped path built, and the memo is keyed by the flag state
so one process cannot serve the other configuration's cell.

---

## §4 PART D — SIX DESTINATIONS, SIX TESTS

`tests/test_m36_dense.py`, **14 tests, all green**; two of them demonstrated RED by disabling the
mechanism they pin (`dense_rows → []` ⇒ D1 fails; `join → input` ⇒ D4 fails).

| # | destination | how it is reached | test |
|---|---|---|---|
| D1 | `place_record` | `rec["dense"]` = one row per node (graph id, owner place, x, y, support, activity, 7 slots), `rec["dense_owner_dropped"]` counts rows whose owner the record dropped, `rec["join"]` = the join's own census | `test_d1_dense_layer_is_in_the_record`, `test_d1_shipped_record_arrays_do_not_move` |
| D2 | `placement_records` (the compact file) | **FORMAT_VERSION 31**: `dense_<view>` (12 float columns) + `denseobj_<view>`, `meta.dense` = 12, absent from a flag-off file. Round trip is exact (`array_equal`). Size: 12 columns × N nodes × 4 B — 1344 nodes add **64.5 KB** before compression on this frame, and **0 B** when the flag is off because the block is not written | `test_d2_dense_block_round_trips`, `test_d2_flag_off_file_has_no_dense_block` |
| D3 | the tree's kernel | `POPULATION_SLOTS` / `POPULATION_AXES` / `CONSTITUTIONAL_RANGES` (theta 6.0 over a 12-cycle by the N/2 convention, the rest 8/8/8/8/8/6) / `CIRCULAR_CYCLE_LENGTHS[cls][0] == 12`; `slot_axes` pairs all 7 by name. A node compared with itself scores **> 0.99**; corrupted content drops it; DENSE_FIELD with differing bins returns a **graded 0.0136**, while an unregistered class returns **exactly 0.0** | `test_d3_class_is_registered_everywhere`, `test_d3_content_moves_similarity` |
| D4 | the graph | 1344 dense nodes in the graph, 1285 `DENSE_IN_PLACE` + 4238 `DENSE_PROXIMITY` edges, every dense node owned once, every vector axial with `source_scale > 0` | `test_d4_join_gives_places_a_degree`, `test_d4_dense_sigs_are_on_the_graph` |
| D5 | determinism | the field's per-node digest is identical at 1, 2 and 4 threads and across a process pool; **max difference 0.0** (digests compared exactly) | `test_d5_determinism_at_1_2_4_workers` |
| D6 | flags off by default | no `dense` key, no id ≥ `1<<20` in a default cell, both switches False | `test_d6_flags_are_off_by_default`, `test_d6_default_record_has_no_dense_block` |

**No silent fallback.** The join's counts close exactly:
`n_dense_in_place_edges + n_refused_no_frame + n_refused_no_place == n_dense` and
`n_dense_attached + n_refused_no_place == n_dense` (`test_b5_every_refusal_is_counted`). The record
never names a place row that does not exist: a dense node whose owner was dropped is written with
`place = -1` and counted in `dense_owner_dropped`.

---

## §5 PART E — WHAT THE LAYER IS WORTH, MEASURED, AND ITS CEILING NAMED

**E1 — the shipped door's score, flag OFF vs ON, same pairs** (`PF.pair_score`, three pairs):

| pair | OFF | ON | delta | seconds OFF / ON |
|---|---|---|---|---|
| apple1-035-045 → apple1-022-000 | 0.35538 | 0.24613 | **−0.10926** | 34.7 / 193.8 |
| apple1-035-045 → apple2-035-045 | 0.34109 | 0.24392 | **−0.09718** | 35.2 / 159.9 |
| cup1-035-045 → cup2-035-045 | 0.48251 | 0.29284 | **−0.18967** | 29.1 / 339.4 |

**The layer, switched into the graph, costs the shipped door 0.10–0.19 of score on these pairs and
5–10× the time.** That is not a defect of the join; it is what the mission says must follow from
scope: the door's arithmetic is out of scope, and its edge term treats every dense node as a place
that ought to repeat across views — the one thing a dense node is explicitly not required to do.
The ceiling is therefore named with numbers: *this representation improves what a place carries; it
does not improve retrieval until the door's edge term distinguishes carried content from placed
anchors, and that is a later mission's arithmetic.*

**E2 — the layer's own information, outside the door.** Activity-weighted mean of the 7 slots per
image, gallery 000-000, query 035-045, 10 objects, chance 0.10: **top-1 = 0.60 (6/10)**, mean rank
of the true object 0.7. So the content the dense layer carries *does* identify the object across a
view change at six times chance, before any matcher, any weight, any retrieval.

---

## §6 GATES

| gate | result |
|---|---|
| `pytest tests -q` | **298 passed, 6 skipped, 4 subtests passed** in 277 s (was 284/6 before this mission's 14) |
| 40-cell structural digest vs `b1b5ac8` | **0 diffs** (`/tmp/dig_base.json` vs `/tmp/dig_head2.json`, 40 keys, nodes+edges+every signature) |
| `m35_pairs.jsonl` | restored to the committed file after the suite (`git checkout --`) |
| bundle | **re-frozen for THIS tree, `code_digest` only** (`f19540f5… → 73baa200…`): `matching/placement_frontend.py` and `matching/placement_records.py` are in `CODE_MODULES` and gained flag-gated code. Verified field-by-field that **no configuration value moved** — only the digest — and the 0-diff digest above is the evidence that flag-OFF behaviour did not move either. |
| deleted | nothing |
| weight/bin sweeps | none: `DENSE_TAU = 0.25`, `DENOM_FLOOR_FRAC = 0.5`, `MIN_SPACING_FRAC = 1.5`, `DENSE_SCALES = (0,2,4)`, `DENSE_LINK_K = 4` were all fixed before their numbers were taken; the one choice made from measurement (the denominator floor) is reported with the failing measurement that forced it and the mechanism, not as a tuning |

---

## §7 CLOSING

The topology now contains the places, and it contains what they carry. Before this mission a cell's
518 places sat at degree 0 holding 0 of its 2,686 edges; a place was a point with a signature and
nothing around it, and the dense content that surrounds it — 88 % of the frame, 88.6 % of the
object on a 40-image mean — existed nowhere in the graph at all. With the join in place, 479–773
places per cell carry 1,216–1,809 dense nodes at 2.3–2.8 nodes per place, 1,285 `DENSE_IN_PLACE`
and 4,238 `DENSE_PROXIMITY` edges on `apple1-035-045`, and the share of degree-0 nodes falls from
763 of 1,506 to 457 of 2,850. What the dense field covers is the content that differs: not a
segmentation of the object, but a set of positions where a centre-surround measurement of luminance
and both colour axes changed, each carrying the local orientation, coherence, difference, colour
and scale at that spot, spaced so that no content wider than the measurement's own σ is left
without one.

The single next repair is the one E1 makes unavoidable: **the door's edge term cannot tell a dense
node carried by a place from a place that must repeat across views**, so switching the layer on
costs 0.10–0.19 of score. The repair is to make `joint_mcs`'s edge term read `DENSE_IN_PLACE`
against `DENSE_IN_PLACE` under the place's own frame and refuse to credit a dense node as an
independent anchor — arithmetic that this mission's scope explicitly excluded, and the reason its
result is a measured cost rather than a claimed gain.
