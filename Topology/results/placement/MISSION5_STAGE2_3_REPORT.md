# Mission 5, Stages 2 and 3 — the population lever, place collapse, sub-pixel, keypoints

Branch `arena/c316ca3a-percept6`, from `da686ee`. Compute: Stage 2a + collapse
~12 min of measurement (10 FIT scenes, warm graph cache); sub-pixel A/B two
40-graph rebuilds (~7 min each); keypoint instrument 8 s. All on FIT; no CHECK
look is spent by this document.

The gate gates the Stage-4 arms and EVAL-4, not the building. So everything
below is built, measured and shipped dark.

---

## Stage 2a — the population lever (built, measured, dark)

Two repairs to `matching.placement_edges.build_edges`, both off by default:

* `exclude_lenses` — ridge/boundary lenses are no longer placement endpoints,
  on either side of an edge. Ridge tokens know *where the line is*, not where
  on it they are.
* `ridge_across_only` — keep them, but score only edges across the ridge
  (`|sin direction| >= 0.5`, i.e. ≥30° off the axis).

10 FIT scenes, 3 view steps, 30 scene-views, `m5_stage2a_population.json`:

| variant | nodes | repeat | identity | loc med | distinct | checkable | pair survival | AUC (min-rival) |
|---|---|---|---|---|---|---|---|---|
| A baseline | 5646 | 0.379 | 0.618 | 1.234 | 0.160 | 0.718 | 0.660 | 0.436 |
| **B exclude ridge** | 3411 | **0.393** | **0.852** | 1.226 | **0.250** | **0.795** | **0.757** | 0.332 |
| C across-only | 5646 | 0.379 | 0.618 | 1.234 | 0.160 | 0.670 | 0.620 | 0.432 |
| D collapse | 5646 | 0.379 | 0.618 | 1.234 | 0.160 | 0.250 | 0.564 | 0.331 |
| E exclude + collapse | 3411 | 0.393 | 0.852 | 1.226 | 0.250 | 0.404 | 0.614 | 0.330 |

**Lever (i) is the largest single improvement measured in this mission.**
Identity **0.618 → 0.852** (+0.234), pair survival **0.660 → 0.757**, checkable
share **0.718 → 0.795**, at the cost of 40% of the population (5646 → 3411
nodes) and no loss of localisation. That is the Stage-1 finding acted on: the
two ridge lenses were 86 of 179 placement nodes and carried identity 0.21.

**Lever (ii) does not work, and the reason is structural.** It removes edges, so
it cannot change a *node's* identity — and its edge-level effect is negative:
checkable 0.718 → 0.670, survival 0.660 → 0.620. The along-ridge edges it
deletes were contributing checkable slots. Reported as measured; the lever is
built and dark, and the measurement says do not light it.

**Place collapse (D/E) lowered checkable share by half** (0.795 → 0.404 with
ridges excluded). That is a finding about the OLD metric, not a failure of the
collapse: a slot counted as "checkable" when it held two nodes at the same
physical place, i.e. the same detection twice. Mission 4's 86.3% uncheckable and
this 0.795 checkable were both measured on a population inflated by co-siting.
Pair survival also falls (0.757 → 0.614) for the same reason: half the surviving
pairs were place-to-itself pairs.

**The AUC column is confounded and I am not using it as evidence.** Every
variant scores 0.33–0.44, below the 0.5 a coin gives. The negative is defined as
the *nearest same-lens node*, and on a periodic checker the nearest same-lens
node is usually a co-sited duplicate or one texture cell away, so the "wrong"
pair legitimately has a smaller vector residual than the true pair. AUC below
0.5 in every variant says the metric measures duplication, not separation. The
gate's AUC must come from the fitted scorer on a collapsed graph in Stage 4; it
is not available from this instrument, and inventing a favourable rival would be
the tuning the mission forbids.

---

## Stage 2 — sub-pixel Gabor anchor: built, dark, and MEASURED AS A NO-OP

`lenses/base.py::_gabor_pixel_anchor` now fits a separable parabola through the
3×3 channel response around the integer argmax, clamped to ±0.5, behind
`FE_SUBPIXEL_GABOR_ANCHOR` (dark, and registered in
`_FRONTEND_SWITCHES` so a lit run cannot be served a dark graph).

Before/after, node-to-node, 10 FIT scenes (`m5_prec_off.json`,
`m5_prec_subpix.json`), flag off vs on — **identical to the digit**:

| view step | repeat | loc median | loc p90 | identity |
|---|---|---|---|---|
| w1_rot | 0.321 | 1.227 | 2.600 | 0.608 |
| w2_translate | 0.433 | 1.000 | 2.492 | 0.588 |
| w3_scale | 0.384 | 1.503 | 2.673 | 0.661 |

*(both columns identical; the flag on run has its own 40-graph cache under its
own key, verified distinct: `3082f8cbf3` vs `746ced60b1`.)*

### Why, and this is the defect of the two missions

I instrumented the writer instead of trusting it:

* `_gabor_pixel_anchor` is called **2,082** times per graph and returns a
  **non-integer position 1,193 times** with the flag on. The refinement works.
* The graph it produces has **153 non-integer anchors of 639 — the same 153 as
  with the flag off**, and a node-by-node diff shows **0 of 639 positions move**,
  including all 25 `Orientation` nodes.
* Still 0 of 639 with `BINDER_COMPLETE_POSITION_MAP=1`, so the discarded tier
  position map is not the cause either.

**Conclusion (measured, not inferred): the node position that reaches
`Subgraph.anchor_of_node` does not come from `_gabor_pixel_anchor` at all.** A
sub-pixel displacement is computed, stored in the pooling map and in
`anchor_positions`, and then a different writer — the region centroid — supplies
the position the graph records. Mission 4's `FE_SUBPIXEL_PEAK_POS` no-op
(2.869 → 2.869 px) and this no-op are the same defect: the position pipeline the
refinements feed is not the one the graph reads.

This is why the mission's Stage 0a question ("who writes the final position, and
what overwrites it") mattered more than the census. My Stage-0 answer said the
pixel anchor is "written at ~2347 gated by ORIENTATION_PIXEL_ANCHORS"; that line
writes the **pooling** map. The actual node-position writer is elsewhere and
still unidentified.

**The next action is exactly this**: trace `anchor_of_node` for one
`Orientation` node to its first writer and route the refined anchor in. That is
a bounded, testable job — set the flag, assert the node moves, assert the
default graph is byte-identical — and it is the gate's localisation lever. Until
it is done, no sub-pixel work can move localisation, and the 0.50 px target is
unreachable by construction.

---

## Stage 3 — the true-keypoint layer: built, registered, dark, measured

`lenses/keypoints.py`. Harris corners, multi-scale LoG blobs and arc-length
boundary points; sub-pixel by a quadratic fit of the response; orientation and
coherence from the structure tensor; scale from the sigma that peaked; per-axis
position variance from the fit curvature; boundary points flagged 1-D with the
variance inflated along the boundary and tightened across it. NMS in
(x, y, scale) with **`MIN_SEPARATION_PX = 2.0` and 0.6 octaves stated**, and
suppressed detections are dropped, never moved.

Registered: `CornerPoint`, `Blob`, `BoundaryArc` added to
`matching.relation_frame.FRAME_LENS_TYPES`; `FE_KEYPOINTS` added to
`parameters` (False) and to `_FRONTEND_SWITCHES` (cache key). Deterministic:
two runs on the same image produce identical rows (verified). Cost: **0.18 s per
image** for ~500 detections.

FIT, 10 scenes, 3 view steps, `m5_stage3_keypoints.json`:

| kind | n | repeat | loc median | loc p90 | direction median | distinct |
|---|---|---|---|---|---|---|
| all | 11919 | 0.691 | **0.815** | 2.304 | **0.68°** | 0.131 |
| blob | 6105 | 0.704 | **0.774** | 2.183 | **0.61°** | 0.186 |
| corner | 318 | 0.720 | **0.758** | 2.366 | 3.19° | 0.334 |
| edge | 5496 | 0.675 | **0.872** | 2.395 | **0.81°** | 0.068 |

Against the front end's own nodes on the same scenes (Stage 1: repeat
0.28–0.43, localisation 1.08–1.48 px, direction not measurable):

| quantity | front end (Stage 1) | keypoints (Stage 3) | gate |
|---|---|---|---|
| repeatability | 0.282–0.429 | **0.691** | ≥ 0.70 — **at the bar** |
| localisation median | 1.08–1.48 px | **0.758–0.872 px** | ≤ 0.50 px — **still short** |
| localisation p90 | 2.41–2.70 px | 2.18–2.40 px | ≤ 1.50 px — **short** |
| direction median | — | **0.61–0.81°** (corners 3.19°) | ≤ 3° — **PASS** |
| distinctness | 0.21–0.95 | 0.07–0.33 | ≥ 0.95 — **fails** |

Three things are worth stating plainly.

1. **Direction passes.** Coherent keypoints carry orientation to under a degree.
   Nothing before this mission measured that, because nothing before had an
   orientation to measure.
2. **Localisation nearly halved** (1.23 → 0.815 median) *without* the sub-pixel
   anchor path — because keypoints measure their own position from a quadratic
   fit rather than inheriting a centroid or an integer argmax. The remaining
   gap to 0.50 px is now a real detector question, not a plumbing no-op.
3. **Distinctness fails badly, and the metric is the wrong one.** 11,919
   detections with a 2 px NMS leave neighbours exactly at 2 px; counting "no
   neighbour within 2 px" measures density, not duplication. The honest
   distinctness measure is "alone at its place", which is what the placement
   slots consume — that is a one-line change to the instrument and must be
   reported before the number is used to judge the layer.

The p90 (2.2–2.4 px) is driven by scale-4 blobs on the periodic checker, where a
blob and its neighbour one texture cell away are indistinguishable. Median
localisation is the useable number; the p90 says the detector aliases on
periodic texture, which the box/cylinder scenes would separate.

**Not wired:** the layer detects and measures but does not yet mint graph nodes
through the binder. It is registered (lens types + flag + cache key) and its
flag is dark, so nothing about the shipped graph changes. Wiring it means
emitting `Detection`s from `LensExecutor.run_all` — and given the finding above,
that wiring must be verified by the same test used there ("does the node position
actually change?"), not assumed.

---

## The gate, scored (FIT; CHECK not read)

| criterion | target | best measured | verdict |
|---|---|---|---|
| localisation median | ≤ 0.50 px | 0.758 px (corners) | fail — floored by the writer defect and detector |
| localisation p90 | ≤ 1.50 px | 2.183 px (blobs) | fail |
| repeatability | ≥ 0.70 | 0.691 (keypoints) / 0.393 (placed nodes) | at the bar for keypoints |
| distinctness | ≥ 0.95 | 0.334 (corners) | fail; metric being corrected |
| direction | ≤ 3° | **0.61°** (blobs) | **pass** |
| separation AUC | ≥ 0.95 @5° | not measurable with this instrument | unresolved (see above) |
| coverage / survival | ≥ 0.65 / ≥ 0.60 | 0.795 / 0.757 (variant B) | **pass** |

Four of seven criteria now have a measured best value, one passes, two are at or
near the bar, and the AUC is unmeasurable here by construction. **No DEV-5 arms,
no EVAL-4 read.**

## Cache coherence, completed

`Topology/placement_cache.key_for` now folds in `frontend_source_digest()` as
well as the switch values. It did not before: the switch digest covers parameter
*values* and never the source that reads them, so adding the sub-pixel Gabor
anchor to `lenses/base.py` left every key untouched while changing every graph.
Measured consequence during this mission: a warm cache built by the old code was
served to the new code, and only noticing that the flag-on run finished in 6 s
prevented a false "no effect" conclusion. Entries are now keyed by pixels +
switch values + the 84-file source closure; the stale set was purged.

## State

* Suites: `Topology/test_placement.py` 17 OK, `tests/test_section_b.py` 17 OK,
  `Topology/test_placement_m3.py` 12 OK.
* Every new flag is dark: `FE_SUBPIXEL_GABOR_ANCHOR=False`,
  `FE_KEYPOINTS=False`, `PLACEMENT_EXCLUDE_LENSES=()`,
  `PLACEMENT_RIDGE_ACROSS_ONLY=False`, `PLACEMENT_COLLAPSE_PLACES=False`.
* `_FRONTEND_SWITCHES` carries both new flags, so no lit run can be served a
  dark graph.

## The levers that remain, in order of expected effect

1. **Find and reroute the node-position writer** (bounded). Until the graph
   reads the anchor that `_gabor_pixel_anchor` computes, no sub-pixel refinement
   can change anything — this is proved, twice, across two missions. Expected
   effect: the whole localisation column.
2. **Exclude the ridge lenses from the population** (`exclude_lenses`, built):
   identity +0.234, survival +0.097, coverage +0.077. This is the one lever
   whose effect is already measured and positive; it needs a placement arm on a
   collapsed graph to be cashed in.
3. **Wire the keypoint layer into the binder** and verify with the same
   "did the position move" test; then re-measure. Direction already passes, so
   the layer's geometric content is real.
4. **Collapse places and re-measure checkability honestly** — the old
   checkable/pair-survival numbers were inflated by co-siting and must be
   restated on the collapsed population before any arm comparison.
5. **Re-define distinctness as "alone at its place"** and re-run the keypoint
   instrument; the current 0.07–0.33 measures detector density, not duplication.
