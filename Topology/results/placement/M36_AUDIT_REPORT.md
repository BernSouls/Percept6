# MISSION 36 — AUDIT OF `perception/`, `lenses/`, `core/`

**Status**: every claim was checked against the code and, where a number was
claimed, against a measurement of this repository. Confirmed defects were fixed
behind OFF-by-default switches (plus two unflagged core fixes), each with a test
that goes RED when the fix is reverted. Claims that did not survive are withdrawn
in writing below (§2). Nothing was fixed on an unverified claim.

**Gates at the end of this pass**

| gate | result |
| --- | --- |
| `pytest tests -q` | **284 passed, 6 skipped, 4 subtests passed** (406.5 s) — the 272 of M35 plus the 12 new audit tests |
| graph drift, all switches OFF | **40 cells (10 objects × 4 views), 0 diffs** on a digest of node count, every signature's (class, relation, quantized), and every edge — tree identical to `d69057d` |
| matcher scores, all switches OFF | 3 M35 pairs re-scored: **+5.8e-4 / +6.4e-4 / +1.2e-3**, and the baseline tree reproduces the stored values to the last digit (§6) |
| determinism | unchanged from M35 (the fixes add no state) |

---

## 1. Verified, confirmed, fixed

### 1.1 A1 — `circularity` was a constant ladder, not a shape measurement

`perception/contour_structure.py`: `circularity` was `m50/m95` where **both** came
from `_mass_at(q) = cum[np.searchsorted(cum, q)]` — a *position in the cumulative
mass array*. The quantity is a property of the cumulative ramp, not of the shape.

Measured (40×40 footprints, both terms computed by the shipped code):

| footprint | shipped `circularity` |
| --- | --- |
| disc (r = 12) | 0.5274 |
| square (25 × 25) | 0.5269 |
| sliver (6 × 38) | 0.5269 |

**Fix** — `BOUNDARY_CIRCULARITY_RADIAL=1`: `isoperimetric_circularity(win)`, the
isoperimetric ratio `4πA/P²` on the 10 %-of-peak footprint, with the perimeter as
the total variation of the smoothed footprint (the coarea identity makes that the
true perimeter; a 4-neighbour side count over-charges a diagonal edge):

| footprint | measured |
| --- | --- |
| disc | 1.000 |
| square | 0.876 |
| ellipse 12 × 32 | 0.736 |
| sliver 6 × 38 | 0.559 |

Test: `test_circularity_moves_with_shape` (RED on revert: the old branch cannot
produce a spread of 0.44 across the three).

### 1.2 A2 — `regularity` ranked a featureless blob above a perfect grating

`perception/rich.py::_compute_regularity` took the peak of the **unnormalised**
autocorrelation over its DC term. The DC term is the sum of squares, i.e.
dominated by the mean. Measured on 24×24 windows with the shipped code:

| window | shipped | fixed |
| --- | --- | --- |
| white noise | 0.096 | 0.000 |
| bright noise (mean 100, σ 5) | 0.874 | 0.000 |
| perfect 6-px grating | 0.875 | **1.000** |
| smooth Gaussian blob | **0.893** | 0.000 |

The blob outranked the grating and noise matched it: the slot was a mean-brightness
and smoothness detector.

**Fix** — `RICH_REGULARITY_CORRELATION=1`: Pearson autocorrelation (mean-subtracted,
variance-normalised, overlap-corrected for the finite window) read along both axes,
taken only **after the profile has gone negative** — a period requires an opposite
phase. Blob and noise never trough, so they score 0; the grating scores 1.
Test: `test_regularity_is_periodicity_not_brightness_or_smoothness`.

### 1.3 A3 — the L6 address discarded the node's content and its edges

`perception/front_end_mechanisms.py::_l6_vector_address` built `sigs[ni] = (kind,)`.
`compute_vector_addresses` reads element 1 as the `TypedSignature` and hands it to
`_content_vector`; element 1 was never there, so **two graphs with the same node
and edge types and completely different geometry hashed to the same address**.
The edge branch was guarded by `isinstance(cg.edges, dict)`; `ConceptGraph.edges`
is a dict (so the live binder path is fine) but a cell's `subgraph.edges` is a
frozenset and was dropped entirely.

**Fix** — `FE_L6_SIGMA_CONTENT=1`: the signature itself is element 1, and an
iterable edge set is accepted as well as a dict.
Test: `test_l6_address_carries_the_node_signature` (two signatures of the same class
must give different content codes).

### 1.4 A4 — `roughness` (`var(joint_p) × 100`) is inverted and compressed

Measured on the shipped cell (256 B4 rich signatures):

* `var(p) × 100` at the **uniform** distribution over the (θ, sf) grid = **0.0**
  (the maximally rough texture scores "smoothest");
* one cell holding 60 % of the energy → **0.877**;
* **17.2 %** of the 256 regions sat at exactly **1.0** (the clip).

**Fix** — `RICH_ROUGHNESS_FULL_SCALE=1`: the normalised entropy `H(p)/log N ∈ [0,1]`
(uniform = 1.0, one cell = 0.0), which is what the slot's name says.
Test: `test_roughness_is_normalised_entropy_not_var_times_100`.

### 1.5 A5 — `contrast_energy` had two values on real data

`contrast = tot_energy / (tot_energy + lum_mean)` compares a **sum over the window**
with a **per-pixel** luminance, and the `< 2 significant cells` gate set `ABSENT`
(→ 0). Measured on the shipped cell, 256 windows:

* **202** exactly 0 (the gate);
* the other **54** all inside **[0.99894, 0.99999]** — a 0.001-wide band at the top.

**Fix** — `RICH_TEXTURE_ABSENT=1`: the energy is divided by the window's pixel count
first, so the ratio compares like with like and keeps its range (measured on the same
cell: 0.030 and 0.238 for two windows differing by ×10 in energy).
Test: `test_texture_contrast_is_not_a_two_valued_slot`.

### 1.6 A6 — a detection did not match itself when its population was flat

`core/population.axis_similarity` is the non-negative part of the *centred* cosine.
A **flat** activity vector (a uniform marginal — an uninformative axis) has a zero
centred vector, so the correlation is undefined and the old code fell through to
**0.0**: two identical populations compared to 0.0, contradicting the function's own
docstring ("Identical -> 1.0"). Measured: a REGION_SHAPE population uniform on every
axis scored 0.0 against itself and 1.0 against a peaked one — i.e. **self-similarity
was 0.0 and unrelated similarity 1.0**.

**Fix** (unflagged — a self-match must be 1.0): the identity case is tested first.
Chance correction still applies to every non-identical pair.
Test: `test_identical_population_scores_one`.

**Consequence for scores**: this is the one fix that is not gated, so it moves
matched-pair scores; see §6. It is a +6e-4 … +1.2e-3 change on the total score.

### 1.7 A7 — one anchor value, two byte strings

`core/place_key._anchor_bytes` tested `isinstance(anchor, int)`, which is **False**
for `np.int64`; a numpy index therefore fell through to `b"s" + str(...)`. Measured:
`_anchor_bytes(42) == b"i42"` but `_anchor_bytes(np.int64(42)) == b"s42"`, so a
salted node whose anchor came out of an `np.nonzero` minted a twin that no lookup
with a Python int could find.

**Fix** (unflagged): the integer test covers `np.integer`; `bool` is still tested
first and keeps its own encoding. Test: `test_anchor_bytes_numpy_scalar_is_the_same_bytes`.

### 1.8 B1 — `image_area` never existed on the observation

`lenses/v4.py:416` and `lenses/derived_common.py:1235` read
`getattr(observation, "image_area", 0.0) or 1.0`. `Observation2D` has **no**
`image_area` (verified: `hasattr` is False; it exposes `image_shape`), so the divisor
was always 1.0 and `log10(area / 1.0)` was positive for every closed outline, pinning
the area bin at its top.

**Measured consequence on the shipped cell: none** — and the reason matters. All 17
REGION_SHAPE polylines of this cell are **open chains**, whose shoelace `area` is 0 by
construction, so `area_norm = log10(max(0/1, 1e-6)) = -6` and the bin is 0 with or
without the fix (verified: area bin = 0, one distinct value, both configurations).
The defect bites **closed rings only**, where it pins them to the top bin.

**Fix** — `LENS_AREA_FROM_SHAPE=1`: the area comes from `image_shape[0] × image_shape[1]`.
Tests: `test_image_area_comes_from_the_frame_shape`.

### 1.9 B3 — `along` was 1.0 for every keypoint

`lenses/keypoints.py` computed `along = min(1.0, 0.5*(vx+vy) + 4.0)` with
`vx, vy ∈ [1e-3, 1]`, so the sum is `≥ 4.001` and `along` is **1.0 by construction**
for every keypoint — no measurement needed. The pair is bound positionally to
`var_x, var_y`.

**Fix** — `KP_ANISOTROPY_RATIO=1`: the two measured per-axis variances are emitted
instead of their mean and a constant. Test: `test_keypoint_variances_are_the_two_axes`.

*Caveat, stated rather than hidden*: `_parabola` clips `var = 1/(2|den|)` to `[1e-3, 1]`
and the M32 comment records that `|den|` is far below 0.5 for every place, so **both**
per-axis variances are themselves often at the cap. M32 already added the un-clipped
`sharp_x/sharp_y/curv_x/curv_y` for that reason; this fix removes the invented `+4.0`,
it does not restore information the clip destroyed.

### 1.10 C2 — three anchor namespaces, three silent misses

The bands are disjoint by construction:

| producer | `alpha` |
| --- | --- |
| field tiling (Color, Texture, Orientation) | region id `u` (0 … a few hundred) |
| CONTOUR_PEAK / BorderOwnership | `PEAK_ANCHOR_BASE + i` = 1e9 + i |
| PEAK_CHAIN | `PEAK_ANCHOR_BASE + CHAIN_ANCHOR_OFFSET + k` = 1e9 + 1e5 + k |
| Corner | 1e9 + 2e5 + k |

`SurfaceLens` looked up `bo_by[chain_anchor]` (peak band) and
`color_by[chain_anchor]` / `tex_by[chain_anchor]` (region band) with a **chain**
anchor; a chain anchor can equal a peak anchor only if a frame has ≥ 100 000 peaks
(it has 94), and a region id only if `u ≈ 1e9` (it is 0…285). `ContourLens` looked up
`bo_by_anchor[a_u]` with an **Orientation** (region) anchor in a peak-keyed map. All
four lookups therefore miss on **every row**; the contour-ownership test `continue`d
before it could run.

**Fix** — `LENS_ANCHOR_RESOLUTION=1` and the new `lenses/_anchors.py`: a chain's
BorderOwnership rows are resolved by **membership** (the member peaks its `rho` names),
and a region-keyed layer is resolved by the **nearest position** (the region the
fragment's centroid falls in). Every resolution is counted in
`lenses._anchors.ANCHOR_RESOLUTION`, so a zero-hit lookup is visible.
Tests: `test_anchor_bands_are_disjoint_and_the_lookups_missed`,
`test_chain_and_region_anchors_resolve_through_their_producer`.

### 1.11 C1 — place nodes are not content tokens (and the deeper cause is structural)

Census of the shipped cell (`data/ETH-80-master/1/1/apple1-035-045.png`, reproduced
twice in this pass): **1,506 nodes, 2,686 edges; 518 place nodes (Blob 261,
BoundaryArc 222, CornerPoint 35) — all 518 at degree 0; 0 of the 2,686 edges touches
a place.** (The mission text's "474 places" reproduces as 518.)

Two causes, and only the first is an audit-sized fix:

1. `CONTENT_TOKEN_LENSES` omitted the place classes, so `_content_node_positions`
   skipped every place. Fixed behind `FE_CONTENT_PLACES=1` (the three classes join
   the set; test `test_place_layers_can_become_content_tokens`).
2. **Nothing in the cell build links a place node at all.** Turning the switch on and
   also enabling `FE_L3_SPINE=1` / `FE_L4_JUNCTIONS=1` in this build path changed
   neither the node nor the edge count (1,514 / 2,938, both before and after): those
   stages are binder stages and do not run in a placement cell. With the flag on, the
   content-token set excludes no place, and the places still have degree 0.

So the finding stands as the **entry point for M36 proper** (the joined topology):
the place layer is minted and never joined, and joining it needs an edge stage, not a
set membership. Nothing was fabricated to hide it.

---

## 2. Withdrawn, in writing

| claim | verdict | evidence |
| --- | --- | --- |
| `_autocorr_peak` "never subtracts the mean; uses a 5×5 DC mask" | **WITHDRAWN** | the code does `e = energy - energy.mean()`, normalises by `var`, and masks `r_dist <= 2.5`. Read at `perception/rich.py:1217`. |
| CONTOUR_PEAK: "8 ranges against 7 axes, names misaligned" | **WITHDRAWN** | registry dump: 8 ranges, 8 `POPULATION_AXES`, and `SLOT_NAMES["CONTOUR_PEAK"] = (theta, sf, curvature, bend, phase, strength, coherence, endstop)` in the same order as the axes. |
| PEAK_CHAIN: "slot 3 is not `closure`" | **WITHDRAWN** | `SLOT_NAMES["PEAK_CHAIN"] = (n_peaks, mean_theta, theta_var, closure, curvature, length)`; index 3 **is** `closure`. Axis 3 of `POPULATION_AXES["PEAK_CHAIN"]` is `closure`. |
| REGION_SHAPE "self-similarity 0.952" | **WITHDRAWN as stated** | measured: a uniform population scored **0.0** against itself, a peaked one **1.0**. 0.952 does not reproduce. The *underlying* defect (identity broken for a flat axis) is real and is fixed at §1.6 — the audit's number was wrong, its direction was right. |
| `SIGNED_DEADZONE` "bins 3 and 4 abstain, so 25 % of the range is dead" | **WITHDRAWN as a defect** | `signed = (q + 0.5)/8·2 − 1`, so q = 3 → −0.125 and q = 4 → +0.125 and `|s| ≤ 0.125` abstains: the two **central** bins, which is the definition of a deadzone. The other 6 of 8 bins vote (measured slot `signed_dir`: 7 distinct values of 8 over 62 BorderOwnership rows). |
| CONTOUR_PEAK `Z_SCALE = 25.0` "against a p50 of 3.238" | **WITHDRAWN as stated** | the code's own instrumentation (`perception/contour_peak.py:299-304`) measures `integration_strength` over 2,643 peaks from 5 ETH-80 images: min 0.029, **p50 17.015**, max 25.446. 3.238 is not a quantity this code measures, and `25.0 = max` gives a median z of 0.68 — a graded, non-saturated confidence. |
| `relatable_completion` "always returns `[]`" | **WITHDRAWN** | no such symbol exists in the live tree (`perception/`, `lenses/`, `core/`, `matching/`, `orchestration/`). |
| `_gate_admissible` "missing `relations`" | **WITHDRAWN** | no such symbol exists in the live tree. |
| `_port_tangent_junction` "only called where it is guaranteed `None`" | **WITHDRAWN** | no such symbol exists in the live tree. |
| `IllusoryContourLens` "±90° error" | **WITHDRAWN on reading** | the lens converts both carrier angles with `carrier_to_tangent` and compares tangents against the chord bearing `phi = atan2(dy,dx) % π`; there is no ±90° term to remove. |

---

## 3. Confirmed, deliberately not fixed here

| finding | why it is not in this commit |
| --- | --- |
| `complexity ≡ 1 − isotropy` (exact complement, `perception/contour_structure.py`) | it is a **redundant definition and a mislabel**, but no `BOUNDARY` signature carries both slots (`BOUNDARY = (circularity, complexity)`), so the matcher does not see two copies of one quantity. A correct replacement (the participation ratio of the orientation marginal, a genuinely different measurement) changes every Boundary node and belongs with M36's measurement pass, not this one. |
| positions normalised by the hard-coded 127.5 / 7.5 | correct for the 256-px field every ETH-80 cell uses (`x/127.5 − 1 ∈ [−1, 1]`, verified). A size-independent normalisation is a corpus change, not a bug fix. |
| the `var = 1/(2|den|)` clip in `_parabola` | real, and already documented in the code (M32) with the un-clipped `sharp_*`/`curv_*` added to carry what the clip destroys. Nothing to fix without changing the keypoint constitution. |
| places at degree 0 | structural; see §1.11. This is M36's joined topology. |

---

## 4. What changed in the tree

| file | change |
| --- | --- |
| `core/place_key.py` | `_anchor_bytes` accepts `np.integer` (unflagged) |
| `core/population.py` | `axis_similarity` identity case → 1.0 (unflagged) |
| `perception/contour_structure.py` | `isoperimetric_circularity()`; the shipped ladder kept reachable under the flag |
| `perception/rich.py` | `_regularity_periodicity()`; entropy roughness; density contrast |
| `perception/front_end_mechanisms.py` | L6 signature content + iterable edges; `PLACE_TOKEN_LENSES` |
| `lenses/_anchors.py` | **new**: the band-aware anchor resolver, with counters |
| `lenses/v4.py`, `lenses/v2.py` | `_image_area`, `_anchors` routing behind the flags |
| `lenses/keypoints.py` | per-axis variances behind `KP_ANISOTROPY_RATIO` |
| `eth80_frontend.py` | the nine switches are in `_FRONTEND_SWITCHES` (a cached graph may not be served across them) |
| `parameters/_text.py` | the nine switches, each with the measurement that motivated it |
| `tests/test_m36_audit_fixes.py` | **new**: 12 tests, one per confirmed fix |

Every switch is **OFF** by default; the shipped numbers stay reachable. The two
unflagged fixes are the two that make a value mean what its own docstring says
(identity → 1.0; one anchor → one byte string).

Note: adding the switches to `_FRONTEND_SWITCHES` moves `switch_suffix()`, so the
graph cache is invalidated once (keys change, values do not — §5).

---

## 5. No drift in the shipped configuration

Ten objects × four views, digest = node count + every signature's
(class, relation, quantized) + every edge `(a, b, type)`, computed in a worktree of
`d69057d` and in the working tree with all nine switches off:

**40 cells, 0 diffs** (`/tmp/digest_base.json` vs `/tmp/digest_new.json`, sample
`apple/1|000-000 → e61d8f39a1857045`).

The M35 anchor gate (`Topology/m34_anchors.py --check`, 640 cells) was **not**
re-run: the switch-suffix change invalidates the whole graph cache, so that check
would rebuild 640 cells from scratch (~17 s each, ≈ 3 h). The 40-cell digest above
is the same comparison on a sample and is the evidence for this commit; the full
re-check is queued behind M36's own build.

---

## 6. Score comparison against M35

The M35 pair scores are reproducible **exactly** by the baseline tree — the three
re-scored pairs from `m35_pairs.jsonl`:

| pair | M35 stored | baseline tree (`d69057d`) | this tree |
| --- | --- | --- | --- |
| `0\|035-045\|5` | 0.3646661689621957 | **0.3646661689621957** | 0.3652495776665506 |
| `0\|035-045\|7` | 0.32293415250473884 | **0.32293415250473884** | 0.32357797210841904 |
| `3\|035-045\|3` | 0.41449762347912317 | **0.41449762347912317** | 0.41573519711848994 |

The deltas (+5.8e-4, +6.4e-4, +1.2e-3) are entirely the population-identity fix
(§1.6): the graphs are byte-identical, so nothing else can move a score. The sign is
the expected one — pairs whose two sides carry an identical flat population axis now
earn 1.0 on that axis instead of 0.0.

Ranking implication: the shifts are ~0.1 % of the score and do not reorder these pairs
(the gaps between them are 0.04 and 0.09). Any M36 number taken on a full set should
be read as "M35 + a +6e-4…+1.2e-3 per-pair offset from the identity fix", not as an
independent replication of the M35 table.

The audited measurement switches are all OFF, so an ON-configuration score table is
not part of this pass; that is the opening measurement of M36.
