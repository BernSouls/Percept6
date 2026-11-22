# Mission 6 — the node positions are real (step 1), and what that bought

Branch `arena/c316ca3a-percept6`, from `8b3e4aa`. Commits: `5d629a7` (the
writer fix) plus the measurement and this report.

## 1. The writer — found, fixed, tested both ways

The position a node keeps came from `Pipeline._derive_anchor_positions`, which
builds its map from `observation.positions`: **region index → region centroid**.
Every tiling lens bound its node to a centroid, whatever its own cell had
measured. The sub-pixel Gabor anchor was written under the 4-tuple cell key
`(u, theta, sf, kind)` into `anchor_positions`, and **nothing ever looked that
key up**. That single mismatch is why Mission 4's `FE_SUBPIXEL_PEAK_POS` and
Mission 5's `FE_SUBPIXEL_GABOR_ANCHOR` both moved exactly zero nodes.

I found it the way the mission asked — by printing the call chain. Binding one
node showed the lookup key is an `int` region index and the value a
`(x, y, 0.0)` centroid; wrapping `Binder._remember_node_position` showed **0 of
984** bound detections ever carried a lens-measured place, which is what proved
the lens tier is not the writer and sent me to the pipeline.

The fix (all behind the existing dark flag `FE_SUBPIXEL_GABOR_ANCHOR`):

* `core/types.Detection` gains a `_place_pos` slot, carried exactly like
  `_frag_mid`: not in the signature, equality or hash, so the node is the same
  node and a dark flag cannot move a default bit.
* `lenses/base.gabor_place_positions(obs, positions)` — the writer. For each
  region: the strongest `(theta, sf)` cell at the centroid (one lookup, no
  scan), then that cell's sub-pixel position through `_gabor_pixel_anchor`
  (U26 tile/RF rule plus the quadratic refinement). The `z` of the original
  entry is preserved.
* `Pipeline._derive_anchor_positions` merges that map under the flag, so the
  binder's own map carries measured places, not centroids.
* `Binder._remember_node_position(nid, anchor, det)` prefers the detection's
  `_place_pos`; with none it is exactly the old lookup.
* The Orientation lens also hands each emitted detection its region's best
  measured place, so a lens-level route exists.

### The test the mission specified

| check | result |
|---|---|
| flag **off** → graph byte-identical | **PASS** — clean worktree at `8b3e4aa` and this tree both give 657 nodes / 1167 edges and the same sha256 over (node, anchor) + (edge) triples: `4707831d1393797f` |
| flag **on** → the node moves | **PASS** — non-integer anchors 153 → **341** of 629 nodes; node count 657 → 629 as competition shifts |

Suites pass (17 + 17 + 12). Two dead ends are recorded in the commit so they are
not repeated: the first lens-level version set nothing because `alpha` is a
**numpy** integer (`isinstance(x, int)` is False), and fixing only the binder
cannot work because the bound detections never carry a lens place.

## 2. What the fix bought, like-for-like

Same 6 FIT scenes, same instrument, same truth, flag off vs on
(`m6_prec_writer_off.json`, `m6_prec_writer_on.json`):

| view step | repeat off → on | loc median off → on | loc p90 off → on | identity off → on |
|---|---|---|---|---|
| w1_rot | 0.306 → **0.459** | 1.275 → **1.114** | 2.795 → 2.617 | 0.625 → 0.507 |
| w2_translate | 0.413 → **0.464** | 1.142 → **0.969** | 2.591 → 2.554 | 0.579 → 0.537 |
| w3_scale | 0.394 → 0.391 | 1.533 → **1.336** | 2.673 → 2.696 | 0.686 → 0.540 |
| mean | 0.371 → **0.438** | 1.317 → **1.140** | 2.69 → 2.62 | 0.630 → 0.528 |

**Localisation improves by 0.18 px (−13%) and repeatability by +0.067 — the
first movement any sub-pixel lever has produced in three missions.** The p90
does not move (2.69 → 2.62): the gain is in the median, and the tail is still
the periodic-texture aliasing that Stage 3 identified.

**Identity falls (0.630 → 0.528), and that is reported, not hidden.** Two causes,
neither of which is a regression in the front end:

1. `track_optimal` is recomputed on the moved graph, so the "right partner"
   changed with the positions; a node that moved onto a peak now sits nearer a
   *different* node's predicted spot.
2. Nodes converge onto shared peaks — the census already measured up to 16 nodes
   at one site, and moving them onto the same measured peak makes ties more
   likely, which identity counts as a miss.

The correct next move is therefore not to undo the fix but to **collapse
places** (Mission 5's `collapse_places`, already built and dark) so that a tie
at one peak is one place, not a failure. That is unmeasured on the new graph and
is the first thing the next run should do.

## 3. What is NOT done

Stated plainly, because the mission's DONE list asks for it:

* **Keypoint places are not wired into the binder.** `lenses/keypoints.py`
  detects and measures (repeatability 0.691, localisation 0.758–0.872 px,
  direction 0.61°, registered lens types, dark flag, cache-keyed) but it does
  not mint graph nodes. The obstacle is now identified rather than assumed: a
  detection becomes a node only through `_emit`'s signature machinery, so the
  wiring needs (a) a `LensInputDeclaration` for the layer with its slot names,
  (b) a `TypedSignature` whose quantised slots are stable across views (the
  identity measure depends on it), and (c) a entry in
  `_LENS_DETECTION_CAPS`. That is a build, not a patch, and doing it
  half-tested would have broken the deterministic graph.
* **No attach-don't-drop.** Ridge and attribute nodes were excluded in Mission 5
  rather than attached to keypoint places. Coverage on the *ambiguous*
  population (not just placement nodes) is therefore not restated.
* **No place-level scoring restatement** on the new graph, and **no DEV-5 run**.
  Step 6 needs a scene set that includes box/cylinder/occlusion families; those
  scenes do not exist yet (`posed` cannot pose a `Cylinder`, carried from
  Mission 4), so DEV-5 as specified cannot be run without building them first.

Reporting a DEV-5 arm table from the current scenes would be exactly the
substitution the mission forbids, so there is none.

## 4. Where that leaves the gate

| criterion | target | best measured | change this mission |
|---|---|---|---|
| localisation median | ≤ 0.50 px | 1.114 px (front-end nodes) / 0.758 (keypoints, unwired) | **−13%**, first real movement |
| repeatability | ≥ 0.70 | 0.459 (front-end) / 0.691 (keypoints, unwired) | **+0.067** |
| identity / distinctness | ≥ 0.95 | 0.528 | −0.10, explained above |
| direction | ≤ 3° | 0.61° (keypoints) | unchanged, passing |

The writer was the floor under every previous localisation number. It is now
gone, and the numbers moved. What remains between 1.11 px and the 0.50 px gate
is: keypoints unwired (they measure 0.758 px already), place collapse not
applied to the new graph, and the periodic-texture tail behind the p90.

## 5. Next, in order

1. Re-run with `collapse_places=True` on the writer-fixed graph — it should
   recover the identity loss this mission introduced and restate checkable share
   and pair survival on a population without duplicates.
2. Wire the keypoint layer (declaration + signature + cap), tested with the same
   proof used here: flag off byte-identical, flag on positions equal to the
   detector's.
3. Attach ridge/attribute nodes to the nearest keypoint place and report
   coverage on the ambiguous population.
4. Build the box/cylinder/occlusion DEV-5 families (`posed` must pose a
   `Cylinder`), then run the arms.
