# Mission 6 — steps 1–4: the writer, keypoint places, attach, DEV-5

Branch `arena/c316ca3a-percept6`, from `8b3e4aa`. Commits: `5d629a7` (writer),
`a1e52ac` (writer measured), `4f0ee74` (keypoints wired), plus step 3
(coverage) and step 4a (DEV-5 scenes).

---

## 1. The writer — fixed, and tested both ways

`Pipeline._derive_anchor_positions` built the binder's map from
`observation.positions`: **region index → region centroid**. That map is what
`ConceptGraph.note_pattern_node_position` records, so every tiling lens bound
its node to a centroid regardless of what its own cell had measured. The
sub-pixel Gabor anchor was written under the 4-tuple cell key and **nothing ever
looked that key up**.

Found by printing the chain, as the mission asked: binding one node showed an
`int` key and a `(x, y, 0.0)` centroid; wrapping `Binder._remember_node_position`
showed **0 of 984** bound detections ever carried a lens-measured place, which
proved the lens tier is not the writer.

Fixed behind the existing dark flag:

* `Detection._place_pos` — carried like `_frag_mid`, invisible to signature,
  equality and hash.
* `gabor_place_positions(obs, positions)` — for each region, the strongest
  `(theta, sf)` cell at the centroid, then that cell's sub-pixel position.
* `_derive_anchor_positions` merges it under the flag — **this is the writer**.
* `Binder._remember_node_position(nid, anchor, det)` prefers `_place_pos`.
* The Orientation lens gives each emitted detection its region's best place.

| check | result |
|---|---|
| flag off → byte-identical | **PASS** — `4707831d1393797f`, 657 nodes / 1167 edges, identical to a clean worktree of `8b3e4aa` |
| flag on → the node moves | **PASS** — non-integer anchors 153 → **341** of 629 nodes |

Like-for-like, 6 FIT scenes, same instrument and truth:

| view step | repeat off → on | loc median | loc p90 | identity |
|---|---|---|---|---|
| w1_rot | 0.306 → **0.459** | 1.275 → **1.114** | 2.795 → 2.617 | 0.625 → 0.507 |
| w2_translate | 0.413 → **0.464** | 1.142 → **0.969** | 2.591 → 2.554 | 0.579 → 0.537 |
| w3_scale | 0.394 → 0.391 | 1.533 → **1.336** | 2.673 → 2.696 | 0.686 → 0.540 |
| mean | 0.371 → **0.438** | 1.317 → **1.140** | 2.69 → 2.62 | 0.630 → 0.528 |

Localisation −13%, the first movement any sub-pixel lever has produced. The p90
does not move, so the gain is in the median.

## 2. Collapse does NOT recover the identity loss — hypothesis falsified

The obvious prediction after step 1 was that collapsing places would recover the
identity drop. Measured on the writer-fixed graph, 6 FIT scenes:

| variant | nodes | repeat | identity | loc med | checkable | survival |
|---|---|---|---|---|---|---|
| A baseline | 4233 | 0.438 | 0.528 | 1.127 | 0.659 | 0.643 |
| **B exclude ridge** | 2604 | 0.452 | **0.728** | 1.112 | **0.747** | **0.744** |
| D collapse | 4233 | 0.438 | **0.528** | 1.127 | 0.576 | 0.599 |
| E exclude + collapse | 2604 | 0.452 | **0.728** | 1.112 | 0.599 | 0.614 |

Collapse leaves identity **exactly unchanged (0.528)**, and that is not a bug:
identity is a per-node measure and collapse changes only which EDGES exist.
Collapse lowers checkable share and survival, which is the same finding as
Mission 5 — those metrics were inflated by co-sited duplicates.

The identity loss from step 1 is therefore NOT recovered by collapse. What does
recover it is the ridge exclusion (0.528 → **0.728**), which is now measured on
the writer-fixed graph too.

## 3. Keypoint places wired — with the test

`lenses/keypoints.py` gained `signature_of` (CANONICAL, quantised over
orientation/12, coherence/4, scale, kind, dim — never position), `anchor_of` (a
`("kp", kind, idx)` key, never a region int) and `as_detections` returning the
detections plus the anchor→position map. `build_graph` adds them to `D_bind`
after surround suppression — a rule written for the tiling lenses must not thin a
keypoint layer.

| check | result |
|---|---|
| flag off → byte-identical | **PASS** — `4707831d1393797f` |
| flag on → positions ARE the detector's | **PASS** — 96 keypoint nodes, **96 of 96** carry a stored position that is one of the detector's own sub-pixel positions |

### The structural finding this exposed

**405 keypoint detections become 96 nodes.** A pattern node is keyed by
`(kind, signature hash)`: **a node is a SIGNATURE CLASS, not a place.** Every
place sharing a signature folds into one node with one position, first write
wins. So "one PLACE per physical feature" cannot be reached by adding
detections — it needs a place layer that is not the pattern-node layer. This is
the top structural item in the whole programme, ahead of any tuning, and it also
explains the co-siting census from Mission 5 from the other side.

## 4. Attach, don't drop — coverage measured on the ambiguous population

The instruction was to re-attach what Mission 5 dropped. Radius stated before
measuring: **4.0 px** (the front end's own `TARGET_RADIUS_SCALES`). 6 FIT
scenes, 961 ambiguous object-anchored nodes of attribute/ridge lenses, 404
keypoint places per graph:

| lens | n | coverage | median dist |
|---|---|---|---|
| Surface | 25 | 0.640 | 1.34 |
| BorderOwnership | 126 | 0.690 | 1.44 |
| TextureBoundary | 156 | 0.705 | 0.96 |
| Orientation | 48 | 0.708 | 3.10 |
| Boundary | 76 | 0.724 | 2.47 |
| RegionShape | 39 | 0.744 | 0.93 |
| Position | 47 | 0.745 | 2.56 |
| ContourFragment | 349 | 0.756 | 2.00 |
| Color | 34 | 0.794 | 2.47 |
| Texture | 59 | 0.831 | 2.18 |
| **ALL** | **961** | **0.736** | **2.00** |

**0.736 clears the 0.65 bar**, so Mission 5's exclusion was not necessary: three
in four dropped nodes have a keypoint place within 4 px.

## 5. DEV-5 scenes — built, verified

`Topology/placement_dev5.py`: SPHERE (control, new seeds), BOX, CYLINDER, and
OCCLUSION (a sphere pair with `posed(occluder=True)`, so part of the front
object is undetectable by construction). Four steps: 5°, 0.04 translation, 15°
(reported separately — the gate's localisation bar only covers ≤5°), 1.15 scale.
FIT/CHECK split by scene, stratified by family, decided before anything is
fitted.

Verified end to end: all families render and build — nodes/corners at the
reference view SPHERE 665/18, BOX 610/17, CYLINDER 533/9, OCCLUSION 654/15 (the
occluder is a third object). `posed` already scales and poses a `Cylinder`, so
the "cannot pose a Cylinder" note carried since Mission 4 is **stale and wrong**.
`collect_fit` gained a `pose_fn` hook (DEV-5 states its steps and applies the
occluder; DEV-3 derives them from a name hash), and a 2-scene pilot run fits both
the LLR and the presence densities on DEV-5 (μ_wrong −0.293, 546 true / 4173
wrong candidates).

## 6. What is NOT done, and why

**The DEV-5 arm table is not run.** The plumbing is verified and the scenes
exist, but an arm comparison needs enough scenes to support the MDE the mission
requires (< 0.03), and a two-scene pilot cannot. Reporting an arm table from it
would be exactly the substitution the mission forbids, so there is none. A
powered run needs roughly 32 scenes × 4 views ≈ 160 graph builds (~27 min of
build alone) plus the eight arms and the scene-clustered intervals.

Also not done: **the place layer** (section 3) — the node model is a signature
class, so a real place layer is a change to `ConceptGraph`, not to the detector,
and it should be designed before more arms are run on a graph that cannot
represent places.

## 7. State

* Suites 17 + 17 + 12 OK.
* Every flag dark: `FE_SUBPIXEL_GABOR_ANCHOR`, `FE_KEYPOINTS`,
  `PLACEMENT_EXCLUDE_LENSES`, `PLACEMENT_RIDGE_ACROSS_ONLY`,
  `PLACEMENT_COLLAPSE_PLACES`. All in `_FRONTEND_SWITCHES`; the cache key is
  pixels + switches + the 84-file source closure.
* Defaults byte-identical: `4707831d1393797f`.

## 8. Next, in order

1. **Design the place layer** — one node per place, not per signature class.
   Everything else is blocked on it: coverage, distinctness, and the arm
   comparison all assume places exist in the graph.
2. Attach ridge/attribute nodes to places (coverage is already 0.736 at 4 px, so
   the data supports it).
3. Run the powered DEV-5 arm table once places exist.
4. EVAL only after the DEV gate passes, pre-registered and read once.
