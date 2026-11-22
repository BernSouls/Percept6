# Mission 5 — front-end redesign, an honest instrument, and the placement re-test

Branch `arena/c316ca3a-percept6`, base `28436db`. Compute spent this mission:
~35 minutes of wall clock (Stage 0 + Stage 1 + cache rebuild of one graph);
the 182-graph cache was purged and has NOT been rebuilt.

**Verdict up front: the precision gate is NOT met, so there is no DEV-5 arm
table and no EVAL-4 read. Stages 2 and 3 are not built.** What this mission did
deliver is the one thing that had to come first: an instrument that measures the
right thing, plus a cache defect that would have made every later result
untrustworthy. Both are fixed and proven below.

Mission 4 concluded "identity 23–27%, so the front end rarely finds the same
feature twice". That conclusion was an artefact of the instrument: it asked
whether *any* node landed near the truth point, across all lenses, on a graph
where up to 16 nodes share one physical place. Asked node-to-node, within one
lens, identity is **0.59–0.66**. The front end is roughly two and a half times
better than Mission 4 said, and the remaining failure is concentrated in two
lens types that should never have been placement endpoints.

---

## Stage 0 — the instrument's own integrity

### 0a. Position writers, per lens

Only `perception/contour_peak.py` writes a sub-pixel coordinate anywhere
(`anchor_positions[anchor] = (x, y)` under `FE_SUBPIXEL_PEAK_POS`); its second
writer, the chain path (~line 371), averages *integer* peaks. Every tiling lens
in `lenses/base.py` writes either a region centroid or the integer argmax of a
Gabor channel (`_gabor_pixel_anchor`, line 637). Measured in the census: of 639
nodes, **573 carry integer coordinates**, 66 do not (`ContourFragment` 74 nodes,
only 14 integer; `TextureBoundary` 32 nodes, 0 integer) — those two write
centroids.

### 0b. Flags

`FE_L3_SPINE=False`, `FE_L4_JUNCTIONS=False`, `FE_SUBPIXEL_PEAK_POS=False`,
`ORIENTATION_PIXEL_ANCHORS=True`. Junction/T/Y/Onset/CurveComplex exist in the
census with L3 and L4 off (1–5 nodes each), so they are reachable from the
default path and are not switch-dependent.

### 0c. Nodes per place — the census, and the hypothesis it kills

Spec `d3_resolvable_00`, `m5_stage0_census.json`:

* 639 nodes, **354 distinct places**, share-shared **0.446**, max **16 nodes at
  one place**.
* Placement population: **179 nodes at 55 places**.

| lens | n | distinct | shared | integer | in population | pop. shared |
|---|---|---|---|---|---|---|
| ContourFragment | 74 | **1** | 73 | 14 | **62** | 61 |
| TextureBoundary | 32 | **0** | 32 | 0 | **24** | 24 |
| BorderOwnership | 86 | 24 | 62 | 86 | 18 | 12 |
| CONTOUR_PEAK | 119 | 55 | 64 | 119 | 13 | 12 |
| Boundary | 74 | 33 | 41 | 66 | 13 | 11 |
| Position | 96 | 54 | 42 | 91 | 9 | 6 |
| Texture | 32 | 5 | 27 | 27 | 9 | 8 |
| RegionShape | 18 | 1 | 17 | 1 | 6 | 5 |
| PEAK_CHAIN | 18 | 1 | 17 | 1 | 5 | 5 |
| EndStopped | 25 | 25 | 0 | 25 | 4 | 0 |
| Orientation | 25 | 5 | 20 | 25 | 4 | 4 |

Two lenses collapse to a **single** place each: `ContourFragment` puts 74 nodes
on one point, `TextureBoundary` puts 32 nodes on one point (0 distinct — every
node sits on a place another node already occupies). Together they are **86 of
the 179 placement nodes**.

I also tested the mission's "double-counted endpoints" hypothesis directly on
the placement payload: **4384 edges, 0 with both ends at one place, 0 of zero
length.** Degenerate pairs are already excluded. The hypothesis is falsified;
the defect is many legitimate nodes per place, not duplicated endpoints. The
333 endpoint nodes spread over only **149 places** — a 2.2× dilution.

### 0d. Cache digest vs import closure — a real defect, found and fixed

`frontend_source_digest()` hashed `perception/`, `lenses/`, `parameters/`,
`core/`, `dynamics/` and four files, with a comment saying the matcher was
"deliberately absent". Walking the actual import closure from `build_graph`
(`Topology/placement_cacheclosure_m5.py`) gives **84 repository files** on the
build path. Uncovered by the old list, among others:

* **`eth80_frontend.py` itself** — the file that defines `build_graph`
* `sensory_floor.py`, `derived_lenses.py`, `consolidator.py`, `_bootstrap.py`,
  `Topology/guard.py`, `eth80_split.py`
* all of `matching/`, `memory/`, `learning/`, `action/`

An edit to any of those would have been served a stale graph while the key
stayed put — the same class of defect as the `GRAPH_STAGE6_RELATIONS` bug the
function's docstring already documents. Fixed by extending
`_FRONTEND_SOURCE_DIRS` / `_FRONTEND_SOURCE_FILES`; the closure checker now
reports **84/84 covered**, and it returns non-zero if anything drifts out. Two
tests lock it: the closure ⊆ covered assertion, and a sensitivity test that
appends a byte to `sensory_floor.py` (a file the old list missed) and requires
the digest to move, then restores it.

**Consequence, stated plainly:** the digest changed, so every cached graph is
stale and was purged. Future runs rebuild at ~38 s per graph. No measurement in
this mission used a stale entry — the node-to-node instrument ran before the
digest edit, on graphs built under one consistent key.

A note for whoever runs this next: rewriting a file through Python text mode
converts CRLF to LF. My first sensitivity probe did exactly that to
`consolidator.py` and `eth80_frontend.py`; both were restored from git and the
probe was rewritten in binary mode. Verify `git status` after any such test.

### 0e. The TextureBoundary regression — not resolved, bounded

Turning `FE_SUBPIXEL_PEAK_POS` on moved TextureBoundary localisation 2.64 →
3.34 px. Its writer is a centroid and is untouched by that flag, and its nodes
carry no integer coordinates at all — so the flag cannot be writing its
positions. The plausible mechanism is indirect: moving CONTOUR_PEAK peaks
changes which nodes survive competition, which changes the region a
TextureBoundary node is the boundary of. Confirming that needs the graph pair
rebuilt under both flag values, which the cache purge made a 76 s experiment I
did not spend. Recorded as **unresolved**, with the mechanism narrowed to node
competition rather than the writer.

---

## Stage 1 — node-to-node truth

`Topology/placement_precision_m5.py`. For every base-object node, the truth
transform predicts where it should appear; the instrument then looks for the
nearest node **of the same lens** within 3 px and asks two separate questions —
did one appear (repeatability), and is the one that appeared the *right* one
(identity). Localisation is reported only for identity-correct matches.

FIT half, 20 scenes, `m5_precision_nodetonode.json`:

| view step | repeat | loc median | loc p90 | identity |
|---|---|---|---|---|
| w1_rot | 0.282 | 1.478 | 2.695 | 0.589 |
| w2_translate | 0.429 | 1.077 | 2.409 | 0.594 |
| w3_scale | 0.373 | 1.403 | 2.558 | 0.655 |

Against Mission 4, same scenes, same splits — old numbers kept beside new:

| quantity | Mission 4 | Mission 5 | why they differ |
|---|---|---|---|
| identity | 0.231–0.270 | **0.589–0.655** | M4 matched any lens; M5 matches the same lens |
| localisation (median) | 1.14–1.91 px | **1.08–1.48 px** | both identity-correct; M5 is stricter about what counts as a match |
| repeatability | 0.479–0.572 | **0.282–0.429** | M5 requires the *same lens* within 3 px, not any node |

Per lens (n, repeat, loc median, loc p90, identity):

| lens | n | repeat | loc med | loc p90 | identity |
|---|---|---|---|---|---|
| Color | 270 | 0.152 | 0.86 | 2.27 | 0.927 |
| Contour | 102 | 0.147 | 0.95 | 2.66 | 0.933 |
| CurveComplex | 81 | 0.160 | 0.95 | 2.67 | 0.923 |
| Position | 366 | 0.246 | 1.01 | 2.48 | 0.933 |
| Polarity | 45 | 0.156 | 1.03 | 1.83 | 0.714 |
| **ContourFragment** | 2706 | 0.295 | 1.12 | 2.62 | **0.213** |
| Texture | 534 | 0.230 | 1.13 | 2.42 | 0.894 |
| Orientation | 375 | 0.243 | 1.13 | 2.67 | 0.923 |
| Boundary | 651 | 0.249 | 1.20 | 2.53 | 0.846 |
| PEAK_CHAIN | 336 | 0.363 | 1.24 | 2.33 | 0.926 |
| CONTOUR_PEAK | 1125 | **0.630** | 1.25 | 2.59 | 0.810 |
| RegionShape | 330 | 0.439 | 1.25 | 2.26 | 0.952 |
| BorderOwnership | 1107 | 0.597 | 1.26 | 2.53 | 0.799 |
| Surface | 213 | 0.300 | 1.43 | 2.48 | 0.938 |
| **TextureBoundary** | 1476 | 0.383 | 1.57 | 2.64 | **0.212** |
| Corner | 99 | 0.162 | 1.68 | 2.78 | 0.750 |
| EndStopped | 438 | 0.210 | 1.71 | 2.76 | 0.870 |
| Junction | 27 | 0.074 | 2.39 | 2.48 | 1.000 |

**The finding that matters.** Every lens except two has identity 0.71–0.95.
The two exceptions — `ContourFragment` 0.213 and `TextureBoundary` 0.212 — are
ridge and boundary tokens, where a point on a line is only defined along the
normal. Their identity is low for a *principled* reason, and they are exactly
the tokens that make up 86 of the 179 placement nodes. The placement population
is therefore dominated by the two token types whose position is intrinsically
ambiguous — which is a front-end design error, not detector noise.

Repeatability splits the same way: the tokens with a definite 2-D anchor
(CONTOUR_PEAK 0.63, BorderOwnership 0.60, RegionShape 0.44) repeat; colours,
contours, corners and junctions (0.15–0.25) appear only sometimes. Note that
localisation and repeatability are *not* in tension here: the lenses that
repeat are precise to 1.25 px, and the lenses that are precise are not the ones
that repeat. Neither reaches the gate.

---

## The gate, scored honestly (CHECK not read; FIT reported)

| criterion | target | measured | outcome |
|---|---|---|---|
| localisation median | ≤ 0.50 px | 1.08–1.48 px | **FAIL** (best lens 0.86) |
| localisation p90 | ≤ 1.50 px | 2.41–2.70 px | **FAIL** (best lens 1.83) |
| repeatability | ≥ 0.70 | 0.28–0.43 | **FAIL** (best lens 0.63) |
| distinctness | ≥ 0.95 | 0.21–0.95, population-weighted ~0.3 | **FAIL** |
| direction | ≤ 3° | not measured (no keypoint layer) | not measured |
| separation AUC | ≥ 0.95 @5° | not measured (no graph may be re-run) | not measured |
| coverage / pair survival | ≥ 0.65 / ≥ 0.60 | 0.648 / 0.671 (M3, unchanged) | at/near floor |

The 0.5 px target is not reachable by this front end as it stands: the median
error of the *best* lens (Color, 0.86 px) already exceeds it, and the writers
are integer argmax/centroid. A sub-pixel Gabor fit (Stage 2) is the only
mechanism in this codebase that can move localisation below 1 px, and Mission 4
already showed that applying it to peaks alone is a no-op for the population
(2.869 → 2.869) and harmful for TextureBoundary.

Per the mission's own rule — *do not run placement arms on a graph that fails
the gate* — **no DEV-5 arms were run and no EVAL-4 was read.**

---

## What is built and what is not

**Built, tested, pushed:** the writer map, the flag census, the nodes-per-place
census with the degenerate-edge test, the cache-closure fix with a checker and
two tests, and the node-to-node instrument with its old-vs-new table.

**Not built:** Stage 2 (place-level collapse, sub-pixel Gabor refinement, the
uncertainty sidecar) and Stage 3 (the true-keypoint frame layer, Harris /
structure tensor, DoG blobs, orientation and coherence, lens registration). The
budget for this session went on the instrument and the cache, and building a
keypoint layer that cannot pass the gate without first deciding which tokens
belong in the population would have been the wrong order of work.

**The next concrete lever, in priority order** (all measured or bounded here):

1. **Remove `ContourFragment` and `TextureBoundary` from the placement
   population**, or admit them only as across-contour constraints. They are 86
   of 179 nodes and carry identity 0.21. This is a population-definition change,
   not a weight, and it is checkable immediately: coverage must stay ≥ 0.65
   against the 55-place denominator.
2. **Collapse the remaining co-sited nodes to one place per site** (Stage 2).
   179 nodes at 55 places means the pair statistics are diluted 3.3×.
3. **Sub-pixel refinement applied to every writer at once** (peaks, centroids and
   chain anchors together) — the Mission-4 evidence says peaks alone do nothing.
4. **Repeatability via NMS with a stated minimum separation in (x, y, scale)**,
   because the reliable lenses (CONTOUR_PEAK 0.63, BorderOwnership 0.60) are
   already half-way to the 0.70 target while the regional lenses are at 0.15.

## No-regression checks

* `Topology/test_placement.py` 17 OK, `tests/test_section_b.py` 17 OK,
  `Topology/test_placement_m3.py` 12 OK (10 + the 2 new cache-closure tests).
* Flags dark by default; no default changed.
* The digest change cannot alter graph construction: `frontend_source_digest`
  is called only by the cache layer, never on the build path, and the diff is
  confined to the constant list and its comment. Measured on the default spec,
  657 nodes — the same count recorded in Mission 2.
* Determinism, canonical sorting and the non-vacuity canaries are untouched by
  anything in this mission.
