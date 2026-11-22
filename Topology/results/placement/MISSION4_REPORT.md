# Mission 4 — Front-end precision

**Verdict: the precision gate is NOT met, and it is not close.** The front end's
positions are 2–4× too coarse, 89 % of the nodes placement uses sit on a
position another node already occupies, and three-quarters of nearest-position
matches are a *different feature*. The largest single reason geometry cannot
beat presence is now measured rather than suspected: **86.3 % of neighbour slots
have no counterpart at all**, so the geometry term is built on about one edge of
evidence per candidate.

Compute: Stage 0 ≈ 0.4 h, Stage 1 ≈ 0.6 h wall clock, lever tests ≈ 0.3 h.

---

## Stage 0a — was the flag on, and does the consumer read the sub-pixel position?

`FE_SUBPIXEL_PEAK_POS` **defaults to `False`**, so every Mission-3 graph was
built with integer grid-site positions. Confirmed structurally: 76.1 % of all
anchors are exact integers.

The **consumer path is correct**. `perception/contour_peak.py` writes the
sub-pixel position into `anchor_positions[...]`, that becomes
`anchor_of_node`, and the placement code reads `anchor_of_node` and nothing
else. There is no bug to fix in the consumer — the payload was simply fed grid
sites.

**Turning it on does not help.** Like-for-like, same four scenes, same poses:

| | flag off | flag on |
|---|---|---|
| localisation median, `w1_rot` | 2.869 px | 2.869 px |
| CONTOUR_PEAK median | 2.20 px | 2.20 px |
| TextureBoundary median | 2.64 px | **3.34 px** (worse) |
| distinct positions | 684/1252 | 683/1251 |

Sub-pixel refinement applies to `CONTOUR_PEAK` only, and the error is not
concentrated there. It is a no-op at the level that matters and a slight loss in
one lens.

---

## Stage 0b — does the absence penalty swamp weak geometry?

**No — and it cannot, because the padding is a per-node constant.** Same
estimator, FIT half only (no CHECK look spent), 2811 near-tie ambiguous nodes:

| arm | accuracy |
|---|---|
| P, padded at `mu_wrong` | 0.3277 |
| P, padded at 0 (neutral) | **0.3277** |
| P_presence, `mu_wrong` | 0.3553 |
| P_presence, neutral | **0.3553** |

`P_neutral − P_mu = +0.0000` **exactly**, and `P − P_presence` is −0.0276 under
both paddings.

The reason is structural, and it is the same defect class as Mission 2's
modifier: the number of slots depends on `u` and on the mapping, never on the
candidate `v`. So the padding term adds the **same constant to every candidate
of a node** and the argmax is arithmetic-invariant. The estimator is left as it
is (the padding still makes scores comparable across nodes), but it is now known
to be incapable of "swamping" anything, and the mission's hypothesis is
falsified rather than left standing.

What the same run *did* measure is the thing that matters: **86.3 % of
neighbour slots are uncheckable** under the top candidate. Geometry averages
about **one** usable edge per candidate, while presence is scored on the edges
that actually exist. That is a coverage fact, not a padding fact.

---

## Stage 0c — how shared are the anchors?

One DEV-3 scene, 639 nodes:

| quantity | value |
|---|---|
| distinct positions | 354 |
| share of nodes sharing an exact position | **0.446** |
| most nodes at a single site | **16** |
| share of nodes on a site shared with a DIFFERENT lens | **0.664** |
| **share of OBJECT-ANCHORED (ambiguous population) nodes on a shared site** | **0.894** |

Per lens, share on a shared site: `TextureBoundary` 1.00, `ContourFragment`
0.99, `BorderOwnership` 0.72, `Texture` 0.84, `Orientation` 0.80,
`CONTOUR_PEAK` 0.54, `Boundary` 0.55, `Position` 0.44, `EndStopped` 0.00.

Nodes are not anchored to the (lattice) tile centre in the sense of a square
grid; they are **allocated per site and many lenses land on the same site**. Two
nodes at one site with different lenses are, for placement purposes, the *same
point*, so the two placement vectors built from them are identical and carry no
extra information. With 89.4 % of the ambiguous population in that condition,
the placement payload is largely degenerate before any noise is considered.

Sub-pixel does not change this (354 distinct either way): the multiplicity is in
the anchor **allocation**, not in the position arithmetic.

---

## Stage 1 — the precision instrument, before any lever

Renderer truth only (used to score, never to match). 20 FIT scenes, 3 view steps.

| view step | repeatability | loc median | loc p90 | **loc (identity-correct)** | p90 | identity | distinct |
|---|---|---|---|---|---|---|---|
| `w1_rot` | 0.479 | 3.380 px | 6.774 | **1.905 px** | 4.108 | 0.247 | 0.554 |
| `w2_translate` | 0.572 | 2.020 px | 5.908 | **1.136 px** | 3.052 | 0.231 | 0.559 |
| `w3_scale` | 0.541 | 2.522 px | 5.301 | **1.724 px** | 3.565 | 0.270 | 0.560 |

The three errors the mission said were being conflated are now separate:

* **LOCALISATION** — given the right node, **1.1–1.9 px** median, 3.0–4.1 px
  p90. This is the detector's own precision. The old "3.6 px" figure was 2.0–3.4
  px because it also absorbed the next error.
* **IDENTITY / wrong node for the feature** — looking at the nearest node to a
  feature's true position, it is **the same feature only 23–27 % of the time**.
  This is the largest single term, and it is a consequence of the 89 % shared
  sites above.
* **REPEATABILITY / missing nodes** — only **48–57 %** of tracked features have
  a node in the other view.

Per-lens localisation (median px, unrestricted): `Polarity` 1.33,
`CONTOUR_PEAK` 1.87, `RegionShape` 1.89, `BorderOwnership` 1.95,
`PEAK_CHAIN` 1.99, `TextureBoundary` 2.49, `Surface` 2.56, `Corner` 2.78,
`ContourFragment` 3.14, `Texture` 3.20, `Color` 3.21, `CurveComplex` 3.33,
`EndStopped` 3.37, `Contour` 3.39, `Position` 3.46, `Orientation` 3.48,
`Boundary` 3.58, `T` 3.68, `Junction` 4.34.

`Junction`, `Corner`, `T` — the content-poor but position-rich types lever 2
wants — are the *worst* localised and the rarest (n = 5, 20, 8).

---

## The gate, scored

| criterion | target | measured | |
|---|---|---|---|
| localisation median | ≤ 0.5 px | 1.14–1.91 px | **FAIL** |
| localisation p90 | ≤ 1.5 px | 3.05–4.11 px | **FAIL** |
| distinctness | ≥ 95 % | 55.4–56.0 % (all nodes); 10.6 % of object-anchored | **FAIL** |
| repeatability | ≥ 70 % | 47.9–57.2 % | **FAIL** |
| direction | ≤ 3° median | not re-measured this mission (Mission 2: 2.12° on coherent nodes) | — |
| separation | AUC ≥ 0.95 @5°, ≥ 0.85 @15° | Mission 3: 0.64–0.82 | **FAIL** |

**No lever was applied and no re-run of the placement test was done**, because
the gate exists precisely to prevent that: re-running the arms on graphs that
fail every axis would have produced another null and spent CHECK looks on it.

---

## The measured floor, and where the remaining error is

Every lever, and the error source it would have to move:

1. **Sub-pixel for every node type (lever 1)** — tested, no effect
   (2.869 → 2.869 px). Error source it does *not* touch: anchor multiplicity.
2. **Content anchors instead of shared sites (lever 2)** — the largest term.
   89.4 % of placement-population nodes sit on a shared site and identity is
   23–27 %; this is where the remaining error overwhelmingly is.
3. **Keypoint detector feeding the machinery (lever 3)** — not attempted. Would
   address the same term as (2) and is the next thing to build.
4. **Direction refinement (lever 4)** — not attempted; Mission 2's rejected sign
   cue says the axial field is already the best-behaved provider.
5. **Per-detection uncertainty (lever 5)** — not attempted; cannot help while
   the dominant error is *which* node, not *where* it is.
6. **Repeatability filtering (lever 6)** — would trade coverage for precision
   and the coverage constraint (≥ 0.65) is already tight at 0.648.
7. **Anything else** — the 86.3 % uncheckable-slot share is the *symptom* of
   (2): with many nodes at one site, the counterpart edge usually does not exist
   because the counterpart node is a different lens at the same point.

**The single largest remaining source is anchor multiplicity — many nodes of
different lenses anchored to one site — which shows up simultaneously as low
distinctness (55 %), low identity (25 %) and high uncheckable-slot share
(86 %).** It is one cause wearing three masks, and it is a front-end allocation
problem, not a numeric-precision problem.

---

## What is NOT claimed

* **No DEV-4 placement run and no EVAL-3 read.** The gate was not met, so by the
  mission's own rule the placement re-run does not happen. No arm table from
  this mission is claimed, and the Mission-3 null stands as the state of the
  evidence.
* Any improvement from any lever. Nothing was applied, so nothing improved.

Suites: `Topology/test_placement.py` 17 OK, `tests/test_section_b.py` 17 OK,
`Topology/test_placement_m3.py` 10 OK. Defaults bit-identical
(`sha256 03fa5d1a3de822d2`).
