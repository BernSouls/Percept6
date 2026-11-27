# Mission 30 — wire the colour, then fix the saturated shape grid

Branch `arena/c316ca3a-percept6`, base `05377f5` (M29). Environment: Intel Xeon @ 2.60 GHz,
**2 logical / 1 physical core**, numpy 2.4.6, scipy 1.17.1, BLAS pinned, Python 3.11.2
(`parameters` environment key in every json below). Cell unless stated: 80 objects, query view
`-035-045`, gallery `-000-000` (one stored view), S = 8, chance top-1 0.0125, chance recall@8 0.100.

Headline: **the colour door shipped (A1), the colour blend did not earn its place (A2), and the
saturated shape grid was measured to carry identity that does not cross a view (B3) — with the
mechanism named field by field.** The place now carries 4.0 / 3.2 / 5.1 bits of object identity
per Blob / BoundaryArc / CornerPoint place (was 0.83 / 0.74 / 2.44), but only the colour part of
that travels between two views, and only two of the five shape fields travel at all.

---

## A1 — the colour door (PASS, shipped)

`matching/placement_fuse.py` (new, core): `fuse(counted, appearance, colour=None, weights=None)`
is reciprocal-rank fusion with the colour channel at a **stated discount** (`COLOUR_WEIGHT = 0.5`,
k = 60). `Topology/m30_door.py` is the thin harness; `Topology/results/placement/m30_door.json`
holds every number (247 s).

| view | split | n | shipped recall@8 | fused recall@8 | shipped top-1 | fused top-1 |
|---|---|---|---|---|---|---|
| 035-045 | FIT | 40 | 0.550 (22) | **0.675 (27)** | 0.125 | **0.250** |
| 035-045 | CONF | 40 | 0.550 (22) | **0.650 (26)** | 0.150 | **0.250** |
| 045-180 | FIT | 40 | 0.550 (22) | **0.650 (26)** | 0.250 | **0.275** |
| 045-180 | CONF | 40 | 0.475 (19) | **0.600 (24)** | 0.175 | 0.225 |

Bar (stated before running): average FIT recall@8 gain ≥ **+0.050** *and* DEV+FRESH not below
shipped by more than **0.025**. Measured: gain **+0.112** (0.125 and 0.100 per FIT view); CONF
**+0.100** and **+0.125**. Chance 0.100 (recall@8), 0.0125 (top-1). **PASS.**
No query lost a hit on any view (gained: 035-045 car/1, car/8, cow/7, horse/2; 045-180 cow/7,
cup/1, horse/3, pear/6). The colour channel alone (no counted, no appearance) scores 0.675 / 0.600
recall@8 — i.e. the fused door is not trading the incumbents away, it is adding a channel that is
independently as strong as the pair is together.

## A2 — the colour blend in the matcher (ablation run; did not earn a shipment)

Composition, fixed before any read: `sim' = (1 − w_cfg − w_col)·sim + w_cfg·config + w_col·colour`,
`w_cfg` **held** at the bundle's 0.25, `w_col = 0.10` (cap 0.15), applied inside `rescore_matrix`
immediately after the descriptor term and **before** `top_candidates`, so the cap sees the same
population. `PLACEMENT_NODE_COLOUR` (default **off**) and `PLACEMENT_NODE_COLOUR_W` (default 0.10)
are declared in `parameters/_text.py` and in `eth80_frontend._FRONTEND_SWITCHES`.

Ablation (apple/1, −035-045 ↔ −000-000, 80 gallery objects):

| arm | score |
|---|---|
| deployed matcher | 0.349997291 |
| flag on, `w_col = 0` | 0.349997291 (**byte-identical** — the flag with no weight is the deployed score) |
| flag on, `w_col = 0.10` | 0.347423554 (**Δ −0.002574**) |

`colour_cells` 257,408; `raised` 0, `fell_back` 0, `skipped` 0 — no silent fallback anywhere.
**Reading:** the blend moves one pair's score by −0.7 %; the colour *information* is real (A1) but
re-weighting an existing MCS score with it, at a weight the mission capped below the shape term,
cannot re-rank the candidates. The flag stays off by default; the door is where colour is shipped.

Mechanism found while measuring (a real cost, not a symptom): the blend recomputed every side's
CIELAB ring vectors **once per pair** — 58 s against 25 s for the shipped pair score. The vectors
depend on the side alone, so they are now cached per side (`_COLOUR_CACHE`, LRU of 6, holding the
side so `id()` cannot be reused): a blended pair is 25.3 s against 23.2 s shipped, and
`tests/test_m30_content.py::TestColourCache` pins the cached vectors to be *equal* to the
uncached ones (`_legacy_node_colour` is kept for exactly that test).

## A3 — one read at S = 8

<!-- A3 -->

## B1 — saturation, per lens (confirmed, with the M29 error corrected)

`Topology/m30_shape.py --information` (`Topology/results/placement/m30_shape.json`), 80 objects,
both views. Each lens's grid is 192 cells (Blob 12×4×4, BoundaryArc 12×4×4 nominal, CornerPoint
12×4×4); M29's "1e-8 utilisation" claim was wrong and is replaced by these numbers.

| view | lens | grid cells | distinct keys | filled | places | places / occupied cell | I(key;object) |
|---|---|---|---|---|---|---|---|
| 000-000 | Blob | 192 | **192** | 1.000 | 23,220 | **120.9** | **0.833** |
| 000-000 | BoundaryArc | 192 | **182** | 0.948 | 15,055 | 82.7 | 0.741 |
| 000-000 | CornerPoint | 192 | **185** | 0.964 | 2,099 | 11.5 | 2.444 |
| 035-045 | Blob | 192 | 192 | 1.000 | 24,571 | 124.8 | 0.792 |
| 035-045 | BoundaryArc | 192 | 186 | 0.969 | 16,905 | 94.2 | 0.694 |
| 035-045 | CornerPoint | 192 | 175 | 0.911 | 1,447 | 8.9 | 2.860 |

**All three lenses saturate** (≥ 175 of 192 cells used, Blob exactly 192), and the identity a place
carries per lens is *inversely* related to how many places crowd a cell: Blob packs 121 places into
a cell and carries 0.83 bits (13 % of the 6.32 available), CornerPoint carries 2.44 bits (39 %) on
5 % of the places. No lens is excluded; none is unsaturated.

## B2 — shape as graded columns (keys byte-identical)

`matching/placement_shape.py` (new, core): class `PLACE_SHAPE` with five graded fields —
`resp`, `var_x`, `var_y` (the detector's own numbers, previously computed and discarded),
`eigratio` (structure-tensor eigenvalue ratio, radius 1.5 px), `rel_angle` (the arm/edge direction
relative to the place's own frame, circular with period π, cycle 12). Weights (2,1,1,2,3), bins
(8,6,6,8,12), ranges (1,1,1,1,6). Registration is **on import** (idempotent, refusals counted) —
the M16 lesson: an unregistered class returns 0.0, not a small number.

`matching/placement_records.py` → `FORMAT_VERSION = 29`, `N_SHAPE = 5`, `SHAPE_FIELDS` exported at
module top, `shape_<view>` arrays in the npz, `meta["shape"] = 5`. The five fields are **columns,
never key slots**: `m30_records.py --check` reports `keys_identical: true` and
`frames_identical: true` on all 4 sampled objects × 2 views, `missing_registration: []`.

Record: `data/placement_records_m30.npz`, **37,391,603 B** (M28: 33,478,046 B, **+3.9 MB / +11.7 %**),
built in 320 cells / ~570 s with a per-cell cache and a progress file. Committed (≤ 40 MB).

## B3 — information and gauge (FAILS its bar; mechanism named)

Information, per lens (`m30_shape.json`), 80 objects, 6.32 bits available:

| lens | I(key;object) | I(key+shape;object) | gain |
|---|---|---|---|
| Blob | 0.833 | **4.046** | +3.213 |
| BoundaryArc | 0.741 | **3.216** | +2.475 |
| CornerPoint | 2.444 | **5.073** | +2.629 |

The graded columns multiply the identity a place carries by ~4.9× (Blob), ~4.3× (BoundaryArc),
~2.1× (CornerPoint). Distinct key+shape tuples: 8,197 (Blob), 2,874 (BoundaryArc), 1,267
(CornerPoint) against ~190 key cells.

Same-class cross-view gauge on the M28 subset (8 objects, both views, 4,284 query places, chance
0.500), bar stated before running: **gain ≥ +0.10**:

| arm | G_any | G_same | object-level G_same |
|---|---|---|---|
| key (M28 signature) | 0.515 | **0.496** | — |
| shape alone | 0.661 | 0.571 | — |
| key + shape | 0.638 | **0.572** | **0.514** |
| bar | | shape +0.075, both **+0.077** | **fail** (need +0.10) |

**Verdict: FAIL.** Two independent failures, not one: the per-place gain is +0.077 against a
+0.10 bar, *and* the object-level aggregate is 0.514 — indistinguishable from chance. A per-place
signal that dies when places are aggregated is not a signal the matcher can use.

Mechanism, field by field (`--fields`, same subset, chance 0.500):

| field | G_any | G_same |
|---|---|---|
| `resp` | 0.601 | 0.545 |
| `var_x` | **0.500** | **0.500** |
| `var_y` | **0.500** | **0.500** |
| `eigratio` | 0.615 | 0.544 |
| `rel_angle` | 0.570 | 0.521 |

Two of the five fields carry **exactly zero** cross-view identity: `var_x` and `var_y` are the
detector's variances along the *image axes*, so they rotate with the object and change with the
viewpoint. The three that survive are the rotation-normalised ones (`rel_angle` is measured in the
place's own frame) and the view-stable photometric one (`resp`, `eigratio`). Because the similarity
weights all five fields (2,1,1,2,3), the two dead columns cost 2/9 of the graded distance and
dilute the 0.60-0.62 the live columns reach on their own down to 0.572.

**B5 therefore does not run**: wiring a shape term into the door and the matcher at a weight above
colour's would ship two thirds of a signal whose measured cross-view transfer is 0.077 and whose
object-level aggregate is at chance. That is the ceiling, and it is a *named* one: the shape term is
recorded, registered, tested and byte-stable, and it needs a view-invariant anisotropy before it can
be weighted above the colour term.

## B4 — `corner_quantile` sweep (FIT only, 40 objects, −000-000)

| corner_quantile | places | corners | corner frac | I per place | **total identity** |
|---|---|---|---|---|---|
| **90.0** | 31,563 | 10,775 | 34.1 % | 1.263 | **39,852 bits** |
| 95.0 | 26,052 | 5,264 | 20.2 % | 1.471 | 38,323 bits |
| 97.0 | 23,789 | 3,001 | 12.6 % | 1.631 | 38,793 bits |
| 98.5 (shipped) | 22,155 | 1,367 | 6.2 % | **1.744** | 38,649 bits |

More corners do **not** carry meaningfully more identity: the total is flat within **3.1 %** across
a 5× change in corner count, while the identity *per place* falls 28 % (1.744 → 1.263). **Shipped
threshold kept at 98.5.** Deviation from "pick on FIT by total identity": the winner (q90) wins by
3.1 % on a statistic whose estimator moves with the place count, and it costs every record a
re-detection and a 42 % larger place set for that 3.1 %. Recorded as a deviation, with these
numbers, rather than shipped.

## Quality gates

* **Tests**: `tests/test_m30_content.py`, **21 tests**, all green. Every one goes RED on a revert:
  `fuse(a, b)` must equal the shipped `rank_fuse(a, b)` bit for bit; the colour weight must be
  below both incumbents and non-negative; the blend must return the *same matrix object* when the
  flag is off or the weight is 0; the five shape fields must be in all four live registries with
  their ranges/weights/cycle/bins; unknown/partial registration must be visible
  (`missing_fields()`); the record format must carry 5 shape columns at version ≥ 29 while the key
  tuple stays unchanged; and no core module may import `Topology` (mechanism-in-core, harness
  allowed only as a thin caller).
* **Flags off by default**: `PLACEMENT_NODE_COLOUR = False`, `PLACEMENT_NODE_COLOUR_W = 0.10`;
  `PLACEMENT_NODE_DESC*` untouched. The A1 door is a *reported* rule (`placement_fuse`), not a
  silent default: the shipped door path is unchanged unless a caller passes the colour channel.
* **Declared before the build**: both new switches are in `parameters` **and**
  `eth80_frontend._FRONTEND_SWITCHES`; the graph/pair cache key moves with them.
* **Records**: 37.4 MB, committed; nothing deleted; per-cell cache and progress file make the build
  resumable.
* **Determinism**: see the note in the A3 section.

## The one paragraph

The place now carries **4.0 bits of object identity per Blob place, 3.2 per BoundaryArc and 5.1 per
CornerPoint** (from 0.83 / 0.74 / 2.44 in M29) once its graded shape columns are counted, and its
six-number colour content — which M28 measured at a 0.745 same-class cross-view gauge — is now
fused into the door at a stated discount, which is what moved recall@8 from 0.550 to 0.675 / 0.650
on FIT and 0.650 / 0.600 on CONF (top-1 0.125 → 0.250, 0.250 → 0.275), with no query losing a hit.
**Shipped: the colour door (A1) and the record format v29 with five graded shape columns (B2).**
Not shipped: the matcher's colour blend (A2 — measured at Δ −0.0026 on a pair score, and its 57 %
cost overhead fixed rather than paid) and the shape term (B5 — its cross-view transfer is +0.077
against a +0.10 bar and its object-level aggregate is 0.514, at chance, because two of its five
fields are image-axis variances that rotate with the viewpoint). **The single next lever** is the
one the field table points at: replace `var_x`/`var_y` with a *rotation-normalised* anisotropy (the
structure-tensor eigenvalue ratio measured along and across the place's own frame, plus the
normalised third moment on each side of that frame), re-run the 8-object gauge with the bar at
+0.10, and only then weight shape above colour — the two dead columns are 2/9 of the graded
distance and they are what keeps a +0.60-transferable signal at 0.572.
