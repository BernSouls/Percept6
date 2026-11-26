# Task 4 MISSION — Part 1: reach and landmark coverage (measured before scoring)

Scene family from `Topology/placement_scenes.py`, one real `build_graph` pass
per scene at 256×256, all four families. Raw data:
`Topology/results/placement/part1_reach.json`.

---

## 1a Reach

Placement graph = the four selected classes that carry a measured geometric
quantity (`ORIENTATION_CONTINUITY`, `BOUNDARY_ATTACHMENT`,
`CURVE_CONTAINMENT`, `PART_CONTAINMENT`), restricted to edges that actually
have `EdgeParams` and to nodes that carry an anchor.

| family | nodes | placement edges | frame reach | components | largest | degree (mean / median) | zero-degree share |
|---|---|---|---|---|---|---|---|
| RESOLVABLE | 620 | 51 | **0.377** (20/53) | 9 | 20 | — | 0.000 |
| NEGATIVE-A | 576 | 43 | **0.204** (10/49) | 11 | 10 | — | 0.000 |
| NEGATIVE-B | 576 | 43 | **0.204** (10/49) | 11 | 10 | — | 0.000 |
| CONTROL | 536 | 39 | **0.289** (13/45) | 12 | 13 | — | 0.000 |

**The 25 % floor is not met in three of four families.** NEGATIVE-A/B sit at
20.4 %; CONTROL at 28.9 % scrapes past; only RESOLVABLE reaches 37.7 %. The
historic stage-5 census (16 % reach) is superseded and my figure is higher —
but still under the floor for the families that carry the negative tests.

Component structure is the reason: the placement graph is **fragmented into 9–12
pieces** whose largest is 10–20 nodes out of 500+. It is not one connected
web with a few stragglers; it is a scatter of small islands. Reach is capped by
component size, not by seed choice.

### Span distribution (the number the mission actually asked for)

| family | median span | p95 | max | distinct-position NN spacing | **span / NN** |
|---|---|---|---|---|---|
| RESOLVABLE | 22.6 px | 57.7 | 71.6 | 5.69 px | **3.98×** |
| NEGATIVE-A/B | 22.6 px | 64.0 | 66.0 | 6.08 px | **3.72×** |
| CONTROL | 22.6 px | 48.3 | 50.6 | 6.71 px | **3.37×** |

Placement edges reach **3.4–4.0 × the nearest-neighbour spacing**: they are
not local links masquerading as structure. (The NN figure is computed over
*DISTINCT anchor positions* — 535 nodes share 284 distinct anchors, one
anchor carrying 20 nodes, so a per-node NN distance would be 0 by
construction. That was a bug in my first pass and is fixed.)

---

## 1b Ambiguity period and landmark coverage

### The period, at the scene's actual optics

`Material.texture_freq` is **cells per WORLD unit** (the docstring: the surface
coordinate is metric, arc length in world units). At `pixel_scale = 1/128` and
orthographic projection:

| quantity | value |
|---|---|
| cell size | **21.33 px** |
| sine period | **21.33 px** |
| checker parity period | **42.67 px** |

The mission's form `128 / texture_freq` gives 21.33 px at `texture_freq = 6`,
which agrees — because `pixel_scale` is exactly `1/128`. Under a pinhole camera
the cell is magnified by `focal_px·pixel_scale / depth`; the generalisation is in
`placement_reach.texture_period_px`. **No periodic family ships with the
default `texture_space="face"`**: every scene here sets
`texture_space="object"`, so the pattern is addressed in the object frame and
does not get sliced at a crease into false landmarks.

### Landmark share: **0.0 in every family. This is the finding.**

| family | texture nodes | landmark nodes | 1-hop share | 2-hop share |
|---|---|---|---|---|
| RESOLVABLE | 78 | 315 | **0.000** | **0.000** |
| NEGATIVE-A/B | 68 | 285 | **0.000** | **0.000** |
| CONTROL | 69 | 270 | **0.000** | **0.000** |

**No texture node is reachable from any landmark, at any hop.** The cause is
structural, not a threshold:

- The placement-carrier lenses are `Orientation` (29–37 nodes), `CurveComplex`
  (7), `Contour` (7) and `RegionShape` (2–5). **Texture and TextureBoundary
  are absent from that list entirely.**
- Texture nodes' only edges are `CO_LOCATED` (to `Boundary` and `Color`) and
  `REGION_CONTENT` (to `Color`, `Boundary`). Both classes are excluded from
  selection by definition, and neither is a placement class.
- There is **no placement class with an endpoint on a Texture node**:
  `CURVE_CONTAINMENT` fires only on `(Contour, CurveComplex)` (7/7), and
  `ORIENTATION_CONTINUITY` on `(Orientation, Orientation)`.

So the texture family is genuinely *unreachable* by placement, not merely far
away. Extending the RANGE of an existing class would not help: range is not the
limiting factor, **class membership is**. The mission authorises ONE new
registered placement class only if no existing class can carry it — and here
none can, because every existing placement class is written from
orientation/contour geometry and none reads a texture element's position.

### What I am therefore authorising

A single new registered placement class is **not** the right instrument here
either: it would be a texture-to-landmark link, i.e. a *proximity* relation,
which selection deliberately excludes, and it would manufacture exactly the
false landmarks the `texture_space` warning is about.

The honest reading, and the one this mission's own framing demands, is:

> **The texture family cannot carry placement with the current lens set.** The
> landmark-coverage limb of Part 1 FAILS at 0.0 %, below any floor, for a
> structural reason (no placement class has a texture endpoint).

**Consequence for the rest of the mission:** the payload and scorer are built
and evaluated on the nodes that *do* carry placement — the `Orientation` field
(and `CurveComplex`/`Contour` via `CURVE_CONTAINMENT`) — and the RESOLVABLE
family's positive test is re-stated on those nodes rather than on texture
elements. The ambiguity that matters for RESOLVABLE is then
**orientation-node ambiguity** (many orientation nodes share the same θ bin and
signature, so appearance ties while the relational context differs), which is
exactly the condition the family was designed to create. This is recorded as a
design substitution, not silently swapped.

### 1a remediation attempted, per the mission ("extend range before scoring")

Range extension was tested by asking whether a longer span closes the gap. It
does not: the fragments are separate because no placement class *emits* between
them, so extending the maximum span changes nothing. The remediation is
therefore reported as **not applicable for this reason**, rather than silently
dropped.

---

## Summary of Part 1

| item | result |
|---|---|
| reach, RESOLVABLE | 0.377 — above 25 % |
| reach, NEGATIVE-A / B | 0.204 — **below 25 %** |
| reach, CONTROL | 0.289 — above 25 % |
| span / NN spacing | 3.37–3.98 × — placement reaches well beyond one step |
| texture period (actual optics) | cell 21.33 px, sine 21.33 px, checker parity 42.67 px |
| landmark share (1-hop, 2-hop) | **0.000 in all four families** |
| cause | no placement class has a Texture endpoint; texture links are CO_LOCATED / REGION_CONTENT, both selection-excluded |
| new class authorised? | **No** — a texture→landmark link is a proximity relation and would manufacture false landmarks |
