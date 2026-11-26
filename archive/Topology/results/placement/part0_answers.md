# Task 4 MISSION — Part 0 answers (measured, cited)

Branch `arena/c316ca3a-percept6`. Every claim below is a code citation or a
measurement made in this session; nothing is inferred.

---

## 0a — Is the walker's admission test MAPPING-DEPENDENT? **NO.**

**Answer: NO.** The walker never composes a stored placement through the
candidate mapping, and never asks where the mapped node *sits*. Every
admission test it runs compares two stored parameter vectors, axis by axis,
in bins.

The admission path is `matching/relation_walker.py::admit` (line 1022), which
for each unmapped neighbour pair calls `note()` → `propose()`, gated by three
functions, all of which read only `_Half.bins`/`_Half.cycles` — the stored
quantized parameters:

| function | line | test |
|---|---|---|
| `_slack(a, b)` | 314 | `da = dict(a.bins)`; for each `ax, vb in b.bins`: `d = abs(va - vb)`; fail if `d > BIN_TOLERANCE` |
| `_agreement(a, b)` | 341 | `max(0.0, 1.0 - d / CONTRADICTION_MARGIN)`, same `a.bins` vs `b.bins` |
| `_contradicts(a, b)` | 362 | `d > CONTRADICTION_MARGIN` on any shared axis |

`_Half` (line 301) holds `bins: Tuple[Tuple[int, float], ...]` — *"the (axis,
bin) of every measured axis"*. There is no position field and no geometry.

`admit` uses `sim(hu.far, hv.far)` for a node-similarity floor and calls
`order_propose`, which is an **ordinal** consistency check on arc position
among siblings; `_order_consistent` (line 501) compares orderings, not
placement.

Grepping the entire file for positional machinery returns **nothing**:

```
$ grep -n "anchor_of_node|world|\.pos\b|pixel|placement|expected_position" \
      matching/relation_walker.py
(no matches)
```

Constants: `BIN_TOLERANCE = 1.0` (line 67), `CONTRADICTION_MARGIN = 4.0`
(line 76). Agreement is therefore *parameter equality within a quantized
bin*, which the mission explicitly excludes: **that is not placement.**

**Host decision: `matching/alignment.py`.** The walker does not decide the
mapping (its flag `P.MATCH_RELATION_WALKER` is off by default and the shipped
path does not use it), so the scorer goes in `matching/alignment.py`, whose
`_relational_agreement` / `_assign` are the authorised change sites. **One
path only** — no scorer in the walker, no duplicate.

---

## 0b — Which selected classes carry placement, and their zero points

Selection excludes CONTACT, CO_LOCATED, PROXIMITY, SURFACE_GROUPING. The
surviving classes that carry geometry, with spans read from
`core/schema.py::EDGE_UNIT_TABLE` (the authority, not the prose):

| class (selected) | slot | unit | span (lo..hi) | zero point | centred? |
|---|---|---|---|---|---|
| ORIENTATION_CONTINUITY | `perp` | median_spacings | **0.0 .. 2.0** | **0.0** | **NO** |
| ORIENTATION_CONTINUITY | `dist` | median_spacings | 0.0 .. 4.0 | 0.0 | NO |
| ORIENTATION_CONTINUITY | `cocirc_asym` | radians | 0.0 .. π/2 | 0.0 = co-circular | NO |
| ORIENTATION_CONTINUITY | `turn` | radians | −π/2 .. +π/2 | 0.0 | YES (circular) |
| BOUNDARY_ATTACHMENT | `perp_dist` | wavelengths | **0.0 .. 1.0** | **0.0** | **NO** |
| BOUNDARY_ATTACHMENT | `tangent_angle` | radians | **0.0 .. π/2** | **0.0** (parallel) | **NO** |
| BOUNDARY_ATTACHMENT | `arc_end_dist` | fraction | 0.0 .. 0.5 | 0.0 | NO |
| CURVE_CONTAINMENT | `polar_angle` | radians | **0.0 .. 2π**, 24 bins | **0.0 = +x of the IMAGE frame** | **NO (not centred)** |
| CURVE_CONTAINMENT | `rel_dist` | wavelengths | 0.0 .. 5.0 | 0.0 | NO |
| CURVE_CONTAINMENT | `scale_ratio` | log2_ratio | −2.0 .. +2.0 | **0.0 = equal size** | YES |
| PART_CONTAINMENT | (same three as CURVE_CONTAINMENT) | | | | |

### The three specific questions asked

**`ORIENTATION_CONTINUITY.perp` (0..2): is 1 the zero? — NO.** The span is
declared `_eslot("perp", "median_spacings", 0.0, 2.0, 8)` and the quantiser is
`floor` (not `signed_centered`, not `signed_log`). A floor quantiser over
`lo=0` places bin 0 at 0.0 and bin 7 at 1.75; the *de-binned* centre is
`lo + (b + 0.5)/n * span`. **The zero point is 0.0, not 1.0.** 1.0 is the
midpoint of the range, which is not a meaningful origin: `perp` is an
unsigned perpendicular offset. It is **not** centred, and no sign is
recoverable from it.

**`BOUNDARY_ATTACHMENT.perp_dist` (0..1): zero point 0.0, unsigned.** Written
by `perception/attachment.py::_measure` as `perp_dist=d / scale`, where
`d` is `poly.project(f.pos)`'s distance to the foot of the perpendicular — a
**norm**, hence non-negative with true zero at 0 (the feature lies on the
host). Span `0.0..1.0` wavelengths. Not centred.

**`BOUNDARY_ATTACHMENT.tangent_angle` (0..π/2): zero point 0.0 = parallel.**
`_measure` computes `axial_difference(f.tangent, host_angle)` — an *axial*
difference, folded to [0, π/2]. Zero means the feature's tangent is parallel
to the host's. Unsigned by construction: the fold discards which side.

**`CURVE/PART_CONTAINMENT.polar_angle` (0..2π, 24 bins): the frame is the
IMAGE frame, not the container's own frame.** From
`orchestration/binder.py` (the `_run_part_containment` writer):

```python
angle = math.atan2(p_pos[1] - c_pos[1], p_pos[0] - c_pos[0]) % (2.0 * math.pi)
```

`p_pos`/`c_pos` are image-plane positions and `atan2` is taken in image
coordinates. The slot is **not** measured relative to the container's
orientation, so it is not a frame-relative quantity: two views that rotate
the scene by φ produce `polar_angle + φ`, and the slot cannot tell that
change from a change of relation. **This is the one selected slot whose frame
is wrong for this mission**, and it is why the payload below is built from
`ORIENTATION_CONTINUITY` and not from containment.

`polar_angle` is also `circular=True, quantiser="round"`, so 24 bins over 2π
is 15° of angular resolution — too coarse to be a placement measurement.

---

## 0c — Share of selected edges whose endpoints are outside CONTENT_TOKEN_LENSES

`CONTENT_TOKEN_LENSES` (`perception/front_end_mechanisms.py:378`) =
{CONTOUR_PEAK, ContourFragment, PEAK_CHAIN, EndStopped, Corner, Junction, T,
Y, RegionShape, IllusoryContour}.

Measured on a rendered sphere (256×256, 486 nodes, 1269 edges, one real
`build_graph` pass), counting edges with **both** endpoints on a content lens:

| class | all | both content | share |
|---|---|---|---|
| ORIENTATION_CONTINUITY | 46 | **0** | **0.0 %** |
| CURVE_CONTAINMENT | 14 | 0 | 0.0 % |
| PART_CONTAINMENT | 4 | 0 | 0.0 % |
| CONTOUR_ORIENTATION | 102 | 39 | 38.2 % |
| TEXTURE_BOUNDARY_EDGE | 267 | 3 | 1.1 % |

**The share of selected placement edges with both endpoints outside the
content tokens is 100 % for every class that carries usable placement.**
`ORIENTATION_CONTINUITY`'s endpoints are `Orientation` ↔ `Orientation` on
46/46 edges: a FRAME lens, not a content token.

Restricting to content kinds would therefore delete *every* placement edge.
The honest resolution is the one the mission's own framing supports: the
**content-token restriction is the wrong instrument here**, because the
placement carriers are the orientation field itself, and `Orientation` nodes
do have what a placement needs — a continuous θ (slot 0, 24 bins over 2π,
`parameters/_text.py:299 ORIENTATION_N_THETA = 24`, observed bins 0..23) and a
full anchor (`anchor_of_node` covers 486/486 nodes). I therefore keep
`Orientation` endpoints and record the restriction as **not applicable**,
with the 0.0 % figure reported rather than silently ignored.

---

## 0d — Selection: ties and the 512 budget

**Two id deciders existed. Both are replaced.**

*Decider 1 — the binder's channel cap*
(`orchestration/binder.py::_channel_process`, line ~4925`):

```python
candidates = nlargest(cap, candidates, key=lambda c: (c[2], _neg_bytes(c[3])))
```

`c[3]` is the channel's "per-channel-unique canonical tuple for M_prev
tracking", and it contains the two node ids. So when two candidates carried
equal signal, the surviving edge was decided by **node id, i.e. build
order** — exactly what the mission forbids.

*Decider 2 — the perceptual selection sort*
(`matching/perceptual_selection.py::edge_key`):

```python
return (-float(graph.weights.get(edge, 1.0)), et.name, a, b, edge[0], edge[1])
```

`node_key(n)` returns `(lens type, signature bytes, n)` — the id is the
**third** component, so *within the same lens type and same signature* the
tie again fell to the node id.

**Replacement (both sites):** the tie is broken by the endpoints' **content
identity** — `(lens type, signature bytes)` for each endpoint, sorted so the
key does not depend on edge direction. Node ids survive only as the final
total-order fallback for content-identical pairs, which cannot occur for
distinct endpoints in practice. In the binder the content key is recovered by
inverting `ConceptGraph._pattern_node_id_by_key` (the `{(lens, sig_hash):
nid}` table each node was minted from), cached per node.

**Measurement (`EDGE_BUDGET_PER_FRAME = 512`), one rendered sphere 256×256:**

| quantity | value |
|---|---|
| total edges emitted | 1269 |
| structural edges (eligible) | 1109 |
| selected | **512** (the budget binds exactly) |
| survive, share of structural | **46.2 %** |
| ORIENTATION_CONTINUITY | 46 emitted → **34 selected** |
| other structural classes dropped | ≥20 each |

Because only 46.2 % of eligible edges survive, the tie decider is not a corner
case: it chooses more than half of what the matcher ever sees.

## 0e — Determinism proven

`frozenset` iteration is process-salted and this branch has already been
burnt by it once (the W-rewiring in the retrieval work). Every structure this
mission builds carries a canonical sort key.

**Proof (fresh-process rerun, bit-identical):** a two-object scene (sphere +
box) rendered and built three times in three separate processes gives

```
nodes 595
all_edges 1fe935e808a9c3e2
selected  8ae201d341ef1a9d
```

three times over — identical node count, identical full edge set, identical
selected set.

New flags entering the front-end cache key: the placement flags are added to
`parameters` and the front-end digest (`eth80_frontend.py::switch_suffix`),
so a cache built under one setting can never be read back under another. To
be confirmed when the flags land (Part 2/3/4).
