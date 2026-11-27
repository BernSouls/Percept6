# M29 — THE REPRESENTATION: WHAT THE NODES AND EDGES ARE, AND HOW MUCH THEY CARRY

A read-only judgement. Every number is measured by `Topology/m29_representation.py`
(`m29_representation.json`) on this checkout; the machine is in every manifest (`environment`).
Chance and n are given beside every rate, as always.

---

## 1. What a node is

A node is a `TypedSignature` (`core/types.py:355`):

```python
kind: str                 # "CANONICAL" | "OPAQUE"
class_: str               # the lens's class name, e.g. "Blob", "CONTOUR_PEAK", "COLOR"
relation: str             # "" for a node
quantized: Tuple[float, ...]      # the QUANTISED reading: one bin per declared slot
population: Tuple[Tuple[float, ...], ...]   # optional graded activity vectors
```

* `quantized` is **the node's identity**: it is what the hash is computed over
  (`_typed_signature_bytes`), so two detections with the same bins are the same node type.  The
  number of slots is declared per class in `core/registries.CONSTITUTIONAL_RANGES` and the bin
  count is `2 × range` (the N-half convention; circular slots declare a cycle length in
  `CIRCULAR_CYCLE_LENGTHS` and wrap).
* `population` is **deliberately excluded from the hash and from byte serialisation** (its own
  comment says so): the bins are the coarse *index*, the population is the *graded truth* the
  similarity reads when the class has registered axes (`core/registries.POPULATION_AXES`).
* **Position is not part of a signature.**  A detection's place in the image rides on the node
  (`Subgraph.anchor_of_node`), never in the signature.  That is the design's central bet: matching
  must not be about where things are, so the representation must be invariant to it.
* `similarity` (`core/types.py:408`) is a per-slot Gaussian over the constitutional range
  (`_similarity_gaussian`, σ from the range), with three overrides: population similarity when both
  sides carry payload, a slot-skipping path when a slot is `ABSENT`, and a class/relation gate that
  returns 0.0 across classes.  **ABSENT is not zero** — an unmeasured slot contributes neither
  distance nor weight.  That distinction is real engineering and it is respected everywhere I read.

## 2. What an edge is

An edge is a triple `(u, v, EdgeType)` plus an `EdgeParams` (`core/schema.py`):

```python
class_     # "EDGE_CO_LOCATED", "EDGE_BORDER_OWNERSHIP", ...
quantized  # one bin per declared parameter slot (1-7 slots)
sign       # +1 / -1
confidence # the writer's own confidence
population # optional graded axes for the edge class
```

There are **36 edge classes** with constitutional ranges, from 1 slot (`EDGE_REGION_CONTENT`,
`EDGE_CONTOUR_ORIENTATION`) to 7 (`EDGE_ORIENTATION_CONTINUITY`), with 14 of them declaring circular
slots at 8/12/24 bins.  An edge class says *what relation between two nodes is being asserted*
(contact, co-location, border ownership, collinearity, concentricity, occlusion, ...) and its slots
say *how much of it*.

## 3. The declared vocabulary (what the system can say)

| | node classes | edge classes |
|---|---|---|
| count | **67** | **36** |
| slots per class | mean 16.9 | mean 3.1 |
| nominal bits per class | mean **34.9**, max 88.6 | mean **12.3**, max 24.5 |
| total nominal bits over classes | 2,338 | 444 |
| with graded population axes | **31 of 67** | registered per class, used where declared |
| lens tiers that produce nodes | **13** (`Peak, Chain, BorderOwnership, Surface, Contour, Junction, Corner, EndStopped, RegionShape, Polarity, TextureBoundary, IllusoryContour, CurveComplex`) | — |

"Nominal bits" is `Σ log2(bins)` over a class's slots: the size of the space the class *could* fill.

## 4. What the front end actually emits (6-image census, view `-000-000`)

| | per image |
|---|---|
| nodes | **1,563** |
| edges | **2,731** (1.75 per node) |
| node classes used | **24 of 67** |
| edge types used | **28 of 36** |
| node degree | mean **3.49**, median **0.0**, max 77, **53.4 % isolated** |
| nodes with a population payload | **69.2 %** |

The node mix is dominated by the three place lenses plus the boundary and peak tiers:
`Blob` 19.8 %, `BoundaryArc` 13.8 %, `CONTOUR_PEAK` 8.3 %, `Boundary` 8.0 %, `Color` 7.3 %,
`BorderOwnership` 5.4 %, `ContourFragment` 3.8 %, and a long tail.

**Half the graph is disconnected.** 53 % of nodes have no edge at all, so whatever identity they
carry is carried by the node alone.  The relational layer is not a graph over the image; it is a
sparse set of local assertions hanging off ~47 % of the nodes.

Per class, what the emitted signatures actually say (nodes per image, distinct signatures observed
across the 6 images, entropy of the signature distribution, slots actually measured, payload share):

| class | nodes/img | distinct sigs | **H(sig) bits** | slots measured | payload |
|---|---|---|---|---|---|
| Blob | 309.3 | 172 | **6.67** | 5.0 | 1.00 |
| BoundaryArc | 216.0 | 103 | **5.67** | 5.0 | 1.00 |
| CONTOUR_PEAK | 130.0 | 636 | **9.21** | 8.0 | 1.00 |
| Boundary | 125.5 | 162 | 6.42 | 3.0 | 0.00 |
| Color | 113.5 | 146 | 6.16 | 3.0 | 0.00 |
| BorderOwnership | 84.7 | 246 | **7.67** | 4.0 | 1.00 |
| ContourFragment | 58.7 | 321 | 8.28 | 4.0 | 1.00 |
| CornerPoint | 45.8 | 105 | 6.36 | 5.0 | 1.00 |
| EndStopped | 34.5 | 90 | 6.25 | 4.0 | 1.00 |
| Corner | 15.5 | 77 | 6.17 | 5.4 | 1.00 |
| Contour | 13.5 | 59 | 5.69 | 27.0 | 1.00 |
| CurveComplex | 8.5 | 40 | 5.10 | 44.0 | 1.00 |

A class's **achieved** entropy is 5–9 bits against a nominal 20–90: the emitted vocabulary uses a
tiny corner of its declared space (172 distinct Blob signatures used out of a nominal 2^34.9).  That
is what one expects from a quantised reading of a smooth world, and it is not a defect by itself —
but it means the *effective* alphabet is ~10^2, not 10^10.

## 5. How much information the nodes carry about identity

The question that matters for retrieval: given a place's key, how many bits do you learn about
*which object this is*?  Measured on the committed records (all 80 objects, n places and n keys
below), with `H(object) = log2 80 = 6.32` bits as the total available:

| view | places | distinct keys | H(key) | H(key &#124; object) | **I(key; object)** | share of 6.32 |
|---|---|---|---|---|---|---|
| `-000-000` | 40,399 | 559 | 7.71 | 6.81 | **0.90 bits** | **14.2 %** |
| `-035-045` | 43,020 | 553 | 7.65 | 6.81 | **0.84 bits** | 13.3 % |

Per lens (view `-000-000`), and this is the sharpest number in the report:

| lens | places | distinct keys | H(key) | **I(key; object)** | share of 6.32 |
|---|---|---|---|---|---|
| Blob | 23,220 | 192 | 6.76 | **0.83 bits** | 13 % |
| BoundaryArc | 15,055 | 182 | 6.04 | **0.74 bits** | 12 % |
| **CornerPoint** | 2,124 | 185 | 6.92 | **2.44 bits** | **39 %** |

The **rarest place class carries the most identity per place** — a corner's content says three times
as much about *which object* as a blob's or an arc's, and it keeps more of it on the query view
(2.86 bits at `-035-045` against 2.44 at `-000-000`, where the other two fall).  The pooled 0.90
bits is a weighted average of the three, and it is dominated by the two numerous classes; read the
other way, 88 % of a place key's entropy is variation that says nothing about *which* object it came
from — it is the local shape of a part, and parts recur across objects.

The pixel-side comparison, on the same 80 objects and the same view: a single coarse appearance key
(4×4×4 colour + 12-bin gradient, the door's own appearance channel) has **I = 6.32 bits**, i.e. it
saturates the task (**77 distinct keys for 80 objects**, I = 6.32 bits of 6.32).  So the node vocabulary, as a per-place
reading, delivers **14 % of what one global colour/gradient histogram of the same photograph
delivers** — the ceiling is not the image, it is the representation's choice of what to keep.

Two more facts that bound it:

* **A place key does not survive a view change.**  The M28 gauge measured the same-class cross-view
  identity AUC of exactly this key at **0.496** against chance 0.500 (n = 4,284 query places) — the
  0.90 bits above is *within-view* identity, and it does not transfer.
* The per-object distribution has ~170 distinct keys per object and H ≈ 6.83 bits *within* an
  object, so the object-level signature is a large, redundant set rather than a compact code; the
  door reads that set (bag-of-keys, count/sqrt norm) and gets recall@8 0.550 (44/80, chance 0.100).

## 6. How much information the edges carry

Per edge class, measured over the same 6 images (n ≈ 2,731 edges per image):

| edge class | edges/img | distinct param tuples | **H(params) bits** | measured slots | mean confidence |
|---|---|---|---|---|---|
| BORDER_OWNERSHIP_EDGE | 311.5 | 236 | **7.30** | 1.00 | 0.935 |
| CO_LOCATED | 252.3 | 7 | **2.14** | 1.00 | 0.995 |
| COLOR_OPPONENT_BORDER | 183.5 | 566 | **8.81** | 1.00 | 0.686 |
| CONTOUR_ORIENTATION | 124.2 | **1** | **0.00** | 1.00 | 1.000 |
| REGION_CONTENT | 121.0 | 2 | 0.00 | 1.00 | — |
| CONTOUR_CONTINUITY | 40.0 | 46 | 5.03 | 1.00 | 0.751 |
| COCIRCULAR | 20.8 | 33 | 4.80 | 1.00 | 0.803 |
| CONTACT | 2.8 | 16 | 3.97 | 1.00 | 0.997 |
| AMODAL_LINK | 8.2 | 18 | 3.88 | 1.00 | 1.000 |
| CONCENTRICITY | 3.3 | 3 | 1.37 | 1.00 | 1.000 |

Aggregate (measured, `Σ edges_per_image × H(params)` over the 28 emitted classes): one image's
relational layer carries **14,733 bits** of parameter variation over **2,731 edges — 5.40 bits per
edge on average**, against a per-class nominal mean of 12.3 bits.  The distribution is brutal:
**11.5 % of edges carry exactly zero bits** (classes whose tuple never varies) and **20.9 % carry
≤ 2.14 bits**.  The graph asserts far more structure than it measures, and the parts it measures
densely (`COLOR_OPPONENT_BORDER`, `BORDER_OWNERSHIP_EDGE`, the contour tier) are not the parts it
asserts most often.

**The busiest edge types are the emptiest.**  `CONTOUR_ORIENTATION` is 124 edges per image whose
parameter tuple is *the same tuple every time* — 4.5 % of all edges carrying exactly zero bits.
`CO_LOCATED` is 252 edges per image with 7 distinct tuples (2.14 bits), `REGION_CONTENT` 121 edges
with 2 tuples (0 bits).  Those three alone are **18.2 % of every edge in the graph** (497.5 of 2,731 per image) and together
carry less than one bit of variation per edge.  The informative classes are fewer and smaller:
`COLOR_OPPONENT_BORDER` (8.81 bits), `BORDER_OWNERSHIP_EDGE` (7.30), and the contour tier
(`CONTOUR_CONTINUITY` 5.03, `COCIRCULAR` 4.80) — which is exactly the tier the matcher's
`EDGE_ORIENTATION_CONTINUITY` schema was designed around.

**What the edges are worth to the score.**  The matcher's score splits into a node term and an edge
term; setting `MATCH_EDGE_WEIGHT = 0` and re-scoring the same pairs measures the split directly
(`m29_representation.json: edge_term`):

| pair (query `apple1-035-045`) | score | score with the edge term at 0 | edge share |
|---|---|---|---|
| **apple1** `-000-000` (the true instance) | 0.349997 | 0.292202 | **16.5 %** |
| apple10 | 0.313906 | 0.257898 | 17.8 % |
| **apple3** | **0.367655** | 0.316021 | 14.0 % |
| apple4 | 0.378389 | 0.325059 | 14.1 % |
| apple5 | 0.364666 | 0.301063 | 17.4 % |
| apple6 | 0.329947 | 0.279459 | 15.3 % |
| apple7 | 0.322934 | 0.274525 | 15.0 % |
| apple9 | 0.360668 | 0.307688 | 14.7 % |

n = 8 pairs (one query against all 8 stored apple instances), mean edge share **15.6 %**, range
14.0–17.8 %.  Note the two facts sitting side by side in that table: the true instance's share
(16.5 %) is mid-distribution, and **two wrong instances score higher than the true one** (apple3
0.3677, apple4 0.3784 against apple1 0.3500) — the same-class confusion M27 measured across the
whole read, visible in a single column here.

The relational layer contributes **15.6 % of the magnitude of a pair score on average, and it
contributes it without regard to whether the pair is right**: the true instance 16.5 %, wrong
instances 14.0–17.8 %, and the ranking the edge term does not fix is the one that matters (two wrong
apples outrank the true one *with the edges in*).  On this evidence the edge term is close to a
constant additive offset: it changes where scores sit, not which object wins.  M27's own loss column
is consistent with that — inside a correct shortlist the matcher loses by tiny margins (median gap
−0.0124), and a term with ~16 % of the magnitude and no discrimination between right and wrong pairs
is exactly where such margins come from.

## 7. The judgement

**What this representation is good at.**  It is *invariant by construction*, and deliberately so:
signatures carry no position, so a place's reading does not depend on where it landed; circular
slots are declared with their cycle length and wrap (the `EDGE_LINE_TERMINATION` audit note that
fixed a 24-cycle to a 12-cycle is exactly the kind of error that would silently destroy an encoding);
`ABSENT` is distinguished from zero everywhere, so "not measured" never reads as "measured as
nothing"; 31 classes carry graded population codes that the similarity reads *instead of* the bins,
so the quantisation is only the index; and class/relation gating makes cross-class comparisons zero
rather than small.  These are the properties most representations get wrong, and this one gets them
right.

**Where it loses.**  Four places, in descending order of how much they cost:

1. **What the node keeps is local shape at a quantised resolution, and identity is not local
   shape.**  A place key is 0.90 bits about *which object* out of 6.32 (14 %), and 0 bits of that
   survives a view change (AUC 0.496).  The representation is a good description of *a part of an
   image* and a poor carrier of *which object* — which is precisely why M28's six colour ring means,
   attached at the place, moved the same-class gauge from 0.496 to 0.745 without changing the graph
   at all.
2. **The busiest relations are constant.**  11.5 % of edges carry zero bits and 18.2 % are three
   classes with ≤ 2.14 bits between them; the edge term as a whole is a **15.6 %** additive offset
   that does not discriminate correct from incorrect pairs (14.0–17.8 % across eight pairs).  A relation that is true of everything tells you nothing; the
   representation asserts a great many of them.
3. **Half the graph is isolated** (53 %), so the relational layer cannot speak about half the nodes,
   and the correspondence engine is matching those nodes on content alone.
4. **Utilisation is ~10⁻⁸ of the declared space** (172 distinct Blob signatures used against a
   nominal 2^34.9; 5–9 achieved bits against 12–25 declared per class).  The vocabulary is far
   larger than the world it is asked to describe; the quantisation, not the space, is the limit.

**What I would change first, if asked.**  Not the geometry of any single lens: the measurement that
pays is *finding the source that carries identity across a view change and attaching it where the
representation already has a slot* — exactly the M28 result (colour at the place: +0.249 same-class
AUC, from 0.496 to 0.745) and exactly what the graph layer's own numbers say is missing (0.90 bits
of 6.32, none of it robust to the view).  Second: prune or fix the constant edge classes, because
they are 18 % of the edge budget and 0 % of its information; a relation that never varies should be
one bit ("present"), not three slots of parameters.

---

## 8. How to reproduce

```
.venv/bin/python Topology/m29_representation.py --registry --info     # vocabulary + information
.venv/bin/python Topology/m29_representation.py --graphs --edges      # what is emitted, and the split
```

`--info` reads the committed records (no rebuild, ~14 s); `--graphs` builds 6 graphs (~80 s, cached
after the first); `--edges` scores 8 pairs twice, once with `MATCH_EDGE_WEIGHT = 0` (~7 min).

*Files:* harness `Topology/m29_representation.py`; results `Topology/results/placement/
m29_representation.json`; the node/edge code is `core/types.py` (signature + similarity),
`core/schema.py` (edge classes, ranges, circular cycles), `core/registries.py` (slot ranges,
population axes), `core/population.py` (chance-corrected population similarity), and the place record
that the door reads is `matching/placement_records.py` + `data/placement_records_m28.npz`.
