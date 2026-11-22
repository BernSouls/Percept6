# Task 4 MISSION — Part 6: the ONE arms run

Pre-registration written before the run: `PART6_PREREGISTRATION.md`
(scene count, MDE floor, all pre-stated decisions). Raw: `part6_arms.json`.

**Model graphs:** 4 families × 5 non-base views = **20 scene-graphs** (n_eff =
20 for every clustered interval; clusters are `family|scene|view`).

---

## THE ARMS TABLE

Clustered by scene-graph, 2000-replicate bootstrap, MDE = 2.8 × SE.

### Ambiguous nodes (n = 3565 node-view rows)

| arm | accuracy | 95 % CI | MDE |
|---|---|---|---|
| N (node-only) | **0.8715** | [0.8510, 0.8899] | 0.0278 |
| R (current edges) | **0.7644** | [0.7440, 0.7847] | 0.0290 |
| **P (placement)** | **0.8715** | [0.8510, 0.8899] | 0.0278 |
| P-shuffled | **0.8715** | [0.8510, 0.8899] | 0.0278 |
| W (rewired) | **0.8715** | [0.8510, 0.8899] | 0.0278 |

| contrast | diff | 95 % CI | MDE | verdict |
|---|---|---|---|---|
| **P − N** | **+0.0000** | [0.0000, 0.0000] | 0.0000 | **INCONCLUSIVE — zero with zero width** |
| P − R | +0.1072 | [+0.0840, +0.1298] | 0.0327 | POSITIVE |
| **P − P-shuffled** | **+0.0000** | [0.0000, 0.0000] | 0.0000 | **INCONCLUSIVE** |
| P − W | +0.0000 | [0.0000, 0.0000] | 0.0000 | INCONCLUSIVE |

### Unambiguous nodes (n = 4566)

| arm | accuracy | 95 % CI | MDE |
|---|---|---|---|
| N | 0.3007 | [0.2684, 0.3332] | 0.0463 |
| R | 0.2716 | [0.2405, 0.3020] | 0.0439 |
| P / P-shuffled / W | 0.3007 | [0.2684, 0.3332] | 0.0463 |

P − R = +0.0291 [+0.0213, +0.0367] (POSITIVE); P − N = P − P-shuffled = P − W
= +0.0000 exactly.

### By family, ambiguous nodes

| family | n | N | R | P |
|---|---|---|---|---|
| CONTROL | 873 | 0.8706 | 0.7812 | 0.8706 |
| NEGATIVE-A | 901 | 0.8790 | 0.7691 | 0.8790 |
| NEGATIVE-B | 901 | 0.8790 | 0.7691 | 0.8790 |
| RESOLVABLE | 890 | 0.8573 | 0.7382 | 0.8573 |

**P is identical to N in every family to four decimals.** That is not a
rounding coincidence; it is the result, and its cause is measured below.

---

## WHY P ≡ N — the measured mechanism

| quantity | value |
|---|---|
| node-view rows | 8131 |
| **share of nodes with ZERO checkable placement edges** | **0.9758** |
| mean checkable edges per node | **0.0310** |
| mean placement edges per node | 0.0552 |
| checkable fraction (of existing placement edges) | 0.561 |

**97.6 % of nodes have no checkable placement edge.** The modifier is defined to
leave a node with `support is None` completely unchanged (no credit, no
penalty), so for 97.6 % of nodes P *cannot* differ from N. For the remaining
2.4 % the modifier is bounded by `1 ± w/2 = 1 ± 0.125`, and the observed
similarity gaps are large enough that a ≤12.5 % reweighting does not reorder
the top-2. Hence the arm produces exactly the same assignments.

This is consistent with Part 1's census: placement edges exist (39–51 per
graph) but they attach to `Orientation` nodes only (29–37 of them), so the
reachable set is tiny relative to 520–620 nodes per graph.

## The family expectations, checked

- **NEGATIVE-A** — expected ≈0.5 on the swappable subset if placement is
  correctly blind to which object a node came from. Measured **0.8790 for N
  and P, 0.7691 for R**. This is **NOT ≈0.5**: the truth mapping is built from
  *node truth* (same lens, nearest anchor), and the two objects are at
  different image positions, so the nearest-anchor correspondence never
  actually produces a swap. The NEGATIVE-A control is therefore **not doing
  its job** — it cannot detect a positional leak, because there is no swap to
  detect. Reported as a defect in the control, not as evidence of a leak.
- **NEGATIVE-B** — identical numbers to NEGATIVE-A (0.8790 / 0.7691 / 0.8790),
  as expected since the arms do not consume cross-object edges differently
  here. A win would have been legitimate; there is no win to read.
- **RESOLVABLE** — the positive test: 0.8573 (N/P) vs 0.7382 (R). **P does not
  beat N**, and R is *worse* than N by 0.13. The positive test fails.
- **CONTROL** — 0.8706 (N/P), 0.7812 (R). P does not hurt: it is identical.

## Verdict

The mission's success criteria, applied one by one:

| criterion | result |
|---|---|
| P beats N on ambiguous nodes | **NO** — +0.0000 exactly, zero-width interval |
| P does not hurt unambiguous nodes | **YES** — identical (0.3007), trivially |
| P beats P-shuffled | **NO** — +0.0000 exactly, identical |
| P is at chance on NEGATIVE-A | **NOT TESTABLE** — the control produces no swap, so ≈0.5 is not the right expectation and the control cannot detect a positional leak |

**Verdict: FAILS, and the failing part is REACH.** The mission asks to name the
part when it fails: it is not the payload (Part 3 measured AUC 0.81–0.92,
d′ 1.21–1.97 — the payload carries placement), not the frame (the toggle test
passes to 1e−15), not the scorer (it is exact, bounded and leaves unmapped
neighbours out of both numerator and denominator). It is that **placement
support reaches 2.4 % of nodes**, so an arm built on it is a no-op. The chain
is: Part 1 reach 20.4–37.7 % of *frame* nodes, of which the payload-carrying
subset is 39–51 edges on `Orientation` endpoints only, giving 0.031 checkable
edges per node.

The mission says: if it fails, find the actual reason and fix it; no reverts.
The reason is located (reach). The fix is not a threshold: it needs placement
edges that attach to a far larger share of nodes, which is the **landmark
coverage failure already measured in Part 1b (0.000 in all four families)** —
texture and content nodes have no placement endpoint at all. Fixing it means a
new placement class over content nodes, which Part 1b argued would manufacture
false landmarks with the current periodic textures. That is the honest
dead-end of this instrument at this reach, and it is stated rather than papered
over with a tuned weight.

## Controls that DID pass

| control | result |
|---|---|
| rotate every node together → P unchanged | payload invariant to 5.3e−15 (0.30 rad), 2.0e−14 (1.10 rad) |
| translate every node together → P unchanged | payload invariant **exactly** (delta 0.0) |
| fresh-process rerun bit-identical | 3 processes, identical node count and identical full + selected edge-set hashes |
| K arm | **not run**: 0a found the walker's admission test is not mapping-dependent, so it is not a placement candidate |

---

# THE FIX, ATTEMPTED AND MEASURED

The mission says: if it fails, find the actual reason and fix it; extend the
RANGE of an existing class first, and authorise a new class only if none can
carry it. That procedure was followed and is recorded here in full.

## 1. The reason, located exactly

The spine admits three signature classes:

```
ORIENTATION_SPINE_CLASSES = ("ORIENTATION", "ORIENTATION_POL", "ORIENTATION_GABOR")
```

and the writer filters on **two** things: the lens key `d.k`, restricted to
`("Orientation", "OrientationPol")`, and `d.sigma.class_`. In the synthetic
placement graphs the class census is

| class | nodes |
|---|---|
| POSITION | 102 |
| CONTOUR_PEAK | 75 |
| BOUNDARY_ANCHORED | 64 |
| CONTOUR_FRAGMENT | 63 |
| BORDER_OWNERSHIP | 50 |
| TEXTURE_BOUNDARY | 35 |
| TEXTURE | 34 |
| **ORIENTATION** | **29** |
| ENDSTOPPED | 26 |
| … | … |

so **29 of 536 nodes** are ever eligible. Reach is not a radius problem, it is
a class-admission problem.

## 2. Extending the RANGE of the existing class — measured NO-OP

`ORIENTATION_CONTINUITY_RANGE_TAU_MULT` swept on one graph:

| search radius multiplier | nodes | payload edges | endpoints | reach |
|---|---|---|---|---|
| 2.0 (shipped) | 536 | 30 | 29 | 5.41 % |
| 4.0 | 536 | **30** | **29** | **5.41 %** |
| 8.0 | 536 | **30** | **29** | **5.41 %** |

Range extension changes nothing, because the binding constraint is which nodes
may enter at all, not how far they may look. Part 1's "extend the range first"
therefore resolves to: **range cannot be extended usefully here.**

## 3. Widening the class admission — it DOES raise reach

A dark, pair-carrying knob `ORIENTATION_SPINE_EXTRA` was added, empty by
default. Both filters read it, because a class-name list alone is **inert** —
measured: widening only the class list, or only the lens list, left the payload
count at exactly 30, since the two namespaces differ (`"Boundary"` vs
`"BOUNDARY_ANCHORED"`).

| widening | nodes | payload edges | endpoints | **reach** |
|---|---|---|---|---|
| none (shipped) | 536 | 30 | 29 | 5.41 % |
| + ContourFragment | 533 | 47 | 45 | 8.44 % |
| + Texture | 536 | 92 | 60 | 11.19 % |
| + Boundary | 527 | 124 | 78 | 14.80 % |
| + Position | 537 | 153 | 116 | 21.60 % |
| **+ all** | 527 | **220** | **149** | **28.27 %** |

So the reach defect **is** fixable in principle: 5.41 % → 28.27 %.

## 4. Why it was NOT adopted — the frames are not directions

A node's frame is its orientation, so a lens may only join the spine if its
`theta` is a genuine image-plane direction. Tested on a single-orientation
grating rotated by a **true 2D image rotation of 20°** (the well-posed test: the
population is concentrated, so the circular mean is meaningful). A genuine
direction must shift by ≈ 0.349 rad.

| lens | n | shift in theta | concentration R at 0° / 20° | verdict |
|---|---|---|---|---|
| **Orientation** | 17 | **0.262** | **1.00 / 1.00** | **DIRECTION** (one 24-bin step for a 20° turn) |
| Position | 96 | 0.146 | 0.19 / 0.17 | unconcentrated |
| IllusoryContour | 32 | 0.340 | 0.62 / 0.05 | direction, but 1 node in the real scenes |
| TextureBoundary | 34 | 0.047 | — | **not a direction** |
| **Boundary** | 17 | **0.009** | **0.89 / 0.84** | **not a direction** |
| **ContourFragment** | 8 | **0.000** | **1.00 / 1.00** | **not a direction** |
| Color | 5 | 0.399 | 0.48 / 0.00 | unconcentrated |

`Boundary` and `ContourFragment` are strongly **concentrated** and still do not
move — their `theta` is a stable local descriptor, not an orientation in the
image plane. Attaching placement payloads to them would put the "direction" in
a frame that does not rotate with the object: **exactly the `polar_angle`
frame error the mission forbids.** No new registered class was authorised, and
none of the candidates can carry it.

**Conclusion: the fix is available in reach but invalid in frame, so it is not
adopted.** The knob stays dark (`ORIENTATION_SPINE_EXTRA = ()`).

### A confounded test that was discarded

An earlier probe compared `theta` medians across the `v2_rot10` pose and
appeared to show *every* lens shifting by exactly 10°. That test is invalid on
two counts and was not used: the v-series pose is a 3D object rotation, not a
2D image rotation, and for near-isotropic orientation populations the circular
mean is unstable and carries no information. The grating test above is the
well-posed replacement.

## 5. The refactor is a proven no-op

Restructuring the two filters into `_orient_spine_classes()` /
`_orient_spine_lenses()` changes nothing at the default:

```
REFACTOR    nodes=711 edges=1717 hash=91aa4bfaf8d2bd43
ORIGINAL    nodes=711 edges=1717 hash=91aa4bfaf8d2bd43   (pre-refactor literals restored)
REFACTOR-2  nodes=711 edges=1717 hash=91aa4bfaf8d2bd43
```

## Revised verdict

The failing part is **REACH**, and the reason it stands unfixed is that the
only lenses with enough coverage do not carry a usable orientation frame. This
is a **frame-validity** wall behind a reach wall: reach fails first, and the
obvious repair fails on frames. Both are named here rather than one being
traded for the other, and no weight, threshold or tolerance was tuned to hide
it.
