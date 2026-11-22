# Task 4 MISSION — Part 2 payload, Part 3 scorer (measured)

## Part 2 — the payload (`matching/placement.py`)

For a placement edge `u -> v`, the payload is the relative placement of `v` in
**u's own frame**:

| field | definition | unit |
|---|---|---|
| `direction` | `atan2(dy,dx) − θ_u`, wrapped to (−π, π] | radians, **signed, full 2π** |
| `distance` | `|v − u| / node_scale(u)` | source-scale units |
| `log_scale` | `log2(scale_v / scale_u)`; 0 = equal | log2 ratio |
| `source_theta` | u's **continuous** orientation | radians |
| `source_scale` | the divisor actually used | px |
| `tolerance` | frozen per view step (Part 3) | scale units |

**`node_scale` is named**: it is the detection's own `Detection.scale` when
available, else the graph's **median nearest-neighbour anchor spacing** — the
same length unit `EDGE_ORIENTATION_CONTINUITY.dist` is declared in
(`core/schema.py`: "median_spacings"). A node with neither has **no scale and
cannot source a placement**; the edge is then uncheckable rather than given a
default.

**Continuous, not quantized.** `θ_u` is the `Orientation` slot-0 bin
**de-binned to its centre** (`(b+0.5)/24·2π`), read once at the boundary
(`debin_theta`); everything downstream is continuous. Identity remains
categorical (lens type + signature), as before.

**Canonical direction and reversal.** The canonical direction is
SOURCE → TARGET, measured in the source's frame. Reversal is **not** the
negation of `direction`: the far end has its own orientation and its own
scale, so reversal is a separate construction — `PlacementVector.reversed()`
raises, directing the caller to build `placement_vector(v, u, …)` instead.
Reading it as `−direction` is the error the docstring blocks.

**TOGGLE TEST (Part 6's first control), measured:**

| transform | payloads matched | worst delta |
|---|---|---|
| rotate every node by 0.30 rad | 30/30 | 5.3e−15 |
| rotate every node by 1.10 rad | 30/30 | 2.0e−14 |
| translate every node by (13, −7) px | 30/30 | **0.0** |

So the payload is invariant to a global rigid move, to machine precision.

## Part 3 — scorer (`matching/placement_scorer.py`)

```
residual = sqrt( (Δψ/tol)² + (Δlog d/tol)² + (Δlog_scale/tol)² )
support  = exp(−0.5·(residual/σ)²)   for residual ≤ 3σ,  else exactly 0
```

Bounded in [0, 1], exactly 1 at zero residual, **exactly 0 beyond 3σ**. Both
payloads are already frame-relative, so **nothing reads image coordinates**.

- **No checkable edges = no credit and no penalty.** A neighbour that is
  unmapped or has no payload leaves *both* numerator and denominator;
  `node_support` returns `None` (unmeasured) when a node has zero checkable
  edges — never 0, never 1.
- **Each neighbour's contribution is capped** (`PLACEMENT_MAX_NEIGHBOUR_CREDIT`).
- **`checkable_fraction` is reported per arm**, so "support by mapping less" is
  a visible number.

### The frozen tolerance (measured before scoring, `tolerance.json`)

`tolerance = max(q90 of the true-correspondence residual at that view step,
the detector floor)`. **The floor is required**, and the data proves it:

| view step | n | median residual | q90 | detector floor | **frozen tolerance** |
|---|---|---|---|---|---|
| v1_rot5 (5°) | 264 | 0.3721 | 2.4852 | 0.1667 | **2.4852** |
| v2_rot10 (10°) | 264 | 0.6122 | 2.4416 | 0.1667 | **2.4416** |
| v3_rot15 (15°) | 264 | 0.8759 | 2.4008 | 0.1667 | **2.4008** |
| v4_translate | 264 | 0.1777 | 0.6039 | 0.1667 | **0.6039** |
| v5_scale | 264 | **0.0000** | **0.0000** | 0.1667 | **0.1667** |

At `v5_scale` the true residual is **exactly 0**, so a data-only tolerance
would be 0 and every edge would be uncheckable. The floor (1 px of
localisation ÷ 6 px median spacing = 0.1667 scale units) is what keeps the
scorer defined there. This is the mission's "the floor is required" clause,
confirmed rather than assumed.

### SEPARATION of true and wrong residuals (the required finding)

TRUE = each query payload edge under the node-level truth correspondence.
WRONG = the same query vector compared against other reference edges of the
same class whose source is a **different** reference node. (A derangement of
the mapping destroys the edge and yields essentially no comparisons —
measured 1 wrong residual against 31 true — so that construction was abandoned
as uninformative, not as a convenience.)

| view step | n_true | n_wrong | **AUC** | **d′** | median true | median wrong |
|---|---|---|---|---|---|---|
| **v1_rot5 (5°)** | 37 | 4434 | **0.9223** | **1.973** | 0.024 | 0.686 |
| **v2_rot10 (10°)** | 31 | 4132 | **0.8786** | **1.673** | 0.205 | 0.728 |
| **v3_rot15 (15°)** | 21 | 4130 | **0.8093** | **1.214** | 0.205 | 0.738 |

**The payload carries placement at every step, and degrades with rotation as
it should.** No step is near 0.5, so no step had to be reported as carrying
nothing; the tolerance was frozen from data before any of this was computed
and was **not** tuned.

### Per-scene, and the one bad cell

| scene | 5° | 10° | 15° |
|---|---|---|---|
| control_single_sphere | 0.890 | 0.854 | 0.864 |
| neg_2_identical | 0.998 | 0.865 | 0.972 |
| neg_cross | 0.998 | 0.865 | 0.972 |
| **resolvable_two_spheres** | 0.891 | 0.892 | **0.635** (d′ 0.50) |

**FINDING, named and not hidden:** the RESOLVABLE scene's separation collapses
to **AUC 0.635, d′ 0.50 at 15°**. Two identical spheres with one partially
occluded means the far sphere's `Orientation` nodes are progressively
swallowed at larger rotation, and the survivors' frames are near the
silhouette where a small pose change swings the bearing a lot. The payload is
still above chance there, but it is the weakest cell in the table and it is
the scene the positive test depends on. This is reported as the step at which
RESOLVABLE's payload nearly stops carrying placement, per the mission's
instruction to name the step rather than tune the tolerance.
