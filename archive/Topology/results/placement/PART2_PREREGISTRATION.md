# Mission 2 — EVAL pre-registration

**Committed before any EVAL graph exists.** No EVAL image has been rendered at
the time of this commit; the graph cache contains DEV graphs only. The code that
runs on EVAL is the same file that ran on DEV (`Topology/placement_arms2.py`,
`--set eval` vs `--set dev`); nothing branches on the set except which scene
list and which frozen tolerance table is loaded.

## 1. Scene set (`Topology/placement_eval_spec.py`)

Disjoint from DEV in shape, material, texture frequency, object size,
separation and pose magnitude:

| scene | family | objects | what it is for |
|---|---|---|---|
| `eval_resolvable` | RESOLVABLE | cylinder (stripes, f=12) + box (checker, f=6) | correct correspondence exists |
| `eval_neg_a` | NEGATIVE-A | two identical spheres r=0.19, striped | swap must be indistinguishable |
| `eval_neg_b` | NEGATIVE-B | the same pair, cross-object edges allowed | identity must beat swap |
| `eval_control` | CONTROL | one noise-textured box | placement must not hurt |

DEV pair was r=0.22 spheres separated 0.52; EVAL pair is r=0.19 separated 0.48
with a different texture frequency. DEV resolvable was sphere+box; EVAL is
cylinder+box.

## 2. Views (`EVAL_VIEWS`)

`e1_rot12`, `e2_rot24`, `e3_rot33`, `e4_translate` (−0.07, +0.05),
`e5_scale` 0.90, `e6_rot15_scale` (15°, shift +0.03, scale 1.06).

Six views × four scenes = 24 scene-views, 24 clusters.

## 3. Frozen constants

| constant | value | how it was fixed |
|---|---|---|
| tolerance, every EVAL view | **1.7009 scale units** | largest DEV q90 (`devfit_tolerance.json`), not interpolated |
| `FRAME_COHERENCE_MIN` | 0.50 | DEV paired frame-accuracy table |
| `TARGET_K` | 16 | DEV k sweep, 65.8 % coverage / 67.1 % survival |
| `TARGET_RADIUS_SCALES` | 4.0 | measured non-binding |
| placement weight `w` | 0.25 | one pre-stated value, never swept |
| `EDGE_BETA` (R / R-fixed) | 0.5 | one pre-stated value |
| ambiguity margin | 0.05 | near-tie rule, see §4 |
| `TRACK_RADIUS` | 8.0 | radius sweep 55/80/126/142/152, N accuracy flat |
| `SWAP_RADIUS_WORLD` | 0.06 | DEV swap build (105 pairs, 0 overlap with identity) |

Taking the **largest** DEV q90 for all EVAL views is the conservative choice:
a larger tolerance admits more matches as "supported", so it cannot manufacture
precision for P.

## 4. Ambiguity label (fixed before the run)

A node is **ambiguous** when its top two similarity candidates are a near-tie,
`margin = sim[0] − sim[1] ≤ 0.05`. Truth-free, observable, and it is the
quantity that decides whether a bounded modifier can overturn the decision.

Part 4's stricter label (`sim[0] ≥ 0.99` **and** `margin ≤ 0.05`) is reported
alongside as `part4_label`; it is not the primary.

**Disclosure:** the switch from Part 4's label to the near-tie label was decided
after seeing the DEV margin table, where P's gain was positive in *every* margin
band (+0.079 at ≤0.02, +0.032, +0.049, +0.029, +0.027, then 0.000 once the gap
is too wide for the modifier to matter). The label describes the population the
arm must win; it was not selected to make a result positive, and the primary
outcome is reported under both labels.

## 5. Truth

`track_optimal` (optimal 1–1 assignment per (object, lens), radius 8.0 scale
units). Truth is used **only** to score. It never enters a score, a target
choice or a threshold.

## 6. Arms — all on the SAME graph

| arm | decision rule |
|---|---|
| **N** | top similarity, no placement (baseline) |
| **R** | `(1−β)·sim + β·presence` — frozen as shipped, presence-only, unnormalised sum (hub bias) |
| **R_fixed** | `(1−β)·sim + β·geometry` — normalised, geometry-verified, disagreement-penalised |
| **P** | `sim · (1 + w·(support(u→v) − 0.5))`, support **candidate-dependent** |
| **P_shuffled** | identical to P, payloads of u shuffled across sources (same marginals) |
| **W** | identical to P, payloads rewired to the wrong partner nodes |

`P_shuffled` is the control that matters: it holds the number of edges, the
class mix and the payload distribution fixed and destroys only the association
between a node and its own payload. If P does not beat `P_shuffled`, P is
reading noise.

## 7. Success criteria (fixed now)

Primary outcome: accuracy on **ambiguous** nodes, EVAL, scene-view clustered
bootstrap, 2000 reps, MDE = 2.8 × clustered SE.

1. **P > N** on ambiguous nodes — effect larger than MDE and the 95 %
   cluster-bootstrap CI excludes 0. *(the mission's question)*
2. **P > P_shuffled** on ambiguous nodes, same rule. *(the signal is real)*
3. **P ≥ R_fixed − MDE** — placement is not worse than the best edge-based arm.
4. **R < N** and **R_fixed > R** on ambiguous nodes — R's harm explained and
   repaired (reported separately, R stays frozen).
5. **P does not harm the unambiguous bucket** by more than MDE.
6. **NEGATIVE-A ≈ 0.5** at score level (identity share of identity+swap
   support), reported per view.
7. Determinism: a fresh-process rerun of the EVAL arms is bit-identical.
8. Defaults bit-identical by hash with all flags dark.

If (1) is INCONCLUSIVE the honest report is INCONCLUSIVE, **not** "no effect".
If (2) fails while (1) holds, the gain is not attributable to the placement
signal and must be reported as such.

## 8. Known instrument gap carried into EVAL (declared, not hidden)

**NEGATIVE-B is not realized.** `intra_only=False` adds 1287 placement edges on
the DEV pair and **every one of them terminates on background** (`object_id
= -1`): at the frozen `TARGET_K`/`TARGET_RADIUS_SCALES` the builder emits no
edge between the two objects, so "cross-object edges allowed" creates no
contrast and NEGATIVE-B reads numerically identical to NEGATIVE-A (verified on
all five DEV views). NEGATIVE-B will therefore be reported as **NOT
ESTABLISHED**, with this reason, and not as a pass. The payload's validity is
carried instead by criterion (2), the `P_shuffled` contrast.

## 9. Reads

**One** EVAL read. A second read requires a recorded cause and is reported as a
second read. DEV reruns are free and are not counted here.

## 10. Amendments recorded BEFORE any EVAL number was read

**A1 (instrument, 2026-11-18).** The first `--set eval` invocation crashed with
`AttributeError: 'Cylinder' object has no attribute 'height'` inside
`placement_scenes.posed`: the primitive stores `half_height`, and no scene had
used a cylinder through `posed` before. The run aborted before computing or
writing any accuracy; `arms_eval.json` did not exist and no EVAL number was
seen. Fixed in `posed` (scale the real attribute). **The pre-registered scene
set, views, constants and criteria are unchanged** — this was an instrument gap
in the pose helper, not a change to the experiment.

**A2 (instrument, 2026-11-18).** `posed` was imported only in the DEV branch, so
`--set eval` raised `UnboundLocalError`. Same status: no numbers produced.
