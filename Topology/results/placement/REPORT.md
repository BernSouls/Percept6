# Task 4 MISSION — PLACEMENT: final report

Branch `arena/c316ca3a-percept6`. Every part below has its own document next to
this one; this file is the map, the DONE-MEANS checklist, and the verdict.

| part | document | status |
|---|---|---|
| 0 — dependence & determinism | `part0_answers.md` | done, determinism proven |
| 1 — reach, span, landmark census | `part1_reach.md` + `part1_reach.json` | done — **below floor, recorded** |
| 2 — payload | `part2_part3.md`, `matching/placement.py` | done — invariant to 1e−15 |
| 3 — scorer | `part2_part3.md`, `matching/placement_scorer.py` | done — AUC 0.81–0.92 |
| 4 — gate | `part4_gate.md` + `part4_gate.json` | done — **UNEVALUATED (no gradient)** |
| 5 — scenes/tracks | `placement_scenes.py`, pre-registration | done — power pre-registered |
| 6 — arms | `part6_arms.md` + `part6_arms.json` | done — **P fails, on REACH** |

---

## DONE MEANS — one line each

| requirement | result |
|---|---|
| Part 0 written | **yes** — 0a NO, host `matching/alignment.py`, K not run; 0b/0c/0d/0e answered |
| determinism proven | **yes** — bit-identical ×3 in 0e; Part 6 rerun in a **fresh process** is bit-identical (`sha256 2131aafdf4328d74`) |
| reach reported | **yes** — RESOLVABLE .377, NEG-A/B .204, CONTROL .289 |
| reach raised if below floor | **attempted, failed, and explained** — see below; floor still missed |
| landmark share reported | **yes** — **0.000** in all four families |
| landmark raised | **no** — structural; the only candidate link would be proximity, which is not authorised |
| payload implemented | **yes** — continuous signed direction in the source's own frame |
| scorer implemented | **yes** — bounded, frozen tolerance, floor, cap, denominator rules |
| gate implemented, flags dark | **yes** — and measured **UNEVALUATED** |
| tests passing | **yes** — `Topology/test_placement.py` 13 OK, `tests/test_section_b.py` 17 OK |
| tracks validated | **yes** — `assert_identity` = 0.0 |
| ambiguity counts | **yes** — P(ambiguous) ungated 0.3674 |
| power pre-registered | **yes** — `PART6_PREREGISTRATION.md`, 20 scene-graphs, MDE stated |
| one Part 6 run, full table | **yes** — `part6_arms.md` |
| verdict names the failing part | **yes** — REACH |

---

## The arms result

Ambiguous nodes (n = 3565), scene-clustered, MDE = 2.8 × SE:

| arm | accuracy | 95 % CI | MDE |
|---|---|---|---|
| N | 0.8715 | [0.8510, 0.8899] | 0.0278 |
| R | 0.7644 | [0.7440, 0.7847] | 0.0290 |
| **P** | **0.8715** | [0.8510, 0.8899] | 0.0278 |
| P-shuffled | 0.8715 | [0.8510, 0.8899] | 0.0278 |
| W | 0.8715 | [0.8510, 0.8899] | 0.0278 |

| contrast | diff | CI | MDE | verdict |
|---|---|---|---|---|
| **P − N** | +0.0000 | [0, 0] | 0.0000 | **INCONCLUSIVE — zero with zero width** |
| **P − P-shuffled** | +0.0000 | [0, 0] | 0.0000 | INCONCLUSIVE |
| P − R | +0.1072 | [+0.0840, +0.1298] | 0.0327 | POSITIVE* |

\* not a win: R is *worse than N* (0.7644 vs 0.8715), so beating R is not
evidence that placement works — it is evidence the current edge term hurts.

**P is bit-identical to N in every family.** The success criteria therefore
fail as follows: P beats N on ambiguous — **no**; P does no harm on unambiguous
— **yes but trivially** (identical); P beats P-shuffled — **no**; P at chance on
NEGATIVE-A — **not testable**, the control produces no swap (see below).

---

## Why: the measured mechanism

| quantity | value |
|---|---|
| share of nodes with zero checkable placement edges | **0.9758** |
| mean checkable edges per node | 0.0310 |
| payload edges per graph | 30–51 |
| nodes eligible for the spine | **29 of 536** |

The modifier leaves a node with `support is None` **exactly unchanged** — that
is the "no credit, no penalty" rule the mission demands — so for 97.6 % of
nodes P *cannot* differ from N. For the other 2.4 % the bound is ±12.5 %, which
does not reorder the top two.

---

## The fix: attempted, measured, rejected

1. **Reason located.** Reach is **class-admission limited**: the spine admits
   `ORIENTATION/POL/GABOR` and the writer filters on the lens key as well, so
   only the 29 `Orientation` detections may ever carry an edge.
2. **Extending RANGE of the existing class — no-op.** `RANGE_TAU_MULT` at
   2.0 / 4.0 / 8.0 gives 30 / 30 / 30 payload edges. Nothing moves.
3. **Widening class admission — works.** Reach 5.41 % → ContourFragment 8.44 %,
   Texture 11.19 %, Boundary 14.80 %, Position 21.60 %, all **28.27 %**.
4. **Rejected: the frames are not directions.** Under a true 2D image rotation
   of 20° on a grating, a genuine image-plane direction must shift ≈ 0.349 rad.
   `Orientation` shifts 0.262 (one 24-bin step) with concentration 1.00;
   `Boundary` shifts **0.009** and `ContourFragment` **0.000** while both stay
   strongly concentrated. Their θ is a stable local descriptor, not an
   orientation. Attaching placement payloads to them is exactly the
   `polar_angle` frame error the mission forbids.
5. The knob (`ORIENTATION_SPINE_EXTRA`) stays **empty**, and the refactor that
   lets it drive both filters is a proven no-op (identical edge hash
   `91aa4bfaf8d2bd43` against the restored original literals).

So the failure is a **reach wall with a frame-validity wall behind it**: reach
fails first, and the obvious repair fails on frames. Neither was traded to hide
the other, and no weight, threshold or tolerance was tuned.

---

## Other findings, recorded rather than smoothed over

### The NEGATIVE-A control is defective
Expected ≈0.5 on the swappable subset; measured **0.8790** for N and P. The
truth map is built from node truth (same lens, nearest anchor), and the two
objects sit at different image positions, so **no swap is ever produced**. The
control cannot detect a positional leak in either direction. Reported as a
defective control — not as evidence of a leak, and not as evidence of its
absence.

### The gate is UNEVALUATED
P(ambiguous | margin quintile) = 0.3987, 0.3108, **0.8166**, 0.2782, 0.0000 —
not monotone. A gate on the margin would suppress placement where ambiguity is
least and permit it where it is worst. `PLACEMENT_GATE_ENABLED` stays False and
the modifier is evaluated ungated. The bin count and threshold were fixed
before looking and were not tuned towards monotonicity.

### RESOLVABLE is the weakest separation cell
AUC 0.635, d′ 0.50 at 15° — the scene the positive test depends on is the one
whose payload separates least well. Named, not tolerated away (tolerance is
frozen at the q90 of true-correspondence residual).

### Landmark share is zero, structurally
No placement class has a Texture endpoint, and `CURVE_CONTAINMENT` fires only
Contour↔CurveComplex. A texture→landmark link would be proximity, so no new
class was authorised.

---

## Bugs found and fixed

All three had made P vacuous; each was caught by a measurement, not by review.

1. **Reference-side key used the query-side orientation.** `node_support`
   looked up the reference partner under `(mapping[u], …)` for both directions
   of a payload edge, so the reference edge never existed and every node scored
   "no payload" (0/40 covered). The reference key must be the edge's **own
   source image** `(mapping[s], mapping[d])` → coverage 11/20.
2. **Only outgoing edges were walked.** Most nodes have none, so support was
   `None` almost always. Both directions are now used; the frame stays the
   source's, so this is not a frame error but two framings of one node.
3. **The ambiguity label was called without similarity rows,** so only the
   exact-signature route ran. Every signature in these graphs is unique (536
   distinct on 536 nodes), so it labelled **nothing** and the ambiguous bucket
   came out empty (n = 0) on the first run.

---

## What would have to change

The instrument is sound end to end and starved of edges. Raising reach
legitimately needs a lens whose θ is a genuine image-plane direction **and**
which appears on many nodes. In these graphs only `Orientation` qualifies, and
it is 5 % of nodes. Either the front end must emit far more orientation-spine
detections, or a lens that already carries a validated direction must be made
to cover content and texture nodes. Neither is a threshold change, and neither
is authorised by this mission — which is why the verdict is a named failure
rather than a tuned pass.
