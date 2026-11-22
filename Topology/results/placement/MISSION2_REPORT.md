# Mission 2 — Placement reach: final report

**One EVAL read.** The pre-registration was committed (`9d7eaac`) before any EVAL
graph existed; the runner asserts every frozen constant against it. The EVAL
arms were run once, and a second invocation was executed **only** as the
determinism control required by the pre-registration (bit-identical output, so
it is the same read, not a second one).

---

## 1. Answer

**Placement-decided correspondence does beat node-only on near-tie nodes — but
the gain cannot be attributed to the placement payload.**

On EVAL, ambiguous nodes (n = 931, 24 scene-view clusters):

| arm | accuracy | 95 % clustered CI | MDE |
|---|---|---|---|
| **N** (node only) | 0.2793 | [0.2469, 0.3158] | 0.0492 |
| **R** (frozen, as shipped) | 0.1869 | [0.1602, 0.2170] | 0.0406 |
| **R_fixed** | 0.2868 | [0.2527, 0.3275] | 0.0535 |
| **P** (placement) | **0.3405** | [0.2974, 0.3925] | 0.0679 |
| **P_shuffled** | 0.3104 | [0.2751, 0.3498] | 0.0534 |
| **W** (rewired) | 0.2492 | [0.2194, 0.2818] | 0.0445 |

| contrast | effect | 95 % CI | MDE | verdict |
|---|---|---|---|---|
| P − N | **+0.0612** | [+0.0221, +0.1026] | 0.0575 | **POSITIVE** |
| P − P_shuffled | +0.0301 | [+0.0010, +0.0610] | 0.0428 | **INCONCLUSIVE** |
| P − R_fixed | +0.0537 | [+0.0036, +0.1110] | 0.0767 | INCONCLUSIVE |
| P − R | +0.1536 | [+0.1088, +0.2002] | 0.0653 | POSITIVE |
| P − W | +0.0913 | [+0.0450, +0.1387] | 0.0670 | POSITIVE |
| R − N | −0.0924 | [−0.1223, −0.0650] | 0.0410 | **NEGATIVE** |
| R_fixed − R | +0.0999 | [+0.0688, +0.1357] | 0.0478 | **POSITIVE** |

Unambiguous nodes (n = 878): N 0.3998, P 0.4089, P−N **+0.0091** [−0.0032,
+0.0263] INCONCLUSIVE — **placement does not harm the unambiguous bucket**.

Net flips over 1797 scored nodes: 301 changed, **100 fixed, 35 broken, 166
neutral, net +65**.

**P − P_shuffled is the control that decides whether this is a placement
result, and it is INCONCLUSIVE.** `P_shuffled` keeps the number of edges, the
class mix and the payload distribution and destroys only the association
between a node and its own payload; it still lands 0.031 above N. So roughly
half of P's advantage over N survives destroying the payload's meaning, and the
part attributable to the payload itself is smaller than the design's floor.
Criterion 2 of the pre-registration is **not met**, and by the mission's own
rule that means the gain is **not attributed to the placement signal**.

---

## 2. The label decides the answer — both are reported

| ambiguity label | n | N | P | P − N |
|---|---|---|---|---|
| **near-tie** (`margin ≤ 0.05`, primary, pre-registered) | 931 | 0.2793 | 0.3405 | **+0.0612 POSITIVE** |
| Part 4 strict (`sim ≥ 0.99` **and** `margin ≤ 0.05`) | 267 | 0.4757 | 0.4644 | −0.0113 INCONCLUSIVE |

Under Part 4's stricter label there is **no gain at all**. The two labels pick
different populations: the strict one keeps only high-similarity nodes, where
the similarity decision is already close to the truth and there is little to
overturn; the near-tie label includes low-similarity ties, where the appearance
decision is a coin flip and placement has room. The switch to the near-tie
label was disclosed in the pre-registration (§4) as having been made after
seeing the DEV margin table, and both labels are reported here as promised.

---

## 3. Controls

**NEGATIVE-A (required ≈ 0.5) — PASS.** Identity share of identity+swap
support, same nodes under both correspondences, on EVAL:

`0.549 / 0.494 / 0.515 / 0.502 / 0.545 / 0.504` (six views), i.e. identity and
swap are indistinguishable, as they must be for two identical objects with
intra-object edges only. DEV: 0.490–0.521 over five views.

**NEGATIVE-B — NOT ESTABLISHED (declared in advance, §8 of the
pre-registration).** `intra_only=False` adds 1287 placement edges on the pair
and **every one of them terminates on background**; at the frozen `TARGET_K=16`
/ `TARGET_RADIUS_SCALES=4.0` the builder emits **no edge between the two
objects**, so "cross-object edges allowed" creates no contrast and NEGATIVE-B
reads numerically identical to NEGATIVE-A on all six EVAL views. It is reported
as an open instrument gap, **not as a pass**.

Running the same contrast on the scene where a real swap *does* exist
(`eval`/`dev` resolvable, different objects) gives identity share 0.47–0.56:
the payload does not prefer the correct object either. This is consistent with
§1 — the payload is informative about *which node*, not about *which object*.

---

## 4. Reach — the first half of the mission

DEV census, committed `0f5b89d` (20 scene-views, object-anchored population):

| quantity | value | 95 % CI | target |
|---|---|---|---|
| coverage of ambiguous nodes under N's mapping | **0.648** | [0.491, 0.769] | 50 %+ **met** |
| pair survival under truth | **0.671** | [0.554, 0.797] | floor 0.60 stated first, met |
| landmark share, texture nodes | **0.999** | [0.984, 1.000] | — |
| coverage under truth (not N) | 0.482 | [0.289, 0.741] | — |
| coverage, all ambiguous incl. background | 0.199 | [0.149, 0.286] | — |

Coverage went 0.076 → 0.648 with `track_optimal` (optimal 1–1 assignment per
object and lens) and a fixed landmark-share indexing bug (0.000 → 0.999).

---

## 5. Frames, tolerance and separation (frozen on DEV, `d1651dc`)

Frame test (`d52e891`): only the anchor orientation field is admitted. Δ_field =
+Δ_pose on both textured scenes (share 0.93/0.92/0.85 and 0.97/0.86/0.83),
NOISE on the untextured control; all 54 lens/provider combinations are 54
STATIC / 16 NOISE / 4 TRACKS, and all four TRACKS are `CONTOUR_PEAK`, which is
STATIC on the control and therefore inadmissible. Coherence floor re-derived to
0.50 from the paired error table (0.30–0.50 → 14.68°, 0.90–1.01 → 2.12°).

Tolerance from DEV q90 of the TRUE residual: 1.4946 / 1.3915 / 1.7009 / 1.0645
/ 1.5406. Separation AUC **0.7555 / 0.7570 / 0.6365 / 0.8217 / 0.6698**
(d′ 0.842 / 0.878 / 0.466 / 1.124 / 0.599). EVAL used **1.7009 everywhere**,
the largest DEV q90, as pre-registered.

**Sign cue tried and rejected.** The field is axial; signing it by the direction
of increasing intensity tracks per node but doubles the true-residual q90
(≈1.5 → ≈2.9) and lowers separation on every view (AUC 0.7113 / 0.6746 / 0.5999
/ 0.7600 / 0.6764). Rejected; the code is kept with its numbers on it.

---

## 6. R's harm, explained; R_fixed repairs it

R is frozen as shipped all mission. Its edge term is **presence-only and
unnormalised** (a sum, not a mean): a candidate is credited for *having*
same-class edges, whatever their geometry, so a node with a busy neighbourhood
wins the assignment without any relation being verified. That is a hub bias,
and it costs accuracy on every population: **R − N = −0.0924 NEGATIVE**
(ambiguous), −0.1036 NEGATIVE (unambiguous).

**R_fixed** replaces it with a normalised, geometry-verified,
disagreement-penalised term: **R_fixed − R = +0.0999 POSITIVE**, and R_fixed is
the only edge arm that gets back to N's level (0.2868 vs N 0.2793). R was never
modified; the repair lives in a separate arm and the two contrasts are reported
separately as required.

---

## 7. Instrument defects found and fixed (all measured, none hidden)

1. **The Part-4 modifier could not change a decision.** `sim·(1 + w·(support(u)
   − 0.5))` multiplies every candidate of a node by the *same* factor, so the
   argmax is arithmetic-invariant. Measured twice: mission 1, and again here on
   the denser graphs — **0 flips of 2373 nodes**, all six arms exactly equal.
   Replaced with `support_for_candidate(u, v)`: candidate-dependent, scoring only
   u's own incident edges in each edge's own source frame, neighbours' images
   from N's mapping, u's image overridden by the candidate.
2. **`total_support` compared different populations** — it summed each
   correspondence over whatever nodes happened to be resolvable (n_identity 35
   vs n_swap 28), manufacturing a spurious "identity beats swap". Now the same
   nodes under both.
3. **`q_inc` was overwritten** by the placement incident index, so R and
   R_fixed read empty edge classes and both collapsed exactly onto N
   (R−N and R_fixed−R both 0.0000, twice). Separated as `ship_inc`.
4. **"broken" counted nodes N also got wrong.** Flips now require N right and P
   wrong (or the reverse), with a neutral bucket; the previous count reported
   net −247 where the truth was net +96 (DEV) and +65 (EVAL).
5. **Verdict rule corrected**: decisive = effect clears the MDE *and* the
   two-level CI excludes zero. The old rule called a CI of [+0.022, +0.072]
   "crosses 0".
6. `placement_scenes.posed` could not scale a `Cylinder` (`height` vs
   `half_height`) and `posed` was imported only on the DEV branch. Both
   crashed the first EVAL invocation **before any number was computed**; recorded
   as amendments A1/A2 in the pre-registration, whose scenes, constants and
   criteria are unchanged.

---

## 8. Pre-registered criteria, pass/fail

| # | criterion | result |
|---|---|---|
| 1 | P > N on ambiguous, effect > MDE and CI excludes 0 | **PASS** (+0.0612) |
| 2 | P > P_shuffled on ambiguous, same rule | **FAIL — INCONCLUSIVE** (+0.0301) |
| 3 | P not worse than R_fixed by more than MDE | **PASS** (P is ahead by 0.0537) |
| 4 | R < N and R_fixed > R | **PASS** (−0.0924, +0.0999) |
| 5 | P does not harm the unambiguous bucket by more than MDE | **PASS** (+0.0091) |
| 6 | NEGATIVE-A ≈ 0.5 at score level | **PASS** (0.494–0.549) |
| 7 | Determinism: fresh-process EVAL rerun bit-identical | **PASS** (`sha256 9d421df10728d0bc`) |
| 8 | Defaults bit-identical with flags dark | **PASS** (`sha256 03fa5d1a3de822d2` on base `d1651dc` and on the tip; 657 nodes, 1456 edges, 0 placement edges) |

**Overall: 6 of 8 pass; criterion 2 — the one that decides whether this is a
placement result — does not.** The honest summary is *reach solved, signal
unproven*.

---

## 9. What would settle it

The payload is informative about *which node* (P beats W and N) but not about
*which object* (§3), and a shuffled payload keeps most of the benefit. That
points at the score, not the reach: `support_for_candidate` averages only u's
own incident edges, so a node with three edges is decided by three numbers, and
the credits are largely explained by degree and coverage rather than by
geometry. The next lever is to make the credit depend on the *relational*
agreement of the candidate neighbourhood (the R_fixed direction, applied
candidate-dependently), not on the count of edges u happens to have — and to
validate it against `P_shuffled` on DEV before any further EVAL read.

No constant was tuned to produce any of the numbers above, and no test, label
or scene was changed after an EVAL number was seen.
