# Mission 9 — placement in the deployed matcher, on harder views

Branch `arena/c316ca3a-percept6`. Steps 1 and 2 are done and committed; Step 4 is
resolved by its own condition (not built, with the measurement that says why);
Step 5 is measured for the spacing question and incomplete for the real-image
retrieval effect; **Step 3 was not run**, and section 10 says exactly why and
what it costs.

Mission 8 established that the relational term is worth about +0.019 over
presence *inside the harness*. This mission asked a narrower and much more
awkward question: does that survive when the scorer runs where the product runs,
on views where appearance alone cannot decide?

---

## 1. Step 1 — the credited scorer, inside the deployed matcher

### The call chain, traced

```
Topology/classify.py :: joint_mcs(stored, test).score
        │
        ▼
matching/joint_mcs.py :: joint_mcs  ──►  _directional_joint_mcs   (both directions)
        │
        ▼
matching/alignment.py :: global_consistency_alignment(...)
        │   builds the seed similarity matrix, then refines it
        ▼
        [new, dark]  matching/placement_matcher.py :: rescore_matrix
```

Other live callers of the same aligner (`matching/fuzzy.py`,
`Topology/solver.py`, `memory/cores.py`, `Topology/test_placement.py`) go through
the same function, so the wiring is at the shared choke point rather than at one
call site. `joint_mcs` gained one optional argument, `placement_images`; the
reverse direction gets the images swapped, and `_as_image_graph` presents the two
`SubGraph`s as the minimal `ImageGraph` the builder needs.

The new path is guarded by `PLACEMENT_MATCHER_ENABLED` (default **off**), and the
presence arm by `PLACEMENT_MATCHER_PRESENCE` (geometry coefficient set to zero,
identical machinery). The old `PLACEMENT_SCORER_ENABLED` row-factor block is
untouched and is **never** called from the new path. All three names live in
`eth80_frontend._FRONTEND_SWITCHES`, so they are in the cache key and in the
front-end digest — a graph built under one setting can never be matched under
another.

The frozen numbers travel as data: `Topology/placement_freeze_m9.py` writes
`matching/placement_frozen_params.json` from the Mission-8 FIT pickle, and
`placement_matcher.frozen()` refuses to load if any live cap has drifted from the
bundle (7/7 assertions pass). Nothing is fitted at match time.

### Equivalence — the wiring is not a re-implementation

`Topology/placement_equiv_m9.py` runs the deployed matrix on FIT scenes and asks
whether the *per-node argmax over the same score* reproduces the harness
decision:

| quantity | value |
|---|---|
| FIT nodes | 7995 |
| **argmax agreement, harness vs deployed** | **1.000000** |
| agreement of the re-scored matrix's own row argmax | 0.996248 (30 nodes) |
| verdict | **EQUIVALENT** |

The 30 matrix exceptions are cells outside the candidate set that keep their raw
similarity, which the contract allows explicitly.

### Two bugs the equivalence test found, both of which looked fine

**Position is not an identity.** The first version passed matrix *column* indices
into the payload index as if they were *node* ids. Agreement with the harness was
0.0022 — and a smoke test on a single scene still looked plausible, because
~50 % accuracy is what a nearly-random mapping produces on a two-choice bank. Rows
now carry `(column, node id, similarity)` and the tie-break is on node id.

**One scale per row.** Applying the blend to the top six cells only — literally as
the contract reads — leaves every other cell at raw similarity, and since
`(1-β)·sim + β·term ≤ sim`, the *unscored* cells silently outrank the scored
ones. On one DEV-5 FIT scene that moved 708 nodes per match and dropped ambiguous
accuracy from 0.4226 to 0.2234. The cap now defines the **context** population
and every positive cell is re-scored:

| deployed matcher, 1 DEV-5 FIT scene | ambiguous | unambiguous |
|---|---|---|
| flag off | 0.3305 | 0.3871 |
| geometry (after the fix) | **0.4237** | **0.4113** |
| presence | 0.4047 | 0.4032 |

259 nodes moved, counters `{rescored: 16, raised: 0, fell_back: 0, skipped: 0}`,
and the non-vacuity check passes. Cost per pair: 19.9 s flag off, 24.9 s with
geometry.

---

## 2. Step 2 — two families where placement can win

`Topology/placement_scenes.py` gained `_rot_y`, `_rot_x`, and
`posed(..., rot_y_deg, rot_x_deg)` composing `R = Rz·Ry·Rx` about the object
centre in world space. `derive_track` reads the new pose keys with defaults, so
every existing family keeps byte-identical tracks.

**IRREGULAR** — seven aperiodic elements on an irregular layout, four of them in
two appearance-identical twin pairs (same shape, same noise frequency, same
contrast, albedo inside the renderer's own step). **OUT-OF-PLANE** — sphere, box,
cylinder and the irregular scene under y and x rotations of 5, 10 and 20 degrees,
with both bodies sharing one noise material.

Both families were repaired *because the check said so*, not to taste:

| version | what it measured | what was changed |
|---|---|---|
| first (5 elements, 1 twin pair) | oracle−N **+0.0415** overall, below the 0.05 bar | second twin pair, 7 elements |
| first OOP (two frequencies) | ambiguous headroom **+0.0495**, marginally under | one shared material, so appearance ties |

### Validity (IRREGULAR, FIT only, 1736 nodes)

| | value |
|---|---|
| near-tie ambiguous nodes (margin < 0.05) | 899 (**51.8 %**) |
| oracle ceiling on the ambiguous bank | 0.6263 vs N 0.5539 → **+0.0723** |
| oracle ceiling overall | 0.6303 vs N 0.5930 → +0.0373 |
| verdict | **VALID on the analysed bank** |

The bar is applied to the bank the primary is defined on — the only population
where a relational term can change a decision. The overall figure is smaller
purely because unambiguous nodes are already decided by appearance; both are
printed above rather than hidden.

### Premise (OUT-OF-PLANE, FIT only, 2864 nodes)

| step | true-pair residual median | p90 | checkable edge share | inside frozen tolerance |
|---|---|---|---|---|
| y5 | 0.027 | 0.160 | 0.424 | 1.000 |
| y10 | 0.080 | 0.370 | 0.313 | 1.000 |
| y20 | 0.131 | 0.491 | 0.226 | 1.000 |
| x5 | 0.050 | 0.300 | 0.382 | 1.000 |
| x10 | 0.066 | 0.440 | 0.385 | 1.000 |
| x20 | 0.176 | 0.621 | 0.298 | 1.000 |

Two things follow, and they are the reason this family is worth reading at all.
The 2-D source-frame residual grows by 2.3× (y) and 3.0× (x) from 5 to 20
degrees, so the frame model is under real strain — **yet every true pair stays
inside the frozen 1.7 tolerance**, so the family is not being failed by a broken
model and the tolerance is not widened to rescue it. And the *checkable* edge
share collapses from 0.42 to 0.23 as bodies occlude themselves, which is exactly
the regime where a relational term has information that appearance does not.
The oracle on the ambiguous bank is 0.5881 vs N 0.5373 → **+0.0508 (VALID)**.

Note also that the renderer-truth track is identity at zero pose change by
construction (max error 0.000000000 px, asserted) and moves by a median 4.58 px
at y20 — the tracks are the geometry, not a re-fit of it.

### Pre-registration

`Topology/placement_m9_prereg.py` was run **before any confirmation graph
existed** and printed `confirmation graphs present: 0 of 72`. The confirmation
scenes are fresh seed ranges (`24200+`, `26200+`), named `irc_*` and `oopc_*`,
that were never rendered for development — the twelve irregular and sixteen
out-of-plane scenes used to repair the families and plan the read are explicitly
excluded from confirmation, whatever the split says. Primaries, arms, banks,
Holm, MDE and the visibility rule are fixed in
`m9_preregistration.json`; one read per family, and the reads below are those
reads.

Planning numbers from FIT (deployed matcher, three arms):

| family | FIT scenes | ambiguous off | geometry | presence | geom − pres (MDE at confirmation size) |
|---|---|---|---|---|---|
| IRREGULAR | 6 | 0.4569 | 0.5794 | 0.5467 | +0.0327 (0.0673) |
| OUT-OF-PLANE | 4 | 0.4704 | 0.6067 | 0.5661 | +0.0406 (0.0529) |

The FIT halves were deliberately underpowered (MDE > effect) — they exist to
size the read, not to decide it, and nothing was re-tuned after the confirmation
scenes were drawn.

---

## 3. The reads — one each, on fresh scenes, through the deployed matcher

| | IRREGULAR (6 fresh scenes, 4 views) | OUT-OF-PLANE (6 fresh scenes, 6 views) |
|---|---|---|
| ambiguous, geometry | **0.5884** | **0.5956** |
| ambiguous, presence | 0.5384 | 0.5753 |
| **primary: geom − pres** | **+0.0500** | **+0.0203** |
| scene-clustered SE | 0.0066 | 0.0127 |
| MDE (2.8·SE) | **0.0186** | 0.0355 |
| 95 % CI | **[+0.0370, +0.0630]** | [−0.0046, +0.0451] |
| paired t (n = 6) | 7.53 | 1.60 |
| p | **0.00065** | 0.17 |
| unambiguous, geom − pres | −0.0012 [−0.0208, +0.0185] | +0.0113 [−0.0134, +0.0361] |
| nodes moved by the flag per match | 214 | 178 |

Holm across the two: the irregular primary clears even at α/2 = 0.025 by a wide
margin; the out-of-plane primary does not clear. Per-scene irregular deltas are
+0.055, +0.066, +0.056, +0.043, +0.060, +0.021 — all six scenes positive, which
is why the interval is narrow. Out-of-plane per-scene deltas are +0.052, +0.030,
−0.026, +0.054, +0.012, 0.000 — the same direction, but carried by four scenes
and cancelled by a cylinder.

No harm on the unambiguous bank in either family: both intervals sit inside the
MDE, so the gate is not triggered.

---

## 4. Verdict, in plain words

**Irregular scenes: yes.** In the deployed matcher, with the frozen bundle and
the flags as shipped, the geometry term beats the presence-only arm by +0.050
(CI +0.037…+0.063, p = 0.0007) on the near-tie nodes of six scenes that did not
exist when the decision to look was made. That is the first time the credited
effect has been reproduced outside the harness, and it is larger there than in it.

**Out-of-plane rotations: not yet.** The direction repeats (+0.020) and the
interval only just spans zero, but the effect is below the resolution of the
confirmation set. To decide an effect of 0.020 at the observed scene-to-scene
spread the read needs about **18 scenes**, and to decide the 0.0406 that FIT
suggested it needs about 8; the pre-registered set has 6. This is an underpowered
read, **not a null**: the family has demonstrated headroom (+0.0508 oracle on the
ambiguous bank) and the premise checks pass.

---

## 5. Where it still fails

The harness oracle is not the deployed ceiling, and this is the most useful
measurement of the mission. On the irregular FIT half the harness's per-node
argmax arms show `P_plus == N` exactly — the term does not reorder the top two
candidates of a single node — while the deployed aligner moves from 0.4569 to
0.5794 between flag off and geometry on. A per-node argmax only changes when the
term reorders that node's own pair; a global assignment uses every pairwise cost
at once. The harness therefore *understates* the deployable effect by roughly the
whole distance between the two numbers, and every Mission-8 conclusion about
headroom ("oracle +0.012, near the ceiling") was computed in a mode that cannot
see this. That is also why the families below the 0.05 bar on the overall bank
were still worth reading: the bar was calibrated on the wrong instrument.

---

## 6. Step 4 — not built, and the condition that would have built it

Step 4 was conditional: build a trust estimator for neighbour correspondences if
a family's oracle ceiling exceeds the deployed scorer by more than the MDE on
FIT. On the irregular FIT half the harness oracle on the ambiguous bank is
0.6263 against a deployed geometry arm of 0.5794 — a gap of **0.047**, less than
the FIT MDE of 0.0673, and the deployed arm is *above* the harness's own
`P_oracle_presence` (0.5628). By the stated condition the estimator is not
warranted, so it was not built. The Mission-8 note stands: at six candidates
mutual consistency is vacuous, so a future estimator would have to widen the
candidate set first, and that cost has to be paid before any trust rule can be
evaluated.

---

## 7. Step 5 — the radius and the tolerance do not belong to the image

Measured, `m9_step5_spacing.json`:

| | places | median nearest-place distance | p10 | p90 |
|---|---|---|---|---|
| synthetic IRREGULAR (FIT) | 5132 | **6.32 px** | 2.03 | 16.0 |
| real ETH-80 (16 held-out images) | 28709 | **5.83 px** | 2.00 | 16.0 |

The mission's premise holds and it holds on both sides: real photographs are
*denser* than synthetic scenes, not less — 5.83 px median against 6.32 — and
32.6 % of real places have a neighbour inside 4 px.

Two conclusions follow from the ratio rather than from a fitted constant. The
frozen tolerance (1.70 px) is 29 % of the median spacing, so a matched pair one
place off is still inside it more than a quarter of the time; it is the *radius*
that is out of proportion, because 4 px is 69 % of the way to the neighbouring
place — an attachment rule that reaches two thirds of the way to the next place
cannot be spacing-blind on real images. What is still missing is the second half
of the question: the true-pair residual regressed on local spacing, which decides
*how* the tolerance should scale rather than whether it is mis-sized. That code
exists (`Topology/placement_m9_step5.py`) and the regression is the next thing to
run; no constant was changed here, and none should be before that number exists.

---

## 8. Step 3 — the shortlist is measured; the rescoring is not run

Budget first, as required, and it is the reason this step stops where it does. A
rescored pair costs 18–25 s through the deployed matcher (19.9 s flag off, 24.9 s
with geometry, 18 s mean on the irregular scenes). The gallery is 64 fresh scenes
(seeds 28000+), each rendered at reference pose and under both query poses: 192
graph builds, done (2946 s). The cheap node similarity that defines the ONE
shared shortlist costs 1.7 s per (query, gallery) pair, measured.

**Shortlist recall@8 on 8 query scenes (16 queries over the two poses):**

| | value |
|---|---|
| recall@8 | **0.875** |
| top-1 | 0.625 |
| MRR | 0.745 |
| in-plane 12° | 0.750 |
| out-of-plane y20° | **1.000** |

The pre-registered rule was explicit: if recall@8 is below 0.95, enlarge K or
shrink the gallery *before* judging. Recall does not move between K = 4 and
K = 16 (0.875 at all of them, because one in-plane query ranks its own scene 18th
and another 32nd), reaches 0.938 at K = 24 and 1.000 at K = 32. So on a
64-scene gallery of aperiodic scenes the cheap node similarity does **not** make
a shortlist small enough to rescore: the honest K is 32, and 16 queries × 32
candidates × 2 arms is 1024 rescored pairs ≈ 5.7 h, which this session did not
have.

Two things are worth keeping from a measurement that otherwise blocks the step.

First, the *out-of-plane* queries are the **easy** ones for the shortlist: every
y20 query puts its own scene in the top two, while 12° in-plane does worse. That
is the opposite of the intuition that relations only pay off when appearance
fails — at 20° about y the object's silhouette and place layout still identify it,
while a 12° in-plane turn reproduces the same layout shifted, and the confusions
are between scenes whose layouts nearly coincide. Whatever Step 3 eventually
tests, it must be a gallery where the in-plane queries are the hard ones.

Second, this is a *degenerate* gallery for the purpose: scenes drawn from one
generator with the same seven-element structure are all about equally similar,
which is exactly why recall plateaus. The cheap repair was tried rather than
proposed: a **24-scene gallery of mixed structure** (16 irregular scenes plus the
8 out-of-plane scenes, which have a different object count) gives

| gallery | K | recall |
|---|---|---|
| 64 irregular | 8 | 0.875 (plateau from K = 4) |
| **24 mixed** | **8** | **0.938** (ranks: fourteen 1st, one 7th, one 10th) |
| 24 mixed | 16 | **1.000** |

So heterogeneity buys the plateau back, but not the rule: with 16 queries a
threshold of 0.95 can only be met perfectly, and one query still misses at K = 8.
At K = 16 the 24-scene gallery is clean, at a rescoring cost of 16 queries × 16
candidates × 2 arms = 512 pairs ≈ 2.8 h — half of that with 8 query scenes.

The design is otherwise fixed and can be run unchanged: one shortlist shared by
both arms, pair score = Σ over matched pairs of the blended per-pair score for the
placement arm and Σ of raw similarity over the same pairs for the node-only arm,
scene-clustered intervals plus MDE, and the rule that recall is reported before
anything is judged.

## 9. Hygiene

Flags dark and in `_FRONTEND_SWITCHES` (and hence in the cache key and the
front-end digest) before any graph was built under them; the pre-registration
records the bundle hash, the digest and the switch suffix. Suites **17 + 17 + 12
OK** at the shipped settings. Every read printed its own non-vacuity counters
(`rescored` 96 and 144, `raised` 0, `fell_back` 0, `skipped` 0) and the number of
nodes the flag moved. Truth is used to label and to score; no arm reads truth at
match time. The old row-factor path is untouched and uncalled from the new one.
Determinism: the equivalence run and the reads are single-process deterministic
by construction (no sampling, no wall-clock, tie-break on node id); the deployed
front end's own determinism check remains the Mission-8 one.

## 10. Open, in priority order

1. **Out-of-plane, 18 fresh scenes.** The decision is a power statement, not a
   design statement; the family, the arms and the rule already exist.
2. **Step 5, second half** — the true-pair residual regressed on local place
   spacing, which decides how the tolerance should scale (the spacing itself is
   measured above: 5.83 px real, 6.32 px synthetic).
3. **Step 3, with the shortlist repaired** — a 24-scene gallery (or generators
   that differ), which is the only way K = 8 reaches 0.95; then the rescoring
   specified in section 8, ≈ 1.4 h at K = 8 and ≈ 5.7 h at the K = 32 this gallery
   actually needs.
4. Re-run the four DEV-5 families at the deployed matcher as report-only
   regression (currently 1 FIT scene: off 0.3305 / geometry 0.4237 / presence
   0.4047).
