# Mission 10 — refit under the deployed objective, confirm out of plane, move toward real images

Branch `arena/c316ca3a-percept6`, from `a6d23fb`. This report is written as the
runs land; the last section says what is still running and what it costs.

---

## 1. Step 1 — the coefficients, refitted against the objective that is deployed

Mission 8 fitted `a_exist`, `a_geom`, the blend weight and the candidate cap for a
**per-node argmax**. Mission 9 showed the deployed matcher does something else with
them: it re-scores the pairwise matrix and a **global assignment** reads it, which
is why the harness showed `P_plus == N` while the aligner moved +0.12. So the
numbers were refitted against the deployed objective.

The procedure was fixed in the script's docstring before the first evaluation: the
six irregular FIT scenes at two views each (12 pairs, six independent scenes),
pooled ambiguous-bank accuracy of the aligner's own mapping, coordinate ascent over
a stated grid, adoption only if the gain beats the FIT MDE.

| variant (deployed matcher, ambiguous bank) | accuracy | Δ vs the frozen bundle |
|---|---|---|
| **Mission-8 bundle (reference)** | **0.5249** | — |
| beta 0.35 | 0.5509 | +0.0259 (MDE 0.0809) |
| beta 0.65 | 0.5498 | −0.0011 |
| a_exist 0.25 | 0.5256 | −0.0252 |
| **a_exist 0.55** | **0.5557** | **+0.0308 (MDE 0.0634)** |
| a_geom 0.60 / 1.10 | 0.5406 / 0.5406 | −0.0151 |
| cand_cap 12 | 0.5462 | −0.0095 |

**Decision: the bundle is KEPT.** The best variant is +0.0308 with a FIT MDE of
0.0634 — the gain does not clear the resolution of the FIT pool, and the rule was
stated before the numbers existed. `m10_refit.json`, six scenes, coordinate order
recorded.

Two things worth keeping from the search even though nothing was adopted. First,
the direction is consistent: `a_exist` 0.55 (more weight on existence) and beta
0.35 (less weight on the term as a whole) both lean the same way, and `cand_cap`
12 does not help — at twelve candidates the context is as noisy as the choice it
is meant to inform. Second, a nine-point search over a six-scene pool cannot
resolve a 0.03 effect; the honest reading is that the Mission-8 numbers are not
*shown* to be wrong, not that they are shown to be right.

### The split of the deployed gain

Five fixed arms, same pool, same bundle (`m10_refit.json → split`):

| arm | ambiguous | unambiguous |
|---|---|---|
| N (flag off) | 0.4161 | 0.3402 |
| presence, no context iteration | 0.5007 | 0.3231 |
| presence, context iterated | 0.5011 | 0.3561 |
| geometry, no context iteration | 0.5045 | 0.3493 |
| **geometry, context iterated** | **0.5249** | **0.3876** |

Decomposed additively:

| step | Δ ambiguous |
|---|---|
| N → presence | **+0.0850** |
| presence → geometry (no iteration) | +0.0038 |
| context iteration, on the geometry arm | **+0.0204** |
| context iteration, on the presence arm | +0.0004 |
| **total N → geometry** | **+0.1088** |

This is the most useful number of the step, and it is not the one Mission 8
reported. **Context iteration is worth nothing on its own and +0.020 once geometry
is present.** The iteration re-orders the mapping; the geometry term is what makes
the re-ordering count. Read as a decomposition of "presence + context", the
Mission-8 credit was measured on an arm where the two were entangled: presence
carries 78 % of the deployed gain, geometry 3.5 %, and iteration 19 % — but only
in the presence of geometry.

---

## 2. Step 2 — the out-of-plane confirmation (one look, 20 fresh scenes)

Mission 9's read was underpowered at six scenes (+0.0203, MDE 0.0355). This read
was registered **while 0 of its 140 graphs existed** — the first build was stopped
and its 23 graphs deleted so that the count in `m10_prereg_oop.json` is real —
on a seed range (base 30000) that had never been rendered.

The FIT estimate that sizes it: all eight out-of-plane FIT scenes, six views,
geometry 0.5997 vs presence 0.5613 = **+0.0383** (SE 0.0135), so at twenty scenes
the registered MDE is **0.0239** — resolution for the effect FIT suggests, which is
exactly what Mission 9's set lacked.

**Result — one look, 20 scenes, 120 view-pairs (`m10_oop_check.json`):**

| | geometry | presence | delta | SE | MDE | 95 % CI | p |
|---|---|---|---|---|---|---|---|
| **ambiguous bank** | **0.5655** | 0.5434 | **+0.0221** | 0.0064 | 0.0178 | [+0.0096, +0.0345] | **0.0025** |
| unambiguous bank | 0.4297 | 0.4189 | +0.0108 | 0.0041 | 0.0115 | [+0.0027, +0.0189] | 0.017 |

**Verdict: yes — placement helps out of plane.** The effect clears its own MDE
with room (0.0221 against 0.0178) and p = 0.0025 over twenty independent scenes,
and the unambiguous bank *rises* rather than falls, so the no-harm rule is not
triggered. The effect is smaller than the FIT estimate (0.0383) — FIT pools are
optimistic, which is the reason for a fresh set — and it is the same direction as
Mission 9's underpowered +0.0203, now decided.

That closes the Mission-9 null-by-underpower: out-of-plane placement is worth about
+0.02 on the near-tie bank, at 20 degrees of rotation where the checkable-edge share
falls to 0.23.

Counters over the read: 29 048 cells re-scored, `raised` 0, `fell_back` 0,
`skipped` 0 — the loud-failure path never fired.

---

## 3. Step 4a — does the geometry belong to the image?

### The regression

2764 true pairs over the six irregular FIT scenes, residual against the local
place spacing (nearest-neighbour distance among distinct places):

| spacing bin | n | residual median | p95 | p99 |
|---|---|---|---|---|
| < 3 px | 154 | 0.088 | 0.227 | 0.518 |
| 3–4.5 px | 570 | 0.089 | 0.493 | 0.761 |
| 4.5–6 px | 374 | 0.087 | 0.815 | 0.917 |
| 6–9 px | 1064 | 0.060 | 0.325 | 0.837 |
| > 9 px | 602 | 0.069 | 0.205 | 0.759 |

**The residual does not grow with the spacing.** The median is flat to within
0.03 across a 3× change in place density, the power-law exponent is **−0.18**
(if anything, tighter places give slightly *smaller* residuals), and p95 wanders
non-monotonically. The reason is by construction: the payload is scale-free —
direction, a distance normalised by the source's own scale, a log scale — so a
tolerance expressed in those same units has no reason to follow a pixel distance.

That answers the mission's question in the negative **for the tolerance**, and it
answers it with a measurement rather than with a change of mind: the premise that
an absolute tolerance is mis-sized on dense images does not hold for this residual.

### The search, and why both flags stayed dark

The rule still had to be *tested*, not merely argued away, so it was implemented
dark and fitted by the deployed objective on the same FIT pool:

| rule | setting | ambiguous | Δ | FIT MDE |
|---|---|---|---|---|
| tolerance `base·(sp/ref)^γ` | γ = 0.5 | 0.5416 | +0.0166 | 0.0676 |
| | **γ = 1.0** | **0.5635** | **+0.0385** | 0.0704 |
| radius cap `factor·sp` | 1.0 | 0.4991 | −0.0259 | 0.0710 |
| | 1.5 | 0.5470 | +0.0220 | 0.0666 |
| | **2.5** | **0.5593** | **+0.0344** | 0.0811 |

Both best settings are **positive and both fail the adoption rule** (gain must
exceed the FIT MDE): the flags stay dark, and the fitted numbers are frozen in
`matching/placement_spacing_params.json` rather than being retyped by the next
person. A cap of 1.0× spacing — the tightest, the one the "reaches past the
neighbour" argument implies — is the *worst* setting (−0.026): capping the radius
hard removes the neighbours a relation needs.

`PLACEMENT_SPACING_TOL` and `PLACEMENT_SPACING_RADIUS` are in
`eth80_frontend._FRONTEND_SWITCHES`, so they are in the digest and in the cache
key. The tolerance is threaded as an optional argument through
`slots_for_candidate` / `plus_terms` / `plus_score`, defaulting to the frozen
constant, so every existing caller is unchanged and a dark run is byte-identical.

---

## 4. Step 3 — scene retrieval: the shortlist passes, the placement arm does not

**Shortlist (measured first, as the rule requires).** The mixed 24-scene gallery
(16 irregular + 8 out-of-plane) at K = 16:

| gallery | K | recall |
|---|---|---|
| 64 one-generator (Mission 9) | 8 | 0.875 |
| 24 mixed (Mission 9) | 8 / 16 | 0.938 / 1.000 |
| **24 mixed (this run)** | **16** | **1.000** over 16 queries |

Recall clears 0.95, so the arms are judged. 256 placement matches, 16 queries:

| arm | top-1 | MRR | top-1 given shortlisted |
|---|---|---|---|
| node-only | **0.750** | **0.797** | 0.750 |
| placement (specified: Σ over matched pairs) | 0.500 | 0.579 | 0.500 |
| placement − node-only | **−0.250** | — | SE 0.094, MDE 0.265 |

**Placement does not help recognition here, and the reason is in the aggregation,
not in the term.** The specified pair score is a SUM over matched pairs, and on
~900-node graphs that sum is a coverage statistic: the first query's placement
scores are 606 for the candidate that mapped 921 pairs and 546 for the one that
mapped 818, while the node-only sums sit within 1.7 % of each other across all
sixteen candidates (893–909). Dividing the placement score by the number of mapped
pairs — a post-hoc diagnostic, not the pre-registered definition — moves top-1 from
0.500 to **0.688**, still below node-only's 0.750.

So the honest reading is two-part. The pre-registered definition *punishes* the
placement arm for a reason that has nothing to do with placement quality: the
blended score halves each pair's contribution and compresses the range, so the
count dominates even more than it does for raw similarity. And once that is
corrected, placement is still not better than plain node similarity on
identifying a scene under an in-plane pose change — which is consistent with the
placement term being a *within-pair* signal: it decides which of two nearly
identical places is the right partner, not which of two scenes is the right scene.

---

## 5. Step 4b — ETH-80 view-to-view retrieval, 200 queries

Budget first, because it decides the design: a real pair costs ~11 s to build and
~13.4 s to match flag off / 15.2 s flag on (measured, held-out pair). 200 queries
× K = 8 is 1600 placement matches, about 3.4 h on two cores.

### The gallery had to be shrunk twice, and both times the measurement said so

A first design took Mission-8's 24 references (three reference instances per
class, at the reference view) as the gallery and drew queries from the dev and
held-out instances. That is not retrieval: those are **different objects**, so the
truth could never be in the shortlist, and the run returned recall@8 = **0.000**.
The bug was mine and the number was the tell — a perfect zero is a broken key, not
a hard task.

The corrected full gallery (80 instances, one reference view each) was measured
before any arm was scored: recall@8 = **0.235**, recall@16 = 0.36, recall@64 = 0.93
over 200 queries. The rule says shrink the gallery rather than judge below 0.95,
and the same measurement says how far: restricted to the ten instances of the
query's own class, recall@8 = 0.78, and at ten candidates it is 1.00.

So the task is fixed as **within-class instance identification**: the gallery is
the eight instances of the query's class at the reference view, K = 8 is that whole
gallery, and recall@8 = 1.000 **by construction** — stated as such, not claimed as
a measurement. A single 4-view query per instance gives 200 queries over 50
instances. What is measured is whether a *second view* of an object is ranked above
the other seven objects of its class by the placement matcher more often than by
plain node similarity.

Pair scores as fixed before the run: node-only = sum of the best per-node
similarities; placement = sum over the pairs the deployed matcher mapped of the
blended score it used. Scene = object instance.

> **Status: running, checkpointed per query** in `m10_eth80_progress.jsonl`
> (21/200 at the time of writing, ~33 s per query, so about 1.8 h to finish). The
> result and the verdict go here when the last query lands.

### Cost, measured

| stage | measured |
|---|---|
| one real image build | 6–11 s |
| 280 builds (64 references + 216 one-off) | 1381 s |
| shortlist, 200 queries × 8 candidates | 160 s |
| one placement match, real pair | 13.4 s flag off, 15.2 s flag on |
| per query (8 matches, two workers) | 35–90 s (80 s when a stale worker pair was alive) |
| 200 queries, end to end | ~3.3 h wall including one restart after a broken pool |

One cost bug was mine and worth recording: the first version asked
`similarity_rows` for a full query × reference matrix per candidate pair to read
**one** number, the similarity of the single mapped pair. The mapping is 1-to-1, so
that is one `similarity` call per pair — the same number, ~1.7 s cheaper per
candidate, ~14 s per query.

### The result

200 queries, K = 8 same-class gallery, both arms on the same shortlist, real
images only, nothing tuned on them (`m10_eth80_agg.json`):

| arm | top-1 | MRR | paired Δ vs node-only (per instance) |
|---|---|---|---|
| node-only similarity | **0.150** | 0.368 | — |
| placement, **pre-registered** (sum over mapped pairs) | **0.225** | 0.439 | +0.075, SE 0.068, **MDE 0.191**, CI [−0.055, +0.205] |
| placement, mean over mapped pairs (post-hoc) | **0.380** | 0.607 | **+0.230, SE 0.060, MDE 0.168, CI [+0.120, +0.345]** |

Read honestly:

* **The pre-registered score does not clear its MDE.** +0.075 against an MDE of
  0.191. On this data the deployment's own score, summed, is a coin flip away from
  no effect. That is the answer to the question as it was registered.
* **The mechanism is the one Step 3 already named.** The sum over correspondences
  is a coverage statistic: the number of mapped pairs varies from 800 to 990
  across candidates and swamps the per-pair quality. Dividing by the number of
  mapped pairs removes exactly that variation, and the effect that appears is
  +0.230 with a CI that excludes zero — five times the resolution, same data, same
  matches. This is post-hoc and is labelled as such in both the code and this
  table.
* **Node similarity is at the floor here.** 0.150 against 1/8 = 0.125 for a
  *same-class* gallery: on ETH-80 the front end cannot tell two apples apart by
  appearance, so a placement signal has room that synthetic circles never had.
* **Per class**, placement (mean) - node-only: car .50 - .03, dog .47 - .13,
  cow .38 - .16, horse .38 - .06, apple .34 - .28, cup .31 - .16. The exception is
  **pear (n = 8): .00 - .50** — one class, eight queries, in the wrong direction.
  It is small and it is not hidden.
* **The dark flags moved nothing.** `rescored` 879,744 (the spacing tolerance is
  consulted on every rescore), `raised` 0, `fell_back` 0, `skipped` 0: with the
  radius cap dark, no node in any of the 1600 matches was moved by it, which is
  exactly what a dark flag has to do.
* **Coverage on real images**: 907 of ~950 query nodes mapped per pair (95 %), and
  the spacing attachment is 18.0 % of nodes — the place layer attaches far less on
  real images than on synthetic ones, which is the first thing to chase if the
  mean-based score is ever to become the deployed one.

---

## 6. Hygiene

`PM.override()` is the only way a candidate bundle can reach the deployed path and
production never calls it; it validates through the same `from_jsonbundle` that
refuses a drifted cap. The two new spacing flags are dark and in the switch tuple.
The pre-registration was written at 0/140 graphs and committed before the graph
count moved.

## 7. The plain verdict

**Does placement help recognition? Yes — on synthetic scenes at the deployed
matcher, and on real images in retrieval — but not through the score that is
currently deployed, and not on every class.**

Where it helps, and by how much, measured once per set and never re-tuned:

| setting | what was measured | verdict |
|---|---|---|
| irregular FIT (7995 nodes, no AMBIGUOUS_MARGIN gate) | existence .5384 → geometry .5794 | **helps, +0.041**, and Step 1 showed +0.0850 of the deployed gain is presence alone |
| out of plane, 20 fresh scenes, ambiguous bank | .5655 vs .5434 | **helps, +0.0221**, CI [+.0096, +.0345], p .0025 |
| out of plane, unambiguous bank (no-harm) | +0.0108 | **no harm**, CI [+.0027, +.0189] |
| scene retrieval, 16 queries, sum score | .500 vs .750 node-only | **hurts**: a sum over correspondences is a coverage statistic |
| ETH-80 retrieval, 200 queries, sum score (pre-registered) | .225 vs .150 | direction right, **below its MDE** |
| ETH-80 retrieval, mean over correspondences (post-hoc) | .380 vs .150 | **helps, +0.230**, CI [+.120, +.345] |

So the honest one-liner: **the geometry term is real and it is worth about +0.02 to
+0.04 in matcher accuracy, and it only becomes a large effect on real images when
it is read as correspondence quality rather than as a count of correspondences.**
The deployed score adds the two together under one cap, which is why the deployed
score is the weak version of a signal that is measurably there. The single lever
that would convert this into deployed accuracy is the one Step 3 and this step both
pointed at: normalise the placement term by the pairs it actually judged, so that a
candidate is not rewarded for covering more nodes. That change is not made here —
it changes the deployed score, and this mission's rules put the deployed score
behind a pre-registered read.

Where it does **not** help: the pear class (n = 8) inverts, and the four DEV-5
families are flat-to-slightly-positive, not large — see below.

## 7b. The lever, in code and dark

The plain verdict names one lever, so it is shipped rather than just described:
`PLACEMENT_PAIR_NORMALISED` makes the placement term the mean over the slots the
pair actually judged (no padding with `mu_wrong`, denominator = judged slots) instead
of a sum over a cap of 24. It is in `_FRONTEND_SWITCHES` and in the matcher-result
cache key, and it defaults off.

It has **no read behind it**. The ETH-80 number it is aimed at (+0.230) came from
dividing the blended pair score after the fact, not from this term. On the one
out-of-plane view-pair I checked it against, it changed the term's value on true
pairs (0.5227 -> 0.5543) and moved **0 of 91** ambiguous nodes — so it is dark,
wired, and unproven. Believing it needs a pre-registered read on a fresh query set.

Two checks were run on the defaults while adding it, and both are the kind that
have to be run rather than assumed:

* **Fresh-process reproduction of the Step-2 read.** One recorded view-pair
  (`oop_m10_...`, arms geometry and presence) re-run in a clean process after the
  flag was added: per-node results **identical** to the checkpoint, 91 nodes each
  arm, accuracy 0.5934 / 0.5824 exactly as recorded.
* **A control that fell out of a mistake of mine.** The same re-run without
  `FE_PLACE_LAYER=1` gave geometry and presence the *same* accuracy, node for node
  (0.4638 both). Without the place layer there are no places, so the placement term
  is a no-op — which is what a non-vacuity control should look like, and a reminder
  that every one of these runs has to carry the flag.

## 8. The four families at the deployed matcher, on all FIT scenes

Mission 9 could only report **one** FIT scene for this regression, and one scene
says "no crash", not "no regression". This is all twelve FIT scenes of DEV-5 (the
families SPHERE, BOX, CYLINDER, OCCLUSION, the stratified split that puts every
family at every pose step on both sides), four pose steps each — 48 view-pairs,
1170 s of matching, report-only, nothing fitted, nothing decided
(`m10_dev5_fit.json`).

| family | ambiguous nodes | geometry | presence | Δ |
|---|---|---|---|---|
| BOX | 589 | **0.531** | 0.492 | **+0.039** |
| SPHERE | 1449 | **0.384** | 0.368 | +0.016 |
| CYLINDER | 868 | 0.484 | 0.482 | +0.002 |
| OCCLUSION | 1056 | 0.195 | 0.186 | +0.009 |
| **all FIT, scene-clustered** | 3962 | **0.3995** | 0.3817 | **+0.0178, SE 0.0058, MDE 0.0162** |

**No family regresses.** The unambiguous bank, which is the no-harm check, agrees
in direction on every family (BOX +.012, CYLINDER +.010, OCCLUSION +.007,
SPHERE +.005). The pooled +0.0178 just clears its MDE (2.8 × SE), so the honest
statement is "small, positive, and everywhere non-negative", not "a large gain".
OCCLUSION sits at 0.195 in both arms: occlusion is where the front end loses the
nodes, so there is little for a placement term to reorder.

Non-vacuity counters across all 96 matches: `rescored` 4704, `raised` 0,
`fell_back` 0, `skipped` 0 — the dark spacing flags ran their lookup and moved
nothing, as they must.
