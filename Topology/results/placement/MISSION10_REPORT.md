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
| per query (8 matches, two workers) | ~33 s |
| 200 queries | ~1.8 h |

---

## 6. Hygiene

`PM.override()` is the only way a candidate bundle can reach the deployed path and
production never calls it; it validates through the same `from_jsonbundle` that
refuses a drifted cap. The two new spacing flags are dark and in the switch tuple.
The pre-registration was written at 0/140 graphs and committed before the graph
count moved.

## 7. Still running, with the cost of finishing

1. ETH-80 — 280 builds (done) + shortlist + 1600 placement matches at ~15 s each;
2. the report's final verdict section, once ETH-80 has run.
