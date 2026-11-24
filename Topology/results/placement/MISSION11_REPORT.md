# Mission 11 — put placement into the deployed pair score, and test recognition against the shipped matcher

Branch `arena/c316ca3a-percept6`, from `8225efc`. This report is written as the runs
land; the last section says what is still running and what it costs.

**The short version.** The deployed pair score (`JointMCS.score`) is computed after
alignment from raw node similarity and knows nothing about the placement evidence
that produced the correspondence — which is why every Mission-10 retrieval number had
to be a bespoke score. Step 1 puts one learned term inside the matcher, dark:
`score' = score_shipped + w · E`, where `E` is the mean placement credit over the
mapped pairs the frozen estimator could actually judge, shrunk by the number judged
so it cannot become a coverage statistic. `w` is fitted on FIT only. The
maximum-likelihood fit turns out to be **degenerate** — it climbs to its bound,
because the shipped score already ranks the FIT truth first on 11 of 14 queries, so
scaling the logits up sharpens them rather than reordering them. That is a
temperature being fitted, not a weight; the report says so and applies a declared
tie-break (FIT top-1, ties to the smaller weight) which lands on **w = 0.375**.

---

## 1. Step 1 — the term, the definition, and the weight

### The formula, written before the fit

For a pair (A, B) with the final mapping M:

```
judged pair   (u,v) in M with at least one slot slots_for_candidate(...) can check
pair credit   c(u,v) = plus_score(u,v) - 0.5          (the estimator's own [0,1] map)
evidence      E = sum over judged pairs of c  /  (n_judged + k),   k = 8
pair score    S' = S_shipped + w * E
```

* **centred so that no evidence is 0** — `plus_score` is centred at 0.5 by
  construction, so a pair with nothing to say contributes 0 and a negative E is
  real evidence against.
* **restricted to judged pairs** — a pair with no checkable slot contributes nothing
  and is not counted; `n_judged` and `share_judged = n_judged / |M|` are exposed on
  the result.
* **coverage-neutral** — the denominator is the number judged, not the number of
  mapped nodes. Mission 10 measured that a sum over correspondences is a coverage
  statistic (mapped pairs ran 606–921 across candidates with flat per-pair quality);
  that is exactly the failure this denominator exists to prevent.
* **shrunk** — `+ k` with k = 8 means one judged pair can move E by at most 1/9 of
  its own value.

Everything is dark: `PLACEMENT_SCORE_EVIDENCE` defaults off, and with it off `score`
is bit-for-bit the shipped number (`score_shipped` carries the same value).

### The weight, and the fact that maximum likelihood does not identify it

196 FIT pairs (14 FIT scenes × 14 candidates; irregular at in-plane 12°, out-of-plane
at y 20°). The ML objective is the retrieval likelihood the mission asks for:

```
w* = argmax_w  sum_q log softmax_c( S_qc + w * E_qc ) [truth]
```

| w | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 | 3 | 10 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| log-likelihood | −36.25 | −36.19 | −36.12 | −35.98 | −35.72 | −35.21 | −34.69 | −31.25 | −17.46 |
| FIT top-1 | 0.786 | 0.857 | 0.857 | **0.929** | 0.857 | 0.643 | 0.643 | 0.643 | 0.571 |

The likelihood is **monotone increasing in w** all the way to any bound. It has to
be: the shipped score already ranks the truth first on 11 of 14 FIT queries, so
multiplying the logits by anything makes those probabilities sharper and raises their
likelihood, while the queries it gets wrong are already wrong at w = 0. Maximum
likelihood is choosing a temperature, not a weight — it is unidentified at the
boundary in this model class, and saying otherwise would be a nicer story than the
data supports.

So the code applies a **declared tie-break**: when ML lands on the bound, take the
weight that maximises FIT top-1 on the same shortlists, ties to the smaller weight.
That is **w = 0.375** (top-1 0.929 against 0.786 at w = 0). Both numbers, the whole
likelihood curve and the top-1 curve are in `m11_fit_w.json` and
`matching/placement_score_params.json`, so the choice can be audited.

Scale check, because a weight is meaningless without one: on the FIT shortlists E has
SD 0.018 and the shipped score's candidate-to-candidate SD is 0.023 — the same order
of magnitude, which is why w = 0.375 is a refinement of the score rather than a
replacement for it.

### Defaults

Flag off, a recorded Mission-10 view-pair reproduces **per node** in both arms
(0.5934 / 0.5824, 91 ambiguous nodes each), and the three suites pass (17 + 12 + 17).
`PLACEMENT_SCORE_EVIDENCE`, `_SHRINK` and `_WEIGHT` are all in `_FRONTEND_SWITCHES`
and therefore in the switch digest every matcher-result cache key folds in.

---

## 2. Step 5 — what limits real-image coverage, measured link by link

Eight fresh ETH-80 apple pairs (reference view vs `-022-270`), one row per pair in
`m11_coverage.json`. The chain that produces a judged pair, link by link:

| link | measured |
|---|---|
| anchors | 100 % |
| anchor coherence ≥ 0.5 at the single 7×7 window | **40.6 %** (median coherence 0.423; **59.3 % of anchors are below the bar**) |
| nodes with ≥ 1 payload edge | **31.7 %** (10.9 incident edges per node) |
| nodes the aligner mapped | 79.6 % |
| **mapped pairs with ≥ 1 judged slot** | **42.6 %** |
| payload edges with both endpoints mapped | 88.9 % |
| place spacing on real images | 5.77 px (synthetic: 6.32 px) |

The limiting link is the **frame provider**, not the tolerance and not the radius:
59 % of real anchors never clear the coherence bar at the 7×7 window, so they never
get a payload edge, so their pairs can never be judged. The module already ships the
multi-scale provider for exactly this reason (`field_multiscale` /
`node_frames_multiscale`, smallest window that clears the bar).

### The fix, shipped dark

`PLACEMENT_FRAME_MULTISCALE` (dark, in `_FRONTEND_SWITCHES`) switches `build_edges`
onto that provider. Same eight pairs, flag off → on:

| quantity | off | on | Δ |
|---|---|---|---|
| nodes with a frame | 40.6 % | **50.4 %** | +9.8 pts (+24 % relative) |
| nodes with ≥ 1 payload edge | 31.7 % | **34.9 %** | **+3.1 pts** |
| incident edges per node | 10.9 | 13.4 | +2.5 |
| **mapped pairs judged** | 42.6 % | **44.9 %** | **+2.3 pts** |
| match time per pair | 24.0 s | 25.8 s | +7.6 % |

**The frame fix does not translate one-for-one, and that is the finding**: +9.8 points
of frames buy only +3.1 points of nodes with an edge, because the next limit is
target availability (`TARGET_K = 16` within `4 × scale`, and a node whose neighbours
all lack frames has nothing to point at), and the judged-slot definition after that.
The flag is dark and **uncredited** — it changes the payload geometry, so believing it
needs a fresh read, which this mission did not spend.

### The pear inversion, traced

Mission 10's pear inversion (.50 → .00 on eight queries) was measured with the
**bespoke** score — the mean over mapped pairs of `0.5·sim + 0.5·term`, whose term is
centred at 0.5 and therefore carries a ≥ 0.25 floor per pair. That score ranks
coverage. This mission's arm C is a different quantity, so the inversion had to be
re-measured rather than inherited: four pear queries and four apple queries, 8-way
same-class gallery, arms A and B and C on the same pairs (`m11_pear.json`).

| class | A | B | C | judged / mapped | share judged | E(truth) | E(wrong) |
|---|---|---|---|---|---|---|---|
| pear | 0.750 | 0.750 | 0.750 | 221 / 878 | **0.254** | −0.0556 | −0.0583 |
| apple | 0.250 | 0.250 | 0.250 | 328 / 813 | **0.410** | −0.0613 | −0.0596 |

Two findings, both of them about the code rather than about pears being hard:

1. **The inversion does not reproduce under the deployed score.** With arms A and C
   built on `JointMCS.score`, pear is the *better* class here (.750 vs apple .250).
   The old inversion was a property of the coverage-floored bespoke score, which is
   exactly why this mission stopped using it.
2. **Pear is the *under-covered* class**: 25.4 % of its mapped pairs are judged
   against 41.0 % for apple, because pear surfaces are smooth, fewer anchors clear the
   frame coherence bar, and fewer payload edges exist to judge. That is the same link
   Step 5 measures, and the multiscale frame provider is the fix aimed at it — dark,
   and worth +2.3 points of judged share in aggregate.

The honest limit: four queries per class. This says what the mechanism is; it does not
say pear is fixed.

---

## 3. Step 3 — the synthetic retrieval read (fresh, pre-registered, one read)

Registered in `m11_prereg_synth.json` before any of these graphs existed: 24 fresh
scenes (seeds 52000+/56000+), 32 queries at 15° in plane and 25° out of plane — poses
chosen so the shipped arm is **not** at ceiling (at 5–12° it already scores .75+, and
a null there would say nothing) — one shortlist for every arm, K = 8 with a declared
K = 16 fallback below recall 0.95.

**Shortlist first, as the rule requires**: node-similarity recall@8 = **0.969**
(in-plane 0.9375, out-of-plane 1.000) over 32 queries, so K stayed 8, and one query
has its truth outside the shortlist — scored as a miss, not dropped.

| arm | top-1 | MRR | margin | top-1 in plane | top-1 out of plane |
|---|---|---|---|---|---|
| **A** shipped (placement off, `JointMCS.score`) | 0.594 | 0.712 | +0.0017 | 0.750 | 0.438 |
| **B** mapping-only (placement on, score unchanged) | 0.688 | 0.792 | +0.0068 | 0.750 | 0.625 |
| **C** full (placement on, `score + w·E`, w = 0.375) | **0.750** | **0.839** | **+0.0176** | **0.8125** | **0.6875** |

Scene-clustered paired contrasts (16 scenes, 4000 draws):

| contrast | Δ top-1 | SE | **MDE** | CI | Δ margin (SE) |
|---|---|---|---|---|---|
| **C − A (primary)** | **+0.156** | 0.060 | 0.168 | **[+0.062, +0.281]** | **+0.0159** (0.0019) |
| B − A | +0.094 | 0.050 | 0.141 | [+0.000, +0.188] | +0.0051 (0.0012) |
| C − B | +0.062 | 0.043 | 0.120 | [+0.000, +0.156] | +0.0108 (0.0011) |

Read by the letter of the pre-registration: **the primary contrast does not clear its
own MDE.** +0.156 against MDE 0.168 — 7 % short. The CI excludes zero, the direction
is the one predicted, and the ordering A < B < C is the one predicted, but this read
does not have the resolution to call +0.156 a gain, and it is not called one. What it
does have is the margin: the paired truth-minus-best-wrong score gap rises
+0.0159 ± 0.0019 from A to C, eight standard errors from zero, and the same for C − B
at +0.0108 ± 0.0011. The effect is real in the score; the top-1 count over 32 queries
is what is underpowered (in-plane is where: .8125 vs .75 is one query).

Compute: 256 pairs × two arms, 12.6 s + 14.5 s per pair, ~1.5 h on two cores.

Coverage note, and it matters for what follows: on these synthetic pairs the estimator
could judge only **130 of 869 mapped pairs (15.0 %)** — the same limitation Step 5
measures on real images.

---

## 4. Step 4 — ETH-80, fresh queries: registered, shortlisted, **not finished**

Registered in `m11_prereg_eth.json` with **0 graphs built**: 200 queries over 64
instances at views Mission 10 never queried (`-022-270`, `-035-045`, `-045-180`,
`-090-045`; overlap with Mission 10's query set: **0**), the frozen w = 0.375, two
galleries (same-class K = 8 and all-class K = 16), arms A/B/C, primary contrast C − A
clustered by instance.

**The shortlist is measured** (56 min, 12 800 pairs): same-class K = 8 recall **1.000**
by construction (chance 1/8), all-class K = 16 recall **0.665** over the 64 references
(chance 1/16 given the truth is shortlisted). The all-class gallery is therefore a
*harder* task than it looks and its recall goes in the table, not in a footnote.

**The read is running and is not finished.** Cost, measured on this machine: 17.1 s
for arm A and 20.0 s for arm C per pair, 1600 pairs for the same-class gallery — about
8.2 h on two cores, and roughly as much again for the all-class gallery. At the time
of writing 200 of 1600 pairs are checkpointed, which is 25 complete queries over 7
instances: **A 0.360, B 0.280, C 0.360**, C − A = **0.000** (MDE 0.342), C − B = +0.071
(MDE 0.129). **That is not the read.** Twenty-five queries cannot resolve a 0.34 MDE,
and the only thing the partial says is that on real images the *mapping* alone (B) does
not help recognition, while the evidence term (C) gives back what B lost — the opposite
of the synthetic picture at the same N, which is exactly why the read has to finish
before anything is claimed. It is recorded so the resume point is exact, and the job is
checkpointed per pair and restarts where it stopped.

What the real-image side *does* have, measured and complete in this mission:

* the **coverage chain** (§2): 42.6 % of mapped pairs judged, 31.7 % of nodes carrying
  a payload edge, 40.6 % of anchors carrying a frame, and the frame provider named as
  the limiting link;
* the **fix** for that link, dark and measured (+2.3 points of judged pairs);
* the **pear mechanism** (§2), traced to coverage rather than to a wrong sign;
* a **fresh shortlist** whose recall is now known for both galleries.

## 5. The plain verdict

**Synthetic scenes: placement helps, and the honest size of the help is "just under
what this read could resolve".** On 32 fresh queries over a fresh 24-scene gallery,
the shipped matcher A scores 0.594 top-1, mapping-only B 0.688, and the full
placement-aware score C 0.750. The pre-registered primary contrast C − A is +0.156
with a CI of [+0.062, +0.281] and an MDE of 0.168: the CI excludes zero, the ordering
and the direction are the predicted ones, and the point estimate is 7 % **below** the
pre-registered MDE, so this mission does not call it a gain. The paired score margin —
a better-powered quantity than a 32-query top-1 count — rises +0.0159 ± 0.0019 from A
to C, eight standard errors from zero, so the effect is real in the score even where
the top-1 count is underpowered.

**Real images: unresolved.** The pre-registered read exists, is registered, has its
shortlist measured, and needs ~8 h of matcher per gallery; it is running and it is not
finished. The mission's DONE item 3 is therefore **not met**, and no real-image claim
is made — neither a gain nor a null. What is established on real images is narrower and
prior: the estimator can judge only 42.6 % of the pairs the aligner maps, that limit is
the frame provider and not the tolerance, and the multiscale provider recovers a third
of the gap (+9.8 points of frames, +2.3 points of judged pairs) without being credited
yet.

**Where it does not help:** pear is under-covered (25.4 % of mapped pairs judged
against 41.0 % for apple) and its Mission-10 inversion was a property of the bespoke
coverage-floored score, not of this one; the in-plane synthetic step is where the
top-1 count is weakest (.8125 vs .75 is one query); and the all-class ETH-80 gallery
shortlists the truth only 66.5 % of the time, so any read there is bounded by recall
before it starts.

## 6. What is running, and what it costs

| job | state | cost |
|---|---|---|
| ETH-80 same-class read (200 queries × 8 × 2 arms) | **running**, 96/1600 pairs checkpointed | ~8.2 h total |
| ETH-80 all-class read (K = 16) | not started | ~7 h more |
| `m11_eth80_agg.py` | rerun any time; counts complete queries only | seconds |

Every row is a `(query, reference)` checkpoint, so both stages resume rather than
recompute.
