# Mission 8 — making placement help beyond presence

Branch `arena/c316ca3a-percept6`.  Every number below is on the FIT half unless it
is explicitly called the fresh read.  Arms live in `Topology/placement_m3.py` and
run on ONE graph per scene; nothing here changes what a deployed graph does, and
the defaults are byte-identical at both flag settings.

## 0. The free FIT reads

`Topology/placement_step0_m8.py`, `Topology/placement_step0b_m8.py`
(16 FIT scenes, places-only endpoints, 64 views).

| arm | FIT |
|---|---|
| N | 0.3145 |
| R | 0.2303 |
| R_fixed | 0.3354 |
| P_presence | 0.3532 |
| P | 0.3591 |
| P_ratio | 0.3440 |
| P_shuffled | 0.3041 |
| W | 0.3133 |
| **P_oracle_presence** | **0.3934** |
| **P_oracle** | **0.3961** |

Decomposition of the geometry headroom:

```
P_oracle - P_oracle_presence = +0.0027   what PERFECT CONTEXT buys
P_oracle_presence - P_presence = +0.0403 what perfect context CANNOT buy
```

That is the mission's central diagnostic: **geometry is not starved of context,
it is starved of information.**  Give the arm perfect neighbour images and the
residual term adds 0.0027 over an existence-only arm that also has perfect
context.  The gap from P_presence to the ceiling is 0.04, and almost all of it
is bought by fixing the neighbour *correspondences* the existence term reads —
not by reading their residuals more carefully.  Per family (N / P / P_presence /
P_oracle / P_oracle_presence): BOX .4918/.5302/.5294/.5596/.5466, CYLINDER
.4732/.5066/.4976/.5167/.5131, OCCLUSION .2359/.2505/.2558/.2883/.2883, SPHERE
.3733/.4026/.3944/.4241/.4241.

### Context audit — and a correction

The first audit compared a *query* node id against *reference* node ids, which
only agrees where the two graphs number their nodes the same way; its "0.027
correct" was an id-agreement rate.  The corrected audit asks the scorer's own
question (does the reference payload hold the edge between N's pick for `u` and
N's pick for the neighbour) over the 4760 neighbour edges P can actually use:

```
N's mapping correct on 7995 mapped nodes : 0.377
usable neighbour edge correspondence     : 0.517 correct / 0.483 wrong
residual |.| median   right 0.096  |  wrong 0.552   (means 0.149 / 0.561)
by family: BOX .594  CYLINDER .594  SPHERE .535  OCCLUSION .368
```

Wrong context does not merely add noise: it manufactures a residual 5.7x larger
than a correct context does, on the very candidates the residual is supposed to
certify.  At 0.483 the confident-context filter is *optional* by the >50% rule,
and it is kept anyway — the rival-count refusal never relaxes.

## 1. Front end: one position per segment (frozen)

Orientation had 30 distinct positions for 45 nodes: every segment of a
multi-segment place shared the region centroid.  The polyline walk already
computed a midpoint at half of each segment's own arc and discarded it; it is
now keyed by `(entity, segment)` and handed to the `Detection` as `_place_pos`
before the region anchor is consulted.

| FIT node-to-node, place graph, 6 scenes | before | after |
|---|---|---|
| w1_rot repeat / identity | 0.297 / 0.644 | 0.297 / 0.652 |
| w2_translate repeat / identity | 0.409 / 0.603 | 0.417 / 0.609 |
| w3_scale repeat / identity | 0.417 / 0.721 | 0.421 / 0.727 |
| Orientation repeat / identity | 0.209 / 0.722 | **0.266 / 0.873** |
| node-weighted identity | 0.639 | 0.647 |
| distinct Orientation positions | 30/45 | **45/45** |

Localisation medians move by at most 0.04 px; every other lens is identical.
Graph size unchanged (913 nodes on the probe scene).  Flag off is byte-identical
across the edit (`c0ab7a1b670d0540` both sides) and flag on is deterministic in
two fresh processes (`65ae879ac7f21f82` twice).  Suites 17+17+12 OK.
**The front end is frozen** (`m8_frontend_freeze.json`).

## 2. P_plus, the context iteration, and one FIT choice

**P_plus** = existence term + geometry term in one estimator: P's slots, P's
`SLOT_CAP`, P's padding, P's `+-LLR_CLIP`, P's `[0,1]` map, with two
coefficients fitted by multinomial maximum likelihood on FIT
(`matching/placement_llr.fit_plus_coefficients`) and then frozen:
`a_exist = 0.40625`, `a_geom = 0.84375` (5361 FIT nodes; identical on the rerun).
`P_plus_presence` is the same machine with `a_geom = 0`.

**Context iteration** (`matching/placement_context.py`): five fixed synchronous
sweeps from N's mapping; a node moves only when the new choice beats its
incumbent by `MARGIN` and is not inside a near-tie (at most one rival within
`RIVAL_TOL = 0.02`); everything else keeps the mapping it came in with.
Thresholds chosen on FIT *context accuracy*, not on arm accuracy:
k = 1 .4304, 2 .4452, 3 .4524, 5 .4550, 8 .4553; margin 0.00 .4398 > 0.01 .4370
> 0.02 .4301.  The mutual-agreement test is OFF because with six candidates it
freezes the mapping (.3998 against .4550) — reported rather than quietly dropped.

FIT table (16 scenes, 64 views, scene-clustered):

| arm | FIT |
|---|---|
| N | 0.3145 |
| P_presence | 0.3532 |
| P | 0.3591 |
| P_plus_presence | 0.3483 |
| P_plus | 0.3679 |
| P_plus_ctx_presence | 0.3583 |
| **P_plus_ctx (chosen)** | **0.3818** |
| P_shuffled | 0.3041 |
| P_oracle | 0.3961 |

| contrast | effect | SE | MDE (2.8·SE) |
|---|---|---|---|
| P_plus_ctx − P_plus_ctx_presence | **+0.0235** | 0.0045 | 0.0126 |
| P_plus_ctx − N | **+0.0701** | 0.0083 | 0.0234 |
| P_plus − P_plus_presence | +0.0191 | 0.0038 | 0.0107 |
| P_plus − N | +0.0568 | 0.0069 | 0.0193 |
| P_plus_ctx − P_plus | +0.0132 | 0.0027 | 0.0076 |
| **P − P_presence (old arm)** | **+0.0063** | 0.0039 | 0.0109 |

Context accuracy on the scored population 0.389 → 0.430 across 64 views, 3 views
worse; 581 moves accepted of 1183 offered.  The undamped control (re-pick
everything, no filter) reaches .4522 on the subsample against the filtered .4394 —
better, and still refused: it is the mechanism that cannot tell a good contextual
move from a bad one, and it is exactly what would amplify a wrong majority.

Unambiguous bank: P_plus_ctx − P_plus = +0.0011 (MDE 0.0015) — the iteration does
not trade away the easy nodes to buy the hard ones.

**Step 2c** — per-edge tolerance from per-place uncertainty: |residual| does not
track the uncertainty proxy (spearman 0.040, n = 416), so no per-edge tolerance is
fitted.  The single frozen `FIXED_TOL` stands.

## 3. Pre-registration (committed before any fresh graph existed)

`m8_preregistration.json` fixes the front-end digest and file hashes, every
switch, the endpoint whitelist, the two coefficients, `FIXED_TOL`, `SLOT_CAP`,
`LLR_CLIP`, `CTX_ITERS/MARGIN/MUTUAL`, `EDGE_BETA`, `CAND_CAP`, the ambiguity
margin, the 16 confirmation scenes, the arms, the two primary contrasts, Holm on
two, the decision rule and the MDE.  The script refuses to write while any graph
of the confirmation set is cached; it reported 0 of 80 present.

Confirmation set: DEV-5 ranks `i = 8..11` of the same generator, all four
families, 16 scenes / 80 graphs.  No scene of this set was fitted, scored or
looked at in Missions 5–8.  (The mission caps the build at 128 scenes / 512
graphs; this is 16 / 80, and it resolves MDE 0.0126 and 0.0234.)

## 4. The fresh read — one look

`m8_read.json`, 16 scenes, 64 views, coordinates and thresholds asserted against
the pre-registration before scoring (all ten checks passed), densities loaded
from the FIT pickle rather than refitted.

| arm | fresh read |
|---|---|
| N | 0.3183 |
| P_presence | 0.3589 |
| P | 0.3683 |
| P_plus_presence | 0.3573 |
| P_plus | 0.3751 |
| P_plus_ctx_presence | 0.3688 |
| **P_plus_ctx** | **0.3874** |
| P_shuffled | 0.3068 |
| P_oracle | 0.3991 |
| P_oracle_presence | 0.3975 |

| primary contrast | effect | 95% CI | p (x2, Holm) | MDE | verdict |
|---|---|---|---|---|---|
| P_plus_ctx − P_plus_ctx_presence | **+0.0190** | [+0.0128, +0.0252] | 0.00012 | 0.0126 | beyond MDE |
| P_plus_ctx − N | **+0.0672** | [+0.0560, +0.0785] | 0.00006 | 0.0234 | beyond MDE |

**VERDICT: CREDITED.**  Both pre-registered primaries clear their MDE in the
pre-registered direction, scene-clustered, with the two-contrast correction
applied.  Both effects reproduce the FIT estimates within a fifth of their own
size (+0.0190 vs +0.0235; +0.0672 vs +0.0701).

Secondaries (uncorrected): P_plus − P_plus_presence +0.0174 (MDE 0.0067),
P_plus − P +0.0065 (MDE 0.0045), P_plus_ctx − P_plus +0.0113 (MDE 0.0083),
P_plus_ctx − P_plus_presence +0.0287, **P − P_presence +0.0094 (MDE 0.0110) —
the old standalone geometry arm still fails**, which is the whole reason the
mission asked for a new machine.

Per family (N / P / P_plus / P_plus_ctx / P_oracle):
BOX .4555/.5178/.5237/.5326/.5490 · CYLINDER .4197/.4570/.4658/.4707/.4745 ·
OCCLUSION .2167/.2515/.2561/.2742/.2970 · SPHERE .2956/.3582/.3662/.3787/.3838.
Per view (N / P_plus / P_plus_ctx): rot5 .3544/.4245/.4371 ·
translate .3738/.4332/.4414 · rot15 .2801/.3365/.3530 · scale .2522/.2911/.3031.
Similarity bands are flat until the top band (top_sim ≥ 0.85, n = 3757):
N .351 / P .410 / P_plus .418 / ctx .432 — the gain is where the candidates
actually compete, not in the long tail.  Net flips over ambiguous nodes:
P_plus_ctx vs N +337/−22 (net +315), vs P_plus +73/−17 (net +56),
P_plus vs P_plus_presence +110/−29 (net +81).  Unambiguous bank:
N .4693 / P_plus .4735 / P_plus_ctx .4762, and P_plus_ctx − N = +0.0077
(MDE 0.0067).

## 5. Real images (ETH-80, non-gating)

`m8_real_on.json`, `m8_real_off.json` — 16 real photographs, one per class pair
(reference + query instance, front view).

| flags | nodes | edges | place-lens nodes | build | attach @4 px | median |
|---|---|---|---|---|---|---|
| off | 725.8 | 2220.6 | 194.4 | 15.3 s | 0.311 | 7.00 px |
| **on** | **969.2** | **2323.4** | **437.6** | **17.1 s** | **0.439** | **5.00 px** |

Nothing collapses: 22 lenses present at both settings, the rarest being Junction
(3), Y (3), T (4), Onset (16) — genuinely rare structure on 256² photographs.
The place layer adds a third more nodes on real images, as it does on synthetic
ones (+39%), at +12% build time.  Attachment coverage is lower than on the
synthetic lattice (0.439 against 0.735) and the median nearest place is 5 px
against 2.13 px — real photographs place places further apart, which is worth
saying rather than hiding.  Retrieval sanity (one query per class, six candidate
references): top-1 same-object 0.750 for N and 0.750 for the chosen variant — no
regression, no gain, which is all a sanity check may claim.

## 6. A cache-key hole found on the way (fixed)

`FE_PLACE_LAYER` was missing from `eth80_frontend._FRONTEND_SWITCHES`, so the
graph cache did not key on it.  It was caught by the step-5 flags-off run
returning the flags-on graphs: identical node and edge counts, 0.0 s of build
time.  Fixed by adding the switch; verified content-neutral on
`apple1-090-000` (pre-fix cache entry and post-fix fresh build both give 904
nodes / 1946 edges / hash `3d9853b6ca8eed2c`; only the key moved, 0.1 s →
17.0 s).  The fresh read was taken with the pre-fix digest recorded in the
freeze file; the fix changes no graph, so the read stands.  Flag fingerprints
after the fix: off `c0ab7a1b670d0540`, on `65ae879ac7f21f82`.

## 7. What is now true, and what is still not

Credited: placement now beats presence by +0.019 on a fresh, pre-registered set,
and beats similarity by +0.067, with the deployed variant `P_plus_ctx` chosen on
FIT only.  The oracle decomposition says why this was hard and where the rest is:
perfect context is worth +0.0027 to the residual term and +0.040 to the existence
term, so the remaining headroom (P_oracle − P_plus_ctx ≈ +0.012 on the fresh
read) is context, not geometry.

The largest remaining error is the *unfixed* context: 51.7% of the neighbour
edges the scorer uses are still wrong, the confident filter only moves 581 of
1183 offered nodes, and the arms' ceiling under truth is 0.399.  The next
mechanical fix is not a better residual — it is a context estimator that decides
*which* neighbour correspondences to trust before the residual is computed, and
the obvious lever is the mutual-consistency test that had to be switched off
here because a six-candidate ballot makes it vacuous.

## 8. Hygiene

Defaults byte-identical (flag off `c0ab7a1b670d0540`, flag on
`65ae879ac7f21f82`, both after the cache fix).  Suites 17 + 17 + 12 OK, at both
flag settings.  Determinism checked in two fresh processes.  Truth is used to
label FIT pools and to score; no arm reads truth at match time, and the oracle
arms are DEV-only and excluded from every contrast in the read.
