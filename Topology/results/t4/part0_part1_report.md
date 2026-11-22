# Task 4 retrieval — Part 0 corrections and Part 1 offline re-analysis

Branch `arena/c316ca3a-percept6`. Logging run: 128 queries × 8 refs × 3 arms,
6778 s, 3072 records (`retrieval_components.jsonl`).

**Every number below is at:** beta = 0.50 (committed; 0.75 is the
TUNING_REPORT winner and is **not** what these runs use), ARM_B1
(`MATCH_SYSTEMATICITY_WEIGHT = 0.5`, `BINDER_COMPLETE_POSITION_MAP = True`,
L3/L4 **ON**), `MATCH_EDGE_WEIGHT` w = 0.10, pooled normalization,
`MATCH_EDGE_DENOMINATOR_MODE = "checkable"`,
`MATCH_COVERAGE_EXPONENT_GAMMA = 0.5`. n = 128 queries (8 objects × 16 views),
n_eff = 8 objects. Intervals are two-level clustered (objects, then views),
2000 bootstrap replicates, MDE = 2.8 × SE.

**Statistics are tie-aware** (pre-registered before results were inspected).
For a query whose same-object ref has score s, with k refs strictly above s
and t refs tied at s: top-1 = 1/t if k = 0 else 0; MRR = (1/t)·Σ 1/(k+j);
AUC = ½ on equality. With no straddling tie these reduce to the usual values.

---

## Part 0 — corrections

### 0a CASE ACCURACY / D1 are withdrawn
`t4_retrieval_measure.py` built its "correspondence" with

```python
counterpart_map(q["pos"], r["pos"], 0, 0, 8.0)   # dx=0, dy=0, radius=8.0 px
```

That is a **same-pixel** proximity test, not a physical correspondence: it
asks whether the query node and the ref node land within 8 px of the *same
image coordinate* at two *different viewpoints*.

Measured with the repaired one-to-one `counterpart_map` (mutual pairs only):
**21.1 % of query nodes acquire a "counterpart" this way** (268/1270 on four
objects; per object apple 18.2 %, car 22.2 %, cup 14.7 %, dog 27.9 %). The
pre-filter rate is higher still (e.g. apple 93 candidate pairs collapsing to
66 mutual). Every number derived from it — CASE ACCURACY, D1, the claim that
"correspondence is the bottleneck", and the B1 case-(c) +0.367 citation — is
**withdrawn**. Nothing in Part 1 depends on it.

Note that the *function* is not at fault: `t4_shift.py::counterpart_map` was
independently repaired to a one-to-one mutual-nearest map with `(mapping, info,
stats)`, and the caller unpacks three values. The defect is the **arguments** —
`dx=dy=0` asserts a pure translation between two different viewpoints, which is
a claim about the geometry, not a measured correspondence.

### 0b Intervals re-derived; both properties are INCONCLUSIVE
Recomputed with object clustering (n_eff = 8), MRR primary. The starting
pairwise top-1 figure `R − W = +0.0469 [+0.0078, +0.0938]` **does not
reproduce**: under object clustering it neither excludes zero nor clears its
own MDE. Reporting the two properties separately, as required:

| property | contrast | MRR diff | 95 % CI | MDE | verdict |
|---|---|---|---|---|---|
| CONSISTENCY | R − W | +0.0193 | [−0.0015, +0.0487] | 0.0359 | INCONCLUSIVE |
| ADDITIVE BENEFIT | R − N | +0.0249 | [−0.0623, +0.1074] | 0.1212 | INCONCLUSIVE |

Neither interval excludes effects above its MDE, and neither excludes zero.
The correct statement is **INCONCLUSIVE at n_eff = 8**, not "no effect".

The CONSISTENCY contrast is the tightest thing in this study (MDE 0.036,
about 1.9 pp of MRR): R is above W in 5/8 objects strictly and 8/8 weakly, yet
the interval still contains zero.

### 0c Provenance
beta 0.50 is **committed**; 0.75 is the TUNING_REPORT winner and was **not**
used. ARM_B1 has L3/L4 **ON**; the shipped configuration has L3/L4 **OFF**
(that is Part 1g, not run here — see "what did not run"). Each table in this
document names beta, arm, edge weight and normalization in its header.

### 0d C1 relabelled
N's node-id-relabel invariance (n = 16, identical under seeds 42 and 7) is
**trivial**: N has no edges, so there is nothing for an edge claim to be
sensitive to. The informative number is R's **3.06 % mapping sensitivity**.

### 0e C2 stays "not object-discriminative" — now measured (1f)
Prior C2 compared same-object vs different-class pairs and could not separate
object identity from class identity. 1f (below) supplies the missing
comparison. beta(same-object) = 0.0235, CI [0.0031, 0.0795], MDE 0.0546 →
INCONCLUSIVE, and only ~10 % of R's advantage over N is object-specific. C2
therefore **does not** become object-discriminative.

### 0f W construction — two defects, one fatal (corrected)
The pre-registered claim was that W strips parameters and so has
`edge_credit ≈ 0`, making 1b vacuous. **That premise is wrong, and the real
defect is worse.** Measured on the actual logged W arm: `edge_credit` mean
**122.7**, not ≈ 0, with a non-zero `matched_edges` in 1024/1024 records.
Stripping params does not zero edge credit — it converts graded credit into
*presence-only topological* credit, because a relation with no measured axes
still counts as evidence that the relation exists. W is therefore a
"rewired topology, no physics" arm, which is a usable control for topology
but **not** a null for edge evidence.

Separately, and fatally for 1b as first run, the W-P builder
(`_rewired_graph_preserve_params`) read parameters from `g.edge_params`, but
the parameters live on `g.subgraph.edge_params`. It silently found none, and
**W-P came out byte-identical to W** (identical edge sets and both with 0
params) — the parameter-preserving control did not exist. Fixed by reading the
subgraph first:

```python
ep = (getattr(sg, "edge_params", None)
      or getattr(g, "edge_params", None) or {})
```

After the fix W-P carries 248/346/274 params (apple/cup/horse refs) on exactly
W's edge set. The rewiring is degree-preserving for 97–100 % of nodes and
preserves the edge-type multiset for some objects but not others (apple, cup,
dog: False), so "degree-matched" is approximately, not exactly, true.

### Cross-check against the other task-4 lineage on this branch
The branch also carries an earlier task-4 mission (`t4_mission_report.md`,
commits `9bbcc69`–`18961ed`) that measured *placement* evidence on dev scenes
with the repaired map at per-kind radii (CONTOUR_PEAK 1.0, Corner 2.06,
PEAK_CHAIN 1.54, RegionShape 1.73, …). Two of its results bear directly on the
paragraph above, so they are reconciled here rather than left to disagree:

- **Parameters are not inert, and the null is not "unmeasured".** That mission
  measured, on n = 6241 STRICT-LEAD pairs, `real − params_shuffled` =
  +0.0016 [−0.0013, +0.0043] but `real − params_dropped` =
  **+0.0072 [+0.0026, +0.0130]**. Parameters therefore carry a **small but real**
  net signal of about 0.7 %, and shuffling them is inert. My 1b
  `W-P − W = −0.0044 [−0.0757, +0.0658]` and
  `R − W-P = +0.0267 [−0.0858, +0.1464]` are both consistent with that small
  positive value but, at n_eff = 8 refs, cannot resolve 0.7 % — a ~7× shortfall
  in MDE.
  My 1b is therefore **underpowered for the effect that is actually there**; its
  intervals must be read as *unresolved*, **not** as "topology is irrelevant".
  That mission had already withdrawn the historical blanket claim "parameters
  non-informative".
- **Edges net harm the pooled objective, and "edges hurt" is not jitter-robust.**
  On dev, `real − no_edge` pooled = **−0.0172 [−0.0214, −0.0131]** at
  beta = 0.50, worsening monotonically with beta (−0.0016 at 0.25, −0.0348 at
  0.75). But that −0.0172 lies **inside** the tie-order jitter envelope
  [−0.0645, +0.0204], so the sign of "edges hurt" is not robust to tie order.
  The same mission measured tie order as material (**3.06 %** mapping change),
  which agrees with 0d.

Read together: this branch's best-powered evidence is that **parameter content
carries a small real identity signal (+0.7 %), the topology swap alone does not
expose it, and the pooled benefit of edges at beta 0.50 is slightly negative**
— none of which changes the Part 3 verdict below, because all of these effects
are far inside the MDE available at 8 objects.

### 0g The rewired arms were not deterministic across processes (fixed)

While re-running the analysis for reproducibility, the W-P edge count for one
fixed ref changed between runs: **351, then 350**, with different edge sets.
Cause: `_rewire_edges` and `_rewired_graph_preserve_params` both derived their
input list from `sg.edges`, which is a `frozenset`. A `frozenset`'s iteration
order depends on string hashing, which Python **salts per process**, so
feeding it to a seeded `random.Random` produced a different rewiring each
process. The `seed` argument was therefore not doing what it appeared to do.
This affects the *shipped* W arm too, not just my W-P control.

Fix (both builders): sort the edge list canonically before shuffling —

```python
edges = sorted(sg.edges, key=lambda e: (e[0], e[1], str(e[2])))
```

After the fix the same ref yields the same hash (`fc23b2b03a16`) in three
consecutive processes, and two consecutive full analysis runs produce
**byte-identical output**.

**This changed a 1b result.** Before the fix, R − W-P on edge-only MRR was
−0.0134; after it, **+0.0267**, with all three arms' point estimates moving.
The pre-fix numbers were one draw from a distribution nobody had controlled,
and the post-fix numbers are reproducible. All 1b figures quoted in this
report are post-fix. The 1a, 1d, 1e and 1f figures are unaffected (they never
touch the rewired arms) and were verified to reproduce exactly.

### Two further findings recorded here (both measured, not assumed)
1. **`MATCH_SEPARATE_TERM_NORMALIZATION` is inert under the committed
   configuration.** `separate` is read only inside the
   `mode == "all" and gamma == 1.0` branch. With the committed
   `MATCH_EDGE_DENOMINATOR_MODE = "checkable"` and
   `MATCH_COVERAGE_EXPONENT_GAMMA = 0.5` the flag has **no effect on the
   score**: 0.561222224194537 with the flag off and on, bit-identical. Under
   `mode = "all", gamma = 1.0` it does bite (0.2538 → 0.5043). An authorised
   flag that cannot move the shipped number is not a usable variant.
2. **The matcher's `edge_term` under "checkable" is two-sided.** The matcher
   accumulates `edge_credit_total = edge_credit + b_credit` (both directions)
   and divides by `(2·min(checkable_a − pres_a, checkable_b − pres_b))^0.5`
   because gamma = 0.5 is applied *after* the mass. A one-sided replica does
   not reproduce it. The replica used for 1b now matches the matcher's own
   `edge_credit` and `edge_term` to < 1e-9 on 4/4 sampled pairs.

---

## Part 1 — offline re-analysis

### 1a Clustered re-statement (MRR primary)

| arm | MRR | 95 % CI | MDE | top-1 | AUC |
|---|---|---|---|---|---|
| **R** | **0.5934** | [0.4215, 0.7813] | 0.2570 | 0.3906 [0.1250, 0.6719] | 0.7679 [0.6417, 0.8907] |
| **N** | **0.5685** | [0.4139, 0.7342] | 0.2288 | 0.3672 [0.1639, 0.5859] | 0.7522 [0.6205, 0.8761] |
| **W** | **0.5741** | [0.3984, 0.7647] | 0.2616 | 0.3750 [0.1172, 0.6562] | 0.7500 [0.6194, 0.8761] |

Paired, object-clustered, on MRR:

| contrast | diff | 95 % CI | MDE | verdict |
|---|---|---|---|---|
| R − N | +0.0249 | [−0.0623, +0.1074] | 0.1212 | INCONCLUSIVE |
| R − W | +0.0193 | [−0.0015, +0.0487] | 0.0359 | INCONCLUSIVE |
| N − W | −0.0057 | [−0.0939, +0.0836] | 0.1267 | INCONCLUSIVE |

Per-object MRR (n = 16 views each):

| object | R | N | W |
|---|---|---|---|
| apple | 0.3295 | 0.4656 | 0.3161 |
| car | 0.4500 | 0.3240 | 0.4073 |
| cow | 0.4479 | 0.3302 | 0.4479 |
| cup | 1.0000 | 0.9688 | 1.0000 |
| dog | 1.0000 | 0.8438 | 1.0000 |
| horse | 0.3683 | 0.5469 | 0.3302 |
| pear | 0.7083 | 0.6771 | 0.6531 |
| tomato | 0.4429 | 0.3914 | 0.4382 |

R > W strictly in 5/8 objects, weakly in 8/8. R > N strictly in 6/8. The
per-object table shows why the intervals are wide: cup and dog are at ceiling
(1.0 for R), and the sign of R − N flips on apple and horse.

### 1b Edge-only ranking under N's fixed mapping

The mapping is taken from the **N** arm, whose graphs have no edges — so the
mapping was formed without ever seeing edge evidence. Edge-only score =
`edge_term` computed under that fixed mapping. Chance top-1 = 1/8 = 0.125.

| arm | edge-only top-1 | 95 % CI | MDE | AUC | mean edge_credit |
|---|---|---|---|---|---|
| R | 0.2266 | [0.1016, 0.3594] | 0.1842 | 0.5915 [0.4553, 0.7221] | 24.688 |
| W | 0.1875 | [0.0938, 0.3047] | 0.1507 | 0.5932 [0.4966, 0.6892] | 17.777 |
| W-P | 0.1797 | [0.1094, 0.2578] | 0.1060 | 0.5993 [0.5324, 0.6641] | 10.565 |

Against chance: **no arm is above it** in top-1 — R low end 0.1016, W 0.0938,
W-P 0.1094, all below 0.125. On AUC all three are above 0.5 with intervals
clear of it (0.59–0.60), and W-P has the **tightest** interval
(0.5993 [0.5324, 0.6641], MDE 0.094).

The arm-to-arm contrasts all have favourable point estimates, and none is
resolved:

| contrast | on MRR | 95 % CI | MDE |
|---|---|---|---|
| R − W | **+0.0223** | [−0.0755, +0.1301] | 0.1469 |
| R − W-P | **+0.0267** | [−0.0858, +0.1464] | 0.1658 |
| W-P − W | −0.0044 | [−0.0757, +0.0658] | 0.1011 |

On top-1 the same ordering holds: R − W = +0.0391 [−0.0861, +0.1719],
R − W-P = +0.0469 [−0.0938, +0.1875].

**Reading.** The point estimates now have R **ahead** of both controls on
edge-only MRR (R − W = +0.0223, R − W-P = +0.0267) and on top-1
(+0.0391, +0.0469); before the determinism fix the same statistics came out
negative. Every interval still contains zero, and every MDE (0.10–0.18) is
several times the estimate, so **nothing here is established** — but the
direction is now consistently favourable to true topology, and it is
consistent with the +0.7 % parameter effect that the better-powered
cross-check measured.

R is nevertheless **not above chance** on edge-only top-1 (0.2266, low end
0.1016 < 0.125). The pre-registered "edges carry identity alone" condition —
R above chance **and** above W-P — therefore **fails on the first limb**, so
F-G does not fire and no gated policy is proposed. All three arms carry real
signal on AUC (0.59–0.60, intervals clear of 0.5), so edges are not
uninformative; they are simply not informative *enough* to beat chance on
top-1 at this mapping.

Note that the arms are ranked on quantities of different magnitude — mean
edge credit R 24.7, W 17.8, W-P 10.6 — so a higher AUC for W-P (0.5993) is
relative to its own distribution, not evidence of stronger absolute credit.

### 1c Density bias (measured fresh)

| arm | ref mean score vs ref edge count | 95 % CI (ref bootstrap) | query mean score vs query edge count |
|---|---|---|---|
| R | **+0.5048** | [−0.3370, +0.9394] | +0.2354 |
| N | undefined | — | undefined |

N has no edges at all, so its edge counts are identically 0 and the
correlation is **undefined, not zero**.

R's ref-level correlation is +0.5048, crossing the pre-registered |corr| ≥ 0.5
threshold — so the **F-S trigger fires**. With only 8 refs, though, the
interval is [−0.337, +0.939]: the point estimate is too poorly determined to
carry the trigger on its own. (Ref edge counts span 351–512, a 1.46× spread;
query counts span 251–512.) See 1d for what happens when the variant it
triggers is actually evaluated.

### 1d Score variants

V0 = the shipped pooled score. V1/V2 = separate-term recombination of the
**logged** terms at w = 0.3 / 0.1:

```
V = 2·((1 − w)·node_term + w·edge_term)
```

These are recombinations of logged quantities, not runs of the matcher's
`MATCH_SEPARATE_TERM_NORMALIZATION` path — which, as established in 0f, is
inert under the committed denominator mode.

**Arm R, MRR:**

| variant | MRR | 95 % CI | MDE |
|---|---|---|---|
| V0 (shipped, w = 0.1) | **0.5934** | [0.4215, 0.7813] | 0.2570 |
| V1 (separate, w = 0.3) | 0.4209 | [0.3215, 0.5338] | 0.1517 |
| V2 (separate, w = 0.1) | 0.4465 | [0.3427, 0.5612] | 0.1560 |
| edge-only anchor | 0.4192 | [0.3201, 0.5328] | 0.1519 |
| node-only anchor | 0.5582 | [0.3876, 0.7515] | 0.2600 |

**Arm N:** V0 = V1 = V2 = node-only = 0.5685 (all identical — N's
`edge_term` is 0, so every recombination collapses onto the node term).
N's edge-only MRR is **exactly 0.33973 = H₈/₈ = Σ(1/k)/8**, the value of an
8-way tie: with no edges every ref scores 0. This is a useful internal check
that the tie-aware statistic is doing what it claims.

**R − N per variant (MRR, object-clustered):**

| variant | diff | 95 % CI | MDE | objects R > N | verdict |
|---|---|---|---|---|---|
| V0 | +0.0249 | [−0.0623, +0.1074] | 0.1212 | 6/8 | INCONCLUSIVE |
| V1 | −0.1475 | [−0.3232, +0.0230] | 0.2473 | 2/8 | INCONCLUSIVE |
| V2 | −0.1220 | [−0.2880, +0.0421] | 0.2358 | 2/8 | INCONCLUSIVE |
| edge-only | +0.0794 | [−0.0196, +0.1931] | 0.1519 | 1/8 | INCONCLUSIVE |
| node-only | −0.0102 | [−0.1093, +0.0830] | 0.1373 | 5/8 | INCONCLUSIVE |

Two things to note. First, **every authorised variant is worse for R than the
shipped V0** (0.42–0.45 vs 0.59). Recombining terms on their own masses
reweights the score drastically, because gamma = 0.5 inflates `edge_term`
(mean 6.84) relative to `node_term` (mean 0.243); V1/V2 therefore largely
discard the node evidence. Second, the edge-only "positive" mean (+0.0794) is
an artifact of N's uniform tie being broken: R wins on mean while losing on
7/8 objects, because N's tie is worth the constant 0.3397 while R's broken
ties are worth 1.0 on one eighth of queries and less on the rest. A mean that
disagrees with 7/8 object signs is not evidence.

### 1e Edge-gated use (fixed rule, no fitting)

N margin = top-1 minus top-2 N score. Cut = lowest tercile of the N margins
on the 8 rank-0 objects, frozen: **cut = 0.004099** (n = 128, 43 queries
below). Margins: min 0.0000, p33 0.0041, p67 0.0119, max 0.0582.

| policy | MRR | 95 % CI | MDE | top-1 |
|---|---|---|---|---|
| Gated (V1 on low-margin, else N) | 0.5618 | [0.4220, 0.7087] | 0.2048 | 0.3672 |
| N throughout | 0.5685 | [0.4139, 0.7342] | 0.2288 | 0.3672 |
| V1(R) throughout | 0.4209 | [0.3215, 0.5338] | 0.1517 | — |

| contrast | diff | 95 % CI | MDE | verdict |
|---|---|---|---|---|
| Gated − N (MRR) | −0.0066 | [−0.0651, +0.0506] | 0.0827 | INCONCLUSIVE |
| Gated − V1(R) (MRR) | +0.1409 | [+0.0074, +0.2795] | 0.1944 | INCONCLUSIVE |

**P(N wrong) by N-margin tercile:** shrunk **0.814**, middle **0.738**,
wide **0.349**. P(N wrong) does rise as the margin shrinks — the diagnostic
that F-G names — and it rises steeply.

The gated *policy* nevertheless delivers nothing: its top-1 is **identical**
to N's (0.3672) and its MRR is marginally lower. The gating rule helps
relative to using V1 everywhere (+0.1409) but does not beat just using N. The
diagnostic is real; the intervention built on it is not.

### 1f Same-object gap (two-way clustered, ref fixed effects)

Δ(q, r) = R score − N score, regressed on a same-object indicator with ref
fixed effects; n_obs = 1024 (8 query-objects × 8 refs × 16 views).

| quantity | value |
|---|---|
| beta(same-object) | **0.0235** |
| bootstrap 95 % CI (two-way: query-object, ref) | [0.0031, 0.0795] |
| MDE | 0.0546 |
| two-way CRVE SE | 0.0078 |
| verdict | **INCONCLUSIVE** |

Descriptive split: Δ on the same-object ref 0.2396 (n = 128); Δ on other refs
0.2162 (n = 896); raw gap 0.0235. **Only ~10 % of R's advantage over N is
object-specific**; ~90 % of it shows up on completely unrelated refs and is
therefore generic (class-level or global), not object discrimination. The
interval does not clear the MDE. C2 stays "not object-discriminative" (0e).

---

## Trigger ledger

| trigger | condition | measured | fired? | outcome |
|---|---|---|---|---|
| **F-S** separate normalization | 1c \|corr\| ≥ 0.5, pooled R | +0.5048 (CI [−0.337, +0.939], 8 refs) | **FIRED** (point estimate) | V1/V2 evaluated in 1d, both **worse** than V0 and both with R−N intervals containing negative values → **DISCARD** |
| **F-G** edge-gated use | 1b edge-only R above chance **AND** above W-P **AND** 1e P(N wrong) rising as margin shrinks | R **not above chance** on top-1 (0.2266, lo 0.1016 < 0.125); R − W-P = **+0.0267** (above, ✓); P(N wrong) rising ✓ | **NOT FIRED** (1 of 3 conditions fails) | no gated policy proposed |

**KEEP RULE** (variant − N interval above zero on MRR **and** R beats W-P on
edge-only or MRR; else INCONCLUSIVE; ties → N): **not met by any variant**.
V1 and V2 both have R − N intervals spanning large negative values
(−0.32, −0.29) and are far worse for R than V0. Every candidate is discarded;
**N remains the default**.

## Committed flag values

Unchanged. `MATCH_EDGE_WEIGHT = 0.1`,
`MATCH_EDGE_DENOMINATOR_MODE = "checkable"`,
`MATCH_COVERAGE_EXPONENT_GAMMA = 0.5`, `MATCH_UNMATCHED_COST = 0.15`,
`MATCH_ALIGN_UNMAPPED_CREDIT = 0.5`,
`MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE = True`,
`MATCH_PRESENCE_ONLY_EDGES_ARE_NOT_EVIDENCE = False`,
`MATCH_SYSTEMATICITY_WEIGHT = 0.5`, `BINDER_COMPLETE_POSITION_MAP = True`,
`MATCH_SEPARATE_TERM_NORMALIZATION` not set in committed code (and, per 0f,
inert there anyway). `structural_similarity.py` untouched. **No flag was
turned on.**

## What did not run, and why

**1g (shipped arm, 136 graphs, L3/L4 OFF) and Part 3 did not run.** The
Part 3 gate was: edge-only R above chance and above W-P with clustered
intervals, **or** a pre-registered variant with a positive mean MRR difference
vs N in ≥ 6/8 objects.

- edge-only R above chance: **no** (edge-only top-1 0.2266, low end 0.1016
  < 0.125). AUC is clear of 0.5 but top-1 is the pre-registered chance test.
- edge-only R above W-P: **yes on the point estimate** (+0.0267 MRR,
  +0.0469 top-1), but the interval [−0.0858, +0.1464] crosses zero, so it is
  not "above" in the clustered sense the gate requires.
- variant positive in ≥ 6/8 objects: V1 2/8, V2 2/8 — **no**. (Only the
  shipped V0 reaches 6/8.)

Both limbs of the gate fail, so the mission's pre-registration says stop. The
outcome is **INCONCLUSIVE at n_eff = 8**, with no variant adopted and nothing
changed in the shipped configuration.

The honest summary: the instrument now exists, is verified to reproduce the
matcher's own components to 1e-9, is deterministic across processes, and the
intervals are correctly clustered.
What the design can say is bounded by its resolution. The effect that the
best-powered measurement on this branch can actually see is about **+0.7 %**
(parameters, n = 6241), and a slightly negative pooled edge contribution
(−0.0172) that is inside its own jitter envelope. At 8 objects the headline
contrast carries an MDE of **0.1212 MRR**, which is roughly **five times** the
effect being chased.

So the result is not "edges do not help". It is that **this design cannot
resolve the effect**, in either direction, on either property, and that the
one gate that would have licensed the larger test fails on both limbs. The
contribution of Part 1 is therefore chiefly the demolition of four invalid
instruments (0a positional CASE ACCURACY, 0f's W as a null, W-P as first
built, 1d's authorised flag being unreachable) plus a correctly clustered,
exactly reproducible baseline that any future design can be compared against.
