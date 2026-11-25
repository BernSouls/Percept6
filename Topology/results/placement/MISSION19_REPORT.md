# MISSION 19 — make the descriptor rank, and make dense rows reachable

Branch `arena/c316ca3a-percept6`, from the M18 tip `e796ae4`.  Pre-registration committed
**alone** at `9e766e7`, before any M19 graph existed (`MISSION19_PREREGISTRATION.md`).
Residuals are **tolunits**, LLR **nats**, shifts **radians**, spacing and texture period
**pixels**, ranks in **places**.  Every rate carries its `n` and its per-image spread.

Two things changed, both behind switches that are OFF by default:

1. **the node descriptor** — the rotation-invariant configuration descriptor, already
   computed for the bijection start, blended into every cell of the similarity matrix
   (`PLACEMENT_NODE_DESC`, K 24, w 0.25, whole row);
2. **nothing else** — the adaptive cap and the deduplicated cap were built, measured and
   **not** carried (§2), and the soft consensus is settled as a registered arm in §3/§5.

---

## 1. THE NUMBERS (15°, four DEV images, n = 1,166 unless stated)

| | top-1 | appearance-only | ctx | geo AUC (gap) | per-image top-1 | sd |
|---|---|---|---|---|---|---|
| **M18 frozen** | 0.462 | 0.059 | 0.459 | 0.796 (0.106) | .410 / .518 / .668 / .402 | 0.108 |
| **M19 design** | **0.569** | **0.310** | **0.553** | **0.826 (0.076)** | .545 / .535 / .788 / .521 | 0.110 |

Per image, `top-1 / appearance-only / ctx / geo`:

```
               M18                                     M19
m11ir_00   0.410 / 0.173 / 0.391 / 0.732        0.545 / 0.276 / 0.506 / 0.812
m11ir_01   0.518 / 0.124 / 0.494 / 0.826        0.535 / 0.341 / 0.524 / 0.802
real|cell  0.668 / 0.054 / 0.685 / 0.882        0.788 / 0.500 / 0.777 / 0.897
real|grass 0.402 / 0.017 / 0.402 / 0.743        0.521 / 0.258 / 0.509 / 0.793
```

0° (n = 2,028): top-1 **0.995 → 0.995** (0.991 per image on m11ir_00, 0.997 on m11ir_01,
1.000 cell, 0.991 grass), geo 0.973 → 0.973.  Guardrails: 0° ≥ 0.995 ✓, 15° ≥ 0.462 ✓,
geo ≥ 0.796 ✓.

**The registered trap, in one line:** `full − appearance-only` falls from **0.403 to 0.259**
because the appearance term got better.  That is the success this mission was for.

Sources: `m18_eval_tbhyb.json` (before), `m19_w25_full.json` (after).

---

## 2. O1 — THE DESCRIPTOR'S ORDERING

### 2.1 The mechanism, measured (not "the descriptor is weak")

The keypoint signature is gated by five quantised slots (theta/12, coherence/4, scale, kind,
dim), so places that share a key are separated only by the graded population term.
`m19_descriptor.json`, 15°, per image:

| cell | median row | share of places whose full quantised key is shared | **true partner's own key occupancy (median)** | distinct keys per row |
|---|---|---|---|---|
| m11ir_00 | 111 | 0.31 | **3** | 41 |
| m11ir_01 | 112 | 0.30 | **3** | 65 |
| real\|cell | 93 | 0.41 | **3** | 46 |
| real\|grass | 315 | 0.49 | **5** | 136 |

**The true partner shares its five-slot key with 2–4 other reference places at 15°.**  A
similarity built from those slots cannot put it first; at best it can break the tie, and the
population term does not.  That is why the measured rank-1 share was 0.059 with a median
rank of 11.

### 2.2 The fix: a per-node geometry that needs no mapping

`config_descriptor` (sorted log neighbour distances, sorted source-frame directions) and
`config_similarity` were already the bijection start's configuration term.  Used as a
**node** descriptor they read only each node's own star, so they are truth-free, they need
no mapping, and they are rotation-invariant.  `PM.apply_node_descriptor` blends
`(1−w)·sim + w·config` into the row.

Measured separability of that descriptor alone: AUC of `config(true)` vs `config(best
impostor)` = **0.712 / 0.732 / 0.840 / 0.792** at 15° and **0.996 / 0.976 / 0.980 / 0.882**
at 0°.

Ordering, measured offline on the same rows (`m19_descriptor.json`, pooled n = 1,166):

| variant | rank-1 share | outside cap 6 | outside cap 24 | median rank |
|---|---|---|---|---|
| base (five slots only) | 0.059 | 0.696 | 0.132 | 11.1 |
| **R2 whole-row, w 0.25 (shipped)** | **0.310** | **0.412** | **0.108** | **4.7** |
| R1 top-24, w 0.50 | 0.334 | 0.388 | 0.132 | 4.1 |
| R1 top-72, w 0.25 | 0.310 | 0.412 | 0.107 | 4.7 |

**Acceptance, all three, met:** appearance-only top-1 **0.059 → 0.310**; median true rank
**11.1 → 4.7**; share outside the operating cap **0.696 → 0.412** (outside cap 24:
0.132 → 0.108).

### 2.3 Which candidate, and why not the best rank-1

R1 (24, 0.50) has the highest rank-1 share (0.334) but the *lowest* end-to-end 15° top-1 of
the three (0.490 vs 0.569, `m19_w50_full.json`) and the widest spread (sd 0.140 vs 0.110):
a weight of 0.5 over-trusts a descriptor whose AUC is 0.71 on two of the four images.  The
registered tie-break (acceptance and guardrails first, then geo AUC, then top-1, then
smaller K, then smaller w) selects **whole-row, w 0.25**.

**Whole-row, not top-K, is a correctness requirement, not a preference.**  A head cell whose
configuration is poor is attenuated to `(1−w)·s`; a tail cell that was never blended keeps
its raw similarity, so on a mixed scale it can outscore an attenuated head cell and become
the row's argmax *without ever being re-ranked*.  Blending every positive cell keeps one
scale.  Pinned by
`test_the_whole_row_blend_matches_a_brute_force_blend` and
`test_the_head_restriction_is_the_thing_that_breaks_the_scale`.

### 2.4 The hypothesis worth testing — answered

Does a better start improve the bijection?  **Yes, but not because it is a better start:**
ctx accuracy rises 0.459 → 0.553 and geo AUC  0.796 → 0.826 *with the same assignment*,
because the bijection is solved over rows whose head is now ordered correctly.  The
context's own contribution per query falls — `full − appearance` 0.403 → 0.259 — which is
the intended direction: the descriptor does more, the geometry rescues less.

---

## 3. O2 — DENSE ROWS AND GRASS

### 3.1 The number

| | M18 | M19 | acceptance |
|---|---|---|---|
| grass top-1 at 15° (n = 656) | 0.402 | **0.521** | > 0.402 ✓ |
| grass appearance-only | 0.017 | **0.258** | — |
| grass median true rank | 15.0 | **7.0** (0.25) / 5.0 (0.50) | — |
| other three images | — | +0.120 / +0.017 / +0.077 | no regression > 0.02 ✓ |

### 3.2 The registered row-size mechanisms were built and measured — none of them delivered it

| mechanism | grass top-1 | top-1 (all) | geo AUC | vs plain design |
|---|---|---|---|---|
| none (plain cap 24 + descriptor) | **0.521** | 0.569 | 0.826 | — |
| M1 adaptive cap, k = 1/8, max 64 | 0.517 | **0.571** | **0.834** | grass −0.004, geo +0.008 |
| M1 adaptive cap, k = 1/4, max 128 | 0.517 | 0.556 | 0.826 | grass −0.004, top-1 −0.013 |
| M2 dedup by quantised key | 0.506 | 0.527 | 0.791 | grass −0.015, ctx −0.100 |

So the grass gain is real but it comes from **ordering**, not from reach: with the node
descriptor the true partner is inside the top 24 for grass far more often (outside-24 share
falls to 0.108 pooled), so widening the cap has nothing left to reach.  M2 *lost* 0.042
top-1 because dedup drops a candidate that duplicates a key already taken — and the true
partner is exactly such a duplicate (§2.1: its key is shared with 2–4 others).  That is the
recall loss the registration demanded be measured, and it is why dedup is not shipped.

**Deviation, recorded:** O2's acceptance asks for a **row-size mechanism** to produce the
grass number.  None of the three registered mechanisms produces it; the descriptor does.
The mechanisms are therefore **not carried**, and their numbers are reported here rather
than a mechanism being shipped to satisfy a sentence.

### 3.3 The three statistics M18 left open (`m19_texture.json`)

| image | texture period | spectral period | places | median place spacing | coherence median [IQR] |
|---|---|---|---|---|---|
| m11ir_00 | 122 px | 256 px (unresolved) | 1,163 | 3.97 px | 3.0 [3.0–4.0] |
| m11ir_01 | 120 px | 256 px (unresolved) | 1,089 | 4.26 px | 3.0 [3.0–4.0] |
| real\|cell | 87 px | 85.3 px | 896 | 4.60 px | 3.0 [1.0–3.0] |
| **real\|grass** | **35 px** | **17.1 px** | **2,669** | **3.02 px** | **2.0 [1.0–4.0]** |

The regime is now characterised rather than guessed: grass has **3× the places** of the
photographs at **the smallest spacing (3.0 px)** and **the shortest period (35 px, 17 px
spectrally)** — roughly **12 places per texture period**.  A period shorter than the spacing
a place's own descriptor spans is exactly the condition under which "the same five slots"
describes a dozen different places, and it is why grass was the cell where the signature
ordering failed hardest (rank-1 0.017) and where a geometry that reads the *neighbourhood*
recovered the most (+0.241).

(The first version of the period estimator read 1 px for every image — the autocorrelation's
central peak.  The fix is a minimum lag of 4; it is in `placement_m19_texture.py`.)

---

## 4. O3 — THE SOFT CONSENSUS, SETTLED BY REGISTRATION

**REJECTED, with the numbers.**  The arm was carried exactly as registered: same design,
same 20 fresh images, soft-on minus soft-off, paired per image.

| | value |
|---|---|
| n (images) | 20 |
| mean paired difference (A2 − A1) | **−0.0374** |
| sd / MDE (α .05, power .80) | 0.0391 / 0.0245 |
| 95 % CI (t, 19 df) | **[−0.0557, −0.0191]** |
| wins / ties / losses | **4 / 0 / 16** |
| worst image | **−0.094** (pre-stated margin: no image may lose more than 0.05) |
| A1 pooled | 0.461 | 
| A2 pooled | 0.419 |
| geo AUC A1 → A2 | 0.789 → 0.808 |

The registration demanded *either* a CI excluding zero in the positive direction *or*
rejection with the number.  This is rejection on both counts: the CI excludes zero on the
**negative** side, and one image loses twice the pre-stated margin.

What makes it worth reporting rather than deleting: on the four DEV images the same arm
measured **+0.016** top-1 with a narrower spread (0.108 → 0.087), and on the fresh images it
raises geo AUC (0.789 → 0.808) while losing top-1.  So the mechanism is not noise — it
improves the geometry's ranking and damages the placement decision: the shift consensus
restricts *rows*, and a row restriction that is right about the rotation can still cost the
final argmax on an image where the descriptor was already right.  M18 saw the same signature
on one photograph (real|cell −0.027 of 184 queries); the fresh read says it is systematic
(16 of 20 images lose).  **Not shipped, flag stays off.**


---

## 5. O4 — THE CAP COST, MEASURED WHERE IT IS PAID

Instrumented inside the harness (`matching/placement_llr.JUDGED`, reset per cell; the
harness records `sec.score`, `sec.total` and the judged-slot counters on every cell):

| design | cap | sec/cell (score) | queries/cell | **pairs scored / query** | **slots judged / query** | top-1 | geo |
|---|---|---|---|---|---|---|---|
| descriptor | 6 | 0.7 | 292 | 710.1 | 3,534.2 | 0.509 | 0.790 |
| descriptor | 24 | 0.7 | 292 | **710.1** | **3,534.2** | 0.571 | 0.834 |
| descriptor + dedup | 24 | 0.6 | 292 | 710.1 | 3,534.2 | 0.527 | 0.791 |

**The cap is free, and now for a measured reason rather than an inference from the perf
pairs:** the harness scores every positive cell of a row regardless of the cap (710 pairs and
3,534 judged slots per query at cap 6 and at cap 24 — identical to the last digit), so the
cap changes only the bijection's candidate list, and top-1 rises 0.509 → 0.571 at the same
cost.  M18's 0.97× on the `perf|` pairs is therefore not a coincidence of that path: the
cap's cost is not in this pipeline's scoring at all.

Caveat, stated rather than hidden: `sec.total` (which includes the graph build when the
cache is cold) was only recorded from the O5 run onward; the numbers above are the
scoring-loop seconds and the exact counters, which are the quantities the cap could affect.

---

## 6. O5 — THE FRESH READ

**Both primaries survive Holm at α = 0.05, in the direction this mission intended.**

Arms, 20 images, 4,116 queries at rot15 (`m19_o5_a1a2.json`, `m19_o5_a0.json`):

| arm | what it is | top-1 | appearance-only | geo AUC |
|---|---|---|---|---|
| **A0** | M18 frozen design (cap 24, M18 table, assignment) | 0.371 | 0.086 | — |
| **A1** | this mission (A0 + node descriptor, K 24, w 0.25, whole row) | **0.461** | — | 0.789 |
| **A2** | A1 + soft consensus | 0.419 | — | 0.808 |
| **A3** | the new descriptor's appearance-only arm (= A1's `app_ok`) | — | **0.222** | — |

| primary | contrast | n | mean | sd | 95 % CI | wins/ties/losses | MDE | p | Holm |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | A1 − A0 | 20 | **+0.0826** | 0.0653 | **[+0.0520, +0.1131]** | 18 / 0 / 2 | 0.0409 | 1.9e−05 | reject H0 (α/2) |
| **P2** | A1 − A3 | 20 | **+0.2483** | 0.0737 | **[+0.2138, +0.2828]** | 20 / 0 / 0 | 0.0462 | 5.1e−12 | reject H0 |

Both effects exceed their own MDE (0.0826 > 0.0409, 0.2483 > 0.0462), so neither read is
vacuous, and Holm (family of 2) rejects both: P2 at 5.1e−12 ≤ 0.025, P1 at 1.9e−05 ≤ 0.05.

**P2 shrank, exactly as the registration said it might** — from M18's +0.2564 to **+0.2483**
— while P1 is the largest improvement of the deployed design the programme has measured.
A smaller P2 with a larger P1 is the success condition written down before the read.

The two images that lose under P1, and by how much:

```
syn|m19ir_07   -0.029      syn|m19ir_11   -0.013
```

Both are synthetic scenes where the descriptor's own appearance arm is already strong
(0.182 and 0.280) and the design's top-1 is low (0.418, 0.475): the config blend moves the
truth out of the argmax on 4 and 2 queries respectively, out of 170 and 163.  Reported, not
smoothed.

Split, never pooled across regimes:

| regime | images | queries | A0 | A1 | A2 | appearance-only (A3) |
|---|---|---|---|---|---|---|
| 16 synthetic (seeds 91000+i) | 16 | 2,486 | 0.432 | **0.506** | 0.475 | 0.252 |
| 4 photographs (brick, coins, camera, hubble) | 4 | 1,630 | 0.277 | **0.393** | 0.333 | 0.177 |

The photographs gain **more** than the synthetic scenes (+0.116 against +0.074): a new
photograph neither the fit nor any DEV number had seen is where the node descriptor pays the
most, which is the opposite of what the DEV cells suggested (`real|cell` was already 0.668
and gained 0.120, while `m11ir_01` gained only 0.017).

Non-vacuity, checked before the verdict: **0** cells with zero queries; appearance-only > 0
in **20 of 20** images (per image 0.132 … 0.365); context rows counted on **4,116** query
rows across all 20 cells; no rate reported without a denominator.


---

## 7. EVERY FAILED ATTEMPT, WITH ITS NUMBER AND MECHANISM

| # | attempt | number | mechanism |
|---|---|---|---|
| 1 | R1 head-only band `(1−w)·s_K` | an unblended tail cell outscored an attenuated head cell (0.386 vs 0.302) | mixed scales: the tail is never attenuated, so the row's argmax could be a cell that was never re-ranked.  Fixed by blending the whole row (§2.3) |
| 2 | weight 0.50 | top-1 0.490, sd 0.140, only 0.490 at 15° against 0.569 at 0.25 | the config AUC is 0.71–0.73 on two images; at w 0.5 the descriptor outvotes a similarity that is right more often |
| 3 | top-K re-rank (K = 24, 72) | 0.569 == whole-row 0.569 (K=24 was already superset-safe) and 0.562 (72, band) | the band restriction interacts with the scale problem above; whole-row is the honest form |
| 4 | M2 dedup by quantised key | grass 0.506 (vs 0.521), top-1 0.527 (vs 0.569), ctx 0.453 (vs 0.553) | the true partner's key is shared with 2–4 others, so dedup removes the truth from the candidate set: a recall loss disguised as a diversity gain |
| 5 | M1 adaptive cap at k = 1/8 and 1/4 | grass 0.517 both, i.e. −0.004; geo +0.008 / +0.000; top-1 +0.002 / −0.013 | with the descriptor the truth is already inside 24 for grass; the cap has nothing left to reach.  Not carried (O2 asked for a row-size mechanism, and this is not one) |
| 6 | texture period estimator, first version | 1 px for every image | it read the autocorrelation's central peak, not a period; fixed with a minimum lag (§3.3) |
| 7 | `_image` for the M19 photographs | IndexError: list index out of range | it read only the frame-validation photo set; the M19 read uses photographs outside it.  Fixed to read the full cache |

---

## 8. QUALITY BARS

- **Defaults byte-identical**: `placement_m12_perf.py --check --workers 2` →
  `identical: True (0 diffs)`, 14 pairs, exit 0, run **after** every M19 change (`--workers 1`, wall 633.8 s; the M18 tip's own check at the end of M18 was the "before").
- **No new fallback at defaults**: `FAILURES` gains one key (`desc_capped`, 0 at defaults)
  and `test_no_new_path_counts_a_fallback_at_defaults` still holds; the descriptor, the
  adaptive cap and the dedup all return their input untouched when their flag is off
  (`test_off_by_default_and_the_matrix_is_the_same_object`,
  `test_off_is_exactly_top_candidates`, `DedupTests.test_off_is_exactly_top_candidates`).
- **Switches declared before use**: `PLACEMENT_NODE_DESC{,_K,_W,_MAXCELLS}`,
  `PLACEMENT_CAP_ADAPTIVE{,_K,_MAX}`, `PLACEMENT_CAP_DEDUP` in `parameters/_text.py` and in
  `eth80_frontend._FRONTEND_SWITCHES`, before any graph was built under them.
- **Tests that go RED if reverted**: 12 in `tests/test_m19_descriptor.py` — the flag-off
  identity, the configuration preference, the whole-row == brute-force blend, the head
  restriction breaking the scale, the truncation counter, determinism, the adaptive cap's
  clamp/monotone/off-identity, and dedup's head/fill/off-identity.
- **Determinism**: ties break on node id in every sort (`(-score, node id)`), the blend is
  closed-form, the descriptor is a pure function of the side; repeated cells reproduce.
- **Units** as stated at the top; `m19_texture.json` states px and coarse-slot units.
- **Manifest + per-cell rows** on every result file through `m15_manifest.write_result`.

---

## 9. DEVIATIONS FROM THE REGISTRATION

1. **O2's mechanism requirement is unmet while its number is met** (§3.2).  All three
   registered mechanisms were built and measured; none produces the grass gain.  The
   report says so and carries none of them, rather than shipping the one with the best
   aggregate and calling it O2.
2. **The whole-row blend replaced the registered "top-K re-rank"** as the shipped form.
   Reason in §2.3: the top-K form is not scale-consistent.  It is a refinement of R1/R2,
   not a widened search, and both forms were measured (0.569 each at K = 24).
3. **The O1 tie-break was applied to whole-row candidates as well as to K/w**, since the
   registration listed R1 and R2 as separate routes.
4. **The adaptive cap's best setting (k = 1/8, max 64) has a higher geo AUC than the shipped
   design** (0.834 vs 0.826) and one cell +0.038, at grass −0.004 (3 queries of 656).  It is
   reported, not shipped: O2 is the outcome it was built for and it does not move it (§9.1
   of the registration says a criterion may not be reinterpreted).

---

## 10. OPEN PROBLEMS, RANKED, AND THE SINGLE NEXT CHANGE

1. **The context's marginal share keeps falling** (0.403 → 0.259 per query on DEV).  The
   geometry is now the tie-breaker rather than the rescuer, and its own ceiling (0.902) is
   unchanged: geo AUC 0.826 of a 0.902 ceiling on DEV, 0.789 on the fresh read.
0. **The fresh read is the honest number and it is lower than DEV** (0.461 against 0.569),
   and the two images that lose under P1 are both synthetic scenes where the appearance arm
   is already strong — the blend has the least to add exactly where the signature already
   orders well.
2. **`m11ir_01` is the weakest cell** (0.535) and its appearance term is the strongest
   (0.341) — the two are related: where the signature already orders well, the config blend
   has little left to add, and the remaining loss is geometry.
3. **The adaptive cap deserves a registered read** (§9.4): +0.008 geo AUC and +0.038 on one
   cell at −3 grass queries is exactly the size of effect that needs the O5 discipline.
4. **Dedup needs a key that the truth does not share** — the failure is in the key, not in
   the idea.
5. **The fresh photographs remain the weak regime** (see §6): real texture is where the
   descriptor's AUC is lowest (0.84 cell, 0.79 grass at 15°) and where every mechanism has
   the least margin.

**The single next change**: give the node descriptor the *scale* term it is missing — the
config descriptor reads distances and directions but not `log_scale`, which is the one slot
the five-slot signature carries and the configuration throws away; a node descriptor that
uses it is the cheapest route to the `m11ir_01`/photograph gap, and it is truth-free at
match time.
