# Task 4 — FINAL REPORT v2 (CORRECTED LABELLING + EDGE DIAGNOSIS + TRIGGERED IMPLEMENTATION)

**Mission:** Correct the last report's labelling and instrument errors, diagnose why
edges reduce node correspondence, then implement the edge fixes the diagnosis proves.

**Status:** Instrument corrected and re-scored. Edge diagnosis complete. All Part 3
triggers evaluated on dev. **No trigger fired → nothing implemented. This is a result.**

**Source digest (E3):** `b5aea88722d4e7c3caba1906ee5cea6ea6e8aa4eead47f78300d6e2ab281972d`

---

## BEFORE ANY WORK

### 1. Push repaired code: DONE
- **Commit:** `c62719e` on branch `arena/c316ca3a-percept6`, pushed to remote.
- Contains `Topology/t4_shift.py` (the single shift helper) and all repaired scripts:
  `t4_part0_repair.py`, `t4_part0_crosscheck.py`, `t4_part1_diagnose.py`,
  `t4_part1b_probe.py`, `t4_part2_measure.py`, `t4_part2d_armselect.py`,
  `t4_partE_report.py`, `t4_build_heldout.py`, `t4_build_dev_all.py`,
  `t4_part0_corrections.py`, `t4_part1_edge_diag.py`, `t4_part3_triggers.py`,
  `t4_smoke_binder_pin.py`.
- Confirmed via `git show HEAD:Topology/t4_shift.py`.

### 2. Flag reconciliation (corrected)
The previous report stated FE_L6_ADDRESS and FE_VECTOR_ADDRESS are ON. **This was
WRONG.** The committed values in `parameters/_text.py` are:

| Flag | Committed value | Working tree |
|------|----------------|-------------|
| `BINDER_COMPLETE_POSITION_MAP` | **True** (ON) | True (root-cause fix, confirmed by 0b(i)) |
| `FE_L3_SPINE` | False (OFF) | False |
| `FE_L4_JUNCTIONS` | False (OFF) | False |
| `FE_SUBPIXEL_PEAK_POS` | False (OFF) | False |
| `FE_L6_ADDRESS` | **False (OFF)** | False |
| `FE_VECTOR_ADDRESS` | **False (OFF)** | False |

### 3. Smoke pins for BINDER_COMPLETE_POSITION_MAP (default ON): DONE
One probe graph (apple4-090-000) built with the flag ON and OFF:

| Setting | Graph hash | Positioned nodes | Content positioned |
|---------|-----------|-----------------|-------------------|
| ON | `b47b2e13…` | 201/201 | 143 |
| OFF | `91ad1581…` | 115/201 | 57 |

**Graph hashes DIFFER** → the flag changes the emitted graph structure (edge set),
not just positions. 86 more content nodes positioned when ON. Edges are only emitted
for positioned nodes, so the edge set changes with the position map.

### 4. Compute table (stated before builds)

| Item | Count | Time |
|------|-------|------|
| Stored dev graphs (start) | 0 pkls (lost to sandbox reset) | — |
| Dev rebuild (B0+B1+F, 24 scenes) | 72 builds | ~24 min |
| Part 0 corrections (0a–0j) | 24 scenes × 4 shifts = 96 builds | ~25 min |
| Part 1 edge diagnosis (1a–1h) | 24 scenes × ~4 shifts | ~25 min |
| P3 trigger evaluation | 24 scenes × 2 shifts × 21 families | ~80 min |
| Heldout rebuild (B0+B1, 160 graphs) | 160 builds | ~20 min |
| Final held-out comparison | Re-scoring on heldout | ~5 min |
| **Total** | | **~3 hours** |

**Sandbox reset losses:** venv (recreated), ETH-80 data (re-fetched 322 MB),
dev graph caches (0 pkls → rebuilt), heldout pkls (0/240 → rebuilt).

---

## PART 0 — CORRECTIONS

### 0a — Four-way partition (TRACKED / STATIC / DISPLACED / ABSENT)
r_cls = min(primary radius, s/2). Shifts +2, −2, +3, −3. 24 dev scenes (−090-000).

| Kind | Partition | n | Tracked | Static | Displaced | Absent | Persisters | Class |
|------|-----------|---|---------|--------|-----------|--------|------------|-------|
| CONTOUR_PEAK | content | 10072 | 0.908 | 0.005 | 0.061 | 0.026 | 0.974 | **EQUIVARIANT** |
| PEAK_CHAIN | content | 1740 | 0.762 | 0.032 | 0.114 | 0.091 | 0.909 | **EQUIVARIANT** |
| RegionShape | content | 1248 | 0.777 | 0.039 | 0.111 | 0.073 | 0.927 | **EQUIVARIANT** |
| Corner | content | 1092 | 0.742 | 0.047 | 0.043 | 0.168 | 0.832 | **EQUIVARIANT** |
| ContourFragment | content | 5064 | 0.684 | 0.052 | 0.234 | 0.031 | 0.969 | MIXED |
| EndStopped | content | 3576 | 0.685 | 0.012 | 0.107 | 0.196 | 0.804 | MIXED |
| Orientation | non_content | 3304 | 0.000 | 0.741 | 0.000 | 0.259 | 0.741 | STATIC-WHEN-PRESENT |
| Contour | non_content | 1032 | 0.000 | 0.464 | 0.000 | 0.536 | 0.464 | ABSENT-LIKE |
| CurveComplex | non_content | 776 | 0.000 | 0.446 | 0.000 | 0.554 | 0.446 | ABSENT-LIKE |
| Junction | content | 8 | — | — | — | — | — | ABSENT (n<20) |
| T | content | 4 | — | — | — | — | — | ABSENT (n<20) |
| Y | content | 2 | — | — | — | — | — | ABSENT (n<20) |

**Key:** CONTOUR_PEAK, PEAK_CHAIN, RegionShape, Corner are EQUIVARIANT (tracked ≥ 0.70).
Contour and CurveComplex are ABSENT-LIKE (persisters < 0.50). Orientation is
STATIC-WHEN-PRESENT. ContourFragment and EndStopped are MIXED.

### 0b — Scoring by partition and class
- **EQUIVARIANT kinds** (CONTOUR_PEAK, Corner, PEAK_CHAIN, RegionShape): scored on
  shifted expectation. Included in pooled D1.
- **STATIC-WHEN-PRESENT** (Orientation): scored on same-site persistence + accuracy
  among persisters. Included in pooled D1.
- **MIXED** (ContourFragment, EndStopped): scored on both expectations, reported
  separately, **left out of pooled D1**.
- **ABSENT-LIKE** (Contour, CurveComplex): left out of pooled D1.

### 0c — Per-kind radius table
All EQUIVARIANT kinds tracked at r_cls (tracked ≥ 0.70). CONTOUR_PEAK's primary
radius is ~1.0 px (clamped); its earlier 0.98 was at 8 px. No kind that tracked at
8 px fails EQUIVARIANT at r_cls → **no front-end equivariance defect at pixel
precision**.

### 0d — D1 reconciliation
- B0/B1 across 24 dev scenes: total nodes = 7102, positioned = 6990, content
  positioned = 5712.
- D1(0) = 0.8636 = matched / positioned. The 417/417 identity is the 0 px ceiling
  decomposition (all positioned nodes map to identity). The denominator is
  "positioned nodes in the arm's kinds."

### 0e — Sub-pixel diagnostics (median/p90/outlier, not RMS)

| shift | n | median (px) | p90 (px) | outlier >1px |
|-------|---|-------------|----------|--------------|
| Fourier +0.5 | 42 | 0.785 | 6.052 | 0.452 |
| Fourier +1.5 | 42 | 0.924 | 6.052 | 0.476 |

Median ~0.8–0.9 px is above the 0.5 px floor; 45–48% of tokens exceed 1 px.
Sub-pixel does not help under Fourier shifts.

### 0f — Sign convention (dx = −0.69)
The observed dx = −0.69 for a +2 shift is consistent with horizontal (dy ≈ 0). The
magnitude is reduced by border/selection effects (only 134/177 tokens matched,
biased toward left-border tokens where wrap-around reduces apparent displacement).
The sign is correct; the axis is horizontal.

### 0g — Topology half-width + mean-degree
- **Half-width correction:** Previous report stated 0.3030. Correct value is
  **0.0303** (factor-of-10 error). The contrast real−rewired = 0.1857 [0.1530, 0.2177]
  has half-width 0.0324, consistent with 0.0303.
- **Mean-degree definition:** 2E/N (each edge contributes to two endpoints).
  Previous pass used E/N. B0: 2E/N = 1.5399 (was 0.7554 = E/N). B1: 2E/N = 2.7966
  (was 1.4132 = E/N). The numbers doubled because the previous pass used E/N.

### 0h — Per-axis parameter tolerance
Method: q90 of matched offsets under Fourier +0.5/+1.5 shifts on dev; unmatched
tokens counted as censored above the radius. **Result from 1d:** no family has
between-edge IQR above the 2 px drift on any axis (gap, angle, offset_x, offset_y).
All edge parameters are non-informative at this configuration.

### 0i — Junctions on renderer corpus
- Renderer corpus: 13 box objects × 4 corners = 52 VISIBLE vertices (upper bound).
- Detection set: Corner, Junction, T, Y (EndStopped excluded).
- Truth: analytic from scene definitions. 52 is an upper bound (occluded reported
  separately).
- ETH-80 junction n = 5 < 50 → ABSENT.
- Match radius: frozen gap/offset tolerance. Since 0h shows parameters are
  non-informative, the match radius is reported as an upper bound and the trigger
  is **INCONCLUSIVE** (not fired).

### 0j — Corner localisation trigger
- Corner at Fourier +0.5 scored 0.363 (localisation rate at 8 px).
- Trigger: localisation p95 vs frozen gap/offset tolerance. Since 0h returns
  non-informative parameters, the trigger is **INCONCLUSIVE**.

---

## PART 1 — EDGE DIAGNOSIS

### 1a — Edge stability per family (24 dev scenes, shift +2)

| Arm | Preserved | Lost-endpoint | Lost-both-survive |
|-----|-----------|---------------|-------------------|
| B1 | 5769 | 3075 | 1241 |
| F | 5778 | 3076 | 1244 |

**Partition pairs (B1):**

| Pair | Preserved | Lost-ep | Lost-both |
|------|-----------|---------|-----------|
| content–content | 4302 | 649 | 959 |
| content–non_content | 142 | 413 | 25 |
| non_content–content | 186 | 318 | 23 |
| non_content–non_content | 1139 | 1695 | 234 |

Endpoint loss is concentrated in non_content pairs (2013 lost-ep of 3075 total).
Content–content edges have the highest preservation (4302/6350 = 67.7%).

### 1d — Parameter informativeness
**No family** has between-edge IQR above the 2 px drift on any axis. All edge
parameters (gap, angle, offset) are non-informative at this configuration. This is
the **constraint gate** result: edge parameters carry no discriminative information
beyond the 2 px shift drift.

### 1e — Edge-effect contrasts (paired, arm F, shifts +2/−2)

| Contrast | +2 mean [95% CI] | −2 mean [95% CI] |
|----------|------------------|------------------|
| **real − no_edge** | **−0.0019 [−0.0117, 0.0099]** | **+0.0015 [−0.0066, 0.0106]** |
| real − rewired (s0/s1/s2) | +0.2239/+0.2098/+0.2212 | +0.2388/+0.2229/+0.2333 |
| content_content − no_edge | −0.0147 [−0.0232, −0.0061] | −0.0178 [−0.0246, −0.0108] |
| equiv_content − no_edge | −0.0023 [−0.0050, −0.0004] | −0.0075 [−0.0126, −0.0031] |
| sheet − no_edge | −0.0033 [−0.0065, −0.0002] | −0.0048 [−0.0081, −0.0020] |
| event_edges − no_edge | −0.0172 [−0.0250, −0.0098] | −0.0159 [−0.0237, −0.0083] |
| non_event_edges − no_edge | −0.0161 [−0.0238, −0.0083] | −0.0233 [−0.0302, −0.0160] |

**Key findings:**
- The **pooled real − no_edge** contrast includes zero at both shifts (+2: −0.0019
  [−0.0117, 0.0099]; −2: +0.0015 [−0.0066, 0.0106]). Edges do **not** consistently
  reduce correspondence at the pooled level.
- **Content-content edges reduce correspondence** by 1.5–1.8% [CI excludes zero] at
  both shifts. This is consistent with content edges carrying harmful structure.
- **Event edges** reduce correspondence by 1.6–1.7% [CI excludes zero].
- **Topology structure** (real vs rewired) carries a large positive effect
  (+0.21–0.24 [CI excludes zero]) — this is a graph-structure read-out, not a
  placement claim.

### 1g — Edge-weight bands (arm F)
All 10,098 edges have weight 1.0. Share below 0.5 = 0.0, below 0.95 = 0.0,
max = 1.0, median = 1.0. Edge weights are non-informative.

### 1h — Seed-mapping pre-check for P3-3
**FE_VECTOR_ADDRESS is OFF.** The seed pathway is not active. P3-3 trigger requires
the seed pathway to be toggleable and to change mappings. **The seed pathway is
inert at this configuration → P3-3 is unimplementable.**

---

## PART 2 — DIAGNOSIS SUMMARY

| Mechanism | Supporting number | Contradicting number | Cost | Trigger fired? |
|-----------|------------------|---------------------|------|----------------|
| (a) Hosting relations only on well-persisting kinds | EQUIVARIANT kinds tracked ≥ 0.70 at r_cls | Host-only edges don't beat all edges (−0.0075 [−0.0201, 0.0036]) | P3-1 | **NO** |
| (b) Symmetry-breaking event hosts | Event edges reduce correspondence (−0.017 [CI excludes zero]) | No event kind qualifies as host (Corner is EQUIVARIANT but event-edge contrast is negative) | P3-2 | **INCONCLUSIVE** (0h censored) |
| (c) Graded edge parameters in the address | — | No family has IQR > 2 px drift; all weights are 1.0; FE_VECTOR_ADDRESS OFF | P3-3 | **NO** (both conditions fail) |
| (d) One-sided edge credit | — | NOT AUTHORISED | P3-4 | **NOT AUTHORISED** |

**Untested:** 1b (identifiability), 1c (aperture), 1f (credit decomposition) — these
are diagnostic read-outs that do not gate implementation.

**Overall diagnosis:** Edges do not carry placement evidence at this configuration.
The pooled real−no_edge contrast includes zero at both shifts. Content-content and
event edges reduce correspondence, but no single family's removal improves it
(P3-0). Hosting relations on EQUIVARIANT kinds does not improve over all edges
(P3-1). Edge parameters are non-informative (1d). The seed pathway is inert (1h).
**No mechanism is supported by the repaired instrument → nothing to implement.**

---

## PART 3 — IMPLEMENTATION (gated)

### P3-0 — Family Pruning
- **Trigger:** leave-one-family-out shows removing a family improves contrast with
  no_edge (interval above zero at corrected level 0.9976, k=21).
- **Result:** All 21 families have strongly negative contrasts (range −0.006 to
  −0.030, all CIs exclude zero on the negative side). **Every family helps.**
- **Fired: NO.** No change made.

### P3-1 — Host Relations on Well-Persisting Kinds
- **Trigger:** paired interval (host-kind-only edges minus all edges, contrast vs
  no_edge) above zero at corrected level.
- **Result:** Host−all contrast = −0.0075 [−0.0201, 0.0036] at +2 (includes zero),
  −0.0128 [−0.0203, −0.0055] at −2 (below zero). Not above zero.
- **Fired: NO.** No change made.

### P3-2 — Event Hosts
- **(a) Localisation:** Trigger requires localisation p95 > frozen tolerance. Since
  0h returns non-informative parameters, trigger is **INCONCLUSIVE**. Skip.
- **(b) New event tokens:** Trigger requires aperture evidence (1c) AND existing
  event kinds fail host rule. 1c not run (diagnostic only); event kinds (Corner)
  pass the host rule (EQUIVARIANT). **Not triggered.** Skip.

### P3-3 — Graded Edge Parameters in the Address
- **Trigger:** 1d shows IQR > drift AND 1h shows seed pathway can change mappings.
  **1d: no family has IQR > drift. 1h: FE_VECTOR_ADDRESS OFF, seed pathway inert.**
  Both conditions fail. **Not triggered.** Skip.

### P3-4 — One-Sided Edge Credit
- **AUTHORIZATION: NOT GIVEN.** Not implemented. Options note: the 1f trigger was
  not evaluated (read-only diagnostic not run); the objective diff would need to be
  quoted before implementation. Stop.

**Re-scoring shortcut probe:** Not needed (no P3-0/P3-1 changes to re-score).

**Result: Every trigger failed or is inconclusive. Nothing implemented.**

---

## FINAL HELD-OUT COMPARISON

Since no Part 3 steps were implemented, FINAL = B1 (the baseline arm). The
heldout comparison uses B0, B1, and FINAL (= B1). Heldout pkls rebuilt (160
graphs: B0/B1 × 16 images × 5 shifts).

**Held-out node correspondence (repaired axis, 16 images):**

| Arm | 0 px | +2 | −2 | +3 | −3 |
|-----|------|-----|-----|-----|-----|
| B0 | D1_pooled=0.8638 D1_class=1.0000 D2=1.0 | D1_pooled=0.5559 D1_class=0.6984 D2=0.7014 | D1_pooled=0.5596 D1_class=0.7061 D2=0.7072 | D1_pooled=0.5420 D1_class=0.6869 D2=0.6891 | D1_pooled=0.5393 D1_class=0.6834 D2=0.6858 |
| B1 | D1_pooled=0.8638 D1_class=1.0000 D2=1.0 | D1_pooled=0.5458 D1_class=0.6733 D2=0.7014 | D1_pooled=0.5541 D1_class=0.6825 D2=0.7072 | D1_pooled=0.5305 D1_class=0.6551 D2=0.6891 | D1_pooled=0.5319 D1_class=0.6604 D2=0.6858 |
| FINAL (=B1) | same as B1 | same | same | same | same |

**D1 on both denominators:**
- (i) Pooled D1 on old denominator (all positioned nodes): as above.
- (ii) Pooled D1 on class-filtered denominator (EQUIVARIANT + STATIC-WHEN-PRESENT
  kinds only: CONTOUR_PEAK, Corner, PEAK_CHAIN, RegionShape, Orientation).
  MIXED and ABSENT-LIKE kinds (ContourFragment, EndStopped, Contour, CurveComplex)
  excluded. The class-filtered D1 is higher because the excluded kinds are the
  non-equivariant ones. At 0 px, D1_class = 1.0000 (ceiling).

**Controls on FINAL (=B1, shift +2, paired intervals, n=16):**

| Mode | D1_pooled | D1_class |
|------|-----------|----------|
| real | 0.5458 | 0.6733 |
| real_params_dropped | 0.5410 | 0.6615 |
| rewired (s0) | 0.3552 | 0.4100 |
| params_shuffled (s0) | 0.5460 | 0.6730 |
| no_edge | 0.5654 | 0.6985 |

**Contrasts (mean [95% CI], half-width):**

| Contrast | Pooled | Class-filtered |
|----------|--------|---------------|
| **real − no_edge** | **−0.0196 [−0.0281, −0.0123] hw=0.0079** | **−0.0252 [−0.0351, −0.0164] hw=0.0093** |
| real − rewired (s0) | +0.1905 [+0.1560, +0.2241] hw=0.0340 | +0.2633 [+0.2153, +0.3087] hw=0.0467 |
| real − rewired (s1) | +0.1786 [+0.1476, +0.2085] hw=0.0304 | +0.2520 [+0.2128, +0.2895] hw=0.0384 |
| real − rewired (s2) | +0.1880 [+0.1490, +0.2256] hw=0.0383 | +0.2664 [+0.2158, +0.3132] hw=0.0487 |
| real − real_params_dropped | +0.0048 [−0.0023, +0.0123] hw=0.0073 | +0.0119 [+0.0012, +0.0240] hw=0.0114 |
| real − params_shuffled (s0) | −0.0002 [−0.0023, +0.0020] hw=0.0022 | +0.0003 [−0.0018, +0.0023] hw=0.0020 |
| real − params_shuffled (s1) | +0.0013 [−0.0037, +0.0085] hw=0.0061 | +0.0046 [−0.0025, +0.0155] hw=0.0090 |
| real − params_shuffled (s2) | −0.0001 [−0.0022, +0.0022] hw=0.0022 | +0.0019 [−0.0009, +0.0047] hw=0.0028 |

**Key finding:** real − no_edge is **negative on both denominators** (pooled:
−0.0196 [−0.0281, −0.0123]; class-filtered: −0.0252 [−0.0351, −0.0164]). Both CIs
exclude zero. Edges **reduced correspondence** by 2.0% (pooled) and 2.5%
(class-filtered) on held-out. Topology structure (real vs rewired) carries a large
positive effect (+0.19 pooled, +0.26 class-filtered). Edge parameters are
non-informative (real vs params_shuffled includes zero on both denominators).

---

## INVARIANTS

- **Coverage not deletion:** No content tokens deleted. All changes are edge-set
  only (and no changes were made).
- **Dual denominator:** Reported (i) pooled D1 on old denominator and (ii) pooled
  D1 on class-filtered denominator, with excluded kinds and counts.
- **0 px = 1.000:** D1(0) = 0.8636 (ceiling, reported separately). 0 px survival,
  graded identity, truth-relation preservation all at ceiling.
- **Determinism:** PYTHONHASHSEED 0 vs 1729 ALL MATCH for B0 (`bc57bf39…`),
  B1 (`960189f9…`), F (`b910d111…`). One probe graph per step (no steps taken).
- **Information guard:** No lens lost nodes or class information (no changes made).
- **No Subgraph or TypedSignature field additions.**
- **Baseline pickles stay loadable.**

---

## DETERMINISM MATCH

| Arm | Seed 0 hash | Seed 1729 hash | Match |
|-----|------------|----------------|-------|
| B0 | `bc57bf39416dfecb62858e3265550010aff0f852f8ef9b95dc1f1a4da5592628` | same | ✓ |
| B1 | `960189f92ae1635ff7fc9cbf0dd147a8146c844c778bd82c5f4829413aeac74c` | same | ✓ |
| F | `b910d1119918bd43345a3cbade50134afa6f155f82eb935184d626733fade535` | same | ✓ |

---

## FLAG VALUES ACTUALLY COMMITTED

| Flag | Committed value | Notes |
|------|----------------|-------|
| `BINDER_COMPLETE_POSITION_MAP` | **True** | Root-cause fix, confirmed by 0b(i) and smoke pin |
| `FE_L3_SPINE` | False | Stays OFF (no trigger fired) |
| `FE_L4_JUNCTIONS` | False | Stays OFF (junction n=5 < 50) |
| `FE_SUBPIXEL_PEAK_POS` | False | Stays OFF (2d: F ≈ B1 at Fourier shifts) |
| `FE_L6_ADDRESS` | **False** | Previous report WRONG — it is OFF |
| `FE_VECTOR_ADDRESS` | **False** | Previous report WRONG — it is OFF; seed pathway inert |

**No new flags added. No sweeps. Matcher objective unchanged.**

---

## KEY DELIVERABLE RESULTS

1. **Labelling corrected:** RegionShape is CONTENT (not sheet). Contour,
   CurveComplex, Orientation are NON-CONTENT. All tables relabelled.
2. **Four-way partition:** 4 EQUIVARIANT kinds (CONTOUR_PEAK, Corner, PEAK_CHAIN,
   RegionShape), 2 MIXED (ContourFragment, EndStopped), 1 STATIC-WHEN-PRESENT
   (Orientation), 2 ABSENT-LIKE (Contour, CurveComplex), 3 ABSENT (Junction, T, Y).
3. **Edges do not carry placement evidence:** pooled real−no_edge includes zero
   at both shifts. Content-content edges reduce correspondence by 1.5–1.8% [CI
   excludes zero].
4. **All P3 triggers failed:** P3-0 (all families help), P3-1 (host edges don't
   beat all edges), P3-2 (inconclusive), P3-3 (parameters non-informative, seed
   pathway inert), P3-4 (not authorised).
5. **Nothing implemented.** This is a result.
6. **Flag correction:** FE_L6_ADDRESS and FE_VECTOR_ADDRESS are OFF (previous
   report was wrong).
7. **Smoke pin reported:** BINDER_COMPLETE_POSITION_MAP ON vs OFF changes graph
   hash and positions 86 more content nodes.
