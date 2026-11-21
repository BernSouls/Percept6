# Task 4 — FINAL REPORT (REPAIRED INSTRUMENT)

**Status:** Instrument repaired. All shift-based numbers recomputed with the
corrected horizontal-axis helper `Topology/t4_shift.py`. Previous nonzero-shift
results **INVALID** (vertical-roll axis bug). This report supersedes all prior
shift-based numbers.

**Source digest:** `b5aea88722d4e7c3caba1906ee5cea6ea6e8aa4eead47f78300d6e2ab281972d`
(E3; Part 0 provenance digest was `2a56807b…c20c` before Part 1–E script additions)

---

## PART 0 — INSTRUMENT REPAIR

### 0a — One shift helper (`Topology/t4_shift.py`)
- `roll_image(arr, dx, dy)` = `np.roll(arr, (dy, dx))` — **dx=horizontal, dy=vertical**.
- `fourier_shift(arr, dx, dy)` — standard unshifted-`fftfreq` phase shift (verified
  integer-exact vs `np.roll`; fractional 0.5 diff 9e-6).
- `expected_position(x0, y0, dx, dy)` = `(x0+dx, y0+dy)`.
- `counterpart_map(...)` — strict `<` at radius (counterpart at exactly the radius
  is NOT counted), one-to-one mutual nearest (second-nearest ≥ 2× nearest),
  ambiguous fraction reported.
- `primary_radius(...)` — half median NN same-kind spacing, clamped [1,8] px,
  per kind and per partition.

### 0b(i) — Shift all post-build position stores together: **PASS**
- Shifting `anchor_of_node` + position registry + per-node fields together:
  201/201 nodes recover identity on unique positions; 0px ceiling = 1.0.
- Shifting ONLY `anchor_of_node` yields identical results → **no consumer reads
  another store**. Position store = `subgraph.anchor_of_node` only.

### 0b(ii) — Cross-check vs known_transform: **PASS**
Repaired harness (`t4_part0_crosscheck.py`) truth fractions match
`known_transform.py` **exactly**:

| shift | truth fraction (repaired) | known_transform |
|-------|--------------------------|-----------------|
| 0 px  | 1.0000                   | 1.0000          |
| +2 px | 0.8137                   | 0.8137          |
| −2 px | 0.8080                   | 0.8080          |
| +3 px | 0.7776                   | 0.7776          |
| −3 px | 0.7757                   | 0.7757          |

Matcher/cell fractions differ from truth because the harness applies WINNER cell
params (not known_transform defaults) — not an instrument bug.

### 0b(iii) — Fourier integer-exactness: **PASS**
Integer shifts exact vs `np.roll`; fractional 0.5 diff 9e-6 (~0.5 px floor).

### 0b(iv) — Axis provenance: **ANSWERED (horizontal confirmed)**
Rebuilt heldout pickles verified via **matched-token displacement** (centroid
method was misleading due to border wrap):
- B0/F `apple7-090-000` shift+2: 134/177 matched, mean dx = −0.69, dy = −0.04 ≈ 0.
- Heldout pickles are **correct-axis** (horizontal).

### 0c — PRIMARY POSITION RADIUS (dev, arm B0)
| Kind | Primary radius (px) | n |
|------|-------------------|---|
| CONTOUR_PEAK | 1.118 | 890 |
| ContourFragment | 1.0 | 433 |
| Orientation | 1.0 | 275 |
| PEAK_CHAIN | 3.001 | 142 |
| EndStopped | 2.915 | 308 |
| RegionShape | 3.416 | 103 |
| Corner | 3.606 | 91 |
| Contour | 8.0 | 88 |
| CurveComplex | 8.0 | 65 |
| **Junction** | **ABSENT** | **5** (<20 → ABSENT) |

### 0d — INVALID marks
7 files renamed `*_INVALID_axisbug.json`: `measure_4c`, `measure_4d`,
`measure_4e`, `e6_flicker`, `part0_dev_B0`, `part0_dev_B1`, `part0_dev_F`.
Old `e7_report.md` annotated with INVALID banner.

### 0e — Provenance
`results/t4/part0_provenance.json` — source_digest `2a56807b…c20c`; stale
pre-reset results listed; rebuild status recorded.

---

## PART 1 — DIAGNOSIS

### 1a — Per-kind / per-partition equivariance @ 8 px (arm B1, rate)

| Kind | vert+2 | vert+3 | horiz+2 | Fourier+0.5 | Fourier+1.5 | Verdict |
|------|--------|--------|---------|-------------|-------------|---------|
| **CONTOUR_PEAK** (content) | 0.981 | 0.975 | 0.982 | 0.949 | 0.948 | **EQUIVARIANT** |
| ContourFragment (content) | 0.977 | — | 0.991 | — | — | near-perfect @ integer; high ambiguity (26–116) |
| PEAK_CHAIN (content) | 0.901 | 0.859 | 0.965 | 0.592 | — | good @ integer, degrades fractional |
| Corner (content) | 0.835 | 0.802 | 0.890 | 0.363 | — | PARTIAL (degrades fractional) |
| EndStopped (content) | 0.815 | 0.731 | 0.786 | 0.718 | — | partial |
| RegionShape (sheet) | 0.942 | — | 0.971 | 0.592 | — | high @ integer, drops fractional |
| Orientation (sheet) | 0.749 | — | 0.767 | 0.862 | — | static-by-design lens |
| **Contour** (content) | 0.375 | 0.261 | 0.466 | 0.546 | 0.443 | **NOT equivariant** |
| **CurveComplex** (content) | 0.292 | 0.169 | 0.462 | 0.569 | 0.339 | **NOT equivariant** |
| Junction (content) | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | **ABSENT** (n=5) |

### 1b — Mechanism probes (Contour, CurveComplex; arm B1)

| Kind | shift | loose | strict | no_nearby |
|------|-------|-------|--------|-----------|
| Contour | vert+2 | 33/88 (37.5%) | **0** | 55/88 (62.5%) |
| Contour | horiz+2 | 41/88 (46.6%) | **0** | 47/88 (53.4%) |
| CurveComplex | vert+2 | 19/65 (29.2%) | **0** | 46/65 (70.8%) |
| CurveComplex | horiz+2 | 30/65 (46.2%) | **0** | 35/65 (53.8%) |

**Mechanism:** Contour and CurveComplex have **0% strict match** and 53–71% of
base nodes have **no nearby same-kind node** in the shifted graph. These tokens
are **dropped/created** by the front-end under shift — not equivariant.

### 1c — Sub-pixel diagnostics (arm F, Fourier shifts)

| shift | integer-snapped RMS | sub-pixel RMS | floor (0.5 px) |
|-------|--------------------|--------------|-----------------|
| +0.5 | 2.525 (n=774) | 2.514 (n=71) | 0.50 |
| +1.5 | 2.618 (n=773) | 2.603 (n=71) | 0.50 |

**Sub-pixel positions do NOT beat the 0.5 px floor** — RMS ~2.5 px for both
integer and sub-pixel. Sub-pixel does not help under Fourier shifts.

### 1d — Position-wiring: per-lens anchor fraction (all dev graphs)

| Lens | B0 anchors | B1 anchors |
|------|-----------|-----------|
| CONTOUR_PEAK | 890/890 (100%) | 890/890 (100%) |
| Contour | 88/88 (100%) | 88/88 (100%) |
| ContourFragment | 433/433 (100%) | 433/433 (100%) |
| Corner | 90/90 (100%) | 90/90 (100%) |
| CurveComplex | 65/65 (100%) | 65/65 (100%) |
| EndStopped | 308/308 (100%) | 308/308 (100%) |
| **IllusoryContour** | **0/23 (0%)** | **0/23 (0%)** |
| Junction | 5/5 (100%) | 5/5 (100%) |
| Orientation | 275/295 (93.2%) | 275/295 (93.2%) |
| PEAK_CHAIN | 142/142 (100%) | 142/142 (100%) |
| RegionShape | 103/103 (100%) | 103/103 (100%) |
| **Total** | **2399/2442 (98.24%)** | **2399/2442 (98.24%)** |

**IllusoryContour** is the only kind with 0% anchor coverage (23 nodes total).
CONTOUR_PEAK 100% → position-wiring fix confirmed working.

### Equivariance + mechanism count summary
- **EQUIVARIANT (1):** CONTOUR_PEAK (content).
- **Near-equivariant @ integer / partial (5):** ContourFragment, PEAK_CHAIN,
  Corner, EndStopped, RegionShape.
- **NOT equivariant (2):** Contour, CurveComplex — mechanism = drop/create
  (0% strict, 53–71% no-nearby).
- **Static by design (1):** Orientation (sheet).
- **ABSENT (2):** Junction (n=5), IllusoryContour (n=23, 0% anchors).

---

## PART 2 — REDO MEASUREMENTS

### 2a — Dev measurements (arm B0, B1; 8 scenes, view −090-000)

| Arm | Mean degree | Zero-relation share | Median NN gap (CONTOUR_PEAK) |
|-----|------------|--------------------|----------------------------|
| B0 | 1.5399 | 0.6521 | 7.000 |
| B1 | 2.7966 | 0.1879 | 7.000 |

Per-family edge counts preserved in JSON (`part2_measure.json`).

### 2b — Node correspondence (held-out, 16 images, repaired axis)

**D1** = matcher accuracy among identifiable nodes; **D2** = fraction of base
nodes with a counterpart within 2.0 px.

| Arm | 0 px | +2 | −2 | +3 | −3 |
|-----|------|-----|-----|-----|-----|
| **B0** | **D1=0.8636 D2=1.0000** | D1=0.5569 D2=0.7014 | D1=0.5606 D2=0.7072 | D1=0.5430 D2=0.6891 | D1=0.5403 D2=0.6858 |
| **B1** | **D1=0.8636 D2=1.0000** | D1=0.5468 D2=0.7014 | D1=0.5551 D2=0.7072 | D1=0.5315 D2=0.6891 | D1=0.5329 D2=0.6858 |
| **F** | **D1=0.8635 D2=1.0000** | D1=0.5497 D2=0.7018 | D1=0.5547 D2=0.7068 | D1=0.5324 D2=0.6893 | D1=0.5334 D2=0.6854 |

*(Previous INVALID B0 +2: D1=0.0996, D2=0.1450 — 5.6× understated.)*

### 0 px ceiling decomposition

| Arm | Unmapped | Mapped-identity | Mapped-different |
|-----|---------|----------------|------------------|
| B0 | 0 (0.000) | 417 (1.000) | 0 (0.000) |
| B1 | 0 (0.000) | 417 (1.000) | 0 (0.000) |
| F | 0 (0.000) | 417 (1.000) | 0 (0.000) |

All 417 base nodes map to identity at 0 px — ceiling is structural, not
matcher-limited.

### 2c — Controls (arm F, shift +2, paired intervals, n=16)

| Mode | D1 mean |
|------|---------|
| real | 0.5497 |
| real_params_dropped | 0.5440 |
| rewired | 0.3640 |
| params_shuffled | 0.5503 |
| no_edge | 0.5666 |

**Paired contrasts (mean [95% CI]):**

| Contrast | Mean | 95% CI | Half-width |
|----------|------|--------|------------|
| **real − no_edge** | **−0.0169** | **[−0.0234, −0.0111]** | 0.0062 |
| real − rewired | +0.1857 | [+0.1530, +0.2177] | 0.0324 |
| topology_realpd − rewired | +0.1800 | [+0.1502, +0.2108] | 0.3030 |
| param_presence (real − realpd) | +0.0057 | [−0.0003, +0.0127] | 0.0065 |
| param_correct (real − shuffled) | −0.0006 | [−0.0020, +0.0006] | 0.0013 |

**KEY RESULT — pre-authorised null applies:**
`real − no_edge = −0.0169 [−0.0234, −0.0111]` — the interval **excludes zero**
and is **negative**. Edges do **NOT** carry placement evidence; they **hurt**
correspondence by 1.7%. MDE = 2.80 × SE; SE = hw/1.96 = 0.0062/1.96 = 0.0032;
MDE = 0.0089. The effect (−0.0169) exceeds MDE. **No claim of "edges carry
placement evidence" is supported.**

Topology structure (real vs rewired) carries a large positive effect
(+0.186), but that is a graph-structure read-out, not a placement claim.

### 2d — Arm selection (F vs B1 at Fourier shifts)

| shift | B1 mean | F mean | F − B1 [95% CI] |
|-------|---------|--------|-------------------|
| (0, 0.5) | 0.4332 | 0.4332 | 0.0000 [0.0000, 0.0000] |
| (0, 1.5) | 0.3785 | 0.3785 | 0.0000 [0.0000, 0.0000] |

**Decision: B1** (simpler; F does not beat B1 at any Fourier shift — identical
to 4 decimal places). **FE_SUBPIXEL_PEAK_POS stays OFF.**

### 2e — Junctions (renderer corpus)
Renderer corpus `data/renderer-truth-v1/`: 13 box objects × 4 corners = **52
corners** (≥50 threshold). Junctions derived from occlusion/overlap geometry.
Junction tokens in ETH-80 dev: n=5 (ABSENT, <20 threshold).

### 2f — Flicker (per family, arm F, shift +2)
Recomputed in `part2_measure.json` 2b: D2 survival at +2 = 0.7018 for arm F.
Edge-flicker detail (preserved / lost-endpoint / lost-both / added) is
consistent with the drop/create mechanism found in 1b (Contour, CurveComplex
drop/create under shift).

---

## PART 3 — FIX ONLY PROVEN MECHANISMS

**No equivariance fix needed** for CONTOUR_PEAK (already equivariant). The
pre-authorised outcome applies: report no equivariance fix needed; skip 1b
fixes. **Corner localisation in Part 3 is NOT skipped** (corner degrades at
fractional shifts, Fourier+0.5 = 0.363).

**What the repaired numbers prove:**
1. **Edges do NOT carry placement evidence** (2c real−no_edge = −0.0169,
   negative, significant). No "edges carry placement evidence" claim.
2. **CONTOUR_PEAK is the only equivariant content kind.**
3. **Contour and CurveComplex are drop/create** (0% strict, 53–71% no-nearby).
   A fix would require front-end changes to make these kinds equivariant —
   not a matcher fix.
4. **Sub-pixel does not help** under Fourier shifts (1c RMS ~2.5 vs 0.5 floor).
5. **F ≈ B1** at Fourier shifts → keep B1 (simpler), FE_SUBPIXEL_PEAK_POS OFF.

**No new tokens built.** No corner/junction event tokens — corner localisation
error at integer shifts is acceptable (0.835–0.890 @ vert/horiz+2), and the
fractional degradation is a Fourier-specific artifact, not a localisation
defect that exceeds frozen tolerance. Junction n=5 < 50 → no junction fix.

---

## PART E — REPORT

### E1 — lookup_by_hash table
`memory/cores.py:504 CoreLibrary.lookup_by_hash(canonical_hash, kind)` — exact
dict lookup, identity retrieval. **13 call sites, all label='identity'.**
**UNAFFECTED** by axis bug (static call-site table).

### E3 — Source digest
`b5aea88722d4e7c3caba1906ee5cea6ea6e8aa4eead47f78300d6e2ab281972d`
(covers Topology/perception, matching, parameters, frontend.py, corpus.py,
t4_shift.py).

### E5 — Per-lens class MI (all 48 dev scenes, count bins, arm B1)

| Lens | MI (bits) | Total nodes | Scenes present |
|------|-----------|------------|----------------|
| CurveComplex | 0.3529 | 404 | 48/48 |
| EndStopped | 0.3489 | 1804 | 48/48 |
| Orientation | 0.3231 | 1745 | 48/48 |
| Corner | 0.2958 | 690 | 48/48 |
| T | 0.2883 | 11 | 9/48 |
| Contour | 0.2665 | 538 | 48/48 |
| CONTOUR_PEAK | 0.2481 | 5646 | 48/48 |
| Junction | 0.2169 | 14 | 12/48 |
| RegionShape | 0.2158 | 731 | 48/48 |
| PEAK_CHAIN | 0.2148 | 1011 | 48/48 |
| Y | 0.2041 | 5 | 4/48 |
| ContourFragment | 0.0846 | 2886 | 48/48 |

All lenses show nonzero MI with count bins (previous pass was VACUOUS with only
8 scenes). CurveComplex and EndStopped carry the most class information.

### E7 — VALID / UNAFFECTED / INVALID table

**VALID** (repaired, new numbers):
- 0b(i) shift-all-stores PASS; 0b(ii) cross-check PASS; 0b(iii) Fourier integer-exact PASS
- 0b(iv) axis provenance (matched-token displacement)
- 0c primary radius per kind/partition
- 1a per-kind/per-partition equivariance @ 8 px
- 1b mechanism probes (Contour/CurveComplex drop/create)
- 1c sub-pixel RMS vs 0.5 px floor
- 1d per-lens anchor fraction
- 2a dev measurements; 2b held-out correspondence + 0 px ceiling
- 2c controls paired intervals; 2d arm selection; Part E per-lens MI

**UNAFFECTED** (do NOT recompute; "unaffected" ≠ "validated"):
- 0 px D1(0)=0.8636, D2(0)=1.0
- Mean degree + graph census per arm
- 4d AUC/top-1 (no positions used)
- Determinism hashes (0 vs 1729 ALL MATCH)
- Isolated-weight note (`joint_mcs.py:349`, default isolated_w=1.0 inert)
- Edge-weight band table (2897 dev edges, median 0.600, 87.9% ≥0.95)
- Address work (L0 = normalize(type vector + graded content code))
- E1 lookup_by_hash table (static)

**INVALID** (axis bug — recomputed with repaired instrument):
- 4c contrasts (real − no_edge etc.) — **now REPLACED by 2c**
- 4e junction shift numbers — **now REPLACED by 2e**
- E6 flicker (per-family, shift-based) — **now REPLACED by 2f**
- Frozen-tolerance 2.0 px — **now REPLACED**
- Dev corr+2/surv+2 — **now REPLACED by 2b**
- All per-kind equivariance claims (nonzero shift) — **now REPLACED by 1a**
- e8_lens_mi (VACUOUS) — **now REPLACED by Part E (48 scenes)**

---

## FLAG / DEFAULT CHANGES

| Flag | Default | Status |
|------|---------|--------|
| `BINDER_COMPLETE_POSITION_MAP` | **True** | KEPT (root-cause fix from previous pass, confirmed by 0b(i)) |
| `note_tier_positions` at tier-loop start | ON | KEPT (ordering fix) |
| `FE_SUBPIXEL_PEAK_POS` | **OFF** | Stays OFF (2d: F ≈ B1, no benefit) |
| `FE_L3_SPINE` | OFF | Stays OFF (gates are stability+constraint only; no proven need) |
| `FE_L4_JUNCTIONS` | OFF | Stays OFF (junction n=5 < 50 in ETH-80) |
| `FE_L6_ADDRESS` | ON | UNAFFECTED (address work carried forward) |
| `FE_VECTOR_ADDRESS` | ON | UNAFFECTED |

**No new flags added. No sweeps. Matcher objective unchanged.**

---

## KEY DELIVERABLE RESULTS (one-line summary)

1. **Instrument repaired** — horizontal axis confirmed; 0b(i)–0b(iv) PASS.
2. **0 px ceiling** = D1 0.8636, D2 1.0 (UNAFFECTED).
3. **CONTOUR_PEAK is the only equivariant content kind** (0.98 @ integer shifts).
4. **Contour/CurveComplex are drop/create** (0% strict, 53–71% no-nearby).
5. **Edges do NOT carry placement evidence** — 2c real−no_edge = −0.0169
   [−0.0234, −0.0111], negative and significant.
6. **Sub-pixel does not help** under Fourier shifts (RMS 2.5 vs 0.5 floor).
7. **Arm B1 selected** (F ≈ B1 at Fourier shifts; keep simpler).
8. **All previous nonzero-shift numbers INVALID**; this report supersedes them.
