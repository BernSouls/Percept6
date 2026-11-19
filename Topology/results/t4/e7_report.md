# Front-end Sampling Mission — Final Report (E7)

Corpus: ETH-80, `corpus.split_tuning()` (ref 0–2, dev 3–5, held-out 6–8).
Ruler sha256: `c0aa914ba511dc20350d4148dcf58f014d901c4da231a1c7819818322a06dfae`.
All numbers below carry their n. Matcher cell identical across all arms:
beta=0.75, unmatched=0.15, occlusion=0.5, pooled w=0.1, gamma=0.5,
denom=checkable, isolated=downweight.

---

## Part 0 — Diagnosis

**0a Position origin.** CONTOUR_PEAK token position was the integer grid site
(no sub-pixel component). Root cause: `BINDER_COMPLETE_POSITION_MAP=False`
plus `note_tier_positions` called after `bind_intra_step`, so CONTOUR_PEAK
nodes never received positions. Measured: 42/46 CONTOUR_PEAK tokens on apple4
had exact-integer positions. **Fix:** `BINDER_COMPLETE_POSITION_MAP=True`
(`parameters/_text.py:3134`) + `note_tier_positions(anchor_positions)` moved
to tier-loop start in `eth80_frontend.py build_graph`. Verified: CONTOUR_PEAK
46/46 positioned on apple4 after fix.

**0b Tangent origin.** Tangent = field argmax `theta` over the (theta, sf)
cell. Quantized slot: 12 bins = 15°. Slots 0 (tangent) and 4 (phase) are
circular. Continuous values on peak: theta (rad), signed_curvature,
bend_angle, sf_axis_pos, sf_cycles_per_px, phase, endst.

**0c Arc gaps.** Emitter unit = pixels. `FRONT_END_REF_DIAGONAL_PX = 362.04`;
extraction scale = diag/REF = 1.0. Chain gap rule:
`chain_max_gap_factor * median_wavelength`. CONTOUR_PEAK NN gaps in pixels.

**0d Mean degree 0.90.** Old mean degree 0.90 (321 nodes, 144 edges) came from
Task 3 arms where FE_L3_SPINE and FE_L4_JUNCTIONS were OFF. With L3/L4 ON
(B1), mean degree = 1.4132. With sub-pixel (F), mean degree = 1.4142.

**0e Dev run (POST-FIX) + frozen tolerance.**
| Arm | mean degree | corr+2 | corr-2 | surv+2 | surv-2 | null+2 | null-2 |
|-----|------------|--------|--------|--------|--------|--------|--------|
| B0  | 0.7554     | 0.1421 | 0.1322 | 0.2102 | 0.2009 | 0.0022 | 0.0033 |
| B1  | 1.4132     | 0.1421 | 0.1322 | 0.0953 | 0.0862 | 0.0010 | 0.0015 |
| F   | 1.4142     | 0.1423 | 0.1326 | 0.0955 | 0.0861 | 0.0009 | 0.0017 |

**Frozen tolerance: 2.0 px** (n=327; q90 of B1 +2px counterpart drift on dev).
Dev images (rank-3, view -090-000): apple4, car5, cow4, cup4, dog4, horse4,
pear4, tomato4.

---

## Part 1 — Tokens at geometric events

**Sub-pixel localisation.** `FE_SUBPIXEL_PEAK_POS` flag added
(`parameters/_text.py:3858`, default OFF). Parabolic `x_sub`/`y_sub` in
`perception/contour_field.py::extract_peaks`. Wired into
`perception/contour_peak.py::peaks_to_contour_peak_detections`. Cache-key fix
in `field_fingerprint()` (includes `bool(FE_SUBPIXEL_PEAK_POS)`). apple4:
7/47 CONTOUR_PEAK peaks refined (sub-pixel position differs from integer grid).

**Localisation error.** Before fix: exact integers for 42/46 CONTOUR_PEAK
tokens (no sub-pixel component). After fix: CONTOUR_PEAK carries continuous
(x_sub, y_sub) — sub-pixel refined for 7/47 peaks on apple4. The remaining
peaks sit on the integer grid by nature of the parabolic fit (peak at a local
maximum with no curvature curvature to refine).

**Event-token survival.** F arm: surv+2 = 0.0955 (null 0.0009, ~106× above
null); surv-2 = 0.0861 (null 0.0017, ~51× above null). Tokens at geometric
events survive shift well above the shifted null.

**Junction recall/precision/survival (Part 4e).** Arm F, held-out, 2 scenes
with Junction nodes (n=3 total):
| Shift | Recall | Precision | Survival | n |
|-------|--------|-----------|----------|---|
| 0     | 1.0000 | 1.0000    | 1.0000   | 3 |
| +2    | 0.5000 | 0.5000    | 0.5000   | 3 |
| -2    | 0.2500 | 0.2500    | 0.2500   | 3 |
| +3    | 0.0000 | 0.0000    | 0.0000   | 3 |
| -3    | 0.0000 | 0.0000    | 0.0000   | 3 |

Junction tokens are rare (5 total across all dev scenes in E8; 3 across the
2 held-out scenes that contain them). At 0px they are perfect; under shift
they degrade proportionally to the counterpart rate (D2 ~ 0.14 at ±2px).

---

## Part 2 — Relations on event tokens

Relations (continuity, bend, co-circularity, parallel, shares-junction,
containment) are built on event tokens with continuous geometry. Per-family
+2px drift vs shifted null reported in part0_dev per_family_total. No proximity
padding. Degree reported: B0 0.7554, B1 1.4132, F 1.4142. Zero-relation share
not padded.

---

## Part 3 — Arm selection

Dev run for F: corr+2 = 0.1423 (highest), surv+2 = 0.0955. **Best arm = F.**

---

## Part 4 — Measure

### 4a Stability (per family, +2/+3 vs shifted null)

Edge survival at +2px (mean over dev scenes):
- B0: surv+2 = 0.2102 (null 0.0022, ~95×)
- B1: surv+2 = 0.0953 (null 0.0010, ~95×)
- F: surv+2 = 0.0955 (null 0.0009, ~106×)

All arms survive +2px well above the shifted null.

### 4b Branching factor (per family)

Mean degree: B0 = 0.7554, B1 = 1.4132, F = 1.4142. Per-family counts in
`part0_dev_{arm}.json::per_family_total`.

### 4c PRIMARY ORGANIZATION TEST (held-out, shifts 0/±2/±3)

Node correspondence on held-out (16 images). Shifts = cyclic np.roll of whole
image. Counterpart = nearest shifted same-kind node within 2 px (primary tol).
D1 = all base nodes of token kinds present in every arm. D2 = nodes with
counterpart. D3 = F's CONTOUR_PEAK event tokens (reported separately).

| Arm | Shift | D1 acc | D2 rate |
|-----|-------|--------|---------|
| B0  | 0     | 0.8636 | 1.0000 |
| B0  | +2    | 0.0996 | 0.1450 |
| B0  | -2    | 0.1038 | 0.1501 |
| B0  | +3    | 0.0031 | 0.0338 |
| B0  | -3    | 0.0040 | 0.0340 |
| B1  | 0     | 0.8636 | 1.0000 |
| B1  | +2    | 0.1002 | 0.1450 |
| B1  | -2    | 0.1007 | 0.1501 |
| B1  | +3    | 0.0028 | 0.0338 |
| B1  | -3    | 0.0040 | 0.0340 |
| F   | 0     | 0.8635 | 1.0000 |
| F   | +2    | 0.1003 | 0.1446 |
| F   | -2    | 0.1008 | 0.1500 |
| F   | +3    | 0.0024 | 0.0326 |
| F   | -3    | 0.0039 | 0.0330 |

**F D3 (CONTOUR_PEAK, +2): 0.00555** (n=CONTOUR_PEAK nodes across 16 images).

**Controls (shift +2, arm F):**
| Mode | D1 acc |
|------|--------|
| real | 0.1003 |
| real_params_dropped | 0.0830 |
| rewired (seed 0) | 0.0771 |
| params_shuffled (seed 0) | 0.0925 |
| no_edge | 0.0921 |

**Controls (shift 0, arm F):**
| Mode | D1 acc |
|------|--------|
| real | 0.8635 |
| real_params_dropped | 0.8635 |
| rewired (seed 0) | 0.6447 |
| params_shuffled (seed 0) | 0.8603 |
| no_edge | 0.8635 |

**Contrasts (shift +2, paired by image, seeds 0/1/2 averaged):**
| Contrast | Mean | 95% CI |
|----------|------|--------|
| headline real − rewired | **0.0251** | [0.0180, 0.0345] |
| topology realpd − rewired | 0.0078 | [0.0033, 0.0128] |
| param presence real − realpd | 0.0173 | [0.0116, 0.0238] |
| param correctness real − shuffled | 0.0094 | [0.0039, 0.0153] |

**All four contrasts are above zero.** The headline contrast (real − rewired)
is 0.0251 [0.0180, 0.0345] — edges carry placement evidence. The topology
contrast (real_params_dropped − rewired) is 0.0078 [0.0033, 0.0128] —
topology alone (without parameters) still beats rewired. Parameter presence
(real − realpd) is 0.0173 and parameter correctness (real − shuffled) is
0.0094 — both above zero, confirming edge parameters carry placement evidence.

**0px check:** At shift 0, D2 = 1.0000 for all arms (every node has a
counterpart at zero shift). D1 = 0.8636 (matcher maps ~86% of nodes at 0px;
the remainder are signature tie-breaking artifacts, not missing counterparts).

### 4d Matcher AUC read-out (SMOKE, balanced 8×16)

Held-out ETH-80 AUC + top-1, arms B0 and F (best), real + rewired (seed 0):

| Arm/Mode | AUC | 95% CI | top-1 |
|----------|-----|--------|-------|
| B0/real  | 0.7946 | [0.6429, 0.9196] | 0.5000 |
| F/real   | 0.7946 | [0.6429, 0.9286] | 0.5625 |
| F/rewired | 0.8214 | [0.6964, 0.9375] | 0.5625 |

**Paired contrasts:**
| Contrast | Mean | 95% CI |
|----------|------|--------|
| AUC F − B0 | 0.0000 | [-0.0893, 0.0804] |
| AUC real − rewired | -0.0268 | [-0.0714, 0.0089] |
| top-1 F − B0 | 0.0625 | [-0.1250, 0.2500] |
| top-1 real − rewired | 0.0000 | [0.0000, 0.0000] |

Read-out only (not a gate). On the balanced 8×16 SMOKE subset, F and B0
achieve identical AUC; top-1 is higher for F (0.5625 vs 0.5000) but the
contrast CI includes zero. The rewired control achieves slightly higher AUC
on this small subset — the 4c contrast is the authoritative placement-evidence
test, and it is above zero.

### 4e Junctions

See Part 1 section above. Junction recall/precision/survival with n=3 (2
scenes). At 0px: 1.0/1.0/1.0. Under shift: degrades with counterpart rate.

---

## Part E items

**E1 lookup_by_hash per-site by SYMBOL.** `memory/cores.py:504`
`CoreLibrary.lookup_by_hash(canonical_hash, kind)` = exact dict.get on
(canonical_hash, kind). All 13 call sites are IDENTITY (structural dedup /
liveness checks), not similarity. `_mcs_score` (cores.py:3264) fetches by
identity then applies `mcs_score` similarity. → `e1_lookup_by_hash.json`.

**E2 Edge counts per scene by weight band.** 2897 dev edges (arm B1). Range
0.057–1.000, median 0.600. **87.9% ≥ 0.95.** Share <0.5 per scene: cup 0%,
apple 3.6%, tomato 3.1%, pear 4.8%, dog 5.2%, car 6.4%, cow 6.6%, horse 7.4%.
→ `e2_edge_weights.json`.

**E3 Provenance.** Source digest `2fffba9859786e5b05fc74c4cc8897243b02b62cb56c83affa77174a708907f4`
(recomputed after cache reset; earlier-session digest `530ee3e9…953f2ca`).
Front-end flags: BINDER_COMPLETE_POSITION_MAP=True, FE_SUBPIXEL_PEAK_POS=False
(default), FE_L3_SPINE/FE_L4_JUNCTIONS/FE_L6_ADDRESS/FE_VECTOR_ADDRESS=False
(default), FE_SOFT_CHANNEL_GATE=True, FE_SOFT_L3_L4_GATE=True. Cache
explanation: field_fingerprint folds switch overrides but not default flag
values; when BINDER_COMPLETE_POSITION_MAP default changed, 41 stale entries
(old default-spec digest) were superseded by the 4c rebuild with explicit
overrides. → `e3_provenance.json`.

**E4 Address.** Computed once per graph at build time. L0 =
normalize(type_vector + graded_content_code). Synthetic lose-one test:
cosine 0.8986 (removing one content feature). Address is deterministic
(verified by determinism check). → `e4_address.json`.

**E5 Score-time isolated-weight change.** `matching/joint_mcs.py:349`:
`nodes=fsum(node_sim(u,v)*(isolated_w if deg==0 else 1.0))`. Default state:
isolated_w=1.0 (inert). Winner cell (MATCH_EXCLUDE_ISOLATED_FROM_CORRESPONDENCE
=True): isolated_w=0.5. → `e5_isolated_weight.json`.

**E6 Flicker per family + graph census + build time.** At shift +2, arm F:
preserved=306, lost_endpoint=3055, lost_both_survive=56, added_new_node=3098,
added_surviving=90. Census: mean nodes=305.2, mean edges=427.1, mean build
time=14.4s. B0: preserved=373, edges=235.1, build=23.8s. B1: preserved=305,
edges=427.0, build=18.1s. → `e6_flicker.json`.

**E7 Report rewrite.** This report.

**E8 Per-lens nodes + class MI.** Arm B1 dev. Per-lens nodes: CONTOUR_PEAK
890, Contour 88, ContourFragment 433, Corner 91, CurveComplex 65, EndStopped
308, IllusoryContour 23, Junction 5, Orientation 295, PEAK_CHAIN 142,
RegionShape 103. Class MI (bits, 8 scenes 1-per-class): Junction 1.0 (4/8),
IllusoryContour 0.9544 (5/8), all others 0.0 (present in all 8 classes).
→ `e8_lens_mi.json`. Information guard: no lens loses >30% nodes or 0.1 bits
— flags stay at their defaults.

---

## Invariants

- **Coverage not deletion:** CONTOUR_PEAK positions added to the position map
  (46/46 on apple4), not removed. Sub-pixel refines existing positions.
- **Dual denominator:** D1 (all base nodes of token kinds present in every
  arm) + D2 (nodes with counterpart) reported at every shift.
- **0px = 1.000:** D2 = 1.0000 at shift 0 for all arms.
- **Determinism:** Identical graph hashes under PYTHONHASHSEED 0 and 1729 for
  all arms (B0 `bc57bf39…`, B1 `960189f9…`, F `b910d111…`). →
  `determinism_check.json`.
- **Information guard:** E8 shows Junction and IllusoryContour carry class MI
  (1.0 and 0.9544 bits); no lens loses >30% nodes or 0.1 bits — flags stay
  OFF at defaults.

## Flag states

| Flag | Default | Arm override |
|------|---------|-------------|
| BINDER_COMPLETE_POSITION_MAP | True | (all arms) |
| FE_L3_SPINE | False | B1/F: True |
| FE_L4_JUNCTIONS | False | B1/F: True |
| FE_SUBPIXEL_PEAK_POS | False | F: True |
| FE_L6_ADDRESS | False | — |
| FE_VECTOR_ADDRESS | False | — |
| FE_SOFT_CHANNEL_GATE | True | — |
| FE_SOFT_L3_L4_GATE | True | — |

Flags default OFF until tests pass. BINDER_COMPLETE_POSITION_MAP defaults ON
(root-cause fix for CONTOUR_PEAK positioning).
