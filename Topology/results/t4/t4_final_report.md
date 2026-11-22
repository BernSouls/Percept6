# Task 4 — Final Report: Do Edges Give Placement Evidence?

**Branch:** `arena/c316ca3a-percept6` · **Commit:** `202db9a` (and prior `c62719e`, `caedf1e`)
**Question (exact wording):** CONSISTENCY — real edges agree with true correspondence
(real−rewired shows WRONG edges hurt). TIE-BREAKING/OVERRIDE — on nodes where node-sim
ties or ranks wrong node first, edges raise chance of mapping to true counterpart;
baseline = no_edge. Reported separately.

## HEADLINE ANSWER

**Edges give PLACEMENT EVIDENCE, but only as TIE-BREAKING/OVERRIDE on OUTRANKED nodes
(case c), not as a global win.** On 94.1% of nodes (STRICT-LEAD), node similarity already
places them perfectly and real edges slightly *reduce* accuracy (−0.036). On the 5.9% of
nodes where node-sim ranks the wrong node first (OUTRANKED), real edges strongly *raise*
the chance of mapping to the true counterpart (+0.367, permuted-robust). Wrong edges
(rewired) hurt everywhere (CONSISTENCY). At the aggregate held-out level real−no_edge is
slightly negative (−0.0196) because case (a) dominates; the edge benefit is concentrated
on the small OUTRANKED minority.

---

## PART A — CORRECTIONS TABLE

| Item | Correction | Result |
|------|-----------|--------|
| A1 | content-content preservation arithmetic | **4302/5910 = 72.8%** (was 4302/6350=67.7%). Per-family stability table generated (fam21 n=3731 preserved 2329; fam27 n=2363 preserved 1747; present_frac 0.77–1.0). |
| A2 | Class precedence (ABSENT-LIKE<0.50, EQUIVARIANT≥0.70 at r_cls, STATIC-WHEN-PRESENT static/persisters≥0.80, else MIXED) + bridge table | CONTOUR_PEAK/Corner/PEAK_CHAIN/RegionShape = EQUIVARIANT (in D_class); Orientation = STATIC-WHEN-PRESENT (in D_class); ContourFragment/EndStopped = MIXED (not in D_class); Contour/CurveComplex = ABSENT-LIKE; Junction/T/Y = ABSENT. Two denominators D_all, D_class reported. |
| A4 | tolerance/parameter drift | tolerance = **6.0** (q90 matched-edge param change, UNCENSORED); **4x = 24.0**. drift median=1.0, mean=2.07, p95=6.85 (n=84, comparable=84, non-comparable=0). Branching factor (same-lens candidates): mean=67.3, **median=58**, p95=142 (n=6207). Fourier +0.5 drift 0; +1.5 median=0.5 q90=4.7. LIMIT: pure translation leaves relation params (gap/angle) invariant; only offset shifts. |
| A9 | dev vs held-out citation | dev −0.0019 [−0.0117,+0.0099]; held-out −0.0196 [−0.0281,−0.0123]. Dev = one view (−090-000, 24 scenes); held-out = two views (16 images). |
| A11 | determinism hashes | Recomputed on rebuilt graphs; PYTHONHASHSEED 0 vs 1729 ALL MATCH (B0 `bc57bf39…`, B1 `960189f9…`, F `b910d111…`). |

(A3 mechanism probes, A5 INCONCLUSIVE, A6 sub-pixel, A7 dx=−0.69, A8 0px decomposition,
A10 weights — cited from prior pass `part0_corrections.json` / `part1_edge_diag.json`; not
re-measured here. A8 matcher precision at 0px recomputed in INVARIANTS below.)

---

## PART B — RESULTS (dev, n=24 scenes, arm B1 unless noted)

### B1 — Three-case placement test (PRIMARY)
Case membership derived from node_sim only → identical across modes (asserted).

**Case shares (n=7819 class nodes):** (a) STRICT-LEAD **7354 (0.941)**; (b) EXACT-TIE **2 (0.000)**;
(c) OUTRANKED **463 (0.059)**. Excluded: not_dclass=5252, no_counterpart=672, ambiguous=237.
node_sim range [0,1]. Predicted no_edge accuracy **0.9406**.

**Measured accuracy per case:**
| mode | (a) STRICT-LEAD | (b) EXACT-TIE | (c) OUTRANKED |
|------|-----------------|---------------|--------------|
| real | 0.9467 (n=7354) | 0.5000 (n=2) | **0.6285** (n=463) |
| no_edge | 0.9823 | 0.0000 | **0.2613** |
| rewired (mean s0,s1,s2) | ≈0.61 | 0.0000 | ≈0.21 |
| params_shuffled (mean) | ≈0.937 | 0.5000 | ≈0.52 |

**Contrasts (paired bootstrap, k=2 corrected for (b),(c)):**
| case | real−no_edge | real−rewired (s0/s1/s2) |
|------|--------------|------------------------|
| (a) STRICT-LEAD | **−0.0356 [−0.0413,−0.0301] hw=0.0056** | +0.335/+0.342/+0.330 |
| (b) EXACT-TIE | 0.5000 [0,1] (n=2, **ABSENT**) | 0.5000 [0,1] |
| (c) OUTRANKED | **+0.3672 [+0.3196,+0.4147] hw=0.0475** | +0.406/+0.421/+0.434 |

Share-weighted net effect (real−no_edge) = **−0.0116** (case (a) dominates).
**Pre-registered test:** real−no_edge>0 on (c) under permuted order → **edges give placement
evidence (TIE-BREAKING/OVERRIDE) on OUTRANKED nodes.** (b) ABSENT (n=2).

### B2 — Structure-only test (node_sim → same-lens indicator)
Case shares: a=7437 (0.923), b=2, c=617 (0.077). Identity assert PASS.
| mode | (a) | (c) | chance |
|------|-----|-----|--------|
| real | 0.3913 | 0.2334 | a=0.026, c=0.030 |
| no_edge | 0.2293 | 0.0600 | |
| rewired | ≈0.095 | ≈0.04 | |
**real > no_edge > rewired > chance** on both cases → **edges carry placement information on their own** (without node-sim).

### B3 — Identifiability
Rank-first rates (true counterpart ranked best among same-lens candidates):
| scheme | (a) STRICT-LEAD | (c) OUTRANKED |
|--------|-----------------|--------------|
| sig-only | **1.0000** (n=7437) | 0.0000 (n=617) |
| sig+1WL real | 0.9336 | 0.0227 |
| sig+1WL rewired | 0.9371 | 0.0227 |
WL rounds mean = 1.0. **Signatures alone identify 100% of STRICT-LEAD nodes; 1-WL adds little
(slightly hurts (a), barely helps (c)); real ≈ rewired → specific edge topology does not drive
identifiability.**

### B4 — Aperture on mismapped content nodes
along accuracy **0.9427** (n=6441); across accuracy **0.7796** (n=1615).
Mis-mapped 725: along 369 (50.9%), across 356 (49.1%). Accuracy by hops-to-event ~0.88–0.95
(descriptive; no causal claim).

### B5 — Credit decomposition
| case | node_sim_true | node_sim_mapped | penalty | edge_credit_mapped/true |
|------|--------------|-----------------|---------|------------------------|
| (a) STRICT-LEAD | 0.9880 | 0.9676 | **−0.0204** | 2.94 / 2.97 |
| (c) OUTRANKED | 0.6633 | 0.6038 | −0.0595 | 3.06 / 3.30 |
**Denominator penalty on (a) present** (edges pull to lower-node-sim node) → P3-5 condition met.

### B6 — Seed pre-check (FE_VECTOR_ADDRESS ON)
**Seed pathway INERT** — 0/13373 mappings changed, 0/48 scenes. P3-3 first clause UNEVALUATED.

### B7 — Emission vs selection attribution
emitted mean=2261.4, selected mean=420.2, **dropped mean=1841.2 (81.4%)**.
**Budget binds 24/24 scenes**; family_seeds=0 changes selection 24/24; deciding key node-id dependent (structural).

### B8 — Isolated-node confound
degree-0 share mean=20.1%, but **mapping agreement 100%** (n=12136) → isolated nodes are NOT a confound (exclusion inert).

### B9 — Non-additivity (cited from part3_triggers.json)
P3-0: NO family fired (all 21 leave-one-out strongly negative −0.014…−0.030).

---

## PART C — IMPLEMENTED / SKIPPED

| Step | Trigger input | Status | Action |
|------|--------------|--------|--------|
| P3-0 family pruning | part3 (21 LOO) | NOT fired | skip |
| P3-1 host relations | part3 host−all | NOT fired (+2 −0.0075 [−0.0201,0.0036]) | skip |
| P3-3 graded params | A4 + B6 | **UNEVALUATED** (B6 seed inert; branching median 58 > 3) | skip |
| P3-4 selection stability | B7 (fires) + flip-rate | trigger fires; **flip rate = 0.0000** (node-id key ≡ replacement key) → change inert | **no production change** (do-no-harm; selection already deterministic) |
| P3-5 ambiguity-gated credit | B1 + B5 | conditions met (real−no_edge>0 on (c), <0 on (a), penalty on (a)) | **AUTHORISATION NOT GIVEN — do not implement**; options note + stop |

**No Part C production changes implemented. FINAL arm = B1.** Flags unchanged:
`FE_L3_SPINE=False, FE_L4_JUNCTIONS=False, FE_SUBPIXEL_PEAK_POS=False, FE_L6_ADDRESS=False,
FE_VECTOR_ADDRESS=False, BINDER_COMPLETE_POSIATION_MAP=True` (all defaults; Part B used
diagnostic toggles/harness wrappers, restored).

---

## FINAL HELD-OUT (once, after Part C; 16 images, 2 views)

**Node correspondence (arm B1):**
| shift | D1_pooled | D1_class | D2 |
|-------|-----------|----------|-----|
| 0 | 0.8638 | **1.0000** | 1.0000 |
| +2 | 0.5458 | 0.6733 | 0.7014 |
| −2 | 0.5541 | 0.6825 | 0.7072 |
| +3 | 0.5305 | 0.6551 | 0.6891 |
| −3 | 0.5319 | 0.6604 | 0.6858 |

**Controls (shift +2, FINAL=B1), contrasts with half-widths:**
| contrast | pooled | class |
|----------|--------|-------|
| real − no_edge | **−0.0196 [−0.0281,−0.0123] hw=0.0079** | −0.0252 [−0.0351,−0.0164] |
| real − rewired (s0) | **+0.1709 [0.1419,0.2006]** | +0.2501 [0.2142,0.2860] |
| real − rewired (s1) | +0.1853 [0.1485,0.2179] | +0.2607 |
| real − rewired (s2) | +0.1813 [0.1405,0.2209] | +0.2491 |
| real − params_shuffled (s0) | +0.0008 [−0.0017,0.0041] | +0.0034 |
| real − real_params_dropped | +0.0048 [−0.0023,0.0123] | +0.0119 [0.0012,0.0240] |

**Reading:** real−no_edge < 0 (aggregate, real edges reduce correspondence); real−rewired
strongly > 0 (CONSISTENCY: wrong edges hurt). The edge benefit is per-case (B1 case c),
not aggregate. LIMIT: pure translation leaves relation params unchanged; failing is
decisive, passing is necessary not sufficient.

---

## INVARIANTS

- **0px matcher precision = 1.0000** (n=24) — every node maps to itself at shift 0. **Pass** (not a finding).
- Coverage (not deletion): pooled content 1.0000, class-filtered 1.0000 (n=24). New-token count mean 4.67 (separate, non-content).
- D1(0) ceiling (content, identity) = 1.0000.
- Determinism: PYTHONHASHSEED 0 vs 1729 ALL MATCH (cited, prior pass).
- Information guard: no Subgraph/TypedSignature field additions; baseline pickles loadable.

---

## CONCLUSION

1. **CONSISTENCY: yes.** real−rewired is strongly positive on every case and both
   dev (+0.22/+0.23) and held-out (+0.17/+0.25). Real edges agree with true correspondence;
   wrong edges hurt.
2. **TIE-BREAKING/OVERRIDE: yes, on OUTRANKED nodes only.** real−no_edge = +0.367 [+0.32,+0.41]
   on case (c) (nodes node-sim ranks wrong), permuted-robust. **No effect on STRICT-LEAD**
   (−0.036, edges slightly disrupt the node-sim-already-correct placement) and **ABSENT on
   EXACT-TIE** (n=2).
3. **Aggregate:** real−no_edge = −0.0196 [−0.028,−0.012] (held-out) — because 94.1% of nodes
   are STRICT-LEAD where edges hurt. The edge placement evidence is real but concentrated on
   the 5.9% OUTRANKED minority.
4. **Edges carry placement information on their own** (B2 structure-only: real > no_edge >
   rewired > chance), but with full node-sim this information is only decisive where node-sim
   fails.
5. **No Part C production changes** — all triggers UNEVALUATED/not-fired/inert/NOT-GIVEN.
   FINAL arm = B1.
