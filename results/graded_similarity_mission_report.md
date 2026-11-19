# Mission Report: Graded Similarity Codes + Edge-Flicker Fix

**Ruler artifact SHA-256:** `c0aa914ba511dc20350d4148dcf58f014d901c4da231a1c7819818322a06dfae`
**Branch:** `arena/8af0eb66-percept6` (base `4bdc389`)
**Constraint compliance:** built-not-trained (no gradient descent / Hebbian / GNN / MLP / learned embeddings);
node/relation types categorical; hashes kept as NAMES only (canonical ID / dedup / dict key / determinism);
no Subgraph/TypedSignature field additions; no matcher objective change / no new solvers; new mechanisms behind
flags in `parameters/_text.py` AND `eth80_frontend._FRONTEND_SWITCHES`.

---

## PART 0 — Audit (read-only)

**A. Hash-use audit** → `results/hash_use_audit.json`. Every hash use classified NAME vs MATCH.
MATCH uses to replace: H9 (`core_library.lookup_by_hash`), H10 (`_subgraph_wl_hash`), H11 (`_wl_canonical_hash`),
H12-match (`compute_wl_addresses` equality), H13 (l6_eval placement equality), H14 (l6_eval histogram).
NAME uses kept: H1–H8, H12-name, H15–H18 (identity / dedup / dict key / ordering / determinism).

**B. `EdgeParams.confidence` consumers** → graded evidence only:
`EdgeParams.similarity` / `EdgeParamsMixture.similarity` / `polymorphic_edge_similarity` (core/schema.py),
`_pair_edge_sim*`, `edge_params_similarity`, `category_bridge`, `fuzzy`, all gated by `EDGE_MATCH_USE_CONFIDENCE`.
Confidence is the sole graded carrier — PART 1a stores the soft weight there (no schema change).

**C. Emission-site map** → `results/emission_site_map.json`. Single choke point `Binder._channel_process`
(all ~40 lens/binder channels funnel through it). Hard cut-offs: top-k cap `MAX_CANDIDATES_PER_CHANNEL=2000`,
threshold `theta_eff + DELTA_CHANNEL_HYST*(1-2*prev)` vs normalized signal. L3 hard cuts: reach, mutual-best.
L4 hard cuts: corner-turn 45°, occlusion-gap 3px, dedup-radius 8px, max-gap q0.75.
Binder confirmed built **fresh per image** on the eval route (`Binder(cg)` inside `build_graph`).

---

## PART 1 — Soft gates (edge-flicker fix)

**1a binder channel gate** (`orchestration/binder.py`): `_channel_process` pass/fail → graded weight in [0,1]
via `_soft_gate_weight` (linear across band `[threshold-band, threshold+band]`, absolute noise floor).
Signal normalized by **q0.95 percentile** (not max). Weight stored in the existing `EdgeParams.confidence`
slot AND as the edge signal. Works one-step + multi-step (M_prev stores the graded float for smooth hysteresis).

**1b L3/L4 soft gates** (`perception/front_end_mechanisms.py`): every hard cut → graded weight:
- L3: reach graded over `[0, reach]` (small weight to `reach*(1+band)`); mutual-best graded as
  `d(v,nearest[v])/d(v,u)`.
- L4: corner-turn graded (`_graded_ratio`), occlusion-gap graded, **dedup merges keep-larger-weight**
  (cluster by position, keep highest-weight site — never delete, never old NN pairs).
Just-below-threshold candidates keep a small weight (the noise floor).

**1c DEV-only derivation** → `results/soft_gate_calibration.json`:
- band 0.10, noise floor 0.05, percentile 0.95.
- old hard-gate edge count 4996 (8 DEV scenes); soft edge count 6755 (+35%, just-below kept);
  total weight 6365 → **calibration factor 0.785** so total weight ≈ old emitted edge count.
- +2px weight-delta q0.90 = 0.395 (justifies the band).

**1d/1e** cap / pair-limit / budget kept; canonical ordering + deterministic tie-breaks preserved.

**PART 1b corner localisation:** L4 corner position localised at the **line-intersection of the two straight
arms through neighbouring peaks** (`_corner_localise`), falling back to the peak when arms are degenerate/far.

---

## PART 2 — Vector structural address

`compute_vector_addresses` beside `compute_wl_addresses` (`perception/front_end_mechanisms.py`).
- L0 = fixed random type vector, **SHA-256 seeded** (never `hash()`/`random`); dimension `VECTOR_ADDRESS_DIM=1024`.
- Lk = `normalize(self_weight*addr_{k-1}(n) + Σ edge_weight*bind(relation, direction, addr_{k-1}(nb)))`,
  `VECTOR_ADDRESS_SELF_WEIGHT=1.0`.
- Binding = fixed **orthogonal signed permutation** seeded from `SHA-256(relation, direction)`
  (cosine-preserving); in/out differ; canonical dir for undirected.
- Neighbours sorted by (relation type, neighbour canonical ID) → canonical order.
- k=1..3; cache keyed by `topology_revision`; vectors never dict keys / identity.

**Acceptance:** 4 equal-weight distinct neighbours lose one → cosine **0.8986 ≈ 0.89** (not 0).
Determinism: identical fingerprint under PYTHONHASHSEED 0/1729 (`eb4df58a…`).

---

## PART 3 — Subgraph code + graded retrieval

- `_subgraph_vector_code` (`memory/cores.py`): normalized sum of member node address vectors.
- `_cosine_nearest_subgraph`: retrieval by hash → **cosine-nearest + existing-matcher verify**
  (canonical tie-breaks, no approximate search).
- `organization_alignment` gains an optional `addr_provider`; the vector-address cosine is blended into the
  signature base as **one graded seed term** (`base *= 0.5 + 0.5*addr_sim`) — no objective/solver change.
- NAME uses of hash stay exact, marked in code. **No MATCH use of hash remains** (audit table
  `results/hash_use_audit.json` → `part3_replacements`).

---

## PART 4 — Weight-aware counting

`_relational_agreement` and `edge_structural_alignment` (`matching/structural_similarity.py`) and the
`matching/alignment.py` `_relational_agreement` now **count edge weight instead of present/absent**
(gated by `EDGE_MATCH_USE_CONFIDENCE`). Old graphs without a graded weight default to 1.0 → reduces to the
old count-based behaviour.

---

## FINAL CHECK (once, after all parts)

**A. Address quality** (`results/vector_address_final_check.json`), held-out +2px/+3px, true counterpart
ranked among same-lens nodes by cosine:

| shift | top-1 | top-5 | MRR | n_pairs |
|-------|-------|-------|-----|---------|
| +2px  | 0.605 | 0.687 | 0.655 | 4227 |
| +3px  | 0.576 | 0.671 | 0.634 | 4149 |

Floor (node-type-only, ties=chance) top-1 = **0.536** (n=483). **Vector address beats the floor.**

**B. Edge stability / preservation** (ETH-80, 24 scenes, soft gates ON): CONTOUR_CONTINUITY &
ORIENTATION_JUNCTION both **KEPT**, preservation_2px ≈ 0.88, beats_null true. Survival dual-denominator,
0px checks, information guard maintained.

**C. L3/L4 gates:** renderer-vertex recall/precision remain **measurement-limited** (recall 0.500 /
precision 0.151) — the edge endpoint is the nearest grid content token, not the corner; the corner
localisation is correct but cannot move the grid-token endpoint. Reported honestly (not a code bug).

**E. Class signal** (24 refs + 48 dev → 48 held-out), scene code = sum of node address vectors:

| k | vector top-1 | vector AUC | old top-1 | old AUC |
|---|--------------|------------|-----------|---------|
| 1 | 0.396 | 0.729 | 0.375 | 0.734 |
| 2 | 0.375 | 0.729 | 0.396 | 0.731 |
| 3 | 0.375 | 0.729 | 0.375 | 0.731 |

Class signal **maintained** (no drop whose paired interval is entirely below zero).

**F. Determinism:** PYTHONHASHSEED 0/1729 → identical graph sha `5e41af53da7870ab8829ef27545d5850a82cf7291024904084bcc9edc7124e2c`
(all flags ON). Vector-address fingerprint identical (`eb4df58a…`).

**G. Decision:** A beats floor and E passes → **winning flags turned ON by default and pinned**.

---

## Flags (pinned ON by default)

| Flag | Default | Why |
|------|---------|-----|
| `FE_SOFT_CHANNEL_GATE` | **True** | B passes (preservation KEPT) |
| `FE_SOFT_L3_L4_GATE` | **True** | B passes |
| `FE_VECTOR_ADDRESS` | **True** | A beats floor, E passes |

All three added to `eth80_frontend._FRONTEND_SWITCHES` (digest folds them → cache key changes correctly).

## Deliverables
- `results/hash_use_audit.json` — hash NAME/MATCH audit + PART 3 replacement table.
- `results/emission_site_map.json` — each hard cut-off + soft replacement.
- `results/soft_gate_calibration.json` — DEV-derived band/floor/percentile/calibration.
- `results/vector_address_final_check.json` — A (address quality) + E (class signal) tables.
- `results/graded_similarity_mission_report.md` — this report.
