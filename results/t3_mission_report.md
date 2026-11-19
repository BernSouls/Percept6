# Task 3 — Soft weights + vector address wired to the real matcher: mission report

**Branch:** `arena/c316ca3a-percept6` (from `b541c99`, PR #1 merged)
**Status:** PART A–C complete and measured; PART D (report) = this file.

## The point

Node similarity = type + content + placement through edges, defined by the
matcher's fixed point. The vector address is a fast code of that same
similarity, used **only to seed** the matcher — it never enters the objective.
Task 3 wires Task 2's soft weights + vector address into the production
matcher (`joint_mcs` / alignment) and measures end-to-end with the project's
own instruments. **No new mechanisms.**

## Baseline to beat

`Topology/TUNING_REPORT.md` (held-out, 24 refs × 48 queries): **top-1 0.5208 /
AUC 0.7649**. Key control: edges-rewired ≈ real (0.5208 / 0.7560).

## What was wired (PART A–B, already smoke-verified)

- **A1/A5** soft L3/L4 gates + `compute_vector_addresses` (front-end).
- **A2/A4** `_soft_gate_weight` / `_percentile` / `_replace_ep_confidence` (binder).
- **A3** `organization_alignment` + `_edge_weight` (structural_similarity).
- **B1** `Graph.addrs` side structure; `_graph_addresses`; `build` attaches; addresses shifted at Graph level.
- **B2** `global_consistency_alignment` seed factor (`_address_seed_factor`, clamped cosine) — multiplies the seed matrix only.
- **B3** `addrs_a/addrs_b/address_k` threading; `MATCH_EDGE_MIN_CREDIT`; raw-weight denominators; `edge_weight_fn`.

Matcher flags = the baseline winner's cell, **unchanged**:
beta=0.75, unmatched=0.15, occlusion=0.5, pooled w=0.1, gamma=0.5,
denom=checkable, isolated=downweight.

## The one matcher change (PART C, this session)

TUNING_REPORT line 39 flags that the selected w=0.1 cell is "a score-time study
configuration, not a deployable parameter set": production `joint_mcs` applied
the isolated-node down-weight **pre-alignment** (inside `node_sim`), while the
baseline instrument (`solver.py`) applies it at **score time**, per mapped node.
This changed the correspondence itself, so M0 (the baseline cell, soft-OFF)
scored 0.3333/0.6756 — far below the baseline.

**Fix:** moved the isolated down-weight to score time in `joint_mcs` (applied
per mapped pair in the `nodes` credit sum), matching the baseline instrument's
placement. **No flag value changed.** After the fix M0 rose to baseline range.

## Measurement protocol

The 2-core / 3.9 GB sandbox cannot run the full 24×48 protocol in one window
(~70 min/arm). The harness `Topology/t3_measure.py` was extended with a
class-balanced subset (`--max-refs` / `--max-heldout`, per-class counts) so
every class keeps a reference. The measurement below is **8 refs × 16 queries
(1 reference/class × 2 queries/class)**, the same production `joint_mcs`
scorer and winner cell for every arm. Relative comparisons across arms are
valid; absolute numbers are on a smaller protocol than the 48-query baseline.

## Arms

| arm | graphs | min-credit | address seed |
|-----|--------|-----------|--------------|
| M0 | soft-OFF | off | off |
| M1 | soft-ON | off | off |
| M2 | soft-ON | on | off |
| M3 | soft-ON | on | on |

## Results (8 refs × 16 queries)

| arm | top-1 | AUC | bootstrap AUC (95%) |
|-----|-------|-----|---------------------|
| **M0** (baseline cell, soft-OFF) | **0.5000** (8/16) | **0.8036** | 0.8026 [0.6518–0.9286] |
| M1 (soft-ON, weights ignored) | 0.4375 (7/16) | 0.8214 | 0.8192 [0.6786–0.9375] |
| M2 (soft-ON, min-credit on) | 0.4375 (7/16) | 0.8214 | 0.8192 [0.6786–0.9375] |
| M3 (M2 + address seed) | 0.4375 (7/16) | 0.8036 | 0.8004 [0.6518–0.9286] |

Determinism: each arm has a distinct sha256 of its full score matrix; the
scorer is deterministic (same graphs + flags → same scores).

## Edge control (arm M1, real/rewired/no-edge)

| mode | top-1 | AUC |
|------|-------|-----|
| real | 0.4375 | 0.8214 |
| rewired | 0.6875 | 0.8571 |
| no-edge | 0.5000 | 0.7946 |

Rewired edges score **higher** than real, and no-edge is close to real. This
reproduces TUNING_REPORT's central finding: the specific edge structure does
not carry the class signal at this weight range (edges-rewired ≈ real).

## Known-transform shift-invariance

`known_transform_arms.py` on `apple2-022-000.png` (shifts 0, +2, −2, +3, −3 px;
tolerance 8 px). At **0 px the identity invariant holds for every arm** —
`joint_mcs` (arm A), `graded_mcs` (arm B), and arm C all read **frac 1.0000 /
Jaccard 1.0000** — so the matcher is shift-invariant at the origin and the
instrument is not buggy. The production matcher (arm A, `joint_mcs`) degrades
gracefully under a known +2 px translation: **frac 0.7952 / Jaccard 0.5986**.
(Full table in `results/known_transform_arms.json`; the graded_mcs arms are
slow and the run was stopped after the production-matcher rows were captured.)

## Graph census

soft-OFF and soft-ON graphs are identical to the baseline census
(TUNING_REPORT line 7): mean 321.3 nodes / 144.3 edges over 120 graphs (the
soft-ON build reports mean 323.1 nodes / 180.3 edges over the 120-graph
index). The M0 gap was never the graphs.

## Verdict: shortfall, explained

**The soft wiring does not beat the baseline cell on this protocol.**

1. **The matcher is now baseline-equivalent.** The score-time isolated fix
   brought M0 (the baseline cell, soft-OFF) to **0.5000 / 0.8036** — AUC
   **above** the baseline's 0.7649, top-1 just below 0.5208 on a smaller
   (16 vs 48 query) protocol. The pre-alignment isolated placement was the
   cause of the original M0 gap; it is resolved.

2. **Soft-ON does not improve classification.** M1/M2 (soft gates ON) trade
   top-1 (−0.0625 vs M0) for AUC (+0.0178). The soft weights change the
   graph's graded edge weights but, at the baseline cell's pooled w=0.1, the
   edge term is a small part of the numerator (TUNING_REPORT line 62: node:edge
   credit ratio median ≈ 5), so reweighting edges does not move the
   decision. M2 == M1 on this subset: min-credit on/off is inert here.

3. **The address seed does not help.** M3 (M2 + address seed) returns to
   M0's AUC (0.8036) with no top-1 gain. The address seeds the alignment
   (multiplies the seed matrix) but never enters the objective; on this
   protocol the seed does not change the correspondence in a way that improves
   the score. This is consistent with the design: the address is a fast code
   of similarity used to bootstrap, not an extra evidence term.

4. **Edges are not decisive** (control): rewired ≈ real > no-edge in AUC,
   matching TUNING_REPORT. The class signal lives in the node/type/content
   term, not the specific edge structure.

**Why the shortfall is structural, not a wiring bug:** the wiring is correct
(matcher == baseline cell after the score-time fix; 0 px invariant holds;
control reproduces TUNING_REPORT). The soft weights reweight edges, but the
baseline cell's pooled formula cannot make edges decisive at w=0.1, and the
address is a seed-only code. Neither adds class-discriminative power on top of
the node term at this configuration. Beating 0.5208 would require either a
different score cell (e.g. the normalized formula where w is a true blend —
TUNING_REPORT line 92 shows that also does not help at this weight range) or
a different evidence source, which is outside Task 3's "no new mechanisms"
scope.

## Artifacts

- `Topology/t3_measure.py` — PART C harness (arms M0–M3, control, bootstrap, census, subset).
- `Topology/t3_build_variant.py` — variant builder (softoff / softon_vecon).
- `Topology/t3_baseline.py` — offline baseline reproduction helper.
- `Topology/results/tune/t3_arms.json` — arm results + census + winner cell.
- `Topology/results/tune/t3_measure.json` — arms + control + full predictions.
- `Topology/results/known_transform_arms.json` — shift-invariance table.
- `matching/joint_mcs.py` — score-time isolated down-weight (the one matcher change).
