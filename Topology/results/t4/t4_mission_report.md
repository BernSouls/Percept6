# Task 4 MISSION REPORT — Repair the truth instrument, then turn measured placement evidence into a net gain

Branch `arena/c316ca3a-percept6` (base `b541c99`, on top of `9bbcc69`).
**Outcome: the truth instrument is repaired and validated; no fix arm passes the
pre-registered keep rule, so NOTHING NEW IS SHIPPED — the unmodified objective is
kept as the reference, and the dev numbers below are the evidence.**

## Compute hours (stated first)

- Dev pkl rebuilds (B0+B1, 48 graphs): ~10.5 min
- Shifted B1 pkl rebuild (48 graphs, ±2): ~11 min
- Part 1 1b+1d (24 scenes × 2 shifts, matcher + node_sim): ~13 min
- Part 1 baseline + F0 arms (24 scenes × 2 shifts × 3 betas × 2 modes): ~14 min
- Part 1 1c jitter (8 scenes × 2 shifts × 3 seeds × 2 betas × 2 modes): ~6 min
- Total this mission ≈ **55 min** of measurement + ~22 min of graph rebuilds.
  Held-out pkls were NOT rebuilt (no arm is kept, so no held-out comparison of a
  new arm is warranted; the reference is unchanged).

## PART 0 — truth instrument repaired

`Topology/t4_shift.py::counterpart_map` repaired: one-to-one **mutual** nearest,
inf-search then radius applied after (`best_d < radius`), greedy smallest-distance
competition resolution, **NON-STRICT** ambiguity (`second_d <= 2*best_d`, no
`bd>0` gate). Added `radius_by_lens={lens:radius}`. Returns `(mapping, info, stats)`.
All 13 `Topology/t4_*.py` callers updated to unpack the 3rd value.

- **0b validation** (`part0b_validate.json`): ALL 4 PASS (T1 rate 1.0000, T2 dup
  ambiguous, T3 lone not flagged, T4 competing resolved=[0]).
- **0c per-kind primary radius frozen** (`primary_radius_dev.json`): CONTOUR_PEAK
  1.0, Contour 1.0, ContourFragment 1.0, Corner 2.0616, CurveComplex 1.0,
  EndStopped 1.5, Junction→8.0, Orientation 1.0, PEAK_CHAIN 1.5413, RegionShape
  1.7324, T→8.0, Y→8.0.
- **0d B1 on PRIMARY** (`b1_primary.json`), repaired map + per-kind radius, dev 24
  scenes ±2, n=7076:
  - PRIMARY: (a) STRICT-LEAD **6241 (0.882)**, (b) EXACT-TIE **0**, (c) OUTRANKED
    **25 (0.004)**. 8px: (a) 6494 (0.918), (c) 139 (0.020).
  - **Old (pre-repair, 8px, old map) → New (PRIMARY, per-kind, repaired):**
    (a) 7354(0.941)→6241(0.882); (b) 2→0; **(c) 463(0.059)→25(0.004)**, −438.
  - **CASE (c) IS ABSENT (<200 n) at both radii → headline restated before arms.**
  - Orientation all 0 (separate); MIXED (a)1416/(b)3/(c)382; ABSENT-LIKE 0.
- **0e order check** (`part0e_order.json`): structural identity 16/16; real
  mapping-change **3.06%**, no_edge **0.08%** → **tie order MATTERS** (F3 relevant).
- **0f/0g**: see `part0_closure_0f_0g.md` (6 claims withdrawn one line each;
  `MATCH_SYSTEMATICITY_WEIGHT` read at fuzzy.py:156, joint_mcs.py:235/:247,
  cores.py:923; isolated-weight handling at joint_mcs.py:171/:195-199/:208/:236/:343-350).

## PART 1 — measurements (gating the fixes)

**1b truth-free ambiguity gradients** (`part1_1b_1d.json`): eps (q90 shortfall of
true counterparts) = **0.0000** (true counterparts saturate node_sim ≈ 1.0).
- K_top: bin1 P(c)=0.0038 (n=6248), bin2 P(c)=0.0556 (n=18). **Usable gradient:
  NO** (top bin n<200).
- K_self: bin0 P(c)=0.0040 (n=6250), bin1 P(c)=0.0 (n=16). **Usable: NO.**
- margin terciles t1=0.0504, t2=0.1220.
→ **F2 (ambiguity gate g(u)) and F4 (ambiguity-aware selection) are UNEVALUATED**
(no truth-free gradient to gate on).

**1c JITTER ENVELOPE** (`part1_1c_jitter.json`), 48 runs each:
- beta=0.50: min −0.0645, median 0.0000, max +0.0204, stdev 0.0208.
- beta=0.25: min −0.0414, median 0.0000, max +0.0204, stdev 0.0136.
The reference pooled delta (−0.0172) is **inside** this envelope → the "edges
hurt" direction is NOT robust to tie-order jitter.

**1d params contrasts** (`part1_1b_1d.json`):
- STRICT-LEAD (n=6241): real−params_shuffled = +0.0016 [−0.0013,+0.0043];
  real−real_params_dropped = **+0.0072 [+0.0026,+0.0130]**.
- OUTRANKED (n=25): real−params_shuffled = −0.04 [−0.12,0.0];
  real−real_params_dropped = −0.04 [−0.16,+0.08].
- Pooled (n=6266): real−dropped = +0.0072 [+0.0022,+0.0124].
→ params carry a **small** net signal (dropping them costs ~0.7%); shuffling is
inert. Edges do NOT fix OUTRANKED (c).

**1a/1e/1f**: not run in full; the decision-relevant quantities are resolved by
1b/1c/1d. 1a (f_true/f_false, P(rel_false>rel_true)) is not separately measurable
because case (c) is ABSENT; 1e cross-shift stability and 1f bandwidth do not
change the keep/discard decision given the failing net below.

## PART 2 — F0 arms (the only lever with a live trigger)

`MATCH_SYSTEMATICITY_WEIGHT` = beta, three pre-registered fixed values. Dev PRIMARY,
n=6266, real vs no_edge (`part1_f0_baseline.json`):

| beta | STRICT-LEAD (a) real−ne | case (c) real−ne (n) | POOLED real−ne | D1 real−ne |
|---|---|---|---|---|
| **0.50 (reference)** | −0.0183 [−0.0221,−0.0141] | +0.24 [+0.08,+0.44] (25) | **−0.0172 [−0.0214,−0.0131]** | −0.0141 |
| 0.25 | −0.0026 [−0.0051,+0.0003] | +0.24 [+0.08,+0.40] (25) | **−0.0016 [−0.0043,+0.0013]** | −0.0010 |
| 0.75 | −0.0357 [−0.0412,−0.0308] | +0.20 [+0.04,+0.40] (25) | −0.0348 [−0.0401,−0.0295] | −0.0278 |

**Per-arm keep-rule evaluation** (share-weighted net interval above zero AND case
(c) lift ≥75% of the ORIGINAL reference's):
- **beta=0.50 (reference)**: net −0.0172, interval entirely below zero → **DISCARD**
  (kept only as the reference).
- **beta=0.25**: net −0.0016, interval **crosses zero (not above)** → **DISCARD**.
  Best arm: cuts the harm from −0.0172 to −0.0016 and makes STRICT-LEAD ≈ neutral
  (interval includes 0), but does NOT reach a net gain. Case (c) lift +0.24 ≥ 0.75×
  reference's +0.24, but case (c) is ABSENT (n=25) so this criterion is not
  independently evaluable.
- **beta=0.75**: net −0.0348 → **DISCARD** (harm increases with beta).

**F1 (absence discount)**: trigger is OUTRANKED harm — case (c) ABSENT (n=25) →
**UNEVALUATED** (no measurable target).
**F2 / F4**: no usable truth-free gradient (1b) → **UNEVALUATED**.
**F3 (selection stability)**: 0e shows tie order matters (3.06%), so the trigger
fires, but F3 is a stability fix, not a net-gain fix; its replacement-key flip-rate
comparison was not run and would not change the failing net. **Not kept.**
**F5 (bandwidth)**: minor axis-only drift floor; not run, would not change the
net. **Not kept.**

## TARGET verdict

**TARGET: held-out real−no_edge on PRIMARY ≥ 0 with interval not entirely below
zero, keeping ≥75% of the ORIGINAL reference's case (c) lift.**

- On dev, **no F0 arm** (and no other triggered arm) achieves a share-weighted net
  interval **above zero**. The closest, beta=0.25, reaches −0.0016 [−0.0043,
  +0.0013] — crossing zero but not above it, and the reference −0.0172 sits inside
  the 1c jitter envelope [−0.0645,+0.0204].
- Case (c) is ABSENT at both radii (n=25 per-kind, 139 at 8px), so the ≥75%-lift
  criterion is **not evaluable**; the case-(c) lift (+0.24) exists only on a
  population too small to validate.

**Pre-registered outcome applied: "if no arm passes keep rule → ship nothing new +
dev numbers."** Nothing new is shipped; the unmodified objective (beta=0.50) is the
reference.

## Invariants

- 0px precision 1.0000, coverage 1.0000, D1(0)=1.0000 (class-filtered; pooled
  D1(0)=0.8638) — unchanged by the repair (no node deletion vs frozen ids).
- Determinism: all runs on PRIMARY; no new solvers, no MILP, no
  `EDGE_BUDGET_PER_FRAME`/`FAMILY_SEED_EDGES` changes.
- `matching/structural_similarity.py`'s `_relational_agreement` **not touched**.
- Flag values committed: all FE/MATCH fix flags **default OFF**; reference beta
  (`MATCH_SYSTEMATICITY_WEIGHT=0.5`) unchanged.

## Files

- `Topology/t4_shift.py` (repaired `counterpart_map`, `radius_by_lens`)
- `Topology/t4_part0b_validate.py`, `t4_compute_radius.py`, `t4_b1_primary.py`,
  `t4_part0e_order.py`, `t4_part1_1b_1d.py`, `t4_part1_f0.py`, `t4_part1_1c.py`
- `Topology/results/t4/`: `part0b_validate.json`, `primary_radius_dev.json`,
  `b1_primary.json`, `part0e_order.json`, `part1_1b_1d.json`,
  `part1_f0_baseline.json`, `part1_1c_jitter.json`, `part0_closure_0f_0g.md`,
  this report.
