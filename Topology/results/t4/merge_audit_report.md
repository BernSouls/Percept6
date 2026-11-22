# Merge audit: arena/c316ca3a-percept6 → main

Merge commit **`e065170`** (pushed to `origin/main`, `8e05a7b..e065170`).
Merged tree verified **identical to branch tip `18961ed`** (350 files).

## A1 — divergence

- **Branch commits since divergence (18):** `b541c99` (Merge PR #1) → `9232c6f`
  (Task 3 soft weights+vector address) → `56e340e` (Front-end sampling) →
  `c62719e` → `1543827` → `caedf1e` → `25b019d` → `8238762` → `33c0d5b` →
  `805bd2f` → `bfa19ac` → `3f6b708` → `8d7659e` → `202db9a` → `32112f6` →
  `9bbcc69` → `5e8f37f` → `18961ed` (Task 4 mission).
- **Main commits since divergence (1):** `8e05a7b` "Add files via upload".
- **Overlap: NONE — true fork.** `git merge-base HEAD origin/main` is empty; main
  is a single uploaded snapshot (136 files), the branch is the developed
  lineage (350 files). Branch is 18 ahead, main 1 ahead; no shared ancestor.

## A2 — file buckets (240 changed files vs main)

- **(i) harness-only** (Topology/ scripts, results JSONs, docs, data, tests):
  `Topology/**` (174), `results/**` (24), `data/**` (17),
  `work/topology/README.md`, `.gitignore`, `01a1edd7-*.patch`,
  `tests/test_section_b.py`.
- **(ii) production, behind a flag that defaults OFF:** the FE_*-gated
  front-end mechanism code — `perception/front_end_mechanisms.py`,
  `perception/contour_field.py`, `perception/contour_peak.py`, `lenses/**`,
  and the `FE_L3_SPINE/FE_L4_JUNCTIONS/FE_L6_ADDRESS/FE_VECTOR_ADDRESS/
  FE_SUBPIXEL_PEAK_POS`-gated code. All OFF in the merged tree → inert.
- **(iii) production DEFAULT change / shipped-path:** `parameters/_text.py`
  (**`BINDER_COMPLETE_POSITION_MAP` False→True** — the one committed flag
  flipped ON, validated; smoke pin below) plus the always-on shipped-pipeline
  changes from Task 2/3/4 — `matching/joint_mcs.py`, `matching/alignment.py`,
  `matching/structural_similarity.py`, `matching/graded_mcs.py`,
  `memory/cores.py`, `orchestration/binder.py`, `learning/fertility.py`,
  `learning/slots.py`, `eth80_frontend.py`, and the new shipped modules
  `core/**`, `memory/**` (new), `matching/**` (new), `parameters/
  {matching,vision,memory,dynamics,__init__}.py`, `_bootstrap.py`,
  `sensory_floor.py`, `concept_graph.py`, `consolidator.py`, `action/**`,
  `dynamics/**`, `diagnostics/**`, `eth80_split.py`, `derived_lenses.py`.
- **(iv) signature/API change to a shared function:** `Topology/t4_shift.py`
  — `counterpart_map` now returns `(mapping, info, stats)` (3 values).

## A3 — counterpart_map callers

Grep of the **entire merged tree**: **34** `counterpart_map(` call sites.
**All 34 unpack exactly 3 values; 0 callers expect 2.** (`t4_measure_4c.py`'s
`_counterpart_map` is a fully local reimplementation returning a dict — it does
not call the module function.)

## B1 — flag table (merged tree)

| flag | committed default | bucket | mission |
|---|---|---|---|
| BINDER_COMPLETE_POSITION_MAP | **True** | iii | Front-end sampling (`56e340e`), smoke-validated |
| FE_SOFT_CHANNEL_GATE | True | ii→ON | Task 3 (PR #1 `b541c99`), calibrated |
| FE_SOFT_L3_L4_GATE | True | ii→ON | Task 3 (PR #1 `b541c99`), calibrated |
| FE_L3_SPINE | False | ii | Task 3 (OFF) |
| FE_L4_JUNCTIONS | False | ii | Task 3 (OFF) |
| FE_L6_ADDRESS | False | ii | Task 3 (OFF) |
| FE_VECTOR_ADDRESS | False | ii | Task 3 (OFF) |
| FE_SUBPIXEL_PEAK_POS | False | ii | Front-end sampling (OFF) |
| MATCH_EDGE_MIN_CREDIT | False | ii | Task 4 (OFF) |
| MATCH_EXCLUDE_ISOLATED_FLOOR | 0.5 | ii (param) | Task 4 |
| MATCH_SYSTEMATICITY_WEIGHT | 0.5 | (reference, unchanged) | — |

## B2 — new mechanism flags OFF; BINDER pin

All NEW mechanism flags are **OFF** except the three explicitly turned ON and
validated: **FE_SOFT_CHANNEL_GATE**, **FE_SOFT_L3_L4_GATE** (Task 3,
soft_gate_calibration.json + section_b_verdict 21 FIXED), **BINDER_COMPLETE_
POSITION_MAP** (Front-end sampling).

**BINDER_COMPLETE_POSITION_MAP smoke pin** (`t4_smoke_binder_pin.py`, re-run on
merged tree): ON graph_hash=`b47b2e133e604bd3` (positioned 201/201,
content 143) vs OFF graph_hash=`91ad15813bbe25df` (positioned 115/201, content
57); **hashes differ, +86 positioned nodes** (+86 content). Flag changes emitted
graph structure, not just positions.

## C — invariants on merged tree

- **C1** pickle load: 30/30 sampled pickles load (Subgraph intact), 0 failures.
- **C2** determinism: graph hashes identical under PYTHONHASHSEED 0 and 1729 —
  B0=`bc57bf39416dfecb`, B1=`960189f92ae1635f`, F=`b910d1119918bd43`.
- **C3** Subgraph & TypedSignature class bodies **identical** to main (no field
  additions).
- **C4** 0px: matcher precision **1.0000** (n=24), pooled content coverage
  **1.0000** (n=24), class-filtered coverage **1.0000** (n=24), D1(0)=**1.0000**
  (n=24). Coverage = not deletion vs frozen ids.
- **C5** `structural_similarity.py::_relational_agreement`: the **opposite
  no-relations convention is intact** (agree_u/agree_v = 1.0 when no relations,
  both sides). The function body was extended by earlier missions with
  weight-aware counting behind `EDGE_MATCH_USE_CONFIDENCE` (default OFF → reduces
  to unit counting) and STAGE 10 denominator/direction fixes — flag-gated, not a
  convention flip.

## D — conflicts and merge

- **D1** main had moved to `8e05a7b` (unrelated snapshot). Merged with
  `--allow-unrelated-histories --no-ff`. **25 add/add conflicts** = exactly the
  25 differing shared files. **Every conflict resolved to the branch
  (developed) version** (`--theirs`), because main is a strict subset snapshot
  with no unique source and the branch tree is the verified merged tree. The one
  main-only file, `learning/__pycache__/fertility.cpython-311.pyc` (build
  artifact), was removed to match branch intent. No invariant check dropped.
- **D2** **Merge commit** `e065170` (non-linear), parents `8e05a7b` (main) +
  `18961ed` (branch). Merged HEAD = `e065170`. Includes all 18 branch commits.

## E — merge report

- **E1** Production behaviour change on main: `BINDER_COMPLETE_POSITION_MAP=True`
  (emits 86 more positioned nodes, graph structure changes — validated). Soft
  gates (`FE_SOFT_CHANNEL_GATE`, `FE_SOFT_L3_L4_GATE`) ON (Task 3, calibrated).
  Remaining FE_* / MATCH_* mechanism flags **OFF** (inert). Harness-only:
  Topology/ scripts, results JSONs, docs.
- **E2** Withdrawn-claims record present: `Topology/results/t4/part0_closure_0f_0g.md`
  (6 bogus claims, one line each) + `t4_mission_report.md` + instrument repair in
  `Topology/t4_shift.py`. All PRESENT in merged tree.
- **E3** Flag values merged: as in B1/E3 table. Invariants C1-C5 all PASS.
  **Merge status: COMPLETE.**
