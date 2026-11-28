# ARCHIVE INDEX -- everything Mission 24 moved out of the way, and why

Mission 24 put the working system in core (`matching/placement_recognise.py`) and moved what the
shipped path does not need here.  **Nothing was deleted**: every entry below was moved with
`git mv`, so its history is intact and `git log --follow` still works.

**Mission 28 moved nothing here.**  Its five new harnesses (`m28_determinism.py`, `m28_gauge.py`,
`m28_records.py`, `m28_door.py`, `m28_speed.py`) are live in `Topology/` and are listed in
`KEEP_MAP.md`; the mission added no file to the archive and deleted none (it did add
`data/placement_records_m28.npz`, the M27 record file plus its six content columns).

The archive is not part of the product.  Nothing in `matching/`, `core/`, `parameters/`,
`perception/`, `lenses/` or a kept test imports anything from here, and nothing here is imported by
the legacy parity check (`Topology/placement_m12_perf.py --check`) or the item-4 verification
(`Topology/verify_m24_equivalence.py`).  The rule that produced this list is in `KEEP_MAP.md` §4.

Two things stayed in place that look like candidates, deliberately:

* **`parameters/_text.py` and every `PLACEMENT_*` flag.**  The front-end switch digest and the M14
  switch-digest test both name them, so deleting a "legacy" flag would silently invalidate the graph
  cache key and re-open the hole that test exists to close.  They are legacy; they are also load-
  bearing.  The shipped entry point does not read them (it applies its own frozen bundle), and
  `tests/test_m24_core.py::TestEntryPointNeedsNoEnv` proves that a run needs no variable set and
  leaks none afterwards.
* **`Topology/fetch_corpus.py`.**  It is how a fresh checkout gets the 330 MB ETH-80 tree; archiving
  the only way to fetch the corpus would make everything above unreachable.  It stays a `Topology/`
  utility on purpose -- it is setup, not product.
* **The M20-M23 harnesses and their result files.**  They are how the numbers in the reports were
  produced, and the item-4 verification imports two of them.  Their scripts are in `Topology/`, not
  here; their result JSONs are in `Topology/results/placement/`.

Columns: file -> mission -> one line on why it is here -> where its result is reported.  Files are
grouped by mission; within a group the reason is the same shape because the rule is the same shape.



**Mission 34 moved nothing here.**  Its six harnesses (`m34_fragments.py`, `m34_door.py`,
`m34_rerank.py`, `m34_audit.py`, `m34_anchors.py`, `m34_wire.py`) are live in `Topology/` and listed
in `KEEP_MAP.md` §3.  It archived nothing and deleted nothing -- but it did find 19 orphan modules
(10,291 lines) that no shipped path can reach, including 3,185 lines of `surface_fragments` machinery;
they are LEFT IN PLACE because the census that names them (`Topology/m34_audit.py --run`) is the point,
and moving them here would hide the finding.  It also commits one new data file,
`data/placement_anchors_m34.npz` (616 KB), because the place record does not carry the fragment
anchors the shipped door reads.

**Mission 33 moved nothing here.**  Its three harnesses (`m33_records.py`, `m33_curvature.py`, and
the M32 ones it reuses) are live in `Topology/` and listed in `KEEP_MAP.md` §3; the mission archived
no file and deleted none.  It did add something to the repository that is not a script:
`data/placement_records_m32.npz` is now COMMITTED (38.03 MB, exempted from `.gitignore`), because a
fresh checkout must be able to load the record file without an eight-minute rebuild.

## Mission 32 -- 16 files

**Mission 32 moved 16 scripts here and nothing else.**  The rule is unchanged (`KEEP_MAP.md` §4):
a `Topology/` file stays if the parity check, the item-4 verification, a report-producing harness or
a kept test needs it.  M32 applied the rule to the M26-M31 range by AST closure and kept exactly the
scripts its own gates or the records/read chain use — `m26_records.py`, `m26_oracle.py`,
`m27_read.py`, `m27_export_records.py`, `m30_read.py`, `m30_records.py`, `m30_parity.py`,
`m31_determinism.py` — and moved the rest.  It also moved `MISSION30_REPORT.md` and
`MISSION31_REPORT.md` from the repository root to `Topology/results/placement/`, next to M14-M29.

| file | mission | why it is here | where its result is reported |
|---|---|---|---|
| `m26_census.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION26_REPORT.md` |
| `m26_matcher.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION26_REPORT.md` |
| `m27_door.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION27_REPORT.md` |
| `m27_drift.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION27_REPORT.md` |
| `m27_profile.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION27_REPORT.md` |
| `m28_determinism.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION28_REPORT.md` |
| `m28_door.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION28_REPORT.md` |
| `m28_gauge.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION28_REPORT.md` |
| `m28_records.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION28_REPORT.md` |
| `m28_speed.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION28_REPORT.md` |
| `m29_representation.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION29_REPORT.md` |
| `m30_door.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION30_REPORT.md` |
| `m30_shape.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION30_REPORT.md` |
| `m31_arms.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION31_REPORT.md` |
| `m31_shape.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION31_REPORT.md` |
| `m31_wire.py` | M32 | in the M26-M31 one-off range, and nothing kept imports it (checked by AST closure over `tests/`, `scripts/`, `_bootstrap.py`, `matching/` and the live `Topology/` scripts); M32's own harnesses and the records/read chain it needs stayed | `archive/../Topology/results/placement/MISSION31_REPORT.md` |

## At a glance

| | files | what |
|---|---|---|
| scripts | 199 | `Topology/*.py` |
| result files | 175 | `Topology/results/placement/*` |
| **total** | **374** | **56.8 MB** |

The working tree went from `Topology/` 68 MB to 12 MB; this directory is the other 55 MB.


## M3 -- 11 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m3.py` | the placement series' M3 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look1.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look1_nodes.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look2.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look2_nodes.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look3.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look3_nodes.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look4.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_look4_nodes.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_smoke.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/m3_smoke_nodes.json` | a M3 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |

## T3 series -- 3 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `t3_baseline.py` | a T3-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `t3_build_variant.py` | a T3-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `t3_measure.py` | a T3-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION3_REPORT.md |

## T4 series -- 46 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `t4_a4_tolerance_drift.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b1_placement.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b1_primary.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b2_structure_only.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b3_identifiability.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b4_aperture.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b5_credit.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b6_seed_precheck.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b7_emission_selection.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_b8_isolated_confound.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_build_dev.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_build_dev_all.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_build_heldout.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_compute_radius.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_determinism.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_e6_flicker.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_final_heldout_compare.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_invariants.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_measure_4c.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_measure_4d.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_measure_4e.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_p3_4_fliprate.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0_corrections.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0_crosscheck.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0_dev.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0_repair.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0b_validate.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part0e_order.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1_1b_1d.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1_1c.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1_diagnose.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1_edge_diag.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1_f0.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part1b_probe.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part2_measure.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part2d_armselect.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_part3_triggers.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_partA_corrections.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_partE_report.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_retrieval_analyze.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_retrieval_build.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_retrieval_c1.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_retrieval_log.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_retrieval_measure.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_shift.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `t4_smoke_binder_pin.py` | a T4-series measurement script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION4_REPORT.md |

## M4 -- 4 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `results/placement/m4_precision_before.json` | a M4 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `results/placement/m4_precision_off4.json` | a M4 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `results/placement/m4_precision_subpixel.json` | a M4 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION4_REPORT.md |
| `results/placement/m4_stage0b_padding.json` | a M4 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION4_REPORT.md |

## M5 -- 6 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `results/placement/m5_prec_off.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |
| `results/placement/m5_prec_subpix.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |
| `results/placement/m5_precision_nodetonode.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |
| `results/placement/m5_stage0_census.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |
| `results/placement/m5_stage2a_population.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |
| `results/placement/m5_stage3_keypoints.json` | a M5 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION5_REPORT.md |

## L6 -- 2 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `l6_eval.py` | an L6 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION6_REPORT.md |
| `l6_smoke.py` | an L6 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION6_REPORT.md |

## M6 -- 4 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `results/placement/m6_attach_coverage.json` | a M6 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION6_REPORT.md |
| `results/placement/m6_collapse_on.json` | a M6 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION6_REPORT.md |
| `results/placement/m6_prec_writer_off.json` | a M6 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION6_REPORT.md |
| `results/placement/m6_prec_writer_on.json` | a M6 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION6_REPORT.md |

## M7 -- 5 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `results/placement/m7_attach_places.json` | a M7 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION7_REPORT.md |
| `results/placement/m7_dev5.json` | a M7 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION7_REPORT.md |
| `results/placement/m7_dev5_fit.json` | a M7 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION7_REPORT.md |
| `results/placement/m7_dev5_places.json` | a M7 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION7_REPORT.md |
| `results/placement/m7_prec_places.json` | a M7 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION7_REPORT.md |

## M8 -- 16 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_step0_m8.py` | a step-0 script of M8; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `placement_step2_m8.py` | a step-2 script of M8; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `placement_step3_m8.py` | a step-3 script of M8; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `placement_step4_m8.py` | a step-4 script of M8; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `placement_step5_m8.py` | a step-5 script of M8; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_frontend_freeze.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_prec_after.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_prec_before.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_preregistration.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_read.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_real_off.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_real_on.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_step0.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_step0b.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_step2.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |
| `results/placement/m8_step2_smoke.json` | a M8 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION8_REPORT.md |

## early (pre-M9) -- 77 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `_bench_solver.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `_calib_tracer.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `aggregate_survival.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `build_renderer_corpus.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `classify.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `cost_probe.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `det_hash.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `diag_winner_cell.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `err_shape.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `frontend.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `known_transform.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `known_transform_arms.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `known_transform_arms_zero.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `loss_diag.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `make_report.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `obj_check.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `object_local_kt.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `part0_aggregate.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `part0_stats.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `pin_ruler.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_arms.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_arms2.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_cacheclosure_m5.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_coverage_m6.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_deployed_m9.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_dev3.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_dev5.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_dev5_fit.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_dev5_run.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_devfit.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_equiv_m9.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_frame_test.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_freeze_m9.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_keypoints_m5.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_precision.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_precision_m5.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_reach.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_separation.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_step0b_m8.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_tolerance.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `provenance.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `section_b_audit.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `soft_gate_calibrate.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `solver.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `step_a7_relations.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `step_a8_selected.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `step_a_ruler.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `step_f_measure.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `step_h_walker.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `survival_table.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `test_placement.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `test_placement_m3.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `transfer_check.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `tune.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `vector_address_final_check.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `winner_details.py` | an early-mission script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/PART2_PREREGISTRATION.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/PART6_PREREGISTRATION.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/REPORT.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/STAGE0_SCENE_LEVEL.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/arms_dev.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/arms_eval.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/census_dev.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/devfit_tolerance.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/eval_nodes.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/frame_test.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part0_answers.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part1_reach.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part1_reach.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part2_part3.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part3_separation.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part4_gate.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part4_gate.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part6_arms.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/part6_arms.md` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/stage0_scene_level.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |
| `results/placement/tolerance.json` | an early result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION3_REPORT.md |

## M9 -- 19 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m9_checks.py` | the placement series' M9 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_m9_gallery.py` | the placement series' M9 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_m9_prereg.py` | the placement series' M9 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_m9_read.py` | the placement series' M9 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `placement_m9_step5.py` | the placement series' M9 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_checks_ir.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_checks_oop.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_checks_smoke.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_deployed.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_equiv.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_ir_check.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_ir_fit.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_ir_smoke.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_oop_check.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_oop_fit.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_preregistration.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_step3_recall.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_step3_recall24.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |
| `results/placement/m9_step5_spacing.json` | a M9 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION9_REPORT.md |

## M10 -- 25 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m10_dev5_fit.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_eth80.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_eth80_agg.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_gallery.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_prereg_oop.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_read_oop.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_refit.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_spacing.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `placement_m10_step3.py` | the placement series' M10 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_dev5_fit.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_dev5_fit_progress.jsonl` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_eth80.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_eth80_agg.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_eth80_progress.jsonl` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_eth80_shortlist.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_oop_check.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_oop_check_progress.jsonl` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_oop_fit.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_prereg_oop.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_refit.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_shortlist.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_spacing_fit.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_spacing_measure.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_step3.json` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |
| `results/placement/m10_step3_progress.jsonl` | a M10 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION10_REPORT.md |

## M11 -- 20 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m11_coverage.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_eth80_agg.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_fit_w.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_pear.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_prereg_eth.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_prereg_synth.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `placement_m11_synth.py` | the placement series' M11 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_coverage.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_coverage_ms.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_eth80.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_eth80_progress.jsonl` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_eth80_shortlist.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_fit_w.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_fit_w_progress.jsonl` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_pear.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_prereg_eth.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_prereg_synth.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_synth.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_synth_read.jsonl` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |
| `results/placement/m11_synth_shortlist.json` | a M11 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION11_REPORT.md |

## M12 -- 26 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m12_e_coverage.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_fit_sweep.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_fit_w.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_frames.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_frames2.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_prereg_real.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_prereg_synth.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_read_synth.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_sigcache.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_synth_shortlist.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_truthcalib.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `placement_m12_truthcheck.py` | the placement series' M12 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_e_coverage.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_fit_w.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_fit_w_progress.jsonl` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_fit_w_sweep.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_frames.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_frames2.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_prereg_real.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_prereg_synth.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_read_real.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_read_real_progress.jsonl` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_real_photos.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_shortlist.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_sigcache.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |
| `results/placement/m12_truth_rule_calib.json` | a M12 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION12_REPORT.md |

## M13 -- 13 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m13_decompose.py` | the placement series' M13 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `placement_m13_frames.py` | the placement series' M13 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `placement_m13_margin.py` | the placement series' M13 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `placement_m13_read.py` | the placement series' M13 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `placement_m13_repeat.py` | the placement series' M13 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_decompose.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_frames.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_margin.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_prereg_read.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_read_real.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_read_real_progress.jsonl` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_repeat.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |
| `results/placement/m13_repeat_boot.json` | a M13 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION13_REPORT.md |

## M15 -- 9 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m15_atlas.py` | the placement series' M15 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `placement_m15_budget.py` | the placement series' M15 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `placement_m15_geometry.py` | the placement series' M15 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `placement_m15_writers.py` | the placement series' M15 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `results/placement/m15_atlas.json` | a M15 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `results/placement/m15_budget.json` | a M15 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `results/placement/m15_geometry.json` | a M15 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `results/placement/m15_item5b.json` | a M15 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION15_REPORT.md |
| `results/placement/m15_position_writers.json` | a M15 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION15_REPORT.md |

## M16 -- 8 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m16_bank.py` | the placement series' M16 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `placement_m16_graphs.py` | the placement series' M16 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `placement_m16_similarity.py` | the placement series' M16 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `placement_m16_summary.py` | the placement series' M16 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `results/placement/m16_bank.json` | a M16 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `results/placement/m16_graphs.json` | a M16 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `results/placement/m16_graphs_summary.json` | a M16 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION16_REPORT.md |
| `results/placement/m16_similarity.json` | a M16 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION16_REPORT.md |

## M17 -- 15 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m17_context.py` | the placement series' M17 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `placement_m17_eval.py` | the placement series' M17 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `placement_m17_fresh.py` | the placement series' M17 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `placement_m17_freshread.py` | the placement series' M17 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `placement_m17_llr.py` | the placement series' M17 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_assign.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_assign2.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_attrib_m16.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_attrib_m17.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_context.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_ctx_sweep.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_fresh.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_freshread.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_llr.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |
| `results/placement/m17_tables.json` | a M17 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION17_REPORT.md |

## M18 -- 23 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m18_bank.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `placement_m18_capcost.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `placement_m18_diagnose.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `placement_m18_fit.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `placement_m18_shift.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `placement_m18_summary.py` | the placement series' M18 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_bank_cap24.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_capcost.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_cons.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_cons_cap24.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_diagnose.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_eval_soft.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_eval_tbexpf.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_eval_tbfine.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_eval_tbhyb.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_fit.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_full_cap24.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_full_cap36.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_full_cap6.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_o5_fresh.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_shift_cap24.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_shift_cap6.json` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |
| `results/placement/m18_tables` | a M18 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION18_REPORT.md |

## M19 -- 17 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m19_descriptor.py` | the placement series' M19 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `placement_m19_fresh.py` | the placement series' M19 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `placement_m19_read.py` | the placement series' M19 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `placement_m19_texture.py` | the placement series' M19 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_cap6.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_desc24.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_desc72.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_descriptor.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_o2_adapt128.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_o2_dedup.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_o5_a0.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_o5_a1a2.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_read.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_texture.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_w25_adapt.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_w25_full.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |
| `results/placement/m19_w50_full.json` | a M19 result file; superseded by the M20-M23 record and not read by any kept code | archive/../Topology/results/placement/MISSION19_REPORT.md |

## M20 -- 4 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m20_attrib.py` | the placement series' M20 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION20_REPORT.md |
| `placement_m20_oop.py` | the placement series' M20 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION20_REPORT.md |
| `placement_m20_parity.py` | the placement series' M20 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION20_REPORT.md |
| `placement_m20_splits.py` | the placement series' M20 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION20_REPORT.md |

## M21 -- 4 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m21_analysis.py` | the placement series' M21 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION21_REPORT.md |
| `placement_m21_before.py` | the placement series' M21 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION21_REPORT.md |
| `placement_m21_k.py` | the placement series' M21 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION21_REPORT.md |
| `placement_m21_yaw.py` | the placement series' M21 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION21_REPORT.md |

## M22 -- 3 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m22_audit.py` | the placement series' M22 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION22_REPORT.md |
| `placement_m22_keysweep.py` | the placement series' M22 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION22_REPORT.md |
| `placement_m22_placeproof.py` | the placement series' M22 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION22_REPORT.md |

## M23 -- 2 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_m23_geovote.py` | the placement series' M23 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION23_REPORT.md |
| `placement_m23_power.py` | the placement series' M23 step script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | archive/../Topology/results/placement/MISSION23_REPORT.md |

## L3/L4 -- 7 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `l3_l4_eth80_expanded.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_eth80_gates.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_eth80_gates_final.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_gates.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_renderer_gates.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_smoke.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `l3_l4_step0.py` | an L3/L4 experiment; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |

## M4/M5 -- 5 files

| file | why it is archived | where its result is reported |
|---|---|---|
| `placement_stage0.py` | an early placement stage script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `placement_stage0_m4.py` | an early placement stage script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `placement_stage0_m5.py` | an early placement stage script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `placement_stage0b_m4.py` | an early placement stage script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
| `placement_stage2a_m5.py` | an early placement stage script; not imported by the shipped path, the parity check, the item-4 verification, a kept harness or a kept test | the M20-M23 record (Topology/results/placement/MISSION2*_REPORT.md) |
