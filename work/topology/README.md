# Front-end fix session — harness location and export adaptations

The user's harness is shipped in this checkout as **`Topology/`** (capital T),
not `work/topology/`.  It is the same code the brief calls "the existing
harness in work/topology": `step_a_ruler.py` (the frozen ruler's *code*),
`step_a7_relations.py`, `step_a8_selected.py`, `survival_table.py`,
`classify.py`, `guard.py`, `corpus.py`, `step_h_walker.py`, `part0_stats.py`,
`pin_ruler.py`, etc.  All measurement scripts are run as

```
.venv/bin/python Topology/<script>.py ...
```

## What was missing from the export and had to be reconstructed

| item | status | reconstruction |
|---|---|---|
| product import graph | broken | `sensory_floor.py`, `derived_lenses.py`, `_bootstrap.py`, `learning/__init__.py`, `action/decode.py`, `diagnostics/` alias shims (pure re-exports of the shipped definitions) |
| `learning/fertility.py`, `learning/slots.py` | absent on the session branch, added by the user on `origin/main` | taken verbatim from `origin/main` |
| `data/ETH-80-master` | absent | `Topology/fetch_corpus.py` downloads the `chenchkx/ETH-80` GitHub mirror into `.cache/` and symlinks it; verified 8 classes x 10 instances x 41 views = 3280 images + 3280 masks |
| `data/renderer-truth-v1` | absent | `Topology/build_renderer_corpus.py` regenerates 15 deterministic 192x192 scenes from the repo's own `perception/synthetic3d.py` renderer |
| `results/step_a_ruler.json`, `results/step_a_ruler_pin.json` | **ABSENT** | the brief's frozen artifact (sha256 `15c2b7c9...`, canonical content `1d8cf807...`) is not in this checkout and cannot be reproduced without the original corpus.  The ruler CODE is used unchanged; the rebuilt artifact is pinned with its own hash and every number in the report cites that hash. |

## Path adaptations (no measurement definitions changed)

`Topology/guard.py` was written for a `work/src` + `work/diagnostics` tree.
In this flat export: `SRC = TREE` (repository root), `source_digest()` walks a
whitelist of product directories instead of the whole root (so writing
results cannot change graph cache keys), and `Path(guard.TREE).parent /
"data"` became `Path(guard.TREE) / "data"` in the eight scripts that name
ETH-80 images.  These are the only edits to the shipped harness.

## Session utilities added under `Topology/`

* `fetch_corpus.py` — download + verify ETH-80 (`results/corpus_verify.json`).
* `build_renderer_corpus.py` — deterministic renderer-truth-v1 corpus.
* `front_end_fix.py` — the flagged front-end mechanisms (content-anchored
  spine, junctions, host/arc records, graded walker seeds).
* `measure_session.py` — the I2 dual-denominator census, relation census by
  endpoint type, junction census and gate logs on one fixed base list.
* `pin_session.py` — pins the rebuilt ruler and the before/after artifacts.

## Frozen files

`matching/alignment.py`, `matching/joint_mcs.py`, `matching/graded_mcs.py`,
`matching/relation_walker.py` are frozen EXCEPT the section B defect fixes,
which are pinned by `tests/test_section_b.py`.
