# KEEP_MAP -- what the shipped path touches, and where each thing lives

Written before anything moved (Mission 24, item 1).  The rule the archive follows is at the
bottom; this table is what the rule was applied to.

The shipped system is: **content-count door + appearance, RRF k = 60, shortlist S = 8, then the
M19 matcher's plain re-order**.  Its one entry point is `matching/placement_recognise.py`
(`build_gallery`, `recognise`), and its one configuration is `matching/placement_bundle.json`.

## 1. Modules the shipped path imports

| file | lives in | note |
|---|---|---|
| `matching/placement_recognise.py` | **core (new)** | the entry point: build_gallery, recognise, order_plain |
| `matching/placement_bundle.py` | **core (new)** | the frozen bundle: load/validate/freeze, refusal on mismatch |
| `matching/placement_gallery.py` | **core (new)** | place records -> Gallery (index + appearance + disk cache) |
| `matching/placement_frontend.py` | **core (new, moved)** | the RECIPE, `cell`, `place_record`, `pair_score` -- was `Topology/placement_m20_eth80.RECIPE/_recipe/_cell` + `Topology/placement_m23_eth80.image_job` |
| `matching/placement_truth.py` | **core (new, moved)** | the exact-transform truth rule -- was `Topology/placement_m12_read_real._forward/_inverse/_truth` |
| `matching/placement_cache.py` | **core (new, moved)** | content-hash graph cache -- was `Topology/placement_cache.py`, which now re-exports it |
| `matching/placement_frontdoor.py` | core (was already) | PlaceIndex, count_scores, appearance, rank_fuse |
| `matching/placement_prefilter.py` | core | `load_params`, `bag_v2`, `standardize`, `score_bags_v2` |
| `matching/placement_edges.py` | core | `node_frames_keypoint` (the place frames) |
| `matching/placement_matcher.py` | core | `build_side_cached`; its `_image_digest` now calls core's `key_for` |
| `matching/joint_mcs.py` | core | the M19 matcher entry point, returns `score_shipped` |
| `matching/m15_manifest.py` | core | writes every result JSON with its manifest |
| `eth80_frontend.py`, `concept_graph.py`, `parameters/`, `core/`, `perception/`, `lenses/` | core (top-level) | the front end that builds graphs; unchanged by this mission |
| `Topology/guard.py` | Topology, **kept** | the measurement guard; core does not import it and the moved code calls it nowhere |

**Before this mission the shipped path imported two Topology modules** (`placement_cache` from
`matching/placement_matcher`, and the truth rule from `matching/m15_places`), so "no imports from
Topology/" could not have passed.  Both moved; `tests/test_m24_core.py::TestNoTopologyImport`
walks the AST of every core package and fails on either.

## 2. Data files the shipped path reads

| file | lives in | why it is frozen |
|---|---|---|
| `matching/placement_bundle.json` | core (new) | the configuration: recipe, door `n_hits`, RRF `k`, S, rerank, M19 weight, provenance, code digest |
| `matching/placement_score_params.json` | core | the M19 record's weight `w = 0.375` (and its fit provenance) |
| `matching/placement_prefilter_params.json` | core | the Mission-21 frozen ruler: per-slot mu/sd/keep + widths sigma 0.35 / 0.50 |
| `.cache/graphcache/*.pkl` | working state (gitignored) | built graphs, keyed on pixels + switch digest + source digest |
| `.cache/placement/gallery-*.pkl` | working state (gitignored) | gallery records, keyed on items + the bundle's code digest |
| `data/ETH-80-master/` | corpus (gitignored symlink) | the images; fetched by `Topology/fetch_corpus.py --fetch` |
| `matching/placement_frozen_params*.json` | core | older frozen M17/M18 bundles: **no shipped code reads them** -- kept because the matcher's own tests pin them |

## 3. What the kept checks need from `Topology/` (kept, in place)

| root | modules it pulls in |
|---|---|
| `Topology/placement_m12_perf.py` (the legacy parity check) | corpus, placement_cache, placement_census, placement_dev, placement_eval_spec, placement_families_m9, placement_gate, placement_m11_eth80, placement_m11_families, placement_scenes, placement_truth, guard |
| `Topology/verify_m24_equivalence.py` (item 4) | placement_m23_eth80, placement_m22_eth80, placement_m20_eth80, placement_m23_stages, placement_m12_read_real |
| the report-producing M20-M23 harnesses | placement_m20_eth80, m22_eth80, m23_eth80, m23_stages, m21_* , m20_fit, m22_power, m12_realphotos |
| the test suite (`tests/`) | 20 Topology modules, all inside the two lists above |
| result files cited by the README / the reports | `m12_perf_baseline.json`, `m20_splits.json`, `m22_*.json`, `m23_*.json`, `m23_read_rows.jsonl`, `m23_read_pairs.jsonl`, MISSION14..23 `.md` |

## 4. The rule that decides the archive

> A `Topology/` file stays if it is (a) the legacy parity check or a module that check imports,
> (b) the item-4 verification or a module it imports, (c) a report-producing harness of M20-M23
> plus its imports, or (d) read/written by a kept test.  Everything else moves to `archive/`
> with `git mv` -- nothing is deleted, and `archive/ARCHIVE_INDEX.md` says which mission each file
> served and where its result is reported.

Applied mechanically: the import closure of the four roots above (parsed with `ast`, resolved
against the tree) is the keep set -- **27 modules**; the other 201 `Topology/*.py` files move.
`parameters/_text.py` is NOT touched by any of this: every legacy `PLACEMENT_*` flag stays, because
the front-end switch digest and the M14 switch-digest test both name them.
